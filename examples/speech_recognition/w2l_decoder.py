#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wav2letter decoders.
"""

from collections import namedtuple, deque
import gc
import itertools as it
import numpy as np
import torch
import os.path as osp
import warnings
from fairseq import tasks
from fairseq.utils import apply_to_sample
import torch.nn.functional as F
from examples.speech_recognition.data.replabels import unpack_replabels

try:
    from wav2letter.common import create_word_dict, load_words, Dictionary
    from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from wav2letter.decoder import (
        CriterionType,
        DecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
        LexiconFreeDecoder,
    )
except:
    warnings.warn(
        "wav2letter python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/wav2letter/wiki/Python-bindings"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        if args.criterion == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.asg_transitions = args.asg_transitions
            self.max_replabel = args.max_replabel
            assert len(self.asg_transitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = filter(lambda x: x >= 0, idxs)
            idxs = unpack_replabels(list(idxs), self.tgt_dict, self.max_replabel)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.silence = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.lexicon = load_words(args.lexicon)
        self.word_dict = create_word_dict(self.lexicon)
        self.unk_word = self.word_dict.get_index("<unk>")

        self.lm = KenLM(args.kenlm_model, self.word_dict)
        self.trie = Trie(self.vocab_size, self.silence)

        start_state = self.lm.start(False)
        for i, (word, spellings) in enumerate(self.lexicon.items()):
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idxs = [tgt_dict.index(token) for token in spelling]
                assert (
                    tgt_dict.unk() not in spelling_idxs
                ), f"{spelling} {spelling_idxs}"
                self.trie.insert(spelling_idxs, word_idx, score)
        self.trie.smear(SmearingMode.MAX)

        self.decoder_opts = DecoderOptions(
            args.beam,
            int(getattr(args, "beam_size_token", len(tgt_dict))),
            args.beam_threshold,
            args.lm_weight,
            args.word_score,
            args.unk_weight,
            args.sil_weight,
            0,
            False,
            self.criterion_type,
        )

        if self.asg_transitions is None:
            N = 768
            # self.asg_transitions = torch.FloatTensor(N, N).zero_()
            self.asg_transitions = []

        self.decoder = LexiconDecoder(
            self.decoder_opts,
            self.trie,
            self.lm,
            self.silence,
            self.blank,
            self.unk_word,
            self.asg_transitions,
            False,    # Lexicon decoder with word-LM
            # True,       # Lexicon decoder with tkn-LM
        )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
        return hypos


class W2lKenLMFreeDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.silence = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        token_dict = Dictionary(args.lexicon)
        self.lm = KenLM(args.kenlm_model, token_dict)

        self.decoder_opts = DecoderOptions(
            args.beam,
            int(getattr(args, "beam_size_token", len(tgt_dict))),
            args.beam_threshold,
            args.lm_weight,
            args.word_score,
            args.unk_weight,
            args.sil_weight,
            0,
            False,
            self.criterion_type,
        )

        if self.asg_transitions is None:
            N = 768
            # self.asg_transitions = torch.FloatTensor(N, N).zero_()
            self.asg_transitions = []

        self.decoder = LexiconFreeDecoder(
            self.decoder_opts,
            self.lm,
            self.silence,
            self.blank,
            self.asg_transitions,   # Lexicon-free decoder with token-LM
        )

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
            # ...表示其他维度由计算机自行推断
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits
    
    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                    }
                    for result in nbest_results
                ]
            )
        return hypos

    def decode_1(self, emissions):
        repetition_penalty = 1.0
        temperature = 0.88
        topk = 0
        topp = 0.85

        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            cur_emissions = emissions[b]
            generated = []
            # 最多生成max_len个token
            for ii in range(T):
                next_token_logits = cur_emissions[ii, :]    # shape=(v)
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= repetition_penalty
                next_token_logits = next_token_logits / temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[self.tgt_dict.index("<unk>")] = -float('Inf')
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token.item() == self.tgt_dict.index("</s>"):  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(generated),
                        "score": 100.0,
                    }
                ]
            )
        return hypos


FairseqLMState = namedtuple("FairseqLMState", ["prefix", "incremental_state", "probs"])


class FairseqLM(LM):
    def __init__(self, dictionary, model):
        LM.__init__(self)
        self.dictionary = dictionary
        self.model = model
        self.unk = self.dictionary.unk()

        self.save_incremental = False  # this currently does not work properly
        self.max_cache = 20_000

        model.cuda()
        model.eval()
        model.make_generation_fast_()

        self.states = {}
        self.stateq = deque()

    def start(self, start_with_nothing):
        state = LMState()
        prefix = torch.LongTensor([[self.dictionary.eos()]])
        incremental_state = {} if self.save_incremental else None
        with torch.no_grad():
            res = self.model(prefix.cuda(), incremental_state=incremental_state)
            probs = self.model.get_normalized_probs(res, log_probs=True, sample=None)

        if incremental_state is not None:
            incremental_state = apply_to_sample(lambda x: x.cpu(), incremental_state)
        self.states[state] = FairseqLMState(
            prefix.numpy(), incremental_state, probs[0, -1].cpu().numpy()
        )
        self.stateq.append(state)

        return state

    def score(self, state: LMState, token_index: int, no_cache: bool = False):
        """
        Evaluate language model based on the current lm state and new word
        Parameters:
        -----------
        state: current lm state
        token_index: index of the word
                     (can be lexicon index then you should store inside LM the
                      mapping between indices of lexicon and lm, or lm index of a word)

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        curr_state = self.states[state]

        def trim_cache(targ_size):
            while len(self.stateq) > targ_size:
                rem_k = self.stateq.popleft()
                rem_st = self.states[rem_k]
                rem_st = FairseqLMState(rem_st.prefix, None, None)
                self.states[rem_k] = rem_st

        if curr_state.probs is None:
            new_incremental_state = (
                curr_state.incremental_state.copy()
                if curr_state.incremental_state is not None
                else None
            )
            with torch.no_grad():
                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cuda(), new_incremental_state
                    )
                elif self.save_incremental:
                    new_incremental_state = {}

                res = self.model(
                    torch.from_numpy(curr_state.prefix).cuda(),
                    incremental_state=new_incremental_state,
                )
                probs = self.model.get_normalized_probs(
                    res, log_probs=True, sample=None
                )

                if new_incremental_state is not None:
                    new_incremental_state = apply_to_sample(
                        lambda x: x.cpu(), new_incremental_state
                    )

                curr_state = FairseqLMState(
                    curr_state.prefix, new_incremental_state, probs[0, -1].cpu().numpy()
                )

            if not no_cache:
                self.states[state] = curr_state
                self.stateq.append(state)

        score = curr_state.probs[token_index].item()

        trim_cache(self.max_cache)

        outstate = state.child(token_index)
        if outstate not in self.states and not no_cache:
            prefix = np.concatenate(
                [curr_state.prefix, torch.LongTensor([[token_index]])], -1
            )
            incr_state = curr_state.incremental_state

            self.states[outstate] = FairseqLMState(prefix, incr_state, None)

        if token_index == self.unk:
            score = float("-inf")

        return outstate, score

    def finish(self, state: LMState):
        """
        Evaluate eos for language model based on the current lm state

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        return self.score(state, self.dictionary.eos())

    def empty_cache(self):
        self.states = {}
        self.stateq = deque()
        gc.collect()


class W2lFairseqLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.silence = tgt_dict.bos()

        self.unit_lm = getattr(args, "unit_lm", False)

        self.lexicon = load_words(args.lexicon) if args.lexicon else None
        self.idx_to_wrd = {}

        checkpoint = torch.load(args.kenlm_model, map_location="cpu")
        lm_args = checkpoint["args"]
        lm_args.data = osp.dirname(args.kenlm_model)
        print(lm_args)
        task = tasks.setup_task(lm_args)
        model = task.build_model(lm_args)
        model.load_state_dict(checkpoint["model"], strict=False)

        self.trie = Trie(self.vocab_size, self.silence)

        self.word_dict = task.dictionary
        self.unk_word = self.word_dict.unk()
        self.lm = FairseqLM(self.word_dict, model)

        self.decoder_opts = DecoderOptions(
            args.beam,
            int(getattr(args, "beam_size_token", len(tgt_dict))),
            args.beam_threshold,
            args.lm_weight,
            args.word_score,
            args.unk_weight,
            args.sil_weight,
            0,
            False,
            self.criterion_type,
        )

        if self.lexicon:
            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                if self.unit_lm:
                    word_idx = i
                    self.idx_to_wrd[i] = word
                    score = 0
                else:
                    word_idx = self.word_dict.index(word)
                    _, score = self.lm.score(start_state, word_idx, no_cache=True)

                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert (
                        tgt_dict.unk() not in spelling_idxs
                    ), f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                [],
                self.unit_lm,
            )
        else:
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
            )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []

        def idx_to_word(idx):
            if self.unit_lm:
                return self.idx_to_wrd[idx]
            else:
                return self.word_dict[idx]

        def make_hypo(result):
            hypo = {"tokens": self.get_tokens(result.tokens), "score": result.score}
            if self.lexicon:
                hypo["words"] = [idx_to_word(x) for x in result.words if x >= 0]
            return hypo

        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append([make_hypo(result) for result in nbest_results])
            self.lm.empty_cache()

        return hypos
