#!/usr/bin/env python3

import os
import re
import string
import jieba
import glob
from pypinyin import pinyin, lazy_pinyin, Style, load_phrases_dict, load_single_dict

load_phrases_dict({'各地': [['gè'], ['dì']]})
load_phrases_dict({'还款': [['huán'], ['kuǎn']]})
# load_phrases_dict({'的话': [['de'], ['huà']]})
# load_phrases_dict({'如果': [['rú'], ['guǒ']]})
# load_phrases_dict({'玩儿': [['wán'], ['ér']]})
# load_phrases_dict({'或是': [['huò'], ['shì']]})
# load_phrases_dict({'隔壁': [['gé'], ['bì']]})

load_single_dict({ord('还'): 'hái, huán'})

alpha_char = string.punctuation + u"abcdefghijklmnopqrstuvwxyz| "
pattern_alpha = re.compile('[%s]' % re.escape(alpha_char))

illegal_char = string.punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：・·「」『』〈〉／－□ '
pattern = re.compile('[%s]' % re.escape(illegal_char))

tkn_dict = []
with open("output/zh/tokens.txt", "r") as fr_tkn:
    for line in fr_tkn:
        tkn_dict.append(line.strip())
tkn_dict = list(set(tkn_dict))
print("len(tkn_dict) = ", len(tkn_dict))


def char_in_dict(sentence):
    for ch in sentence:
        if ch in tkn_dict:
            continue
        else:
            return False
    return True


def main():
    src_trans_file = "output/all20_pyl/test.txt"
    dataset_name = "test"
    dest_dir = "output/all20_pyl"
    discard_cnt = 0
    all_cnt = 0

    wav_trans_dict = {}
    with open(src_trans_file, "r") as fr:
        for line in fr:
            wav_name, cur_trans = line.strip().split("<--->")
            wav_trans_dict[wav_name + ".wav"] = cur_trans

    with open(os.path.join(dest_dir, dataset_name + '.tsv'), 'r') as tsv_fr, open(os.path.join(dest_dir, dataset_name + '.ltr.txt'), 'w') as ltr_fw, open(os.path.join(dest_dir, dataset_name + '.wrd.txt'), 'w') as wrd_fw:
        root = next(tsv_fr).strip()
        for line in tsv_fr:
            all_cnt += 1
            wav_name, wav_len = line.strip().split("\t")
            cur_trans = wav_trans_dict.get(wav_name, "")
            true_sentence = pattern.sub(u'', cur_trans)
            if not char_in_dict(true_sentence):
                print("cur %s trans is not purely dict char, as %s" % (wav_name, true_sentence))
                discard_cnt += 1
                continue

            # pieces = encode_pieces(sp, true_sentence, sample=False)
            seg_list = jieba.lcut(true_sentence)
            seg_pos_list = []
            cur_pos = 0
            wrd_str = ""
            for cur_seg in seg_list:
                cur_pos += len(cur_seg)
                seg_pos_list.append(cur_pos)
                wrd_str += cur_seg + " "

            py_list = pinyin(true_sentence, style=Style.TONE3, heteronym=False)

            ltr_str = ""
            ltr_index = 0
            for wrd in py_list:
                cur_wrd = wrd[0].replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                ltr_index += 1
                if ltr_index in seg_pos_list:
                    ltr_str += cur_wrd + " | "
                else:
                    ltr_str += cur_wrd + " "
            print(ltr_str[:-1], file=ltr_fw)
            print(wrd_str[:-1], file=wrd_fw)
    print("discard_cnt = %d, all_cnt = %d " % (discard_cnt, all_cnt))


def generate_lexicon_file():
    basedir = '/devdata/home/pishichan/code/asr/data/mandarin/'
    # basedir = "/home/psc/Desktop/code/asr/data/mandarin/"
    trans_list = []
    search_path = os.path.join(basedir, '*/*/*/' + "trans.txt")
    for fname in glob.iglob(search_path, recursive=True):
        trans_path = os.path.realpath(fname)
        with open(trans_path, 'r') as fr:
            for cur_line in fr:
                trans_list.append(cur_line.strip().split("<--->")[1])
    print("trans_cnt = %d " % len(trans_list))

    discard_cnt = 0
    lexicon_list = []
    for cur_trans in trans_list:
        true_sentence = pattern.sub(u'', cur_trans)
        if not char_in_dict(true_sentence):
            print("cur trans is not purely dict char, as %s" % true_sentence)
            discard_cnt += 1
            continue
        seg_list = jieba.lcut(true_sentence)
        for cur_wrd in seg_list:
            if len(cur_wrd) == 1:
                py_list = pinyin(cur_wrd, style=Style.TONE3,  heteronym=True)
                ll_all = []
                for ll in py_list[0]:
                    l1 = ll.replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                    if l1 not in ll_all:
                        ll_all.append(l1)
                        w_str = cur_wrd + "\t" + l1 + " |"
                        lexicon_list.append(w_str)
            else:
                py_list = pinyin(cur_wrd, style=Style.TONE3, heteronym=False)
                w_str = cur_wrd + "\t"
                for cur_py in py_list:
                    l1 = cur_py[0].replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                    w_str += l1 + " "
                lexicon_list.append(w_str + "|")
    lexicon_file = "output/all20_pyl/lexicon_pyl.txt"
    lexicon_list = list(set(lexicon_list))
    lexicon_list.sort()
    with open(lexicon_file, "w") as fw:
        for w_str in lexicon_list:
            fw.write(w_str + "\n")


if __name__ == "__main__":
    # main()
    generate_lexicon_file()