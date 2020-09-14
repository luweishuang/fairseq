#!/usr/bin/env python3

import soundfile
import os
import re
import string
from pypinyin import pinyin, lazy_pinyin, Style

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
    src_trans_file = "output/all50_py/train.txt"
    dataset_path = '/devdata/home/pishichan/code/asr/data/mandarin'
    dataset_name = "train"
    dest_dir = "output/all50_py"
    discard_cnt = 0
    all_cnt = 0
    with open(os.path.join(dest_dir, dataset_name + '.tsv'), 'w') as tsv_fw, open(os.path.join(dest_dir, dataset_name + '.ltr.txt'), 'w') as ltr_fw, open(os.path.join(dest_dir, dataset_name + '.wrd.txt'), 'w') as wrd_fw:
        print(dataset_path, file=tsv_fw)
        with open(src_trans_file, "r") as fr:
            for line in fr:
                all_cnt += 1
                wav_name, cur_trans = line.strip().split("<--->")

                true_sentence = pattern.sub(u'', cur_trans)
                if not char_in_dict(true_sentence):
                    print("cur %s trans is not purely dict char, as %s" % (wav_name, true_sentence))
                    discard_cnt += 1
                    continue

                audio_name = wav_name + ".wav"
                audio_path = os.path.join(dataset_path, audio_name)

                frames = soundfile.info(audio_path).frames
                print('{}\t{}'.format(audio_name, frames), file=tsv_fw)

                py_list = pinyin(true_sentence, style=Style.TONE3, heteronym=False)
                ltr_str = ""
                wrd_str = ""
                for wrd in py_list:
                    cur_wrd = wrd[0].replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                    ltr_str += " ".join(list(cur_wrd)) + " | "
                    wrd_str += cur_wrd + " "
                if '' == pattern_alpha.sub(u'', ltr_str) and '' == pattern_alpha.sub(u'', wrd_str):
                    print(ltr_str[:-1], file=ltr_fw)
                    print(wrd_str[:-1], file=wrd_fw)
                else:
                    print("contain illeagle char, ", ltr_str[:-1], wrd_str[:-1])
    print("discard_cnt = %d, all_cnt = %d " % (discard_cnt, all_cnt))


if __name__ == "__main__":
    main()

