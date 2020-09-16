#!/usr/bin/env python3

import soundfile
import os
import re
import glob
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
    src_trans_file = "output/all20_pyw/dev.txt"
    dataset_name = "dev"
    dest_dir = "output/all20_pyw"
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

            py_list = pinyin(true_sentence, style=Style.TONE3, heteronym=False)
            ltr_str = ""
            wrd_str = true_sentence
            for wrd in py_list:
                cur_wrd = wrd[0].replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                ltr_str += cur_wrd + " "
            if '' == pattern_alpha.sub(u'', ltr_str):
                print(ltr_str[:-1], file=ltr_fw)
                print(wrd_str, file=wrd_fw)
            else:
                print("contain illeagle char, ", ltr_str[:-1], wrd_str)
    print("discard_cnt = %d, all_cnt = %d " % (discard_cnt, all_cnt))


def write_dict_file():
    dict_list = []
    base_dir = "output/all20_pyw"

    for cur_zh in tkn_dict:
        py_list = pinyin(cur_zh, style=Style.TONE3, heteronym=True)
        for wrd in py_list[0]:
            # cur_wrd = wrd.replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
            cur_wrd = wrd.replace(" ", "")
            dict_list.append(cur_wrd)
    dict_list = list(set(dict_list))
    print("len(dict_list) = ", len(dict_list))
    # process_names = ["train", "test", "dev"]
    # for cur_pro in process_names:
    #     cur_file = os.path.join(base_dir, cur_pro + ".ltr.txt")
    #     with open(cur_file, "r") as fr:
    #         for line in fr:
    #             dict_list.extend(line.strip().split(" "))
    # dict_list = list(set(dict_list))
    # print("len(dict_list) = ", len(dict_list))
    dict_list.sort()
    dict_file = os.path.join(base_dir, "dict.ltr.txt")
    with open(dict_file, "w") as fw:
        for cur_t in dict_list:
            fw.write(cur_t + " 636\n")


if __name__ == "__main__":
    # main()
    # write_dict_file()

    base_dir = "output/all20_pyw"
    process_names = ["train", "test", "dev"]
    for cur_pro in process_names:
        cur_file = os.path.join(base_dir, cur_pro + ".wrd.txt")
        cur_save_file = os.path.join(base_dir, cur_pro + "_new.wrd.txt")
        with open(cur_file, "r") as fr, open(cur_save_file, "w") as fw:
            for line in fr:
                new_str = ""
                for cur_w in line.strip():
                    new_str += cur_w + " "
                fw.write(new_str[:-1] + "\n")

    print("done ")