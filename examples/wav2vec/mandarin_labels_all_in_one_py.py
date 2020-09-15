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


def generate_lexicon_file():
    basedir = "/home/psc/Desktop/code/asr/data/mandarin/"
    wrd_list_all = []
    discard_cnt = 0
    search_path = os.path.join(basedir, '*/*/*/' + "trans.txt")
    for fname in glob.iglob(search_path, recursive=True):
        trans_path = os.path.realpath(fname)
        with open(trans_path, 'r') as fr:
            for cur_line in fr:
                wav_name, cur_trans = cur_line.strip().split("<--->")
                true_sentence = pattern.sub(u'', cur_trans)
                if not char_in_dict(true_sentence):
                    print("cur %s trans is not purely dict char, as %s" % (wav_name, true_sentence))
                    discard_cnt += 1
                    continue

                py_list = pinyin(true_sentence, style=Style.TONE3, heteronym=False)
                for wrd in py_list:
                    cur_wrd = wrd[0].replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                    wrd_list_all.append(cur_wrd)

    wrd_list_all = list(set(wrd_list_all))
    wrd_list_all.sort()
    print("len(wrd_list_all) = ", len(wrd_list_all))

    lexicon_file = "lexicon.txt"
    with open(lexicon_file, "w") as fw:
        for cur_wrd in wrd_list_all:
            w_str = cur_wrd + "\t"
            for cur_char in cur_wrd:
                w_str += cur_char + " "
            fw.write(w_str[:-1] + " |\n")


def generate_lexicon_simple():
    base_dir = "output/all20_py"
    wrd_list_all = []
    process_names = ["train", "test", "dev"]
    for cur_name in process_names:
        cur_file = os.path.join(base_dir, cur_name + ".wrd.txt")
        with open(cur_file, "r") as fr:
            for line in fr:
                wrd_list_all += line.strip().split(" ")

    wrd_list_all = list(set(wrd_list_all))
    wrd_list_all.sort()
    print("len(wrd_list_all) = ", len(wrd_list_all))

    lexicon_file = os.path.join(base_dir, "lexicon.txt")
    with open(lexicon_file, "w") as fw:
        for cur_wrd in wrd_list_all:
            w_str = cur_wrd + "\t"
            for cur_char in cur_wrd:
                w_str += cur_char + " "
            fw.write(w_str[:-1] + " |\n")


def generate_py_test():
    src_file = "output/test/yitu-chat.ltr"
    dst_file = src_file.replace(".ltr", "_py.ltr")
    discard_cnt = 0
    with open(src_file, "r") as fr, open(dst_file, "w") as fw:
        for line in fr:
            cur_trans = line.strip().replace(" ", "")
            true_sentence = pattern.sub(u'', cur_trans)
            if not char_in_dict(true_sentence):
                print("cur trans is not purely dict char, as %s" % (true_sentence))
                discard_cnt += 1
                continue
            wrd_str = ""
            py_list = pinyin(true_sentence, style=Style.TONE3, heteronym=False)
            for wrd in py_list:
                cur_wrd = wrd[0].replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(" ", "")
                wrd_str += cur_wrd + " "
            fw.write(wrd_str[:-1] + "\n")


if __name__ == "__main__":
    # main()
    # generate_lexicon_file()
    # generate_lexicon_simple()
    generate_py_test()

