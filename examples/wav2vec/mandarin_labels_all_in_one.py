#!/usr/bin/env python3

import soundfile
import os
import re
import string


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
    src_trans_file = "output/all/test.txt"
    dataset_path = '/devdata/home/pishichan/code/asr/data/mandarin/'
    dataset_name = "test"
    dest_dir = "output/all"

    discard_cnt = 0
    all_cnt = 0
    with open(os.path.join(dest_dir, dataset_name + '.tsv'), 'w') as tsv_fw, open(os.path.join(dest_dir, dataset_name + '.ltr.txt'), 'w') as ltr_fw:
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

                char_list = []
                for cur_char in true_sentence:
                    char_list.append(cur_char)
                print(" ".join(char_list), file=ltr_fw)
    print("discard_cnt = %d, all_cnt = %d " % (discard_cnt, all_cnt))


if __name__ == "__main__":
    main()


'''
yitu-chat
discard_cnt = 97, all_cnt = 2021 
yitu-reverb
discard_cnt = 46, all_cnt = 976 
'''
