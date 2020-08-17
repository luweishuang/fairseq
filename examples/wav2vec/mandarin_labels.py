#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""
import glob
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="output/zh")
    args = parser.parse_args()

    search_path = os.path.join(args.data_dir, '*.tsv')
    for cur_file in glob.iglob(search_path, recursive=True):
        tsv_file = os.path.realpath(cur_file)
        output_name = os.path.basename(cur_file).replace(".tsv", "")
        transcriptions = {}
        with open(tsv_file, "r") as tsv, open(os.path.join(args.data_dir, output_name + ".ltr.txt"), "w") as ltr_out,\
                open(os.path.join(args.data_dir, output_name + ".wrd.txt"), "w") as wrd_out:
            root = next(tsv).strip()
            for line in tsv:
                line = line.strip()
                dir = os.path.dirname(line)
                if dir not in transcriptions:
                    path = os.path.join(root, os.path.dirname(dir), "trans.txt")
                    assert os.path.exists(path)
                    texts = {}
                    with open(path, "r") as trans_f:
                        for tline in trans_f:
                            items = tline.strip().split("<--->")
                            char_list = []
                            for cur_char in items[1:]:
                                char_list.append(cur_char)
                            texts[items[0]] = " ".join(char_list)
                    transcriptions[dir] = texts
                part = os.path.basename(line).split(".")[0]
                assert part in transcriptions[dir]
                print(" ".join(list(transcriptions[dir][part].replace(" ", "|"))), file=wrd_out)
                print(
                    " ".join(list(transcriptions[dir][part].replace(" ", "|"))),
                    file=ltr_out,
                )


def write_token_2_dict():
    tkn_file = "output/zh/tokens.txt"
    dict_file = "output/zh/dict.ltr.txt"
    cnt = 94802
    with open(tkn_file, "r") as tkn_fr, open(dict_file, "w") as dict_fw:
        for line in tkn_fr:
            wrt_line = line.strip() + " " + str(cnt) + "\n"
            dict_fw.write(wrt_line)


if __name__ == "__main__":
    # write_token_2_dict()
    main()
