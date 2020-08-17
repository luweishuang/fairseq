#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="output/zh/train.tsv")
    parser.add_argument("--output-dir", default="output/zh")
    parser.add_argument("--output-name", default="train")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}

    with open(args.tsv, "r") as tsv, open(
            os.path.join(args.output_dir, args.output_name + ".ltr.txt"), "w"
    ) as ltr_out, open(
        os.path.join(args.output_dir, args.output_name + ".wrd.txt"), "w"
    ) as wrd_out:
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
            l1 = transcriptions[dir][part].replace(" ", "")
            print(transcriptions[dir][part], file=wrd_out)
            print(
                " ".join(list(transcriptions[dir][part].replace(" ", "|"))),
                file=ltr_out,
            )


if __name__ == "__main__":
    main()
