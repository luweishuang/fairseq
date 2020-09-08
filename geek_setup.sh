#! /bin/bash

# setup dataset for training wav2vec_ctc model
cd /input
mkdir -p data/asr/mandarin
cd /data/asr/mandarin
unzip -d /input/data/asr/mandarin all1.zip

cd /root/code/asr/fairseq/examples/wav2vec
mkdir -p output
