#! /bin/bash

# setup dataset for training wav2vec_ctc model
cd /input
mkdir -p data/asr/mandarin
cd /data/asr/mandarin
unzip -d /input/data/asr/mandarin all1.zip

cd /root
mkdir -p code/asr
cd /root/code/asr
cp -r /data/code/asr/fairseq .
cp -r /data/code/asr/apex .

# install apex and fairseq by hand
#cd apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./

#cd ../fairseq
#pip install --editable ./ -i https://pypi.doubanio.com/simple



