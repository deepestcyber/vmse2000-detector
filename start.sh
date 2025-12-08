#!/bin/bash

dev=0

if [ "$#" -gt 0 ]; then
    dev="$1"
fi

source ~/envs/vmse/bin/activate
python vmse2kv3/detector_v2.py \
        --input-device "$dev" \
        --model models/ggml-base.bin \
        vmse2kv3/assets/swear_words.txt
