#!/bin/bash

dev=0

if [ "$#" -gt 0 ]; then
    dev="$1"
fi

poetry run python vmse2kv3/detectorv2.py \
        --input-device "$dev" \
        --model models/ggml-base.bin \
