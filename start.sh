#!/bin/bash

dev=0

if [ "$#" -gt 0 ]; then
    dev="$1"
fi


source ~/envs/vmse/bin/activate
source openvino/openvino_toolkit_ubuntu22_2025.4.0.20398.8fdad55727d_arm64/setupvars.sh

python vmse2kv3/detector_v2.py \
        -ind "$dev" \
        -m models/ggml-basectx768.bin \
	-bd 150 \
	-qt 16 \
	-actx 768 \
        vmse2kv3/assets/swear_words.txt
