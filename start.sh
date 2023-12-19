#!/bin/bash

source openvino/l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64/setupvars.sh

poetry run python vmse2kv3/detector.py \
        --device 0 \
        --model pywhispercpp/extern/whispercpp/models/ggml-base.bin \
        --audio-ctx 512
