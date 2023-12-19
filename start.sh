#!/bin/bash

# save the cli arguments as the execution of the source statement
# clears the parameters.
args="$@"
readonly args

source openvino/l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64/setupvars.sh
set -- "$args"

dev=0

if [ "$#" -gt 0 ]; then
    dev="$1"
fi

poetry run python vmse2kv3/detector.py \
        --device "$dev" \
        --model pywhispercpp/extern/whispercpp/models/ggml-base.bin \
        --audio-ctx 512
