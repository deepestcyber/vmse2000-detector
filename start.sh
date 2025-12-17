#!/bin/bash

dev=0

if [ "$#" -gt 0 ]; then
    dev="$1"
fi


source ~/envs/vmse/bin/activate
source openvino/openvino_toolkit_ubuntu22_2025.4.0.20398.8fdad55727d_arm64/setupvars.sh

# This is very specific and assumes that we applied the asoundrc of
# ./vmse2kv3/assets/asoundrc. It sets the recording volume of the ps3 eye cam
# to the given value (max=255). This roughly corresponds to
# value/255 * 30dB gain.
if arecord -L | grep -q CameraB409241; then
    amixer -c CameraB409241 cset numid=3 220
else
    echo "Warning: Not setting volume since we didn't find the ps3eye"
    echo "recording device (i.e. the underlying device name)."
fi

# small is still a bit too slow - would need a bit more context i guess
if false; then
python vmse2kv3/detector_v2.py \
        -ind "$dev" \
        -m models/ggml-smallctx768.bin \
       -bd 150 \
       -qt 16 \
       -actx 768 \
        vmse2kv3/assets/swear_words.txt
fi

python vmse2kv3/detector_v2.py \
        -ind "$dev" \
        -m models/ggml-basectx768.bin \
	-bd 150 \
	-qt 16 \
	-actx 768 \
        vmse2kv3/assets/swear_words.txt
