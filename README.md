# Setup

## Known issues

- some words like 'Nutte' and 'Schlampe' are not well processed
    **needs re-evaluation**
- having a virtual microphone and playing back recorded audios (via
  the same microphone than live streaming) seems to improve recognition.
  Maybe the recorded samples / the virtual mic have a higher sample rate?
    **needs re-evaluation**
- it may be worthwhile to, instead of using a fixed `k` to use a small
  value at the beginning (~100) and then get broader and broader so that
  we don't drop candidates too quickly.
- more methodical testing is needed
  * a few samples with good quality and known contents are needed
    (clean + unclean)
  * small benchmark script would help to run tests automatically

## Contents

- detector v1 script (based on python whisper.cpp from 2023 using bazel & hacks)
- detector v2 script based on maintained pywhispercpp
- direct dependencies (pinned; whispercpp python bindings)

## Installation

Setup venv, install requirements and build whisper.cpp python bindings:

    python -m venv <envdir>
    source activate <envdir>/bin/activate

    pip install .

    # initialize pywhispercpp repo
    (cd pywhispercpp; git submodule update --init --recursive)

    pip install ./pywhispercpp

**Important**: If you want fast whisper.cpp using OpenVINO (you do!) then
we need to do extra steps. If you don't want that, you can go to the next
section.

### Installing pywhispercpp + OpenVINO

TODO take from raspi

## Downloading the model

For the model you can go to `pywhispercpp/whispercpp/models/` and run the
download script, e.g.

```bash
pywhispercpp/whisper.cpp/models/download-ggml-model.sh base
```

The "base" model is recommended for a Raspberry Pi 5.


### Optimizations

#### OpenVINO

We may want to squeeze all the performance we can get out of our compute
platform (currently: Raspberry Pi 4). OpenVINO is a good candidate for this.

I focus on aarch64 for this step - it should be easier when using x86.
The main problem is actually the python whispercpp integration and bazel,
of course.

First, download the [2023.2.0 release](https://github.com/openvinotoolkit/openvino/releases/tag/2023.2.0)
of OpenVINO (for aarch64).

```bash
mkdir openvino
cd openvino
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/linux/l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64.tgz
tar -xf l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64.tgz
```

**Important:** The path of the unpacked directory should now be
`/home/pi/code/v3/openvino/l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64/`.
This is important because the modifications to the bazel build env of pywhispercpp
are relying on this information.

The submodule of pywhispercpp already points to the fork that has the
necessary bazel changes, you need to change to that branch, though.
Of course, a rebuild is also necessary.

```bash
cd pywhispercpp
git switch feature/openvino
./tools/bazel build //:whispercpp_wheel
```

After that you need to uninstall and reinstall the resulting wheel like
in the initial installation:

```bash
poetry run pip uninstall whispercpp
poetry run pip install pywhispercpp/bazel-bin/whispercpp-0.0.17-cp310-cp310-manylinux2014_aarch64.whl
```

Now you should be able to run models with OpenVINO encoder support.
You may need to set the `LD_LIBRARY_PATH` appropriately:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/openvino/l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64/runtime/lib/:$PWD/openvino/l_openvino_toolkit_debian9_2023.2.0.13089.cfd42bd2cb0_arm64/runtime/3rdparty/tbb/lib/"
```

Note that if you want to use smaller audio context sizes you *must* convert
the model with the modified openvino conversion script
[`convert-whisper-to-openvino-audioctx.py`](./scripts/convert-whisper-to-openvino-audioctx.py)
which allows for the `--audio-context` parameter:

```bash
poetry shell
cd pywhispercpp/extern/whispercpp/models
cp ../../../scripts/convert-whisper-to-openvino-audioctx.py .
[if not already: pip install -r openvino-conversion-requirements.txt]
python convert-whisper-to-openvino-audioctx.py --model base --audio-context 512
```

You can then deploy/use the resulting `ggml-base-encoder-openvino.{xml,bin}`
files by simply placing them in the models directory of your choice and
selecting `ggml-base` as your model.
