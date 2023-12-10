# Setup

## Install python whispercpp bindings

    poetry install
    poetry shell

    # initialize pywhispercpp repo
    cd pywhispercpp
    git submodule update --init --recursive

this possibly fails with the message

> fatal: Fetched in submodule path 'whispercpp/bindings/ios', but it did not contain c9d5095f0c64455b201f1cd0b547efcf093ee7c3. Direct fetching of that commit failed.

in that case simply remove the bindings from that repo by running

    cd extern/whisper.cpp
    git rm bindings/ios
    cd ../..
    git submodule update --init --recursive

continue installation

    sudo apt install libsdl2-mixer-dev
    ./tools/bazel build //:whispercpp_wheel

the wheel can then be installed

    cd ..
    poetry run pip install pywhispercpp/bazel-bin/whispercpp-0.0.17-cp310-cp310-manylinux2014_x86_64.whl

## Model setup

Models are stored in `~/.local/share/whispercpp/`.

TODO: document which models to download and/or what to do with them
