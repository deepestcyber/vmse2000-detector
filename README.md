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

if installing on an ARM64 architecture the path will be, of course, different, e.g.:

    pywhispercpp/bazel-bin/whispercpp-0.0.17-cp310-cp310-manylinux2014_aarch64.whl

The runtime is now safe.
You can now proceed downloading the model.

## Downloading the model

This project uses the whisper.cpp submodule provided by the python wrapper
as root for all things whisper - including the models.

For the base model you can go to `pywhispercpp/extern/whispercpp/models/`
and run the download script, e.g.

```bash
cd pywhispercpp/extern/whispercpp/models/
./download-ggml-model.sh base
```


### Notes on fixing python version

Normally the repository would include a `.python-version` file that is
telling poetry (and pyenv) to use a specific python version. Sadly though
this is not possible with bazel or at least I haven't found a way.
Below are the things I found, it builds but the binary is still built for
3.9 instead of 3.10. So I decided to use whatever bazel finds on the system
(3.9 on raspbian). Maybe it is possible doing this when the poetry env
is activated, that's the one thing I didn't try.

#### How to (not) use pyenv with bazel

Note that if bazel is building `py39` images (even though the python version
is set via pyenv/`.python-version` file to 3.10) it may be that bazel uses
the system python instead. check, e.g., via `sh -lc 'python --version'`.
If it does, activate pyenv in your `~/.profile` file so that your `/bin/sh`
also uses pyenv.

If that does not help, well, pull your sleeves up and get ready to mess
with bazel. Basically: add a new python toolchain for 3.10 (using the fact
that a pyenv python is available in PATH under `python3.10`). Use the following
diff and the following command:

    ln -s `which python3.10` python3.10

diff:

```diff
diff --git a/BUILD.bazel b/BUILD.bazel
index 3dec535..e7f585d 100644
--- a/BUILD.bazel
+++ b/BUILD.bazel
@@ -7,9 +7,32 @@ load("@bazel_skylib//lib:selects.bzl", "selects")
 load("@rules_python//python:packaging.bzl", "py_wheel")
 load("@com_github_bentoml_plugins//rules/py:packaging.bzl", "py_package")
 load("@python_abi//:abi.bzl", "python_abi")
+load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

 package(default_visibility = ["//visibility:public"])

+
+py_runtime(
+    name = "python-pyenv",
+    interpreter = "python3.10",
+    python_version = "PY3",
+)
+
+
+py_runtime_pair(
+    name = "python-pyenv-runtime-pair",
+    py2_runtime = None,
+    py3_runtime = ":python-pyenv",
+)
+
+toolchain(
+    name = "python-pyenv-toolchain",
+    toolchain = ":python-pyenv-runtime-pair",
+    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
+)
+
+
+
 filegroup(
     name = "pyproject",
     srcs = ["pyproject.toml"],
```

now run the build like this:

    ./tools/bazel build --extra_toolchains=//:python-pyenv-toolchain //:whispercpp_wheel

### Optimizations

#### Compile flags

The bazel run compiles all the binary stuff which only contains optimizations
for `x86_64` (ssl/avx2) and not for `aarm64`. We can change this by
modifying the `BUILD.bazel` file.

Relevant compile flags are `-march=native -mcpu=native -mtune=native`.
Note that we don't explicitly add NEON since that is included in -O3 for
aarch64 anyway.

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
