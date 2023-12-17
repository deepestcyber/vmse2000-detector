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


### Notes on fixing python version

Normally the repository would include a `.python-version` file that is
telling poetry (and pyenv) to use a specific python version. Sadly though
this is not possible with bazel or at least I haven't found a way.
Below are the things I found, it builds but the binary is still built for
3.9 instead of 3.10. So I decided to use whatever bazel finds on the system
(3.9 on raspbian).

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

### Optimizations during compilation

The bazel run compiles all the binary stuff which only contains optimizations
for `x86_64` (ssl/avx2) and not for `aarm64` (+NEON). We can change this by
modifying the `BUILD.bazel` file.

Relevant compile flags are `-march=native -mcpu=native -mtune=native -mfpu=neon-vfpv4`


## Model setup

Models are stored in `~/.local/share/whispercpp/`.

TODO: document which models to download and/or what to do with them
