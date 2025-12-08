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

    sudo apt install libsdl2-mixer-dev libportaudio2

    python -m venv <envdir>
    source activate <envdir>/bin/activate

    pip install -r requirements.txt

    # initialize pywhispercpp repo
    (cd pywhispercpp; git submodule update --init --recursive)

    pip install ./pywhispercpp

**Important**: If you want fast whisper.cpp using OpenVINO (you do!) then
we need to do extra steps. If you don't want that, you can go to the next
section.

### Installing pywhispercpp + OpenVINO

Download the OpenVINO SDK e.g. via https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-linux.html.

```
pi@vmse2000:~/code/v3 $ ls -l openvino/
total 75M
-rw-r--r--  1 pi pi  37M Dec  1 11:03 openvino_toolkit_ubuntu22_2025.4.0.20398.8fdad55727d_arm64.tgz
```

**IMPORTANT**: as of the time of writing this library will miss a crucial invocation
of `whisper_ctx_init_openvino_encoder`. This is not very hard to do, the patch can be found
at the end of this section. *Apply the patch after cloning and before installing*.
For your convenience, this repo uses a fork with the patch applied.

```
git clone https://github.com/absadiki/pywhispercpp.git

export WHISPER_OPENVINO=1
source openvino/openvino_toolkit*/setupvars.sh

pip install --force-reinstall ./pywhispercpp
pip install sounddevice~=0.4.6
```

Now that is not enough, of course. The built wheel will bundle the core OpenVINO library
but not all the other parts of the library. The core runtime will have an ELF rpath set,
a relative path where it will look for other modules of the library. There will be ni
other libraries and therefore we won't be able to parse XML files, leading to this error:

```
whisper_ctx_init_openvino_encoder_with_state: loading OpenVINO model from '/home/pi/code/v3/model
s/ggml-base-encoder-openvino.xml'
whisper_ctx_init_openvino_encoder_with_state: first run on a device may take a while ...
whisper_openvino_init: path_model = /home/pi/code/v3/models/ggml-base-encoder-openvino.xml, devic
e = CPU, cache_dir = /home/pi/code/v3/models/ggml-base-encoder-openvino-cache
in openvino encoder compile routine: exception: Exception from src/inference/src/cpp/core.cpp:97:
Exception from src/inference/src/model_reader.cpp:155:
Unable to read the model: /home/pi/code/v3/models/ggml-base-encoder-openvino.xml Please check tha
t model format: xml is supported and the model is correct. Available frontends:


whisper_ctx_init_openvino_encoder_with_state: failed to init OpenVINO encoder from '/home/pi/code
/v3/models/ggml-base-encoder-openvino.xml'
```

I found that the simplest remedy is to copy the runtime libraries to the python module's
library folder and replace the runtime library with the original.

1. Find the venv directory + pywhispercpp libs module
3. Copy stuff over
4. Symlink core library

Step 1

```
$ poetry run python -c 'import pywhispercpp; print(pywhispercpp)'
<module 'pywhispercpp' from '/home/pi/.cache/pypoetry/virtualenvs/vmse2kv3-d5r8ZGfy-py3.13/lib/python3.13/site-packages/pywhispercpp/__init__.py'>
```

So the libs will probably be at `/home/pi/.cache/pypoetry/virtualenvs/vmse2kv3-d5r8ZGfy-py3.13/lib/python3.13/site-packages/pywhispercpp.libs/`.

Copy all the libraries and replace linked runtime library:

```
cp openvino/openvino_toolkit_.../runtime/lib/aarch64/libopenvino* /home/pi/.cache/pypoetry/virtualenvs/vmse2kv3-d5r8ZGfy-py3.13/lib/python3.13/site-packages/pywhispercpp.libs/
cd /home/pi/.cache/pypoetry/virtualenvs/vmse2kv3-d5r8ZGfy-py3.13/lib/python3.13/site-packages/pywhispercpp.libs
ln -sf libopenvino.so.2025.4.0 libopenvino-f2621866.so.2025.4.0
```

**Important**: This is an example, `libopenvino-f26...` might be called differently.

This should run now!



## Downloading the model

For the model you can go to `pywhispercpp/whispercpp/models/` and run the
download script, e.g.

```bash
pywhispercpp/whisper.cpp/models/download-ggml-model.sh base
```

The "base" model is recommended for a Raspberry Pi 5.


## Patch for adding `whisper_ctx_init_openvino_encoder`

```diff
diff --git a/pywhispercpp/model.py b/pywhispercpp/model.py
index 39944f1..50ddc79 100644
--- a/pywhispercpp/model.py
+++ b/pywhispercpp/model.py
@@ -254,8 +254,10 @@ class Model:
         :return:
         """
         logger.info("Initializing the model ...")
+
         with utils.redirect_stderr(to=self.redirect_whispercpp_logs_to):
             self._ctx = pw.whisper_init_from_file(self.model_path)
+            pw.whisper_ctx_init_openvino_encoder(self._ctx)

     def _set_params(self, kwargs: dict) -> None:
         """
@@ -384,4 +386,4 @@ class Model:
         Free up resources
         :return: None
         """
-        pw.whisper_free(self._ctx)
\ No newline at end of file
+        pw.whisper_free(self._ctx)
diff --git a/src/main.cpp b/src/main.cpp
index 68c0ea5..2329397 100644
--- a/src/main.cpp
+++ b/src/main.cpp
@@ -226,6 +226,11 @@ int whisper_full_wrapper(
     return whisper_full(ctx_w->ptr, params, samples_ptr, n_samples);
 }

+int whisper_ctx_init_openvino_encoder_wrapper(struct whisper_context_wrapper * ctx_w) {
+    whisper_ctx_init_openvino_encoder(ctx_w->ptr, nullptr, "CPU", nullptr);
+    return 0;
+}
+
 int whisper_full_parallel_wrapper(
         struct whisper_context_wrapper * ctx_w,
         struct whisper_full_params   params,
@@ -666,6 +671,10 @@ PYBIND11_MODULE(_pywhispercpp, m) {
                                                                                 "This contains probabilities, timestamps, etc.");

     m.def("whisper_full_get_token_p", &whisper_full_get_token_p_wrapper, "Get the probability of the specified token in the specified segment.");
+
+    ////////////////////////////////////////////////////////////////////////////
+
+    m.def("whisper_ctx_init_openvino_encoder", &whisper_ctx_init_openvino_encoder_wrapper, "Initialize OpenVINO encoder.");

     ////////////////////////////////////////////////////////////////////////////

```

