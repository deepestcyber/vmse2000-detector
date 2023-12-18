#!/usr/bin/env python
"""
This script is a modified version of
`whisper.cpp/models/convert-whisper-to-openvino.py`.

The major difference is that we can specify the audio context size to use
so that we can use OpenVINO and a smaller audio context for faster streaming
performance. If we don't limit the audio context *here* we will be stuck
with the given audio context size of 1500 elements.
"""

import argparse
import io
import torch
import os
from openvino.tools import mo
from openvino.runtime import serialize
import shutil
from typing import Optional, Union
from whisper import Whisper, ModelDimensions
from whisper import _MODELS, _ALIGNMENT_HEADS, _download

def convert_encoder(hparams, encoder, mname):
    encoder.eval()

    mel = torch.zeros((1, hparams.n_mels, hparams.n_audio_ctx*2))

    onnx_folder=os.path.join(os.path.dirname(__file__),"onnx_encoder")

    #create a directory to store the onnx model, and other collateral that is saved during onnx export procedure
    if not os.path.isdir(onnx_folder):
        os.makedirs(onnx_folder)

    onnx_path = os.path.join(onnx_folder, "whisper_encoder.onnx")

    torch.onnx.export(
        encoder,
        mel,
        onnx_path,
        input_names=["mel"],
        output_names=["output_features"]
    )

    # use model optimizer to convert onnx to OpenVINO IR format
    encoder_model = mo.convert_model(onnx_path, compress_to_fp16=True)
    serialize(encoder_model, xml_path=os.path.join(os.path.dirname(__file__),"ggml-" + mname + "-encoder-openvino.xml"))

    #cleanup
    if os.path.isdir(onnx_folder):
        shutil.rmtree(onnx_folder)


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
    audio_context: Optional[int] = None,
) -> Whisper:
    """
    Load a Whisper ASR model but be able to specify the audio context
    of the model!

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory
    audio_context: int
        whether to limit the audio context to this size. Will speed up computation
        by a lot at a loss of quality. Cannot be higher than the model's
        audio context.

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])

    if audio_context is not None:
        dims.n_audio_ctx = audio_context
        checkpoint["model_state_dict"]['encoder.positional_embedding'] = checkpoint["model_state_dict"]['encoder.positional_embedding'][:audio_context]

    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model to convert (e.g. tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3)", required=True)
    parser.add_argument('--audio-context', type=int, default=None, help="Parameter to artificially limit the audio context to speed up inference. Leave None to keep audio context as is")
    args = parser.parse_args()

    if args.model not in ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"]:
        raise ValueError("Invalid model name")

    whisper = load_model(args.model, audio_context=args.audio_context).cpu()
    hparams = whisper.dims

    encoder = whisper.encoder

    # Convert encoder to onnx
    convert_encoder(hparams, encoder, args.model)
