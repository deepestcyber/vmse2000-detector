"""
This script is a modified version of
`whisper.cpp/models/convert-whisper-to-openvino.py`.

The major difference is that we can specify the audio context size to use
so that we can use OpenVINO and a smaller audio context for faster streaming
performance. If we don't limit the audio context *here* we will be stuck
with the given audio context size of 1500 elements.
"""
import argparse
import torch
from whisper import load_model
import os
from openvino.tools import mo
from openvino.frontend import FrontEndManager
from openvino.runtime import serialize
import shutil

def convert_encoder(hparams, encoder, mname, audio_ctx=1500):
    encoder.eval()

    mel = torch.zeros((1, hparams.n_mels, audio_ctx*2))

    encoder.positional_embedding = encoder.positional_embedding[:audio_ctx]

    onnx_folder = os.path.join(os.path.dirname(__file__), "onnx_encoder")

    #create a directory to store the onnx model, and other collateral that is saved during onnx export procedure
    if not os.path.isdir(onnx_folder):
        os.makedirs(onnx_folder)

    onnx_path = os.path.join(onnx_folder, "whisper_encoder.onnx")

    # Export the PyTorch model to ONNX
    torch.onnx.export(
        encoder,
        mel,
        onnx_path,
        input_names=["mel"],
        output_names=["output_features"]
    )

    # Convert ONNX to OpenVINO IR format using the frontend
    fem = FrontEndManager()
    onnx_fe = fem.load_by_framework("onnx")
    onnx_model = onnx_fe.load(onnx_path)
    ov_model = onnx_fe.convert(onnx_model)

    # Serialize the OpenVINO model to XML and BIN files
    serialize(ov_model, xml_path=os.path.join(os.path.dirname(__file__), "ggml-" + mname + "-encoder-openvino.xml"))

    # Cleanup
    if os.path.isdir(onnx_folder):
        shutil.rmtree(onnx_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model to convert (e.g. tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, large-v3-turbo)", required=True)
    parser.add_argument("--audio-ctx", type=int, default=1500, help="audio context size, useful when using a non-default -audio-ctx arg for whisper")
    args = parser.parse_args()

    if args.model not in ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]:
        raise ValueError("Invalid model name")

    whisper = load_model(args.model).cpu()
    hparams = whisper.dims

    encoder = whisper.encoder

    # Convert encoder to onnx
    convert_encoder(hparams, encoder, args.model, audio_ctx=args.audio_ctx)
