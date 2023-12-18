import numpy as np

from termcolor import colored

import whispercpp
from whispercpp import Whisper, api, audio
from typing import Any, List, Optional


transcript: List[str] = []

def stream_transcribe(
    whisper: Whisper,
    swear_words: list[str],
    firmware_address: tuple[str,str],
    device_id: int = 0,
    sample_rate: Optional[int] = None,
    **kwargs: Any,
) -> list[str]:
    """Streaming transcription from microphone. Note that this function is blocking.

    Args:
        length_ms (int, optional): Length of audio to transcribe in milliseconds. Defaults to 10000.
        device_id (int, optional): Device ID of the microphone. Defaults to 0. Use
                                   ``whispercpp.utils.available_audio_devices()`` to list all available devices.
        sample_rate: (int, optional): Sample rate to be passed to Whisper.

    Returns:
        A generator of all transcripted text from given audio device.
    """
    if sample_rate is None:
        sample_rate = api.SAMPLE_RATE
    if "length_ms" not in kwargs:
        kwargs["length_ms"] = 5000
    if "step_ms" not in kwargs:
        kwargs["step_ms"] = 700

    if kwargs["step_ms"] < 500:
        raise ValueError("step_ms must be >= 500")

    ac = audio.AudioCapture(kwargs["length_ms"])
    if not ac.init_device(device_id, sample_rate):
        raise RuntimeError("Failed to initialize audio capture device.")

    whisper.params.on_new_segment(
        _store_transcript_handler,
        dict(
            firmware_address=firmware_address,
            swear_words=swear_words,
        ),
    )

    try:
        ac.stream_transcribe(whisper.context, whisper.params, **kwargs)
    except KeyboardInterrupt:
        # handled from C++
        pass
    return transcript


def token_to_str(ctx: api.Context, tid: int) -> str:
    # this exists since the given ctx.token_to_str cannot deal with
    # all unicode characters apparently :)
    return str(ctx.token_to_bytes(tid), 'utf8', 'ignore')


def colored_token(token_str, token_prob):
    colors = ['red', 'light_red', 'yellow', 'light_yellow', 'green']
    idx = int(token_prob / (1/len(colors)))

    return colored(token_str, colors[idx])


def send_word(address, word):
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(word, address)


def process_merged_tokens(firmware_address, tokens, swear_words):
    for token, p in tokens:
        print(f"'{colored_token(token, p)}'", end='')
    print()

    for token, p in tokens:
        word = token.lower().strip()
        if word in swear_words:
            print("!" * 30, f"swear word detected (p={p})")
            send_word(firmware_address, word)


def _store_transcript_handler(
    ctx: api.Context, n_new: int, kwargs,
):
    import string

    swear_words: list[str] = kwargs['swear_words']
    firmware_address: tuple[str,str] = kwargs['firmware_address']
    segment = ctx.full_n_segments() - n_new

    # collected tokens over all segments
    all_tokens = []

    # decoder segments contain a bunch of tokens
    for segment in range(ctx.full_n_segments()):
        num_tokens = ctx.full_n_tokens(segment)

#        print('decoded tokens:', [ctx.token_to_str(ctx.full_get_token_id(segment, i)) for i in range(num_tokens)])

        def merge_tokens(merge_token):
            token = ''.join(merge_token[0])
            p = np.mean(merge_token[1])
            return token, p

        merge_token = None

        for token_idx in range(num_tokens):
            token_data = ctx.full_get_token_data(segment, token_idx)
            token = token_to_str(ctx, token_data.id)

            print("debug:", colored_token("'"+token+"'", token_data.p))

            # we attempt to merge tokens that are likely to be part of a
            # larger word. the vocabulary seems to be ordered in such a way
            # that tokens may start with a whitespace but don't end with one.
            # this way we can just see if the next token starts with a
            # punctuation token in which case we stop merging.

            if merge_token is None:
                merge_token = ([token], [token_data.p])
                continue

            # this token is mergeable, collect it for merging
            if token[0] not in string.whitespace + string.punctuation:
                merge_token[0].append(token)
                merge_token[1].append(token_data.p)
                continue

            # this token is not mergeable so
            # - merging of the previous token is done
            # - merging of possible future tokens begins now
            all_tokens.append(merge_tokens(merge_token))
            merge_token = ([token], [token_data.p])


        # collect leftover merge tokens so they get processed as well.
        if merge_token:
            all_tokens.append(merge_tokens(merge_token))


        # final processing of tokens
        process_merged_tokens(firmware_address, all_tokens, swear_words)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--step', type=int, default=4000)
    parser.add_argument('--length', type=int, default=8000)
    parser.add_argument('--audio-ctx', type=int, default=512)
    parser.add_argument('--language', type=str, default='de')
    parser.add_argument('--model', type=str, default='pywhispercpp/extern/whispercpp/models/ggml-base.bin')
    parser.add_argument('--host', type=str, default='localhost',
        help='address to send detected words to',
    )
    parser.add_argument('--port', type=int, default=1800,
        help='port to send detected words to',
    )

    args = parser.parse_args()

    swear_words = [
        "fuck",
        "ass",
        "shit",
        "eleven",
        "scheiß",
        "scheiße",
        "hölle",
        "fick",
        "kack",
        "kacke",
        "kackscheiß",
        "hure",
        "bratze",
    ]

    print(whispercpp.utils.available_audio_devices())

    whisper = Whisper.from_pretrained(args.model)
    print(whisper.context.sys_info())

    stream_transcribe(
        whisper,
        swear_words,
        device_id=args.device,
        step_ms=args.step,
        length_ms=args.length,

        language=args.language,
        audio_ctx=args.audio_ctx,

        firmware_address=(args.host, args.port),
    )

