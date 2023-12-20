import string
from typing import Any, List, Optional

import numpy as np
from termcolor import colored
import whispercpp
from whispercpp import Whisper, api, audio


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

    whisper.params.on_new_logits(
        _logits_callback,
        dict(swear_words=swear_words),
    )

    try:
        ac.stream_transcribe(whisper.context, whisper.params, **kwargs)
    except KeyboardInterrupt:
        # handled from C++
        pass
    return transcript


def tokenize(text: str):
    from vmse2kv3.openai_tokenizer import get_tokenizer
    tokenizer = get_tokenizer(True)
    return tokenizer.encode(text)


from functools import lru_cache

@lru_cache
def get_word_tokens(words):
    def augment_word(w):
        yield ' ' + w
        yield ' ' + w[0].upper() + w[1:]
        yield w[0].upper() + w[1:]

    return {
        word: tokenize(word)
        for raw_word in words
        for word in augment_word(raw_word)
    }



active_words = {}


def _logits_callback(ctx: api.Context, n_tokens: int, logits: np.array, kwargs):
    global active_words

    swear_words = kwargs.get('swear_words')
    top_k = 30
    top_token_ids = logits.argsort()[-top_k:]

    if n_tokens == 0:
        active_words = {}

    logsumexp = np.ma.masked_invalid(logits)
    logsumexp = np.exp(logits - logits.max()).sum()
    logsumexp = np.log(logsumexp) + logits.max()

    for word, tokens in get_word_tokens(swear_words).items():
        if word in active_words:
            idx = active_words[word]['idx']
            # only collect proba if there are still tokens missing
            if idx < len(tokens):
                if tokens[idx] in top_token_ids:
                    log_proba = logits[tokens[idx]] - logsumexp
                    active_words[word]['p'] *= np.exp(log_proba)
                active_words[word]['idx'] += 1
        else:
            if tokens[0] in top_token_ids:
                # p is the conditional probability discounted by the amount
                #   of tokens - this is done to penalize words that are
                #   only found partially (e.g. the first 2 of 5 tokens)
                # idx is the token index of the current word that was
                #     last processed
                log_proba = logits[tokens[0]] - logsumexp
                active_words[word] = {'p': np.exp(log_proba), 'idx': 0, 'len': len(tokens)}


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
        word = token.lower().strip(string.punctuation + string.whitespace)
        if word in swear_words:
            print("!" * 30, f"swear word detected (p={p})")
            send_word(firmware_address, bytes(word, 'utf-8'))


def _store_transcript_handler(
    ctx: api.Context, n_new: int, kwargs,
):
    swear_words: list[str] = kwargs['swear_words']
    firmware_address: tuple[str,str] = kwargs['firmware_address']
    segment = ctx.full_n_segments() - n_new

    # collected tokens over all segments
    all_tokens = []

    print(sorted(active_words.items(), key=lambda x: -x[1]['p']))

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

            print("debug:", colored_token("'"+token+"'", token_data.p), token_data.id)

            # we attempt to merge tokens that are likely to be part of a
            # larger word. the vocabulary seems to be ordered in such a way
            # that tokens may start with a whitespace but don't end with one.
            # this way we can just see if the next token starts with a
            # punctuation token in which case we stop merging.

            if merge_token is None:
                merge_token = ([token], [token_data.p])
                continue

            # this token is mergeable, collect it for merging
            if not token or token[0] not in string.whitespace + string.punctuation:
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

    top_idcs = np.array(word_probas).argsort()[-3:]
    print("from probas: ", top_idcs, [f'{word_words[i]}:{word_probas[i]*100:.3f}' for i in top_idcs])


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

    swear_words = (
        "ass",
        "arsch",
        "bitch",
        "bratze",
        "eleven",
        "fick",
        "fuck",
        "hölle",
        "hure",
        "kack",
        "kacke",
        "kackscheiß",
        "kackbratze",
        "nutte",
        "titten",
        "penis",
        "scheiß",
        "scheiße",
        "schlampe",
        "shit",
        "spack",
        "spacken",
        "wichser",
    )

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

