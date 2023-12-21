from functools import lru_cache
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
    # invoked on every new 'segment' (i.e. an inference run on a audio snippet).
    # called for every predicted token (n_tokens is the number of tokens that
    # were already predicted before, so n_tokens == 0 signifies the first call
    # in the decoding process).

    global active_words

    swear_words = kwargs.get('swear_words')
    top_k = 100
    sorted_tokens = logits.argsort()
    top_token_ids = sorted_tokens[-top_k:]

    if n_tokens == 0:
        active_words = {}

    logsumexp = np.ma.masked_invalid(logits)
    logsumexp = np.exp(logits - logits.max()).sum()
    logsumexp = np.log(logsumexp) + logits.max()

    all_probas = np.exp(logits - logsumexp)
    min_p = all_probas.min()
    max_p = all_probas.max()
    p99_p = np.median(all_probas)

    """
    we want to find the conditional probability of words in a list
    *if* they are in a certain confidence threshold (defined by a quantile (top_k)).

    - what happens if we think we already found a word beginning but find a
      beginning token that matches the word but with higher probability?
      -> greedy or non-greedy (i.e. reset)?
      => let's go with greedy for now as it is easier to implement
         (i.e. stick with what we already found)
      -> it happens quite often that the first token matched is not the
         best one
      => we should at least implement it so that we take the
         most probable start and, with it, overriding active words.

    - what happens if we don't gather all tokens of a word (left-overs)?
      => word should be discarded

    - what happens if an intermediate token is not in the top-k list?
      -> discard word, discount probability or take probability?
      -> discard would be 'more correct', discounting would allow for one-off
         errors to be corrected but more difficult to tune. taking the
         probability anyway would disable the top-k filtering which would
         make the whole thing harder to tune.
      => let's go with discard for now

    - how to guarantee token continuity? i.e. enforcing that the tokens of
      a word need to be found directly one after another, not distributed
      over the whole sequence.
      => by deciding to drop tokens that are not in the top-k token list
         we solve this problem implicitly as either the word is active
         and the token is present or the token is not present, then the word
         is not active anymore.

    - conditional probabilities need to be normalized to length
      => after gathering the active words we divide the gathered conditional
         probabilities by the number of tokens.

    - we might need a 'probability floor' to get a sense how high the
      conditional probability is in relation to what is possible at most/worst
      => track max/min p(x_{t} | x_{t-1,...,0})
      -> this does not work as the probabilities are way to low to be taken
         over the whole sequence.
      => instead we may track the min/max probas at time of capturing a token.
         we also may need to skip special tokens.
    """

    def activate_word(word, p):
        active_words[word] = {
            'p': p,
            'idx': 0,
            'len': len(tokens),
            'p99_p': p99_p,
            'min_p': min_p,
            'max_p': max_p,
            'pos': [n_tokens],
        }

    for word, tokens in get_word_tokens(swear_words).items():

        if word in active_words:
            # in non-greedy sampling we would check if the token matches the
            # beginning token with higher probability. we don't do non-greedy.
            idx = active_words[word]['idx'] + 1

            t0_proba = np.exp(logits[tokens[0]] - logsumexp)

            if (tokens[0] in top_token_ids and
                t0_proba >= active_words[word]['p']):
                print(f're-activating {word}; found better')
                activate_word(word, t0_proba)
                continue

            # only collect proba if there are still tokens missing, if we
            # collected all tokens of a word, we're done.
            if idx >= len(tokens):
                continue

            if tokens[idx] in top_token_ids:
                log_proba = logits[tokens[idx]] - logsumexp
                active_words[word]['p'] *= np.exp(log_proba)
                active_words[word]['idx'] += 1
                active_words[word]['min_p'] *= min_p
                active_words[word]['max_p'] *= max_p
                active_words[word]['p99_p'] *= p99_p
                active_words[word]['pos'].append(n_tokens)
                continue
            else:
                # the token did not make it into the top-k tokens, thus we
                # cannot correctly compute the conditional probability of
                # the word and we decided NOT to discount or to take the
                # probability.
                k = np.where(sorted_tokens == tokens[idx])[0][0]
                print(f'dropping word {word}, idx: {idx}, k: {k}')
                del active_words[word]

                # pass through since we might re-activate with this token now.

        if tokens[0] in top_token_ids:
            # p is the conditional probability discounted by the amount
            #   of tokens - this is done to penalize words that are
            #   only found partially (e.g. the first 2 of 5 tokens).
            #   to get the correct proba, you need to multiply by 'idx'+1
            # idx is the token index of the current word that was
            #   last processed
            log_proba = logits[tokens[0]] - logsumexp
            activate_word(word, np.exp(log_proba))


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


@lru_cache
def normalize_word(w: str) -> str:
    return w.lower().strip(string.punctuation + string.whitespace)


def process_merged_tokens(firmware_address, tokens, swear_words):
    for token, p in tokens:
        print(f"'{colored_token(token, p)}'", end='')
    print()

    for token, p in tokens:
        word = normalize_word(token)
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

    for word in active_words:
        active_words[word]['p'] /= active_words[word]['len']
        active_words[word]['p99_p'] /= active_words[word]['len']
    print('normalized:', sorted(active_words.items(), key=lambda x: -x[1]['p']))

    for word in active_words:
        active_words[word]['p'] /= active_words[word]['max_p']
    print('    scaled:', sorted(active_words.items(), key=lambda x: -x[1]['p']))

    # join all similar words into one proba
    comb_p = {}
    for word, data in active_words.items():
        w = normalize_word(word)
        if w not in comb_p:
            comb_p[w] = {'p': 0, 'count': 0}
        comb_p[w]['p'] += data['p']
        comb_p[w]['count'] += 1
    comb_p = {w: p['p'] / p['count'] for w, p in comb_p.items()}

    print('  combined:', sorted(comb_p.items(), key=lambda x: -x[1]))

    print('    chosen:', [
        (w,p) for w,p in sorted(comb_p.items(), key=lambda x: -x[1])
        if p >= 0.3
    ])

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

