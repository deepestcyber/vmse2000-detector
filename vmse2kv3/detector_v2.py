#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import importlib.metadata
import queue
import time
from typing import Callable
from functools import partial
import re
import string
import socket
import threading

import numpy as np
import sounddevice as sd
import pywhispercpp.constants as constants
import _pywhispercpp as pw
import logging
from pywhispercpp.model import Model
from termcolor import colored

from g2p_de import G2p
from phonsim import sequence_distance


# Graphene to phoneme converter.
g2p = G2p()

# list of words that are often times confused with very similarly sounding
# words and should be considered for phoneme correction. we use this list
# as a pre-filter to keep the false-positives from phoneme correction at bay.
#
# Each word is assigned a custom threshold at which a correction is applied.
WORDS_TO_CORRECT = {
    "nutte": 0.6,
    "cyber": 0.6,
    "kackbratze": 0.8,
    "fotze": 0.7,
    "arschgeige": 0.9,
    "arschgesicht": 0.9,
    "ass": 0.9,
    "bimbo": 0.9,
    "bonze": 0.9,
    "asshole": 0.9,
}


def token_to_str(ctx, tid: int) -> str:
    # this exists since the given ctx.token_to_str cannot deal with
    # all unicode characters apparently :)
    return str(pw.whisper_token_to_bytes(ctx, tid), 'utf8', 'ignore')


def colored_token(token_str, token_prob):
    colors = ['red', 'light_red', 'yellow', 'light_yellow', 'green']
    idx = int(token_prob / (1/len(colors)))

    if idx >= len(colors):
        idx = len(colors) - 1

    return colored(token_str, colors[idx])


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def soundex_german(source, size=4):
    code_map = {
        'a': '0',  'e': '0', 'i': '0', 'o': '0', 'u': '0', 'ä': '0', 'ö': '0', 'ü': '0', 'y': '0', 'j': '0', 'h': '0',
        'b': '1', 'p': '1', 'f': '1', 'v': '1', 'w': '1',
        'c': '2', 'g': '2', 'k': '2', 'q': '2', 'x': '2', 's': '2', 'z': '2', 'ß': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6',
        'ch': '7',
    }
    t = [source[0]]

    for c in re.split('(ch|.)', source):
        if not c:
            continue
        digit = code_map[c]
        if digit and digit != t[-1]:
            t.append(digit)

    for _ in range(size - len(t)):
        t.append('0')

    return ''.join(t)


def compare_phonemes(word1, word2):
    code1 = g2p(word1)
    code2 = g2p(word2)

    return sequence_distance(code1, code2), [
        f"{code1} vs. {code2}"
    ]

    # old:

    code1 = cologne_encode(word1)
    code2 = cologne_encode(word2)

    if not (code1.startswith(code2) or code2.startswith(code1)):
        return 2.0, []

    # more complex version with way worse specificity and sensitivity below:
    weight = {
        "soundex": 0.2,
        "metaphone": 0.5,
        "nysiis": 0.1,
        "cologne": 0.2,
    }

    algorithms = {
        "soundex": soundex_german,
        "metaphone": metaphone,
        "nysiis": nysiis,
        "cologne": cologne_encode,
    }

    trace = []

    total = 0.0
    for entry, algo in algorithms.items():
        code1 = algo(word1)
        code2 = algo(word2)

        lev = levenshtein (code1, code2)
        currentWeight = weight[entry]
        if False:
            print ("comparing %s with %s for %s (%0.2f: weight %0.2f)" % (code1, code2, entry, lev, currentWeight))

        subtotal = lev * currentWeight
        trace.append([(code1, code2), entry, lev, subtotal])
        total += subtotal

    return total, trace


class SwearWordDetector:

    def __init__(
        self,
        model='tiny',
        input_device: int = None,
        queue_threshold: int = 16,
        block_duration: int = 30,
        swear_words: list[str] | None = None,
        firmware_address: tuple[str, int] | None = None,
        phoneme_invocation_threshold = 0.5,
        phoneme_correction_threshold = 1.0,
        temperature=0.0,
        **model_params,
    ):

        """
        :param model: whisper.cpp model name or a direct path to a`ggml` model
        :param input_device: The input device (aka microphone), keep it None to take the default
        :param queue_threshold: The inference won't be running until the data queue is having at least `queue_threshold` elements
        :param block_duration: minimum time audio updates in ms
        :param model_params: any other parameter to pass to the whsiper.cpp model see ::: pywhispercpp.constants.PARAMS_SCHEMA
        """

        self.input_device = input_device
        self.sample_rate = constants.WHISPER_SAMPLE_RATE  # same as whisper.cpp
        self.channels = 1  # same as whisper.cpp
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration / 1000)
        self.q = queue.Queue()
        self.transcribe_queue = queue.Queue()

        if not swear_words:
            raise ValueError("Need swear words for detection.")

        self.swear_words = swear_words
        self.queue_threshold = queue_threshold
        self.firmware_address = firmware_address
        self.phoneme_invocation_threshold = phoneme_invocation_threshold
        self.phoneme_correction_threshold = phoneme_correction_threshold

        self.pwccp_model = Model(
            model,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            single_segment=True,
            no_context=True,
            temperature=temperature,
            **model_params,
        )

        # Since we want token probabilities (non-averaged) it is more efficient
        # to implement the new segment handler ourselves.
        pw.assign_new_segment_callback(
            self.pwccp_model._params,
            partial(self.on_new_segment, instance=self),
        )

    def _audio_callback(self, indata, frames, time, status):
        # Ideally called from sounddevice InputStream thread
        if status:
            logging.warning(F"underlying audio stack warning:{status}")

        assert frames == self.block_size

        self.q.put(indata.copy())

        if self.q.qsize() > self.queue_threshold:
            audio_data = np.array([])
            while self.q.qsize() > 0:
                audio_data = np.append(audio_data, self.q.get())
            self.transcribe_queue.put(audio_data.copy())

    def _transcribe_speech(self, audio_data):
        # Appending zeros to the audio data as a workaround for short audio packets
        #audio_data = np.concatenate([audio_data, np.zeros((int(self.sample_rate) + 10))])

        # running the inference, will call the new segment callback
        # (self.on_new_segment) when new segments are detected.
        a = time.time()
        self.pwccp_model.transcribe(audio_data)
        b = time.time()
        print("transcribe time:",b-a, "record time:", len(audio_data) / 16000)

    def send_word(self, word):
        word = bytes(word, 'utf-8')

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(word, self.firmware_address)

    def process_merged_tokens(self, detected_words: list[tuple[str, float]]):
        for token, p in detected_words:
            print(f"'{colored_token(token, p)}'", end='')
        print()

        for token, p in detected_words:
            word = token.lower().strip(string.punctuation + string.whitespace)
            if word in self.swear_words:
                print("!" * 30, f"swear word '{word}' detected (p={p})")
                if self.firmware_address:
                    self.send_word(word)

    def new_segment_callback(self, ctx, n_new):
        n_segments = pw.whisper_full_n_segments(ctx)
        start = n_segments - n_new

        if not n_segments:
            return

        # collected tokens over all segments
        all_tokens = []

        # decoder segments contain a bunch of tokens
        for segment in range(start, n_segments):
            num_tokens = pw.whisper_full_n_tokens(ctx, segment)

            def merge_tokens(merge_token):
                token = ''.join(merge_token[0])
                p = np.mean(merge_token[1])
                return token, p

            merge_token = None

            for token_idx in range(num_tokens):
                token_data = pw.whisper_full_get_token_data(ctx, segment, token_idx)
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

            # check individual tokens for phonetic similarity with swear
            # words if their probability is low.
            for token, proba in all_tokens:
                if proba >= self.phoneme_invocation_threshold:
                    continue

                if proba <= 0.15:
                    continue

                token = token.lower().strip()
                if not token.isalpha():
                    continue

                found_match = False

                for swear_word, threshold in WORDS_TO_CORRECT.items():
                    score, trace = compare_phonemes(token, swear_word)
                    if score < threshold:
                        print(f"{token} (p={proba}) => {swear_word} because of phonemes! ({score})")
                        print(trace)
                        all_tokens.append((swear_word, 1 - score))
                        found_match = True
                        break

                if found_match:
                    break

            # final processing of tokens
            self.process_merged_tokens(all_tokens)

    def transcription_process(self, *args, **kwargs):
        while True:
            audio_data = self.transcribe_queue.get()

            print('transcribing')
            self._transcribe_speech(np.array(audio_data))
            print('end of transcribing')

    @staticmethod
    def on_new_segment(ctx, n_new, user_data, instance) -> None:
        instance.new_segment_callback(ctx, n_new)

    def start(self) -> None:
        if False:
            data = Model._load_audio('/home/pi/code/whisper.cpp/samples/jfk.wav')

            for i in range(0, len(data), self.block_size):
                frame = data[i:i+self.block_size]

                self._audio_callback(frame, self.block_size, 0, 0)


            return

        t = threading.Thread(target=self.transcription_process)
        t.start()

        with sd.InputStream(
                device=self.input_device,  # the default input device
                channels=self.channels,
                samplerate=constants.WHISPER_SAMPLE_RATE,
                blocksize=self.block_size,
                callback=self._audio_callback):

            try:
                logging.info(f"Listening ... (CTRL+C to stop)")
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logging.info("Stopping")

    @staticmethod
    def available_devices():
        return sd.query_devices()


def _main():
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    parser.add_argument('-m', '--model', default='tiny.en', type=str, help="Whisper.cpp model, default to %(default)s")
    parser.add_argument('-ind', '--input_device', type=int, default=None,
                        help=f'Id of The input device (aka microphone)\n'
                             f'available devices {SwearWordDetector.available_devices()}')
    parser.add_argument('-bd', '--block_duration', default=150, type=int,
                        help=f"minimum time audio updates in ms, default to %(default)s")
    parser.add_argument('-qt', '--queue_threshold', default=16, type=int,
                        help="number of items in the audio queue before transcribing. default %(default)s")
    parser.add_argument('-actx', '--audio_context', default=1500, type=int,
                        help=(
                            "Audio context window size. Lower is faster. Note for OpenVINO: Must be supported "
                            "by the model. Use the custom OpenVINO conversion script in ./scripts/ to build "
                            "a model that supports lower audio contexts."),
    )
    parser.add_argument('--phoneme-correction-threshold', type=float, default=1.0,
        help=(
            'Threshold to reach to cause a token to be corrected with a swear '
            'word. The score that is compared against this threshold is a '
            'weighted combination of the levensthein distance on the output '
            'of different phoneme translation algorithms, so the higher the '
            'score, the higher the dissimilarity.'
        ),
    )
    parser.add_argument('--phoneme-invocation-threshold', type=float, default=0.5,
        help=(
            'Threshold for the word probability to invoke the phoneme '
            'error correction (i.e. finding a swear word that sounds similar). '
            'This threshold is compared against the conditional probability of '
            'the word\'s tokens as detected by whisper.'
        ),
    )
    parser.add_argument('--host', type=str, default='0.0.0.0',
        help='address to send detected words to',
    )
    parser.add_argument('--port', type=int, default=1800,
        help='port to send detected words to',
    )
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('word_list', type=argparse.FileType('r'),
        help="Path to the newline-separated list of swear words to be detected.",
    )

    args = parser.parse_args()

    swear_words = [n.strip() for n in args.word_list]

    detector = SwearWordDetector(
        model=args.model,
        input_device=args.input_device,
        block_duration=args.block_duration,
        queue_threshold=args.queue_threshold,
        language="de",
        swear_words=swear_words,
        firmware_address=(args.host, args.port),
        phoneme_invocation_threshold=args.phoneme_invocation_threshold,
        phoneme_correction_threshold=args.phoneme_correction_threshold,
        audio_ctx=args.audio_context,
        temperature=args.temperature,
        #greedy={'best_of': 5},
        #params_sampling_strategy=0, # 0 = greedy, 1 = beam search
    )
    detector.start()


if __name__ == '__main__':
    _main()
