#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import importlib.metadata
import queue
import time
from typing import Callable
from functools import partial
import string
import socket

import numpy as np
#import sounddevice as sd
import pywhispercpp.constants as constants
import _pywhispercpp as pw
import logging
from pywhispercpp.model import Model
from termcolor import colored


def token_to_str(ctx, tid: int) -> str:
    return pw.whisper_token_to_str(ctx, tid)

    # this exists since the given ctx.token_to_str cannot deal with
    # all unicode characters apparently :)
    return str(pw.whisper_token_to_bytes(ctx, tid), 'utf8', 'ignore')


def colored_token(token_str, token_prob):
    colors = ['red', 'light_red', 'yellow', 'light_yellow', 'green']
    idx = int(token_prob / (1/len(colors)))

    if idx >= len(colors):
        idx = len(colors) - 1

    return colored(token_str, colors[idx])


class SwearWordDetector:

    def __init__(
        self,
        model='tiny',
        input_device: int = None,
        silence_threshold: int = 8,
        q_threshold: int = 16,
        block_duration: int = 30,
        swear_words: list[str] | None = None,
        firmware_address: tuple[str, int] | None = None,
        **model_params,
    ):

        """
        :param model: whisper.cpp model name or a direct path to a`ggml` model
        :param input_device: The input device (aka microphone), keep it None to take the default
        :param q_threshold: The inference won't be running until the data queue is having at least `q_threshold` elements
        :param block_duration: minimum time audio updates in ms
        :param model_params: any other parameter to pass to the whsiper.cpp model see ::: pywhispercpp.constants.PARAMS_SCHEMA
        """

        self.input_device = input_device
        self.sample_rate = constants.WHISPER_SAMPLE_RATE  # same as whisper.cpp
        self.channels = 1  # same as whisper.cpp
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration / 1000)
        self.q = queue.Queue()

        if not swear_words:
            raise ValueError("Need swear words for detection.")

        self.swear_words = swear_words
        self.q_threshold = q_threshold

        self.firmware_address = firmware_address

        self.pwccp_model = Model(
            model,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            single_segment=True,
            no_context=True,
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
        audio_data = map(lambda x: (x + 1) / 2, indata)  # normalize from [-1,+1] to [0,1]
        audio_data = np.fromiter(audio_data, np.float16)
        audio_data = audio_data.tobytes()
        self.q.put(indata.copy())

        if self.q.qsize() > self.q_threshold:
            self._transcribe_speech()

    def _transcribe_speech(self):
        audio_data = np.array([])
        while self.q.qsize() > 0:
            # get all the data from the q
            audio_data = np.append(audio_data, self.q.get())

        # Appending zeros to the audio data as a workaround for short audio packets
        audio_data = np.concatenate([audio_data, np.zeros((int(self.sample_rate) + 10))])

        # running the inference, will call the new segment callback
        # (self.on_new_segment) when new segments are detected.
        self.pwccp_model.transcribe(audio_data)

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

            # final processing of tokens
            self.process_merged_tokens(all_tokens)

    @staticmethod
    def on_new_segment(ctx, n_new, user_data, instance) -> None:
        instance.new_segment_callback(ctx, n_new)

    def start(self) -> None:
        data = Model._load_audio('./data/nutteschlampekacksheisse_16k.wav')

        for i in range(0, len(data), self.block_size):
            frame = data[i:i+self.block_size]

            self._audio_callback(frame, self.block_size, 0, 0)


        return

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
        return []  # FIXME
        return sd.query_devices()


def _main():
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    parser.add_argument('-m', '--model', default='tiny.en', type=str, help="Whisper.cpp model, default to %(default)s")
    parser.add_argument('-ind', '--input_device', type=int, default=None,
                        help=f'Id of The input device (aka microphone)\n'
                             f'available devices {SwearWordDetector.available_devices()}')
    parser.add_argument('-bd', '--block_duration', default=150, type=int,
                        help=f"minimum time audio updates in ms, default to %(default)s")
    parser.add_argument('--host', type=str, default='localhost',
        help='address to send detected words to',
    )
    parser.add_argument('--port', type=int, default=1800,
        help='port to send detected words to',
    )
    parser.add_argument('word_list', type=argparse.FileType('r'),
        help="Path to the newline-separated list of swear words to be detected.",
    )

    args = parser.parse_args()

    swear_words = [n.strip() for n in args.word_list]

    detector = SwearWordDetector(
        model=args.model,
        input_device=args.input_device,
        block_duration=args.block_duration,
        language="de",
        swear_words=swear_words,
        firmware_address=(args.host, args.port),
    )
    detector.start()


if __name__ == '__main__':
    _main()
