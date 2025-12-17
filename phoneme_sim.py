"""Quick benchmark to evaluate phoneme correction on a small dataset.

Measures accuracy, specificity and sensitivity.

Our goal is to have a sensitivity of 1 since false-positives are our
nemesis. But it would be good to have a specificity > 0.5 as well :)

Some notes:
- at first we tried using very abstract methods like cologne phonetics,
  soundex, metaphone, ... (via the phonetics and cologne_phonetics)
  packages. This was working OK but is not very granular since graphemes
  are mapped rather broadly to respective classes. Paired with an edit
  distance and you get difference of 1 between possitive matches as well
  as negative matches. Thresholding becomes very hard.

- to mitigate the granularity we use `g2p_de`, a grapheme to phoneme mapper
  employing a lexicon as well as a GRU. having phonemes we can create a
  similarity matrix phoneme -> phoneme and use that for our similarity score.
  this is much more granular and we can threshold so that specificity is OK
  and sensitivity is 1. since the similarity matrix is created from features
  and code written bei an LLM it is likely that it contains quite a few
  errors but it is OK for now.

- since g2p_de utilizes a GRU to encode the word and map it to phonemes, we
  can take the encoded hidden state of two words and compute a similarity
  between them. this seems to work quite well and seems to have (no real
  word testing yet) to have a higher specificity than the phoneme similarity
  matrix approach. it should also be less expensive computationally.

"""

import itertools
import re
from pprint import pprint

from phonetics import metaphone, soundex, nysiis
import numpy as np
from cologne_phonetics import encode as _cologne_encode

from vmse2kv3.phonsim import sequence_distance
from g2p_de import G2p

g2p = G2p()


def cologne_encode(s):
    return _cologne_encode(s)[0][1]


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



# for: / max(l1, l2)
words_to_correct = {
    "nutte": 0.30,
    "cyber": 0.15,
    "kackbratze": 0.50,
    "fotze": 0.15,
    "arschgeige": 0.51,
    "arschgesicht": 0.49,
    "ass": 0.15,
    "bimbo": 0.39,
    "bonze": 0.40,
    "asshole": 0.56,
}



# for: no normalization
words_to_correct = {
    "nutte": 0.6,
    "cyber": 0.6,
    "kackbratze": 0.8,
    "fotze": 0.7,
    "arschgeige": 0.9,
    "arschgesicht": 0.9,
    "ass": 0.5,
    "bimbo": 0.9,
    "bonze": 0.9,
    "asshole": 0.9,
}

# for / min(l1, l2)
words_to_correct = {
    "nutte": 0.20,
    "cyber": 0.13,
    "kackbratze": 0.50,
    "fotze": 0.15,
    "arschgeige": 0.51,
    "arschgesicht": 0.49,
    "ass": 0.02,
    "bimbo": 0.397,
    "bonze": 0.40,
    "asshole": 0.49,
}

# for: / l2
words_to_correct = {
    "nutte": 0.20,
    "cyber": 0.13,
    "kackbratze": 0.20,
    "hackfresse": 0.30,
    "fotze": 0.15,
    "arschgeige": 0.43,
    "arschgesicht": 0.42,
    "ass": 0.02,
    "bimbo": 0.26,
    "bonze": 0.18,
    "asshole": 0.32,
    "crypto": 0.20,
    "bitcoin": 0.36,
}

# for g2p embedding sim
_words_to_correct = {
    "nutte": 0.24,
    "cyber": 0.28,
    "kackbratze": 0.20,
    "hackfresse": 0.30,
    "fotze": 0.24,
    "arschgeige": 0.23,
    "arschgesicht": 0.28,
    "ass": 0.02,
    "bimbo": 0.26,
    "bonze": 0.20,
    "asshole": 0.30,
    "crypto": 0.20,
    "bitcoin": 0.33,
}


def g2p_embedding_compare(word1, word2):
    def get_hidden(word):
        enc = g2p.encode(word)
        enc = g2p.gru(enc, len(word) + 1, g2p.enc_w_ih, g2p.enc_w_hh,
                       g2p.enc_b_ih, g2p.enc_b_hh, h0=np.zeros((1, g2p.enc_w_hh.shape[-1]), np.float32))
        return enc[:, -1, :]

    h1 = get_hidden(word1) / len(word1)
    h2 = get_hidden(word2) / len(word2)

    m1 = np.sqrt((h1 ** 2).sum())
    m2 = np.sqrt((h2 ** 2).sum())

    return 1 - ((h1 * h2).sum() / (m1 * m2))



def compare(word1, word2, weight):
    if word2 not in words_to_correct:
        return 100, ['word not in correct list']

    # 78% acc, 46% sens., 100% spec. - promising!
    #return g2p_embedding_compare(word1, word2), []

    code1 = cologne_encode(word1)
    code2 = cologne_encode(word2)

    #print(word1, word2)
    #print(g2p(word1), g2p(word2))

    #if not (code1.startswith(code2) or code2.startswith(code1)):
    #    return 100, []

    code1 = g2p(word1)
    code2 = g2p(word2)

    #return sequence_distance(code1, code2) / min(len(code1), len(code2)), [
    return sequence_distance(code1, code2) / len(code2), [
        f"{code1} vs. {code2}"
    ]


    trace = []

    algorithms = {
        #"soundex": soundex,
        "soundex": soundex_german,
        "metaphone": metaphone,
        "nysiis": nysiis,
        "cologne": cologne_encode,
    }

    total = 0.0
    for entry, algo in algorithms.items():
        code1 = algo(word1)
        code2 = algo(word2)

        lev = levenshtein (code1, code2) / len(code2)
        currentWeight = weight[entry]
        #print ("comparing %s with %s for %s (%0.2f: weight %0.2f)" % (code1, code2, entry, lev, currentWeight))
        subtotal = lev * currentWeight
        trace.append([(code1, code2), entry, lev, subtotal])
        total += subtotal

    return total, trace


with open('data/phoneme.txt') as f:
    phoneme_data = []
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        phoneme_data.append(line.split(' '))


base_threshold = 0.6

weight = {
    "soundex": 0.2,
    "metaphone": 0.5,
    "nysiis": 0.1,
    "cologne": 0.2,
}

tp = 0
fp = 0
fn = 0
tn = 0

for is_good, w1, w2 in phoneme_data:
    is_good = int(is_good)
    score, trace = compare(w1, w2, weight)

    threshold = words_to_correct.get(w2, base_threshold)

    if score <= threshold and is_good:
        tp += 1

    if score > threshold and is_good:
        print(f'false-negative: {w1} {w2}: {score}')
        print(pprint(trace))
        fn += 1

    if score <= threshold and not is_good:
        print(f'false-positive: {w1} {w2}: {score}')
        print(pprint(trace))
        fp += 1

    if score > threshold and not is_good:
        tn += 1

print(f"Accuracy: {(tp+tn) / (fp+fn+tp+tn)}")
print(f"Sensitivity: TP / (FN+TP): {tp / (fn + tp)}")
print(f"Specificity: TN / (FP+TN): {tn / (fp + tn)}")

