import itertools
import re
from pprint import pprint

from phonetics import metaphone, soundex, nysiis
import numpy as np
from cologne_phonetics import encode as _cologne_encode


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


def compare(word1, word2, weight):
    algorithms = {
        #"soundex": soundex,
        "soundex": soundex_german,
        "metaphone": metaphone,
        "nysiis": nysiis,
        "cologne": cologne_encode,
    }

    code1 = cologne_encode(word1)
    code2 = cologne_encode(word2)

    if not (code1.startswith(code2) or code2.startswith(code1)):
        return 2.0, []

    trace = []
    total = 0.0
    for entry, algo in algorithms.items():
        code1 = algo(word1)
        code2 = algo(word2)

        lev = levenshtein (code1, code2)
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


threshold = 1.0

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


"""
words = ["kacke", "nutte", "kacka", "nutter", "note"]

for word1, word2 in itertools.product(words, words):
    if word1 == word2: continue
    compare(word1, word2)
"""
