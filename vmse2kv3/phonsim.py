# Thanks ChatGPT.
#
# Without you this would have taken a lot longer. It would have been
# less error-prone and I would be able to tune it a lot better but
# I wouldn't have done it in time. Maybe.
#
# Good prototype.

import math
import itertools
import json

#######################################################################
# 1) German phoneme inventory (atomic; affricates, diphthongs included)
#######################################################################

PHONEMES = [
    'ts', 'ə', 'iː', 'oː', 'pf', 'aj', 'd', 'tʃ', 'm', 'œ', 'z', 'ɛ', 'ɲ',
    't', 'ɟ', 'n̩', 'b', 'ɪ', 'kʰ', 'h', 'eː', 'ɔ', 'f', 'v', 'l̩', 'n', 'x',
    'yː', 'p', 'c', 'aː', 'ç', 'uː', 'ʃ', 'øː', 'a', 'l', 'j', 'ɔʏ', 'cʰ',
    'aw', 'ŋ', 'ɐ', 'ʊ', 'pʰ', 'ʁ', 's', 'ʏ', 'ɡ', 'tʰ', 'k', 'm̩'
]

#######################################################################
# 2) Feature system
#
# All phonemes use the same flat feature vector. Missing features = 0.
#
# For vowels:
#   V, height, backness, rounded, central, long
#
# For consonants:
#   C, sonorant, voiced, place_labial, place_alv, place_post, place_pal, place_velar,
#   manner_stop, manner_fric, manner_aff, manner_nasal, manner_lat, manner_approx,
#   aspiration
#
#######################################################################

# Helper templates
VOWEL = {
    'V':1,'C':0,
    'height':0,'back':0,'round':0,'central':0,'long':0,
    'sonorant':0,'voiced':0,
    'place_labial':0,'place_alv':0,'place_post':0,'place_pal':0,'place_velar':0,
    'manner_stop':0,'manner_fric':0,'manner_aff':0,
    'manner_nasal':0,'manner_lat':0,'manner_approx':0,
    'asp':0
}

CONS = {
    'V':0,'C':1,
    'height':0,'back':0,'round':0,'central':0,'long':0,
    'sonorant':0,'voiced':0,
    'place_labial':0,'place_alv':0,'place_post':0,'place_pal':0,'place_velar':0,
    'place_glottal':0,
    'manner_stop':0,'manner_fric':0,'manner_aff':0,
    'manner_nasal':0,'manner_lat':0,'manner_approx':0,
    'asp':0
}

#######################################################################
# 3) Feature vectors for every phoneme (Option A — atomic diphthongs/affricates)
#
# Values chosen to reflect German phonology realistically.
#######################################################################

FV = {}

def V(height, back, round, central, long=False):
    f = VOWEL.copy()
    f['height'] = height
    f['back'] = back
    f['round'] = round
    f['central'] = central
    f['long'] = 1 if long else 0
    f['sonorant'] = 1
    return f

def C(voiced, place, manner, sonorant=0, asp=0):
    f = CONS.copy()
    f['voiced'] = voiced
    f['asp'] = asp
    if place == 'labial': f['place_labial']=1
    if place == 'alv':    f['place_alv']=1
    if place == 'post':   f['place_post']=1
    if place == 'pal':    f['place_pal']=1
    if place == 'velar':  f['place_velar']=1
    if place == 'glottal': f['place_glottal']=1

    if manner=='stop': f['manner_stop']=1
    if manner=='fric': f['manner_fric']=1
    if manner=='aff':  f['manner_aff']=1
    if manner=='nasal':f['manner_nasal']=1
    if manner=='lat':  f['manner_lat']=1
    if manner=='approx':f['manner_approx']=1

    f['sonorant'] = sonorant
    return f

# --- vowels ---
FV['iː'] = V(0.0,0.0,0,0,long=True)
FV['ɪ']  = V(0.15,0.05,0,0)
FV['yː'] = V(0.0,0.0,1,0,long=True)
FV['ʏ']  = V(0.15,0.05,1,0)
FV['eː'] = V(0.30,0.10,0,0,long=True)
FV['ɛ']  = V(0.45,0.15,0,0)
FV['øː'] = V(0.30,0.10,1,0,long=True)
FV['œ']  = V(0.45,0.15,1,0)
FV['aː'] = V(1.00,0.60,0,0,long=True)
FV['a']  = V(1.00,0.60,0,0)
FV['ɐ']  = V(0.85,0.55,0,0.60)
FV['ə']  = V(0.55,0.45,0,0.65)
FV['uː'] = V(0.0,1.0,1,0,long=True)
FV['ʊ']  = V(0.40,0.80,1,0)
FV['oː'] = V(0.30,0.95,1,0,long=True)
FV['ɔ']  = V(0.45,0.95,1,0)

# diphthongs (atomic)
FV['aj'] = V(0.85,0.25,0,0.15)  # composite mid-frontish low-ish diphthong
FV['aw'] = V(0.85,0.50,0,0.10)
FV['ɔʏ'] = V(0.60,0.55,1,0.20)

# --- consonants ---
FV['p']  = C(0,'labial','stop')
FV['pʰ'] = C(0,'labial','stop',asp=1)
FV['b']  = C(1,'labial','stop')

FV['t']  = C(0,'alv','stop')
FV['tʰ'] = C(0,'alv','stop',asp=1)
FV['d']  = C(1,'alv','stop')

FV['k']  = C(0,'velar','stop')
FV['kʰ'] = C(0,'velar','stop',asp=1)
FV['ɡ']  = C(1,'velar','stop')

# affricates treated as atomic
FV['ts'] = C(0,'alv','aff')
FV['pf'] = C(0,'labial','aff')
FV['tʃ']= C(0,'post','aff')

# fricatives
FV['s']  = C(0,'alv','fric')
FV['z']  = C(1,'alv','fric')
FV['ʃ']  = C(0,'post','fric')
FV['ç']  = C(0,'pal','fric')
FV['x']  = C(0,'velar','fric')
FV['h']  = C(0,'glottal','fric')  # neutral place but keep alv

# nasals
FV['m']  = C(1,'labial','nasal',sonorant=1)
FV['n']  = C(1,'alv','nasal',sonorant=1)
FV['ŋ']  = C(1,'velar','nasal',sonorant=1)
FV['ɲ']  = C(1,'pal','nasal',sonorant=1)

# syllabic sonorants
FV['n̩'] = C(1,'alv','nasal',sonorant=1)
FV['m̩'] = C(1,'labial','nasal',sonorant=1)
FV['l̩'] = C(1,'alv','lat',sonorant=1)

# liquids & glides
FV['l']  = C(1,'alv','lat',sonorant=1)
FV['ʁ']  = C(1,'velar','approx',sonorant=1)
FV['j']  = C(1,'pal','approx',sonorant=1)

FV['ɟ'] = C(1,'pal','stop')
FV['f'] = C(0,'labial','fric')
FV['v'] = C(1,'labial','fric')
FV['c']  = C(0,'pal','stop')
FV['cʰ'] = C(0,'pal','stop', asp=1)

#######################################################################
# 4) Weights
#######################################################################

WEIGHTS = {
    # vowel features
    'height':  0.30,
    'back':    0.25,
    'round':   0.15,
    'central': 0.20,
    'long':    0.10,

    # consonant features
    'voiced':       0.15,
    'sonorant':     0.10,
    'place_labial': 0.10,
    'place_alv':    0.10,
    'place_post':   0.10,
    'place_pal':    0.10,
    'place_velar':  0.10,
    'place_glottal': 0.25,
    'manner_stop':  0.15,
    'manner_fric':  0.15,
    'manner_aff':   0.15,
    'manner_nasal': 0.15,
    'manner_lat':   0.15,
    'manner_approx':0.15,
    'asp':          0.05,
}

#######################################################################
# 5) Similarity computation
#######################################################################

def raw_distance(p, q):
    P, Q = FV[p], FV[q]

    # vowel–consonant penalty
    if P['V'] != Q['V']:
        return 999.0  # gives similarity 0

    # weighted Euclidean distance
    s = 0.0
    for k, w in WEIGHTS.items():
        dv = P.get(k,0) - Q.get(k,0)
        s += w * (dv * dv)
    return math.sqrt(s)

# maximum possible distance for normalization
MAX_D = math.sqrt(sum(w for w in WEIGHTS.values()))

def similarity(p, q):
    d = raw_distance(p, q)
    if d >= MAX_D:
        return 0.0
    return max(0.0, 1 - d / MAX_D)

#######################################################################
# 6) Full similarity matrix
#######################################################################

def similarity_matrix():
    M = {}
    for p in PHONEMES:
        M[p] = {}
        for q in PHONEMES:
            M[p][q] = round(similarity(p,q), 4)
    return M

#######################################################################
# 7) Sequence distance (phonological Levenshtein)
#######################################################################

def sequence_distance(seq1, seq2):
    """
    Computes distance between phoneme sequences using DP.
    Insert/delete cost = 1
    Substitution cost = 1 - similarity(p,q)
    """
    n, m = len(seq1), len(seq2)

    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0] = i
    for j in range(1,m+1): dp[0][j] = j

    for i in range(1,n+1):
        for j in range(1,m+1):
            sub_cost = 1 - similarity(seq1[i-1], seq2[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + sub_cost
            )
    return dp[n][m]

#######################################################################
# 8) Demo
#######################################################################

if __name__ == "__main__":
    M = similarity_matrix()
    print(json.dumps(M, ensure_ascii=False, indent=2))

    # example sequence distance
    print("distance(['n','ʊ','tʰ','ɐ'], ['n','ʊ','tʰ','ə']) =",
          sequence_distance(['n','ʊ','tʰ','ɐ'], ['n','ʊ','tʰ','ə']))

