import re
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
from numpy import nan

from camel_tools.utils.charmap import CharMapper

global roots_counter

normalize_map = CharMapper({
    '<': 'A',
    '>': 'A',
    '|': 'A',
    '{': 'A',
    'Y': 'y'
})

root_error = 0

def preprocess_func(form):
    form = normalize_map(form)
    form = [c if c not in 'uioa~' else '' for c in form]
    return form

def propose_pattern(lemma_egy, lemma_norm_dediac_egy, pattern_lemma_msa, root):
    root = root.split('.')
    pattern_lemma_dediac_norm_msa = ''.join(preprocess_func(pattern_lemma_msa))
    dediac2lemma = []
    offset = 0
    for i, p in enumerate(lemma_egy):
        if offset < len(lemma_norm_dediac_egy) and p == lemma_norm_dediac_egy[offset]:
            dediac2lemma.append(i)
            offset += 1
    lemma2dediac = {i: offset for offset, i in enumerate(dediac2lemma)}
    
    proposed_pattern = []
    for i, c in enumerate(lemma_egy):
        if i in lemma2dediac:
            if c in "><&}{'|Y":
                proposed_pattern.append(c)
            else:
                proposed_pattern.append(pattern_lemma_dediac_norm_msa[lemma2dediac[i]])
        else:
            proposed_pattern.append(c)
    proposed_pattern = ''.join(proposed_pattern)
    
    try:
        reconstructed = []
        for p in proposed_pattern:
            if p.isdigit():
                reconstructed.append(root[int(p) - 1])
            else:
                reconstructed.append(p)
        assert ''.join(reconstructed) == lemma_egy
        return proposed_pattern
    except:
        global root_error
        root_error += 1
        return 'ROOT_ERROR'

data_msa_nom = pd.read_csv(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/data/camel-morph-msa/nom_msa_red/MSA-Nom-LEX.csv')
data_egy_nom = pd.read_csv(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/data/camel-morph-egy/nom_egy_red/EGY-Nom-LEX.csv')
data_egy_verb = pd.read_csv(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/data/camel-morph-egy/all_aspects_egy/EGY-Verb-LEX-PV.csv')
data_pal = pd.read_csv(
    '/Users/chriscay/Downloads/PACL-Letter-Split - Phrase-Lemma-Add.csv')
data_msa_nom = data_msa_nom.replace(nan, '', regex=True)
data_egy_nom = data_egy_nom.replace(nan, '', regex=True)
data_egy_verb = data_egy_verb.replace(nan, '', regex=True)
data_pal = data_pal.replace(nan, '', regex=True)

pattern2freq = {}
for _, row in tqdm(data_msa_nom.iterrows(), total=len(data_msa_nom.index)):
    pattern2freq.setdefault(row['PATTERN_LEMMA'], 0)
    pattern2freq[row['PATTERN_LEMMA']] += 1

dediac_norm_lemma2root_msa = {}
lemma_norm_egy2root_msa = {}
backoff = {}
# data_msa_nom = data_msa_nom[data_msa_nom['LEMMA'] == 'fukAhap']
for _, row in tqdm(data_msa_nom.iterrows(), total=len(data_msa_nom.index)):
    lemma_dediac_norm_msa = preprocess_func(row['LEMMA'])
    info = dediac_norm_lemma2root_msa.setdefault(''.join(lemma_dediac_norm_msa), [set(), ''])
    info[0].add(row['ROOT'])
    info[1] = row['PATTERN_LEMMA']
    
    lemma_norm_msa = normalize_map(row['LEMMA'])
    info = lemma_norm_egy2root_msa.setdefault(lemma_norm_msa, [set(), ''])
    info[0].add(row['ROOT'])
    info[1] = row['PATTERN_LEMMA']
    
    pattern_dediac_norm = ''.join(preprocess_func(row['PATTERN_LEMMA']))
    match_pattern = ''.join('([^Ap])' if p.isdigit() else p for p in pattern_dediac_norm)
    if pattern2freq[row['PATTERN_LEMMA']] > 1:
        regexes = backoff.setdefault(len(pattern_dediac_norm), {})
        pattern_dediac_norm_digits = re.sub(r'\D', '', pattern_dediac_norm)
        info = regexes.setdefault(match_pattern, [set(), ''])
        info[0].add(pattern_dediac_norm_digits)
        info[1] = row['PATTERN_LEMMA']

roots_msa_nom = set(data_msa_nom['ROOT'].replace(r'[wy]', '#', regex=True).values.tolist())
roots_egy_verb = set(data_egy_verb['ROOT'].replace(r'[wy]', '#', regex=True).values.tolist())
roots_egy_nom = set(data_egy_nom['ROOT'].replace(r'[wy]', '#', regex=True).values.tolist())
gold_roots = roots_msa_nom | roots_egy_verb | roots_egy_nom

def get_pattern(lemma_egy, lemma_dediac_norm_egy, pattern_lemma_msa, roots_msa):
    roots_msa = list(roots_msa)
    num_roots = len(roots_msa)
    num_roots = num_roots if num_roots <= 2 else '3+'
    if num_roots == 1:
        if roots_msa[0] == 'NTWS':
            proposed_pattern = 'NTWS'
        elif pattern_lemma_msa == 'ERROR':
            proposed_pattern = 'NO_MSA_PATTERN'
        else:
            proposed_pattern = propose_pattern(lemma_egy, lemma_dediac_norm_egy, pattern_lemma_msa, roots_msa[0])
    else:
        ranking = np.argmax(np.array([roots_counter[root] for root in roots_msa]))
        proposed_pattern = propose_pattern(
            lemma_egy, lemma_dediac_norm_egy, pattern_lemma_msa, roots_msa[ranking.item()])
    
    return proposed_pattern, num_roots

def filter_roots(possible_roots):
    possible_roots_with_hash, possible_roots_no_hash = [], []
    for possible_root in possible_roots:
        if '#' in possible_root[0]:
            possible_roots_with_hash.append(possible_root)
        else:
            possible_roots_no_hash.append(possible_root)
    if possible_roots_no_hash:
        possible_roots_filtered = possible_roots_no_hash
    else:
        possible_roots_filtered = possible_roots_no_hash + possible_roots_with_hash
    possible_roots_filtered = [(root, pattern) for root, pattern in possible_roots_filtered
                                if any(c != '#' for c in root.split('.'))]
    return possible_roots_filtered

# data_egy_nom = data_egy_nom[data_egy_nom['LEMMA'] == 'mitqaw~il']
roots_egy = []
roots_counter = Counter()
for _, row in tqdm(data_pal.iterrows(), total=len(data_pal.index)):
    if not row['LEMMA']:
        roots_egy.append((row['LEMMA'], '', 'NO_LEMMA', '', ''))
        continue
    lemma_dediac_norm_egy = ''.join(preprocess_func(row['LEMMA']))
    lemma_norm_egy = normalize_map(row['LEMMA'])
    lemma_dediac_egy = re.sub(r'[uioa~]', '', row['LEMMA'])
    # Normalized MSA lemmas lookup for root
    if lemma_norm_egy in lemma_norm_egy2root_msa:
        roots_msa, pattern_lemma_msa = lemma_norm_egy2root_msa[lemma_norm_egy]
        roots_msa = list(roots_msa)
        proposed_pattern, num_roots = get_pattern(row['LEMMA'], lemma_dediac_egy, pattern_lemma_msa, roots_msa)
        roots_egy.append((row['LEMMA'], roots_msa, f'LOOKUP_DIAC-{num_roots}', pattern_lemma_msa, proposed_pattern))
        roots_counter.update(roots_msa)
    # Normalized and dediacritized MSA lemmas lookup for root
    elif lemma_dediac_norm_egy in dediac_norm_lemma2root_msa:
        roots_msa, pattern_lemma_msa = dediac_norm_lemma2root_msa[lemma_dediac_norm_egy]
        roots_msa = list(roots_msa)
        proposed_pattern, num_roots = get_pattern(
            row['LEMMA'], lemma_dediac_egy, pattern_lemma_msa, roots_msa)
        roots_egy.append((row['LEMMA'], list(roots_msa), f'LOOKUP_DEDIAC-{num_roots}', pattern_lemma_msa, proposed_pattern))
        roots_counter.update(roots_msa)
    # Abstract MSA lemmas backoff (regex method)
    else:
        possible_roots = []
        if len(lemma_dediac_norm_egy) not in backoff:
            roots_egy.append((row['LEMMA'], '', 'BACKOFF_NO_LEN', '', ''))
            continue
        for regex, info in backoff[len(lemma_dediac_norm_egy)].items():
            patterns_dediac_norm_digits_msa, pattern_lemma_msa = info
            match = re.match(f'^{regex}$', lemma_dediac_norm_egy)
            if match:
                for pattern in patterns_dediac_norm_digits_msa:
                    root = []
                    for i, digit in enumerate(pattern):
                        root.append('#' if int(digit) != i + 1 else match.groups()[i])
                    if 2 < len(root) < 6:
                        possible_roots.append(('.'.join(root), pattern_lemma_msa))
        
        if possible_roots:
            possible_roots_map = {}
            for possible_root, pattern_lemma_msa in possible_roots:
                possible_roots_map[re.sub(r'[wy]', '#', possible_root)] = (possible_root, pattern_lemma_msa)
            possible_roots_map_set = set(possible_roots_map)
            possible_roots_gold_lookup = possible_roots_map_set & gold_roots
            if possible_roots_gold_lookup:
                possible_roots_filtered = filter_roots([possible_roots_map[root] for root in possible_roots_gold_lookup])
                if not possible_roots_filtered:
                    roots_egy.append((row['LEMMA'], '', 'NO_ROOT', '', ''))
                else:
                    roots = [root for root, _ in possible_roots_filtered]
                    pattern_lemma_msa = [pattern for _, pattern in possible_roots_filtered][0]
                    proposed_pattern, num_roots = get_pattern(row['LEMMA'], lemma_dediac_norm_egy, pattern_lemma_msa, roots)
                    roots_counter.update(roots)
                    roots_egy.append(
                        (row['LEMMA'], roots, f'BACKOFF-{num_roots}', pattern_lemma_msa, proposed_pattern))
            else:
                possible_roots = [(root, pattern) for root, pattern in possible_roots if any(c != '#' for c in root.split('.'))]
                possible_roots = filter_roots(possible_roots)
                if not possible_roots:
                    roots_egy.append((row['LEMMA'], '', 'NO_ROOT', '', ''))
                else:
                    roots = [root for root, _ in possible_roots]
                    pattern_lemma_msa = [pattern for _, pattern in possible_roots][0]
                    proposed_pattern, num_roots = get_pattern(row['LEMMA'], lemma_dediac_norm_egy, pattern_lemma_msa, roots)
                    roots_counter.update(roots)
                    roots_egy.append((row['LEMMA'], roots, f'BACKOFF_NO_GOLD_ROOT-{num_roots}', pattern_lemma_msa, proposed_pattern))
        else:
            roots_egy.append((row['LEMMA'], '', 'NO_ROOT', '', ''))

with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/pal_roots.tsv', 'w') as f:
    # print('LEMMA', 'ROOT', 'STATUS', sep='\t', file=f)
    print('ROOT', 'STATUS', 'PATTERN_LEMMA_PROPOSED', sep='\t', file=f)
    # print('LEMMA', 'ROOT', 'STATUS', 'PATTERN_LEMMA_PROPOSED', sep='\t', file=f)
    # print('ROOT', sep='\t', file=f)
    for x in roots_egy:
        # print(x[0], ' '.join(x[1]), x[2], sep='\t', file=f)
        print(' '.join(x[1]), x[2], x[4], sep='\t', file=f)
        # print(x[0], ' '.join(x[1]), x[2], x[4], sep='\t', file=f)
        # print(' '.join(x[1]), sep='\t', file=f)
c = 0
