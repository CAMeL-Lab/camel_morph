import re
from tqdm import tqdm

import pandas as pd
from numpy import nan

from camel_tools.utils.charmap import CharMapper

normalize_map = CharMapper({
    '<': 'A',
    '>': 'A',
    '|': 'A',
    '{': 'A',
    'Y': 'y'
})

def preprocess_func(form):
    form = normalize_map(form)
    form = re.sub(r"[uioa~]", '', form)
    return form

data_msa = pd.read_csv(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/data/camel-morph-msa/nom_msa_red/MSA-Nom-LEX.csv')
data_egy = pd.read_csv(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/data/camel-morph-egy/nom_egy_red/EGY-Nom-LEX.csv')
data_msa = data_msa.replace(nan, '', regex=True)
data_egy = data_egy.replace(nan, '', regex=True)

lemma2root_msa = {}
backoff = {}
for _, row in tqdm(data_msa.iterrows(), total=len(data_msa.index)):
    lemma_dediac_norm = preprocess_func(row['LEMMA'])
    lemma2root_msa.setdefault(lemma_dediac_norm, set()).add(row['ROOT'])
    pattern_dediac_norm = preprocess_func(row['PATTERN_LEMMA'])
    match_pattern = ''.join('([^Ap])' if p.isdigit() else p for p in pattern_dediac_norm)
    regexes = backoff.setdefault(len(pattern_dediac_norm), {})
    pattern_dediac_norm_digits = re.sub(r'\D', '', pattern_dediac_norm)
    regexes.setdefault(match_pattern, set()).add(pattern_dediac_norm_digits)


data_egy = data_egy[data_egy['LEMMA'] == '*awqiy~ap']
roots_egy = []
for _, row in tqdm(data_egy.iterrows(), total=len(data_egy.index)):
    if not row['LEMMA']:
        roots_egy.append((row['LEMMA'], '', 'NO_LEMMA'))
        continue
    lemma_dediac_norm = preprocess_func(row['LEMMA'])
    if lemma_dediac_norm in lemma2root_msa:
        roots_egy.append((row['LEMMA'], list(lemma2root_msa[lemma_dediac_norm]), 'LOOKUP'))
    else:
        possibilities = []
        if len(lemma_dediac_norm) not in backoff:
            roots_egy.append((row['LEMMA'], '', 'BACKOFF_NO_LEN'))
            continue
        for regex, patterns_dediac_norm_digits in backoff[len(lemma_dediac_norm)].items():
            match = re.match(f'^{regex}$', lemma_dediac_norm)
            if match:
                for pattern in patterns_dediac_norm_digits:
                    root = []
                    for i, digit in enumerate(pattern):
                        root.append('#' if int(digit) != i + 1 else match.groups()[i])
                    if 2 < len(root) < 6:
                        possibilities.append('.'.join(root))
        if possibilities:
            roots_egy.append((row['LEMMA'], list(set(possibilities)), 'BACKOFF'))
        else:
            roots_egy.append((row['LEMMA'], '', 'NO_ROOT'))

with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/egy_nom_roots.tsv', 'w') as f:
    print('LEMMA', 'ROOT', 'STATUS', sep='\t', file=f)
    for x in roots_egy:
        print(x[0], ' '.join(x[1]), x[2], sep='\t', file=f)
c = 0
