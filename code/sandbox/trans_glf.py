import re

import pandas as pd

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex
from camel_tools.utils.dediac import dediac_bw
from sklearn.model_selection import train_test_split

DEFAULT_NORMALIZE_MAP = CharMapper({u'\u0671': u'\u0627'})

bw2ar = CharMapper.builtin_mapper('bw2ar')

glf = pd.read_csv('data/GLF-LEX-CV.csv')
egy = pd.read_csv('data/EGY-LEX-PV.csv')
msa = pd.read_csv('data/MSA-LEX-PV.csv')

def process_lemma(lemma):
    return DEFAULT_NORMALIZE_MAP(bw2ar(re.sub(r'[aoiu]', '', strip_lex(lemma))))


lemma2trans_msa, lemma2trans_msa_unnorm  = {}, {}
for _, row in msa.iterrows():
    lemma = process_lemma(row['LEMMA'])
    trans = 'trans' if re.search(r'\btrans\b', row['COND-S']) else 'intrans'
    lemma2trans_msa.setdefault(lemma, []).append(trans)
    lemma2trans_msa_unnorm.setdefault(row['LEMMA'], []).append(trans)

mixed_trans_msa = {lemma: trans for lemma, trans in lemma2trans_msa_unnorm.items() if set(trans) == {'trans', 'intrans'}}

lemma2trans_egy = {}
for _, row in egy.iterrows():
    lemma = process_lemma(row['LEMMA'])
    lemma2trans_egy.setdefault(lemma, []).append('trans' if re.search(r'\btrans\b', row['COND-S']) else 'intrans')

normalized2lemmas_glf = {}
for _, row in glf.iterrows():
    lemma_norm = process_lemma(row['LEMMA'])
    normalized2lemmas_glf.setdefault(lemma_norm, []).append(bw2ar(row['LEMMA']))

transitivity = []
for _, row in glf.iterrows():
    lemma, gloss = row['LEMMA'], row['GLOSS']
    lemma_processed = process_lemma(lemma)
    intrans_score = 0
    if lemma.startswith('Ain'):
        intrans_score += 1
    elif lemma_processed in lemma2trans_egy:
        if set(lemma2trans_egy[lemma_processed]) == {'intrans'}:
            intrans_score += 4
    elif lemma_processed in lemma2trans_msa:
        if set(lemma2trans_msa[lemma_processed]) == {'intrans'}:
            intrans_score += 3

    evidence = re.findall(r'(to)?\bbe\b', gloss)
    if evidence:
        counter_evidence = re.findall(r'to (?!be)', gloss)
        intrans_score += (len(evidence) * 2 - len(counter_evidence) * 1)
    
    if intrans_score > 0:
        transitivity.append(('intrans', intrans_score))
    else:
        transitivity.append(('trans', intrans_score))

with open('sandbox/trans_glf.tsv', 'w') as f:
    for t in transitivity:
        print(*t, sep='\t', file=f)
pass
