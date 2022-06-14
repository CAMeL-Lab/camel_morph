import json
import re
import pandas as pd
import os
from numpy import nan
from collections import Counter
from pprint import pprint

import camel_morph.db_maker_utils as db_maker_utils

valid_morphemes = {
    'msa': {
        '[QUES]': 'prc',
        '[CONJ]': 'prc',
        '[VPART.P]': 'prc',
        '[VPART.I]': 'prc',
        '[VPART.J]': 'prc',
        '[VPART.S]': 'prc',
        '[VPART.E]': 'prc',
        '[PVBuff]': 'post-buff',
        '[PVSuff]': 'suff',
        '[IVPref.1S]': 'pref',
        '[IVPref.1P]': 'pref',
        '[IVPref.2MS]': 'pref',
        '[IVPref.2MP]': 'pref',
        '[IVPref.2FS]': 'pref',
        '[IVPref.2FP]': 'pref',
        '[IVPref.2D]': 'pref',
        '[IVPref.3MS]': 'pref',
        '[IVPref.3MD]': 'pref',
        '[IVPref.3MP]': 'pref',
        '[IVPref.3FS]': 'pref',
        '[IVPref.3FD]': 'pref',
        '[IVPref.3FP]': 'pref',
        '[IVPreBuff]': 'pre-buff',
        '[IVBuff]': 'post-buff',
        '[IVSuff.I.1P]': 'suff',
        '[IVSuff.I.1S]': 'suff',
        '[IVSuff.I.2D]': 'suff',
        '[IVSuff.I.2FP]': 'suff',
        '[IVSuff.I.2FS]': 'suff',
        '[IVSuff.I.2MP]': 'suff',
        '[IVSuff.I.2MS]': 'suff',
        '[IVSuff.I.3FD]': 'suff',
        '[IVSuff.I.3FP]': 'suff',
        '[IVSuff.I.3FS]': 'suff',
        '[IVSuff.I.3MD]': 'suff',
        '[IVSuff.I.3MP]': 'suff',
        '[IVSuff.I.3MS]': 'suff',
        '[IVSuff.J.1P]': 'suff',
        '[IVSuff.J.1S]': 'suff',
        '[IVSuff.J.2D]': 'suff',
        '[IVSuff.J.2FP]': 'suff',
        '[IVSuff.J.2FS]': 'suff',
        '[IVSuff.J.2MP]': 'suff',
        '[IVSuff.J.2MS]': 'suff',
        '[IVSuff.J.3FD]': 'suff',
        '[IVSuff.J.3FP]': 'suff',
        '[IVSuff.J.3FS]': 'suff',
        '[IVSuff.J.3MD]': 'suff',
        '[IVSuff.J.3MP]': 'suff',
        '[IVSuff.J.3MS]': 'suff',
        '[IVSuff.S.1P]': 'suff',
        '[IVSuff.S.1S]': 'suff',
        '[IVSuff.S.2D]': 'suff',
        '[IVSuff.S.2FP]': 'suff',
        '[IVSuff.S.2FS]': 'suff',
        '[IVSuff.S.2MP]': 'suff',
        '[IVSuff.S.2MS]': 'suff',
        '[IVSuff.S.3FD]': 'suff',
        '[IVSuff.S.3FP]': 'suff',
        '[IVSuff.S.3FS]': 'suff',
        '[IVSuff.S.3MD]': 'suff',
        '[IVSuff.S.3MP]': 'suff',
        '[IVSuff.S.3MS]': 'suff',
        '[IVSuff.E.1P]': 'suff',
        '[IVSuff.E.1S]': 'suff',
        '[IVSuff.E.2D]': 'suff',
        '[IVSuff.E.2FP]': 'suff',
        '[IVSuff.E.2FS]': 'suff',
        '[IVSuff.E.2MP]': 'suff',
        '[IVSuff.E.2MS]': 'suff',
        '[IVSuff.E.3FD]': 'suff',
        '[IVSuff.E.3FP]': 'suff',
        '[IVSuff.E.3FS]': 'suff',
        '[IVSuff.E.3MD]': 'suff',
        '[IVSuff.E.3MP]': 'suff',
        '[IVSuff.E.3MS]': 'suff',
        '[IVSuff.X.1P]': 'suff',
        '[IVSuff.X.1S]': 'suff',
        '[IVSuff.X.2D]': 'suff',
        '[IVSuff.X.2FP]': 'suff',
        '[IVSuff.X.2FS]': 'suff',
        '[IVSuff.X.2MP]': 'suff',
        '[IVSuff.X.2MS]': 'suff',
        '[IVSuff.X.3FD]': 'suff',
        '[IVSuff.X.3FP]': 'suff',
        '[IVSuff.X.3FS]': 'suff',
        '[IVSuff.X.3MD]': 'suff',
        '[IVSuff.X.3MP]': 'suff',
        '[IVSuff.X.3MS]': 'suff',
        '[CVBuff]': 'post-buff',
        '[CVSuff]': 'suff',
        '[CVSuff.E]': 'suff',
        '[CVSuff.X]': 'suff',
        '[PRON]': 'enc',
        '[PRON2]': 'enc'
    },
    'egy': {
        '[CONJ]': 'prc',
        '[PRENEG]': 'prc',
        '[VPART]': 'prc',
        '[PVBuff]': 'post-buff',
        '[PVSuff]': 'suff',
        '[IVPref.1S]': 'pref',
        '[IVPref.1P]': 'pref',
        '[IVPref.2MS]': 'pref',
        '[IVPref.2FS]': 'pref',
        '[IVPref.2P]': 'pref',
        '[IVPref.3MS]': 'pref',
        '[IVPref.3P]': 'pref',
        '[IVPref.3FS]': 'pref',
        '[IVPreBuff]': 'pre-buff',
        '[IVBuff]': 'post-buff',
        '[IVSuff.1P]': 'suff',
        '[IVSuff.1S]': 'suff',
        '[IVSuff.2MS]': 'suff',
        '[IVSuff.2FS]': 'suff',
        '[IVSuff.2P]': 'suff',
        '[IVSuff.3MS]': 'suff',
        '[IVSuff.3FS]': 'suff',
        '[IVSuff.3P]': 'suff',
        '[CVBuff]': 'post-buff',
        '[CVSuff]': 'suff',
        '[PRON]': 'enc',
        '[LPRON]': 'enc',
        '[NEG]': 'enc'
    }
}

with open('configs/config.json') as f:
    config = json.load(f)
    config_local_msa = config['local']['msa_cr']
    config_local_egy = config['local']['egy_cr']
    data_dir = config['global']['data_dir']

with open('/Users/chriscay/Downloads/suff_alomorphs.tsv') as f:
    salam = [line.strip() for line in f.readlines()]

morph_specs = {'msa': 'msa_cr', 'egy': 'egy_cr'}
cond_counts = {'msa': 'msa_cr', 'egy': 'egy_cr'}
for variant, config_name in morph_specs.items():
    SHEETS, _ = db_maker_utils.read_morph_specs(config, config_name)
    morph_specs[variant] = SHEETS

morph_counts = {}
for variant, sheets in morph_specs.items():
    morph, lexicon = sheets['morph'], sheets['lexicon']
    morph_counts_variant = morph_counts.setdefault(variant, {})
    morphemes = morph.drop_duplicates(subset=['CLASS', 'FUNC'])
    allomorphs = morph.drop_duplicates(subset=['CLASS', 'FUNC', 'FORM'])
    morph_counts_variant['morpheme'] = Counter(
        list(map(lambda x: valid_morphemes[variant][x], morphemes['CLASS'].values.tolist())))
    morph_counts_variant['allomorph'] = Counter(
        list(map(lambda x: valid_morphemes[variant][x], allomorphs['CLASS'].values.tolist())))
    
    cond_t = set([term for ct in morph['COND-T'].values.tolist() + lexicon['COND-T'].values.tolist()
                for cond in ct.split() for term in cond.split('||')])
    cond_s = set([term for cs in morph['COND-S'].values.tolist() + lexicon['COND-S'].values.tolist()
                for cond in cs.split() for term in cond.split('||')])
    valid_conditions = set([cond for cond in cond_t & cond_s if cond != '_'])
    morph_counts_variant['valid_conditions'] = {
        'count': len(valid_conditions), 'conditions': valid_conditions}

pprint(morph_counts)
