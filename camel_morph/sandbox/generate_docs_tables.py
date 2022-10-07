import re
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    from .. import db_maker_utils
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph import db_maker_utils
    from camel_morph.debugging.create_repr_lemmas_list import get_lemma2prob


def generate_table(lexicon, pos, dialect, pos2lemma2prob, db_lexprob):
    cond_s2cond_t2feats2rows = {}
    for _, row in lexicon.iterrows():
        cond_t, cond_s = '||'.join(sorted(row['COND-T'].split('||'))), row['COND-S']
        feats = {feat.split(':')[0]: feat.split(':')[1] for feat in row['FEAT'].split()}
        gen = feats.get('gen', '')
        num = feats.get('num', '')
        pos_ = re.search(r'pos:(\S+)', row['FEAT']).group(1)
        row_cond_s = {'LEMMA': row['LEMMA'], 'FORM': row['FORM']}
        cond_s2cond_t2feats2rows.setdefault(cond_s, {}).setdefault(cond_t, {}).setdefault((gen, num, pos_), []).append(row_cond_s)
    
    pos2lemma2prob_ = _get_pos2lemma2prob(db_lexprob, pos, cond_s2cond_t2feats2rows, pos2lemma2prob)
    
    errors = []
    table = [['CLASS', 'VAR', 'Entry', 'Freq', 'Lemma', 'Stem', 'POS', 'MS', 'MD', 'MP', 'FS', 'FD', 'FP', 'Other lemmas']]
    for cond_s, cond_t2feats2rows in cond_s2cond_t2feats2rows.items():
        table.append([cond_s if cond_s else '-', dialect, 'Definition', '', '', '', '', '', '', '', '', '', '', ''])
        paradigm_ = {}
        rows_cond_s = {}
        for cond_t, feats2rows in cond_t2feats2rows.items():
            for feats, rows in feats2rows.items():
                gen, num, pos_ = feats
                for cond_t_ in cond_t.split('||'):
                    best_index = (-np.array([pos2lemma2prob_[pos_][re.sub(r'[aiuo]', '', info['LEMMA'])]
                                        for info in rows])).argsort()[:len(rows)][0]
                    row = rows[best_index]
                    other_lemmas = [row['LEMMA'] for row in rows[:best_index] + rows[best_index + 1:]][:20]
                    lemma, stem = bw2ar(strip_lex(row['LEMMA'])), bw2ar(row['FORM'])
                    feats_, feats_enc_ = _process_and_generate_feats(gen, num, pos_, cond_t_)
                    analyses, _ = generator.generate(lemma, feats_, debug=True)
                    analyses_enc, _ = generator.generate(lemma, feats_enc_, debug=True)
                    analyses = [a for a in analyses if a[0]['stem'] == stem]
                    analyses_enc = [a for a in analyses_enc if a[0]['stem'] == stem]

                    if len(analyses) > len(analyses_enc):
                        analyses_enc_ = []
                        for _ in range(len(analyses) - len(analyses_enc)):
                            analyses_enc_.append((None, None, None, None))
                        analyses_enc = analyses_enc_
                    elif len(analyses) < len(analyses_enc):
                        analyses_ = []
                        for _ in range(len(analyses_enc) - len(analyses)):
                            analyses_.append((None, None, None, None))
                        analyses = analyses_
                        
                    analyses = sorted(analyses, key=lambda a: a[0]['cas'] if a[0] is not None else 'z')
                    analyses_enc = sorted(analyses_enc, key=lambda a: a[0]['cas'] if a[0] is not None else 'z')

                    col, col_pos = table[0].index(cond_t_), table[0].index('POS')
                    cases = set([a[0]['cas'] for a in analyses + analyses_enc if a[0] is not None])
                    cases = cases if cases else set(['-'])
                    suffixes = {}
                    for case in cases:
                        row_cond_s = rows_cond_s.setdefault(lemma, {}).setdefault(
                            case, [cond_s, dialect, f'Example (cas:{case})' if case in 'ang' else 'Example', 
                            str(len(rows)), lemma, stem, '', '', '', '', '', '', '', ' '.join(other_lemmas)])
                        multiple_gen, a_pos_ = [], []
                        for (a, _, _, suff_cat), (a_enc, _, _, suff_cat_enc) in zip(analyses, analyses_enc):
                            if (a is None or a['cas'] == case) and (a_enc is None or a_enc['cas'] == case):
                                a_diac = a['diac'] if a is not None else 'CHECK'
                                a_enc_diac = a_enc['diac'] if a_enc is not None else 'CHECK'
                                a_gen = a_enc['gen'] if a_enc is not None else a['gen']
                                a_num = a_enc['num'] if a_enc is not None else a['num']
                                multiple_gen.append(f'{a_diac} / {a_enc_diac} ({a_gen}{a_num})')
                                a_pos_.append(a['pos'].upper() if a is not None else a_enc['pos'].upper())

                                suffixes = {}
                                if suff_cat is not None:
                                    for suff in db.suffix_cat_hash[suff_cat]:
                                        suffixes.setdefault('', []).append(ar2bw(suff['diac']))
                                else:
                                    suffixes.setdefault('', []).append('CHECK')
                                if suff_cat_enc is not None:
                                    for suff in db.suffix_cat_hash[suff_cat_enc]:
                                        if suff['enc0'] == '3ms_poss':
                                            suffixes.setdefault('enc', []).append(ar2bw(suff['diac']))
                                else:
                                    suffixes.setdefault('enc', []).append('CHECK')
                                
                        row_cond_s[col] = ' // '.join(multiple_gen)
                        row_cond_s[col_pos] = ' / '.join(a_pos_)
                        for k, v in suffixes.items():
                            paradigm_.setdefault(case, {}).setdefault(cond_t_, {}).setdefault(k, set()).add('/'.join(v))
        
        for case, cond_t2suffixes in paradigm_.items():
            table.append([cond_s if cond_s else '-', dialect, f'Paradigm (cas:{case})' if case in 'ang' else 'Paradigm', '', '**', 'X', '', '', '', '', '', '', '', ''])
            for cond_t_, suffixes in cond_t2suffixes.items():
                col = table[0].index(cond_t_)
                cell = []
                for k in ['', 'enc']:
                    if len(suffixes[k]) > 1:
                        cell.append('X' + '+{' + ', '.join(v if v else 'Ø' for v in suffixes[k]) + '}')
                    elif len(suffixes[k]) == 1:
                        cell_ = list(suffixes[k])[0]
                        cell.append('X+' + (cell_ if cell_ else 'Ø'))
                    else:
                        cell.append('CHECK')
                
                table[-1][col] = ' / '.join(cell)
        
        if not paradigm_:
            table.append([cond_s if cond_s else '-', dialect, f'Paradigm (cas:{case})' if case in 'ang' else 'Paradigm', '', '**', 'X', '', '', '', '', '', '', '', ''])

        table += [[c if c else '-' for c in row]
                    for row in sorted([row_case for row_ in rows_cond_s.values() for row_case in row_.values()],
                                      key=lambda x: int(x[3]), reverse=True)]
    
    return table

def _process_and_generate_feats(gen, num, pos, cond_t):
    feats_ = {'pos': pos, 'stt': 'i'}
    if gen not in ['', '-']:
        feats_.update({'gen': gen})
    else:
        feats_.update({'gen': cond_t[0].lower()})
    if num not in ['', '-']:
        feats_.update({'num': num})
    else:
        feats_.update({'num': cond_t[1].lower()})
    
    feats_ = {k: '' if v == '-' else v for k, v in feats_.items()}
    feats_enc_ = {**feats_, **{'enc0': '3ms_poss'}}
    feats_enc_['stt'] = 'c'

    return feats_, feats_enc_


def _get_pos2lemma2prob(db_lexprob, pos, cond_s2cond_t2feats2rows, pos2lemma2prob):
    unique_lemma_classes = {(cond_s, cond_t, feats): {'lemmas': [{'lemma': row['LEMMA']} for row in rows]}
                            for cond_s, cond_t2feats2rows in cond_s2cond_t2feats2rows.items()
                            for cond_t, feats2rows in cond_t2feats2rows.items()
                            for feats, rows in feats2rows.items()}
    pos2lemma2prob_ = {}
    for pos_ in pos:
        pos2lemma2prob_[pos_] = get_lemma2prob(
            pos_, db_lexprob, {k: v for k, v in unique_lemma_classes.items() if k[2][2] == pos_},
            pos2lemma2prob[pos_] if pos2lemma2prob else None,
            strip_lex, bw2ar, DEFAULT_NORMALIZE_MAP)
    
    return pos2lemma2prob_




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='default_config', nargs='+',
                        type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
    parser.add_argument("-output_dir", default='',
                        type=str, help="Path of the directory to output the lemmas to.")
    parser.add_argument("-output_name", default='',
                        type=str, help="Name of the file to output the representative lemmas to. File will be placed in a directory called conjugation/repr_lemmas/")
    parser.add_argument("-db", default='',
                        type=str, help="Name of the DB file which will be used for the retrieval of lexprob.")
    parser.add_argument("-db_dir", default='',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-lexprob",  default='',
                        type=str, help="Custom lexical probabilities file which contains two columns (lemma, frequency).")
    parser.add_argument("-pos", default='',
                        type=str, help="POS of the lemmas.")
    parser.add_argument("-pos_type", default='', choices=['verbal', 'nominal', ''],
                        type=str, help="POS type of the lemmas.")
    parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    with open(args.config_file) as f:
        config = json.load(f)
    config_name = args.config_name[0]
    config_local = config['local'][config_name]
    config_global = config['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.morphology.analyzer import DEFAULT_NORMALIZE_MAP
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.generator import Generator
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.utils import strip_lex

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    db_name = config_local['db']
    db_dir = config_global['db_dir']
    db_dir = os.path.join(db_dir, f"camel-morph-{config_local['dialect']}")
    db = MorphologyDB(os.path.join(db_dir, db_name), flags='gd')
    generator = Generator(db)

    SHEETS, _ = db_maker_utils.read_morph_specs(config, config_name, process_morph=False)
    lexicon = SHEETS['lexicon']
    lexicon['LEMMA'] = lexicon.apply(lambda row: re.sub('lex:', '', row['LEMMA']), axis=1)
    lexicon = lexicon[lexicon['COND-T'].str.contains(r'^((MS|MD|MP|FS|FD|FP)(\|\|)?)+$', regex=True)]

    pos_type = args.pos_type if args.pos_type else config_local['pos_type']
    if pos_type == 'verbal':
        pos = 'verb'
    elif pos_type == 'nominal':
        pos = args.pos if args.pos else config_local.get('pos')
    elif pos_type == 'other':
        pos = args.pos
    if pos:
        lexicon = lexicon[lexicon['FEAT'].str.contains(f'pos:{pos}\\b', regex=True)]
        pos = [pos]
    else:
        lexicon = lexicon[lexicon['FEAT'].str.contains(r'pos:\S+', regex=True)]
        pos = list(set([x[0].split(':')[1] for x in lexicon['FEAT'].str.extract(r'(pos:\S+)').values.tolist()]))

    pos2lemma2prob, db_lexprob = None, None
    if config_local.get('lexprob'):
        with open(config_local['lexprob']) as f:
            freq_list_raw = f.readlines()
            if len(freq_list_raw[0].split('\t')) == 2:
                pos2lemma2prob = dict(map(lambda x: (x[0], int(x[1])),
                                    [line.strip().split('\t') for line in freq_list_raw]))
                pos2lemma2prob = {'verb': pos2lemma2prob}
            elif len(freq_list_raw[0].split('\t')) == 3:
                pos2lemma2prob = {}
                for line in freq_list_raw:
                    line = line.strip().split('\t')
                    lemmas = pos2lemma2prob.setdefault(line[1], {})
                    lemmas[line[0]] = int(line[2])
            else:
                raise NotImplementedError
        
        for pos_ in pos:
            total = sum(pos2lemma2prob[pos_].values())
            pos2lemma2prob[pos_] = {lemma: freq / total for lemma, freq in pos2lemma2prob[pos_].items()}
    elif args.db:
        db_lexprob = MorphologyDB(args.db)
    else:
        db_lexprob = MorphologyDB.builtin_db()

    table = generate_table(lexicon, pos, config_local['dialect'].upper(), pos2lemma2prob, db_lexprob)

    with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/docs_table.tsv', 'w') as f:
        for row in table:
            print(*row, sep='\t', file=f)

