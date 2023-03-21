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

COND_T_CLASSES = ['MS', 'MD', 'MP', 'FS', 'FD', 'FP']
FUNC_GEN_FROM_FORM_RE = re.compile(r'([MF])[SDP](?=$|\|| )')
FUNC_NUM_FROM_FORM_RE = re.compile(r'[MF]([SDP])(?=$|\|| )')
FORM_FEATS_RE = re.compile(r'[MF][SDP](?=$|\|\|| )')

def generate_row(cond_s='', variant='', entry_type='', freq='', lemma='',
                 lemma_bw='', stem='', pos='', ms='', md='', mp='', fs='',
                 fd='', fp='', other_lemmas=''):
    return [cond_s, variant, entry_type, freq, lemma, lemma_bw, stem,
            pos, ms, md, mp, fs, fd, fp, other_lemmas]


def generate_table(lexicon, pos, dialect, pos2lemma2prob, db_lexprob):
    cond_s2cond_t2feats2rows = _get_structured_lexicon_classes(lexicon)
    
    pos2lemma2prob_ = None
    if pos2lemma2prob is not None or db_lexprob is not None:
        pos2lemma2prob_ = _get_pos2lemma2prob(db_lexprob, pos, cond_s2cond_t2feats2rows, pos2lemma2prob)
    
    table = [generate_row(cond_s='CLASS', variant='VAR', entry_type='Entry', freq='Freq', lemma='Lemma',
                          lemma_bw='Lemma_bw', stem='Stem', pos='POS', ms='MS', md='MD', mp='MP', fs='FS',
                          fd='FD', fp='FP', other_lemmas='Other lemmas')]
    col_pos, col_MS, col_freq = table[0].index('POS'), table[0].index('MS'), table[0].index('Freq')
    for cond_s, cond_t2feats2rows in cond_s2cond_t2feats2rows.items():
        form_feats2rows_cond_s = {}
        for cond_t, feats2rows in cond_t2feats2rows.items():
            for feats, rows in feats2rows.items():
                gen, num, pos_ = feats
                if pos2lemma2prob_ is not None:
                    best_index = (-np.array([pos2lemma2prob_[pos_][re.sub(r'[aiuo]', '', info['LEMMA'])]
                                                for info in rows])).argsort()[:len(rows)][0]
                else:
                    best_index = 0

                row = rows[best_index]
                other_lemmas = [row['LEMMA'] for row in rows[:best_index] + rows[best_index + 1:]][:20]
                lemma, stem = bw2ar(strip_lex(row['LEMMA'])), bw2ar(row['FORM'])

                form_feat2case2info, cases = get_suffixes_across_form_feats(lemma, stem, gen, num, pos_, cond_t,
                                                                            actpart=True if 'ACTPART' in cond_s else False)
                for case in cases:
                    form_feats_ = []
                    for case2info in form_feat2case2info:
                        form_feats_.append(case2info[case]['suffixes'] if case2info[case] else '')
                    
                    row_form_feats = form_feats2rows_cond_s.setdefault(
                        tuple(form_feats_), {}).setdefault(
                            case, {}).setdefault(
                                feats, generate_row(
                                    cond_s=cond_s, variant=dialect,
                                    entry_type=f'Example (cas:{case})' if case in 'nag' else 'Example', 
                                    freq=str(len(rows)), lemma=lemma, lemma_bw=ar2bw(lemma), stem=stem,
                                    other_lemmas=' '.join(other_lemmas)))
                    
                    for col_form_feat, case2info in enumerate(form_feat2case2info):
                        if case2info[case]:
                            row_form_feats[col_MS + col_form_feat] = case2info[case]['content']
                            row_form_feats[col_pos] = case2info[case]['pos']
        
        form_feats_cluster_map = get_form_feats_cluster_map(set(form_feats2rows_cond_s))
        
        form_feats2rows_cond_s_refactored = {}
        for form_feats, case2rows in form_feats2rows_cond_s.items():
            key = form_feats_cluster_map[form_feats] if form_feats in form_feats_cluster_map else form_feats
            form_feats2rows_cond_s_refactored.setdefault(key, []).append(case2rows)

        table_ = []
        for form_feats, rows in form_feats2rows_cond_s_refactored.items():
            table_.append(generate_row(cond_s=cond_s if cond_s else '-', variant=dialect, entry_type='Definition'))
            table_.append(generate_row(cond_s=cond_s if cond_s else '-', variant=dialect, entry_type='Suffixes',
                                       lemma='**', lemma_bw='**', stem='X', ms=form_feats[0], md=form_feats[1],
                                       mp=form_feats[2], fs=form_feats[3], fd=form_feats[4], fp=form_feats[5]))
            row2case = {}
            for case2feats2row in rows:
                for case, feats2row in case2feats2row.items():
                    for feats, row in feats2row.items():
                        row2case[tuple(row)] = case

            for row, case in sorted(row2case.items(), key=lambda x: int(x[0][col_freq]), reverse=True):
                table_.append([c if c else '-' for c in row])

        table += table_
    
    return table


def _get_structured_lexicon_classes(lexicon):
    cond_s2cond_t2feats2rows = {}
    for _, row in lexicon.iterrows():
        
        cond_t, cond_s = row['COND-T'], row['COND-S']
        cond_t_dict = {'form': [], 'other': []}
        for conds_ in cond_t.split():
            cond_t_dict['form' if FORM_FEATS_RE.search(conds_) else 'other'].append(conds_)
        
        cond_t_form = cond_t_dict['form']
        assert len(cond_t_form) == 1
        cond_t_form = '||'.join(cond for cond in sorted(cond_t_form[0].split('||')) if cond)
        cond_s_pp = cond_s
        if cond_t_dict['other']:
            cond_s_pp += (' ' if cond_s_pp else '') + f"COND-T:{' '.join(sorted(cond_t_dict['other']))}"

        feats = {feat.split(':')[0]: feat.split(':')[1] for feat in row['FEAT'].split()}
        
        gen = feats.get('gen', '')
        if not gen:
            genders = FUNC_GEN_FROM_FORM_RE.findall(cond_t_form)
            if len(set(genders)) == 1:
                gen = genders[0].lower()

        num = feats.get('num', '')
        if not num:
            numbers = FUNC_NUM_FROM_FORM_RE.findall(cond_t_form)
            if len(set(numbers)) == 1:
                num = numbers[0].lower()
        
        pos_ = re.search(r'pos:(\S+)', row['FEAT']).group(1)
        row_cond_s = {'LEMMA': row['LEMMA'], 'FORM': row['FORM']}
        cond_s2cond_t2feats2rows.setdefault(cond_s_pp, {}).setdefault(cond_t_form, {}).setdefault((gen, num, pos_), []).append(row_cond_s)
    
    return cond_s2cond_t2feats2rows

def get_suffixes_across_form_feats(lemma, stem, gen, num, pos_, cond_t, actpart):
    form_feat2case2info = [{'n': '', 'a': '', 'g': '', 'u': ''} for _ in range(6)]
    cases_seen = set()
    for cond_t_ in cond_t.split('||'):
        analyses, analyses_enc = _generate_analyses(lemma, stem, gen, num, pos_, cond_t_, actpart)
        cases = set([a[0]['cas'] for a in analyses + analyses_enc if a[0] is not None])
        cases_seen.update(cases)
        cases = cases if cases else set(['-'])

        col_form_feat = COND_T_CLASSES.index(cond_t_)
        suffix, suffix_enc = {'diac': 'CHECK-ZERO'}, {'diac': 'CHECK-ZERO'}
        for case in cases:
            multiple_gen, a_pos_ = [], []
            for (a, _, _, suff_cat), (a_enc, _, _, suff_cat_enc) in zip(analyses, analyses_enc):
                if a is not None and a_enc is not None:
                    assert a['cas'] == a_enc['cas'] and a['gen'] == a_enc['gen'] and \
                        a['num'] == a_enc['num'] and a['pos'] == a_enc['pos']
                
                if not ((a is None or a['cas'] == case) and (a_enc is None or a_enc['cas'] == case)):
                    continue
                a_diac = a['diac'] if a is not None else 'CHECK'
                a_enc_diac = a_enc['diac'] if a_enc is not None else 'CHECK'
                a_gen = a_enc['gen'] if a_enc is not None else a['gen']
                a_num = a_enc['num'] if a_enc is not None else a['num']
                multiple_gen.append(f'{a_diac} / {a_enc_diac} ({a_gen}{a_num})')
                a_pos_.append(a['pos'].upper() if a is not None else a_enc['pos'].upper())

                if suff_cat is not None:
                    suffixes = db.suffix_cat_hash[suff_cat]
                    if len(suffixes) == 1:
                        suffix = suffixes[0]
                    else:
                        suffix = {'diac': 'CHECK-GT-1'}
                else:
                    suffix = {'diac': 'CHECK-ZERO'}
                
                if suff_cat_enc is not None:
                    suffixes_enc = [suff for suff in db.suffix_cat_hash[suff_cat_enc]
                                    if suff['enc0'] == '3ms_poss']
                    if len(suffixes_enc) == 1:
                        suffix_enc = suffixes_enc[0]
                    else:
                        suffix_enc = {'diac': 'CHECK-GT-1'}
                else:
                    suffix_enc = {'diac': 'CHECK-ZERO'}

                break
            
            form_feat2case2info[col_form_feat][case] = {
                'suffixes': ' / '.join(f"X+{ar2bw(suff) if suff else 'Ø'}" for suff in (suffix['diac'], suffix_enc['diac'])),
                'content':' // '.join(multiple_gen),
                'pos': ' / '.join(a_pos_)
            }
    
    return form_feat2case2info, cases_seen

def _generate_analyses(lemma, stem, gen, num, pos_, cond_t_, actpart):
    feats_, feats_enc_ = _process_and_generate_feats(gen, num, pos_, cond_t_, actpart)
    analyses, _ = generator.generate(lemma, feats_, debug=True)
    analyses_enc, _ = generator.generate(lemma, feats_enc_, debug=True)
    analyses = [a for a in analyses if a[0]['stem'] == stem]
    analyses_enc = [a for a in analyses_enc if a[0]['stem'] == stem]

    if len(analyses) > len(analyses_enc):
        for _ in range(len(analyses) - len(analyses_enc)):
            analyses_enc.append((None, None, None, None))
    elif len(analyses) < len(analyses_enc):
        for _ in range(len(analyses_enc) - len(analyses)):
            analyses.append((None, None, None, None))
        
    analyses = sorted(analyses, key=lambda a: a[0]['cas'] if a[0] is not None else 'z')
    analyses_enc = sorted(analyses_enc, key=lambda a: a[0]['cas'] if a[0] is not None else 'z')

    return analyses, analyses_enc

def get_form_feats_cluster_map(form_feats_set):
    slots2form_feats = {}
    for form_feats in form_feats_set:
        slots2form_feats.setdefault(sum(1 for suff in form_feats if suff), []).append(form_feats)
    
    clusters = {}
    already_clustered = set()
    for i in reversed(range(6)):
        i += 1
        if not slots2form_feats.get(i):
            continue
        for form_feats_len_i in slots2form_feats[i]:
            already_clustered.add(form_feats_len_i)
            remaining_form_feats = form_feats_set - already_clustered
            for remaining_form_feats_ in remaining_form_feats:
                if all(suff == form_feats_len_i[j] for j, suff in enumerate(remaining_form_feats_) if suff):
                    clusters.setdefault(form_feats_len_i, []).append(remaining_form_feats_)
                    already_clustered.add(remaining_form_feats_)
    
    cluster_map = {}
    for form_feats, form_feats_mapped in clusters.items():
        for form_feats_ in form_feats_mapped:
            cluster_map[form_feats_] = form_feats
    
    return cluster_map

def _process_and_generate_feats(gen, num, pos, cond_t, actpart):
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
    feats_enc_ = {**feats_, **{'enc0': '3ms_poss' if not actpart else '3ms_dobj'}}
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
            pos2lemma2prob[pos_] if pos2lemma2prob else None)
    
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
    parser.add_argument("-most_prob_lemma", default=False, action='store_true',
                        help="Whether or not to use the most probable lemma.")
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

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.generator import Generator
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.utils import strip_lex

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    output_dir = args.output_dir if args.output_dir \
                    else os.path.join(config_global['debugging'], config_global['docs_tables_dir'])
    output_dir = os.path.join(output_dir, f"camel-morph-{config_local['dialect']}")
    os.makedirs(output_dir, exist_ok=True)

    db_name = config_local['db']
    db_dir = config_global['db_dir']
    db_dir = os.path.join(db_dir, f"camel-morph-{config_local['dialect']}")
    db = MorphologyDB(os.path.join(db_dir, db_name), flags='gd')
    generator = Generator(db)

    SHEETS, _ = db_maker_utils.read_morph_specs(config, config_name, process_morph=False, lexicon_cond_f=False)
    lexicon = SHEETS['lexicon']
    lexicon['LEMMA'] = lexicon.apply(lambda row: re.sub('lex:', '', row['LEMMA']), axis=1)
    assert lexicon['COND-T'].str.contains(r'((MS|MD|MP|FS|FD|FP)(\|\|)?)+', regex=True).values.tolist()

    pos_type = args.pos_type if args.pos_type else config_local['pos_type']
    if pos_type == 'verbal':
        pos = 'verb'
    elif pos_type == 'nominal':
        pos = args.pos if args.pos else config_local.get('pos')
    elif pos_type == 'other':
        pos = args.pos
    
    assert all(lexicon['FEAT'].str.contains(r'pos:\S+', regex=True))
    if pos:
        lexicon = lexicon[lexicon['FEAT'].str.contains(f'pos:{pos}\\b', regex=True)]
        pos = [pos]
    else:
        pos = list(set([x[0].split(':')[1] for x in lexicon['FEAT'].str.extract(r'(pos:\S+)').values.tolist()]))

    pos2lemma2prob, db_lexprob = None, None
    if args.most_prob_lemma:
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

    outputs = generate_table(lexicon, pos, config_local['dialect'].upper(), pos2lemma2prob, db_lexprob)

    output_name = args.output_name if args.output_name else config_local['docs_debugging']['docs_tables']
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output, sep='\t', file=f)

