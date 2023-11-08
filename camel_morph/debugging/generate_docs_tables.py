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


class Row:
    INDEX2COL = ['cond_s', 'variant', 'entry_type', 'freq', 'cas',
                 'lemma', 'lemma_bw', 'stem', 'buffer', 'pos', 'ms', 'md',
                 'mp', 'fs', 'fd', 'fp', 'other_lemmas']
    COL2INDEX = {col: i for i, col in enumerate(INDEX2COL)}
    
    def __init__(self, cond_s='', variant='', entry_type='', freq='', lemma='',
                 lemma_bw='', stem='', buffer='', pos='', cas='', ms='', md='', mp='', fs='',
                 fd='', fp='', other_lemmas='') -> None:
        self.cond_s = cond_s
        self.variant = variant
        self.entry_type = entry_type
        self.freq = freq
        self.lemma = lemma
        self.lemma_bw = lemma_bw
        self.stem = stem
        self.buffer = buffer
        self.pos = pos
        self.cas = cas
        self.ms = ms
        self.md = md
        self.mp = mp
        self.fs = fs
        self.fd = fd
        self.fp = fp
        self.other_lemmas = other_lemmas

    def generate_list(self):
        return [getattr(self, col) for col in Row.INDEX2COL]


def generate_table(lexicon, pos, dialect, pos2lemma2prob, db_lexprob):
    cond_s2cond_t2feats2rows = _get_structured_lexicon_classes(lexicon)
    
    pos2lemma2prob_ = None
    if pos2lemma2prob is not None or db_lexprob is not None:
        pos2lemma2prob_ = _get_pos2lemma2prob(db_lexprob, pos, cond_s2cond_t2feats2rows, pos2lemma2prob)
    
    row = Row(cond_s='CLASS', variant='VAR', entry_type='Entry', cas='cas', freq='Freq', lemma='Lemma',
              lemma_bw='Lemma_bw', stem='Stem', buffer='Buffer(s)', pos='POS', ms='MS', md='MD', mp='MP', fs='FS',
              fd='FD', fp='FP', other_lemmas='Other lemmas').generate_list()
    table = [row]
    col_pos, col_MS, col_freq = table[0].index('POS'), table[0].index('MS'), table[0].index('Freq')
    for cond_s, cond_t2feats2rows in cond_s2cond_t2feats2rows.items():
        form_feats2rows_cond_s = {}
        for cond_t, feats2rows in cond_t2feats2rows.items():
            for feats, rows in feats2rows.items():
                gen, num, pos_ = feats
                if pos2lemma2prob_ is not None:
                    best_index = (-np.array([pos2lemma2prob_[pos_][re.sub(r'[aiuo]', '', strip_lex(info['LEMMA']))]
                                                for info in rows])).argsort()[:len(rows)][0]
                else:
                    best_index = 0

                row = rows[best_index]
                other_lemmas = [bw2ar(row['LEMMA']) for row in rows[:best_index] + rows[best_index + 1:]][:20]
                lemma_stripped = bw2ar(strip_lex(row['LEMMA']))
                lemma, stem = bw2ar(row['LEMMA']), bw2ar(row['FORM'])

                form_feat2case2info, cases = get_suffixes_across_form_feats(lemma, stem, gen, num, pos_, cond_t,
                                                                            actpart=True if 'ACTPART' in cond_s else False)
                for case in cases:
                    form_feats, buffers = [], set()
                    for case2info in form_feat2case2info:
                        form_feats.append(case2info[case]['suffixes'] if case2info[case] else '')
                        if case2info[case]:
                            buffers.add(case2info[case]['buffers'])
                    
                    row_form_feats = form_feats2rows_cond_s.setdefault(
                        tuple(form_feats), {}).setdefault(
                            case, {}).setdefault(
                                feats, Row(
                                    cond_s=cond_s, variant=dialect,
                                    entry_type='Example',
                                    buffer=' // '.join(sorted(buffers)),
                                    cas=case, 
                                    freq=str(len(rows)),
                                    lemma=lemma_stripped, lemma_bw=ar2bw(lemma_stripped), stem=stem,
                                    other_lemmas=' '.join(other_lemmas)).generate_list())
                        
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
            table_.append(Row(cond_s=cond_s, variant=dialect, entry_type='Definition').generate_list())
            case_combs = sorted(set(''.join(sorted(set(cas for cas in row))) for row in rows))
            for case_comb in case_combs:
                table_.append(Row(
                    cond_s=cond_s, variant=dialect, entry_type='Suffixes',
                    cas=case_comb, lemma='**', lemma_bw='**', stem='X',
                    ms=form_feats[0], md=form_feats[1], mp=form_feats[2],
                    fs=form_feats[3], fd=form_feats[4], fp=form_feats[5]).generate_list())
                row2case = {}
                for case2feats2row in rows:
                    case_comb_ = ''.join(sorted(set(cas for cas in case2feats2row)))
                    if case_comb_ != case_comb:
                        continue
                    if len(case_comb_) > 1 and all({k: [vv for i, vv in enumerate(v) if i != Row.COL2INDEX['cas']]
                                for k, v in case2feats2row[cas].items()} for cas in case_comb_):
                        for feats, row in case2feats2row[case_comb_[0]].items():
                            row[Row.COL2INDEX['cas']] = case_comb_
                            row2case[tuple(row)] = case_comb_
                    else:
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
        cond_t = '||'.join(sorted(cond_t.split('||')))
        cond_s = ' '.join(sorted(cond_s.split()))
        cond_s = cond_s if cond_s else '-'

        feats = {feat.split(':')[0]: feat.split(':')[1] for feat in row['FEAT'].split()}
        
        gen = feats.get('gen', '')
        if not gen:
            genders = FUNC_GEN_FROM_FORM_RE.findall(cond_t)
            if len(set(genders)) == 1:
                gen = genders[0].lower()

        num = feats.get('num', '')
        if not num:
            numbers = FUNC_NUM_FROM_FORM_RE.findall(cond_t)
            if len(set(numbers)) == 1:
                num = numbers[0].lower()
        
        pos_ = re.search(r'pos:(\S+)', row['FEAT']).group(1)
        row_cond_s = {'LEMMA': row['LEMMA'], 'FORM': row['FORM']}
        cond_s2cond_t2feats2rows.setdefault(cond_s, {}).setdefault(cond_t, {}).setdefault((gen, num, pos_), []).append(row_cond_s)
    
    return cond_s2cond_t2feats2rows

def get_suffixes_across_form_feats(lemma, stem, gen, num, pos_, cond_t, actpart):
    form_feat2case2info = [{'n': '', 'a': '', 'g': '', 'u': ''} for _ in range(6)]
    cases_seen = set()
    for cond_t_ in cond_t.split('||'):
        col_form_feat = COND_T_CLASSES.index(cond_t_)
        for case in ['n', 'a', 'g', 'u']:
            feats = dict(pos=pos_, gen=gen, num=num, cas=case)
            analyses = _generate_analyses(lemma, stem, feats, cond_t_, actpart)
            cases = set([a[0]['cas'] for a in sum(analyses, []) if a[0] is not None])
            cases_seen.update(cases)
            cases = cases if cases else set(['-'])
            multiple_generations = []
            for a_debug in zip(*analyses):
                analyses_ = [analyses_[0] for analyses_ in a_debug]
                suffixes_feats = [analyses_[6] if analyses_[6] is not None else {'diac': 'CHECK-ZERO'}
                                for analyses_ in a_debug]
                buffers = [a_.get('cm_buffer', '') if a_ is not None else 'CHECK-ZERO'
                           for a_ in analyses_]
                
                if None not in analyses_:
                    assert all(len(set(a_[f] for a_ in analyses_)) == 1
                               for f in ['cas', 'gen', 'num', 'pos'])
                if not all(a_ is None or a_['cas'] == case for a_ in analyses_):
                    continue
                
                diacs = [a_['diac'] if a_ is not None else 'CHECK' for a_ in analyses_]
                gens = set(a_['gen'] for a_ in analyses_ if a_ is not None)
                nums = set(a_['num'] for a_ in analyses_ if a_ is not None)
                pos = set(a_['pos'] for a_ in analyses_ if a_ is not None)
                assert len(gens) == len(nums) == len(pos) == 1
                a_gen, a_num, pos = next(iter(gens)), next(iter(nums)), next(iter(pos)).upper()
                cell = ' / '.join(diacs) + f' ({a_gen}{a_num})'
                multiple_generations.append(cell)
                break
            
            form_feat2case2info[col_form_feat][case] = {
                'suffixes': ' / '.join(f"X+{ar2bw(suff['diac']) if suff['diac'] else 'Ø'}"
                                       for suff in suffixes_feats),
                'content':' // '.join(multiple_generations),
                'pos': pos,
                'buffers':  ' / '.join('X' + ('+' if buff else '') + ar2bw(buff) for buff in buffers)
            }
    
    return form_feat2case2info, cases_seen

def _generate_analyses(lemma, stem, feats, cond_t_, actpart):
    feats_generate = _process_and_generate_feats(feats, cond_t_, actpart)
    analyses = []
    for feats_ in feats_generate:
        analyses_, _ = generator.generate(lemma, feats_, debug=True)
        analyses_ = [a for a in analyses_ if a[0]['cm_stem'] == stem]
        analyses.append(analyses_)

    pad_len = max(len(analyses_) for analyses_ in analyses)
    for analyses_ in analyses:
        analyses_ += [(None,)*7] * (pad_len - len(analyses_))
    
    sort_by_case = lambda a: (a[0]['cm_stem'] + a[0].get('cm_buffer', ''), a[0]['cas']) \
                             if a[0] is not None else (u'\u06FF', 'z')
    for analyses_ in analyses:
        analyses_ = sorted(analyses_, key=sort_by_case)

    return analyses

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

def _process_and_generate_feats(feats, cond_t, actpart):
    feats_ = {'pos': feats['pos'], 'stt': 'i'}
    if feats['gen'] not in ['', '-']:
        feats_.update({'gen': feats['gen']})
    else:
        feats_.update({'gen': cond_t[0].lower()})
    if feats['num'] not in ['', '-']:
        feats_.update({'num': feats['num']})
    else:
        feats_.update({'num': cond_t[1].lower()})
    if 'cas' in feats:
        feats_['cas'] = feats['cas']
    
    feats_ = {k: '' if v == '-' else v for k, v in feats_.items()}
    feats_enc_3ms = {**feats_, **{'enc0': '3ms_poss' if not actpart else '3ms_dobj'}}
    feats_enc_3ms['stt'] = 'c'
    feats_enc_1s = {**feats_, **{'enc0': '1s_poss' if not actpart else '1s_dobj'}}
    feats_enc_1s['stt'] = 'c'


    return feats_, feats_enc_3ms, feats_enc_1s


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

    SHEETS, _ = db_maker_utils.read_morph_specs(
        config, config_name, process_morph=False, lexicon_cond_f=False)
    lexicon = SHEETS['lexicon']
    lexicon['LEMMA'] = lexicon.apply(lambda row: re.sub('lex:', '', row['LEMMA']), axis=1)

    pos_type = args.pos_type if args.pos_type else config_local['pos_type']
    if pos_type == 'verbal':
        POS = 'verb'
        lexicon['COND-S'] = lexicon['COND-S'].replace(r'ditrans', '', regex=True)
    elif pos_type == 'nominal':
        POS = args.pos if args.pos else config_local.get('pos')
        lexicon['COND-S'] = lexicon['COND-S'].replace(r'L#', '', regex=True)
        assert lexicon['COND-T'].str.contains(
            r'((MS|MD|MP|FS|FD|FP)(\|\|)?)+', regex=True).values.tolist()
    elif pos_type == 'other':
        POS = args.pos
    
    assert all(lexicon['FEAT'].str.contains(r'pos:\S+', regex=True))
    if POS:
        lexicon = lexicon[lexicon['FEAT'].str.contains(f'pos:{POS}\\b', regex=True)]
        POS = [POS]
    else:
        POS = list(set([x[0].split(':')[1]
                for x in lexicon['FEAT'].str.extract(r'(pos:\S+)').values.tolist()]))

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
            
            for pos_ in POS:
                total = sum(pos2lemma2prob[pos_].values())
                pos2lemma2prob[pos_] = {lemma: freq / total
                    for lemma, freq in pos2lemma2prob[pos_].items()}
        elif args.db:
            db_lexprob = MorphologyDB(args.db)
        else:
            db_lexprob = MorphologyDB.builtin_db()

    outputs = generate_table(
        lexicon, POS, config_local['dialect'].upper(), pos2lemma2prob, db_lexprob)

    output_name = args.output_name if args.output_name else \
        config_local['docs_debugging']['docs_tables']
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output, sep='\t', file=f)

