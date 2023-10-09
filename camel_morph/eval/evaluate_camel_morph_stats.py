import argparse
import sys
import os
import re
from collections import Counter
from itertools import product
import pickle

import gspread
from numpy import nan
import pandas as pd
from tqdm import tqdm

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.debugging.download_sheets import download_sheets
from camel_morph.debugging.debug_lemma_paradigms import regenerate_signature_lex_rows, _strip_brackets
from camel_morph.utils.utils import get_config_file
from camel_morph import db_maker, db_maker_utils

parser = argparse.ArgumentParser()
parser.add_argument("-config_file_main", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name_main", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-no_download", default=False,
                    action='store_true', help="Do not download data.")
parser.add_argument("-no_build_db", default=False,
                    action='store_true', help="Do not the DB.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args([] if "__file__" not in globals() else None)


POS = ['noun', 'adj', 'adj_comp']

lex_keys = ['diac', 'lex']
lex_pos_keys = [*lex_keys, 'pos']
proclitic_keys = ['prc0', 'prc1', 'prc2', 'prc3']
enclitic_keys = ['enc0', 'enc1']
clitic_keys = [*proclitic_keys, *enclitic_keys]
feats_oblig = ['asp', 'mod', 'vox', 'per', 'num', 'gen', 'cas', 'stt']
form_keys = ['form_num', 'form_gen']
essential_keys = [*lex_pos_keys, *feats_oblig, *clitic_keys]
essential_keys_no_lex_pos = [k for k in essential_keys if k not in lex_pos_keys]
essential_keys_form_feats = essential_keys + form_keys
essential_keys_form_feats_no_lex_pos = essential_keys_no_lex_pos + form_keys
essential_keys_form_feats_no_clitics = lex_pos_keys + feats_oblig + form_keys


def _get_analysis_counts(db):
    stem_cat_hash, stem_hash_diac_only = {}, {}
    for match in db.stem_hash:
        if match == 'NOAN':
            continue
        for cat_analysis in db.stem_hash[match]:
            cat, analysis = cat_analysis
            if POS and analysis['pos'] not in POS:
                continue
            stem_cat_hash.setdefault(cat, []).append(analysis)
            stem_hash_diac_only.setdefault(cat, []).append(analysis.get('diac', ''))

    X_cat_hash_no_clitics, X_hash_diac_only = {}, {}
    for morph_type in ['prefix', 'suffix']:
        clitic_keys_ = proclitic_keys if morph_type == 'prefix' else enclitic_keys
        for cat, analyses in getattr(db, f'{morph_type}_cat_hash').items():
            for analysis in analyses:
                if all(analysis.get(k, '0') == '0' for k in clitic_keys_):
                    X_cat_hash_no_clitics.setdefault(morph_type, {}).setdefault(
                        cat, []).append(analysis)
        
        for cat, analyses in getattr(db, f'{morph_type}_cat_hash').items():
            for analysis in analyses:
                X_hash_diac_only.setdefault(morph_type, {}).setdefault(
                    cat, []).append(analysis.get('diac', ''))
                
    def _get_feats_dict(feats, morpheme_type):
        if morpheme_type in memoize and feats in memoize[morpheme_type]:
            feats_ = memoize[morpheme_type][feats]
        else:
            feats_ = {feat: feats[i]
                    for i, feat in enumerate(essential_keys_form_feats_no_clitics)
                    if feats[i] != 'N/A'}
            memoize_ = memoize.setdefault(morpheme_type, {})
            memoize_[feats] = feats_
        return feats_

    memoize = {}
    unique_analyses_no_clitics = set()
    counts, compat_entries, cmplx_morphs = {}, {}, {}
    for cat_A in tqdm(db.prefix_suffix_compat):
        for cat_C in db.prefix_suffix_compat[cat_A]:
            if cat_A in db.prefix_stem_compat and cat_A in db.prefix_cat_hash:
                for cat_B in db.prefix_stem_compat[cat_A]:
                    if cat_B in db.stem_suffix_compat and cat_B in stem_cat_hash:
                        if cat_C in db.stem_suffix_compat[cat_B] and cat_C in db.suffix_cat_hash:
                                A_counts = len(db.prefix_cat_hash[cat_A])
                                B_counts = len(stem_cat_hash[cat_B])
                                C_counts = len(db.suffix_cat_hash[cat_C])
                                counts.setdefault('analyses', {}).setdefault(
                                    (cat_A, cat_B, cat_C), A_counts * B_counts * C_counts)
                                
                                cmplx_morphs.setdefault('prefix', set()).add(cat_A)
                                cmplx_morphs.setdefault('stem', set()).add(cat_B)
                                cmplx_morphs.setdefault('suffix', set()).add(cat_C)
                                
                                A_counts_diac_only = len(set(X_hash_diac_only['prefix'][cat_A]))
                                B_counts_diac_only = len(set(stem_hash_diac_only[cat_B]))
                                C_counts_diac_only = len(set(X_hash_diac_only['suffix'][cat_C]))
                                counts.setdefault('forms', {}).setdefault(
                                    (cat_A, cat_B, cat_C), A_counts_diac_only * B_counts_diac_only * C_counts_diac_only)

                                if cat_A in X_cat_hash_no_clitics['prefix'] and cat_C in X_cat_hash_no_clitics['suffix']:
                                    A_counts_no_clitics = len(X_cat_hash_no_clitics['prefix'][cat_A])
                                    C_counts_no_clitics = len(X_cat_hash_no_clitics['suffix'][cat_C])
                                    counts.setdefault('no_clitics', {}).setdefault(
                                        (cat_A, cat_B, cat_C), A_counts_no_clitics * B_counts * C_counts_no_clitics)
                                    
                                    feat_combs_A = [tuple(a.get(k, 'N/A') for k in essential_keys_form_feats_no_clitics)
                                                    for a in X_cat_hash_no_clitics['prefix'][cat_A]]
                                    feat_combs_B = [tuple(a.get(k, 'N/A') for k in essential_keys_form_feats_no_clitics)
                                                    for a in stem_cat_hash[cat_B]]
                                    feat_combs_C = [tuple(a.get(k, 'N/A') for k in essential_keys_form_feats_no_clitics)
                                                    for a in X_cat_hash_no_clitics['suffix'][cat_C]]
                                    product_ = product(feat_combs_A, feat_combs_B, feat_combs_C)
                                    for feats_A, feats_B, feats_C in product_:
                                        feats_A_ = _get_feats_dict(feats_A, 'A')
                                        feats_B_ = _get_feats_dict(feats_B, 'B')
                                        feats_C_ = _get_feats_dict(feats_C, 'C')
                                        pos_B = feats_B_['pos']
                                        merged = merge_features(db, feats_A_, feats_B_, feats_C_)
                                        feat_comb = tuple([merged.get(feat, db.defaults[pos_B][feat])
                                                           for feat in essential_keys_form_feats_no_clitics])
                                        unique_analyses_no_clitics.add(feat_comb)

                                compat_entries.setdefault('AB', set()).add((cat_A, cat_B))
                                compat_entries.setdefault('BC', set()).add((cat_B, cat_C))
                                compat_entries.setdefault('AC', set()).add((cat_A, cat_C))

    return counts, compat_entries, cmplx_morphs, unique_analyses_no_clitics


def _underscore_soundness(lemmas_db_camel):
    lemmas_db_camel_, bad_entries = {}, {}
    for lemma_pos in lemmas_db_camel:
        lemmas_db_camel_.setdefault(
            (lemma_pos[0].split('_')[0], lemma_pos[1]), []).append(lemma_pos)
    for lemma_pos, lemmas_pos in lemmas_db_camel_.items():
        if len(lemmas_pos) == 1:
            if re.search(r'\d', lemmas_pos[0][0]):
                bad_entries[lemma_pos] = lemmas_pos
        else:
            if any(not bool(re.search(r'\d', lemma_pos[0])) for lemma_pos in lemmas_pos):
                bad_entries[lemma_pos] = lemmas_pos
            else:
                numbers = [int(lemma_pos[0].split('_')[1]) for lemma_pos in lemmas_pos]
                numbers = sorted(numbers)
                if numbers[0] != 1 or sorted(numbers) != list(range(min(numbers), max(numbers)+1)):
                    bad_entries[lemma_pos] = lemmas_pos


def _get_lex_specs(lexicon_specs, feats):
    return set((feats_[0].split(':')[1], feats_[1].split('/')[1].lower(), *feats_[2:])
                for feats_ in lexicon_specs[feats].values.tolist())

def _get_lex_db(db, feats):
    return set(
        tuple(ar2bw(analysis[feat]) if feat in ['lex', 'stem'] else analysis[feat]
            for feat in feats)
        for _, analyses in db.stem_hash.items()
        for cat, analysis in analyses
        if analysis['pos'] in POS)

def _get_lex_db_calima(db_calima, feats):
    lex_db_calima = _get_lex_db(db_calima, feats)
    lex_db_calima_underscore = [
        (feats_[0].split('_')[0], *feats_[1:]) for feats_ in lex_db_calima
        if int(feats_[0].split('_')[1]) > 1]
    lex_db_calima_ = set()
    for feats_ in lex_db_calima:
        feats_stripped = (feats_[0].split('_')[0], *feats_[1:])
        if feats_stripped in lex_db_calima_underscore:
            lex_db_calima_.add(feats_)
        else:
            lex_db_calima_.add(feats_stripped)
    lex_db_calima = lex_db_calima_
    return lex_db_calima


def _aggregate_results(lemmas_specs, lemmas_db_camel, lemmas_db_calima):
    systems = [('specs', lemmas_specs), ('db_camel', lemmas_db_camel),
               ('db_calima', lemmas_db_calima)]
    lemma_counts = {}
    for system, lemmas_ in systems:
        for pos in POS:
            lemma_counts.setdefault(pos, {}).setdefault(system,
                sum(1 for lemma_pos in lemmas_ if lemma_pos[1] == pos))
    return lemma_counts


def get_number_of_lemmas(lexicon_specs, db_camel, db_calima):
    lemmas_specs = _get_lex_specs(lexicon_specs, ['LEMMA', 'BW'])
    lemmas_db_camel = _get_lex_db(db_camel, ['lex', 'pos'])
    lemmas_db_calima = _get_lex_db_calima(db_calima, ['lex', 'pos'])
    assert len(lemmas_specs) == len(lemmas_db_camel)
    lemma_counts = _aggregate_results(
        lemmas_specs, lemmas_db_camel, lemmas_db_calima)

    return lemma_counts


def get_number_of_stems(lexicon_specs, db_camel, db_calima):
    stems_specs = _get_lex_specs(lexicon_specs, ['LEMMA', 'BW', 'FORM'])
    stems_db_camel = _get_lex_db(db_camel, ['lex', 'pos', 'diac'])
    stems_db_calima = _get_lex_db_calima(db_calima, ['lex', 'pos', 'diac'])
    assert len(stems_specs) != len(stems_db_camel)
    stem_counts = _aggregate_results(
        stems_specs, stems_db_camel, stems_db_calima)

    return stem_counts


def get_specs_stats(morph_specs, lexicon_specs, order_specs):
    db_prefix_morphemes = set(map(tuple, morph_specs[morph_specs['CLASS'].isin(
        CLASS_MAP['DBPrefix'])][['CLASS', 'FUNC']].values.tolist()))
    db_prefix_allomorphs = set(map(tuple, morph_specs[morph_specs['CLASS'].isin(
        CLASS_MAP['DBPrefix'])][['CLASS', 'FUNC', 'FORM']].values.tolist()))
    
    db_suffix_morphemes = set(map(tuple, morph_specs[morph_specs['CLASS'].isin(
        CLASS_MAP['DBSuffix'])][['CLASS', 'FUNC']].values.tolist()))
    db_suffix_allomorphs = set(map(tuple, morph_specs[morph_specs['CLASS'].isin(
        CLASS_MAP['DBSuffix'])][['CLASS', 'FUNC', 'FORM']].values.tolist()))
    
    stem_buffers = set(map(tuple, morph_specs[morph_specs['CLASS'].isin(
        CLASS_MAP['Buffer'])][['CLASS', 'FUNC', 'FORM']].values.tolist()))
    
    morph_classes = set(lexicon_specs['CLASS'].tolist()) | \
                    set(morph_specs['CLASS'].tolist())
    
    def get_conds_from_specs(specs):
        conditions = set()
        for _, row in specs.iterrows():
            for cond_x in [row['COND-S'], row['COND-T'], row['COND-F']]:
                for cond in cond_x.split():
                    for comp in cond.split('||'):
                        comp = re.sub(r'^!', '', comp)
                        if comp not in ['', '_']:
                            conditions.add(comp)
        return conditions

    conditions_morph = get_conds_from_specs(morph_specs)
    conditions_lexicon = get_conds_from_specs(lexicon_specs)
    assert conditions_lexicon - conditions_morph == set()

    
    order_lines = []
    for _, order in order_specs.iterrows():
        kill = False
        for classes in [order['PREFIX'], order['STEM'], order['SUFFIX']]:
            for morph_class in classes.split():
                if morph_class not in morph_classes:
                    kill = True
        if not kill:
            order_lines.append(tuple(order[morph_type]
                for morph_type in ['PREFIX', 'STEM', 'SUFFIX']))

    specs_stats = dict(
        db_prefix_morphemes=len(db_prefix_morphemes),
        db_prefix_allomorphs=len(db_prefix_allomorphs),
        db_suffix_morphemes=len(db_suffix_morphemes),
        db_suffix_allomorphs=len(db_suffix_allomorphs),
        stem_buffers=len(stem_buffers),
        conditions=len(conditions_lexicon|conditions_morph),
        order_lines=len(order_lines)
    )
    
    return specs_stats


def get_db_stats(db_camel, db_calima):
    cmplx_morph_count, compat_count, analyses_count = {}, {}, {}
    unique_analyses_no_clitics_ = {}
    for system, db in [('calima', db_calima), ('camel', db_camel)]:
        analyses_count_, compat, cmplx_morphs, unique_analyses_no_clitics = _get_analysis_counts(db)

        for morph_type in ['prefix', 'suffix']:
            X_cat_hash = getattr(db, f'{morph_type}_cat_hash')
            count = sum(len(X_cat_hash[cat]) for cat in cmplx_morphs[morph_type])
            cmplx_morph_count.setdefault(system, {}).setdefault(morph_type, count)
        
        analyses_count[system] = {k: sum(v.values())
                                  for k, v in analyses_count_.items()}
        
        compat_count[system] = sum(len(compats) for compats in compat.values())

        unique_analyses_no_clitics_[system] = unique_analyses_no_clitics

    return cmplx_morph_count, compat_count, analyses_count


def create_stats_table(lemma_counts, stem_counts, specs_stats,
                       cmplx_morph_count, compat_count, analyses_count):
    table = []
    # (a) section
    row_total = []
    for count_type in ['specs', 'db_camel', 'db_calima']:
        lemmas = sum(counts[count_type] for counts in lemma_counts.values())
        stems = sum(counts[count_type] for counts in stem_counts.values())
        row_total.append(f'{lemmas:,} ({stems:,})')
    table.append(row_total)

    for pos, counts in lemma_counts.items():
        row = []
        for count_type in ['specs', 'db_camel', 'db_calima']:
            row.append(f'{counts[count_type]:,} ({stem_counts[pos][count_type]:,})')
        table.append(row)

    # (b) section
    for morph_type in ['prefix', 'suffix']:
        table.append([str(specs_stats[f'db_{morph_type}_morphemes']) + ' (' +
                      str(specs_stats[f'db_{morph_type}_allomorphs']) + ')', '', ''])
    
    table.append([specs_stats['stem_buffers'], '', ''])
    table.append([specs_stats['conditions'], '', ''])
    table.append([specs_stats['order_lines'], '', ''])

    # (c) section
    table.append(['', compat_count['camel'], compat_count['calima']])
    for morph_type in ['prefix', 'suffix']:
        table.append(['', cmplx_morph_count['camel'][morph_type],
                      cmplx_morph_count['calima'][morph_type]])
        
    # (d) section
    for count_type in ['forms', 'analyses', 'no_clitics']:
        table.append(['', analyses_count['camel'][count_type],
                      analyses_count['calima'][count_type]])
    
    sh = sa.open(config_local['debugging']['stats_spreadsheet'])
    sheet = sh.worksheet(config_local['debugging']['stats_sheet'])
    sheet.batch_update(
        [{'range': config_local['debugging']['table_range'], 'values': table}])


def get_stem_count_per_lemma(db_camel, db_calima):
    stem_counts = {}
    for system, db in [('camel', db_camel), ('calima', db_calima)]:
        for analyses in db.stem_hash.values():
            for _, analysis in analyses:
                lemma, pos, diac = analysis['lex'], analysis['pos'], analysis['diac']
                if pos in POS:
                    stem_counts.setdefault(system, {}).setdefault(
                        (lemma, pos), set()).add(diac)
        hist = sorted(Counter([len(stems)
            for stems in stem_counts[system].values()]).items())
        with open(f'scratch_files/stem_counts_per_lemma_{system}.tsv', 'w') as f:
            for line in hist:
                print(*line, sep='\t', file=f)


def get_basic_lemma_paradigms(lexicon_specs):
    """Get frequencies of basic paradigms (bp)"""
    sh = sa.open('camel-morph-msa-nom')
    sheet = sh.worksheet('Lemma-Class-Reference-noun')
    reference_bp = pd.DataFrame(sheet.get_all_records())

    def _parse_signature(signature):
        return tuple(zip(*[[_strip_brackets(s_) for s_ in s.split(']-[')]
                           for s in signature.split()[:-1]]))

    signature2bp = {}
    for _, row in reference_bp.iterrows():
        if row['id']:
            signature2bp[_parse_signature(row['signature'])] = row['id']
    
    lemma_pos2signature = regenerate_signature_lex_rows(
        config, config_name, lexicon_specs=lexicon_specs)

    pos2bp_freq, lemma_pos_used = {}, set()
    for _, row in lexicon_specs.iterrows():
        lemma_pos = (row['LEMMA'].split(':')[1], row['POS'])
        if lemma_pos in lemma_pos_used:
            continue
        bp_freq = pos2bp_freq.setdefault(lemma_pos[1], Counter())
        signature = _parse_signature(lemma_pos2signature[lemma_pos])
        if signature in signature2bp:
            bp_freq.update([signature2bp[signature]])
        else:
            signature_ = []
            for i, s in enumerate(signature):
                signature_.append(s)
                if i == 0 and s[0] in ['MS||MD', 'FS||FD']:
                    continue
                if tuple(signature_) in signature2bp:
                    bp_freq.update([signature2bp[tuple(signature_)]])
                    break
            else:
                bp_freq.update(['else'])
        lemma_pos_used.add(lemma_pos)
    
    sh = sa.open('CamelMorph-Nominals-Paper-Assets')
    sheet = sh.worksheet('Lemma-Paradigms-Index')
    table_paper = pd.DataFrame(sheet.get_all_records())
    table = []
    for bp in [bp_id for bp_id in table_paper['id'].tolist() if bp_id]:
        row = [pos2bp_freq[pos].get(bp, 0) for pos in POS]
        row.append(sum(row))
        table.append(row)
    sheet.batch_update([{'range': 'K2:N33', 'values': table}])

    pass


if __name__ == "__main__":
    config_name = args.config_name_main
    config = get_config_file(args.config_file_main)
    config_local = config['local'][config_name]
    config_global = config['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.utils import merge_features

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    sa = gspread.service_account(config_global['service_account'])

    CLASS_MAP = config_local['class_map']
    db_name = config_local['db']
    db_dir = os.path.join(
        config_global['db_dir'], f"camel-morph-{config_local['dialect']}", db_name)
    db_camel = MorphologyDB(db_dir, 'dag')
    db_calima = MorphologyDB(args.msa_baseline_db, 'dag')

    # stem_counts = get_stem_count_per_lemma(db_camel, db_calima)
    
    if not args.no_download:
        print()
        download_sheets(config=config,
                        config_name=config_name,
                        service_account=sa)

    if not args.no_build_db:
        print('Building DB...')
        SHEETS = db_maker.make_db(config, config_name)
        print()
    else:
        SHEETS, _ = db_maker_utils.read_morph_specs(
            config, config_name, lexicon_cond_f=False)

    lexicon_specs, morph_specs, order_specs = SHEETS['lexicon'], SHEETS['morph'], SHEETS['order']
    lexicon_specs = lexicon_specs.replace(nan, '', regex=True)
    morph_specs = morph_specs.replace(nan, '', regex=True)
    order_specs = order_specs.replace(nan, '', regex=True)

    # get_basic_lemma_paradigms(lexicon_specs)

    lemma_counts = get_number_of_lemmas(
        lexicon_specs, db_camel, db_calima)

    stem_counts = get_number_of_stems(
        lexicon_specs, db_camel, db_calima)
    
    specs_stats = get_specs_stats(morph_specs, lexicon_specs, order_specs)

    cmplx_morph_count, compat_count, analyses_count = get_db_stats(
        db_camel, db_calima)

    stats_table = create_stats_table(lemma_counts, stem_counts, specs_stats,
                                     cmplx_morph_count, compat_count, analyses_count)
    
    pass