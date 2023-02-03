import os
import sys
from tqdm import tqdm
import itertools
import pickle
import re

from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

essential_keys = ['diac', 'lex', 'pos', 'asp', 'mod', 'vox', 'per', 'num', 'gen',
                  'cas', 'stt', 'prc0', 'prc1', 'prc2', 'prc3', 'enc0']
feats_oblig = ['asp', 'mod', 'vox', 'per', 'num', 'gen', 'cas', 'stt']
essential_keys_no_lex_pos = [k for k in essential_keys if k not in ['diac', 'lex', 'pos']]
feat2index = {k: i for i, k in enumerate(essential_keys_no_lex_pos)}

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    
    size = str(size / (1000*1000)) + 'MB'
    return size


def _reverse_compat_table(XY):
    YX = {}
    for X_cat, Y_cats in XY.items():
        for Y_cat in Y_cats:
            YX.setdefault(Y_cat, set()).add(X_cat)
    return YX

def _get_cat2analyses(X):
    cat2analyses = {}
    for match, analyses in X.items():
        for cat, analysis in analyses:
            cat2analyses.setdefault(cat, []).append((match, analysis))
    return cat2analyses


def calculate_number_of_possible_words(db):
    all_compat = []
    for a, B in tqdm(db.prefix_stem_compat.items()):
        for b in B:
            for c in db.stem_suffix_compat.get(b, set()):
                if c in db.prefix_suffix_compat.get(a, set()):
                    all_compat.append((a, b, c))

    cat2analyses = [_get_cat2analyses(db.prefix_hash),
                     _get_cat2analyses(db.stem_hash),
                     _get_cat2analyses(db.suffix_hash)]
    total = 0
    for compat in all_compat:
        total_ = 1
        for i, cat2analyses_i in enumerate(cat2analyses):
            total_ *= len(cat2analyses_i.get(compat[i], []))
        total += total_
    
    return total


def _get_cat2pos(X):
    cat2pos = {}
    for match, analyses in X.items():
        for cat, analysis in analyses:
            if 'pos' in analysis:
                cat2pos.setdefault(cat, set()).add(analysis['pos'])
    return cat2pos


def _get_clitic2value2pos(X, Y, XY, clitics):
    clitic2value2cat = {}
    for match, analyses in X.items():
        for cat, analysis in analyses:
            for clitic in clitics:
                if clitic in analysis:
                    clitic2value2cat.setdefault(clitic, {}).setdefault(
                        analysis[clitic], set()).add(cat)

    cat2pos_Y = _get_cat2pos(Y)
    clitic2value2pos = {}
    for clitic, value2cat in clitic2value2cat.items():
        for value, X_cats in value2cat.items():
            for X_cat in X_cats:
                for Y_cat in XY.get(X_cat):
                    clitic2value2pos.setdefault(clitic, {}).setdefault(
                        value, set()).update(cat2pos_Y.get(Y_cat, set()))
    
    return clitic2value2pos


def _get_clitic2value2pos_joined(db_baseline):
    prc2value2pos = _get_clitic2value2pos(X=db_baseline.prefix_hash,
                                         Y=db_baseline.stem_hash,
                                         XY=db_baseline.prefix_stem_compat,
                                         clitics=['prc0', 'prc1', 'prc2', 'prc3'])
    suffix_stem_compat = _reverse_compat_table(db_baseline.stem_suffix_compat)
    enc2value2pos = _get_clitic2value2pos(X=db_baseline.suffix_hash,
                                         Y=db_baseline.stem_hash,
                                         XY=suffix_stem_compat,
                                         clitics=['enc0'])
    clitic2value2pos = {**prc2value2pos, **enc2value2pos}
    return clitic2value2pos


def get_pos2clitic_combs(db_baseline):
    clitic2value2pos = _get_clitic2value2pos_joined(db_baseline)

    pos2clitic2value = {}
    for clitic, value2pos in clitic2value2pos.items():
        for value, X_pos in value2pos.items():
            for X_pos_ in X_pos:
                pos2clitic2value.setdefault(X_pos_, {}).setdefault(
                    clitic, set()).add(value)
    
    for pos, clitic2value in pos2clitic2value.items():
        for clitic in clitic2value:
            pos2clitic2value[pos][clitic].add(db_baseline.defaults[pos][clitic])

    pos2feats = {}
    for pos, clitic2value in pos2clitic2value.items():
        for clitic, value in clitic2value.items():
            clitic_values = list(clitic2value.items())
            clitic_combinations = itertools.product(
                *[values for _, values in clitic_values])
            feats = [{**{clitic:comb[i]  for i, (clitic, _) in enumerate(clitic_values)},
                      **{'pos': pos}}
                     for comb in clitic_combinations]
            pos2feats[pos] = feats
    
    for pos, feats in pos2feats.items():
        feats.append({'pos': pos})

    return pos2feats


def _get_pos2obligfeats(db_baseline):
    pos2obligfeats = {}
    for pos, analysis_default in db_baseline.defaults.items():
        feats_oblig_pos = {
            feat for feat in feats_oblig
            if analysis_default[feat] not in [None, 'na']}
        
        # Had to enforce asp because for some reason it is na in the DB defaults
        if pos == 'verb':
            feats_oblig_pos.add('asp')
        
        feats_oblig_pos = list(feats_oblig_pos)

        combinations = []
        for feat in feats_oblig_pos:
            values = [value for value in db_baseline.defines[feat]
                    if value != 'na']
            combinations.append(values)
        
        # Except rare cases to reduce search space (these cases will appear
        # as misses later on in the evaluation and can be looked up separately)
        if 'num' in feats_oblig_pos:
            num_index = feats_oblig_pos.index('num')
            combinations[num_index] = [v for v in combinations[num_index]
                                    if v not in ['b', 'u']]
        if 'gen' in feats_oblig_pos:
            gen_index = feats_oblig_pos.index('gen')
            combinations[gen_index] = [v for v in combinations[gen_index]
                                if v not in ['b']]
        if 'vox' in feats_oblig_pos:
            vox_index = feats_oblig_pos.index('vox')
            combinations[vox_index] = [v for v in combinations[vox_index]
                                    if v not in ['u']]
        
        product = itertools.product(*combinations)

        feat_combs_pos = []
        for comb in product:
            comb_ = {}
            for i, f in enumerate(feats_oblig_pos):
                comb_[f] = comb[i]
            feat_combs_pos.append(comb_)
        
        pos2obligfeats[pos] = feat_combs_pos

    return pos2obligfeats


def get_closest_key(analysis_key_compare, analysis_keys):
    analysis_key_compare = analysis_key_compare.split('+')
    index2similarity = {}
    for analysis_index, analysis in enumerate(analysis_keys):
        for index, f in enumerate(analysis.split('+')):
            if f == analysis_key_compare[index]:
                index2similarity.setdefault(analysis_index, 0)
    sorted_indexes = sorted(
        index2similarity.items(), key=lambda x: x[1], reverse=True)
    best_key_index = sorted_indexes[0]
    best_key = [analysis_keys[best_key_index]]
    return best_key


def get_index2lemmas_pos(report_dir):
    with open(os.path.join(report_dir, 'lemmas_pos.pkl'), 'rb') as f:
        lemmas_pos = pickle.load(f)
    lemmas_pos = {index: lemma_pos for index, lemma_pos in lemmas_pos}
    return lemmas_pos


def get_results_debug_eval(report_dir):
    with open(os.path.join(report_dir, 'results_debug_eval.pkl'), 'rb') as f:
        results_debug_eval = pickle.load(f)
    return results_debug_eval


def get_full_report(report_dir):
    report = {}
    for report_name in sorted(os.listdir(report_dir)):
        if not bool(re.match(r'\d+\.pkl', report_name)):
            continue
        with open(os.path.join(report_dir, report_name), 'rb') as f:
            report_ = pickle.load(f)
            join_reports(main_report=report, report_to_add=report_)
    return report


def join_reports(main_report, report_to_add):
    for comparison_type, comparison in report_to_add.items():
        if comparison_type in ['camel', 'baseline']:
            for feats, lemmas_ in comparison.items():
                lemmas = main_report.setdefault(
                    comparison_type, {}).setdefault(feats, [])
                lemmas += lemmas_
        else:
            for feats, info_ in comparison.items():
                info = main_report.setdefault(
                    comparison_type, {}).setdefault(feats, {})
                for metric, value_ in info_.items():
                    info.setdefault(metric, 0 if type(value_) is int else [0, []])
                    if type(value_) is int:
                        main_report[comparison_type][feats][metric] += value_
                    else:
                        main_report[comparison_type][feats][metric][0] += value_[0]
                        main_report[comparison_type][feats][metric][1] += value_[1]