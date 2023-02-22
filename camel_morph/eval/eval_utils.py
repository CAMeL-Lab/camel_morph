import os
import sys
from tqdm import tqdm
import itertools
import pickle
import re
import numpy as np
from collections import Counter

from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

lex_keys = ['diac', 'lex']
lex_pos_keys = [*lex_keys, 'pos']
proclitic_keys = ['prc0', 'prc1', 'prc2', 'prc3']
enclitic_keys = ['enc0', 'enc1']
clitic_keys = [*proclitic_keys, *enclitic_keys]
feats_oblig = ['asp', 'mod', 'vox', 'per', 'num', 'gen', 'cas', 'stt']
form_keys = ['form_num', 'form_gen']
essential_keys = [*lex_pos_keys, *feats_oblig, *clitic_keys]
essential_keys_no_lex = [k for k in essential_keys if k not in lex_keys]
essential_keys_no_lex_pos = [k for k in essential_keys if k not in lex_pos_keys]
essential_keys_no_lex_pos_clitic_indexes = [
    i for i, k in enumerate(essential_keys_no_lex_pos) if k in clitic_keys]
essential_keys_form_no_lex_pos = essential_keys_no_lex_pos + form_keys
feat2index = {k: i for i, k in enumerate(essential_keys_no_lex_pos)}

mat_names = ['diac_mat_baseline', 'diac_mat_system', 'system_only_mat',
             'baseline_only_mat', 'no_intersect_mat']

sukun_ar, fatHa_ar = 'ْ', 'َ'
sukun_regex = re.compile(sukun_ar)
aA_regex = re.compile(f'(?<!^[وف]){fatHa_ar}ا')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\033[35m'
    colors = {
        'header': HEADER,
        'blue': OKBLUE,
        'cyan': OKCYAN,
        'green': OKGREEN,
        'warning': WARNING,
        'fail': FAIL,
        'orange':ORANGE}

def color(text, color_):
    text = bcolors.colors[color_] + text
    text += bcolors.ENDC if not text.endswith(bcolors.ENDC) else ''
    return text

def bold(text):
    text = bcolors.BOLD + str(text)
    text += bcolors.ENDC if not text.endswith(bcolors.ENDC) else ''
    return text

def underline(text):
    text = bcolors.UNDERLINE + text
    text += bcolors.ENDC if not text.endswith(bcolors.ENDC) else ''
    return text


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


def get_all_lemmas_from_db(db):
    lemmas_pos = set()
    for match, analyses in db.stem_hash.items():
        for cat, analysis in analyses:
            lemmas_pos.add((analysis['lex'], analysis['pos']))
    return lemmas_pos


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


def _get_feat2value2pos(X, Y, XY, feats):
    feat2value2cat = {}
    for match, analyses in X.items():
        for cat, analysis in analyses:
            for feat in feats:
                if feat in analysis:
                    feat2value2cat.setdefault(feat, {}).setdefault(
                        analysis[feat], set()).add(cat)

    cat2pos_Y = _get_cat2pos(Y)
    feat2value2pos = {}
    for feat, value2cat in feat2value2cat.items():
        for value, X_cats in value2cat.items():
            for X_cat in X_cats:
                for Y_cat in XY.get(X_cat, []):
                    feat2value2pos.setdefault(feat, {}).setdefault(
                        value, set()).update(cat2pos_Y.get(Y_cat, set()))
    
    return feat2value2pos


def get_clitic2value2pos_joined(db):
    """Get a mapping of possible (DB-compatible according to compatibility tables)
    clitic-value pairs according to POS.
    """
    proclitics = [f for f in db.defines if 'prc' in f]
    prc2value2pos = _get_feat2value2pos(X=db.prefix_hash,
                                         Y=db.stem_hash,
                                         XY=db.prefix_stem_compat,
                                         feats=proclitics)
    suffix_stem_compat = _reverse_compat_table(db.stem_suffix_compat)
    enclitics = [f for f in db.defines if 'enc' in f]
    enc2value2pos = _get_feat2value2pos(X=db.suffix_hash,
                                         Y=db.stem_hash,
                                         XY=suffix_stem_compat,
                                         feats=enclitics)
    clitic2value2pos = {**prc2value2pos, **enc2value2pos}
    return clitic2value2pos


def get_pos2clitic_combs(db, clitic2value2pos=None, clitic_value_pos=None):
    if clitic2value2pos is None:
        clitic2value2pos = get_clitic2value2pos_joined(db)

    pos2clitic2value = {}
    for clitic, value2pos in clitic2value2pos.items():
        for value, X_pos in value2pos.items():
            for X_pos_ in X_pos:
                if clitic_value_pos is not None and \
                    (clitic, value, X_pos_) not in clitic_value_pos:
                    continue
                pos2clitic2value.setdefault(X_pos_, {}).setdefault(
                    clitic, set()).add(value)
        
    for pos, clitic2value in pos2clitic2value.items():
        for clitic in clitic2value:
            pos2clitic2value[pos][clitic].add(db.defaults[pos][clitic])

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


def get_pos2obligfeats(db):
    pos2obligfeats = {}
    for pos, analysis_default in db.defaults.items():
        feats_oblig_pos = {
            feat for feat in feats_oblig
            if analysis_default[feat] not in [None, 'na']}
        
        # Had to enforce asp because for some reason it is na in the DB defaults
        if pos == 'verb':
            feats_oblig_pos.add('asp')
        
        feats_oblig_pos = list(feats_oblig_pos)

        combinations = []
        for feat in feats_oblig_pos:
            values = [value for value in db.defines[feat]
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


def preprocess_lex_features(analysis, return_analysis=False):
    for k in ['lex', 'diac']:
        analysis[k] = sukun_regex.sub('', analysis[k])
        analysis[k] = aA_regex.sub('A', analysis[k])
    if return_analysis:
        return analysis


def load_results_debug_eval(report_dir):
    with open(os.path.join(report_dir, 'results_debug_eval.pkl'), 'rb') as f:
        results_debug_eval = pickle.load(f)
    return results_debug_eval


def construct_feats(analysis_key, pos):
    return {**{k: analysis_key[i] for i, k in enumerate(essential_keys_no_lex_pos)
               if analysis_key[i] != 'na'},
            **{'pos': pos}}


def load_matrices(report_dir):
    with open(os.path.join(report_dir, 'matrices.pkl'), 'rb') as f:
        MATRICES = pickle.load(f)

    return MATRICES

def load_pos2feat_value_pairs(report_dir):
    with open(os.path.join(report_dir, 'pos2feat_value_pairs_baseline.pkl'), 'rb') as f:
        pos2feat_value_pairs_baseline = pickle.load(f)
    with open(os.path.join(report_dir, 'pos2feat_value_pairs_system.pkl'), 'rb') as f:
        pos2feat_value_pairs_system = pickle.load(f)
    POS2FEAT_VALUE_PAIRS = dict(
        baseline=pos2feat_value_pairs_baseline,
        system=pos2feat_value_pairs_system
    )
    return POS2FEAT_VALUE_PAIRS


def _get_cat2feat_combs(X):
    cat2feat_combs = {}
    for analyses in X.values():
        for cat, analysis in analyses:
            feats = tuple([analysis.get(feat, 'N/A')
                            for feat in essential_keys_form_no_lex_pos])
            cat2feat_combs.setdefault(cat, Counter()).update([feats])
    return cat2feat_combs

def _get_pos2cat2feat_combs(X):
    pos2cat2feat_combs = {}
    for analyses in X.values():
        for cat, analysis in analyses:
            feats = tuple([analysis.get(feat, 'N/A')
                            for feat in essential_keys_form_no_lex_pos])
            pos2cat2feat_combs.setdefault(
                analysis['pos'], {}).setdefault(cat, {}).setdefault(feats, 0)
            feats_plus_lex = feats + tuple([analysis['lex']])
            count = [
                tuple([a[1].get(feat, 'N/A') for feat in essential_keys_form_no_lex_pos + ['lex']])
                for a in analyses].count(feats_plus_lex)
            pos2cat2feat_combs[analysis['pos']][cat][feats] += count
    return pos2cat2feat_combs

def get_pos2possible_feat_combs(db, POS, merge_features_fn):
    cat2feat_combs_A = _get_cat2feat_combs(db.prefix_hash)
    pos2cat2feat_combs_B = _get_pos2cat2feat_combs(db.stem_hash)
    cat2feat_combs_C = _get_cat2feat_combs(db.suffix_hash)

    def _get_feats_dict(feats, morpheme_type):
        if morpheme_type in memoize and feats in memoize[morpheme_type]:
            feats_ = memoize[morpheme_type][feats]
        else:
            feats_ = {feat: feats[i]
                    for i, feat in enumerate(essential_keys_form_no_lex_pos)
                    if feats[i] != 'N/A'}
            memoize_ = memoize.setdefault(morpheme_type, {})
            memoize_[feats] = feats_
        return feats_

    memoize = {}
    pos2possible_feat_combs = {}
    for cat_A, feat_combs_A_count in tqdm(cat2feat_combs_A.items()):
        for pos_B, cat2feat_combs_B_count in pos2cat2feat_combs_B.items():
            if pos_B not in POS:
                continue
            for cat_B, feat_combs_B_count in cat2feat_combs_B_count.items():
                for cat_C, feat_combs_C_count in cat2feat_combs_C.items():
                    if (cat_B in db.prefix_stem_compat.get(cat_A, []) and
                        cat_C in db.prefix_suffix_compat.get(cat_A, []) and
                        cat_C in db.stem_suffix_compat.get(cat_B, [])):
                        feat_combs_A = feat_combs_A_count.keys()
                        feat_combs_B = feat_combs_B_count.keys()
                        feat_combs_C = feat_combs_C_count.keys()
                        product = itertools.product(feat_combs_A, feat_combs_B, feat_combs_C)
                        for feats_A, feats_B, feats_C in product:
                            feats_A_ = _get_feats_dict(feats_A, 'A')
                            feats_B_ = _get_feats_dict(feats_B, 'B')
                            feats_C_ = _get_feats_dict(feats_C, 'C')
                            merged = merge_features_fn(db, feats_A_, feats_B_, feats_C_)
                            feat_comb = tuple([merged.get(feat, db.defaults[pos_B][feat])
                                               for feat in essential_keys_no_lex_pos])
                            pos2possible_feat_combs.setdefault(pos_B, {}).setdefault(feat_comb, 0)
                            pos2possible_feat_combs[pos_B][feat_comb] += feat_combs_A_count[feats_A] * \
                                                                         feat_combs_B_count[feats_B] * \
                                                                         feat_combs_C_count[feats_C]
    return pos2possible_feat_combs


def _get_clitic_value_pos(clitic2value2pos, POS):
    clitic_value_pos = set(
    (clitic, value, pos)
    for clitic, value2pos_baseline in clitic2value2pos.items()
    for value, poses in value2pos_baseline.items()
    for pos in poses if pos in POS)
    return clitic_value_pos


def get_clitic_value_pos_old(db_system, db_baseline, POS):
    clitic2value2pos_baseline = get_clitic2value2pos_joined(db_baseline)
    clitic2value2pos_system = get_clitic2value2pos_joined(db_system)
    
    clitic_value_pos_baseline = _get_clitic_value_pos(clitic2value2pos_baseline, POS)
    clitic_value_pos_system = _get_clitic_value_pos(clitic2value2pos_system, POS)
    clitic_value_pos_intersect = clitic_value_pos_baseline & clitic_value_pos_system
    clitic_value_pos = dict(
        clitic_value_pos_baseline=clitic_value_pos_baseline,
        clitic_value_pos_system=clitic_value_pos_system,
        clitic_value_pos_intersect=clitic_value_pos_intersect
    )

    def _sanity_check():
        pos2cliticfeats_baseline_intersect = get_pos2clitic_combs(
            db_baseline, clitic2value2pos_baseline, clitic_value_pos_intersect)
        pos2cliticfeats_system_intersect = get_pos2clitic_combs(
            db_system, clitic2value2pos_system, clitic_value_pos_intersect)

        pos2cliticfeats_baseline_intersect_set = set(
            [tuple(sorted((f, v) for f, v in feats.items()))
            for pos in POS for feats in pos2cliticfeats_baseline_intersect[pos]])
        pos2cliticfeats_system_intersect_set = set(
            [tuple(sorted((f, v) for f, v in feats.items()))
            for pos in POS for feats in pos2cliticfeats_system_intersect[pos]])
        assert pos2cliticfeats_system_intersect_set >= pos2cliticfeats_baseline_intersect_set
    
    _sanity_check()
    return clitic_value_pos


def get_pos2feat_value_pairs(pos2possible_feat_combs):
    pos2feat_value_pairs = {}
    for pos, possible_feat_combs in pos2possible_feat_combs.items():
        for feat_comb in possible_feat_combs:
            for i, value in enumerate(feat_comb):
                feat = essential_keys_no_lex_pos[i]
                pos2feat_value_pairs.setdefault(pos, Counter()).update([f'{feat}:{value}'])
    return pos2feat_value_pairs



def get_pos2cliticfeats(pos2possible_feat_combs, POS):
    pos2cliticfeats = {}
    for pos, possible_feat_combs in pos2possible_feat_combs.items():
        if pos not in POS:
            continue
        clitic_feats = set()
        for feat_comb in possible_feat_combs:
            clitic_feats_ = tuple(f for i, f in enumerate(feat_comb) if i in essential_keys_no_lex_pos_clitic_indexes)
            clitic_feats.add(clitic_feats_)
        for clitic_feats_ in clitic_feats:
            clitic_feats_dict = {f: clitic_feats_[i] for i, f in enumerate(clitic_keys)}
            clitic_feats_dict = {**clitic_feats_dict, **{'pos': pos}}
            pos2cliticfeats.setdefault(pos, []).append(clitic_feats_dict)
    
    return pos2cliticfeats


def harmonize_defaults(db_baseline, db_system, POS, report=False):
    for pos in POS:
        defaults_baseline, defaults_system = db_baseline.defaults[pos], db_system.defaults[pos]
        features_union = set(defaults_baseline) | set(defaults_system)
        for feat in features_union:
            if pos is None:
                continue
            if feat not in defaults_system:
                defaults_system[feat] = defaults_baseline[feat]
                if report:
                    print(f'Added {feat}:{defaults_baseline[feat]} as default in SYSTEM')
            if feat not in defaults_baseline:
                defaults_baseline[feat] = defaults_system[feat]
                if report:
                    print(f'Added {feat}:{defaults_system[feat]} as default in BASELINE')
            assert defaults_system[feat] == defaults_baseline[feat]
        
    features_union = set(db_baseline.defines) | set(db_system.defines)
    for feat in features_union:
        if feat not in db_baseline.defines:
            db_baseline.defines[feat] = db_system.defines[feat]
            if report:
                print(f"Added values {' '.join(db_system.defines[feat])} for {feat} in BASELINE")
            continue
        if feat not in db_system.defines:
            db_system.defines[feat] = db_baseline.defines[feat]
            if report:
                print(f"Added values {' '.join(db_baseline.defines[feat])} for {feat} in SYSTEM")
            continue

        baseline_values, system_values = db_baseline.defines[feat], db_system.defines[feat]
        if baseline_values is None and system_values is None:
            continue
        baseline_values_set = set(baseline_values if baseline_values is not None else [])
        system_values_set = set(system_values if system_values is not None else [])
        baseline_only = baseline_values_set - system_values_set
        system_only = system_values_set - baseline_values_set
        if baseline_only:
            db_system.defines[feat] += list(baseline_only)
            if report:
                print(f"Added values {' '.join(baseline_only)} for {feat} in SYSTEM")
        if system_only:
            db_baseline.defines[feat] += list(system_only)
            if report:
                print(f"Added values {' '.join(system_only)} for {feat} in SYSTEM")


def get_pos(pos, db_baseline, db_system):
    if pos == 'other':
        POS = sorted(pos for pos in set(db_baseline.defaults) & set(db_system.defaults)
                    if pos not in ['verb', 'noun', 'noun_quant', 'noun_num', 'noun_prop'
                                    'adj', 'adj_num', 'adj_comp', None])
    else:
        POS = [pos]
    
    return POS