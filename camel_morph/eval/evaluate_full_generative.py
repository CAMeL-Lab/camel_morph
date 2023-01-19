import sys
import argparse
import json
from tqdm import tqdm
import pickle
import itertools
import random
import multiprocessing

essential_keys = ['diac', 'lex', 'pos', 'asp', 'mod', 'vox', 'per', 'num', 'gen',
                  'prc0', 'prc1', 'prc2', 'prc3', 'enc0']
essential_keys_no_lex = [k for k in essential_keys if k not in ['diac', 'lex']]

parser = argparse.ArgumentParser()
parser.add_argument("-output_dir", default='eval_files',
                    type=str, help="Path of the directory to output evaluation results.")
parser.add_argument("-config_file", default='configs/config.json',
                    type=str, help="Config file specifying which sheets to use.")
parser.add_argument("-msa_config_name", default='all_aspects_msa',
                    type=str, help="Config name which specifies the path of the MSA Camel DB.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-multiprocessing", default=True, action='store_true',
                    help="Whether or not to use multiprocessing.")
parser.add_argument("-multiprocessing", default=True, action='store_true',
                    help="Whether to load or to compute results.")
parser.add_argument("-n", default=100,
                    type=int, help="Number of verbs to input to the two compared systems.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")

random.seed(42)

args = parser.parse_args()

with open(args.config_file) as f:
    config = json.load(f)
    config_local = config['local']
    config_msa = config_local[args.msa_config_name]

if args.camel_tools == 'local':
    camel_tools_dir = config['global']['camel_tools']
    sys.path.insert(0, camel_tools_dir)

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.generator import Generator
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_bw


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


def get_all_lemmas_from_db(db):
    lemma_pos = set()
    for match, analyses in db.stem_hash.items():
        for cat, analysis in analyses:
            lemma_pos.add((analysis['lex'], analysis['pos']))
    return lemma_pos


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


def get_pos2feats(db_baseline):
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


def generate_all_possible_words_from_lemma(lemma_pos):
    lemma, pos = lemma_pos
    lemma_ar = bw2ar(lemma)
    generations = []
    zeroes = []
    for feats in pos2feats[pos]:
        generations_ = generator_baseline.generate(lemma_ar, feats)
        if len(generations_) == 0:
            zeroes.append(feats)
        generations += generations_
    
    generations_ = []
    for g in generations:
        generations_.append({k: g[k] for k in essential_keys})
    generations = generations_

    return generations


bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

path_db_baseline = args.msa_baseline_db
db_baseline = MorphologyDB(path_db_baseline)
analyzer_baseline = Analyzer(db_baseline)

db_baseline_gen = MorphologyDB(path_db_baseline, flags='g')
generator_baseline = Generator(db_baseline_gen)

pos2feats = get_pos2feats(db_baseline)

if __name__ == "__main__":
    lemmas_pos = list(get_all_lemmas_from_db(db_baseline))[:args.n]
    
    if args.load_results:
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
    
        with open('results_multiproc.pkl', 'rb') as f:
            final_results = pickle.load(f)
    else:
        if args.multiprocessing:
            results_multiproc = {}
            chunk_size = 2000
            lemma_chunked = [lemmas_pos[i:i + chunk_size] for i in range(0, len(lemmas_pos), chunk_size)]
            for chunk in lemma_chunked:
                with multiprocessing.Pool(8) as p:
                    results = list(tqdm(p.imap(generate_all_possible_words_from_lemma, chunk),
                                        total=len(chunk), position=0, desc='main', leave=False))
                for result in results:
                    for analysis in result:
                        results_multiproc.setdefault(
                            tuple([analysis[k] for k in essential_keys_no_lex]), set()).add(
                            (analysis['lex'], analysis['diac']))
            
            with open('results_multiproc.pkl', 'wb') as f:
                pickle.dump(results_multiproc, f)
        else:
            results = {}
            for lemma_pos in tqdm(lemmas_pos):
                results[lemma_pos] = generate_all_possible_words_from_lemma(lemma_pos)

            with open('results.pkl', 'wb') as f:
                pickle.dump(results, f)

    pass