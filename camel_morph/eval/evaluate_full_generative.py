import sys
import os
import argparse
import json
from tqdm import tqdm
import pickle
import random
import re
import multiprocessing
import cProfile, pstats
import shutil
import numpy as np

import eval_utils

essential_keys = ['diac', 'lex', 'pos', 'asp', 'mod', 'vox', 'per', 'num', 'gen',
                  'cas', 'stt', 'prc0', 'prc1', 'prc2', 'prc3', 'enc0']
feats_oblig = ['asp', 'mod', 'vox', 'per', 'num', 'gen', 'cas', 'stt']
essential_keys_no_lex_pos = [k for k in essential_keys if k not in ['diac', 'lex', 'pos']]

parser = argparse.ArgumentParser()
parser.add_argument("-output_dir", default='eval_files',
                    type=str, help="Path of the directory to output evaluation results.")
parser.add_argument("-config_file", default='configs/config_default.json',
                    type=str, help="Config file specifying which sheets to use.")
parser.add_argument("-db_system", required=True,
                        type=str, help="Path of the system DB file we will be evaluating against the baseline.")
parser.add_argument("-pos", required=True,
                    type=str, help="Part-of-speech to perform the evaluation on.")
parser.add_argument("-baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the baseline DB file we will be comparing against.")
parser.add_argument("-multiprocessing", default=False, action='store_true',
                    help="Whether or not to use multiprocessing.")
parser.add_argument("-report_dir", default='eval_files/report_default',
                    type=str, help="Path of the directory containing partial reports (matrices) generated by the full generative evaluation.")
parser.add_argument("-possible_feat_combs", default='',
                    type=str, help="Paths of the file containing the union of possible feature combinations between the baseline and the system.")
parser.add_argument("-test_mode", default=False, action='store_true',
                    help="Only test mode.")
parser.add_argument("-profiling", default=False, action='store_true',
                    help="Run profiling.")
parser.add_argument("-n_cpu", default=8,
                    type=int, help="Number of cores to use.")
parser.add_argument("-n", default=1000000,
                    type=int, help="Number of inputs to the two compared systems.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args()

random.seed(42)

with open(args.config_file) as f:
    config = json.load(f)

if args.camel_tools == 'local':
    camel_tools_dir = config['global']['camel_tools']
    sys.path.insert(0, camel_tools_dir)

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.generator import Generator
from camel_tools.utils.charmap import CharMapper

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')


def get_all_possible_feat_combs(id_lemma_pos, oblig_spec=False):
    _, (lemma_ar, pos) = id_lemma_pos

    if not args.multiprocessing:
        global possible_feat_combs
    else:
        possible_feat_combs = set()
    
    generations_baseline, generations_camel, _ = produce_generations(lemma_ar, pos, oblig_spec)
    generations_baseline_ = _process_generations(generations_baseline)
    generations_camel_ = _process_generations(generations_camel)
    possible_feat_combs.update(set(generations_camel_))
    possible_feat_combs.update(set(generations_baseline_))
    
    return possible_feat_combs
    

def _process_generations(generations):
    generations_ = {}
    for g in generations:
        eval_utils._preprocess_lex_features(g)
        key = tuple([g.get(k, 'na') for k in essential_keys_no_lex_pos])
        generations_.setdefault(key, []).append((g['lex'], g['diac'], g['pos']))
    return generations_


def produce_generations(lemma_ar, pos, oblig_spec):
    generations_baseline, generations_camel = [], []
    failed = {}
    if oblig_spec:
        for feats_oblig in tqdm(pos2obligfeats[pos]):
            for feats_clitic in pos2cliticfeats[pos]:
                feats_all = {**feats_oblig, **feats_clitic}
                try:
                    generations_baseline += generator_baseline.generate(lemma_ar, feats_all, legacy=True)
                except:
                    failed.setdefault('baseline', []).append((lemma_ar, feats_all))
                if feats_all.get('prc0') not in ['mA_neg', 'lA_neg']:
                    try:
                        generations_camel += generator_camel.generate(lemma_ar, feats_all)
                    except:
                        failed.setdefault('camel', []).append((lemma_ar, feats_all))
    else:
        for feats_clitic in pos2cliticfeats[pos]:
            try:
                generations_baseline += generator_baseline.generate(lemma_ar, feats_clitic, legacy=True)
            except:
                failed.setdefault('baseline', []).append((lemma_ar, feats_clitic))
            if feats_clitic.get('prc0') not in ['mA_neg', 'lA_neg']:
                try:
                    generations_ = generator_camel.generate(lemma_ar, feats_clitic)
                    generations_camel += generations_
                except:
                    failed.setdefault('camel', []).append((lemma_ar, feats_clitic))
    
    return generations_baseline, generations_camel, failed


def generate_all_possible_words_from_lemma(id_lemma_pos, oblig_spec=False):
    if not args.multiprocessing:
        global failed
        global recall_mat_camel, recall_mat_baseline
        global camel_minus_baseline_mat, baseline_minus_camel_mat, no_intersect_mat
    else:
        failed = {}
        recall_mat_camel = np.zeros((1, len(index2analysis)), dtype='uint8')
        recall_mat_baseline = np.zeros((1, len(index2analysis)), dtype='uint8')
        camel_minus_baseline_mat = np.zeros((1, len(index2analysis)), dtype='uint8')
        baseline_minus_camel_mat = np.zeros((1, len(index2analysis)), dtype='uint8')
        no_intersect_mat = np.zeros((1, len(index2analysis)), dtype='bool')
    
    lemma_id, (lemma_ar, pos) = id_lemma_pos

    generations_baseline, generations_camel, failed_ = produce_generations(lemma_ar, pos, oblig_spec)
    if not args.multiprocessing:
        for system, fails_ in failed_.items():
            fails = failed.setdefault(system, [])
            fails += fails_
    else:
        failed = failed_

    generations_baseline_ = _process_generations(generations_baseline)
    generations_camel_ = _process_generations(generations_camel)

    generations_camel_set, generations_baseline_set = set(generations_camel_), set(generations_baseline_)
    camel_minus_baseline = generations_camel_set - generations_baseline_set
    baseline_minus_camel = generations_baseline_set - generations_camel_set
    intersection = generations_camel_set & generations_baseline_set
    
    for k in camel_minus_baseline:
        if not args.multiprocessing:
            recall_mat_camel[lemma_id][analysis2index[k]] += 1
        else:
            recall_mat_camel[0][analysis2index[k]] += 1
    for k in baseline_minus_camel:
        if not args.multiprocessing:
            recall_mat_baseline[lemma_id][analysis2index[k]] += 1
        else:
            recall_mat_baseline[0][analysis2index[k]] += 1
    for k in intersection:
        lemma_diac_pos_baseline_set = set(generations_baseline_[k])
        lemma_diac_pos_camel_set = set(generations_camel_[k])

        if not args.multiprocessing:
            recall_mat_camel[lemma_id][analysis2index[k]] += len(lemma_diac_pos_camel_set)
            recall_mat_baseline[lemma_id][analysis2index[k]] += len(lemma_diac_pos_baseline_set)
        else:
            recall_mat_camel[0][analysis2index[k]] += len(lemma_diac_pos_camel_set)
            recall_mat_baseline[0][analysis2index[k]] += len(lemma_diac_pos_baseline_set)
        
        camel_baseline_lex_intersect = lemma_diac_pos_camel_set & lemma_diac_pos_baseline_set
        baseline_minus_camel_lex = lemma_diac_pos_baseline_set - lemma_diac_pos_camel_set
        camel_minus_baseline_lex = lemma_diac_pos_camel_set - lemma_diac_pos_baseline_set
        if camel_baseline_lex_intersect:
            if lemma_diac_pos_camel_set != lemma_diac_pos_baseline_set:
                if camel_minus_baseline_lex:
                    if not args.multiprocessing:
                        camel_minus_baseline_mat[lemma_id][analysis2index[k]] += len(camel_baseline_lex_intersect)
                    else:
                        camel_minus_baseline_mat[0][analysis2index[k]] += len(camel_baseline_lex_intersect)
                if baseline_minus_camel_lex:
                    if not args.multiprocessing:
                        baseline_minus_camel_mat[lemma_id][analysis2index[k]] += len(camel_baseline_lex_intersect)
                    else:
                        baseline_minus_camel_mat[0][analysis2index[k]] += len(camel_baseline_lex_intersect)
        else:
            if not args.multiprocessing:
                no_intersect_mat[lemma_id][analysis2index[k]] = True
            else:
                no_intersect_mat[0][analysis2index[k]] = True

    result = dict(lemma_id=lemma_id,
                  recall_mat_camel=recall_mat_camel,
                  recall_mat_baseline=recall_mat_baseline,
                  camel_minus_baseline_mat=camel_minus_baseline_mat,
                  baseline_minus_camel_mat=baseline_minus_camel_mat,
                  no_intersect_mat=no_intersect_mat,
                  failed=failed)

    return result


path_db_baseline = args.baseline_db
db_baseline = MorphologyDB(path_db_baseline)

db_baseline_gen = MorphologyDB(path_db_baseline, flags='g')
generator_baseline = Generator(db_baseline_gen)

path_db_camel = os.path.join(args.db_system)
db_camel_gen = MorphologyDB(path_db_camel, flags='g')
generator_camel = Generator(db_camel_gen)

DEFINES = {k: v if v is None else [vv for vv in v if vv != 'na']
           for k, v in generator_baseline._db.defines.items()}

pos2cliticfeats = eval_utils.get_pos2clitic_combs(db_baseline)
pos2obligfeats = eval_utils.get_pos2obligfeats(db_baseline)
lemmas_pos_baseline = eval_utils.get_all_lemmas_from_db(db_baseline)
del db_baseline
lemmas_pos_baseline = set([lemma_pos for lemma_pos in lemmas_pos_baseline
                           if lemma_pos[1] == args.pos])
lemmas_pos_camel = eval_utils.get_all_lemmas_from_db(MorphologyDB(path_db_camel))
lemmas_pos_camel = set([lemma_pos for lemma_pos in lemmas_pos_camel
                        if lemma_pos[1] == args.pos])
lemmas_intersect = lemmas_pos_camel & lemmas_pos_baseline

lemmas_pos = list(lemmas_intersect)[:args.n]
lemmas_pos = [(i, lemma_pos) for i, lemma_pos in enumerate(lemmas_pos)]

if args.possible_feat_combs:
    with open(args.possible_feat_combs, 'rb') as f:
        possible_feat_combs = pickle.load(f)
    index2analysis = list(possible_feat_combs)
    analysis2index = {analysis: i for i, analysis in enumerate(index2analysis)}
    recall_mat_baseline = np.zeros((len(lemmas_pos), len(index2analysis)), dtype='uint8')
    recall_mat_camel = np.zeros((len(lemmas_pos), len(index2analysis)), dtype='uint8')
    camel_minus_baseline_mat = np.zeros((len(lemmas_pos), len(index2analysis)), dtype='uint8')
    baseline_minus_camel_mat = np.zeros((len(lemmas_pos), len(index2analysis)), dtype='uint8')
    no_intersect_mat = np.zeros((len(lemmas_pos), len(index2analysis)), dtype='bool')
else:
    possible_feat_combs = set()

eval_with_clitics = {}
failed = {}

if args.profiling:
    profiler = cProfile.Profile()
    profiler.enable()

if __name__ == "__main__":
    if os.path.isdir(args.report_dir):
        if args.possible_feat_combs:
            for file_name in os.listdir(args.report_dir):
                if file_name != 'possible_feat_combs.pkl':
                    os.remove(os.path.join(args.report_dir, file_name))
        else:
            shutil.rmtree(args.report_dir)
    else:
        os.makedirs(args.report_dir)

    with open(os.path.join(args.report_dir, 'lemmas_pos.pkl'), 'wb') as f:
        pickle.dump(lemmas_pos, f)
    
    if args.multiprocessing:
        if args.possible_feat_combs:
            with multiprocessing.Pool(args.n_cpu) as p:
                results = list(tqdm(p.imap(generate_all_possible_words_from_lemma, lemmas_pos),
                                    total=len(lemmas_pos), smoothing=0.2))
            for result in results:
                lemma_id = result['lemma_id']
                recall_mat_baseline[lemma_id] = result['recall_mat_baseline']
                recall_mat_camel[lemma_id] = result['recall_mat_camel']
                camel_minus_baseline_mat[lemma_id] = result['camel_minus_baseline_mat']
                baseline_minus_camel_mat[lemma_id] = result['baseline_minus_camel_mat']
                no_intersect_mat[lemma_id] = result['no_intersect_mat']
                for system, fails_ in result['failed'].items():
                    fails = failed.setdefault(system, [])
                    fails += fails_
        else:
            with multiprocessing.Pool(args.n_cpu) as p:
                results = list(tqdm(p.imap(get_all_possible_feat_combs, lemmas_pos),
                                    total=len(lemmas_pos), smoothing=0.2))
            for result in results:
                possible_feat_combs.update(result)
    else:
        results = []
        for id_lemma_pos in tqdm(lemmas_pos):
            if args.possible_feat_combs:
                generate_all_possible_words_from_lemma(id_lemma_pos)
            else:
                get_all_possible_feat_combs(id_lemma_pos)

        with open(os.path.join(args.report_dir, 'results_serial.pkl'), 'wb') as f:
            pickle.dump(eval_with_clitics, f)

    if not args.possible_feat_combs:
        with open(os.path.join(args.report_dir, 'possible_feat_combs.pkl'), 'wb') as f:
            pickle.dump(possible_feat_combs, f)
    else:
        with open(os.path.join(args.report_dir, 'failed.pkl'), 'wb') as f:
            pickle.dump(failed, f)
        np.save(os.path.join(args.report_dir, 'recall_mat_baseline'), recall_mat_baseline)
        np.save(os.path.join(args.report_dir, 'recall_mat_camel'), recall_mat_camel)
        np.save(os.path.join(args.report_dir, 'camel_minus_baseline_mat'), camel_minus_baseline_mat)
        np.save(os.path.join(args.report_dir, 'baseline_minus_camel_mat'), baseline_minus_camel_mat)
        np.save(os.path.join(args.report_dir, 'no_intersect_mat'), no_intersect_mat)
        with open(os.path.join(args.report_dir, 'index2analysis.pkl'), 'wb') as f:
            pickle.dump(index2analysis, f)
    
    print('Done.')