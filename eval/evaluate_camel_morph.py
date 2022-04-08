import sys
import os
import argparse
import json
from tqdm import tqdm
import pickle
import re
import itertools
import random

import pandas as pd
from numpy import nan
pd.options.mode.chained_assignment = None  # default='warn'

sukun_regex = re.compile('o')
aA_regex = re.compile(r'(?<!^[wf])aA')
verb_bw_regex = re.compile(r'[PIC]V')

essential_keys = ['diac', 'lex', 'pos', 'asp', 'mod', 'vox', 'per', 'num', 'gen',
                  'prc0', 'prc1', 'prc2', 'prc3', 'enc0', 'enc1']

def _preprocess_magold_data(gold_data):
    # gold_data = gold_data.split('\n--------------\n')
    gold_data = gold_data.split('--------------\nSENTENCE BREAK\n--------------\n')[:-1]
    gold_data = [sentence.split('\n--------------\n') for sentence in gold_data]
    gold_data = [{'sentence': ex[0].split('\n')[0], 'words': [ex[0].split('\n')[1:]] + [x.split('\n') for x in ex[1:]]} for ex in gold_data]
    gold_data_ = {}
    word_start = len(";;WORD ")
    sentence_start = len(";;; SENTENCE ")
    for example in tqdm(gold_data):
        for analysis in example['words']:
            word = analysis[0][word_start:]
            analysis_key = analysis[4]
            ldc = analysis[1]

            pos_type = 'verbal' if verb_bw_regex.search(ldc) else 'other'
            if pos_type == 'verbal':
                analysis_ = {}
                for field in analysis[4].split()[1:]:
                    field = field.split(':')
                    analysis_[field[0]] = ''.join(field[1:])

                word = {
                            'info': {
                                'sentence': example['sentence'][sentence_start:],
                                'word': word,
                                'magold': analysis[1:4]
                            },
                            'analysis':analysis_
                }

            gold_data_.setdefault(pos_type, []).append(word)

    return gold_data_

def _preprocess_camel_tb_data(data):
    data = data.split('\n')
    data_fl = {}
    for line in data:
        line = line.split()
        if len(line) == 2:
            freq = line[0] if line[0].isdigit() else line[1]
            word = line[0] if not line[0].isdigit() else line[1]
            data_fl[word] = freq
    
    return data_fl


def _preprocess_analysis(analysis, optional_keys=[], gen_norm=True):
    pred = []
    for k in essential_keys + optional_keys:
        if k in ['lex', 'diac']:
            pred.append(strip_lex(ar2bw(analysis[k])))
            pred[-1] = sukun_regex.sub('', pred[-1])
            pred[-1] = aA_regex.sub('A', pred[-1])
        elif gen_norm and k == 'gen':
            pred.append('m' if analysis[k] == 'u' else analysis[k])
        else:
            pred.append(analysis.get(k, 'na'))
    return tuple(pred)


def print_errors(errors, results_path):
    errors_ = []
    i = 0
    for error in errors:
        e_gold = error['gold']
        if not error['pred']:
            pred = pd.DataFrame([('-',)*len(essential_keys)])
        else:
            if e_gold in error['pred']:
                continue
            analyses_pred, index2similarity = [], {}
            for analysis_index, analysis in enumerate(error['pred']):
                analysis_ = []
                for index, f in enumerate(analysis):
                    if f == e_gold[index]:
                        analysis_.append(f)
                        index2similarity.setdefault(analysis_index, 0)
                        index2similarity[analysis_index] += 1
                    else:
                        analysis_.append(f'[{f}]')
                analyses_pred.append(tuple(analysis_))
            sorted_indexes = sorted(
                index2similarity.items(), key=lambda x: x[1], reverse=True)
            analyses_pred = [analyses_pred[analysis_index]
                                for analysis_index, _ in sorted_indexes]

            pred = pd.DataFrame(analyses_pred)

        gold = pd.DataFrame([e_gold])
        example = pd.concat([gold, pred], axis=1)
        ex_col = pd.DataFrame([(f"{i} {error['word']['info']['word']}",)]*len(example.index))
        extra_info = pd.DataFrame([(error['word']['info']['sentence'], *error['word']['info']['magold'])] )
        example = pd.concat([ex_col, example, extra_info], axis=1)
        errors_.append(example)
        i += 1
    
    errors = pd.concat(errors_)
    errors = errors.replace(nan, '', regex=True)
    errors.columns = ['filter'] + essential_keys + essential_keys + ['sentence', 'ldc', 'rank', 'starline']
    errors.to_csv(results_path, sep='\t')


def evaluate_verbs_recall(data, results_path):
    correct, total = 0, 0
    errors = []
    mod_index = essential_keys.index('mod')
    gen_index = essential_keys.index('gen')
    data_ = {}
    for word_info in data['verbal']:
        data_[(word_info['info']['word'], tuple(word_info['info']['magold'][0].split(' # ')[1:4]))] = word_info
    
    for (word, _), word_info in tqdm(data_.items()):
        total += 1
        word_dediac = bw2ar(word)
        analyses_pred = analyzer_camel.analyze(word_dediac)
        analyses_pred = set([_preprocess_analysis(analysis, gen_norm=False) for analysis in analyses_pred])
        analysis_gold = _preprocess_analysis(word_info['analysis'], gen_norm=False)

        if analysis_gold in analyses_pred:
            correct += 1
        else:
            for pred, gold in itertools.product(analyses_pred, [analysis_gold]):
                if any(pred[i] != gold[i] for i in range(len(essential_keys))
                        if i not in [mod_index, gen_index]):
                    continue
                if list(gold).count('u') > 1:
                    raise NotImplementedError
                else:
                    if gold[mod_index] == 'u':
                        correct += 1
                    elif gold[gen_index] == 'u' or pred[gen_index] == 'u':
                        correct += 1
                    else:
                        raise NotImplementedError
                    break
            else:
                errors.append({'word': word_info,
                               'pred': analyses_pred,
                               'gold': analysis_gold})
    
    print_errors(errors, results_path)


def print_comparison(words, analyses_words, status, results_path, bw=False):
    assert len(analyses_words) == len(status) == len(words)

    def qc_automatic(camel):
        for i, k in enumerate(essential_keys):
            baseline_both = pd.concat([baseline[k], both[k]])
            feat_contained = camel[k].isin(baseline_both)
            if k not in ['lex', 'diac']:
                qc = f'{k}:' + camel[k]
                camel.loc[~feat_contained, 'qc'] += ' ' + qc[~feat_contained]
            else:
                camel.loc[~feat_contained, 'qc'] += (' ' if i else '') + k
                # Checks for a full row match in the grammatical features between baseline and camel (if it is not empty)
                # The idea is to spot variation in spelling in the lexical features
                if not baseline.empty:
                    non_lex_feats = [c for c in camel.columns if c not in [k] + ['bw', 'filter', 'status', 'status-global', 'qc']]
                    common_feat_rows_filter_camel = camel[non_lex_feats].isin(
                        baseline[non_lex_feats]).all(axis=1)
                    camel.loc[common_feat_rows_filter_camel, 'qc'] += f'check-{k}-features' 
        return camel

    analysis_results = []
    count = 1
    for (word, analyses_word, status_word) in zip(words, analyses_words, status):
        if status_word == ['NOAN']:
            continue
        columns = ['filter'] + essential_keys + (['bw'] if bw else []) + ['status']
        analyses_word = [analysis + ((ar2bw(analyses_word[analysis]['bw']),) if bw else ())
                            for analysis in analyses_word]
        analyses_word = pd.DataFrame(analyses_word)
        status_word = pd.DataFrame(status_word)
        example = pd.concat([analyses_word, status_word], axis=1)
        ex_col = pd.DataFrame([(f'{count} {dediac_bw(word)}',)]*len(example.index))
        example = pd.concat([ex_col, example], axis=1)
        example.columns = columns
        camel = example[example['status'] == 'CAMEL']
        baseline = example[example['status'] == 'BASELINE']
        both = example[example['status'] == 'BOTH']
        camel['status-global'], baseline['status-global'], both['status-global'] = '', '', ''
        camel['qc'], baseline['qc'], both['qc'] = '', '', ''
        camel.sort_values(by=['lex'], inplace=True)
        baseline.sort_values(by=['lex'], inplace=True)
        both.sort_values(by=['lex'], inplace=True)
        if baseline.empty or camel.empty:
            empty_row = pd.DataFrame([('-',)*len(camel.columns)], columns=camel.columns)
            empty_row.loc[0, 'filter'] = f'{count} {dediac_bw(word)}'
        if not both.empty:
            both['status-global'] = 'full'
            if not baseline.empty and not camel.empty:
                camel['status-global'] = 'mixed'
                camel = qc_automatic(camel)
                baseline['status-global'] = 'mixed'
                both['status-global'] = 'mixed'
            elif not baseline.empty and camel.empty:
                camel = empty_row
                camel['status-global'] = 'noadd-camel'
                baseline['status-global'] = 'only-baseline'
                both['status-global'] = 'mixed'
            elif baseline.empty and not camel.empty:
                camel['status-global'] = 'only-camel'
                camel = qc_automatic(camel)
                baseline = empty_row
                baseline['status-global'] = 'noadd-baseline'
                both['status-global'] = 'mixed'
            else:
                camel = empty_row
                camel['status-global'] = 'noadd-camel'
                baseline = empty_row.copy()
                baseline['status-global'] = 'noadd-baseline'
                both['status-global'] = 'full'
                
        else:
            if not camel.empty:
                camel['status-global'] = 'only-camel'
                if not baseline.empty:
                    camel = qc_automatic(camel)
            else:
                camel = empty_row
                camel['status-global'] = 'noan-camel'

            if not baseline.empty:
                baseline['status-global'] = 'only-baseline'
            else:
                baseline = empty_row
                baseline['status-global'] = 'noan-baseline'
            
        example = pd.concat([both, camel, baseline])
        analysis_results.append(example)
        count += 1

    analysis_results = pd.concat(analysis_results)
    analysis_results = analysis_results.replace(nan, '', regex=True)
    analysis_results.columns = columns + ['status-global', 'qc']
    analysis_results.to_csv(results_path, index=False, sep='\t')


def evaluate_verbs_analyzer_comparison(data, n, results_path):
    words, analyses, status = [], [], []
    data = [word for word in list(data) if word]
    pbar = tqdm(total=min(n, len(data)))
    random.shuffle(data)
    count = 0
    for word in data:
        if count == n:
            break
        word_dediac = bw2ar(word)
        analyses_camel = analyzer_camel.analyze(word_dediac)
        analyses_baseline = analyzer_baseline.analyze(word_dediac)
        analyses_camel = {_preprocess_analysis(analysis): analysis
                          for analysis in analyses_camel if analysis['pos'] == 'verb'}
        analyses_baseline = {_preprocess_analysis(analysis): analysis
                             for analysis in analyses_baseline if analysis['pos'] == 'verb'}
        analyses_camel_set, analyses_baseline_set = set(analyses_camel), set(analyses_baseline)
        
        words.append(ar2bw(word))
        
        if analyses_camel_set == analyses_baseline_set == set():
            analyses.append([])
            status.append(['NOAN'])
            continue
        
        count += 1
        
        camel_minus_baseline = analyses_camel_set - analyses_baseline_set
        baseline_minus_camel = analyses_baseline_set - analyses_camel_set
        intersection = analyses_camel_set & analyses_baseline_set
        analyses_camel.update(analyses_baseline)
        union = analyses_camel
        
        analyses.append(union)
        union = list(analyses_camel.keys())
        
        status_ = []
        for analysis in union:
            if analysis in intersection:
                status_.append('BOTH')
            elif analysis in camel_minus_baseline:
                status_.append('CAMEL')
            elif analysis in baseline_minus_camel:
                status_.append('BASELINE')
            else:
                raise NotImplementedError
        status.append(status_)
        
        pbar.update(1)
    pbar.close()

    with open('eval/status_compare.tsv', 'w') as f:
        for line in status:
            print(*line, sep='\t', file=f)

    print_comparison(words, analyses, status, results_path, bw=True)

def compare_stats(compare_results):
    stats = {'noan': 0, 'only_baseline': 0, 'only_camel': 0, 'equal': 0,
             'disjoint': 0, 'overlap': 0, 'camel_superset': 0, 'baseline_superset': 0}
    stats = {
        'noan': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'only_baseline': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'only_camel': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'equal': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'disjoint': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'overlap': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'camel_superset': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0},
        'baseline_superset': {'word_count': 0, 'analyses_camel': 0, 'analyses_baseline': 0, 'overlap': 0}
    }
    for example in compare_results:
        example_set = set(example)
        both_count = example.count('BOTH')
        if example == ['NOAN']:
            stats['noan']['word_count'] += 1
        elif example_set == {'BASELINE'}:
            stats['only_baseline']['word_count'] += 1
            stats['only_baseline']['analyses_baseline'] += len(example)
        elif example_set == {'CAMEL'}:
            stats['only_camel']['word_count'] += 1
            stats['only_camel']['analyses_camel'] += len(example)
        elif example_set == {'BOTH'}:
            stats['equal']['word_count'] += 1
            stats['equal']['overlap'] += both_count
            stats['equal']['analyses_camel'] += len(example)
            stats['equal']['analyses_baseline'] += len(example)
        elif example_set == {'BOTH', 'CAMEL', 'BASELINE'}:
            stats['overlap']['word_count'] += 1
            stats['overlap']['overlap'] += both_count
            stats['overlap']['analyses_camel'] += both_count + example.count('CAMEL')
            stats['overlap']['analyses_baseline'] += both_count + example.count('BASELINE')
        elif example_set == {'BOTH', 'BASELINE'}:
            stats['baseline_superset']['word_count'] += 1
            stats['baseline_superset']['overlap'] += both_count
            stats['baseline_superset']['analyses_camel'] += both_count
            stats['baseline_superset']['analyses_baseline'] += both_count + example.count('BASELINE')
        elif example_set == {'BOTH', 'CAMEL'}:
            stats['camel_superset']['word_count'] += 1
            stats['camel_superset']['overlap'] += both_count
            stats['camel_superset']['analyses_baseline'] += both_count
            stats['camel_superset']['analyses_camel'] += both_count + example.count('CAMEL')
        elif example_set == {'CAMEL', 'BASELINE'}:
            stats['disjoint']['word_count'] += 1
            stats['disjoint']['analyses_camel'] += example.count('CAMEL')
            stats['disjoint']['analyses_baseline'] += example.count('BASELINE')
        else:
            raise NotImplementedError
        
        with open('eval/stats_compare_camel_tb.tsv', 'w') as f:
            for row, info in stats.items():
                info = [info[k] for k in ['word_count', 'analyses_camel', 'analyses_baseline', 'overlap']]
                print(row, *info, sep='\t', file=f)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", required=True,
                        type=str, help="Path of the file containing the data to evaluate on.")
    parser.add_argument("-preprocessing", required=True, choices=['camel_tb', 'magold'],
                        type=str, help="Preprocessing to use for the data.")
    parser.add_argument("-db_dir", default='db_iterations',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-config_file", required=True,
                        type=str, help="Config file specifying which sheets to use.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-baseline_db", required=True,
                        type=str, help="Path of the baseline DB file we will be comparing against.")
    parser.add_argument("-eval_mode", required=True, choices=['recall', 'compare', 'compare_stats'],
                        type=str, help="What evaluation to perform.")
    parser.add_argument("-results_path", required=True,
                        type=str, help="Path of the output file containing the comparison/recall results.")
    parser.add_argument("-n", default=100,
                        type=int, help="Number of verbs to input to the two compared systems.")
    parser.add_argument("-camel_tools", default='',
                        type=str, help="Path of the directory containing the camel_tools modules.")
    
    random.seed(42)

    args = parser.parse_args()

    if args.camel_tools:
        sys.path.insert(0, args.camel_tools)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.utils.dediac import dediac_bw
    from camel_tools.morphology.utils import strip_lex

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    with open(args.config_file) as f:
        config = json.load(f)['local'][args.config_name]

    db_camel = MorphologyDB(os.path.join(args.db_dir, config['output']))
    analyzer_camel = Analyzer(db_camel)
    db_baseline = MorphologyDB(args.baseline_db)
    analyzer_baseline = Analyzer(db_baseline)

    with open(args.data_path) as f:
        data = f.read()
    
    if args.preprocessing == 'magold':
        data = _preprocess_magold_data(data)

    elif args.preprocessing == 'camel_tb':
        data = _preprocess_camel_tb_data(data)

    if args.eval_mode == 'recall':
        evaluate_verbs_recall(data, args.results_path)
    elif args.eval_mode == 'compare':
        evaluate_verbs_analyzer_comparison(data, args.n, args.results_path)
    elif args.eval_mode == 'compare_stats':
        with open('eval/status_compare.tsv') as f:
            compare_results = [line.strip().split('\t') for line in f.readlines()]
            compare_stats(compare_results)


pass
