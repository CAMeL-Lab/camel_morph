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

essential_keys = ['source', 'diac', 'lex', 'pos', 'asp', 'mod', 'vox', 'per', 'num', 'gen',
                  'prc0', 'prc1', 'prc1.5', 'prc2', 'prc3', 'enc0', 'enc1', 'enc2']


def _preprocess_magold_data(gold_data):
    gold_data = gold_data.split(
        '--------------\nSENTENCE BREAK\n--------------\n')[:-1]
    gold_data = [sentence.split('\n--------------\n')
                 for sentence in gold_data]
    gold_data = [{'sentence': ex[0].split('\n')[0], 'words': [ex[0].split(
        '\n')[1:]] + [x.split('\n') for x in ex[1:]]} for ex in gold_data]
    gold_data_ = {}
    word_start, sentence_start = len(";;WORD "), len(";;; SENTENCE ")
    for example in tqdm(gold_data):
        for info in example['words']:
            word = info[0][word_start:]
            ldc = info[1]

            pos_type = 'verbal' if verb_bw_regex.search(ldc) else 'other'
            if pos_type == 'verbal':
                analysis_ = {}
                for field in info[4].split()[1:]:
                    field = field.split(':')
                    analysis_[field[0]] = ''.join(field[1:])

                word = {
                    'info': {
                        'sentence': example['sentence'][sentence_start:],
                        'word': word,
                        'magold': info[1:4]
                    },
                    'analysis': analysis_
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


def _preprocess_ldc_dediac(ldc_diac):
    analyzer_input = re.sub(r'\(null\)', '', ldc_diac)
    analyzer_input = re.sub(r'uwA\+', 'uw', analyzer_input)
    analyzer_input = re.sub(r'[aiuo~FKN`\+_]', '', analyzer_input)
    return analyzer_input


def _preprocess_analysis(analysis, optional_keys=[]):
    if analysis['prc0'] in ['mA_neg', 'lA_neg']:
        analysis['prc1.5'] = analysis['prc0']
        analysis['prc0'] = '0'

    pred = []
    for k in essential_keys + optional_keys:
        if k in ['lex', 'diac']:
            stripped = ar2bw(analysis[k])
            if k == 'lex':
                stripped = re.sub(r'(-[uia]{1,3})?(_\d)?$', '', stripped)
            stripped = stripped.replace('_', '')
            pred.append(stripped)
            pred[-1] = sukun_regex.sub('', pred[-1])
            pred[-1] = aA_regex.sub('A', pred[-1])
        elif k == 'gen':
            pred.append('m' if analysis[k] == 'u' else analysis[k])
        elif k == 'prc2':
            pred.append(re.sub(r'^w[ai]', 'w', analysis[k]))
        elif k == 'prc1':
            pred.append(re.sub(r'^[hH]', 'h', analysis[k]))
        elif re.match(r'prc\d|enc\d', k):
            pred.append(analysis.get(k, '0'))
        else:
            pred.append(analysis.get(k, 'na'))
    return tuple(pred)


def recall_print(errors, correct_cases, drop_cases, results_path):
    errors_ = []
    i = 1
    source_index = essential_keys.index('source')
    for label, cases in [('correct', correct_cases), ('wrong', errors), ('drop', drop_cases)]:
        for case in cases:
            e_gold = case['gold']
            if not case['pred']:
                pred = pd.DataFrame([('-',)*len(essential_keys)])
                label_ = 'noan'
            else:
                label_ = label
                analyses_pred, index2similarity = [], {}
                for analysis_index, analysis in enumerate(case['pred']):
                    analysis_ = []
                    for index, f in enumerate(analysis):
                        if f == e_gold[index]:
                            analysis_.append(f)
                            index2similarity.setdefault(analysis_index, 0)
                            index2similarity[analysis_index] += (
                                1.01 if analysis[source_index] == 'main' else 1)
                        else:
                            analysis_.append(f'[{f}]')
                    analyses_pred.append(tuple(analysis_))
                sorted_indexes = sorted(
                    index2similarity.items(), key=lambda x: x[1], reverse=True)
                analyses_pred = [analyses_pred[analysis_index]
                                 for analysis_index, _ in sorted_indexes][0:1]

                pred = pd.DataFrame(analyses_pred)

            gold = pd.DataFrame([e_gold])
            example = pd.concat([gold, pred], axis=1)
            ex_col = pd.DataFrame(
                [(f"{i} {case['word']['info']['word']}", label_)]*len(example.index))
            extra_info = pd.DataFrame(
                [(bw2ar(case['word']['info']['sentence']), *case['word']['info']['magold'], case['count'])])
            example = pd.concat([ex_col, extra_info, example], axis=1)
            errors_.append(example)
            i += 1

    errors = pd.concat(errors_)
    errors = errors.replace(nan, '', regex=True)
    errors.columns = ['filter', 'label', 'sentence', 'ldc',
                      'rank', 'starline', 'freq'] + essential_keys + essential_keys
    errors.to_csv(results_path, index=False, sep='\t')


def evaluate_verbs_recall(data, eval_mode):
    source_index = essential_keys.index('source')
    lex_index = essential_keys.index('lex')
    diac_index = essential_keys.index('diac')
    essential_keys_ = [k for k in essential_keys if k != 'source']
    excluded_indexes = [source_index]
    if 'no_lex' in eval_mode:
        print('Excluding lexical features from evaluation.')
        essential_keys_ = [
            k for k in essential_keys_ if k not in ['lex', 'diac']]
        excluded_indexes.append(lex_index)
        excluded_indexes.append(diac_index)
    mod_index = essential_keys_.index('mod')
    gen_index = essential_keys_.index('gen')

    if 'ldc_dediac' in eval_mode:
        print('Analyzer input: LDC DEDIAC')
    elif 'raw' in eval_mode:
        print('Analyzer input: RAW')

    data_, counts = {}, {}
    for word_info in data['verbal']:
        key = (word_info['info']['word'], tuple(
            word_info['info']['magold'][0].split(' # ')[1:4]))
        counts.setdefault(key, 0)
        counts[key] += 1
        data_[key] = word_info

    correct, total = 0, 0
    errors, correct_cases, drop_cases = [], [], []
    data_ = list(data_.items())[:args.n]
    pbar = tqdm(total=len(data_))
    for (word, ldc_bw), word_info in data_:
        total += 1
        if 'raw' in eval_mode:
            analyzer_input = word
        elif 'ldc_dediac' in eval_mode:
            analyzer_input = _preprocess_ldc_dediac(ldc_bw[0])

        analyzer_input = bw2ar(analyzer_input)

        analyses_pred = analyzer_camel.analyze(analyzer_input)
        for analysis in analyses_pred:
            analysis['source'] = 'main'
        analyses_pred = set([_preprocess_analysis(analysis)
                             for analysis in analyses_pred])
        analysis_gold = _preprocess_analysis(word_info['analysis'])

        match = re.search(r'ADAM|CALIMA|SAMA', word_info['analysis']['gloss'])
        if match:
            analysis_gold = (match.group().lower(),) + \
                analysis_gold[source_index + 1:]

        if msa_camel_analyzer is not None:
            analyses_msa_pred = msa_camel_analyzer.analyze(analyzer_input)
            for analysis in analyses_msa_pred:
                analysis['source'] = 'msa'
            analyses_msa_pred = set([_preprocess_analysis(
                analysis) for analysis in analyses_msa_pred])
            analyses_pred = analyses_pred | analyses_msa_pred

        analyses_pred_no_source = set(
            [tuple([f for i, f in enumerate(analysis) if i not in excluded_indexes])
                for analysis in analyses_pred])
        analysis_gold_no_source = tuple(
            [f for i, f in enumerate(analysis_gold) if i not in excluded_indexes])

        is_error = False
        if analysis_gold_no_source in analyses_pred_no_source:
            correct += 1
        elif ldc_bw[0] == '[NONE]' or ldc_bw[1] == '[NONE]':
            drop_cases.append({'word': word_info,
                               'pred': analyses_pred,
                               'gold': analysis_gold,
                               'count': counts[(word, ldc_bw)]})
        else:
            for pred, gold in itertools.product(analyses_pred_no_source, [analysis_gold_no_source]):
                if any(pred[i] != gold[i] for i in range(len(essential_keys_))
                        if i not in [mod_index, gen_index]):
                    continue
                if list(gold).count('u') > 1:
                    raise NotImplementedError
                else:
                    if gold[mod_index] == 'u':
                        correct += 1
                    elif gold[gen_index] != pred[gen_index] or \
                            gold[mod_index] != pred[mod_index]:
                        continue
                    else:
                        raise NotImplementedError
                    break
            else:
                is_error = True
                errors.append({'word': word_info,
                               'pred': analyses_pred,
                               'gold': analysis_gold,
                               'count': counts[(word, ldc_bw)]})

        if not is_error:
            correct_cases.append({'word': word_info,
                                  'pred': analyses_pred,
                                  'gold': analysis_gold,
                                  'count': counts[(word, ldc_bw)]})
        pbar.set_description(f'{len(correct_cases)/total:.1%} (recall)')
        pbar.update(1)

    pbar.close()

    print(
        f"Type space recall: {sum(case['count'] for case in correct_cases)/(sum(case['count'] for case in correct_cases) + sum(case['count'] for case in errors))}")

    recall_print(errors, correct_cases, drop_cases, f'eval/{eval_mode}.tsv')


def compare_print(words, analyses_words, status, results_path, bw=False):
    assert len(analyses_words) == len(status) == len(words)

    def qc_automatic(camel):
        for i, k in enumerate(essential_keys):
            if k in ['source']:
                continue
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
                    non_lex_feats = [c for c in camel.columns if c not in [
                        k] + ['bw', 'filter', 'status', 'status-global', 'qc']]
                    common_feat_rows_filter_camel = camel[non_lex_feats].isin(
                        baseline[non_lex_feats]).all(axis=1)
                    camel.loc[common_feat_rows_filter_camel,
                              'qc'] += f'check-{k}-features'
        return camel

    analysis_results = []
    count = 1
    for (word, analyses_word, status_word) in zip(words, analyses_words, status):
        if status_word == ['NOAN']:
            continue
        columns = ['filter'] + essential_keys + \
            (['bw'] if bw else []) + ['status']
        analyses_word = [(analyses_word[analysis]['source'],) + analysis + ((ar2bw(analyses_word[analysis]['bw']),) if bw else ())
                         for analysis in analyses_word]
        analyses_word = pd.DataFrame(analyses_word)
        status_word = pd.DataFrame(status_word)
        example = pd.concat([analyses_word, status_word], axis=1)
        ex_col = pd.DataFrame(
            [(f'{count} {dediac_bw(word)}',)]*len(example.index))
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
            empty_row = pd.DataFrame(
                [('-',)*len(camel.columns)], columns=camel.columns)
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


def evaluate_verbs_analyzer_comparison(data, n, eval_mode):
    words, analyses, status = [], [], []
    pbar = tqdm(total=min(n, len(data)))
    random.shuffle(data)
    count = 0
    source_index = essential_keys.index('source')
    for word in data:
        if count == n:
            break
        word_dediac = bw2ar(word)
        analyses_camel = analyzer_camel.analyze(word_dediac)
        for analysis in analyses_camel:
            analysis['source'] = 'camel'
        analyses_baseline = analyzer_baseline.analyze(word_dediac)
        for analysis in analyses_baseline:
            match = re.search(r'ADAM|CALIMA|SAMA', analysis['gloss'])
            analysis['source'] = match.group().lower() if match else 'na'
        analyses_camel = {tuple([f for i, f in enumerate(_preprocess_analysis(analysis)) if i != source_index]): analysis
                          for analysis in analyses_camel if analysis['pos'] == 'verb'}
        analyses_baseline = {tuple([f for i, f in enumerate(_preprocess_analysis(analysis)) if i != source_index]): analysis
                             for analysis in analyses_baseline if analysis['pos'] == 'verb'}
        analyses_camel_set, analyses_baseline_set = set(
            analyses_camel), set(analyses_baseline)

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
                source = analyses_baseline[analysis]['source']
                if source == 'sama':
                    status_.append('BASELINE-SAMA')
                elif source == 'adam':
                    status_.append('BASELINE-ADAM')
                elif source == 'calima':
                    status_.append('BASELINE-CALIMA')
                else:
                    status_.append('BASELINE-NA')
            else:
                raise NotImplementedError
        status.append(status_)

        pbar.update(1)
    pbar.close()

    with open('eval/status_compare.tsv', 'w') as f:
        for line in status:
            print(*line, sep='\t', file=f)

    compare_print(words, analyses, status, f'eval/{eval_mode}.tsv', bw=True)

    return status


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
        example = [e for e in example if e in [
            'BASELINE', 'CAMEL', 'BOTH', 'NOAN']]
        example_set = set(example)
        both_count = example.count('BOTH')
        if example == ['NOAN'] or not example:
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
            stats['overlap']['analyses_camel'] += both_count + \
                example.count('CAMEL')
            stats['overlap']['analyses_baseline'] += both_count + \
                example.count('BASELINE')
        elif example_set == {'BOTH', 'BASELINE'}:
            stats['baseline_superset']['word_count'] += 1
            stats['baseline_superset']['overlap'] += both_count
            stats['baseline_superset']['analyses_camel'] += both_count
            stats['baseline_superset']['analyses_baseline'] += both_count + \
                example.count('BASELINE')
        elif example_set == {'BOTH', 'CAMEL'}:
            stats['camel_superset']['word_count'] += 1
            stats['camel_superset']['overlap'] += both_count
            stats['camel_superset']['analyses_baseline'] += both_count
            stats['camel_superset']['analyses_camel'] += both_count + \
                example.count('CAMEL')
        elif example_set == {'CAMEL', 'BASELINE'}:
            stats['disjoint']['word_count'] += 1
            stats['disjoint']['analyses_camel'] += example.count('CAMEL')
            stats['disjoint']['analyses_baseline'] += example.count('BASELINE')
        else:
            raise NotImplementedError

    with open('eval/stats_compare_results.tsv', 'w') as f:
        for row, info in stats.items():
            info = [info[k] for k in ['word_count',
                                      'analyses_camel', 'analyses_baseline', 'overlap']]
            print(row, *info, sep='\t', file=f)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-egy_magold_path", default='eval/ARZ-All-train.113012.magold',
                        type=str, help="Path of the file containing the EGY MAGOLD data to evaluate on.")
    parser.add_argument("-msa_magold_path", default='eval/ATB123-train.102312.calima-msa-s31_0.3.0.magold',
                        type=str, help="Path of the file containing the MSA MAGOLD data to evaluate on.")
    parser.add_argument("-camel_tb_path", default='eval/camel_tb_uniq_types.txt',
                        type=str, help="Path of the file containing the MSA CAMeLTB data to evaluate on.")
    parser.add_argument("-db_dir", default='db_iterations',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-config_file", default='config.json',
                        type=str, help="Config file specifying which sheets to use.")
    parser.add_argument("-egy_config_name", default='all_egy_order-v4',
                        type=str, help="Config name which specifies the path of the EGY Camel DB.")
    parser.add_argument("-msa_config_name", default='all_msa_order-v4',
                        type=str, help="Config name which specifies the path of the MSA Camel DB.")
    parser.add_argument("-msa_baseline_db", default='eval/calima-msa-s31_0.4.2.utf8.db',
                        type=str, help="Path of the MSA baseline DB file we will be comparing against.")
    parser.add_argument("-egy_baseline_db", default='eval/calima-egy-c044_0.2.0.utf8.db',
                        type=str, help="Path of the EGY baseline DB file we will be comparing against.")
    parser.add_argument("-eval_mode", required=True,
                        choices=['recall_msa_magold_raw', 'recall_msa_magold_ldc_dediac',
                                 'recall_egy_magold_raw', 'recall_egy_magold_ldc_dediac', 'recall_egy_union_msa_magold_raw', 'recall_egy_union_msa_magold_ldc_dediac',
                                 'recall_egy_magold_raw_no_lex', 'recall_egy_magold_ldc_dediac_no_lex',
                                 'recall_msa_magold_ldc_dediac_backoff', 'recall_egy_magold_ldc_dediac_backoff',
                                 'compare_camel_tb_msa_raw', 'compare_camel_tb_egy_raw'],
                        type=str, help="What evaluation to perform.")
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

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    with open(args.config_file) as f:
        config = json.load(f)
        config_local = config['local']
        config_egy = config_local[args.egy_config_name]
        config_msa = config_local[args.msa_config_name]

    print('Eval mode:', args.eval_mode)
    if 'msa' in args.eval_mode and 'egy' not in args.eval_mode:
        camel_db_path = os.path.join(args.db_dir, config_msa['output'])
    elif 'egy' in args.eval_mode:
        camel_db_path = os.path.join(args.db_dir, config_egy['output'])
    else:
        raise NotImplementedError

    db_camel = MorphologyDB(camel_db_path)
    print('CAMeL DB path:', camel_db_path)

    if 'backoff' in args.eval_mode:
        analyzer_camel = Analyzer(db_camel, backoff='SMART')
        print('Using SMARTBACKOFF mode.')
    else:
        analyzer_camel = Analyzer(db_camel)

    if 'compare' in args.eval_mode:
        print('Baseline DB path:', args.baseline_db)
        db_baseline = MorphologyDB(args.baseline_db)
        analyzer_baseline = Analyzer(db_baseline)

    if 'msa' in args.eval_mode and 'egy' not in args.eval_mode:
        if 'magold' in args.eval_mode:
            data_path = args.msa_magold_path
        elif 'camel_tb' in args.eval_mode:
            data_path = args.camel_tb_path
        else:
            raise NotImplementedError
    elif 'egy' in args.eval_mode:
        if 'magold' in args.eval_mode:
            data_path = args.egy_magold_path
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print('Data file path:', data_path)
    with open(data_path) as f:
        data = f.read()

    print('Preprocessing data...', end=' ')
    if 'magold' in args.eval_mode:
        print('using dataset:', 'MAGOLD')
        data = _preprocess_magold_data(data)
    elif 'camel_tb' in args.eval_mode:
        print('using dataset:', 'CAMeL TB')
        data = _preprocess_camel_tb_data(data)
    else:
        raise NotImplementedError

    if 'recall' in args.eval_mode:
        print('Eval mode:', 'RECALL')
        msa_camel_analyzer = None
        if 'egy_union_msa' in args.eval_mode and args.msa_config_name:
            print('Using union of EGY and MSA analyses.')
            msa_camel_db = MorphologyDB(os.path.join(
                args.db_dir, config_msa['output']))
            msa_camel_analyzer = Analyzer(msa_camel_db)

        evaluate_verbs_recall(data, args.eval_mode)

    elif 'compare' in args.eval_mode:
        print('Eval mode:', 'COMPARE')
        if 'magold' in args.eval_mode:
            data = [example['info']['word'] for example in data['verbal']]
        elif 'camel_tb' in args.eval_mode:
            print('Eval mode:', 'COMPARE')
            data = [word for word in list(data) if word]
        else:
            raise NotImplementedError

        status = evaluate_verbs_analyzer_comparison(
            data, args.n, args.eval_mode)

        with open('eval/status_compare.tsv') as f:
            compare_results = [line.strip().split('\t')
                               for line in f.readlines()]
            compare_stats(compare_results)
    else:
        raise NotImplementedError
