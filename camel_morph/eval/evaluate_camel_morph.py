# MIT License
#
# Copyright 2022 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import os
import argparse
import json
from tqdm import tqdm
import re
import itertools
import random
from collections import OrderedDict

import gspread
import pandas as pd
from numpy import nan
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser()
parser.add_argument("-egy_magold_path", default='eval_files/ARZ-All-train.113012.magold',
                    type=str, help="Path of the file containing the EGY MAGOLD data to evaluate on.")
parser.add_argument("-msa_magold_path", default='eval_files/ATB123-train.102312.calima-msa-s31_0.3.0.magold',
                    type=str, help="Path of the file containing the MSA MAGOLD data to evaluate on.")
parser.add_argument("-camel_tb_path", default='eval_files/camel_tb_uniq_types.txt',
                    type=str, help="Path of the file containing the MSA CAMeLTB data to evaluate on.")
parser.add_argument("-output_dir", default='eval_files',
                    type=str, help="Path of the directory to output evaluation results.")
parser.add_argument("-config_file", default='camel_morph/configs/config_default.json',
                    type=str, help="Config file specifying which sheets to use.")
parser.add_argument("-egy_config_name", default='all_aspects_egy',
                    type=str, help="Config name which specifies the path of the EGY Camel DB.")
parser.add_argument("-msa_config_name", default='all_aspects_msa',
                    type=str, help="Config name which specifies the path of the MSA Camel DB.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-egy_baseline_db", default='eval_files/calima-egy-c044_0.2.0.utf8.db',
                    type=str, help="Path of the EGY baseline DB file we will be comparing against.")
parser.add_argument("-spreadsheet", default='',
                    type=str, help="Spreadsheet to write the results to.")
parser.add_argument("-compare_stats_cell", default='',
                    type=str, help="Cell at which to start printing the compare results in the sheet.")

parser.add_argument("-pos_or_type", default='', choices=['verbal', 'nominal', 'other',
                                                            'noun', 'noun_num', 'noun_quant', 'noun_prop',
                                                            'adj', 'adj_num', 'adj_comp'],
                    type=str, help="POS or POS type to evaluate.")
"""['recall_msa_magold_raw', 'recall_msa_magold_ldc_dediac',
'recall_egy_magold_raw', 'recall_egy_magold_ldc_dediac',
'recall_egy_union_msa_magold_raw', 'recall_egy_union_msa_magold_ldc_dediac', 'recall_egy_union_msa_magold_calima_dediac',
'recall_egy_magold_raw_no_lex', 'recall_egy_magold_ldc_dediac_no_lex',
'recall_msa_magold_ldc_dediac_backoff', 'recall_egy_magold_ldc_dediac_backoff',
'compare_camel_tb_msa_raw', 'compare_camel_tb_egy_raw']"""
parser.add_argument("-eval_mode", default='',
                    type=str, help="What evaluation to perform.")
parser.add_argument("-n", default=1000000000,
                    type=int, help="Number of instances to evaluate.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")

random.seed(42)
np.random.seed(42)

args, _ = parser.parse_known_args()

with open(args.config_file) as f:
    config = json.load(f)
    configs = config['local']

    config_local_egy = configs.get(args.egy_config_name)
    config_local_msa = configs.get(args.msa_config_name)

if args.camel_tools == 'local':
    camel_tools_dir = config['global']['camel_tools']
    sys.path.insert(0, camel_tools_dir)

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_bw

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

sukun_regex = re.compile('o')
aA_regex = re.compile(r'(?<!^[wf])aA')
poss_regex = re.compile(r'_POSS')

GOLD_DROP_RE = re.compile(r'None|DEFAULT|TBupdate|no_rule')

ESSENTIAL_KEYS = ['source', 'diac', 'lex', 'pos', 'asp', 'mod',
                  'vox', 'per', 'num', 'gen', 'stt', 'cas',
                  'prc0', 'prc1', 'prc1.5', 'prc2', 'prc3',
                  'enc0', 'enc1', 'enc2', 'stem_seg']

FIELD2SENTENCE_INDEX = {f: i
    for i, f in enumerate(['sentence'])}
FIELD2INFO_INDEX = {f: i
    for i, f in enumerate(['word', 'ldc', 'ranking', 'starline', 'calima'])}
FIELD2LDC_INDEX = {f: i
    for i, f in enumerate(['word', 'diac', 'lex', 'bw', 'gloss'])}

POS_TYPE, POS_OR_TYPE = '', ''
ATB_POS_ALL, CAMEL_POS_ALL = {}, {}


def _load_analysis(analysis):
    analysis_ = {}
    for field in analysis:
        field = field.split(':')
        analysis_[field[0]] = ''.join(field[1:])
    return analysis_


def _preprocess_magold_data(gold_data, POS=None,
                            load_analysis_fn=_load_analysis,
                            pos_type=None,
                            field2sentence_index=FIELD2SENTENCE_INDEX,
                            field2info_index=FIELD2INFO_INDEX,
                            field2ldc_index=FIELD2LDC_INDEX):
    global POS_TYPE
    pos_type = POS_TYPE
    ldc_tag_exists = field2info_index is not None and 'ldc' in field2info_index
    if not ldc_tag_exists:
        print('WARNING: Using POS from analysis and not BW tag choose which words to analyze.')
    gold_data = gold_data.split(
        '--------------\nSENTENCE BREAK\n--------------\n')[:-1]
    gold_data = [sentence.split('\n--------------\n')
                 for sentence in gold_data]

    words_start_index = max(field2sentence_index.values()) + 1
    gold_data_ = []
    for example in tqdm(gold_data):
        example_info_list = example[0].split('\n')
        sentence_str = example_info_list[field2sentence_index['sentence']][len(';;; SENTENCE '):]
        words_info = [example_info_list[words_start_index:]] + [x.split('\n') for x in example[words_start_index:]]
        for info in words_info:
            analysis = load_analysis_fn(
                info[field2info_index['calima']].split()[1:])
            if POS is not None:
                if ldc_tag_exists:
                    ldc = info[field2info_index['ldc']]
                    ldc_split = ldc.split(' # ')
                    ldc_BW_components = set(
                        ldc_split[field2ldc_index['bw']].split('+'))
                    intersect = ldc_BW_components & ATB_POS
                    if pos_type == 'other':
                        if not bool(intersect) or \
                           bool(ldc_BW_components & ATB_POS_ALL['nominal']) or \
                           bool(ldc_BW_components & ATB_POS_ALL['verbal']):
                            continue
                    else:
                        if not bool(intersect):
                            continue
                else:
                    if analysis['pos'] not in POS:
                        continue
            
            magold = OrderedDict(
                (f, info[i]) for f, i in field2info_index.items()
                if f in ['ldc', 'ranking', 'starline', 'calima'])
            
            word_info = {
                'info': {
                    'sentence': sentence_str,
                    'word': info[0][len(';;WORD '):],
                    'magold': magold
                },
                'analysis': analysis
            }
            gold_data_.append(word_info)

    return gold_data_


def _preprocess_camel_tb_data(data):
    data = data.split('\n')
    data_fl = OrderedDict()
    for line in data:
        line = line.split()
        if len(line) == 2:
            freq = line[0] if line[0].isdigit() else line[1]
            word = line[0] if not line[0].isdigit() else line[1]
            data_fl[word] = freq

    return data_fl


def _preprocess_ldc_dediac(ldc_diac):
    analyzer_input = re.sub(r'\(null\)', '', re.sub(r'_\d$', '', ldc_diac))
    analyzer_input = re.sub(r'uwA\+', 'uw', analyzer_input)
    analyzer_input = re.sub(r'[aiuo~FKN`\+_]', '', analyzer_input)
    return analyzer_input


def _preprocess_ldc_bw(bw):
    """Filters out .VN in NOUN.VN and ADJ.VN"""
    bw = '+'.join(comp.split('.')[0] for comp in bw.split('+'))
    bw = poss_regex.sub('', bw)
    return bw


def _preprocess_lex_features(lex_feat, f=None):
    lex_feat = ar2bw(lex_feat)
    if f == 'lex':
        lex_feat = re.sub(r'(-[uia]{1,3})?(_\d)?|[][]|-', '', lex_feat)
    lex_feat = lex_feat.replace('_', '')
    lex_feat = sukun_regex.sub('', lex_feat)
    lex_feat = aA_regex.sub('A', lex_feat)
    return lex_feat


def _preprocess_analysis(analysis,
                         defaults,
                         essential_keys=ESSENTIAL_KEYS,
                         optional_keys=[]):
    if analysis['prc0'] in ['mA_neg', 'lA_neg']:
        analysis['prc1.5'] = analysis['prc0']
        analysis['prc0'] = '0'

    pred = []
    for k in essential_keys + optional_keys:
        if k in ['lex', 'diac']:
            pred.append(_preprocess_lex_features(analysis[k], k))
        elif k == 'gen':
            pred.append('m' if analysis[k] == 'u' else analysis[k])
        elif k == 'prc2':
            pred.append(re.sub(r'^([wf])[ai]', r'\1', analysis[k]))
        elif k == 'prc1':
            x = re.sub(r'^[hH]', 'h', analysis[k])
            x = re.sub(r'^([bl])[ai]', r'\1', x)
            pred.append(x)
        elif re.match(r'prc\d|enc\d', k):
            pred.append(analysis.get(k, defaults.get(k, '0')))
        elif k == 'bw':
            bw = [x.split('/')[1] for x in ar2bw(analysis['bw']).split('+')
                  if x and 'STEM' not in x]
            bw = '+'.join(x for x in bw if x not in ['CASE_DEF_U', 'CASE_INDEF_U'])
            bw = poss_regex.sub('', bw)
            pred.append(bw)
        else:
            pred.append(analysis.get(k, defaults.get(k, 'na')))
    return tuple(pred)


def recall_print(examples, results_path,
                 essential_keys=ESSENTIAL_KEYS):
    essential_keys_no_bw_source = [
        k for k in essential_keys if k not in ['bw', 'source']]
    outputs_ = []
    i = 1
    for label, examples_ in examples.items():
        for example in examples_:
            pred = pd.DataFrame(example['pred'])
            gold = pd.DataFrame([example['gold']])
            row = pd.concat([gold, pred], axis=1)
            ex_col = pd.DataFrame(
                [(f"{i} {example['word']['info']['word']}", label,
                  example['match'])]*len(row.index))
            extra_info = pd.DataFrame(
                [(bw2ar(example['word']['info']['sentence']),
                  *example['word']['info']['magold'].values(), example['count'])])
            row = pd.concat([ex_col, extra_info, row], axis=1)
            # Avoids duplicate indexes issue (generates errors)
            row.columns = range(row.columns.size)
            outputs_.append(row)
            i += 1

    outputs = pd.concat(outputs_)
    outputs = outputs.replace(nan, '', regex=True)
    outputs.columns = ['filter', 'label', 'match', 'sentence',
        *example['word']['info']['magold'], 'freq'] + \
        [f'{k}_g' for k in essential_keys] + essential_keys
    outputs['calima_mismatches'] = outputs[essential_keys_no_bw_source].apply(
        lambda row: row.str.contains(']')).sum(axis=1)
    sample_pool_mask = outputs['label'].isin(['wrong', 'noan'])
    sample_size = min(100, int(0.1*sample_pool_mask.sum()))
    sample_filter = set(outputs[sample_pool_mask].sample(
        sample_size, weights='freq', random_state=42)['filter'].tolist())
    outputs['sampled'] = 0
    outputs.loc[outputs['filter'].isin(sample_filter), 'sampled'] = 1

    outputs.reset_index(drop=True, inplace=True)
    outputs.to_csv(results_path, index=False, sep='\t')

    if sh is not None:
        sheet_name = f'{DIALECT}-Recall-{POS_OR_TYPE}'
        print(f'Uploading outputs to sheet {sheet_name} in spreadsheet {sh.title}... ', end='')
        upload(df=outputs, sheet_name=sheet_name, template_sheet_name='Recall-template')
        print('Done.')

    return outputs


def filter_and_rank_analyses(analyses_pred, analysis_gold, analysis_gold_ldc,
                             essential_keys,
                             mode='sama'):
    source_index = essential_keys.index('source')
    lex_index = essential_keys.index('lex')
    diac_index = essential_keys.index('diac')
    bw_index = essential_keys.index('bw')
    analyses_pred_, index2similarity = [], {}
    for analysis_index, analysis in enumerate(analyses_pred):
        analysis_ = []
        for index, f in enumerate(analysis):
            if f == analysis_gold[index]:
                analysis_.append(f)
                index2similarity.setdefault(analysis_index, 0)
                if index == bw_index:
                    index2similarity[analysis_index] += len(analysis) \
                        if mode == 'bw' else 1
                else:
                    index2similarity[analysis_index] += (
                        1.01 if analysis[source_index] == 'main' else 1)
            else:
                analysis_.append(f'[{f}]')

            if index == diac_index and f != analysis_gold_ldc[0] or \
                index == lex_index and f != analysis_gold_ldc[1]:
                analysis_[-1] = f'({analysis_[-1]})'
            elif index == bw_index and f != analysis_gold_ldc[2]:
                ldc_gold_set = set(analysis_gold_ldc[2].split('+'))
                bw_pred_comp_ = []
                for bw_pred_comp in f.split('+'):
                    if bw_pred_comp not in ldc_gold_set:
                        bw_pred_comp = f'({bw_pred_comp})'
                    bw_pred_comp_.append(bw_pred_comp)
                bw_pred_comp = '+'.join(bw_pred_comp_)
                if '[' in analysis_[-1]:
                    analysis_[-1] = f'[{bw_pred_comp}]'
                else:
                    analysis_[-1] = bw_pred_comp

        analyses_pred_.append(tuple(analysis_))
    sorted_indexes = sorted(
        index2similarity.items(), key=lambda x: x[1], reverse=True)
    analyses_pred_ = [analyses_pred_[analysis_index]
                        for analysis_index, _ in sorted_indexes]
    
    return analyses_pred_


def replace_values_with_empty(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                if key not in ['match', 'ldc', 'ranking', 'gold']:
                    obj[key] = ''
            elif isinstance(value, tuple):
                obj[key] = ('',) * len(obj[key])
            else:
                replace_values_with_empty(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = ''
            elif isinstance(item, tuple):
                obj[i] = ('',) * len(obj[i])
            else:
                replace_values_with_empty(item)


def evaluate_recall(data, n, eval_mode, output_path, analyzer_camel,
                    msa_camel_analyzer=None, best_analysis=True,
                    pos_type=None,
                    print_recall=True,
                    essential_keys=ESSENTIAL_KEYS,
                    field2ldc_index=FIELD2LDC_INDEX):
    essential_keys.insert(0, 'bw')
    bw_index = essential_keys.index('bw')
    source_index = essential_keys.index('source')
    lex_index = essential_keys.index('lex')
    diac_index = essential_keys.index('diac')
    stem_seg_index = essential_keys.index('stem_seg')
    essential_keys_ = [k for k in essential_keys if k not in ['bw', 'source']]
    excluded_indexes = [source_index, bw_index]
    if 'no_lex' in eval_mode:
        print('Excluding lexical features from evaluation.')
        essential_keys_ = [
            k for k in essential_keys_ if k not in ['lex', 'diac', 'stem_seg']]
        excluded_indexes.append(lex_index)
        excluded_indexes.append(diac_index)
        excluded_indexes.append(stem_seg_index)
    mod_index = essential_keys_.index('mod')
    gen_index = essential_keys_.index('gen')

    if 'ldc_dediac' in eval_mode:
        print('Analyzer input: LDC DEDIAC')
    elif 'raw' in eval_mode:
        print('Analyzer input: RAW')
    elif 'calima_dediac' in eval_mode:
        print('Analyzer input: CALIMA DEDIAC')

    ldc_index2field = {ldc_index: field
                       for field, ldc_index in field2ldc_index.items()}
    data_unique, counts = OrderedDict(), {}
    for word_info in data:
        if 'ldc' in word_info['info']['magold']:
            ldc = word_info['info']['magold']['ldc'].split(' # ')
            ldc = {ldc_index2field[i]: comp for i, comp in enumerate(ldc)}
            # if ldc['lex'] != '[mumav~il_1]' or ldc['diac'] != 'mumav~iliy':
            #     continue
            diac_lemma_bw = tuple(ldc[f] for f in ['diac', 'lex', 'bw'])
        else:
            diac_lemma_bw = tuple(word_info['analysis'][f]
                                  for f in ['diac', 'lex', 'bw'])
        key = (word_info['info']['word'], diac_lemma_bw)
        counts.setdefault(key, 0)
        counts[key] += 1
        data_unique[key] = word_info

    correct, total = 0, 0
    examples = OrderedDict()
    data_unique = list(data_unique.items())[:n]
    pbar = tqdm(total=len(data_unique))
    for (word, ldc), word_info in data_unique:
        diac_ldc, lex_ldc, bw_ldc = ldc
        total += 1
        analysis_gold = _preprocess_analysis(word_info['analysis'],
                                             analyzer_camel._db.defaults,
                                             essential_keys)

        if 'raw' in eval_mode:
            analyzer_input = word
        elif 'ldc_dediac' in eval_mode:
            analyzer_input = _preprocess_ldc_dediac(diac_ldc)
        elif 'calima_dediac' in eval_mode:
            analyzer_input = _preprocess_ldc_dediac(analysis_gold[diac_index])

        analyzer_input = bw2ar(analyzer_input)

        analyses_pred_ = analyzer_camel.analyze(analyzer_input)
        for analysis in analyses_pred_:
            analysis['source'] = 'main'
        analyses_pred = set(
            [_preprocess_analysis(analysis, analyzer_camel._db.defaults, essential_keys)
             for analysis in analyses_pred_])

        match = re.search(r'ADAM|CALIMA|SAMA', word_info['analysis']['gloss'])
        if match:
            analysis_gold = (match.group().lower(),) + \
                analysis_gold[source_index + 1:]

        if msa_camel_analyzer is not None:
            analyses_msa_pred = msa_camel_analyzer.analyze(analyzer_input)
            for analysis in analyses_msa_pred:
                analysis['source'] = 'msa'
            analyses_msa_pred = set(
                [_preprocess_analysis(analysis, msa_camel_analyzer._db.defaults, essential_keys)
                 for analysis in analyses_msa_pred])
            analyses_pred = analyses_pred | analyses_msa_pred

        analyses_pred_no_source = set(
            [tuple([f for i, f in enumerate(analysis) if i not in excluded_indexes])
                for analysis in analyses_pred])
        analysis_gold_no_source = tuple(
            [f for i, f in enumerate(analysis_gold) if i not in excluded_indexes])
        
        analysis_gold_ldc = (_preprocess_lex_features(diac_ldc),
                             _preprocess_lex_features(lex_ldc, f='lex'),
                             _preprocess_ldc_bw(bw_ldc))
        analyses_pred_ldc = {(a[diac_index], a[lex_index], a[bw_index])
                             for a in analyses_pred}

        mode = 'bw' if 'ldc_dediac_match' in eval_mode else 'sama'
        analyses_pred = filter_and_rank_analyses(
            analyses_pred, analysis_gold, analysis_gold_ldc, essential_keys, mode)
        if best_analysis:
            analyses_pred = analyses_pred[0:1]
        else:
            #FIXME: might need fixing; why are we excluding source, lex, etc.?
            analyses_pred = [
                analysis for analysis in analyses_pred
                if all(analysis[i] == analysis_gold[i] for i, k in enumerate(essential_keys)
                        if k not in ['source', 'lex', 'diac', 'stem_seg'])]
        match = ''
        if analysis_gold_ldc in analyses_pred_ldc:
            match = 'ldc'
        else:
            for i, f in enumerate(['diac', 'lex', 'bw']):
                if analysis_gold_ldc[i] in [a[i] for a in analyses_pred_ldc]:
                    match += (' ' if match else '') + f'ldc:{f}'
        
        if analysis_gold_no_source in analyses_pred_no_source:
            match += ' feats' if match else 'feats'
        
        is_error = False
        if 'ldc_dediac_match' not in eval_mode and \
                analysis_gold_no_source in analyses_pred_no_source or \
            'ldc_dediac_match' in eval_mode and \
                analysis_gold_ldc in analyses_pred_ldc:
            correct += 1
        elif pos_type == 'verbal':
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
        else:
            is_error = True

        is_drop = bool(GOLD_DROP_RE.search(' '.join([diac_ldc, lex_ldc])))
        label = ('wrong' if is_error else 'correct') if len(analyses_pred) else 'noan'
        if is_drop:
            label = 'drop-' + label
        examples.setdefault(label, []).append(
            {'word': word_info,
             'match': match,
             'pred': analyses_pred,
             'gold': analysis_gold,
             'count': counts[(word, ldc)]})
        
        correct = len(examples.get('correct', []))
        wrong = len(examples.get('wrong', []))
        total = correct + wrong
        recall_type = (correct / total) if total else 0
        pbar.set_description(f'{recall_type:.1%} (recall)')
        pbar.update(1)

    pbar.close()
    correct = sum(e['count'] for e in examples.get('correct', []))
    wrong = sum(e['count'] for e in examples.get('wrong', []))
    total = correct + wrong
    recall_token = (correct / total) if total else 0
    print(f"Token space recall: {recall_token:.2%}")

    no_print_cases = [e for l, examples_ in examples.items() for e in examples_
                      if l == 'correct' or 'drop' in l]
    if print_recall:
        replace_values_with_empty(no_print_cases)
        recall_print(examples, output_path, essential_keys=essential_keys)

    correct_cases = examples['correct']
    return correct_cases


def compare_print(words, analyses_words, status, results_path,
                  bw=False, essential_keys=ESSENTIAL_KEYS):
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
    
    columns = ['filter'] + essential_keys + (['bw'] if bw else []) + ['status']
    columns_extra = ['status-global', 'qc', 'cond_format']
    columns_all = columns + columns_extra

    analysis_results = []
    count = 1
    for (word, analyses_word, status_word) in zip(words, analyses_words, status):
        if status_word == ['NOAN']:
            continue
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
        baseline = example[example['status'].str.contains('BASELINE')]
        both = example[example['status'] == 'BOTH']
        camel['status-global'], baseline['status-global'], both['status-global'] = '', '', ''
        camel['qc'], baseline['qc'], both['qc'] = '', '', ''
        camel.sort_values(by=['lex'], inplace=True)
        baseline.sort_values(by=['lex'], inplace=True)
        both.sort_values(by=['lex'], inplace=True)
        if baseline.empty or camel.empty or both.empty:
            empty_row = pd.DataFrame(
                [('-',)*len(camel.columns)], columns=camel.columns)
            empty_row.loc[0, 'filter'] = f'{count} {dediac_bw(word)}'
        if not both.empty:
            if not baseline.empty and not camel.empty:
                camel['status-global'] = 'both-camel-entry'
                camel = qc_automatic(camel)
                baseline['status-global'] = 'both-baseline-entry'
                both['status-global'] = 'both-shared-entry'
            elif not baseline.empty and camel.empty:
                camel = empty_row.copy()
                camel['status-global'] = 'baseline-super-noadd-camel'
                baseline['status-global'] = 'baseline-super-entry'
                both['status-global'] = 'baseline-super-shared-entry'
            elif baseline.empty and not camel.empty:
                camel['status-global'] = 'camel-super-entry'
                camel = qc_automatic(camel)
                baseline = empty_row.copy()
                baseline['status-global'] = 'camel-super-noadd-baseline'
                both['status-global'] = 'camel-super-shared-entry'
            else:
                camel = empty_row
                camel['status-global'] = 'exact-match-noadd-camel'
                baseline = empty_row.copy()
                baseline['status-global'] = 'exact-match-noadd-baseline'
                both['status-global'] = 'exact-match'
        else:
            both = empty_row.copy()
            if not baseline.empty and not camel.empty:
                if len(baseline.index) < len(camel.index):
                    status_ = 'camel-only-disj'
                else:
                    status_ = 'baseline-only-disj'
                both['status-global'] = status_
                camel['status-global'] = status_ + 'camel-entry'
                camel = qc_automatic(camel)
                baseline['status-global'] = status_ + 'baseline-entry'
            elif not baseline.empty and camel.empty:
                camel = empty_row.copy()
                camel['status-global'] = 'baseline-only'
                baseline['status-global'] = 'baseline-only-entry'
            elif baseline.empty and not camel.empty:
                baseline = empty_row.copy()
                baseline['status-global'] = 'camel-only'
                camel['status-global'] = 'camel-only-entry'
                camel = qc_automatic(camel)
            else:
                # Should never happen because it is the case where nothing is generated
                # on either side.
                raise NotImplementedError

        example = pd.concat([both, camel, baseline])
        example['cond_format'] = count % 2
        analysis_results.append(example)
        count += 1

    analysis_results = pd.concat(analysis_results)
    analysis_results = analysis_results.replace(nan, '', regex=True)
    analysis_results.columns = columns_all
    analysis_results = analysis_results[columns_all]
    analysis_results.to_csv(results_path, index=False, sep='\t')

    if sh is not None:
        upload(df=analysis_results,
               sheet_name=f"{DIALECT}-Compare-{POS_OR_TYPE}",
               template_sheet_name='Compare-template')


def evaluate_analyzer_comparison(data, n, output_path,
                                 analyzer_camel, analyzer_baseline,
                                 essential_keys=ESSENTIAL_KEYS):
    words, analyses, status = [], [], []
    pbar = tqdm(total=min(n, len(data)))
    random.shuffle(data)
    count = 0
    source_index = essential_keys.index('source')
    for word in data:
        if count == n:
            break
        word_ar = bw2ar(word)
        analyses_camel = analyzer_camel.analyze(word_ar)
        for analysis in analyses_camel:
            analysis['source'] = 'camel'
        analyses_baseline = analyzer_baseline.analyze(word_ar)
        for analysis in analyses_baseline:
            match = re.search(r'ADAM|CALIMA|SAMA', analysis['gloss'])
            analysis['source'] = match.group().lower() if match else 'na'
        analyses_camel = {
            tuple([f for i, f in enumerate(_preprocess_analysis(analysis, analyzer_camel._db.defaults, essential_keys))
                   if i != source_index]): analysis
            for analysis in analyses_camel if analysis['pos'] in CAMEL_POS}
        analyses_baseline = {
            tuple([f for i, f in enumerate(_preprocess_analysis(analysis, analyzer_baseline._db.defaults, essential_keys))
                   if i != source_index]): analysis
            for analysis in analyses_baseline if analysis['pos'] in CAMEL_POS}
        analyses_camel_set, analyses_baseline_set = set(
            analyses_camel), set(analyses_baseline)

        words.append(ar2bw(word_ar))

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

    compare_print(words, analyses, status, output_path, bw=True,
                  essential_keys=essential_keys)

    return status


def compare_stats(compare_results):
    stats = {}
    for example in compare_results:
        example = [e.split('-')[0] for e in example]
        example_set = set(example)
        both_count = example.count('BOTH')
        if example == ['NOAN'] or not example:
            stats.setdefault('noan', {}).setdefault('word_count', 0)
            stats['noan']['word_count'] += 1
        elif example_set == {'BASELINE'} or example_set == {'CAMEL'}:
            system = list(example_set)[0].lower()
            stats.setdefault(f'only_{system}', {}).setdefault('word_count', 0)
            stats.setdefault(f'only_{system}', {}).setdefault(f'analyses_{system}', 0)
            stats[f'only_{system}']['word_count'] += 1
            stats[f'only_{system}'][f'analyses_{system}'] += len(example)
        elif example_set == {'BOTH'}:
            stats.setdefault('equal', {}).setdefault('word_count', 0)
            stats.setdefault('equal', {}).setdefault('overlap', 0)
            stats.setdefault('equal', {}).setdefault('analyses_camel', 0)
            stats.setdefault('equal', {}).setdefault('analyses_baseline', 0)
            stats['equal']['word_count'] += 1
            stats['equal']['overlap'] += both_count
            stats['equal']['analyses_camel'] += len(example)
            stats['equal']['analyses_baseline'] += len(example)
        elif example_set == {'BOTH', 'CAMEL', 'BASELINE'}:
            stats.setdefault('overlap', {}).setdefault('word_count', 0)
            stats.setdefault('overlap', {}).setdefault('overlap', 0)
            stats.setdefault('overlap', {}).setdefault('analyses_camel', 0)
            stats.setdefault('overlap', {}).setdefault('analyses_baseline', 0)
            stats['overlap']['word_count'] += 1
            stats['overlap']['overlap'] += both_count
            stats['overlap']['analyses_camel'] += both_count + example.count('CAMEL')
            stats['overlap']['analyses_baseline'] += both_count + example.count('BASELINE')
        elif example_set == {'BOTH', 'BASELINE'} or example_set == {'BOTH', 'CAMEL'}:
            system = list(example_set - {'BOTH'})[0].lower()
            system_other = 'camel' if system == 'baseline' else 'baseline'
            stats.setdefault(f'{system}_superset', {}).setdefault('word_count', 0)
            stats.setdefault(f'{system}_superset', {}).setdefault('overlap', 0)
            stats.setdefault(f'{system}_superset', {}).setdefault(f'analyses_camel', 0)
            stats.setdefault(f'{system}_superset', {}).setdefault(f'analyses_baseline', 0)
            stats[f'{system}_superset']['word_count'] += 1
            stats[f'{system}_superset']['overlap'] += both_count
            stats[f'{system}_superset'][f'analyses_{system_other}'] += both_count
            stats[f'{system}_superset'][f'analyses_{system}'] += both_count + example.count(system.upper())
        elif example_set == {'CAMEL', 'BASELINE'}:
            stats.setdefault('disjoint', {}).setdefault('word_count', 0)
            stats.setdefault('disjoint', {}).setdefault('analyses_camel', 0)
            stats.setdefault('disjoint', {}).setdefault('analyses_baseline', 0)
            stats['disjoint']['word_count'] += 1
            stats['disjoint']['analyses_camel'] += example.count('CAMEL')
            stats['disjoint']['analyses_baseline'] += example.count('BASELINE')
        else:
            raise NotImplementedError

    header_col = ['word_count', 'analyses_camel', 'analyses_baseline', 'overlap']
    header_row = ['noan', 'only_baseline', 'only_camel', 'equal', 'disjoint',
                  'overlap', 'camel_superset', 'baseline_superset']
    table = []
    with open(os.path.join(output_dir, 'stats_compare_results_1.tsv'), 'w') as f:
        table.append([POS_OR_TYPE, *header_col])
        print(POS_OR_TYPE, *header_col, sep='\t', file=f)
        for row in header_row:
            table.append([])
            row_ = stats.get(row)
            for col in header_col:
                table[-1].append(row_.get(col, 0) if row_ is not None else 0)
            table[-1].insert(0, row)
            print(*table[-1], sep='\t', file=f)

    if sh is not None:
        sheet = sh.worksheet(title='Stats-Compare')
        start_col, start_row = args.compare_stats_cell[0], args.compare_stats_cell[1:]
        start_number = ord(start_col) - ord('A')
        end_number = start_number + len(header_col)
        end_row = int(start_row) + len(header_row)
        end_col = (chr(ord('A') + end_number // 27) if end_number >= 26 else '') + \
                     chr(ord('A') + end_number % 26)
        sheet.update(f'{start_col}{start_row}:{end_col}{end_row}', table)


    return stats


def load_required_pos(required_pos_or_type):
    with open('misc_files/atb2camel_pos.json') as f:
        pos_type2atb2camel_pos = json.load(f)
        pos_type2atb2camel_pos['any'] = {**pos_type2atb2camel_pos['verbal'],
            **pos_type2atb2camel_pos['nominal'], **pos_type2atb2camel_pos['other']}
    
    ATB_POS, CAMEL_POS = set(), set()
    for pos_type, atb_pos2camel_pos in pos_type2atb2camel_pos.items():
        if pos_type != 'not_mappable':
            atb_pos2camel_pos = {
                atb: set(camel) if type(camel) is list else set([camel])
                for atb, camel in atb_pos2camel_pos.items()}
            if required_pos_or_type in ['verbal', 'nominal', 'other', 'any']:
                if pos_type == required_pos_or_type:
                    ATB_POS.update(atb_pos2camel_pos.keys())
                    CAMEL_POS.update(*[map(str.lower, camel_pos)
                                       for camel_pos in atb_pos2camel_pos.values()])
            else:
                ATB_POS.update([atb
                                for atb, camel in atb_pos2camel_pos.items()
                                if required_pos_or_type.upper() in camel])
                CAMEL_POS.update([required_pos_or_type])
    
    return ATB_POS, CAMEL_POS


def upload(df, sheet_name, template_sheet_name):
    sheets = sh.worksheets()
    new_sheet = False
    if sheet_name in [sheet.title for sheet in sheets]:
        sheet = sh.worksheet(title=sheet_name)
    else:
        sheet_default_id = [(sheet.id, sheet.index)
                            for sheet in sheets if sheet.title == template_sheet_name][0]
        sheet = sh.duplicate_sheet(sheet_default_id[0], sheet_default_id[1] + 1,
                                    new_sheet_name=sheet_name)
        new_sheet = True
    sheet_df = pd.DataFrame(sheet.get_all_records())
    assert list(sheet_df.columns[:len(df.columns)]) == list(df.columns)
    if not new_sheet:
        sheet.batch_clear(['A:BC'])
    sheet.update('A:BC', [df.columns.values.tolist()] + df.values.tolist())
    if len(sheet_df.index) > len(df.index) and not new_sheet:
        sheet.delete_rows(len(df.index) + 2, len(sheet_df.index) + 1)
    else:
        sheet.set_basic_filter('A:BC')


if __name__ == "__main__":
    if args.spreadsheet:
        sa = gspread.service_account(config['global']['service_account'])
        sh = sa.open(args.spreadsheet)
    else:
        sh = None

    print()
    print('Eval mode:', args.eval_mode)
    if 'msa' in args.eval_mode and 'egy' not in args.eval_mode:
        DIALECT = 'MSA'
        config_local = config_local_msa
        camel_db_path = os.path.join('databases/camel-morph-msa', config_local['db'])

    elif 'egy' in args.eval_mode:
        DIALECT = 'EGY'
        config_local = config_local_egy
        camel_db_path = os.path.join('databases/camel-morph-egy', config_local_egy['db'])
    else:
        raise NotImplementedError
    
    POS_TYPE = config_local.get('pos_type')
    if args.pos_or_type:
        POS_OR_TYPE = args.pos_or_type
    elif config_local.get('pos'):
        POS_OR_TYPE = config_local['pos']
    else:
        if POS_TYPE:
            POS_OR_TYPE = POS_TYPE
        else:
            POS_OR_TYPE = 'any'

    ATB_POS, CAMEL_POS = load_required_pos(POS_OR_TYPE)
    for pos_type in ['nominal', 'verbal', 'other']:
        atb_pos_, camel_pos_ = load_required_pos(pos_type)
        ATB_POS_ALL[pos_type], CAMEL_POS_ALL[pos_type] = atb_pos_, camel_pos_

    output_dir = os.path.join(args.output_dir, POS_OR_TYPE)
    os.makedirs(output_dir, exist_ok=True)

    db_camel = MorphologyDB(camel_db_path)
    print('CAMeL DB path:', camel_db_path)

    if 'backoff' in args.eval_mode:
        analyzer_camel = Analyzer(db_camel, backoff='SMART')
        print('Using SMARTBACKOFF mode.')
    else:
        analyzer_camel = Analyzer(db_camel)

    if 'compare' in args.eval_mode:
        if 'msa' in args.eval_mode:
            db_baseline_path = args.msa_baseline_db
        elif 'egy' in args.eval_mode:
            db_baseline_path = args.egy_baseline_db
        else:
            raise NotImplementedError
        print('Baseline DB path:', db_baseline_path)
        db_baseline = MorphologyDB(db_baseline_path)
        analyzer_baseline = Analyzer(db_baseline, legacy=True)

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

    print(f'POS (type): {POS_OR_TYPE}')

    print('Preprocessing data...', end=' ')
    if 'magold' in args.eval_mode:
        print('using dataset:', 'MAGOLD')
        data = _preprocess_magold_data(data, CAMEL_POS)
    elif 'camel_tb' in args.eval_mode:
        print('using dataset:', 'CAMeL TB')
        data = _preprocess_camel_tb_data(data)
    else:
        raise NotImplementedError

    output_path = os.path.join(output_dir, f'{args.eval_mode}.tsv')
    if 'recall' in args.eval_mode:
        print('Eval mode:', 'RECALL')
        msa_camel_analyzer = None
        if 'egy_union_msa' in args.eval_mode and args.msa_config_name:
            print('Using union of EGY and MSA analyses.')
            msa_camel_db = MorphologyDB(os.path.join(
                'databases/camel-morph-msa', config_local_msa['db']))
            msa_camel_analyzer = Analyzer(msa_camel_db)

        evaluate_recall(data, args.n, args.eval_mode, output_path,
                        analyzer_camel, msa_camel_analyzer,
                        pos_type=POS_TYPE)

    elif 'compare' in args.eval_mode:
        print('Eval mode:', 'COMPARE')
        if 'magold' in args.eval_mode:
            data = [example['info']['word'] for example in data]
        elif 'camel_tb' in args.eval_mode:
            data = [word for word in list(data) if word]
        else:
            raise NotImplementedError

        status = evaluate_analyzer_comparison(
            data, args.n, output_path, analyzer_camel, analyzer_baseline)

        compare_stats(status)
    else:
        raise NotImplementedError
    print()
