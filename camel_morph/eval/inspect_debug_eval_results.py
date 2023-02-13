import re
import sys
import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import itertools
from collections import Counter
import pickle
from textwrap import wrap

import eval_utils
from eval_utils import color, bold, underline

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='eval_files/report_default',
                    type=str, help="Config file specifying which sheets to use.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-camel_db", default='databases/camel-morph-msa/XYZ_msa_all_v1.0.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-report_dir", default='eval_files/report_default',
                    type=str, help="Paths of the directory containing partial reports generated by the full generative evaluation.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args()

with open('configs/config_default.json') as f:
    config = json.load(f)

if args.camel_tools == 'local':
    camel_tools_dir = config['global']['camel_tools']
    sys.path.insert(0, camel_tools_dir)

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.generator import Generator
from camel_tools.utils.charmap import CharMapper

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

db_baseline_gen = MorphologyDB(args.msa_baseline_db, flags='g')
generator_baseline = Generator(db_baseline_gen)

db_camel_gen = MorphologyDB(args.camel_db, flags='g')
generator_camel = Generator(db_camel_gen)

try:
    results_debug_eval = eval_utils.load_results_debug_eval(args.report_dir)
except:
    pass

try:
    info = eval_utils.load_matrices(args.report_dir)
    recall_mat_baseline = info['recall_mat_baseline']
    recall_mat_camel = info['recall_mat_camel']
    camel_minus_baseline_mat = info['camel_minus_baseline_mat']
    baseline_minus_camel_mat = info['baseline_minus_camel_mat']
    no_intersect_mat = info['no_intersect_mat']
    index2analysis, analysis2index = info['index2analysis'], info['analysis2index']
except:
    pass

index2lemmas_pos = eval_utils.load_index2lemmas_pos(args.report_dir)

with open('eval_files/results_multiproc_all_4/failed.pkl', 'rb') as f:
    failed = pickle.load(f)

# faulty_lemma2analysis_matching = {}
# for diff, analyses in results_debug_eval['baseline']['baseline'].items():
#     if re.search(r'enc0', diff):
#         for analysis, indexes in analyses.items():
#             if len(indexes) > 1000:
#                 continue
#             for index in indexes:
#                 faulty_lemma2analysis_matching.setdefault(
#                     index2lemmas_pos[index], {}).setdefault(diff, []).append(analysis)
                
# analyses = [generator_camel.generate(lemma, eval_utils.construct_feats('p+i+a+3+p+m+na+na+0+0+0+0+0'.split('+'), 'verb'))
#             for lemma, _ in faulty_lemma2analysis_matching]
# lemmas_intrans = []
# for i, analyses_ in enumerate(analyses):
#     match = re.search(r'\[CS:([^\]]+)\]', analyses_[0]['stemcat']).group(1)
#     if 'intrans' in match:
#         lemmas_intrans.append(analyses_[0]['lex'])

# not_validated_sorting = {}
# for error_type, analyses_indexes in results_debug_eval['baseline']['not_validated'].items():
#     for analysis, indexes_ in analyses_indexes.items():
#         if 'neg' in analysis:
#             indexes = not_validated_sorting.setdefault('neg', {}).setdefault(analysis, [])
#             indexes += indexes_
#         elif '1+' in analysis and re.search(r'1[ps]_dobj', analysis):
#             indexes = not_validated_sorting.setdefault('per:1+1_dobj', {}).setdefault(analysis, [])
#             indexes += indexes_
#         elif re.match(r'i\+u', analysis):
#             indexes = not_validated_sorting.setdefault('asp:i+mod:u', {}).setdefault(analysis, [])
#             indexes += indexes_
#         elif 'prep' in analysis:
#             indexes = not_validated_sorting.setdefault('prep', {}).setdefault(analysis, [])
#             indexes += indexes_
#         elif re.match(r'na', analysis):
#             indexes = not_validated_sorting.setdefault('asp:na', {}).setdefault(analysis, [])
#             indexes += indexes_
#         else:
#             indexes = not_validated_sorting.setdefault('unk', {}).setdefault(analysis, [])
#             indexes += indexes_

# faulty_lemma2analysis_unk = {}
# for analysis, indexes in not_validated_sorting['unk'].items():
#     for index in indexes:
#         faulty_lemma2analysis_unk.setdefault(index2lemmas_pos[index], []).append(analysis)
# pass
report_title = 'Evaluation Report - Camel Morph - Verbs'
try:
    terminal_size_col = os.get_terminal_size().columns
except:
    terminal_size_col = len(report_title)
print()
print('=' * terminal_size_col)
print(report_title)
print('=' * terminal_size_col)
print()
baseline_path = color(bold(args.msa_baseline_db), 'cyan')
camel_path = color(bold(args.camel_db), 'cyan')
print(bold(underline('DBs used for analysis')))
print('CALIMA: ' + color(bold(args.msa_baseline_db), 'cyan'))
print('Camel: ' + color(bold(args.camel_db), 'cyan'))
print()
print(bold(underline('Verb Lemmas overlap between CALIMA and Camel')))
print()
lemmas_pos_baseline = eval_utils.get_all_lemmas_from_db(MorphologyDB(args.msa_baseline_db))
lemmas_baseline = set([lemma_pos[0] for lemma_pos in lemmas_pos_baseline if lemma_pos[1] == 'verb'])
lemmas_pos_camel = eval_utils.get_all_lemmas_from_db(MorphologyDB(args.camel_db))
lemmas_camel = set([lemma_pos[0] for lemma_pos in lemmas_pos_camel if lemma_pos[1] == 'verb'])
rows = []
header = ['A . B', 'Result', '# lemmas', '(%)']
lemmas_baseline_minus_camel = lemmas_baseline - lemmas_camel
lemmas_camel_minus_baseline = lemmas_camel - lemmas_baseline
lemmas_intersect = lemmas_camel & lemmas_baseline
lemmas_union = lemmas_camel | lemmas_baseline
rows.append(['CALIMA - Camel',
             f'{len(lemmas_baseline_minus_camel):,}',
             f'{len(lemmas_baseline):,} in A',
             f'{len(lemmas_baseline_minus_camel) / len(lemmas_baseline):.2%}'])
rows.append(['Camel - Calima',
             f'{len(lemmas_camel_minus_baseline):,}',
             f'{len(lemmas_camel):,} in A',
             f'{len(lemmas_camel_minus_baseline) / len(lemmas_camel):.2%}'])
rows.append(['Camel ∩ Calima',
             bold(color(f'{len(lemmas_intersect):,}', 'green')),
             f'{len(lemmas_union):,} in A ∪ B',
             f'{len(lemmas_intersect) / len(lemmas_union):.2%}'])
print(tabulate(rows, tablefmt='fancy_grid', headers=header))
print()
print('CALIMA - Camel:')
print('\n'.join(wrap(' '.join(sorted(map(ar2bw, lemmas_baseline_minus_camel))), 100)))
print()
print('Camel - CALIMA:')
print('\n'.join(wrap(' '.join(sorted(map(ar2bw, lemmas_camel_minus_baseline))), 100)))
print()
# Trun off entries in baseline for which Camel has no lemma
recall_mat_baseline[~np.any(recall_mat_camel, axis=1)] = 0
print(bold(underline('Overlap statistics of generated diacs between CALIMA and Camel')))
print()

mask_not_equal_0_baseline = recall_mat_baseline != 0
mask_not_equal_0_camel = recall_mat_camel != 0

rows = []
num_valid_feats = int(np.sum(np.any(recall_mat_camel|recall_mat_baseline, axis=0)))
num_valid_lemmas = int(np.sum(np.any(recall_mat_camel|recall_mat_baseline, axis=1)))
slots_total = num_valid_feats * num_valid_lemmas
slots_filled_mask = mask_not_equal_0_camel | mask_not_equal_0_baseline
slots_filled_total = int(np.sum(slots_filled_mask))
rows.append(['Number of slots filled by at least one of the systems (0-x, x-0, x-y)',
             bold(color(f'{slots_filled_total:,}', 'warning')),
             f'{slots_filled_total/slots_total:.0%}'])
rows.append(['Number of slots filled by none of the systems (0-0)',
             f'{slots_total - slots_filled_total:,}',
             f'{(slots_total - slots_filled_total)/slots_total:.0%}'])
rows.append(['Total number of slots per system',
             f'{slots_total:,}' + ' (' + bold(color(f'\n{num_valid_lemmas:,} ', 'green')) + '× ' +
             bold(color(f'{num_valid_feats:,}', 'green')) + ')',
             f'{1:.0%}'])
assert len(recall_mat_baseline) == len(recall_mat_camel)
print('Number of lemmas evaluated on: ' + bold(color(f'{len(recall_mat_baseline):,}', 'green')) +' (Camel ∩ CALIMA)')
print('Total number of feature combinations across both systems: ' + bold(color(f'{num_valid_feats:,}', 'green')))
print(tabulate(rows, tablefmt='fancy_grid', maxcolwidths=[40, 17, None]))
print()
print(bold(underline(f'Distribution of feature combination availability across systems (0-x, x-0, x-y)')))
print(color('Note: A slot is a matrix cell representing a lemma and a feature combinatinon from which one a more diacs were generated.', 'warning'))
print()
rows = []
header = ['Desc', 'CALIMA', 'Camel', 'Slots', '(%)', 'Lemmas', '(%)', 'Feat combs', '(%)']
# Slot space
match = int(np.sum(mask_not_equal_0_baseline & mask_not_equal_0_camel))
overgeneration = int(np.sum((recall_mat_baseline == 0) & mask_not_equal_0_camel))
undergeneration = int(np.sum((recall_mat_camel == 0) & mask_not_equal_0_baseline))
total_dist = overgeneration + undergeneration + match

any_feats_baseline = np.any(recall_mat_baseline, axis=0)
any_feats_camel = np.any(recall_mat_camel, axis=0)
any_lemmas_baseline = np.any(recall_mat_baseline, axis=1)
any_lemmas_camel = np.any(recall_mat_camel, axis=1)
# Feature space
feats_baseline_only = int(np.sum(any_feats_baseline & ~any_feats_camel))
feats_camel_only = int(np.sum(any_feats_camel & ~any_feats_baseline))
feats_both = int(np.sum(any_feats_camel & any_feats_baseline))
feats_total = feats_baseline_only + feats_camel_only + feats_both
# Lemma space
lemmas_baseline_only = int(np.sum(any_lemmas_baseline & ~any_lemmas_camel))
lemmas_camel_only = int(np.sum(any_lemmas_camel & ~any_lemmas_baseline))
lemmas_both = int(np.sum(any_lemmas_camel & any_lemmas_baseline))
lemmas_total = lemmas_baseline_only + lemmas_camel_only + lemmas_both

assert lemmas_total == num_valid_lemmas

rows.append(['Overgeneration',
             '0', 'x',
             f'{overgeneration:,}',
             f'{overgeneration/slots_filled_total:.0%}',
             f'{lemmas_camel_only:,}', f'{lemmas_camel_only/lemmas_total:.0%}',
             f'{feats_camel_only:,}', f'{feats_camel_only/feats_total:.0%}'])
rows.append(['Undergeneration',
             'x', '0',
             f'{undergeneration:,}',
             f'{undergeneration/slots_filled_total:.0%}',
             f'{lemmas_baseline_only:,}', f'{lemmas_baseline_only/lemmas_total:.0%}',
             f'{feats_baseline_only:,}', f'{feats_baseline_only/feats_total:.0%}'])
rows.append(['Match',
             'x', 'y',
             bold(color(f'{match:,}', 'cyan')),
             f'{match/slots_filled_total:.0%}',
             f'{lemmas_both:,}', f'{lemmas_both/lemmas_total:.0%}',
             f'{feats_both:,}', f'{feats_both/feats_total:.0%}'])
rows.append(['Total',
             '-', '-',
             bold(color(f'{total_dist:,}', 'warning')),
             f'{total_dist/slots_filled_total:.0%}',
             bold(color(f'{lemmas_total:,}', 'green')), f'{1:.0%}',
             bold(color(f'{feats_total:,}', 'green')), f'{1:.0%}'])
print(tabulate(rows, tablefmt='fancy_grid', headers=header))
print()

print(bold(underline('Breakdown of the x-y set (coverage of CALIMA by Camel Morph)')))
print(color('Note: # diac here is the number of unique diacs generated, and not the number of analyses generated which could generally be more.', 'warning'))
print(color('Note: Slots here is the number of feature combinations that both systems were able to generate for listed lemmas.', 'warning'))
print(color('Note: A slot is a matrix cell representing a lemma and a feature combinatinon from which one a more diacs were generated.', 'warning'))
print(color('Note: Slot space means we are counting number of slots while in diac space were are counting number of diacs..', 'warning'))
print('\n'.join(wrap(color(('Note: Top number in recall distribution (last columns) indicates recall (diac space) by Camel of CALIMA and bottom is in slot'
                            'space (displayed only if different). Total recall in slot space basically represents the sum of all categories (in slot space) minus no_intersect.'), 'warning'), 100)))
print(color(f'Note: All total recall values are micro-averaged.', 'warning'))
print()
# exact_match = sum(info['matching'] for info in eval_with_clitics['both'].values())
# total = sum(info['total'] for info in eval_with_clitics['both'].values())
num_diac_baseline_possible, num_diac_camel_possible = [], []
for x in range(1, np.max(recall_mat_baseline) + 1):
    if x in recall_mat_baseline:
        num_diac_baseline_possible.append(x)
for x in range(1, np.max(recall_mat_baseline) + 1):
    if x in recall_mat_camel:
        num_diac_camel_possible.append(x)

rows = []
header = ['# diac (CALIMA)', '# diac (Camel)', 'Slots', '(%)', 'Lemmas', '(%)', 'Feat combs', '(%)',
          'Example', 'No intersec', 'Exact match', 'Baseline super', 'Camel super', 'Intersec', 'Recall']

match_comb_ = 0
combinations = list(itertools.product(num_diac_baseline_possible, num_diac_camel_possible))
combinations.append(('-', '-'))
for combination in tqdm(combinations):
    num_diac_baseline, num_diac_camel = combination
    if combination != ('-', '-'):
        match_comb_mask = (recall_mat_baseline == num_diac_baseline) & (recall_mat_camel == num_diac_camel)
    else:
        match_comb_mask = (recall_mat_baseline != 0) & (recall_mat_camel != 0)
    num_lemmas_match = int(np.sum(np.any(match_comb_mask, axis=1)))
    num_feats_match = int(np.sum(np.any(match_comb_mask, axis=0)))
    match_comb = int(np.sum(match_comb_mask))
    if match_comb == 0:
        continue
    match_comb_ += match_comb

    no_intersect = int(np.sum(no_intersect_mat[match_comb_mask]))
    exact_match_indexes = ((baseline_minus_camel_mat == 0) &
                           (camel_minus_baseline_mat == 0) &
                           (no_intersect_mat == False) &
                           match_comb_mask)
    exact_match_indexes_sum = int(np.sum(exact_match_indexes))
    exact_match = int(np.sum(recall_mat_camel[exact_match_indexes]))
    camel_superset_indexes = ((camel_minus_baseline_mat != 0) &
                              (baseline_minus_camel_mat == 0) &
                              match_comb_mask)
    camel_superset_indexes_sum = int(np.sum(camel_superset_indexes))
    if combination != ('-', '-') and combination[0] >= combination[1]:
        assert camel_superset_indexes_sum == 0
    camel_superset = int(np.sum(recall_mat_camel[camel_superset_indexes]))
    baseline_superset_indexes = ((camel_minus_baseline_mat == 0) &
                                 (baseline_minus_camel_mat != 0) &
                                 match_comb_mask)
    baseline_superset_indexes_sum = int(np.sum(baseline_superset_indexes))
    if combination != ('-', '-') and combination[1] >= combination[0]:
        assert baseline_superset_indexes_sum == 0
    baseline_superset = int(np.sum(recall_mat_camel[baseline_superset_indexes]))
    intersect_indexes = ((baseline_minus_camel_mat != 0) &
                         (camel_minus_baseline_mat != 0) &
                         match_comb_mask)
    intersect_indexes_sum = int(np.sum(intersect_indexes))
    assert np.sum(camel_minus_baseline_mat[intersect_indexes]) == \
           np.sum(baseline_minus_camel_mat[intersect_indexes])
    intersect = int(np.sum(camel_minus_baseline_mat[intersect_indexes]))
    if combination != ('-', '-'):
        if combination[0] == combination[1]:
            assert exact_match == \
            np.sum(recall_mat_baseline[exact_match_indexes]) == \
            exact_match_indexes_sum * combination[0] == \
            exact_match_indexes_sum * combination[1]
            assert camel_superset == baseline_superset == 0
        else:
            assert exact_match == 0
    coverage_total_x_y = int(
        no_intersect + exact_match + camel_superset +
        baseline_superset + intersect)

    match_comb_indexes = np.where(match_comb_mask)
    example_coord = (match_comb_indexes[0][0], match_comb_indexes[1][0])
    lemma = index2lemmas_pos[example_coord[0]][0]
    feats = index2analysis[example_coord[1]]
    example_forms_camel = generator_camel.generate(
        lemma, eval_utils.construct_feats(feats.split('+'), 'verb'))
    example_forms_camel = ','.join(
        set(eval_utils._preprocess_lex_features(form, True)['diac'] for form in example_forms_camel))
    example_forms_baseline = generator_baseline.generate(
        lemma, eval_utils.construct_feats(feats.split('+'), 'verb'))
    example_forms_baseline = ','.join(
        set(eval_utils._preprocess_lex_features(form, True)['diac'] for form in example_forms_baseline))

    coverage_x_y_dist = [no_intersect/coverage_total_x_y,
                         exact_match/coverage_total_x_y,
                         baseline_superset/coverage_total_x_y,
                         camel_superset/coverage_total_x_y,
                         intersect/coverage_total_x_y]
    slot_total_x_y = (no_intersect + exact_match_indexes_sum +
                      baseline_superset_indexes_sum +
                      camel_superset_indexes_sum +
                      intersect_indexes_sum)
    assert slot_total_x_y == match_comb
    slot_x_y_dist = [no_intersect/slot_total_x_y,
                     exact_match_indexes_sum/slot_total_x_y,
                     baseline_superset_indexes_sum/slot_total_x_y,
                     camel_superset_indexes_sum/slot_total_x_y,
                     intersect_indexes_sum/slot_total_x_y]
    coverage_highest_index = np.array(coverage_x_y_dist).argmax()
    slot_highest_index = np.array(slot_x_y_dist).argmax()
    coverage_x_y_dist_str = [
        (f'{x:.1%}' if i != coverage_highest_index else bold(color(f'{x:.1%}', 'green'))) +
        ((f'\n{y:.1%}' if i != slot_highest_index else '\n' + bold(color(f'{y:.1%}', 'cyan'))) if x != y else '')
        for i, (x, y) in enumerate(zip(coverage_x_y_dist, slot_x_y_dist))]

    total_diac_baseline = np.sum(recall_mat_baseline[match_comb_mask])
    if combination != ('-', '-'):
        assert slot_total_x_y * combination[0] == total_diac_baseline
    recall_diac = (coverage_total_x_y - no_intersect) / total_diac_baseline
    recall_slot = (slot_total_x_y - no_intersect) / match_comb

    example_str = (bold(color('lex:', 'warning')) + ar2bw(lemma) + bold(color('x', 'fail')) + '\n' +
                   bold(color('feats:', 'warning')) + feats + '\n' +
                   bold(color('camel:', 'warning')) + ar2bw(example_forms_camel) + bold(color('x', 'fail')) + '\n' +
                   bold(color('baseline:', 'warning')) + ar2bw(example_forms_baseline)) + bold(color('x', 'fail'))

    rows.append([combination[0], combination[1],
                f'{match_comb:,}', f'{match_comb/match:.1%}',
                f'{num_lemmas_match:,}', f'{num_lemmas_match/num_valid_lemmas:.1%}',
                f'{num_feats_match:,}', f'{num_feats_match/num_valid_feats:.1%}',
                example_str if combination != ('-', '-') else bold(color('Recall (micro)', 'fail')),
                *coverage_x_y_dist_str,
                bold(color(f'{recall_diac:.1%}', 'blue')) +
                (('\n' + bold(color(f'{recall_slot:.1%}', 'blue')) if recall_diac != recall_slot else ''))])

rows = sorted(rows, key=lambda row: int(row[2].replace(',', '')) if row[0] != '-' else -1,
              reverse=True)
# rows.append(['-', '-',
#              bold(color(f'{match_comb_:,}', 'cyan')), f'{match_comb_/match:.0%}',
#              '-', '-', '-', '-', 'Recall (micro)', '-', '-', '-', '-', '-', '-'])
print(tabulate(rows, tablefmt='fancy_grid', headers=header,
               maxcolwidths=[5, 5, None, None, None, None, 5, 5, 40, 7, 7, 7, 7, 7, None]))

# max_index = recall_mat_baseline.shape[0]
# comb2freq = Counter()
# for feats, info in tqdm(eval_with_clitics['both'].items()):
#     for index in info['intersect_baseline'][1]:
#         if index >= max_index:
#             continue
#         comb = (recall_mat_baseline[index][analysis2index[feats]],
#                 recall_mat_camel[index][analysis2index[feats]])
#         comb2freq.update([comb])
print()