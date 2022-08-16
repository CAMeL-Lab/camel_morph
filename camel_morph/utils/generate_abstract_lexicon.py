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


import json
import os
import re
import argparse
from collections import Counter
from tqdm import tqdm
import sys

import gspread
import pandas as pd

from camel_morph.utils.utils import add_check_mark_online, consonants_bw
try:
    from ..debugging.download_sheets import download_sheets
except:
    from camel_morph.debugging.download_sheets import download_sheets

header = ['PATTERN_ABSTRACT', 'PATTERN_DEF', 'ROOT', 'ROOT_SUB', 'DEFINE', 'CLASS', 'PATTERN',
          'LEMMA', 'LEMMA_SUB', 'FORM', 'FORM_SUB', 'BW', 'BW_SUB', 'GLOSS', 'FEAT', 'COND-T', 'COND-F', 'COND-S',
          'MATCH', 'COMMENTS', 'STATUS']

def generate_sub_regex(pattern, root_class, pos2match, join_symbol='', root=None):
    pos2sub = []
    match_groups = 1
    for i, r in enumerate(root_class):
        if r[0] == 'c':
            regex_replace_ = f"\\{match_groups}"
            match_groups += 1
        elif r.isdigit():
            regex_replace_ = pos2match[i]
        else:
            regex_replace_ = r if not root else root[i]
        
        pos2sub.append(regex_replace_)

    sub_regex = join_symbol.join(pos2sub[int(p) - 1] if p.isdigit() else p for p in pattern)
    return sub_regex


def generate_match_field(pattern, root_class, global_exclusions, local_exclusions, local_additions):
    pos2match = []
    match_groups = 0
    for i, r in enumerate(root_class):
        regex_match = (global_exclusions | local_exclusions[i]) - local_additions[i]
        regex_match = f"([^{''.join(regex_match)}])"
        
        if r[0] == 'c':
            regex_match_ = regex_match
            match_groups += 1
        elif r.isdigit():
            regex_match_ = f"\\{match_groups}"
        else:
            regex_match_ = r

        pos2match.append(regex_match_)
    
    match_regex = []
    digits_used = set()
    for p in pattern:
        p_is_digit = p.isdigit()
        if p_is_digit:
            r = root_class[int(p) - 1]
            if p not in digits_used or not r.isdigit() and r[0] != 'c':
                match_regex.append(pos2match[int(p) - 1])
                digits_used.add(p)
            elif p in digits_used and r[0] == 'c':
                match_regex.append(f"\\{[x[0] for x in root_class[:int(p)]].count('c')}")
        else:
            match_regex.append(p)
    match_regex = ''.join(match_regex)
    return match_regex, pos2match

def test_regex(match, sub, text, gold):
    try:
        generated, n = re.subn(match, sub, text)
    except:
        return 'Regex Error'
    if n != 1 or generated != gold:
        return 'Regex Error'
    else:
        return ''

def check_correct_patterns(patterns_to_test, root, root_class):
    for x_name, (x, x_pattern) in patterns_to_test.items():
        reconstructed = []
        for c in x_pattern:
            if c.isdigit():
                r = root_class[int(c) - 1]
                if r != 'c' and not r.isdigit():
                    if len(r) > 1:
                        reconstructed.append(root[int(c) - 1])
                    else:
                        reconstructed.append(r)
                else:
                    reconstructed.append(root[int(c) - 1])
            else:
                reconstructed.append(c)
        reconstructed = ''.join(reconstructed)
        if reconstructed != x:
            return f'{x_name} Pattern Error'
    return ''


def generate_abstract_stem(row,
                           global_exclusions, local_exclusions, local_additions,
                           form_preprocessed='', form_pattern_preprocessed='',
                           root_class_preprocessed='',
                           cond_s='', cond_t=''):
    """Root class has four types of phenomena (which can occur simultaneously):
        (1) Letter made explicit, e.g., c.y.c
        (2) Digit letter for certain types of gem, e.g., c.c.2
        (3) Local exclusion: c.c.c-n
        (4) Local addition: c.c+w.c
    Global exclusion occurs automatically and relies on the letters made explicit in (1)

    Args:
        row (pd.Series): pandas row
        global_exclusions (dict): computed automatically from explicit radicals in root class. Defaults to ''.
        local_exclusions (dict): computed from explicit radicals following `-` in root class. Defaults to ''.
        local_additions (dict): computed from explicit radicals following `+` in root class.
        form_preprocessed (str, optional): form used to provide as input for testing correctness of match/sub regexes. For Arabic will usually be the normalized, dediacritized pattern. Defaults to ''.
        form_pattern_preprocessed (str, optional): form pattern used to generate the match_regex. For Arabic, will usually be the normalized, dediacritized pattern. Defaults to ''.
        root_class_preprocessed (list, optional): preprocessed root class used to generate the match_regex. For Arabic, will usually be the normalized, dediacritized version. Defaults to ''.
        cond_s (str, optional): preprocessed COND-S to use for abstract stem. Defaults to ''.
        cond_t (str, optional): preprocessed COND-T to use for abstract stem.. Defaults to ''.

    Returns:
        dict: dictionary containaining the new columns of the abstract entry
    """
    messages = []
    columns = {}
    columns['DEFINE'] = 'SMARTBACKOFF'
    columns['CLASS'] = row['CLASS']
    columns['COND-T'] = cond_t if cond_t else row['COND-T-BACKOFF']
    columns['COND-S'] = cond_s if cond_s else row['COND-S-BACKOFF']
    columns['COND-F'] = ''
    columns['PATTERN'] = row['PATTERN']
    columns['FEAT'] = row['FEAT']
    columns['GLOSS'] = 'na'
    
    abstract_pattern = row['PATTERN_ABSTRACT']
    lemma_stripped = re.sub(r'^lex:|(-.)?(_\d)?$', '', row['LEMMA'])
    pattern_lemma_stripped = re.sub(r'^lex:|(-.)?(_\d)?$', '', row['PATTERN_LEMMA'])
    form_ = form_preprocessed if form_preprocessed else row['FORM']
    
    root = row['ROOT'].split('.')
    root_class = row['ROOT_CLASS'].split('.')
    
    assert len(root_class) == len(root)

    patterns_to_test = {
        'lemma': (lemma_stripped, pattern_lemma_stripped),
        'form': (row['FORM'], row['PATTERN'])
    }
    messages.append(check_correct_patterns(patterns_to_test, root, root_class))
    
    # MATCH #########################################################
    form_pattern_ = form_pattern_preprocessed if form_pattern_preprocessed else row['PATTERN']
    match_form, pos2match = generate_match_field(
        form_pattern_, root_class_preprocessed, global_exclusions, local_exclusions, local_additions)
    columns['MATCH'] = f"^{match_form}$"

    # FORM #########################################################
    form_sub = generate_sub_regex(row['PATTERN'], root_class, pos2match)
    messages.append(test_regex(match_form, form_sub, form_, row['FORM']))

    columns['FORM'] = form_sub
    columns['FORM_EX'] = row['FORM']
    bw = row['BW'].split('/')
    columns['BW'] = f"{form_sub}/{bw[1] if len(bw) == 2 else bw[0]}"

    # LEMMA #########################################################
    columns['PATTERN_LEMMA'] = row['PATTERN_LEMMA']
    columns['PATTERN_ABSTRACT'] = abstract_pattern

    lemma_sub = generate_sub_regex(pattern_lemma_stripped, root_class, pos2match)
    messages.append(test_regex(match_form, lemma_sub, form_, lemma_stripped))
    
    extra_info = re.search(r'-.', row['LEMMA'])
    columns['LEMMA'] = f"lex:{lemma_sub}{extra_info.group() if extra_info else ''}"
    columns['LEMMA_EX'] = row['LEMMA']

    # ROOT ##########################################################
    root_concrete = row['ROOT'].split('.')
    pattern_root = ''.join(map(str, range(1, len(root_concrete) + 1)))
    root_sub = generate_sub_regex(pattern_root, root_class, pos2match, join_symbol='.', root=root)
    messages.append(test_regex(match_form, root_sub, form_, row['ROOT']))

    columns['ROOT'] = root_sub
    columns['ROOT_EX'] = row['ROOT']

    if any(messages):
        return ' '.join(messages)
    else:
        return columns

def process_cond_t(row):
    cond_t = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                              for cond in row['COND-T'].split()]))
    return cond_t

def process_cond_s(row):
    cond_s = re.sub(r'hamzated|hollow|defective|ditrans', '', row['COND-S'])
    cond_s = re.sub(r'intrans', 'trans', cond_s) if 'vox:p' not in row['FEAT'] else cond_s
    cond_s = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                              for cond in cond_s.split()]))
    return cond_s

def get_exclusions(unique_abstract_types, preprocess_func):
    global_exclusions = set()
    for rows in unique_abstract_types.values():
        row_index, row = rows[0]
        root_class = row['ROOT_CLASS'].split('.')
        local_exclusions = {
            '-': {i: set() for i in range(len(root_class))},
            '+': {i: set() for i in range(len(root_class))}
        }
        for i, r in enumerate(root_class):
            if r != 'c' and not r.isdigit():
                match = re.match(r'c([-\+][^\.\+]+)([-\+][^\.]+)?', r)
                if match:
                    match = match.groups()
                    if match[0][0] == '-':
                        exclusions_, additions_ = match[0][1:], match[1][1:] if match[1] else ''
                    elif match[0][0] == '+':
                        additions_, exclusions_ = match[0][1:], match[1][1:] if match[1] else ''
                    local_exclusions['-'][i].update(exclusions_)
                    local_exclusions['+'][i].update(additions_)
                else:
                    global_exclusions.add(r)
        row['LOCAL_EXCLUSIONS'] = local_exclusions
        rows[0] = (row_index, row)
    
    all_radicals = set(consonants_bw + 'A')
    global_exclusions = set([preprocess_func(c) for c in global_exclusions])
    for c in global_exclusions:
        assert len(c) == 1 and c in all_radicals
    
    return global_exclusions, unique_abstract_types


def generate_abstract_lexicon(lexicon, spreadsheet=None, sheet=None):
    from camel_tools.utils.charmap import CharMapper

    normalize_map = CharMapper({
        '<': 'A',
        '>': 'A',
        '|': 'A',
        '{': 'A',
        'Y': 'y'
    })
    
    lexicon['COND-S-BACKOFF'] = lexicon.apply(process_cond_s, axis=1)
    lexicon['COND-T-BACKOFF'] = lexicon.apply(process_cond_t, axis=1)
    
    abstract_entries = []
    errors_indexes, messages = [], []
    unique_abstract_types = {}
    for row_index, row in tqdm(lexicon.iterrows(), total=len(lexicon.index)):
        key = (row['PATTERN_ABSTRACT'], row['PATTERN_LEMMA'], row['PATTERN'],
               row['ROOT_CLASS'], row['FEAT'], row['COND-T-BACKOFF'], row['COND-S-BACKOFF'])
        unique_abstract_types.setdefault(key, []).append((row_index, row))

    def preprocess_func(form):
        form = normalize_map(form)
        form = re.sub(r"[uioa~]", '', form)
        return form

    global_exclusions, unique_abstract_types = get_exclusions(unique_abstract_types, preprocess_func)
    
    messages = [''] * len(lexicon.index)
    for rows in unique_abstract_types.values():
        row_index, row = rows[0]
        form_norm_dediac = preprocess_func(row['FORM'])
        form_pattern_norm_dediac = preprocess_func(row['PATTERN'])
        root_class_preprocessed = preprocess_func(row['ROOT_CLASS']).split('.')
        columns = generate_abstract_stem(row=row,
                                         global_exclusions=global_exclusions,
                                         local_exclusions=row['LOCAL_EXCLUSIONS']['-'],
                                         local_additions=row['LOCAL_EXCLUSIONS']['+'],
                                         form_preprocessed=form_norm_dediac,
                                         form_pattern_preprocessed=form_pattern_norm_dediac,
                                         root_class_preprocessed=root_class_preprocessed)
        
        if type(columns) is dict:
            abstract_entries.append(columns)
        else:
            for row_index, _ in rows:
                errors_indexes.append(row_index)
                messages[row_index] = columns

    if sheet and spreadsheet and errors_indexes:
        add_check_mark_online(lexicon, spreadsheet, sheet,
                              messages=messages, mode='backoff',
                              status_col_name='STATUS_CHRIS')

    entry2freq = Counter(
        [tuple([entry.get(h) for h in header]) for entry in abstract_entries])
    comments_index = header.index('COMMENTS')
    abstract_lexicon = [[field if i != comments_index else freq for i, field in enumerate(entry)]
                        for entry, freq in entry2freq.items()]
    abstract_lexicon = pd.DataFrame(abstract_lexicon)
    abstract_lexicon.columns = header

    lexicon.drop('COND-T-BACKOFF', axis=1, inplace=True)
    lexicon.drop('COND-S-BACKOFF', axis=1, inplace=True)

    return abstract_lexicon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", default='',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-data_dir", default='',
                        type=str, help="Path of the directory where the sheets are.")
    parser.add_argument("-get_patterns_from_sheet", default=False,
                        action='store_true', help="Get patterns from sheet instead of generating them on the fly.")
    parser.add_argument("-output_name", default='',
                        type=str, help="Name of the file to output the abstract lexicon to.")
    parser.add_argument("-camel_tools", default='',
                        type=str, help="Path of the directory containing the camel_tools modules.")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args()
    if args.camel_tools:
        sys.path.insert(0, args.camel_tools)
        from camel_tools.utils.charmap import CharMapper
        from camel_tools.morphology.utils import strip_lex
    import camel_morph.db_maker_utils as db_maker_utils

    with open(args.config_file) as f:
        config = json.load(f)
    config_local = config['local'][args.config_name]
    config_global = config['global']

    data_dir = args.data_dir if args.data_dir else config_global['data_dir']
    output_name = args.output_name if args.output_name else \
        (next(iter(config_local['backoff'].values())) if config_local.get('backoff') else 'ABSTRACT-LEX.csv')

    sa = gspread.service_account(args.service_account)
    sh = sa.open(config_local['lexicon']['spreadsheet'])
    download_sheets(save_dir=data_dir,
                    config=config,
                    config_name=args.config_name,
                    service_account=sa)

    SHEETS, _ = db_maker_utils.read_morph_specs(config, args.config_name)
    lexicon = SHEETS['lexicon']

    abstract_lexicon = generate_abstract_lexicon(lexicon,
                                                 sh, config_local['lexicon']['sheets'][0])

    abstract_lexicon.to_csv(os.path.join(data_dir, output_name))
