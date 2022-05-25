import json
import os
import re
import argparse
from collections import Counter
from tqdm import tqdm
import sys

import gspread
import pandas as pd
from numpy import nan

from utils import add_check_mark_online
from download_sheets import download_sheets
import db_maker_utils

root_class_list, patterns_list = [], []

header = ['PATTERN_ABSTRACT', 'PATTERN_DEF', 'ROOT', 'ROOT_SUB', 'DEFINE', 'CLASS', 'PATTERN',
          'LEMMA', 'LEMMA_SUB', 'FORM', 'FORM_SUB', 'BW', 'BW_SUB', 'GLOSS', 'FEAT', 'COND-T', 'COND-F', 'COND-S',
          'MATCH', 'COMMENTS', 'STATUS']

def generate_sub_regex(pattern, root, root_class, join_symbol=''):
    sub = []
    counter, stop_counting = 0, False
    for c in pattern:
        if c.isdigit():
            if int(c) - 2 and root_class[int(c) - 1] == root_class[int(c) - 2]:
                if not root_class[int(c) - 1].isdigit():
                    sub.append(concrete_root_radical(c, root, root_class))
                    continue
            elif re.match(r'^\d$', root_class[int(c) - 1]):
                counter += 1
            elif not stop_counting and re.match(r'\d\d', root_class[int(c) - 1]):
                counter += 1
                stop_counting = True
            elif not stop_counting and re.match(r'\d\w\d', root_class[int(c) - 1]):
                counter += 1
                stop_counting = True
            else:
                sub.append(concrete_root_radical(c, root, root_class))
                continue
            sub.append(f'\\{counter}')
        else:
            sub.append(c)
    sub = join_symbol.join(sub)
    return sub

def concrete_root_radical(index, root, root_class):
    interdigitation = root_class[int(index) - 1][0]
    interdigitation = interdigitation if not interdigitation.isdigit() else root[int(index) - 1]
    return interdigitation


def generate_match_field(pattern, root_class, regex_replace, final_radical_regex_replace):
    match_form_ = pattern
    for i, r in enumerate(root_class):
        generic_pos = i + 1
        regex_match_ = str(generic_pos)
        r_ = int(r) if r.isdigit() else r

        if len(r) > 1:
            if re.match(r'\d\d', r):
                regex_match_ = r
                regex_replace_ = regex_replace if generic_pos != len(root_class) else final_radical_regex_replace
                parenth_grps = re.findall(r'\([^\(\)]+\)', match_form_)
                regex_replace_ += f"\\\{len(parenth_grps) + 1}"
            elif re.match(r'\d\w\d', r):
                regex_match_ = r
                regex_replace_ = regex_replace if generic_pos != len(root_class) else final_radical_regex_replace
                regex_replace_ += r[1]
                parenth_grps = re.findall(r'\([^\(\)]+\)', match_form_)
                regex_replace_ += f"\\\{len(parenth_grps) + 1}"
            else:
                regex_match_ = str(generic_pos) * 2
                regex_replace_ = r
        elif i + 1 < len(root_class) and r == root_class[i + 1]:
            regex_replace_ = (regex_replace if generic_pos != len(root_class) - 1 else final_radical_regex_replace) \
                if type(r_) is int else r_
        elif i >= 1 and r == root_class[i - 1]:
            parenth_grps = re.findall(r'\([^\(\)]+\)', match_form_)
            regex_replace_ = f"\\\{len(parenth_grps)}" if type(r_) is int else r_
        else:
            regex_replace_ = (regex_replace if generic_pos != len(root_class) else final_radical_regex_replace) \
                if type(r_) is int else r_
        
        match_form_ = re.sub(regex_match_, regex_replace_.replace('|', 'A'), match_form_, count=1)
    
    return match_form_

def test_regex(match, sub, text, gold):
    generated, n = re.subn(match, sub, text)
    if n != 1 or normalize_map(generated) != normalize_map(gold):
        return 'Regex Error'

def check_correct_patterns(patterns_to_test, root, root_class):
    for x_name, (x, x_pattern) in patterns_to_test.items():
        reconstructed = []
        for c in x_pattern:
            if c.isdigit():
                reconstructed.append(concrete_root_radical(c, root, root_class))
            else:
                reconstructed.append(c)
        reconstructed = ''.join(reconstructed)
        if reconstructed != x:
            return f'{x_name} Pattern Error'


def generate_abstract_stem(row, regex_replace='(.)', final_radical_regex_replace='(.)',
                           form_preprocessed=None, form_pattern_preprocessed=None,
                           cond_s=None, cond_t=None):
    messages = []
    columns = {}
    columns['DEFINE'] = 'BACKOFF'
    columns['CLASS'] = row['CLASS']
    columns['COND-T'] = cond_t if cond_t else row['COND-T']
    columns['COND-S'] = cond_s if cond_s else row['COND-S']
    columns['PATTERN'] = row['PATTERN']
    columns['FEAT'] = row['FEAT']
    columns['GLOSS'] = 'na'
    
    abstract_pattern = row['PATTERN_ABSTRACT']
    lemma_stripped = strip_lex(row['LEMMA']).split('lex:')[1]
    pattern_lemma_stripped = strip_lex(row['PATTERN_LEMMA'])
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
    match_form = generate_match_field(form_pattern_, root_class, regex_replace, final_radical_regex_replace)
    columns['MATCH'] = f"^{match_form}$"

    # FORM #########################################################
    form_sub = generate_sub_regex(row['PATTERN'], root, root_class)
    messages.append(test_regex(match_form, form_sub, form_, row['FORM']))

    columns['FORM_SUB'] = form_sub
    columns['FORM_EX'] = row['FORM']
    columns['BW_SUB'] = f"{form_sub}/{row['BW'].split('/')[1]}"

    # LEMMA #########################################################
    columns['PATTERN_LEMMA'] = row['PATTERN_LEMMA']
    columns['PATTERN_ABSTRACT'] = abstract_pattern

    lemma_sub = generate_sub_regex(pattern_lemma_stripped, root, root_class)
    messages.append(test_regex(match_form, lemma_sub, form_, lemma_stripped))
    
    extra_info = re.search(r'-.', row['LEMMA'])
    columns['LEMMA_SUB'] = f"lex:{lemma_sub}{extra_info.group() if extra_info else ''}"
    columns['LEMMA_EX'] = row['LEMMA']

    # ROOT ##########################################################
    root_row = row['ROOT'].split('.')
    root_sub = generate_sub_regex(
        ''.join(map(str, range(1, len(root_row) + 1))), root_row, root_class, join_symbol='.')
    messages.append(test_regex(match_form, root_sub, form_, row['ROOT']))

    columns['ROOT_SUB'] = root_sub
    columns['ROOT_EX'] = row['ROOT']

    if any(messages):
        return ' '.join(messages)
    else:
        return columns


def generate_abstract_lexicon(lexicon, spreadsheet, sheet, override_explicit):
    abstract_entries = []
    errors_indexes, messages = [], []
    t_explicit = bool(lexicon['COND-S'].str.contains('#t').any())
    n_explicit = bool(lexicon['COND-S'].str.contains('#n').any())
    unique_abstract_types = {}
    for row_index, row in tqdm(lexicon.iterrows(), total=len(lexicon.index)):
        cond_t = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-T'].split()]))
        cond_s = re.sub(r'ditrans', '', row['COND-S'])
        cond_s = re.sub(r'intrans', 'trans', cond_s) if 'vox:p' not in row['FEAT'] else cond_s
        cond_s = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in cond_s.split()]))
        lexicon.loc[row_index, 'COND-S'] = cond_s
        lexicon.loc[row_index, 'COND-T'] = cond_t
        key = (row['PATTERN_ABSTRACT'], row['PATTERN_LEMMA'], row['PATTERN'], row['ROOT_CLASS'], row['FEAT'], cond_t, cond_s)
        unique_abstract_types.setdefault(key, []).append((row_index, lexicon.loc[row_index]))
    
    for rows in unique_abstract_types.values():
        row_index, row = rows[0]
        lemma_stripped = strip_lex(row['LEMMA']).split('lex:')[1]
        override_n_t = override_explicit and 'gem' in row['COND-S'] and bool(re.search(r'[nt]~?$', lemma_stripped))
        if not t_explicit and n_explicit or override_n_t:
            final_radical_regex_replace = '([^nwyA}&])'
        elif t_explicit and not n_explicit:
            final_radical_regex_replace = '([^twyA}&])'
        elif not t_explicit and not n_explicit:
            final_radical_regex_replace = '([^wyA}&])'
        else:
            final_radical_regex_replace = '([^ntwyA}&])'
        form_norm_dediac = normalize_map(row['FORM'])
        form_norm_dediac = re.sub(r"[uioa~]", '', form_norm_dediac)
        form_pattern_norm_dediac = normalize_map(row['PATTERN'])
        form_pattern_norm_dediac = re.sub(r"[uioa~]", '', form_pattern_norm_dediac)
        
        columns = generate_abstract_stem(row=row,
                                         regex_replace='([^wyA}&])',
                                         final_radical_regex_replace=final_radical_regex_replace,
                                         form_preprocessed=form_norm_dediac,
                                         form_pattern_preprocessed=form_pattern_norm_dediac)
        
        if type(columns) is dict:
            abstract_entries.append(columns)
            messages += [''] * len(rows)
        else:
            for row_index, _ in rows:
                errors_indexes.append(row_index)
                messages.append(columns)
    
    if errors_indexes:
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

    return abstract_lexicon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", default='',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-lexicon_path", default='',
                        type=str, help="Path of the lexicon to load.")
    parser.add_argument("-data_dir", default="data",
                        type=str, help="Path of the directory where the sheets are.")
    parser.add_argument("-get_patterns_from_sheet", default=False,
                        action='store_true', help="Get patterns from sheet instead of generating them on the fly.")
    parser.add_argument("-output_dir", default='data',
                        type=str, help="Path of the directory to output the lemmas to.")
    parser.add_argument("-output_name", default='ABSTRACT-LEX.csv',
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
    normalize_map = CharMapper({
        '<': 'A',
        '>': 'A',
        '|': 'A',
        '{': 'A',
        'Y': 'y'
    })

    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)

        sa = gspread.service_account(args.service_account)
        sh = sa.open(config['local'][args.config_name]['lexicon']['spreadsheet'])
        download_sheets(save_dir=args.data_dir,
                        config=config,
                        config_name=args.config_name,
                        service_account=sa)

        SHEETS, _ = db_maker_utils.read_morph_specs(config, args.config_name)
        SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
            lambda row: re.sub(r'hamzated|hollow|defective', '', row['COND-S']), axis=1)
        SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
            lambda row: re.sub(r' +', ' ', row['COND-S']), axis=1)
        lexicon = SHEETS['lexicon']
        lexicon = lexicon[lexicon['FORM'] != 'DROP']
        get_patterns_from_sheet = config['local'][args.config_name]['lexicon'].get('get_patterns_from_sheet')
        override_explicit = config['local'][args.config_name]['lexicon'].get('override_explicit')
    elif args.lexicon_path:
        lexicon = pd.read_csv(args.lexicon_path)
        lexicon = lexicon.replace(nan, '', regex=True)
    else:
        raise NotImplementedError

    abstract_lexicon = generate_abstract_lexicon(lexicon,
                                                 sh, config['local'][args.config_name]['lexicon']['sheets'][0],
                                                 override_explicit)

    abstract_lexicon.to_csv(os.path.join(args.output_dir, args.output_name))
