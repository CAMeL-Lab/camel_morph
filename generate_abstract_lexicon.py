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

from utils import add_check_mark_online, consonants_bw
from download_sheets import download_sheets

root_class_list = []

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

def test_regex(match, sub, text, gold, normalize_map):
    try:
        generated, n = re.subn(match, sub, text)
    except:
        return 'Regex Error'
    if n != 1 or normalize_map(generated) != normalize_map(gold):
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
                           global_exclusions='', local_exclusions='', local_additions='',
                           form_preprocessed=None, form_pattern_preprocessed=None,
                           cond_s=None, cond_t=None,
                           normalize_map=None):
    messages = []
    columns = {}
    columns['DEFINE'] = 'BACKOFF'
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
    # root_class = re.sub(r'2\.2', 'c.2', row['ROOT_CLASS'])
    # root_class = re.sub(r'(?<!c\.)\d', 'c', root_class).split('.')
    root_class = row['ROOT_CLASS'].split('.')
    # Root class has four types of phenomena (cna occur simultaneously):
    # (1) Letter made explicit, e.g., c.y.c
    # (2) Digit letter for when there is a gem, e.g., c.1.1
    # (3) Local exclusion: c.c.c-n
    # (4) Local addition: c.c+w.c
    # Global exclusion occurs automatically and relies on the letters made explicit in (1)
    
    assert len(root_class) == len(root)

    patterns_to_test = {
        'lemma': (lemma_stripped, pattern_lemma_stripped),
        'form': (row['FORM'], row['PATTERN'])
    }
    messages.append(check_correct_patterns(patterns_to_test, root, root_class))
    
    # MATCH #########################################################
    form_pattern_ = form_pattern_preprocessed if form_pattern_preprocessed else row['PATTERN']
    root_class_ = list(map(normalize_map, root_class))
    match_form, pos2match = generate_match_field(
        form_pattern_, root_class_, global_exclusions, local_exclusions, local_additions)
    columns['MATCH'] = f"^{match_form}$"

    # FORM #########################################################
    form_sub = generate_sub_regex(row['PATTERN'], root_class, pos2match)
    messages.append(test_regex(match_form, form_sub, form_, row['FORM'], normalize_map))

    columns['FORM'] = form_sub
    columns['FORM_EX'] = row['FORM']
    bw = row['BW'].split('/')
    columns['BW'] = f"{form_sub}/{bw[1] if len(bw) == 2 else bw[0]}"

    # LEMMA #########################################################
    columns['PATTERN_LEMMA'] = row['PATTERN_LEMMA']
    columns['PATTERN_ABSTRACT'] = abstract_pattern

    lemma_sub = generate_sub_regex(pattern_lemma_stripped, root_class, pos2match)
    messages.append(test_regex(match_form, lemma_sub, form_, lemma_stripped, normalize_map))
    
    extra_info = re.search(r'-.', row['LEMMA'])
    columns['LEMMA'] = f"lex:{lemma_sub}{extra_info.group() if extra_info else ''}"
    columns['LEMMA_EX'] = row['LEMMA']

    # ROOT ##########################################################
    root_concrete = row['ROOT'].split('.')
    pattern_root = ''.join(map(str, range(1, len(root_concrete) + 1)))
    root_sub = generate_sub_regex(pattern_root, root_class, pos2match, join_symbol='.', root=root)
    messages.append(test_regex(match_form, root_sub, form_, row['ROOT'], normalize_map))

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
    t_explicit = bool(lexicon['COND-S-BACKOFF'].str.contains('#t').any())
    n_explicit = bool(lexicon['COND-S-BACKOFF'].str.contains('#n').any())
    unique_abstract_types = {}
    for row_index, row in tqdm(lexicon.iterrows(), total=len(lexicon.index)):
        # root_class = re.sub(r'2\.2', 'c.2', row['ROOT_CLASS'])
        # root_class = re.sub(r'(?<!c\.)\d', 'c', root_class).split('.')
        # root_class_list.append(root_class)
        key = (row['PATTERN_ABSTRACT'], row['PATTERN_LEMMA'], row['PATTERN'],
               row['ROOT_CLASS'], row['FEAT'], row['COND-T-BACKOFF'], row['COND-S-BACKOFF'])
        unique_abstract_types.setdefault(key, []).append((row_index, row))

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
                if re.match(r'c([-\+].)+', r):
                    for exclusion in [r[x : x + 2] for x in range(1, len(r), 2)]:
                        local_exclusions[exclusion[0]][i].add(exclusion[1])
                else:
                    global_exclusions.add(r)
        row['LOCAL_EXCLUSIONS'] = local_exclusions
        rows[0] = (row_index, row)
    
    all_radicals = set(consonants_bw + 'A')
    global_exclusions = set([normalize_map(c) for c in global_exclusions])
    for c in global_exclusions:
        assert len(c) == 1 and c in all_radicals
    
    messages = [''] * len(lexicon.index)
    for rows in unique_abstract_types.values():
        row_index, row = rows[0]
        form_norm_dediac = normalize_map(row['FORM'])
        form_norm_dediac = re.sub(r"[uioa~]", '', form_norm_dediac)
        form_pattern_norm_dediac = normalize_map(row['PATTERN'])
        form_pattern_norm_dediac = re.sub(r"[uioa~]", '', form_pattern_norm_dediac)
        
        columns = generate_abstract_stem(row=row,
                                         global_exclusions=global_exclusions,
                                         local_exclusions=row['LOCAL_EXCLUSIONS']['-'],
                                         local_additions=row['LOCAL_EXCLUSIONS']['+'],
                                         form_preprocessed=form_norm_dediac,
                                         form_pattern_preprocessed=form_pattern_norm_dediac,
                                         normalize_map=normalize_map)
        
        if type(columns) is dict:
            abstract_entries.append(columns)
        else:
            for row_index, _ in rows:
                errors_indexes.append(row_index)
                messages[row_index] = columns
    
    with open('sandbox/root_classes.tsv', 'w') as f:
        for root_class in root_class_list:
            print('.'.join(root_class), file=f)

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
    parser.add_argument("-config_name", default='',
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
    import db_maker_utils

    with open(args.config_file) as f:
        config = json.load(f)
    config_local = config['local'][args.config_name]
    config_global = config['global']

    data_dir = args.data_dir if args.data_dir else config_global['data-dir']
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
