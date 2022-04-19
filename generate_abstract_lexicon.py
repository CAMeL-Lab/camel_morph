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

from utils import assign_pattern, add_check_mark_online
from download_sheets import download_sheets
import db_maker_utils

header = ['PATTERN_ABS', 'PATTERN_DEF', 'ROOT', 'ROOT_SUB', 'DEFINE', 'CLASS', 'PATTERN',
          'LEMMA', 'LEMMA_SUB', 'FORM', 'FORM_SUB', 'BW', 'BW_SUB', 'GLOSS', 'FEAT', 'COND-T', 'COND-F', 'COND-S',
          'MATCH', 'COMMENTS', 'STATUS']

def generate_substitution_regex(pattern, gem_c_suff_form=False):
    sub, radicalindex2grp = [], {}
    parenth_grp = 1
    for c in pattern:
        if c.isdigit():
            sub.append(f'\\{parenth_grp}')
            radicalindex2grp[int(c)] = parenth_grp
            parenth_grp += 1
        else:
            sub.append(c)
    if gem_c_suff_form and len(radicalindex2grp) >= 2:
        sub[-1] = [s for s in sub if re.search(r'\d', s)][-2]
    sub = ''.join(sub)
    return sub, radicalindex2grp


def generate_abstract_stem(row, get_patterns_from_sheet, t_explicit, n_explicit, override_explicit):
    columns = {}
    columns['DEFINE'] = 'BACKOFF'
    columns['CLASS'] = row['CLASS']
    cond_t = ' '.join(sorted(row['COND-T'].split()))
    columns['COND-T'] = cond_t
    cond_s = re.sub('intrans', 'trans', row['COND-S']) if 'vox:p' not in row['FEAT'] else row['COND-S']
    cond_s = ' '.join(sorted([cond for cond in cond_s.split()]))
    columns['COND-S'] = cond_s
    columns['PATTERN'] = row['PATTERN']
    columns['FEAT'] = row['FEAT']
    columns['GLOSS'] = 'na'

    lemma_ex_stripped = strip_lex(row['LEMMA']).split('lex:')[1]
    if not get_patterns_from_sheet:
        result = assign_pattern(
            lemma_ex_stripped, root=row['ROOT'].split('.'))
        lemma_pattern_surf = result['pattern_surf']
        abstract_pattern = result['pattern_abstract']
    else:
        lemma_pattern_surf = row['PATTERN_LEMMA']
        abstract_pattern = row['PATTERN_ABSTRACT']
    # FORM ##########################################################
    index2radical = row['ROOT'].split('.')
    radical2index = {}
    for i, r in enumerate(index2radical, start=1):
        radical2index.setdefault(r, []).append(i)
    
    form_pattern_dediac, form_pattern = row['PATTERN'], row['PATTERN']
    form_reconstructed = []
    for c in row['PATTERN']:
        form_reconstructed.append(index2radical[int(c) - 1] if c.isdigit() else c)
    form_reconstructed = ''.join(form_reconstructed)
    if not (form_reconstructed == row['FORM']):
        return 'Error 1'
    form_dediac = normalize_map(row['FORM'])
    form_dediac = re.sub(r"[uioa~]", '', form_dediac)
    form_pattern_dediac = normalize_map(form_pattern_dediac)
    form_pattern_dediac = re.sub(r"[uioa~]", '', form_pattern_dediac)

    match_form, match_form_diac = form_pattern_dediac, form_pattern
    max_form_digit = re.findall(r'\d', match_form)
    max_form_digit = max(int(d) for d in max_form_digit) if max_form_digit else None
    if not t_explicit and not n_explicit:
        n_t = None
    elif not t_explicit and n_explicit:
        n_t = '#n' in cond_s
    elif t_explicit and not n_explicit:
        n_t = '#t' in cond_s
    else:
        n_t = '#n' in cond_s or '#t' in cond_s
    if n_t:
        if lemma_ex_stripped[-1] != '~':
            r = lemma_ex_stripped[-1]
            index = radical2index[r][-1]
        else:
            r = lemma_ex_stripped[-2]
            # c-suff
            if form_pattern[-1] == '~' and abstract_pattern != '{i1o2a3~':
                index = radical2index[r][-2] if len(radical2index[r]) > 1 else radical2index[r][-1]
            # v-suff
            else:
                index = radical2index[r][-1]
        
        match_form, n0 = re.subn(str(index), r, match_form)
        match_form_diac, n1 = re.subn(str(index), r, form_pattern)
        if form_pattern[-1] == '~':
            if not (n0 == n1 == 1):
                return 'Error 2'
        else:
            if not (n0 == n1 == form_pattern.count(str(index))):
                return 'Error 3'
    
    n0 = 0
    not_t_n = '#t' not in cond_s and '#n' not in cond_s
    generic_last_radical = max_form_digit == len(index2radical) and ('gem' in cond_s or True)
    gem_c_suff = True if 'c-suff' in cond_t and 'gem' in cond_s else False
    form_sub, radicalindex2grp = generate_substitution_regex(
        match_form_diac, gem_c_suff_form=not_t_n and generic_last_radical and gem_c_suff)
    
    override_n_t = override_explicit and 'gem' in cond_s and bool(re.search(r'[nt]~?$', lemma_ex_stripped))
    if not_t_n and generic_last_radical and len(radicalindex2grp) >= 2:
        if not t_explicit and not n_explicit or override_n_t:
            regex_replace = '([^wyA}&])'
        elif not t_explicit and n_explicit:
            regex_replace = '([^nwyA}&])'
        elif t_explicit and not n_explicit:
            regex_replace = '([^twyA}&])'
        else:
            regex_replace = '([^ntwyA}&])'
        regex_match = r'\d$' if not gem_c_suff else r'\d\d$'
        if gem_c_suff:
            penultimate_replace_grp = re.findall(r'\d', form_sub)[-2]
        regex_replace = regex_replace if not gem_c_suff else f"{regex_replace}\\\{penultimate_replace_grp}"
        match_form, n0 = re.subn(regex_match, regex_replace, match_form)
        if n0 == 1 and gem_c_suff:
            n0 = 2
        else:
            if not (n0 == 1):
                return 'Error 4'

    match_form, n1 = re.subn(r'(?<!\\)\d', '([^wyA}&])', match_form)
    n_match = n0 + n1
    columns['MATCH'] = f"^{match_form}$"
    
    match_form_diac_parenth = re.escape(match_form_diac)
    match_form_diac_parenth = re.sub(r'\d', '(.)', match_form_diac_parenth)

    digits = [int(d) for d in re.findall(r'\d', form_sub)]
    max_digit = max(digits) if digits else 0
    if not (max_digit <= n_match):
        return 'Error 5'
    
    form_gen, n = re.subn(match_form_diac_parenth, form_sub, row['FORM'])
    if not (n == 1 and form_gen == row['FORM']):
        return 'Error 6'
    form_gen, n = re.subn(match_form, form_sub, form_dediac)
    if not (n == 1 and normalize_map(form_gen) == normalize_map(row['FORM'])):
        return 'Error 7'
    columns['FORM_SUB'] = form_sub
    columns['FORM'] = match_form_diac
    columns['FORM_EX'] = row['FORM']

    columns['BW'] = f"{match_form_diac}/{row['BW'].split('/')[1]}"
    columns['BW_SUB'] = f"{form_sub}/{row['BW'].split('/')[1]}"

    # LEMMA #########################################################
    columns['PATTERN_DEF'] = lemma_pattern_surf
    columns['PATTERN_ABS'] = abstract_pattern
    lemma_reconstructed = []
    for c in lemma_pattern_surf:
        lemma_reconstructed.append(index2radical[int(c) - 1] if c.isdigit() else c)
    lemma_reconstructed = ''.join(lemma_reconstructed)
    if not (lemma_reconstructed == lemma_ex_stripped):
        return 'Error 8'

    if n_t:
        lemma_pattern_surf = re.sub(str(index), r, lemma_pattern_surf)

    match_lemma_parenth = re.escape(lemma_pattern_surf)
    match_lemma_parenth = re.sub(r'\d', '(.)', match_lemma_parenth)
    lemma_sub, radicalindex2grp = generate_substitution_regex(lemma_pattern_surf)
    digits = [int(d) for d in re.findall(r'\d', lemma_sub)]
    max_digit = max(digits) if digits else 0
    if not (max_digit <= n_match):
        return 'Error 9'
    
    lemma_gen, n = re.subn(match_lemma_parenth, lemma_sub, lemma_ex_stripped)
    if not (n == 1 and lemma_gen == lemma_ex_stripped):
        return 'Error 10'
    lemma_gen, n = re.subn(match_form, lemma_sub, form_dediac)
    if not (n == 1 and normalize_map(lemma_gen) == normalize_map(lemma_ex_stripped)):
        return 'Error 11'
    Eayn_diac = re.search(r'-.', row['LEMMA'])
    columns['LEMMA_SUB'] = f"lex:{lemma_sub}{Eayn_diac.group() if Eayn_diac else ''}"
    columns['LEMMA'] = f"lex:{lemma_pattern_surf}{Eayn_diac.group() if Eayn_diac else ''}"
    columns['LEMMA_EX'] = row['LEMMA']

    # ROOT ##########################################################
    root, root_sub = [], []
    root_split = row['ROOT'].split('.')
    for i, r in enumerate(root_split, start=1):
        if not t_explicit and not n_explicit:
            radical_nt = False
        elif not t_explicit and n_explicit:
            radical_nt = (r == 'n')
        elif t_explicit and not n_explicit:
            radical_nt = (r == 't')
        else:
            radical_nt = (r in ['n', 't'])

        if r in ['>', 'w', 'y'] or i == len(root_split) and radical_nt or \
                i == len(root_split) - 1 and root_split[i - 1] == root_split[i] and 'gem' in cond_s and radical_nt:
            root.append(r)
            root_sub.append(r)
        else:
            root.append(str(i))
            grp = radicalindex2grp.get(i)
            if grp is None:
                if r == root_split[i - 2]:
                    grp = radicalindex2grp.get(i - 1)
                else:
                    return 'Error 12'
            root_sub.append(f'\{grp}')

    root = '.'.join(root)
    root_sub = '.'.join(root_sub)
    match_root_parenth = re.escape(root)
    match_root_parenth = re.sub(r'\d', '(.)', match_root_parenth)
    digits = [int(d) for d in re.findall(r'\d', root_sub)]
    max_digit = max(digits) if digits else 0
    if not (max_digit <= n_match):
        return 'Error 13'
    try:
        root_gen, n = re.subn(match_root_parenth, root_sub, row['ROOT'])
    except:
        try:
            root_gen, n = re.subn(match_form, root_sub, form_dediac)
        except:
            return 'Error 14'
    if not (n == 1 and root_gen == row['ROOT']):
        return 'Error 15'

    columns['ROOT_SUB'] = root_sub
    columns['ROOT'] = root
    columns['ROOT_EX'] = row['ROOT']

    return columns


def generate_abstract_lexicon(lexicon, spreadsheet, sheet, get_patterns_from_sheet, override_explicit):
    abstract_entries = []
    errors_indexes = []
    t_explicit = bool(lexicon['COND-S'].str.contains('#t').any())
    n_explicit = bool(lexicon['COND-S'].str.contains('#n').any())
    for row_index, row in tqdm(lexicon.iterrows(), total=len(lexicon.index)):
        columns = generate_abstract_stem(row, get_patterns_from_sheet,
                                         t_explicit, n_explicit, override_explicit)
        if type(columns) is dict:
            abstract_entries.append(columns)
        else:
            errors_indexes.append(row_index)

    if errors_indexes:
        add_check_mark_online(lexicon, spreadsheet, sheet,
                              indexes=errors_indexes, mode='backoff',
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
        download_sheets(lex=None, specs=None, save_dir=args.data_dir,
                        config_file=args.config_file, config_name=args.config_name,
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
                                                 get_patterns_from_sheet,
                                                 override_explicit)

    abstract_lexicon.to_csv(os.path.join(args.output_dir, args.output_name))
