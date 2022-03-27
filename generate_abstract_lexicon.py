import json
import os
import re
import argparse
from collections import Counter
from tqdm import tqdm

import pandas as pd
from numpy import nan

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex

from utils import assign_pattern
import db_maker_utils

header = ['PATTERN_ABS', 'PATTERN_DEF', 'ROOT', 'ROOT_SUB', 'DEFINE', 'CLASS', 'PATTERN',
          'LEMMA', 'LEMMA_SUB', 'FORM', 'FORM_SUB', 'BW', 'BW_SUB', 'GLOSS', 'FEAT', 'COND-T', 'COND-F', 'COND-S',
          'MATCH', 'COMMENTS', 'STATUS']

normalize_map = CharMapper({
    '<': 'A',
    '>': 'A',
    '|': 'A',
    '{': 'A',
    'Y': 'y'
})

def generate_substitution_regex(pattern):
    sub, radicalindex2grp = [], {}
    parenth_grp = 1
    for c in pattern:
        if c.isdigit():
            sub.append(f'\\{parenth_grp}')
            radicalindex2grp[int(c)] = parenth_grp
            parenth_grp += 1
        else:
            sub.append(c)
    sub = ''.join(sub)
    return sub, radicalindex2grp

def generate_abstract_lexicon(lexicon):
    abstract_entries = []
    for _, row in tqdm(lexicon.iterrows(), total=len(lexicon.index)):
        columns = {}
        columns['DEFINE'] = 'BACKOFF'
        columns['CLASS'] = row['CLASS']
        cond_t = ' '.join(sorted(row['COND-T'].split()))
        columns['COND-T'] = cond_t
        cond_s = ' '.join(sorted([cond for cond in row['COND-S'].split() if 'trans' not in cond]))
        columns['COND-S'] = cond_s
        columns['PATTERN'] = row['PATTERN']
        columns['FEAT'] = row['FEAT']
        columns['GLOSS'] = 'na'

        lemma_ex_stripped = strip_lex(row['LEMMA']).split('lex:')[1]
        result = assign_pattern(
            lemma_ex_stripped, root=row['ROOT'].split('.'))
        lemma_pattern_surf = result['pattern_surf']
        abstract_pattern = result['pattern_abstract']
        # FORM ##########################################################
        index2radical = row['ROOT'].split('.')
        radical2index = {}
        for i, r in enumerate(index2radical, start=1):
            radical2index.setdefault(r, []).append(i)
        
        form_pattern_dediac, form_pattern = row['PATTERN'], row['PATTERN']
        form_dediac = normalize_map(row['FORM'])
        form_dediac = re.sub(r"[uioa~]", '', form_dediac)
        form_pattern_dediac = normalize_map(form_pattern_dediac)
        form_pattern_dediac = re.sub(r"[uioa~]", '', form_pattern_dediac)

        match_form, match_form_diac = form_pattern_dediac, form_pattern
        n_t = re.search(r'[nt]~?$', lemma_ex_stripped)
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
                assert n0 == n1 == 1
            else:
                assert n0 == n1 == form_pattern.count(str(index))
        
        n0 = 0
        if '#t' in cond_s or '#n' in cond_s:
            match_form, n0 = re.subn(r'\d$', '([^ntwy])', match_form)
        match_form, n1 = re.subn(r'\d', '([^wy])', match_form)
        n_match = n0 + n1
        columns['MATCH'] = f"^{match_form}$"
        
        match_form_diac_parenth = re.escape(match_form_diac)
        match_form_diac_parenth = re.sub(r'\d', '(.)', match_form_diac_parenth)

        form_sub, _ = generate_substitution_regex(match_form_diac)

        digits = [int(d) for d in re.findall(r'\d', form_sub)]
        max_digit = max(digits) if digits else 0
        assert max_digit <= n_match
        
        form_gen, n = re.subn(match_form_diac_parenth, form_sub, row['FORM'])
        assert n == 1 and form_gen == row['FORM']
        form_gen, n = re.subn(match_form, form_sub, form_dediac)
        assert n == 1 and normalize_map(form_gen) == normalize_map(row['FORM'])
        columns['FORM_SUB'] = form_sub
        columns['FORM'] = match_form_diac
        columns['FORM_EX'] = row['FORM']

        columns['BW'] = f"{match_form_diac}/{row['BW'].split('/')[1]}"
        columns['BW_SUB'] = f"{form_sub}/{row['BW'].split('/')[1]}"

        # LEMMA #########################################################
        columns['PATTERN_DEF'] = lemma_pattern_surf
        columns['PATTERN_ABS'] = abstract_pattern

        if n_t:
            lemma_pattern_surf = re.sub(str(index), r, lemma_pattern_surf)

        match_lemma_parenth = re.escape(lemma_pattern_surf)
        match_lemma_parenth = re.sub(r'\d', '(.)', match_lemma_parenth)
        lemma_sub, radicalindex2grp = generate_substitution_regex(lemma_pattern_surf)
        digits = [int(d) for d in re.findall(r'\d', lemma_sub)]
        max_digit = max(digits) if digits else 0
        assert max_digit <= n_match
        
        lemma_gen, n = re.subn(match_lemma_parenth, lemma_sub, lemma_ex_stripped)
        assert n == 1 and lemma_gen == lemma_ex_stripped
        lemma_gen, n = re.subn(match_form, lemma_sub, form_dediac)
        assert n == 1 and normalize_map(lemma_gen) == normalize_map(lemma_ex_stripped)
        columns['LEMMA_SUB'] = f"lex:{lemma_sub}"
        columns['LEMMA'] = f"lex:{lemma_pattern_surf}"
        columns['LEMMA_EX'] = row['LEMMA']

        # ROOT ##########################################################
        root, root_sub = [], []
        # count = 1
        root_split = row['ROOT'].split('.')
        for i, r in enumerate(root_split, start=1):
            if r in ['>', 'w', 'y'] or i == len(root_split) and r in ['n', 't'] or \
                i == len(root_split) - 1 and root_split[i - 1] == root_split[i] and 'gem' in cond_s:
                root.append(r)
                root_sub.append(r)
            else:
                root.append(str(i))
                grp = radicalindex2grp.get(i)
                if grp is None:
                    if r == root_split[i - 2]:
                        grp = radicalindex2grp.get(i - 1)
                    else:
                        raise NotImplementedError
                root_sub.append(f'\{grp}')
                # if i < len(root_split) and root_split[i] != r:
                # if count < n_match or i < len(root_split) - 1:
                    # count += 1

        root = '.'.join(root)
        root_sub = '.'.join(root_sub)
        match_root_parenth = re.escape(root)
        match_root_parenth = re.sub(r'\d', '(.)', match_root_parenth)
        digits = [int(d) for d in re.findall(r'\d', root_sub)]
        max_digit = max(digits) if digits else 0
        assert max_digit <= n_match
        try:
            root_gen, n = re.subn(match_root_parenth, root_sub, row['ROOT'])
        except:
            try:
                root_gen, n = re.subn(match_form, root_sub, form_dediac)
            except:
                raise NotImplementedError
        assert n == 1 and root_gen == row['ROOT']

        columns['ROOT_SUB'] = root_sub
        columns['ROOT'] = root
        columns['ROOT_EX'] = row['ROOT']

        abstract_entries.append(columns)
    
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
    parser.add_argument("-output_dir", default='data',
                        type=str, help="Path of the directory to output the lemmas to.")
    parser.add_argument("-output_name", default='ABSTRACT-LEX.csv',
                        type=str, help="Name of the file to output the abstract lexicon to.")
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
        SHEETS, _ = db_maker_utils.read_morph_specs(config, args.config_name)
        SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
            lambda row: re.sub(r'hamzated|hollow|defective', '', row['COND-S']), axis=1)
        SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
            lambda row: re.sub(r' +', ' ', row['COND-S']), axis=1)
        lexicon = SHEETS['lexicon']
        lexicon = lexicon[lexicon['FORM'] != 'DROP']
    elif args.lexicon_path:
        lexicon = pd.read_csv(args.lexicon_path)
        lexicon = lexicon.replace(nan, '', regex=True)
    else:
        raise NotImplementedError

    abstract_lexicon = generate_abstract_lexicon(lexicon)

    abstract_lexicon.to_csv(os.path.join(args.output_dir, args.output_name))
