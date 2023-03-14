import os
import argparse
import pandas as pd
import gspread
import pickle
import numpy as np
import re
import json
from numpy import nan

from camel_morph.utils.utils import add_check_mark_online

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-well_formedness", default=False,
                    action='store_true', help="Whether or not to perform some well-formedness checks before generating.")
parser.add_argument("-spreadsheet", required=True,
                    type=str, help="Google Sheets spreadsheet name to output to.")
parser.add_argument("-sheet", required=True,
                    type=str, help="Google Sheets sheet name to output to.")
args = parser.parse_args([] if "__file__" not in globals() else None)

FIELDS_MAP = dict(
    lemma='LEMMA',
    form='FORM',
    cond_t='COND-T',
    cond_s='COND-S',
    gloss='GLOSS',
    line='LINE',
    pos='POS',
    gen='GEN',
    num='NUM',
    rat='RAT',
    cas='CAS',
    stem_count='STEM_COUNT'
)

COLUMNS_OUTPUT = ['LINE', 'STEM_COUNT', 'META_INFO', 'FREQ', 'LEMMA',
                  'FORM', 'COND-T', 'COND-S', 'GLOSS', 'POS', 'GEN',
                  'NUM', 'RAT', 'CAS', 'SIGNATURE', 'FLAGS']

def _strip_brackets(info):
    if info[0] == '[':
        info = info[1:]
    if info[-1] == ']':
        info = info[:-1]
    return info

def _split_field(field):
    field_split = field.split(']-[')
    for i in [0, -1]:
        field_split[i] = _strip_brackets(field_split[i])
    return field_split


def well_formedness_check():
    nom_lex = pd.read_csv('data/camel-morph-msa/noun_msa_lemma_paradigms/MSA-Nom-LEX.csv')
    nom_lex = nom_lex.replace(nan, '')
    essential_columns = ['ROOT', 'LEMMA', 'FORM', 'GLOSS', 'FEAT', 'COND-T', 'COND-S']
    # Duplicate entries
    nom_lex_essential = [tuple(row) for row in nom_lex[essential_columns].values.tolist()]
    nom_lex_essential_set = set(nom_lex_essential)
    assert len(nom_lex_essential) == len(nom_lex_essential_set)
    # Glosses should be merged
    essential_columns_no_gloss = [col for col in essential_columns if col != 'GLOSS']
    messages = []
    key2indexes = {}
    for i, row in nom_lex.iterrows():
        key = tuple(row[essential_columns_no_gloss])
        key2indexes.setdefault(key, []).append(i)
    
    index2message = {}
    for key, indexes in key2indexes.items():
        for j, index in enumerate(indexes):
            if len(key2indexes[key]) > 1:
                if j:
                    index2message[index] = 'delete'
                else:
                    index2message[index] = '###'.join(nom_lex.loc[index, 'GLOSS']
                                                      for index in key2indexes[key])
            else:
                messages.append('')

    messages = [index2message[i] if i in index2message else '' for i in range(len(nom_lex.index))]

    if set(messages) != {''}:
        add_check_mark_online(rows=nom_lex,
                            spreadsheet='camel-morph-msa-nom-other',
                            sheet='MSA-Nom-LEX',
                            status_col_name='STATUS_CHRIS',
                            write='overwrite',
                            messages=messages)


def generate_lex_rows(repr_lemmas):
    rows = {}
    for lemma_paradigm, lemmas_info in repr_lemmas.items():
        for lemma_info in lemmas_info['lemmas']:
            stem_mask = [info for info in lemma_info['meta_info'].split()
                            if 'stem' in info][0]
            stem_mask = stem_mask.split(':')[1].split('-')
            # Add all lex fields
            for field, field_header in FIELDS_MAP.items():
                values = str(lemma_info[field])
                if ']-[' in values:
                    values_ = _split_field(values)
                else:
                    values_ = [_strip_brackets(values)] * len(stem_mask)
                
                for i, stem_id in enumerate(stem_mask):
                    if field not in ['gen', 'num']:
                        values_[i] = '' if values_[i] == '-' else values_[i]
                    rows.setdefault(field_header.upper(), []).append(values_[i])
            # Add signature
            for i, stem_id in enumerate(stem_mask):
                signature = f"{lemma_info['cond_t']} {lemma_info['gen']} {lemma_info['num']}"
                rows.setdefault('SIGNATURE', []).append(signature)
            # Metainfo
            for i, stem_id in enumerate(stem_mask):
                meta_info = lemma_info['meta_info']
                rows.setdefault('META_INFO', []).append(meta_info)
            # Add frequency
            for i, stem_id in enumerate(stem_mask):
                rows.setdefault('FREQ', []).append(lemmas_info['freq'])

            # Check for stem well-formedness, i.e., at least one stem in a lemma system
            # should match the prefix of the lemma.
            well_formed = any(True if re.match(stem, lemma_info['lemma_ar']) else False
                            for stem in _split_field(lemma_info['form_ar']))
            
            for i, stem_id in enumerate(stem_mask):
                rows.setdefault('FLAGS', []).append('check_stems' if not well_formed else '')
    
    return rows


if __name__ == "__main__":
    with open(args.config_file) as f:
        config = json.load(f)
    config_name = args.config_name
    config_local = config['local'][config_name]
    config_global = config['global']

    repr_lemmas_path = os.path.join(
        'sandbox_files/camel-morph-msa', f'repr_lemmas_{args.config_name}.pkl')
    with open(repr_lemmas_path, 'rb') as f:
        repr_lemmas = pickle.load(f)

    if args.well_formedness:
        well_formedness_check()

    rows = generate_lex_rows(repr_lemmas)

    repr_lemmas = pd.DataFrame(rows)
    repr_lemmas = repr_lemmas.replace(np.nan, '', regex=True)
    repr_lemmas = repr_lemmas[COLUMNS_OUTPUT]
    sa = gspread.service_account(config_global['service_account'])
    sh = sa.open(args.spreadsheet)
    sheets = sh.worksheets()
    if args.sheet in [sheet.title for sheet in sheets]:
        worksheet = sh.worksheet(title=args.sheet)
    else:
        worksheet = sh.add_worksheet(title=args.sheet, rows="100", cols="20")
    
    worksheet.clear()
    worksheet.update(
        [repr_lemmas.columns.values.tolist()] + repr_lemmas.values.tolist())