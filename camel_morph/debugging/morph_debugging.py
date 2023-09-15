import argparse
import sys
import os

import pandas as pd
import gspread

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.debugging.download_sheets import download_sheets
from camel_morph.utils.utils import get_config_file
from camel_morph.debugging.create_repr_lemmas_list import create_repr_lemmas_list
from camel_morph.debugging.generate_conj_table import create_conjugation_tables
from camel_morph.debugging.paradigm_debugging import automatic_bank_annotation
from camel_morph.debugging.upload_sheets import upload_sheet
from camel_morph import db_maker

parser = argparse.ArgumentParser()
parser.add_argument("-config_file_main", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name_main", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-no_download", default=False,
                    action='store_true', help="Do not download data.")
parser.add_argument("-no_build_db", default=False,
                    action='store_true', help="Do not the DB.")
parser.add_argument("-feats", default='',
                    type=str, help="Features to generate the conjugation tables for.")
parser.add_argument("-lemma_debug", default=[], action='append',
                    type=str, help="Lemma (without _1) to debug. Use the following format after the flag: lemma pos:val gen:val num:val")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args([] if "__file__" not in globals() else None)


def _get_df(table):
    table_ = {}
    for row in table[1:]:
        for h, value in row.items():
            table_.setdefault(h, []).append(value)
    table = pd.DataFrame(table_)
    return table


if __name__ == "__main__":
    config_name = args.config_name_main
    config = get_config_file(args.config_file_main)
    config_local = config['local'][config_name]
    config_global = config['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)
    
    sa = gspread.service_account(config_global['service_account'])
    
    if not args.no_download:
        print()
        download_sheets(config=config,
                        config_name=config_name,
                        service_account=sa)
    
    print('\nExtracting representative lemma groups...', end=' ')
    repr_lemmas = create_repr_lemmas_list(config=config,
                                          config_name=config_name,
                                          feats_bank=args.feats)
    print('Done.\n')
    
    if not args.no_build_db:
        print('Building DB...')
        db_maker.make_db(config, config_name)
        print()

    HEADER = [
        'line', 'status', 'count', 'signature', 'lemma', 'diac_ar', 'diac', 'freq',
        'qc', 'warnings', 'comments', 'pattern', 'stem', 'bw', 'gloss', 'pos',
        'cond-s', 'cond-t', 'pref-cat', 'stem-cat', 'suff-cat', 'feats', 'debug', 'color'
    ]

    print('Building inflection table...', end=' ')
    conj_table = create_conjugation_tables(config=config,
                                           config_name=config_name,
                                           paradigm_key=args.feats,
                                           repr_lemmas=repr_lemmas,
                                           HEADER=HEADER)
    conj_table = _get_df(conj_table)
    print('Done.\n')
    
    print('Querying bank and automatically quality checking inflection table...', end=' ')
    outputs = automatic_bank_annotation(config=config,
                                        config_name=config_name,
                                        feats=args.feats,
                                        new_conj_table=conj_table,
                                        sa=sa,
                                        HEADER=HEADER)
    outputs = _get_df(outputs)
    print('Done.\n')
    
    upload_sheet(config=config,
                 config_name=config_name,
                 feats=args.feats,
                 sheet=outputs,
                 input_dir='paradigm_debugging_dir',
                 mode='backup',
                 sa=sa)
    upload_sheet(config=config,
                 config_name=config_name,
                 feats=args.feats,
                 sheet=outputs,
                 input_dir='banks_dir',
                 mode='backup',
                 sa=sa)
    
