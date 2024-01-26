import argparse
import sys
import os

import pandas as pd
import gspread

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.debugging.download_sheets import download_sheets
from camel_morph.utils.utils import Config
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


if __name__ == "__main__":
    config = Config(args.config_file_main, args.config_name_main)

    if args.camel_tools == 'local':
        sys.path.insert(0, config.camel_tools)
    
    sa = gspread.service_account(config.service_account)
    
    if not args.no_download:
        print()
        download_sheets(config=config,
                        service_account=sa)
        
    if not args.no_build_db:
        print('Building DB...')
        db_maker.make_db(config)
        print()
        
    feats = args.feats
    if not args.feats:
        feats = list(config.debugging.feats)
    
    for feats_ in feats:
        config = Config(args.config_file_main, args.config_name_main, feats_)
        print('\nExtracting representative lemma groups...', end=' ')
        repr_lemmas = create_repr_lemmas_list(config=config)
        print('Done.\n')

        HEADER = [
            'line', 'status', 'count', 'signature', 'lemma', 'diac_ar', 'diac', 'freq',
            'qc', 'warnings', 'comments', 'pattern', 'stem', 'bw', 'gloss', 'pos',
            'cond-s', 'cond-t', 'pref-cat', 'stem-cat', 'suff-cat', 'feats', 'debug', 'color'
        ]

        print('Building inflection table...', end=' ')
        conj_table_df = create_conjugation_tables(config=config,
                                               paradigm_key=feats_,
                                               repr_lemmas=repr_lemmas,
                                               HEADER=HEADER)
        print('Done.\n')
        
        print('Querying bank and automatically quality checking inflection table...', end=' ')
        outputs_df, bank = automatic_bank_annotation(config=config,
                                                  new_system_results=conj_table_df,
                                                  sa=sa,
                                                  header=HEADER)
        print('Done.\n')
        
        upload_sheet(config=config,
                     sheet=outputs_df,
                     input_dir='paradigm_debugging_dir',
                     mode='backup',
                     sa=sa)
        upload_sheet(config=config,
                     sheet=bank.to_df(),
                     input_dir='banks_dir',
                     mode='backup',
                     sa=sa)
    
