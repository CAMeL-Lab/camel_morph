import argparse
import sys
import os

import gspread
import pandas as pd

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.utils.utils import Config, sheet2df
from camel_morph.debugging.download_sheets import download_sheets

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args([] if "__file__" not in globals() else None)


class Lexeme:
    def __init__(self) -> None:
        pass
        
class Stem:
    def __init__(self) -> None:
        pass


def automatic_fixes(config:Config):
    lexicon_sheets_paths = config.get_sheets_paths('lexicon', with_labels=True)
    for sheet_path in lexicon_sheets_paths:
        sheet_df = pd.read_csv(sheet_path, na_filter=False)


def manual_fixes():
    pass


if __name__ == "__main__":
    config = Config(args.config_file, args.config_name)

    download_sheets(config=config)


    if args.camel_tools == 'local':
        sys.path.insert(0, config.camel_tools)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.utils import merge_features

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    sa = gspread.service_account(config.service_account)

    automatic_fixes(config)
    manual_fixes()
