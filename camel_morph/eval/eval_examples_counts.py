import argparse
import os
import sys
import json
from itertools import combinations

from tqdm import tqdm
import gspread

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.utils.utils import Config
from camel_morph import db_maker, db_maker_utils
from camel_morph.debugging.download_sheets import download_sheets
from camel_morph.eval.evaluate_camel_morph_stats import get_analysis_counts

parser = argparse.ArgumentParser()
parser.add_argument("-config_file_main", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name_main", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-no_download", default=False,
                    action='store_true', help="Do not download data.")
parser.add_argument("-no_build_db", default=False,
                    action='store_true', help="Do not the DB.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args([] if "__file__" not in globals() else None)

if __name__ == "__main__":
    config = Config(args.config_file_main, args.config_name_main)

    if args.camel_tools == 'local':
        sys.path.insert(0, config.camel_tools)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.utils.normalize import normalize_alef_ar, \
        normalize_alef_maksura_ar, normalize_teh_marbuta_ar
    from camel_tools.utils.dediac import dediac_ar

    sa = gspread.service_account(config.service_account)

    with open(config.class_map) as f:
        CLASS_MAP = json.load(f)
        class_map_rev = {class_: complx_morph_type
                        for complx_morph_type, classes in CLASS_MAP.items()
                        for class_ in classes}

    if not args.no_download:
            print()
            download_sheets(config=config,
                            service_account=sa)
    if not args.no_build_db:
        print('Building DB...')
        SHEETS = db_maker.make_db(config)
        print()
    else:
        SHEETS, _ = db_maker_utils.read_morph_specs(config, lexicon_cond_f=False)
        
    db = MorphologyDB(config.get_db_path(), 'dag')

    info = get_analysis_counts(db, forms=True, ids=True)
    forms, ids = info['forms'], info['ids']

    normalization_fns = [
        normalize_teh_marbuta_ar, normalize_alef_maksura_ar, normalize_alef_ar]
    analyses = set()
    for word in forms:
        word = dediac_ar(word)
        analyses.add(word)
        for x in range(1, 4):
            for fns in combinations(normalization_fns, x):
                word_ = word
                for fn in fns:
                    word_ = fn(word_)
                analyses.add(word_)

    ids_class_map = {}
    for complx_morph_type, info in ids.items():
        for allomorph in info['split']:
            morph_class, id_ = allomorph.split(':')
            ids_class_map.setdefault(class_map_rev[morph_class], []).append(allomorph)

    morph = SHEETS['morph']
    morph.loc[morph['LINE'] == '', 'LINE'] = '-1'
    morph['LINE'] = morph['LINE'].astype(float).astype(int)
    conditions = set()
    for morph_type, info in ids.items():
        for allomorph in info['split']:
            morph_class, id_ = allomorph.split(':')
            if 'STEM' in morph_class:
                continue
            conditions_ = morph[(morph['LINE'] == int(id_)) & (morph['CLASS'] == morph_class)][['COND-S', 'COND-T']].values.tolist()[0]
            for cond in conditions_:
                if cond != '_':
                    for cond_or in cond.split():
                        for cond_ in cond_or.split('||'):
                            conditions.add(cond_)
