import argparse
import sys
import os
import re
from collections import Counter
from itertools import product
import pickle
import cProfile, pstats

import gspread
from numpy import nan
import pandas as pd
from tqdm import tqdm
from time import time

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.debugging.download_sheets import download_sheets
from camel_morph.utils.utils import get_config_file, get_db_path
from camel_morph import db_maker
from camel_morph.eval.eval_utils import getsize
from camel_morph.eval.evaluate_camel_morph import _preprocess_camel_tb_data

parser = argparse.ArgumentParser()
parser.add_argument("-config_file_main", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name_main", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-config_file_factored", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name_factored", default='default_config',
                    type=str, help="Name of the configuration of the factored DB to load from the config file.")
parser.add_argument("-no_download", default=False,
                    action='store_true', help="Do not download data.")
parser.add_argument("-no_build_db", default=False,
                    action='store_true', help="Do not the DB.")
parser.add_argument("-msa_baseline_db", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the MSA baseline DB file we will be comparing against.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
parser.add_argument("-run_profiling", default=False,
                    action='store_true', help="Run execution time profiling for the make_db().")
args = parser.parse_args([] if "__file__" not in globals() else None)


if __name__ == "__main__":
    config_name = args.config_name_main
    config = get_config_file(args.config_file_main)
    config_local = config['local'][config_name]
    config_global = config['global']

    config_name_factored = args.config_name_factored
    config_factored = get_config_file(args.config_file_factored)
    config_local_factored = config_factored['local'][config_name_factored]

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.analyzer import Analyzer

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    sa = gspread.service_account(config_global['service_account'])
    
    if not args.no_download:
        download_sheets(config=config, config_name=config_name, service_account=sa)
        download_sheets(config=config_factored, config_name=config_name_factored, service_account=sa)

    if not args.no_build_db:
        print('Building DBs...')
        db_maker.make_db(config, config_name)
        db_maker.make_db(config_factored, config_name_factored)
        print()

    with open('eval_files/camel_tb_uniq_types.txt') as f:
        data = f.read()
    data = {k: int(v) for k, v in list(_preprocess_camel_tb_data(data).items())}

    t0 = time()
    db_camel_unfactored = MorphologyDB(get_db_path(config, config_name), 'dag')
    t1 = time()
    db_camel_factored = MorphologyDB(
        get_db_path(config_factored, config_name_factored), 'dag')
    t2 = time()
    db_calima = MorphologyDB(args.msa_baseline_db, 'dag')
    t3 = time()
    
    print(f'Time to load unfactored: {t1 - t0:.3f}s')
    print(f'Time to load factored: {t2 - t1:.3f}s')
    print(f'Time to load Calima: {t3 - t2:.3f}s')

    analyzer_unfactored = Analyzer(db_camel_unfactored)
    analyzer_factored = Analyzer(db_camel_factored)
    analyzer_calima = Analyzer(db_calima)
    analyzer_unfactored_cache = Analyzer(db_camel_unfactored, cache_size=5000)
    analyzer_factored_cache = Analyzer(db_camel_factored, cache_size=5000)
    analyzer_calima_cache = Analyzer(db_calima, cache_size=5000)
    
    analyzers = [
        ('unfactored', analyzer_unfactored, analyzer_unfactored_cache),
        ('factored', analyzer_factored, analyzer_factored_cache),
        ('calima', analyzer_calima, analyzer_calima_cache)
    ]

    print(f'Size unfactored: {getsize(db_camel_unfactored)}')
    print(f'Size factored: {getsize(db_camel_factored)}')
    print(f'Size Calima: {getsize(db_calima)}')

    for name, analyzer, analyzer_cache in analyzers:
        info = {}
        profiler = cProfile.Profile()
        profiler.enable()
        for token, freq in tqdm(data.items(), desc=name):
            # No caching
            t0 = time()    
            analyses = analyzer.analyze(token)
            t1 = time()
            info.setdefault('runtime', 0)
            info['runtime'] += t1 - t0
            # # With caching
            t0 = time()
            analyses = analyzer_cache.analyze(token)
            t1 = time()
            info.setdefault('runtime_cache', 0)
            info['runtime_cache'] += t1 - t0

            if args.run_profiling:
                continue

            if not analyses:
                info.setdefault('oov', 0)
                info['oov'] += 1
                info.setdefault('oov_token', 0)
                info['oov_token'] += int(freq)
            else:
                info.setdefault('recall', 0)
                info['recall'] += 1
                info.setdefault('recall_token', 0)
                info['recall_token'] += int(freq)
                info.setdefault('num_analyses', 0)
                info['num_analyses'] += int(len(analyses))
        
        profiler.disable()
        with open(f'scratch_files/profiling_loading_time_db/profiling_{name}_analysis.tsv', 'w') as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
            print(stats.print_stats())
        
        if args.run_profiling:
            continue
        print((f"{name.upper()} oov: {info['oov']/(info['oov']+info['recall']):.1%}; "
               f"oov_token: {info['oov_token']/(info['oov_token']+info['recall_token']):.1%}; "
               f"num_analyses: {info['num_analyses']:,}; "
               f"runtime_cache: {info['runtime_cache']:.2f}s"
               f"runtime: {info['runtime']:.2f}s"))