import argparse
import sys
import os
import re
from collections import OrderedDict

import gspread
import pandas as pd

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.utils.utils import add_check_mark_online

parser = argparse.ArgumentParser()
parser.add_argument("-spreadsheet_name", default='',
                    type=str, help="Name of the Google spreadsheet to write to.")
parser.add_argument("-sheet_name", action='append',
                    help="Name of the Google sheet to write to.")
parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
args = parser.parse_args([] if "__file__" not in globals() else None)



def fix_underscores(sheet_df, sh, sheet):
    bad_entries = {}
    lemma_stripped_pos2index2lemma, lemmas_fixed = {}, []
    for row_index, row in sheet_df.iterrows():
        lemma, pos = row['LEMMA'], re.search(r'pos:([^ ]+)', row['FEAT']).group(1)
        lemma_stripped = lemma.split('_')[0]
        lemma_stripped_pos2index2lemma.setdefault(
            (lemma_stripped, pos), {}).setdefault(row_index, lemma)
    
    for lemma_stripped_pos, index2lemma in lemma_stripped_pos2index2lemma.items():
        underscore2row_indexes = {}
        for row_index, lemma in index2lemma.items():
            match_ = re.match(r'[^_]+_?(\d+)?', lemma)
            assert match_.group(1) is None or int(match_.group(1)) >= 1
            und_old = int(match_.group(1)) if match_.group(1) is not None else 0
            underscore2row_indexes.setdefault(und_old, []).append(row_index)
        underscore2row_indexes = sorted(underscore2row_indexes.items(), key=lambda x: x[0])
        
        underscores = [und for und, _ in underscore2row_indexes]
        if underscores[0] > 1 or underscores != list(range(min(underscores), max(underscores)+1)) or \
            0 in underscores and len(set(underscores)) > 1 or \
            underscores[0] == 1 and len(set(underscores)) == 1:
            bad_entries[lemma_stripped_pos] = index2lemma

        reindexed_und2row_indexes = OrderedDict()
        und_index_new = 0
        for _, row_indexes in underscore2row_indexes:
            if len(underscore2row_indexes) > 1:
                und_index_new += 1
            for row_index in row_indexes:
                reindexed_und2row_indexes.setdefault(und_index_new, []).append(row_index)
        
        for reindexed_und, row_indexes in reindexed_und2row_indexes.items():
            lemma_ = lemma_stripped_pos[0] + (f'_{reindexed_und}' if reindexed_und else '')
            for row_index in row_indexes:
                lemma = lemma_stripped_pos2index2lemma[lemma_stripped_pos][row_index]
                lemmas_fixed.append((row_index, lemma_))
        
        pass
        
    lemmas_fixed = [lemma for _, lemma in sorted(lemmas_fixed)]
    lemmas_diff = [(old, new) for old, new in zip(sheet_df['LEMMA'].tolist(), lemmas_fixed)
                   if old != new]
    
    if lemmas_diff:
        add_check_mark_online(rows=sheet_df,
                            spreadsheet=sh,
                            worksheet=sheet,
                            status_col_name='LEMMA',
                            write='overwrite',
                            messages=lemmas_fixed)


if __name__ == "__main__":
    sa = gspread.service_account(args.service_account)
    sh = sa.open(args.spreadsheet_name)

    for sheet_name in args.sheet_name:
        sheet = sh.worksheet(args.sheet_name)
        sheet_df = pd.DataFrame(sheet.get_all_records())
        fix_underscores(sheet_df, sh, sheet)