# MIT License
#
# Copyright 2022 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
from collections import OrderedDict
import os
import csv
import sys
import re
import json

import gspread
import pandas as pd
from numpy import nan

try:
    from ..utils.utils import Config
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph.utils.utils import Config

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-feats", default='',
                    type=str, help="Features to generate the conjugation tables for.")
parser.add_argument("-gsheet", default='',
                    type=str, help="Name of the manually annotated paradigms gsheet, the annotations of which will go in the bank.")
parser.add_argument("-spreadsheet", default='',
                    type=str, help="Name of the spreadsheet in which that sheet is.")
parser.add_argument("-new_system_results",  default='',
                    type=str, help="Path of the conjugation tables file generated after fixes were made to the specification sheets. This file will be automatically annotated using the information in the bank.")
parser.add_argument("-bank_dir",  default='',
                    type=str, help="Directory in which the annotation banks are.")
parser.add_argument("-bank_name",  default='',
                    type=str, help="Name of the annotation bank to use.")
parser.add_argument("-bank_spreadsheet", default='Paradigm-Banks',
                    type=str, help="Name of the spreadsheet in which that sheet is.")
parser.add_argument("-output_name",
                    type=str, help="Name of the file to output the automatically bank-annotated paradigm tables to.")
parser.add_argument("-output_dir", default='',
                    type=str, help="Path of the directory to output the paradigm tables to.")
parser.add_argument("-mode", default='debugging',
                    type=str, help="Path of the directory to output the paradigm tables to.")
parser.add_argument("-process_key", default='', choices=['', 'extra_energetic'],
                    type=str, help="Flag used to process the key before cross-checking it with bank entries while performing automatic bank annotation. Useful in cases like energetic and extra energetic when same diacs are shared by multiple paradigm slots.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")
parser.add_argument("-service_account", default='',
                    type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
args, _ = parser.parse_known_args()

feats = args.feats if args.feats else None
config = Config(args.config_file, args.config_name, feats)

if args.camel_tools == 'local':
    sys.path.insert(0, config.camel_tools)

HEADER = [
    'line', 'status', 'count', 'signature', 'lemma', 'diac_ar', 'diac', 'freq',
    'qc', 'comments', 'pattern', 'stem', 'bw', 'gloss', 'cond-s', 'cond-t',
    'pref-cat', 'stem-cat', 'suff-cat', 'feats', 'debug', 'color'
]
QUERY_KEYS = ['SIGNATURE', 'LEMMA']
VALUE_KEY = 'DIAC'

class AnnotationBank:
    UNKOWN = 'UNK'
    GOOD = 'OK'
    PROBLEM = 'PROB'
    TAGS = {UNKOWN, GOOD, PROBLEM}
    HEADER_INFO = ['QC', 'COMMENTS', 'STATUS']

    def __init__(self,
                 bank_path,
                 query_keys=QUERY_KEYS,
                 value_key=VALUE_KEY,
                 annotated_sheet=None,
                 gsheet_info=None,
                 sa=None):
        self._bank_path = bank_path
        self.sa = sa
        self._bank, self._unkowns = OrderedDict(), OrderedDict()
        self.query_keys, self.value_key = query_keys, value_key
        self.header_key = self.query_keys + [self.value_key]
        self.header = self.header_key + AnnotationBank.HEADER_INFO
        self.key2index = {k: i for i, k in enumerate(self.header_key)}
        
        if gsheet_info is not None:
            self._gsheet_name = gsheet_info['gsheet_name']
            self._spreadsheet = gsheet_info['spreadsheet']
            self.sa = gsheet_info['sa']
            sh = self.sa.open(self._spreadsheet)
            worksheet = sh.worksheet(title=self._gsheet_name)
            bank = pd.DataFrame(worksheet.get_all_records())
            self._read_bank_from_df(bank)
        else:
            if os.path.exists(bank_path):
                self._read_bank_from_tsv()

        if annotated_sheet is not None:
            self._update_bank(annotated_sheet)

    def get(self, key, default=None):
        return self._bank[key] if key in self._bank else default
        

    def __contains__(self, key):
        return key in self._bank

    def __setitem__(self, key, item):
        self._bank[key] = item

    def __getitem__(self, key):
        return self._bank[key]

    def _update_bank(self, annotated_sheet):
        annotated_sheet['QC'] = annotated_sheet['QC'].replace(
            '', AnnotationBank.UNKOWN, regex=True)
        annotated_sheet['QC'] = annotated_sheet['QC'].str.strip()
        annotated_sheet['QC'] = annotated_sheet['QC'].str.upper()
        if 'LEMMA' in annotated_sheet.columns:
            annotated_sheet['LEMMA'] = annotated_sheet['LEMMA'].replace(
                r'_\d', '', regex=True)
        assert all([qc in AnnotationBank.TAGS
                    for qc in annotated_sheet['QC'].values.tolist()]), \
            f'Get rid of all tags not belonging to {AnnotationBank.TAGS}'
        assert set(self.header).issubset(annotated_sheet.columns)
        
        for _, row in annotated_sheet.iterrows():
            key_annot = tuple([row[h] for h in self.header_key])
            if row['QC'] != AnnotationBank.UNKOWN:
                self._bank[key_annot] = {
                    h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO
                                                if h != 'STATUS']}
                self._bank[key_annot]['STATUS'] = ''
            else:
                self._unkowns[key_annot] = {
                    h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO
                                                if h != 'STATUS']}

        self._save_bank()

    def _save_bank(self):
        with open(self._bank_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(self.header)
            writer.writerows([[k for k in key] + [self._bank[key].get(h, '')
                                                  for h in AnnotationBank.HEADER_INFO]
                                for key in self._bank])

    def _upload_gsheet(self, df=None, sh=None, sheet=None):
        sh = self.sa.open(sh if sh is not None else self._spreadsheet)
        worksheet = sh.worksheet(title=sheet if sheet is not None else self._gsheet_name)
        worksheet.clear()
        bank = df if df is not None else self.to_df()
        worksheet.update(
            [bank.columns.values.tolist()] + bank.values.tolist())

    def _read_bank_from_tsv(self):
        bank = pd.read_csv(self._bank_path, delimiter='\t')
        bank = bank.replace(nan, '', regex=True)
        if 'LEMMA' in bank.columns:
            bank['LEMMA'] = bank['LEMMA'].replace(r'_\d', '', regex=True)
        self._bank = OrderedDict(
            (tuple([row[h] for h in self.header_key]), 
                {h: row.get(h, '') for h in AnnotationBank.HEADER_INFO})
            for _, row in bank.iterrows())

    def _read_bank_from_df(self, df):
        if 'STATUS' in df.columns:
            df = df[df['STATUS'] != 'DELETE']
        for _, row in df.iterrows():
            info = {h: row.get(h, '') for h in AnnotationBank.HEADER_INFO}
            self._bank[tuple([row[h] for h in self.header_key])] = info

    def to_df(self):
        columns = self.header_key + [h for h in AnnotationBank.HEADER_INFO]
        if self._bank:
            bank_df = pd.DataFrame([
                list(k) + [v[k] for k in AnnotationBank.HEADER_INFO]
                for k, v in self._bank.items()], columns=columns)
        else:
            bank_df = pd.DataFrame(columns=columns)
        return bank_df

def _process_key(key, mode):
    if mode == 'extra_energetic':
        signature = key[0]
        signature = re.sub(r'(I2D\..)X', r'\1E', signature)
        signature = re.sub(r'(I2FP\..)X', r'\1E', signature)
        signature = re.sub(r'(I3FD\..)X', r'\1E', signature)
        signature = re.sub(r'(I3FP\..)X', r'\1E', signature)
        signature = re.sub(r'(I3MD\..)X', r'\1E', signature)
        key = (signature, *key[1:])
    
    return key

def automatic_bank_annotation(config:Config,
                              new_system_results:pd.DataFrame,
                              sa=None,
                              mode='debugging',
                              query_keys=QUERY_KEYS,
                              value_key=VALUE_KEY,
                              bank_dir=None,
                              process_key=None,
                              header=HEADER,
                              header_upper=True):
    bank_path, annotated_sheet = setup(config, sa, mode, bank_dir)
    if annotated_sheet is None:
        annotated_sheet = new_system_results
    bank = AnnotationBank(bank_path, annotated_sheet=annotated_sheet,
                          sa=sa, query_keys=query_keys, value_key=value_key)
    if header is None:
        header = bank.header
    
    indexes_query = set(index for key, index in bank.key2index.items()
                        if key in bank.query_keys)
    partial_keys = {}
    for k, info in bank._bank.items():
        query_keys = tuple(k_ for i, k_ in enumerate(k) if i in indexes_query)
        value_key = k[bank.key2index[bank.value_key]]
        partial_keys.setdefault(query_keys, []).append(
            {**info, **{bank.value_key: value_key}})

    outputs = []
    for _, row in new_system_results.iterrows():
        key = tuple(row[k] for k in bank.header_key)
        if process_key is not None:
            key = _process_key(key, process_key)
        row = row.to_dict()
        info = bank.get(key)
        if info is not None:
            row['QC'] = info['QC']
            if info['QC'] == AnnotationBank.PROBLEM:
                pass
            key_partial = tuple(row[k] for k in bank.query_keys)
            if len(partial_keys[key_partial]) > 1:
                warnings = [
                    f"{AnnotationBank.GOOD}-MULT:{info[bank.value_key]}"
                    for info in partial_keys[key_partial]
                    if info['QC'] == AnnotationBank.GOOD and
                    key[bank.key2index[bank.value_key]] != info[bank.value_key]]
                if warnings:
                    row['WARNINGS'] = ' '.join(warnings)
        else:
            for k, info in bank._bank.items():
                key_partial = tuple(key[bank.key2index[k_]] for k_ in bank.query_keys)
                k_partial = tuple(k[bank.key2index[k_]] for k_ in bank.query_keys)
                if key_partial == k_partial:
                    row['QC'] = f"({k[-1]})[{info['QC']}]>({row[bank.value_key]})[{AnnotationBank.UNKOWN}]"
                    break
            else:
                row['QC'] = AnnotationBank.UNKOWN
        comment = bank._unkowns.get(key)
        if comment is None:
            comment = info['COMMENTS'] if info is not None else ''
        else:
            comment = comment['COMMENTS']
        row['COMMENTS'] = comment

        output_ordered = OrderedDict()
        for k in header:
            if header_upper:
                k = k.upper()
            output_ordered[k] = row.get(k, '')
        
        outputs.append(output_ordered)

    outputs_ = {}
    for row in outputs[1:]:
        for h, value in row.items():
            outputs_.setdefault(h, []).append(value)
    columns = map(str.upper, header) if header_upper else header
    outputs_df = pd.DataFrame(outputs_, columns=columns)

    return outputs_df, bank

def bank_cleanup_checker(bank_path, gsheet_info, mode, annotated_sheet=None):
    mode_ = ''
    if len(mode.split('_')) > 2:
        mode_ = '_'.join(mode.split('_')[2:])
    def fetch_bank():
        bank = AnnotationBank(bank_path, gsheet_info=gsheet_info)
        bank_partial_key = {}
        for key, info in bank._bank.items():
            bank_partial_key.setdefault(key[:-1], {}).setdefault(key[-1], []).append(info)
        return bank, bank_partial_key
    
    if mode_ == 'freeze_table_as_bank':
        bank = AnnotationBank(bank_path)
        bank._bank = OrderedDict()
        bank._update_bank(annotated_sheet)
    else:
        print('\nBeginning cleanup...')
        fixed = False
        while not fixed:
            bank, bank_partial_key = fetch_bank()
            over_generations = {}
            for partial_key, diacs in bank_partial_key.items():
                for diac, infos in diacs.items():
                    if len(infos) > 1:
                        over_generations[partial_key + (diac,)] = infos
            
            if len(over_generations) != 0:
                _add_check_mark_online(bank, over_generations)
                input('Inspect CHECK instances in Google Sheets then press Enter to reevaluate:')
            else:
                print('Overgeneration: OK')
                fixed = True

        fixed = False
        while not fixed:
            bank, bank_partial_key = fetch_bank()
            conflicting_annotations = {}
            bank_mt1_diac = {partial_key: diacs for partial_key, diacs in bank_partial_key.items() if len(diacs) > 1}
            for partial_key, diacs in bank_mt1_diac.items():
                if [diac[0]['QC'] for diac in diacs.values()].count('OK') > 1:
                    for diac, infos in diacs.items():
                        conflicting_annotations[partial_key + (diac,)] = infos[0]
            
            _add_check_mark_online(bank, conflicting_annotations)
            if len(conflicting_annotations) != 0:
                input('Inspect CHECK instances in Google Sheets then press Enter to reevaluate:')
            else:
                print('Conflicting annotations: OK')
                fixed = True

        bank = AnnotationBank(bank_path, gsheet_info=gsheet_info, download_bank=True)
        bank._save_bank()

        print('Cleanup completed.\n')


def _add_check_mark_online(bank, error_cases):
    bank_df = bank.to_df()

    bank_df['STATUS'] = ''
    for i, row in bank_df.iterrows():
        key = tuple(row[k] for k in bank.header_key)
        if key in error_cases:
            bank_df.loc[i, 'STATUS'] = 'CHECK'

    bank._upload_gsheet(bank_df)


def setup(config:Config, sa, mode, bank_dir):
    if bank_dir is None:
        bank_dir = args.bank_dir if args.bank_dir else config.get_banks_dir_path()
    os.makedirs(bank_dir, exist_ok=True)
    bank_name = args.bank_name if args.bank_name \
        else config.debugging.debugging_feats.bank
    bank_path = os.path.join(bank_dir, bank_name)
    
    spreadsheet = args.spreadsheet if args.spreadsheet \
        else config.debugging.debugging_spreadsheet
    if sa is None:
        sa = gspread.service_account(config.service_account)
    sh = sa.open(spreadsheet)

    gsheet = args.gsheet if args.gsheet \
        else config.debugging.debugging_feats.debugging_sheet
    worksheets = sh.worksheets()
    worksheet = [sheet for sheet in worksheets if sheet.title == gsheet]
    annotated_sheet = None
    if worksheet:
        annotated_sheet = pd.DataFrame(worksheet[0].get_all_records())

    if mode.startswith('bank_cleanup'):
        bank_spreadsheet = args.bank_spreadsheet if args.bank_spreadsheet \
            else config.banks_spreadsheet
        gsheet_info = {
            'gsheet_name': bank_path.split('/')[-1].split('.')[0],
            'spreadsheet': bank_spreadsheet,
            'sa': sa}
        bank_cleanup_checker(bank_path, gsheet_info, mode, annotated_sheet)
        sys.exit()

    return bank_path, annotated_sheet


if __name__ == "__main__":
    output_dir = args.output_dir if args.output_dir else config.get_paradigm_debugging_dir_path()
    os.makedirs(output_dir, exist_ok=True)

    output_name = args.output_name if args.output_name \
        else config.debugging.debugging_feats.paradigm_debugging
    output_path = os.path.join(output_dir, output_name)

    service_account = args.service_account if args.service_account else config.service_account
    sa = gspread.service_account(service_account)

    if args.new_system_results:
        new_sys_results_path = args.new_system_results
    else:
        new_sys_results_path = config.get_tables_dir_path()
        new_sys_results_path = os.path.join(
            new_sys_results_path, config.debugging.debugging_feats.conj_tables)

        new_sys_results_table = pd.read_csv(new_sys_results_path, delimiter='\t')
        new_sys_results_table = new_sys_results_table.replace(nan, '', regex=True)

    #FIXME: do something about process_key (which is used from debugging
    # command energetic and extra energetic)
    outputs_df, bank = automatic_bank_annotation(config=config,
                                                 process_key=args.process_key,
                                                 new_system_results=new_sys_results_table,
                                                 sa=sa)
    outputs_df.to_csv(output_path, sep='\t')
