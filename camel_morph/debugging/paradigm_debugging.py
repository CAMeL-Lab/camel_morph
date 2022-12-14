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

header = ["line", "status", "count", "signature", "lemma", "diac_ar", "diac", "freq",
          "qc", "warnings", "comments", "pattern", "stem", "bw", "gloss", "cond-s",
          "cond-t", "pref-cat", "stem-cat", "suff-cat", "feats", "debug", "color"]

class AnnotationBank:
    UNKOWN = 'UNK'
    GOOD = 'OK'
    PROBLEM = 'PROB'
    TAGS = {UNKOWN, GOOD, PROBLEM}
    HEADER_KEY = ['SIGNATURE', 'LEMMA', 'DIAC']
    HEADER_INFO = ['QC', 'COMMENTS', 'STATUS']
    HEADER = HEADER_KEY + HEADER_INFO

    def __init__(self, bank_path, annotated_paradigms=None, gsheet_info=None, download_bank=False):
        self._bank_path = bank_path
        if gsheet_info is not None:
            self._gsheet_name = gsheet_info['gsheet_name']
            self._spreadsheet = gsheet_info['spreadsheet']
        self._bank, self._unkowns = OrderedDict(), OrderedDict()

        if not download_bank:
            if os.path.exists(bank_path):
                self._read_bank_from_tsv()
        else:
            sh = sa.open(self._spreadsheet)
            worksheet = sh.worksheet(title=self._gsheet_name)
            bank = pd.DataFrame(worksheet.get_all_records())
            self._read_bank_from_df(bank)

        if annotated_paradigms is not None:
            self._update_bank(annotated_paradigms)

    def get(self, key, default=None):
        return self._bank[key] if key in self._bank else default
        

    def __contains__(self, key):
        return key in self._bank

    def __setitem__(self, key, item):
        self._bank[key] = item

    def __getitem__(self, key):
        return self._bank[key]

    def _update_bank(self, annotated_paradigms):
        annotated_paradigms['QC'] = annotated_paradigms['QC'].replace(
            '', AnnotationBank.UNKOWN, regex=True)
        annotated_paradigms['QC'] = annotated_paradigms['QC'].str.strip()
        annotated_paradigms['QC'] = annotated_paradigms['QC'].str.upper()
        annotated_paradigms['LEMMA'] = annotated_paradigms['LEMMA'].replace(r'_\d', '', regex=True)
        assert all([qc in AnnotationBank.TAGS for qc in annotated_paradigms['QC'].values.tolist()]), \
            f"Get rid of all tags not belonging to {AnnotationBank.TAGS}"
        assert set(AnnotationBank.HEADER).issubset(annotated_paradigms.columns)
        
        for _, row in annotated_paradigms.iterrows():
            key_annot = tuple([row[h] for h in AnnotationBank.HEADER_KEY])
            if row['QC'] != AnnotationBank.UNKOWN:
                self._bank[key_annot] = {h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO if h != 'STATUS']}
                self._bank[key_annot]['STATUS'] = ''
            else:
                self._unkowns[key_annot] = {h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO if h != 'STATUS']}

        self._save_bank()

    def _save_bank(self):
        with open(self._bank_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(AnnotationBank.HEADER)
            writer.writerows([[k for k in key] + [self._bank[key].get(h, '') for h in AnnotationBank.HEADER_INFO]
                                for key in self._bank])

    def _upload_gsheet(self, df=None):
        sa = gspread.service_account(
            "/Users/chriscay/.config/gspread/service_account.json")
        sh = sa.open(self._spreadsheet)
        worksheet = sh.worksheet(title=self._gsheet_name)
        worksheet.clear()
        bank = df if df is not None else self.to_df()
        worksheet.update(
            [df.columns.values.tolist()] + df.values.tolist())

    def _read_bank_from_tsv(self):
        bank = pd.read_csv(self._bank_path, delimiter='\t')
        bank = bank.replace(nan, '', regex=True)
        bank['LEMMA'] = bank['LEMMA'].replace(r'_\d', '', regex=True)
        self._bank = OrderedDict(
            (tuple([row[h] for h in AnnotationBank.HEADER_KEY]), 
                {h: row.get(h, '') for h in AnnotationBank.HEADER_INFO})
            for _, row in bank.iterrows())

    def _read_bank_from_df(self, df):
        if 'STATUS' in df.columns:
            df = df[df['STATUS'] != 'DELETE']
        for _, row in df.iterrows():
            info = {h: row.get(h, '') for h in AnnotationBank.HEADER_INFO}
            self._bank[tuple([row[h] for h in AnnotationBank.HEADER_KEY])] = info

    def to_df(self):
        bank = pd.DataFrame([list(k) + [v[k] for k in AnnotationBank.HEADER_INFO]
                                for k, v in self._bank.items()])
        bank.columns = AnnotationBank.HEADER
        return bank

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

def automatic_bank_annotation(bank_path, annotated_paradigms, new_conj_tables, process_key):
    bank = AnnotationBank(bank_path, annotated_paradigms)
    lemmas = [entry[1] for entry in bank._bank]
    strip = True if not any(['-' in lemma for lemma in lemmas]) else False
    partial_keys = {}
    for k, info in bank._bank.items():
        partial_keys.setdefault(k[:-1], []).append({**info, **{'DIAC': k[-1]}})

    outputs = []
    for _, row in new_conj_tables.iterrows():
        lemma = row['LEMMA'] if not strip else strip_lex(row['LEMMA'])
        key = (row['SIGNATURE'], lemma, row['DIAC'])
        if process_key:
            key = _process_key(key, process_key)
        row = row.to_dict()
        info = bank.get(key)
        if info is not None:
            row['QC'] = info['QC']
            if info['QC'] == AnnotationBank.PROBLEM:
                pass
            key_ = key[:-1]
            if len(partial_keys[key_]) > 1:
                warnings = [f"{AnnotationBank.GOOD}-MULT:{info['DIAC']}" for info in partial_keys[key_]
                                if info['QC'] == AnnotationBank.GOOD and key[-1] != info['DIAC']]
                if warnings:
                    row['WARNINGS'] = ' '.join(warnings)
        else:
            for k, info in bank._bank.items():
                if key[:-1] == k[:-1]:
                    row['QC'] = f"({k[-1]})[{info['QC']}]>({row['DIAC']})[{AnnotationBank.UNKOWN}]"
                    break
            else:
                row['QC'] = AnnotationBank.UNKOWN
        comment = bank._unkowns.get(key)
        if comment is None:
            comment = info['COMMENTS']
        else:
            comment = comment['COMMENTS']
        row['COMMENTS'] = comment

        output_ordered = OrderedDict()
        for k in header:
            output_ordered[k.upper()] = row.get(k.upper(), '')
        
        outputs.append(output_ordered)

    outputs.insert(0, OrderedDict((i, x) for i, x in enumerate(map(str.upper, header))))
    return outputs

def bank_cleanup_checker(bank_path, gsheet_info):
    def fetch_bank():
        bank = AnnotationBank(bank_path, gsheet_info=gsheet_info, download_bank=True)
        bank_partial_key = {}
        for key, info in bank._bank.items():
            bank_partial_key.setdefault(key[:-1], {}).setdefault(key[-1], []).append(info)
        return bank, bank_partial_key

    print('\nBeginning cleanup...')
    fixed = False
    while not fixed:
        bank, bank_partial_key = fetch_bank()
        over_generations = {}
        for partial_key, diacs in bank_partial_key.items():
            for diac, infos in diacs.items():
                if len(infos) > 1:
                    over_generations[partial_key + (diac,)] = infos
        
        _add_check_mark_online(bank, over_generations)
        if len(over_generations) != 0:
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
        if (row['SIGNATURE'], row['LEMMA'], row['DIAC']) in error_cases:
            bank_df.loc[i, 'STATUS'] = 'CHECK'

    bank._upload_gsheet(bank_df)


if __name__ == "__main__":
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
    parser.add_argument("-new_conj",  default='',
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
    parser.add_argument("-mode", default='debugging', choices=['debugging', 'bank_cleanup'],
                        type=str, help="Path of the directory to output the paradigm tables to.")
    parser.add_argument("-process_key", default='', choices=['', 'extra_energetic'],
                        type=str, help="Flag used to process the key before cross-checking it with bank entries while performing automatic bank annotation. Useful in cases like energetic and extra energetic when same diacs are shared by multiple paradigm slots.")
    parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)
    config_local = config['local'][args.config_name]
    config_global = config['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)
        
    from camel_tools.morphology.utils import strip_lex
    
    bank_dir = args.bank_dir if args.bank_dir else os.path.join(
        config_global['debugging'], config_global['banks_dir'], f"camel-morph-{config_local['dialect']}")
    os.makedirs(bank_dir, exist_ok=True)
    bank_name = args.bank_name if args.bank_name else config_local['debugging']['feats'][args.feats]['bank']
    bank_path = os.path.join(bank_dir, bank_name)

    if args.mode == 'bank_cleanup':
        bank_spreadsheet = args.bank_spreadsheet if args.bank_spreadsheet else config_global['banks_spreadsheet']
        gsheet_info = {
            'gsheet_name': bank_path.split('/')[-1].split('.')[0],
            'spreadsheet': bank_spreadsheet}
        bank_cleanup_checker(bank_path, gsheet_info)
        sys.exit()

    output_dir = args.output_dir if args.output_dir else os.path.join(
        config_global['debugging'], config_global['paradigm_debugging_dir'], f"camel-morph-{config_local['dialect']}")
    os.makedirs(output_dir, exist_ok=True)
    output_name = args.output_name if args.output_name else config_local['debugging']['feats'][args.feats]['paradigm_debugging']
    output_path = os.path.join(output_dir, output_name)

    service_account = args.service_account if args.service_account else config_global['service_account']
    sa = gspread.service_account(service_account)
    spreadsheet = args.spreadsheet if args.spreadsheet else config_local['debugging']['debugging_spreadsheet']
    sh = sa.open(spreadsheet)

    gsheet = args.gsheet if args.gsheet else config_local['debugging']['feats'][args.feats]['debugging_sheet']
    worksheet = sh.worksheet(title=gsheet)
    annotated_paradigms = pd.DataFrame(worksheet.get_all_records())
    annotated_paradigms.to_csv(output_path)

    if args.new_conj:
        new_conj_path = args.new_conj
    else:
        new_conj_path = os.path.join(config_global['debugging'], config_global['tables_dir'],
                                     f"camel-morph-{config_local['dialect']}")
        new_conj_path = os.path.join(new_conj_path, config_local['debugging']['feats'][args.feats]['conj_tables'])
    
    new_conj_tables = pd.read_csv(new_conj_path, delimiter='\t')
    new_conj_tables = new_conj_tables.replace(nan, '', regex=True)

    outputs = automatic_bank_annotation(bank_path=bank_path,
                                        annotated_paradigms=annotated_paradigms,
                                        new_conj_tables=new_conj_tables,
                                        process_key=args.process_key)

    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output.values(), sep='\t', file=f)
