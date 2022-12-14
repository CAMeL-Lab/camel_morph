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
import os
import csv
import sys
import re
import json

import gspread
import pandas as pd
from numpy import nan

COND_S = 'CLASS'
VARIANT = 'VAR'
ROW_TYPE = 'Entry'
ROW_TYPE_DEF = 'Definition'
ROW_TYPE_SUFF = 'Suffixes'
ROW_TYPE_EX = 'Example'
FREQ = 'Freq'
LEMMA = 'Lemma'
LEMMA_BW = 'Lemma_bw'
STEM = 'Stem'
POS = 'POS'
MS, MD, MP, FS, FD, FP = 'MS', 'MD', 'MP', 'FS', 'FD', 'FP'
OTHER_LEMMAS = 'Other lemmas'
QC = 'QC'
COMMENTS = 'COMMENTS'

header = [COND_S, VARIANT, ROW_TYPE, FREQ, LEMMA, LEMMA_BW, STEM, POS, MS,
          MD, MP, FS, FD, FP, OTHER_LEMMAS, QC, COMMENTS]

class AnnotationBank:
    UNKOWN = 'UNK'
    GOOD = 'OK'
    PROBLEM = 'PROB'
    TAGS = {UNKOWN, GOOD, PROBLEM}
    HEADER_KEY = [COND_S, LEMMA, STEM, POS, MS, MD, MP, FS, FD, FP, ROW_TYPE_SUFF]
    HEADER_INFO = [QC, COMMENTS, ROW_TYPE_DEF]
    HEADER_SHEET_ESSENTIAL = [COND_S, LEMMA, STEM, POS, MS, MD, MP, FS, FD, FP, QC, COMMENTS]
    HEADER = HEADER_KEY + HEADER_INFO

    def __init__(self, bank_path, annotated_paradigms=None, gsheet_info=None, download_bank=False):
        self._bank_path = bank_path
        if gsheet_info is not None:
            self._gsheet_name = gsheet_info['gsheet_name']
            self._spreadsheet = gsheet_info['spreadsheet']
        self._bank, self._unkowns = {}, {}

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
        annotated_paradigms[QC] = annotated_paradigms[QC].replace(
            '', AnnotationBank.UNKOWN, regex=True)
        annotated_paradigms[QC] = annotated_paradigms[QC].str.strip()
        annotated_paradigms[QC] = annotated_paradigms[QC].str.upper()
        assert all([qc in AnnotationBank.TAGS for qc in annotated_paradigms[QC].values.tolist()]), \
            f"Get rid of all tags not belonging to {AnnotationBank.TAGS}"
        assert set(AnnotationBank.HEADER_SHEET_ESSENTIAL).issubset(annotated_paradigms.columns)
        
        definition, suffixes = '', ()
        for i, row in annotated_paradigms.iterrows():
            if row[ROW_TYPE] == ROW_TYPE_DEF:
                definition = row[FREQ]
                assert annotated_paradigms.iloc[i + 1][ROW_TYPE] == ROW_TYPE_SUFF
            elif row[ROW_TYPE] == ROW_TYPE_SUFF:
                suffixes = tuple(row[[MS, MD, MP, FS, FD, FP]].values.tolist())
                assert annotated_paradigms.iloc[i + 1][ROW_TYPE] == ROW_TYPE_EX
            elif row[ROW_TYPE] == ROW_TYPE_EX:
                key_annot = tuple([row[h] for h in AnnotationBank.HEADER_KEY if h != ROW_TYPE_SUFF] + [suffixes])
                if row[QC] != AnnotationBank.UNKOWN:
                    self._bank[key_annot] = {
                        h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO if h != ROW_TYPE_DEF]}
                    self._bank[key_annot][ROW_TYPE_DEF] = definition
                else:
                    self._unkowns[key_annot] = {
                        h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO if h != ROW_TYPE_DEF]}
                    self._unkowns[key_annot][ROW_TYPE_DEF] = definition
            else:
                raise NotImplementedError

        self._save_bank()

    def _save_bank(self):
        with open(self._bank_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(AnnotationBank.HEADER)
            for example_key, info in self._bank.items():
                writer.writerow([k if type(k) is str else '|||'.join(k) for k in example_key] +
                                [info.get(h, '') for h in AnnotationBank.HEADER_INFO])

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
        self._bank = {}
        for _, row in bank.iterrows():
            suffixes = tuple(row[ROW_TYPE_SUFF].split('|||'))
            key_annot = tuple([row[h] for h in AnnotationBank.HEADER_KEY if h != ROW_TYPE_SUFF] + [suffixes])
            self._bank[key_annot] = {h: row.get(h, '') for h in AnnotationBank.HEADER_INFO}

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


def automatic_bank_annotation(bank_path, annotated_paradigms, new_docs_tables):
    bank = AnnotationBank(bank_path, annotated_paradigms)

    outputs = []
    suffixes, suffixes_row = (), {}
    examples_, definitions_ = [], set()
    for i, row in new_docs_tables.iterrows():
        if row[ROW_TYPE] == ROW_TYPE_DEF:
            assert new_docs_tables.iloc[i + 1][ROW_TYPE] == ROW_TYPE_SUFF
        elif row[ROW_TYPE] == ROW_TYPE_SUFF:
            suffixes = tuple(row[[MS, MD, MP, FS, FD, FP]].values.tolist())
            assert new_docs_tables.iloc[i + 1][ROW_TYPE] == ROW_TYPE_EX
            suffixes_row = {h: row.get(h, '') for h in header}
        elif row[ROW_TYPE] == ROW_TYPE_EX:
            key_new = tuple([row[h] for h in AnnotationBank.HEADER_KEY if h != ROW_TYPE_SUFF] + [suffixes])
            row = row.to_dict()
            info = bank.get(key_new)
            if info is not None:
                row[QC] = info[QC]
            else:
                # Equal but the signature suffixes of the groups not exactly equal but a superset of current signature
                break_true = False
                for key_bank, info in bank._bank.items():
                    if key_new[:10] == key_bank[:10] and all(k_new == k_bank for k_new, k_bank in zip(key_new[10], key_bank[10])
                                                             if k_new not in ' -' and k_bank not in ' -'):
                        row[QC] = info[QC]
                        break_true = True
                        break
                # Only the cond_s, lemma, stem and pos are equal
                if not break_true:
                    for key_bank, info in bank._bank.items():
                        if key_new[:4] == key_bank[:4]:
                            row[QC] = f"({' | '.join([f'{k}:{key_bank[i + 4]}' for i, k in enumerate(AnnotationBank.HEADER_KEY[4:-1])])})[{info[QC]}] > [{AnnotationBank.UNKOWN}]"
                            break
                    else:
                        row[QC] = AnnotationBank.UNKOWN
            
            row_unk = bank._unkowns.get(key_new)
            if row_unk is None:
                comment, definition_ = info[COMMENTS], info[ROW_TYPE_DEF]
            else:
                comment, definition_ = row_unk[COMMENTS], row_unk[ROW_TYPE_DEF]
            row[COMMENTS] = comment
            row[ROW_TYPE_DEF] = definition_
            definitions_.add(definition_)

            output_ordered = {}
            for h in header:
                output_ordered[h] = row.get(h, '') if h != FREQ else int(row.get(h, ''))
            examples_.append(output_ordered)

            if i == len(new_docs_tables.index) - 1 or new_docs_tables.iloc[i + 1][ROW_TYPE] == ROW_TYPE_DEF:
                definitions_valid = [definition_ for definition_ in definitions_ if definition_]
                if len(definitions_valid) <= 1:
                    definition_row = {}
                    for j, h in enumerate(header):
                        definition_row[h] = row.get(h, '') if j < 3 else ''
                    definition_row[ROW_TYPE] = ROW_TYPE_DEF
                    definition_row[FREQ] = definitions_valid[0] if len(definitions_valid) else ''
                else:
                    raise NotImplementedError
                
                outputs += [definition_row, suffixes_row] + examples_
                examples_, definitions_ = [], set()
        else:
            raise NotImplementedError

    outputs.insert(0, {i: x for i, x in enumerate(header)})
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-gsheet", default='',
                        type=str, help="Name of the manually annotated paradigms gsheet, the annotations of which will go in the bank.")
    parser.add_argument("-spreadsheet", default='',
                        type=str, help="Name of the spreadsheet in which that sheet is.")
    parser.add_argument("-new_docs",  default='',
                        type=str, help="Path of the documentation tables generated after fixes were made to the specification sheets. This file will be automatically annotated using the information in the bank.")
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
        config_global['debugging'], config_global['docs_banks_dir'], f"camel-morph-{config_local['dialect']}")
    os.makedirs(bank_dir, exist_ok=True)
    bank_name = args.bank_name if args.bank_name else config_local['docs_debugging']['bank']
    bank_path = os.path.join(bank_dir, bank_name)

    output_dir = args.output_dir if args.output_dir else os.path.join(
        config_global['debugging'], config_global['docs_debugging_dir'], f"camel-morph-{config_local['dialect']}")
    os.makedirs(output_dir, exist_ok=True)
    output_name = args.output_name if args.output_name else config_local['docs_debugging']['output_name']
    output_path = os.path.join(output_dir, output_name)

    service_account = args.service_account if args.service_account else config_global['service_account']
    sa = gspread.service_account(service_account)
    spreadsheet = args.spreadsheet if args.spreadsheet else config_local['docs_debugging']['debugging_spreadsheet']
    sh = sa.open(spreadsheet)

    gsheet = args.gsheet if args.gsheet else config_local['docs_debugging']['debugging_sheet']
    worksheet = sh.worksheet(title=gsheet)
    annotated_paradigms = pd.DataFrame(worksheet.get_all_records())
    annotated_paradigms.to_csv(output_path)

    if args.new_docs:
        new_docs_path = args.new_docs
    else:
        new_docs_dir = os.path.join(config_global['debugging'], config_global['docs_tables_dir'],
                                     f"camel-morph-{config_local['dialect']}")
        os.makedirs(new_docs_dir, exist_ok=True)
        new_docs_path = os.path.join(new_docs_dir, config_local['docs_debugging']['docs_tables'])
    
    new_docs_tables = pd.read_csv(new_docs_path, delimiter='\t')
    new_docs_tables = new_docs_tables.replace(nan, '', regex=True)

    outputs = automatic_bank_annotation(bank_path=bank_path,
                                        annotated_paradigms=annotated_paradigms,
                                        new_docs_tables=new_docs_tables)

    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output.values(), sep='\t', file=f)
