import argparse
from collections import OrderedDict
import os
import csv
import sys

import gspread
import pandas as pd
from numpy import nan

from camel_tools.morphology.utils import strip_lex

header = ["line", "status", "count", "signature", "lemma", "diac_ar", "diac", "freq",
          "qc", "comments", "pattern", "stem", "bw", "gloss", "cond-s", "cond-t", "pref-cat",
          "stem-cat", "suff-cat", "feats", "debug", "color"]

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
        self._bank = OrderedDict()

        if not download_bank:
            if os.path.exists(bank_path):
                self._read_bank_from_tsv()
        else:
            sa = gspread.service_account(
                "/Users/chriscay/.config/gspread/service_account.json")
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

        lemmas = [entry[1] for entry in self._bank]
        strip = True if not any(['-' in lemma for lemma in lemmas]) else False
        if strip:
            annotated_paradigms['LEMMA'] = annotated_paradigms['LEMMA'].replace(r'-.', '', regex=True)
        
        keys_to_del = []
        for _, row in annotated_paradigms.iterrows():
            key_annot = tuple([row[h] for h in AnnotationBank.HEADER_KEY])
            if row['QC'] != AnnotationBank.UNKOWN:
                if key_annot not in self._bank:
                    for key_bank in self._bank:
                        if key_annot[:-1] == key_bank[:-1]:
                            # Correction case (give precedence to newer annotations)
                            if row['QC'] == self._bank[key_bank]['QC'] == AnnotationBank.GOOD:
                                keys_to_del.append(key_bank)
                    for key_bank in keys_to_del:
                        if key_bank in self._bank:
                            del self._bank[key_bank]
                self._bank[key_annot] = {h: row.get(h, '') for h in [h for h in AnnotationBank.HEADER_INFO if h != 'STATUS']}
                self._bank[key_annot]['STATUS'] = ''

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

def automatic_bank_annotation(bank_path, annotated_paradigms, new_conj_tables):
    bank = AnnotationBank(bank_path, annotated_paradigms)
    lemmas = [entry[1] for entry in bank._bank]
    strip = True if not any(['-' in lemma for lemma in lemmas]) else False

    outputs = []
    for _, row in new_conj_tables.iterrows():
        lemma = row['LEMMA'] if not strip else strip_lex(row['LEMMA'])
        key = (row['SIGNATURE'], lemma, row['DIAC'])
        row = row.to_dict()
        info = bank.get(key)
        if info is not None:
            row['QC'] = info['QC']
            row['COMMENTS'] = info['COMMENTS']
        else:
            for k, info in bank._bank.items():
                if key[:-1] == k[:-1]:
                    row['QC'] = f"({k[-1]})[{info['QC']}]>({row['DIAC']})[{AnnotationBank.UNKOWN}]"
                    break
            else:
                row['QC'] = AnnotationBank.UNKOWN

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
    parser.add_argument("-gsheet", default='',
                        type=str, help="Name of the manually annotated paradigms gsheet, the annotations of which will go in the bank.")
    parser.add_argument("-spreadsheet", default='',
                        type=str, help="Name of the spreadsheet in which that sheet is.")
    parser.add_argument("-use_local",  default='',
                        type=str, help="Use local sheet instead of downloading. Add the name of the conjugation table path to use.")
    parser.add_argument("-new_conj",  default='',
                        type=str, help="Path of the conjugation tables file generated after fixes were made to the specification sheets. This file will be automatically annotated using the information in the bank.")
    parser.add_argument("-bank_dir",  default='conjugation_local/banks',
                        type=str, help="Directory in which the annotation banks are.")
    parser.add_argument("-bank_name",  default='',
                        type=str, help="Name of the annotation bank to use.")
    parser.add_argument("-bank_spreadsheet", default='Paradigm-Banks',
                        type=str, help="Name of the spreadsheet in which that sheet is.")
    parser.add_argument("-output_name",
                        type=str, help="Name of the file to output the automatically bank-annotated paradigm tables to.")
    parser.add_argument("-output_dir", default='conjugation/paradigm_debugging',
                        type=str, help="Path of the directory to output the paradigm tables to.")
    parser.add_argument("-mode", default='debugging', choices=['debugging', 'bank_cleanup'],
                        type=str, help="Path of the directory to output the paradigm tables to.")
    args = parser.parse_args()

    bank_path = os.path.join(args.bank_dir, f"{args.bank_name}.tsv")

    if args.mode == 'bank_cleanup':
        gsheet_info = {
            'gsheet_name': bank_path.split('/')[-1].split('.')[0],
            'spreadsheet': args.bank_spreadsheet}
        bank_cleanup_checker(bank_path, gsheet_info)
        sys.exit()

    def create_dir_if_does_not_exist(directory):
        outer_dir = directory.split('/')[0]
        if not os.path.exists(outer_dir):
            os.mkdir(outer_dir)
            os.mkdir(directory)
        elif not os.path.exists(directory):
            os.mkdir(directory)

    if args.use_local:
        create_dir_if_does_not_exist(args.use_local)
    create_dir_if_does_not_exist(args.bank_dir)
    create_dir_if_does_not_exist(args.output_dir)

    sa = gspread.service_account(
                "/Users/chriscay/.config/gspread/service_account.json")
    sh = sa.open(args.spreadsheet)

    if args.use_local:
        print('Using local sheet...')
        annotated_paradigms = pd.read_csv(args.use_local, delimiter='\t')
        annotated_paradigms = annotated_paradigms.replace(nan, '', regex=True)
    else:
        worksheet = sh.worksheet(title=args.gsheet)
        annotated_paradigms = pd.DataFrame(worksheet.get_all_records())
        annotated_paradigms.to_csv(os.path.join(args.output_dir, args.output_name))

    new_conj_tables = pd.read_csv(args.new_conj, delimiter='\t')
    new_conj_tables = new_conj_tables.replace(nan, '', regex=True)

    outputs = automatic_bank_annotation(bank_path=bank_path,
                                        annotated_paradigms=annotated_paradigms,
                                        new_conj_tables=new_conj_tables)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output.values(), sep='\t', file=f)
