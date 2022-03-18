import argparse
from collections import OrderedDict
import os
import csv

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

    def __init__(self, bank_path, annotated_paradigms=None):
        self._bank_path = bank_path
        self._bank = OrderedDict()

        if os.path.exists(bank_path):
            self._read_bank()

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
        assert set(['SIGNATURE', 'LEMMA', 'DIAC', 'QC', 'COMMENTS']).issubset(annotated_paradigms.columns)
        
        for _, row in annotated_paradigms.iterrows():
            key = (row['SIGNATURE'], row['LEMMA'], row['DIAC'])
            if row['QC'] != AnnotationBank.UNKOWN:
                if key not in self._bank:
                    for k in self._bank:
                        if key[:-1] == k[:-1]:
                            self._bank[k]['QC'] = row['QC']
                self._bank[key] = {'QC': row['QC'],
                                    'COMMENTS': row.get('COMMENTS', '')}

        self._save_bank()

    def _save_bank(self):
        with open(self._bank_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['SIGNATURE', 'LEMMA', 'DIAC', 'QC', 'COMMENTS'])
            writer.writerows([[k for k in key] + [self._bank[key][h] for h in ['QC', 'COMMENTS']]
                                for key in self._bank])

    def _read_bank(self):
        bank = pd.read_csv(self._bank_path, delimiter='\t')
        bank = bank.replace(nan, '', regex=True)
        bank['LEMMA'] = bank['LEMMA'].replace(r'_\d', '', regex=True)
        self._bank = OrderedDict(
            ((row['SIGNATURE'], row['LEMMA'], row['DIAC']), 
                {'QC': row['QC'], 'COMMENTS': row.get('COMMENTS', '')})
            for _, row in bank.iterrows())

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
    parser.add_argument("-output_name",
                        type=str, help="Name of the file to output the automatically bank-annotated paradigm tables to.")
    parser.add_argument("-output_dir", default='conjugation/paradigm_debugging',
                        type=str, help="Path of the directory to output the paradigm tables to.")
    args = parser.parse_args()

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

    outputs = automatic_bank_annotation(bank_path=os.path.join(args.bank_dir, f"{args.bank_name}.tsv"),
                                        annotated_paradigms=annotated_paradigms,
                                        new_conj_tables=new_conj_tables)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output.values(), sep='\t', file=f)
