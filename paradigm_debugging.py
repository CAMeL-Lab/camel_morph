import argparse
from collections import OrderedDict
import os
import csv

import gspread
import pandas as pd
from numpy import nan

header = ["line", "status", "count", "signature", "lemma", "diac_ar", "diac", "freq",
          "qc", "pattern", "stem", "bw", "gloss", "cond-s", "cond-t", "pref-cat",
          "stem-cat", "suff-cat", "feats", "debug", "color"]

class AnnotationBank:
    UNKOWN = 'UNK'
    GOOD = 'OK'
    CORRECTION = 'CORRECT'
    PROBLEM = 'PROB'
    TAGS = {UNKOWN, GOOD, CORRECTION, PROBLEM}

    def __init__(self, bank_path, annotated_paradigms):
        self._bank_path = bank_path
        self._bank = OrderedDict()

        if os.path.exists(bank_path):
            self._read_bank()

        self._update_bank(annotated_paradigms)

    def __contains__(self, key):
        return key in self._bank

    def __setitem__(self, key, item):
        self._bank[key] = item

    def __getitem__(self, key):
        return self._bank[key]

    def _update_bank(self, annotated_paradigms):
        annotated_paradigms['QC'] = annotated_paradigms['QC'].replace(
            '', AnnotationBank.UNKOWN, regex=True)
        assert all([qc in AnnotationBank.TAGS for qc in annotated_paradigms['QC'].values.tolist()]), \
            f"Get rid of all tags not belonging to {AnnotationBank.TAGS}"
        
        for _, row in annotated_paradigms.iterrows():
            key = (row['SIGNATURE'], row['LEMMA'], row['DIAC'], row['QC'])
            if key not in self._bank and row['QC'] != AnnotationBank.UNKOWN:
                self._bank[key] = 1
            elif row['QC'] == AnnotationBank.CORRECTION:
                for current_key in self._bank:
                    if current_key[:-1] == key[:-1]:
                        del self._bank[current_key]
                        self._bank[key] = 1
        self._save_bank()

    def _save_bank(self):
        with open(self._bank_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([[k for k in key] for key in self._bank])

    def _read_bank(self):
        with open(self._bank_path) as f:
            csvreader = csv.reader(f)
            self._bank = OrderedDict((tuple(row), 1) for row in csvreader)

def automatic_bank_annotation(bank_path, annotated_paradigms):
    bank = AnnotationBank(bank_path, annotated_paradigms)

    outputs = []
    for _, row in annotated_paradigms.iterrows():
        key = (row['SIGNATURE'], row['LEMMA'], row['DIAC'], row['QC'])
        row = row.to_dict()
        if key in bank:
            row['QC'] = bank[key]
        #TODO: add elifs here
        else:
            row['QC'] = AnnotationBank.UNKOWN

        output_ordered = OrderedDict()
        for h in header:
            output_ordered[h.upper()] = row.get(h.upper(), '')
        
        outputs.append(output_ordered)

    outputs.insert(0, OrderedDict((i, x) for i, x in enumerate(map(str.upper, header))))
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gsheet", default='',
                        type=str, help="Name of the manually annotated paradigms gsheet.")
    parser.add_argument("-spreadsheet", default='',
                        type=str, help="Name of the spreadsheet in which that sheet is.")
    parser.add_argument("-use_local",  default='',
                        type=str, help="Use local sheet instead of downloading (for testing). Add the name of the directory in which the sheet is as an argument.")
    parser.add_argument("-bank_dir",  default='conjugation_local/banks',
                        type=str, help="Directory in which the annotation banks are.")
    parser.add_argument("-bank_name",  default='conjugation_local/paradigm_debugging',
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
        annotated_paradigms = pd.read_csv(os.path.join(args.use_local, args.gsheet))
        annotated_paradigms = annotated_paradigms.replace(nan, '', regex=True)
    else:
        worksheet = sh.worksheet(title=args.gsheet)
        annotated_paradigms = pd.DataFrame(worksheet.get_all_records())
        annotated_paradigms.to_csv(os.path.join(args.output_dir, args.output_name))

    outputs = automatic_bank_annotation(bank_path=os.path.join(args.bank_dir, f"{args.bank_name}.csv"),
                                        annotated_paradigms=annotated_paradigms)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        for output in outputs:
            print(*output.values(), sep='\t', file=f)
