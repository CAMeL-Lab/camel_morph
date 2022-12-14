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


import os
import argparse
import re
import sys
import json

import gspread
import gspread_formatting
import pandas as pd
from numpy import nan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", default='tables_dir', choices=['banks_dir', 'paradigm_debugging_dir', 'tables_dir', 'docs_banks_dir', 'docs_debugging_dir', 'docs_tables_dir'],
                        type=str, help="Directory in which local sheet is contained (in CSV format).")
    parser.add_argument("-dir", default='',
                        type=str, help="Directory in which local sheet is contained (in CSV format).")
    parser.add_argument("-config_file", default='configs/config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-feats", default='',
                        type=str, help="Features to generate the conjugation tables for.")
    parser.add_argument("-file_name", default='',
                        type=str, help="Name of the CSV file to write to the cloud.")
    parser.add_argument("-spreadsheet_name", default='',
                        type=str, help="Name of the spreadsheet to write the CSV file to (and to format).")
    parser.add_argument("-gsheet_name", default='',
                        type=str, help="Name of the gsheet to write the CSV file to (and to format).")
    parser.add_argument("-mode", default='prompt', choices=['backup', 'overwrite', 'prompt'],
                        type=str, help="Mode that decides what to do if sheet which is being uploaded already exists.")
    parser.add_argument("-insert_index", default=False, action='store_true',
                        help="Insert an index column.")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = json.load(f)
    if args.config_name:
        config_local = config['local'][args.config_name]
    config_global = config['global']

    if 'debugging' in config_local:
        spreadsheet_name = args.spreadsheet_name if args.spreadsheet_name else config_local['debugging']['debugging_spreadsheet']
        gsheet_name = args.gsheet_name if args.gsheet_name else config_local['debugging']['feats'][args.feats]['debugging_sheet']
    elif 'docs_debugging' in config_local:
        spreadsheet_name = args.spreadsheet_name if args.spreadsheet_name else config_local['docs_debugging']['debugging_spreadsheet']
        gsheet_name = args.gsheet_name if args.gsheet_name else config_local['docs_debugging']['debugging_sheet']

    if args.file_name:
        file_name = args.file_name
    elif args.input_dir == 'banks_dir':
        file_name = config_local['debugging']['feats'][args.feats]['bank']
        gsheet_name = args.gsheet_name if args.gsheet_name else config_local['debugging']['feats'][args.feats]['bank'].split('.')[0]
        spreadsheet_name = args.spreadsheet_name if args.spreadsheet_name else config_global['banks_spreadsheet']
    elif args.input_dir == 'docs_banks_dir':
        file_name = config_local['docs_debugging']['bank']
        gsheet_name = args.gsheet_name if args.gsheet_name else config_local['docs_debugging']['bank'].split('.')[0]
        spreadsheet_name = args.spreadsheet_name if args.spreadsheet_name else config_global['banks_spreadsheet']
    elif args.input_dir == 'paradigm_debugging_dir':
        file_name = config_local['debugging']['feats'][args.feats]['paradigm_debugging']
    elif args.input_dir == 'docs_debugging_dir':
        file_name = config_local['docs_debugging']['output_name']
    elif args.input_dir == 'tables_dir':
        file_name = config_local['debugging']['feats'][args.feats]['conj_tables']
    elif args.input_dir == 'docs_tables_dir':
        file_name = config_local['docs_debugging']['docs_tables']
    
    input_dir = args.dir if args.dir \
        else os.path.join(config_global['debugging'], config_global[args.input_dir], f"camel-morph-{config_local['dialect']}")
    sheet_csv = pd.read_csv(os.path.join(input_dir, file_name), sep='\t')
    sheet_csv = sheet_csv.replace(nan, '', regex=True)
    if args.insert_index:
        sheet_csv.insert(0, 'Index', range(len(sheet_csv)))
    
    service_account = args.service_account if args.service_account else config_global['service_account']
    sa = gspread.service_account(service_account)
    sh = sa.open(spreadsheet_name)
    
    try:
        worksheet = sh.add_worksheet(title=gsheet_name, rows="100", cols="20")
    except gspread.exceptions.APIError as e:
        if re.search(r'name.*already exists', e.args[0]['message']):
            if args.mode == 'prompt':
                response = input('A sheet with this name already exists. Do you still want to overwrite it? [y/n]: ')
            elif args.mode == 'overwrite':
                response = 'y'
            elif args.mode == 'backup':
                response = 'b'
                gsheet_name_bu = gsheet_name + '-Backup'

            if response == 'y':
                worksheet = sh.worksheet(title=gsheet_name)
                worksheet.clear()
                print('Sheet content will be overwritten.')
            elif response == 'b':
                print(f'Previous sheet will be backed up under the name {gsheet_name_bu} (effectively overwriting the old backup sheet).')
                if gsheet_name_bu in [sheet.title for sheet in sh.worksheets()]:
                    worksheet_bu = sh.worksheet(title=gsheet_name_bu)
                    sh.del_worksheet(worksheet_bu)
                worksheet = sh.worksheet(title=gsheet_name)
                sh.duplicate_sheet(source_sheet_id=worksheet.id,
                                   insert_sheet_index=worksheet.index + 1,
                                   new_sheet_name=gsheet_name_bu)
                worksheet.clear()
            elif response == 'n':
                print('Process aborted.')
                sys.exit()
        else:
            raise NotImplementedError
    
    worksheet.update(
        [sheet_csv.columns.values.tolist()] + sheet_csv.values.tolist())

    if args.input_dir in ['tables_dir', 'paradigm_debugging_dir']:
        rule1 = gspread_formatting.ConditionalFormatRule(
            ranges=[gspread_formatting.GridRange.from_a1_range('A:S', worksheet)],
            booleanRule=gspread_formatting.BooleanRule(
                condition=gspread_formatting.BooleanCondition('CUSTOM_FORMULA', ['=$U1=0']),
                format=gspread_formatting.CellFormat(
                    backgroundColor=gspread_formatting.Color(217/255, 234/255, 211/255))
            )
        )
        rule2 = gspread_formatting.ConditionalFormatRule(
            ranges=[gspread_formatting.GridRange.from_a1_range('A:S', worksheet)],
            booleanRule=gspread_formatting.BooleanRule(
                condition=gspread_formatting.BooleanCondition('CUSTOM_FORMULA', ['=$U1=1']),
                format=gspread_formatting.CellFormat(
                    backgroundColor=gspread_formatting.Color(255/255, 242/255, 204/255))
            )
        )
        rules = gspread_formatting.get_conditional_format_rules(worksheet)
        rules.clear()
        rules.append(rule1)
        rules.append(rule2)
        rules.save()

        worksheet.freeze(rows=1, cols=11)

    else:
        worksheet.freeze(rows=1)
