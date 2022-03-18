import os
import argparse
import re
import sys

import gspread
import gspread_formatting
import pandas as pd
from numpy import nan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default='conjugation/tables',
                        type=str, help="Directory in which local sheet is contained (in CSV format).")
    parser.add_argument("-file_name", required=True,
                        type=str, help="Name of the CSV file to write to the cloud.")
    parser.add_argument("-spreadsheet_name", required=True,
                        type=str, help="Name of the spreadsheet to write the CSV file to (and to format).")
    parser.add_argument("-gsheet_name", required=True,
                        type=str, help="Name of the gsheet to write the CSV file to (and to format).")
    parser.add_argument("-formatting", default='', choices=['conj_tables', 'bank'],
                        type=str, help="How to format the sheet.")
    parser.add_argument("-mode", default='prompt', choices=['backup', 'overwrite', 'prompt'],
                        type=str, help="Mode that decides what to do if sheet which is being uploaded already exists.")
    args = parser.parse_args()

    sheet_csv = pd.read_csv(os.path.join(args.dir, args.file_name), sep='\t')
    sheet_csv = sheet_csv.replace(nan, '', regex=True)
    sa = gspread.service_account(
        "/Users/chriscay/.config/gspread/service_account.json")
    sh = sa.open(args.spreadsheet_name)
    try:
        worksheet = sh.add_worksheet(title=args.gsheet_name, rows="100", cols="20")
    except gspread.exceptions.APIError as e:
        if re.search(r'name.*already exists', e.args[0]['message']):
            if args.mode == 'prompt':
                response = input('A sheet with this name already exists. Do you still want to overwrite it? [y/n]: ')
            elif args.mode == 'overwrite':
                response = 'y'
            elif args.mode == 'backup':
                response = 'b'
                gsheet_name_bu = args.gsheet_name + '-Backup'

            if response == 'y':
                worksheet = sh.worksheet(title=args.gsheet_name)
                worksheet.clear()
                print('Sheet content will be overwritten.')
            elif response == 'b':
                print(f'Previous sheet will be backed up under the name {gsheet_name_bu} (effectively overwriting the old backup sheet).')
                if gsheet_name_bu in [sheet.title for sheet in sh.worksheets()]:
                    worksheet_bu = sh.worksheet(title=gsheet_name_bu)
                    sh.del_worksheet(worksheet_bu)
                worksheet = sh.worksheet(title=args.gsheet_name)
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

    if args.formatting == 'conj_tables':
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

        worksheet.freeze(rows=1, cols=10)

    else:
        worksheet.freeze(rows=1)
