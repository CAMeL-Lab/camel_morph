import os
import argparse
import json

import gspread
import pandas as pd

sa = gspread.service_account(
    "/Users/chriscay/.config/gspread/service_account.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lex", nargs='+', action='append',
                        type=str, help="Name of the lexicon gsheets to download followed by the desired individual sheets.")
    parser.add_argument("-specs", nargs='+', action='append',
                        type=str, help="Name of the specs spreadsheet to download followed by the desired individual sheets.")
    parser.add_argument("-save_dir", default="data",
                        type=str, help="Path of the directory to save the csv files to.")
    parser.add_argument("-config_file",
                        type=str, help="Config file specifying which sheets to use.")
    parser.add_argument("-config_name",
                        type=str, help="Name of the configuration to load from the config file.")
    args = parser.parse_args()

    
    if args.lex != None or args.specs != None:
        lex = {'spreadsheets': [ss[0] for ss in args.lex],
               'sheets': [ss[1:] for ss in args.lex]}
        specs = {'spreadsheet': [ss[0] for ss in args.specs],
                 'sheets': [ss[1:] for ss in args.specs]}
    else:
        with open(args.config_file) as f:
            config = json.load(f)
        specs = {
            'spreadsheets': [config['global']['specs']['spreadsheet']],
            'sheets': [
                config['global']['specs']['sheets'] + list(config['local'][args.config_name]['specs'].values())]
        }
        lex = {
            'spreadsheets': [config['local'][args.config_name]['lexicon']['spreadsheet']],
            'sheets': [config['local'][args.config_name]['lexicon']['sheets']]
        }

    for spreadsheets in [lex, specs]:
        for i, spreadsheet_name in enumerate(spreadsheets['spreadsheets']):
            spreadsheet = sa.open(spreadsheet_name)
            for sheet_name in spreadsheets['sheets'][i]:
                if 'PASS' in sheet_name:
                    continue
                sheet = spreadsheet.worksheet(sheet_name)
                sheet = pd.DataFrame(sheet.get_all_records())
                sheet.to_csv(os.path.join(args.save_dir, f"{sheet_name}.csv"))
