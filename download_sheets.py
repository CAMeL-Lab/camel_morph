import os
import argparse
import json
from re import L
import time

import gspread
import pandas as pd

def download_sheets(lex, specs, save_dir, config, config_name, service_account):
    if type(service_account) is str:
        sa = gspread.service_account(service_account)
    else:
        sa = service_account

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if lex or specs:
        lex = {'spreadsheets': [ss[0] for ss in lex],
               'sheets': [ss[1:] for ss in lex]}
        specs = {'spreadsheets': [ss[0] for ss in specs],
                 'sheets': [ss[1:] for ss in specs]}
    else:
        specs_local = []
        for sheets in config['local'][config_name]['specs'].values():
            if type(sheets) is str:
                specs_local.append(sheets)
            elif type(sheets) is dict:
                for sheet in sheets.values():
                    specs_local.append(sheet)
                    
        specs = {
            'spreadsheets': [config['global']['specs']['spreadsheet']],
            'sheets': [
                config['global']['specs']['sheets'] + specs_local]
        }
        lex = {
            'spreadsheets': [config['local'][config_name]['lexicon']['spreadsheet']],
            'sheets': [config['local'][config_name]['lexicon']['sheets']]
        }

    for spreadsheets in [lex, specs]:
        for i, spreadsheet_name in enumerate(spreadsheets['spreadsheets']):
            spreadsheet = sa.open(spreadsheet_name)
            for sheet_name in spreadsheets['sheets'][i]:
                if 'PASS' in sheet_name:
                    continue
                not_downloaded = True
                while not_downloaded:
                    try:
                        sheet = spreadsheet.worksheet(sheet_name)
                        not_downloaded = False
                    except gspread.exceptions.APIError as e:
                        if 'Quota exceeded' in e.args[0]['message']:
                            print('Quota exceeded, waiting for 15 seconds and then retrying...')
                            time.sleep(15)
                        else:
                            raise NotImplementedError
                sheet = pd.DataFrame(sheet.get_all_records())
                sheet.to_csv(os.path.join(save_dir, f"{sheet_name}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lex", nargs='+', action='append', default=[],
                        type=str, help="Name of the lexicon gsheets to download followed by the desired individual sheets.")
    parser.add_argument("-specs", nargs='+', action='append', default=[],
                        type=str, help="Name of the specs spreadsheet to download followed by the desired individual sheets.")
    parser.add_argument("-save_dir", default='',
                        type=str, help="Path of the directory to save the csv files to.")
    parser.add_argument("-config_file", default='config.json',
                        type=str, help="Config file specifying which sheets to use.")
    parser.add_argument("-config_name", default='',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)
    config_local = config['local'][args.config_name]
    config_global = config['global']

    save_dir = args.save_dir if args.save_dir else config_global['data-dir']
    service_account = args.service_account if args.service_account else config_global['service_account']

    download_sheets(args.lex, args.specs, save_dir,
                    config, args.config_name, service_account)
