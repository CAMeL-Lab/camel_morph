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
import sys
import time
import gspread
import pandas as pd
from collections import OrderedDict

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) -
                                    1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)
from camel_morph.utils.utils import Config


def download_sheets(specs=None, save_dir=None, config:Config=None, service_account=None):
    if service_account is None:
        sa = gspread.service_account(config.service_account)
    elif type(service_account) is str:
        sa = gspread.service_account(service_account)
    else:
        sa = service_account

    specs_ = {}
    if specs is not None:
        for s in specs:
            assert len(s) >= 2
            spreadsheet = s[0]
            specs_.setdefault(spreadsheet, OrderedDict()).update(
                {sheet: None for sheet in s[1:]})

    for spreadsheet, sheets in config.get_spreadsheet2sheets().items():
        specs_.setdefault(spreadsheet, OrderedDict()).update(
            {sheet: None for sheet in sheets})
    
    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for spreadsheet_name, sheets in specs_.items():
        spreadsheet = sa.open(spreadsheet_name)
        for sheet_name in sheets:
            print(f'Downloading {spreadsheet_name} -> {sheet_name} ...')
            not_downloaded = True
            while not_downloaded:
                try:
                    sheet = spreadsheet.worksheet(sheet_name)
                    not_downloaded = False
                except gspread.exceptions.APIError as e:
                    if 'Quota exceeded' in e.args[0]['message']:
                        print('Quota exceeded, waiting for 30 seconds and then retrying...')
                        time.sleep(30)
                    else:
                        raise NotImplementedError
            sheet = pd.DataFrame(sheet.get_all_records())
            if save_dir is not None:
                output_dir = save_dir
            else:
                output_dir = config.get_data_dir_path()
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{sheet_name}.csv')
            sheet.to_csv(output_path)
    
    print(f'Files saved to: {output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-specs", nargs='+', action='append', default=[],
                        type=str, help="Name of the specs spreadsheet to download followed by the desired individual sheets.")
    parser.add_argument("-save_dir", default='',
                        type=str, help="Path of the directory to save the csv files to.")
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args()

    print(args.specs)
    sys.exit()
    config = None
    if args.config_file and args.config_name:
        config = Config(args.config_file, args.config_name)
        assert not args.save_dir

    download_sheets(args.specs, args.save_dir, config, args.service_account)
