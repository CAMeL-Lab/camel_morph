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


import re
import argparse
import json
import os

import pandas as pd
from numpy import nan

from camel_morph.utils.utils import assign_pattern

errors, missing = {}, {}

def generate_passive(LEXICON, patterns_path):
    from camel_tools.morphology.utils import strip_lex
    passive_patterns = pd.read_csv(patterns_path)
    passive_patterns = passive_patterns.replace(nan, '', regex=True)
    passive_patterns['COND-S-ESSENTIAL-Act'] = passive_patterns.apply(
        lambda row: re.sub(r' ?(gem|hamzated|hollow|defective) ?', '', row['COND-S-Act']), axis=1)
    passive_patterns['COND-S-ESSENTIAL-Pass'] = passive_patterns.apply(
        lambda row: re.sub(r' ?(gem|hamzated|hollow|defective) ?', '', row['COND-S-Pass']), axis=1)
    passive_patterns_map = {}
    for _, row in passive_patterns.iterrows():
        info = dict(regex_match=row['REGEX-match'],
                    regex_sub=row['REGEX-sub'],
                    cond_t_pass=row['COND-T-Pass'],
                    cond_s_pass=row['COND-S-ESSENTIAL-Pass'])
        key = (row['Pattern'], row['COND-T-Act'], row['COND-S-ESSENTIAL-Act'])
        passive_patterns_map.setdefault(key, []).append(info)

    soundness_pattern = re.compile(r'(hollow|defective|gem|hamzated)')

    def assign_pattern_wrapper(row):
        result = assign_pattern(strip_lex(row['LEMMA']), root=row['ROOT'].split('.'))
        pattern = result['pattern_conc']
        error = result['error']
        if error:
            errors.setdefault(error, []).append(row['LEMMA'])
        return pattern if pattern else nan

    def get_info(row):
        infos = passive_patterns_map.get((row['PATTERN-DEF'], row['COND-T'], row['COND-S-ESSENTIAL']))
        if infos != None:
            if len(infos) == 1:
                info = infos[0]
                info['regex_sub'] = info['regex_sub'].replace('$', '\\')
                return info
            else:
                for info in infos:
                    if re.match(info['regex_match'], row['FORM']):
                        info['regex_sub'] = info['regex_sub'].replace('$', '\\')
                        return info
        else:
            missing.setdefault((row['PATTERN-DEF'], row['COND-T'], row['COND-S-ESSENTIAL']), []).append(row.to_dict())
            return nan

    def get_pattern(row):
        return re.sub(row['PATTERN-MAP']['regex_match'],
                      row['PATTERN-MAP']['regex_sub'],
                      row['PATTERN'])

    def get_soundness(row):
        match = soundness_pattern.search(row['COND-S'])
        return match.group(1) if match else ''

    LEXICON = LEXICON[~LEXICON['COND-S'].str.contains("Frozen")]

    LEXICON_PASS = LEXICON.copy()
    LEXICON_PASS['PATTERN-DEF'] = LEXICON_PASS.apply(assign_pattern_wrapper, axis=1)
    LEXICON_PASS = LEXICON_PASS[LEXICON_PASS['PATTERN-DEF'].notna()]
    LEXICON_PASS['COND-T'] = LEXICON_PASS['COND-T'].str.strip()
    LEXICON_PASS['COND-S-ESSENTIAL'] = LEXICON_PASS.apply(
        lambda row: re.sub(r'ditrans|trans|intrans|gem|hamzated|hollow|defective', '', row['COND-S']), axis=1)
    LEXICON_PASS['COND-S-ESSENTIAL'] = LEXICON_PASS['COND-S-ESSENTIAL'].str.strip()
    LEXICON_PASS['PATTERN-MAP'] = LEXICON_PASS.apply(get_info, axis=1)
    LEXICON_PASS = LEXICON_PASS[LEXICON_PASS['PATTERN-MAP'].notna()]
    LEXICON_PASS['FORM'] = LEXICON_PASS.apply(
        lambda row: re.sub(row['PATTERN-MAP']['regex_match'],
                           row['PATTERN-MAP']['regex_sub'],
                           row['FORM']), axis=1)
    if 'PATTERN' in LEXICON_PASS.columns:
        LEXICON_PASS['PATTERN'] = LEXICON_PASS.apply(get_pattern, axis=1)
    LEXICON_PASS['SOUND'] = LEXICON_PASS.apply(get_soundness, axis=1)
    # All passive forms should be intransitive
    LEXICON_PASS['COND-T'] = LEXICON_PASS.apply(
        lambda row: row['PATTERN-MAP']['cond_t_pass'], axis=1)
    LEXICON_PASS['COND-S-ESSENTIAL-Pass'] = LEXICON_PASS.apply(
        lambda row: row['PATTERN-MAP']['cond_s_pass'], axis=1)
    LEXICON_PASS['COND-S-ESSENTIAL-Pass'] = LEXICON_PASS['COND-S-ESSENTIAL-Pass'].str.strip()
    LEXICON_PASS['COND-S'] = LEXICON_PASS.apply(
        lambda row: row['COND-S-ESSENTIAL-Pass'] +
                    (' ' if row['COND-S-ESSENTIAL-Pass'] else '') +
                    row['SOUND'] +
                    (' ' if row['SOUND'] else '') + "intrans", axis=1)
    
    LEXICON_PASS['BW'] = LEXICON_PASS.apply(
        lambda row: re.sub(r'(.V)', r'\1_PASS', row['BW']), axis=1)
    LEXICON_PASS['FEAT'] = LEXICON_PASS.apply(
        lambda row: re.sub(r'vox:a', r'vox:p', row['FEAT']), axis=1)

    LEXICON_PASS.drop('PATTERN-DEF', axis=1, inplace=True)
    LEXICON_PASS.drop('COND-S-ESSENTIAL', axis=1, inplace=True)
    LEXICON_PASS.drop('PATTERN-MAP', axis=1, inplace=True)
    LEXICON_PASS.drop('SOUND', axis=1, inplace=True)

    return LEXICON_PASS
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", required=True,
                        type=str, help="Path of active aspect lexicon to generate the passive from.")
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-patterns", required=True,
                        type=str, help="Path of file which contains the passive pattern maps.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)
    data_dir = config['global']['data_dir']

    LEXICON = pd.read_csv(os.path.join(data_dir, args.input_file))
    # Replace spaces in BW and GLOSS with '#'; skip commented rows and empty lines
    LEXICON = LEXICON[LEXICON.DEFINE == 'LEXICON']
    LEXICON = LEXICON.replace(nan, '', regex=True)
    LEXICON['GLOSS'] = LEXICON['GLOSS'].replace('\s+', '#', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' +', ' ', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' $', '', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' +', ' ', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' $', '', regex=True)

    LEXICON_PASS = generate_passive(LEXICON, os.path.join(data_dir,args.patterns))
    
    output_name = re.sub(r'(.*).csv', r'\1-PASS.csv', args.input_file)
    output_path = os.path.join(data_dir, output_name)
    LEXICON_PASS.to_csv(output_path)
