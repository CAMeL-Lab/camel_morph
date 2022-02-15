import re
import argparse
import json
import os
from tqdm import tqdm

import pandas as pd
from numpy import nan
from camel_tools.morphology.utils import strip_lex

from utils import assign_pattern

errors = {}

def generate_passive(LEXICON, patterns_path):
    patterns = pd.read_csv(patterns_path)
    patterns = patterns.replace(nan, '', regex=True)
    patterns_map = {}
    for _, row in patterns.iterrows():
        info = dict(regex_match=row['REGEX-match'],
                    regex_sub=row['REGEX-sub'],
                    cond_map=row['COND-map'])
        patterns_map[(row['Pattern'], row['COND-ST'])] = info

    cond_t_pattern = re.compile(r'([cv]-suff)')
    cond_s_pattern = re.compile(r'(?:.* \+ )?(.*)')
    one_or_more_pluses = re.compile(r' +')

    def assign_pattern_wrapper(row):
        pattern, _, _, error = assign_pattern(strip_lex(row['LEMMA']))
        if error:
            errors.setdefault(error, []).append(row['LEMMA'])
        return pattern if pattern else nan

    def get_info(row):
        info = patterns_map.get((row['PATTERN-DEF'], row['COND-TS']))
        if info != None:
            info['regex_sub'] = info['regex_sub'].replace('$', '\\')
            return info
        else:
            return nan

    def get_cond_t_pass(row):
        match = cond_t_pattern.search(row['COND-TS'])
        return match.group(1) if match else ''

    def get_cond_s_pass(row):
        match = cond_s_pattern.search(row['COND-TS'])
        return match.group(1) if match else ''
    
    LEXICON_PASS = LEXICON.copy()
    LEXICON_PASS['PATTERN-DEF'] = LEXICON_PASS.apply(assign_pattern_wrapper, axis=1)
    LEXICON_PASS = LEXICON_PASS[LEXICON_PASS['PATTERN-DEF'].notna()]
    LEXICON_PASS['COND-T'] = LEXICON_PASS['COND-T'].str.strip()
    LEXICON_PASS['COND-T'] = LEXICON_PASS.apply(
        lambda row: one_or_more_pluses.sub(' ', row['COND-T']), axis=1)
    LEXICON_PASS['COND-S-NO-TRANS'] = LEXICON_PASS.apply(
        lambda row: re.sub(r'\btrans|intrans', '', row['COND-S']), axis=1)
    LEXICON_PASS['COND-S-NO-TRANS'] = LEXICON_PASS.apply(
        lambda row: one_or_more_pluses.sub(' ', row['COND-S-NO-TRANS']), axis=1)
    LEXICON_PASS['COND-S-NO-TRANS'] = LEXICON_PASS['COND-S-NO-TRANS'].str.strip()
    LEXICON_PASS['COND-TS'] = LEXICON_PASS['COND-T']
    LEXICON_PASS['COND-TS'] = LEXICON_PASS.apply(
        lambda row: row['COND-TS'] + f"{' + ' if row['COND-T'] else ''}{row['COND-S-NO-TRANS']}", axis=1)
    LEXICON_PASS['PATTERN-MAP'] = LEXICON_PASS.apply(get_info, axis=1)
    LEXICON_PASS = LEXICON_PASS[LEXICON_PASS['PATTERN-MAP'].notna()]
    LEXICON_PASS['FORM'] = LEXICON_PASS.apply(
        lambda row: re.sub(row['PATTERN-MAP']['regex_match'],
                           row['PATTERN-MAP']['regex_sub'],
                           row['FORM']), axis=1)
    LEXICON_PASS['COND-S'] = LEXICON_PASS['COND-S'].str.strip()
    # All passive forms should be intransitive
    LEXICON_PASS['COND-TS'] = LEXICON_PASS.apply(
        lambda row: row['PATTERN-MAP']['cond_map'] + ' intrans', axis=1)
    LEXICON_PASS['COND-T'] = LEXICON_PASS.apply(get_cond_t_pass, axis=1)
    LEXICON_PASS['COND-S'] = LEXICON_PASS.apply(get_cond_s_pass, axis=1)
    LEXICON_PASS['COND-S'] = LEXICON_PASS['COND-S'].str.strip()
    
    LEXICON_PASS['BW'] = LEXICON_PASS.apply(
        lambda row: re.sub(r'(.V)', r'\1_PASS', row['BW']), axis=1)
    LEXICON_PASS['FEAT'] = LEXICON_PASS.apply(
        lambda row: re.sub(r'vox:a', r'vox:p', row['FEAT']), axis=1)

    LEXICON_PASS.drop('PATTERN-DEF', axis=1, inplace=True)
    LEXICON_PASS.drop('COND-S-NO-TRANS', axis=1, inplace=True)
    LEXICON_PASS.drop('COND-TS', axis=1, inplace=True)
    LEXICON_PASS.drop('PATTERN-MAP', axis=1, inplace=True)

    return LEXICON_PASS
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", required=True,
                        type=str, help="Path of active aspect lexicon to generate the passive from.")
    parser.add_argument("-config_file", required=True,
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-patterns", required=True,
                        type=str, help="Path of file which contains the passive pattern maps.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)
    data_dir = config['global']['data-dir']

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
