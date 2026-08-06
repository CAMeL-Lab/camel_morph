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


import argparse
import re
from typing import Dict, List, Tuple

import pandas as pd
from camel_tools.morphology.utils import strip_lex
from numpy import nan

from ..almor_schema import CATIB6_PASSIVE_VERB
from .utils import assign_pattern


PatternKey = Tuple[str, str, str]
PatternInfo = Dict[str, str]
PatternMap = Dict[PatternKey, List[PatternInfo]]

_SOUNDNESS_TERMS_RE = re.compile(r'(hollow|defective|gem|hamzated)')
_PATTERN_COND_S_CLEAN_RE = re.compile(
    r' ?(gem|hamzated|hollow|defective) ?'
)
_PASSIVE_COND_S_CLEAN_RE = re.compile(
    r'ditrans|trans|intrans|gem|hamzated|hollow|defective'
)


def _load_pattern_map(patterns_path: str) -> PatternMap:
    patterns = pd.read_csv(patterns_path, na_filter=False)
    patterns['COND-S-ESSENTIAL'] = patterns['COND-S'].str.replace(
        _PATTERN_COND_S_CLEAN_RE, '', regex=True
    )
    patterns['COND-S-ESSENTIAL-PASS'] = patterns['COND-S-PASS'].str.replace(
        _PATTERN_COND_S_CLEAN_RE, '', regex=True
    )

    pattern_map: PatternMap = {}
    for _, row in patterns.iterrows():
        key = (row['PATTERN'], row['COND-T'], row['COND-S-ESSENTIAL'])
        info = {
            'regex_match': row['MATCH'],
            'regex_sub': row['SUB'].replace('$', '\\'),
            'cond_t_pass': row['COND-T-PASS'],
            'cond_s_pass': row['COND-S-ESSENTIAL-PASS'],
        }
        pattern_map.setdefault(key, []).append(info)
    return pattern_map


def _assign_pattern_definition(row: pd.Series):
    result = assign_pattern(strip_lex(row['LEMMA']), root=row['ROOT'].split('.'))
    return result['pattern_conc'] or nan


def _select_pattern_info(row: pd.Series, pattern_map: PatternMap):
    key = (row['PATTERN-DEF'], row['COND-T'], row['COND-S-ESSENTIAL'])
    candidates = pattern_map.get(key)
    if not candidates:
        return nan
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if re.match(candidate['regex_match'], row['FORM']):
            return candidate
    return nan


def _apply_pattern_map(row: pd.Series, column: str) -> str:
    pattern_info = row['PATTERN-MAP']
    return re.sub(
        pattern_info['regex_match'],
        pattern_info['regex_sub'],
        row[column],
    )


def _get_soundness(cond_s: str) -> str:
    match = _SOUNDNESS_TERMS_RE.search(cond_s)
    return match.group(1) if match else ''


def _build_passive_cond_s(row: pd.Series) -> str:
    return ' '.join(filter(None, (
        row['COND-S-ESSENTIAL-PASS'],
        row['SOUND'],
        'intrans',
    )))


def generate_passive(lexicon: pd.DataFrame, patterns_path: str) -> pd.DataFrame:
    """Generate passive lexicon rows using the configured pattern rules."""
    pattern_map = _load_pattern_map(patterns_path)

    passive = lexicon.loc[
        ~lexicon['COND-S'].str.contains('Frozen', na=False)
    ].copy()
    passive['PATTERN-DEF'] = passive.apply(_assign_pattern_definition, axis=1)
    passive = passive[passive['PATTERN-DEF'].notna()].copy()
    passive['COND-T'] = passive['COND-T'].str.strip()
    passive['COND-S-ESSENTIAL'] = passive['COND-S'].str.replace(
        _PASSIVE_COND_S_CLEAN_RE, '', regex=True
    ).str.strip()
    passive['PATTERN-MAP'] = passive.apply(
        _select_pattern_info,
        axis=1,
        pattern_map=pattern_map,
    )
    passive = passive[passive['PATTERN-MAP'].notna()].copy()

    passive['FORM'] = passive.apply(
        _apply_pattern_map, axis=1, column='FORM'
    )
    if 'PATTERN' in passive.columns:
        passive['PATTERN'] = passive.apply(
            _apply_pattern_map, axis=1, column='PATTERN'
        )

    passive['SOUND'] = passive['COND-S'].apply(_get_soundness)
    passive['COND-T'] = passive['PATTERN-MAP'].apply(
        lambda info: info['cond_t_pass']
    )
    passive['COND-S-ESSENTIAL-PASS'] = passive['PATTERN-MAP'].apply(
        lambda info: info['cond_s_pass'].strip()
    )
    passive['COND-S'] = passive.apply(_build_passive_cond_s, axis=1)
    passive['BW'] = passive['BW'].str.replace(
        r'(.V)', r'\1_PASS', regex=True
    )
    passive['FEAT'] = passive['FEAT'].str.replace(
        r'vox:a', r'vox:p', regex=True
    )
    if 'CATIB6' in passive.columns:
        passive['CATIB6'] = CATIB6_PASSIVE_VERB

    return passive.drop(columns=[
        'PATTERN-DEF',
        'COND-S-ESSENTIAL',
        'PATTERN-MAP',
        'SOUND',
    ])
            

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file", required=True,
                        type=str, help="Path of active aspect lexicon to generate the passive from.")
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Deprecated compatibility option; ignored.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Deprecated compatibility option; ignored.")
    parser.add_argument("-patterns", required=True,
                        type=str, help="Path of file which contains the passive pattern maps.")
    args = parser.parse_args()

    lexicon = pd.read_csv(args.input_file, na_filter=False)
    # Replace spaces in BW and GLOSS with '#'; skip commented rows and empty lines
    lexicon = lexicon[lexicon.DEFINE == 'LEXICON'].copy()
    lexicon['GLOSS'] = lexicon['GLOSS'].replace(r'\s+', '#', regex=True)
    lexicon['COND-S'] = lexicon['COND-S'].replace(r' +', ' ', regex=True)
    lexicon['COND-S'] = lexicon['COND-S'].replace(r' $', '', regex=True)
    lexicon['COND-T'] = lexicon['COND-T'].replace(r' +', ' ', regex=True)
    lexicon['COND-T'] = lexicon['COND-T'].replace(r' $', '', regex=True)

    passive = generate_passive(lexicon, args.patterns)
    
    output_path = re.sub(r'(.*)\.csv$', r'\1-PASS.csv', args.input_file)
    passive.to_csv(output_path)


if __name__ == "__main__":
    main()
