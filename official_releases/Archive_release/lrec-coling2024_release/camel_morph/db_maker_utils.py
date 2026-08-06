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
import os
import sys
from typing import Dict, List, Optional, Union, Set, Tuple
from itertools import product
from collections import Counter
from itertools import combinations
import pickle
from tqdm import tqdm

import pandas as pd
from numpy import nan

try:
    from camel_morph.utils.generate_passive import generate_passive
    from camel_morph.utils.generate_abstract_lexicon import generate_abstract_lexicon
    from camel_morph.utils.utils import Config
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) -
                                      1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from utils.generate_passive import generate_passive
    from utils.generate_abstract_lexicon import generate_abstract_lexicon
    from utils.utils import Config

EMPTY_ROW = dict(DEFINE=['MORPH'], CLASS=['[EMPTY]'], LINE=-1) #, FEAT='prc0:0 prc1:0 prc1.5:0 prc2:0 prc3:0 enc0:0 enc1:0')
SPECS_HEADER_REQUIRED = dict(
    order=['EXCLUDE', 'DEFINE', 'CLASS', 'PREFIX', 'STEM', 'SUFFIX'],
    morph=['EXCLUDE', 'DEFINE', 'CLASS', 'FUNC', 'FORM',
           'BW', 'GLOSS', 'FEAT', 'COND-T', 'COND-S',
           'CLASS', 'FUNC', 'FORM', 'BW', 'GLOSS']
)
ORDER_FIELDS = ['PREFIX', 'STEM', 'SUFFIX']
ORDER_FIELDS_SHORT = ['PREFIX-SHORT', 'STEM-SHORT', 'SUFFIX-SHORT']

def read_morph_specs(config:Config,
                     lexicon_df: Optional[pd.DataFrame] = None,
                     process_morph:bool=True,
                     lexicon_cond_f:bool=True) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[str, int]]]:
    """
    Method which loads and processes the `csv` sheets that are specified in the
    specific configuration of the config file. Outputs a dictionary which contains
    the 7 main dataframes which will be used throughout the DB making process.
    It preprocesses all of the sheets, concatenates MORPH sheets together (same
    for ORDER and LEXICON), computes the COND-F column for MORPH and LEXICON sheets.

    Args:
        config (Dict): dictionary containing all the necessary information to build 
        the `db` file.
        lexicon_df (pd.DataFrame): if this is specified, then the passed dataframe will be
        processed instead of the path that is specified in the config. Defaults to None.
        process_morph (bool): whether or not to process MORPH specs. Defaults to True.
        lexicon_cond_f (bool): whether or not to convert COND-T conditions to COND-F when necessary. Defaults to True.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[str, int]]]: dictionary which contains
        the 7 main dataframes which will be used throughout the DB making process, and an inventory
        of condition definitions and their corresponding vectors which will be useful in the later
        pruning process.

    """
    # Imported here to avoid disturbing other files' camel_tools importing which
    # should happen from the fork and not the pip installed version. If imported
    # globally, then the official camel_tools will be loaded everywhere, even in
    # other files.
    from camel_tools.utils.charmap import CharMapper
    safebw2ar = CharMapper.builtin_mapper('safebw2ar')
    
    if process_morph:
        ABOUT = pd.read_csv(config.get_sheets_paths('about')[0], na_filter=False)
        HEADER = pd.read_csv(config.get_sheets_paths('header')[0], index_col=0, na_filter=False)

        def _read_specs(specs_type):
            """Currently, different morph/order pair files are treated as independent entities,
            and their morpheme classes will be augmented with a different index that is unique to
            that pair. This will prevent interactions from happening between
            concatenated morph/order files. This may not be a desirable feature, as for example,
            one would maybe like to recycle, say, the >a_ques morpheme between nominals and verbs.
            This is impossible in the current setup, and we will have two different question morphemes,
            one for verbs, and one for nominals. This method would have to be altered in
            order to allow morpheme sharing between concatenated files.
            """
            specs_paths = config.get_sheets_paths(specs_type, with_labels=True)
            specs = None
            for specs_path, label in specs_paths:
                specs_ = pd.read_csv(specs_path, na_filter=False)
                assert set(SPECS_HEADER_REQUIRED[specs_type]) - set(specs_.columns) == set()
                if specs_type == 'order':
                    order_fields_short_ = [
                        f for f in ORDER_FIELDS_SHORT if f in specs_.columns]
                    order_fields = order_fields_short_ if order_fields_short_ else ORDER_FIELDS
                    unique_order_lines_ = Counter()
                    for line in specs_[order_fields].values.tolist():
                        if all(c.strip() for c in line):
                            unique_order_lines_.update([tuple(line)])
                    assert sum(unique_order_lines_.values()) == len(unique_order_lines_)
                    columns = ORDER_FIELDS + order_fields_short_
                elif specs_type == 'morph':
                    columns = ['CLASS']
                for col in columns:
                    if label:
                        specs_[col] = specs_.apply(
                            lambda row: re.sub(r'\]', f'-{label}]', row[col]), axis=1)
                specs = pd.concat([specs, specs_]).fillna('') if specs is not None else specs_
                
            specs.reset_index(drop=True, inplace=True)
            specs = specs.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            return specs
    
        ORDER, MORPH = _read_specs('order'), _read_specs('morph')
    else:
        ABOUT, HEADER, MORPH, ORDER = [None] * 4
    
    lexicon_sheet_names: List[Union[str, pd.DataFrame]] = None
    if lexicon_df is not None:
        lexicon_sheet_names = {None: lexicon_df}.items()
    else:
        lexicon_sheet_names = config.get_sheets_list('lexicon', with_labels=True)

    # `passive_patterns_sheets` contain regex transformation rules that specify a passive verb form
    # for each active form. This option is used when the passive verb entries are not included (frozen)
    # into the verb lexicon sheets. Therefore, if the verb lexicon only contains active forms, and
    # we want to generate passive forms on the fly, these patterns should be specified in the config. This
    # option is only useful for debugging, but final versions of the lexicon contain the generated passive verb
    # forms, therefore there is no need to specify this.
    passive_patterns_sheets: Optional[Dict] = None
    if config.passive is not None:
        passive_patterns_sheets = {
            act: pass_ for act2pass in config.passive.values()
            for act, pass_ in act2pass.items()}
    backoff_sheets: Optional[Dict] = None
    if config.backoff is not None:
        if type(config.backoff) is dict:
            backoff_sheets = {
                act: abs_ for act2abs in config.backoff.values()
                for act, abs_ in act2abs.items()}
        else:
            backoff_sheets = config.backoff
    
    # Process LEXICON sheet
    # Loop over the specified lexicon (and backoff lexicon if present) sheets to concatenate those into
    # a unified dataframe.
    LEXICON = {'concrete': None, 'backoff': None, 'smart_backoff': None}
    for lexicon_sheet_name, label in lexicon_sheet_names:
        if type(lexicon_sheet_name) is str:
            # Setting dtype to silence warning of mixed types
            LEXICON_ = pd.read_csv(config.get_sheet_path_from_name(lexicon_sheet_name),
                                   dtype=object, na_filter=False)
            if label:
                LEXICON_['CLASS'] = LEXICON_.apply(
                    lambda row: re.sub(r'\]', f'-{label}]', row['CLASS']), axis=1)
        else:
            LEXICON_ = label
        # Only use entries in which the `DEFINE` has 'LEXICON' specified.
        BACKOFF_ = LEXICON_[LEXICON_.DEFINE == 'BACKOFF']
        LEXICON_ = LEXICON_[LEXICON_.DEFINE == 'LEXICON']
        if 'COND-F' not in LEXICON_.columns:
            LEXICON_['COND-F'], BACKOFF_['COND-F'] = '', ''
        if passive_patterns_sheets is not None:
            # Generate passive verb lexicon on the fly from the generation patterns.
            passive_patterns = passive_patterns_sheets.get(lexicon_sheet_name)
            if passive_patterns:
                print(f'\nGenerating automatic passive entries from {passive_patterns}...', end='')
                #TODO: should import this method using importlib and a path from the local config
                LEXICON_PASS = generate_passive(
                    LEXICON_, config.get_sheet_path_from_name(passive_patterns))
                LEXICON_ = pd.concat([LEXICON_, LEXICON_PASS], ignore_index=True).fillna('')
                print(f'Generated {len(LEXICON_PASS.index)} entries... Done.')

        if backoff_sheets is not None:
            if backoff_sheets == 'auto':
                # Generate backoff lexicon from the language-agnostic method relying on the
                # root class, lemma pattern, and form pattern.
                SMART_BACKOFF_ = generate_abstract_lexicon(LEXICON_)
            elif backoff_sheets.get(lexicon_sheet_name):
                # Allow the specification of an already generated backoff lexicon and load it.
                backoff_filename = f"{backoff_sheets[lexicon_sheet_name]}.csv"
                SMART_BACKOFF_ = pd.read_csv(
                    config.get_sheet_path_from_name(backoff_filename), na_filter=False)
            LEXICON['smart_backoff'] = pd.concat([LEXICON['smart_backoff'], SMART_BACKOFF_]).fillna('') \
                if LEXICON['smart_backoff'] is not None else SMART_BACKOFF_

        if len(LEXICON_.index):
            LEXICON['concrete'] = pd.concat([LEXICON['concrete'], LEXICON_]).fillna('') \
                if LEXICON['concrete'] is not None else LEXICON_
        if len(BACKOFF_.index):
            LEXICON['backoff'] = pd.concat([LEXICON['backoff'], BACKOFF_]).fillna('') \
                if LEXICON['backoff'] is not None else BACKOFF_

    for lex_type in ['concrete', 'backoff']:
        if LEXICON[lex_type] is None:
            continue
        LEXICON[lex_type] = LEXICON[lex_type].astype(str)
        LEXICON[lex_type] = LEXICON[lex_type].apply(
            lambda x: x.str.strip() if x.dtype == 'object' and x.dtype not in [float, int] else x)
        LEXICON[lex_type].reset_index(drop=True, inplace=True)
        LEXICON[lex_type] = LEXICON[lex_type].apply(
            lambda x: x.str.strip() if x.dtype == 'object' else x)
        if 'BW' not in LEXICON[lex_type].columns:
            LEXICON[lex_type]['BW'] = LEXICON[lex_type]['FEAT'].str.extract(
                r'pos:(\S+)')[0].apply(str.upper)
        LEXICON[lex_type]['BW'] = LEXICON[lex_type]['BW'].replace('\s+', '#', regex=True)
        LEXICON[lex_type].loc[LEXICON[lex_type]['BW'] == '', 'BW'] = \
            LEXICON[lex_type]['FEAT'].str.extract(r'pos:(\S+)')[0].apply(str.upper)
        LEXICON[lex_type].loc[LEXICON[lex_type]['BW'].str.contains('\+'), 'BW'] = \
            LEXICON[lex_type]['BW']
        LEXICON[lex_type].loc[~LEXICON[lex_type]['BW'].str.contains('\+|/', regex=True), 'BW'] = \
            LEXICON[lex_type]['FORM'] + '/' + LEXICON[lex_type]['BW']
        LEXICON[lex_type]['LEMMA'] = 'lex:' + LEXICON[lex_type]['LEMMA']
        LEXICON[lex_type] = LEXICON[lex_type] if len(LEXICON[lex_type].index) != 0 else None
    
        for col in ['D3SEG', 'D3TOK', 'ATBSEG', 'ATBTOK']:
            if col not in LEXICON[lex_type]:
                LEXICON[lex_type][col] = ''
        
        source = 'lex' if lex_type == 'concrete' else 'backoff'
        if 'SOURCE' in LEXICON[lex_type].columns:
            LEXICON[lex_type].loc[LEXICON[lex_type]['SOURCE'] == '', 'SOURCE'] = source
        else:
            LEXICON[lex_type]['SOURCE'] = source
        
        if 'LINE' not in LEXICON[lex_type].columns:
            LEXICON[lex_type]['LINE'] = ''

    # Process POSTREGEX sheet
    # Compiles the regex match expression from the sheet into a regex match expression that is
    # suitable for storing into the DB in Arabic script. Expects Safe BW transliteration in the sheet.
    POSTREGEX = None
    if config.postregex is not None:
        POSTREGEX = pd.read_csv(config.get_sheets_paths('postregex')[0], na_filter=False)
        POSTREGEX = POSTREGEX[(POSTREGEX.DEFINE == 'POSTREGEX') &
                              (POSTREGEX.VARIANT == config.dialect.upper())]
        for i, row in POSTREGEX.iterrows():
            POSTREGEX.at[i, 'MATCH'] = _bw2ar_regex(row['MATCH'], safebw2ar)
            POSTREGEX.at[i, 'REPLACE'] = _bw2ar_regex(
                ''.join(re.sub(r'\$', r'\\', row['REPLACE'])), safebw2ar)
    
    # This splits rows with or-ed COND-T expressions, i.e., expands the COND-T into
    # mujltiple rows. A COND-T expression (cell) is always a conjunction of terms (one
    # condition or a disjunction of conditions). For example, a row with COND-T `a1 b1||b2`
    # would be expanded into two rows, each with COND-T `a1 b1` and `a1 b2` respectively.
    # It is useful for debugging nominals, since we would like to debug at the
    # form gender-number level.
    if config.split_or == True:
        assert LEXICON['smart_backoff'] is None
        for lex_type in ['concrete', 'backoff']:
            if LEXICON[lex_type] is None:
                continue
            LEXICON_ = []
            for _, row in LEXICON[lex_type].iterrows():
                terms = {'disj': [], 'other': []}
                for term in row['COND-T'].split():
                    terms['disj' if '||' in term else 'other'].append(term)
                
                if terms['disj']:
                    terms_other = ' '.join(terms['other'])
                    for disj_terms in product(*[t.split('||') for t in terms['disj']]):
                        row_ = row.to_dict()
                        row_['COND-T'] = f"{' '.join(disj_terms)} {terms_other}".strip()
                        LEXICON_.append(row_)
                else:
                    LEXICON_.append(row.to_dict())
            LEXICON[lex_type] = pd.DataFrame(LEXICON_)

    exclusions: List[str] = config.exclude if config.exclude is not None else []
    
    # Process ORDER sheet
    if ORDER is not None:
        ORDER = ORDER[ORDER.DEFINE == 'ORDER']  # skip comments & empty lines
        for exclusion in exclusions:
            ORDER = ORDER[~ORDER.EXCLUDE.str.contains(f'(?:^|\s){exclusion}(?:\s|$)')]

    cond2class = None
    if MORPH is not None:
        # Dictionary which groups conditions into classes (used later to
        # do partial compatibility which is useful from pruning out incoherent
        # suffix/prefix/stem combinations before performing full compatibility
        # in which (prefix, stem, suffix) instances are tried out individually).
        class2cond = MORPH[MORPH.DEFINE == 'CONDITIONS']
        class2cond: Dict[str, List[str]] = {
            cond_class["CLASS"]: [cond for cond in cond_class["FUNC"].split() if cond]
            for _, cond_class in class2cond.iterrows()}
        # Reverses the dictionary (k -> v and v -> k) so that individual conditions
        # which belonged to a class are now keys with that class as a value. In addition,
        # each condition gets its corresponding one-hot vector (actually stored as an int
        # because bitwise operations can only be performed on ints) computed based on the
        # other conditions within the same class (useful for pruning later).
        #NOTE: all conditions appearing in the lexicon and morph file must be included
        # in the condition defitions in the Morph sheet if the pruning option is set to true
        # or else KeyError exceptions will be thrown in the DB making process.
        cond2class = {
            cond: (cond_class, 
                int(''.join(['1' if i == index else '0' for index in range (len(cond_s))]), 2)
            )
            for cond_class, cond_s in class2cond.items()
                for i, cond in enumerate(cond_s)}
        
        # Process MORPH sheet
        MORPH = MORPH[MORPH.DEFINE == 'MORPH']
        MORPH = pd.concat([MORPH, pd.DataFrame(EMPTY_ROW)], ignore_index=True).fillna('')
        MORPH = MORPH.astype(str)
        MORPH = MORPH.apply(lambda x: x.str.strip()
                            if x.dtype == 'object' and x.dtype not in [float, int] else x)
        MORPH = MORPH.replace('\s+', ' ', regex=True)

        MORPH['COND-S'] = MORPH['COND-S'].replace('[\[\]]', '', regex=True)
        MORPH.loc[MORPH['COND-S'] == '', 'COND-S'] = '_'
        MORPH.loc[MORPH['COND-T'] == '', 'COND-T'] = '_'
        # Replace spaces in BW and GLOSS with '#'
        MORPH['BW'] = MORPH['BW'].replace('\s+', '#', regex=True)
        MORPH.loc[MORPH['BW'] == '', 'BW'] = '_'
        MORPH.loc[MORPH['FORM'] == '', 'FORM'] = '_'
        MORPH['GLOSS'] = MORPH['GLOSS'].replace('\s+', '#', regex=True)
        if 'COND-F' not in MORPH.columns:
            MORPH['COND-F'] = ''
        MORPH = process_morph_specs(MORPH, exclusions) if process_morph else MORPH

    # cont'd: Process LEXICON sheet
    for lex_type in ['concrete', 'backoff']:
        if LEXICON[lex_type] is None:
            continue
        LEXICON[lex_type] = LEXICON[lex_type].replace('\s+', ' ', regex=True)
        LEXICON[lex_type]['GLOSS'] = LEXICON[lex_type]['GLOSS'].replace('\s+', '#', regex=True)
        # Generate Lexicon COND-F if necessary
        if lexicon_cond_f and any('!' in cond_t for cond_t in LEXICON[lex_type]['COND-T'].values.tolist()):
            print('Lexicon sheet COND-F populated')
            for i, row in LEXICON[lex_type].iterrows():
                cond_t = row['COND-T'].split()
                cond_t_, cond_f_ = [], []
                for ct in cond_t:
                    if ct.startswith('!'):
                        cond_f_.append(ct[1:])
                    else:
                        cond_t_.append(ct)
                LEXICON[lex_type].at[i, 'COND-T'] = ' '.join(cond_t_)
                LEXICON[lex_type].at[i, 'COND-F'] = ' '.join(cond_f_)
    
    #NOTE: temporary fix to the next `if` statement; read below FIXME note to see
    # what the problem is. Here we delete some unused conditions manually, but
    # there might be others which we do not know of, which is why the next `if`
    # statement is needed.
    if LEXICON['concrete'] is not None:
        LEXICON['concrete']['COND-S'] = LEXICON['concrete']['COND-S'].replace(
            r'hollow|defective', '', regex=True)
        LEXICON['concrete']['COND-S'] = LEXICON['concrete']['COND-S'].replace(
            r' +', ' ', regex=True)
    
    #FIXME: below `if` statement currently not being used anywhere because behavior is
    # still not completely correct. The way unused conditions are being deleted needs
    # to be corrected. The correct behavior is: if a condition appears in COND-S of
    # LEXICON, then it must necessarily appear in COND-T of MORPH, otherwise it will
    # be unused and should therefore be deleted. Similarly, if a condition appears in
    # COND-T of LEXICON, then it must appear in COND-S of MORPH, else it will be unused
    # and should be deleted. Deletion of one condition in a cell should not affect the
    # rest of the conditions in that cell. Take into account that there might be
    # disjunctions in COND-T of either sheets while performing deletion.
    
    # Gets rid of unused conditions, i.e., use only the conditions which are in the
    # intersection of the collective (concatenated across morph and lexicon sheets)
    # COND-T and COND-S columns.
    #TODO: apply this to backoff too
    clean_conditions: Optional[str] = config.clean_conditions
    if clean_conditions:
        conditions = {}
        for f in ['COND-T', 'COND-S', 'COND-F']:
            conditions[f] = set(
                [term for ct in MORPH[f].values.tolist() + LEXICON['concrete'][f].values.tolist()
                        for cond in ct.split() for term in cond.split('||')])
        conditions_aggr = conditions['COND-S'] & (conditions['COND-T'] | conditions['COND-F'])
        conditions_used = set([cond for cond in conditions_aggr if cond != '_'])
        #TODO: process MORPH file here too.
        for df in [LEXICON['concrete']]:
            for f in ['COND-T', 'COND-S', 'COND-F']:
                conditions = set([term for ct in df[f].values.tolist()
                            for cond in ct.split() for term in cond.split('||')])
                conditions_unused = '|'.join(conditions - conditions_used)
                assert all(not bool(re.search(r'[\$\|\^\*\+><}{][)(\?\!\\]', cond))
                            for cond in conditions_unused)
                if conditions_unused:
                    print(f'Deleting unused conditions: {conditions_unused}')
                    conditions_unused = '^(' + conditions_unused + ')$'
                    df[f] = df.apply(lambda row: ' '.join(re.sub(conditions_unused, '', cond)
                                                          for cond in row[f].split()), axis=1)

                df[f] = df[f].replace(' +', ' ', regex=True)
                df[f] = df[f].replace(' $', '', regex=True)

    SHEETS = dict(about=ABOUT, header=HEADER, order=ORDER, morph=MORPH,
                  lexicon=LEXICON['concrete'], smart_backoff=LEXICON['smart_backoff'],
                  postregex=POSTREGEX, backoff=LEXICON['backoff'])
    
    
    return SHEETS, cond2class


def process_morph_specs(MORPH:pd.DataFrame, exclusions: List[str]) -> pd.DataFrame:
    """
    Method which preprocesses the MORPH sheet by cleaning it and generating
    the COND-F column by processing each morpheme one at a time. A morpheme is defined
    by its (CLASS, FUNC) tuple whereas an allomorph is defined by its (CLASS, FUNC, FORM).
    Hence, all lines having the same (CLASS, FUNC) combination belong to the same morpheme,
    and similarly for allomorphs. COND-F is generated by (1) inferring the meaning
    of the `else` marker in the scope of a morpheme, since the scope of the `else`
    marker is exactly that, and (2) by taking all negated conditions, i.e., preceded
    by `!` to the COND-F column. The way the `else` is processed is in the following way:
    the COND-T of each morpheme is treated as a table (i.e., with columns and rows) and
    the behavior only matches the intended one provided conditions across the rows of a
    morpheme are written in a way which is consistent to and aware of all the below definitions:
    Example (x1, x2, ... are conditions):
        CLASS   FUNC    COND-T                  COND-F
        X       Y       A       B       C
    1   CLS1    F1      x1      x2      x3      (x4; from default else of x1)
    2   CLS1    F1      x1      x2      else    x3 (x4; from default else of x1)
    3   CLS1    F1      x4      else    else    (x1; from default else of x4)
    4   CLS1    F1      else    x5      else    x1 x4
    5   CLS1    F1      else    else    else    x1 x4 x5
    
    The above table represents the scope of a dummy morpheme in the MORPH sheet. Rows 1 to 5 belong to
    the same morpheme becaus they have the same CLASS and FUNC values. The rows are actual
    rows in the sheet. Columns X and Y are actual columns in the sheet, whereas A, B, and C are not
    actual sheet columns; the conditions in the COND-T column are written in such a way as to emulate
    a table structure. To understand how the `else` markers populates the COND-F column, we define the
    Selector Range (SR). It is the horizontal range which comes to the left of any cell, e.g., the SR
    of B2 is X2:A2 (i.e., CLS1 F1 x1), and the SR of C4 is X4:B4. Formally, the SR of Dn is Xn:Cn where
    X is the first col to the left, and going right, C is the col just before D.
    Also, we define the Range of Action (RoA) of an `else` marker as a certain subset of the same column
    it lies in. The cells (conditions) in that subset will appear in the COND-F of the allomorph (row) whose COND-T
    contains that `else`. The subset is chosen by picking all the conditions which have the same SR as the `else`
    (which means that they are also in the same column as the `else`). For example, the RoA of the `else` in C2
    is C1:2, because the only cell that has the same SR as the C2 is C1. All cells in columns have the same SR
    to their left so their RoA is the whole col A. The `else` in B3 has no effect (RoA is null) since no other
    cell has the same SR (similarly for C3, C4, and C5). Formally, the RoA of an `else` in Dn is all cells
    which have SR equal to Xn:Cn.
    Therefore, all conditions (excluding `else`) that lie in the RoA of an `else` will go to COND-F of the
    row the `else` is in.

    Additionally, each condition carries an inherent "default `else`", meaning that all conditions (excluding
    `else` and itself) that lie in the RoA of that condition will go to COND-F of the row that that condition
    is in. For example, see rows 1 to 3 in which the COND-F holds x4 or x1, which are the direct result of
    this "default else".
    
    Note columns A, B, ... within COND-T represent a conjunction of the terms in it. For example, row 1 in COND-T
    is interpreted as x1 AND x2 AND x3. Also note that conditions (x1, x2, ...) can be be one of three things:
        1. a single condition (x1).
        2. a disjunction of conditions, e.g., x1||x11||..., with || standing for OR.
        3. a negation of either 1. or 2., e.g., !x1 or !x1||x11||..., represented in the aforementioned way.
           Any negation is taken by default to COND-F, e.g., if we have !x1 in COND-T, it becomes x1 in COND-F.
           Note that a negation of a disjunction is equivalent to a conjunction of the negations by De Morgan's law.
           So within COND-T !x1||x11||... can be interpreted as !x1 !x11 ...

    Args:
        MORPH (pd.DataFrame): morph sheet dataframe
        exclusions (List[str]): labels to search for in EXCLUDE column, indicating that 
        the line should be dropped.

    Returns:
        pd.DataFrame: processed morph dataframe.
    """
    MORPH_CLASSES = MORPH.CLASS.unique()
    MORPH_CLASSES = MORPH_CLASSES[MORPH_CLASSES != '_']
    MORPH_MORPHEMES = MORPH.FUNC.unique()
    MORPH_MORPHEMES = MORPH_MORPHEMES[MORPH_MORPHEMES != '_']
    # TODO: refactor in a way to go through only valid morphemes, it is not necessary
    # to search for those, simply iterate over them as they are in MORPH sheet.
    for CLASS in MORPH_CLASSES:
        for MORPHEME in MORPH_MORPHEMES: 
            # Get the unique list of the true conditions in all the allomorphs.
            # We basically want to convert the 'COND-T' column into n columns
            # where n = maximum number of conditions for a single allomorph
            cond_t = MORPH[(MORPH.FUNC == MORPHEME) & 
                                (MORPH.CLASS == CLASS)]['COND-T'].str.split(pat=' ', expand=True)
            if cond_t.empty:
                continue
            cond_t = cond_t.replace(r'\b_\b', '', regex=True)

            if len(cond_t.iloc[:][:]) == 1 and cond_t.iloc[0][0] == '':
                continue
            # Go through each column in the true conditions
            for col in cond_t:
                if col == 0:
                    # Unique col[0] and remove fluff
                    cond_t_current = get_clean_set(cond_t[col])
                    MORPH = get_morph_cond_f(cond_t[col], cond_t_current, MORPH)
                else:
                    # Create temp_T by merging the the condTrue for col 0-col, put it in 
                    # col[0] and remove 1:col. This is basically to group allomorph with 
                    # similar general conditions. Take the unique of what is now col[0] and
                    # remove the fluff.
                    if col == 1:
                        cond_t_temp = cond_t.copy()
                    else:
                        cond_t_temp = cond_t.copy()
                        cond_t_temp = cond_t_temp.replace([None], '')

                        cond_t_temp[0] = cond_t_temp.loc[:,:col-1].agg(' '.join, axis=1)
                        cond_t_temp = cond_t_temp.drop(cond_t_temp.columns[1:col], axis=1)
                        cond_t_temp.columns = range(cond_t_temp.shape[1])
                    if len(cond_t_temp.columns) == 1:
                        continue

                    cond_groups = cond_t_temp[0].unique()
                    cond_groups = cond_groups[cond_groups!= '_']
                    # Go through each group of allomorphs
                    for cond_group in cond_groups:
                        cond_t_current = get_clean_set(cond_t_temp[cond_t_temp[0] == cond_group][1])
                        MORPH = get_morph_cond_f(
                            cond_t_temp[cond_t_temp[0] == cond_group][1], cond_t_current, MORPH)
            
            for idx, _ in MORPH[(MORPH.FUNC == MORPHEME) & (MORPH.CLASS == CLASS)].iterrows():
                # If there is a negated condition in the set of true conditions for the 
                # current allomorph, remove the negation and add it to the false 
                # conditions list for the current alomorph.
                cond_t_almrph = MORPH.loc[idx, 'COND-T'].split(' ')
                cond_f_almrph = MORPH.loc[idx, 'COND-F'].split(' ')
                cond_s_negated = [x for x in cond_t_almrph if x.startswith('!')]
                if cond_s_negated:
                    for cond_negated in cond_s_negated:
                        cond_f_almrph.append(cond_negated[1:])
                        # remove from the true conditions
                        cond_t_almrph.remove(cond_negated)
                cond_t_almrph = [y for y in cond_t_almrph if y not in ['', '_', 'else', None]]
                cond_f_almrph = [y for y in cond_f_almrph if y not in ['', '_', 'else', None]]
                MORPH.loc[idx, 'COND-T'] = ' '.join(cond_t_almrph)
                MORPH.loc[idx, 'COND-F'] = ' '.join(cond_f_almrph)

    for exclusion in exclusions:
        MORPH = MORPH[~MORPH.EXCLUDE.str.contains(f'(?:^|\s){exclusion}(?:\s|$)')]

    MORPH = MORPH.drop_duplicates([h for h in SPECS_HEADER_REQUIRED['morph']
                                   if h not in ['EXCLUDE', 'DEFINE']])
    
    return MORPH


def get_clean_set(cond_col: pd.Series) -> Set[str]:
    """Cleans up the conjunction of terms in COND-T field, keeping only valid
    conditions (not else or empty)."""
    morph_cond_t = cond_col.tolist()
    morph_cond_t = [
        y for y in morph_cond_t if y not in ['', '_', 'else', None]]
    morph_cond_t = set(morph_cond_t)

    return morph_cond_t


def get_morph_cond_f(morph_cond_t: pd.Series,
                     cond_t_current: Set[str],
                     MORPH: pd.DataFrame) -> pd.DataFrame:
    """Get COND-F based on the RoA (see defition in documentation of
    `process_morph_specs()`"""
    # Go through each allomorph
    for idx, entry in morph_cond_t.iteritems():
        # If we have no true condition for the allomorph (aka can take anything)
        if entry is None:
            continue
        elif entry != 'else':
            cond_f_almrph, _ = _get_cond_false(cond_t_current, [entry])
        elif entry == 'else':
            cond_f_almrph = cond_t_current
            # Finally, populate the 'COND-F' cell with the false conditions
        MORPH.loc[idx, 'COND-F'] = MORPH.loc[idx, 'COND-F'] + \
            ' ' + ' '.join(cond_f_almrph)
        MORPH.loc[idx, 'COND-F'] = re.sub(r'\b_\b', '', MORPH.loc[idx, 'COND-F'])
    
    return MORPH

def _get_cond_false(cond_t_all, cond_t_almrph):
    """
    The COND-F for the current allomorph is whatever COND-T in the morpheme set that
    does not belong to the set of true conditions to the current allomorph.
    """
    cond_f_almrph = cond_t_all - set(cond_t_almrph)
    # If we end up with no false condition, set it to '_'
    if not cond_f_almrph:
        cond_f_almrph = ['_']
    # If we end up with no true condition, set it to '_'
    if not cond_t_almrph:
        cond_t_almrph = ['_']

    return cond_f_almrph, cond_t_almrph


def _bw2ar_regex(regex, bw2ar):
    """ Converts regex expression from the sheet to Arabic while taking care not to
    convert characters which are special regex characters in the process. This expects
    the input not to be ambiguous e.g.,Safe BW.
    """
    #FIXME: make sure that symbols being compiled into regex are not mistaken
    # for special characters. The below does not do that properly, but was enough
    # for the EGY sheets.
    match_ = []
    for match in re.split(r'(\\.|[\|}{\*\$_])', regex):
        match = match if re.match(r'(\\.)|[\|}{\*\$_]', match) else bw2ar(match)
        match_.append(match)
    return ''.join(match_)


def _reverse_compat_table(XY):
    YX = {}
    for X_cat, Y_cats in XY.items():
        for Y_cat in Y_cats:
            YX.setdefault(Y_cat, set()).add(X_cat)
    return YX


def factorize_categories(prefix_stem_compat,
                         stem_suffix_compat,
                         prefix_suffix_compat,
                         test=''):
    stem_prefix_compat = _reverse_compat_table(prefix_stem_compat)
    suffix_stem_compat = _reverse_compat_table(stem_suffix_compat)
    suffix_prefix_compat = _reverse_compat_table(prefix_suffix_compat)
    assert len(prefix_stem_compat) == len(prefix_suffix_compat) and \
           len(stem_prefix_compat) == len(stem_suffix_compat) and \
           len(suffix_stem_compat) == len(suffix_prefix_compat)

    combs = dict(
        prefix=list(combinations(prefix_stem_compat, 2)),
        stem=list(combinations(stem_suffix_compat, 2)),
        suffix=list(combinations(suffix_stem_compat, 2))
    )
    
    if not test:
        equivalences = {}
        for pcat_1, pcat_2 in tqdm(combs['prefix']):
            if prefix_stem_compat[pcat_1] == prefix_stem_compat[pcat_2]:
                if prefix_suffix_compat[pcat_1] == prefix_suffix_compat[pcat_2]:
                    equivalences.setdefault(pcat_1, set()).add(pcat_2)
        for xcat_1, xcat_2 in tqdm(combs['stem']):
            if stem_suffix_compat[xcat_1] == stem_suffix_compat[xcat_2]:
                if stem_prefix_compat[xcat_1] == stem_prefix_compat[xcat_2]:
                    equivalences.setdefault(xcat_1, set()).add(xcat_2)
        for scat_1, scat_2 in tqdm(combs['suffix']):
            if suffix_stem_compat[scat_1] == suffix_stem_compat[scat_2]:
                if suffix_prefix_compat[scat_1] == suffix_prefix_compat[scat_2]:
                    equivalences.setdefault(scat_1, set()).add(scat_2)
    else:
        with open(test, 'rb') as f:
            equivalences = pickle.load(f)
    
    if not equivalences:
        print(('WARNING: Tried to factorize categories but found that they are already '
               'optimally factorized'))
        return {}

    equivalences_ = {}
    done = set()
    while done != {True}:
        done = set()
        for cat, cats_eq in (equivalences_ if equivalences_ else equivalences).items():
            cats_eq_ = set()
            for cat_eq in cats_eq:
                cats_eq_.update(equivalences.get(cat_eq, {cat_eq}))
            if cats_eq != cats_eq_:
                done.add(False)
            else:
                done.add(True)
            equivalences_[cat] = cats_eq_

    assert all(len(v) == 1 for v in equivalences_.values())
    assert set.union(*equivalences_.values()) & set(equivalences_) == set()
    equivalences_ = {k: next(iter(v)) for k, v in equivalences_.items()}

    return equivalences_


def factorize_compatibility_lines(prefix_stem_compat,
                                  stem_suffix_compat,
                                  prefix_suffix_compat,
                                  equivalences):

    def _rebuild_compat_reduced(X_Y_compat):
        X_Y_compat_ = {}
        for X_cat, Y in X_Y_compat.items():
            X_cat_ = equivalences.get(X_cat, X_cat)
            Y_ = set()
            for Y_cat in Y:
                Y_.add(equivalences.get(Y_cat, Y_cat))
            X_Y_compat_[X_cat_] = Y_
        return X_Y_compat_

    prefix_stem_compat_ = _rebuild_compat_reduced(prefix_stem_compat)
    stem_suffix_compat_ = _rebuild_compat_reduced(stem_suffix_compat)
    prefix_suffix_compat_ = _rebuild_compat_reduced(prefix_suffix_compat)
    
    def _get_cats_map(X_Y_compat):
        X_Y_compat_set = set(X_Y_compat)
        X_type = set(X[0] for X in X_Y_compat_set)
        assert len(X_type) == 1 and X_type <= {'P', 'X', 'S'}
        X_type = next(iter(X_type))
        X_cats_sorted = sorted(X_Y_compat_set)
        X_cat_map = {}
        for i, X_cat in enumerate(X_cats_sorted):
            X_cat_new = f'{X_type}{str(i + 1).zfill(5)}'
            X_cat_map[X_cat] = X_cat_new
        return X_cat_map

    prefix_cat_map = _get_cats_map(prefix_stem_compat_)
    stem_cat_map = _get_cats_map(stem_suffix_compat_)
    suffix_stem_compat = _reverse_compat_table(stem_suffix_compat_)
    suffix_cat_map = _get_cats_map(suffix_stem_compat)

    def _reindex_compat_categories(X_Y_compat, X_cat_map, Y_cat_map):
        X_Y_compat_ = {}
        for X_cat, Y in X_Y_compat.items():
            for Y_cat in Y:
                X_Y_compat_.setdefault(X_cat_map[X_cat], set()).add(Y_cat_map[Y_cat])
        return X_Y_compat_

    prefix_stem_compat_ = _reindex_compat_categories(
        prefix_stem_compat_, prefix_cat_map, stem_cat_map)
    stem_suffix_compat_ = _reindex_compat_categories(
        stem_suffix_compat_, stem_cat_map, suffix_cat_map)
    prefix_suffix_compat_ = _reindex_compat_categories(
        prefix_suffix_compat_, prefix_cat_map, suffix_cat_map)

    return prefix_stem_compat_, stem_suffix_compat_, prefix_suffix_compat_, \
           prefix_cat_map, stem_cat_map, suffix_cat_map
           

def reindex_cat(X_cat, X_cat_map, equivalences):
    X_cat_new = X_cat_map[equivalences.get(X_cat, X_cat)]
    return X_cat_new
