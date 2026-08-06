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

import pandas as pd
from numpy import nan

try:
    from utils.utils.generate_passive import generate_passive
    from utils.utils.generate_abstract_lexicon import generate_abstract_lexicon
except:
    pass

def read_morph_specs(config:Dict, config_name:str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[str, int]]]:
    """
    Method which loads and processes the `csv` sheets that are specified in the
    specific configuration of the config file. Outputs a dictionary which contains
    the 7 main dataframes which will be used throughout the DB making process.

    Args:
        config (Dict): dictionary containing all the necessary information to build 
        the `db` file.
        config_name (str): key of the specific ("local") configuration to get information 
        from in the config file.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[str, int]]]: dictionary which contains
        the 7 main dataframes which will be used throughout the DB making process, and an inventory
        of condition definitions and their corresponding vectors which will be useful in the later
        pruning process.

    """
    # Imported here to avoid disturbing other files' camel_tools importing which
    # should happen from the fork and not the pip installed version.
    from camel_tools.utils.charmap import CharMapper
    safebw2ar = CharMapper.builtin_mapper('safebw2ar')
    
    # Specific configuration from the list of configurations in config['local']
    local_specs: Dict = config['local'][config_name]
    # Directory where all the specification sheets are present
    data_dir: str = os.path.join(config['global']['data_dir'], f"camel-morph-{local_specs['dialect']}", config_name)
    
    ABOUT = pd.read_csv(os.path.join(data_dir, 'About.csv'))
    HEADER = pd.read_csv(os.path.join(data_dir, 'Header.csv'))
    
    # If statement to deal with the case when there is no spreadsheet specified in the local
    # configuration to get the sheets from. If a `spreadsheet` key is present in the local `specs`
    # in the local configuration, then ORDER/MORPH are present in a dict under the key `sheets`,
    # otherwise, they are just sitting in the `specs` dict.
    order_filename = f"{local_specs['specs']['sheets']['order']}.csv"
    ORDER = pd.read_csv(os.path.join(data_dir, order_filename))
    morph_filename = f"{local_specs['specs']['sheets']['morph']}.csv"
    MORPH = pd.read_csv(os.path.join(data_dir, morph_filename))
    
    lexicon_sheets: List[str] = local_specs['lexicon']['sheets']
    # `passive_patterns_sheets` contain regex transformation rules that specify a passive verb form
    # for each active form. This option is used when the passive verb entries are not included (frozen)
    # into the verb lexicon sheets. Therefore, if the verb lexicon only contains active forms, and
    # we want to generate passive forms on the fly, these patterns should be specified in the config. This
    # option is only useful for debugging, but final versions of the lexicon contain the generated passive verb
    # forms, therefore there is no need to specify this.
    if local_specs['specs'].get('spreadsheet'):
        passive_patterns_sheets: Optional[Dict] = local_specs['specs']['sheets'].get('passive')
    else:
        passive_patterns_sheets = local_specs['specs'].get('passive')
    backoff_sheets: Optional[Dict] = local_specs['lexicon'].get('backoff')
    
    # Process LEXICON sheet
    # Loop over the specified lexicon (and backoff lexicon if present) sheets to concatenate those into
    # a unified dataframe.
    LEXICON, SMART_BACKOFF, BACKOFF = None, None, None
    for lexicon_sheet_name in lexicon_sheets:
        LEXICON_ = pd.read_csv(os.path.join(data_dir, f"{lexicon_sheet_name}.csv"))
        # Only use entries in which the `DEFINE` has 'LEXICON' specified.
        BACKOFF_ = LEXICON_[LEXICON_.DEFINE == 'BACKOFF']
        BACKOFF_ = BACKOFF_.replace(nan, '', regex=True)
        LEXICON_ = LEXICON_[LEXICON_.DEFINE == 'LEXICON']
        LEXICON_ = LEXICON_.replace(nan, '', regex=True)
        if 'COND-F' not in LEXICON_.columns:
            LEXICON_['COND-F'] = ''
            BACKOFF_['COND-F'] = ''

        if passive_patterns_sheets:
            # Generate passive verb lexicon on the fly from the generation patterns.
            passive_patterns = passive_patterns_sheets.get(lexicon_sheet_name)
            if passive_patterns:
                LEXICON_PASS = generate_passive(
                    LEXICON_, os.path.join(data_dir, f"{passive_patterns}.csv"))
                LEXICON_ = pd.concat([LEXICON_, LEXICON_PASS], ignore_index=True)

        if backoff_sheets:
            if backoff_sheets == 'auto':
                # Generate backoff lexicon from the language-agnostic method relying on the
                # root class, lemma pattern, and form pattern.
                SMART_BACKOFF_ = generate_abstract_lexicon(LEXICON_)
            elif backoff_sheets.get(lexicon_sheet_name):
                # Allow the specification of an already generated backoff lexicon and load it.
                backoff_filename = f"{backoff_sheets[lexicon_sheet_name]}.csv"
                SMART_BACKOFF_ = pd.read_csv(os.path.join(data_dir, backoff_filename))
                SMART_BACKOFF_ = SMART_BACKOFF_.replace(nan, '', regex=True)
            SMART_BACKOFF = pd.concat([SMART_BACKOFF, SMART_BACKOFF_])

        LEXICON_['BW'] = LEXICON_['FORM'] + '/' + LEXICON_['BW']
        LEXICON_['LEMMA'] = 'lex:' + LEXICON_['LEMMA']
        LEXICON = pd.concat([LEXICON, LEXICON_]) if LEXICON is not None else LEXICON_
        
        BACKOFF_['BW'] = BACKOFF_['FORM'] + '/' + BACKOFF_['BW']
        BACKOFF = pd.concat([BACKOFF, BACKOFF_]) if BACKOFF is not None else BACKOFF_

    BACKOFF = BACKOFF if len(BACKOFF.index) != 0 else None


    # Process POSTREGEX sheet
    #TODO: make sure that symbols being compiled into regex are not mistaken for special characters.
    # This was already done but does not cover unlikely cases.
    # Compiles the regex match expression from the sheet into a regex match expression that is
    # suitable for storing into the DB in Arabic script. Expects Safe BW transliteration in the sheet.
    POSTREGEX = None
    postregex_path: Optional[str] = local_specs['specs']['sheets'].get('postregex')
    if postregex_path:
        POSTREGEX = pd.read_csv(os.path.join(data_dir, 'PostRegex.csv'))
        POSTREGEX = POSTREGEX[(POSTREGEX.DEFINE == 'POSTREGEX') & (POSTREGEX.VARIANT == local_specs['dialect'].upper())]
        POSTREGEX = POSTREGEX.replace(nan, '', regex=True)
        for i, row in POSTREGEX.iterrows():
            POSTREGEX.at[i, 'MATCH'] = _bw2ar_regex(row['MATCH'], safebw2ar)
            POSTREGEX.at[i, 'REPLACE'] = _bw2ar_regex(
                ''.join(re.sub(r'\$', r'\\', row['REPLACE'])), safebw2ar)
    
    # Useful for debugging of nominals, since many entries are inconsistent in terms of their COND-T
    # This splits rows with or-ed COND-T expressions into 
    #FIXME: does not handle cases when there is a conjunction with or-ed in COND-T
    if local_specs.get('split_or') == True:
        assert SMART_BACKOFF is None
        LEXICON_ = []
        for _, row in LEXICON.iterrows():
            if '||' in row['COND-T'] and ' ' not in row['COND-T']:
                for term in row['COND-T'].split('||'):
                    row_ = row.to_dict()
                    row_['COND-T'] = term
                    LEXICON_.append(row_)
            else:
                LEXICON_.append(row.to_dict())
        LEXICON = pd.DataFrame(LEXICON_)

    exclusions: List[str] = local_specs['specs'].get('exclude', [])
    
    # Process ORDER sheet
    ORDER = ORDER[ORDER.DEFINE == 'ORDER']  # skip comments & empty lines
    ORDER = ORDER.replace(nan, '', regex=True)
    for exclusion in exclusions:
        ORDER = ORDER[~ORDER.EXCLUDE.str.contains(exclusion)]

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
    MORPH = process_morph_specs(MORPH, exclusions)

    # cont'd: Process LEXICON sheet
    LEXICON['GLOSS'] = LEXICON['GLOSS'].replace('\s+', '#', regex=True)
    
    # Get rid of unused conditions
    LEXICON['COND-S'] = LEXICON.apply(
        lambda row: re.sub(r'hamzated|hollow|defective', '', row['COND-S']), axis=1)

    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' +', ' ', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' $', '', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' +', ' ', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' $', '', regex=True)

    SHEETS = dict(about=ABOUT, header=HEADER, order=ORDER, morph=MORPH,
                  lexicon=LEXICON, smart_backoff=SMART_BACKOFF, postregex=POSTREGEX,
                  backoff=BACKOFF)
    
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
    the same morpheme because they have the same CLASS and FUNC values. The rows are actual
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
    # Skip commented rows
    MORPH = MORPH[MORPH.DEFINE == 'MORPH']
    MORPH = MORPH.replace(nan, '', regex=True)
    MORPH = MORPH.replace('^\s+', '', regex=True)
    MORPH = MORPH.replace('\s+$', '', regex=True)
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
    MORPH_CLASSES = MORPH.CLASS.unique()
    MORPH_CLASSES = MORPH_CLASSES[MORPH_CLASSES != '_']
    MORPH_MORPHEMES = MORPH.FUNC.unique()
    MORPH_MORPHEMES = MORPH_MORPHEMES[MORPH_MORPHEMES != '_']
    # TODO: inefficient, should go through only valid morphemes, it is not necessary
    # to search for those, simply iterate over them as they are in MORPH sheet.
    for CLASS in MORPH_CLASSES:
        for MORPHEME in MORPH_MORPHEMES: 
            # Get the unique list of the true conditions in all the allomorphs.
            # We basically want to convert the 'COND-T' column in to n columns
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
                    # similar general conditions. Take the unique of what is now col[0] and remove the fluff.
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
        MORPH = MORPH[~MORPH.EXCLUDE.str.contains(exclusion)]
    
    return MORPH


def get_clean_set(cond_col: pd.Series) -> Set[str]:
    """Cleans up the conjunction of terms in COND-T field, keeping only valid conditions (not else or empty)."""
    morph_cond_t = cond_col.tolist()
    morph_cond_t = [
        y for y in morph_cond_t if y not in ['', '_', 'else', None]]
    morph_cond_t = set(morph_cond_t)

    return morph_cond_t


def get_morph_cond_f(morph_cond_t: pd.Series,
                     cond_t_current: Set[str],
                     MORPH: pd.DataFrame) -> pd.DataFrame:
    """Get COND-F based on the RoA (see defition in documentation of `process_morph_specs()`"""
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
    """ Converts regex expression from the sheet to Arabic while taking care not to convert characters
    which are special regex characters in the process. This expects the input not to be ambiguous e.g.,
    Safe BW"""
    match_ = []
    for match in re.split(r'(\\.|[\|}{\*\$_])', regex):
        match = match if re.match(r'(\\.)|[\|}{\*\$_]', match) else bw2ar(match)
        match_.append(match)
    return ''.join(match_)
