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
from tqdm import tqdm
import argparse
import itertools
from time import strftime, gmtime, process_time
from functools import partial
import cProfile, pstats
import sys
from typing import Dict, Tuple, List, Optional
import importlib
import pickle

import pandas as pd

try:
    # Needed for when db_maker() needs to be imported by another script
    from . import db_maker_utils
    from .utils.utils import essential_keys_form_feats, Config
    from .debugging.download_sheets import download_sheets
except:
    # Needed for when db_maker() is ran from within this script
    import db_maker_utils
    from utils.utils import essential_keys_form_feats, Config
    from debugging.download_sheets import download_sheets

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-output_dir", default='',
                    type=str, help="Path of the directory to output the DBs to.")
parser.add_argument("-debug_lemma", default='',
                    type=str, help="Only keep specified lemma in the lexicon (for debugging).")
parser.add_argument("-download", default=False,
                    action='store_true', help="Whether or not to download the data before doing anything. This should be done in case data was changed in Google Sheets since last time this script was ran.")
parser.add_argument("-run_profiling", default=False,
                    action='store_true', help="Run execution time profiling for the make_db().")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")
args, _ = parser.parse_known_args()

config = Config(args.config_file, args.config_name)

if args.camel_tools == 'local':
    sys.path.insert(0, config.camel_tools)

from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_bw
from camel_tools.morphology.utils import strip_lex


normalize_map = CharMapper({
    '<': 'A',
    '>': 'A',
    '|': 'A',
    '{': 'A',
    'Y': 'y'
})

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

_required_verb_stem_feats = ['pos', 'asp', 'per', 'gen', 'num', 'vox', 'mod']
# vox needs to be added even for nominals, otherwise the generator would throw errors
_required_nom_stem_feats = ['pos', 'form_gen', 'form_num', 'gen', 'num', 'stt', 'cas', 'vox']
_clitic_feats = ['enc0', 'enc1', 'enc2', 'prc0', 'prc1', 'prc1.5', 'prc2', 'prc3']

PRE_POST_REGEX_SYMBOL = re.compile(r'[#@]')
PRE_POST_REGEX_SYMBOL_SMARTBACKOFF = re.compile(r'^\^|\$$|[#@]')
HAMZA_WASL_RE = re.compile(r'\{')
CAPHI_UNDERSCORE_RE_1 = re.compile(r'_+')
CAPHI_UNDERSCORE_RE_2 = re.compile(r'^_|_$')
SEG_TOK_SCHEMES = ['D3SEG', 'D3TOK', 'ATBSEG', 'ATBTOK']

"""
Useful terms to know for a better understanding of the comments:
    - Spreadsheet: a collection of sheets
    - Class (or CLASS or morpheme class): a label given to a morpheme linking it to a broad
    grouping of other morphemes which is meant to define how they are positioned (order)
    relative to other morphemes in different classes. Only morphemes of the different classes
    can be concatenated together.
    - Function: general tag restricting a morpheme to specific functional grammatical features
    - Form (or surface form or diac): the surface form or realization of a specific morpheme.
    - Morpheme: defined by its (CLASS, FUNC) tuple, i.e., there are as many
    morphemes in our specifications as there are (CLASS, FUNC) tuples. Note: when we say morpheme,
    for the purpose of this camel_morph, we also mean the buffer "morphemes" even though they are
    not technically morphemes in the linguistic sense.
    - Allomorph: defined by its (CLASS, FUNC, FORM).
    - Affix: suffix or prefix.
    - Order: concatenative order of morphemes based on classes.
    - Complex morpheme: in the ORDER sheet, the content of the three PREFIX (A),
    STEM (B), SUFFIX (C) columns defines the mandatory order of morphemes within
    these three fields. Since a prefix/suffix can be made of more than one morpheme,
    then we call this a complex morheme.
    - Complex morpheme class: different from Class (above), it is the concatenation along the
    COND-S, COND-T, and COND-F of all the conditions of the morphemes forming the complex morpheme.
    In a sence, it is a signature of how this complex morpheme behaves with other morphemes.
    - Match: match field used to retrieve words from a database once it is compiled
    - Category: label given to a valid complex morpheme, that gives information about its compatibility
    with other complex morphemes
"""

def make_db(config:Config, output_path:Optional[str]=None):
    """
    Main function which takes in a set of specifications from `csv` files (downloadable
    from Google Sheets) and which, from the latter, prints out a `db` file in the ALMOR format,
    useable by the Camel Tools Analyzer and Generator engines to produce word analyses/generations.
    The config file is any json object which at the highest level contains: (1) global
    specifications, i.e., that apply no matter what the present local configuration contains;
    (2) local specifications, from which we source all the details that are specific to the current
    DB we are trying to build. The format of these two is dictated by the current code and how
    it reads that information. Any changes to the format should be accompanied by changes to the code
    and vice versa.

    Args:
        config (Config): dictionary containing all the necessary information to build the `db` file.
        output_path (str): path of the output DB. Defaults to None
    """
    if args.download:
        print()
        download_sheets(config=config)
    
    morph2caphi = None
    if config.caphi is not None:
        caphi_module = importlib.import_module(config.caphi)
        morph2caphi = {morph_type: getattr(caphi_module, f'caphi_{morph_type}')
                       for morph_type in ['DBPrefix', 'DBStem', 'DBSuffix']}
    
    logprob: Dict[str] = config.logprob
    if logprob is not None:
        with open(logprob, 'rb') as f:
            logprob = pickle.load(f)
        pos2lex2logprob = {}
        for (pos, lex), logprob_ in logprob['pos_lex'].items():
            pos2lex2logprob.setdefault(pos, {}).setdefault(lex, logprob_)
        
        for pos, lex2logprob in pos2lex2logprob.items():
            pos2lex2logprob[pos] = dict(sorted(
                lex2logprob.items(), key=lambda x: x[1], reverse=True))
        logprob['pos2lex2logprob'] = pos2lex2logprob
    
    c0 = process_time()
    
    print("\nLoading and processing sheets... [1/4]")
    SHEETS, cond2class = db_maker_utils.read_morph_specs(config)

    if args.debug_lemma:
        SHEETS['lexicon'] = SHEETS['lexicon'][
            SHEETS['lexicon']['LEMMA'] == f'lex:{args.debug_lemma}']
    
    print("\nValidating combinations... [2/4]")
    cat2id: bool = config.cat2id if config.cat2id is not None else False
    defaults: bool = config.defaults if config.defaults is not None else True
    
    db = construct_almor_db(SHEETS, config.pruning,
        cond2class, cat2id, defaults, morph2caphi, logprob)

    print("\nCollapsing categories and reindexing... [3/4]")
    reindex: bool = config.reindex if config.reindex is not None else False
    if reindex:
        db, _ = collapse_and_reindex_categories(db, collapse_morphemes=False)
    
    print("\nGenerating DB file... [4/4]")
    if output_path is None:
        output_path = config.get_db_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print_almor_db(output_path, db)
    
    c1 = process_time()
    print(f"\nTotal time required: {strftime('%M:%S', gmtime(c1 - c0))}")
    return SHEETS


def construct_almor_db(SHEETS:Dict[str, pd.DataFrame],
                       pruning:bool,
                       cond2class:Dict,
                       cat2id:bool=False,
                       defaults:Optional[bool]=None,
                       morph2caphi:Optional[Dict]=None,
                       logprob:Optional[Dict]=None) -> Dict:
    """
    Function which takes care of the condition validation process, i.e., deciding which
    (complex) morphemes are compatible, and prints them and their computed categories in
    ALMOR format.

    Args:
        SHEETS (Dict[str, pd.DataFrame]): dictionary which contains the 7 main dataframes which will 
        be used throughout the DB making process.
        pruning (bool): whether or not to perform pruning which is the preprocessing step of determining
        which morphemes are not compatible as a combination of complex morpheme, to reduce the number
        of complex morphemes which will then in turn be validated.
        cond2class (Dict): inventory of condition definitions and their corresponding vectors which will
        be useful in the pruning process.
        cat2id (bool): whether or not to convert the category names to IDs (makes them smaller,
        and thus makes the DB file size smaller, but eliminates the debug info contained in them).
        defaults (bool): whether or not to add defaults per POS for the DB lines. Defaults to None.
        morph2caphi (Dict): maps to the different methods to use to convert diac to CAPHI based
        on the complex morpheme type. Defaults to None.
        logprob (Dict): dictionary containing the log probablities of different features, extracted
        from a corpus. Defaults to None.

    Returns:
        Dict: Database which contains entries (values) for each section (keys).
    """
    ORDER, MORPH, LEXICON = SHEETS['order'], SHEETS['morph'], SHEETS['lexicon']
    ABOUT, HEADER, POSTREGEX = SHEETS['about'], SHEETS['header'], SHEETS['postregex']
    BACKOFF, SMART_BACKOFF = SHEETS['backoff'], SHEETS['smart_backoff']

    #TODO: classes are compiled into the category name of each morpheme, hence, if the class
    # names are long, then categories will be long and this will reduce readability of the 
    # latter (for debugging purposes). Therefore we rely on short names, the mapping of which
    # is currently provided manually from within the same order sheet. Should try to provide these
    # short names automatically as adding them manually is problematic because currently, we need to 
    # add a short name for each class field (PREIFX, STEM, SUFFIX columns). Also, we need to make sure
    # that every cell in the class fields (in the order file) containaining a specific order of classes
    # should have the same short name for that order. Hence, this needs to be done manually for the moment,
    # which is why we should consider it being done automatically, as this is often the source of bugs,
    # for example if some order is changed in the class fields and forgetting to change the associated
    # short name.
    short_cat_maps = None
    if {'PREFIX-SHORT', 'STEM-SHORT', 'SUFFIX-SHORT'} <= set(ORDER.columns):
        short_cat_maps = _get_short_cat_name_maps(ORDER) 

    # One-time filling of the About, Header, and PostRegex sections of the DB
    db = {}
    db['OUT:###ABOUT###'] = list(ABOUT['CONTENT'])
    if POSTREGEX is not None:
        db['OUT:###POSTREGEX###'] = [
            'MATCH\t' + '\t'.join([match for match in POSTREGEX['MATCH'].values.tolist()]),
            'REPLACE\t' + '\t'.join([replace for replace in POSTREGEX['REPLACE'].values.tolist()])
        ]
    
    header_, defaults_ = _read_header_file(HEADER)
    db['OUT:###HEADER###'] = header_

    defaults_ = defaults_ if defaults else None
    cat2id = {} if cat2id else None
    
    def construct_process(lexicon: pd.DataFrame,
                          order: pd.DataFrame,
                          cmplx_stem_memoize: Dict[str, str],
                          stems_section_title: str):
        """ Process which is ran for each ORDER line, in which plausible complex morphemes
        are generated and then tested (validated) against each other across the prefix/stem/suffix
        boundary. Complex prefixes/stems/suffixes which are compatible with each other are added as
        entries in the DB. 
        """
        # Complex morphemes generation (within the prefix/stem/suffix boundary)
        cmplx_prefix_classes = gen_cmplx_morph_combs(
            order['PREFIX'], MORPH, lexicon, cond2class,
            pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
        cmplx_suffix_classes = gen_cmplx_morph_combs(
            order['SUFFIX'], MORPH, lexicon, cond2class,
            pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
        cmplx_stem_classes = gen_cmplx_morph_combs(
            order['STEM'], MORPH, lexicon, cond2class,
            cmplx_morph_memoize=cmplx_stem_memoize,
            pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
        
        cmplx_type_empty = set()
        if not cmplx_stem_classes: cmplx_type_empty.add('Stem')
        if not cmplx_suffix_classes: cmplx_type_empty.add('Suffix')
        if not cmplx_prefix_classes: cmplx_type_empty.add('Prefix')
        if cmplx_type_empty:
            cmplx_type_empty = '/'.join(cmplx_type_empty)
            tqdm.write((f"WARNING: {order['SUFFIX-SHORT']}: {cmplx_type_empty} class " 
                        'is empty; proceeding to process next order line.'))
            return db
        
        cmplx_morph_classes = dict(
            cmplx_prefix_classes=(cmplx_prefix_classes, order['PREFIX'] if order['PREFIX'] else '[EMPTY]'),
            cmplx_suffix_classes=(cmplx_suffix_classes, order['SUFFIX'] if order['SUFFIX'] else '[EMPTY]'),
            cmplx_stem_classes=(cmplx_stem_classes, order['STEM']))
        
        # Complex morphemes validation or word generation (across the prefix/stem/suffix boundary)
        db_ = cross_cmplx_morph_validation(
            cmplx_morph_classes, order['CLASS'].lower(), short_cat_maps, defaults_,
            stems_section_title, cat2id, morph2caphi, logprob)
        for section, contents in db_.items():
            # if 'BACKOFF' in stems_section_title and section != stems_section_title:
            #     assert set(contents) <= set(db[section])
            if section != 'OUT:###STEMBACKOFF###':
                db.setdefault(section, {}).update(contents)
            else:
                for backoff_mode, cats in contents.items():
                    db.setdefault(section, {}).setdefault(backoff_mode, set()).update(cats)
    
    # For memoization to work as intended, same-aspect order lines should be placed next
    # to each other in the ORDER file, and since the stem part of the order usually stays
    # the same at the aspect level, then it makes sense to avoid recomputing all the combinations
    # each time and same them in the memo. dict.
    for name, SHEET in [('Concrete', LEXICON), ('Backoff', BACKOFF)]:
        if SHEET is not None:
            print(f'\n{name} lexicon')
            pbar = tqdm(total=len(list(ORDER.iterrows())))
            cmplx_stem_memoize = {}
            order_stem_prev = ''
            for _, order in ORDER.iterrows():
                col = 'SUFFIX-SHORT' if 'SUFFIX-SHORT' in ORDER.columns else 'SUFFIX'
                pbar.set_description(order[col])
                if order['STEM'] != order_stem_prev:
                    cmplx_stem_memoize = {}
                    order_stem_prev = order['STEM']
                construct_process(SHEET, order, cmplx_stem_memoize,
                                stems_section_title='OUT:###STEMS###')
                pbar.update(1)
            pbar.close()

    stem_backoffs_ = {}
    if 'OUT:###STEMBACKOFF###' in db:
        for backoff_mode, cats in db['OUT:###STEMBACKOFF###'].items():
            stem_backoffs_[('STEMBACKOFF', backoff_mode, ' '.join(cats))] = 1
    db['OUT:###STEMBACKOFF###'] = stem_backoffs_
            
    #TODO: maybe this should also be included in the above loop, but more study is needed
    if SMART_BACKOFF is not None:
        print('Smart Backoff lexicon')
        pbar = tqdm(total=len(list(ORDER.iterrows())))
        for _, order in ORDER.iterrows():
            pbar.set_description(order['SUFFIX-SHORT'])
            construct_process(SMART_BACKOFF, order, {},
                              stems_section_title='OUT:###SMARTBACKOFF###')
            pbar.update(1)
        pbar.close()

    return db

def cross_cmplx_morph_validation(cmplx_morph_classes: Dict,
                                 pos_type: str,
                                 short_cat_maps: Optional[Dict]=None,
                                 defaults: Dict=None,
                                 stems_section_title: str='OUT:###STEMS###',
                                 cat2id:Optional[Dict]=None,
                                 morph2caphi:Optional[Dict]=None,
                                 logprob:Optional[Dict]=None) -> Dict:
    """Method which takes in classes of complex morphemes, and validates them against each other
    in a three-loop fashion, one for each of prefix, stem, and suffix. Instead of going over all
    individual combinations, we loop over "classes" of them (since all combinations belonging to
    the same class behave similarly from a validation POV and it is much less costly to do so),
    and whenever a combination is validated, all the individual combinations belonging to this
    class are added to the DB (once).

    Args:
        cmplx_morph_classes (Dict): keys are unique classes of condition combinations (3-tuple of 3-tuple),
        (PREF(COND-S,T,F), STEM(COND-S,T,F), SUFF(COND-S,T,F)) and values are all the combinations that have these conditions.
        pos_type (str): 'nominal' or 'verbal' to choose the default values of features for DB (from Header).
        short_cat_maps (Optional[Dict], optional): mapping from the actual class name (in PREFIX,
        STEM, or SUFFIX column or ORDER) to its short name (PREFIX-SHORT, STEM-SHORT, and
        SUFFIX-SHORT). Defaults to None.
        defaults (Dict, optional): default values of features for DB (from Header). Defaults to None.
        stems_section_title (_type_, optional): title of the section that will appear in the DB. Defaults to 'OUT:###STEMS###'.
        morph2caphi (Dict): maps to the different methods to use to convert diac to CAPHI based
        on the complex morpheme type. Defaults to None.
        logprob (Dict): dictionary containing the log probablities of different features, extracted
        from a corpus. Defaults to None.

    Returns:
        Dict: Database in progress
    """
    db = {}
    db['OUT:###STEMBACKOFF###'] = {}
    db['OUT:###PREFIXES###'] = {}
    db['OUT:###SUFFIXES###'] = {}
    db[stems_section_title] = {}
    db['OUT:###TABLE AB###'] = {}
    db['OUT:###TABLE BC###'] = {}
    db['OUT:###TABLE AC###'] = {}

    cmplx_prefix_classes, cmplx_prefix_seq = cmplx_morph_classes['cmplx_prefix_classes']
    cmplx_suffix_classes, cmplx_suffix_seq = cmplx_morph_classes['cmplx_suffix_classes']
    cmplx_stem_classes, cmplx_stem_seq = cmplx_morph_classes['cmplx_stem_classes']
    
    cat_memoize = {'stem': {}, 'suffix': {}, 'prefix': {}}
    compatibility_memoize = {}
    for cmplx_stem_cls, cmplx_stems in cmplx_stem_classes.items():
        # `cmplx_stem_cls` = (cmplx_stem['COND-S'], cmplx_stem['COND-T'], cmplx_stem['COND-F'])
        # All entries in `cmplx_stems` have the same cat
        stem_cond_s = ' '.join([f['COND-S'] for f in cmplx_stems[0]])
        stem_cond_t = ' '.join([f['COND-T'] for f in cmplx_stems[0]])
        stem_cond_f = ' '.join([f['COND-F'] for f in cmplx_stems[0]])

        for cmplx_prefix_cls, cmplx_prefixes in cmplx_prefix_classes.items():
            prefix_cond_s = ' '.join([f['COND-S'] for f in cmplx_prefixes[0]])
            prefix_cond_t = ' '.join([f['COND-T'] for f in cmplx_prefixes[0]])
            prefix_cond_f = ' '.join([f['COND-F'] for f in cmplx_prefixes[0]])

            for cmplx_suffix_cls, cmplx_suffixes in cmplx_suffix_classes.items():
                suffix_cond_s = ' '.join([f['COND-S'] for f in cmplx_suffixes[0]])
                suffix_cond_t = ' '.join([f['COND-T'] for f in cmplx_suffixes[0]])
                suffix_cond_f = ' '.join([f['COND-F'] for f in cmplx_suffixes[0]])

                valid = check_compatibility(' '.join([prefix_cond_s, stem_cond_s, suffix_cond_s]),
                                            ' '.join([prefix_cond_t, stem_cond_t, suffix_cond_t]),
                                            ' '.join([prefix_cond_f, stem_cond_f, suffix_cond_f]),
                                            compatibility_memoize)
                if valid:
                    stem_cat, prefix_cat, suffix_cat = None, None, None
                    update_info_stem = dict(pos_type=pos_type,
                                            cmplx_morph_seq=cmplx_stem_seq,
                                            cmplx_morph_cls=cmplx_stem_cls,
                                            cmplx_morph_type='stem',
                                            cmplx_morphs=cmplx_stems,
                                            conditions=(stem_cond_s, stem_cond_t, stem_cond_f),
                                            db_section=stems_section_title)
                    update_info_prefix = dict(pos_type=pos_type,
                                              cmplx_morph_seq=cmplx_prefix_seq,
                                              cmplx_morph_cls=cmplx_prefix_cls,
                                              cmplx_morph_type='prefix',
                                              cmplx_morphs=cmplx_prefixes,
                                              conditions=(prefix_cond_s, prefix_cond_t, prefix_cond_f),
                                              db_section='OUT:###PREFIXES###')
                    update_info_suffix = dict(pos_type=pos_type,
                                              cmplx_morph_seq=cmplx_suffix_seq,
                                              cmplx_morph_cls=cmplx_suffix_cls,
                                              cmplx_morph_type='suffix',
                                              cmplx_morphs=cmplx_suffixes,
                                              conditions=(suffix_cond_s, suffix_cond_t, suffix_cond_f),
                                              db_section='OUT:###SUFFIXES###')
                    
                    for update_info in [update_info_stem, update_info_prefix, update_info_suffix]:
                        update_db(db, update_info, cat_memoize, short_cat_maps, defaults, cat2id,
                                  morph2caphi, logprob)
                    # If morph class cat has already been computed previously, then cat is still `None`
                    # (because we will not go again in the morph for loop) and we need to retrieve the
                    # computed value. 
                    # FIXME: stem_cat seems to always be None at this point, so there is no need for
                    # the if statement 
                    stem_cat = stem_cat if stem_cat else cat_memoize['stem'][cmplx_stem_cls]
                    prefix_cat = prefix_cat if prefix_cat else cat_memoize['prefix'][cmplx_prefix_cls]
                    suffix_cat = suffix_cat if suffix_cat else cat_memoize['suffix'][cmplx_suffix_cls]

                    db['OUT:###TABLE AB###'][(prefix_cat, stem_cat)] = 1
                    db['OUT:###TABLE BC###'][(stem_cat, suffix_cat)] = 1
                    db['OUT:###TABLE AC###'][(prefix_cat, suffix_cat)] = 1
    # Turn this on to make sure that every entry is only set once (can also be used to catch
    # double entries in the lexicon sheets)
    # assert [1 for items in db.values() for item in items if item != 1] == []
    return db

def update_db(db: Dict,
              update_info: Dict,
              cat_memoize: Dict,
              short_cat_maps: Optional[Dict]=None,
              defaults: Optional[Dict]=None,
              cat2id:Optional[Dict]=None,
              morph2caphi:Optional[Dict]=None,
              logprob:Optional[Dict]=None):
    """If a combination of complex prefix/suffix/stem is valid, then each of the complex morphemes
    in that combination will be added as an entry in the DB by this method. Default feature values
    are taken from the Header sheet and are assigned to features which are set to appear in the
    analysis (DB entry). Since the outer loop in `cross_cmplx_morph_validation()` only validates
    at the complex morpheme class level (and not at the individual complex morpheme), we then need
    to add all of the complex morphemes belonging to that class to the DB. Because complex stems
    often share compatibility with complex prefixes/suffixes, there is no reason to overwrite
    the suffixes/prefixes/stems repetitively, so we keep track of which classes have already been
    added to avoid adding them again (which is costly).

    Args:
        db (Dict): database in progress
        update_info (Dict): keys are:
        - `pos_type` ('nominal' or 'verbal')
        - `cmplx_morph_seq` (space-separated classes forming the complex morpheme, e.g., [STEM-PV] [PVBuff])
        - `cmplx_morph_cls` n-tuple of 3-tuples M1(COND-S,T,F), M2(COND-S,T,F), ... s.t. `n` is
        `len(cmplx_morph_seq).split()`, in other words, it is the number of morphemes in the complex morpheme.
        - `cmplx_morph_type` ('prefix', 'stem', or 'suffix')
        - `cmplx_morphs` (list of list of dataframes <-> examples for complex morphemes in the order of `cmplx_morph_seq`)
        - `conditions` (3-tuple of the complex morpheme (COND-S, COND-T, COND-F), with
        space-separated conditions), and `db_section`.
        cat_memoize (Dict): dictionary keeping track of which complex morpheme categories have already been
        added to the DB.
        short_cat_maps (Optional[Dict], optional): mapping from the actual class name (in PREFIX, STEM, or
        SUFFIX column or ORDER) to its short name (PREFIX-SHORT, STEM-SHORT, and SUFFIX-SHORT). If not
        specified, the actuall class name is used Defaults to None.
        defaults (Optional[Dict], optional): default values of features parsed from the Header sheet (same ones
        which usually appear in the beginning of any DB file). They are used to specify feature values for DB entries
        for features whose value was not specified in the sheets. Defaults to None.
        morph2caphi (Dict): maps to the different methods to use to convert diac to CAPHI based
        on the complex morpheme type. Defaults to None.
        logprob (Dict): dictionary containing the log probablities of different features, extracted
        from a corpus. Defaults to None.
    """
    cmplx_morph_seq = update_info['cmplx_morph_seq']
    cmplx_morph_cls = update_info['cmplx_morph_cls']
    cmplx_morph_type = update_info['cmplx_morph_type']
    cmplx_morphs = update_info['cmplx_morphs']
    cond_s, cond_t, cond_f = update_info['conditions']
    db_section = update_info['db_section']
    
    if cmplx_morph_type == 'stem':
        short_cat_map = short_cat_maps['stem'] if short_cat_maps is not None else None
        _generate = _generate_stem
    elif cmplx_morph_type in ['prefix', 'suffix']:
        short_cat_map = short_cat_maps['prefix' if cmplx_morph_type == 'prefix' else 'suffix'] \
                            if short_cat_maps is not None else None
        _generate = partial(_generate_affix, cmplx_morph_type)
    else:
        raise NotImplementedError

    required_feats = _choose_required_feats(update_info['pos_type'])
    # This if statement implements early stopping which entails that if we have already 
    # logged a specific prefix/stem/suffix entry, we do not need to do it again. Entry
    # generation (and more specifically `dediac()`) is costly.
    if cat_memoize[cmplx_morph_type].get(cmplx_morph_cls) is None:
        for cmplx_morph in cmplx_morphs:
            morph_entry = _generate(
                cmplx_morph_seq, required_feats, cmplx_morph, cond_s, cond_t, cond_f,
                short_cat_map, defaults if defaults != False else None,
                cat2id, morph2caphi, logprob)
            if defaults != False:
                morph_entry_analysis_str = ' '.join(f"{k}:{morph_entry['analysis'][k]}"
                    for k in defaults['order'] if morph_entry['analysis'].get(k) is not None)
            else:
                morph_entry_analysis_str = ' '.join(f"{k}:{v if v is not None else ''}"
                    for k, v in  morph_entry['analysis'])
            morph_entry_ = tuple(morph_entry[x] for x in ['match', 'cat']) + (morph_entry_analysis_str,)
            db[db_section].setdefault(morph_entry_, 0)
            db[db_section][morph_entry_] += 1

            if morph_entry['match'] == 'NOAN':
                for backoff_mode in morph_entry['analysis']['backoff_modes'].split():
                    db['OUT:###STEMBACKOFF###'].setdefault(backoff_mode, set()).add(morph_entry['cat'])
        cat_memoize[cmplx_morph_type][cmplx_morph_cls] = morph_entry['cat']


def _generate_cat_field(cmplx_morph_type: str, cmplx_morph_class: str,
                cmplx_morph_cond_s: str, cmplx_morph_cond_t: str, cmplx_morph_cond_f: str,
                short_cat_map: Optional[Dict]=None,
                cat2id:Optional[Dict]=None):
    """This function creates the category for matching using classes and conditions"""
    if short_cat_map:
        cmplx_morph_class = short_cat_map[cmplx_morph_class]
    cmplx_morph_cond_s = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_s.split()) if cond != '_'])
    cmplx_morph_cond_s = cmplx_morph_cond_s if cmplx_morph_cond_s else '-'
    cmplx_morph_cond_t = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_t.split()) if cond != '_'])
    cmplx_morph_cond_t = cmplx_morph_cond_t if cmplx_morph_cond_t else '-'
    cmplx_morph_cond_f = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_f.split()) if cond != '_'])
    cmplx_morph_cond_f = cmplx_morph_cond_f if cmplx_morph_cond_f else '-'
    cat = f"{cmplx_morph_type}:{cmplx_morph_class}_[CS:{cmplx_morph_cond_s}]_[CT:{cmplx_morph_cond_t}]_[CF:{cmplx_morph_cond_f}]"
    if cat2id is not None:
        cat2id_morph_type = cat2id.setdefault(cmplx_morph_type, {})
        if cat in cat2id_morph_type:
            cat = cat2id_morph_type[cat]
        else:
            cat_ = f'{cmplx_morph_type}{str(len(cat2id_morph_type) + 1).zfill(5)}'
            cat2id[cmplx_morph_type][cat] = cat_
            cat = cat_
    return cat

def _convert_bw_tag(bw_tag:str, backoff:bool=False):
    """Create complex BW tag"""
    if bw_tag == '':
        return bw_tag
    bw_elements = bw_tag.split('+')
    utf8_bw_tag = []
    for element in bw_elements:
        parts = element.split('/')
        if 'null' in parts[0]:
            bw_lex = parts[0]
        else:
            bw_lex = parts[0] if backoff else bw2ar(parts[0])
        bw_pos = parts[1]
        utf8_bw_tag.append('/'.join([bw_lex, bw_pos]))
    return '+'.join(utf8_bw_tag)

def _generate_match_field(diac):
    #NOTE: For EGY nominals, postregex symbol is @ while for verbs it is #
    #NOTE: Maybe this is unnecessary and EGY can use #; should look into this
    diac_ = PRE_POST_REGEX_SYMBOL.sub('', diac)
    diac_ = diac_.replace('_', '')
    diac_ = dediac_bw(diac_)
    diac_ = normalize_teh_marbuta_bw(diac_)
    diac_ = normalize_alef_maksura_bw(diac_)
    diac_ = normalize_alef_bw(diac_)
    return diac_


def _generate_caphi(morpheme, caphi_list, caphi_copy, morph2caphi, cmplx_morpheme_type):
    copy_list = [m[caphi_copy] for m in morpheme]
    if len(set(caphi_list)) > 1 and '' in caphi_list:
        raise NotImplementedError
    # TODO: implemented but not debugged
    value = []
    if set(caphi_list) == {''} and morph2caphi is not None:
        value.append(morph2caphi[cmplx_morpheme_type](''.join(copy_list)))
    else:
        for i, v in enumerate(caphi_list):
            if v == '_':
                continue
            elif v:
                value.append(v)
            else:
                if copy_list[i] and copy_list[i] != '_':
                    if morph2caphi is not None:
                        value.append(morph2caphi[cmplx_morpheme_type](copy_list[i]))
    value = ' '.join(value).strip().replace(' ', '_')
    value = CAPHI_UNDERSCORE_RE_1.sub('_', value)
    value = CAPHI_UNDERSCORE_RE_2.sub('', value)
    return value


def _generate_affix(affix_type: str,
                    cmplx_morph_seq: str,
                    required_feats: List[str],
                    affix: List[Dict],
                    affix_cond_s: str, affix_cond_t: str, affix_cond_f: str,
                    short_cat_map: Optional[Dict]=None,
                    defaults: Dict=None,
                    cat2id:Optional[Dict]=None,
                    morph2caphi:Optional[Dict]=None,
                    logprob:Optional[Dict]=None) -> Dict[str, str]:
    """From the CamelMorph specifications, loads the affix information
    of multiple morphemes appearing in the prefix/suffix portion of the order line
    and which are deemed to be compatible with each other to form a complex affix, and
    generates from them the 3 fields needed to store the complex affix as an entry
    in the DB, namely, (1) the match field, (2) the category field, and (3) the analysis.

    Args:
        affix_type (str): 'prefix' or 'suffix'
        cmplx_morph_seq (str): space-separated sequence of classes that predefine the
        order of the morphemes to be assembled for the cartesian product.
        required_feats (List[str]): features that should be included in the analysis. Not used here.
        affix (List[Dict]): individual analyses (dict) of the morphemes in the complex affix.
        affix_cond_s (str): COND-S of complex affix (concat of COND-S of individual morphemes)
        affix_cond_t (str): COND-T of complex affix (concat of COND-T of individual morphemes)
        affix_cond_f (str): COND-F of complex affix (concat of COND-F of individual morphemes)
        short_cat_map (Optional[Dict], optional): mapping from the actual class name (in PREFIX,
        STEM, or SUFFIX column or ORDER) to its short name (PREFIX-SHORT, STEM-SHORT, and
        SUFFIX-SHORT). Defaults to None.
        defaults (Dict, optional): default values of features parsed from the Header sheet. 
        Not used here. Defaults to None.
        backoff (bool): whether or not to add the correct category or just the same category
        to all stem entries. Defaults to False.
        morph2caphi (Dict): maps to the different methods to use to convert diac to CAPHI based
        on the complex morpheme type. Defaults to None.
        logprob (Dict): dictionary containing the log probablities of different features, extracted
        from a corpus. Defaults to None.

    Returns:
        Dict[str, str]: dict containing the 3 fields needed to store the complex affix as an entry in the DB.
    """
    affix_match, analysis = _read_affix(affix, affix_type)
    affix_type = 'P' if affix_type == 'prefix' else 'S'
    acat = _generate_cat_field(affix_type, cmplx_morph_seq, affix_cond_s, affix_cond_t,
                       affix_cond_f, short_cat_map, cat2id)
    analysis['bw'] = _convert_bw_tag(analysis['bw'])
    affix_type_ = 'DBPrefix' if affix_type == 'P' else 'DBSuffix'
    
    for col in SEG_TOK_SCHEMES:
        col = col.lower()
        tok_copy = defaults['tokenization'][col]
        value = ''.join(
            v if v else (affix[i][tok_copy] if affix[i][tok_copy] != '_' else '')
            for i, v in enumerate(analysis[col]))
        analysis[col] = value
    
    analysis['caphi'] = _generate_caphi(
        affix, analysis['caphi'], defaults['transcription']['caphi'], morph2caphi, affix_type_)
    
    for f in ['diac', 'd3seg', 'd3tok', 'atbseg', 'atbtok']:
        analysis[f] = bw2ar(analysis[f])

    affix = {'match': bw2ar(affix_match), 'cat': acat, 'analysis': analysis}
    return affix


def _generate_stem(cmplx_morph_seq: str,
                   required_feats: List[str],
                   stem: List[Dict],
                   stem_cond_s: str, stem_cond_t: str, stem_cond_f: str,
                   short_cat_map: Optional[Dict]=None,
                   defaults: Dict=None,
                   cat2id:Optional[Dict]=None,
                   morph2caphi:Optional[Dict]=None,
                   logprob:Optional[Dict]=None) -> Dict[str, str]:
    """Same as `_generate_affix()` but slightly different.

    Args:
        cmplx_morph_seq (str): space-separated sequence of classes that predefines the
        order of the morphemes to be assembled for the cartesian product.
        required_feats (List[str]): features that should be included in the analysis.
        stem (List[Dict]): individual analyses (dict) of the morphemes in the complex stem.
        stem_cond_s (str): COND-S of complex stem (concat of COND-S of individual morphemes)
        stem_cond_t (str): COND-T of complex stem (concat of COND-T of individual morphemes)
        stem_cond_f (str): COND-F of complex stem (concat of COND-F of individual morphemes)
        short_cat_map (Optional[Dict], optional): _description_. Defaults to None.
        short_cat_map (Optional[Dict], optional): mapping from the actual class name (in PREFIX,
        STEM, or SUFFIX column or ORDER) to its short name (PREFIX-SHORT, STEM-SHORT, and
        SUFFIX-SHORT). Defaults to None.
        defaults (Dict, optional): default values of features parsed from the Header sheet. Defaults to None.
        morph2caphi (Dict): maps to the different methods to use to convert diac to CAPHI based
        on the complex morpheme type. Defaults to None.
        logprob (Dict): dictionary containing the log probablities of different features, extracted
        from a corpus. Defaults to None.

    Returns:
        Dict[str, str]: _description_
    """
    stem_match, analysis, backoff = _read_stem(stem)
    analysis['bw'] = _convert_bw_tag(analysis['bw'], backoff)
    
    if defaults is not None:
        for f in required_feats + _clitic_feats:
            if f not in analysis or analysis[f] == '_':
                analysis[f] = defaults['defaults'][analysis['pos']][f]

    if '-' in analysis['lex'] and analysis['pos'] == 'verb':
        part1, part2 = analysis['lex'].split('-')
        mid_root_diac, underscore = None, None
        if '_' in part2:
            mid_root_diac, underscore = part2.split('_')
        analysis['lex'] = part1 + f'_{underscore}'
        analysis['mid_root_diac'] = mid_root_diac
    
    if backoff == 'smart':
        match = db_maker_utils._bw2ar_regex(stem_match, bw2ar)
    elif backoff == 'vanilla':
        match = stem_match
        analysis['backoff_modes'] = analysis['lex']
        analysis['lex'] = 'NOAN'
    else:
        match = bw2ar(stem_match)

    xcat = _generate_cat_field('X', cmplx_morph_seq, stem_cond_s, stem_cond_t,
                               stem_cond_f, short_cat_map, cat2id)
    
    if not backoff:
        for col in SEG_TOK_SCHEMES:
            col = col.lower()
            tok_copy = defaults['tokenization'][col]
            value = ''.join(
                v if v else (stem[i][tok_copy] if stem[i][tok_copy] != '_' else '')
                for i, v in enumerate(analysis[col]))
            if value != analysis['diac']:
                analysis[col] = value
            else:
                del analysis[col]
        
        analysis['caphi'] = _generate_caphi(
            stem, analysis['caphi'], defaults['transcription']['caphi'], morph2caphi, 'DBStem')
        
        if logprob is not None:
            for f in ['lex', 'pos_lex']:
                lex_ = tuple(analysis[f_]
                             if f_ != 'lex' else strip_lex(analysis[f_])
                             for f_ in f.split('_'))
                logprob_ = f'{logprob[f][lex_]:.6f}' if lex_ in logprob[f] else '-99'
                analysis[f'{f}_logprob'] = logprob_

        bw2ar_columns = ['lex', 'diac', 'cm_stem', 'cm_buffer', 'root',
                         'd3seg', 'd3tok', 'atbseg', 'atbtok',
                         'pattern', 'pattern_abstract']
        for f in bw2ar_columns:
            if f in analysis:
                if analysis[f] == 'NTWS' or analysis[f] is None:
                    continue
                analysis[f] = bw2ar(analysis[f])

    stem = {'match': match, 'cat': xcat, 'analysis': analysis}
    return stem

def _read_affix(affix: List[Dict], affix_type: str) -> Tuple[str, Dict]:
    """From the CamelMorph specifications, loads the affix information
    of multiple morphemes appearing in the prefix/suffix portion of the order line
    and which are deemed to be compatible with each other to form a complex affix, and
    generates from them the fields needed to store the complex affix as an entry
    in the DB.

    Args:
        affix (List[Dict]): individual analyses (dict) of the morphemes in the complex affix.
        affix_type (str): 'prefix' or 'suffix'

    Returns:
        Tuple[str, Dict]: information to store in the DB.
    """
    analysis = {}
    analysis['bw'] = '+'.join(m['BW'] for m in affix if m['BW'] != '_')
    affix_gloss = '+'.join(m['GLOSS'] for m in affix if m['GLOSS'] != '_')
    analysis['gloss'] = affix_gloss if affix_gloss else '_'
    affix_feat = {feat.split(':')[0]: feat.split(':')[1]
                  for m in affix for feat in m['FEAT'].split()}
    analysis = {**analysis, **affix_feat}

    analysis['diac'] = ''.join(m['FORM'] for m in affix if m['FORM'] != '_')
    
    for col in SEG_TOK_SCHEMES + ['CAPHI']:
        analysis[col.lower()] = [m.get(col, '') for m in affix]
    
    source = [m['SOURCE'] for m in affix if m.get('SOURCE')]
    if source and any(source):
        analysis['source'] = source[0]
    else:
        analysis['source'] = 'lex'
    affix_type = 'pref' if affix_type == 'prefix' else 'suff'
    analysis[f'cm_{affix_type}_ids'] = '+'.join(
        m['CLASS'] + ':' + str(int(float(m['LINE'] if m['LINE'] else -1))) for m in affix)

    affix_match = _generate_match_field(analysis['diac'])
    return affix_match, analysis


def _read_stem(stem: List[Dict]) -> Tuple[str, Dict]:
    """Same as `_read_affix()`. Treated slightly differently than affixes which is why it has a
    method of its own.

    Args:
        stem (List[Dict]): individual analyses (dict) of the morphemes in the complex stem.

    Returns:
        Tuple[str, Dict]: information to store in the DB
    """
    analysis = {}
    analysis['bw'] = '+'.join(s['BW'] for s in stem if s['BW'] != '_')
    stem_gloss = '+'.join(s['GLOSS'] for s in stem if 'LEMMA' in s)
    analysis['gloss'] = stem_gloss if stem_gloss else '_'
    analysis['lex'] = '+'.join(
        s['LEMMA'].split(':')[1] for s in stem if 'LEMMA' in s)
    stem_feat = {feat.split(':')[0]: feat.split(':')[1]
                for s in stem for feat in s['FEAT'].split()}
    analysis = {**analysis, **stem_feat}

    analysis['diac'] = ''.join(s['FORM'] for s in stem if s['FORM'] != '_')
    
    for col in ['ROOT', 'PATTERN_ABSTRACT', 'PATTERN', 'SOURCE']:
        feat = [s[col] for s in stem if s.get(col)]
        if feat and any(feat):
            analysis[col.lower()] = feat[0]
        elif col == 'SOURCE':
            analysis['source'] = 'lex'
        else:
            analysis[col.lower()] = 'NTWS'

    analysis['cm_stem_ids'] = '+'.join(
        s['CLASS'] + ':' + str(int(float(s['LINE'] if s['LINE'] else -1)))
        for s in stem)
    analysis['cm_stem'], analysis['cm_buffer'] = stem[0]['FORM'], None
    if len(stem) > 1 and len(stem) == 2 and stem[1]['FORM'] not in ['_', '']:
        analysis['cm_buffer'] = stem[1]['FORM']

    stem_defines = set(s['DEFINE'] for s in stem)
    if 'SMARTBACKOFF' in stem_defines:
        assert stem_defines <= {'MORPH', 'SMARTBACKOFF'}
        backoff = 'smart'
        stem_match = []
        for s in stem:
            if s['FORM'] != '_':
                if s['DEFINE'] == 'SMARTBACKOFF':
                    stem_match.append(
                        PRE_POST_REGEX_SYMBOL_SMARTBACKOFF.sub('', s['MATCH']))
                else:
                    stem_match.append(_generate_match_field(s['FORM']))
        stem_match = f"^{''.join(stem_match)}$"
    elif 'BACKOFF' in  stem_defines:
        backoff = 'vanilla'
        assert stem_defines <= {'MORPH', 'BACKOFF'}
        stem_match = 'NOAN'
    else:
        backoff = None
        stem_match = _generate_match_field(analysis['diac'])
        for col in SEG_TOK_SCHEMES + ['CAPHI']:
            analysis[col.lower()] = [s.get(col, '') for s in stem]

    return stem_match, analysis, backoff


def _read_compatibility_tables(X_Y_compat):
    X_Y_compat_ = {}
    for X_cat, Y_cat in X_Y_compat:
        X_Y_compat_.setdefault(X_cat, set()).add(Y_cat)
    return X_Y_compat_


def _write_compatibility_tables(X_Y_compat):
    X_Y_compat_ = []
    for X_cat, Y in X_Y_compat.items():
        for Y_cat in Y:
            X_Y_compat_.append((X_cat, Y_cat))
    return X_Y_compat_


def _reindex_morpheme_table_cats(X, X_cat_map, equivalences):
    X_ = []
    for X_entry in X:
        X_cat = X_entry[1]
        X_cat_new = db_maker_utils.reindex_cat(X_cat, X_cat_map, equivalences)
        X_.append((X_entry[0], X_cat_new, X_entry[2]))
    return X_


def _reindex_backoff_stem_cats(mode2cats, X_cat_map, equivalences):
    mode2cats_ = {}
    for entry_type ,backoff_mode, cats in mode2cats:
        cats_new = []
        for cat in cats.split():
            cat_new = db_maker_utils.reindex_cat(cat, X_cat_map, equivalences)
            cats_new.append(cat_new)
        mode2cats_[(entry_type, backoff_mode, ' '.join(cats_new))] = 1
    return mode2cats_


def collapse_and_reindex_morphemes(A, B, C, AB, BC, AC):
    entries = {}
    for name, entries_ in [('A', A), ('B', B), ('C', C)]:
        for match_, cat, analysis in entries_:
            analysis_ = {}
            for f_v in analysis.split():
                f_v = f_v.split(':')
                f, v = f_v[0], ':'.join(f_v[1:])
                analysis_[f] = v
            analysis_key = tuple(analysis_.get(k, 'NA') for k in essential_keys_form_feats)
            entries.setdefault(name, {}).setdefault(analysis_key, {}).setdefault(
                cat, []).append(((match_, analysis)))
    
    duplicates = {}
    problematic = {}
    for name, entries_ in entries.items():
        for analysis_key, cat2entries in entries_.items():
            if len(cat2entries) > 1:
                for cat in cat2entries:
                    problematic.setdefault(name, {}).setdefault(
                        analysis_key, []).append(cat)
                for cat, match_analyses in cat2entries.items():
                    if len(match_analyses) > 1:
                        duplicates.setdefault(name, {}).setdefault(
                            analysis_key, []).append(cat)
    
    CB = db_maker_utils._reverse_compat_table(BC)
    CA = db_maker_utils._reverse_compat_table(AC)
    BA = db_maker_utils._reverse_compat_table(AB)
    
    XYZ_info = {}
    for name, entries_ in problematic.items():
        if name == 'C':
            max_cat_index = int(re.search(r'[A-Z]+(\d+)', sorted(CB)[-1]).group(1)[1:])
            info = _get_mapping_for_table_Z(entries_, AB, BC, AC, 'S', max_cat_index)
        elif name == 'B':
            max_cat_index = int(re.search(r'[A-Z]+(\d+)', sorted(BA)[-1]).group(1)[1:])
            info = _get_mapping_for_table_Z(entries_, AC, CB, AB, 'X', max_cat_index)
        elif name == 'A':
            max_cat_index = int(re.search(r'[A-Z]+(\d+)', sorted(AB)[-1]).group(1)[1:])
            info = _get_mapping_for_table_Z(entries_, BC, CA, BA, 'P', max_cat_index)
        
        XYZ_info[name] = info

    A_, B_, C_, AB_, BC_, AC_ = _reindex_morphemes_after_collapse(
        XYZ_info, entries, A, B, C, AB, BC, AC)

    return A_, B_, C_, AB_, BC_, AC_, XYZ_info


def _reindex_morphemes_after_collapse(collapsing_info, entries, A, B, C, AB, BC, AC):
    A_, B_, C_ = set(), set(), set()
    for name, X_, X in [['A', A_, A], ['B', B_, B], ['C', C_, C]]:
        X_info = collapsing_info[name][0]
        xnew2Xold = {x_cat_new: set(info['cats']) for x_cat_new, info in X_info.items()}
        xold2Xnew = db_maker_utils._reverse_compat_table(xnew2Xold)
        for x_match_old, x_cat_old, x_analysis_old in X:
            for x_cat_new in xold2Xnew.get(x_cat_old, [x_cat_old]):
                X_.add((x_match_old, x_cat_new, x_analysis_old))
        
        # kill_entries = set()
        # for cat_new, info in X_info.items():
        #     cats_old, analysis_key = info['cats'], info['analysis_key']
        #     for cat_old in cats_old:
        #         entries_old = entries[name][analysis_key][cat_old]
        #         for match_, analysis in entries_old:
        #             X_.add((match_, cat_new, analysis))
        #             kill_entries.add((match_, cat_old, analysis))
        # assert kill_entries <= set(X)
        # X_.update({entry for entry in X if entry not in kill_entries})

    
    AB_, BC_, AC_ = {}, {}, {}
    for name, XY_, XY in [['AB', AB_, AB], ['BC', BC_, BC], ['AC', AC_, AC]]:
        name_src, name_tgt = name
        X_info_src, X_info_tgt = collapsing_info[name_src][0], collapsing_info[name_tgt][0]
        xnew2Xold = {x_cat_new: set(info['cats']) for x_cat_new, info in X_info_src.items()}
        xold2Xnew = db_maker_utils._reverse_compat_table(xnew2Xold)
        ynew2Yold = {y_cat_new: set(info['cats']) for y_cat_new, info in X_info_tgt.items()}
        yold2Ynew = db_maker_utils._reverse_compat_table(ynew2Yold)
        for x_old, Y_old in XY.items():
            for x_new in xold2Xnew.get(x_old, [x_old]):
                for y_old in Y_old:
                    XY_.setdefault(x_new, set()).update(yold2Ynew.get(y_old, [y_old]))

    # assert  len(A_) <= len(A) and len(B_) <= len(B) and len(C_) <= len(C)
    assert [sum(len(Y) for Y in XY_.values()) == sum(len(Y) for Y in XY.values())
            for XY_, XY in [(AB_, AB), (BC_, BC), (AC_, AC)]]
    
    return A_, B_, C_, AB_, BC_, AC_


def _get_mapping_for_table_Z(entries, XY, YZ, XZ, name, max_cat_index):
    Z_info, failed = {}, {}
    for analysis_key, cats in entries.items():
        XY_, YZ_, XZ_ = {}, {}, {}
        YZ_new, XZ_new = {}, {}
        cat_new = f'{name}{str(max_cat_index + len(Z_info) + 1).zfill(5)}'
        for cat in cats:
            for y, Z_ in YZ.items():
                if cat in Z_:
                    YZ_.setdefault(y, set()).add(cat)
                    YZ_new.setdefault(y, set()).add(cat_new)
            for x, Z_ in XZ.items():
                if cat in Z_:
                    XZ_.setdefault(x, set()).add(cat)
                    XZ_new.setdefault(x, set()).add(cat_new)
            for x, Y_ in XY.items():
                if x in XZ_:
                    for y in Y_:
                        if y in YZ_:
                            XY_.setdefault(x, set()).add(y)
        
        combinations, combinations_new = set(), set()
        COMBINATIONS = [(combinations, (XY_, YZ_, XZ_)),
                        (combinations_new, (XY_, YZ_new, XZ_new))]
        for combinations_, (XY_, YZ_, XZ_) in COMBINATIONS:
            for x, Y_ in XY_.items():
                for y in Y_:
                    for c in YZ_.get(y, []):
                        for x, Z_ in XZ_.items():
                            if c in Z_:
                                combinations_.add((x, y, c))
        
        combinations_reindexed = {
            comb[:2] + (next(iter(combinations_new))[2],) for comb in combinations}
        if combinations_reindexed <= combinations_new:
            Z_old_cats = [tuple(sorted(info['cats'])) for info in Z_info.values()]
            if tuple(sorted(cats)) not in Z_old_cats:
                Z_info[cat_new] = {}
                Z_info[cat_new]['cats'] = cats
                #FIXME: allow multiple analysis keys to be appended
                Z_info[cat_new]['analysis_key'] = analysis_key
            operation = 'equal'
        elif combinations_reindexed & combinations_new:
            operation = 'intersect'
        else:
            operation = 'no_intersect'
        if operation != 'equal':
            info = failed.setdefault(operation, {}).setdefault(analysis_key, {})
            info['cats'] = cats
            info['combinations'] = combinations
            info['combinations_reindexed'] = combinations_reindexed
            info['combinations_new'] = combinations_new
    
    return Z_info, failed


def collapse_and_reindex_categories(db, collapse_morphemes):
    prefix_stem_compat_ = _read_compatibility_tables(db['OUT:###TABLE AB###'])
    stem_suffix_compat_ = _read_compatibility_tables(db['OUT:###TABLE BC###'])
    prefix_suffix_compat_ = _read_compatibility_tables(db['OUT:###TABLE AC###'])

    prefixes_ = db['OUT:###PREFIXES###']
    stems_ = db['OUT:###STEMS###']
    suffixes_ = db['OUT:###SUFFIXES###']
    backoff_stems_ = db['OUT:###STEMBACKOFF###']

    print('Factorization Round 1')
    equivalences = db_maker_utils.factorize_categories(
        prefix_stem_compat_, stem_suffix_compat_, prefix_suffix_compat_)
    debug_info = None
    
    i = 0
    while i < 1 or equivalences != {}:
        if equivalences:
            prefix_stem_compat_, stem_suffix_compat_, prefix_suffix_compat_, \
                prefix_cat_map, stem_cat_map, suffix_cat_map = \
                    db_maker_utils.factorize_compatibility_lines(
                        prefix_stem_compat_, stem_suffix_compat_, prefix_suffix_compat_, equivalences)

            prefixes_ = _reindex_morpheme_table_cats(prefixes_, prefix_cat_map, equivalences)
            stems_ = _reindex_morpheme_table_cats(stems_, stem_cat_map, equivalences)
            suffixes_ = _reindex_morpheme_table_cats(suffixes_, suffix_cat_map, equivalences)
            backoff_stems_ = _reindex_backoff_stem_cats(backoff_stems_, stem_cat_map, equivalences)
        #TODO: deal with backoff stems here also
        # TODO: the following collapsing is still unstable and is not being
        # used for the moment. Should be debugged.
        if collapse_morphemes:
            collapsed = collapse_and_reindex_morphemes(
                prefixes_, stems_, suffixes_,
                prefix_stem_compat_, stem_suffix_compat_, prefix_suffix_compat_)
            prefixes_, stems_, suffixes_, prefix_stem_compat_, stem_suffix_compat_, \
                prefix_suffix_compat_, debug_info = collapsed
        
        print(f'Factorization Round {i + 2}')
        equivalences = db_maker_utils.factorize_categories(
            prefix_stem_compat_, stem_suffix_compat_, prefix_suffix_compat_)
        
        i += 1

    prefix_stem_compat_ = _write_compatibility_tables(prefix_stem_compat_)
    stem_suffix_compat_ = _write_compatibility_tables(stem_suffix_compat_)
    prefix_suffix_compat_ = _write_compatibility_tables(prefix_suffix_compat_)

    db['OUT:###PREFIXES###'] = prefixes_
    db['OUT:###STEMS###'] = stems_
    db['OUT:###SUFFIXES###'] = suffixes_
    db['OUT:###TABLE AB###'] = prefix_stem_compat_
    db['OUT:###TABLE BC###'] = stem_suffix_compat_
    db['OUT:###TABLE AC###'] = prefix_suffix_compat_
    db['OUT:###STEMBACKOFF###'] = backoff_stems_

    collapse_and_reindex_debug = dict(
        equivalences=equivalences if equivalences else None,
        prefix_cat_map=prefix_cat_map if equivalences else None,
        stem_cat_map=stem_cat_map if equivalences else None,
        suffix_cat_map=suffix_cat_map if equivalences else None,
        morpheme_collapsing_debug_info=debug_info
    )

    return db, collapse_and_reindex_debug


def print_almor_db(output_path, db):
    """Create output file in ALMOR DB format"""

    with open(output_path, 'w') as f:
        for x in db['OUT:###HEADER###']:
            print(x, file=f)

        print('###STEMBACKOFF###', file=f)
        for x in db['OUT:###STEMBACKOFF###']:
            print(*x, sep=' ', file=f)
        
        postregex = db.get('OUT:###POSTREGEX###')
        if postregex:
            print('###POSTREGEX###', file=f)
            for x in postregex:
                print(x, file=f)

        for section in ['PREFIXES', 'STEMS', 'SUFFIXES']:
            assert f'OUT:###{section}###' in db, (
                f'Empty {section} section. Something might be wrong with the sheets.')     

        print('###PREFIXES###', file=f)
        for x in db['OUT:###PREFIXES###']:
            print(*x, sep='\t', file=f)
            
        print('###SUFFIXES###', file=f)
        for x in db['OUT:###SUFFIXES###']:
            print(*x, sep='\t', file=f)
        
        underscore_ar = re.compile('ـ')
        print('###STEMS###', file=f)
        for x in db['OUT:###STEMS###']:
            # Fixes weird underscore generated by bw2ar()
            x = (*x[:2], underscore_ar.sub('_', x[2]))
            print(*x, sep='\t', file=f)

        smart_backoff = db.get('OUT:###SMARTBACKOFF###')
        if smart_backoff:
            print('###SMARTBACKOFF###', file=f)
            for x in db['OUT:###SMARTBACKOFF###']:
                print(*x, sep='\t', file=f)
            
        print('###TABLE AB###', file=f)
        for x in db['OUT:###TABLE AB###']:
            print(*x, sep=' ', file=f)
            
        print('###TABLE BC###', file=f)
        for x in db['OUT:###TABLE BC###']:
            print(*x, sep=' ', file=f)
            
        print('###TABLE AC###', file=f)
        for x in db['OUT:###TABLE AC###']:
            print(*x, sep=' ', file=f)


def _get_short_cat_name_maps(ORDER: pd.DataFrame) -> Dict:
    """Because the categories are made up of the ORDER class names among other things,
    in order to reduce the visual length of these categories while maintaining meaning
    for debugging purposes, the following short names are used. There is a corresponding
    short name for each CLASS item across the ORDER rows.

    Args:
        ORDER (pd.DataFrame): ORDER sheet

    Returns:
        Dict: mapping from the actual class name (in PREFIX, STEM, or SUFFIX column or ORDER)
        to its short name (PREFIX-SHORT, STEM-SHORT, and SUFFIX-SHORT).
    """
    map_p, map_x, map_s = {}, {}, {}
    map_word = {}
    for _, row in ORDER.iterrows():
        p, x, s = row['PREFIX'], row['STEM'], row['SUFFIX']
        p = '[EMPTY]' if p == '' else p
        s = '[EMPTY]' if s == '' else s
        p_short, x_short, s_short = row['PREFIX-SHORT'], row['STEM-SHORT'], row['SUFFIX-SHORT']
        p_short = '[EMPTY]' if p_short == '' else p_short
        s_short = '[EMPTY]' if s_short == '' else s_short
        def check_soundness(x, map_x, x_short):
            if x in map_x:
                assert map_x[x] == x_short, 'Every complex morpheme class sequence should have a unique short cat name.'
            else:
                assert x_short not in map_x.values(), 'Clashing short cats.'
        check_soundness(p, map_p, p_short)
        check_soundness(x, map_x, x_short)
        check_soundness(s, map_s, s_short)
        map_p[p], map_x[x], map_s[s] = p_short, x_short, s_short
        map_word.setdefault((p_short, x_short, s_short), 0)
        map_word[(p_short, x_short, s_short)] += 1
    short_cat_maps = dict(prefix=map_p, stem=map_x, suffix=map_s)
    # Make sure that the short order names are unique
    assert sum(map_word.values()) == len(map_word), 'Short order names are not unique.'
    return short_cat_maps


def gen_cmplx_morph_combs(cmplx_morph_seq: str,
                          MORPH: pd.DataFrame, LEXICON: pd.DataFrame,
                          cond2class: Optional[Dict[str, Tuple[str, int]]]=None,
                          cmplx_morph_memoize: Optional[Dict]=None,
                          pruning_cond_s_f: bool=True,
                          pruning_same_class_incompat: bool=True) -> Dict[Tuple[Tuple[str]], List[List[pd.DataFrame]]]:
    """Method which works within the scope of a PREFIX/STEM/SUFFIX order field. BW for example
    confounds prefixes (suffixes) and proclitics (enclitics) within the PREFIX (SUFFIX) field.
    [Side note]: In our case, we have an additional [Buffer] class which could be considered as
    part of any of the three, giving exactly the same result (different DB, but same resulting
    analyses/generations) in all three cases.
    Thus, morphemes at the PREFIX/STEM/SUFFIX order field level are called complex morphemes.
    This method generates all combinations of complex morphemes by combining regular morphemes,
    as per the order lines specifications. It can do so in a naive way by generating all possible
    combinations, or by using the morphemes' conditions to reduce the space of possibilities to
    a set of more plausible combinations. So for example, if we have for the SUFFIX field the
    following order: `[Buffer] [NSuff.XXIN]`, and [Buffer] and [NSuff.XXIN] contain 40 and 30
    allomorphs respectively, then the space of possibilities is 40 x 30 = 1,200 complex morphemes.
    Many of those may be decuded to be implausible thanks to the simple heuristic of checking
    whether the conditions of their component allomorphs are contradictory.

    Args:
        cmplx_morph_seq (str): space-separated sequence of classes that predefine the order of the morphemes
        to be assembled for the cartesian product.
        MORPH (pd.DataFrame): morph specs
        LEXICON (pd.DataFrame): lexicon specs
        cond2class (Optional[Dict[str, Tuple[str, int]]], optional): inventory
        of condition definitions and their corresponding vectors which will be useful in the later
        pruning process. Defaults to None.
        cmplx_morph_memoize (Optional[Dict], optional): dictionary mainly used for stems to avoid recomputing
        all combinations of stem classes. Defaults to None.
        pruning_cond_s_f (bool, optional): whether or not to perform pruning based on COND-S and COND-F
        compatibility. Defaults to True.
        pruning_same_class_incompat (bool, optional): whether or not to perform pruning based on wether complex
        morphemes set conditions which belong to the same class of of conditions (different from morpheme class).
        Defaults to True.

    Returns:
        Dict[Tuple[Tuple[str]], List[List[pd.DataFrame]]]: keys are unique classes of condition combinations, and values
        are all the combinations that have these conditions.
    """
    if cmplx_morph_memoize:
        return cmplx_morph_memoize
    
    if not cmplx_morph_seq:
        cmplx_morph_seq = '[EMPTY]'

    cmplx_morph_classes = []
    for cmplx_morph_cls in cmplx_morph_seq.split():
        sheet = LEXICON if 'STEM' in cmplx_morph_cls else MORPH
        instances = []
        for _, row in sheet[sheet.CLASS == cmplx_morph_cls].iterrows():
            if 'STEM' in cmplx_morph_cls and (row['FORM'] == '' or row['FORM'] == "DROP"):
                continue
            instances.append(row.to_dict())
        if not instances:
            return {}
        cmplx_morph_classes.append(instances)
    
    cmplx_morphs = [list(t) for t in itertools.product(*[mc for mc in cmplx_morph_classes if mc])]
    cmplx_morph_categorized = {}
    for seq in cmplx_morphs:
        seq_cond_cat = [(morph['COND-S'], morph['COND-T'], morph['COND-F']) for morph in seq]
        cmplx_morph_categorized.setdefault(tuple(seq_cond_cat), []).append(seq)
    
    # Performing partial compatibility tests to prune out incoherent combinations
    if pruning_cond_s_f or pruning_same_class_incompat:
        # Prune out incoherent classes
        complex_morph_categorized_ = {}
        for seq_class, seq_instances in cmplx_morph_categorized.items():
            cond_s_seq = {
                cond for part in seq_class for cond in part[0].split()}
            cond_t_seq = {
                cond for part in seq_class for cond in part[1].split()}
            cond_f_seq = {
                cond for part in seq_class for cond in part[2].split()}
            # If any condition appears in COND-S and COND-F of the combination sequence,
            # then the sequence should be pruned out since it is incoherent.
            if pruning_cond_s_f and cond_s_seq.intersection(cond_f_seq) != set():
                continue
            # If any two conditions belonging to the same condition class appear in COND-T
            # of the combination sequence, then the sequence should be pruned out since it
            # is incoherent.
            if pruning_same_class_incompat:
                # If or-ed (||) COND-T did not exist, this would be as simple as checking
                # whether two conditions of the same class are present in COND-T of the combination
                # sequence, and disqualifying the latter based on that since two morphemes cannot
                # coherently require some condition to be true if they are of the same class 
                # (e.g., a combination sequence (suffix/prefix/stem) cannot both require #t and #-a>
                # since these conditions are contradictory). But or-ed conditions require us to follow
                # a different approach implemented below.
                coherence = {}
                is_not_coherent = False
                for cond in cond_t_seq:
                    # Disregard if default condition
                    if cond == '_':
                        continue
                    elif '||' in cond:
                        cond_s = cond.split('||')
                        or_terms = [cond2class[cond][1] for cond in cond_s]
                        # Based on the assumption that all terms belong to the same class
                        cond_onehot_or = int('0' * or_terms[0], 2)
                        for or_term in or_terms:
                            cond_onehot_or = cond_onehot_or | or_term
                        cond_class = cond2class[cond_s[0]]
                        cond_onehot = cond_onehot_or
                    else:
                        cond_class, cond_onehot = cond2class[cond]
                    cond_onehot_and = coherence.setdefault(cond_class, cond_onehot)
                    coherence[cond_class] = cond_onehot_and & cond_onehot
                for morph in coherence.values():
                    if morph == 0:
                        is_not_coherent = True
                        continue
                if is_not_coherent:
                    continue
            complex_morph_categorized_[seq_class] = seq_instances
        
        cmplx_morph_categorized = complex_morph_categorized_
    
    cmplx_morph_memoize = cmplx_morph_categorized
    
    return cmplx_morph_categorized


def check_compatibility (cond_s: str, cond_t: str, cond_f: str,
                         compatibility_memoize: Dict[str, bool]) -> bool:
    """Method which, based on COND-S (conditions set by the morpheme), COND-T (conditions
    required to be set by the concatenating morpheme(s)), and COND-F (conditions required not
    to be set by the concatenating morpheme(s)), decides whether a combination of
    complex morphemes is compatible together (across the the prefix/stem/suffix boundary).
    If two morphemes are concatenated to form a complex morpheme, then the new system they
    form (the complex morpheme) now shares the fingerprint (COND-S, COND-T, and COND-F) of
    both the morphemes. Furthermore, all COND-S of the complex prefix, stem, and suffix are
    concatenated (same for COND-T, and COND-F) since for two complex morphemes to be compatible
    with each other, their conditions must be evaluated collectively across the complex morphemes,
    along the two COND-T and COND-F axes, based on their collective identity (COND-S). In other
    words, if any COND-S of the word fromed by the complex morpheme system is present in COND-F,

    Args:
        cond_s (str): concatenation of COND-S of complex prefix, complex stem, and complex suffix
        cond_t (str): concatenation of COND-T of complex prefix, complex stem, and complex suffix
        cond_f (str): concatenation of COND-F of complex prefix, complex stem, and complex suffix
        compatibility_memoize (Dict[str, bool]): dictionary keeping track of combinations that were
        previously validated to avoid recomputing them a second time.

    Returns:
        bool: whether a combination of complex morphemes is valid or not. If it is valid, all the
        complex morphemes in it are secured a place in the DB.
    """
    #TODO: inefficient, in future create a Conditions class and vectorize everything
    # This method takes up about half of the runtime profile.
    key = f'{cond_s}\t{cond_t}\t{cond_f}'
    if key in compatibility_memoize:
      return compatibility_memoize[key]
    else:
      compatibility_memoize[key] = ''
    # Remove all nil items (indicated as "_")
    cs = [cond for cond in cond_s.split()]
    ct = [cond for cond in cond_t.split()]
    cf = [cond for cond in cond_f.split()]

    valid = True
    # Conditions required to be set by the concatenating morpheme(s)
    for t in ct:
        # Supports cases where we have a disjunction of condition terms
        validor = False
        for ort in t.split('||'):
            validor = validor or ort in cs
        # If any of the conditions present in COND-T is not present in COND-S
        # then the combination in invalid 
        valid = valid and validor
        if not valid:
            compatibility_memoize[key] = valid
            return valid
    # Conditions required NOT to be set by the concatenating morpheme(s)
    for f in cf:
        for orf in f.split('||'):
            valid = valid and orf not in cs
        if not valid:
            compatibility_memoize[key] = valid
            return valid

    compatibility_memoize[key] = valid
    return valid                     


def _choose_required_feats(pos_type):
    if pos_type == 'verbal':
        required_feats = _required_verb_stem_feats
    elif pos_type == 'nominal':
        required_feats = _required_nom_stem_feats
    elif pos_type in ['other', 'any']:
        required_feats = list(set(_required_nom_stem_feats + _required_verb_stem_feats))
    else:
        raise NotImplementedError
    return required_feats


def _read_header_file(header:pd.DataFrame):
    header_ = []
    order = list(header.columns)[1:]
    defines = {}
    for _, row in header[header['DEFINE'] == 'DEFINE'].iterrows():
        for feat in order:
            if row[feat]:
                defines.setdefault(feat, []).append(row[feat])
    
    header_.append('###DEFINES###')
    for feat in order:
        line = f'DEFINE {feat} ' + ' '.join(f'{feat}:{v}' for v in defines[feat])
        header_.append(line)
    
    defaults = {}
    for _, row in header[header['DEFINE'] == 'DEFAULT'].iterrows():
        for feat in order:
            defaults.setdefault(row['pos'], {}).setdefault(feat, row[feat])
    
    header_.append('###DEFAULTS###')
    for pos, feat2value in defaults.items():
        line = ' '.join(f'{feat}:{feat2value[feat]}' for feat in order)
        line = f'DEFAULT pos:{pos} ' + line
        header_.append(line)

    header_.append('###ORDER###')
    header_.append('ORDER ' + ' '.join(order))

    tokenization = {}
    for _, row in header[header['DEFINE'] == 'TOKENIZATION'].iterrows():
        for feat in order:
            if row[feat]:
                tokenization.setdefault(feat, row[feat])
    
    header_.append('###TOKENIZATIONS###')
    header_.append('TOKENIZATION ' + ' '.join(o for o in order if o in tokenization))

    transcription = {}
    for _, row in header[header['DEFINE'] == 'TRANSCRIPTION'].iterrows():
        for feat in order:
            if row[feat]:
                transcription.setdefault(feat, row[feat])

    defaults = {'defaults': defaults, 'order': order,
                'tokenization': tokenization, 'transcription': transcription}

    return header_, defaults


if __name__ == "__main__":
    if args.run_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    
    output_dir = args.output_dir if args.output_dir else config.get_db_dir_path()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, config.db)

    make_db(config, output_path)
    
    if args.run_profiling:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
