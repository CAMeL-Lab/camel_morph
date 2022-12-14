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
import json
import argparse
import itertools
from time import strftime, gmtime, process_time
from functools import partial
import cProfile, pstats
import sys
from typing import Dict, Tuple, List, Optional

import pandas as pd

import db_maker_utils

_required_verb_stem_feats = ['pos', 'asp', 'per', 'gen', 'num', 'vox', 'mod']
_required_nom_stem_feats = ['pos', 'form_gen', 'form_num', 'gen', 'num', 'stt', 'cas']
_clitic_feats = ['enc0', 'enc1', 'enc2', 'prc0', 'prc1', 'prc1.5', 'prc2', 'prc3']

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

def make_db(config:Dict, config_name:str, output_dir:str):
    """
    Main function which takes in a set of specifications from `csv` files (downloadable
    from Google Sheets) and which, from the latter, prints out a `db` file in the ALMOR format,
    useable by the Camel Tools Analyzer and Generator engines to produce word analyses/generations.

    Args:
        config (Dict): dictionary containing all the necessary information to build the `db` file.
        config_name (str): key of the specific ("local") configuration to get information from in the config file.
        output_dir (str): path of the directory to output the `db` file to.
    """
    config_local = config['local'][config_name]
    c0 = process_time()
    
    print("\nLoading and processing sheets... [1/3]")
    SHEETS, cond2class = db_maker_utils.read_morph_specs(config, config_name)
    
    print("\nValidating combinations... [2/3]")
    cat2id = config_local.get('cat2id', False)
    db = construct_almor_db(SHEETS, config_local['pruning'], cond2class, cat2id)
    
    print("\nGenerating DB file... [3/3]")
    output_dir = os.path.join(output_dir, f"camel-morph-{config_local['dialect']}")
    os.makedirs(output_dir, exist_ok=True)
    print_almor_db(os.path.join(output_dir, config_local['db']), db)
    
    c1 = process_time()
    print(f"\nTotal time required: {strftime('%M:%S', gmtime(c1 - c0))}")


def construct_almor_db(SHEETS:Dict[str, pd.DataFrame],
                       pruning:bool,
                       cond2class:Dict,
                       cat2id:bool=False) -> Dict:
    """
    Function which takes care of the condition validation process, i.e., deciding which
    (complex) morphemes are compatible, and print them and their computed categories in
    ALMOR format.

    Args:
        SHEETS (Dict[str, pd.DataFrame]): dictionary which contains the 7 main dataframes which will 
        be used throughout the DB making process.
        pruning (bool): whether or not to perform pruning which is the preprocessing step of determining
        which morphemes are not compatible as a combination of complex morpheme, to reduce the number
        of complex morphemes which will then in turn be validated.
        cond2class (Dict): inventory of condition definitions and their corresponding vectors which will
        be useful in the pruning process.

    Returns:
        Dict: Database which contains entries (values) for each section (keys).
    """
    ORDER, MORPH, LEXICON = SHEETS['order'], SHEETS['morph'], SHEETS['lexicon']
    ABOUT, HEADER, POSTREGEX = SHEETS['about'], SHEETS['header'], SHEETS['postregex']
    BACKOFF, SMART_BACKOFF = SHEETS['backoff'], SHEETS['smart_backoff']

    short_cat_maps = _get_short_cat_name_maps(ORDER) if 'PREFIX-SHORT' in ORDER.columns else None

    # One-time filling of the About, Header, and PostRegex sections of the DB
    db = {}
    db['OUT:###ABOUT###'] = list(ABOUT['CONTENT'])
    db['OUT:###HEADER###'] = list(HEADER['CONTENT'])
    if POSTREGEX is not None:
        db['OUT:###POSTREGEX###'] = [
            'MATCH\t' + '\t'.join([match for match in POSTREGEX['MATCH'].values.tolist()]),
            'REPLACE\t' + '\t'.join([replace for replace in POSTREGEX['REPLACE'].values.tolist()])
        ]
    
    defaults = _process_defaults(db['OUT:###HEADER###'])
    cat2id = {} if cat2id else None
    
    def construct_process(lexicon: pd.DataFrame,
                          order: pd.DataFrame,
                          cmplx_stem_memoize: Dict[str, str],
                          stems_section_title: str,
                          backoff:bool=False):
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
        
        cmplx_morph_classes = dict(
            cmplx_prefix_classes=(cmplx_prefix_classes, order['PREFIX']),
            cmplx_suffix_classes=(cmplx_suffix_classes, order['SUFFIX']),
            cmplx_stem_classes=(cmplx_stem_classes, order['STEM']))
        
        # Complex morphemes validation or word generation (across the prefix/stem/suffix boundary)
        db_ = cross_cmplx_morph_validation(
            cmplx_morph_classes, order['CLASS'].lower(), short_cat_maps, defaults, stems_section_title, cat2id, backoff)
        for section, contents in db_.items():
            # if 'BACKOFF' in stems_section_title and section != stems_section_title:
            #     assert set(contents) <= set(db[section])
            db.setdefault(section, {}).update(contents)
    
    # For memoization to work as intended, same-aspect order lines should be placed next
    # to each other in the ORDER file, and since the stem part of the order usually stays
    # the same at the aspect level, then it makes sense to avoid recomputing all the combinations
    # each time and same them in the memo. dict.
    if LEXICON is not None:
        print('Concrete lexicon')
        pbar = tqdm(total=len(list(ORDER.iterrows())))
        cmplx_stem_memoize = {}
        order_stem_prev = ''
        for _, order in ORDER.iterrows():
            pbar.set_description(order['SUFFIX-SHORT'])
            if order['STEM'] != order_stem_prev:
                cmplx_stem_memoize = {}
                order_stem_prev = order['STEM']
            construct_process(LEXICON, order, cmplx_stem_memoize,
                            stems_section_title='OUT:###STEMS###')
            pbar.update(1)
        pbar.close()
    #FIXME: Currently these backoff entries do not interact with the other morphemes.
    # They are just inserted here for backward compatibility with the Analyzer engine.
    if BACKOFF is not None:
        print('Backoff lexicon')
        pbar = tqdm(total=len(list(ORDER.iterrows())))
        cmplx_stem_memoize = {}
        order_stem_prev = ''
        for _, order in ORDER.iterrows():
            pbar.set_description(order['SUFFIX-SHORT'])
            if order['STEM'] != order_stem_prev:
                cmplx_stem_memoize = {}
                order_stem_prev = order['STEM']
            construct_process(BACKOFF, order, cmplx_stem_memoize,
                            stems_section_title='OUT:###STEMS###', backoff=True)
            pbar.update(1)
        pbar.close()
        
    
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
                                 backoff:bool=False) -> Dict:
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
        backoff (bool): whether or not to add the correct category or just the same category
        to all stem entries. Defaults to False.

    Returns:
        Dict: Database in progress
    """
    db = {}
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
                        update_db(db, update_info, cat_memoize, short_cat_maps, defaults, cat2id, backoff)
                    # If morph class cat has already been computed previously, then cat is still `None`
                    # (because we will not go again in the morph for loop) and we need to retrieve the
                    # computed value. 
                    # FIXME: stem_cat seems to always be None at this point, so there is no need for
                    # the if statement 
                    stem_cat = stem_cat if stem_cat else cat_memoize['stem'][cmplx_stem_cls]
                    prefix_cat = prefix_cat if prefix_cat else cat_memoize['prefix'][cmplx_prefix_cls]
                    suffix_cat = suffix_cat if suffix_cat else cat_memoize['suffix'][cmplx_suffix_cls]

                    db['OUT:###TABLE AB###'][prefix_cat + " " + stem_cat] = 1
                    db['OUT:###TABLE BC###'][stem_cat + " " + suffix_cat] = 1
                    db['OUT:###TABLE AC###'][prefix_cat + " " + suffix_cat] = 1
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
              backoff:bool=False):
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
        backoff (bool): whether or not to add the correct category or just the same category
        to all stem entries. Defaults to False.
    """
    cmplx_morph_seq = update_info['cmplx_morph_seq']
    cmplx_morph_cls = update_info['cmplx_morph_cls']
    cmplx_morph_type = update_info['cmplx_morph_type']
    cmplx_morphs = update_info['cmplx_morphs']
    cond_s, cond_t, cond_f = update_info['conditions']
    db_section = update_info['db_section']
    defaults_ = None
    if defaults:
        defaults_ = defaults['defaults']
    
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
            morph_entry = _generate(cmplx_morph_seq,
                                    required_feats,
                                    cmplx_morph,
                                    cond_s, cond_t, cond_f,
                                    short_cat_map,
                                    defaults_,
                                    cat2id,
                                    backoff)
            db[db_section].setdefault('\t'.join(morph_entry.values()), 0)
            db[db_section]['\t'.join(morph_entry.values())] += 1
        cat_memoize[cmplx_morph_type][cmplx_morph_cls] = morph_entry['cat']


def _create_cat(cmplx_morph_type: str, cmplx_morph_class: str,
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

def _generate_affix(affix_type: str,
                    cmplx_morph_seq: str,
                    required_feats: List[str],
                    affix: List[Dict],
                    affix_cond_s: str, affix_cond_t: str, affix_cond_f: str,
                    short_cat_map: Optional[Dict]=None,
                    defaults: Dict=None,
                    cat2id:Optional[Dict]=None,
                    backoff:bool=False) -> Dict[str, str]:
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

    Returns:
        Dict[str, str]: dict containing the 3 fields needed to store the complex affix as an entry in the DB.
    """
    affix_match, affix_diac, affix_gloss, affix_feat, affix_bw = _read_affix(affix)
    affix_type = 'P' if affix_type == 'prefix' else 'S'
    acat = _create_cat(
        affix_type, cmplx_morph_seq, affix_cond_s, affix_cond_t, affix_cond_f, short_cat_map, cat2id)
    ar_pbw = _convert_bw_tag(affix_bw)
    affix_feat = ' '.join([f"{feat}:{val}" for feat, val in affix_feat.items()])
    affix = {
        'match': bw2ar(affix_match),
        'cat': acat,
        'analysis': f"diac:{bw2ar(affix_diac)} bw:{ar_pbw} gloss:{affix_gloss.strip()} {affix_feat}"
    }
    return affix


def _generate_stem(cmplx_morph_seq: str,
                   required_feats: List[str],
                   stem: List[Dict],
                   stem_cond_s: str, stem_cond_t: str, stem_cond_f: str,
                   short_cat_map: Optional[Dict]=None,
                   defaults: Dict=None,
                   cat2id:Optional[Dict]=None,
                   backoff:bool=False) -> Dict[str, str]:
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
        backoff (bool): whether or not to add the correct category or just the same category
        to all stem entries. Defaults to False.

    Returns:
        Dict[str, str]: _description_
    """
    stem_match, stem_diac, stem_lex, stem_gloss, stem_feat, stem_bw, root, smart_backoff = _read_stem(stem)
    xcat = _create_cat(
        'X', cmplx_morph_seq, stem_cond_s, stem_cond_t, stem_cond_f, short_cat_map, cat2id)
    ar_xbw = _convert_bw_tag(stem_bw, backoff)
    if defaults:
        stem_feat = [f"{feat}:{stem_feat[feat]}"
                        if feat in stem_feat and stem_feat[feat] != '_'
                        else f"{feat}:{defaults[stem_feat['pos']][feat]}" 
                        for feat in required_feats + _clitic_feats]
    else:
        stem_feat = [f"{feat}:{val}" for feat, val in stem_feat.items()]
    stem_feat = ' '.join(stem_feat)
    
    if smart_backoff:
        match = db_maker_utils._bw2ar_regex(stem_match, bw2ar)
        stem_diac, stem_lex, root = bw2ar(stem_diac), bw2ar(stem_lex), bw2ar(root)
    elif backoff:
        match = 'NOAN'
    else:
        match = bw2ar(stem_match)
        stem_diac, stem_lex, root = bw2ar(stem_diac), bw2ar(stem_lex), bw2ar(root)

    stem = {
        'match': match,
        'cat': xcat,
        'analysis': (f"diac:{stem_diac} bw:{ar_xbw} lex:{stem_lex} "
                  f"root:{root} gloss:{stem_gloss.strip()} {stem_feat} ")
    }
    return stem

def _read_affix(affix: List[Dict]) -> Tuple[str, Dict]:
    """From the CamelMorph specifications, loads the affix information
    of multiple morphemes appearing in the prefix/suffix portion of the order line
    and which are deemed to be compatible with each other to form a complex affix, and
    generates from them the fields needed to store the complex affix as an entry
    in the DB.

    Args:
        affix (List[Dict]): individual analyses (dict) of the morphemes in the complex affix.

    Returns:
        Tuple[str, Dict]: information to store in the DB.
    """
    affix_bw = '+'.join([m['BW'] for m in affix if m['BW'] != '_'])
    affix_diac = ''.join([m['FORM'] for m in affix if m['FORM'] != '_'])
    affix_gloss = '+'.join([m['GLOSS'] for m in affix if m['GLOSS'] != '_'])
    affix_gloss = affix_gloss if affix_gloss else '_'
    affix_feat = {feat.split(':')[0]: feat.split(':')[1]
             for m in affix for feat in m['FEAT'].split()}
    affix_match = normalize_alef_bw(normalize_alef_maksura_bw(
        normalize_teh_marbuta_bw(dediac_bw(re.sub(r'[#@]', '', affix_diac)))))
    return affix_match, affix_diac, affix_gloss, affix_feat, affix_bw


def _read_stem(stem: List[Dict]) -> Tuple[str, Dict]:
    """Same as `_read_affix()`. Treated slightly differently than affixes which is why it has a
    method of its own.

    Args:
        stem (List[Dict]): individual analyses (dict) of the morphemes in the complex stem.

    Returns:
        Tuple[str, Dict]: information to store in the DB
    """
    stem_bw = '+'.join([s['BW'] for s in stem if s['BW'] != '_'])
    stem_gloss = '+'.join([s['GLOSS'] for s in stem if 'LEMMA' in s])
    stem_gloss = stem_gloss if stem_gloss else '_'
    stem_lex = '+'.join([s['LEMMA'].split(':')[1] for s in stem if 'LEMMA' in s])
    stem_feat = {feat.split(':')[0]: feat.split(':')[1]
                for s in stem for feat in s['FEAT'].split()}
    root = [s['ROOT'] for s in stem if s.get('ROOT')][0]
    
    smart_backoff = False
    if not any([s['DEFINE'] == 'SMARTBACKOFF' for s in stem]):
        stem_diac = ''.join([s['FORM']for s in stem if s['FORM'] != '_'])
        #NOTE: for nominals, postregex symbol is @ while for verbs it is #
        stem_match = normalize_alef_bw(normalize_alef_maksura_bw(
            normalize_teh_marbuta_bw(dediac_bw(re.sub(r'[#@]', '', stem_diac)))))
    else:
        smart_backoff = True
        stem_diac = ''.join([s['FORM'] for s in stem if s['FORM'] != '_'])
        stem_match = []
        for s in stem:
            if s['FORM'] != '_':
                if s['DEFINE'] == 'SMARTBACKOFF':
                    stem_match.append(re.sub(r'^\^|\$$|[#@]', '', s['MATCH']))
                else:
                    stem_match.append(normalize_alef_bw(normalize_alef_maksura_bw(
                                      normalize_teh_marbuta_bw(dediac_bw(re.sub(r'[#@]', '', s['FORM']))))))
        stem_match = f"^{''.join(stem_match)}$"

    return stem_match, stem_diac, stem_lex, stem_gloss, stem_feat, stem_bw, root, smart_backoff

def print_almor_db(output_path, db):
    """Create output file in ALMOR DB format"""
    with open(output_path, "w") as f:
        for x in db['OUT:###HEADER###']:
            print(x, file=f)

        print("###POSTREGEX###", file=f)
        postregex = db.get('OUT:###POSTREGEX###')
        if postregex:
            for x in postregex:
                print(x, file=f)

        print("###PREFIXES###", file=f)
        for x in db['OUT:###PREFIXES###']:
            print(x, file=f)
            
        print("###SUFFIXES###", file=f)
        for x in db['OUT:###SUFFIXES###']:
            print(x, file=f)
            
        print("###STEMS###", file=f)
        for x in db['OUT:###STEMS###']:
            # Fixes weird underscore generated by bw2ar()
            x = re.sub('ـ', '_', x)
            print(x, file=f)

        print("###SMARTBACKOFF###", file=f)
        smart_backoff = db.get('OUT:###SMARTBACKOFF###')
        if smart_backoff:
            for x in db['OUT:###SMARTBACKOFF###']:
                print(x, file=f)
            
        print("###TABLE AB###", file=f)
        for x in db['OUT:###TABLE AB###']:
            print(x, file=f)
            
        print("###TABLE BC###", file=f)
        for x in db['OUT:###TABLE BC###']:
            print(x, file=f)
            
        print("###TABLE AC###", file=f)
        for x in db['OUT:###TABLE AC###']:
            print(x, file=f)


def _get_short_cat_name_maps(ORDER: pd.DataFrame) -> Dict:
    """Because the categories are made out of the ORDER class names among other things,
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
        p_short, x_short, s_short = row['PREFIX-SHORT'], row['STEM-SHORT'], row['SUFFIX-SHORT']
        map_p[p], map_x[x], map_s[s] = p_short, x_short, s_short
        map_word.setdefault((p_short, x_short, s_short), 0)
        map_word[(p_short, x_short, s_short)] += 1
    short_cat_maps = dict(prefix=map_p, stem=map_x, suffix=map_s)
    # Make sure that the short order names are unique
    assert sum(map_word.values()) == len(map_word), "Short order names are not unique."
    return short_cat_maps


def gen_cmplx_morph_combs(cmplx_morph_seq: str,
                          MORPH: pd.DataFrame, LEXICON: pd.DataFrame,
                          cond2class: Optional[Dict[str, Tuple[str, int]]]=None,
                          cmplx_morph_memoize: Optional[Dict]=None,
                          pruning_cond_s_f: bool=True,
                          pruning_same_class_incompat: bool=True) -> Dict[Tuple[Tuple[str]], List[List[pd.DataFrame]]]:
    """Method which works within the scope of a prefix/stem/suffix, c.f., across the boundary.
    It generates all combinations of complex morphemes by combining regular morphemes. It can do
    so in a naive way by generating all possible combinations or by using conditions to reduce those
    to a set of more plausible combinations in a postprocessing fashion after the initial cartesian
    product has been generated.

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

    cmplx_morph_classes = []
    for cmplx_morph_cls in cmplx_morph_seq.split():
        sheet = LEXICON if 'STEM' in cmplx_morph_cls else MORPH
        instances = []
        for _, row in sheet[sheet.CLASS == cmplx_morph_cls].iterrows():
            if 'STEM' in cmplx_morph_cls and (row['FORM'] == '' or row['FORM'] == "DROP"):
                continue
            instances.append(row.to_dict())
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


def _process_defaults(header):
    _order = [line.split()[1:] for line in header
                if line.startswith('ORDER')][0]
    _defaults = [{f: d for f, d in [f.split(':') for f in line.split()[1:]]}
                    for line in header if line.startswith('DEFAULT')]
    _defaults = {d['pos']: d for d in _defaults}
    for pos in _defaults:
        _defaults[pos]['enc1'] = _defaults[pos]['enc0']
    defaults = {'defaults': _defaults, 'order': _order}
    return defaults

def _choose_required_feats(pos_type):
    if pos_type == 'verbal':
        required_feats = _required_verb_stem_feats
    elif pos_type == 'nominal':
        required_feats = _required_nom_stem_feats
    elif pos_type == 'other':
        required_feats = list(set(_required_nom_stem_feats + _required_verb_stem_feats))
    else:
        raise NotImplementedError
    return required_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-output_dir", default='',
                        type=str, help="Path of the directory to output the DBs to.")
    parser.add_argument("-run_profiling", default=False,
                        action='store_true', help="Run execution time profiling for the make_db().")
    parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)
    config_local =  config['local'][args.config_name]
    config_global =  config['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.utils.dediac import dediac_bw

    normalize_map = CharMapper({
        '<': 'A',
        '>': 'A',
        '|': 'A',
        '{': 'A',
        'Y': 'y'
    })

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    if args.run_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    
    output_dir = args.output_dir if args.output_dir else config_global['db_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    make_db(config, args.config_name, output_dir)
    
    if args.run_profiling:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
