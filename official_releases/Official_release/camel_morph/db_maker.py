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

# =============================================================================
# Imports
# =============================================================================
import argparse
import cProfile
import importlib
import itertools
import os
import pickle
import pstats
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw
from camel_tools.utils.dediac import dediac_bw
from camel_tools.morphology.utils import strip_lex

from . import db_maker_utils
from . import db_maker_runtime as runtime
from .db_to_json import export_db_to_json
from .almor_schema import (
    BACKOFF_SMART, BACKOFF_VANILLA,
    BW2AR_AFFIX_FIELDS, BW2AR_STEM_FIELDS, CAPHI_COLUMN,
    CAPHI_MORPH_TYPES, CAT_TYPE_PREFIX, CAT_TYPE_STEM,
    CAT_TYPE_SUFFIX, COL_CLASS, COL_CONTENT, COL_MATCH, COL_PREFIX,
    COL_PREFIX_SHORT, COL_REPLACE, COL_STEM, COL_STEM_SHORT,
    COL_SUFFIX, COL_SUFFIX_SHORT, COMPATIBILITY_SECTIONS,
    DB_SECTION_ABOUT, DB_SECTION_HEADER, DB_SECTION_POSTREGEX,
    DB_SECTION_PREFIXES, DB_SECTION_SMART_BACKOFF,
    DB_SECTION_STEM_BACKOFF, DB_SECTION_STEMS, DB_SECTION_SUFFIXES,
    DB_SECTION_TABLE_AB, DB_SECTION_TABLE_AC, DB_SECTION_TABLE_BC,
    DROP_FORM, EMPTY_CONDITION, EMPTY_FIELD, EMPTY_MORPH_CLASS,
    MORPHEME_SECTIONS, MORPH_TYPE_PREFIX, MORPH_TYPE_STEM,
    MORPH_TYPE_SUFFIX, NO_ANALYSIS, NOT_WRITTEN, POS_TAG_SCHEMES,
    POS_VERB, SEG_TOK_SCHEMES, SHORT_ORDER_COLUMNS, SOURCE_LEXICON,
    STEM_METADATA_COLUMNS, almor_output_header,
)
from .debugging.download_sheets import download_sheets
from .utils.utils import Config, essential_keys_form_feats


def _make_export_metadata(config: Config) -> Dict[str, str]:
    """Build top-level JSON ``meta`` from the active db-make configuration.

    Keys are emitted in alphabetical order: ``dbName``, ``dbVersion``,
    ``description``, ``regexFormat``.
    """
    regex_format = 'python-re'

    db_filename = os.path.basename(config.db)
    db_base = os.path.splitext(db_filename)[0]
    # Extract the human-friendly database name (strip the `_vX.Y.Z...` part).
    db_name = db_base.split('_v', 1)[0] if '_v' in db_base else db_base

    # Extract numeric DB version from something like "..._v1.2.2_annex.db".
    m = re.search(r'_v([0-9]+(?:\.[0-9]+)*)', db_base)
    db_version = m.group(1) if m else 'unknown'
    if db_version != 'unknown' and '.' not in db_version:
        # Normalize "1" -> "1.0" to match the naming convention in db filenames.
        db_version = f'{db_version}.0'

    # Same label used to filter POSTREGEX sheet rows by VARIANT.
    dialect = (config.dialect or '').upper()
    description = f'morph db of {dialect}' if dialect else 'morph db'

    return {
        'dbName': db_name,
        'dbVersion': db_version,
        'description': description,
        'regexFormat': regex_format,
    }


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI args. Accepts both hyphen and underscore flag spellings."""
    parser = argparse.ArgumentParser(
        description="Build a CAMeL Morph ALMOR database."
    )

    parser.add_argument(
        "-config-file",
        dest="config_file",
        required=True,
        help="Configuration file containing database configurations.",
    )
    parser.add_argument(
        "-config-name",
        dest="config_name",
        required=True,
        help="Configuration name inside the configuration file.",
    )
    parser.add_argument(
        "-output-dir",
        dest="output_dir",
        default=None,
        help="Directory where the generated database will be written.",
    )
    parser.add_argument(
        "-debug-lemma", 
        dest="debug_lemma",
        default=None,
        help="Restrict the lexicon to one lemma for debugging.",
    )
    parser.add_argument(
        "-download",
        action="store_true",
        help="Download the specification sheets before building.",
    )
    parser.add_argument(
        "-run-profiling", 
        dest="run_profiling",
        action="store_true",
        help="Profile database construction.",
    )
    parser.add_argument(
        "-camel-tools",
        dest="camel_tools",
        default=None,
        choices=[runtime.CAMEL_TOOLS_LOCAL, runtime.CAMEL_TOOLS_OFFICIAL],
        help="Override automatic selection of the local or official camel_tools installation.",
    )

    return parser.parse_args(argv)


def _configured_bool(config: Config, name: str) -> bool:
    value = getattr(config, name, None)
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be configured as a boolean')
    return value


def _configured_positive_int(config: Config, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be configured as a positive integer')
    return value

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

def make_db(
    config: Config,
    output_path: Optional[str] = None,
    *,
    download: bool = False,
    debug_lemma: Optional[str] = None,
    json_output_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main function which takes in a set of specifications from `csv` files (downloadable
    from Google Sheets) and which, from the latter, prints out a `db` file in the ALMOR format,
    useable by the Camel Tools Analyzer and Generator engines to produce word analyses/generations.
    Always also writes a CM-schema JSON export of that DB.
    The config file is any json object which at the highest level contains: (1) global
    specifications, i.e., that apply no matter what the present local configuration contains;
    (2) local specifications, from which we source all the details that are specific to the current
    DB we are trying to build. The format of these two is dictated by the current code and how
    it reads that information. Any changes to the format should be accompanied by changes to the code
    and vice versa.

    Args:
        config (Config): dictionary containing all the necessary information to build the `db` file.
        output_path (str): path of the output DB. Defaults to None
        json_output_path (str): path of the output JSON.
            Defaults to ``config.get_db_json_path()``.
    """
    if download:
        print()
        download_sheets(config=config)
    
    morph2caphi = None
    if config.caphi is not None:
        caphi_module = importlib.import_module(config.caphi)
        morph2caphi = {
            morph_type: getattr(caphi_module, f'caphi_{morph_type}')
            for morph_type in CAPHI_MORPH_TYPES
        }
    
    logprob: Dict[str] = config.logprob
    if logprob is not None and logprob != runtime.LOGPROB_RETURN_ALL:
        with open(logprob, 'rb') as f:
            logprob = pickle.load(f)
        pos2lex2logprob = {}
        for (pos, lex), logprob_ in logprob['pos_lex'].items():
            pos2lex2logprob.setdefault(pos, {}).setdefault(lex, logprob_)
        
        for pos, lex2logprob in pos2lex2logprob.items():
            pos2lex2logprob[pos] = dict(sorted(
                lex2logprob.items(), key=lambda x: x[1], reverse=True))
        logprob['pos2lex2logprob'] = pos2lex2logprob
    elif logprob == runtime.LOGPROB_RETURN_ALL:
        logprob = None
    
    
    print("\nLoading and processing sheets... [1/5]")
    SHEETS, cond2class = db_maker_utils.read_morph_specs(config)

    if debug_lemma or config.restrict_db_to_lemma:
        lemma = debug_lemma if debug_lemma else config.restrict_db_to_lemma
        SHEETS['lexicon'] = SHEETS['lexicon'][
            SHEETS['lexicon']['LEMMA'] == f'lex:{lemma}']
    
    print("\nValidating combinations... [2/5]")
    cat2id = _configured_bool(config, 'cat2id')
    defaults = _configured_bool(config, 'defaults')
    pruning = _configured_bool(config, 'pruning')
    
    n_workers = _configured_positive_int(config, 'n_workers')
    db = construct_almor_db(SHEETS, pruning,
        cond2class, cat2id, defaults, morph2caphi, logprob, n_workers=n_workers)

    print("\nCollapsing categories and reindexing... [3/5]")
    reindex = _configured_bool(config, 'reindex')
    if reindex:
        db, _ = collapse_and_reindex_categories(db, collapse_morphemes=False)
    
    print("\nGenerating DB file... [4/5]")
    if output_path is None:
        output_path = config.get_db_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print_almor_db(output_path, db)

    print("\nExporting DB to JSON... [5/5]")
    if json_output_path is None:
        json_output_path = config.get_db_json_path()
    metadata = _make_export_metadata(config)
    export_db_to_json(
        output_path,
        json_output_path,
        metadata=metadata,
        dialect=(config.dialect or '').upper(),
    )
    print(f"Wrote JSON: {json_output_path}")

    return SHEETS


def _merge_partial_db(db: Dict, db_: Dict) -> None:
    """Merge one ORDER-line partial DB into the accumulating DB.
    Same logic as the original construct_process merge.
    """
    for section, contents in db_.items():
        if section != DB_SECTION_STEM_BACKOFF:
            db.setdefault(section, {}).update(contents)
        else:
            for backoff_mode, cats in contents.items():
                db.setdefault(section, {}).setdefault(backoff_mode, set()).update(cats)


def _map_cat_to_id(cat: str, cat2id: Dict) -> str:
    """Same ID assignment as in _generate_cat_field when cat2id is enabled."""
    if ':' not in cat:
        return cat
    morph_type = cat.split(':', 1)[0]
    cat2id_morph_type = cat2id.setdefault(morph_type, {})
    if cat in cat2id_morph_type:
        return cat2id_morph_type[cat]
    cat_ = f'{morph_type}{str(len(cat2id_morph_type) + 1).zfill(5)}'
    cat2id[morph_type][cat] = cat_
    return cat_


def _remap_partial_db_cats(db_partial: Dict, cat2id: Dict) -> Dict:
    """Apply cat2id mapping to a worker partial DB (long category names → IDs)."""
    remapped = {}
    for section, contents in db_partial.items():
        if section == DB_SECTION_STEM_BACKOFF:
            remapped[section] = {
                backoff_mode: {_map_cat_to_id(c, cat2id) for c in cats}
                for backoff_mode, cats in contents.items()
            }
        elif section in COMPATIBILITY_SECTIONS:
            remapped[section] = {
                (_map_cat_to_id(a, cat2id), _map_cat_to_id(b, cat2id)): v
                for (a, b), v in contents.items()
            }
        else:
            remapped[section] = {
                (match, _map_cat_to_id(cat, cat2id), analysis): v
                for (match, cat, analysis), v in contents.items()
            }
    return remapped


# Per-process state set once via ProcessPoolExecutor initializer (avoids re-pickling
# MORPH/cond2class/etc. on every ORDER line).
_WORKER_SHARED = {}


def _init_construct_worker(lexicon, morph, cond2class, pruning, short_cat_maps, defaults_,
                           morph2caphi, logprob, order_suffix_col):
    _WORKER_SHARED['lexicon'] = lexicon
    _WORKER_SHARED['morph'] = morph
    _WORKER_SHARED['cond2class'] = cond2class
    _WORKER_SHARED['pruning'] = pruning
    _WORKER_SHARED['short_cat_maps'] = short_cat_maps
    _WORKER_SHARED['defaults_'] = defaults_
    _WORKER_SHARED['morph2caphi'] = morph2caphi
    _WORKER_SHARED['logprob'] = logprob
    _WORKER_SHARED['order_suffix_col'] = order_suffix_col


def _construct_process_worker(args):
    """Worker for one ORDER line. Returns (order_index, partial_db_or_None, warning_or_None).

    Always builds with cat2id=None so category names are stable across processes.
    The main process remaps to IDs after merge when cat2id is enabled.
    """
    order_index, order_sequence_dict, stems_section_title = args
    order_sequence = pd.Series(order_sequence_dict)

    lexicon = _WORKER_SHARED['lexicon']
    morph = _WORKER_SHARED['morph']
    cond2class = _WORKER_SHARED['cond2class']
    pruning = _WORKER_SHARED['pruning']
    short_cat_maps = _WORKER_SHARED['short_cat_maps']
    defaults_ = _WORKER_SHARED['defaults_']
    morph2caphi = _WORKER_SHARED['morph2caphi']
    logprob = _WORKER_SHARED['logprob']
    order_suffix_col = _WORKER_SHARED['order_suffix_col']

    cmplx_prefix_classes = gen_cmplx_morph_combs(
        order_sequence[COL_PREFIX], morph, lexicon, cond2class,
        pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
    cmplx_suffix_classes = gen_cmplx_morph_combs(
        order_sequence[COL_SUFFIX], morph, lexicon, cond2class,
        pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
    cmplx_stem_classes = gen_cmplx_morph_combs(
        order_sequence[COL_STEM], morph, lexicon, cond2class,
        cmplx_morph_memoize={},
        pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)

    cmplx_type_empty = set()
    if not cmplx_stem_classes: cmplx_type_empty.add('Stem')
    if not cmplx_suffix_classes: cmplx_type_empty.add('Suffix')
    if not cmplx_prefix_classes: cmplx_type_empty.add('Prefix')
    if cmplx_type_empty:
        cmplx_type_empty = '/'.join(cmplx_type_empty)
        warning = (f"WARNING: {order_sequence[order_suffix_col]}: {cmplx_type_empty} class "
                   'is empty; proceeding to process next order line.')
        return order_index, None, warning

    cmplx_morph_classes = dict(
        cmplx_prefix_classes=(
            cmplx_prefix_classes,
            order_sequence[COL_PREFIX] if order_sequence[COL_PREFIX] else EMPTY_MORPH_CLASS,
        ),
        cmplx_suffix_classes=(
            cmplx_suffix_classes,
            order_sequence[COL_SUFFIX] if order_sequence[COL_SUFFIX] else EMPTY_MORPH_CLASS,
        ),
        cmplx_stem_classes=(cmplx_stem_classes, order_sequence[COL_STEM]),
    )

    db_ = cross_cmplx_morph_validation(
        cmplx_morph_classes, order_sequence[COL_CLASS].lower(), short_cat_maps, defaults_,
        stems_section_title, None, morph2caphi, logprob)
    return order_index, db_, None


def construct_almor_db(SHEETS:Dict[str, pd.DataFrame],
                       pruning:bool,
                       cond2class:Dict,
                       cat2id:bool=False,
                       defaults:Optional[bool]=None,
                       morph2caphi:Optional[Dict]=None,
                       logprob:Optional[Dict]=None,
                       n_workers:Optional[int]=None) -> Dict:
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
        n_workers (Optional[int]): process count for ORDER-line validation. If <= 1, runs
        in-process (needed for useful cProfile output). Must be configured explicitly.

    Returns:
        Dict: Database which contains entries (values) for each section (keys).
    """
    if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 1:
        raise ValueError('n_workers must be a positive integer')

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
    if SHORT_ORDER_COLUMNS <= set(ORDER.columns):
        short_cat_maps = _get_short_cat_name_maps(ORDER) 

    # One-time filling of the About, Header, and PostRegex sections of the DB
    db = {}
    db[DB_SECTION_ABOUT] = list(ABOUT[COL_CONTENT])
    if POSTREGEX is not None:
        db[DB_SECTION_POSTREGEX] = [
            'MATCH\t' + '\t'.join(POSTREGEX[COL_MATCH].values.tolist()),
            'REPLACE\t' + '\t'.join(POSTREGEX[COL_REPLACE].values.tolist()),
        ]
    
    header_, defaults_ = _read_header_file(HEADER)
    db[DB_SECTION_HEADER] = header_

    defaults_ = defaults_ if defaults else None
    cat2id = {} if cat2id else None
    order_suffix_col = (
        COL_SUFFIX_SHORT if COL_SUFFIX_SHORT in ORDER.columns else COL_SUFFIX
    )
    order_rows = [(i, row.to_dict()) for i, (_, row) in enumerate(ORDER.iterrows())]

    def _consume_order_result(order_index, db_, warning, pbar):
        if warning:
            tqdm.write(warning)
        pbar.set_description(str(order_rows[order_index][1][order_suffix_col]))
        pbar.update(1)
        if db_ is None:
            return
        if cat2id is not None:
            db_ = _remap_partial_db_cats(db_, cat2id)
        _merge_partial_db(db, db_)

    def _run_order_rows_parallel(lexicon, stems_section_title, pbar):
        """Run ORDER lines with n_workers processes (or in-process if n_workers <= 1)."""
        worker_args = [
            (i, order_dict, stems_section_title)
            for i, order_dict in order_rows
        ]

        # In-process path: required for useful main-process cProfile / single-line profiling.
        if n_workers <= 1:
            _init_construct_worker(
                lexicon, MORPH, cond2class, pruning, short_cat_maps, defaults_,
                morph2caphi, logprob, order_suffix_col)
            for args in worker_args:
                order_index, db_, warning = _construct_process_worker(args)
                _consume_order_result(order_index, db_, warning, pbar)
            return

        results = [None] * len(worker_args)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_construct_worker,
            initargs=(lexicon, MORPH, cond2class, pruning, short_cat_maps, defaults_,
                      morph2caphi, logprob, order_suffix_col),
        ) as executor:
            futures = {
                executor.submit(_construct_process_worker, args): args[0]
                for args in worker_args
            }
            for future in as_completed(futures):
                order_index, db_, warning = future.result()
                results[order_index] = (db_, warning)
                if warning:
                    tqdm.write(warning)
                pbar.set_description(str(order_rows[order_index][1][order_suffix_col]))
                pbar.update(1)
        for db_, _warning in results:
            if db_ is None:
                continue
            if cat2id is not None:
                db_ = _remap_partial_db_cats(db_, cat2id)
            _merge_partial_db(db, db_)

    # For memoization to work as intended, same-aspect order lines should be placed next
    # to each other in the ORDER file, and since the stem part of the order usually stays
    # the same at the aspect level, then it makes sense to avoid recomputing all the combinations
    # each time and same them in the memo. dict.
    # NOTE: with parallel workers, per-STEM memoization across ORDER lines is not shared;
    # each worker starts with an empty stem memo dict.
    for name, SHEET in [('Concrete', LEXICON), ('Backoff', BACKOFF)]:
        if SHEET is not None:
            print(f'\n{name} lexicon')
            pbar = tqdm(total=len(order_rows))
            _run_order_rows_parallel(SHEET, DB_SECTION_STEMS, pbar)
            pbar.close()

    stem_backoffs_ = {}
    if DB_SECTION_STEM_BACKOFF in db:
        for backoff_mode, cats in db[DB_SECTION_STEM_BACKOFF].items():
            stem_backoffs_[('STEMBACKOFF', backoff_mode, ' '.join(cats))] = 1
    db[DB_SECTION_STEM_BACKOFF] = stem_backoffs_
            
    #TODO: maybe this should also be included in the above loop, but more study is needed
    if SMART_BACKOFF is not None:
        print('Smart Backoff lexicon')
        pbar = tqdm(total=len(order_rows))
        _run_order_rows_parallel(SMART_BACKOFF, DB_SECTION_SMART_BACKOFF, pbar)
        pbar.close()

    return db

def cross_cmplx_morph_validation(cmplx_morph_classes: Dict,
                                 pos_type: str,
                                 short_cat_maps: Optional[Dict]=None,
                                 defaults: Dict=None,
                                 stems_section_title: str=DB_SECTION_STEMS,
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
        stems_section_title (_type_, optional): title of the section that will appear in the DB.
        morph2caphi (Dict): maps to the different methods to use to convert diac to CAPHI based
        on the complex morpheme type. Defaults to None.
        logprob (Dict): dictionary containing the log probablities of different features, extracted
        from a corpus. Defaults to None.

    Returns:
        Dict: Database in progress
    """
    db = {}
    db[DB_SECTION_STEM_BACKOFF] = {}
    db[DB_SECTION_PREFIXES] = {}
    db[DB_SECTION_SUFFIXES] = {}
    db[stems_section_title] = {}
    db[DB_SECTION_TABLE_AB] = {}
    db[DB_SECTION_TABLE_BC] = {}
    db[DB_SECTION_TABLE_AC] = {}

    cmplx_prefix_classes, cmplx_prefix_seq = cmplx_morph_classes['cmplx_prefix_classes']
    cmplx_suffix_classes, cmplx_suffix_seq = cmplx_morph_classes['cmplx_suffix_classes']
    cmplx_stem_classes, cmplx_stem_seq = cmplx_morph_classes['cmplx_stem_classes']
    
    cat_memoize = {'stem': {}, 'suffix': {}, 'prefix': {}}
    # Reused COND-S buffer: almost all stem×prefix×suffix triples are unique, so a
    # compatibility memo dict would only add allocation overhead on this hot path.
    cs_buf = set()

    # Pre-parse COND-S/T/F once per complex-morpheme class (avoids re-join/re-split
    # on every stem×prefix×suffix triple — the old hot path in check_compatibility).
    stem_cond_info = {}
    for cmplx_stem_cls, cmplx_stems in cmplx_stem_classes.items():
        stem_cond_s = ' '.join([f['COND-S'] for f in cmplx_stems[0]])
        stem_cond_t = ' '.join([f['COND-T'] for f in cmplx_stems[0]])
        stem_cond_f = ' '.join([f['COND-F'] for f in cmplx_stems[0]])
        stem_cond_info[cmplx_stem_cls] = (
            cmplx_stems, stem_cond_s, stem_cond_t, stem_cond_f,
            _parse_condition_fingerprint(stem_cond_s, stem_cond_t, stem_cond_f))

    prefix_cond_info = {}
    for cmplx_prefix_cls, cmplx_prefixes in cmplx_prefix_classes.items():
        prefix_cond_s = ' '.join([f['COND-S'] for f in cmplx_prefixes[0]])
        prefix_cond_t = ' '.join([f['COND-T'] for f in cmplx_prefixes[0]])
        prefix_cond_f = ' '.join([f['COND-F'] for f in cmplx_prefixes[0]])
        prefix_cond_info[cmplx_prefix_cls] = (
            cmplx_prefixes, prefix_cond_s, prefix_cond_t, prefix_cond_f,
            _parse_condition_fingerprint(prefix_cond_s, prefix_cond_t, prefix_cond_f))

    suffix_cond_info = {}
    for cmplx_suffix_cls, cmplx_suffixes in cmplx_suffix_classes.items():
        suffix_cond_s = ' '.join([f['COND-S'] for f in cmplx_suffixes[0]])
        suffix_cond_t = ' '.join([f['COND-T'] for f in cmplx_suffixes[0]])
        suffix_cond_f = ' '.join([f['COND-F'] for f in cmplx_suffixes[0]])
        suffix_cond_info[cmplx_suffix_cls] = (
            cmplx_suffixes, suffix_cond_s, suffix_cond_t, suffix_cond_f,
            _parse_condition_fingerprint(suffix_cond_s, suffix_cond_t, suffix_cond_f))

    for cmplx_stem_cls, (cmplx_stems, stem_cond_s, stem_cond_t, stem_cond_f,
                         (stem_cs, stem_ct, stem_cf)) in stem_cond_info.items():
        # `cmplx_stem_cls` = (cmplx_stem['COND-S'], cmplx_stem['COND-T'], cmplx_stem['COND-F'])
        # All entries in `cmplx_stems` have the same cat
        for cmplx_prefix_cls, (cmplx_prefixes, prefix_cond_s, prefix_cond_t, prefix_cond_f,
                               (prefix_cs, prefix_ct, prefix_cf)) in prefix_cond_info.items():
            #TODO: should probably move this loop to be the outermost one (instead of stem) 
            # and should check if there are interactions between morpheme class/condition
            # pairs between prefix and the stem/suffix. If there are none, then there would
            # be no need to loop multiple times to validate. The idea is to reduce the number
            # of complex morpheme classes we are looping over. If adding a condition which
            # is internal classes to one complex morpheme category, i.e., only appears in
            # in morpheme classes that appear in only one of complex prefix, suffix, or stem,
            # then this condition should not have interactions with the other two complex
            # morpheme categories.
            for cmplx_suffix_cls, (cmplx_suffixes, suffix_cond_s, suffix_cond_t, suffix_cond_f,
                                   (suffix_cs, suffix_ct, suffix_cf)) in suffix_cond_info.items():
                cs_buf.clear()
                cs_buf.update(prefix_cs)
                cs_buf.update(stem_cs)
                cs_buf.update(suffix_cs)
                valid = _check_compatibility_with_cs(
                    cs_buf, prefix_ct, stem_ct, suffix_ct, prefix_cf, stem_cf, suffix_cf)
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
                                              db_section=DB_SECTION_PREFIXES)
                    update_info_suffix = dict(pos_type=pos_type,
                                              cmplx_morph_seq=cmplx_suffix_seq,
                                              cmplx_morph_cls=cmplx_suffix_cls,
                                              cmplx_morph_type='suffix',
                                              cmplx_morphs=cmplx_suffixes,
                                              conditions=(suffix_cond_s, suffix_cond_t, suffix_cond_f),
                                              db_section=DB_SECTION_SUFFIXES)
                    
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

                    db[DB_SECTION_TABLE_AB][(prefix_cat, stem_cat)] = 1
                    db[DB_SECTION_TABLE_BC][(stem_cat, suffix_cat)] = 1
                    db[DB_SECTION_TABLE_AC][(prefix_cat, suffix_cat)] = 1
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

    # This if statement implements early stopping which entails that if we have already 
    # logged a specific prefix/stem/suffix entry, we do not need to do it again. Entry
    # generation (and more specifically `dediac()`) is costly.
    if cat_memoize[cmplx_morph_type].get(cmplx_morph_cls) is None:
        for cmplx_morph in cmplx_morphs:
            morph_entry = _generate(
                cmplx_morph_seq, cmplx_morph, cond_s, cond_t, cond_f,
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

            if morph_entry['match'] == NO_ANALYSIS:
                for backoff_mode in morph_entry['analysis']['backoff_modes'].split():
                    db[DB_SECTION_STEM_BACKOFF].setdefault(backoff_mode, set()).add(
                        morph_entry['cat']
                    )
        cat_memoize[cmplx_morph_type][cmplx_morph_cls] = morph_entry['cat']


def _generate_cat_field(cmplx_morph_type: str, cmplx_morph_class: str,
                cmplx_morph_cond_s: str, cmplx_morph_cond_t: str, cmplx_morph_cond_f: str,
                short_cat_map: Optional[Dict]=None,
                cat2id:Optional[Dict]=None):
    """This function creates the category for matching using classes and conditions"""
    if short_cat_map:
        cmplx_morph_class = short_cat_map[cmplx_morph_class]
    cmplx_morph_cond_s = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_s.split()) if cond != EMPTY_FIELD])
    cmplx_morph_cond_s = (
        cmplx_morph_cond_s if cmplx_morph_cond_s else EMPTY_CONDITION
    )
    cmplx_morph_cond_t = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_t.split()) if cond != EMPTY_FIELD])
    cmplx_morph_cond_t = (
        cmplx_morph_cond_t if cmplx_morph_cond_t else EMPTY_CONDITION
    )
    cmplx_morph_cond_f = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_f.split()) if cond != EMPTY_FIELD])
    cmplx_morph_cond_f = (
        cmplx_morph_cond_f if cmplx_morph_cond_f else EMPTY_CONDITION
    )
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
            bw_lex = parts[0] if backoff else runtime.BW2AR(parts[0])
        bw_pos = parts[1]
        utf8_bw_tag.append('/'.join([bw_lex, bw_pos]))
    return '+'.join(utf8_bw_tag)

def _generate_match_field(diac):
    # Strip postregex markers from the lookup key only; surface fields keep them
    # for POSTREGEX (see runtime.PRE_POST_REGEX_SYMBOL).
    diac_ = runtime.PRE_POST_REGEX_SYMBOL.sub('', diac)
    diac_ = diac_.replace('_', '')
    diac_ = dediac_bw(diac_)
    diac_ = normalize_teh_marbuta_bw(diac_)
    diac_ = normalize_alef_maksura_bw(diac_)
    diac_ = normalize_alef_bw(diac_)
    return diac_


def _generate_caphi(morpheme, caphi_list, caphi_copy, morph2caphi, cmplx_morpheme_type):
    copy_list = [m[caphi_copy] if m[caphi_copy] != EMPTY_FIELD else '' for m in morpheme]
    if len(set(caphi_list)) > 1 and '' in caphi_list:
        raise NotImplementedError
    value = []
    if set(caphi_list) == {''} and morph2caphi is not None:
        value.append(morph2caphi[cmplx_morpheme_type](''.join(copy_list)))
    else:
        for i, v in enumerate(caphi_list):
            if v == EMPTY_FIELD:
                continue
            elif v:
                value.append(v)
            else:
                if copy_list[i] and copy_list[i] != EMPTY_FIELD:
                    if morph2caphi is not None:
                        value.append(morph2caphi[cmplx_morpheme_type](copy_list[i]))
    value = ' '.join(value).strip().replace(' ', '_')
    value = runtime.CAPHI_UNDERSCORE_RE_1.sub('_', value)
    value = runtime.CAPHI_UNDERSCORE_RE_2.sub('', value)
    return value


def _join_sheet_pos_tags(values) -> str:
    """Join non-empty per-morpheme UD/CATIB6 sheet values with '+'."""
    if isinstance(values, str):
        return values
    return '+'.join(v for v in values if v and v != EMPTY_FIELD)


def _assign_ud_catib_from_sheets(analysis: Dict) -> None:
    """Set analysis['ud'] / analysis['catib6'] from MORPH/LEX sheet columns.

    Values are read per morpheme (like D3SEG/ATBTOK) and concatenated with '+'.
    Empty sheet cells mean the morpheme contributes no POS tag (e.g. case/NSUFF).
    No BW→POS fallback: UD/CATiB must come from the sheets.
    """
    # Always assign (including '') so inflectional affixes do not leak raw BW tags
    analysis['ud'] = _join_sheet_pos_tags(analysis.get('ud', []))
    analysis['catib6'] = _join_sheet_pos_tags(analysis.get('catib6', []))


def _generate_affix(affix_type: str,
                    cmplx_morph_seq: str,
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
    affix_type = CAT_TYPE_PREFIX if affix_type == 'prefix' else CAT_TYPE_SUFFIX
    acat = _generate_cat_field(affix_type, cmplx_morph_seq, affix_cond_s, affix_cond_t,
                       affix_cond_f, short_cat_map, cat2id)
    analysis['bw'] = _convert_bw_tag(analysis['bw'])
    _assign_ud_catib_from_sheets(analysis)
    affix_type_ = (
        MORPH_TYPE_PREFIX if affix_type == CAT_TYPE_PREFIX else MORPH_TYPE_SUFFIX
    )
    
    for col in SEG_TOK_SCHEMES:
        col = col.lower()
        tok_copy = defaults['tokenization'][col]
        value = ''.join(
            v if v else (
                affix[i][tok_copy] if affix[i][tok_copy] != EMPTY_FIELD else ''
            )
            for i, v in enumerate(analysis[col]))
        analysis[col] = value
    
    analysis['caphi'] = _generate_caphi(
        affix, analysis['caphi'], defaults['transcription']['caphi'], morph2caphi, affix_type_)
    
    for f in BW2AR_AFFIX_FIELDS:
        analysis[f] = runtime.BW2AR(analysis[f])

    affix = {
        # affix_match already had postregex markers stripped in _generate_match_field.
        'match': runtime.BW2AR(affix_match),
        'cat': acat,
        'analysis': analysis,
    }
    return affix


def _generate_stem(cmplx_morph_seq: str,
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
    _assign_ud_catib_from_sheets(analysis)

    if defaults is not None:
        pos_defaults = defaults['defaults'].get(analysis['pos'], {})

        for f, default in pos_defaults.items():
            if default in [None, '', '*']:
                continue

            if analysis.get(f) in [None, '', EMPTY_FIELD]:
                analysis[f] = default

    if '-' in analysis['lex'] and analysis['pos'] == POS_VERB:
        part2 = analysis['lex'].split('-')[1]
        mid_root_diac = part2.split('_')[0] if '_' in part2 else part2
        analysis['mid_root_diac'] = mid_root_diac
    
    if backoff == BACKOFF_SMART:
        match = db_maker_utils._bw2ar_regex(stem_match, runtime.BW2AR)
    elif backoff == BACKOFF_VANILLA:
        match = stem_match
        analysis['backoff_modes'] = analysis['lex']
        analysis['lex'] = NO_ANALYSIS
    else:
        match = runtime.BW2AR(stem_match)

    xcat = _generate_cat_field(CAT_TYPE_STEM, cmplx_morph_seq, stem_cond_s, stem_cond_t,
                               stem_cond_f, short_cat_map, cat2id)
    
    if not backoff:
        for col in SEG_TOK_SCHEMES:
            col = col.lower()
            tok_copy = defaults['tokenization'][col]
            value = ''.join(
                v if v else (
                    stem[i][tok_copy] if stem[i][tok_copy] != EMPTY_FIELD else ''
                )
                for i, v in enumerate(analysis[col]))
            if value != analysis['diac']:
                analysis[col] = value
            else:
                del analysis[col]
        
        analysis['caphi'] = _generate_caphi(
            stem,
            analysis['caphi'],
            defaults['transcription']['caphi'],
            morph2caphi,
            MORPH_TYPE_STEM,
        )
        
        if logprob is not None:
            for f in runtime.LOGPROB_FEATURES:
                lex_ = tuple(analysis[f_]
                             if f_ != 'lex' else strip_lex(analysis[f_])
                             for f_ in f.split('_'))
                analysis[f'{f}_logprob'] = (
                    f'{logprob[f][lex_]:.6f}'
                    if lex_ in logprob[f]
                    else runtime.MISSING_LOGPROB
                )

        for f in BW2AR_STEM_FIELDS:
            if f in analysis:
                if analysis[f] == NOT_WRITTEN or analysis[f] is None:
                    continue
                analysis[f] = runtime.BW2AR(analysis[f])

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
    analysis['bw'] = '+'.join(
        m['BW'] for m in affix if m['BW'] != EMPTY_FIELD
    )
    analysis['gloss'] = '+'.join(m['GLOSS'] for m in affix
                                 if m['GLOSS'] and m['GLOSS'] != EMPTY_FIELD)
    affix_feat = {feat.split(':')[0]: feat.split(':')[1]
                  for m in affix for feat in m['FEAT'].split()}
    analysis = {**analysis, **affix_feat}

    analysis['diac'] = ''.join(
        m['FORM'] for m in affix if m['FORM'] != EMPTY_FIELD
    )
    
    for col in (*SEG_TOK_SCHEMES, *POS_TAG_SCHEMES, CAPHI_COLUMN):
        analysis[col.lower()] = [m.get(col, '') for m in affix]
    
    source = [m['SOURCE'] for m in affix if m.get('SOURCE')]
    if source and any(source):
        analysis['source'] = source[0]
    else:
        analysis['source'] = SOURCE_LEXICON
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
    analysis['bw'] = '+'.join(
        s['BW'] for s in stem if s['BW'] != EMPTY_FIELD
    )
    analysis['gloss'] = '+'.join(s['GLOSS'] for s in stem
                                 if s['GLOSS'] and s['GLOSS'] != EMPTY_FIELD)
    analysis['lex'] = '+'.join(
        s['LEMMA'].split(':')[1] for s in stem if 'LEMMA' in s)
    stem_feat = {feat.split(':')[0]: feat.split(':')[1]
                for s in stem for feat in s['FEAT'].split()}
    analysis = {**analysis, **stem_feat}

    analysis['diac'] = ''.join(
        s['FORM'] for s in stem if s['FORM'] != EMPTY_FIELD
    )
    
    for col in STEM_METADATA_COLUMNS:
        feat = [s[col] for s in stem if s.get(col)]
        if feat and any(feat):
            analysis[col.lower()] = feat[0]
        elif col == 'SOURCE':
            analysis['source'] = SOURCE_LEXICON
        else:
            analysis[col.lower()] = NOT_WRITTEN

    analysis['cm_stem_ids'] = '+'.join(
        s['CLASS'] + ':' + str(int(float(s['LINE'] if s['LINE'] else -1)))
        for s in stem)
    analysis['cm_stem'], analysis['cm_buffer'] = stem[0]['FORM'], None
    if len(stem) == 2 and stem[1]['FORM'] not in [EMPTY_FIELD, '']:
        analysis['cm_buffer'] = stem[1]['FORM']

    stem_defines = set(s['DEFINE'] for s in stem)
    if 'SMARTBACKOFF' in stem_defines:
        assert stem_defines <= {'MORPH', 'SMARTBACKOFF'}
        backoff = BACKOFF_SMART
        stem_match = []
        for s in stem:
            if s['FORM'] != EMPTY_FIELD:
                if s['DEFINE'] == 'SMARTBACKOFF':
                    stem_match.append(
                        runtime.PRE_POST_REGEX_SYMBOL_SMARTBACKOFF.sub('', s['MATCH']))
                else:
                    stem_match.append(_generate_match_field(s['FORM']))
        stem_match = f"^{''.join(stem_match)}$"
    elif 'BACKOFF' in  stem_defines:
        backoff = BACKOFF_VANILLA
        assert stem_defines <= {'MORPH', 'BACKOFF'}
        stem_match = NO_ANALYSIS
    else:
        backoff = None
        stem_match = _generate_match_field(analysis['diac'])
        for col in (*SEG_TOK_SCHEMES, CAPHI_COLUMN):
            analysis[col.lower()] = [s.get(col, '') for s in stem]

    # UD/CATIB6 come from sheets for all stem types (concrete + backoff).
    for col in POS_TAG_SCHEMES:
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
    prefix_stem_compat_ = _read_compatibility_tables(db[DB_SECTION_TABLE_AB])
    stem_suffix_compat_ = _read_compatibility_tables(db[DB_SECTION_TABLE_BC])
    prefix_suffix_compat_ = _read_compatibility_tables(db[DB_SECTION_TABLE_AC])

    prefixes_ = db[DB_SECTION_PREFIXES]
    stems_ = db[DB_SECTION_STEMS]
    suffixes_ = db[DB_SECTION_SUFFIXES]
    backoff_stems_ = db[DB_SECTION_STEM_BACKOFF]

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
        #TODO: the following (morpheme) collapsing is still unstable and is not being
        # used for the moment. Should be debugged. Its purpose is to be ran after
        # category collapsing/reindexing and collapse redundant entries based on the
        # new categories.
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

    db[DB_SECTION_PREFIXES] = prefixes_
    db[DB_SECTION_STEMS] = stems_
    db[DB_SECTION_SUFFIXES] = suffixes_
    db[DB_SECTION_TABLE_AB] = prefix_stem_compat_
    db[DB_SECTION_TABLE_BC] = stem_suffix_compat_
    db[DB_SECTION_TABLE_AC] = prefix_suffix_compat_
    db[DB_SECTION_STEM_BACKOFF] = backoff_stems_

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
        for x in db[DB_SECTION_HEADER]:
            print(x, file=f)

        print(almor_output_header(DB_SECTION_STEM_BACKOFF), file=f)
        for x in db[DB_SECTION_STEM_BACKOFF]:
            print(*x, sep=' ', file=f)
        
        postregex = db.get(DB_SECTION_POSTREGEX)
        if postregex:
            print(almor_output_header(DB_SECTION_POSTREGEX), file=f)
            for x in postregex:
                print(x, file=f)

        for section_name, section_key in MORPHEME_SECTIONS:
            if section_key not in db:
                raise ValueError(
                    f'Empty {section_name} section. Something might be wrong with the sheets.'
                )

        print(almor_output_header(DB_SECTION_PREFIXES), file=f)
        for x in db[DB_SECTION_PREFIXES]:
            print(*x, sep='\t', file=f)
            
        print(almor_output_header(DB_SECTION_SUFFIXES), file=f)
        for x in db[DB_SECTION_SUFFIXES]:
            print(*x, sep='\t', file=f)
        
        underscore_ar = re.compile('ـ')
        print(almor_output_header(DB_SECTION_STEMS), file=f)
        for x in db[DB_SECTION_STEMS]:
            # Fixes weird underscore generated by bw2ar()
            x = (*x[:2], underscore_ar.sub('_', x[2]))
            print(*x, sep='\t', file=f)

        smart_backoff = db.get(DB_SECTION_SMART_BACKOFF)
        if smart_backoff:
            print(almor_output_header(DB_SECTION_SMART_BACKOFF), file=f)
            for x in db[DB_SECTION_SMART_BACKOFF]:
                print(*x, sep='\t', file=f)
            
        print(almor_output_header(DB_SECTION_TABLE_AB), file=f)
        for x in db[DB_SECTION_TABLE_AB]:
            print(*x, sep=' ', file=f)
            
        print(almor_output_header(DB_SECTION_TABLE_BC), file=f)
        for x in db[DB_SECTION_TABLE_BC]:
            print(*x, sep=' ', file=f)
            
        print(almor_output_header(DB_SECTION_TABLE_AC), file=f)
        for x in db[DB_SECTION_TABLE_AC]:
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
        p, x, s = row[COL_PREFIX], row[COL_STEM], row[COL_SUFFIX]
        p = EMPTY_MORPH_CLASS if p == '' else p
        s = EMPTY_MORPH_CLASS if s == '' else s
        p_short = row[COL_PREFIX_SHORT]
        x_short = row[COL_STEM_SHORT]
        s_short = row[COL_SUFFIX_SHORT]
        p_short = EMPTY_MORPH_CLASS if p_short == '' else p_short
        s_short = EMPTY_MORPH_CLASS if s_short == '' else s_short
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
        cmplx_morph_seq = EMPTY_MORPH_CLASS

    cmplx_morph_classes = []
    for cmplx_morph_cls in cmplx_morph_seq.split():
        sheet = LEXICON if 'STEM' in cmplx_morph_cls else MORPH
        instances = []
        for _, row in sheet[sheet.CLASS == cmplx_morph_cls].iterrows():
            if 'STEM' in cmplx_morph_cls and (
                row['FORM'] == '' or row['FORM'] == DROP_FORM
            ):
                continue
            instances.append(row.to_dict())
        if not instances:
            return {}
        cmplx_morph_classes.append(instances)
    
    cmplx_morphs = [list(t) for t in itertools.product(*[mc for mc in cmplx_morph_classes if mc])]
    cmplx_morph_categorized = {}
    for seq in cmplx_morphs:
        #TODO: maybe can reduce number of classes by uniquing and sorting?
        # Maybe there is even no need for a 3-tuple, they could all be in one string
        # (and thus reducing compilation time). I don't really know if this
        # is conceptually possible. Should try it, and see if the same DB is produced
        cmplx_morph_class = [(morph['COND-S'], morph['COND-T'], morph['COND-F']) for morph in seq]
        cmplx_morph_categorized.setdefault(tuple(cmplx_morph_class), []).append(seq)
    
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
                    if cond == EMPTY_FIELD:
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


def _parse_condition_fingerprint(cond_s: str, cond_t: str, cond_f: str):
    """Parse COND-S/T/F strings once into set / alt-tuples for fast checks."""
    cs = frozenset(cond_s.split())
    ct = tuple(tuple(t.split('||')) for t in cond_t.split())
    cf = tuple(tuple(f.split('||')) for f in cond_f.split())
    return cs, ct, cf


def _check_compatibility_with_cs(cs, prefix_ct, stem_ct, suffix_ct,
                                 prefix_cf, stem_cf, suffix_cf) -> bool:
    """Compatibility check against an already-built COND-S set."""
    # COND-T: each term must have at least one alternative present in COND-S
    for terms in (prefix_ct, stem_ct, suffix_ct):
        for alts in terms:
            for ort in alts:
                if ort in cs:
                    break
            else:
                return False
    # COND-F: no alternative may be present in COND-S
    for terms in (prefix_cf, stem_cf, suffix_cf):
        for alts in terms:
            for orf in alts:
                if orf in cs:
                    return False
    return True


def check_compatibility(cond_s: str, cond_t: str, cond_f: str,
                        compatibility_memoize: Dict) -> bool:
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
        compatibility_memoize (Dict): dictionary keeping track of combinations that were
        previously validated to avoid recomputing them a second time.

    Returns:
        bool: whether a combination of complex morphemes is valid or not. If it is valid, all the
        complex morphemes in it are secured a place in the DB.
    """
    key = (cond_s, cond_t, cond_f)
    cached = compatibility_memoize.get(key)
    if cached is not None:
        return cached
    cs = set(cond_s.split())
    ct = tuple(tuple(t.split('||')) for t in cond_t.split())
    cf = tuple(tuple(f.split('||')) for f in cond_f.split())
    empty = ()
    valid = _check_compatibility_with_cs(cs, ct, empty, empty, cf, empty, empty)
    compatibility_memoize[key] = valid
    return valid


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


def run_profiled(
    config: Config,
    output_path: str,
    *,
    download: bool,
    debug_lemma: Optional[str],
    json_output_path: Optional[str],
) -> None:
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        make_db(
            config,
            output_path,
            download=download,
            debug_lemma=debug_lemma,
            json_output_path=json_output_path,
        )
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats()


def main() -> None:
    args =  parse_args()
    config = Config(args.config_file, args.config_name)

    output_dir = args.output_dir or config.get_db_dir_path()
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, config.db)
    json_output_path = os.path.join(output_dir, config.db_json) if config.db_json else None

    if args.run_profiling:
        run_profiled(
            config=config,
            output_path=output_path,
            download=args.download,
            debug_lemma=args.debug_lemma,
            json_output_path=json_output_path,
        )
    else:
        make_db(
            config,
            output_path,
            download=args.download,
            debug_lemma=args.debug_lemma,
            json_output_path=json_output_path,
        )


if __name__ == "__main__":
    main()
