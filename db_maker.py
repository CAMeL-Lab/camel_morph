###########################################
#CalimaStar DB Maker
#Habash, Khalifa, 2020-2021
#
# This code takes in a linguistitically informed 
# morphology specification, and outputs a DB
# in the ALMOR format to be used in CamelTools
#
###########################################

import re
import os
from tqdm import tqdm
import json
import argparse
import itertools
from time import strftime, gmtime, process_time
from functools import partial
import cProfile, pstats

from camel_tools.utils.dediac import dediac_bw
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw
from generate_passive import generate_passive

import pandas as pd
import numpy as np

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

_required_verb_stem_feats = ['pos', 'asp', 'per', 'gen', 'num', 'vox', 'mod']
_required_nom_stem_feats = ['pos', 'form_gen', 'form_num', 'gen', 'num', 'stt', 'cas']
_clitic_feats = ['enc0', 'prc0', 'prc1', 'prc2', 'prc3']

###########################################
#Input File: XLSX containing specific worksheets: About, Header, Order, Morph, Lexicon
#inputfilename="CamelMorphDB-Nov-2021.xlsx"
# Primary function 
# e.g. make_db("CamelMorphDB-Nov-2021.xlsx", "config_pv_msa_order-v4")
###########################################

def make_db(config_file, config_name, output_dir):
    """Initializes `ABOUT`, `HEADER`, `ORDER`, `MORPH`, and `LEXICON`"""
    c0 = process_time()
    
    print("\nLoading and processing sheets... [1/3]")
    #Config file specifies which sheets to read in the xlsx spreadsheet
    with open(config_file) as f:
        config = json.load(f)
    SHEETS, cond2class = read_morph_specs(config, config_name)
    
    print("\nValidating combinations... [2/3]")
    db = construct_almor_db(
        SHEETS, config['local'][config_name]['pruning'], cond2class)
    
    print("\nGenerating DB file... [3/3]")
    print_almor_db(
        os.path.join(output_dir, config['local'][config_name]['output']), db)
    
    c1 = process_time()
    print(f"\nTotal time required: {strftime('%M:%S', gmtime(c1 - c0))}")


def read_morph_specs(config, config_name):
    """Read Input file containing morphological specifications"""
    data_dir = config['global']['data-dir']
    local_specs = config['local'][config_name]

    ABOUT = pd.read_csv(os.path.join(data_dir, 'About.csv'))
    HEADER = pd.read_csv(os.path.join(data_dir, 'Header.csv'))
    order_filename = f"{local_specs['specs']['order']}.csv"
    ORDER = pd.read_csv(os.path.join(data_dir, order_filename))
    morph_filename = f"{local_specs['specs']['morph']}.csv"
    MORPH = pd.read_csv(os.path.join(data_dir, morph_filename))
    lexicon_sheets = local_specs['lexicon']['sheets']
    LEXICON = pd.concat([pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
                         for name in lexicon_sheets])
    # Replace spaces in BW and GLOSS with '#'; skip commented rows and empty lines
    LEXICON = LEXICON[LEXICON.DEFINE == 'LEXICON']
    LEXICON = LEXICON.replace(np.nan, '', regex=True)
    LEXICON['GLOSS'] = LEXICON['GLOSS'].replace('\s+', '#', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' +', ' ', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' $', '', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' +', ' ', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' $', '', regex=True)

    patterns_path = local_specs['specs'].get('passive')
    if patterns_path:
        LEXICON_PASS = generate_passive(LEXICON, os.path.join(data_dir, f"{patterns_path}.csv"))
        LEXICON = pd.concat([LEXICON, LEXICON_PASS])

    LEXICON['BW'] = LEXICON['FORM'] + '/' + LEXICON['BW']
    LEXICON['LEMMA'] = 'lex:' + LEXICON['LEMMA']

    POSTREGEX = None
    postregex_path = local_specs['specs'].get('postregex')
    if postregex_path:
        POSTREGEX = pd.read_csv(os.path.join(data_dir, 'PostRegex.csv'))
        POSTREGEX = POSTREGEX[POSTREGEX.DEFINE == 'POSTREGEX']
        POSTREGEX = POSTREGEX.replace(np.nan, '', regex=True)
        POSTREGEX = _process_postregex(POSTREGEX)
    
    if local_specs.get('split_or') == True:
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

    #Process all the components:
    ORDER = ORDER[ORDER.DEFINE == 'ORDER']  # skip comments & empty lines
    ORDER = ORDER.replace(np.nan, '', regex=True)

    # Dictionary which groups conditions into classes (used later to
    # do partial compatibility which is useful from pruning out incoherent
    # suffix/prefix/stem combinations before performing full compatibility
    # in which (prefix, stem, suffix) instances are tried out individually).
    class2cond = MORPH[MORPH.DEFINE == 'CONDITIONS']
    class2cond = {cond_class["CLASS"]:
                            [cond for cond in cond_class["FUNC"].split() if cond]
                        for _, cond_class in class2cond.iterrows()}
    # Reverses the dictionary (k -> v and v -> k) so that individual conditions
    # which belonged to a class are now keys with that class as a value. In addition,
    # each condition gets its corresponding one-hot vector (actually stored as an int
    # because bitwise operations can only be performed on ints) computed based on the
    # other conditions within the same class (useful for pruning later).
    cond2class = {
        cond: (cond_class, 
               int(''.join(['1' if i == index else '0' for index in range (len(cond_s))]), 2)
        )
        for cond_class, cond_s in class2cond.items()
            for i, cond in enumerate(cond_s)}

    if os.path.exists('morph_cache/morph_sheet_prev.pkl'):
        MORPH_prev = pd.read_pickle('morph_cache/morph_sheet_prev.pkl')
        if MORPH.equals(MORPH_prev):
            MORPH = pd.read_pickle('morph_cache/morph_sheet_processed.pkl')
            SHEETS = dict(about=ABOUT, header=HEADER,
                          order=ORDER, morph=MORPH, lexicon=LEXICON, postregex=POSTREGEX)
            return SHEETS, cond2class
    MORPH.to_pickle('morph_cache/morph_sheet_prev.pkl')
    
    # Skip commented rows and empty lines
    MORPH = MORPH[MORPH.DEFINE == 'MORPH']
    MORPH = MORPH.replace(np.nan, '', regex=True)
    MORPH = MORPH.replace('^\s+', '', regex=True)
    MORPH = MORPH.replace('\s+$', '', regex=True)
    MORPH = MORPH.replace('\s+', ' ', regex=True)
    # add FUNC and FEAT to COND-S
    MORPH['COND-S'] = MORPH['COND-S'].replace('[\[\]]', '', regex=True)
    MORPH.loc[MORPH['COND-S'] == '', 'COND-S'] = '_'
    MORPH.loc[MORPH['COND-T'] == '', 'COND-T'] = '_'
    # Replace spaces in BW and GLOSS with '#'
    MORPH['BW'] = MORPH['BW'].replace('\s+', '#', regex=True)
    MORPH.loc[MORPH['BW'] == '', 'BW'] = '_'
    MORPH.loc[MORPH['FORM'] == '', 'FORM'] = '_'
    MORPH['GLOSS'] = MORPH['GLOSS'].replace('\s+', '#', regex=True)
    # Retroactively generate the condFalse by creating the complementry distribution
    # of all the conditions within a single morpheme (all the allomorphs)
    # This is perfomed here once instead of on the fly.
    # Get all the classes in Morph
    MORPH_CLASSES = MORPH.CLASS.unique()
    MORPH_CLASSES = MORPH_CLASSES[MORPH_CLASSES != '_']
    # Get all the morphemes in Morph
    MORPH_MORPHEMES = MORPH.FUNC.unique()
    MORPH_MORPHEMES = MORPH_MORPHEMES[MORPH_MORPHEMES != '_']
    # Go through each class
    for CLASS in MORPH_CLASSES:
        # Go through each morpheme
        for MORPHEME in MORPH_MORPHEMES: 

            # Get the unique list of the true conditions in all the allomorphs.
            # We basically want to convert the 'COND-T' column in to n columns
            #   where n = maximum number of conditions for a single allomorph
            cond_t = MORPH[(MORPH.FUNC == MORPHEME) & 
                                (MORPH.CLASS == CLASS)]['COND-T'].str.split(pat=' ', expand=True)
            if cond_t.empty:
                continue
            cond_t = cond_t.replace(['_'], '')

            if len(cond_t.iloc[:][:]) == 1 and cond_t.iloc[0][0] == '':
                continue
            # Go through each column in the true conditions
            for col in cond_t:
              
                # if we are at the first columns
                if col == 0:
                    #TODO: uniq col[0] and remove fluff -> c_T
                    cond_t_current = get_clean_set(cond_t[col])
                    # Go through each allomorph
                    MORPH = get_morph_cond_f(cond_t[col], cond_t_current, MORPH)
                # we are in col > 0
                else:
                    #TODO: create temp_T by merging the the condTrue for col 0-col, put it in 
                    # col[0] and remove 1:col. This is basically to group allomorph with 
                    # similar general conditions
                    # Take the uniq of what is now col[0] and remove the fluff
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
                #   current allomorph, remove the negation and add it to the false 
                #   conditions list for the current alomorph.
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
    MORPH.to_pickle('morph_cache/morph_sheet_processed.pkl')
    SHEETS = dict(about=ABOUT, header=HEADER, order=ORDER, morph=MORPH, lexicon=LEXICON, postregex=POSTREGEX)
    return SHEETS, cond2class
   

def construct_almor_db(SHEETS, pruning, cond2class):
    ORDER, MORPH, LEXICON = SHEETS['order'], SHEETS['morph'], SHEETS['lexicon']
    ABOUT, HEADER, POSTREGEX = SHEETS['about'], SHEETS['header'], SHEETS['postregex']

    short_cat_maps = _get_short_cat_name_maps(ORDER)

    db = {}
    db['OUT:###ABOUT###'] = list(ABOUT['Content'])
    db['OUT:###HEADER###'] = list(HEADER['Content'])
    if POSTREGEX is not None:
        db['OUT:###POSTREGEX###'] = [
            "###POSTREGEX###",
            'MATCH\t' + '\t'.join([match for match in POSTREGEX['MATCH'].values.tolist()]),
            'REPLACE\t' + '\t'.join([replace for replace in POSTREGEX['REPLACE'].values.tolist()])
        ]
    
    defaults = _process_defaults(db['OUT:###HEADER###'])
    
    compatibility_memoize = {}
    cmplx_morph_memoize = {'stem': {}, 'suffix': {}, 'prefix': {}}
    stem_pattern = re.compile(r'\[(STEM(?:-[PIC]V)?)\]')
    # Currently not used
    # TODO: see how different order lines can benefit from each other's information
    # when it comes to affixes
    prefix_pattern = re.compile(r'(\[(IVPref)\.\d\w{1,2}\])')
    suffix_pattern = re.compile(r'(\[(IVSuff)\.\w\.\d\w{1,2}\])')
    
    pbar = tqdm(total=len(list(ORDER.iterrows())))
    for _, order in ORDER.iterrows():
        cmplx_prefix_classes = gen_cmplx_morph_combs(
            order['PREFIX'], MORPH, LEXICON, cond2class, prefix_pattern,
            cmplx_morph_memoize=cmplx_morph_memoize['prefix'],
            pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
        cmplx_suffix_classes = gen_cmplx_morph_combs(
            order['SUFFIX'], MORPH, LEXICON, cond2class, suffix_pattern,
            cmplx_morph_memoize=cmplx_morph_memoize['suffix'],
            pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
        cmplx_stem_classes = gen_cmplx_morph_combs(
            order['STEM'], MORPH, LEXICON, cond2class, stem_pattern,
            cmplx_morph_memoize=cmplx_morph_memoize['stem'],
            pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
        
        cmplx_morph_classes = dict(
            cmplx_prefix_classes=(cmplx_prefix_classes, order['PREFIX']),
            cmplx_suffix_classes=(cmplx_suffix_classes, order['SUFFIX']),
            cmplx_stem_classes=(cmplx_stem_classes, order['STEM']))
        
        db_ = populate_db(
            cmplx_morph_classes, order['CLASS'].lower(), compatibility_memoize,
            short_cat_maps, defaults)
        for section, contents in db_.items():
            db.setdefault(section, {}).update(contents)
        
        pbar.update(1)
    pbar.close()

    return db

def populate_db(cmplx_morph_classes,
                pos_type,
                compatibility_memoize,
                short_cat_maps=None,
                defaults=None):
    db = {}
    db['OUT:###PREFIXES###'] = {}
    db['OUT:###SUFFIXES###'] = {}
    db['OUT:###STEMS###'] = {}
    db['OUT:###TABLE AB###'] = {}
    db['OUT:###TABLE BC###'] = {}
    db['OUT:###TABLE AC###'] = {}

    cmplx_prefix_classes, cmplx_prefix_seq = cmplx_morph_classes['cmplx_prefix_classes']
    cmplx_suffix_classes, cmplx_suffix_seq = cmplx_morph_classes['cmplx_suffix_classes']
    cmplx_stem_classes, cmplx_stem_seq = cmplx_morph_classes['cmplx_stem_classes']

    cat_memoize = {'stem': {}, 'suffix': {}, 'prefix': {}}
    for cmplx_stem_cls, cmplx_stems in cmplx_stem_classes.items():
        # `stem_class` = (stem['COND-S'], stem['COND-T'], stem['COND-F'])
        # All `stem_comb` in `stem_combs` have the same cat 
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
                                            db_section='OUT:###STEMS###')
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
                        update_db(db, update_info, cat_memoize, short_cat_maps, defaults)
                    # If morph class cat has already been computed previously, then cat is still `None`
                    # (because we will not go again in the morph for loop) and we need to retrieve the
                    # computed value. 
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

def update_db(db,
              update_info,
              cat_memoize,
              short_cat_maps=None,
              defaults=None):
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
        short_cat_map = short_cat_maps['stem']
        _generate = _generate_stem
    elif cmplx_morph_type in ['prefix', 'suffix']:
        short_cat_map = short_cat_maps['prefix' if cmplx_morph_type == 'prefix' else 'suffix']
        _generate = partial(_generate_affix, cmplx_morph_type)
    else:
        raise NotImplementedError

    required_feats = _choose_required_feats(update_info['pos_type'])
    # This if statement implements early stopping which entails that if we have already 
    # logged a specific prefix/stem/suffix entry, we do not need to do it again. Entry
    # generation (and more specifically `dediac()`) is costly.
    if cat_memoize[cmplx_morph_type].get(cmplx_morph_cls) is None:
        for cmplx_morph in cmplx_morphs:
            morph_entry, mcat = _generate(cmplx_morph_seq,
                                          required_feats,
                                          cmplx_morph,
                                          cond_s, cond_t, cond_f,
                                          short_cat_map,
                                          defaults_)
            db[db_section].setdefault('\t'.join(morph_entry.values()), 0)
            db[db_section]['\t'.join(morph_entry.values())] += 1
        cat_memoize[cmplx_morph_type][cmplx_morph_cls] = mcat


def _create_cat(cmplx_morph_type, cmplx_morph_class,
                cmplx_morph_cond_s, cmplx_morph_cond_t, cmplx_morph_cond_f,
                short_cat_map=None):
    """This function creates the category for matching using classes and conditions"""
    if short_cat_map:
        cmplx_morph_class = short_cat_map[cmplx_morph_class]
    cmplx_morph_cond_s = '+'.join(
        [cond for cond in cmplx_morph_cond_s.split() if cond != '_'])
    cmplx_morph_cond_s = cmplx_morph_cond_s if cmplx_morph_cond_s else '-'
    cmplx_morph_cond_t = '+'.join(
        [cond for cond in cmplx_morph_cond_t.split() if cond != '_'])
    cmplx_morph_cond_t = cmplx_morph_cond_t if cmplx_morph_cond_t else '-'
    cmplx_morph_cond_f = '+'.join(
        [cond for cond in cmplx_morph_cond_f.split() if cond != '_'])
    cmplx_morph_cond_f = cmplx_morph_cond_f if cmplx_morph_cond_f else '-'
    cat = f"{cmplx_morph_type}:{cmplx_morph_class}_[CS:{cmplx_morph_cond_s}]_[CT:{cmplx_morph_cond_t}]_[CF:{cmplx_morph_cond_f}]"
    return cat

def _convert_bw_tag(bw_tag):
    """Convert BW tag from BW2UTF8"""
    if bw_tag == '':
        return bw_tag
    bw_elements = bw_tag.split('+')
    utf8_bw_tag = []
    for element in bw_elements:
        parts = element.split('/')
        if 'null' in parts[0]:
            bw_lex = parts[0]
        else:
            bw_lex = bw2ar(parts[0])
        bw_pos = parts[1]
        utf8_bw_tag.append('/'.join([bw_lex, bw_pos]))
    return '+'.join(utf8_bw_tag)

def _generate_affix(affix_type,
                    cmplx_morph_seq,
                    required_feats,
                    affix,
                    affix_cond_s, affix_cond_t, affix_cond_f,
                    short_cat_map=None,
                    defaults=None):
    affix_match, affix_diac, affix_gloss, affix_feat, affix_bw = _read_affix(affix)
    affix_type = "P" if affix_type == 'prefix' else 'S'
    acat = _create_cat(
        affix_type, cmplx_morph_seq, affix_cond_s, affix_cond_t, affix_cond_f, short_cat_map)
    ar_pbw = _convert_bw_tag(affix_bw)
    affix_feat = ' '.join([f"{feat}:{val}" for feat, val in affix_feat.items()])
    affix = {
        'match': bw2ar(affix_match),
        'cat': acat,
        'feats': f"diac:{bw2ar(affix_diac)} bw:{ar_pbw} gloss:{affix_gloss.strip()} {affix_feat}"
    }
    return affix, acat


def _generate_stem(cmplx_morph_seq,
                   required_feats,
                   stem,
                   stem_cond_s, stem_cond_t, stem_cond_f,
                   short_cat_map=None,
                   defaults=None):
    stem_match, stem_diac, stem_lex, stem_gloss, stem_feat, stem_bw = _read_stem(stem)
    xcat = _create_cat(
        "X", cmplx_morph_seq, stem_cond_s, stem_cond_t, stem_cond_f, short_cat_map)
    ar_xbw = _convert_bw_tag(stem_bw)
    if defaults:
        stem_feat = [f"{feat}:{stem_feat[feat]}"
                        if feat in stem_feat and stem_feat[feat] != '_'
                        else f"{feat}:{defaults[stem_feat['pos']][feat]}" 
                        for feat in required_feats + _clitic_feats]
    else:
        stem_feat = [f"{feat}:{val}" for feat, val in stem_feat.items()]
    stem_feat = ' '.join(stem_feat)
    #TODO: strip lex before transliteration (else underscore will be handled wrong)
    stem = {
        'match': bw2ar(stem_match),
        'cat': xcat,
        'feats': (f"diac:{bw2ar(stem_diac)} bw:{ar_xbw} lex:{bw2ar(stem_lex)} "
                  f"gloss:{stem_gloss.strip()} {stem_feat} ")
    }
    return stem, xcat

def _read_affix(affix):
    affix_bw = '+'.join([m['BW'] for m in affix if m['BW'] != '_'])
    affix_diac = ''.join([m['FORM'] for m in affix if m['FORM'] != '_'])
    affix_gloss = '+'.join([m['GLOSS'] for m in affix if m['GLOSS'] != '_'])
    affix_gloss = affix_gloss if affix_gloss else '_'
    affix_feat = {feat.split(':')[0]: feat.split(':')[1]
             for m in affix for feat in m['FEAT'].split()}
    affix_match = normalize_alef_bw(normalize_alef_maksura_bw(
        normalize_teh_marbuta_bw(dediac_bw(affix_diac))))
    return affix_match, affix_diac, affix_gloss, affix_feat, affix_bw

def _read_stem(stem):
    stem_bw = '+'.join([s['BW'] for s in stem if s['BW'] != '_'])
    stem_diac = ''.join([s['FORM'] for s in stem if s['FORM'] != '_'])
    stem_gloss = '+'.join([s['GLOSS'] for s in stem if 'LEMMA' in s])
    stem_gloss = stem_gloss if stem_gloss else '_'
    stem_lex = '+'.join([s['LEMMA'].split(':')[1] for s in stem if 'LEMMA' in s])
    stem_feat = {feat.split(':')[0]: feat.split(':')[1]
                for s in stem for feat in s['FEAT'].split()}
    stem_match = normalize_alef_bw(normalize_alef_maksura_bw(
        normalize_teh_marbuta_bw(dediac_bw(stem_diac))))
    return stem_match, stem_diac, stem_lex, stem_gloss, stem_feat, stem_bw

def print_almor_db(output_filename, db):
    """Create output file in ALMOR DB format"""
    with open(output_filename, "w") as f:
        for x in db['OUT:###HEADER###']:
            print(x, file=f)

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
            
        print("###TABLE AB###", file=f)
        for x in db['OUT:###TABLE AB###']:
            print(x, file=f)
            
        print("###TABLE BC###", file=f)
        for x in db['OUT:###TABLE BC###']:
            print(x, file=f)
            
        print("###TABLE AC###", file=f)
        for x in db['OUT:###TABLE AC###']:
            print(x, file=f)


def _get_short_cat_name_maps(ORDER):
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


def gen_cmplx_morph_combs(cmplx_morph_seq,
                          MORPH, LEXICON,
                          cond2class=None,
                          morph_pattern=None,
                          cmplx_morph_memoize=None,
                          pruning_cond_s_f=True,
                          pruning_same_class_incompat=True):
    """This function exands the specification of morph classes
    into all possible combinations of specific morphemes.
    Input is space separated class name: e.g., '[QUES] [CONJ] [PREP]'
    Ouput is a list of lists of dictionaries; where each dictionary
    specifies a morpheme; the list order of the dictionaries indicate 
    order of morphemes.
    In other words, it generates a Cartesian product of their combinations
    as specified by the allowed morph classes"""
    if cmplx_morph_memoize != None:
        pattern_match = morph_pattern.search(cmplx_morph_seq)
        if pattern_match:
            seq_key = pattern_match.group(1)
        else:
            seq_key = cmplx_morph_seq
        
        if 'STEM' in cmplx_morph_seq and cmplx_morph_memoize.get(seq_key) != None:
            return cmplx_morph_memoize[seq_key]

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
                # If or-ed (||) COND-T cond_s did not exist, this would be as simple as checking
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
    
    if 'STEM' in cmplx_morph_seq and cmplx_morph_memoize != None:
        cmplx_morph_memoize[seq_key] = cmplx_morph_categorized
    
    return cmplx_morph_categorized

#Remember to eliminate all non match affixes/stems
def check_compatibility (cond_s, cond_t, cond_f, compatibility_memoize):
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
    # Things that need to be true
    for t in ct:
        #OR condition check
        validor = False
        for ort in t.split('||'):
            validor = validor or ort in cs
        #AND Check
        valid = valid and validor
        if not valid:  # abort if we hit an invalid condition
            compatibility_memoize[key] = valid
            return valid
    # Things that need to be false
    for f in cf:
        for orf in f.split('||'):
            valid = valid and orf not in cs
        if not valid:  # abort if we hit an invalid condition
            compatibility_memoize[key] = valid
            return valid

    compatibility_memoize[key] = valid
    return valid                     


def get_clean_set(cond_col):
    morph_cond_t = cond_col.tolist()
    morph_cond_t = [
        y for y in morph_cond_t if y not in ['', '_', 'else', None]]
    morph_cond_t = set(morph_cond_t)  # make it a set

    return morph_cond_t


def get_morph_cond_f(morph_cond_t, cond_t_current, MORPH):
    # Go through each allomorph
    for idx, entry in morph_cond_t.iteritems():
        # If we have no true condition for the allomorph (aka can take anything)
        if entry is None:
            continue
        elif entry != 'else':
            #TODO: create condFalse for the morpheme by c_T - entry
            cond_f_almrph, _ = _get_cond_false(
                cond_t_current, [entry])
        elif entry == 'else':
            # TODO: condFalse = c_T
            cond_f_almrph = cond_t_current
            # Finally, populate the 'COND-F' cell with the false conditions
        MORPH.loc[idx, 'COND-F'] = MORPH.loc[idx, 'COND-F'] + \
            ' ' + ' '.join(cond_f_almrph)
        MORPH.loc[idx, 'COND-F'] = MORPH.loc[idx, 'COND-F'].replace('_', '')
    
    return MORPH

def _get_cond_false(cond_t_all, cond_t_almrph):
    # The false conditions for the current allomorph is whatever true condition
    #   in the morpheme set that does not belong to the set of true conditions
    #   to the current allomorph.
    cond_f_almrph = cond_t_all - set(cond_t_almrph)
    # If there is a negated condition in the set of true conditions for the
    #   current allomorph, remove the negation and add it to the false
    #   conditions list for the current alomorph.
    # negated_cond = [x for x in almrph_condTrue if x.startswith('!')]
    # if negated_cond:
    #   for cond in negated_cond:
    #     almrph_condFalse.add(cond[1:])
    #     # remove from the true conditions
    #     almrph_condTrue.remove(cond)

    # If we end up with no false condition, set it to '_'
    if not cond_f_almrph:
        cond_f_almrph = ['_']
    # If we end up with no true condition, set it to '_'
    if not cond_t_almrph:
        cond_t_almrph = ['_']

    return cond_f_almrph, cond_t_almrph


def _process_postregex(postregex):
    for i, row in postregex.iterrows():
        match_ = []
        for match in re.split(r'(\\.|[\|}{\*\$_])', row['MATCH']):
            match = match if re.match(r'(\\.)', match) else bw2ar(match)
            match_.append(match)

        postregex.at[i, 'MATCH'] = ''.join(match_)
        postregex.at[i, 'REPLACE'] = ''.join(re.sub(r'\$', r'\\', row['REPLACE']))
    
    return postregex


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
    else:
        raise NotImplementedError
    return required_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", required=True,
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-output_dir", default='db_iterations',
                        type=str, help="Path of the directory to output the DBs to.")
    parser.add_argument("-run_profiling", default=False,
                        action='store_true', help="Run execution time profiling for the make_db().")
    args = parser.parse_args()
    
    if args.run_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    make_db(args.config_file, args.config_name, args.output_dir)
    
    if args.run_profiling:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
