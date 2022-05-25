###########################################
#CalimaStar DB Maker
#Habash, Khalifa, 2020-2021
#
# This code takes in a linguistitically informed 
# morphology specification, and outputs a DB
# in the ALMOR format to be used in CamelTools
#
###########################################

import db_maker_utils
from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_bw
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

_required_verb_stem_feats = ['pos', 'asp', 'per', 'gen', 'num', 'vox', 'mod']
_required_nom_stem_feats = ['pos', 'form_gen', 'form_num', 'gen', 'num', 'stt', 'cas']
_clitic_feats = ['enc0', 'enc1', 'enc2', 'prc0', 'prc1', 'prc1.5', 'prc2', 'prc3']

###########################################
#Input File: XLSX containing specific worksheets: About, Header, Order, Morph, Lexicon
#inputfilename="CamelMorphDB-Nov-2021.xlsx"
# Primary function 
# e.g. make_db("CamelMorphDB-Nov-2021.xlsx", "config_pv_msa_order-v4")
###########################################

def make_db(config, config_name, output_dir):
    """Initializes `ABOUT`, `HEADER`, `ORDER`, `MORPH`, and `LEXICON`"""
    c0 = process_time()
    
    print("\nLoading and processing sheets... [1/3]")
    #Config file specifies which sheets to read in the xlsx spreadsheet
    SHEETS, cond2class = db_maker_utils.read_morph_specs(config, config_name)
    
    print("\nValidating combinations... [2/3]")
    db = construct_almor_db(
        SHEETS, config['local'][config_name]['pruning'], cond2class)
    
    print("\nGenerating DB file... [3/3]")
    print_almor_db(
        os.path.join(output_dir, config['local'][config_name]['db']), db)
    
    c1 = process_time()
    print(f"\nTotal time required: {strftime('%M:%S', gmtime(c1 - c0))}")


def construct_almor_db(SHEETS, pruning, cond2class):
    ORDER, MORPH, LEXICON = SHEETS['order'], SHEETS['morph'], SHEETS['lexicon']
    ABOUT, HEADER, POSTREGEX = SHEETS['about'], SHEETS['header'], SHEETS['postregex']
    BACKOFF = SHEETS['backoff']

    short_cat_maps = _get_short_cat_name_maps(ORDER)

    db = {}
    db['OUT:###ABOUT###'] = list(ABOUT['Content'])
    db['OUT:###HEADER###'] = list(HEADER['Content'])
    if POSTREGEX is not None:
        db['OUT:###POSTREGEX###'] = [
            'MATCH\t' + '\t'.join([match for match in POSTREGEX['MATCH'].values.tolist()]),
            'REPLACE\t' + '\t'.join([replace for replace in POSTREGEX['REPLACE'].values.tolist()])
        ]
    
    defaults = _process_defaults(db['OUT:###HEADER###'])
    
    def construct_process(lexicon, stems_section_title):
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
            pbar.set_description(order['SUFFIX-SHORT'])
            cmplx_prefix_classes = gen_cmplx_morph_combs(
                order['PREFIX'], MORPH, lexicon, cond2class, prefix_pattern,
                cmplx_morph_memoize=cmplx_morph_memoize['prefix'],
                pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
            cmplx_suffix_classes = gen_cmplx_morph_combs(
                order['SUFFIX'], MORPH, lexicon, cond2class, suffix_pattern,
                cmplx_morph_memoize=cmplx_morph_memoize['suffix'],
                pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
            cmplx_stem_classes = gen_cmplx_morph_combs(
                order['STEM'], MORPH, lexicon, cond2class, stem_pattern,
                cmplx_morph_memoize=cmplx_morph_memoize['stem'],
                pruning_cond_s_f=pruning, pruning_same_class_incompat=pruning)
            
            cmplx_morph_classes = dict(
                cmplx_prefix_classes=(cmplx_prefix_classes, order['PREFIX']),
                cmplx_suffix_classes=(cmplx_suffix_classes, order['SUFFIX']),
                cmplx_stem_classes=(cmplx_stem_classes, order['STEM']))
            
            db_ = populate_db(
                cmplx_morph_classes, order['CLASS'].lower(), compatibility_memoize,
                short_cat_maps, defaults, stems_section_title)
            for section, contents in db_.items():
                # if 'BACKOFF' in stems_section_title and section != stems_section_title:
                #     assert set(contents) <= set(db[section])
                db.setdefault(section, {}).update(contents)
            
            pbar.update(1)
        pbar.close()
    
    print('Concrete lexicon')
    construct_process(LEXICON, stems_section_title='OUT:###STEMS###')
    if BACKOFF is not None:
        print('Backoff lexicon')
        construct_process(BACKOFF, stems_section_title='OUT:###SMARTBACKOFF###')

    return db

def populate_db(cmplx_morph_classes,
                pos_type,
                compatibility_memoize,
                short_cat_maps=None,
                defaults=None,
                stems_section_title='OUT:###STEMS###'):
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
        [cond for cond in sorted(cmplx_morph_cond_s.split()) if cond != '_'])
    cmplx_morph_cond_s = cmplx_morph_cond_s if cmplx_morph_cond_s else '-'
    cmplx_morph_cond_t = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_t.split()) if cond != '_'])
    cmplx_morph_cond_t = cmplx_morph_cond_t if cmplx_morph_cond_t else '-'
    cmplx_morph_cond_f = '+'.join(
        [cond for cond in sorted(cmplx_morph_cond_f.split()) if cond != '_'])
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
    stem_match, stem_diac, stem_lex, stem_gloss, stem_feat, stem_bw, root, backoff = _read_stem(stem)
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
        'match': db_maker_utils._bw2ar_regex(stem_match, bw2ar) if backoff else bw2ar(stem_match),
        'cat': xcat,
        'feats': (f"diac:{bw2ar(stem_diac)} bw:{ar_xbw} lex:{bw2ar(stem_lex)} "
                  f"root:{bw2ar(root)} gloss:{stem_gloss.strip()} {stem_feat} ")
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
        normalize_teh_marbuta_bw(dediac_bw(re.sub(r'[#@]', '', affix_diac)))))
    return affix_match, affix_diac, affix_gloss, affix_feat, affix_bw

def _read_stem(stem):
    stem_bw = '+'.join([s['BW_SUB'] if s['DEFINE'] == 'BACKOFF' else s['BW']
                        for s in stem if s['BW'] != '_'])
    stem_gloss = '+'.join([s['GLOSS'] for s in stem if 'LEMMA' in s])
    stem_gloss = stem_gloss if stem_gloss else '_'
    stem_lex = '+'.join([s['LEMMA_SUB'].split(':')[1] if s['DEFINE'] == 'BACKOFF' else s['LEMMA'].split(':')[1]
                            for s in stem if 'LEMMA' in s])
    stem_feat = {feat.split(':')[0]: feat.split(':')[1]
                for s in stem for feat in s['FEAT'].split()}
    root = [s['ROOT_SUB'] if s['DEFINE'] == 'BACKOFF' else s['ROOT']
            for s in stem if s.get('ROOT')][0]
    
    backoff = False
    if not any([s['DEFINE'] == 'BACKOFF' for s in stem]):
        stem_diac = ''.join([s['FORM']for s in stem if s['FORM'] != '_'])
        stem_match = normalize_alef_bw(normalize_alef_maksura_bw(
            normalize_teh_marbuta_bw(dediac_bw(re.sub(r'[#@]', '', stem_diac)))))
    else:
        backoff = True
        stem_diac = ''.join([s['FORM_SUB'] if s['DEFINE'] == 'BACKOFF' else s['FORM']
                             for s in stem if s['FORM'] != '_'])
        stem_match = ''.join([re.sub(r'^\^|\$$|[#@]', '', s['MATCH'])
                              if s['DEFINE'] == 'BACKOFF' else \
                                normalize_alef_bw(normalize_alef_maksura_bw(
                                normalize_teh_marbuta_bw(dediac_bw(re.sub(r'[#@]', '', s['FORM'])))))
                              for s in stem if s['FORM'] != '_'])
        stem_match = f'^{stem_match}$'

    return stem_match, stem_diac, stem_lex, stem_gloss, stem_feat, stem_bw, root, backoff

def print_almor_db(output_filename, db):
    """Create output file in ALMOR DB format"""
    with open(output_filename, "w") as f:
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
        backoff = db.get('OUT:###SMARTBACKOFF###')
        if backoff:
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
    parser.add_argument("-config_file", default='config.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
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
