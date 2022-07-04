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


import pandas as pd
from numpy import nan
import re
import argparse
import sys
import json
import os

import gspread

try:
    from camel_morph.utils.utils import assign_pattern, add_check_mark_online
    from camel_morph.debugging.download_sheets import download_sheets
except:
    from camel_morph.camel_morph.utils.utils import assign_pattern, add_check_mark_online
    from camel_morph.camel_morph.debugging.download_sheets import download_sheets

two_stem_well_formedness_options_dialect = {
    'msa': [('c-suff', 'v-suff')],
    'egy': [('c-suff', 'n-suff||v-suff')],
    'glf': [('c-suff', 'n-suff||v-suff'), ('c-suff||n-suff', 'v-suff'), ('n-suff', 'v-suff')]
}
two_stem_well_formedness_options = None


def two_stem_lemma_well_formedness(lemma, rows, iv=None):
    global two_stem_well_formedness_options
    stems_cond_t_sorted = tuple(sorted([row['cond-t'] for row in rows]))
    lemma_uniq_glosses = set([row['gloss'] for row in rows])
    lemma_uniq_conditions = set([tuple([row[k] for k in ['cond-t', 'cond-s']]) for row in rows])
    lemma_uniq_cond_s = set([row['cond-s'] for row in rows])
    lemma_uniq_feats = set([row['feats'] for row in rows])
    lemma_uniq_forms = set([row['form'] for row in rows])
    lemma_trans = tuple(sorted([re.search(r'trans|intrans', row['cond-s']).group() for row in rows]))
    stems_cond_s_no_trans = set([re.sub(r' ?(trans|intrans) ?', '', row['cond-s']) for row in rows])
    if (len(lemma_uniq_glosses) == 1 and stems_cond_t_sorted in two_stem_well_formedness_options and
        len(lemma_uniq_forms) == len(rows) and len(lemma_uniq_cond_s) == 1):
        return True, 'legit'
    elif (len(lemma_uniq_glosses) == 1 and stems_cond_t_sorted in two_stem_well_formedness_options and
            len(lemma_uniq_forms) == len(rows) and len(lemma_uniq_cond_s) == 2):
        if len(set(lemma_trans)) == 1:
            return True, 'legit'
        else:
            return False, 'legit'
    elif (len(lemma_uniq_glosses) != 1 and len(lemma_uniq_conditions) == 1 and len(lemma_uniq_feats) == 1 and 
            len(lemma_uniq_forms) == 1):
        if len(lemma_uniq_feats) == 1 and 'asp:i' in rows[0]['feats'] and len(strip_lex(lemma)) == 4 and lemma[0] == '|':
            message = 'legit'
        else:
            if iv is not None:
                if ('asp:p' in rows[0]['feats'] or 'asp:c' in rows[0]['feats']) and len(strip_lex(lemma)) == 4 and lemma[0] == '|':
                    lemma_iv = iv[iv['LEMMA'] == lemma]
                    lemma2info = get_lemma2info(lemma_iv)
                    message = two_stem_lemma_well_formedness(lemma, lemma2info[lemma][2])[1]
                    if message == 'error':
                        raise NotImplementedError
                else:
                    message = 'merge_case'
            else:
                message = 'merge_case'
        return True, message
    elif (len(lemma_uniq_glosses) != 1 and len(lemma_uniq_conditions) == 2 and len(lemma_uniq_feats) == 1 and
            len(stems_cond_s_no_trans) == 1):
        if lemma_trans == ('intrans', 'trans'):
            return True, 'legit'
    # Verbs like |laf
    elif len(lemma_uniq_feats) == 1 and 'asp:i' in rows[0]['feats'] and len(strip_lex(lemma)) == 4 and lemma[0] == '|':
        if len(lemma_uniq_forms) == 2 and len(lemma_uniq_glosses) == 2:
            return True, 'legit'
    return False, 'error'

def well_formedness_check(verbs, lemma2info, spreadsheet=None, sheet=None):
    elementary_feats = ['lemma', 'form', 'root', 'gloss', 'feats', 'cond-s', 'cond-t']
    # Duplicate entries are not allowed
    error_cases = {}
    for lemma, info in lemma2info.items():
        if info[0] > 1:
            uniq_entries = set([tuple([row[k] for k in elementary_feats]) for row in info[2]])
            if len(uniq_entries) != info[0]:
                error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')
    
    # No entry contains a double diactric lemma (inherited from BW)
    error_cases = {}
    for lemma, info in lemma2info.items():
        if re.search(r'-[uia]{2,}', lemma):
            error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')

    # COND-S must contain transitivity information
    error_cases = {}
    for lemma, info in lemma2info.items():
        if all([bool(re.search(r'trans|intrans', row['cond-s']))
                        for row in info[2]]):
            continue
        error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')

    # If exactly one stem is associated to a lemma, then its COND-T must be empty.
    error_cases = {}
    for lemma, info in lemma2info.items():
        if info[0] == 1:
            if not info[2][0]['cond-t']:
                continue
            error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')

    # If exactly two stems are associated to a lemma, then they must be 
    # one c-suff, one v-suff stem, and have the same gloss (e.g., mad~, madad).
    error_cases = {}
    for lemma, info in lemma2info.items():
        if info[0] == 2:
            if two_stem_lemma_well_formedness(lemma, info[2])[0]:
                continue
            error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')
    
    # If more than two stems are associated to a lemma, then they must be 
    # two c-suff and one v-suff PV stem as this does not happend in IV and CV
    # (e.g., laj~, lajij, lajaj).
    error_cases = {}
    for lemma, info in lemma2info.items():
        if info[0] > 2:
            gloss2rows = {}
            for row in info[2]:
                gloss2rows.setdefault(row['gloss'], []).append(row)
            for rows in gloss2rows.values():
                if len(rows) % 2 == 0:
                    if two_stem_lemma_well_formedness(lemma, rows)[0]:
                        continue
                else:
                    lemma_uniq_feats = set([row['feats'] for row in rows])
                    if len(lemma_uniq_feats) == 1 and 'asp:p' in rows[0]['feats']:
                        stems_cond_t_sorted = tuple(sorted([row['cond-t'] for row in rows]))
                        if stems_cond_t_sorted == ('c-suff', 'c-suff', 'v-suff'):
                            continue
                error_cases[lemma] = info
                break
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')

    # All COND-S associated to a lemma must belong to the stem condition categories
    # and they must be unique
    error_cases = {}
    stem_cond_cats = ['[STEM-X]', '[X-STEM]', '[X-ROOT]', '[X-TRANS]', '[PREF-X]', '[LEMMA]']
    for lemma, info in lemma2info.items():
        if (all([cond2class[cond][0] in stem_cond_cats
                        for row in info[2] for cond in row['cond-s'].split()]) and
           all([len(set(row['cond-s'].split())) == len(row['cond-s'].split())
                        for row in info[2]])):
            continue
        error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, spreadsheet, sheet, error_cases, mode='well-formedness')

    #TODO: add root/pattern/form concordance check
    #TODO: check that there are no letters other than defective letters in patterns
    #TODO: add #n/#t/gem check
    #TODO: add diacritic rules
    #TODO: all 1a2a3 forms should have a dash

def get_lemma2info(verbs):
    lemma2info = {}
    for i, row in verbs.iterrows():
        lemma = row['LEMMA']
        count_suff = lemma2info.setdefault(lemma, [0, [], []])
        count_suff[0] += 1
        count_suff[1].append(True if row['COND-T'] else False)
        count_suff[2].append(
            {'index': i, 'lemma': row['LEMMA'], 'form': row['FORM'], 'gloss': row['GLOSS'],
            'cond-s': row['COND-S'], 'cond-t': row['COND-T'], 'feats': row['FEAT'],
            'root': row['ROOT']})
    return lemma2info


def form_pattern_well_formedness(verbs):
    verbs['COND-S'] = verbs.apply(
        lambda row: re.sub(r'(hamzated|hollow|defective|ditrans|trans|intrans)', '', row['COND-S']), axis=1)
    verbs['COND-S'] = verbs.apply(
        lambda row: re.sub(r' +', ' ', row['COND-S']), axis=1)
    verbs['COND-S'] = verbs.apply(
        lambda row: re.sub(r'^ $', '', row['COND-S']), axis=1)
    class2pattern = {}
    for i, row in verbs.iterrows():
        result = assign_pattern(strip_lex(row['LEMMA']), root=row['ROOT'].split('.'))
        pattern = result['pattern_conc']
        Eayn_diac = re.search(r'[^-]+(?:-(.))?', row['LEMMA']).group(1)
        info = {'index': i, 'pattern': row['PATTERN'], 'cond_s': row['COND-S'],
                'cond_t': row['COND-T'], 'lemma_pattern': pattern,
                'lemma': row['LEMMA'], 'root': row['ROOT']}
        class2pattern.setdefault(
            (pattern, row['COND-S'], row['COND-T'], Eayn_diac), []).append(info)

    errors = {}
    for stem_class, lemmas in class2pattern.items():
        if len(set([info['pattern'] for info in lemmas])) != 1:
            pattern2lemmas = {}
            for info in lemmas:
                pattern2lemmas.setdefault(info['pattern'], []).append(info)
            errors[stem_class] = pattern2lemmas
    
    return


def well_formedness_all_aspects(verbs, strictness, spreadsheet=None, sheets=None):
    iv, cv = verbs.get('iv'), verbs.get('cv')
    if iv is not None and cv is not None:
        assert len(iv.index) == len(cv.index)

    merging_info, merge_indexes = {}, {}
    extlemma2infos = {}
    for asp, verbs_asp in verbs.items():
        lemma2info = get_lemma2info(verbs_asp)
        well_formedness_check(verbs_asp, lemma2info, spreadsheet, sheets[asp])
        merge_indexes[asp], merging_info[asp] = merge_same_feats_diff_gloss_lemmas(lemma2info, iv)
        extlemma2infos_asp = extlemma2infos.setdefault(asp, {})
        for lemma, info in lemma2info.items():
            for row in info[2]:
                transitivity = re.search(r'ditrans|trans|intrans', row['cond-s']).group()
                if 'Frozen' in row['cond-s']:
                    continue
                if strictness == 'high':
                    extlemma = (row['lemma'], transitivity, row['gloss'])
                elif strictness == 'low':
                    extlemma = (row['lemma'], row['gloss'])
                extlemma2infos_asp.setdefault(extlemma, []).append(info[2])
    
    if merging_info.get('pv') is not None:
        merging_cases_pv = set([lemma[0] for lemma in merging_info['pv']])
    if merging_info.get('iv') is not None:
        merging_cases_iv = set([lemma[0] for lemma in merging_info['iv']])
    if merging_info.get('cv') is not None:
        merging_cases_cv = set([lemma[0] for lemma in merging_info['cv']])
    
    if merging_info.get('pv') is not None and merging_info.get('iv') is not None:
        assert merging_cases_pv == merging_cases_iv
    if merging_info.get('iv') is not None and merging_info.get('cv') is not None:
        assert merging_cases_iv == merging_cases_cv

    if merging_info.get('pv') is not None and merging_info.get('iv') is not None:
        missing_iv_pv = [extlemma2infos['iv'][gloss] for gloss in set(extlemma2infos['iv']) - set(extlemma2infos['pv'])]
        assert len(missing_iv_pv) == 0
        missing_pv_iv = [extlemma2infos['pv'][gloss] for gloss in set(extlemma2infos['pv']) - set(extlemma2infos['iv'])]
        assert len(missing_pv_iv) == 0
    if merging_info.get('iv') is not None and merging_info.get('cv') is not None:
        missing_iv_cv = [extlemma2infos['iv'][gloss] for gloss in set(extlemma2infos['iv']) - set(extlemma2infos['cv'])]
        assert len(missing_iv_cv) == 0
        missing_cv_iv = [extlemma2infos['cv'][gloss] for gloss in set(extlemma2infos['cv']) - set(extlemma2infos['iv'])]
        assert len(missing_cv_iv) == 0
    if merging_info.get('pv') is not None and merging_info.get('cv') is not None:
        missing_pv_cv = [extlemma2infos['pv'][gloss] for gloss in set(extlemma2infos['pv']) - set(extlemma2infos['cv'])]
        assert len(missing_pv_cv) == 0
        missing_cv_pv = [extlemma2infos['cv'][gloss] for gloss in set(extlemma2infos['cv']) - set(extlemma2infos['pv'])]
        assert len(missing_cv_pv) == 0

    return merge_indexes

def merge_same_feats_diff_gloss_lemmas(lemma2info, iv=None):
    row_indexes_to_merge = []
    lemmas_info = []
    for lemma, info in lemma2info.items():
        if info[0] == 2:
            if two_stem_lemma_well_formedness(lemma, info[2], iv)[1] == 'merge_case':
                row_indexes_to_merge.append(tuple(
                    [int(row['index']) for row in sorted(info[2], key=lambda row: row['gloss'])]))
                lemmas_info.append((lemma, info[2][0]['cond-t'], info[2][0]['cond-s']))
        elif info[0] > 2:
            if len(info[2]) % 2 == 0:
                gloss2rows = {}
                for row in info[2]:
                    gloss2rows.setdefault(row['gloss'], []).append(row)
                rows_ = []
                for rows in gloss2rows.values():
                    merged = {k: set([row[k] for row in rows]) for k in rows[0]}
                    merged = {k: next(iter(v)) if len(v) == 1 else '#'.join(sorted([str(vv) for vv in v]))
                                for k, v in merged.items()}
                    rows_ .append(merged)
                if two_stem_lemma_well_formedness(lemma, rows_, iv)[1] == 'merge_case':
                    # assert all([row['cond-t'] in ['c-suff', 'v-suff'] for rows in gloss2rows.values() for row in rows])
                    rows_to_merge = {}
                    for rows in gloss2rows.values():
                        for row in rows:
                            rows_to_merge.setdefault(row['cond-t'], []).append(row)

                    for rows in rows_to_merge.values():
                        row_indexes_to_merge.append(tuple(
                            [int(row['index']) for row in sorted(rows, key=lambda row: row['gloss'])]))
                        lemmas_info.append((lemma, rows[0]['cond-t'], rows[0]['cond-s']))
            else:
                if two_stem_lemma_well_formedness(lemma, info[2], iv)[1] == 'merge_case':
                    raise NotImplementedError
    
    return row_indexes_to_merge, lemmas_info

def msa_verbs_lexicon_well_formedness(verbs, strictness, spreadsheet=None, sheets=None, fix_mode='manual'):
    # for verbs in [verbs['pv'], verbs['iv'], verbs['cv']]:
    #     form_pattern_well_formedness(verbs)
    
    merge_indexes = well_formedness_all_aspects(verbs, strictness, spreadsheet, sheets)
    if fix_mode == 'manual':
        for asp, merging_indexes_asp in merge_indexes.items():
            indexes = [index for indexes in merge_indexes[asp] for index in indexes]
            assert len(merging_indexes_asp) == 0, add_check_mark_online(
                verbs[asp], spreadsheet, sheets[asp], indexes=indexes, mode='well-formedness')
    elif fix_mode == 'automatic':
        for asp, merging_indexes_asp in merge_indexes.items():
            for merge_indexes_tuple in merging_indexes_asp:
                index_to_keep = merge_indexes_tuple[0]
                for merge_index in merge_indexes_tuple[1:]:
                    verbs[asp].at[index_to_keep, 'GLOSS'] += f";{verbs[asp].at[merge_index, 'GLOSS']}"
                    verbs[asp] = verbs[asp].drop([merge_index])
    
        merge_indexes = well_formedness_all_aspects(verbs, strictness, spreadsheet, sheets)

    for verbs_asp in verbs.values():
        verbs_asp['#LEMMA_AR'] = verbs_asp['LEMMA'].apply(bw2ar)
    verbs = pd.concat([verbs_asp for verbs_asp in verbs.values()])
    return verbs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lex", action='append', default=[],
                        type=str, help="Paths of the verb lexicon(s) that need(s) to be checked (against each other) for well-formedness.")
    parser.add_argument("-config_file", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use.")
    parser.add_argument("-config_name", default='default_config',
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-data_dir", default="data",
                        type=str, help="Path of the directory where the sheets are.")
    parser.add_argument("-strictness", default='low', choices=['low', 'high'],
                        type=str, help="Strictness level of the check.")
    parser.add_argument("-output_path", default='',
                        type=str, help="Path to output the new sheet to.")
    parser.add_argument("-camel_tools", default='',
                        type=str, help="Path of the directory containing the camel_tools modules.")
    parser.add_argument("-fix_mode", default='manual', choices=['manual', 'automatic'],
                        type=str, help="Mode to specify how things should be fixed (manually with status filling or automatically).")
    parser.add_argument("-service_account", default='',
                        type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
    args = parser.parse_args()

    if args.camel_tools:
        sys.path.insert(0, args.camel_tools)
    
    from camel_tools.morphology.utils import strip_lex
    from camel_tools.utils.charmap import CharMapper

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    
    if args.config_file and args.config_name:
        with open(args.config_file) as f:
            config = json.load(f)
            config_local = config['local'][args.config_name]
        lexicon_sheets = config_local['lexicon']['sheets']
    else:
        lexicon_sheets = args.lex
    
    two_stem_well_formedness_options = two_stem_well_formedness_options_dialect[config_local['dialect']]

    sa = gspread.service_account(args.service_account)
    sh = sa.open(config_local['lexicon']['spreadsheet'])
    download_sheets(save_dir=args.data_dir,
                    config=config,
                    config_name=args.config_name,
                    service_account=sa)

    verbs, sheets = {}, {}
    for sheet in lexicon_sheets:
        asp = re.search(r'[cip]v', sheet, re.I).group().lower()
        verbs[asp] = pd.read_csv(os.path.join(args.data_dir, f'{sheet}.csv'))
        verbs[asp] = verbs[asp].replace(nan, '', regex=True)
        verbs[asp] = verbs[asp][verbs[asp].DEFINE == 'LEXICON']
        sheets[asp] = sheet

    morph = pd.read_csv(os.path.join(args.data_dir, f"{config_local['specs']['morph']}.csv"))
    class2cond = morph[morph.DEFINE == 'CONDITIONS']
    class2cond = {cond_class["CLASS"]:
                            [cond for cond in cond_class["FUNC"].split() if cond]
                  for _, cond_class in class2cond.iterrows()}
    cond2class = {
        cond: (cond_class,
               int(''.join(
                   ['1' if i == index else '0' for index in range(len(cond_s))]), 2)
               )
        for cond_class, cond_s in class2cond.items()
        for i, cond in enumerate(cond_s)}

    verbs_ = msa_verbs_lexicon_well_formedness(verbs=verbs,
                                               strictness=args.strictness,
                                               spreadsheet=sh,
                                               sheets=sheets,
                                               fix_mode=args.fix_mode)

    if args.output_path:
        verbs_.to_csv(args.output_path)
