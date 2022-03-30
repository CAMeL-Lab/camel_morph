import pandas as pd
from numpy import nan
import re
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-pv", default='data/MSA-LEX-PV.csv',
                    type=str, help="Path of the PV verbs.")
parser.add_argument("-iv", default='data/MSA-LEX-IV.csv',
                    type=str, help="Path of the IV verbs.")
parser.add_argument("-cv", default='data/MSA-LEX-CV.csv',
                    type=str, help="Path of the CV verbs.")
parser.add_argument("-strictness", default='low', choices=['low', 'high'],
                    type=str, help="Strictness level of the check.")
parser.add_argument("-output_path", default='',
                    type=str, help="Path of the CV verbs.")
parser.add_argument("-camel_tools", default='',
                type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args()

if args.camel_tools:
    sys.path.insert(0, args.camel_tools)

import gspread
from camel_tools.morphology.utils import strip_lex
from camel_tools.utils.charmap import CharMapper

bw2ar = CharMapper.builtin_mapper('bw2ar')

def add_check_mark_online(verbs, error_cases):
    return
    status = []
    filtered = verbs[~verbs['LEMMA'].isin(error_cases)]
    sa = gspread.service_account(
        "/Users/chriscay/.config/gspread/service_account.json")
    sh = sa.open('msa-verb-lex')
    worksheet = sh.worksheet(title='MSA-LEX-PV')
    worksheet.update('R2:R10634', [['CHECK'] if i in filtered.index else [
                     'OK'] for i in range(len(verbs['LEMMA']))])

def two_stem_lemma_well_formedness(lemma, rows, iv=None):
        stems_cond_t_sorted = tuple(sorted([row['cond-t'] for row in rows]))
        lemma_uniq_glosses = set([row['gloss'] for row in rows])
        lemma_uniq_conditions = set([tuple([row[k] for k in ['cond-t', 'cond-s']]) for row in rows])
        lemma_uniq_cond_s = set([row['cond-s'] for row in rows])
        lemma_uniq_feats = set([row['feats'] for row in rows])
        lemma_uniq_forms = set([row['form'] for row in rows])
        lemma_trans = tuple(sorted([re.search(r'trans|intrans', row['cond-s']).group() for row in rows]))
        stems_cond_s_no_trans = set([re.sub(r' ?(trans|intrans) ?', '', row['cond-s']) for row in rows])
        if (len(lemma_uniq_glosses) == 1 and stems_cond_t_sorted == ('c-suff', 'v-suff') and len(lemma_uniq_forms) == len(rows) and
            len(lemma_uniq_cond_s) == 1):
            return True, 'legit'
        elif (len(lemma_uniq_glosses) == 1 and stems_cond_t_sorted == ('c-suff', 'v-suff') and len(lemma_uniq_forms) == len(rows) and
              len(lemma_uniq_cond_s) == 2):
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

def well_formedness_check(verbs, lemma2info, iv):
    elementary_feats = ['lemma', 'form', 'gloss', 'feats', 'cond-s', 'cond-t']
    # Duplicate entries are not allowed
    error_cases = {}
    for lemma, info in lemma2info.items():
        if info[0] > 1:
            uniq_entries = set([tuple([row[k] for k in elementary_feats]) for row in info[2]])
            if len(uniq_entries) != info[0]:
                error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, error_cases)
    
    # No entry contains a double diactric lemma (inherited from BW)
    error_cases = {}
    for lemma, info in lemma2info.items():
        if re.search(r'-[uia]{2,}', lemma):
            error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, error_cases)

    # COND-S must contain transitivity information
    error_cases = {}
    for lemma, info in lemma2info.items():
        if all([bool(re.search(r'trans|intrans', row['cond-s']))
                        for row in info[2]]):
            continue
        error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, error_cases)

    # If exactly two stems are associated to a lemma, then they must be 
    # one c-suff, one v-suff stem, and have the same gloss (e.g., mad~, madad).
    error_cases = {}
    for lemma, info in lemma2info.items():
        if info[0] == 2:
            if two_stem_lemma_well_formedness(lemma, info[2])[0]:
                continue
            error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, error_cases)
    
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
    assert len(error_cases) == 0, add_check_mark_online(verbs, error_cases)

    # All COND-S associated to a lemma must belong to the stem condition categories
    # and they must be unique
    morph = pd.read_csv('data/MSA-MORPH-Verbs-v4-Red.csv')
    class2cond = morph[morph.DEFINE == 'CONDITIONS']
    class2cond = {cond_class["CLASS"]:
                            [cond for cond in cond_class["FUNC"].split() if cond]
                        for _, cond_class in class2cond.iterrows()}
    cond2class = {
        cond: (cond_class, 
               int(''.join(['1' if i == index else '0' for index in range (len(cond_s))]), 2)
        )
        for cond_class, cond_s in class2cond.items()
            for i, cond in enumerate(cond_s)}

    error_cases = {}
    stem_cond_cats = ['[STEM-X]', '[X-STEM]', '[X-ROOT]', '[X-TRANS]', '[PREF-X]', '[LEMMA]']
    for lemma, info in lemma2info.items():
        if (all([cond2class[cond][0] in stem_cond_cats
                        for row in info[2] for cond in row['cond-s'].split()]) and
           all([len(set(row['cond-s'].split())) == len(row['cond-s'].split())
                        for row in info[2]])):
            continue
        error_cases[lemma] = info
    assert len(error_cases) == 0, add_check_mark_online(verbs, error_cases)

    #TODO: make sure (lemma, gloss) are unique
    #TODO: add root/pattern/form concordance check
    #TODO: check that there are no letters other than defective letters in patterns
    #TODO: add #n/#t/gem check
    #TODO: (maybe later) add check that all forms belonging to same defective pattern must have same shape
    #TODO: add diacritic rules

def get_lemma2info(verbs):
    lemma2info = {}
    for i, row in verbs.iterrows():
        lemma = row['LEMMA']
        count_suff = lemma2info.setdefault(lemma, [0, [], []])
        count_suff[0] += 1
        count_suff[1].append(True if row['COND-T'] else False)
        count_suff[2].append(
            {'index': i, 'lemma': row['LEMMA'], 'form': row['FORM'], 'gloss': row['GLOSS'],
            'cond-s': row['COND-S'], 'cond-t': row['COND-T'], 'feats': row['FEAT']})
    return lemma2info

def well_formedness_all_aspects(pv, iv, cv, strictness):
    assert len(iv.index) == len(cv.index)

    merging_info, merge_indexes = {}, {}
    extlemma2infos = {}
    for asp, verbs_asp in [('pv', pv), ('iv', iv), ('cv', cv)]:
        verbs_asp = verbs_asp.replace(nan, '', regex=True)
        lemma2info = get_lemma2info(verbs_asp)
        well_formedness_check(verbs_asp, lemma2info, iv)
        merge_indexes[asp], merging_info[asp] = merge_same_feats_diff_gloss_lemmas(lemma2info, iv)
        extlemma2infos_asp = extlemma2infos.setdefault(asp, {})
        for lemma, info in lemma2info.items():
            for row in info[2]:
                transitivity = re.search(r'trans|intrans', row['cond-s']).group()
                if 'Frozen' in row['cond-s']:
                    continue
                if strictness == 'high':
                    extlemma = (row['lemma'], transitivity, row['gloss'])
                elif strictness == 'low':
                    extlemma = (row['lemma'], row['gloss'])
                extlemma2infos_asp.setdefault(extlemma, []).append(info[2])
    
    merging_cases_pv = set([lemma[0] for lemma in merging_info['pv']])
    merging_cases_iv = set([lemma[0] for lemma in merging_info['iv']])
    merging_cases_cv = set([lemma[0] for lemma in merging_info['cv']])
    assert merging_cases_pv == merging_cases_iv == merging_cases_cv

    missing_iv_pv = [extlemma2infos['iv'][gloss] for gloss in set(extlemma2infos['iv']) - set(extlemma2infos['pv'])]
    assert len(missing_iv_pv) == 0
    missing_iv_cv = [extlemma2infos['iv'][gloss] for gloss in set(extlemma2infos['iv']) - set(extlemma2infos['cv'])]
    assert len(missing_iv_cv) == 0
    missing_pv_iv = [extlemma2infos['pv'][gloss] for gloss in set(extlemma2infos['pv']) - set(extlemma2infos['iv'])]
    assert len(missing_pv_iv) == 0
    missing_pv_cv = [extlemma2infos['pv'][gloss] for gloss in set(extlemma2infos['pv']) - set(extlemma2infos['cv'])]
    assert len(missing_pv_cv) == 0
    missing_cv_iv = [extlemma2infos['cv'][gloss] for gloss in set(extlemma2infos['cv']) - set(extlemma2infos['iv'])]
    assert len(missing_cv_iv) == 0
    missing_cv_pv = [extlemma2infos['cv'][gloss] for gloss in set(extlemma2infos['cv']) - set(extlemma2infos['pv'])]
    assert len(missing_cv_pv) == 0

    return merge_indexes

def merge_same_feats_diff_gloss_lemmas(lemma2info, iv):
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
                    assert all([row['cond-t'] in ['c-suff', 'v-suff'] for rows in gloss2rows.values() for row in rows])
                    rows_to_merge = {'c-suff': [], 'v-suff': []}
                    for rows in gloss2rows.values():
                        for row in rows:
                            rows_to_merge[row['cond-t']].append(row)

                    for rows in rows_to_merge.values():
                        row_indexes_to_merge.append(tuple(
                            [int(row['index']) for row in sorted(rows, key=lambda row: row['gloss'])]))
                        lemmas_info.append((lemma, rows[0]['cond-t'], rows[0]['cond-s']))
            else:
                if two_stem_lemma_well_formedness(lemma, info[2], iv)[1] == 'merge_case':
                    raise NotImplementedError
    
    return row_indexes_to_merge, lemmas_info

def msa_verbs_lexicon_well_formedness(verbs, strictness):

    merge_indexes = well_formedness_all_aspects(verbs['pv'], verbs['iv'], verbs['cv'], strictness)

    for asp, merging_indexes_asp in merge_indexes.items():
        for merge_indexes_tuple in merging_indexes_asp:
            index_to_keep = merge_indexes_tuple[0]
            for merge_index in merge_indexes_tuple[1:]:
                verbs[asp].at[index_to_keep, 'GLOSS'] += f";{verbs[asp].at[merge_index, 'GLOSS']}"
                verbs[asp] = verbs[asp].drop([merge_index])

    merge_indexes = well_formedness_all_aspects(verbs['pv'], verbs['iv'], verbs['cv'], strictness)

    for verbs_asp in verbs.values():
        verbs_asp['#LEMMA_AR'] = verbs_asp['LEMMA'].apply(bw2ar)
    verbs = pd.concat([verbs_asp for verbs_asp in verbs.values()])
    return verbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pv", default='data/MSA-LEX-PV.csv',
                        type=str, help="Path of the PV verbs.")
    parser.add_argument("-iv", default='data/MSA-LEX-IV.csv',
                        type=str, help="Path of the IV verbs.")
    parser.add_argument("-cv", default='data/MSA-LEX-CV.csv',
                        type=str, help="Path of the CV verbs.")
    parser.add_argument("-strictness", default='low', choices=['low', 'high'],
                        type=str, help="Strictness level of the check.")
    parser.add_argument("-output_path", default='',
                        type=str, help="Path of the CV verbs.")
    args = parser.parse_args()
    
    pv = pd.read_csv(args.pv)
    iv = pd.read_csv(args.iv)
    cv = pd.read_csv(args.cv)
    verbs = {'pv': pv, 'iv': iv, 'cv': cv}

    verbs_ = msa_verbs_lexicon_well_formedness(verbs=verbs, strictness=args.strictness)

    if args.output_path:
        verbs_.to_csv(args.output_path)
