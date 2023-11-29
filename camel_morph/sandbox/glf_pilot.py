import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import pickle
from copy import deepcopy
from collections import Counter

import pandas as pd
from numpy import nan

from glf_pilot_utils import FEATS_INFLECT
try:
    from .. import db_maker, db_maker_utils
    from ..debugging.generate_docs_tables import _get_structured_lexicon_classes_nom
    from ..eval import evaluate_camel_morph
    from ..utils import utils
    from ..debugging.download_sheets import download_sheets
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph import db_maker, db_maker_utils
    from camel_morph.debugging.generate_docs_tables import _get_structured_lexicon_classes_nom
    from camel_morph.eval import evaluate_camel_morph
    from camel_morph.utils import utils
    from camel_morph.debugging.download_sheets import download_sheets

from glf_pilot_utils import POS_NOM, DEFAULT_NORMALIZE_MAP

HEADER_SHEET = ['ROOT', 'PATTERN', 'DEFINE', 'CLASS', 'LEMMA', 'FORM',
                'BW', 'GLOSS', 'FREQ', 'COND-S', 'COND-T', 'FEAT',
                'STATUS', 'COMMENTS']

FIELD2SENTENCE_INDEX = {f: i
    for i, f in enumerate(['sentence', 'raw_sentence'])}
FIELD2INFO_INDEX = {f: i
    for i, f in enumerate(['word', 'starline'])}

ESSENTIAL_KEYS = [
    k for k in evaluate_camel_morph.ESSENTIAL_KEYS if k not in ['stt', 'cas']]
gen_keys_exclusions = [
    'source', 'lex', 'diac', 'stem_seg', 'gen', 'num', 'vox', 'rat']
gen_feat_keys = [
    k for k in ESSENTIAL_KEYS if k not in gen_keys_exclusions]

lex_index = ESSENTIAL_KEYS.index('lex')
stem_seg_index = ESSENTIAL_KEYS.index('stem_seg')
prc1_index = ESSENTIAL_KEYS.index('prc1')
prc2_index = ESSENTIAL_KEYS.index('prc2')


def _load_analysis(analysis):
    analysis_ = {}
    for field in analysis:
        field = field.split(':')
        feat, value = field[0], ''.join(field[1:])
        value = re.sub(r'^no$', r'na', value)
        if feat in ['gen', 'num']:
            feat = f'form_{feat}'
        elif feat in ['fgen', 'fnum']:
            feat = feat[1:]
        analysis_[feat] = value
    return analysis_


def _get_id2info(stem_classes):
    id2info = {}
    for (cond_s, cond_t, _), info in stem_classes.items():
        id2info[info['LEMMA']] = {
            'cond_s': cond_s, 'cond_t': cond_t, 'freq': info['FREQ']}
    return id2info


def _filter_and_process_abstract_entries(lexicon, config_glf, config_name_glf):
    SHEETS, _ = db_maker_utils.read_morph_specs(config_glf,
                                                config_name_glf,
                                                lexicon_cond_f=False)
    morph = SHEETS['morph']
    valid_conditions = set(cond_
                           for conds in morph['COND-T'].values.tolist()
                           for cond in conds.split()
                           for cond_ in cond.split('||'))
    
    deleted_conditions = set()
    for i, row in lexicon.iterrows():
        cond_s = row['COND-S']
        cond_s_ = []
        for cond in cond_s.split():
            if cond in valid_conditions:
                cond_s_.append(cond)
            else:
                deleted_conditions.add(cond)
        cond_s_ = ' '.join(cond_s_)
        lexicon.loc[i, 'COND-S'] = cond_s_

    print(f"Conditions not being used and discarded: {' '.join(deleted_conditions)}")
    
    return lexicon


def get_backoff_stems_from_egy(config_glf,
                               config_name_glf,
                               processing_mode='automatic'):
    if processing_mode == 'automatic':
        SHEETS, _ = db_maker_utils.read_morph_specs(config_egy,
                                                    config_name_egy,
                                                    process_morph=False,
                                                    lexicon_cond_f=False)
        lexicon = SHEETS['lexicon']
        lexicon['LEMMA'] = lexicon.apply(
            lambda row: re.sub('lex:', '', row['LEMMA']), axis=1)
    
    elif processing_mode == 'manual':
        lex_path = utils.get_lex_paths(config_glf, config_name_glf)[0]
        lexicon = pd.read_csv(lex_path)
        lexicon = lexicon.replace(nan, '')
    
    else:
        raise NotImplementedError
    
    lexicon_processed = _filter_and_process_abstract_entries(
        lexicon, config_glf, config_name_glf)
    
    cond_s2cond_t2feats2rows = _get_structured_lexicon_classes_nom(lexicon_processed)
    stem_classes = {}
    for cond_s, cond_t2feats2rows in cond_s2cond_t2feats2rows.items():
        for cond_t, feats2rows in cond_t2feats2rows.items():
            for feats, rows in feats2rows.items():
                row_ = rows[0]
                row_['COND-S'], row_['COND-T'] = cond_s, cond_t
                row_['FEAT'] = ' '.join(f"{f}:{feats[i] if feats[i] else '-'}"
                                        for i, f in enumerate(['gen', 'num', 'pos']))
                row_['BW'] = re.search(r'pos:(\S+)', row_['FEAT']).group(1)
                row_['ROOT'] = 'PLACEHOLDER'
                row_['PATTERN'] = 'PLACEHOLDER'
                row_['DEFINE'] = 'BACKOFF'
                row_['CLASS'] = '[STEM]'
                row_['GLOSS'] = 'no'
                if processing_mode == 'automatic':
                    freq = len(rows)
                elif processing_mode == 'manual':
                    freq = int(lexicon_processed[lexicon_processed['LEMMA'] == rows[0]['LEMMA']]['FREQ'])
                row_['FREQ'] = freq
                stem_classes[(cond_s, cond_t, feats)] = row_

    lexicon_processed = pd.DataFrame(stem_classes.values())
    lex_path_name, lex_path_ext = os.path.splitext(lex_path)
    lex_path_modif = lex_path_name + '_modif' + lex_path_ext
    lexicon_processed.to_csv(lex_path_modif)
    
    with open(args.output_backoff_lex, 'w') as f:
            print(*HEADER_SHEET, sep='\t', file=f)
            stem_classes_ = sorted(
                stem_classes.values(), key=lambda row: row['FREQ'], reverse=True)
            for i, row in enumerate(stem_classes_, start=1):
                if processing_mode == 'automatic':
                    row['FORM'], row['LEMMA'] = f'stem{i}', f'lemma{i}'
                print(*[row.get(h, '') for h in HEADER_SHEET], sep='\t', file=f)

    return stem_classes


def reverse_processing(analysis):
    analysis = list(analysis)
    analysis[prc2_index] = re.sub(r'^([wf])', r'\1a', analysis[prc2_index])
    analysis[prc1_index] = re.sub(r'^([bl])', r'\1i', analysis[prc1_index])
    return tuple(analysis)


def _strip_brackets(info):
    if info[0] == '[' and info[-1] == ']':
        info = info[1:-1]
    return info


def add_information_to_rows(outputs, stem_classes):
    id2info = _get_id2info(stem_classes)
    outputs['Lexicon Freq'], outputs['COND-T'], outputs['COND-S'] = '', '', ''
    for i, row in outputs.iterrows():
        info = id2info[_strip_brackets(row['lex'])]
        outputs.loc[i, 'Lexicon Freq'] = info['freq']
        outputs.loc[i, 'COND-T'] = info['cond_t']
        outputs.loc[i, 'COND-S'] = info['cond_s']
    
    return outputs


def get_generated_form_counts_from_gumar(stem_classes, possible_analyses, outputs):
    with open(args.gumar_pkl, 'rb') as f:
        gumar = pickle.load(f)
    
    gumar_ = {}
    for analysis, lemma2diac_stem2count in tqdm(gumar.items()):
        if analysis[0].upper() not in POS_NOM:
            continue
        for lemma, diac_stem2count in lemma2diac_stem2count.items():
            analysis_ = tuple(k for i, k in enumerate(analysis) if i in [0, 5, 6])
            gumar_.setdefault(analysis_, {}).setdefault(
                lemma, Counter()).update(diac_stem2count)

    id2info = _get_id2info(stem_classes)

    form_counts = []
    for example in tqdm(possible_analyses):
        e_gold = reverse_processing(example['gold'])
        lemma = bw2ar(e_gold[lex_index])
        stemid2form2count = {}
        for pred in example['pred']:
            pred = tuple(p if p != 'no' else 'na' for p in pred)
            pred_inflect = tuple(pred[i] for i, f in enumerate(ESSENTIAL_KEYS)
                                 if f in ['pos', 'num', 'gen'])
            stem_class = _strip_brackets(pred[lex_index])
            stem_id = 'stem' + re.search(r'(\d+)$', stem_class).group()
            stem = _strip_brackets(pred[stem_seg_index])
            stemid2form2count.setdefault(stem_id, ['', {}])
            stemid2form2count[stem_id][0] = stem
            feats = {k: e_gold[ESSENTIAL_KEYS.index(k)] for k in gen_feat_keys}
            for form_gen_num in id2info[stem_class]['cond_t'].split('||'):
                form_gen, form_num = list(form_gen_num.lower())
                feats['form_gen'], feats['form_num'] = form_gen, form_num
                analyses, messages = generator.generate(stem_class, feats, debug=True)
                stemid2form2count[stem_id][1].setdefault(form_gen_num, 0)
                if analyses:
                    analysis = analyses[0][0]
                    diac = re.sub(stem_id, stem, analysis['diac'])
                    diac = dediac_ar(diac)
                    count_gumar = 0
                    pred_inflect = (pred_inflect[0], form_num, form_gen)
                    if gumar.get(pred_inflect) is not None:
                        if gumar[pred_inflect].get(lemma) is not None:
                            if gumar[pred_inflect][lemma].get((diac, stem)) is not None:
                                count_gumar = gumar[pred_inflect][lemma][(diac, stem)]
                    stemid2form2count[stem_id][1][form_gen_num] += count_gumar
        form_counts.append(stemid2form2count)

    form_counts_lu = {(str(i + 1), f'[lemma{stem_abstract[4:]}]'): info
                      for i, example in enumerate(form_counts)
                      for stem_abstract, info in example.items()}
    outputs['COND-T_counts'] = outputs.apply(
        lambda row: '||'.join(f'{cond_t}:{count}'
            for cond_t, count in form_counts_lu[(row['filter'].split()[0], row['lex'])][1].items()), axis=1)
                      
    return outputs, form_counts




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_egy", default='camel_morph/configs/config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets` for the EGY lexicon.")
    parser.add_argument("-config_name_egy", default='default_config', nargs='+',
                        type=str, help="Name of the configuration to load from the config file for the EGY lexicon.")
    parser.add_argument("-config_file_glf", default='camel_morph/configs/config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name_glf", default='default_config', nargs='+',
                        type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
    parser.add_argument("-backoff_stems", default='',
                        type=str, help="Specified how to get the backoff stems (either computed automatically from the EGY lexicon, or loaded from a sheet).")
    parser.add_argument("-gumar_dir", default='',
                        type=str, help="Path to the (big) unannotated Gumar dataset.")
    parser.add_argument("-db", default='',
                        type=str, help="Name of the DB file which will be used for the retrieval of lexprob.")
    parser.add_argument("-gumar_pkl", default='',
                        type=str, help="Path used to load and save the Gumar object which is read from the numerous Gumar files.")
    parser.add_argument("-magold_path", default='',
                        type=str, help="Path of the training MAGOLD data that is used for filtering of the backoff analyses.")
    parser.add_argument("-output_backoff_lex", default='',
                        type=str, help="Path to output the GLF backoff lexicon that was obtained by uniquing on the EGY concrete lexicon.")
    parser.add_argument("-possible_analyses_filtered_path", default='',
                        type=str, help="Path to output the possible analyses generated by the backoff analyzer, filtered based on the gold analysis.")
    parser.add_argument("-db_dir", default='',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-build_glf_db", default=False, action='store_true',
                        help="Whether or not to build the GLF DB before starting the process.")
    parser.add_argument("-download_glf_sheets", default=False, action='store_true',
                        help="Whether or not to download sheets for the GLF DB before starting the process.")
    parser.add_argument("-eval_mode", required=True,
                        choices=['recall_glf_magold_raw_no_lex'],
                        type=str, help="What evaluation to perform.")
    parser.add_argument("-n", default=1000,
                        type=int, help="Number of verbs to input to the two compared systems.")
    parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    with open(args.config_file_egy) as f:
        config_egy = json.load(f)
    config_name_egy = args.config_name_egy[0]
    config_local_egy = config_egy['local'][config_name_egy]
    config_global_egy = config_egy['global']

    with open(args.config_file_glf) as f:
        config_glf = json.load(f)
    config_name_glf = args.config_name_glf[0]
    config_local_glf = config_glf['local'][config_name_glf]
    config_global_glf = config_glf['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global_egy['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.morphology.generator import Generator
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.utils.dediac import dediac_ar

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    if args.download_glf_sheets:
        print()
        data_dir = utils.get_data_dir_path(config_glf, config_name_glf)
        download_sheets(save_dir=data_dir,
                        config=config_glf,
                        config_name=config_name_glf,
                        service_account=config_global_glf['service_account'])

    print('\nFetching (or computing) backoff stems... ')
    stem_classes = get_backoff_stems_from_egy(
        config_glf, config_name_glf, processing_mode=args.backoff_stems)
    print('Done.')

    if args.build_glf_db:
        sheet_name = config_glf['local'][config_name_glf]['lexicon']['sheets'][0]
        config_glf_modif = deepcopy(config_glf)
        config_glf_modif_local = config_glf_modif['local'][config_name_glf]
        config_glf_modif_local['lexicon']['sheets'][0] = sheet_name + '_modif'
        db_maker.make_db(config_glf_modif, config_name_glf)
        print()

    db_name = config_local_glf['db']
    db_dir = utils.get_db_dir_path(config_glf, config_name_glf)
    db = MorphologyDB(os.path.join(db_dir, db_name), flags='a')
    db_gen = MorphologyDB(os.path.join(db_dir, db_name), flags='gd')
    generator = Generator(db_gen)
    analyzer = Analyzer(db, backoff='NOAN-ONLY_ALL')
    
    with open(args.magold_path) as f:
        ma_gold = f.read()

    POS = set(map(str.lower, POS_NOM))
    print('Loading MAGOLD data... ')
    data = evaluate_camel_morph._preprocess_magold_data(ma_gold,
                                                        POS,
                                                        _load_analysis,
                                                        field2sentence_index=FIELD2SENTENCE_INDEX,
                                                        field2info_index=FIELD2INFO_INDEX,
                                                        field2ldc_index=None)
    print('Done.\n')
    
    print('Computing all possible analyses...')
    possible_analyses = evaluate_camel_morph.evaluate_recall(data,
                                                             args.n,
                                                             args.eval_mode,
                                                             args.possible_analyses_filtered_path,
                                                             analyzer,
                                                             msa_camel_analyzer=None,
                                                             k_best_analyses=False,
                                                             print_recall=False,
                                                             essential_keys=ESSENTIAL_KEYS)
    print('Done.\n')
    
    print(f'Printing possible analyses to {args.possible_analyses_filtered_path}... ', end='')
    outputs = evaluate_camel_morph.recall_print(errors=[],
                                                correct_cases=possible_analyses,
                                                drop_cases=[],
                                                results_path=args.possible_analyses_filtered_path,
                                                essential_keys=ESSENTIAL_KEYS)
    outputs = add_information_to_rows(outputs, stem_classes)
    print('Done.\n')
    
    print('Getting form counts from generated words accrding to valid backoff analyses... ', end='')
    outputs, form_counts = get_generated_form_counts_from_gumar(
        stem_classes, possible_analyses, outputs)
    print('Done.\n')

    outputs.to_csv(args.possible_analyses_filtered_path, sep='\t', index=False)