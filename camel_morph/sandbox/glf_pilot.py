import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import pickle
from collections import Counter

import pandas as pd
from numpy import nan

try:
    from .. import db_maker_utils
    from ..debugging.generate_docs_tables import _get_structured_lexicon_classes
    from ..eval import evaluate_camel_morph
    from ..utils import utils
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph import db_maker_utils
    from camel_morph.debugging.generate_docs_tables import _get_structured_lexicon_classes
    from camel_morph.eval import evaluate_camel_morph
    from camel_morph.utils import utils

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


def get_backoff_stems_from_egy(processing_mode='automatic'):
    if processing_mode == 'automatic':
        SHEETS, _ = db_maker_utils.read_morph_specs(config_egy,
                                                    config_name_egy,
                                                    process_morph=False,
                                                    lexicon_cond_f=False)
        lexicon = SHEETS['lexicon']
        lexicon['LEMMA'] = lexicon.apply(
            lambda row: re.sub('lex:', '', row['LEMMA']), axis=1)
        with open(args.output_backoff_lex, 'w') as f:
            print(*HEADER_SHEET, sep='\t', file=f)
            lexicon = sorted(
                lexicon.values(), key=lambda row: row['FREQ'], reverse=True)
            for i, row in enumerate(lexicon, start=1):
                row['FORM'], row['LEMMA'] = f'stem{i}', f'lemma{i}'
                print(*[row.get(h, '') for h in HEADER_SHEET], sep='\t', file=f)
    
    elif processing_mode == 'manual':
        lex_path = utils.get_lex_paths(config_glf, config_name_glf)[0]
        lexicon = pd.read_csv(lex_path)
        lexicon = lexicon.replace(nan, '')
    
    else:
        raise NotImplementedError
    
    cond_s2cond_t2feats2rows = _get_structured_lexicon_classes(lexicon)
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
                row_['FREQ'] = len(rows)
                stem_classes[(cond_s, cond_t, feats)] = row_

    return stem_classes


def filter_analyses(examples):
    source_index = ESSENTIAL_KEYS.index('source')
    processed = []
    for example in examples:
        e_gold = example['gold']
        analyses_pred, index2similarity = [], {}
        for analysis_index, analysis in enumerate(example['pred']):
            analysis_ = []
            for index, f in enumerate(analysis):
                if f == e_gold[index]:
                    analysis_.append(f)
                    index2similarity.setdefault(analysis_index, 0)
                    index2similarity[analysis_index] += (
                        1.01 if analysis[source_index] == 'main' else 1)
                else:
                    analysis_.append(f'[{f}]')
            analyses_pred.append(tuple(analysis_))
        sorted_indexes = sorted(
            index2similarity.items(), key=lambda x: x[1], reverse=True)
        analyses_pred = [analyses_pred[analysis_index]
                            for analysis_index, _ in sorted_indexes]
        analyses_pred = [analysis for analysis in analyses_pred
                            if all(analysis[i] == e_gold[i]
                                   for i, k in enumerate(ESSENTIAL_KEYS)
                                if k not in ['source', 'lex', 'diac', 'stem_seg'])]
        processed.append({'word': example['word']['info']['word'],
                          'gold': e_gold,
                          'pred': analyses_pred})
    
    return processed


def reverse_processing(analysis):
    analysis = list(analysis)
    analysis[prc2_index] = re.sub(r'^([wf])', r'\1a', analysis[prc2_index])
    analysis[prc1_index] = re.sub(r'^([bl])', r'\1i', analysis[prc1_index])
    return tuple(analysis)


def _strip_brackets(info):
    if info[0] == '[' and info[-1] == ']':
        info = info[1:-1]
    return info

def get_generated_form_counts_from_gumar(stem_classes, possible_analyses):
    with open(args.gumar_pkl, 'rb') as f:
        gumar = pickle.load(f)

    id2info = {}
    for (cond_s, cond_t, _), info in stem_classes.items():
        id2info[info['LEMMA']] = {
            'cond_s': cond_s, 'cond_t': cond_t, 'freq': info['FREQ']}

    form_counts = []
    for example in tqdm(possible_analyses):
        e_gold = reverse_processing(example['gold'])
        stemid2form2count = {}
        for pred in example['pred']:
            pred = reverse_processing(pred)
            lemma = _strip_brackets(pred[lex_index])
            feats = {k: e_gold[ESSENTIAL_KEYS.index(k)] for k in gen_feat_keys}
            for form_gen_num in id2info[lemma]['cond_t'].split('||'):
                form_gen, form_num = list(form_gen_num.lower())
                feats['form_gen'], feats['form_num'] = form_gen, form_num
                analyses, messages = generator.generate(lemma, feats, debug=True)
                stem_id = 'stem' + re.search(r'(\d+)$', lemma).group()
                stem = _strip_brackets(pred[stem_seg_index])
                stemid2form2count.setdefault(stem_id, ['', {}])
                stemid2form2count[stem_id][0] = stem
                stemid2form2count[stem_id][1].setdefault(form_gen_num, 0)
                if analyses:
                    analysis = analyses[0][0]
                    diac = re.sub(stem_id, stem, analysis['diac'])
                    diac = dediac_ar(DEFAULT_NORMALIZE_MAP.map_string(diac))
                    stemid2form2count[stem_id][1][form_gen_num] += gumar[diac]
        form_counts.append(stemid2form2count)
                      
    return form_counts


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
    parser.add_argument("-gumar_inspect_path", default='',
                        type=str, help="Path to the Gumar dataset to inspect.")
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

    db_name = config_local_glf['db']
    db_dir = config_global_glf['db_dir']
    db_dir = os.path.join(db_dir, f"camel-morph-{config_local_glf['dialect']}")
    db = MorphologyDB(os.path.join(db_dir, db_name), flags='a')
    db_gen = MorphologyDB(os.path.join(db_dir, db_name), flags='gd')
    generator = Generator(db_gen)
    analyzer = Analyzer(db, backoff='NOAN-ONLY_ALL')

    stem_classes = get_backoff_stems_from_egy(processing_mode=args.backoff_stems)
    
    with open(args.magold_path) as f:
        ma_gold = f.read()

    POS = set(map(str.lower, POS_NOM))
    data = evaluate_camel_morph._preprocess_magold_data(ma_gold,
                                                        POS,
                                                        _load_analysis,
                                                        field2sentence_index=FIELD2SENTENCE_INDEX,
                                                        field2info_index=FIELD2INFO_INDEX,
                                                        field2ldc_index=None)
    
    possible_analyses = evaluate_camel_morph.evaluate_recall(data,
                                                             args.n,
                                                             args.eval_mode,
                                                             args.possible_analyses_filtered_path,
                                                             analyzer,
                                                             msa_camel_analyzer=None,
                                                             best_analysis=False,
                                                             essential_keys=ESSENTIAL_KEYS)
    filtered_possible_analyses = filter_analyses(possible_analyses)
    
    if os.path.exists(args.possible_analyses_filtered_path):
        answer = input((f'Do you want to load the previous file {args.possible_analyses_filtered_path} (l) '
                        'file or overwrite the previous one (o)? '))
        if answer == 'o':
            with open(args.possible_analyses_filtered_path, 'w') as f:
                print(*HEADER_SHEET, sep='\t', file=f)
                for i, row in enumerate(lexicon, start=1):
                    row['FORM'], row['LEMMA'] = f'stem{i}', f'lemma{i}'
                    print(*[row.get(h, '') for h in HEADER_SHEET], sep='\t', file=f)
            with open(args.possible_analyses_filtered_path, 'wb') as f:
                pickle.dump(filtered_possible_analyses, f)
        elif answer == 'l':
            with open(args.possible_analyses_filtered_path, 'rb') as f:
                filtered_possible_analyses = pickle.load(f)
        else:
            raise NotImplementedError
    
    form_counts = get_generated_form_counts_from_gumar(
        stem_classes, filtered_possible_analyses)
    pass