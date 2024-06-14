import pickle
import sys
import os
import argparse

from tqdm import tqdm
import pandas as pd
import gspread

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.utils.utils import Config
from camel_morph.eval.evaluate_camel_morph import load_required_pos
from camel_morph.eval.evaluate_camel_morph_stats import get_analysis_counts
from camel_morph.eval.eval_utils import essential_keys_form_no_lex_pos as essential_keys

essential_keys.insert(0, 'pos')

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-db_system", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the system DB file which we want to compare with the baseline.")
parser.add_argument("-db_baseline", default='eval_files/calima-msa-s31_0.4.2.utf8.db',
                    type=str, help="Path of the baseline DB file we will be comparing against.")
parser.add_argument("-gen_output_dir", default='sandbox_files/db_map_diff',
                    type=str, help="Path of the directory to which the generation pickles should be output.")
parser.add_argument("-pos", default='any',
                    type=str, help="Restrict diff to a certain POS.")
parser.add_argument("-morph_type", required=True,
                    type=str, help="Restrict diff to a certain complex morpheme type.")
parser.add_argument("-diff_output_sheet", default='',
                    type=str, help="Spreadsheet and sheet to output the diff to.")
parser.add_argument("-test_mode", default=False,
                    action='store_true', help="Load generated analyses from pickle file instead of generating them.")
parser.add_argument("-service_account", default='',
                    type=str, help="Path of the JSON file containing the information about the service account used for the Google API.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")
args = parser.parse_args([] if "__file__" not in globals() else None)


config = Config(args.config_file)
if args.camel_tools == 'local':
    sys.path.insert(0, config.camel_tools)

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.database import MorphologyDB

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')


def get_stem_cat_hash(db):
    stem_cat_hash = {}
    for match in db.stem_hash:
        if match == 'NOAN':
            continue
        for cat_analysis in db.stem_hash[match]:
            cat, analysis = cat_analysis
            if CAMEL_POS and analysis['pos'] not in CAMEL_POS:
                continue
            stem_cat_hash.setdefault(cat, []).append(analysis)
    return stem_cat_hash


def get_info_and_morphemes(system, db, morph_type):
    gen_path = os.path.join(
        args.gen_output_dir, f'inspect_morphemes_{system}.pkl')
    if not args.test_mode:
        with open(gen_path, 'wb') as f:
            info = get_analysis_counts(db, camel_pos=CAMEL_POS)
            pickle.dump(info, f)
    else:
        with open(gen_path, 'rb') as f:
            info = pickle.load(f)
    
    morphemes = {
        x for x in set(tuple(xx.get(k, 'NA') for k in essential_keys)
                       for x in info['cmplx_morphs'][morph_type]
                       for xx in getattr(db, f'{morph_type}_cat_hash')[x])}
    
    return morphemes, info


def get_feats2attribution(morph_type,
                          morphemes_src,
                          analysis_counts_src,
                          db_src,
                          morphemes_tgt=None):
    morph_type2index = dict(prefix=0, stem=1, suffix=2)
    if morphemes_tgt is None:
        src_minus_tgt = morphemes_src
    else:
        src_minus_tgt = morphemes_src - morphemes_tgt
    morph_cat_hash = getattr(db_src, f'{morph_type}_cat_hash')

    feats2attribution = {}

    for cat_morph, count in tqdm(analysis_counts_src.items()):
        cat = cat_morph[morph_type2index[morph_type]]
        for morph in morph_cat_hash[cat]:
            morph_feats = tuple(morph.get(k, 'NA') for k in essential_keys)
            if morphemes_tgt is None or morph_feats in src_minus_tgt:
                feats2attribution.setdefault(morph_feats, 0)
                feats2attribution[morph_feats] += count / len(morph_cat_hash[cat])

    return feats2attribution


def diff_per_cmplx_morph_type(morph_type):
    morphemes_baseline, info_baseline = get_info_and_morphemes(
        'baseline', db_baseline, morph_type)
    morphemes_system, info_system = get_info_and_morphemes(
        'system', db_system, morph_type)
    
    analysis_counts_baseline = info_baseline['analysis_counts']['analyses']
    analysis_counts_system = info_system['analysis_counts']['analyses']

    feats2attribution_baseline = get_feats2attribution(
        morph_type,
        morphemes_baseline,
        analysis_counts_baseline,
        db_baseline)
    
    feats2attribution_system = get_feats2attribution(
        morph_type,
        morphemes_system,
        analysis_counts_system,
        db_system)
    
    common_feat_combs = set(feats2attribution_baseline) & set(
        feats2attribution_system)
    baseline_minus_system_combs = set(feats2attribution_baseline) - set(
        feats2attribution_system)
    system_minus_baseline_combs = set(feats2attribution_system) - set(
        feats2attribution_baseline)
    
    feat_combs = [('common', common_feat_combs),
                  ('baseline_minus_system', baseline_minus_system_combs),
                  ('system_minus_baseline', system_minus_baseline_combs)]
    
    diff = {}
    for name, feat_combs_ in feat_combs:
        for feat_comb in feat_combs_:
            baseline_count, system_count = 0, 0
            if name != 'system_minus_baseline':
                baseline_count = feats2attribution_baseline[feat_comb]
                diff.setdefault(name, {}).setdefault(
                    feat_comb, {}).setdefault('baseline', baseline_count)
            if name != 'baseline_minus_system':
                system_count = feats2attribution_system[feat_comb]
                diff.setdefault(name, {}).setdefault(
                    feat_comb, {}).setdefault('system', system_count)
            
            if system_count == 0 and baseline_count == 0:
                delta = 0
            elif system_count != 0 and baseline_count == 0:
                delta = 'new'
            else:
                delta = (system_count - baseline_count) / baseline_count
            diff.setdefault(name, {}).setdefault(
                feat_comb, {}).setdefault('delta', delta)
    
    return diff
    

def augment_db_stem_cat_hash(db):
    stem_cat_hash = get_stem_cat_hash(db)
    db.stem_cat_hash = stem_cat_hash

if __name__ == '__main__':
    sa = gspread.service_account(args.service_account)

    ATB_POS, CAMEL_POS, POS_OR_TYPE = load_required_pos(args.pos, '')

    db_baseline = MorphologyDB(args.db_baseline, 'dag')
    db_system = MorphologyDB(args.db_system, 'dag')

    augment_db_stem_cat_hash(db_baseline)
    augment_db_stem_cat_hash(db_system)

    diff = diff_per_cmplx_morph_type(args.morph_type)

    header = ['Diff', *essential_keys, 'Baseline counts',
              'System counts', 'Δ counts (%)']
    pass
    output = []
    for name, feat2counts in diff.items():
        for feat_comb, system2count in feat2counts.items():
            row = [name,
                   *feat_comb,
                   int(system2count.get('baseline', 0)),
                   int(system2count.get('system', 0)),
                   system2count['delta']]
            output.append(row)
    
    output_df = pd.DataFrame(output)
    output_df.columns = header

    diff_output_sheet = args.diff_output_sheet.split()
    assert len(diff_output_sheet) == 2
    sh_name, sheet_name = diff_output_sheet

    sh = sa.open(sh_name)
    sheets = sh.worksheets()

    if sheet_name in [sheet.title for sheet in sheets]:
        sheet = sh.worksheet(title=sheet_name)
    else:
        sheet = sh.add_worksheet(title=sheet_name, rows=1, cols=1)

    sheet.update([output_df.columns.values.tolist()] +
                 output_df.values.tolist())
    





