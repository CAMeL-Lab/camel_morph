import json
import re
from collections import OrderedDict
from tqdm import tqdm
import argparse
import os
import pickle
import gspread
import pandas as pd
from numpy import nan

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.generator import Generator
from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex

from utils import assign_pattern

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

_test_features = ['pos', 'asp', 'vox', 'per', 'gen', 'num',
                 'mod', 'cas', 'enc0', 'prc0', 'prc1', 'prc2', 'prc3']
# <POS>.<A><P><G><N>.<S><C><V><M>
SIGNATURE_PATTERN = re.compile(
    r'([^\.]+)\.([^\.]{,4})\.?([^\.]{,4})\.?P?(\d{,2})?\.?E?(\d{,2})?')

sig2feat = {
    'feats0': {
        'pos': ["ABBREV", "ADJ", "ADJ_COMP", "ADJ_NUM", "ADV",
                "ADV_INTERROG", "ADV_REL", "CONJ", "CONJ_SUB",
                "DIGIT", "FORIEGN", "INTERJ", "NOUN", "NOUN_NUM",
                "NOUN_PROP", "NOUN_QUANT", "PART", "PART_CONNECT",
                "PART_DET", "PART_EMPHATIC", "PART_FOCUS", "PART_FUT",
                "PART_INTERROG", "PART_NEG", "PART_PROG", "PART_RC",
                "PART_RESTRICT", "PART_VERB", "PART_VOC", "PREP",
                "PRON", "PRON_DEM", "PRON_EXCLAM", "PRON_INTERROG",
                "PRON_REL", "PUNC", "VERB", "VERB_NOM", "VERB_PSEUDO"]},
    'feats1': {
        'asp': ['P', 'I', 'C'],
        'per': ['1', '2', '3'], 
        'gen': ['M', 'F'], 
        'num': ['S', 'D', 'Q']},
    'feats2': {
        'stt': ['D', 'I', 'C'],
        'cas': ['N', 'G', 'A'],
        'vox': ['A', 'P'],
        'mod': ['S', 'I', 'J', 'E']},
    'feats3': {
        'prc0': ['0'],
        'prc1': ['1'],
        'prc2': ['2'],
        'prc3': ['3']
    },
    'feats4': {
        'enc0': {
            'VERB':['3ms_dobj'],
            'NOM': ['3ms_poss']}
    }
}

header = ["line", "status", "count", "signature", "lemma", "diac_ar", "diac", "freq",
          "qc", "pattern", "stem", "bw", "gloss", "cond-s", "cond-t", "pref-cat",
          "stem-cat", "suff-cat", "feats", "debug", "color"]

def parse_signature(signature, pos):
    match = SIGNATURE_PATTERN.search(signature)
    feats0, feats1, feats2, feats3, feats4 = match.groups()
    feats = {'feats1': feats1, 'feats2': feats2, 'feats3': feats3, 'feats4': feats4}
    pos_type = feats0
    feats_ = {'pos': pos}
    for sig_component, comp_content in feats.items():
        if comp_content:
            for feat, possible_values in sig2feat[sig_component].items():
                if (pos_type == 'VERB' and feat in ['stt', 'cas']) or \
                    (pos_type == 'NOM' and feat in ['per', 'asp', 'mod', 'vox']):
                    continue
                if sig_component in ['feats3', 'feats4']:
                    # FIXME: this currently works for only one choice of enc0, 
                    # need to make it generic
                    clitic_type = 'prc' if sig_component[-1] == '3' else 'enc'
                    for comp_part in comp_content:
                        comp_part = f'{clitic_type}{comp_part}'
                        for possible_value in possible_values[pos_type]:
                            feats_[comp_part] = possible_value
                else:
                    for possible_value in possible_values:
                        feat_present = comp_content.count(possible_value)
                        if feat_present:
                            feats_[feat] = ('P' if feat == 'num' and possible_value == 'Q' else possible_value).lower()
                            break
                        else:
                            feats_[feat] = 'u'
    return feats_

def expand_paradigm(paradigms, pos_type, paradigm_key):
    paradigm = paradigms[pos_type][paradigm_key]
    paradigm_ = paradigm['paradigm'][:]
    if pos_type == 'verbal':
        if paradigm.get('passive'):
            paradigm_ += [re.sub('A', 'P', signature)
                          for signature in paradigm['paradigm']]
        else:
            paradigm_ = paradigm['paradigm'][:]
    elif pos_type == 'nominal':
        pass
    else:
        raise NotImplementedError
    
    if paradigm['enclitics']:
        if pos_type == 'verbal':
            paradigm_ += [signature + '.E0' for signature in paradigm_]
        elif pos_type == 'nominal':
            paradigm_ += [signature + '.E0'
                          for signature in paradigm_ if 'D' not in signature.split('.')[2]]
        else:
            raise NotImplementedError
            
    return paradigm_

def filter_and_status(outputs):
    # If analysis is the same except for stemgloss, filter out (as duplicate)
    signature_outputs = []
    for output_no_gloss, outputs_same_gloss in outputs.items():
        output = outputs_same_gloss[0]
        signature_outputs.append(output)

    gloss2outputs = {}
    for so in signature_outputs:
        gloss2outputs.setdefault(so['gloss'], []).append(so)
    signature_outputs_ = []
    for outputs in gloss2outputs.values():
        for output in outputs:
            output['count'] = len(outputs)
            output['status'] = 'OK-ONE' if len(outputs) == 1 else 'CHECK-GT-ONE'
            signature_outputs_.append(output)
            if len(set([tuple([o['diac'], o['bw']]) for o in outputs])) == 1:
                break

    if len(signature_outputs_) > 1:
        if len(set([(so['diac'], so['bw']) for so in signature_outputs_])) == 1 or \
            '-' in so['lemma'] and \
            len(set([(so['pref-cat'], so['stem-cat'], so['suff-cat']) for so in signature_outputs_])) == 1:
            for signature_output in signature_outputs_:
                signature_output['status'] = 'OK-GT-ONE'
    
    return signature_outputs_

def create_conjugation_tables(lemmas,
                              pos_type,
                              paradigm_key,
                              paradigms,
                              generator):
    lemmas_conj = []
    for info in tqdm(lemmas):
        lemma, form = info['lemma'], info['form']
        pos, gen, num = info['pos'], info['gen'], info['num']
        cond_s, cond_t = info['cond_s'], info['cond_t']
        lemma = strip_lex(lemma)
        pattern, _, _, _ = assign_pattern(lemma)
        lemma = bw2ar(lemma)
        
        if pos_type == 'nominal' and paradigm_key == None:
            paradigm_key = f"gen:{gen} num:{num}"

        paradigm = expand_paradigm(paradigms, pos_type, paradigm_key)
        outputs = {}
        for signature in paradigm:
            features = parse_signature(signature, pos)
            # Using altered local copy of generator.py in camel_tools
            analyses, debug_message = generator.generate(lemma, features, debug=True)
            prefix_cats = [a[1] for a in analyses]
            stem_cats = [a[2] for a in analyses]
            suffix_cats = [a[3] for a in analyses]
            analyses = [a[0] for a in analyses]
            debug_info = dict(analyses=analyses,
                              form=form,
                              cond_s=cond_s,
                              cond_t=cond_t,
                              prefix_cats=prefix_cats,
                              stem_cats=stem_cats,
                              suffix_cats=suffix_cats,
                              lemma=info['lemma'],
                              pattern=pattern,
                              pos=pos,
                              freq=info.get('freq'),
                              debug_message=debug_message)
            outputs[signature] = debug_info
        lemmas_conj.append(outputs)
    
    return lemmas_conj

def process_outputs(lemmas_conj):
    conjugations = []
    color, line = 0, 1
    for paradigm in lemmas_conj:
        for signature, info in paradigm.items():
            output = {}
            features = parse_signature(signature, info['pos'])
            signature = re.sub('Q', 'P', signature)
            output['signature'] = signature
            output['stem'] = info['form']
            output['lemma'] = info['lemma']
            output['pattern'] = info['pattern']
            output['cond-s'] = info['cond_s']
            output['cond-t'] = info['cond_t']
            output['color'] = color
            output['freq'] = info['freq']
            output['debug'] = ' '.join([m[1] for m in info['debug_message']])
            output['qc'] = ''
            if info['analyses']:
                outputs = OrderedDict()
                for i, analysis in enumerate(info['analyses']):
                    output_ = output.copy()
                    output_['diac'] = ar2bw(analysis['diac'])
                    output_['diac_ar'] = analysis['diac']
                    output_['bw'] = ar2bw(analysis['bw'])
                    output_['pref-cat'] = info['prefix_cats'][i]
                    output_['stem-cat'] = info['stem_cats'][i]
                    output_['suff-cat'] = info['suffix_cats'][i]
                    output_['feats'] = ' '.join(
                        [f"{feat}:{analysis[feat]}" for feat in _test_features if feat in analysis])
                    output_duplicates = outputs.setdefault(tuple(output_.values()), [])
                    output_['gloss'] = analysis['stemgloss']
                    output_duplicates.append(output_)
                outputs_filtered = filter_and_status(outputs)
                for output in outputs_filtered:
                    output['line'] = line
                    line += 1
                    if 'E0' in signature and features.get('vox') and features['vox'] == 'p':
                        output['status'] = 'CHECK-E0-PASS'
                for i, output in enumerate(outputs_filtered):
                    output_ = OrderedDict()
                    for h in header:
                        output_[h.upper()] = output[h]
                    outputs_filtered[i] = output_
                conjugations += outputs_filtered
            else:
                output_ = output.copy()
                output_['count'] = 0
                if 'E0' in signature and 'intrans' in info['cond_s']:
                    output_['status'] = 'OK-ZERO-E0-INTRANS'
                elif 'E0' in signature and features.get('vox') and features['vox'] == 'p':
                    output_['status'] = 'OK-ZERO-E0-PASS'
                elif 'C' in signature and features['vox'] == 'p':
                    output_['status'] = 'OK-ZERO-CV-PASS'
                elif ('' in signature or 'C3' in signature) and features['asp'] == 'c':
                    output_['status'] = 'OK-ZERO-CV-PER'
                elif features.get('vox') and features['vox'] == 'p':
                    output_['status'] = 'CHECK-ZERO-PASS'
                else:
                    output_['status'] = 'CHECK-ZERO'
                output_['line'] = line
                line += 1
                output_ordered = OrderedDict()
                for h in header:
                    output_ordered[h.upper()] = output_.get(h, '')
                conjugations.append(output_ordered)
            color = abs(color - 1)
    
    conjugations.insert(0, OrderedDict((i, x) for i, x in enumerate(map(str.upper, header))))
    return conjugations

def triplet2row(prev_outputs, new_outputs):
    prev_full_key, new_full_key = {}, {}
    prev_partial_key, new_partial_key = {}, {}
    # Each dict key will contain a list of size # of analyses because (signature, lemma) is
    # unique for each paradigm slot (regardless of diac).
    for _, row in prev_outputs.iterrows():
        key = (row['SIGNATURE'], row['LEMMA'], row['DIAC'])
        row = row.to_dict()
        assert row['QC'] != 'PROBLEM' or row['QC'] != '', \
            f"'{row['QC']}' found. Get rid of all QC tags different from 'PROBLEM' before creating a new sheet."
        row_ = OrderedDict()
        for h in header:
            row_[h.upper()] = row.get(h.upper(), '')
        prev_full_key.setdefault(key, []).append(row_)
        prev_partial_key.setdefault(key[:-1], []).append(row_)

    for row in new_outputs:
        key = (row['SIGNATURE'], strip_lex(row['LEMMA']), row['DIAC'])
        new_full_key.setdefault(key, []).append(row)
        new_partial_key.setdefault(key[:-1], []).append(row)
    
    return prev_full_key, new_full_key, prev_partial_key, new_partial_key


def quality_check(prev_outputs, new_outputs):
    prev_full_key, new_full_key, prev_partial_key, new_partial_key = triplet2row(
        prev_outputs, new_outputs)

    outputs = []
    for key, rows in new_full_key.items():
        if key in prev_full_key:
            for row in rows:
                if len(rows) > 1 and len(prev_full_key[key]) == 1:
                    row['QC'] = 'PROBLEM-GT-1'
                # If row (or at least one row in the case of `count` > 1)
                # in previous sheet was annotated as PROBLEM
                elif any([r['QC'] == 'PROBLEM' for r in prev_full_key[key]]):
                    row['QC'] = 'PROBLEM'
                else:
                    row['QC'] = ''
                row['COLOR'] = ''
                outputs.append(row)
        else:
            if key[:-1] in prev_partial_key:
                for row in rows:
                    if any([r['QC'] == 'PROBLEM' for r in prev_partial_key[key[:-1]]]):
                        row['QC'] = 'PROBLEM-CHANGED'
                    else:
                        row['QC'] = 'GOOD-CHANGED'
                    row['COLOR'] = ''
                    outputs.append(row)

    outputs.insert(0,
        OrderedDict((i, h.upper()) for i, h in enumerate(header)))
    return outputs

def quality_check_merge(prev_outputs, new_outputs):
    _, new_full_key, prev_partial_key, new_partial_key = triplet2row(
        prev_outputs, new_outputs)
    prev_full_key = OrderedDict()
    for key, prev_rows in prev_partial_key.items():
        new_rows = new_partial_key.get(key)
        if new_rows != None and new_rows[0]['QC'] == '' and prev_rows[0]['QC'] == 'PROBLEM':
            assert len(new_rows) == 1 and len(prev_rows) == 1, 'Unhandled case.'
        else:
            for row in prev_rows:
                prev_full_key.setdefault(key + (row['DIAC'],), []).append(row)
    
    merged = OrderedDict()
    for key, rows in new_full_key.items():
        merged[key] = rows
    for key, rows in prev_full_key.items():
        if key not in merged:
            merged[key] = rows
    outputs = []
    for rows in merged.values():
        for row in rows:
            outputs.append(row)

    outputs.insert(0,
        OrderedDict((i, h.upper()) for i, h in enumerate(header)))
    return outputs

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-paradigms", required=True,
                        type=str, help="Configuration file containing the sets of paradigms from which we generate conjugation tables.")
    parser.add_argument("-db", required=True,
                        type=str, help="Name of the DB file which will be used with the generation module.")
    parser.add_argument("-pos_type", required=True, choices=['verbal', 'nominal'],
                        type=str, help="POS type of the lemmas for which we want to generate a representative sample.")
    parser.add_argument("-asp", choices=['p', 'i', 'c'],
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-mod", choices=['i', 's', 'j', 'e'], default='',
                        type=str, help="Mood to generate the conjugation tables for.")
    parser.add_argument("-vox", choices=['a', 'p'], default='',
                        type=str, help="Voice to generate the conjugation tables for.")
    parser.add_argument("-dialect", choices=['msa', 'glf', 'egy'], required=True,
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-repr_lemmas",
                        type=str, help="Name of the file in conjugation/repr_lemmas/ from which to load the representative lemmas from.")
    parser.add_argument("-output_name",
                        type=str, help="Name of the file to output the conjugation tables to in conjugation/tables/ directory.")
    parser.add_argument("-output_dir", default='conjugation/tables',
                        type=str, help="Path of the directory to output the tables to.")
    parser.add_argument("-lemmas_dir", default='conjugation/repr_lemmas',
                        type=str, help="Path of the directory to output the tables to.")
    parser.add_argument("-db_dir", default='db_iterations',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-lemma_debug", default=[], action='append',
                        type=str, help="Lemma (without _1) to debug. Use the following format after the flag: lemma pos:val gen:val num:val")
    parser.add_argument("-qc", default=[], action='append',
                        type=str, help="Evaluate quality check. Add spreadheet_name followed by gsheet name as arguments. Alternatively, add just local CSV path to avoid redownloading.")
    parser.add_argument("-qc_merge", default=[], action='append',
                        type=str, help="Merge previous QC sheet with latest annotated QC sheet.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    conj_dir = args.output_dir.split('/')[0]
    if not os.path.exists(conj_dir):
        os.mkdir(conj_dir)
        os.mkdir(args.output_dir)
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    assert bool(args.qc) ^ bool(args.qc_merge) or bool(args.qc) == False, 'Only one of -qc or -qc_merge can be used at the same time.'
    if args.qc or args.qc_merge:
        sa = gspread.service_account(
                "/Users/chriscay/.config/gspread/service_account.json")
        spreadsheet_sheet = args.qc[0].split() if args.qc else args.qc_merge[0].split()
        sh = sa.open(spreadsheet_sheet[0])

    if args.qc_merge:
        sheet_names = [sheet.title for sheet in sh.worksheets()]
        sheet_names = [
            sheet_name for sheet_name in sheet_names if f'{spreadsheet_sheet[1]}-' in sheet_name]
        assert sheet_names, 'No other sheet to merge with.'
        latest_index = max(int(sheet_name.split('-')[-1]) for sheet_name in sheet_names)
        if len(sheet_names) == 1:
            prev_index = ''
        else:
            prev_index = f'-{str(latest_index - 1)}'
        latest_index = f'-{str(latest_index)}'
        prev_sheet = sh.worksheet(title=spreadsheet_sheet[1] + prev_index)
        latest_sheet = sh.worksheet(title=spreadsheet_sheet[1] + latest_index)
        prev_sheet = pd.DataFrame(prev_sheet.get_all_records())
        latest_sheet = pd.DataFrame(latest_sheet.get_all_records()).values.tolist()
        latest_sheet = [OrderedDict((h.upper(), row[i]) for i, h in enumerate(header)) for row in latest_sheet]
        outputs = quality_check_merge(prev_sheet, latest_sheet)

    else:
        db = MorphologyDB(os.path.join(args.db_dir, args.db), flags='g')
        generator = Generator(db)
        
        with open(args.paradigms) as f:
            paradigms = json.load(f)[args.dialect]
        asp = f"asp:{args.asp}"
        mod = f" mod:{args.mod}" if args.asp in ['i', 'c'] and args.mod else ''
        vox = f" vox:{args.vox}" if args.vox else ''
        if args.pos_type == 'verbal':
            paradigm_key = asp + mod + vox
        elif args.pos_type == 'nominal':
            paradigm_key = None
            raise NotImplementedError
        else:
            raise NotImplementedError

        if args.lemma_debug:
            lemma_debug = args.lemma_debug[0].split()
            lemma = lemma_debug[0]
            feats = {feat.split(':')[0]: feat.split(':')[1] for feat in lemma_debug[1:]}
            lemmas = [dict(form='',
                        lemma=lemma.replace('\\', ''),
                        cond_t='',
                        cond_s='',
                        pos=feats['pos'],
                        gen=feats['gen'],
                        num=feats['num'])]
        else:
            lemmas_path = os.path.join(args.lemmas_dir, args.repr_lemmas)
            with open(lemmas_path, 'rb') as f:
                lemmas = pickle.load(f)
                lemmas = list(lemmas.values())
            
        lemmas_conj = create_conjugation_tables(lemmas=lemmas,
                                                pos_type=args.pos_type,
                                                paradigm_key=paradigm_key,
                                                paradigms=paradigms,
                                                generator=generator)
        outputs = process_outputs(lemmas_conj)

    if args.qc:
        if len(spreadsheet_sheet) == 2:
            worksheet = sh.worksheet(title=spreadsheet_sheet[1])
            paradigm_debug = pd.DataFrame(worksheet.get_all_records())
            paradigm_debug.to_csv(spreadsheet_sheet[1])
        elif len(spreadsheet_sheet) == 1:
            paradigm_debug = pd.read_csv(spreadsheet_sheet[0])
            paradigm_debug = paradigm_debug.replace(nan, '', regex=True)
        else:
            raise NotImplementedError
        outputs = quality_check(prev_outputs=paradigm_debug, new_outputs=outputs[1:])
    
    if not args.lemma_debug:
        output_path = os.path.join(args.output_dir, args.output_name)
        with open(output_path, 'w') as f:
            for output in outputs:
                print(*output.values(), sep='\t', file=f)
