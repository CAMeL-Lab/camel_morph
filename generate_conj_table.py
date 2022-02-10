import json
import re
from collections import OrderedDict
from tqdm import tqdm
import argparse
import os
import pickle

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.generator import Generator
from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex

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
        output['count'] = len(outputs)
        signature_outputs.append(output)
    # From the remaining, keep only one of each same-stemgloss outputs
    gloss2outputs = {}
    for so in signature_outputs:
        gloss2outputs.setdefault(so['gloss'], []).append(so)
    signature_outputs_ = []
    for outputs in gloss2outputs.values():
        # Just pick first one since they are supposedly the same
        output = outputs[0]
        output['count'] = len(gloss2outputs)
        output['status'] = 'OK-ONE' if len(gloss2outputs) == 1 else 'CHECK-GT-ONE'
        signature_outputs_.append(output)

    if len(signature_outputs_) > 1:
        if len(set([tuple([so['diac'], so['bw']]) for so in signature_outputs_])) == 1:
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
        lemma = bw2ar(strip_lex(lemma))
        form = bw2ar(form)
        if pos_type == 'nominal' and paradigm_key == None:
            paradigm_key = f"gen:{gen} num:{num}"

        paradigm = expand_paradigm(paradigms, pos_type, paradigm_key)
        outputs = {}
        for signature in paradigm:
            features = parse_signature(signature, pos)
            # Using altered local copy of generator.py in camel_tools
            analyses = generator.generate(lemma, features, debug=True)
            prefix_cats = [a[1] for a in analyses]
            stem_cats = [a[2] for a in analyses]
            suffix_cats = [a[3] for a in analyses]
            analyses = [a[0] for a in analyses]
            debug_info = dict(analyses=analyses,
                              form=ar2bw(form),
                              cond_s=cond_s,
                              cond_t=cond_t,
                              prefix_cats=prefix_cats,
                              stem_cats=stem_cats,
                              suffix_cats=suffix_cats,
                              lemma=ar2bw(lemma),
                              pos=pos,
                              freq=info['freq'])
            outputs[signature] = debug_info
        lemmas_conj.append(outputs)
    
    return lemmas_conj

def process_outputs(lemmas_conj):
    conjugations = []
    header = ["status", "signature", "color", "lemma", "stem", "diac", "bw", "freq",
              "gloss", "count", "cond-s", "cond-t", "pref-cat", "stem-cat", "suff-cat", "feats"]
    color = 0
    for paradigm in lemmas_conj:
        for signature, info in paradigm.items():
            output = {}
            features = parse_signature(signature, info['pos'])
            signature = re.sub('Q', 'P', signature)
            output_['signature'] = signature
            output['stem'] = info['form']
            output_['lemma'] = info['lemma']
            output['cond-s'] = info['cond_s']
            output['cond-t'] = info['cond_t']
            output['color'] = color
            output['freq'] = info['freq']
            if info['analyses']:
                outputs = OrderedDict()
                for i, analysis in enumerate(info['analyses']):
                    assert output_['lemma'] == ar2bw(analysis['lex'])
                    output_ = output.copy()
                    output_['diac'] = ar2bw(analysis['diac'])
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
                    if 'E0' in signature and features.get('vox') and features['vox'] == 'p':
                        output_['status'] = 'CHECK-E0-PASS'
                conjugations += [[output[key] for key in header] for output in outputs_filtered]
            else:
                output_ = output.copy()
                output_['diac'] = ''
                output_['bw'] = ''
                output_['pref-cat'] = ''
                output_['stem-cat'] = ''
                output_['suff-cat'] = ''
                output_ ['feats'] = ''
                output_['gloss'] = ''
                output_['count'] = 0
                if 'E0' in signature and 'intrans' in info['cond_s']:
                    output_['status'] = 'OK-ZERO-E0-INTRANS'
                elif 'E0' in signature and features.get('vox') and features['vox'] == 'p':
                    output_['status'] = 'OK-ZERO-E0-PASS'
                elif features.get('vox') and features['vox'] == 'p':
                    output_['status'] = 'CHECK-ZERO-PASS'
                else:
                    output_['status'] = 'CHECK-ZERO'
                conjugations.append([output_[key] for key in header])
            color = abs(color - 1)
    
    conjugations.insert(0, list(map(str.upper, header)))
    return conjugations


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
    parser.add_argument("-repr_lemmas", required=True,
                        type=str, help="Name of the file in conjugation/repr_lemmas/ from which to load the representative lemmas from.")
    parser.add_argument("-output_name", required=True,
                        type=str, help="Name of the file to output the conjugation tables to in conjugation/tables/ directory.")
    parser.add_argument("-output_dir", default='conjugation/tables',
                        type=str, help="Path of the directory to output the tables to.")
    parser.add_argument("-lemmas_dir", default='conjugation/repr_lemmas',
                        type=str, help="Path of the directory to output the tables to.")
    parser.add_argument("-db_dir", default='db_iterations',
                        type=str, help="Path of the directory to load the DB from.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    conj_dir = args.output_dir.split('/')[0]
    if not os.path.exists(conj_dir):
        os.mkdir(conj_dir)
        os.mkdir(args.output_dir)
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

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

    lemmas_path = os.path.join(args.lemmas_dir, args.repr_lemmas)
    with open(lemmas_path, 'rb') as f:
        lemmas = pickle.load(f)
        lemmas = list(lemmas.values())
    
    lemmas_conj = create_conjugation_tables(lemmas=lemmas,
                                            pos_type=args.pos_type,
                                            paradigm_key=paradigm_key,
                                            paradigms=paradigms,
                                            generator=generator)
    processed_outputs = process_outputs(lemmas_conj)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        for output in processed_outputs:
            print(*output, sep='\t', file=f)
