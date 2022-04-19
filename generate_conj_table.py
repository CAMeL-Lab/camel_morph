import json
import re
from collections import OrderedDict
from tqdm import tqdm
import argparse
import os
import pickle
import sys

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
        'mod': ['S', 'I', 'J', 'E', 'X']},
    'feats3': {
        'prc0': ['0'],
        'prc1': ['1'],
        'prc2': ['2'],
        'prc3': ['3']
    },
    'feats4': {
        'enc0': {
            'VERB':['3ms_dobj'],
            'NOM': ['3ms_poss']
        },
        'enc1': {
            'VERB':['3ms_dobj'],
            'NOM': ['3ms_poss']
        }
    }
}

header = ["line", "status", "count", "signature", "lemma", "diac_ar", "diac", "freq",
          "qc", "comments", "pattern", "stem", "bw", "gloss", "cond-s", "cond-t", "pref-cat",
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
    
    if paradigm.get('enclitics'):
        paradigm_ += [signature + '.E0' for signature in paradigm_]
            
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
    count = 0
    for outputs in gloss2outputs.values():
        for output in outputs:
            count += 1
            signature_outputs_.append(output)
            if len(set([tuple([o['diac'], o['bw']]) for o in outputs])) == 1 or \
                    len(re.findall(r'\[.+?\]', output['cond-s'])) == 6 and len(outputs) == 2:
                break
    for so in signature_outputs_:
        so['count'] = count
        so['status'] = 'OK-ONE' if count == 1 else 'CHECK-GT-ONE'

    if len(signature_outputs_) > 1:
        if len(set([(so['diac'], so['bw']) for so in signature_outputs_])) == 1 or \
            '-' in so['lemma'] and \
            len(set([(so['pref-cat'], re.sub('intrans', 'trans', so['stem-cat']), so['suff-cat']) for so in signature_outputs_])) == 1:
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
        lemma, form, gloss = info['lemma'], info['form'], info.get('gloss', '_')
        pos, gen, num = info['pos'], info.get('gen', '_'), info.get('num', '_')
        cond_s, cond_t = info['cond_s'], info['cond_t']
        lemma_raw = lemma[:]
        lemma = strip_lex(lemma)
        pattern = None
        if pos_type == 'verbal':
            pattern = assign_pattern(lemma)['pattern_conc']
        elif pos_type == 'nominal':
            match = re.search(r'([MF][SDP])', cond_t)
            form_gen, form_num = None, None
            if match:
                form_gen, form_num = match.groups()[0].lower()
            
            num = process_nom_gen_num_(
                num, form_num, form, cond_t, cond_s, gloss, lemma, pattern, pos, info.get('freq'))
            gen = process_nom_gen_num_(
                gen, form_gen, form, cond_t, cond_s, gloss, lemma, pattern, pos, info.get('freq'))
            if type(num) is dict or type(gen) is dict:
                outputs['NOM.MS.DN'] = num if type(num) is dict else gen 
                lemmas_conj.append(outputs)
                continue
            
            paradigm_key = f'gen:{gen} num:{num}'

        lemma_raw = bw2ar(lemma_raw)

        paradigm = expand_paradigm(paradigms, pos_type, paradigm_key)
        outputs = {}
        for signature in paradigm:
            features = parse_signature(signature, _strip_brackets(pos))
            # Using altered local copy of generator.py in camel_tools
            analyses, debug_message = generator.generate(lemma_raw, features, debug=True)
            prefix_cats = [a[1] for a in analyses]
            stem_cats = [a[2] for a in analyses]
            suffix_cats = [a[3] for a in analyses]
            analyses = [a[0] for a in analyses]
            debug_info = dict(analyses=analyses,
                              gloss=gloss,
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

def _strip_brackets(info):
    if info[0] == '[' and info[-1] == ']':
        info = info[1:-1]
    return info

def process_nom_gen_num_(feat, form_feat,
                         form=None, cond_t=None, cond_s=None, gloss=None,
                         lemma=None, pattern=None, pos=None, freq=None):
    feat_ = _strip_brackets(feat)
    if feat_ == '-':
        if form_feat:
            feat_ = form_feat
        else:
            debug_info = dict(analyses=[],
                              gloss=gloss,
                              form=form,
                              cond_s=cond_s,
                              cond_t=cond_t,
                              prefix_cats=[],
                              stem_cats=[],
                              suffix_cats=[],
                              lemma=lemma,
                              pattern=pattern,
                              pos=pos,
                              freq=freq,
                              debug_message='')
            return debug_info
    return feat_


def process_outputs(lemmas_conj, pos_type):
    conjugations = []
    color, line = 0, 1
    for paradigm in lemmas_conj:
        for signature, info in paradigm.items():
            output = {}
            form = _strip_brackets(info['form'])
            pos = _strip_brackets(info['pos'].upper())
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
                    if pos_type == 'nominal':
                        stem_bw = f"{bw2ar(form)}/{pos}"
                        if stem_bw not in analysis['bw']:
                            continue
                    elif pos_type == 'verbal':
                        if info['lemma'] != ar2bw(analysis['lex']) or _strip_brackets(info['gloss']) != analysis['stemgloss']:
                            continue
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
                        output_[h.upper()] = output.get(h, '')
                    outputs_filtered[i] = output_
                conjugations += outputs_filtered
            else:
                output_ = output.copy()
                output_['count'] = 0
                zero_check = re.findall(r'intrans|trans', info['cond_s'])
                if 'E0' in signature and len(set(zero_check)) == 1 and zero_check[0] == 'intrans':
                    output_['status'] = 'OK-ZERO-E0-INTRANS'
                elif 'E0' in signature and features.get('vox') and features.get('vox') == 'p':
                    output_['status'] = 'OK-ZERO-E0-PASS'
                elif 'C' in signature and features.get('vox') == 'p':
                    output_['status'] = 'OK-ZERO-CV-PASS'
                elif ('C1' in signature or 'C3' in signature) and features.get('asp') == 'c':
                    output_['status'] = 'OK-ZERO-CV-PER'
                elif 'Frozen' in output['cond-s'] and features.get('vox') == 'p':
                    output_['status'] = 'OK-ZERO-FROZEN-PASS'
                elif features.get('vox') == 'p':
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-paradigms", default='',
                        type=str, help="Configuration file containing the sets of paradigms from which we generate conjugation tables.")
    parser.add_argument("-config_file", default='config.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-db", default='',
                        type=str, help="Name of the DB file which will be used with the generation module.")
    parser.add_argument("-pos_type", default='', choices=['verbal', 'nominal', ''],
                        type=str, help="POS type of the lemmas for which we want to generate a representative sample.")
    parser.add_argument("-feats", required=True,
                        type=str, help="Features to generate the conjugation tables for.")
    parser.add_argument("-dialect", default='', choices=['msa', 'glf', 'egy', ''],
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-repr_lemmas", default='',
                        type=str, help="Name of the file in conjugation/repr_lemmas/ from which to load the representative lemmas from.")
    parser.add_argument("-output_name", default='',
                        type=str, help="Name of the file to output the conjugation tables to in conjugation/tables/ directory.")
    parser.add_argument("-output_dir", default='',
                        type=str, help="Path of the directory to output the tables to.")
    parser.add_argument("-lemmas_dir", default='',
                        type=str, help="Path of the directory to output the tables to.")
    parser.add_argument("-db_dir", default='',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-lemma_debug", default=[], action='append',
                        type=str, help="Lemma (without _1) to debug. Use the following format after the flag: lemma pos:val gen:val num:val")
    parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    with open(args.config_file) as f:
        config = json.load(f)
    config_local = config['local'][args.config_name]
    config_global = config['global']

    if args.camel_tools == 'local':
        camel_tools_dir = config_global['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.generator import Generator
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.utils import strip_lex

    from utils import assign_pattern

    bw2ar = CharMapper.builtin_mapper('bw2ar')
    ar2bw = CharMapper.builtin_mapper('ar2bw')

    output_dir = args.output_dir if args.output_dir else config_global['tables_dir']
    conj_dir = output_dir.split('/')[0]
    if not os.path.exists(conj_dir):
        os.mkdir(conj_dir)
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        os.mkdir(output_dir)

    db_name = args.db if args.db else config_local['output']
    db_dir = args.db_dir if args.db_dir else config_global['db_dir']
    db = MorphologyDB(os.path.join(db_dir, db_name), flags='g')
    generator = Generator(db)
    
    paradigms = args.paradigms if args.paradigms else config_global['paradigms_config']
    dialect = args.dialect if args.dialect else config_local['dialect']
    with open(paradigms) as f:
        paradigms = json.load(f)[dialect]
    
    pos_type = args.pos_type if args.pos_type else config_local['pos_type']

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
        lemmas_dir = args.lemmas_dir if args.lemmas_dir else config_global['repr_lemmas_dir']
        repr_lemmas = args.repr_lemmas if args.repr_lemmas else config_local['repr_lemmas']
        lemmas_path = os.path.join(lemmas_dir, repr_lemmas)
        with open(lemmas_path, 'rb') as f:
            lemmas = pickle.load(f)
            lemmas = list(lemmas.values())
        
    lemmas_conj = create_conjugation_tables(lemmas=lemmas,
                                            pos_type=pos_type,
                                            paradigm_key=args.feats,
                                            paradigms=paradigms,
                                            generator=generator)
    outputs = process_outputs(lemmas_conj, pos_type)
    
    if not args.lemma_debug:
        output_name = args.output_name if args.output_name else config_local['debugging']['feats'][args.feats]['conj_tables']
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, 'w') as f:
            for output in outputs:
                print(*output.values(), sep='\t', file=f)
