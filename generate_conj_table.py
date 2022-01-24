import json
import re
from tqdm import tqdm
import argparse

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
    r'([^\.]+)\.([^\.]{,4})\.([^\.]{,4})\.?P?(\d{,2})?\.?E?(\d{,2})?')
ASPECT_PATTERN = re.compile(r'CV|IV|PV')

sig2feat = {
    'feats0': {
        'pos': ['VERB']},
    'feats1': {
        'asp': ['P', 'I', 'C'],
        'per': ['1', '2', '3'], 
        'gen': ['M', 'F'], 
        'num': ['S', 'D', 'Q']},
    'feats2': {
        'stt': ['D', 'I', 'C'],
        'cas': ['N', 'G', 'A'],
        'vox': ['A', 'P'],
        'mod': ['S', 'I', 'J']},
    'feats3': {
        'prc0': ['0'],
        'prc1': ['1'],
        'prc2': ['2'],
        'prc3': ['3']
    },
    'feats4': {
        'enc0': ['3ms_dobj']
    }
}

def generate_feats(signature):
    match = SIGNATURE_PATTERN.search(signature)
    feats0, feats1, feats2, feats3, feats4 = match.groups()
    feats = {'feats0': feats0, 'feats1': feats1,
             'feats2': feats2, 'feats3': feats3, 'feats4': feats4}
    pos = feats0
    feats_ = {}
    for sig_component, comp_content in feats.items():
        if comp_content:
            for feat, possible_values in sig2feat[sig_component].items():
                if (pos == 'VERB' and feat in ['stt', 'cas']) or \
                    (pos in ['NOUN', 'ADJ'] and feat in ['mod', 'vox']):
                    continue
                if sig_component in ['feats3', 'feats4']:
                    clitic_type = 'prc' if sig_component[-1] == '3' else 'enc'
                    for comp_part in comp_content:
                        comp_part = f'{clitic_type}{comp_part}'
                        for possible_value in possible_values:
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

def expand_paradigm(paradigms, aspect):
    paradigm_ = paradigms[aspect]['paradigm'][:]
    if paradigms[aspect]['passive']:
        p_ = []
        for signature in paradigms[aspect]['paradigm']:
            signature = re.sub('A', 'P', signature)
            p_.append(signature)
        paradigm_ += p_
    else:
        paradigm_ = paradigms[aspect]['paradigm']
    
    if paradigms[aspect]['enclitics']:
        p_ = []
        for signature in paradigms[aspect]['paradigm']:
            signature += '.E0'
            p_.append(signature)
        paradigm_ += p_
            
    return paradigm_

def create_conjugation_tables(lemmas_file_name,
                              paradigm_key,
                              paradigms,
                              output_file_name,
                              generator):
    with open(lemmas_file_name) as f:
        lemmas_conj = []
        for info in tqdm(f.readlines()):
            lemma, form, cond_s, cond_t = info.strip().split(',')
            lemma = bw2ar(strip_lex(lemma))
            form = bw2ar(form)
            paradigm = expand_paradigm(paradigms, paradigm_key)
            paradigm_ = {}
            for signature in paradigm:
                features = generate_feats(signature)
                # Using altered local copy of generator.py in camel_tools
                analyses = generator.generate(lemma, features, debug=True)
                prefix_cats = [a[1] for a in analyses]
                stem_cats = [a[2] for a in analyses]
                suffix_cats = [a[3] for a in analyses]
                analyses = [a[0] for a in analyses]
                paradigm_[signature] = {
                    'analyses': analyses,
                    'debug': (ar2bw(form), cond_s, cond_t, prefix_cats, stem_cats, suffix_cats)}
            lemmas_conj.append(paradigm_)

    conjugations = []
    header = ["SIGNATURE", "LEMMA", "STEM", "COND-S", "COND-T",
              "PREFIX-CAT", "STEM-CAT", "SUFFIX-CAT", "FEATURES", "DIAC", "BW", "COUNT"]

    for paradigm in lemmas_conj:
        for signature, info in paradigm.items():
            for i, analysis in enumerate(info['analyses']):
                signature = re.sub('Q', 'P', signature)
                output = [signature, ar2bw(analysis['lex']), *info['debug'][:3]]
                output += [info['debug'][3][i], info['debug'][4][i], info['debug'][5][i]]
                output.append(' '.join(
                    [f"{feat}:{analysis[feat]}" for feat in _test_features if feat in analysis]))
                output += [ar2bw(analysis['diac']), ar2bw(analysis['bw']), str(len(info['analyses']))]
                conjugations.append(output)

    with open(output_file_name, 'w') as f:
        print(*header, sep='\t', file=f)
        for output in conjugations:
            print(*output, sep='\t', file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-paradigms", required=True,
                        type=str, help="Configuration file containing the sets of paradigms from which we generate conjugation tables.")
    parser.add_argument("-db", required=True,
                        type=str, help="DB file which will be used with the generation module.")
    parser.add_argument("-asp", required=True, choices=['p', 'i', 'c'],
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-mod", choices=['i', 's', 'j'],
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-repr_lemmas", required=True,
                        type=str, help="Name of the file from which to load the representative lemmas from.")
    parser.add_argument("-output_name", required=True,
                        type=str, help="Name of the file to output the conjugation tables to.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    db = MorphologyDB(args.db, flags='g')
    generator = Generator(db)

    with open(args.paradigms) as f:
        paradigms = json.load(f)
    asp = f"asp:{args.asp}"
    mod = f" mod:{args.mod}" if args.asp == 'i' else ''
    create_conjugation_tables(lemmas_file_name=args.repr_lemmas,
                              paradigm_key=asp + mod,
                              paradigms=paradigms,
                              output_file_name=args.output_name,
                              generator=generator)
