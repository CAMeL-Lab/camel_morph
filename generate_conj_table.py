import json
import re
from tqdm import tqdm
import argparse
import os

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
        'mod': ['S', 'I', 'J']},
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
        if paradigm['passive']:
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
            paradigm_ += [signature + '.E0'
                            for signature in paradigm['paradigm']]
        elif pos_type == 'nominal':
            paradigm_ += [signature + '.E0'
                          for signature in paradigm_ if 'D' not in signature.split('.')[2]]
        else:
            raise NotImplementedError
            
    return paradigm_

def create_conjugation_tables(lemmas_file_name,
                              pos_type,
                              paradigm_key,
                              paradigms,
                              output_file_name,
                              generator):
    with open(lemmas_file_name) as f:
        lemmas = f.readlines()
        lemmas_conj = []
        for info in tqdm(lemmas):
            lemma, form, pos, gen, num, cond_s, cond_t = info.strip().split(',')
            lemma = bw2ar(strip_lex(lemma))
            form = bw2ar(form)
            if pos_type == 'nominal' and paradigm_key == None:
                paradigm_key = f"gen:{gen} num:{num}"

            paradigm = expand_paradigm(paradigms, pos_type, paradigm_key)
            paradigm_ = {}
            for signature in paradigm:
                features = parse_signature(signature, pos)
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

    for lemma_i, paradigm in enumerate(lemmas_conj):
        for signature, info in paradigm.items():
            if info['analyses']:
                for i, analysis in enumerate(info['analyses']):
                    signature = re.sub('Q', 'P', signature)
                    output = [signature, ar2bw(analysis['lex']), *info['debug'][:3]]
                    output += [info['debug'][3][i], info['debug'][4][i], info['debug'][5][i]]
                    output.append(' '.join(
                        [f"{feat}:{analysis[feat]}" for feat in _test_features if feat in analysis]))
                    output += [ar2bw(analysis['diac']), ar2bw(analysis['bw']), str(len(info['analyses']))]
                    conjugations.append(output)
            else:
                signature = re.sub('Q', 'P', signature)
                output = [signature, strip_lex(lemmas[lemma_i].strip().split(',')[0]), *info['debug'][:3]]
                output += ['', '', '', '', '', '', str(len(info['analyses']))]
                conjugations.append(output)


    with open(os.path.join('conjugation/tables/', output_file_name), 'w') as f:
        print(*header, sep='\t', file=f)
        for output in conjugations:
            print(*output, sep='\t', file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-paradigms", required=True,
                        type=str, help="Configuration file containing the sets of paradigms from which we generate conjugation tables.")
    parser.add_argument("-db", required=True,
                        type=str, help="DB file which will be used with the generation module.")
    parser.add_argument("-pos_type", required=True, choices=['verbal', 'nominal'],
                        type=str, help="POS type of the lemmas for which we want to generate a representative sample.")
    parser.add_argument("-asp", choices=['p', 'i', 'c'],
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-mod", choices=['i', 's', 'j'], default='',
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-dialect", choices=['msa', 'glf', 'egy'], required=True,
                        type=str, help="Aspect to generate the conjugation tables for.")
    parser.add_argument("-repr_lemmas", required=True,
                        type=str, help="Name of the file in conjugation/repr_lemmas/ from which to load the representative lemmas from.")
    parser.add_argument("-output_name", required=True,
                        type=str, help="Name of the file to output the conjugation tables to in conjugation/tables/ directory.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if not os.path.exists('conjugation'):
        os.mkdir('conjugation')
        os.mkdir('conjugation/tables')
    elif not os.path.exists('conjugation/tables'):
        os.mkdir('conjugation/tables')

    db = MorphologyDB(os.path.join('db_iterations', args.db), flags='g')
    generator = Generator(db)
    
    with open(args.paradigms) as f:
        paradigms = json.load(f)[args.dialect]
    asp = f"asp:{args.asp}"
    mod = f" mod:{args.mod}" if args.asp == 'i' and args.mod else ''
    if args.pos_type == 'verbal':
        paradigm_key = asp + mod
    elif args.pos_type == 'nominal':
        paradigm_key = None
    else:
        raise NotImplementedError

    create_conjugation_tables(lemmas_file_name=os.path.join('conjugation/repr_lemmas', args.repr_lemmas),
                              pos_type=args.pos_type,
                              paradigm_key=paradigm_key,
                              paradigms=paradigms,
                              output_file_name=args.output_name,
                              generator=generator)
