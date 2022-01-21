import sys
import json
import re

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.generator import Generator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex

from db_maker import gen_cmplx_morph_combs, read_morph_specs

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

test_features = ['pos', 'asp', 'vox', 'per', 'gen', 'num',
                 'mod', 'cas', 'enc0', 'prc0', 'prc1', 'prc2', 'prc3']
# <POS>.<A><P><G><N>.<S><C><V><M>
SIGNATURE_PATTERN = re.compile(
    r'([^\.]+)\.([^\.]{,4})\.([^\.]{,4})\.?P?(\d{,2})?\.?E?(\d{,2})?')
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

def create_representative_verbs_list():
    SHEETS, cond2class, _ = read_morph_specs(sys.argv[1])
    MORPH, LEXICON = SHEETS['morph'], SHEETS['lexicon']
    cmplx_stem_classes = gen_cmplx_morph_combs(
        '[IVPreBuff] [STEM-IV]', MORPH, LEXICON, cond2class)

    representative_verbs = [
        (c[0][0]['#LEMMA'], c[0][0]['FORM'], c[0][0]['COND-S'], c[0][0]['COND-T'])
            for c in cmplx_stem_classes.values()]
    with open('reprensentative_verbs_pv.csv', 'w') as f:
        for info in representative_verbs:
            print(*info, file=f, sep=',')

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
    paradigm_ = paradigms['PV']['paradigm'][:]
    if aspect == "PV":
        if paradigms['PV']['passive']:
            p_ = []
            for signature in paradigms['PV']['paradigm']:
                signature = re.sub('A', 'P', signature)
                p_.append(signature)
            paradigm_ += p_
        else:
            paradigm_ = paradigms['PV']['paradigm']
        
        if paradigms['PV']['enclitics']:
            p_ = []
            for signature in paradigms['PV']['paradigm']:
                signature += '.E0'
                p_.append(signature)
            paradigm_ += p_
            
    return paradigm_

db = MorphologyDB(sys.argv[2], flags='g')
generator = Generator(db)

db_camel = MorphologyDB(sys.argv[2])
analyzer_camel = Analyzer(db_camel)

with open('conjugation/paradigms.json') as f:
    paradigms = json.load(f)

def create_conjugation_tables(file_name, aspect):
    with open(file_name) as f:
        lemmas_conj = []
        for info in f:
            lemma, form, cond_s, cond_t = info.strip().split(',')
            lemma = bw2ar(strip_lex(lemma))
            form = bw2ar(form)
            paradigm = expand_paradigm(paradigms, aspect)
            paradigm_ = {}
            for signature in paradigm:
                features = generate_feats(signature)
                analyses = generator.generate(lemma, features)
                paradigm_[signature] = {'analyses': analyses, 'debug': (ar2bw(form), cond_s, cond_t)}
            lemmas_conj.append(paradigm_)

    conjugations = []
    header = ["SIGNATURE", "LEMMA", "STEM", "COND-S", "COND-T", "FEATURES", "DIAC", "BW", "COUNT"]

    for paradigm in lemmas_conj:
        for signature, info in paradigm.items():
            for analysis in info['analyses']:
                signature = re.sub('Q', 'P', signature)
                output = [signature, ar2bw(analysis['lex']), *info['debug']]
                output.append(' '.join([f"{feat}:{analysis[feat]}" for feat in test_features]))
                output += [ar2bw(analysis['diac']), ar2bw(analysis['bw']), str(len(info['analyses']))]
                conjugations.append(output)

    with open('conjugation/conjugations_pv_v2.0_clitics.tsv', 'w') as f:
        print(*header, sep='\t', file=f)
        for output in conjugations:
            print(*output, sep='\t', file=f)

create_representative_verbs_list()
for aspect in ['PV', 'IV']:
    file_name = f'representative_verbs_{aspect.lower()}.csv'
    # create_conjugation_tables(file_name, aspect)
