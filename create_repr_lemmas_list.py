import argparse
import json
import os
import pickle
import numpy as np
import re

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

import db_maker

nominals = ["ABBREV", "ADJ", "ADJ_COMP", "ADJ_NUM", "ADV",
            "ADV_INTERROG", "ADV_REL",
            "FORIEGN", "INTERJ", "NOUN", "NOUN_NUM",
            "NOUN_PROP", "NOUN_QUANT",
            "PRON", "PRON_DEM", "PRON_EXCLAM", "PRON_INTERROG",
            "PRON_REL", "VERB_NOM", "VERB_PSEUDO"]
nominals = [n.lower() for n in nominals]

ar2bw = CharMapper.builtin_mapper('ar2bw')
bw2ar = CharMapper.builtin_mapper('bw2ar')
db = MorphologyDB.builtin_db()
analyzer = Analyzer(db)

def create_repr_lemmas_list(config_file,
                            config_name,
                            pos_type):
    with open(config_file) as f:
        config = json.load(f)
    SHEETS, _ = db_maker.read_morph_specs(config, config_name)
    SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
        lambda row: re.sub(r'hamzated|hollow|defective', '', row['COND-S']), axis=1)
    SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
        lambda row: re.sub(r' +', ' ', row['COND-S']), axis=1)

    lemmas_uniq = {}
    lemmas_stripped_uniq = {}
    for _, row in SHEETS['lexicon'].iterrows():
        #TODO: see if need to exclude anything for noms
        feats = tuple([feat for feat in row['FEAT'].split() if 'vox' not in feat])
        lemmas_uniq.setdefault(row['LEMMA'], []).append(
            (row['COND-T'], row['COND-S'], feats, row['FORM'], row['GLOSS']))
        lemmas_stripped_uniq.setdefault(strip_lex(row['LEMMA'].split(':')[1]), []).append(
            (row['COND-T'], row['COND-S'], feats, row['FORM'], row['GLOSS']))
    uniq_lemma_classes = {}
    for lemma, stems in lemmas_uniq.items():
        lemmas_cond_sig = tuple(sorted([stem[:3] for stem in stems]))
        feats = {}
        for stem in stems:
            stem_feats = {
                feat.split(':')[0]: feat.split(':')[1] for feat in stem[2]}
            for feat, value in stem_feats.items():
                if feats.get(feat) != None and value != feats[feat]:
                    feats[feat] += f'+{value}'
                else:
                    feats[feat] = value
        form = ''.join([(f"[{stem[3]}]({i})") for i, stem in enumerate(stems)])
        cond_t_ = {}
        for i, stem in enumerate(stems):
            cond_t_.setdefault(stem[0], []).append(f'({i})')
        cond_t = ''.join([f"[{ct if ct else '-'}]{''.join(indexes)}" for ct, indexes in cond_t_.items()])
        cond_s_ = {}
        for i, stem in enumerate(stems):
            cond_s_.setdefault(stem[1], []).append(f'({i})')
        cond_s = ''.join([f"[{cs if cs else '-'}]{''.join(indexes)}" for cs, indexes in cond_s_.items()])
        info = dict(form=form,
                    lemma=lemma.split(':')[1],
                    cond_t=cond_t,
                    cond_s=cond_s,
                    pos=feats['pos'],
                    gen=feats['gen'],
                    num=feats['num'],
                    gloss=stem[4])
        uniq_lemma_classes.setdefault(lemmas_cond_sig, {'freq': 0, 'lemmas': []})
        uniq_lemma_classes[lemmas_cond_sig]['freq'] += 1
        uniq_lemma_classes[lemmas_cond_sig]['lemmas'].append(info)
    
    lemma2prob = {}
    for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
        for info in lemmas_info['lemmas']:
            lemma_ar = bw2ar(strip_lex(info['lemma']))
            analyses = analyzer.analyze(lemma_ar)
            if pos_type == 'verbal':
                analyses_filtered = [a  for a in analyses
                    if a['lex'] == lemma_ar and a['pos'] == 'verb' and
                    a['per'] == '3' and a['num'] == 's' and a['gen'] == 'm']
            #TODO: update this before generating for nouns
            elif pos_type == 'nominal':
                analyses_filtered = [a  for a in analyses
                    if a['pos'] in nominals and
                    a['stemgloss'] == info['gloss']]
            lemma2prob[info['lemma']] = analyses_filtered

    for lemma, analyses in lemma2prob.items():
        if len(analyses) > 1:
            assert len(set([a['lex'] for a in analyses])) == 1, 'Cannot discard analysis'
            lemma2prob[lemma] = [analyses[0]]
    for lemma, analyses in lemma2prob.items():
        if len(analyses) <= 1:
            if len(analyses) == 1:
                lemma2prob[lemma] = analyses[0]['pos_lex_logprob']
            else:
                lemma2prob[lemma] = -99.0
        else:
            raise 'Still more than one analysis after filtering and discarding'
    
    assert all([any([True if info['lemma'] in lemma2prob else False for info in lemmas_info['lemmas']])
        for lemmas_info in uniq_lemma_classes.values()]), \
            'Some classes do not contain any representative after filtering'
    for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
        lemmas = [info['lemma'] for info in lemmas_info['lemmas']]
        best_indexes = (-np.array([lemma2prob[lemma] for lemma in lemmas])).argsort()[:len(lemmas)]
        for best_index in best_indexes:
            best_lemma_info = lemmas_info['lemmas'][best_index]
            best_lemma_info['freq'] = lemmas_info['freq']
            lemma = best_lemma_info['lemma']
            lemma_stripped = strip_lex(lemma)
            if not ('-' in lemma and
                    len(lemmas_stripped_uniq[lemma_stripped]) > 2 and
                    all([stem[0] == '' for stem in lemmas_stripped_uniq[lemma_stripped]])):
                break
        uniq_lemma_classes[lemmas_cond_sig] = best_lemma_info
        
    return uniq_lemma_classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", required=True,
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-output_dir", default='conjugation/repr_lemmas',
                        type=str, help="Path of the directory to output the lemmas to.")
    parser.add_argument("-output_name", required=True,
                        type=str, help="Name of the file to output the representative lemmas to. File will be placed in a directory called conjugation/repr_lemmas/")
    parser.add_argument("-pos_type", required=True, choices=['verbal', 'nominal'],
                        type=str, help="POS type of the lemmas.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    conj_dir = args.output_dir.split('/')[0]
    if not os.path.exists(conj_dir):
        os.mkdir(conj_dir)
        os.mkdir(args.output_dir)
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    uniq_lemma_classes = create_repr_lemmas_list(config_file=args.config_file,
                                                 config_name=args.config_name,
                                                 pos_type=args.pos_type)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(uniq_lemma_classes, f)
        
