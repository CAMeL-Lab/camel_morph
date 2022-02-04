import argparse
import json
import os
import pickle

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex

import db_maker

ar2bw = CharMapper.builtin_mapper('ar2bw')

def create_repr_lemmas_list(config_file,
                            config_name):
    with open(config_file) as f:
        config = json.load(f)
    SHEETS, _ = db_maker.read_morph_specs(config, config_name)

    lemmas_uniq = {}
    for _, row in SHEETS['lexicon'].iterrows():
        feats = tuple(row['FEAT'].split())
        if 'vox:p' in feats:
            continue
        lemmas_uniq.setdefault(row['LEMMA'], []).append(
            (row['COND-T'], row['COND-S'], feats, row['FORM']))
    uniq_lemma_classes = {}
    for lemma, stems in lemmas_uniq.items():
        lemmas_cond_sig = tuple([stem[:3] for stem in stems])
        feats = {}
        for stem in stems:
            stem_feats = {
                feat.split(':')[0]: feat.split(':')[1] for feat in stem[2]}
            for feat, value in stem_feats.items():
                if feats.get(feat) != None and value != feats[feat]:
                    feats[feat] += f'+{value}'
                else:
                    feats[feat] = value
        form = '+'.join(set([stem[3] for stem in stems]))
        cond_t = '+'.join(set([stem[0] if stem[0] else '_' for stem in stems]))
        cond_t = cond_t if cond_t != '+' else ''
        cond_s = '+'.join(set([stem[1] if stem[1] else '_' for stem in stems]))
        cond_s = cond_s if cond_s != '+' else ''
        info = dict(form=form,
                    lemma=lemma.split(':')[1],
                    cond_t=cond_t,
                    cond_s=cond_s,
                    pos=feats['pos'],
                    gen=feats['gen'],
                    num=feats['num'])
        uniq_lemma_classes.setdefault(lemmas_cond_sig, []).append(info)
    
    # Get rid of multi-gloss lemmas (to avoid duplicates in conjugation table debugging)
    lemmas_info_dict = {}
    for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
        for info in lemmas_info:
            lemmas_info_dict.setdefault(strip_lex(info['lemma']), []).append(info)
    
    for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
        lemmas_info_ = []
        for info in lemmas_info:
            if len(lemmas_info_dict[strip_lex(info['lemma'])]) == 1:
                lemmas_info_.append(info)
        if lemmas_info_ == []:
            lemmas_info_.append(lemmas_info[0])
        uniq_lemma_classes[lemmas_cond_sig] = lemmas_info_

    return {k: v[0] for k, v in uniq_lemma_classes.items()}


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
    args = parser.parse_args([] if "__file__" not in globals() else None)

    conj_dir = args.output_dir.split('/')[0]
    if not os.path.exists(conj_dir):
        os.mkdir(conj_dir)
        os.mkdir(args.output_dir)
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    uniq_lemma_classes = create_repr_lemmas_list(config_file=args.config_file,
                                                 config_name=args.config_name)

    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(uniq_lemma_classes, f)
        
