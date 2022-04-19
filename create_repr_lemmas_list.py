import argparse
import json
import os
import pickle
import numpy as np
import re
import sys

from paradigm_debugging import AnnotationBank

import db_maker_utils

nominals = ["ABBREV", "ADJ", "ADJ_COMP", "ADJ_NUM", "ADV",
            "ADV_INTERROG", "ADV_REL",
            "FORIEGN", "INTERJ", "NOUN", "NOUN_NUM",
            "NOUN_PROP", "NOUN_QUANT",
            "PRON", "PRON_DEM", "PRON_EXCLAM", "PRON_INTERROG",
            "PRON_REL", "VERB_NOM", "VERB_PSEUDO"]
nominals = [n.lower() for n in nominals]

def create_repr_lemmas_list(lexicon,
                            class_keys,
                            extended_lemma_keys,
                            pos_type,
                            bank=None,
                            info_display_format='compact',
                            lemma2prob=None):
    lemmas_uniq, lemmas_stripped_uniq = get_extended_lemmas(lexicon, extended_lemma_keys)
    uniq_lemma_classes = {}
    for lemma, stems in lemmas_uniq.items():
        info = {}
        if info_display_format == 'compact':
            for i, stem in enumerate(stems):
                for k in stem:
                    values = info.setdefault(k, {})
                    values.setdefault(stem[k], []).append(f'({i})')
            for k in stems[0]:
                info[k] = ''.join([f"[{ct if ct else '-'}]{''.join(indexes)}"
                                    for ct, indexes in sorted(info[k].items(), key=lambda x: x[0])])
        
        elif info_display_format == 'expanded':
            for k in stems[0]:
                if k in ['cond_t', 'cond_s']:
                    info[k] = ''.join(
                        sorted([f"[{stem[k]}]" if stem[k] else '[-]' for stem in stems]))
                else:
                    info[k] = ''.join(
                        list(set(sorted([f"[{stem[k]}]" if stem[k] else '[-]' for stem in stems]))))
            info['lemma'] = info['lemma'][1:-1]
        
        lemmas_cond_sig = [{k: stem.get(k) for k in class_keys} for stem in stems]
        lemmas_cond_sig = tuple(
            sorted([tuple(stem.values()) for stem in lemmas_cond_sig]))
        uniq_lemma_classes.setdefault(lemmas_cond_sig, {'freq': 0, 'lemmas': []})
        uniq_lemma_classes[lemmas_cond_sig]['freq'] += 1
        uniq_lemma_classes[lemmas_cond_sig]['lemmas'].append(info)

    uniq_lemma_classes = get_highest_prob_lemmas(
        pos_type, uniq_lemma_classes, lemmas_stripped_uniq, bank, lemma2prob)
    
    return uniq_lemma_classes


def get_extended_lemmas(lexicon, extended_lemma_keys):
    lemmas_uniq = {}
    lemmas_stripped_uniq = {}

    def get_info(row):
        feats = {feat.split(':')[0]: feat.split(':')[1]
                 for feat in row['FEAT'].split()}
        cond_t = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-T'].split()]))
        cond_s = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-S'].split()]))
        lemma = row['LEMMA'].split(':')[1]
        info = dict(lemma=lemma,
                    form=row['FORM'],
                    cond_t=cond_t,
                    cond_s=cond_s,
                    gloss=row['GLOSS'],
                    index=row['index'])
        info.update(feats)
        extended_lemma = tuple([lemma] + [info[k]
                                          for k in extended_lemma_keys[1:]])
        lemmas_uniq.setdefault(extended_lemma, []).append(info)
        lemmas_stripped_uniq.setdefault(strip_lex(lemma), []).append(info)
    
    lexicon["index"] = np.arange(lexicon.shape[0])
    lexicon.apply(get_info, axis=1)
    
    return lemmas_uniq, lemmas_stripped_uniq
    

def get_highest_prob_lemmas(pos_type, uniq_lemma_classes, lemmas_stripped_uniq, bank, lemma2prob=None):
    if lemma2prob is None:
        db = MorphologyDB.builtin_db()
        lemma2prob = {}
        for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
            for info in lemmas_info['lemmas']:
                lemma_stripped = strip_lex(info['lemma'])
                lemma_ar = bw2ar(lemma_stripped)
                normalized_lemma_ar = DEFAULT_NORMALIZE_MAP(dediac_ar(lemma_ar))
                matches = db.stem_hash.get(normalized_lemma_ar, [])
                db_entries = [db_entry[1] for db_entry in matches]

                if pos_type == 'verbal':
                    entries_filtered = [e  for e in db_entries
                        if strip_lex(e['lex']) == lemma_ar and e['pos'] == 'verb']

                elif pos_type == 'nominal':
                    entries_filtered = [e  for e in db_entries
                        if e['lex'] == lemma_ar and e['pos'] == info['pos'][1:-1]]

                if len(entries_filtered) >= 1:
                    lemma2prob[lemma_stripped] = max([float(a['pos_lex_logprob']) for a in db_entries])
                else:
                    lemma2prob[lemma_stripped] = -99.0
    else:
        lemma2prob = {strip_lex(lemma): prob for lemma, prob in lemma2prob.items()}
        for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
            for info in lemmas_info['lemmas']:
                lemma_stripped = strip_lex(info['lemma'])
                if lemma_stripped not in lemma2prob:
                    lemma2prob[lemma_stripped] = 0

    if bank is not None:
        old_lemmas = set([strip_lex(entry[1]) for entry in bank._bank])

    uniq_lemma_classes_ = {}
    for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
        lemmas = [strip_lex(info['lemma']) for info in lemmas_info['lemmas']]
        done = False
        if bank is not None:
            common_lemmas = old_lemmas.intersection(set(lemmas))
            if common_lemmas:
                uniq_lemma_classes_[lemmas_cond_sig] = [info for info in lemmas_info['lemmas']
                                                        if strip_lex(info['lemma']) in common_lemmas][0]
                uniq_lemma_classes_[lemmas_cond_sig]['freq'] = lemmas_info['freq']
                done = True

        best_indexes = (-np.array([lemma2prob[lemma] for lemma in lemmas])).argsort()[:len(lemmas)]
        for best_index in best_indexes:
            best_lemma_info = lemmas_info['lemmas'][best_index]
            best_lemma_info['freq'] = lemmas_info['freq']
            lemma = best_lemma_info['lemma']
            lemma_stripped = strip_lex(lemma)
            if not ('-' in lemma and
                    len(lemmas_stripped_uniq[lemma_stripped]) > 2 and
                    all([stem['cond_t'] == '' for stem in lemmas_stripped_uniq[lemma_stripped]])):
                break
        
        if not done:
            uniq_lemma_classes_[lemmas_cond_sig] = best_lemma_info
        
        other_lemmas, i = [], 0
        while len(other_lemmas) < 5 and i < len(best_indexes):
            if lemmas[best_indexes[i]] != best_lemma_info['lemma']:
                other_lemmas.append(lemmas[best_indexes[i]])
            i += 1
        other_lemmas = ' '.join(other_lemmas)
        uniq_lemma_classes_[lemmas_cond_sig]['other_lemmas'] = other_lemmas
        
    return uniq_lemma_classes_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", default='config.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-output_dir", default='',
                        type=str, help="Path of the directory to output the lemmas to.")
    parser.add_argument("-output_name", default='',
                        type=str, help="Name of the file to output the representative lemmas to. File will be placed in a directory called conjugation/repr_lemmas/")
    parser.add_argument("-pos_type", default='', choices=['verbal', 'nominal', ''],
                        type=str, help="POS type of the lemmas.")
    parser.add_argument("-banks_dir",  default='',
                        type=str, help="Directory in which the annotation banks are.")
    parser.add_argument("-bank",  default='',
                        type=str, help="Name of the annotation bank to use.")
    parser.add_argument("-lexprob",  default='',
                        type=str, help="Custom lexical probabilities file which contains two columns (lemma, frequency).")
    parser.add_argument("-display_format", default='compact', choices=['compact', 'expanded'],
                        type=str, help="Display format of the debug info for each representative lemma.")
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

    from camel_tools.utils.charmap import CharMapper
    from camel_tools.morphology.utils import strip_lex
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import DEFAULT_NORMALIZE_MAP
    from camel_tools.utils.dediac import dediac_ar

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    output_dir = args.output_dir if args.output_dir else config_global['repr_lemmas_dir']
    conj_dir = output_dir.split('/')[0] 
    if not os.path.exists(conj_dir):
        os.mkdir(conj_dir)
        os.mkdir(output_dir)
    elif not os.path.exists(output_dir):
        os.mkdir(output_dir)

    SHEETS, _ = db_maker_utils.read_morph_specs(config, args.config_name)
    SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
        lambda row: re.sub(r'hamzated|hollow|defective', '', row['COND-S']), axis=1)
    SHEETS['lexicon']['COND-S'] = SHEETS['lexicon'].apply(
        lambda row: re.sub(r' +', ' ', row['COND-S']), axis=1)

    class_keys = config_local.get('class_keys')
    extended_lemma_keys = config_local.get('extended_lemma_keys')
    if class_keys == None:
        class_keys = ['cond_t', 'cond_s']
    if extended_lemma_keys == None:
        extended_lemma_keys = ['lemma']

    banks_dir = args.banks_dir if args.banks_dir else config_global['banks_dir']
    bank = args.bank if args.bank else config_local['bank']
    bank = AnnotationBank(bank_path=os.path.join(banks_dir, bank))

    lemma2prob = None
    if config_local.get('lexprob'):
        with open(config_local['lexprob']) as f:
            lemma2prob = dict(map(lambda x: (x[0], int(x[1])),
                                  [line.strip().split('\t') for line in f.readlines()]))
            total = sum(lemma2prob.values())
            lemma2prob = {lemma: freq / total for lemma, freq in lemma2prob.items()}

    uniq_lemma_classes = create_repr_lemmas_list(lexicon=SHEETS['lexicon'],
                                                 class_keys=class_keys,
                                                 extended_lemma_keys=extended_lemma_keys,
                                                 pos_type=args.pos_type if args.pos_type else config_local['pos_type'],
                                                 bank=bank,
                                                 info_display_format=args.display_format,
                                                 lemma2prob=lemma2prob)

    output_name = args.output_name if args.output_name else config_local['repr_lemmas']
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(uniq_lemma_classes, f)
        
