# MIT License
#
# Copyright 2022 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import json
import os
import pickle
import numpy as np
import re
import sys

try:
    from .. import db_maker_utils
    from ..debugging.paradigm_debugging import AnnotationBank
    from ..utils import utils
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph import db_maker_utils
    from camel_morph.debugging.paradigm_debugging import AnnotationBank
    from camel_morph.utils import utils

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='config_default.json',
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config',
                    type=str, help="Name of the configuration to load from the config file.")
parser.add_argument("-output_dir", default='',
                    type=str, help="Path of the directory to output the lemmas to.")
parser.add_argument("-output_name", default='',
                    type=str, help="Name of the file to output the representative lemmas to. File will be placed in a directory called conjugation/repr_lemmas/")
parser.add_argument("-pos_type", default='', choices=['verbal', 'nominal', ''],
                    type=str, help="POS type of the lemmas.")
parser.add_argument("-feats", default='',
                    type=str, help="Features to generate the conjugation tables for.")
parser.add_argument("-db", default='',
                    type=str, help="Path of DB to use to get the lexical/POS probabilities.")
parser.add_argument("-pos", default=[], nargs='+',
                    type=str, help="POS of the lemmas.")
parser.add_argument("-banks_dir",  default='',
                    type=str, help="Directory in which the annotation banks are.")
parser.add_argument("-bank",  default='',
                    type=str, help="Name of the annotation bank to use.")
parser.add_argument("-lexprob",  default='',
                    type=str, help="Custom lexical probabilities file which contains two columns (lemma, frequency).")
parser.add_argument("-display_format", default='compact', choices=['compact', 'unique', 'expanded'],
                    type=str, help="Display format of the debug info for each representative lemma.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                    type=str, help="Path of the directory containing the camel_tools modules.")
args, _ = parser.parse_known_args([] if "__file__" not in globals() else None)

if args.camel_tools == 'local':
    camel_tools_dir = 'camel_morph/camel_tools'
    sys.path.insert(0, camel_tools_dir)

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex
from camel_tools.morphology.database import MorphologyDB

ar2bw = CharMapper.builtin_mapper('ar2bw')
bw2ar = CharMapper.builtin_mapper('bw2ar')

POS_NOMINAL = utils.POS_NOMINAL

cond_t_sort_order = {'MS':0, 'MD':1, 'MP':2, 'FS':3, 'FD':4, 'FP':5}
inspect_cases = {}

def _sort_stems_nominals(stems):
    global inspect_cases
    stems_cond_t = set(stem['cond_t'] for stem in stems)
    
    stems_ = None
    if {'MS||MD', 'MS||MD||FP'} <= stems_cond_t:
        case_ = '2'
        sort_order, broken_plural = [], False
        for stem in stems:
            if stem['cond_t'] == 'MS||MD':
                sort_order.append(0)
            elif stem['cond_t'] == 'MS' and not broken_plural:
                broken_plural = True
                sort_order.append(1)
            elif stem['cond_t'] == 'MS||MD||FP':
                sort_order.append(2 if 'MS' in stems_cond_t else 1)
            else:
                extras_index = 3 if 'MS' in stems_cond_t else 2
                sort_order.append(max(extras_index, len(sort_order)))
        stems_ = [None] * len(stems)
        for i, index in enumerate(sort_order):
            stems_[index] = stems[i]
    else:
        if {'MP', 'MS||MD'} <= stems_cond_t:
            sort_fn = lambda stem: (
                len(stem['cond_t']),
                1 if stem['cond_t'] == 'MP' else -cond_t_sort_order[stem['cond_t'][:2]])
            case_ = '1'
        elif stems_cond_t == {'MS', 'MD||FP'}:
            sort_fn = lambda stem: -len(stem['cond_t'])
            case_ = '3'
        else:
            sort_fn = lambda stem: (
                len(stem['cond_t']), -cond_t_sort_order[stem['cond_t'][:2]])
            case_ = '4'

    if stems_ is None:
        stems_ = sorted(stems, reverse=True, key=sort_fn)
    inspect_cases.setdefault(case_, set()).add(
        tuple(stem['cond_t'] for stem in stems_))
    
    return stems_
                              

def create_repr_lemmas_list(config:utils.Config,
                            lexicon=None,
                            lemma2prob=None,
                            lexicon_is_processed=False):
    POS, info_display_format, bank, lexprob_db, exclusions, lemma2prob = setup(
        config, lemma2prob)

    if not lexicon_is_processed:
        SHEETS, _ = db_maker_utils.read_morph_specs(
            config, lexicon_df=lexicon,
            process_morph=False, lexicon_cond_f=False)
        lexicon = SHEETS['lexicon']

    lexicon = lexicon.replace('ditrans', 'trans')
    if POS:
        lexicon = lexicon[lexicon['FEAT'].str.extract(r'pos:(\S+)').isin(POS)[0]]
    
    if config.class_keys is None:
        class_keys = ['cond_t', 'cond_s']
    else:
        class_keys = config.class_keys
    if config.extended_lemma_keys is None:
        extended_lemma_keys = ['lemma']
    else:
        extended_lemma_keys = config.extended_lemma_keys

    lemmas_uniq, lemmas_stripped_uniq = get_extended_lemmas(
        lexicon, extended_lemma_keys)
    uniq_lemma_classes = {}
    #FIXME: should not be looping over the unique extended lemmas here.
    # Currently, extending lemmas WILL have an effect on the number of 
    # classes (contrary to what is stated in the method documentation
    # of extended lemmas). To avoid this, we should be looping on a uniqued
    # list of entries based on `class_keys`.
    for lemma, stems in lemmas_uniq.items():
        info = {}
        info_union_feats = {f for stem in stems for f in stem}
        if info_display_format == 'compact':
            for i, stem in enumerate(stems):
                for k in stem:
                    values = info.setdefault(k, {})
                    values.setdefault(stem[k], []).append(f'({i})')
            for k in info_union_feats:
                info[k] = ''.join([
                    f"[{ct if ct else '-'}]{''.join(indexes)}"
                    for ct, indexes in sorted(info[k].items(), key=lambda x: x[0])])
        elif info_display_format == 'unique':
            for k in info_union_feats:
                if k in ['cond_t', 'cond_s']:
                    info[k] = ''.join(
                        sorted([f"[{stem[k]}]" if stem.get(k) else '[-]' for stem in stems]))
                else:
                    info[k] = ''.join(list(set(
                        sorted([f"[{stem[k]}]" if stem.get(k) else '[-]' for stem in stems]))))
            info['lemma'] = info['lemma'][1:-1]
        elif info_display_format == 'expanded':
            if stems[0]['pos'] in POS_NOMINAL:
                stems = _sort_stems_nominals(stems)
            for k in info_union_feats:
                if k in ['lemma', 'pos']:
                    info[k] = '-'.join(
                        set([stem[k] if stem.get(k) else '[-]' for stem in stems]))
                else:
                    info[k] = '-'.join(
                        f'[{stem[k]}]' if stem.get(k) else '[-]' for stem in stems)
            
            info['stem_count'] = len(stems)
            info['lemma_ar'] = bw2ar(info['lemma'])
            info['form_ar'] = bw2ar(info['form'])

            lemma_p = ' lemma:#p' if 'p' in info['lemma'] else ''
            form2index = {}
            index = 0
            for stem in stems:
                form = stem['form']
                if form not in form2index:
                    form2index[form] = str(index)
                    index += 1
                stem['meta_info'] = (form2index[form], lemma_p)
            info['meta_info'] = 'stem:' + '-'.join(
                stem['meta_info'][0] for stem in stems)
            info['meta_info'] += lemma_p
        
        lemmas_cond_sig = [{k: stem.get(k, '') for k in class_keys} for stem in stems]
        lemmas_cond_sig = tuple(
            sorted([tuple(stem.values()) for stem in lemmas_cond_sig]))
        uniq_lemma_classes.setdefault(lemmas_cond_sig, {'freq': 0, 'lemmas': []})
        uniq_lemma_classes[lemmas_cond_sig]['freq'] += 1
        uniq_lemma_classes[lemmas_cond_sig]['lemmas'].append(info)

    if lemma2prob == 'return_all':
        return uniq_lemma_classes
    
    uniq_lemma_classes = get_highest_prob_lemmas(
        POS, uniq_lemma_classes, lemmas_stripped_uniq, bank,
        lemma2prob, lexprob_db, exclusions)
    
    return uniq_lemma_classes


def get_extended_lemmas(lexicon, extended_lemma_keys):
    """Method that uniques lemmas, not only based on the lemma, but
    also with some extensions that would be seen as part of the lemma.
    For example, if we want to unique lemmas, but also take POS into
    account (e.g., كاتِب as noun and كاتِب as noun_act), and not have them
    confused due to ambiguity, the concept of extended lemma is used. These extensions
    are specified in the form of features which are appended to the actual lemma
    and which will thus be considered in the uniquing process.
    This does not affect the the number or nature of classes which the lemmas
    will be divided into (`classe_keys`), but will affect how the lemmas will
    be distributed into those."""
    lemmas_uniq = {}
    lemmas_stripped_uniq = {}

    def get_info(row):
        feats = {feat.split(':')[0]: feat.split(':')[1]
                 for feat in row['FEAT'].split()}
        cond_t = ' '.join(sorted('||'.join(
            sorted([part for part in cond.split('||')],
                key=lambda x: cond_t_sort_order.get(x, 0))) for cond in row['COND-T'].split()))
        cond_s = ' '.join(sorted('||'.join(sorted([part for part in cond.split('||')]))
                                 for cond in row['COND-S'].split()))
        lemma = row['LEMMA'].split(':')[1]
        info = dict(lemma=lemma,
                    form=row['FORM'],
                    cond_t=cond_t,
                    morph_class=row['CLASS'],
                    cond_s=cond_s,
                    gloss=row['GLOSS'],
                    bw=row['BW'],
                    index=row['index'],
                    line=row.get('LINE'))
        info.update(feats)
        extended_lemma = tuple([lemma] + [info.get(k, '')
                                          for k in extended_lemma_keys[1:]])
        lemmas_uniq.setdefault(extended_lemma, []).append(info)
        lemmas_stripped_uniq.setdefault(strip_lex(lemma), []).append(info)
    
    lexicon["index"] = np.arange(lexicon.shape[0])
    lexicon.apply(get_info, axis=1)
    
    return lemmas_uniq, lemmas_stripped_uniq


def get_lemma2prob(POS, db, uniq_lemma_classes, lemma2prob):
    if lemma2prob is None:
        lemma2prob = {}
        for lemmas_info in uniq_lemma_classes.values():
            for info in lemmas_info['lemmas']:
                lemma_stripped = strip_lex(info['lemma'])
                lemma_ar = bw2ar(lemma_stripped)
                analyses = db.lemma_hash.get(lemma_ar, [])
                analyses_filtered = [a  for a in analyses
                    if strip_lex(a['lex']) == lemma_ar and a['pos'] in POS]
                if len(analyses_filtered) >= 1:
                    lemma2prob[lemma_stripped] = max(
                        [float(a['pos_lex_logprob']) for a in analyses_filtered])
                else:
                    lemma2prob[lemma_stripped] = -99.0
    else:
        lemma2prob_ = {}
        for lemma, prob in lemma2prob.items():
            lemma = strip_lex(lemma)
            if lemma in lemma2prob_:
                if prob > lemma2prob_[lemma]:
                    lemma2prob_[lemma] = prob
            else:
                lemma2prob_[lemma] = prob
        lemma2prob = lemma2prob_
        for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
            for info in lemmas_info['lemmas']:
                lemma_stripped = strip_lex(info['lemma'])
                if lemma_stripped not in lemma2prob:
                    lemma2prob[lemma_stripped] = 0
    
    return lemma2prob

def get_highest_prob_lemmas(POS,
                            uniq_lemma_classes,
                            lemmas_stripped_uniq=None,
                            bank=None,
                            lemma2prob=None,
                            db=None,
                            exclusions=[]):
    lemma2prob = get_lemma2prob(POS, db, uniq_lemma_classes, lemma2prob)
    
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
            if (not ('-' in lemma and
                len(lemmas_stripped_uniq[lemma_stripped]) > 2 and
                all([stem['cond_t'] == '' for stem in lemmas_stripped_uniq[lemma_stripped]]))
                or lemmas_stripped_uniq is None) \
                and lemma_stripped not in exclusions:
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
        uniq_lemma_classes_[lemmas_cond_sig]['other_lemmas_ar'] = bw2ar(other_lemmas)
        
    return uniq_lemma_classes_


def setup(config:utils.Config, lemma2prob):
    pos_type = args.pos_type if args.pos_type else config.pos_type
    if pos_type == 'verbal':
        pos = ['verb']
    elif pos_type == 'nominal':
        pos = args.pos if args.pos else config.pos
        pos = pos if pos is not None else POS_NOMINAL
    elif pos_type == 'other':
        pos = args.pos
    else:
        pos = config.pos
    POS = pos if type(pos) is list else [pos]

    if config.exclusions is not None:
        with open(config.exclusions) as f:
            exclusions = json.load(f)
            exclusions = sum([exclusions.get(pos, []) for pos in POS], [])
    else:
        exclusions = []

    bank = None
    if config.debugging.feats is not None:
        banks_dir = args.banks_dir if args.banks_dir else config.get_banks_dir_path()
        bank = args.bank if args.bank else config.debugging.debugging_feats.bank
    lexprob_db = args.db if args.db else config.debugging.lexprob_db
    if bank:
        bank = AnnotationBank(bank_path=os.path.join(banks_dir, bank))

    if config.logprob:
        if config.logprob == 'return_all':
            lemma2prob = 'return_all'
        else:
            with open(config.logprob) as f:
                freq_list_raw = f.readlines()
                if len(freq_list_raw[0].split('\t')) == 2:
                    pos2lemma2prob = dict(map(lambda x: (x[0], int(x[1])),
                                        [line.strip().split('\t') for line in freq_list_raw]))
                    pos2lemma2prob = {'verb': pos2lemma2prob}
                elif len(freq_list_raw[0].split('\t')) == 3:
                    pos2lemma2prob = {}
                    for line in freq_list_raw:
                        line = line.strip().split('\t')
                        lemmas = pos2lemma2prob.setdefault(line[1], {})
                        lemmas[line[0]] = int(line[2])
                else:
                    raise NotImplementedError
                pos2lemma2prob[''] = {'': 1}
                total = sum(pos2lemma2prob[pos].values() for pos in POS)
                lemma2prob = {lemma: freq / total
                              for pos in POS
                              for lemma, freq in pos2lemma2prob[pos].items()}
    elif lexprob_db is not None:
        lexprob_db = MorphologyDB(lexprob_db, flags='g')
    else:
        lexprob_db = MorphologyDB.builtin_db(flags='g')

    display_format = (config.debugging.display_format 
                      if config.debugging.display_format is not None
                      else args.display_format)

    return POS, display_format, bank, lexprob_db, exclusions, lemma2prob


if __name__ == "__main__":
    config = utils.Config(args.config_file, args.config_name)
    
    output_dir = args.output_dir if args.output_dir else config.get_repr_lemmas_dir_path()
    os.makedirs(output_dir, exist_ok=True)

    uniq_lemma_classes = create_repr_lemmas_list(config=config)

    output_name = args.output_name if args.output_name else config.get_repr_lemmas_file_name()
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(uniq_lemma_classes, f)
        
