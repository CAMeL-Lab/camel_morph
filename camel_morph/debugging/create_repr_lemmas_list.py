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
from itertools import takewhile

try:
    from .. import db_maker_utils
    from ..utils import utils
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph import db_maker_utils
    from camel_morph.debugging.paradigm_debugging import AnnotationBank

configs_dir = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'configs')

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default=os.path.join(configs_dir, 'config_default.json'),
                    type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config', nargs='+',
                    type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
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
parser.add_argument("-pos", default='',
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

config = utils.get_config_file(args.config_file)
config_name = args.config_name[0] if type(args.config_name) is list else args.config_name
config_local = config['local'][config_name]
config_global = config['global']

if args.camel_tools == 'local':
    camel_tools_dir = configs_dir = os.path.join(
        '/'.join(os.path.dirname(__file__).split('/')[:-1]), 'camel_tools')
    sys.path.insert(0, camel_tools_dir)

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.utils import strip_lex
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import DEFAULT_NORMALIZE_MAP

ar2bw = CharMapper.builtin_mapper('ar2bw')
bw2ar = CharMapper.builtin_mapper('bw2ar')

nominals = ['ABBREV', 'ADJ', 'ADJ_COMP', 'ADJ_NUM', 'ADV', 'ADV_INTERROG', 'ADV_REL',
            'FORIEGN', 'INTERJ', 'NOUN', 'NOUN_NUM', 'NOUN_PROP', 'NOUN_QUANT',
            'PRON', 'PRON_DEM', 'PRON_EXCLAM', 'PRON_INTERROG',
            'PRON_REL', 'VERB_NOM', 'VERB_PSEUDO']
nominals = [n.lower() for n in nominals]

cond_t_sort_order = {'MS':0, 'MD':1, 'MP':2, 'FS':3, 'FD':4, 'FP':5}

def _sort_stems_nominals(stems):
    if (len(stems) in [2, 3] and
        set(stem['cond_t'] for stem in stems) <= {'MS', 'FS', 'MS||MD||FP'}):
        return stems.sort(key=lambda stem: len(stem['cond_t']), reverse=True)
    elif set(stem['cond_t'] for stem in stems) == {'MS', 'MD||FP'}:
        return stems.sort(key=lambda stem: -len(stem['cond_t']), reverse=True)
    else:
        stems.sort(reverse=True,
            key=lambda stem: (-100,) if stem['cond_t'] == 'MS||MD||FP' else
                (len(stem['cond_t']), -cond_t_sort_order[stem['cond_t'][:2]]))
                              

def create_repr_lemmas_list(config,
                            config_name,
                            pos,
                            lexicon=None,
                            bank=None,
                            info_display_format='compact',
                            lemma2prob=None,
                            db=None):
    SHEETS, _ = db_maker_utils.read_morph_specs(
        config, config_name, lexicon_sheet=lexicon,
        process_morph=False, lexicon_cond_f=False)
    lexicon = SHEETS['lexicon']
    lexicon = lexicon.replace('ditrans', 'trans')
    if pos:
        lexicon = lexicon[lexicon['FEAT'].str.contains(
            f'pos:{pos}\\b', regex=True)]
    
    config_local = config['local'][config_name]
    class_keys = config_local.get('class_keys')
    extended_lemma_keys = config_local.get('extended_lemma_keys')
    if class_keys == None:
        class_keys = ['cond_t', 'cond_s']
    if extended_lemma_keys == None:
        extended_lemma_keys = ['lemma']

    lemmas_uniq, lemmas_stripped_uniq = get_extended_lemmas(
        lexicon, extended_lemma_keys)
    uniq_lemma_classes = {}
    for lemma, stems in lemmas_uniq.items():
        info = {}
        if info_display_format == 'compact':
            for i, stem in enumerate(stems):
                for k in stem:
                    values = info.setdefault(k, {})
                    values.setdefault(stem[k], []).append(f'({i})')
            for k in stems[0]:
                info[k] = ''.join([
                    f"[{ct if ct else '-'}]{''.join(indexes)}"
                    for ct, indexes in sorted(info[k].items(), key=lambda x: x[0])])
        elif info_display_format == 'unique':
            for k in stems[0]:
                if k in ['cond_t', 'cond_s']:
                    info[k] = ''.join(
                        sorted([f"[{stem[k]}]" if stem[k] else '[-]' for stem in stems]))
                else:
                    info[k] = ''.join(list(set(
                        sorted([f"[{stem[k]}]" if stem[k] else '[-]' for stem in stems]))))
            info['lemma'] = info['lemma'][1:-1]
        elif info_display_format == 'expanded':
            if pos in nominals:
                _sort_stems_nominals(stems)
            for k in stems[0]:
                if k in ['lemma', 'pos']:
                    info[k] = '-'.join(
                        set([stem[k] if stem[k] else '[-]' for stem in stems]))
                else:
                    info[k] = '-'.join(
                        f'[{stem[k]}]' if stem[k] else '[-]' for stem in stems)
            
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
        
        lemmas_cond_sig = [{k: stem.get(k) for k in class_keys} for stem in stems]
        lemmas_cond_sig = tuple(
            sorted([tuple(stem.values()) for stem in lemmas_cond_sig]))
        uniq_lemma_classes.setdefault(lemmas_cond_sig, {'freq': 0, 'lemmas': []})
        uniq_lemma_classes[lemmas_cond_sig]['freq'] += 1
        uniq_lemma_classes[lemmas_cond_sig]['lemmas'].append(info)

    if lemma2prob == 'return_all':
        return uniq_lemma_classes
    
    uniq_lemma_classes = get_highest_prob_lemmas(
        pos, uniq_lemma_classes, lemmas_stripped_uniq, bank, lemma2prob, db)
    
    return uniq_lemma_classes


def get_extended_lemmas(lexicon, extended_lemma_keys):
    lemmas_uniq = {}
    lemmas_stripped_uniq = {}

    def get_info(row):
        feats = {feat.split(':')[0]: feat.split(':')[1]
                 for feat in row['FEAT'].split()}
        cond_t = ' '.join(sorted('||'.join(
            sorted([part for part in cond.split('||')],
                key=lambda x: cond_t_sort_order[x])) for cond in row['COND-T'].split()))
        cond_s = ' '.join(sorted('||'.join(sorted([part for part in cond.split('||')]))
                                 for cond in row['COND-S'].split()))
        lemma = row['LEMMA'].split(':')[1]
        info = dict(lemma=lemma,
                    form=row['FORM'],
                    cond_t=cond_t,
                    cond_s=cond_s,
                    gloss=row['GLOSS'],
                    index=row['index'],
                    line=row.get('LINE'))
        info.update(feats)
        extended_lemma = tuple([lemma] + [info[k]
                                          for k in extended_lemma_keys[1:]])
        lemmas_uniq.setdefault(extended_lemma, []).append(info)
        lemmas_stripped_uniq.setdefault(strip_lex(lemma), []).append(info)
    
    lexicon["index"] = np.arange(lexicon.shape[0])
    lexicon.apply(get_info, axis=1)
    
    return lemmas_uniq, lemmas_stripped_uniq


def get_lemma2prob(pos, db, uniq_lemma_classes, lemma2prob):
    if lemma2prob is None:
        lemma2prob = {}
        for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
            for info in lemmas_info['lemmas']:
                lemma_stripped = re.sub(r'[aiuo]', '', strip_lex(info['lemma']))
                lemma_ar = bw2ar(lemma_stripped)
                normalized_lemma_ar = DEFAULT_NORMALIZE_MAP(lemma_ar)
                matches = db.stem_hash.get(normalized_lemma_ar, [])
                db_entries = [db_entry[1] for db_entry in matches]
                entries_filtered = [e  for e in db_entries
                    if strip_lex(re.sub(r'[َُِْ]', '', e['lex'])) == lemma_ar and e['pos'] == pos]
                if len(entries_filtered) >= 1:
                    lemma2prob[lemma_stripped] = max([float(a['pos_lex_logprob']) for a in db_entries])
                else:
                    lemma2prob[lemma_stripped] = -99.0
    else:
        lemma2prob_ = {}
        for lemma, prob in lemma2prob.items():
            lemma = re.sub(r'[aiuo]', '', strip_lex(lemma))
            if lemma in lemma2prob_:
                if prob > lemma2prob_[lemma]:
                    lemma2prob_[lemma] = prob
            else:
                lemma2prob_[lemma] = prob
        lemma2prob = lemma2prob_
        for lemmas_cond_sig, lemmas_info in uniq_lemma_classes.items():
            for info in lemmas_info['lemmas']:
                lemma_stripped = re.sub(r'[aiuo]', '', strip_lex(info['lemma']))
                if lemma_stripped not in lemma2prob:
                    lemma2prob[lemma_stripped] = 0
    
    return lemma2prob

def get_highest_prob_lemmas(pos,
                            uniq_lemma_classes,
                            lemmas_stripped_uniq=None,
                            bank=None,
                            lemma2prob=None,
                            db=None):
    lemma2prob = get_lemma2prob(pos, db, uniq_lemma_classes, lemma2prob)
    
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

        best_indexes = (-np.array([lemma2prob[re.sub(r'[aiuo]', '', lemma)] for lemma in lemmas])).argsort()[:len(lemmas)]
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


def lcp(strings):
    "Longest common prefix"
    def allsame(strings_):
        return len(set(strings_)) == 1
    
    return ''.join(i[0] for i in takewhile(allsame, zip(*strings)))


if __name__ == "__main__":
    output_dir = args.output_dir if args.output_dir else os.path.join(config_global['debugging'], config_global['repr_lemmas_dir'])
    output_dir = os.path.join(output_dir, f"camel-morph-{config_local['dialect']}")
    os.makedirs(output_dir, exist_ok=True)
    
    pos_type = args.pos_type if args.pos_type else config_local['pos_type']
    if pos_type == 'verbal':
        pos = 'verb'
    elif pos_type == 'nominal':
        pos = args.pos if args.pos else config_local.get('pos')
    elif pos_type == 'other':
        pos = args.pos

    if 'exclusions' in config_local:
        with open(config_local['exclusions']) as f:
            exclusions = json.load(f)
            exclusions = exclusions[pos] if pos in exclusions else []
    else:
        exclusions = []

    banks_dir = args.banks_dir if args.banks_dir else os.path.join(
        config_global['debugging'], config_global['banks_dir'], f"camel-morph-{config_local['dialect']}")
    bank = args.bank if args.bank else (config_local['debugging']['feats'][args.feats]['bank'] if 'debugging' in config_local else None)
    if bank:
        bank = AnnotationBank(bank_path=os.path.join(banks_dir, bank))

    lemma2prob, db = None, None
    if config_local.get('lexprob'):
        if config_local['lexprob'] == 'return_all':
            lemma2prob = 'return_all'
        else:
            with open(config_local['lexprob']) as f:
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
                total = sum(pos2lemma2prob[pos].values())
                lemma2prob = {lemma: freq / total for lemma, freq in pos2lemma2prob[pos].items()}
    elif args.db:
        db = MorphologyDB(args.db)
    else:
        db = MorphologyDB.builtin_db()

    uniq_lemma_classes = create_repr_lemmas_list(config=config,
                                                 config_name=config_name,
                                                 pos=pos,
                                                 bank=bank,
                                                 info_display_format=args.display_format,
                                                 lemma2prob=lemma2prob,
                                                 db=db)
    excluded_classes = set()
    if type(args.config_name) is list and len(args.config_name) > 1:
        for config_name_ in args.config_name[1:]:
            with open(os.path.join(output_dir, f'repr_lemmas_{config_name_}.pkl'), 'rb') as f:
                excluded_classes.update(pickle.load(f).keys())
    uniq_lemma_classes = {k: v for k, v in uniq_lemma_classes.items() if k not in excluded_classes}

    output_name = args.output_name if args.output_name else f'repr_lemmas_{config_name}.pkl'
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'wb') as f:
        pickle.dump(uniq_lemma_classes, f)
        
