import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import pickle
from collections import Counter

import pandas as pd

try:
    from .. import db_maker_utils
except:
    file_path = os.path.abspath(__file__).split('/')
    package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
    sys.path.insert(0, package_path)
    from camel_morph import db_maker_utils
    from camel_morph.debugging.generate_docs_tables import _get_structured_lexicon_classes
    from camel_morph.eval import evaluate_camel_morph

POS = {"ABBREV", "ADJ", "ADJ_COMP", "ADJ_NUM", "ADV", "ADV_INTERROG", "ADV_REL", "CONJ", "CONJ_SUB", "DIGIT", "FORIEGN", "INTERJ", "NOUN", "NOUN_NUM", "NOUN_PROP", "NOUN_QUANT", "PART", "PART_CONNECT", "PART_DET", "PART_EMPHATIC",
       "PART_FOCUS", "PART_FUT", "PART_INTERROG", "PART_NEG", "PART_PROG", "PART_RC", "PART_RESTRICT", "PART_VERB", "PART_VOC", "PREP", "PRON", "PRON_DEM", "PRON_EXCLAM", "PRON_INTERROG", "PRON_REL", "PUNC", "VERB", "VERB_NOM", "VERB_PSEUDO",
       "NOUN_ACT", "NOUN_PASS", "ADJ/NOUN", "UNKNOWN"}
# POS_NOM = {"ADJ", "ADJ_COMP", "ADJ_NUM", "ADV", "NOUN", "NOUN_NUM", "NOUN_QUANT", "ADV_REL", 'VERB_NOM'}
POS_NOM = {"ADJ", "ADJ_COMP", "ADV", "NOUN"}

essential_keys = evaluate_camel_morph.essential_keys
gen_feat_keys = [k for k in essential_keys if k not in ['source', 'lex', 'diac', 'stem_seg', 'gen', 'num', 'vox', 'rat']]

lex_index = essential_keys.index('lex')
stem_seg_index = essential_keys.index('stem_seg')
prc1_index = essential_keys.index('prc1')
prc2_index = essential_keys.index('prc2')

def _preprocess_magold_data(gold_data):
    # List of sentences
    gold_data = gold_data.split('--------------\nSENTENCE BREAK\n--------------\n')[:-1]
    # List of words
    gold_data = sum([sent.split('\n--------------\n')[1:] for sent in tqdm(gold_data, desc='List of words')], [])
    # List of word analyses
    gold_data = [line for word in tqdm(gold_data, desc='List of word analyses')
                      for line in word.strip().split('\n') if line.startswith('*')]
    # Unique analyses
    gold_data = set([tuple(x.split(' ')[1:]) for x in gold_data])
    gold_data = [{f.split(':')[0]: f.split(':')[1] for f in g} for g in gold_data]

    return gold_data

def _preprocess_magold_data_nominals(gold_data):
    gold_data = gold_data.split(
        '--------------\nSENTENCE BREAK\n--------------\n')[:-1]
    gold_data_ = {}
    word_start, sentence_start = len(";;WORD "), len(";;; SENTENCE ")
    for infos in tqdm(gold_data):
        infos = infos.split('\n--------------\n')
        sentence, _, word, analysis = infos[0].strip().split('\n')
        infos = ['\n'.join([word, analysis])] + infos[1:]
        for info in infos:
            word, analysis = info.strip().split('\n')
            word = word[word_start:]
            analysis_ = {}
            drop = True
            for field in analysis.split()[1:]:
                field = field.split(':')
                analysis_[field[0]] = ''.join(field[1:])
                analysis_[field[0]] = re.sub(r'^no$', r'na', analysis_[field[0]])
                if field[0] == 'pos' and field[1].upper() not in POS_NOM:
                    break
            else:
                drop = False
            
            if not drop:
                word = {
                    'info': {
                        'sentence': sentence[sentence_start:],
                        'word': word,
                        'magold': [' #  #  #  # ', '', '']
                    },
                    'analysis': analysis_
                }
                gold_data_.setdefault('nominal', []).append(word)

    return gold_data_

def gumar_inspect():
    morphemes_nom = Counter()
    data_xml = ET.parse('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/Arabic NLP/annotated-gumar-corpus/TRAIN_annotated_Gumar_corpus.xml').getroot()
    for idx, sentence in tqdm(enumerate(data_xml)):
        for info in sentence[2]:
            token = info.attrib
            analysis = token['baseword_pos'].split(':')
            if len(analysis) == 2:
                pos_baseword, feat = analysis
            else:
                pos_baseword, feat = analysis[0], ''
            if pos_baseword in POS_NOM:
                for feat in ['enc0', 'enc1', 'enc2', 'enc3', 'prc0', 'prc1', 'prc2', 'prc3']:
                    form = token.get(f'{feat}_form')
                    pos = token.get(f'{feat}_pos')
                    morphemes_nom.update([(form, pos, feat)])

    with open('sandbox_files/glf_nom_clitics.tsv', 'w') as f:
        for (form, pos, feat), freq in morphemes_nom.items():
            print(form, pos, feat, freq, sep='\t', file=f)
    pass


def get_backoff_stems(mode='print'):
    SHEETS, _ = db_maker_utils.read_morph_specs(config_glf, config_name_glf_docs, process_morph=False, lexicon_cond_f=False)
    lexicon = SHEETS['lexicon']
    lexicon['LEMMA'] = lexicon.apply(lambda row: re.sub('lex:', '', row['LEMMA']), axis=1)
    cond_s2cond_t2feats2rows = _get_structured_lexicon_classes(lexicon)
    stem_classes = {}
    for cond_s, cond_t2feats2rows in cond_s2cond_t2feats2rows.items():
        for cond_t, feats2rows in cond_t2feats2rows.items():
            for feats, rows in feats2rows.items():
                row_ = rows[0]
                row_['COND-S'], row_['COND-T'] = cond_s, cond_t
                row_['FEAT'] = ' '.join(f"{f}:{feats[i] if feats[i] else '-'}" for i, f in enumerate(['gen', 'num', 'pos']))
                row_['BW'] = re.search(r'pos:(\S+)', row_['FEAT']).group(1)
                row_['ROOT'] = 'PLACEHOLDER'
                row_['PATTERN'] = 'PLACEHOLDER'
                row_['DEFINE'] = 'BACKOFF'
                row_['CLASS'] = '[STEM]'
                row_['GLOSS'] = 'no'
                row_['FREQ'] = len(rows)
                stem_classes[(cond_s, cond_t, feats)] = row_
    
    if mode == 'print':
        with open('sandbox_files/glf_backoff_lexicon_pilot.tsv', 'w') as f:
            header = ['ROOT', 'PATTERN', 'DEFINE', 'CLASS', 'LEMMA', 'FORM', 'BW', 'GLOSS', 'FREQ', 'COND-S', 'COND-T', 'FEAT', 'STATUS', 'COMMENTS']
            print(*header, sep='\t', file=f)
            for i, row in enumerate(sorted(stem_classes.values(), key=lambda row: row['FREQ'], reverse=True), start=1):
                row['FORM'], row['LEMMA'] = f'stem{i}', f'lemma{i}'
                print(*[row.get(h, '') for h in header], sep='\t', file=f)
    elif mode == 'return':
        return stem_classes
    else:
        raise NotImplementedError


def test_backoff(ma_gold):
    for analysis in ma_gold:
        word = analysis['diac']
        analyses = analyzer.analyze(word)
    pass

def disambig_experiment():
    unfactored = BERTUnfactoredDisambiguator.pretrained('glf')

    gumar_dir = '/Users/chriscay/Downloads/Gumar'
    for file_name in tqdm(os.listdir(gumar_dir)):
        with open(os.path.join(gumar_dir, file_name)) as f:
            for line in tqdm(f.readlines()):
                sentence = []
                for token in line.strip().split():
                    token = re.sub(f"^[^{''.join(AR_LETTERS_CHARSET)}]+|[^{''.join(AR_LETTERS_CHARSET)}]+$", '', token)
                    token_split = token.split()
                    for token in token_split:
                        token = re.sub(r'(.+?)\1+', r'\1', token)
                        sentence.append(token)
                disambig = unfactored.disambiguate(sentence)
            pass

    diacritized = [d.analyses[0].analysis['diac'] for d in disambig]
    print(' '.join(diacritized))


def generate_gumar_pickle():
    gumar_dir = '/Users/chriscay/Downloads/Gumar'
    gumar = Counter()
    for file_name in tqdm(os.listdir(gumar_dir)):
        with open(os.path.join(gumar_dir, file_name)) as f:
            for line in f.readlines():
                for token in line.strip().split():
                    token = re.sub(f"^[^{''.join(AR_LETTERS_CHARSET)}]+|[^{''.join(AR_LETTERS_CHARSET)}]+$", '', token)
                    if token and all(c in AR_LETTERS_CHARSET for c in token):
                        token = re.sub(r'(.+?)\1+', r'\1', token)
                        token = DEFAULT_NORMALIZE_MAP.map_string(token)
                        gumar.update([token])
    
    with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/gumar.pkl', 'wb') as f:
        pickle.dump(gumar, f)

    sys.exit()


def process_analyses(examples):
    source_index = essential_keys.index('source')
    processed = []
    for example in examples:
        e_gold = example['gold']
        analyses_pred, index2similarity = [], {}
        for analysis_index, analysis in enumerate(example['pred']):
            analysis_ = []
            for index, f in enumerate(analysis):
                if f == e_gold[index]:
                    analysis_.append(f)
                    index2similarity.setdefault(analysis_index, 0)
                    index2similarity[analysis_index] += (
                        1.01 if analysis[source_index] == 'main' else 1)
                else:
                    analysis_.append(f'[{f}]')
            analyses_pred.append(tuple(analysis_))
        sorted_indexes = sorted(
            index2similarity.items(), key=lambda x: x[1], reverse=True)
        analyses_pred = [analyses_pred[analysis_index]
                            for analysis_index, _ in sorted_indexes]
        analyses_pred = [analysis for analysis in analyses_pred
                            if all(analysis[i] == e_gold[i] for i, k in enumerate(essential_keys)
                                if k not in ['source', 'lex', 'diac', 'stem_seg'])]
        processed.append({'word': example['word']['info']['word'], 'gold': e_gold, 'pred': analyses_pred})
    
    return processed


def reverse_processing(analysis):
    analysis = list(analysis)
    analysis[prc2_index] = re.sub(r'^([wf])', r'\1a', analysis[prc2_index])
    analysis[prc1_index] = re.sub(r'^([bl])', r'\1i', analysis[prc1_index])
    return tuple(analysis)


def _strip_brackets(info):
    if info[0] == '[' and info[-1] == ']':
        info = info[1:-1]
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file_egy", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name_egy", default='default_config', nargs='+',
                        type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
    parser.add_argument("-config_file_glf", default='config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name_glf", default='default_config', nargs='+',
                        type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
    parser.add_argument("-config_name_glf_docs", default='default_config', nargs='+',
                        type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
    parser.add_argument("-output_dir", default='eval_files',
                        type=str, help="Path of the directory to output evaluation results.")
    parser.add_argument("-db", default='',
                        type=str, help="Name of the DB file which will be used for the retrieval of lexprob.")
    parser.add_argument("-db_dir", default='',
                        type=str, help="Path of the directory to load the DB from.")
    parser.add_argument("-eval_mode", required=True,
                        choices=['recall_glf_magold_raw_no_lex'],
                        type=str, help="What evaluation to perform.")
    parser.add_argument("-n", default=1000,
                        type=int, help="Number of verbs to input to the two compared systems.")
    parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    with open(args.config_file_egy) as f:
        config_egy = json.load(f)
    config_name_egy = args.config_name_egy[0]
    config_local_egy = config_egy['local'][config_name_egy]
    config_global_egy = config_egy['global']

    with open(args.config_file_glf) as f:
        config_glf = json.load(f)
    config_name_glf = args.config_name_glf[0]
    config_local_glf = config_glf['local'][config_name_glf]
    config_global_glf = config_glf['global']

    config_name_glf_docs = args.config_name_glf_docs[0]

    if args.camel_tools == 'local':
        camel_tools_dir = config_global_egy['camel_tools']
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.morphology.generator import Generator
    from camel_tools.utils.charmap import CharMapper
    from camel_tools.tokenizers.word import simple_word_tokenize
    from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
    from camel_tools.utils.charsets import AR_LETTERS_CHARSET
    from camel_tools.utils.dediac import dediac_ar

    DEFAULT_NORMALIZE_MAP = CharMapper({
        u'\u0625': u'\u0627',
        u'\u0623': u'\u0627',
        u'\u0622': u'\u0627',
        u'\u0671': u'\u0627',
        u'\u0649': u'\u064a',
        u'\u0629': u'\u0647',
        u'\u0640': u''
    })

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    # gumar_inspect()
    # get_backoff_stems()
    # test_backoff(ma_gold)

    db_name = config_local_glf['db']
    db_dir = config_global_glf['db_dir']
    db_dir = os.path.join(db_dir, f"camel-morph-{config_local_glf['dialect']}")
    db = MorphologyDB(os.path.join(db_dir, db_name), flags='a')
    db_gen = MorphologyDB(os.path.join(db_dir, db_name), flags='gd')
    generator = Generator(db_gen)
    analyzer = Analyzer(db, backoff='NOAN-ONLY_ALL')

    # generate_gumar_pickle()

    stem_classes = get_backoff_stems('return')
    header = ['ROOT', 'PATTERN', 'DEFINE', 'CLASS', 'LEMMA', 'FORM', 'BW', 'GLOSS', 'FREQ', 'COND-S', 'COND-T', 'FEAT', 'STATUS', 'COMMENTS']
    for i, row in enumerate(sorted(stem_classes.values(), key=lambda row: row['FREQ'], reverse=True), start=1):
        row['FORM'], row['LEMMA'] = f'stem{i}', f'lemma{i}'
    id2info = {}
    for (cond_s, cond_t, feat), info in stem_classes.items():
        id2info[info['LEMMA']] = {
            'cond_s': cond_s, 'cond_t': cond_t, 'freq': info['FREQ']}

    with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/gumar_processed_analyses.pkl', 'rb') as f:
        processed = pickle.load(f)
    with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/gumar_form_counts.pkl', 'rb') as f:
        form_counts = pickle.load(f)

    lemma2forms = {}
    for i in range(len(processed)):
        for stem, analyses in form_counts[i].values():
            lemma2forms.setdefault(processed[i]['gold'][lex_index], set()).add(stem)

    disambig_experiment()
    with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/gumar.pkl', 'rb') as f:
        gumar = pickle.load(f)
    
    with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/GA_80_nvls_train.utf8.magold') as f:
        ma_gold = f.read()
    data = _preprocess_magold_data_nominals(ma_gold)
    # disambig_experiment(data)
    output_path = os.path.join(args.output_dir, args.eval_mode)
    examples = evaluate_camel_morph.evaluate_recall(data, args.n, args.eval_mode, output_path, 'nominal',
                                         analyzer, None, ar2bw, bw2ar, best_analysis=False)
    processed = process_analyses(examples)
    
    form_counts = []
    for example in tqdm(processed):
        e_gold = reverse_processing(example['gold'])
        stemid2form2count = {}
        for pred in example['pred']:
            pred = reverse_processing(pred)
            lemma = _strip_brackets(pred[lex_index])
            feats = {k: e_gold[essential_keys.index(k)] for k in gen_feat_keys}
            for form_gen_num in id2info[lemma]['cond_t'].split('||'):
                form_gen, form_num = list(form_gen_num.lower())
                feats['form_gen'], feats['form_num'] = form_gen, form_num
                analyses, messages = generator.generate(lemma, feats, debug=True)
                stem_id = 'stem' + re.search(r'(\d+)$', lemma).group()
                stem = _strip_brackets(pred[stem_seg_index])
                stemid2form2count.setdefault(stem_id, ['', {}])
                stemid2form2count[stem_id][0] = stem
                stemid2form2count[stem_id][1].setdefault(form_gen_num, 0)
                if analyses:
                    analysis = analyses[0][0]
                    diac = re.sub(stem_id, stem, analysis['diac'])
                    diac = dediac_ar(DEFAULT_NORMALIZE_MAP.map_string(diac))
                    stemid2form2count[stem_id][1][form_gen_num] += gumar[diac]
        form_counts.append(stemid2form2count)
        pass                        
    pass