import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import pickle
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-config_file", default='camel_morph/configs/config_default.json',
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
parser.add_argument("-config_name", default='default_config', nargs='+',
                    type=str, help="Name of the configuration to load from the config file. If more than one is added, then lemma classes from those will not be counted in the current list.")
parser.add_argument("-camel_tools", default='local', choices=['local', 'official'],
                        type=str, help="Path of the directory containing the camel_tools modules.")
args, _ = parser.parse_known_args([] if "__file__" not in globals() else None)

with open(args.config_file) as f:
    config = json.load(f)
    config_name = args.config_name
    config_local = config['local'][config_name]
    config_global = config['global']

if args.camel_tools == 'local':
    camel_tools_dir = config_global['camel_tools']
    sys.path.insert(0, camel_tools_dir)

from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.charsets import AR_LETTERS_CHARSET

POS_NOM = {'ADJ', 'ADJ_COMP', 'ADV', 'NOUN'}
# POS_NOM = {'ADJ', 'ADJ_COMP', 'ADJ_NUM', 'ADV', 'NOUN',
#            'NOUN_NUM', 'NOUN_QUANT', 'ADV_REL', 'VERB_NOM'}

FEATS_INFLECT = ['pos', 'asp', 'mod', 'vox', 'per', 'num', 'gen',
                 'form_num', 'form_gen', 'prc0', 'prc1', 'prc2',
                 'prc3', 'enc0', 'enc1', 'enc2']

DEFAULT_NORMALIZE_MAP = CharMapper({
        u'\u0625': u'\u0627',
        u'\u0623': u'\u0627',
        u'\u0622': u'\u0627',
        u'\u0671': u'\u0627',
        u'\u0649': u'\u064a',
        u'\u0629': u'\u0647',
        u'\u0640': u''
    })


def generate_gumar_pickle():
    NON_ARABIC_CHAR_RE = re.compile(
        f"^[^{''.join(AR_LETTERS_CHARSET)}]+|[^{''.join(AR_LETTERS_CHARSET)}]+$")
    gumar = Counter()
    for file_name in tqdm(os.listdir(args.gumar_dir)):
        with open(os.path.join(args.gumar_dir, file_name)) as f:
            for line in f.readlines():
                for token in line.strip().split():
                    token = NON_ARABIC_CHAR_RE.sub('', token)
                    if token and all(c in AR_LETTERS_CHARSET for c in token):
                        token = re.sub(r'(.+?)\1+', r'\1', token)
                        token = DEFAULT_NORMALIZE_MAP.map_string(token)
                        gumar.update([token])
    
    with open(args.gumar_pkl, 'wb') as f:
        pickle.dump(gumar, f)


def gumar_inspect(output_path):
    morphemes_nom = Counter()
    data_xml = ET.parse((args.gumar_inspect_path)).getroot()
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

    with open(output_path, 'w') as f:
        for (form, pos, feat), freq in morphemes_nom.items():
            print(form, pos, feat, freq, sep='\t', file=f)