#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Salam Khalifa"
__email__ = "sk6184@nyu.edu"

import re
from collections import OrderedDict

import pandas as pd
import gspread

from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.charmap import CharMapper

from camel_morph.debugging.paradigm_debugging import automatic_bank_annotation
from camel_morph.utils.utils import Config
from camel_morph.debugging.upload_sheets import upload_sheet

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')


def caphi_DBPrefix(bw_word):
    # print(bw_word)
    if bw_word == 'Al':
        Phon_word = '2 a l'
    else:
        Phon_word = re.sub(r'o', ' ', bw_word)  # NOTE: Christian addition
        Phon_word = re.sub(r'[><&\}\']', ' 2 ', Phon_word)  # turn all hamzas to glottal stop /2/
        Phon_word = re.sub(r'\|', ' 2 aa ', Phon_word)  # turn all hamzas to glottal stop /2/
        Phon_word = re.sub(r'([aiu])A', r' \1 ', Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel
        Phon_word = re.sub(r'^A', '2', Phon_word)  # replace initial 'A' with glottal stop /2/ NOT IN SUFF

        Phon_word = re.sub(r"A", r' aa ', Phon_word)  # replace 'A' with long vowel 'aa'
        Phon_word = re.sub(r'E', ' 3 ', Phon_word)
        Phon_word = re.sub(r'H', ' 7 ', Phon_word)
        Phon_word = re.sub(r'\*', ' dh ', Phon_word)
        Phon_word = re.sub(r'Z', ' dh. ', Phon_word)
        Phon_word = re.sub(r'\$', ' sh ', Phon_word)
        Phon_word = re.sub(r'g', ' gh ', Phon_word)
        Phon_word = re.sub(r'v', ' th ', Phon_word)
        Phon_word = re.sub(r'S', ' s. ', Phon_word)
        Phon_word = re.sub(r'T', ' t. ', Phon_word)
        Phon_word = re.sub(r'D', ' d. ', Phon_word)
        Phon_word = re.sub(r'x', ' kh ', Phon_word)
        Phon_word = Phon_word.replace('#', '-') # NOTE: Christian addition

        Phon_word = re.sub(r"(d\.|t\.|s\.|sh|kh|gh|dh[.]*|th|tsh|[^iuaoe-])", r' \1 ', Phon_word)

        Phon_word = re.sub(' +', ' ', Phon_word)

    Phon_word = re.sub(' ', '_', Phon_word)
    Phon_word = re.sub('(^_|_$)', '', Phon_word)

    return Phon_word

def caphi_DBSuffix(bw_word):
    Phon_word = re.sub(r'[><&\}\']', ' 2 ', bw_word)  # turn all hamzas to glottal stop /2/
    Phon_word = re.sub(r'\|', ' 2 aa ', Phon_word)  # turn all hamzas to glottal stop /2/

    Phon_word = re.sub(r'o', ' ', Phon_word)  # remove sukun

    Phon_word = re.sub(r'([aiu])p$', r' \1 ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])p([aiu])$', r' \1 t \2 ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])pF$', r' \1 t a n ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])pK$', r' \1 t i n ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])pN$', r' \1 t u n ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])A', r' \1 ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'aY', r' aa ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([iua])wA', r'\1 w', Phon_word)
    Phon_word = re.sub(r'([^iua])wA', r' uu ', Phon_word)

    Phon_word = re.sub(r'[A]*F$', r' a n ', Phon_word)
    # Phon_word = re.sub(r'[A]*F(\s*#)', r' a n\1', Phon_word)
    Phon_word = re.sub(r'K$', r' a n ', Phon_word)
    # Phon_word = re.sub(r'K(\s*#)', r' i n\1', Phon_word)
    Phon_word = re.sub(r'N$', r' u n ', Phon_word)
    # Phon_word = re.sub(r'N(\s*#)', r' u n\1', Phon_word)

    Phon_word = re.sub(r"A", r' aa ', Phon_word)  # replace 'A' with long vowel 'aa'
    Phon_word = re.sub(r'(.*)iy(\s*[^iau~]+)', r'\1 ii \2', Phon_word)  # replace 'y' with long vowel 'ii'
    Phon_word = re.sub(r'(.*)uw([^iau~]+)', r'\1 uu \2', Phon_word)

    Phon_word = re.sub(r'([^iau])\s*~', r'\1\1', Phon_word)  # double the letter when shadda

    Phon_word = re.sub(r'E', ' 3 ', Phon_word)
    Phon_word = re.sub(r'H', ' 7 ', Phon_word)
    Phon_word = re.sub(r'\*', ' dh ', Phon_word)
    Phon_word = re.sub(r'Z', ' dh. ', Phon_word)
    Phon_word = re.sub(r'\$', ' sh ', Phon_word)
    Phon_word = re.sub(r'g', ' gh ', Phon_word)
    Phon_word = re.sub(r'v', ' th ', Phon_word)
    Phon_word = re.sub(r'S', ' s. ', Phon_word)
    Phon_word = re.sub(r'T', ' t. ', Phon_word)
    Phon_word = re.sub(r'D', ' d. ', Phon_word)
    Phon_word = re.sub(r'x', ' kh ', Phon_word)

    Phon_word = re.sub(r"(d\.|t\.|s\.|sh|kh|gh|dh[.]*|th|tsh|[^iuaoe])", r' \1 ', Phon_word)

    Phon_word = re.sub(' +', ' ', Phon_word)

    Phon_word = re.sub(' +', ' ', Phon_word)

    Phon_word = re.sub(' ', '_', Phon_word)
    Phon_word = re.sub('(^_|_$)', '', Phon_word)

    return Phon_word


def caphi_DBStem(bw_word):
    # Phon_word = SBW_word
    # Phon_word = re.sub(r' ', '#', SBW_word)  # turn all hamzas to glottal stop /2/
    Phon_word = re.sub(r'[><&\}\']', ' 2 ', bw_word)  # turn all hamzas to glottal stop /2/
    Phon_word = re.sub(r'\|', ' 2 aa ', Phon_word)  # turn all hamzas to glottal stop /2/
    Phon_word = re.sub(r'(^A|^{)', ' 2 ', Phon_word)  # turn all hamzas to glottal stop /2/

    Phon_word = re.sub(r'o', ' ', Phon_word)  # remove sukun

    Phon_word = Phon_word.replace('All~`h', ' aa')  # NOTE: Christian addition
    Phon_word = Phon_word.replace('`', ' aa')  # NOTE: Christian addition

    Phon_word = re.sub(r'([aiu])p$', r' \1 ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])p([aiu])$', r' \1 t \2 ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])pF$', r' \1 t a n ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])pK$', r' \1 t i n ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'([aiu])pN$', r' \1 t u n ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel


    Phon_word = re.sub(r'([aiu])A', r' \1 ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    Phon_word = re.sub(r'aY', r' aa ',
                       Phon_word)  # if there is a vowel before the ta marbuta or Alif maqsura or Alif, replace with vowel

    # Phon_word = re.sub(r'([aiu])A(\s*#)', r' \1\2 ',
    #                    Phon_word)  # if there is a vowel before the ta marbuta, replace with vowel
    # Phon_word = re.sub(r'([^aiu])A(\s*#)', r' \1 a\2 ',
    #                    Phon_word)  # if there is no vowel before ta marbuta, replace with 'a'

    # # dialect specific (will not occur in MSA anyways)
    # Phon_word = re.sub(r'uA', r' oo ', Phon_word)
    # Phon_word = re.sub(r'iA', r' ee ', Phon_word)

    Phon_word = re.sub(r'[u]w$', ' uu ', Phon_word)  # replace ending long vowel with short 'w' with 'u'
    # Phon_word = re.sub(r'[u]w(\s*#)', r' u\1 ',
    #                    Phon_word)  # replace ending long vowel with short 'w' with 'u' (in a multi word expression)

    Phon_word = re.sub(r'^A', '2', Phon_word)  # replace initial 'A' with glottal stop /2/ NOT IN SUFF
    # Phon_word = re.sub(r'(#\s*)A', r'\1 2',
    #                    Phon_word)  # replace initial 'A' with glottal stop /2/. (in a multi word expression)

    # Phon_word = re.sub(r'([^iau]) *X', r'\1\1', Phon_word)  # double the letter when shadda

    Phon_word = re.sub(r'[i]y$', ' ii ', Phon_word)  # replace ending long vowel with short 'y' with 'i'
    # Phon_word = re.sub(r'[i]y(\s*#)', ' i\1 ', Phon_word)  # replace ending long vowel with short 'y' with 'i'


    Phon_word = re.sub(r'([iua])yA', r'\1 y aa', Phon_word)
    Phon_word = re.sub(r'([iua])wA', r'\1 w aa', Phon_word) # NOT IN SUFF
    Phon_word = re.sub(r'[A]*F$', r' a n ', Phon_word)
    # Phon_word = re.sub(r'[A]*F(\s*#)', r' a n\1', Phon_word)
    Phon_word = re.sub(r'K$', r' a n ', Phon_word)
    # Phon_word = re.sub(r'K(\s*#)', r' i n\1', Phon_word)
    Phon_word = re.sub(r'N$', r' u n ', Phon_word)
    # Phon_word = re.sub(r'N(\s*#)', r' u n\1', Phon_word)

    # Phon_word = re.sub(r"([\S ]*)A([\S ]*)", r'\1 aa \2', Phon_word)  # replace 'A' with long vowel 'aa'
    Phon_word = re.sub(r"A", r' aa ', Phon_word)  # replace 'A' with long vowel 'aa'
    Phon_word = re.sub(r'(.*)iy(\s*[^iau~]+)', r'\1 ii \2', Phon_word)  # replace 'y' with long vowel 'ii'
    Phon_word = re.sub(r'(.*)uw([^iau~]+)', r'\1 uu \2', Phon_word)

    Phon_word = re.sub(r'([^iau])\s*~', r'\1\1', Phon_word)  # double the letter when shadda

    Phon_word = re.sub(r'E', ' 3 ', Phon_word)
    Phon_word = re.sub(r'H', ' 7 ', Phon_word)
    Phon_word = re.sub(r'\*', ' dh ', Phon_word)
    Phon_word = re.sub(r'Z', ' dh. ', Phon_word)
    Phon_word = re.sub(r'\$', ' sh ', Phon_word)
    Phon_word = re.sub(r'g', ' gh ', Phon_word)
    Phon_word = re.sub(r'v', ' th ', Phon_word)
    Phon_word = re.sub(r'S', ' s. ', Phon_word)
    Phon_word = re.sub(r'T', ' t. ', Phon_word)
    Phon_word = re.sub(r'D', ' d. ', Phon_word)
    Phon_word = re.sub(r'x', ' kh ', Phon_word)

    # Phon_word = re.sub(r'P', ' p ', Phon_word)
    # Phon_word = re.sub(r'G', ' g ', Phon_word)
    # Phon_word = re.sub(r'B', ' v ', Phon_word)
    # Phon_word = re.sub(r'J', ' tsh ', Phon_word)

    #Phon_word = re.sub(r"(d\.|t\.|s\.|sh|kh|gh|dh|dh\.|th|tsh|[^iuaoe])", r' \1 ', Phon_word)
    Phon_word = re.sub(r"(d\.|t\.|s\.|sh|kh|gh|dh[.]*|th|tsh|[^iuaoe])", r' \1 ', Phon_word)

    Phon_word = re.sub(' +', ' ', Phon_word)

    Phon_word = re.sub(' +', ' ', Phon_word)

    Phon_word = re.sub(' ', '_', Phon_word)
    Phon_word = re.sub('(^_|_$)', '', Phon_word)

    return Phon_word


def generate_new_caphi_df():
    db = MorphologyDB('eval_files/calima-msa-s31_0.4.2.utf8.db')

    output = OrderedDict()

    info = [
        # ('prefix', db.prefix_hash, caphi_DBPrefix),
        ('stem', db.stem_hash, caphi_DBStem),
        # ('suffix', db.suffix_hash, caphi_DBSuffix)
    ]

    for m_type, x_hash, fn in info:
        for ms in x_hash.values():
            for _, m in ms:
                if not m or 'diac' not in m:
                    continue
                diac = ar2bw(m['diac'])
                output[(m_type, diac, fn(diac), m.get('caphi', ''))] = 1

    header = ['Type', 'diac', 'Generated CAPHI', 'Original CAPHI']
    with open('scratch_files/caphi/caphi_debug_output.tsv', 'w') as f:
        print(*header, sep='\t', file=f)
        for info in output:
            print(*info, sep='\t', file=f)
    
    output_df = pd.DataFrame(output.keys(), columns=header)
    return output_df

if __name__ == "__main__":
    config = Config('config_test.json', 'caphi_debug', feats='caphi_debug')
    sa = gspread.service_account(config.service_account)
    annotated = pd.read_csv(
        'scratch_files/caphi/caphi_debug_output_annotated.csv', na_filter=False)
    
    new_system_results = generate_new_caphi_df()
    outputs_df, bank = automatic_bank_annotation(
        config,
        new_system_results,
        header=None,
        header_upper=False,
        query_keys=['Type', 'diac', 'Original CAPHI'],
        value_key='Generated CAPHI',
        bank_dir='scratch_files/caphi')
    upload_sheet(config=config,
                 sheet=outputs_df,
                 mode='backup',
                 sa=sa)
    # bank_name = config.debugging.feats['caphi_debug'].bank.split('.')[0]
    # upload_sheet(config=config,
    #              sheet=bank.to_df(),
    #              spreadsheet_name=config.debugging.debugging_spreadsheet,
    #              gsheet_name=bank_name,
    #              mode='backup',
    #              sa=sa)
    pass
