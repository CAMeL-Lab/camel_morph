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


import re
import gspread
import os
import json
from itertools import takewhile

import pandas as pd

consonants_bw = "['|>&<}bptvjHxd*rzs$SDTZEgfqklmnhwy]"
double_cons = re.compile('{}{}'.format(consonants_bw, consonants_bw))
CONS_CLUSTER_IMPLEM = False

POS_NOMINAL = [
    'abbrev', 'adj', 'adj_comp', 'adj_num', 'adv', 'adv_interrog',
    'adv_rel', 'foriegn', 'interj', 'noun', 'noun_num', 'noun_prop',
    'noun_quant', 'pron', 'pron_dem', 'pron_exclam', 'pron_interrog',
    'pron_rel', 'verb_nom', 'verb_pseudo']

lex_keys = ['diac', 'lex']
lex_pos_keys = [*lex_keys, 'pos']
proclitic_keys = ['prc0', 'prc1', 'prc2', 'prc3']
enclitic_keys = ['enc0', 'enc1']
clitic_keys = [*proclitic_keys, *enclitic_keys]
feats_oblig = ['asp', 'mod', 'vox', 'per', 'num', 'gen', 'cas', 'stt']
form_keys = ['form_num', 'form_gen']
essential_keys = [*lex_pos_keys, *feats_oblig, *clitic_keys]
essential_keys_no_lex_pos = [k for k in essential_keys if k not in lex_pos_keys]
essential_keys_form_feats = essential_keys + form_keys
essential_keys_form_feats_no_lex_pos = essential_keys_no_lex_pos + form_keys
essential_keys_form_feats_no_clitics = lex_pos_keys + feats_oblig + form_keys


def patternize_root(root, dc=None, surface_form=False):
    """Will patternize denuded roots (except patterns which are inherently
    geminate which it treats as a root), while keeping defective letters and
    gemination apparent."""
    pattern = []
    soundness = []
    c = 0

    for char in root:
        if char in [">", "&", "<", "}", "'"]:
            pattern.append(">" if not surface_form else char)
            c += 1
            soundness.append('mhmz')
        elif char in ["w", "Y", "y"]:
            pattern.append(char)
            c += 1
            soundness.append('def')
        elif char in ["a", "i", "u", "o"]:
            pattern.append(char)
        elif char == "~":
            pattern.append(char)
            soundness.append('gem')
        elif char == "A":
            if c == 2:
                pattern.append("aA")
            else:
                pattern.append("aAo")
            c += 1
            soundness.append('def')
        else:
            c += 1
            pattern.append(str(c))
            if dc and char == "#":
                pattern.append(str(c))

    soundness_ = f"mhmz{soundness.count('mhmz')}+" if soundness.count(
        'mhmz') else ''
    soundness_ += f"def{soundness.count('def')}+" if soundness.count(
        'def') else ''
    soundness_ += f"gem+" if soundness.count('gem') else ''
    soundness = "sound" if soundness_ == '' else soundness_[:-1]

    return pattern, soundness


def correct_soundness(soundness):
    """For abstract patterns which are inherently geminate (e.g., 1a2~a3).
    """
    soundness = re.sub(r'\+{0,1}gem', '', soundness)
    return soundness if soundness else 'sound'


def analyze_pattern(lemma, root=None, surface_form=False):
    lemma_raw = lemma
    lemma = re.sub(r'\|', '>A', lemma)
    lemma = re.sub(r'aA', 'A', lemma)
    dc = None
    if CONS_CLUSTER_IMPLEM:
        contains_double_cons = double_cons.search(lemma)
        if contains_double_cons:
            if len(contains_double_cons.regs) > 1:
                raise NotImplementedError
            start, end = contains_double_cons.regs[0][0], contains_double_cons.regs[0][1]
            dc = contains_double_cons[0]
            lemma = lemma[:start] + '#' + lemma[end:]

    lemma_undiac = re.sub(r'[auio]', '', lemma)
    num_letters_lemma = len(lemma_undiac)

    exception = is_exception(lemma)
    if exception:
        return exception
    
    result = {'pattern': None,
              'pattern_abstract': None,
              'soundness': None,
              'error': None}
    # Triliteral (denuded)
    # 1a2a3
    if num_letters_lemma == 3:
        pattern, soundness = patternize_root(lemma, dc, surface_form)
        abstract_pattern = "1a2a3"
    # Triliteral (augmented) and quadriliteral (denuded and augmented)
    elif num_letters_lemma > 3:
        if num_letters_lemma == 4:
            # 1a2~3 (tri)
            if lemma[3] == "~" and lemma[1] != "A":
                pattern, soundness = patternize_root(lemma, dc, surface_form)
                soundness = correct_soundness(soundness)
                abstract_pattern = "1a2~a3"
            # 1A2a3 (tri)
            elif lemma[1] == "A":
                lemma_ = lemma[:1] + lemma[2:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                pattern.insert(1, "A")
                abstract_pattern = "1A2a3"
            # >a1o2a3 (tri) [has precedence over the next clause]
            elif lemma[0] == ">" and dc is None and (len(root) == 3 if root else True):
                lemma_ = lemma[2:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                pattern.insert(0, ">a")
                abstract_pattern = ">a1o2a3"
            # 1a2o3a4 (quad)
            elif lemma[3] == "o":
                pattern, soundness = patternize_root(lemma, dc, surface_form)
                abstract_pattern = "1a2o3a4"
            else:
                result['error'] = '4'
                return result

        elif num_letters_lemma == 5:
            if lemma[0] == "t":
                # ta1A2a3 (tri)
                if lemma[3] == "A":
                    lemma_ = lemma[2] + lemma[4:]
                    pattern, soundness = patternize_root(lemma_, dc, surface_form)
                    pattern.insert(0, "ta")
                    pattern.insert(2, "A")
                    abstract_pattern = "ta1A2a3"
                # ta1a2~3 (tri) or ta1a2o3a4 (quad)
                elif lemma[5] == "~" or lemma[5] == "o":
                    lemma_ = lemma[2:]
                    pattern, soundness = patternize_root(lemma_, dc, surface_form)
                    soundness = correct_soundness(soundness)
                    pattern.insert(0, "ta")
                    abstract_pattern = "ta1a2~3" if lemma[5] == "~" else "ta1a2o3a4"
                else:
                    result['error'] = '5+t'
                    return result
            # {ino1a2a3 (tri)
            elif lemma.startswith("{ino") and (lemma[4] == root[0] if root else True) or \
                 lemma.startswith("{im~"):
                if lemma.startswith("{im~"):
                    lemma_ = lemma[2] + "o" + lemma[5:]
                else:
                    lemma_ = lemma[4:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                if lemma.startswith("{im~"):
                    pattern.insert(0, "{i")
                    pattern[2] = '~a'
                else:
                    pattern.insert(0, "{ino")
                abstract_pattern = "{ino1a2a3"
            # {i1o2a3~ (tri) [has precedence over the next clause]
            elif lemma[0] == "{" and lemma[-1] == "~" and (lemma[4] == root[1] if root else True):
                lemma_ = lemma[2:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                soundness = correct_soundness(soundness)
                pattern.insert(0, "{i")
                abstract_pattern = "{i1o2a3~"
            # {i1ota2a3 (tri)
            elif lemma[0] == "{" and (lemma[4] in ["t", "T"] and lemma[4] not in ["m"] or
                                      lemma[3] == "~" or lemma[2] == 'z'):
                abstract_pattern = "{i1ota2a3"
                if len(lemma) == 7:
                    lemma_ = lemma[2:4] + lemma[5:]
                elif lemma[3] == "~":
                    if len(lemma) == 6:
                        lemma_ = lemma[2] + "o" + lemma[4:]
                    else:
                        lemma_ = lemma[2] + "o" + lemma[5:]
                else:
                    lemma_ = lemma[2:4] + lemma[6:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                pattern.insert(0, "{i")
                if lemma[3] == "~":
                    pattern[2] = "~"
                    if len(lemma) != 6:
                        pattern[2] = "~a"
                    if root and root[0] != 't':
                        pattern[1] = 't'
                elif len(lemma) in [6, 7]:
                    pattern.insert(3, "t")
                else:
                    pattern.insert(3, "ta")
            else:
                result['error'] = '5'
                return result
        elif num_letters_lemma == 6:
            # {isota1o2a3 (tri)
            if lemma.startswith("{iso") and lemma[4] == 't':
                lemma_ = lemma[6:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                pattern.insert(0, "{isota")
                abstract_pattern = "{isota1o2a3"
            # {i1oEawo2a3 (tri)
            elif lemma.startswith("{i") and lemma[6:8] == "wo":
                lemma_ = lemma[2:4] + lemma[8:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                pattern.insert(0, "{i")
                pattern.insert(3, "2awo")
                abstract_pattern = "{i1o2awo2a3"
            # {i1o2a3a4~ (quad)
            elif lemma[-1] == "~":
                lemma_ = lemma[2:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                soundness = correct_soundness(soundness)
                if soundness == "def1":
                    pattern[3] = "aAo"
                pattern.insert(0, "{i")
                abstract_pattern = "{i1o2a3a4~"
            # {i1o2ano3a4 (quad)
            elif lemma[6:8] == "no":
                lemma_ = lemma[2:6] + lemma[8:]
                pattern, soundness = patternize_root(lemma_, dc, surface_form)
                pattern.insert(0, "{i")
                pattern.insert(5, "no")
                abstract_pattern = "{i1o2ano3a4"
            else:
                result['error'] = '6'
                return result
        else:
            result['error'] = '>4'
            return result
    # If there are less than 3 letters (maybe there is a problem)
    else:
        result['error'] = '<3'
        return result

    pattern = ''.join(pattern)

    if surface_form:
        pattern = re.sub(r'>aAo|>A', '|', pattern)
        pattern = re.sub(r'a?Ao?', 'A', pattern)
        if abstract_pattern == "{i1ota2a3" and '~' not in lemma_raw or \
            abstract_pattern == "{i1ota2a3" and lemma[3] != "~" and len(lemma) not in [6, 7]:
            pattern = pattern[:4] + lemma_raw[4] + pattern[5:]
        elif abstract_pattern == "{i1ota2a3"and lemma[0] == "{" and (lemma[4] in ["t", "T"] or 
            lemma[3] == "~" or lemma[2] == 'z'):
            if root and root[0] in ['d', 'D', 'v', 'T', 'Z']:
                pattern = pattern[:2] + '1' + pattern[3:]
    result['pattern'] = re.sub(r'([^a])A', r'\1aA', pattern)
    result['pattern_abstract'] = abstract_pattern
    result['soundness'] = soundness
    return result

def is_exception(lemma):
    exceptions = {
        ">anojolaz": {'pattern': '>a2o3o4a5', 'pattern_abstract': '1a2o3o4a5', 'soundness': 'sound', 'error': None},
        "ta>anojolaz": {'pattern': 'ta>a2o3o4a5', 'pattern_abstract': 'ta1a2o3o4a5', 'soundness': 'sound', 'error': None}
    }
    return exceptions.get(lemma)


def assign_pattern(lemma, root=None):
    info = analyze_pattern(lemma, root)
    info_surf = analyze_pattern(lemma, root, surface_form=True)

    result = {'pattern_conc': info['pattern'],
              'pattern_surf': info_surf['pattern'],
              'pattern_abstract': info['pattern_abstract'],
              'soundness': info['soundness'],
              'error': info['error']}

    return result


def analyze_pattern_egy(root, stem):
    i = 0
    tmp_stem = stem
    first_part = ''
    second_part = stem

    for char in root:
        i = i+1
        if char not in 'yw><&C{}C':
            # print tmp_stem,root,char,i
            if char == '$':
                char = '\$'
            if char == '*':
                char = '\*'
            second_part = re.sub(char, str(i), second_part, 1)
            # print root, second_part
        # when w is a consonant that is the first letter
        elif char == 'w' and second_part.startswith('w') and i == 1:
            second_part = re.sub(char, str(i), second_part, 1)
        elif char == 'y' and i == 1:
            second_part = re.sub(char, str(i), second_part, 1)
        elif (char == 'y' or char == 'w') and i == len(root) and (second_part.endswith('A') or second_part.endswith('a')):
            second_part = second_part[:-1] + 'aY'
            stem = stem[:-1] + 'aY'
        elif (char == 'y' or char == 'w') and i == len(root) and (second_part.endswith('A') or second_part.endswith('a')):
            second_part = second_part[:-1] + 'aY'
            stem = stem[:-1] + 'aY'
        elif (char == 'y' or char == 'w') and i == len(root) and (second_part.endswith('A') or second_part.endswith('a')):
            second_part = second_part[:-1] + 'A'
            stem = stem[:-1] + 'A'
        elif second_part.endswith('i'):
            second_part = second_part[:-1] + 'iy'
            stem = stem[:-1] + 'iy'
        elif char == 'y':       # when y in the root is a consonant
            if not re.search('iy(?!a|i|u|\~)', second_part, 1):
                second_part = re.sub(r'y', str(i), second_part, 1)
                # print second_part
            else:
                second_part = re.sub(
                    r'([aui\~])y([aiu\~])', r'\g<1>'+str(i)+r'\g<2>', second_part)
        elif char == 'w':       # when w in the root is a consonant
            if not re.search('uw(?!a|i|u|\~)', second_part):
                second_part = re.sub(r'w', str(i), second_part, 1)
        elif char == 'C' and i == len(root):
            # print 'Salam'
            if second_part.endswith('a'):
                # print 'in',root, i, char,second_part,stem
                second_part = second_part[:-1] + 'A'
                stem = stem[:-1] + 'A'
                # print 'out',second_part
            elif second_part.endswith('i'):
                second_part = second_part[:-1] + 'iy'
                stem = stem[:-1] + 'iy'
        else:
            pass
        if str(i) in second_part:
            # print second_part, root
            first_part = first_part + second_part.split(str(i))[0]+str(i)
            second_part = second_part.split(str(i))[1]
            # print first_part,second_part
        else:
            second_part = second_part

    tmp_stem = first_part+second_part
    # print second_part

    # hardcamel_morphd stuff
    if stem == 'AiftataH':
        tmp_stem = 'Ai1ta2a3'
    elif stem == 'yiftitiH':
        tmp_stem = 'yi1ti2i3'
    elif stem == 'Aistashal' or stem == 'Aistaslam':
        tmp_stem = 'Aista12a3'
    elif stem == 'yistashal' or stem == 'yistaslam':
        tmp_stem = 'yista12i3'

    return tmp_stem


def index2col_letter(index):
    column_letter = chr(ord('A') - 1 + index // 26) if index >= 26 else ''
    column_letter += chr(ord('A') + index % 26)
    return column_letter


def col_letter2index(col_letter):
    index = 0
    for i, col_letter_ in enumerate(col_letter[::-1]):
        index += (26 ** i) * (ord(col_letter_) - ord('A') + (1 if i else 0))
    return index


def add_check_mark_online(rows,
                          spreadsheet,
                          worksheet,
                          error_cases=None,
                          indexes=None,
                          messages=None,
                          mode=None,
                          write='append',
                          status_col_name='STATUS',
                          service_account='/Users/chriscay/.config/gspread/service_account.json'):
    assert bool(error_cases) ^ bool(indexes) ^ bool(messages)
    if error_cases is not None:
        filtered = rows[rows['LEMMA'].isin(error_cases)]
        indexes = list(filtered.index)

    if type(spreadsheet) is str:
        sa = gspread.service_account(service_account)
        spreadsheet = sa.open(spreadsheet)

    if type(worksheet) is str:
        worksheet = spreadsheet.worksheet(title=worksheet)
    header = worksheet.row_values(1)
    header_count = header.count(status_col_name)
    if header_count == 0:
        worksheet.insert_cols([[status_col_name]])
        header = worksheet.row_values(1)
    elif header_count > 1:
        raise NotImplementedError

    status_column_index = header.index(status_col_name)
    column_letter = index2col_letter(status_column_index)

    status_old = worksheet.col_values(status_column_index + 1)[1:]
    lemmas = worksheet.col_values(header.index('LEMMA') + 1)[1:]
    status_old += [''] * (len(lemmas) - len(status_old))
    assert len(lemmas) == len(status_old) == len(rows['LEMMA'])
    col_range = f'{column_letter}2:{len(rows.index) + 1}'
    
    if indexes:
        if mode:
            check, ok = f'{mode}:CHECK', f'{mode}:OK'
        else:
            check, ok = 'CHECK', 'OK'
        assert set(status_old) <= {check, ok, ''}
        status_new = [[check] if i in indexes else ([ok] if status_old[i] != check else [check])
                            for i in range(len(rows['LEMMA']))]
    elif messages:
        assert len(status_old) == len(lemmas) == len(messages)
        if mode:
            mode = f'{mode}:'
        else:
            mode = ''
        if write == 'overwrite':
            status_new = [[f'{mode}{message}'] if message else ['']
                            for message in messages]
        elif write == 'append':
            status_new = [[f"{s}{' ' if s else ''}" + f'{mode}{message}'] if message else [s + '']
                            for s, message in zip(status_old, messages)]
    else:
        raise NotImplementedError
        
    worksheet.update(col_range, status_new)

def strip_brackets(info):
    if info[0] == '[' and info[-1] == ']':
        info = info[1:-1]
    return info


def get_data_dir_path(config, config_name):
    dialect = config['local'][config_name]['dialect']
    return os.path.join('data',
                        f'camel-morph-{dialect}',
                        config_name)


def sheet2df(sheet):
    return pd.DataFrame(sheet.get_all_records())


def lcp(strings):
    "Longest common prefix"
    def allsame(strings_):
        return len(set(strings_)) == 1
    
    return ''.join(i[0] for i in takewhile(allsame, zip(*strings)))


class Debugging:
    def __init__(self, debugging, feats=None) -> None:
        if debugging is None:
            debugging = {}
        self.bank = debugging.get('bank')
        self.sheets = debugging.get('sheets')
        self.debugging_sheet = debugging.get('debugging_sheet')
        self.debugging_spreadsheet = debugging.get('debugging_spreadsheet')
        self.paradigm_debugging = debugging.get('paradigm_debugging')
        self.display_format = debugging.get('display_format')
        self.lexprob_db = debugging.get('lexprob_db')
        self.docs_bank = debugging.get('docs_bank')
        self.docs_output_name = debugging.get('docs_output_name')
        self.docs_debugging_spreadsheet = debugging.get('docs_debugging_spreadsheet')
        self.docs_debugging_sheet = debugging.get('debugging_sheet')
        self.docs_tables = debugging.get('docs_tables')
        self.insert_index = debugging.get('insert_index')
        self.conj_tables = debugging.get('conj_tables')
        self.pos_display = debugging.get('pos_display')
        self.stats_spreadsheet = debugging.get('stats_spreadsheet')
        self.stats_sheet = debugging.get('stats_sheet')
        self.table_start_cell = debugging.get('table_start_cell')
        self.feats = {feats_: Debugging(debugging_)
                      for feats_, debugging_ in debugging.get('feats', {}).items()}
        self.feats = self.feats if self.feats != {} else None

        self.debugging_feats = None
        if feats is not None:
            self.debugging_feats = self.feats[feats]

    def __repr__(self) -> str:
        return str(self.__dict__)
        

class Config:
    GLOBAL_SHEET_TYPES = ['about', 'header']
    SHEET_TYPES_ESSENTIAL = ['order', 'morph', 'lexicon']
    SHEET_TYPES_OPTIONAL = ['postregex', 'passive', 'backoff']
    LOCAL_SHEET_TYPES = SHEET_TYPES_ESSENTIAL + SHEET_TYPES_OPTIONAL
    SHEET_TYPES = GLOBAL_SHEET_TYPES + SHEET_TYPES_ESSENTIAL + \
        SHEET_TYPES_OPTIONAL

    def __init__(self, config_file, config_name=None, feats=None) -> None:
        self._config_file = config_file
        self._config = self.read_config()
        self._config_name = config_name
        self._config_global = self.read_config_global()
        self._config_local = None
        if config_name is not None:
            self._config_local = self.read_config_local(config_name, feats)


    def read_config(self):
        configs_dir = os.path.join(
            '/'.join(os.path.dirname(__file__).split('/')[:-1]), 'configs')
        with open(os.path.join(configs_dir, self._config_file)) as f:
            config = json.load(f)
        return config
    
    
    def read_config_local(self, config_name, feats=None):
        if config_name not in self._config['local']:
            return
        config_local = self._config['local'][config_name]
        self._config_local = config_local
        # Dialect being used for the configuration (for storing debugging files
        # and DB files in corresponding directories)
        self.dialect = config_local.get('dialect')
        # Whether or not to perform pruning before compatibility validation
        # (pruning refers to the process of eliminating easily eliminable
        # combinations of allomorphs based on their condition classes)
        self.pruning = config_local.get('pruning')
        # Wether or not to perform reindexing of categories and collapsing
        # of morphemes after compatibilities have been determined through validation.
        self.reindex = config_local.get('reindex')
        # Names of the specification sheets (ORDER, MORPH, LEXICON)
        self.specs = self._read_specs()
        # Name of the (output) DB associated with the configuration
        self.db = config_local.get('db')
        # POS type (nominal, verbal, other, any) associated with the configuration
        self.pos_type = config_local.get('pos_type')
        # POS (CAPHI) associated with the configuration (if any)
        self.pos = config_local.get('pos')
        # Features to use to augment the unique lemmas while choosing
        # representative lemmas
        self.extended_lemma_keys = config_local.get('extended_lemma_keys')
        # Features to unique on while choosing representative lemmas
        self.class_keys = config_local.get('class_keys')
        # Path of file containing lemmas to exclude while extracting
        # representative lemmas
        self.exclusions = config_local.get('exclusions')
        # Whether or not to split lexicon lines based on ORed (OR operation)
        # COND-T terms
        self.split_or = config_local.get('split_or')
        # Whether or not to transform categories from the debugging format
        # to a generic ID format. This should be be activated if short
        # morpheme names are used, otherwise it will mess with category factorization
        self.cat2id = config_local.get('cat2id')
        # Whether or not to fill in some unspecified features with their default values
        self.defaults = config_local.get('defaults')
        # Whether or not to automatically delete conditions which are not 
        # being utilized from COND-S and COND-T by cross-checking between 
        # MORPH and LEXICON sheets
        self.clean_conditions = config_local.get('clean_conditions')
        # Script to load for caphi conversion
        self.caphi = config_local.get('caphi')
        # Object containing log probabilities of lemmas (or lexpos)
        self.logprob = config_local.get('logprob')
        # Kill all lexicon entries which do not have the specified lemma
        self.restrict_db_to_lemma = config_local.get('restrict_db_to_lemma')
        # Path of class map object mapping morpheme classes to complex morpheme classes
        self.class_map = config_local.get('class_map')
        # Information for miscellaneous debugging utilities
        self.debugging = Debugging(config_local.get('debugging'), feats)
        return config_local

    def read_config_global(self):
        config_global = self._config['global']
        self._config_global = config_global
        # Directory where all sheets from Google Sheets are downloaded
        self.data_dir = config_global.get('data_dir')
        # Directory where the Camel Morph DBs are output
        self.db_dir = config_global.get('db_dir')
        # Directory where the debugging sheets are output
        self.debugging_dir = config_global.get('debugging_dir')
        # Directory where the documentation debugging sheets are output
        self.docs_debugging_dir = config_global.get('docs_debugging_dir')
        # Directory within the debugging dir where the documentation banks are output
        self.docs_banks_dir = config_global.get('docs_banks_dir')
        # Directory within the debugging dir where the inflection tables for doc are output
        self.docs_tables_dir = config_global.get('docs_tables_dir')
        # Directory within the debugging dir where the representative lemmas
        # are output
        self.repr_lemmas_dir = config_global.get('repr_lemmas_dir')
        # Names of the global specification sheets (ABOUT, HEADER)
        self.specs_global = self._read_specs_global()
        # Directory within the debugging dir where the debugging inflection
        # tables are output
        self.tables_dir = config_global.get('tables_dir')
        # Directory within the debugging dir where automatically bank-debugged
        # inflection tables are output
        self.paradigm_debugging_dir = config_global.get('paradigm_debugging_dir')
        # Directory within the debugging dir where the banks are output
        self.banks_dir = config_global.get('banks_dir')
        # Path of an alternative fork of camel_tools to used instead of the
        # official release
        self.camel_tools = config_global.get('camel_tools')
        # Path to the JSON file containing the credentials to the Google Cloud
        # account used to perform sheet operations via the gspread API
        self.service_account = config_global.get('service_account')
        # Paradigm slots used as part of the morph_debugging process
        self.paradigms_config = config_global.get('paradigms_config')
        # Spreadsheet to which banks are uploaded
        self.banks_spreadsheet = config_global.get('banks_spreadsheet')
        return config_global

    def _read_specs(self):
        # Essential specification sheets
        self.order = self._config_local['specs']['order']
        self.morph = self._config_local['specs']['morph']
        self.lexicon = self._config_local['specs']['lexicon']
        # Sheets containing POSTREGEX regex strings to compile into the DB
        self.postregex = self._config_local['specs'].get('postregex')
        # Keywords used to exclude entries containing them in the EXCLUDE column
        self.exclude = self._config_local['specs'].get('exclude')
        # Sheets used to automatically generate passive entries for PV and IV verbs
        # based on regex rules
        self.passive = self._config_local['specs'].get('passive')
        # Sheets containing smart backoff entries to use for DB compilation
        self.backoff = self._config_local['specs'].get('backoff')
        return self._config_local['specs']
    
    def _read_specs_global(self):
        # Essential specification global specification sheets
        self.about = self._config_global['specs']['about']
        self.header = self._config_global['specs']['header']
        return self._config_global['specs']
    
    def get_dialect_project_dir_path(self, dialect=None):
        return f'camel-morph-{dialect if dialect is not None else self.dialect}'


    def get_spreadsheet2sheets(self,
                               sheet_types=[],
                               config_name=None,
                               with_labels=False):
        if sheet_types == []:
            sheet_types = Config.SHEET_TYPES
        else:
            if type(sheet_types) is str:
                assert sheet_types in Config.SHEET_TYPES
                sheet_types = [sheet_types]
            elif type(sheet_types) is list:
                assert set(sheet_types) <= set(Config.SHEET_TYPES)
            else:
                raise NotImplementedError
        
        if config_name is not None:
            config_local = self._config['local'][config_name]
        else:
            config_local = self._config_local

        spreadsheet2sheets = {}
        for sheet_type in sheet_types:
            if Config.GLOBAL_SHEET_TYPES:
                spreadsheet2sheets_ = getattr(self, sheet_type)
            else:
                spreadsheet2sheets_ = config_local['specs'][sheet_type]
            if spreadsheet2sheets_ is None:
                continue
            for spreadsheet, sheets in spreadsheet2sheets_.items():
                sheets_ = []
                if type(sheets) is str:
                    sheets_.append((sheets, '') if with_labels else sheets)
                elif type(sheets) is list:
                    for sheet in sheets:
                        sheets_.append((sheet, '') if with_labels else sheet)
                elif type(sheets) is dict:
                    for sheet, label in sheets.items():
                        if sheet_type in Config.SHEET_TYPES_OPTIONAL:
                            sheet, label = label, sheet
                        sheets_.append((sheet, label) if with_labels else sheet)
                else:
                    raise NotImplementedError
                for sheet in sheets_:
                    spreadsheet2sheets.setdefault(spreadsheet, []).append(sheet)

        return spreadsheet2sheets
    
    def get_sheets_list(self,
                        sheet_types=None,
                        config_name=None,
                        with_labels=False):
        spreadsheet2sheets = self.get_spreadsheet2sheets(
            sheet_types, config_name, with_labels)
        sheets = sum(spreadsheet2sheets.values(), [])
        return sheets
    
    def get_sheets_paths(self,
                         sheet_types=[],
                         config_name=None,
                         with_labels=False,
                         ext='csv'):
        data_dir_path = self.get_data_dir_path()
        sheet_paths = []
        for sheet_name in self.get_sheets_list(sheet_types, config_name, with_labels):
            if with_labels:
                sheet_name_, label = sheet_name
                sheet_path = os.path.join(data_dir_path, f'{sheet_name_}.{ext}')
                sheet_paths.append((sheet_path, label))
            else:
                sheet_path = os.path.join(data_dir_path, f'{sheet_name}.{ext}')
                sheet_paths.append(sheet_path)

        return sheet_paths
    
    def get_sheet_path_from_name(self,
                                 sheet_name,
                                 ext='csv'):
        return os.path.join(self.get_data_dir_path(), f'{sheet_name}.{ext}')

    def get_repr_lemmas_file_name(self):
        return f'repr_lemmas_{self._config_name}.pkl'
    
    def get_repr_lemmas_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.repr_lemmas_dir, self.get_dialect_project_dir_path())
    
    def get_repr_lemmas_path(self):
        return os.path.join(self.get_repr_lemmas_dir_path(), self.get_repr_lemmas_file_name())

    def get_db_dir_path(self):
        return os.path.join(self.db_dir, self.get_dialect_project_dir_path())

    def get_db_path(self):
        db_name = self._config_local['db']
        return os.path.join(self.get_db_dir_path(), db_name)

    def get_data_dir_path(self):
        return os.path.join(self.data_dir, self.get_dialect_project_dir_path(), self._config_name)
    
    def get_banks_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.banks_dir, self.get_dialect_project_dir_path())
    
    def get_tables_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.tables_dir, self.get_dialect_project_dir_path())

    def get_docs_banks_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.docs_banks_dir, self.get_dialect_project_dir_path())
    
    def get_paradigm_debugging_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.paradigm_debugging_dir, self.get_dialect_project_dir_path())

    def get_docs_debugging_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.docs_debugging_dir, self.get_dialect_project_dir_path())
    
    def get_docs_tables_dir_path(self):
        return os.path.join(
            self.debugging_dir, self.docs_tables_dir, self.get_dialect_project_dir_path())
