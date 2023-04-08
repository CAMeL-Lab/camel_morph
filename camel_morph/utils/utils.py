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

consonants_bw = "['|>&<}bptvjHxd*rzs$SDTZEgfqklmnhwy]"
double_cons = re.compile('{}{}'.format(consonants_bw, consonants_bw))
CONS_CLUSTER_IMPLEM = False

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
    result['pattern'] = pattern
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
    column_letter = (chr(ord('A') + status_column_index // 27) if status_column_index >= 26 else '') + \
        chr(ord('A') + status_column_index % 26)

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

def _strip_brackets(info):
    if info[0] == '[' and info[-1] == ']':
        info = info[1:-1]
    return info


def get_config_file(config_file):
    configs_dir = os.path.join(
        '/'.join(os.path.dirname(__file__).split('/')[:-1]), 'configs')
    with open(os.path.join(configs_dir, config_file)) as f:
        config = json.load(f)
    return config