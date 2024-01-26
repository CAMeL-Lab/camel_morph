"""
This code is to sync the magold files to the Calima MSA database.
We read the magold file, and for every word:
    - analyze the word using the CALIMA Star analyzer with the most current CALIMA MSA database
    - of all the analyses pick the one that matches the star analysis from the magold the most
    - the backoff mode that is used for this code will be ADD_PROP
"""

import os
import re
import argparse

from tqdm import tqdm

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

from camel_tools.utils.charmap import CharMapper

CLITICS = frozenset(['enc0', 'prc0', 'prc1', 'prc2', 'prc3'])

EQ_FEATS = frozenset(['asp', 'cas', 'mod', 'per', 'pos', 'stt', 'vox'])
# EQ_FEATS = frozenset(['asp', 'cas', 'enc0', 'mod', 'per', 'pos',
#                       'prc0', 'prc1', 'prc2', 'prc3',
#                       'stt', 'vox'])
DISTANCE_FEATS = frozenset(['diac', 'lex'])

DIACS = ['a', 'u', 'i', 'o', '~', 'F', 'K', 'N', '_', '`']

PRINT_FEATS = ['diac', 'lex', 'bw', 'gloss', 'pos', 
                'prc3', 'prc2', 'prc1', 'prc0', 
                'per', 'asp', 'vox', 'mod', 
                'form_gen', 'gen', 'form_num', 'num', 
                'stt', 'cas', 'enc0', 'rat', 
                'source', 'stem', 'stemcat', 'stemgloss', 'caphi',
                'catib6', 'ud', 'root', 'pattern', 
                'd3seg', 'atbseg', 'd2seg', 'd1seg', 
                'd1tok', 'd2tok', 'atbtok', 'd3tok', 
                'pos_logprob', 'lex_logprob', 'pos_lex_logprob']

AR2BW_MAP = CharMapper.builtin_mapper('ar2bw')
BW2AR_MAP = CharMapper.builtin_mapper('bw2ar')

split_lemma = re.compile('([_-])')
replace_us = re.compile('ـ')

def __split_feats(featline):
    toks = featline.split()
    feats = {}
    for t in toks:
        subtoks = t.split(':')
        if subtoks[0] == 'bw':
            feats[subtoks[0]] = __convert_BW_tag_to_utf8(':'.join(subtoks[1:]))
            feats[subtoks[0]] = replace_us.sub('_', feats[subtoks[0]])
        elif subtoks[0] == 'lex':
            feats[subtoks[0]] = __convert_lex_to_utf8(':'.join(subtoks[1:]))
            # feats[subtoks[0]] = replace_us.sub('_', feats[subtoks[0]])
        elif subtoks[0] == 'diac':
            feats[subtoks[0]] = BW2AR_MAP(':'.join(subtoks[1:]))
            feats[subtoks[0]] = replace_us.sub('_', feats[subtoks[0]])
        else:
            feats[subtoks[0]] = ':'.join(subtoks[1:])
    return feats

def __remove_extra_pluses(string):
    if string.startswith('+'):
        string = string[1:]
    if string.endswith('+'):
        string = string[:len(string) - 1]
    return string    

def __convert_lex_to_utf8(lex):
    lemma_toks = split_lemma.split(lex)
    lex_part = BW2AR_MAP(lemma_toks[0])
    utf8_lex = lex_part
    return utf8_lex

def __convert_BW_tag_to_utf8(BW_tag):

    if BW_tag == '//PUNC' or BW_tag == '+/PUNC' or BW_tag == '_/PUNC':
        return BW_tag

    BW_tag = __remove_extra_pluses(BW_tag)

    BW_elements = BW_tag.split('+')
    utf8_BW_tag = []
    # print(BW_tag, BW_elements)
    for element in BW_elements:
        parts = element.split('/')
        if 'null' in parts[0]:
            BW_lex = parts[0]
        else:
            BW_lex = BW2AR_MAP(parts[0])
        BW_pos = parts[1]
        utf8_BW_tag.append('/'.join([BW_lex,BW_pos]))
    return '+'.join(utf8_BW_tag)

def __levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1], (distances[0]-distances[-1])/distances[0]

def __score_analysis(feats, analysis):

    verbose = False
    if feats['diac'] == '_':
        verbose = True

    score = 0
    for f in EQ_FEATS:
        iseq = analysis.get(f, 'na') == feats.get(f, 'na')
        score += (1 if iseq else 0)
        if not iseq and verbose:
            print(f, analysis.get(f, 'na'), feats.get(f, 'na'))

    for f in CLITICS:
        iseq = ('0' if analysis.get(f, 'na') ==
                'na' else analysis.get(f, 'na')) == feats.get(f, 'na')
        score += (1 if iseq else 0)
        if not iseq and verbose:
            print(f, analysis.get(f, 'na'), feats.get(f, 'na'))

    for f in DISTANCE_FEATS:
        levenshtein, distance_score = __levenshteinDistance(feats[f], analysis[f])
        
        if f == 'diac' and feats[f] == '_':
            levenshtein, distance_score = __levenshteinDistance(AR2BW_MAP(feats[f]), AR2BW_MAP(analysis[f]))
        
        iseq = feats[f] == analysis[f]
        if f == 'diac' and feats[f] == '_':
            iseq = AR2BW_MAP(feats[f]) == AR2BW_MAP(analysis[f])

        # score += (1 if iseq else 0)
        score += distance_score


        if not iseq and verbose:
            print(f, analysis.get(f, 'na'), feats.get(f, 'na'))

    if feats['bw'] == analysis['bw']:
        score += 1
    else:
        if verbose:
            print('bw', analysis.get(f, 'na'), feats.get(f, 'na'))

    if feats.get('pos') == 'verb':
        feats['form_gen'], feats['form_num'] = feats['gen'], feats['num']
    
    if feats['form_gen'] == analysis['gen']:
        score += 1
    else:
        if verbose:
            print('gen', analysis.get(f, 'na'), feats.get(f, 'na'))

    if feats['form_num'] == analysis['num']:
        score += 1
    else:
        if verbose:
            print('num', analysis.get(f, 'na'), feats.get(f, 'na'))

    if feats['gloss'] == analysis['gloss']:
        score += 1
    elif feats['stemgloss'] == analysis['gloss']:
        score += 1
    else:
        g_levenshtein,g_distance_score = __levenshteinDistance(feats['gloss'], analysis['gloss'])
        sg_levenshtein,sg_distance_score = __levenshteinDistance(feats['stemgloss'], analysis['gloss'])
        if g_distance_score > sg_distance_score:
            score += g_distance_score
        else:
            score += sg_distance_score

        if verbose:
            print('gloss', analysis.get(f, 'na'), feats.get(f, 'na'))

    return score

def __select_match(analyses, top_analysis):
    max_score = -1
    max_analysis = None

    for analysis in analyses:
        score = __score_analysis(analysis, top_analysis)
        if score > max_score:
            max_score = score
            max_analysis = analysis

    return max_analysis, max_score


def synchronize(magold, ext, db_path, output_dir):
    db = MorphologyDB(db_path, 'a')
    analyzer = Analyzer(db, 'ADD_PROP')
    input_file = open(magold, 'r')
    result = open(os.path.join(
        output_dir, magold.replace('.magold', '.' + ext)), 'w+')
    debug_output = open(os.path.join(
        output_dir, magold.replace('.magold', '.' + ext + '.debug')), 'w+')

    word = ''
    top_analysis = None
    star_line = ''

    number_of_words = 0
    number_of_perfect_matches = 0

    score_total = 0
    for line in (pbar := tqdm(input_file.readlines())):

        if line.startswith(';;; SENTENCE'):
            result.write(line)

        elif line.startswith(';;WORD'):
            result.write(line)
            word = line.replace(';;WORD ', '').rstrip()
            number_of_words += 1
            if word not in DIACS:
                word = BW2AR_MAP(word)

        elif line.startswith(';;PATB:'):
            result.write(line)

        elif line.startswith(';;'):
            result.write(line)

        elif line.startswith('---'):
            result.write(line)

        elif line.startswith('SENTENCE BREAK'):
            result.write(line)

        elif line.startswith('*'):
            result.write(';;STAR_LINE ' + line)
            top_analysis = __split_feats(' '.join(line.split(' ')[1:]))
            star_line = line

            analyses = analyzer.analyze(word)

            best_match, score = __select_match(analyses, top_analysis)

            output_line = '*' + "%.7f" % (score/18)
            for feat in PRINT_FEATS:
                if feat in best_match:
                    output_line += ' ' + feat + ':' + str(best_match[feat])
                # else:
                #     print(feat + '\t' + str(best_match))
                    # print(best_match)
                    # break
            result.write(AR2BW_MAP(output_line.rstrip() + '\n'))

            if score != 18:
                debug_output.write(word + '\n' + star_line.rstrip() +
                                   '\n' + AR2BW_MAP(output_line.rstrip()) + '\n\n')
            else:
                number_of_perfect_matches += 1

            perf_matches_avg = '%.2f' % (
                (number_of_perfect_matches / number_of_words) * 100) + '%'
            score_total += score
            score_avg = f'{score_total / number_of_words:.1f}'
            pbar.set_description(
                f'perf_avg:{perf_matches_avg} score_avg:{score_avg}')

    result.close()
    debug_output.close()

    print('Perfect matches (%) = ' + '%.2f' %
          ((number_of_perfect_matches / number_of_words) * 100))
    print('No matches (words) = ' +
          str(number_of_words - number_of_perfect_matches))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-magold", required=True,
                        type=str, help="Path of the MSA MAGOLD data containing the synched PATB (LDC) analyses with the SAMA/MADA analyses.")
    parser.add_argument("-db", required=True,
                        type=str, help="Path of the DB to be synced with the MAGOLD data.")
    parser.add_argument("-ext", required=True,
                        type=str, help="Extension to append to the name of the original MAGOLD file for the output MAGOLD file name.")
    parser.add_argument("-output_dir", default='',
                        type=str, help="Directory to output the resulting synced MAGOLD file to.")
    args = parser.parse_args()

    synchronize(args.magold, args.ext, args.db, args.output_dir)
