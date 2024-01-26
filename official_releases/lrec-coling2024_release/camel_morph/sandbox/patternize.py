import re
import sys

import pandas as pd
from numpy import nan
import random


sys.path.append(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph')

from camel_morph.utils.utils import assign_pattern
dubious = set()
exceptions = {
    '{ivonayoni': '{i1o2a3oni',
    '<inosAn': '<i2o3An',
    'bunoyAn': '1u2o3An'}

visited = set()

def assign_pattern_nom(lemma, root):
    if lemma in exceptions:
        return {'error': None, 'pattern_surf': exceptions[lemma]}

    radical2index = {}
    for i, r in enumerate(root, start=1):
        radical2index.setdefault(r, []).append(str(i))
    pattern = []
    for i, c in enumerate(lemma):
        if c in {'A', 'p', '{', '&', '<', '>', "'", '}', '|', 'i', 'a', 'u', 'o'}:
            pattern.append(c)
            visited.add('1')
        elif (lemma[i - 1 : i + 2] in ['oti', 'oTi'] or lemma[i - 2 : i + 2] == 'zodi') and \
            lemma[0] == '{':
            pattern.append(c)
            visited.add('2')
        elif i == 2 and lemma[: i + 2] == '{ino' and root[0] != 'n':
            pattern.append(c)
            visited.add('3')
        elif (i == 2 and lemma[: i + 2] == '{iso' or \
             i == 4 and lemma[: i + 2] == '{isoti') and root[0] != 's':
            pattern.append(c)
            visited.add('4')
        elif i == 0 and lemma[:3] in ['mam', 'mum', 'mim'] and root[0] == 'm':
            pattern.append(c)
            visited.add('6')
        elif (i == 0 or i == 2) and lemma.startswith('muta'):
            pattern.append(c)
            visited.add('7')
        elif (i == 0 or i == 2 or i == 4) and lemma.startswith('musota'):
            pattern.append(c)
            visited.add('8')
        elif (i == 0 or i == 2) and lemma.startswith('muno'):
            pattern.append(c)
            visited.add('9')
        elif i == 0 and lemma[:2] in ['ma', 'mu', 'mi'] and root[0] != 'm':
            pattern.append(c)
            visited.add('10')
        elif i == 0 and lemma.startswith('ta') and root[0] != 't':
            pattern.append(c)
            visited.add('11')
        elif c in radical2index:
            if radical2index[c]:
                pattern.append(radical2index[c][0])
                del radical2index[c][0]
                visited.add('12')
            elif not radical2index[c] and i == len(lemma) - 1 and lemma[-1] == lemma[-3]:
                pattern.append(pattern[-2])
                visited.add('13')
                continue
            elif not radical2index[c] and lemma[i - 3 : i + 1] in [f'{c}iy{c}', f'{c}uw{c}'] :
                pattern.append(pattern[-3])
                visited.add('14')
                continue
            elif i == len(lemma) - 2 and lemma[i:] == 'y~' or \
                 i == len(lemma) - 4 and lemma[i:] == 'y~ap':
                pattern.append(c)
                visited.add('15')
            elif lemma.count('n') == 2 and root.count('n') != 2 and \
                 re.match(r'.+An(iy~)?', lemma) and not lemma.endswith('nAn'):
                pattern.append(c)
                visited.add('16')
            elif lemma.count('w') == 2 and root.count('n') != 2 and \
                 re.match(r'.+wi(iy~)?', lemma):
                pattern.append(c)
                visited.add('17')
            elif lemma.count(c) == 2 and root.count(c) != 2 and \
                    re.match(f'.+{c}uw{c}(ap)?', lemma):
                pattern.append(c)
                visited.add('18')
            elif re.match(f'.+{root[1]}uw{root[2]}(iy~)?', lemma):
                pattern.append(c)
                visited.add('19')
            else:
                return {'error': 'error', 'pattern_surf': None}
        else:
            visited.add('21')
            pattern.append(c)
    pattern = ''.join(pattern)
    
    return {'error': None, 'pattern_surf': pattern}


data = pd.read_csv(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/data/camel-morph-msa/nom_msa_red/MSA-Nom-LEX.csv')
data = data.replace(nan, '', regex=True)
with open('/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/lemmas_patternized.csv', 'w') as fout:
    conditions = {}
    patterns, pattern_NA = {}, {}
    subpatterns = {}
    used = set()
    c = 0
    for line_i, row in data.iterrows():
        lemma, form, root, pos = row['LEMMA'], row['FORM'], row['ROOT'], row['BW']
        gloss, feats = row['GLOSS'], row['FEAT']
        cond_t = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-T'].split()]))
        cond_s = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-S'].split()]))
        lemma_raw = lemma.split('-')[0].split('_')[0]
        feats_dict = {f.split(':')[0]: f.split(':')[1] for f in feats.split()}
        feats_key = tuple([feats_dict[f] for f in ['gen', 'num']])
        key = (lemma, root, form, feats_key, cond_t, cond_s)
        # if key in used:
        #     continue

        used.add(key)
        if root != 'NTWS':
            lemma = re.sub(r'\|', '>A', lemma_raw)
            # pattern, abstract_pattern, soundness, error = assign_pattern(lemma, root=root.split('.'))
            # result = assign_pattern(lemma, root=root.split('.'))
            root_ = root.split('.')
            result = assign_pattern_nom(lemma, root=root_)
            # pattern_conc = result['pattern_conc']
            pattern_surf = result['pattern_surf']
            pattern_surf = pattern_surf.replace('>A', '|') if pattern_surf else pattern_surf
            # pattern_abstract = result['pattern_abstract']
            error = result['error']
            # output = (lemma_raw, root, pattern_surf)

            output = (pattern_surf,)
        else:
            error, output = None, ('NTWS',)
        
        if error:
            c += 1
            info = patterns.setdefault(error, {'freq': 0, 'lemmas': []})
            info['freq'] += 1
            info['lemmas'].append((lemma_raw, form, root, pos, gloss, feats, cond_t, cond_s))
            print('ERROR', file=fout)
            print(lemma, root, error, output)
        else:
            reconstructed = []
            if root != 'NTWS':
                for p in pattern_surf:
                    if p.isdigit():
                        reconstructed.append(root_[int(p) - 1])
                    else:
                        reconstructed.append(p)
                if ''.join(reconstructed) != lemma_raw:
                    l = 0
            info = patterns.setdefault(output[0], {'freq': 0, 'lemmas': []})
            info['freq'] += 1
            info['lemmas'].append((lemma_raw, form, root, pos, gloss, feats, cond_t, cond_s))
            print(*output, file=fout, sep='\t')

dubious = {x[0]: x[1] for x in sorted(patterns.items(), key=lambda x: x[1]['freq'])}
dubious = {k: {'freq': v['freq'], 'lemmas': [l for l in v['lemmas'] if {'w', 'y'} & set(l[2])]} for k, v in dubious.items()}
dubious = {k: v for k, v in dubious.items() if v['lemmas']}
dubious_ = {}
for k, v in dubious.items():
    dubious_.setdefault(v['freq'], 0)
    dubious_[v['freq']] += 1
proportions = {k: round(v / sum(dubious_.values()) * 100) for k, v in dubious_.items()}

while sum(proportions.values()) != 100:
    proportions[random.choice(list(proportions.keys()))] += 1

dubious = [random.sample([random.choice([(k, l[0], l[2]) for l in v['lemmas']])
                            for k, v in dubious.items() if v['freq'] == x], y)
            for x, y in proportions.items()]
with open('sandbox/reham_root_pattern_checking.tsv', 'w') as f:
    for pattern, info in dubious.items():
        for lemma, form, root, pos, gloss, feats, cond_t, cond_s in info['lemmas']:
            print(lemma, pattern, '', info['freq'], root, '', '', form, pos,
                  gloss, feats, cond_t, cond_s, sep='\t', file=f)

# [(i, len({k: v for k, v in dubious.items() if v['freq'] > i and k != 'NTWS'}), random.choice([random.choice([(k, l[0], l[2]) for l in v['lemmas']]) for k, v in {k: v for k, v in dubious.items() if k != 'NTWS'}.items() if v['freq'] > i])) for i in list(range(20)) + list(range(20, 2120, 27))]
c = 0
