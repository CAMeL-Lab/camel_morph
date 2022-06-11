import pandas as pd
import re

def generate_substitution_regex(pattern, gem_c_suff_form=False):
    sub, radicalindex2grp = [], {}
    parenth_grp = 1
    for c in pattern:
        if c.isdigit():
            sub.append(f'\\{parenth_grp}')
            radicalindex2grp[int(c)] = parenth_grp
            parenth_grp += 1
        else:
            sub.append(c)
    if gem_c_suff_form and len(radicalindex2grp) >= 2:
        sub[-1] = [s for s in sub if re.search(r'\d', s)][-2]
    sub = ''.join(sub)
    return sub, radicalindex2grp

def generate_root_substitution_regex(row, t_explicit, n_explicit, cond_s, radicalindex2grp):
    root, root_sub = [], []
    root_split = row['ROOT'].split('.')
    for i, r in enumerate(root_split, start=1):
        if not t_explicit and not n_explicit:
            radical_nt = False
        elif not t_explicit and n_explicit:
            radical_nt = (r == 'n')
        elif t_explicit and not n_explicit:
            radical_nt = (r == 't')
        else:
            radical_nt = (r in ['n', 't'])

        if r in ['>', 'w', 'y'] or i == len(root_split) and radical_nt or \
                i == len(root_split) - 1 and root_split[i - 1] == root_split[i] and 'gem' in cond_s and radical_nt:
            root.append(r)
            root_sub.append(r)
        else:
            if i - 2 >= 0 and root_split[i - 1] == root_split[i - 2]:
                root.append(root[-1])
            else:
                root.append(str(i))
            grp = radicalindex2grp.get(i)
            if grp is None:
                if r == root_split[i - 2]:
                    grp = radicalindex2grp.get(i - 1)
                else:
                    return 'Error 12'
            root_sub.append(f'\{grp}')

    return '.'.join(root), '.'.join(root_sub)

def concretize_pattern(pattern, root):
    pattern_ = []
    for c in pattern:
        if c in 'wy' and c in root:
            pattern_.append(str(root.index(c) + 1))
        else:
            pattern_.append(c)
    pattern_ = ''.join(pattern_)
    return pattern_


lexicon = pd.read_csv('data/EGY-LEX-CV.csv')

t_explicit = bool(lexicon['COND-S'].str.contains('#t').any())
n_explicit = bool(lexicon['COND-S'].str.contains('#n').any())

patterns = []
for _, row in lexicon.iterrows():
    root = row['ROOT'].split('.')
    form_pattern = concretize_pattern(row['PATTERN'], root)
    lemma_pattern = concretize_pattern(row['PATTERN_LEMMA'], root)
    _, radicalindex2grp = generate_substitution_regex(row['PATTERN_LEMMA'])
    root_class, _ = generate_root_substitution_regex(
        row, t_explicit, n_explicit, row['COND-S'], radicalindex2grp)

    patterns.append((root_class, lemma_pattern, form_pattern))

with open('sandbox/egy_root_class_form_pattern.tsv', 'w') as f:
    for x in patterns:
        print(*x, sep='\t', file=f)
