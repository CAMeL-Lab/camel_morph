from logging import getLevelName
import re
from camel_tools.morphology.utils import strip_lex

from camel_morph.utils.utils import assign_pattern, add_check_mark_online

import pandas as pd
from numpy import nan

# verbs_iv = pd.read_csv('data/MSA-LEX-CV.csv')
# verbs_pv = pd.read_csv('data/MSA-LEX-PV.csv')

from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar

ar2bw = CharMapper.builtin_mapper('ar2bw')
bw2ar = CharMapper.builtin_mapper('bw2ar')
safebw2bw = CharMapper.builtin_mapper('safebw2bw')

errors = []

def check(x, pattern, index2radical):
    form = []
    for c in pattern:
        form.append(index2radical[int(c) - 1] if c.isdigit() else c)
    
    return False if ''.join(form) != x else True

def form_qc():
    nominals = pd.read_csv('data/MSA-LEX-Nom.csv')
    qc = []
    for _, row in nominals.iterrows():
        ok = True

        index2radical = row['ROOT'].split('.')
        radical2index = {r: i + 1 for i, r in enumerate(index2radical)}
        pattern = row['PATTERN']
        
        # try:
        ok = ok and check(row['FORM'], pattern, index2radical)
        #     matches = list(re.finditer(r"[^\dAwy><\}\{\&uaio\|~']", row['PATTERN']))
        #     if matches:
        #         for match in matches:
        #             start, end = match.span()
        #             index = radical2index.get(pattern[start])
        #             if index:
        #                 pattern = pattern[:start] + str(radical2index[pattern[start]]) + pattern[end:]
        #                 ok = ok and check(row['FORM'], pattern, index2radical)
        #             else:
        #                 ok = False
        #         qc.append(pattern if ok else f"CHECK:{pattern}")
        #     else:
        qc.append('OK' if ok else 'CHECK')
        # except:
        #     errors.append((row['LEMMA'], row['PATTERN']))

    with open('sandbox/pattern_root_qc.tsv', 'w') as f:
        for line in qc:
            print(line, file=f)
    pass

def lemma_qc():
    qc = []
    for _, row in verbs.iterrows():
        index2radical = row['ROOT'].split('.')

        lemma = strip_lex(row['LEMMA'])
        pattern = assign_pattern(lemma)['pattern_conc']

        lemma_ = []
        for c in pattern:
            lemma_.append(index2radical[int(c) - 1] if c.isdigit() else c)
        
        qc.append('OK' if ''.join(lemma_) == lemma else 'CHECK')

    with open('sandbox/pattern_root_qc.tsv', 'w') as f:
        for line in qc:
            print(line, file=f)
    pass

def use_pv_patterns():
    ext_lemma2root = {}
    for _, row in verbs_pv.iterrows():
        ext_lemma2root.setdefault((row['LEMMA'], row['GLOSS']), []).append(row['ROOT'])

    for k, v in ext_lemma2root.items():
        assert len(v) <= 3 and len(set(v)) == 1

    iv_roots = []
    for _, row in verbs_iv.iterrows():
        iv_roots.append(ext_lemma2root[(row['LEMMA'], row['GLOSS'])][0])

    with open('sandbox/cv_roots.tsv', 'w') as f:
        for root in iv_roots:
            print(root, file=f)
    pass

def get_root():
    roots = pd.read_csv('data/Roots.csv')
    nominals = pd.read_csv('data/MSA-LEX-Nom.csv')
    lemma2root = {}
    for _, row in roots.iterrows():
        lemma2root.setdefault(strip_lex(row['Arabic Lemma']), []).append(row['ROOT'])
    lemma2root = {lemma: root[0] for lemma, root in lemma2root.items() if len(set(root)) == 1}

    roots, status = [], []
    for _, row in nominals.iterrows():
        root = lemma2root.get(bw2ar(strip_lex(row['LEMMA'])))
        if root is None:
            root = row['ROOT']
            status.append('root')
            roots.append(root)
        else:
            status.append('')
            root = ar2bw(root)
            if root != 'NTWS':
                root = re.sub(r"['<}{&]", '>', root)
                root = '.'.join(list(root))
            roots.append(root)
            if root == 'NTWS':
                continue
        
        index2radical = row['ROOT'].split('.')
        pattern = row['PATTERN']
        ok = check(row['FORM'], pattern, index2radical)
        status[-1] += '' if ok else (' pattern' if status[-1] else 'pattern')

    with open('sandbox/nom_roots.tsv', 'w') as f:
        for root, s in zip(roots, status):
            print(root, s, sep='\t', file=f)
    pass


def inspect_lemmas():
    nominals = pd.read_csv('data/MSA-LEX-Nom.csv')
    nominals = nominals.replace(nan, '', regex=True)
    classes = {}
    for _, row in nominals.iterrows():
        feats = {f.split(':')[0]: f.split(':')[1] for f in row['FEAT'].split()}
        pos = feats['pos']
        feats = {k: v for k, v in feats.items() if k != 'pos'}
        cond_t = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-T'].split()]))
        cond_s = ' '.join(sorted(['||'.join(sorted([part for part in cond.split('||')]))
                                  for cond in row['COND-S'].split()]))
        key = (cond_s, cond_t, *[feats[f] for f in ['gen', 'num', 'rat', 'cas']])
        classes.setdefault(pos, {}).setdefault(key, []).append(row.to_dict())
    pass

def indentify_duplicates():
    nominals = pd.read_csv('data/MSA-LEX-Nom.csv')
    nominals = nominals.replace(nan, '', regex=True)
    elementary_feats = ['LEMMA', 'FORM', 'ROOT', 'GLOSS', 'FEAT', 'COND-S', 'COND-T']
    row2indexes = {}
    for i, row in nominals.iterrows():
        row_elem = tuple([row[k] for k in elementary_feats])
        row2indexes.setdefault(row_elem, []).append(i)
    
    status = [''] * len(nominals.index)
    for row, indexes in {k: v for k, v in row2indexes.items() if len(v) > 1}.items():
        longest_pattern_index = sorted(indexes, key=lambda i: len(nominals.iloc[[i]]['PATTERN']), reverse=True)
        status[longest_pattern_index[0]] = f"{longest_pattern_index[0]}-keep"
        for index in longest_pattern_index[1:]:
            status[index] = f"{longest_pattern_index[0]}-delete"

    add_check_mark_online(nominals, 'msa-nom-other-lex', 'MSA-LEX-Nom', mode='duplicate', messages=status, status_col_name='STATUS_TEST')
    pass

def _status_feats_mismatch(cond_t, gen, num):
    #FIXME: to be revised
    form_feats = set([x.strip() for cond in cond_t.split() for x in cond.split('||') if re.search(r'[MF][SDP]', x)])
    if ({'MS', 'FS'} <= form_feats or {'MD', 'FD'} <= form_feats or {'MP', 'FP'} <= form_feats) and gen != '-':
        return 'GEN'
    elif ({'MS', 'MD'} <= form_feats or {'MS', 'MP'} <= form_feats or {'MD', 'MP'} <= form_feats or 
          {'FS', 'FD'} <= form_feats or {'FS', 'FP'} <= form_feats or {'FD', 'FP'} <= form_feats) and num != '-':
        return 'NUM'

def identify_gender_mismatch():
    nominals = pd.read_csv('data/MSA-LEX-Nom.csv')
    nominals = nominals.replace(nan, '', regex=True)
    status = []
    for i, row in nominals.iterrows():
        feats = {f.split(':')[0]: f.split(':')[1] for f in row['FEAT'].split()}
        gen, num = feats['gen'], feats['num']
        if _status_feats_mismatch(row['COND-T'], gen, num):
            status.append('check')
        else:
            status.append('')
    add_check_mark_online(nominals, 'msa-nom-other-lex', 'MSA-LEX-Nom', mode='feat_mismatch', messages=status, status_col_name='STATUS_CHRIS')


def identify_gender_mismatch_lemma_based():
    nominals = pd.read_csv('data/MSA-LEX-Nom.csv')
    nominals = nominals.replace(nan, '', regex=True)
    elementary_feats = ['LEMMA', 'FORM', 'ROOT', 'FEAT', 'COND-S']
    row2indexes = {}
    for i, row in nominals.iterrows():
        row_elem = tuple([row[k] for k in elementary_feats])
        row2indexes.setdefault(row_elem, []).append({'index': i, 'cond-t': row['COND-T'], 'feats': row['FEAT']})
    
    status = [''] * len(nominals.index)
    for row, infos in row2indexes.items():
        if len(infos) == 1:
            continue
        feats = {f.split(':')[0]: f.split(':')[1] for f in infos[0]['feats'].split()}
        gen, num = feats['gen'], feats['num']
        if _status_feats_mismatch(' '.join([info['cond-t'] for info in infos]), gen, num):
            for info in infos:
                status[info['index']] = f"{infos[0]['index']}-check"
    
    add_check_mark_online(nominals, 'msa-nom-other-lex', 'MSA-LEX-Nom',
                          mode='feat_mismatch', messages=status, status_col_name='STATUS_CHRIS')


# identify_gender_mismatch_lemma_based()
# identify_gender_mismatch()
# indentify_duplicates()
# inspect_lemmas()
get_root()
# form_qc()
# use_pv_patterns()
