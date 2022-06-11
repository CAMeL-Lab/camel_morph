import re
from collections import Counter

from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar

ar2bw = CharMapper.builtin_mapper('ar2bw')

dev = 'sandbox/GA_10_nvls_dev.utf8.magold'
test = 'sandbox/GA_10_nvls_test.utf8.magold'
train = 'sandbox/GA_80_nvls_train.utf8.magold'

with open(train) as train_f, open(dev) as dev_f, open(test) as test_f:
    lemmas, lemmas_dediac = [], []
    for file in [train_f, dev_f, test_f]:
        lemmas += re.findall(r'lex:([^ ]+).+pos:([^ ]+)', file.read())
        lemmas_dediac = list(map(lambda x: (dediac_ar(x[0]), x[1]), lemmas))
    
    counter = Counter(lemmas)
    counter_dediac = Counter(lemmas_dediac)

with open('data/glf_verbs_lexprob.tsv', 'w') as f:
    for lemma, count in counter.most_common():
        print(ar2bw(lemma[0]), lemma[1], count, sep='\t', file=f)
