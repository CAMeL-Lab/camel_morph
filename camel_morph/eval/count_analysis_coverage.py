# usage: python count_analysis_coverage.py freq_list_file db_file
# freq_list_file format: word\tfreq
# db_file: regular .db file

import sys
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB

db = MorphologyDB(sys.argv[2])
msa_analyzer = Analyzer(db)

def analyze_words(filename):
    words = {}
    with open(filename, 'r') as f:
        for line in f:
            elements = line.strip().split('\t')
            words[elements[0]] = {}
            words[elements[0]]['freq'] = int(elements[1])
            words[elements[0]]['anls'] = 0
            analyses = msa_analyzer.analyze(elements[0])
            if analyses != []:
                words[elements[0]]['anls'] += len(analyses)
    return words



def main():
    words = analyze_words(sys.argv[1])
    print(len(words))

    type_anls_bool = 0
    token_anls_bool = 0
    type_anls_count = 0
    token_anls_count = 0
    oov_words = []
    for word in words:
        if words[word]['anls'] > 0:
            type_anls_bool += 1
            token_anls_bool += words[word]['freq']
            type_anls_count += words[word]['anls']
            token_anls_count += words[word]['freq'] * words[word]['anls']
        else:
            oov_words.append(word)
    print("word types with analysis", type_anls_bool)
    print("word tokens with analysis", token_anls_bool)
    print("---")
    print("number of total analysis for types", type_anls_count)
    print("number of total analysis for token", token_anls_count)
    with open(f'{sys.argv[1]}.oov_types', 'w') as f:
        f.write('\n'.join(oov_words))

if __name__ == "__main__":
    main()