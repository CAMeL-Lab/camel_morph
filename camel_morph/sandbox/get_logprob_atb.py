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

import argparse
import os
import sys
from collections import Counter
from math import log
import pickle

file_path = os.path.abspath(__file__).split('/')
package_path = '/'.join(file_path[:len(file_path) - 1 - file_path[::-1].index('camel_morph')])
sys.path.insert(0, package_path)

from camel_morph.eval.evaluate_camel_morph import _preprocess_magold_data
from camel_tools.morphology.utils import strip_lex

parser = argparse.ArgumentParser()
parser.add_argument("-magold_path", default='eval_files/ATB123-train.102312.calima-msa-s31_0.3.0.magold',
                    type=str, help="Path of the file containing the MSA MAGOLD data to evaluate on.")
parser.add_argument("-output_dir", default='eval_files',
                    type=str, help="Path of the directory to output evaluation results.")


args, _ = parser.parse_known_args()

if __name__ == "__main__":
    data_path = args.magold_path
    with open(data_path) as f:
        data = f.read()
    
    FIELD2INFO_INDEX = {f: i
        for i, f in enumerate(['word', 'ranking', 'starline', 'calima'])}
    
    data = _preprocess_magold_data(data, field2info_index=FIELD2INFO_INDEX)
    feat_freq = {}
    for feat in ['lex', 'pos_lex']:
        feat_freq_ = Counter()
        feat_freq_.update([
            tuple(example['analysis'][feat_]
                  if feat_ != 'lex' else strip_lex(example['analysis'][feat_])
                  for feat_ in feat.split('_'))
            for example in data])
        feat_freq[feat] = feat_freq_

    logprob = {}
    for feat in ['lex', 'pos_lex']:
        feat_freq_ = feat_freq[feat]
        total = sum(feat_freq_.values())
        for v, freq in feat_freq_.items():
            logprob.setdefault(feat, {}).setdefault(v, log(freq / total))

    with open(args.output_dir, 'wb') as f:
        pickle.dump(logprob, f)
