from collections import Counter
import re
from tqdm import tqdm
import sys

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator
from camel_tools.utils.dediac import dediac_bw

from nltk import edit_distance

bw2ar = CharMapper.builtin_mapper('bw2ar')
bw2ar_translit = Transliterator(bw2ar)
ar2bw = CharMapper.builtin_mapper('ar2bw')
ar2bw_translit = Transliterator(ar2bw)

db_camel = MorphologyDB(sys.argv[2])
analyzer_camel = Analyzer(db_camel)
db_calima = MorphologyDB(sys.argv[3])
analyzer_calima = Analyzer(db_calima)

output_inspection = {}

def _postprocess_gold(gold):
    gold = tuple([':'.join(f.split(':')[1:]) for f in gold])
    diac = re.sub(r'o', '', gold[0])
    # Remove -[uai] and _[12]
    lex = re.sub(r'([^-_]+)(-\w+)?(_\d)?', r'\1', gold[1])
    bw = gold[2]
    bw = '+'.join([morph.split('/')[1]
                   for morph in bw.split('+') if 'STEM' not in morph])
    pos = gold[3]
    gold = (diac, lex, bw, pos)

    return gold

def _postprocess_pred(pred):
    pred_ = []
    for analysis in pred:
        diac = ar2bw_translit.transliterate(analysis['diac'])
        diac = re.sub(r'o', '', diac)
        lex = re.sub(r'_\d', '', ar2bw_translit.transliterate(analysis['lex']))
        bw = ar2bw_translit.transliterate(analysis['bw'])
        asp = re.search(r'([PIC])V\+', bw)
        asp = (asp.group(1) + 'V') if asp else ''
        bw = '+'.join([morph.split('/')[1]
                       for morph in bw.split('+') if 'STEM' not in morph])
        bw = re.sub(r'XV', asp, bw)
        if bw.startswith('CONJ') and diac.startswith(('wa', 'sa', 'fa')):
            diac = diac[:2] + re.sub(r'aA', 'A', diac[2:])
        else:
            diac = re.sub(r'aA', 'A', diac)
        pos = analysis['pos']
        # IV-specific pre-processing
        bw = re.sub(r"IVSUFF_SUBJ:\dMP_MOOD:I", "IVSUFF_SUBJ:MP_MOOD:I", bw)
        bw = re.sub(r"IVSUFF_SUBJ:\d\w+_MOOD:([ISJ]+)", r"IVSUFF_MOOD:\1", bw)
        bw = bw.replace("SUB_CONJ", "PREP")
        
        pred_.append((diac, lex, bw, pos))
    pred = set(pred_)

    return pred

def recall_stats(preds, gold, func_name, inspection='all'):
    total = len(preds)
    correct = 0
    assert len(preds) == len(gold)
    for pred, gold_inst in tqdm(zip(preds, gold), total=len(preds), desc=func_name):
        form_dediac, pred_inst, analyzer = pred
        if {gold_inst}.intersection(pred_inst):
            correct += 1
        else:
            # Adds pred which is most similar to gold to `output_inspection`
            oov_failed = output_inspection.setdefault('recall', dict())
            form_dediac = ar2bw_translit.transliterate(form_dediac)
            most_similar = [None, 1000]
            if inspection == 'most_similar':
                for analysis in pred_inst:
                    ed = edit_distance(' '.join(analysis), ' '.join(gold_inst))
                    if ed < most_similar[1]:
                        most_similar = [analysis, ed]
            oov_failed.setdefault(analyzer, []).append(
                {'input': form_dediac, 'most_similar': most_similar[0],
                'output': pred_inst, 'gold': gold_inst})
    
    report = ""

    return f"{correct / total:.1%}", report

def oov_stats(preds, gold, func_name):
    total = len(preds)
    oov = 0
    for i, pred in enumerate(tqdm(preds, desc=func_name)):
        form_dediac, pred_inst, _ = pred
        if len(pred_inst) == 0:
            oov += 1
        else:
            oov_failed = output_inspection.setdefault('oov', dict())
            form_dediac = ar2bw_translit.transliterate(form_dediac)
            oov_failed.setdefault(pred[2], []).append((form_dediac, gold[i]))

    report = ""

    return f"{oov / total:.1%}", report


def predict(test_pv_uniq):
    preds_camel, preds_calima, gold = [], [], []
    for ex in test_pv_uniq:
        # Simulate real-world inference setting
        form_diac = ex[0].split(':')[1]
        form_dediac = dediac_bw(form_diac)
        form_dediac = bw2ar_translit.transliterate(form_dediac)
        # Generate analyses
        pred_camel = analyzer_camel.analyze(form_dediac)
        pred_calima = analyzer_calima.analyze(form_dediac)
        pred_camel = _postprocess_pred(pred_camel)
        pred_calima = _postprocess_pred(pred_calima)
        preds_camel.append((form_dediac, pred_camel, 'camel'))
        preds_calima.append((form_dediac, pred_calima, 'calima'))
        # join is to deal with bw tags which have ':' in the tag
        gold_inst = _postprocess_gold(ex)
        gold.append(gold_inst)
    
    return preds_camel, preds_calima, gold

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        test_set = f.read().split('\n--------------\n')
        test_set = [ex.split('\n') for ex in test_set if ex.startswith(";;WORD")]
        test_set = [tuple([field for i, field in enumerate(
            ex[4].split()) if 1 <= i <= 3 or i == 5]) for ex in test_set]
        test_pv = [ex for ex in test_set if "IV" in ex[2]]
        test_pv_uniq = Counter(test_pv)
    
    preds_camel, preds_calima, gold =  predict(test_pv_uniq)
    preds = {'CAMeL': preds_camel, 'CALIMA': preds_calima}

    # Evaluation
    eval_functions = {"OOV rate": oov_stats,
                      "Correct analysis recall rate": recall_stats}
    for func_name, eval_function in eval_functions.items():
        print()
        camel_result, camel_report = eval_function(preds['CAMeL'], gold, func_name)
        print(f"CAMeL DB:", camel_result)
        if camel_report:
            print(camel_report)
        calima_result, calima_report = eval_function(preds['CALIMA'], gold, func_name)
        print(f"CALIMA DB:", calima_result)
        if calima_report:
            print(calima_report)
    pass
