import argparse
import json

from camel_tools.utils.charmap import CharMapper

import db_maker

ar2bw = CharMapper.builtin_mapper('ar2bw')

def create_representative_verbs_list(input_filename,
                                     config_file,
                                     config_name,
                                     cmplx_morph_seq,
                                     pos_type,
                                     output_name):
    with open(config_file) as f:
        config = json.load(f)[config_name]
    SHEETS, cond2class = db_maker.read_morph_specs(input_filename, config)
    MORPH, LEXICON, HEADER = SHEETS['morph'], SHEETS['lexicon'], SHEETS['header']
    
    defaults = db_maker._process_defaults(list(HEADER['Content']))
    required_feats = db_maker._choose_required_feats(pos_type)

    cmplx_stem_classes = db_maker.gen_cmplx_morph_combs(
        cmplx_morph_seq, MORPH, LEXICON, cond2class,
        pruning_cond_s_f=False, pruning_same_class_incompat=False)
    
    representative_verbs = []
    for cmplx_stems in cmplx_stem_classes.values():
        stem_cond_s = ' '.join([f['COND-S'] for f in cmplx_stems[0]])
        stem_cond_t = ' '.join([f['COND-T'] for f in cmplx_stems[0]])
        stem_cond_f = ' '.join([f['COND-F'] for f in cmplx_stems[0]])
        info = db_maker._generate_stem(cmplx_morph_seq,
                                       required_feats,
                                       cmplx_stems[0],
                                       stem_cond_s, stem_cond_t, stem_cond_f,
                                       short_cat_map=None,
                                       defaults=defaults['defaults'])
        info = [feat.split(':')[1] for feat in info[0]['feats'].split()
                if feat.split(':')[0] in ['lex', 'diac']]
        representative_verbs.append((ar2bw(info[1]), ar2bw(info[0]), stem_cond_s, stem_cond_t))
    with open(output_name, 'w') as f:
        for info in representative_verbs:
            print(*info, file=f, sep=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-specs_sheets", required=True,
                        type=str, help="Path of Excel spreadsheet containing all lexicon, morph, etc. sheets.")
    parser.add_argument("-cmplx_morph", required=True,
                        type=str, help="Complex stem/prefix/suffix class combination to generate the instances for.")
    parser.add_argument("-pos_type", required=True, choices=['verbal', 'nominal'],
                        type=str, help="POS type of the lemmas for which we want to generate a representative sample.")
    parser.add_argument("-config_file", required=True,
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-output_name", required=True,
                        type=str, help="Name of the file to output the representative lemmas to.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    create_representative_verbs_list(input_filename=args.specs_sheets,
                                     config_file=args.config_file,
                                     config_name=args.config_name,
                                     cmplx_morph_seq=args.cmplx_morph,
                                     pos_type=args.pos_type,
                                     output_name=args.output_name)
