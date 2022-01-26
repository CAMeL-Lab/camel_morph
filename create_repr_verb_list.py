import argparse

from camel_tools.utils.charmap import CharMapper

from db_maker import gen_cmplx_morph_combs, read_morph_specs, _generate_stem

ar2bw = CharMapper.builtin_mapper('ar2bw')

def create_representative_verbs_list(specs_sheets, config_file, config_name, cmplx_morph, output_name):
    SHEETS, cond2class, _ = read_morph_specs(specs_sheets, config_file, config_name)
    MORPH, LEXICON, HEADER = SHEETS['morph'], SHEETS['lexicon'], SHEETS['header']
    _order = [line.split()[1:] for line in list(HEADER['Content']) if line.startswith('ORDER')][0]
    _defaults = [{f: d for f, d in [f.split(':') for f in line.split()[1:]]}
                 for line in list(HEADER['Content']) if line.startswith('DEFAULT')]
    _defaults = {d['pos']: d for d in _defaults}
    defaults = {'defaults': _defaults, 'order': _order}
    defaults_ = defaults['defaults']['verb']
    stem_feats = ['per', 'gen', 'num', 'mod', 'cas', 'enc0', 'prc0', 'prc1', 'prc2', 'prc3']
    defaults_ = {sf: defaults_[sf] for sf in stem_feats}
    defaults_['enc1'] = defaults_['enc0']

    cmplx_stem_classes = gen_cmplx_morph_combs(
        cmplx_morph, MORPH, LEXICON, cond2class)
    
    representative_verbs = []
    for cmplx_stems in cmplx_stem_classes.values():
        xconds = ' '.join([f['COND-S'] for f in cmplx_stems[0]])
        xcondt = ' '.join([f['COND-T'] for f in cmplx_stems[0]])
        xcondf = ' '.join([f['COND-F'] for f in cmplx_stems[0]])
        info = _generate_stem(cmplx_morph,
                              cmplx_stems[0],
                              xconds, xcondt, xcondf,
                              short_cat_map=None,
                              defaults=defaults_)
        info = [feat.split(':')[1] for feat in info[0]['feats'].split() if feat.split(':')[0] in ['lex', 'diac']]
        representative_verbs.append((ar2bw(info[1]), ar2bw(info[0]), xconds, xcondt))
    with open(output_name, 'w') as f:
        for info in representative_verbs:
            print(*info, file=f, sep=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-specs_sheets", required=True,
                        type=str, help="Excel spreadsheet containing all lexicon, morph, etc. sheets.")
    parser.add_argument("-cmplx_morph", required=True,
                        type=str, help="Complex stem/prefix/suffix class combination to generate the instances for.")
    parser.add_argument("-config_file", required=True,
                        type=str, help="Config file specifying which sheets to use from `specs_sheets`.")
    parser.add_argument("-config_name", required=True,
                        type=str, help="Name of the configuration to load from the config file.")
    parser.add_argument("-output_name", required=True,
                        type=str, help="Name of the file to output the representative lemmas to.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    create_representative_verbs_list(
        args.specs_sheets, args.config_file, args.config_name, args.cmplx_morph, args.output_name)
