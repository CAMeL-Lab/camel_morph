from itertools import zip_longest
import re

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.generator import Generator

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

def __split_BW_tag(BW_tag):
  """
  this function takes a BW tag and splits it into the lexical part and the POS
  tag part for easier analysis
  """
  if BW_tag == '':
    return BW_tag
  BW_elements = BW_tag.split('+')
  lex_BW_tag = []
  pos_BW_tag = []
  for element in BW_elements:
      parts = element.split('/')
      if 'null' in parts[0]:
          BW_lex = parts[0]
      else:
          BW_lex = bw2ar(parts[0])
      BW_pos = parts[1]
      lex_BW_tag.append(BW_lex)
      pos_BW_tag.append(BW_pos)

  return '+'.join(lex_BW_tag), '+'.join(pos_BW_tag)


def __comp_analyses(analyses, anls_dict, lemma, mode):
  """
  this function takes a list of analyses and compresses them in CAMELPOS like
  format (all the same except for the aspect where a V is added for readability)
  """

  feats = ['diac', 'lex', 'bw']  # , 'pos'
  comp_feats = ['asp', 'per', 'gen', 'num', 'vox', 'mod', 'enc0']
  feat_translit = ['diac', 'lex', 'bw']

  anls_dict[lemma]['anls'] = []
  anls_dict[lemma]['anls_lax'] = []
  for anls in analyses:
    word = ar2bw(anls['diac'])
    bw = ''
    c_anls = ''
    for feat in feat_translit:
      if feat == 'bw':
        anls[feat] = re.sub('/([PIC]V)\+(\S+)/STEM', r'\2/\1', anls[feat])
        anls[feat] = re.sub('\+(\S+)/STEM\+(\S+/[PIC]V)', r'+\1\2', anls[feat])
        if 'XVSUFF' in anls[feat]:
          anls[feat] = re.sub('XV', anls['asp'].upper()+'V', anls[feat])
        bw = ar2bw(anls[feat])
      # print(ar2bw(anls[feat]), end = '\t')

    for feat in comp_feats:
      if feat == 'asp':
        # print(anls[feat].upper()+'V', end = '')
        c_anls = c_anls + anls[feat].upper()+'V'
      elif feat == 'enc0':
        if feat in anls:
          if anls[feat] != '0':
            # print('+' + anls[feat].split('_')[0].upper(), end = '')
            c_anls = c_anls + '+PRON/' + anls[feat].split('_')[0].upper()
        else:
          continue
      elif feat == 'vox':
        c_anls = c_anls + '.' + anls[feat].upper()
      elif anls['asp'] == 'p' and feat == 'mod':
        c_anls = c_anls + 'U'
      else:
        # print(anls[feat].upper(), end = '')
        c_anls = c_anls + anls[feat].upper()
    # print()
    c_anls = re.sub('1M([SP])', r'1U\1', c_anls)
    c_anls = re.sub('2MD', r'2UD', c_anls)
    lex_bw, pos_bw = __split_BW_tag(bw)
    anls_dict[lemma]['anls'].append(f'{c_anls}\t{word}\t{lex_bw}\t{pos_bw}')
    anls_dict[lemma]['anls_lax'].append(
        f'{c_anls}\t{word.replace("o", "")}\t{lex_bw.replace("o", "")}\t{pos_bw.lower()}')

    if mode == 'SAMA':
      if c_anls.startswith('IV') and ('.AU' in c_anls):
        c_anls_j = re.sub('AU', 'AJ', c_anls)
        c_anls_s = re.sub('AU', 'AS', c_anls)
        if c_anls_j not in anls_dict[lemma]:
          anls_dict[lemma][c_anls_j] = {}
        if c_anls_s not in anls_dict[lemma]:
          anls_dict[lemma][c_anls_s] = {}
        anls_dict[lemma][c_anls_j][word.replace(
            "o", "")] = f'{word}\t{ar2bw(lex_bw)}\t{pos_bw}'
        anls_dict[lemma][c_anls_s][word.replace(
            "o", "")] = f'{word}\t{ar2bw(lex_bw)}\t{pos_bw}'
        continue

    if c_anls not in anls_dict[lemma]:
      anls_dict[lemma][c_anls] = {}
    anls_dict[lemma][c_anls][word.replace(
        "o", "")] = f'{word}\t{ar2bw(lex_bw)}\t{pos_bw}'

    # anls_dict[lemma][c_anls]['word'] = word
    # anls_dict[lemma][c_anls]['bw'] = bw


def __generate(lemma, generator):
  """
  this function take a lemma and a generator instance and generate analyses
  according to the below features. it generates the analyses for each set of
  features and concatenate it to the the same list.
  """
  features = {
      'pos': 'verb',
      'asp': asp,
      'vox': vox
  }
  features_1s = {
      'pos': 'verb',
      'enc0': '1s_dobj',
      'asp': asp,
      'vox': vox
  }
  features_1p = {
      'pos': 'verb',
      'enc0': '1p_dobj',
      'asp': asp,
      'vox': vox
  }
  features_2ms = {
      'pos': 'verb',
      'enc0': '2ms_dobj',
      'asp': asp,
      'vox': vox
  }
  features_2fs = {
      'pos': 'verb',
      'enc0': '2fs_dobj',
      'asp': asp,
      'vox': vox
  }
  features_3ms = {
      'pos': 'verb',
      'enc0': '3ms_dobj',
      'asp': asp,
      'vox': vox
  }
  features_3fs = {
      'pos': 'verb',
      'enc0': '3fs_dobj',
      'asp': asp,
      'vox': vox
  }
  features_2mp = {
      'pos': 'verb',
      'enc0': '2mp_dobj',
      'asp': asp,
      'vox': vox
  }
  features_2fp = {
      'pos': 'verb',
      'enc0': '2fp_dobj',
      'asp': asp,
      'vox': vox
  }
  features_3mp = {
      'pos': 'verb',
      'enc0': '3mp_dobj',
      'asp': asp,
      'vox': vox
  }
  features_3fp = {
      'pos': 'verb',
      'enc0': '3fp_dobj',
      'asp': asp,
      'vox': vox
  }
  features_2d = {
      'pos': 'verb',
      'enc0': '2d_dobj',
      'asp': asp,
      'vox': vox
  }
  features_3d = {
      'pos': 'verb',
      'enc0': '3d_dobj',
      'asp': asp,
      'vox': vox
  }

  analyses = generator.generate(lemma, features)
  analyses.extend(generator.generate(lemma, features_1s))
  analyses.extend(generator.generate(lemma, features_1p))
  analyses.extend(generator.generate(lemma, features_3ms))
  analyses.extend(generator.generate(lemma, features_3fs))
  analyses.extend(generator.generate(lemma, features_3mp))
  analyses.extend(generator.generate(lemma, features_3fp))
  analyses.extend(generator.generate(lemma, features_2ms))
  analyses.extend(generator.generate(lemma, features_2fs))
  analyses.extend(generator.generate(lemma, features_2mp))
  analyses.extend(generator.generate(lemma, features_2fp))
  analyses.extend(generator.generate(lemma, features_2d))
  analyses.extend(generator.generate(lemma, features_3d))

  return analyses


def generate_inf(db_name, og_calima):
  """
  this is the main function that calls the other functions and does the final 
  printing of the analyses of two given databases.
  """
  new_anls = {}
  og_anls = {}
  db = MorphologyDB(db_name, flags='g')
  og_db = MorphologyDB(og_calima, flags='g')
  generator = Generator(db)
  og_generator = Generator(og_db)

  lemmas = db.lemma_hash
  output_file = open(f'temp_PAN_generation_anls.tsv', 'w')
  output_file.write(
      f'lemma\tPOS\tPAN-diac\tPAN-BW_lex\tPAN-BW_pos\tSAMA-diac\tSAMA-BW_lex\tSAMA-BW_pos\tdiac_match\tbw_lex_match\tbw_pos_match\tanls_comb_rep\tstatus\n')
  for lemma in lemmas:
    new_anls[lemma] = {}
    og_anls[lemma] = {}

    new_analyses = __generate(lemma, generator)
    og_analyses = __generate(lemma, og_generator)

    print(ar2bw(lemma))
    print('############')
    # output_file.write('############\n')
    __comp_analyses(new_analyses, new_anls, lemma, 'PAN')
    __comp_analyses(og_analyses, og_anls, lemma, 'SAMA')

    output_file.write(
        f'{ar2bw(lemma)}\t# of Analysis\t PAN: {len(new_anls[lemma]["anls"])}\t')
    output_file.write(f'SAMA {len(og_anls[lemma]["anls"])}\n')

    shared_anls = set(new_anls[lemma]["anls"]).intersection(
        set(og_anls[lemma]["anls"]))
    shared_anls_lax = set(new_anls[lemma]["anls_lax"]).intersection(
        set(og_anls[lemma]["anls_lax"]))
    output_file.write(
        f'{ar2bw(lemma)}\tShared Analysis\t (exact): {len(shared_anls)}\t')
    output_file.write(f'(lax): {len(shared_anls_lax)}\n')

    # if not new_analyses or not og_analyses:
    #   output_file.write('############\n')
    #   continue
    all_anls = set(new_anls[lemma]["anls"]).union(set(og_anls[lemma]["anls"]))

    anls_in_new_only = list(set(new_anls[lemma]["anls"]) - shared_anls)
    anls_in_calima_only = list(set(og_anls[lemma]["anls"]) - shared_anls)

    all_feat_comb = set(new_anls[lemma].keys()).union(
        set(og_anls[lemma].keys()))
    all_feat_comb.remove('anls')
    all_feat_comb.remove('anls_lax')

    for feat_comb in all_feat_comb:
      # new_only = [x for x in anls_in_new_only if feat_comb == x.split('\t')[0]]
      # calima_only = [x for x in anls_in_calima_only if feat_comb == x.split('\t')[0]]
      new_only = []
      calima_only = []
      # pan_words = new_anls[lemma][feat_comb].keys()
      calima_words = []
      if feat_comb in og_anls[lemma]:
        calima_words = og_anls[lemma][feat_comb].keys()

      if feat_comb in new_anls[lemma]:
        for wrd in new_anls[lemma][feat_comb]:
          w_match = 'na'
          bw_lex_match = 'na'
          bw_pos_match = 'na'
          check = 'No match'
          num_anls_comb = len(new_anls[lemma][feat_comb])
          if wrd in calima_words:
            if new_anls[lemma][feat_comb][wrd].split('\t')[0] == og_anls[lemma][feat_comb][wrd].split('\t')[0]:
              w_match = 'exact'
            else:
              w_match = 'lax'
            if new_anls[lemma][feat_comb][wrd].split('\t')[1] == og_anls[lemma][feat_comb][wrd].split('\t')[1]:
              bw_lex_match = 'exact'
            elif new_anls[lemma][feat_comb][wrd].split('\t')[1].replace('o', '') == og_anls[lemma][feat_comb][wrd].split('\t')[1].replace('o', ''):
              bw_lex_match = 'lax'
            if new_anls[lemma][feat_comb][wrd].split('\t')[2] == og_anls[lemma][feat_comb][wrd].split('\t')[2]:
              bw_pos_match = 'exact'
            elif new_anls[lemma][feat_comb][wrd].split('\t')[2].lower() == og_anls[lemma][feat_comb][wrd].split('\t')[2].lower():
              bw_pos_match = 'lax'
            if w_match == bw_lex_match == bw_pos_match == 'exact':
              check = 'good'
            if len(new_anls[lemma][feat_comb]) > 1:
              check = 'Overgeneration'
            elif check == 'good':
              check = 'match'
            elif w_match in ['lax', 'exact']:
              check = 'match'
            elif w_match == 'na':
              check = 'No match'
            output_file.write(
                f'{ar2bw(lemma)}\t{feat_comb}\t{new_anls[lemma][feat_comb][wrd]}\t{og_anls[lemma][feat_comb][wrd]}\t{w_match}\t{bw_lex_match}\t{bw_pos_match}\t\t{check}\n')
            og_anls[lemma][feat_comb].pop(wrd)
          else:
            new_only.append(f'{new_anls[lemma][feat_comb][wrd]}')

      if feat_comb in og_anls[lemma]:
        for wrd in og_anls[lemma][feat_comb]:
          calima_only.append(f'{og_anls[lemma][feat_comb][wrd]}')

      if not new_only:
          check = 'No Generation'
      elif not calima_only:
          check = 'No SAMA'
      for a, b in zip_longest(new_only, calima_only, fillvalue='\t\t'):
        if b == '\t\t':
          check = 'No SAMA'
        elif a == '\t\t' and feat_comb in new_anls[lemma] and len(new_anls[lemma][feat_comb]) == 1:
          check = 'SAMA overgeneration'
        elif a == '\t\t' and feat_comb not in new_anls[lemma]:
          chack = 'No Generation'
        output_file.write(
            f'{ar2bw(lemma)}\t{feat_comb}\t{a}\t{b}\tna\tna\tna\t\t{check}\n')
      # for a, b in zip_longest(new_only,calima_only, fillvalue='\t\t'):
      #   a = '\t'.join(a.split('\t')[1:])
      #   b = '\t'.join(b.split('\t')[1:])
      #   output_file.write(f'{ar2bw(lemma)}\t{feat_comb}\t{a}\t{b}\n')

      # print(len(all_feat_comb))

    print('############')
    output_file.write('############\n')
  output_file.close()


#####Generate inflections######
asp_map = {'PV': 'p', 'IV': 'i', 'CV': 'c'}
vox = 'a'
asp = asp_map['PV']

generate_inf(f'XYZ.db', '')
