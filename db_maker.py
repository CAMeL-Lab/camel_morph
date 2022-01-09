###########################################
#CalimaStar DB Maker
#Habash, Khalifa, 2020-2021
#
# This code takes in a linguistitically informed 
# morphology specification, and outputs a DB
# in the ALMOR format to be used in CamelTools
#
###########################################
#Notes: TBR
#DB File Reader
#ComplexAffix constructor
#AlmorDB printer
#Compatibility checker
###########################################
#Issues:
# Backoff
# Add CAMEL POS
# Alternate matches
# add all tokenizations... and other features
###########################################

import re
from tqdm import tqdm
import itertools

from camel_tools.utils.dediac import dediac_bw
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw

import pandas as pd
import numpy as np

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

verbose = False
# Pre-compiled regex statements for efficiency
# Statements in checkCompatibility
_INIT_UNDERSCORE = re.compile("^\_")
_SPACE_UNDERSCORE = re.compile(" \_")
_UNDERSCORE_SPACE = re.compile("\_ ")
# Statements in constructAlmorDB
_PLUS_UNDERSCORE = re.compile("\+\_")
_UNDERSCORE_PLUS_START = re.compile("^(\_\+)+")
_UNDERSCORE_or_PLUS = re.compile("[_\+]")
# Statements in createCat
_SPACE_or_PLUS_MANY = re.compile("[\s\_]+")

#Memoization dictionary for compatibilities for chaching
comp_memoi = {}
# Parsed sheets
About, Header, Order, Morph, Lexicon = None, None, None, None, None

###########################################
#Input File: XLSX containing specific worksheets: About, Header, Order, Morph, Lexicon
#inputfilename="CamelMorphDB-Nov-2021.xlsx"
# Primary function 
# e.g. makeDB("CamelMorphDB-Nov-2021.xlsx")
###########################################

def makeDB(inputfilename):
    About, Header, Order, Morph, Lexicon, outputfilename = readMorphSpec(inputfilename)
    db = constructAlmorDB()
    printAlmorDB(outputfilename, db)


def readMorphSpec(inputfilename):
    """Read Input file containing morphological specifications"""
    global About, Header, Order, Morph, Lexicon
    #Read the full CamelDB xlsx file
    FullSpec = pd.ExcelFile(inputfilename)
    #Identify the Params sheet, which specifies which sheets to read in the xlsx spreadsheet
    Params = pd.read_excel(FullSpec, 'Params')
    #Read all the components:
    #Issue - need to allow multiple Morph, lex sheets to be read.
    About = pd.read_excel(
        FullSpec, Params[Params['Component'] == 'About'].Sheetname.values[0])
    Header = pd.read_excel(
        FullSpec, Params[Params['Component'] == 'Header'].Sheetname.values[0])
    Order = pd.read_excel(
        FullSpec, Params[Params['Component'] == 'Order'].Sheetname.values[0])
    Morph = pd.read_excel(
        FullSpec, Params[Params['Component'] == 'Morph'].Sheetname.values[0])
    Lexicon = pd.read_excel(
        FullSpec, Params[Params['Component'] == 'Lexicon'].Sheetname.values[0])
    outputfilename = Params[Params['Component']
                            == 'Output'].Sheetname.values[0]

    #Process all the components:
    Order = Order[Order.DEFINE == 'ORDER']  # skip comments & empty lines
    Order = Order.replace(np.nan, '', regex=True)

    Morph = Morph[Morph.DEFINE == 'MORPH']  # skip comments & empty lines
    Morph = Morph.replace(np.nan, '', regex=True)
    Morph = Morph.replace('^\s+', '', regex=True)
    Morph = Morph.replace('\s+$', '', regex=True)
    Morph = Morph.replace('\s+', ' ', regex=True)

    # add FUNC and FEAT to COND-S
    Morph['COND-S'] = Morph['COND-S']
    Morph['COND-S'] = Morph['COND-S'].replace('[\[\]]', '', regex=True)
    # Replace spaces in BW and GLOSS with '#'
    Morph['BW'] = Morph['BW'].replace('\s+', '#', regex=True)
    Morph['GLOSS'] = Morph['GLOSS'].replace('\s+', '#', regex=True)

    # Replace spaces in BW and GLOSS with '#' skip comments & empty lines
    Lexicon = Lexicon[Lexicon.DEFINE == 'LEXICON']
    Lexicon = Lexicon.replace(np.nan, '', regex=True)
    Lexicon['BW'] = Lexicon['BW'].replace('\s+', '#', regex=True)
    Lexicon['GLOSS'] = Lexicon['GLOSS'].replace('\s+', '#', regex=True)
    Lexicon['COND-S'] = Lexicon['COND-S'].replace(' +', ' ', regex=True)
    Lexicon['COND-S'] = Lexicon['COND-S'].replace(' $', '', regex=True)
    Lexicon['COND-T'] = Lexicon['COND-T'].replace(' +', ' ', regex=True)
    Lexicon['COND-T'] = Lexicon['COND-T'].replace(' $', '', regex=True)

    # Retroactively generate the condFalse by creating the complementry distribution
    # of all the conditions within a single morpheme (all the allomorphs)
    # This is perfomed here once instead of on the fly.

    # Get all the CLASSes in Morph
    all_classes = Morph.CLASS.unique()
    all_classes = all_classes[all_classes != '_']

    # Get all the morphemes in Morph
    all_morphemes = Morph.FUNC.unique()
    all_morphemes = all_morphemes[all_morphemes != '_']
    # Go through each class
        
    for CLASS in all_classes:
        # Go through each morpheme
        for morpheme in all_morphemes: 

            # Get the unique list of the true conditions in all the allomorphs.
            # We basically want to convert the 'COND-T' column in to n columns
            #   where n = maximum number of conditions for a single allomorph
            condTrue_set = Morph[(Morph.FUNC == morpheme) & 
                                (Morph.CLASS == CLASS)]['COND-T'].str.split(pat=' ', expand=True)
            if condTrue_set.empty:
                continue
            condTrue_set = condTrue_set.replace(['_'], '')

            if len(condTrue_set.iloc[:][:]) == 1 and condTrue_set.iloc[0][0] == '':
                continue
            # Go through each column in the true conditions
            for col in condTrue_set:
              
                # if we are at the first columns
                if col == 0:
                    #TODO: uniq col[0] and remove fluff -> c_T
                    current_condTrue_set = __get_clean_set(condTrue_set[col])
                    # Go through each allomorph
                    __get_morph_condFalse(condTrue_set[col], current_condTrue_set, Morph)

                # we are in col > 0
                else:
                    #TODO: create temp_T by merging the the condTrue for col 0-col, put it in 
                    # col[0] and remove 1:col. This is basically to group allomorph with 
                    # similar general conditions
                    # Take the uniq of what is now col[0] and remove the fluff
                    if col == 1:
                        temp_condTrue_set = condTrue_set.copy()
                    else:
                        temp_condTrue_set = condTrue_set.copy()
                        temp_condTrue_set = temp_condTrue_set.replace([None], '')

                        temp_condTrue_set[0] = temp_condTrue_set.loc[:,:col-1].agg(' '.join, axis=1)
                        temp_condTrue_set = temp_condTrue_set.drop(temp_condTrue_set.columns[1:col], axis=1)
                        temp_condTrue_set.columns = range(temp_condTrue_set.shape[1])
                    if len(temp_condTrue_set.columns) == 1:
                        continue


                    cound_groups = temp_condTrue_set[0].unique()
                    cound_groups = cound_groups[cound_groups!= '_']
                    # Go through each group of allomorphs
                    for group in cound_groups:
                        current_condTrue_set = __get_clean_set(temp_condTrue_set[temp_condTrue_set[0] == group][1])
                        __get_morph_condFalse(temp_condTrue_set[temp_condTrue_set[0] == 
                                                                group][1], current_condTrue_set, Morph)
            
            for idx, morpheme_entry in Morph[(Morph.FUNC == morpheme) & (Morph.CLASS == CLASS)].iterrows():
                # If there is a negated condition in the set of true conditions for the 
                #   current allomorph, remove the negation and add it to the false 
                #   conditions list for the current alomorph.
                almrph_condTrue = Morph.loc[idx, 'COND-T'].split(' ')
                almrph_condFalse = Morph.loc[idx, 'COND-F'].split(' ')
                negated_cond = [x for x in almrph_condTrue if x.startswith('!')]
                if negated_cond:
                    for cond in negated_cond:
                        almrph_condFalse.append(cond[1:])
                        # remove from the true conditions
                        almrph_condTrue.remove(cond)
                almrph_condTrue = [y for y in almrph_condTrue if y not in ['', '_', 'else', None]]
                almrph_condFalse = [y for y in almrph_condFalse if y not in ['', '_', 'else', None]]
                Morph.loc[idx, 'COND-T'] = ' '.join(almrph_condTrue)
                Morph.loc[idx, 'COND-F'] = ' '.join(almrph_condFalse)

    return About, Header, Order, Morph, Lexicon, outputfilename
   

def createCat(XMorphType, XClass, XSet, XTrue, XFalse):
    """This function creates the category for matching using classes and conditions"""
    cat = XMorphType + _SPACE_or_PLUS_MANY.sub("#", XClass + " " + XSet + " T:" + XTrue)
    return cat

def __convert_BW_tag(BW_tag):
    """Convert BW tag from BW2UTF8"""
    if BW_tag == '':
        return BW_tag
    BW_elements = BW_tag.split('+')
    utf8_BW_tag = []
    for element in BW_elements:
        parts = element.split('/')
        if 'null' in parts[0]:
            BW_lex = parts[0]
        else:
            BW_lex = bw2ar(parts[0])
        BW_pos = parts[1]
        utf8_BW_tag.append('/'.join([BW_lex, BW_pos]))
    return '+'.join(utf8_BW_tag)

def constructAlmorDB():
    db = {}
    for orderIndex, order in Order.iterrows():
        db_ = _populate_db(order)
        db.update(db_)
    return db

def _populate_db(order):
    db = {}
    db['OUT:###ABOUT###'] = list(About['Content'])
    db['OUT:###HEADER###'] = list(Header['Content'])
    db['OUT:###PREFIXES###'] = {}
    db['OUT:###SUFFIXES###'] = {}
    db['OUT:###STEMS###'] = {}
    db['OUT:###TABLE AB###'] = {}
    db['OUT:###TABLE BC###'] = {}
    db['OUT:###TABLE AC###'] = {}

    print(order["VAR"], order["COND-T"],
              order["PREFIX"], order["STEM"], order["SUFFIX"], sep=" ; ", end='\n')

    prefix_classes = expandSeq(order['PREFIX'], Morph)
    suffix_classes = expandSeq(order['SUFFIX'], Morph)
    stem_classes = expandSeq(order['STEM'], Lexicon)

    # pbar = tqdm(total=sum(1 for _, stems in stem_classes.items() for _ in stems))
    for stem_class, stem_combs in tqdm(stem_classes.items()):  # LEXICON
        #`stem_class` = (stem['COND-S'], stem['COND-T'], stem['COND-F'])
        # len(stem_comb) = 1 always
        xconds = stem_combs[0][0]['COND-S']
        xcondt = stem_combs[0][0]['COND-T']
        xcondf = stem_combs[0][0]['COND-F']

        for prefix_class, prefix_combs in prefix_classes.items():
            pconds = ' '.join([f['COND-S'] for f in prefix_combs[0]])
            pcondt = ' '.join([f['COND-T'] for f in prefix_combs[0]])
            pcondf = ' '.join([f['COND-F'] for f in prefix_combs[0]])

            for suffix_class, suffix_combs in suffix_classes.items():
                sconds = ' '.join([f['COND-S'] for f in suffix_combs[0]])
                scondt = ' '.join([f['COND-T'] for f in suffix_combs[0]])
                scondf = ' '.join([f['COND-F'] for f in suffix_combs[0]])

                valid = checkCompatibility(' '.join([pconds, xconds, sconds]),
                                        ' '.join([pcondt, xcondt, scondt]),
                                        ' '.join([pcondf, xcondf, scondf]))
                if valid:
                    if verbose:
                        print(valid, ":", '+'.join([m['FORM']
                            for m in prefix_combs[0]]) + stem_combs[0][0]['FORM'] + '+'.join(
                            [m['FORM'] for m in suffix_combs[0]]))
                    
                    combination = _read_combination(
                        prefix_combs[0], suffix_combs[0], stem_combs[0][0],
                        pconds, pcondt, pcondf, sconds, scondt, scondf, xconds, xcondt, xcondf)
                    db = _update_db(combination, db)

                    for stem in stem_combs[1:]:
                        stem_entry, _ = _generate_stem(stem[0], xconds, xcondt, xcondf)
                        db['OUT:###STEMS###'][stem_entry] = 1
                    for prefix in prefix_combs[1:]:
                        prefix_entry, _ = _generate_prefix(prefix, pconds, pcondt, pcondf)
                        db['OUT:###PREFIXES###'][prefix_entry] = 1
                    for suffix in suffix_combs[1:]:
                        suffix_entry, _ = _generate_suffix(suffix, sconds, scondt, scondf)
                        db['OUT:###SUFFIXES###'][suffix_entry] = 1
    return db

def _update_db(combination, db):
    db['OUT:###PREFIXES###'][combination["prefix"]] = 1
    db['OUT:###SUFFIXES###'][combination["suffix"]] = 1
    db['OUT:###STEMS###'][combination["stem"]] = 1
    db['OUT:###TABLE AB###'][combination["table_ab"]] = 1
    db['OUT:###TABLE BC###'][combination["table_bc"]] = 1
    db['OUT:###TABLE AC###'][combination["table_ac"]] = 1
    return db

def _read_combination(prefix, suffix, stem,
                      pconds, pcondt, pcondf,
                      sconds, scondt, scondf,
                      xconds, xcondt, xcondf):
    prefix, pcat = _generate_prefix(prefix, pconds, pcondt, pcondf)
    suffix, scat = _generate_suffix(suffix, sconds, scondt, scondf)
    stem, xcat = _generate_stem(stem, xconds, xcondt, xcondf)
    
    combination = dict(prefix=prefix,
                       suffix=suffix,
                       stem=stem,
                       table_ab=pcat+" "+xcat,
                       table_bc=xcat+" "+scat,
                       table_ac=pcat+" "+scat)
    return combination

def _generate_prefix(prefix, pconds, pcondt, pcondf):
    pclass, pmatch, pdiac, pgloss, pfeat, pbw = _read_prefix(prefix)
    pcat = createCat("P:", pclass, pconds, pcondt, pcondf)
    ar_pbw = __convert_BW_tag(pbw)
    prefix = bw2ar(pmatch) + '\t' + pcat + '\t' + 'diac:' + bw2ar(pdiac) + \
        ' bw:' + ar_pbw + ' gloss:' + pgloss.strip() + ' ' + pfeat.strip()
    return prefix, pcat

def _generate_suffix(suffix, sconds, scondt, scondf):
    sclass, smatch, sdiac, sgloss, sfeat, sbw = _read_suffix(suffix)
    scat = createCat("S:", sclass, sconds, scondt, scondf)
    ar_sbw = __convert_BW_tag(sbw)
    suffix = bw2ar(smatch) + '\t' + scat + '\t' + 'diac:' + bw2ar(sdiac) + \
                ' bw:' + ar_sbw + ' gloss:' + sgloss.strip() + ' ' + sfeat.strip()
    return suffix, scat

def _generate_stem(stem, xconds, xcondt, xcondf):
    xclass, xmatch, xdiac, xlex, xgloss, xfeat, xbw = _read_stem(stem)
    xcat = createCat("X:", xclass, xconds, xcondt, xcondf)
    ar_xbw = __convert_BW_tag(xbw)
    stem = bw2ar(xmatch) + '\t' + xcat + '\t' + 'diac:' + bw2ar(xdiac) + \
            ' bw:' + ar_xbw + ' lex:' + bw2ar(xlex) + ' gloss:' + \
            xgloss.strip() + ' ' + xfeat.strip()
    return stem, xcat


def _read_prefix(prefix):
    pclass = '+'.join([m['CLASS'] for m in prefix])
    pbw = '+'.join([m['BW'] for m in prefix])
    pform = '+'.join([m['FORM'] for m in prefix])
    pgloss = '+'.join([m['GLOSS'] for m in prefix])
    pfeat = ' '.join([m['FEAT'] for m in prefix])
    pbw = _PLUS_UNDERSCORE.sub("", _UNDERSCORE_PLUS_START.sub("", pbw))
    if pbw == '_':
        pbw = ''
    pgloss = _PLUS_UNDERSCORE.sub("", _UNDERSCORE_PLUS_START.sub("", pgloss))
    pdiac = _UNDERSCORE_or_PLUS.sub("", pform)
    pmatch = normalize_alef_bw(normalize_alef_maksura_bw(
        normalize_teh_marbuta_bw(dediac_bw(pdiac))))
    return pclass, pmatch, pdiac, pgloss, pfeat, pbw

def _read_suffix(suffix):
    sclass = '+'.join([m['CLASS'] for m in suffix])
    sbw = '+'.join([m['BW'] for m in suffix])
    sform = '+'.join([m['FORM'] for m in suffix])
    sgloss = '+'.join([m['GLOSS'] for m in suffix])
    sfeat = ' '.join([m['FEAT'] for m in suffix])
    sbw = _PLUS_UNDERSCORE.sub("", _UNDERSCORE_PLUS_START.sub("", sbw))
    if sbw == '_':
        sbw = ''
    sgloss = _PLUS_UNDERSCORE.sub("", _UNDERSCORE_PLUS_START.sub("", sgloss))
    sdiac = _UNDERSCORE_or_PLUS.sub("", sform)
    smatch = normalize_alef_bw(normalize_alef_maksura_bw(
        normalize_teh_marbuta_bw(dediac_bw(sdiac))))
    return sclass, smatch, sdiac, sgloss, sfeat, sbw

def _read_stem(stem):
    xbw = stem['BW']
    xclass = stem['CLASS']
    xform = stem['FORM']
    xgloss = stem['GLOSS']
    xlex = stem['LEMMA'].split(':')[1]
    xfeat = stem['FEAT'].strip()
    xbw = _PLUS_UNDERSCORE.sub("", _UNDERSCORE_PLUS_START.sub("", xbw))
    xdiac = _UNDERSCORE_or_PLUS.sub("", xform)
    xmatch = normalize_alef_bw(normalize_alef_maksura_bw(
        normalize_teh_marbuta_bw(dediac_bw(xdiac))))
    return xclass, xmatch, xdiac, xlex, xgloss, xfeat, xbw


def printAlmorDB(outputfilename, db):
    """Create output file in ALMOR DB format"""

    fout = open(outputfilename, "w")
    
    # for x in db['OUT:###ABOUT###']:
    #     fout.write(x+"\n")
    
    for x in db['OUT:###HEADER###']:
        fout.write(x + "\n")

    fout.write("###PREFIXES###\n")
    for x in db['OUT:###PREFIXES###'].keys():
        fout.write(x + "\n")
        
    fout.write("###SUFFIXES###\n")
    for x in db['OUT:###SUFFIXES###'].keys():
        fout.write(x + "\n")
        
    fout.write("###STEMS###\n")
    for x in db['OUT:###STEMS###'].keys():
        fout.write(x + "\n")
        
    fout.write("###TABLE AB###\n")
    for x in db['OUT:###TABLE AB###'].keys():
        fout.write(x + "\n")
        
    fout.write("###TABLE BC###\n")
    for x in db['OUT:###TABLE BC###'].keys():
        fout.write(x + "\n")
        
    fout.write("###TABLE AC###\n")
    for x in db['OUT:###TABLE AC###'].keys():
        fout.write(x + "\n")
        
    fout.close()

def expandSeq(MorphClass, Morph, pruning=True):
    """This function exands the specification of Morph Classes
    into all possible combinations of specific Morphemes.
    Input is space separated class name: e.g., '[QUES] [CONJ] [PREP]'
    Ouput is a list of lists of dictionaries; where each dictionary
    specifies a morpheme; the list order of the dictionaries indicate 
    order of morphemes.
    In other words, it generates a Cartesian product of their combinations
    as specified by the allowed Morph Classes"""
    MorphClasses = [[m.to_dict() for i, m in Morph[Morph.CLASS == c].iterrows()]
                        for c in MorphClass.split()]
    MorphSeqs = [list(t) for t in itertools.product(*MorphClasses)]
    MorphSeqsCategorized = {}
    for seq in MorphSeqs:
        seq_conds_cat = [(position['COND-S'], position['COND-T'], position['COND-F'])
                            for position in seq]
        MorphSeqsCategorized.setdefault(tuple(seq_conds_cat), []).append(seq)
    
    if pruning:
        # Prune out incoherent classes
        MorphSeqsCategorized_ = {}
        for seq_class, seq_instances in MorphSeqsCategorized.items():
            cond_s_seq = {
                cond for part in seq_class for cond in part[0].split() if cond}
            cond_f_seq = {
                cond for part in seq_class for cond in part[2].split() if cond}
            if cond_s_seq.intersection(cond_f_seq) == set():
                MorphSeqsCategorized_[seq_class] = seq_instances
        MorphSeqsCategorized = MorphSeqsCategorized_
    
    return MorphSeqsCategorized

#Remember to eliminate all non match affixes/stems
def checkCompatibility (condSet,condTrue,condFalse):
    #order cond: order-lexicon pairing? pos
    #Cond True is an anded list of ORs (||) 

    #Truth conditions:
    # AND : A B C
    # OR  : A||B||C
    # combined: A B||C

    #some limitations here.
    
    #TODO: optimize by caching; by breaking when no point of continuing.
    key = f'{condSet}\t{condTrue}\t{condFalse}'
    if key in comp_memoi:
      return comp_memoi[key]
    else:
      comp_memoi[key] = ''
    # Remove all nil items (indicated as "_")
    cs = _INIT_UNDERSCORE.sub("", _SPACE_UNDERSCORE.sub(
        "", _UNDERSCORE_SPACE.sub("", condSet))).split()
    ct = _INIT_UNDERSCORE.sub("", _SPACE_UNDERSCORE.sub(
        "", _UNDERSCORE_SPACE.sub("", condTrue))).split()
    cf = _INIT_UNDERSCORE.sub("", _SPACE_UNDERSCORE.sub(
        "", _UNDERSCORE_SPACE.sub("", condFalse))).split()

    cs.sort()
    ct.sort()
    cf.sort()

    valid = True

    # Things that need to be true
    for t in ct:
        #OR condition check
        validor = False
        for ort in t.split('||'):
            validor = validor or ort in cs

        #AND Check
        valid = valid and validor
        if not valid:  # abort if we hit an invalid condition
            comp_memoi[key] = valid
            return valid

    # Things that need to be false
    for f in cf:
        for orf in f.split('||'):
            valid = valid and orf not in cs
        if not valid:  # abort if we hit an invalid condition
            comp_memoi[key] = valid
            return valid

    comp_memoi[key] = valid
    return valid                     
                                   
def __get_condFalse(all_condTrue, almrph_condTrue):
  # The false conditions for the current allomorph is whatever true condition
  #   in the morpheme set that does not belong to the set of true conditions 
  #   to the current allomorph.

  almrph_condFalse = all_condTrue - set(almrph_condTrue)

  # If there is a negated condition in the set of true conditions for the 
  #   current allomorph, remove the negation and add it to the false 
  #   conditions list for the current alomorph.
  # negated_cond = [x for x in almrph_condTrue if x.startswith('!')]
  # if negated_cond:
  #   for cond in negated_cond:
  #     almrph_condFalse.add(cond[1:])
  #     # remove from the true conditions
  #     almrph_condTrue.remove(cond)
  
  # If we end up with no false condition, set it to '_'
  if not almrph_condFalse: almrph_condFalse = ['_']
  # If we end up with no true condition, set it to '_'
  if not almrph_condTrue: almrph_condTrue = ['_']

  return almrph_condFalse, almrph_condTrue


def __get_morph_condFalse(morph_condTrue, current_condTrue_set, Morph):
  # Go through each allomorph
  for idx, entry in morph_condTrue.iteritems():
    
    # If we have no true condition for the allomorph (aka can take anything)
    if entry is None:
      continue
    elif entry != 'else':
      #TODO: create condFalse for the morpheme by c_T - entry
      almrph_condFalse, almrph_condTrue = __get_condFalse(
          current_condTrue_set, [entry])

    elif entry == 'else':
      # TODO: condFalse = c_T
      almrph_condFalse = current_condTrue_set
      almrph_condTrue = ['_']
    
    # Finally, populate the 'COND-F' cell with the false conditions
    Morph.loc[idx, 'COND-F'] = Morph.loc[idx, 'COND-F'] + \
        ' ' + ' '.join(almrph_condFalse)
    Morph.loc[idx, 'COND-F'] = Morph.loc[idx, 'COND-F'].replace('_', '')
  pass

def __get_clean_set(cnd_col):
  mrph_condTrue = cnd_col.tolist()
  mrph_condTrue = [y for y in mrph_condTrue if y not in ['', '_', 'else', None]]
  mrph_condTrue = set(mrph_condTrue)  # make it a set
  
  return mrph_condTrue
