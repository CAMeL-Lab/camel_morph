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

from camel_tools.utils.dediac import dediac_bw
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_alef_bw, normalize_alef_maksura_bw, normalize_teh_marbuta_bw

import pandas as pd
import numpy as np

bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')

#verbose = True
verbose = False

### pre compiled regex statements for efficiency

# statements in checkCompatibility
_INIT_UNDERSCORE = re.compile("^\_")
_SPACE_UNDERSCORE = re.compile(" \_")
_UNDERSCORE_SPACE = re.compile("\_ ")


### statements in constructAlmorDB
_PLUS_UNDERSCORE = re.compile("\+\_")
_UNDERSCORE_PLUS_START = re.compile("^(\_\+)+")
_UNDERSCORE_or_PLUS = re.compile("[_\+]")

### statements in createCat
_SPACE_or_PLUS_MANY = re.compile("[\s\_]+")



#/gdrive/MyDrive/\ NYUAD-CAMEL/Projects/Pan-Arabic-Analyzer/CSTAR-PanArab/CODE/CSTAR-DB-Jan-2021-Exp.xlsx
###########################################
#Input File: XLSX containing specific worksheets: About, Header, Order, Morph, Lexicon 
#inputfilename="CamelMorphDB-Nov-2021.xlsx"
###########################################


###########################################
#Memoization dictionary for compatibilities for chaching
comp_memoi = {}


###########################################
# Primary function 
# e.g. makeDB("CamelMorphDB-Nov-2021.xlsx")
###########################################

def makeDB(inputfilename):
    About, Header, Order, Morph, Lexicon, outputfilename = readMorphSpec(inputfilename)
    db = constructAlmorDB(About, Header, Order, Morph, Lexicon)
    printAlmorDB(outputfilename, db)

###########################################
#Read Input file containing morphological specifications
###########################################


def readMorphSpec(inputfilename):

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
    Morph['COND-S'] = Morph['COND-S'] + ' ' + Morph['FUNC'] + ' ' + Morph['FEAT']
    Morph['COND-S'] = Morph['COND-S'].replace('[\[\]]', '', regex=True)
    #Expand-ElseConditions
    # Replace spaces in BW and GLOSS with '#'
    Morph['BW'] = Morph['BW'].replace('\s+', '#', regex=True)
    Morph['GLOSS'] = Morph['GLOSS'].replace('\s+', '#', regex=True)

    # Replace spaces in BW and GLOSS with '#'
    # skip comments & empty lines
    Lexicon = Lexicon[Lexicon.DEFINE == 'LEXICON']
    Lexicon = Lexicon.replace(np.nan, '', regex=True)
    Lexicon['BW'] = Lexicon['BW'].replace('\s+', '#', regex=True)
    Lexicon['GLOSS'] = Lexicon['GLOSS'].replace('\s+', '#', regex=True)
    Lexicon['COND-S'] = Lexicon['COND-S'].replace(' +', ' ', regex=True)
    Lexicon['COND-S'] = Lexicon['COND-S'].replace(' $', '', regex=True)
    Lexicon['COND-T'] = Lexicon['COND-T'].replace(' +', ' ', regex=True)
    Lexicon['COND-T'] = Lexicon['COND-T'].replace(' $', '', regex=True)

    # ############SALAM##############
    ###########################################
    # Retroactively generate the condFalse by creating the complementry distribution
    # of all the conditions within a single morpheme (all the allomorphs)
    # This is perfomed here once instead of on the fly.
    ###########################################
    # Get all the CLASSes in Morph
    all_classes = Morph.CLASS.unique()
    all_classes = all_classes[all_classes != '_']

    # Get all the morphemes in Morph
    all_morphemes = Morph.FUNC.unique()
    all_morphemes = all_morphemes[all_morphemes != '_']
    # Go through each class
    # with open('test_morph.tsv', 'a') as f:
    #     for classs, cond in zip(Morph['FUNC'], Morph['COND-T']):
    #         print(classs, cond, file=f, sep='\t')

        
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
                    #print(current_condTrue_set)
                    # Go through each allomorph
                    __get_morph_condFalse(condTrue_set[col], current_condTrue_set, Morph)

                # we are in col > 0
                else:
                    #TODO: create temp_T by merging the the condTrue for col 0-col, put it in 
                    #      col[0] and remove 1:col. This is basically to group allomorph with 
                    #       similar general conditions
                    # Take the uniq of what is now col[0] and remove the fluff
                    if col == 1:# and len(condTrue_set.columns) == 2:
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
                    # print(temp_condTrue_set)
                    # print(cound_groups)
                    # Go through each group of allomorphs
                    for group in cound_groups:
                        current_condTrue_set = __get_clean_set(temp_condTrue_set[temp_condTrue_set[0] == group][1])
                        # print(temp_condTrue_set[temp_condTrue_set[0] == group][1])
                        # print(current_condTrue_set)
                        __get_morph_condFalse(temp_condTrue_set[temp_condTrue_set[0] == 
                                                                group][1], current_condTrue_set, Morph)
            
            for idx, morpheme_entry in Morph[(Morph.FUNC == morpheme) & (Morph.CLASS == CLASS)].iterrows():
                # If there is a negated condition in the set of true conditions for the 
                #   current allomorph, remove the negation and add it to the false 
                #   conditions list for the current alomorph.
                almrph_condTrue = Morph.loc[idx, 'COND-T'].split(' ')
                almrph_condFalse = Morph.loc[idx, 'COND-F'].split(' ')
                # if '||' in Morph.loc[idx, 'COND-F']:
                #   print(idx, morpheme, Morph.loc[idx, 'COND-T'])
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
            # if '||' in Morph.loc[idx, 'COND-F']:
            #     print(idx, morpheme, Morph.loc[idx, 'COND-T'])
    # with open('test_morph.tsv', 'a') as f:
    #     for classs, condt, condf in zip(Morph['FUNC'], Morph['COND-T'], Morph['COND-F']):
    #         print(classs, condt, condf, file=f, sep='\t')
    return About, Header, Order, Morph, Lexicon, outputfilename
   

###########################################
#Construct Category:
# This function creates the category for matching
# using classes and conditions
###########################################
def createCat(XMorphType, XClass, XSet, XTrue, XFalse):

    # cat=XMorphType+re.sub("[\s\_]+","#",XClass+" "+XSet+" T:"+XTrue+" F:"+XFalse)
    cat = XMorphType + _SPACE_or_PLUS_MANY.sub("#", XClass + " " + XSet + " T:" + XTrue)
    return cat

###########################################
#Convert BW tag from BW2UTF8:
###########################################


def __convert_BW_tag(BW_tag):
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
        #print(parts[1])
        BW_pos = parts[1]
        utf8_BW_tag.append('/'.join([BW_lex, BW_pos]))
    return '+'.join(utf8_BW_tag)


###########################################
#Read Input file containing morphological specifications
###########################################
def constructAlmorDB(About, Header, Order, Morph, Lexicon):

    db = {}  # All DB components to be created

    db['OUT:###ABOUT###'] = list(About['Content'])

    db['OUT:###HEADER###'] = list(Header['Content'])

    #initialize ALMORDB Components
    db['OUT:###PREFIXES###'] = {}
    db['OUT:###SUFFIXES###'] = {}
    db['OUT:###STEMS###'] = {}
    db['OUT:###TABLE AB###'] = {}
    db['OUT:###TABLE BC###'] = {}
    db['OUT:###TABLE AC###'] = {}
    
    for orderIndex, order in Order.iterrows():
        # no used currently > use to limit lexicon reading
        cond = order['COND-T']
        print(order["VAR"], order["COND-T"],
              order["PREFIX"], order["STEM"], order["SUFFIX"], sep=" ; ", end='\n')

        #TODO - Caching candidate (memoize after the application of code below)
        preSet = expandMorphSeq(order['PREFIX'], Morph)
        sufSet = expandMorphSeq(order['SUFFIX'], Morph)
        #stemSet= [stem.to_dict() for i,stem in Lexicon.iterrows()]
        stemSet = [stem.to_dict()
                   for i, stem in Lexicon[Lexicon.CLASS == order['STEM']].iterrows()]

        cond2combs = {}
        
        for k in tqdm(stemSet):  #LEXICON
            xconds = k['COND-S']
            xcondt = k['COND-T']
            xcondf = k['COND-F']
            xcond_cat = " && ".join([xconds, xcondt, xcondf])
            stem_combinations = cond2combs.setdefault(xcond_cat, [])
            if stem_combinations:
                stem, _ = _generate_stem(k, xconds, xcondt, xcondf)
                db['OUT:###STEMS###'][stem] = 1
                continue

            for i in preSet:
                # print("here", i)
                pconds = ' '.join([m['COND-S'] for m in i])

                # pconds=' '.join([m['COND-S'] for m in i])

                pcondt = ' '.join([m['COND-T'] for m in i])
                pcondf = ' '.join([m['COND-F'] for m in i])

                # pbw=re.sub("\+\_","",re.sub("^(\_\+)+","",pbw)) 

                # pgloss=re.sub("\+\_","",re.sub("^(\_\+)+","",pgloss))

                #TODO: keep splits in diac till a later stage
                # pdiac=re.sub("[_\+]","",pform)

                #matching string for the BW-6T algorithm
                # pmatch=re.sub("[aiuoFKN~\#]","",pdiac)

                #TODO: suboptimal? recreate based on matching.
                #pcat="PrefCat:"+re.sub("[\s\_]+","#",pclass+pcondt)
                #pcat="PrefCat:"+re.sub("[\s\_]+","#",pclass+pcondt+pconds+pcondf)

                for j in sufSet:  
                    sconds = ' '.join([m['COND-S'] for m in j])
                    scondt = ' '.join([m['COND-T'] for m in j])
                    scondf = ' '.join([m['COND-F'] for m in j])

                    # sbw=re.sub("\+\_","",re.sub("^(\_\+)+","",sbw))


                    # sgloss=re.sub("\+\_","",re.sub("^(\_\+)+","",sgloss))
                    # sdiac=re.sub("[_\+]","",sform)
                    # smatch=re.sub("[aiuoFKN~\#]","",sdiac)

                    #scat="SuffCat:"+re.sub("[\s\_]+","#",sclass+scondt)
                    #scat="SuffCat:"+re.sub("[\s\_]+","#",sclass+scondt+sconds+scondf)

                    valid = checkCompatibility(' '.join([pconds, xconds, sconds]),
                                               ' '.join([pcondt, xcondt, scondt]),
                                               ' '.join([pcondf, xcondf, scondf]))
                    if valid:
                        
                        if verbose:
                            #print(valid,":",'+'.join([m['BW'] for m in i]),k['BW'],'+'.join([m['BW'] for m in j]))
                            #print(valid,":",'+'.join([m['FORM'] for m in i]),k['FORM'],''.join([m['FORM'] for m in j]))
                            print(valid,":", '+'.join([m['FORM'] for m in i])+k['FORM']+'+'.join([m['FORM'] for m in j]))
                            
                            #print("\nCOND-S =",' '.join([pconds,xconds,sconds]), 
                            #  "\nCOND-T =",' '.join([pcondt,xcondt,scondt]),
                            #  "\nCOND-F =",' '.join([pcondf,xcondf,scondf]),"\n####")
                        
                        combination = _read_combination(
                            i, j, k, pconds, pcondt, pcondf, sconds, scondt, scondf, xconds, xcondt, xcondf)
                        stem_combinations.append(combination)
                        db = _update_db(combination, db)

    return(db)

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
    ###SALAM###
    ## in the current almor/BAMA db implementation basic verb 
    ##  stems have per:3 by default, so I'm adding it here and 
    ##  then we should decide if we need to add it inthe lexicon
    ##  table or change the db algo
    #if 'pos:verb' in xfeat:
    #    xfeat = xfeat+' per:3 enc0:0 vox:a mod:u'

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

###########################################
#Create output file in ALMOR DB format
###########################################
def printAlmorDB(outputfilename, db):
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

###########################################



########################################### OK
# expandMorphSeq
#
# This function exands the specification of Morph Classes
# into all possible combinations of specific Morphemes.
# Input is space separated class name: e.g., '[QUES] [CONJ] [PREP]'
# Ouput is a list of lists of dictionaries; where each dictionary
# specifies a morpheme; the list order of the dictionaries indicate 
# order of morphemes.
# The function not only retrieves the morphemes but generates
# a Cartesian product of their combinations as specified by the 
# allowed Morph Classes
###########################################

def expandMorphSeq(MorphClass, Morph):
    MorphSeqs = []

    for c in MorphClass.split():
        # print('c='+c)
        if MorphSeqs == []:
            MorphSeqs = [[m.to_dict()]
                         for i, m in Morph[Morph.CLASS == c].iterrows()]
            # print(MorphSeqs)
        else:
            TEMP = []
            for morph in [m.to_dict() for i, m in Morph[Morph.CLASS == c].iterrows()]:
                for mseq in MorphSeqs:
                    x = mseq.copy()
                    x.extend([morph])
                    TEMP.append(x)
            ###SALAM###
            # if TEMP remains empty it will overrite everything
            if TEMP:
              MorphSeqs = TEMP
            # print(MorphSeqs)

    return MorphSeqs
###########################################


###########################################
#Remember to eliminate all non match affixes/stems
def checkCompatibility (condSet,condTrue,condFalse):
    #order cond: order-lexicon pairing? pos
    #Cond True is an anded list of ORs (||) 
    
###BAD BAD BAD
#  if "else" in condTrue:
#    return(True)
###BAD BAD BAD

    #Truth conditions:
    # AND : A B C
    # OR  : A||B||C
    # combined: A B||C

    #some limitations here.
    
    #TODO: optimize by caching; by breaking when no point of continuing.
    # cs=_INIT_UNDERSCORE.sub("",_SPACE_UNDERSCORE.sub("",_UNDERSCORE_SPACE.sub("",condSet)))
    # ct=_INIT_UNDERSCORE.sub("",_SPACE_UNDERSCORE.sub("",_UNDERSCORE_SPACE.sub("",condTrue)))
    # cf=_INIT_UNDERSCORE.sub("",_SPACE_UNDERSCORE.sub("",_UNDERSCORE_SPACE.sub("",condFalse)))
    key = f'{condSet}\t{condTrue}\t{condFalse}'
    # key = f'{cs}\t{ct}\t{cf}'
    if key in comp_memoi:
      return comp_memoi[key]
    else:
      comp_memoi[key] = ''
    # Remove all nil items (indicated as "_")
    # cs=re.sub("^\_","",re.sub(" \_","",re.sub("\_ ","",condSet))).split()
    # ct=re.sub("^\_","",re.sub(" \_","",re.sub("\_ ","",condTrue))).split()
    # cf=re.sub("^\_","",re.sub(" \_","",re.sub("\_ ","",condFalse))).split()
    cs = _INIT_UNDERSCORE.sub("", _SPACE_UNDERSCORE.sub(
        "", _UNDERSCORE_SPACE.sub("", condSet))).split()
    ct = _INIT_UNDERSCORE.sub("", _SPACE_UNDERSCORE.sub(
        "", _UNDERSCORE_SPACE.sub("", condTrue))).split()
    cf = _INIT_UNDERSCORE.sub("", _SPACE_UNDERSCORE.sub(
        "", _UNDERSCORE_SPACE.sub("", condFalse))).split()

    cs.sort()
    ct.sort()
    cf.sort()
    
    #print("Set="+str(cs)) 
   # print("NeedTrue="+str(ct)) 
   # print("NeedFalse="+str(cf)) 
    valid = True

    # Things that need to be true
    for t in ct:
        #print("NEED TRUE " + t)

        #OR condition check
        validor = False
        for ort in t.split('||'):
            validor = validor or ort in cs

        #AND Check
        valid = valid and validor
        if not valid:  # abort if we hit an invalid condition
            comp_memoi[key] = valid
            return valid
        #print(valid)

    # Things that need to be false
    for f in cf:
        #print("NEED False " + f)
        for orf in f.split('||'):
            valid = valid and orf not in cs
        #print(valid)
        if not valid:  # abort if we hit an invalid condition
            comp_memoi[key] = valid
            return valid

    #print("===")

    comp_memoi[key] = valid
    return valid

#Set=_ _ _ C:Tri C:MS C:MD C:L-initial _
#NeedTrue=_ _ _ _ C:FD
#NeedFalse=_ _ C:li/PREP _ _
###########################################
                                   
                                   
                                   
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
    # Morph.loc[idx, 'COND-T'] = ' '.join(almrph_condTrue)
  pass
  # return almrph_condFalse, almrph_condTrue 


###########################################

def __get_clean_set(cnd_col):
  mrph_condTrue = cnd_col.tolist()
  mrph_condTrue = [y for y in mrph_condTrue if y not in ['', '_', 'else', None]]
  mrph_condTrue = set(mrph_condTrue)  # make it a set
  
  return mrph_condTrue
