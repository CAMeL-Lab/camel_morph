import re

import pandas as pd
import os
from numpy import nan

from camel_tools.utils.charmap import CharMapper

from generate_passive import generate_passive
from generate_abstract_lexicon import generate_abstract_lexicon

bw2ar = CharMapper.builtin_mapper('bw2ar')

def read_morph_specs(config, config_name):
    """Read Input file containing morphological specifications"""
    data_dir = config['global']['data-dir']
    local_specs = config['local'][config_name]

    ABOUT = pd.read_csv(os.path.join(data_dir, 'About.csv'))
    HEADER = pd.read_csv(os.path.join(data_dir, 'Header.csv'))
    order_filename = f"{local_specs['specs']['order']}.csv"
    ORDER = pd.read_csv(os.path.join(data_dir, order_filename))
    morph_filename = f"{local_specs['specs']['morph']}.csv"
    MORPH = pd.read_csv(os.path.join(data_dir, morph_filename))
    lexicon_sheets = local_specs['lexicon']['sheets']
    LEXICON = pd.concat([pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
                         for name in lexicon_sheets])
    # Replace spaces in BW and GLOSS with '#'; skip commented rows and empty lines
    LEXICON = LEXICON[LEXICON.DEFINE == 'LEXICON']
    LEXICON = LEXICON.replace(nan, '', regex=True)
    LEXICON['GLOSS'] = LEXICON['GLOSS'].replace('\s+', '#', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' +', ' ', regex=True)
    LEXICON['COND-S'] = LEXICON['COND-S'].replace(' $', '', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' +', ' ', regex=True)
    LEXICON['COND-T'] = LEXICON['COND-T'].replace(' $', '', regex=True)

    patterns_path = local_specs['specs'].get('passive')
    if patterns_path:
        LEXICON_PASS = generate_passive(LEXICON, os.path.join(data_dir, f"{patterns_path}.csv"))
        LEXICON = pd.concat([LEXICON, LEXICON_PASS])

    BACKOFF = None
    backoff = local_specs['lexicon'].get('backoff')
    if backoff:
        if backoff == 'auto':
            BACKOFF = generate_abstract_lexicon(LEXICON)
        else:
            backoff_filename = f"{backoff}.csv"
            BACKOFF = pd.read_csv(os.path.join(data_dir, backoff_filename))
            BACKOFF = BACKOFF.replace(nan, '', regex=True)

    LEXICON['BW'] = LEXICON['FORM'] + '/' + LEXICON['BW']
    LEXICON['LEMMA'] = 'lex:' + LEXICON['LEMMA']

    POSTREGEX = None
    postregex_path = local_specs['specs'].get('postregex')
    if postregex_path:
        POSTREGEX = pd.read_csv(os.path.join(data_dir, 'PostRegex.csv'))
        POSTREGEX = POSTREGEX[POSTREGEX.DEFINE == 'POSTREGEX']
        POSTREGEX = POSTREGEX.replace(nan, '', regex=True)
        for i, row in POSTREGEX.iterrows():
            POSTREGEX.at[i, 'MATCH'] = _bw2ar_regex(row['MATCH'])
            POSTREGEX.at[i, 'REPLACE'] = ''.join(re.sub(r'\$', r'\\', row['REPLACE']))
    
    if local_specs.get('split_or') == True:
        assert BACKOFF is None
        LEXICON_ = []
        for _, row in LEXICON.iterrows():
            if '||' in row['COND-T'] and ' ' not in row['COND-T']:
                for term in row['COND-T'].split('||'):
                    row_ = row.to_dict()
                    row_['COND-T'] = term
                    LEXICON_.append(row_)
            else:
                LEXICON_.append(row.to_dict())
        LEXICON = pd.DataFrame(LEXICON_)

    #Process all the components:
    ORDER = ORDER[ORDER.DEFINE == 'ORDER']  # skip comments & empty lines
    ORDER = ORDER.replace(nan, '', regex=True)

    # Dictionary which groups conditions into classes (used later to
    # do partial compatibility which is useful from pruning out incoherent
    # suffix/prefix/stem combinations before performing full compatibility
    # in which (prefix, stem, suffix) instances are tried out individually).
    class2cond = MORPH[MORPH.DEFINE == 'CONDITIONS']
    class2cond = {cond_class["CLASS"]:
                            [cond for cond in cond_class["FUNC"].split() if cond]
                        for _, cond_class in class2cond.iterrows()}
    # Reverses the dictionary (k -> v and v -> k) so that individual conditions
    # which belonged to a class are now keys with that class as a value. In addition,
    # each condition gets its corresponding one-hot vector (actually stored as an int
    # because bitwise operations can only be performed on ints) computed based on the
    # other conditions within the same class (useful for pruning later).
    cond2class = {
        cond: (cond_class, 
               int(''.join(['1' if i == index else '0' for index in range (len(cond_s))]), 2)
        )
        for cond_class, cond_s in class2cond.items()
            for i, cond in enumerate(cond_s)}

    if os.path.exists('morph_cache/morph_sheet_prev.pkl'):
        MORPH_prev = pd.read_pickle('morph_cache/morph_sheet_prev.pkl')
        if MORPH.equals(MORPH_prev):
            MORPH = pd.read_pickle('morph_cache/morph_sheet_processed.pkl')
            SHEETS = dict(about=ABOUT, header=HEADER, order=ORDER, morph=MORPH,
                          lexicon=LEXICON, backoff=BACKOFF, postregex=POSTREGEX)
            return SHEETS, cond2class
    MORPH.to_pickle('morph_cache/morph_sheet_prev.pkl')
    
    # Skip commented rows and empty lines
    MORPH = MORPH[MORPH.DEFINE == 'MORPH']
    MORPH = MORPH.replace(nan, '', regex=True)
    MORPH = MORPH.replace('^\s+', '', regex=True)
    MORPH = MORPH.replace('\s+$', '', regex=True)
    MORPH = MORPH.replace('\s+', ' ', regex=True)
    # add FUNC and FEAT to COND-S
    MORPH['COND-S'] = MORPH['COND-S'].replace('[\[\]]', '', regex=True)
    MORPH.loc[MORPH['COND-S'] == '', 'COND-S'] = '_'
    MORPH.loc[MORPH['COND-T'] == '', 'COND-T'] = '_'
    # Replace spaces in BW and GLOSS with '#'
    MORPH['BW'] = MORPH['BW'].replace('\s+', '#', regex=True)
    MORPH.loc[MORPH['BW'] == '', 'BW'] = '_'
    MORPH.loc[MORPH['FORM'] == '', 'FORM'] = '_'
    MORPH['GLOSS'] = MORPH['GLOSS'].replace('\s+', '#', regex=True)
    # Retroactively generate the condFalse by creating the complementry distribution
    # of all the conditions within a single morpheme (all the allomorphs)
    # This is perfomed here once instead of on the fly.
    # Get all the classes in Morph
    MORPH_CLASSES = MORPH.CLASS.unique()
    MORPH_CLASSES = MORPH_CLASSES[MORPH_CLASSES != '_']
    # Get all the morphemes in Morph
    MORPH_MORPHEMES = MORPH.FUNC.unique()
    MORPH_MORPHEMES = MORPH_MORPHEMES[MORPH_MORPHEMES != '_']
    # Go through each class
    for CLASS in MORPH_CLASSES:
        # Go through each morpheme
        for MORPHEME in MORPH_MORPHEMES: 

            # Get the unique list of the true conditions in all the allomorphs.
            # We basically want to convert the 'COND-T' column in to n columns
            #   where n = maximum number of conditions for a single allomorph
            cond_t = MORPH[(MORPH.FUNC == MORPHEME) & 
                                (MORPH.CLASS == CLASS)]['COND-T'].str.split(pat=' ', expand=True)
            if cond_t.empty:
                continue
            cond_t = cond_t.replace(['_'], '')

            if len(cond_t.iloc[:][:]) == 1 and cond_t.iloc[0][0] == '':
                continue
            # Go through each column in the true conditions
            for col in cond_t:
              
                # if we are at the first columns
                if col == 0:
                    #TODO: uniq col[0] and remove fluff -> c_T
                    cond_t_current = get_clean_set(cond_t[col])
                    # Go through each allomorph
                    MORPH = get_morph_cond_f(cond_t[col], cond_t_current, MORPH)
                # we are in col > 0
                else:
                    #TODO: create temp_T by merging the the condTrue for col 0-col, put it in 
                    # col[0] and remove 1:col. This is basically to group allomorph with 
                    # similar general conditions
                    # Take the uniq of what is now col[0] and remove the fluff
                    if col == 1:
                        cond_t_temp = cond_t.copy()
                    else:
                        cond_t_temp = cond_t.copy()
                        cond_t_temp = cond_t_temp.replace([None], '')

                        cond_t_temp[0] = cond_t_temp.loc[:,:col-1].agg(' '.join, axis=1)
                        cond_t_temp = cond_t_temp.drop(cond_t_temp.columns[1:col], axis=1)
                        cond_t_temp.columns = range(cond_t_temp.shape[1])
                    if len(cond_t_temp.columns) == 1:
                        continue


                    cond_groups = cond_t_temp[0].unique()
                    cond_groups = cond_groups[cond_groups!= '_']
                    # Go through each group of allomorphs
                    for cond_group in cond_groups:
                        cond_t_current = get_clean_set(cond_t_temp[cond_t_temp[0] == cond_group][1])
                        MORPH = get_morph_cond_f(
                            cond_t_temp[cond_t_temp[0] == cond_group][1], cond_t_current, MORPH)
            
            for idx, _ in MORPH[(MORPH.FUNC == MORPHEME) & (MORPH.CLASS == CLASS)].iterrows():
                # If there is a negated condition in the set of true conditions for the 
                #   current allomorph, remove the negation and add it to the false 
                #   conditions list for the current alomorph.
                cond_t_almrph = MORPH.loc[idx, 'COND-T'].split(' ')
                cond_f_almrph = MORPH.loc[idx, 'COND-F'].split(' ')
                cond_s_negated = [x for x in cond_t_almrph if x.startswith('!')]
                if cond_s_negated:
                    for cond_negated in cond_s_negated:
                        cond_f_almrph.append(cond_negated[1:])
                        # remove from the true conditions
                        cond_t_almrph.remove(cond_negated)
                cond_t_almrph = [y for y in cond_t_almrph if y not in ['', '_', 'else', None]]
                cond_f_almrph = [y for y in cond_f_almrph if y not in ['', '_', 'else', None]]
                MORPH.loc[idx, 'COND-T'] = ' '.join(cond_t_almrph)
                MORPH.loc[idx, 'COND-F'] = ' '.join(cond_f_almrph)
    MORPH.to_pickle('morph_cache/morph_sheet_processed.pkl')
    SHEETS = dict(about=ABOUT, header=HEADER, order=ORDER, morph=MORPH,
                  lexicon=LEXICON, backoff=BACKOFF, postregex=POSTREGEX)
    return SHEETS, cond2class


def get_clean_set(cond_col):
    morph_cond_t = cond_col.tolist()
    morph_cond_t = [
        y for y in morph_cond_t if y not in ['', '_', 'else', None]]
    morph_cond_t = set(morph_cond_t)  # make it a set

    return morph_cond_t


def get_morph_cond_f(morph_cond_t, cond_t_current, MORPH):
    # Go through each allomorph
    for idx, entry in morph_cond_t.iteritems():
        # If we have no true condition for the allomorph (aka can take anything)
        if entry is None:
            continue
        elif entry != 'else':
            #TODO: create condFalse for the morpheme by c_T - entry
            cond_f_almrph, _ = _get_cond_false(
                cond_t_current, [entry])
        elif entry == 'else':
            # TODO: condFalse = c_T
            cond_f_almrph = cond_t_current
            # Finally, populate the 'COND-F' cell with the false conditions
        MORPH.loc[idx, 'COND-F'] = MORPH.loc[idx, 'COND-F'] + \
            ' ' + ' '.join(cond_f_almrph)
        MORPH.loc[idx, 'COND-F'] = MORPH.loc[idx, 'COND-F'].replace('_', '')
    
    return MORPH

def _get_cond_false(cond_t_all, cond_t_almrph):
    # The false conditions for the current allomorph is whatever true condition
    #   in the morpheme set that does not belong to the set of true conditions
    #   to the current allomorph.
    cond_f_almrph = cond_t_all - set(cond_t_almrph)
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
    if not cond_f_almrph:
        cond_f_almrph = ['_']
    # If we end up with no true condition, set it to '_'
    if not cond_t_almrph:
        cond_t_almrph = ['_']

    return cond_f_almrph, cond_t_almrph


def _bw2ar_regex(regex):
    match_ = []
    for match in re.split(r'(\\.|[\|}{\*\$_])', regex):
        match = match if re.match(r'(\\.)|[\|}{\*\$_]', match) else bw2ar(match)
        match_.append(match)
    return ''.join(match_)
