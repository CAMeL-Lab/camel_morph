"""Shared runtime resources and fixed behavior used by ``db_maker``.

Unlike :mod:`almor_schema`, these values belong to the builder implementation;
they do not describe the input-sheet or ALMOR output format.
"""

import re

from camel_tools.utils.charmap import CharMapper


# Cached transliteration resource.
BW2AR = CharMapper.builtin_mapper('bw2ar')
SAFEBW2AR = CharMapper.builtin_mapper('safebw2ar')

# Builder transformation rules.
# Postregex markers stripped from the undiacritized match/lookup key only
PRE_POST_REGEX_SYMBOL = re.compile(r'[#@]|%_?[mn]|%[من]')
PRE_POST_REGEX_SYMBOL_SMARTBACKOFF = re.compile(r'^\^|\$$|[#@]|%_?[mn]|%[من]')
CAPHI_UNDERSCORE_RE_1 = re.compile(r'_+')
CAPHI_UNDERSCORE_RE_2 = re.compile(r'^_|_$')

# Fixed runtime sentinels and supported CLI choices.
LOGPROB_RETURN_ALL = 'return_all'
LOGPROB_FEATURES = ('lex', 'pos_lex')
MISSING_LOGPROB = '-99'

CAMEL_TOOLS_LOCAL = 'local'
CAMEL_TOOLS_OFFICIAL = 'official'
