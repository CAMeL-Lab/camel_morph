"""Stable names and sentinel values used by the ALMOR and sheet formats.

This module describes the data contract consumed and produced by ``db_maker``.
It intentionally contains no database-building logic or runtime configuration.
"""

from types import MappingProxyType


DB_SECTION_KEY_PREFIX = 'OUT:'


def _section_key(name: str) -> str:
    return f'{DB_SECTION_KEY_PREFIX}###{name}###'


def almor_output_header(section_key: str) -> str:
    """Return the printable ALMOR header for an internal section key."""
    if not section_key.startswith(DB_SECTION_KEY_PREFIX):
        raise ValueError(f'Invalid database section key: {section_key}')
    return section_key[len(DB_SECTION_KEY_PREFIX):]


DB_SECTION_ABOUT = _section_key('ABOUT')
DB_SECTION_HEADER = _section_key('HEADER')
DB_SECTION_POSTREGEX = _section_key('POSTREGEX')
DB_SECTION_PREFIXES = _section_key('PREFIXES')
DB_SECTION_STEMS = _section_key('STEMS')
DB_SECTION_SUFFIXES = _section_key('SUFFIXES')
DB_SECTION_STEM_BACKOFF = _section_key('STEMBACKOFF')
DB_SECTION_SMART_BACKOFF = _section_key('SMARTBACKOFF')
DB_SECTION_TABLE_AB = _section_key('TABLE AB')
DB_SECTION_TABLE_BC = _section_key('TABLE BC')
DB_SECTION_TABLE_AC = _section_key('TABLE AC')

COMPATIBILITY_SECTIONS = frozenset({
    DB_SECTION_TABLE_AB,
    DB_SECTION_TABLE_BC,
    DB_SECTION_TABLE_AC,
})
MORPHEME_SECTIONS = (
    ('PREFIXES', DB_SECTION_PREFIXES),
    ('STEMS', DB_SECTION_STEMS),
    ('SUFFIXES', DB_SECTION_SUFFIXES),
)

# Input sheet columns.
COL_PREFIX = 'PREFIX'
COL_STEM = 'STEM'
COL_SUFFIX = 'SUFFIX'
COL_CLASS = 'CLASS'
COL_CONTENT = 'CONTENT'
COL_MATCH = 'MATCH'
COL_REPLACE = 'REPLACE'
COL_PREFIX_SHORT = 'PREFIX-SHORT'
COL_STEM_SHORT = 'STEM-SHORT'
COL_SUFFIX_SHORT = 'SUFFIX-SHORT'
SHORT_ORDER_COLUMNS = frozenset({
    COL_PREFIX_SHORT,
    COL_STEM_SHORT,
    COL_SUFFIX_SHORT,
})

SEG_TOK_SCHEMES = (
    'D1SEG', 'D1TOK', 'D2SEG', 'D2TOK',
    'D3SEG', 'D3TOK', 'ATBSEG', 'ATBTOK',
)
POS_TAG_SCHEMES = ('UD', 'CATIB6')
SHEET_SCHEME_COLUMNS = SEG_TOK_SCHEMES + POS_TAG_SCHEMES
LEXICON_REQUIRED_COLUMNS = frozenset({
    'COND-S',
    'COND-T',
    'FORM',
    'LEMMA',
})

EMPTY_MORPH_ROW = (
    ('DEFINE', 'MORPH'),
    ('CLASS', '[EMPTY]'),
    ('LINE', -1),
)
SPECS_HEADER_REQUIRED = MappingProxyType({
    'order': ('EXCLUDE', 'DEFINE', 'CLASS', 'PREFIX', 'STEM', 'SUFFIX'),
    'morph': (
        'EXCLUDE', 'DEFINE', 'CLASS', 'FUNC', 'FORM',
        'BW', 'GLOSS', 'FEAT', 'COND-T', 'COND-S',
    ),
})
ORDER_FIELDS = (COL_PREFIX, COL_STEM, COL_SUFFIX)
ORDER_FIELDS_SHORT = (COL_PREFIX_SHORT, COL_STEM_SHORT, COL_SUFFIX_SHORT)

# Values with fixed meaning in the sheets and generated database.
EMPTY_FIELD = '_'
EMPTY_CONDITION = '-'
EMPTY_MORPH_CLASS = '[EMPTY]'
NO_ANALYSIS = 'NOAN'
NOT_WRITTEN = 'NTWS'
DROP_FORM = 'DROP'
BACKOFF_SMART = 'smart'
BACKOFF_VANILLA = 'vanilla'
SOURCE_LEXICON = 'lex'
POS_VERB = 'verb'
CATIB6_PASSIVE_VERB = 'VRB-PASS'

MORPH_TYPE_PREFIX = 'DBPrefix'
MORPH_TYPE_STEM = 'DBStem'
MORPH_TYPE_SUFFIX = 'DBSuffix'
CAPHI_MORPH_TYPES = (
    MORPH_TYPE_PREFIX,
    MORPH_TYPE_STEM,
    MORPH_TYPE_SUFFIX,
)
CAT_TYPE_PREFIX = 'P'
CAT_TYPE_STEM = 'X'
CAT_TYPE_SUFFIX = 'S'

BW2AR_AFFIX_FIELDS = (
    'diac', 'd3seg', 'd3tok', 'atbseg', 'atbtok',
    'd2seg', 'd2tok', 'd1tok', 'd1seg',
)
BW2AR_STEM_FIELDS = (
    'lex', 'diac', 'cm_stem', 'cm_buffer', 'root',
    'd3seg', 'd3tok', 'atbseg', 'atbtok',
    'pattern', 'pattern_abstract',
)
STEM_METADATA_COLUMNS = ('ROOT', 'PATTERN_ABSTRACT', 'PATTERN', 'SOURCE')
CAPHI_COLUMN = 'CAPHI'
