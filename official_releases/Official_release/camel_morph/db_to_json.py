"""Convert an ALMOR ``.db`` file into structured JSON.

Faithful conversion: values are taken as-is from the ``.db``. The only
intentional changes are:

* Section / field renames to the CM schema JSON shape
* ``TABLE AB`` / ``TABLE BC`` / ``TABLE AC`` → compat pair lists
* ``POSTREGEX`` → embedded under ``definitions[feature].postregex`` for
  each affected feature (sheet rules + camel_tools built-in rewrites)
* ``definitions`` enriched with ``dtype`` / ``required`` / ``nullable``,
  plus ``default`` when catch-all defaults provide a concrete value
* Top-level ``schemaVersion`` then ``meta``, then keys in ALMOR .db section
  order (definitions → defaults → order → tokenizations → stemBackoffs →
  prefixes → suffixes → stems → …)
* Definition fields: required → nullable → dtype → values → default → postregex
* Feature / defaults / stemBackoffs keys alphabetical; feats keys alphabetical
* Formatting: UTF-8, Unix newlines, tab indent, sorted+uniqued enum values,
  feature dictionaries on one line

CLI::

    python -m camel_morph.db_to_json path/to/file.db -o path/to/file.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA_VERSION = 1
REGEX_FORMAT = 'python-re'

SECTION_RE = re.compile(r'^###(.+?)###\s*$')
OPEN_SENTINEL = '*open*'
STAR_SENTINEL = '*'

# Core analysis identity features (always present after a successful analyze/generate).
# Generator also hard-requires ``pos`` as an input feature.
REQUIRED_FEATURES = frozenset({'diac', 'lex', 'pos', 'pattern'})

FLOAT_FEATURES = frozenset({'pos_logprob', 'lex_logprob', 'pos_lex_logprob'})


#
# `POSTREGEX` in the ALMOR DB is compiled from the configured POSTREGEX sheet.
# Rules are attached under `definitions[feature].postregex` for each affected
# feature present in the DB (diac, caphi, pattern, tokenization/segmentation, …),
# combined with camel_tools built-in rewrite steps for that feature.
#
_POSTREGEX_FEATURE_NAME_ALIASES = frozenset({
    # User-facing name used in this task.
    'abs_pattern',
})

# Top-level keys follow ALMOR .db section order (after schemaVersion / meta).
_TOP_LEVEL_KEY_ORDER = (
    'schemaVersion',
    'meta',
    'definitions',       # DEFINES
    'defaults',          # DEFAULTS
    'order',             # ORDER
    'tokenizations',     # TOKENIZATIONS
    'stemBackoffs',      # STEMBACKOFF
    'prefixes',          # PREFIXES
    'suffixes',          # SUFFIXES
    'stems',             # STEMS
    'smartBackoffs',     # SMARTBACKOFF
    'prefixStemCompat',  # TABLE AB
    'stemSuffixCompat',  # TABLE BC
    'prefixSuffixCompat',  # TABLE AC
)

# Fields under each definitions.<feature> entry (not alphabetical).
_DEFINITION_FIELD_ORDER = (
    'required',
    'nullable',
    'dtype',
    'values',
    'default',
    'postregex',
)

# Nested maps whose *keys* stay alphabetical (feature / pos / mode names).
_ALPHA_KEY_PARENTS = frozenset({
    'definitions', 'defaults', 'stemBackoffs', 'meta',
})


def _ordered_keys(
    obj: Mapping[str, Any],
    *,
    preferred: Sequence[str],
) -> List[str]:
    """Emit keys in ``preferred`` order, then any leftovers in insertion order."""
    keys = list(obj.keys())
    ordered = [key for key in preferred if key in obj]
    ordered.extend(key for key in keys if key not in preferred)
    return ordered


def _sorted_object_keys(
    obj: Mapping[str, Any],
    *,
    parent_key: Optional[str] = None,
) -> List[str]:
    """Return object keys in the export order.

    - Top-level: ``schemaVersion``, ``meta``, then .db section order
    - ``definitions`` / ``defaults`` / ``stemBackoffs`` / ``meta``: A–Z keys
    - Each feature definition: required → nullable → dtype → values → default → postregex
    - Everything else: preserve insertion order
    """
    if parent_key is None:
        return _ordered_keys(obj, preferred=_TOP_LEVEL_KEY_ORDER)

    # Feature definition objects always carry required + nullable + dtype.
    if (
        'required' in obj
        and 'nullable' in obj
        and 'dtype' in obj
        and parent_key not in _ALPHA_KEY_PARENTS
    ):
        return _ordered_keys(obj, preferred=_DEFINITION_FIELD_ORDER)

    if parent_key in _ALPHA_KEY_PARENTS:
        return sorted(obj.keys())

    return list(obj.keys())


def _order_definition_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Order a single feature definition's fields."""
    return {
        key: entry[key]
        for key in _ordered_keys(entry, preferred=_DEFINITION_FIELD_ORDER)
    }
def _infer_postregex_applies_to(definitions: Mapping[str, Any]) -> List[str]:
    available = set(definitions.keys())
    candidates = set()

    for feat in ['diac', 'caphi', 'pattern', 'pattern_abstract', *_POSTREGEX_FEATURE_NAME_ALIASES]:
        if feat in available:
            candidates.add(feat)

    # tokenization/segmentation schemes (e.g., d1tok, d2seg, atbseg, atbtok)
    for feat in available:
        if re.match(r'^d\d+(seg|tok)$', feat):
            candidates.add(feat)
        elif re.match(r'^atb(seg|tok)$', feat):
            candidates.add(feat)
        elif feat in {'dseg', 'dtok', 'abtseg', 'abttok'}:
            candidates.add(feat)

    return sorted(candidates)


def _load_camel_tools_utils():
    """
    Load `camel_tools.morphology.utils` from the vendored `camel_tools/` directory.

    We import only when POSTREGEX export is requested, so normal conversion stays
    fast. This lets us reuse camel_tools' exact rewrite regex patterns instead
    of duplicating their huge unicode regex strings.
    """
    camel_tools_root = os.path.join(os.path.dirname(__file__), 'camel_tools')
    if camel_tools_root not in sys.path:
        sys.path.insert(0, camel_tools_root)
    import camel_tools.morphology.utils as camel_utils  # type: ignore[import-not-found]
    return camel_utils


_CAMEL_TOOLS_BUILTIN_POSTREGEX_RULES_CACHE: Dict[tuple[str, str], List[Dict[str, str]]] = {}


def _normalize_dialect(dialect: Optional[str]) -> str:
    """Uppercase dialect label (matches config ``dialect`` / POSTREGEX VARIANT)."""
    return (dialect or 'MSA').strip().upper() or 'MSA'


def _dialect_rewrite_fn_name(dialect: str) -> str:
    return f'rewrite_diac_camel_morph_{dialect.lower()}'


def _uses_interleaved_sheet_diac_rewrites(dialect: Optional[str] = None) -> bool:
    """True when camel_tools applies sheet POSTREGEX in the middle of diac cleanup.

    Driven only by the config dialect: look up
    ``rewrite_diac_camel_morph_<dialect>`` in camel_tools. MSA applies builtins
    only; other dialects that define that function (e.g. egy, pal) interleave
    sheet rules between builtin steps.
    """
    dialect_key = _normalize_dialect(dialect)
    # MSA is the baseline: built-in rewrite rules for MSA correspond to the
    # non-interleaved pipeline. If some camel_tools version doesn't expose the
    # MSA rewrite function, we still want JSON export to succeed.
    if dialect_key == 'MSA':
        return False

    u = _load_camel_tools_utils()
    fn = getattr(u, _dialect_rewrite_fn_name(dialect_key), None)
    if fn is None:
        raise ValueError(
            f'No camel_tools diac rewrite for dialect {dialect_key!r} '
            f'(expected {_dialect_rewrite_fn_name(dialect_key)}). '
            f'Set local config "dialect" to a supported value.'
        )
    msa_fn = getattr(u, 'rewrite_diac_camel_morph_msa', None)
    if msa_fn is None:
        # Can't compare; best-effort: if dialect-specific rewrite exists, it
        # likely implies interleaving.
        return True

    return fn is not msa_fn


def _rule(rx: re.Pattern[str], repl: str) -> Dict[str, str]:
    return {'pattern': rx.pattern, 'replacement': repl}


def _camel_tools_builtin_postregex_rules_for_feature(
    feature: str,
    dialect: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Rules that `camel_tools` applies internally (not the configured POSTREGEX
    sheet rules).

    This targets the MSA behavior in `rewrite_diac_camel_morph_msa`, plus the
    shared token/pattern/caphi rewrites used by camel_tools.
    """
    dialect_key = _normalize_dialect(dialect)
    cache_key = (feature, dialect_key)
    if cache_key in _CAMEL_TOOLS_BUILTIN_POSTREGEX_RULES_CACHE:
        return _CAMEL_TOOLS_BUILTIN_POSTREGEX_RULES_CACHE[cache_key]

    interleaved = (
        _uses_interleaved_sheet_diac_rewrites(dialect_key)
        if feature == 'diac' else False
    )

    u = _load_camel_tools_utils()

    if feature == 'diac':
        if interleaved:
            # Prefix only; sheet rules are inserted in ``_compose_feature_postregex``.
            rules = [
                _rule(u._REWRITE_DIAC_RE_CM_1, '\\1' + '\u0651'),
                _rule(u._REWRITE_DIAC_RE_CM_2, ''),
                _rule(u._REWRITE_DIAC_RE_5, ''),
            ]
        else:
            rules = [
                _rule(u._REWRITE_DIAC_RE_1, '\\1' + '\u0651'),
                _rule(u._REWRITE_DIAC_RE_2, ''),
                _rule(u._REWRITE_DIAC_RE_3, '\u0627' + '\\1'),
                _rule(u._REWRITE_DIAC_RE_4, '\u0627'),
                _rule(u._REWRITE_DIAC_RE_5, ''),
                _rule(u._REWRITE_DIAC_RE_6, '\u0651'),
            ]
    elif feature == 'caphi':
        rules = [
            _rule(u._REWRITE_CAPHI_RE_1, '\\2\\2'),
            _rule(u._REWRITE_CAPHI_RE_2, '\\1_\\1'),
            _rule(u._REWRITE_CAPHI_RE_3, 'ii_\\1'),
            _rule(u._REWRITE_CAPHI_RE_4, 'uu_\\1'),
            _rule(u._REWRITE_CAPHI_RE_5, '\\1'),
            _rule(u._REWRITE_CAPHI_RE_6, '\\1_\\2'),
            _rule(u._REWRITE_CAPHI_RE_7, 'uu\\1'),
            _rule(u._REWRITE_CAPHI_RE_8, 't_\\1'),
            _rule(u._REWRITE_CAPHI_RE_9, 'aa_'),
            _rule(u._REWRITE_CAPHI_RE_10, '_'),
            _rule(u._REWRITE_CAPHI_RE_11, '_'),
            _rule(u._REWRITE_CAPHI_RE_12, ''),
        ]
    elif feature == 'pattern' or feature == 'pattern_abstract':
        # camel_tools rewrite_pattern removes definite-article marker from diac-like strings.
        rules = [_rule(u._REWRITE_DIAC_RE_2, '')]
    else:
        # Tokenization/segmentation schemes: camel_tools uses rewrite_tok_1 and
        # rewrite_tok_2 depending on the scheme.
        tok_schemes_1 = {'d1tok', 'd2tok', 'atbtok', 'd1seg', 'd2seg', 'd3seg', 'atbseg'}
        tok_schemes_2 = {'d3tok', 'd3seg'}
        rules = []
        if feature in tok_schemes_1:
            rules.extend([
                _rule(u._REWRITE_DIAC_RE_1, '\\1' + '\u0651'),
                _rule(u._REWRITE_DIAC_RE_2, ''),
                _rule(u._REWRITE_DIAC_RE_3, '\u0627' + '\\1'),
            ])
        if feature in tok_schemes_2:
            rules.extend([
                _rule(u._REWRITE_DIAC_RE_3, '\u0627' + '\\1'),
            ])

    _CAMEL_TOOLS_BUILTIN_POSTREGEX_RULES_CACHE[cache_key] = rules
    return rules


def _compose_feature_postregex(
    feature: str,
    sheet_rules: Sequence[Dict[str, str]],
    dialect: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Combine camel_tools builtins with sheet POSTREGEX in dialect order."""
    dialect_key = _normalize_dialect(dialect)
    sheet_rules = list(sheet_rules)

    if feature == 'diac' and _uses_interleaved_sheet_diac_rewrites(dialect_key):
        # Matches non-MSA rewrite_diac_camel_morph_<dialect>:
        #   CM_1, CM_2, RE_5  →  sheet POSTREGEX  →  RE_3, RE_4, RE_6
        u = _load_camel_tools_utils()
        prefix = _camel_tools_builtin_postregex_rules_for_feature('diac', dialect_key)
        suffix = [
            _rule(u._REWRITE_DIAC_RE_3, '\u0627' + '\\1'),
            _rule(u._REWRITE_DIAC_RE_4, '\u0627'),
            _rule(u._REWRITE_DIAC_RE_6, '\u0651'),
        ]
        return prefix + sheet_rules + suffix

    return (
        _camel_tools_builtin_postregex_rules_for_feature(feature, dialect_key)
        + sheet_rules
    )

def parse_almor_db(path: str) -> Dict[str, List[str]]:
    """Read an ALMOR ``.db`` file into ``{section_name: [raw_lines...]}``."""
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None

    with open(path, encoding='utf-8', newline='\n') as f:
        for raw in f:
            line = raw.rstrip('\n').rstrip('\r')
            match = SECTION_RE.match(line)
            if match:
                current = match.group(1)
                sections.setdefault(current, [])
                continue
            if current is None:
                if line.strip():
                    raise ValueError(
                        f'Content before first section header in {path!r}: {line!r}'
                    )
                continue
            sections[current].append(line)

    return sections


def _parse_feat_tokens(feat_blob: str) -> Dict[str, str]:
    """Parse ``feat:value feat:value ...`` into a dict.

    Empty values (``diac:``) are preserved as ``""``. Values may themselves
    contain ``:`` (e.g. ``cm_pref_ids:[QUES-N]:1+[CONJ-N]:3``).
    """
    analysis: Dict[str, str] = {}
    if not feat_blob or not feat_blob.strip():
        return analysis

    for token in feat_blob.split():
        if ':' not in token:
            raise ValueError(f"Malformed feature token (missing ':'): {token!r}")
        key, value = token.split(':', 1)
        analysis[key] = value
    return analysis


def _sorted_unique(values: Sequence[str]) -> List[str]:
    return sorted(set(values))


def _parse_defines_raw(lines: Sequence[str]) -> Dict[str, Optional[List[str]]]:
    """Parse DEFINES into ``{feat: None for open, sorted unique values for closed}``."""
    defines: Dict[str, Optional[List[str]]] = {}
    for line in lines:
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens or tokens[0] != 'DEFINE' or len(tokens) < 2:
            raise ValueError(f'Malformed DEFINE line: {line!r}')
        feat = tokens[1]
        values = tokens[2:]
        if len(values) == 1 and values[0] == f'{feat}:{OPEN_SENTINEL}':
            defines[feat] = None
            continue

        closed_values: List[str] = []
        for item in values:
            prefix = f'{feat}:'
            if not item.startswith(prefix):
                raise ValueError(
                    f'DEFINE value {item!r} does not start with {prefix!r}'
                )
            closed_values.append(item[len(prefix):])
        defines[feat] = _sorted_unique(closed_values)
    return defines


def _build_definitions(
    defines_raw: Mapping[str, Optional[List[str]]],
    *,
    star_defaults: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Build definitions from DEFINE data + catch-all defaults.

    Feature names are alphabetical. Each definition's fields are ordered as::

        required → nullable → dtype → values → default → postregex
    """
    definitions: Dict[str, Dict[str, Any]] = {}

    for feat in sorted(defines_raw.keys()):
        values = defines_raw[feat]
        required = feat in REQUIRED_FEATURES
        entry: Dict[str, Any] = {
            'required': required,
            'nullable': not required,
            'dtype': (
                'float' if feat in FLOAT_FEATURES
                else 'str' if values is None
                else 'str-enum'
            ),
        }
        if values is not None:
            entry['values'] = list(values)

        default = star_defaults.get(feat)
        if default is not None and default not in (STAR_SENTINEL, OPEN_SENTINEL):
            if feat in FLOAT_FEATURES:
                try:
                    entry['default'] = float(default)
                except ValueError:
                    entry['default'] = default
            else:
                entry['default'] = default

        definitions[feat] = _order_definition_entry(entry)

    return definitions


def _parse_defaults(lines: Sequence[str]) -> Dict[str, Dict[str, str]]:
    defaults: Dict[str, Dict[str, str]] = {}
    for line in lines:
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens or tokens[0] != 'DEFAULT':
            raise ValueError(f'Malformed DEFAULT line: {line!r}')
        feats = _parse_feat_tokens(' '.join(tokens[1:]))
        pos = feats.get('pos')
        if pos is None:
            raise ValueError(f'DEFAULT line missing pos: {line!r}')
        # Keep every value exactly as in the .db (including "*" / "*open*").
        defaults[pos] = {key: value for key, value in sorted(feats.items())}
    return defaults


def _parse_order(lines: Sequence[str]) -> List[str]:
    for line in lines:
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens or tokens[0] != 'ORDER':
            raise ValueError(f'Malformed ORDER line: {line!r}')
        return tokens[1:]
    return []


def _parse_tokenizations(lines: Sequence[str]) -> List[str]:
    for line in lines:
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens or tokens[0] != 'TOKENIZATION':
            raise ValueError(f'Malformed TOKENIZATION line: {line!r}')
        return tokens[1:]
    return []


def _parse_stembackoff(lines: Sequence[str]) -> Dict[str, List[str]]:
    stembackoff: Dict[str, List[str]] = {}
    for line in lines:
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens or tokens[0] != 'STEMBACKOFF' or len(tokens) < 2:
            raise ValueError(f'Malformed STEMBACKOFF line: {line!r}')
        mode = tokens[1]
        stembackoff[mode] = tokens[2:]
    return stembackoff


def _parse_postregex(lines: Sequence[str]) -> Dict[str, Any]:
    """Parse ALMOR POSTREGEX into a single ``postregex`` object.
    """
    matches: List[str] = []
    replaces: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('\t')
        kind = parts[0]
        values = parts[1:]
        if kind == 'MATCH':
            matches = values
        elif kind == 'REPLACE':
            replaces = values
        else:
            raise ValueError(f'Malformed POSTREGEX line: {line!r}')

    if len(matches) != len(replaces):
        raise ValueError(
            f'POSTREGEX MATCH/REPLACE length mismatch: '
            f'{len(matches)} vs {len(replaces)}'
        )

    rules = [
        {'pattern': match, 'replacement': replace}
        for match, replace in zip(matches, replaces)
    ]

    # `db_maker` and `camel_tools` support multiple spellings of the
    # `%m`/`%n` placeholders. Some surface features (notably CAPHI) may still
    # contain the Buckwalter placeholders (with an optional underscore),
    # while the configured POSTREGEX sheet may be represented using Arabic
    # letters (`%م`/`%ن`).
    #
    # To keep this transformation config-driven (from the POSTREGEX sheet) but
    # robust across feature spellings, we expand simple placeholder-only rules.
    expanded_rules: List[Dict[str, str]] = []
    for rule in rules:
        expanded_rules.append(rule)
        pattern = rule.get('pattern')
        replacement = rule.get('replacement')
        if pattern == '%م':
            expanded_rules.append({'pattern': r'%_?m', 'replacement': replacement})
        elif pattern == '%ن':
            expanded_rules.append({'pattern': r'%_?n', 'replacement': replacement})

    return {'rules': expanded_rules}


def _parse_table(lines: Sequence[str], section_name: str) -> List[List[str]]:
    pairs: List[List[str]] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f'Malformed {section_name} pair line: {line!r}')
        pairs.append([parts[0], parts[1]])
    return pairs


def _parse_morpheme_line(line: str) -> Dict[str, Any]:
    """Parse ``form\\tcategory\\tfeat:val ...`` into a morpheme object."""
    parts = line.split('\t')
    if len(parts) < 2:
        raise ValueError(f'Malformed morpheme line (need form\\tcat\\t...): {line!r}')
    ortho = parts[0]
    cat = parts[1]
    feat_blob = parts[2] if len(parts) > 2 else ''
    if len(parts) > 3:
        feat_blob = '\t'.join(parts[2:])

    feats = {
        key: value
        for key, value in sorted(_parse_feat_tokens(feat_blob).items())
    }
    # Same column order as the .db line: form, category, features.
    return {'ortho': ortho, 'cat': cat, 'feats': feats}


def _parse_morpheme_section(lines: Sequence[str]) -> List[Dict[str, Any]]:
    return [_parse_morpheme_line(line) for line in lines if line.strip()]


def almor_sections_to_json(
    sections: Mapping[str, Sequence[str]],
    *,
    dialect: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert parsed ALMOR sections into the JSON object.

    ``dialect`` (e.g. ``MSA``, ``EGY``, ``PAL``) selects the camel_tools
    builtin rewrite rules embedded under ``definitions.*.postregex``.
    """
    dialect = _normalize_dialect(dialect)
    required_sections = (
        'DEFINES', 'PREFIXES', 'SUFFIXES', 'STEMS',
        'TABLE AB', 'TABLE BC', 'TABLE AC',
    )
    missing = [name for name in required_sections if name not in sections]
    if missing:
        raise ValueError(f'DB is missing required section(s): {", ".join(missing)}')

    defaults = (
        _parse_defaults(sections['DEFAULTS'])
        if 'DEFAULTS' in sections else {}
    )
    tokenizations = (
        _parse_tokenizations(sections['TOKENIZATIONS'])
        if 'TOKENIZATIONS' in sections else []
    )
    defines_raw = _parse_defines_raw(sections['DEFINES'])

    result: Dict[str, Any] = {
        'schemaVersion': SCHEMA_VERSION,
        'definitions': _build_definitions(
            defines_raw,
            star_defaults=defaults.get('*', {}),
        ),
    }

    if defaults:
        result['defaults'] = defaults
    if 'ORDER' in sections:
        result['order'] = _parse_order(sections['ORDER'])
    if tokenizations:
        result['tokenizations'] = tokenizations
    if 'STEMBACKOFF' in sections:
        result['stemBackoffs'] = _parse_stembackoff(sections['STEMBACKOFF'])
    if 'POSTREGEX' in sections:
        postregex = _parse_postregex(sections['POSTREGEX'])
        if postregex['rules']:
            # Embed rules under each affected feature only (no top-level
            # postregex section). Order follows the dialect's camel_tools
            # rewrite pipeline.
            for feat in _infer_postregex_applies_to(result['definitions']):
                feat_def = result['definitions'].get(feat)
                if not feat_def:
                    continue
                feat_def['postregex'] = _compose_feature_postregex(
                    feat, postregex['rules'], dialect,
                )
                result['definitions'][feat] = _order_definition_entry(feat_def)

    # Morpheme sections follow .db order: PREFIXES, SUFFIXES, STEMS, …
    result['prefixes'] = _parse_morpheme_section(sections['PREFIXES'])
    result['suffixes'] = _parse_morpheme_section(sections['SUFFIXES'])
    result['stems'] = _parse_morpheme_section(sections['STEMS'])

    if 'SMARTBACKOFF' in sections:
        result['smartBackoffs'] = _parse_morpheme_section(sections['SMARTBACKOFF'])

    result['prefixStemCompat'] = _parse_table(sections['TABLE AB'], 'TABLE AB')
    result['stemSuffixCompat'] = _parse_table(sections['TABLE BC'], 'TABLE BC')
    result['prefixSuffixCompat'] = _parse_table(sections['TABLE AC'], 'TABLE AC')

    return result
def db_file_to_json(
    db_path: str,
    *,
    dialect: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse ``db_path`` and return the JSON object."""
    return almor_sections_to_json(parse_almor_db(db_path), dialect=dialect)

# ---------------------------------------------------------------------------
# JSON serialization (tabs, Unix newlines, inline feature dicts)
# ---------------------------------------------------------------------------

def _is_feature_value_dict(obj: Any) -> bool:
    """True for dicts whose keys are feature names and values are scalars."""
    if not isinstance(obj, dict) or not obj:
        return False
    return all(
        isinstance(key, str) and not isinstance(value, (dict, list))
        for key, value in obj.items()
    )


def _encode_scalar(value: Any, *, ensure_ascii: bool) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii)


def _encode_inline_object(obj: Mapping[str, Any]) -> str:
    """Encode a dict on one line with keys in ascending lexicographical order."""
    parts = [
        f'{json.dumps(key, ensure_ascii=False)}: {_encode_scalar(obj[key], ensure_ascii=False)}'
        for key in sorted(obj.keys())
    ]
    return '{' + ', '.join(parts) + '}'


def _encode(
    value: Any,
    *,
    indent: int,
    parent_key: Optional[str] = None,
    inline_feature_dicts: bool = False,
    in_postregex: bool = False,
) -> str:
    tab = '\t' * indent

    if isinstance(value, dict):
        if inline_feature_dicts and _is_feature_value_dict(value):
            return _encode_inline_object(value)
        if not value:
            return '{}'

        keys = _sorted_object_keys(value, parent_key=parent_key)

        lines = ['{']
        for i, key in enumerate(keys):
            child = value[key]
            child_inline = key == 'feats' or parent_key == 'defaults'
            child_in_postregex = in_postregex or key == 'postregex'
            encoded_child = _encode(
                child,
                indent=indent + 1,
                parent_key=key,
                inline_feature_dicts=child_inline,
                in_postregex=child_in_postregex,
            )
            comma = ',' if i < len(keys) - 1 else ''
            key_json = json.dumps(key, ensure_ascii=False)
            lines.append(f'{tab}\t{key_json}: {encoded_child}{comma}')
        lines.append(f'{tab}}}')
        return '\n'.join(lines)
    if isinstance(value, list):
        if not value:
            return '[]'

        if all(isinstance(item, str) for item in value):
            lines = ['[']
            for i, item in enumerate(value):
                comma = ',' if i < len(value) - 1 else ''
                lines.append(
                    f'{tab}\t{_encode_scalar(item, ensure_ascii=in_postregex and parent_key in {"pattern", "replacement"})}{comma}'
                )
            lines.append(f'{tab}]')
            return '\n'.join(lines)

        if all(
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(x, str) for x in item)
            for item in value
        ):
            lines = ['[']
            for i, item in enumerate(value):
                comma = ',' if i < len(value) - 1 else ''
                pair = (
                    f'[{_encode_scalar(item[0], ensure_ascii=in_postregex and parent_key in {"pattern", "replacement"})}, '
                    f'{_encode_scalar(item[1], ensure_ascii=in_postregex and parent_key in {"pattern", "replacement"})}]'
                )
                lines.append(f'{tab}\t{pair}{comma}')
            lines.append(f'{tab}]')
            return '\n'.join(lines)

        lines = ['[']
        for i, item in enumerate(value):
            comma = ',' if i < len(value) - 1 else ''
            encoded_item = _encode(
                item,
                indent=indent + 1,
                parent_key=parent_key,
                inline_feature_dicts=False,
                in_postregex=in_postregex,
            )
            if '\n' in encoded_item:
                item_lines = encoded_item.split('\n')
                lines.append(f'{tab}\t{item_lines[0]}')
                lines.extend(item_lines[1:])
                lines[-1] = lines[-1] + comma
            else:
                lines.append(f'{tab}\t{encoded_item}{comma}')
        lines.append(f'{tab}]')
        return '\n'.join(lines)

    return _encode_scalar(
        value,
        ensure_ascii=in_postregex and parent_key in {'pattern', 'replacement'},
    )


def dumps_db_json(data: Mapping[str, Any]) -> str:
    """Serialize JSON with tab indent and inline feature dicts."""
    return _encode(data, indent=0, parent_key=None) + '\n'


def write_db_json(data: Mapping[str, Any], json_path: str) -> None:
    """Write JSON to ``json_path`` (UTF-8, ``\\n``)."""
    parent = os.path.dirname(json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(dumps_db_json(data))


def export_db_to_json(
    db_path: str,
    json_path: Optional[str] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    dialect: Optional[str] = None,
) -> str:
    """Convert ``db_path`` to JSON and write it.

    Returns the output JSON path. Defaults to the same basename with ``.json``.
    ``dialect`` selects camel_tools builtin rewrite rules (defaults to the
    dialect implied by ``meta.description``, else ``MSA``).

    Top-level key order: ``schemaVersion``, ``meta``, then .db section order.
    Definition fields: required → nullable → dtype → values → default → postregex.
    """
    if json_path is None:
        root, _ = os.path.splitext(db_path)
        json_path = root + '.json'

    if dialect is None and metadata:
        # Prefer an explicit dialect field if a caller still passes one;
        # otherwise recover from ``description`` ("morph db of MSA").
        dialect = metadata.get('dialect')
        if not dialect:
            description = str(metadata.get('description') or '')
            prefix = 'morph db of '
            if description.lower().startswith(prefix):
                dialect = description[len(prefix):].strip()
    dialect = _normalize_dialect(dialect)

    data = db_file_to_json(db_path, dialect=dialect)

    # Assemble in .db-like top-level order.
    ordered: Dict[str, Any] = {
        'schemaVersion': data.pop('schemaVersion', SCHEMA_VERSION),
    }
    if metadata:
        # meta keys: dbName, dbVersion, description, regexFormat
        meta_order = ('dbName', 'dbVersion', 'description', 'regexFormat')
        ordered['meta'] = {
            key: metadata[key]
            for key in _ordered_keys(metadata, preferred=meta_order)
        }
    for key in _TOP_LEVEL_KEY_ORDER:
        if key in ('schemaVersion', 'meta'):
            continue
        if key in data:
            ordered[key] = data.pop(key)
    for key, value in data.items():
        ordered[key] = value

    write_db_json(ordered, json_path)
    return json_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert an ALMOR .db file into JSON (faithful Option 1 shape).',
    )
    parser.add_argument('db_path', help='Path to the ALMOR .db file to convert.')
    parser.add_argument(
        '-o', '--output',
        dest='output',
        default=None,
        help='Output JSON path (default: same basename with .json).',
    )
    parser.add_argument(
        '--dialect',
        dest='dialect',
        default=None,
        help=(
            'Dialect label (same as local config "dialect", e.g. msa). '
            'Selects camel_tools rewrite_diac_camel_morph_<dialect>. Default: msa.'
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = export_db_to_json(
        args.db_path,
        args.output,
        dialect=args.dialect,
    )
    print(f'Wrote JSON: {out}')


if __name__ == '__main__':
    main()
