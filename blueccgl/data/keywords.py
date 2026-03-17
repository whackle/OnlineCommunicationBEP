"""
Conflict keyword tiers for the Israel-Palestine conflict filter.

Usage:
    from blueccgl.data.keywords import CONFLICT_RE, match_keywords, TIER_PATTERNS

    matched = match_keywords("IDF strikes Rafah hospital")
    # → {"Tier1_Core": ["rafah", "idf"]}
"""

from __future__ import annotations

import re
from typing import NamedTuple


# ── Keyword tiers ────────────────────────────────────────────────────────────

TIER1_CORE: list[str] = [
    r"\bgaza\b", r"\bhamas\b", r"\bisrael\b", r"\bpalestine\b", r"\bpalestinian\b",
    r"\bidf\b", r"\bnetanyahu\b", r"\bceasefire\b", r"\bhostage\b", r"\bhostages\b",
    r"\bwest\s+bank\b", r"\brafah\b", r"\bgenocide\b", r"\boccupation\b",
    r"\bzionist\b", r"\bzionism\b", r"\bapartheid\b", r"\bnakba\b", r"\bintifada\b",
    r"\bal[\-\s]aqsa\b", r"\bal[\-\s]shifa\b", r"\bunrwa\b",
]

TIER2_HASHTAGS: list[str] = [
    r"#freepalestine", r"#standwithisrael", r"#gazaunderattack", r"#ceasefirenow",
    r"#bringthemhome", r"#fromtherivertothesea", r"#october7", r"#gazagenocide",
    r"#istandwithisrael", r"#savegaza", r"#freegaza", r"#stopthegenocide",
    r"#israelhamaswar", r"#israelpalestine", r"#gazawar",
]

TIER3_ARABIC: list[str] = [
    r"فلسطين", r"غزة", r"حماس", r"طوفان.?الأقصى",
    r"المقاومة", r"شهداء", r"الأقصى", r"إبادة",
]

TIER4_HEBREW: list[str] = [
    r"חרבות.?ברזל", r"עזה", r"חמאס", r"חטופים", r"ביחד.?ננצח",
]

TIER5_ENTITIES: list[str] = [
    r"\bkhan\s+younis\b", r"\bjabalia\b", r"\bnuseirat\b", r"\bdeir\s+el[\-\s]balah\b",
    r"\bbe'eri\b", r"\bsderot\b", r"\bhezbollah\b", r"\bhouthi\b", r"\bicj\b",
    r"\bprcs\b", r"\bal[\-\s]ahli\b", r"\bnasser\s+hospital\b",
    r"\brefaat\s+alareer\b", r"\bhind\s+rajab\b", r"\bwael\s+dahdouh\b",
    r"\baaron\s+bushnell\b",
    # Note: "gallant" excluded — too ambiguous as a common English word.
    r"\bmuhammad\s+deif\b", r"\babu\s+obeida\b", r"\bismail\s+haniyeh\b",
    r"\bmarwan\s+issa\b",
]

# Ordered tier list: (tier_name, patterns)
TIER_PATTERNS: list[tuple[str, list[str]]] = [
    ("Tier1_Core",      TIER1_CORE),
    ("Tier2_Hashtags",  TIER2_HASHTAGS),
    ("Tier3_Arabic",    TIER3_ARABIC),
    ("Tier4_Hebrew",    TIER4_HEBREW),
    ("Tier5_Entities",  TIER5_ENTITIES),
]

# Single compiled regex for fast screening (any tier match)
_ALL_PATTERNS = TIER1_CORE + TIER2_HASHTAGS + TIER3_ARABIC + TIER4_HEBREW + TIER5_ENTITIES
CONFLICT_RE: re.Pattern = re.compile("|".join(_ALL_PATTERNS), re.IGNORECASE)

# Per-tier compiled patterns for match_keywords()
_TIER_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in pats])
    for name, pats in TIER_PATTERNS
]

# Co-occurrence gate: these patterns must co-occur with at least one OTHER keyword.
# A post matching only one of these is likely metaphorical/off-topic.
_COOCCURRENCE_REQUIRED: re.Pattern = re.compile(
    r"\bhostages?\b", re.IGNORECASE
)

# All patterns EXCEPT the gated ones — used to verify co-occurrence
_ALL_EXCEPT_GATED = [
    p for p in _ALL_PATTERNS
    if not re.fullmatch(r"\\bhostages?\\b", p)  # exclude hostage/hostages patterns
]
_NON_GATED_RE: re.Pattern = re.compile(
    "|".join(
        p for p in _ALL_PATTERNS
        if p not in (r"\bhostage\b", r"\bhostages\b")
    ),
    re.IGNORECASE,
)


def is_conflict(text: str) -> bool:
    """
    Returns True if the text is conflict-related.
    Applies a co-occurrence gate: posts that only match 'hostage(s)'
    must also match at least one other keyword to pass.
    """
    if not CONFLICT_RE.search(text):
        return False
    # If the only match is hostage/hostages, require another keyword too
    if _COOCCURRENCE_REQUIRED.search(text) and not _NON_GATED_RE.search(text):
        return False
    return True


def match_keywords(text: str) -> dict[str, list[str]]:
    """
    Return all matched keywords grouped by tier.
    Only called for posts that passed is_conflict() screening.

    Returns e.g. {"Tier1_Core": ["rafah", "idf"], "Tier2_Hashtags": ["#ceasefirenow"]}
    """
    result: dict[str, list[str]] = {}
    for tier_name, patterns in _TIER_COMPILED:
        hits = []
        for pat in patterns:
            m = pat.search(text)
            if m:
                hits.append(m.group(0).lower())
        if hits:
            result[tier_name] = hits
    return result


def tiers_matched(text: str) -> list[str]:
    """Return list of tier names that matched (for stats)."""
    return [
        tier_name
        for tier_name, patterns in _TIER_COMPILED
        if any(p.search(text) for p in patterns)
    ]
