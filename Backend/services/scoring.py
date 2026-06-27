"""
services/scoring.py
──────────────────────────────────────────────────────────────────────────────
Smart, regex-driven scoring for Experience and Education sections.
Replaces the heuristic keyword-count approach in matcher.py with logic that
produces scores on a 0-100 scale consistent with the rest of the pipeline.
"""

import re

from utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that capture an explicit year count from resume text.
# Listed most-specific first so the first match wins.
_YOE_PATTERNS = [
    re.compile(r"(\d+)\s*\+\s*years?",               re.IGNORECASE),  # "5+ years"
    re.compile(r"(\d+)\s*years?\s+of\s+experience",  re.IGNORECASE),  # "3 years of experience"
    re.compile(r"experience\s+of\s+(\d+)\s*years?",  re.IGNORECASE),  # "experience of 4 years"
    re.compile(r"(\d+)\s*years?\s+experience",        re.IGNORECASE),  # "2 years experience"
    re.compile(r"(\d+)\s*years?\s+(?:in|of|with)",   re.IGNORECASE),  # "6 years in ML"
]

# Fallback signals when no explicit year count is found.
_EXPERIENCE_FALLBACK_KEYWORDS = [
    "internship", "intern", "project", "worked on", "worked at",
    "developed", "built", "designed", "implemented", "contributed",
    "freelance", "part-time", "full-time", "contract",
]

# Raw multipliers (0–1 scale); multiplied by 100 at the end.
_YOE_SCORE_MAP = [
    (0,  0,  0.30),   # 0 years explicit mention  → 30
    (1,  2,  0.60),   # 1-2 years                 → 60
    (3,  4,  0.80),   # 3-4 years                 → 80
    (5,  99, 1.00),   # 5+ years                  → 100
]


def _yoe_to_score(years: int) -> float:
    for lo, hi, score in _YOE_SCORE_MAP:
        if lo <= years <= hi:
            return score
    return 1.0  # safety: anything above range is senior


def calculate_experience_score(text: str) -> float:
    """
    Return an experience score in the 0–100 range.

    Strategy
    --------
    1. Try every YOE regex pattern; collect the maximum year value found.
    2. Map that value through _YOE_SCORE_MAP.
    3. If no explicit years are found, fall back to keyword presence (0.5 → 50).
    4. If nothing found at all → 0.3 → 30 (very junior / no evidence).
    """
    text_lower = text.lower()

    # ── Step 1: explicit year extraction ─────────────────────────────────────
    found_years: list[int] = []
    for pattern in _YOE_PATTERNS:
        for match in pattern.finditer(text_lower):
            try:
                found_years.append(int(match.group(1)))
            except (IndexError, ValueError):
                pass

    if found_years:
        max_years = max(found_years)
        raw = _yoe_to_score(max_years)
        score = round(raw * 100, 2)
        logger.debug(f"Experience: explicit YOE={max_years} → score={score}")
        return score

    # ── Step 2: fallback keyword check ───────────────────────────────────────
    hits = [kw for kw in _EXPERIENCE_FALLBACK_KEYWORDS if kw in text_lower]
    if hits:
        score = 50.0
        logger.debug(f"Experience: no YOE found, fallback keywords={hits} → score={score}")
        return score

    # ── Step 3: nothing found ─────────────────────────────────────────────────
    logger.debug("Experience: no signals found → score=30.0")
    return 30.0


# ══════════════════════════════════════════════════════════════════════════════
# EDUCATION
# ══════════════════════════════════════════════════════════════════════════════

# Each tuple: (compiled regex, base score multiplier)
# Listed highest-degree first so the best matching degree wins.
_DEGREE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bph\.?\s*d\b|doctorate|doctoral",                re.IGNORECASE), 1.0),
    (re.compile(r"\bm\.?\s*tech\b|master(?:s|'s)?\b|m\.?\s*sc\b|mba\b|m\.?\s*eng\b",
                re.IGNORECASE), 0.9),
    (re.compile(r"\bb\.?\s*tech\b|bachelor(?:s|'s)?\b|b\.?\s*sc\b|b\.?\s*eng\b|"
                r"\bb\.?\s*e\.?\b|undergraduate\b",
                re.IGNORECASE), 0.7),
    (re.compile(r"\bdiploma\b|associate\b|foundation\b",            re.IGNORECASE), 0.5),
]

# Relevant field keywords that grant a +0.1 bonus on top of the degree score.
_RELEVANT_FIELDS = [
    "computer", "computing", "software", "information technology", "it",
    "artificial intelligence", "machine learning", "data science", "data",
    "electronics", "electrical", "mathematics", "statistics", "physics",
    "engineering", "cybersecurity", "network", "cognitive", "ai",
]

_EDUCATION_NO_DEGREE_SCORE = 0.4   # raw multiplier when no degree keyword found
_EDUCATION_FIELD_BONUS     = 0.1
_EDUCATION_MAX_MULTIPLIER  = 1.0


def calculate_education_score(text: str) -> float:
    """
    Return an education score in the 0–100 range.

    Strategy
    --------
    1. Scan for degree keywords (highest first); take the first match.
    2. Check whether the surrounding context mentions a relevant field.
    3. Apply +0.1 bonus if relevant field found, capped at 1.0.
    4. Multiply by 100 and return.
    """
    text_lower = text.lower()

    # ── Step 1: degree detection ──────────────────────────────────────────────
    base_multiplier = _EDUCATION_NO_DEGREE_SCORE
    degree_found    = False

    for pattern, score_mult in _DEGREE_PATTERNS:
        if pattern.search(text_lower):
            base_multiplier = score_mult
            degree_found    = True
            logger.debug(f"Education: matched pattern '{pattern.pattern}' → base={score_mult}")
            break   # take the highest-priority (first) match only

    if not degree_found:
        logger.debug("Education: no degree keyword found → base=0.4")

    # ── Step 2: relevant field bonus ──────────────────────────────────────────
    field_bonus = 0.0
    matched_field = next((f for f in _RELEVANT_FIELDS if f in text_lower), None)
    if matched_field:
        field_bonus = _EDUCATION_FIELD_BONUS
        logger.debug(f"Education: relevant field '{matched_field}' → +{field_bonus}")

    # ── Step 3: final score ───────────────────────────────────────────────────
    final_multiplier = min(base_multiplier + field_bonus, _EDUCATION_MAX_MULTIPLIER)
    score = round(final_multiplier * 100, 2)
    logger.debug(f"Education: final score={score}")
    return score