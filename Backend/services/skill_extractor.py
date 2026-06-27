import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"


def _load_skills() -> Dict[str, List[str]]:
    with open(_SKILLS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Load once at import time ──────────────────────────────────────────────────
_SKILLS_DB: Dict[str, List[str]] = _load_skills()
_ALL_SKILLS: Set[str] = {
    skill.lower()
    for skills in _SKILLS_DB.values()
    for skill in skills
}


def extract_skills(text: str) -> List[str]:
    """Return a list of skills found in *text* using keyword matching."""
    text_lower = text.lower()
    found: Set[str] = set()

    for skill in _ALL_SKILLS:
        # Use word-boundary-like match so 'r' doesn't match 'rust'
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)

    result = sorted(found)
    logger.debug(f"Extracted {len(result)} skills from text.")
    return result


def get_all_skills() -> Set[str]:
    """Expose the full skill set (for missing-skill computation)."""
    return _ALL_SKILLS