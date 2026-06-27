"""
services/skill_weights.py
──────────────────────────────────────────────────────────────────────────────
Domain-aware weighted skill scoring.

Weight tiers
------------
3  → Core / must-have  (missing one hurts the score significantly)
2  → Important         (expected for mid-senior roles)
1  → Nice-to-have      (same as the default for all unrecognised skills)

Score formula
-------------
  total_weight   = Σ weight(skill) for skill in jd_skills
  matched_weight = Σ weight(skill) for skill in matched_skills
  skill_score    = (matched_weight / total_weight) × 100
"""

from typing import Dict, List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Default weight applied to every skill not listed below ───────────────────
DEFAULT_WEIGHT = 1

# ── Domain weight tables ──────────────────────────────────────────────────────

AI_ML_WEIGHTS: Dict[str, int] = {
    # Core — tier 3
    "python":           3,
    "machine learning": 3,
    "deep learning":    3,
    # Important — tier 2
    "pytorch":          2,
    "tensorflow":       2,
    "nlp":              2,
    "sql":              2,
    "data analysis":    2,
    "statistics":       2,
    "linear algebra":   2,
    # Nice-to-have — tier 1
    "pandas":           1,
    "numpy":            1,
    "scikit-learn":     1,
    "matplotlib":       1,
    "keras":            1,
    "hugging face":     1,
    "mlops":            1,
    "docker":           1,
}

WEB_DEV_WEIGHTS: Dict[str, int] = {
    # Core — tier 3
    "javascript":   3,
    "html":         3,
    "css":          3,
    # Important — tier 2
    "react":        2,
    "typescript":   2,
    "node.js":      2,
    "rest api":     2,
    "git":          2,
    # Nice-to-have — tier 1
    "vue":          1,
    "angular":      1,
    "next.js":      1,
    "graphql":      1,
    "docker":       1,
    "postgresql":   1,
}

DATA_WEIGHTS: Dict[str, int] = {
    # Core — tier 3
    "sql":          3,
    "python":       3,
    "data analysis":3,
    # Important — tier 2
    "excel":        2,
    "tableau":      2,
    "power bi":     2,
    "statistics":   2,
    "pandas":       2,
    # Nice-to-have — tier 1
    "numpy":        1,
    "r":            1,
    "spark":        1,
    "hadoop":       1,
    "etl":          1,
}

DEVOPS_WEIGHTS: Dict[str, int] = {
    # Core — tier 3
    "docker":       3,
    "kubernetes":   3,
    "linux":        3,
    # Important — tier 2
    "aws":          2,
    "azure":        2,
    "gcp":          2,
    "ci/cd":        2,
    "terraform":    2,
    "git":          2,
    # Nice-to-have — tier 1
    "ansible":      1,
    "jenkins":      1,
    "prometheus":   1,
    "grafana":      1,
    "bash":         1,
}

# ── Domain → weight table mapping ─────────────────────────────────────────────
# Keys must match the strings returned by services.domain_detector.detect_domain()
_DOMAIN_WEIGHTS: Dict[str, Dict[str, int]] = {
    "ai_ml":        AI_ML_WEIGHTS,
    "web_dev":      WEB_DEV_WEIGHTS,
    "data":         DATA_WEIGHTS,
    "devops":       DEVOPS_WEIGHTS,
    # "general" falls through to the empty dict → all skills get DEFAULT_WEIGHT
}


def _get_weights(domain: str) -> Dict[str, int]:
    """Return the weight table for *domain*, falling back to an empty dict."""
    table = _DOMAIN_WEIGHTS.get(domain, {})
    if table:
        logger.debug(f"Weighted scoring: using '{domain}' weight table ({len(table)} entries).")
    else:
        logger.debug(f"Weighted scoring: domain '{domain}' has no table — all weights={DEFAULT_WEIGHT}.")
    return table


# ── Public API ────────────────────────────────────────────────────────────────

def compute_weighted_skills_score(
    resume_skills: List[str],
    jd_skills:     List[str],
    domain:        str = "general",
) -> Tuple[float, List[str], List[str]]:
    """
    Compute a weighted skill score (0–100) that gives more impact to
    critical skills than to nice-to-haves.

    Parameters
    ----------
    resume_skills : skills extracted from the candidate's resume
    jd_skills     : skills extracted from the (possibly expanded) JD
    domain        : role domain string from domain_detector (e.g. "ai_ml")

    Returns
    -------
    (skill_score, matched_skills, missing_skills)
        skill_score   : float 0–100
        matched_skills: sorted list of skills present in both resume and JD
        missing_skills: sorted list of JD skills absent from the resume
    """
    if not jd_skills:
        return 0.0, [], []

    weights = _get_weights(domain)

    resume_set = {s.lower() for s in resume_skills}
    jd_set     = {s.lower() for s in jd_skills}

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)

    total_weight   = sum(weights.get(s, DEFAULT_WEIGHT) for s in jd_set)
    matched_weight = sum(weights.get(s, DEFAULT_WEIGHT) for s in matched)

    raw_score  = matched_weight / max(total_weight, 1)
    skill_score = round(raw_score * 100, 2)

    logger.debug(
        f"Weighted skills: matched_w={matched_weight}/{total_weight} "
        f"({len(matched)} skills) → score={skill_score}"
    )
    return skill_score, matched, missing