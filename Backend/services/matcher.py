import re
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer, util
from config import EMBEDDING_MODEL
from services.skill_extractor import extract_skills
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Load model once ───────────────────────────────────────────────────────────
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
_MODEL = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded.")

# ── Section detection keywords ─────────────────────────────────────────────────
_EXPERIENCE_KEYWORDS = [
    "experience", "work history", "employment", "internship", "worked at",
    "years of experience", "professional background", "career"
]
_EDUCATION_KEYWORDS = [
    "education", "degree", "university", "college", "bachelor", "master",
    "phd", "b.tech", "m.tech", "b.e.", "m.e.", "b.sc", "m.sc",
    "mba", "diploma", "graduation", "coursework"
]


def compute_semantic_score(resume_text: str, jd_text: str) -> float:
    """Cosine similarity (0-100) between resume and JD embeddings."""
    emb_resume = _MODEL.encode(resume_text, convert_to_tensor=True)
    emb_jd = _MODEL.encode(jd_text, convert_to_tensor=True)
    cosine = float(util.cos_sim(emb_resume, emb_jd))
    score = round(max(0.0, min(cosine, 1.0)) * 100, 2)
    logger.debug(f"Semantic score: {score}")
    return score


def compute_skills_score(
    resume_skills: List[str], jd_skills: List[str]
) -> Tuple[float, List[str], List[str]]:
    """
    Returns:
        skills_score (0-100),
        matched_skills,
        missing_skills
    """
    if not jd_skills:
        return 0.0, [], []

    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    score = round((len(matched) / len(jd_set)) * 100, 2) if jd_set else 0.0
    logger.debug(f"Skills score: {score}  matched={len(matched)}  missing={len(missing)}")
    return score, matched, missing


def compute_experience_score(resume_text: str) -> float:
    """Heuristic experience score based on keyword presence & year mentions."""
    text_lower = resume_text.lower()
    keyword_hits = sum(1 for kw in _EXPERIENCE_KEYWORDS if kw in text_lower)
    # Count year-range patterns like "2019 – 2023" or "Jan 2020 - Present"
    year_patterns = re.findall(
        r"\b(20\d{2}|19\d{2})\b.{0,10}(present|current|20\d{2}|19\d{2})",
        text_lower,
    )
    roles = len(year_patterns)
    score = min(100.0, keyword_hits * 10 + roles * 15)
    logger.debug(f"Experience score: {score}")
    return round(score, 2)


def compute_education_score(resume_text: str) -> float:
    """Heuristic education score based on keyword presence."""
    text_lower = resume_text.lower()
    hits = sum(1 for kw in _EDUCATION_KEYWORDS if kw in text_lower)
    score = min(100.0, hits * 12)
    logger.debug(f"Education score: {score}")
    return round(score, 2)