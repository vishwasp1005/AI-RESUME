from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_suggestions(
    missing_skills: List[str],
    ats_score: float,
    semantic_score: float,
    experience_score: float,
    education_score: float,
) -> List[str]:
    """Generate actionable resume improvement suggestions."""
    suggestions: List[str] = []

    # ── Missing skills ──────────────────────────────────────────────────────────
    if missing_skills:
        top = missing_skills[:6]
        suggestions.append(
            f"Add the following missing skills to your resume: {', '.join(top)}."
        )

    # ── Semantic alignment ──────────────────────────────────────────────────────
    if semantic_score < 50:
        suggestions.append(
            "Rewrite your professional summary to more closely mirror the language "
            "and keywords used in the job description."
        )
    elif semantic_score < 70:
        suggestions.append(
            "Refine your resume wording to better align with the job description's "
            "terminology and priorities."
        )

    # ── Experience ──────────────────────────────────────────────────────────────
    if experience_score < 40:
        suggestions.append(
            "Expand your work experience section with quantifiable achievements, "
            "specific tools used, and clear date ranges."
        )
    elif experience_score < 65:
        suggestions.append(
            "Strengthen your experience bullet points with measurable outcomes "
            "(e.g. 'Reduced latency by 30%')."
        )

    # ── Education ──────────────────────────────────────────────────────────────
    if education_score < 30:
        suggestions.append(
            "Add your educational background including degree, institution, "
            "graduation year, and relevant coursework."
        )

    # ── Overall score ───────────────────────────────────────────────────────────
    if ats_score < 40:
        suggestions.append(
            "Consider a significant resume overhaul: tailor every section specifically "
            "to this job description."
        )
    elif ats_score < 60:
        suggestions.append(
            "Use action verbs at the start of bullet points to improve readability "
            "and ATS keyword matching."
        )

    # ── Universal tips ──────────────────────────────────────────────────────────
    suggestions.append(
        "Ensure your resume is saved as a clean, text-readable PDF (avoid scanned images)."
    )
    suggestions.append(
        "Include a tailored professional summary at the top that directly addresses "
        "the role requirements."
    )

    logger.debug(f"Generated {len(suggestions)} suggestions.")
    return suggestions
