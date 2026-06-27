"""
routes/analyze.py
──────────────────────────────────────────────────────────────────────────────
All existing endpoints preserved.
New addition: smart JD expansion via jd_enhancer before any processing.
"""

import hashlib
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import (
    MATCH_AVERAGE,
    MATCH_EXCELLENT,
    MATCH_GOOD,
    WEIGHT_EDUCATION,
    WEIGHT_EXPERIENCE,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILLS,
)
from models.schema import AnalysisData, AnalysisResponse, SectionScores
from services.domain_detector import detect_domain
from services.domain_skills import (
    build_domain_evidence,
    build_jd_analysis,
    build_score_breakdown,
    classify_skills,
    classify_hybrid_skills,
    compute_role_confidence,        # ← NEW
    detect_domain_from_skills,
    detect_multi_domain,
    evaluate_strict_ats,
    extract_jd_skills,
    extract_project_signals,
    filter_domain_skills,
    HybridClassifiedSkills,
    normalize_list,
    simulate_score_improvement,
    smart_suggestions,
    split_skills,
)
from services.jd_enhancer import expand_job_description
from services.llm_service import generate_ai_insight, generate_comparison_insight
from services.matcher import compute_semantic_score   # semantic resume↔JD score
from services.scoring import (
    calculate_education_score,
    calculate_experience_score,
)
from services.parser import extract_text_from_pdf
from services.skill_extractor import extract_skills
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _get_match_level(score: float) -> str:
    if score >= MATCH_EXCELLENT:
        return "Excellent"
    elif score >= MATCH_GOOD:
        return "Good"
    elif score >= MATCH_AVERAGE:
        return "Average"
    else:
        return "Poor"


def _get_hire_decision(score: float) -> str:
    if score > 75:
        return "Strong Hire"
    elif score >= 40:
        return "Consider"
    else:
        return "Reject"


def _interpreted_role(original_jd: str) -> str:
    """Return a clean display label from the original short input."""
    return original_jd.strip().rstrip(".,;:")[:60].title()


# ── Role lock system ───────────────────────────────────────────────────────────
# Maps human-readable role names (lowercase, normalised) to DOMAINS keys.
# STRICT TAXONOMY: ui_ux_designer, frontend_developer, devops_engineer, ai_engineer
# are DISTINCT — never auto-merged by partial matching.
_ROLE_ALIASES: dict = {
    # Software engineering variants
    "software engineer":         "software_engineer",
    "software developer":        "software_engineer",
    "software dev":              "software_engineer",
    "swe":                       "software_engineer",
    # Full-stack — maps to dedicated full_stack_engineer domain
    "full stack":                "full_stack_engineer",
    "fullstack":                 "full_stack_engineer",
    "full stack engineer":       "full_stack_engineer",
    "full stack developer":      "full_stack_engineer",
    "fullstack engineer":        "full_stack_engineer",
    "fullstack developer":       "full_stack_engineer",
    "full-stack engineer":       "full_stack_engineer",
    "full-stack developer":      "full_stack_engineer",
    # Frontend — maps to strict frontend_developer (NOT ui_ux)
    "frontend engineer":         "frontend_developer",
    "frontend developer":        "frontend_developer",
    "frontend dev":              "frontend_developer",
    "front end developer":       "frontend_developer",
    "front-end developer":       "frontend_developer",
    "react developer":           "frontend_developer",
    "web developer":             "frontend_developer",
    # Backend
    "backend engineer":          "backend",
    "backend developer":         "backend",
    "backend dev":               "backend",
    "back end developer":        "backend",
    "back-end developer":        "backend",
    "api developer":             "backend",
    "server side developer":     "backend",
    # UI/UX — STRICTLY separate from frontend
    "ui ux designer":            "ui_ux_designer",
    "ui/ux designer":            "ui_ux_designer",
    "ux designer":               "ui_ux_designer",
    "ui designer":               "ui_ux_designer",
    "product designer":          "ui_ux_designer",
    "ux researcher":             "ui_ux_designer",
    "interaction designer":      "ui_ux_designer",
    "visual designer":           "ui_ux_designer",
    "user experience designer":  "ui_ux_designer",
    # AI / ML — distinct from data analyst
    "ai engineer":               "ai_engineer",
    "ai/ml engineer":            "ai_engineer",
    "ml engineer":               "ai_engineer",
    "ai ml engineer":            "ai_engineer",
    "machine learning engineer": "ai_engineer",
    "deep learning engineer":    "ai_engineer",
    "nlp engineer":              "ai_engineer",
    "computer vision engineer":  "ai_engineer",
    # AI/ML research (broader — maps to ai_ml not ai_engineer)
    "data scientist":            "ai_ml",
    "research engineer":         "ai_ml",
    # Data — strictly separate from AI engineer
    "data analyst":              "data_analyst",
    "data analytics":            "data_analyst",
    "business analyst":          "data_analyst",
    "bi analyst":                "data_analyst",
    "data engineer":             "data",
    "etl developer":             "data",
    # DevOps / Cloud — maps to strict devops_engineer
    "devops engineer":           "devops_engineer",
    "cloud engineer":            "devops_engineer",
    "site reliability engineer": "devops_engineer",
    "sre":                       "devops_engineer",
    "platform engineer":         "devops_engineer",
    "infrastructure engineer":   "devops_engineer",
    "mlops engineer":            "devops_engineer",
    # Mobile
    "mobile developer":          "mobile_developer",
    "ios developer":             "mobile_developer",
    "android developer":         "mobile_developer",
    "react native developer":    "mobile_developer",
    # Product
    "product manager":           "product_manager",
    "pm":                        "product_manager",
    "technical product manager": "product_manager",
}

# Valid domain keys — must stay in sync with domain_skills.DOMAINS
_VALID_DOMAINS: frozenset = frozenset([
    "software_engineer", "frontend", "backend",
    "ai_ml", "data_analyst", "devops_cloud",
    "web_dev", "data", "devops",
    # Strict taxonomy additions
    "frontend_developer", "ui_ux_designer",
    "devops_engineer", "ai_engineer",
    "mobile_developer", "product_manager",
    "full_stack_engineer",        # ← NEW
])

# Domains that are STRICTLY distinct — partial-match fallback is DISABLED for these.
_STRICT_DISTINCT_ROLES: frozenset = frozenset([
    "ui_ux_designer", "frontend_developer",
    "ai_engineer", "mobile_developer", "product_manager",
    "full_stack_engineer",        # ← NEW
])

# Pairs that should NEVER be treated as the same domain
_FORBIDDEN_DOMAIN_PAIRS: set = {
    frozenset({"ui_ux_designer",     "frontend_developer"}),
    frozenset({"ui_ux_designer",     "frontend"}),
    frozenset({"ai_engineer",        "data_analyst"}),
    frozenset({"ai_engineer",        "devops_engineer"}),
    frozenset({"backend",            "devops_engineer"}),
    frozenset({"frontend_developer", "backend"}),
    frozenset({"full_stack_engineer","data_analyst"}),    # ← NEW
    frozenset({"full_stack_engineer","devops_engineer"}), # ← NEW
    frozenset({"full_stack_engineer","ui_ux_designer"}),  # ← NEW
}


def _normalize_user_role(user_role: Optional[str]) -> Optional[str]:
    """
    Coerce a free-text role string into a DOMAINS key.

    Resolution order:
      1. Direct DOMAINS key match  (e.g. "ui_ux_designer")
      2. _ROLE_ALIASES lookup      (e.g. "ux designer" → "ui_ux_designer")
      3. Partial key match — DISABLED for _STRICT_DISTINCT_ROLES to prevent
         "ui" accidentally matching "devops" or "frontend" matching "ui_ux"
      4. None — unrecognised; caller falls back to auto-detected domain
    """
    if not user_role:
        return None
    raw = user_role.lower().strip().rstrip(".,;:")

    # 1. Direct key match
    if raw in _VALID_DOMAINS:
        return raw

    # 2. Alias lookup (highest precision)
    if raw in _ROLE_ALIASES:
        return _ROLE_ALIASES[raw]

    # 3. Partial match — only for non-strict roles
    for key in _VALID_DOMAINS:
        if key in _STRICT_DISTINCT_ROLES:
            continue   # NEVER partial-match strict taxonomy roles
        if raw in key or key in raw:
            return key

    logger.warning(f"User role '{user_role}' not recognised — falling back to auto-detection.")
    return None


def _detect_role_confusion(
    user_role:      Optional[str],
    final_domain:   str,
    detected_domain: str,
) -> bool:
    """
    Detect when auto-detection would have assigned a FORBIDDEN domain pair
    relative to the user-selected role — signals that the system would have
    made a wrong domain collapse without the role lock.

    Returns True when role confusion was present (and role lock prevented it).

    Example:
      user_role = "ui_ux_designer", detected = "frontend_developer"
      → frozenset({"ui_ux_designer", "frontend_developer"}) is a FORBIDDEN pair
      → role_confusion_detected = True
    """
    if not user_role or final_domain == detected_domain:
        return False   # no override happened — no confusion to report
    pair = frozenset({final_domain, detected_domain})
    return pair in _FORBIDDEN_DOMAIN_PAIRS


def resolve_domain(
    user_role:       Optional[str],
    detected_domain: str,
    jd_domain:       Optional[str] = None,
) -> tuple:    # (final_domain, override_applied, domain_source)
    """
    DOMAIN RESOLUTION PIPELINE — strict 3-step priority order.

    Step 1 — USER ROLE (highest priority, never overridden)
    Step 2 — JD domain   (used only when user_role is absent)
    Step 3 — Fallback detected domain (last resort)

    Returns
    ───────
    (final_domain, override_applied, domain_source)
      final_domain     — the LOCKED domain key used for ALL downstream analysis
      override_applied — True when user_role was present and took precedence
      domain_source    — "user_role" | "jd" | "fallback"
    """
    # Step 1 — user role (strict lock)
    normalised = _normalize_user_role(user_role)
    if normalised:
        logger.info(
            f"[DOMAIN LOCK] Step 1 — user_role='{user_role}' → '{normalised}' "
            f"(auto-detected '{detected_domain}' ignored, domain_locked=True)"
        )
        return normalised, True, "user_role"

    # Step 2 — JD domain (only when no user role)
    if jd_domain and jd_domain != detected_domain:
        logger.info(
            f"[DOMAIN LOCK] Step 2 — jd_domain='{jd_domain}' used "
            f"(fallback '{detected_domain}' overridden)"
        )
        return jd_domain, False, "jd"

    # Step 3 — fallback from skill overlap detection
    logger.info(
        f"[DOMAIN LOCK] Step 3 fallback — detected_domain='{detected_domain}' used"
    )
    return detected_domain, False, "fallback"


def _detect_cross_domain_note(
    resume_skills:    List[str],
    final_domain:     str,
    detected_domain:  str,
    domain_source:    str,
) -> Optional[str]:
    """
    When the user-selected role conflicts with what the resume suggests,
    return a transparent note for the UI.  Returns None when no mismatch.

    Example:
      Role = "software_engineer", resume is AI/ML heavy
      → "Resume shows cross-domain exposure in ai_ml, but evaluation
         is based on software_engineer role as requested."
    """
    # Only relevant when user_role overrides detection
    if domain_source != "user_role":
        return None
    if detected_domain == final_domain:
        return None

    # Check if resume has a material number of skills from the detected domain
    from services.domain_skills import DOMAINS, normalize
    detected_cfg  = DOMAINS.get(detected_domain, {})
    detected_core = {normalize(s) for s in detected_cfg.get("core", [])}
    resume_norm   = {normalize(s) for s in resume_skills}
    overlap       = detected_core & resume_norm

    if len(overlap) >= 2:   # meaningful cross-domain exposure
        detected_label = detected_domain.replace("_", " ").title()
        final_label    = final_domain.replace("_", " ").title()
        return (
            f"Resume shows cross-domain exposure in {detected_label} "
            f"({', '.join(sorted(overlap)[:3])}), but evaluation is based on "
            f"{final_label} role as requested. Cross-domain skills are noted "
            f"but do not affect the primary domain score."
        )
    return None


# ── NEW: Adaptive skill filtering for short JD inputs ─────────────────────────
# Ordered by how universally relevant they are across common roles.
# When the JD is a bare job title we cap analysis to these core skills so the
# missing-skills list stays meaningful and ATS scores stay realistic.
_PRIORITY_SKILLS_BY_DOMAIN = [
    # Data / ML
    "python", "machine learning", "deep learning", "sql", "tensorflow",
    "pytorch", "nlp", "data analysis", "pandas", "numpy", "scikit-learn",
    # Software engineering
    "javascript", "typescript", "react", "node.js", "java", "c++",
    "docker", "kubernetes", "git", "rest api",
    # Cloud / DevOps
    "aws", "azure", "gcp", "ci/cd", "linux",
    # General
    "communication", "problem solving", "agile", "testing",
]

_SHORT_JD_WORD_LIMIT = 5   # titles at or below this word count get filtered
_CORE_SKILL_CAP      = 8   # max skills kept from the expanded JD


def filter_core_skills(skills: list) -> list:
    """
    From a large extracted skill list keep only the top _CORE_SKILL_CAP skills
    that appear in the priority list (order preserved).
    Falls back to the first _CORE_SKILL_CAP skills if none match priorities.
    """
    skills_lower = [s.lower() for s in skills]
    prioritised = [s for s in _PRIORITY_SKILLS_BY_DOMAIN if s in skills_lower]

    if prioritised:
        logger.debug(f"Adaptive filter: {len(skills)} → {len(prioritised[:_CORE_SKILL_CAP])} core skills")
        return prioritised[:_CORE_SKILL_CAP]

    # Fallback: no priority match (unlikely) — just truncate
    logger.debug(f"Adaptive filter fallback: capping at {_CORE_SKILL_CAP} skills")
    return skills[:_CORE_SKILL_CAP]


async def _run_pipeline(
    resume_text:      str,            # ← pre-extracted text, NEVER re-parsed here
    filename:         str,
    jd:               str,
    jd_skills:        list,
    domain:           str,
    secondary_domain: Optional[str] = None,
    request_id:       str           = "",
) -> dict:
    """
    Stateless analysis pipeline.

    CONTRACT (enforced by architecture):
    ─────────────────────────────────────
    • resume_text is extracted ONCE by the caller (endpoint).
    • This function NEVER reads UploadFile, never calls extract_text_from_pdf,
      and never opens a PDF stream.  Any such call here is a bug.
    • Every local variable is function-scoped — no shared mutable state.

    Raises RuntimeError when resume_text is empty (caller should abort immediately).
    """
    rid = request_id or str(uuid.uuid4())[:8]

    logger.info(
        f"[{rid}] Pipeline start: '{filename}' "
        f"| chars={len(resume_text)} | domain='{domain}'"
        f"{'/' + secondary_domain if secondary_domain else ''}"
    )

    # Guard — abort immediately, do not continue with empty text
    if not resume_text.strip():
        raise RuntimeError(
            f"Resume text for '{filename}' is empty. "
            "Ensure the file is not scanned or image-based."
        )

    # ── Debug snippet — confirms text is from the correct PDF ────────────────
    snippet = resume_text[:200].replace("\n", " ")
    logger.debug(f"[{rid}] Text snippet: {snippet!r}")

    # ── Skill extraction (text only — no PDF re-read) ─────────────────────────
    resume_skills      = extract_skills(resume_text)
    resume_skills_norm = normalize_list(resume_skills)

    logger.info(
        f"[{rid}] '{filename}' skills extracted ({len(resume_skills)}): "
        f"{sorted(resume_skills)[:20]}"
    )

    # ── Project-context validation ────────────────────────────────────────────
    # Re-evaluate domain against the FULL resume text (not just skill tokens).
    # "AI Resume Scanner using LLMs" is detected here even if skills.json
    # didn't extract it. Logs a mismatch warning if project context strongly
    # disagrees with the JD-detected domain.
    project_signals = extract_project_signals(resume_text)
    if project_signals:
        # Find the project-context winner
        proj_primary = max(project_signals, key=lambda d: project_signals[d])
        logger.info(
            f"[{rid}] Project-context signals: "
            + ", ".join(f"{d}={s:.0f}" for d, s in
                        sorted(project_signals.items(), key=lambda x: -x[1])[:4])
        )
        if proj_primary != domain and project_signals[proj_primary] >= 24.0:
            logger.warning(
                f"[{rid}] Domain mismatch: JD-detected='{domain}' but resume "
                f"project context strongly suggests '{proj_primary}' "
                f"(boost={project_signals[proj_primary]:.0f}pts). "
                f"Using locked domain='{domain}' — set user_role to override."
            )

    # ── Resume-based domain detection (independent of JD) ───────────────────
    # This is the "don't follow JD blindly" check.
    # We detect what domain the resume ITSELF suggests, then compare to the
    # JD-locked domain.  If they agree → high alignment.
    # If they disagree → we return both and flag it — the user can decide.
    resume_domain, resume_secondary, _ = detect_multi_domain(
        resume_skills_norm,
        resume_text=resume_text,
    )

    # JD alignment — how well the resume's actual profile matches the JD role
    if resume_domain == domain:
        jd_alignment = "aligned"
    elif secondary_domain and resume_domain == secondary_domain:
        jd_alignment = "partial"
    elif resume_secondary and resume_secondary == domain:
        jd_alignment = "partial"
    else:
        jd_alignment = "mismatch"

    if jd_alignment == "mismatch":
        logger.info(
            f"[{rid}] JD domain='{domain}' but resume suggests '{resume_domain}' "
            f"— returning both; candidate may be applying cross-domain."
        )

    # Numeric role confidence (0-100) based on resume evidence strength
    confidence_scores    = compute_role_confidence(
        resume_skills_norm, domain, secondary_domain
    )
    primary_confidence   = confidence_scores["primary"]
    secondary_confidence = confidence_scores["secondary"]

    logger.info(
        f"[{rid}] Role confidence: {domain}={primary_confidence}% "
        f"secondary={secondary_domain!r}={secondary_confidence}% "
        f"alignment={jd_alignment}"
    )

    semantic_score = compute_semantic_score(resume_text, jd)
    classified = classify_hybrid_skills(
        jd_skills, resume_skills_norm, domain, secondary_domain
    )

    skills_score   = classified.skill_score()
    matched_skills = classified.all_matched
    missing_skills = classified.all_missing[:8]

    logger.debug(
        f"[{rid}] Skills: "
        f"core {len(classified.matched_core)}/{len(classified.matched_core)+len(classified.missing_core)} "
        f"jd {len(classified.matched_jd)}/{len(classified.matched_jd)+len(classified.missing_jd)} "
        f"opt {len(classified.matched_optional)}/{len(classified.matched_optional)+len(classified.missing_optional)} "
        f"sec={len(classified.secondary_matched)} score={skills_score}"
    )

    # ── Build domain evidence (returned in response for UI + debugging) ────────
    domain_evidence = build_domain_evidence(
        primary_domain  = domain,
        skill_evidence  = classified.matched_core + classified.matched_jd,
        combo_hits      = [],    # combo data not available inside pipeline; logged at detection
        project_signals = project_signals,
        resume_skills   = resume_skills_norm,
    )

    experience_score = calculate_experience_score(resume_text)
    education_score  = calculate_education_score(resume_text)

    ats_score = round(
        WEIGHT_SEMANTIC  * semantic_score
        + WEIGHT_SKILLS  * skills_score
        + WEIGHT_EXPERIENCE * experience_score
        + WEIGHT_EDUCATION  * education_score,
        2,
    )
    match_level = _get_match_level(ats_score)

    logger.info(
        f"[{rid}] '{filename}' → ATS={ats_score} | "
        f"Semantic={semantic_score} Skills={skills_score} "
        f"Exp={experience_score} Edu={education_score}"
    )

    # ── Smart tiered suggestions (hybrid-aware, never empty) ─────────────────
    suggestions = smart_suggestions(classified, ats_score, secondary_domain)

    # ── Explainable score breakdown (deterministic) ───────────────────────────
    score_breakdown = build_score_breakdown(classified)

    # ── Strict ATS evaluation — finds gaps the JD alone may not surface ───────
    strict_eval = evaluate_strict_ats(
        resume_skills=resume_skills_norm,
        classified=classified,
        domain=domain,
        ats_score=ats_score,
    )
    critical_missing = strict_eval["critical_missing"]
    weak_areas       = strict_eval["weak_areas"]
    readiness_level  = strict_eval["readiness_level"]

    # ── Score improvement simulator ───────────────────────────────────────────
    score_simulation = simulate_score_improvement(
        classified=classified,
        jd_skills=jd_skills,
        resume_skills=resume_skills_norm,
        domain=domain,
        secondary_domain=secondary_domain,
        current_ats_score=ats_score,
        current_skill_score=skills_score,
        weight_skills=WEIGHT_SKILLS,
    )

    ai_insight = generate_ai_insight(
        resume_text=resume_text,
        jd_text=jd,
        ats_score=ats_score,
        match_level=match_level,
        domain=domain,
        matched_skills=matched_skills,
        missing_skills=critical_missing,
        weak_areas=weak_areas,
        inferred_skills=classified.inferred_skills,   # ← NEW
    )

    return dict(
        filename=filename,
        ats_score=ats_score,
        match_level=match_level,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        inferred_skills=dict(classified.inferred_skills),
        critical_missing=critical_missing,
        weak_areas=weak_areas,
        readiness_level=readiness_level,
        domain_evidence=domain_evidence,
        score_breakdown=score_breakdown,
        score_simulation=score_simulation,
        section_scores=dict(
            semantic=semantic_score,
            skills=skills_score,
            experience=experience_score,
            education=education_score,
        ),
        suggestions=suggestions,
        ai_insight=ai_insight,
        primary_domain=domain,
        domain=domain,
        secondary_domain=secondary_domain,
        is_hybrid=secondary_domain is not None,
        # ── Role confidence (new) ─────────────────────────────────────────
        primary_confidence=primary_confidence,
        secondary_confidence=secondary_confidence,
        resume_detected_domain=resume_domain,
        resume_detected_secondary=resume_secondary,
        jd_alignment=jd_alignment,
    )


# ── Single resume ──────────────────────────────────────────────────────────────
@router.post("/analyze")
async def analyze_resume(
    file:      UploadFile       = File(..., description="PDF resume"),
    jd:        str              = Form(..., description="Job description text"),
    user_role: Optional[str]   = Form(None, description="User-selected role (e.g. 'software_engineer')"),
):
    # ── STEP 1: Read PDF bytes ONCE, hash, extract text — never re-read ─────────
    request_id = str(uuid.uuid4())[:12]
    raw_bytes  = await file.read()   # single and final read of the stream
    file_hash  = hashlib.sha256(raw_bytes).hexdigest()[:16]
    filename   = file.filename or "resume.pdf"

    logger.info(
        f"[{request_id}] === Single analysis started ==="
        f" file='{filename}' size={len(raw_bytes)}B hash={file_hash}"
        f" user_role={user_role!r}"
    )
    logger.debug(f"[{request_id}] PDF bytes read once — stream will not be touched again.")

    # ── STEP 2: Extract text ONCE — all downstream receives this string ────────
    try:
        resume_text = extract_text_from_pdf(raw_bytes, filename)
        logger.info(
            f"[{request_id}] PDF parsed successfully: {len(resume_text)} chars"
            f" | parse_count=1 (enforced)"
        )
    except Exception as exc:
        logger.error(f"[{request_id}] PDF extraction failed: {exc}")
        return JSONResponse(
            content=AnalysisResponse(
                status="error",
                error=f"Could not extract text from '{filename}'. "
                      "Ensure the file is not scanned or image-based."
            ).model_dump(),
            status_code=422,
        )

    if not resume_text.strip():
        logger.error(f"[{request_id}] Extracted text is empty — aborting.")
        return JSONResponse(
            content=AnalysisResponse(
                status="error",
                error=f"No readable text found in '{filename}'. "
                      "Ensure the file is not a scanned image."
            ).model_dump(),
            status_code=422,
        )

    # Confirm text origin for debugging stale-data reports
    snippet = resume_text[:120].replace("\n", " ")
    logger.debug(f"[{request_id}] Text snippet: {snippet!r}")

    try:
        # ── NEW: expand short JD before any processing ───────────────────
        original_jd = jd.strip()
        jd, jd_was_expanded = expand_job_description(original_jd)
        interpreted_role = _interpreted_role(original_jd) if jd_was_expanded else None
        if jd_was_expanded:
            logger.info(f"[{request_id}] JD auto-expanded. Interpreted role: '{interpreted_role}'")
        # ────────────────────────────────────────────────────────────────

        # ── Deterministic JD skill extraction ────────────────────────────────
        jd_skills_det = extract_jd_skills(jd)
        jd_skills_ext = extract_skills(jd)
        jd_skills_raw = normalize_list(jd_skills_det + jd_skills_ext)

        # Adaptive filter for short/title-only JD inputs
        if len(original_jd.split()) <= _SHORT_JD_WORD_LIMIT:
            jd_skills_raw = filter_core_skills(jd_skills_raw)
            logger.info(f"[{request_id}] Short JD — jd_skills capped to {len(jd_skills_raw)} core skills.")

        # Deterministic multi-domain detection — JD skills + resume project context
        detected_domain, detected_secondary, jd_confidence = detect_multi_domain(
            jd_skills_raw,
            resume_text=resume_text,    # ← project context from full resume text
        )
        logger.info(
            f"[{request_id}] Domain detected: primary='{detected_domain}' "
            f"secondary={detected_secondary!r} (confidence={jd_confidence})"
        )

        # ── DOMAIN RESOLUTION PIPELINE (3-step strict priority) ──────────────
        final_domain, override_applied, domain_source = resolve_domain(
            user_role, detected_domain, jd_domain=detected_domain
        )
        final_secondary = detected_secondary if detected_secondary != final_domain else None
        # ─────────────────────────────────────────────────────────────────────

        # Remove off-domain skills using FINAL primary domain
        jd_skills = filter_domain_skills(jd_skills_raw, final_domain)
        logger.info(f"Domain filter '{final_domain}': {len(jd_skills)} skills retained.")

        jd_analysis = build_jd_analysis(jd_skills, final_domain, jd_confidence, final_secondary)
        # ────────────────────────────────────────────────────────────────────

        result = await _run_pipeline(
            resume_text=resume_text,        # ← pre-extracted string, NOT UploadFile
            filename=filename,
            jd=jd,
            jd_skills=jd_skills,
            domain=final_domain,
            secondary_domain=final_secondary,
            request_id=request_id,
        )

        # Cross-domain mismatch note — transparent to UI when role overrides detection
        cross_domain_note = _detect_cross_domain_note(
            resume_skills=result.get("matched_skills", []),
            final_domain=final_domain,
            detected_domain=detected_domain,
            domain_source=domain_source,
        )

        # Role confusion flag — True when domain lock prevented a forbidden collapse
        role_confusion_detected = _detect_role_confusion(
            user_role, final_domain, detected_domain
        )

        response = AnalysisResponse(
            status="success",
            data=AnalysisData(
                filename=result["filename"],
                ats_score=result["ats_score"],
                match_level=result["match_level"],
                matched_skills=result["matched_skills"],
                missing_skills=result["missing_skills"],
                section_scores=SectionScores(**result["section_scores"]),
                suggestions=result["suggestions"],
                ai_insight=result["ai_insight"],
                domain=result["domain"],
            ),
            error=None,
        )
        logger.info(
            f"[{request_id}] === Single analysis complete ==="
            f" ATS={result['ats_score']} hash={file_hash}"
        )

        # Attach expansion metadata alongside the typed schema fields
        response_dict = response.model_dump()
        response_dict["jd_expanded"]        = jd_was_expanded
        response_dict["interpreted_role"]   = interpreted_role
        response_dict["score_breakdown"]    = result["score_breakdown"]
        response_dict["score_simulation"]   = result["score_simulation"]
        response_dict["jd_analysis"]        = jd_analysis
        # ── Domain resolution provenance (full pipeline transparency) ─────
        response_dict["selected_role"]           = user_role or None
        response_dict["user_role"]               = user_role or None
        response_dict["jd_domain"]               = detected_domain
        response_dict["detected_domain"]         = detected_domain
        response_dict["final_domain"]            = final_domain
        response_dict["final_domain_used"]       = final_domain
        response_dict["domain_source"]           = domain_source
        response_dict["domain_locked"]           = True
        response_dict["override_applied"]        = override_applied
        response_dict["cross_domain_note"]       = cross_domain_note
        response_dict["role_confusion_detected"] = role_confusion_detected
        # ─────────────────────────────────────────────────────────────────
        response_dict["primary_domain"]          = result["primary_domain"]
        response_dict["secondary_domain"]        = result["secondary_domain"]
        response_dict["is_hybrid"]               = result["is_hybrid"]
        response_dict["critical_missing"]        = result["critical_missing"]
        response_dict["weak_areas"]              = result["weak_areas"]
        response_dict["readiness_level"]         = result["readiness_level"]
        response_dict["inferred_skills"]         = result["inferred_skills"]
        response_dict["domain_evidence"]          = result["domain_evidence"]
        # ── Role confidence + JD alignment ────────────────────────────────
        response_dict["primary_confidence"]       = result["primary_confidence"]
        response_dict["secondary_confidence"]     = result["secondary_confidence"]
        response_dict["resume_detected_domain"]   = result["resume_detected_domain"]
        response_dict["resume_detected_secondary"]= result["resume_detected_secondary"]
        response_dict["jd_alignment"]             = result["jd_alignment"]
        # ── Request isolation identifiers ─────────────────────────────────
        response_dict["request_id"]               = request_id
        response_dict["file_hash"]                = file_hash
        return response_dict

    except RuntimeError as exc:
        logger.error(f"RuntimeError: {exc}")
        return JSONResponse(
            content=AnalysisResponse(status="error", error=str(exc)).model_dump(),
            status_code=422,
        )
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            content=AnalysisResponse(
                status="error", error="An unexpected error occurred. Please try again."
            ).model_dump(),
            status_code=500,
        )


# ── Multi-resume comparison ────────────────────────────────────────────────────
@router.post("/analyze-multiple")
async def analyze_multiple_resumes(
    files:     List[UploadFile] = File(..., description="One or more PDF resumes"),
    jd:        str              = Form(..., description="Job description text"),
    user_role: Optional[str]   = Form(None, description="User-selected role (e.g. 'software_engineer')"),
):
    request_id = str(uuid.uuid4())[:12]
    logger.info(
        f"[{request_id}] === Multi-analysis started: {len(files)} file(s)"
        f" user_role={user_role!r} ==="
    )

    if not files:
        return JSONResponse(
            content={"status": "error", "data": [], "error": "No files uploaded."},
            status_code=422,
        )

    # ── NEW: expand short JD before any processing ───────────────────────────
    original_jd = jd.strip()
    jd, jd_was_expanded = expand_job_description(original_jd)
    interpreted_role = _interpreted_role(original_jd) if jd_was_expanded else None
    if jd_was_expanded:
        logger.info(f"[{request_id}] JD auto-expanded. Interpreted role: '{interpreted_role}'")
    # ────────────────────────────────────────────────────────────────────────

    # ── Deterministic JD skill extraction ────────────────────────────────────
    jd_skills_det = extract_jd_skills(jd)
    jd_skills_ext = extract_skills(jd)
    jd_skills_raw = normalize_list(jd_skills_det + jd_skills_ext)

    if len(original_jd.split()) <= _SHORT_JD_WORD_LIMIT:
        jd_skills_raw = filter_core_skills(jd_skills_raw)
        logger.info(f"[{request_id}] Short JD (multi) — capped to {len(jd_skills_raw)} core skills.")

    # JD-based domain detection — resume_text added per-file inside the loop
    detected_domain, detected_secondary, jd_confidence = detect_multi_domain(
        jd_skills_raw,
        # resume_text omitted here — each file's text is processed in the loop below
    )
    logger.info(
        f"[{request_id}] Domain detected: primary='{detected_domain}' "
        f"secondary={detected_secondary!r} (confidence={jd_confidence})"
    )

    # ── DOMAIN RESOLUTION PIPELINE (3-step strict priority) ──────────────────
    final_domain, override_applied, domain_source = resolve_domain(
        user_role, detected_domain, jd_domain=detected_domain
    )
    final_secondary = detected_secondary if detected_secondary != final_domain else None
    # ─────────────────────────────────────────────────────────────────────────

    jd_skills = filter_domain_skills(jd_skills_raw, final_domain)
    logger.info(f"Domain filter '{final_domain}' (multi): {len(jd_skills)} skills retained.")

    jd_analysis = build_jd_analysis(jd_skills, final_domain, jd_confidence, final_secondary)
    # ─────────────────────────────────────────────────────────────────────────

    results = []
    errors  = []

    for upload_file in files:
        fname = upload_file.filename or "resume.pdf"
        try:
            # ── STEP 1: Read each PDF ONCE — stream is single-use ────────────
            file_bytes   = await upload_file.read()
            file_hash_f  = hashlib.sha256(file_bytes).hexdigest()[:16]
            logger.info(
                f"[{request_id}] Reading '{fname}'"
                f" size={len(file_bytes)}B hash={file_hash_f} | parse_count=1"
            )

            # ── STEP 2: Extract text ONCE per file ───────────────────────────
            resume_text_f = extract_text_from_pdf(file_bytes, fname)
            logger.info(
                f"[{request_id}] '{fname}' parsed: {len(resume_text_f)} chars"
            )

            if not resume_text_f.strip():
                raise RuntimeError(
                    f"No readable text in '{fname}'. "
                    "Ensure file is not scanned or image-based."
                )

            # ── STEP 3: Pass text string — never re-read file ─────────────────
            result = await _run_pipeline(
                resume_text=resume_text_f,
                filename=fname,
                jd=jd,
                jd_skills=jd_skills,
                domain=final_domain,
                secondary_domain=final_secondary,
                request_id=request_id,
            )
            results.append(result)
        except RuntimeError as exc:
            logger.error(f"[{request_id}] Failed to process '{fname}': {exc}")
            errors.append({"filename": fname, "error": str(exc)})
        except Exception as exc:
            logger.error(f"[{request_id}] Unexpected error for '{fname}': {exc}", exc_info=True)
            errors.append({"filename": fname, "error": "Unexpected processing error."})

    results.sort(key=lambda r: r["ats_score"], reverse=True)

    ranked = []
    for rank, r in enumerate(results, start=1):
        ranked.append({
            "rank":             rank,
            "filename":         r["filename"],
            "ats_score":        r["ats_score"],
            "match_level":      r["match_level"],
            "hire_decision":    _get_hire_decision(r["ats_score"]),
            "readiness_level":  r["readiness_level"],
            "matched_skills":   r["matched_skills"],
            "missing_skills":   r["missing_skills"],
            "inferred_skills":  r["inferred_skills"],
            "critical_missing": r["critical_missing"],
            "weak_areas":       r["weak_areas"],
            "domain_evidence":  r["domain_evidence"],
            "primary_confidence":        r["primary_confidence"],
            "secondary_confidence":      r["secondary_confidence"],
            "resume_detected_domain":    r["resume_detected_domain"],
            "jd_alignment":              r["jd_alignment"],
            "score_breakdown":  r["score_breakdown"],
            "score_simulation": r["score_simulation"],
            "section_scores":   r["section_scores"],
            "suggestions":      r["suggestions"],
            "ai_insight":       r["ai_insight"],
            "primary_domain":   r["primary_domain"],
            "domain":           r["domain"],
            "secondary_domain": r["secondary_domain"],
            "is_hybrid":        r["is_hybrid"],
        })

    comparison_insight: dict = {}
    if len(ranked) >= 2:
        try:
            comparison_insight = generate_comparison_insight(ranked[:3], jd)
            logger.info("Comparison insight generated successfully.")
        except Exception as exc:
            logger.error(f"Comparison insight failed: {exc}", exc_info=True)

    logger.info(
        f"=== Multi-analysis complete: {len(ranked)} processed, {len(errors)} failed ==="
    )

    role_confusion_detected_multi = _detect_role_confusion(
        user_role, final_domain, detected_domain
    )

    return {
        "status":             "success",
        "data":               ranked,
        "comparison_insight": comparison_insight,
        "errors":             errors,
        "error":              None,
        "jd_expanded":        jd_was_expanded,
        "interpreted_role":   interpreted_role,
        "jd_analysis":        jd_analysis,
        # ── Domain resolution provenance ──────────────────────────────────
        "selected_role":              user_role or None,
        "user_role":                  user_role or None,
        "jd_domain":                  detected_domain,
        "detected_domain":            detected_domain,
        "final_domain":               final_domain,
        "final_domain_used":          final_domain,
        "domain_source":              domain_source,
        "domain_locked":              True,
        "override_applied":           override_applied,
        "role_confusion_detected":    role_confusion_detected_multi,
        "secondary_domain":           final_secondary,
        "is_hybrid":                  final_secondary is not None,
        "request_id":                 request_id,    # ← NEW
    }


# ── Export PDF report ──────────────────────────────────────────────────────────
class ExportReportRequest(BaseModel):
    results:            List[Any]
    jd:                 str = ""
    comparison_insight: Optional[dict] = None


@router.post("/export-report")
async def export_report(body: ExportReportRequest):
    """Accept ranked results + optional comparison insight, return a downloadable PDF."""
    logger.info(f"=== Export report: {len(body.results)} candidates ===")
    try:
        pdf_bytes = _build_pdf(body.results, body.jd, body.comparison_insight or {})
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="resume_comparison_report.pdf"'
            },
        )
    except ImportError:
        return JSONResponse(
            content={"status": "error", "error": "reportlab is not installed. Run: pip install reportlab"},
            status_code=500,
        )
    except Exception as exc:
        logger.error(f"PDF export error: {exc}", exc_info=True)
        return JSONResponse(
            content={"status": "error", "error": "Failed to generate PDF report."},
            status_code=500,
        )


def _build_pdf(results: list, jd: str, comparison_insight: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    C_ACCENT  = colors.HexColor("#6366f1")
    C_GREEN   = colors.HexColor("#22c55e")
    C_ORANGE  = colors.HexColor("#f59e0b")
    C_RED     = colors.HexColor("#ef4444")
    C_DARK    = colors.HexColor("#1e293b")
    C_MUTED   = colors.HexColor("#64748b")
    C_BG_CARD = colors.HexColor("#f8fafc")
    C_BG_TOP  = colors.HexColor("#eef2ff")
    C_BORDER  = colors.HexColor("#e2e8f0")
    C_WHITE   = colors.white

    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    sTitle  = S("sTitle",  fontSize=22, fontName="Helvetica-Bold", textColor=C_ACCENT,  spaceAfter=4)
    sMeta   = S("sMeta",   fontSize=8,  textColor=C_MUTED,         spaceAfter=2)
    sH2     = S("sH2",     fontSize=12, fontName="Helvetica-Bold", textColor=C_DARK,    spaceBefore=10, spaceAfter=6)
    sH3     = S("sH3",     fontSize=10, fontName="Helvetica-Bold", textColor=C_DARK,    spaceBefore=6,  spaceAfter=4)
    sNormal = S("sNormal", fontSize=8,  textColor=C_DARK,          leading=12)
    sMuted  = S("sMuted",  fontSize=8,  textColor=C_MUTED,         leading=12)
    sInsLbl = S("sInsLbl", fontSize=9,  fontName="Helvetica-Bold", textColor=C_DARK,    spaceAfter=3)
    sInsTxt = S("sInsTxt", fontSize=8,  textColor=C_MUTED,         leading=13,          spaceAfter=4)

    def decision_color(score):
        if score > 75:  return C_GREEN
        if score >= 40: return C_ORANGE
        return C_RED

    def hire_label(score):
        if score > 75:  return "Strong Hire"
        if score >= 40: return "Consider"
        return "Reject"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
        title="Resume Comparison Report",
    )
    story = []

    story.append(Paragraph("🎯 Resume Comparison Report", sTitle))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}  ·  "
        f"{len(results)} resume(s) analysed", sMeta,
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=8))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("📊 Candidate Rankings", sH2))
    tbl_data = [["Rank", "Resume", "ATS Score", "Match Level", "Decision"]]
    for r in results:
        sc = r.get("ats_score", 0)
        tbl_data.append([f"#{r.get('rank','')}", r.get("filename",""), f"{sc}%", r.get("match_level",""), hire_label(sc)])

    col_w = [1.2*cm, 6.0*cm, 2.2*cm, 2.8*cm, 3.3*cm]
    tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
    tbl_style = [
        ("BACKGROUND",    (0,0), (-1,0),  C_ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0),  8),
        ("FONTSIZE",      (0,1), (-1,-1), 8),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), C_DARK),
        ("BACKGROUND",    (0,1), (-1,1),  C_BG_TOP),
        ("FONTNAME",      (0,1), (-1,1),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,2), (-1,-1), [C_WHITE, C_BG_CARD]),
        ("GRID",          (0,0), (-1,-1), 0.4, C_BORDER),
        ("ALIGN",         (0,0), (0,-1),  "CENTER"),
        ("ALIGN",         (2,0), (2,-1),  "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]
    for i, r in enumerate(results, start=1):
        tbl_style.append(("TEXTCOLOR", (4,i), (4,i), decision_color(r.get("ats_score",0))))
        tbl_style.append(("FONTNAME",  (4,i), (4,i), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(tbl_style))
    story.append(tbl)
    story.append(Spacer(1, 0.6*cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))
    story.append(Paragraph("📋 Candidate Details", sH2))
    for r in results:
        rank = r.get("rank", "")
        sc   = r.get("ats_score", 0)
        story.append(Paragraph(
            f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else f'#{rank}'}  "
            f"{r.get('filename','')}   |   ATS: {sc}%   |   {r.get('match_level','')}   |   {hire_label(sc)}",
            sH3,
        ))
        ss = r.get("section_scores", {})
        if ss:
            story.append(Paragraph(
                f"Semantic: {ss.get('semantic',0)}%  ·  Skills: {ss.get('skills',0)}%  ·  "
                f"Experience: {ss.get('experience',0)}%  ·  Education: {ss.get('education',0)}%", sMuted,
            ))
        matched = r.get("matched_skills", [])
        missing = r.get("missing_skills", [])
        if matched: story.append(Paragraph(f"✅ Matched Skills: {', '.join(matched[:12])}", sNormal))
        if missing: story.append(Paragraph(f"⚠️ Missing Skills: {', '.join(missing[:12])}", sNormal))
        for s in r.get("suggestions", [])[:4]:
            story.append(Paragraph(f"   • {s}", sMuted))
        story.append(Spacer(1, 0.35*cm))

    if comparison_insight:
        story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=8))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("🤖 AI Comparison Insights", sH2))
        for label, key in [
            ("🥇 Why Top Candidate Wins",  "best_candidate_reason"),
            ("⚖️ Strengths Comparison",     "strengths_comparison"),
            ("⚠️ Weaknesses Comparison",    "weaknesses_comparison"),
            ("🎯 Final Recommendation",     "final_recommendation"),
        ]:
            val = comparison_insight.get(key, "").strip()
            if val:
                story.append(Paragraph(label, sInsLbl))
                story.append(Paragraph(val, sInsTxt))
                story.append(Spacer(1, 0.2*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Paragraph(
        "Generated by ResumeAI Scanner · Powered by FastAPI + Sentence Transformers + Ollama",
        S("footer", fontSize=7, textColor=C_MUTED, alignment=TA_CENTER, spaceBefore=6),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()