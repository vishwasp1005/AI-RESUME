import json
from typing import Dict, Optional

import httpx

from config import MODEL_NAME, OLLAMA_URL, TIMEOUT
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Single-resume prompt — strict but context-aware, plain-text output ───────
_PROMPT_TEMPLATE = """
You are a strict but context-aware ATS evaluator for the "{domain}" role.
Be accurate, not aggressive. Identify real gaps — do not invent them.

EVALUATION RULES:
- Evaluate ONLY skills relevant to the "{domain}" domain
- DO NOT mark a skill missing if it is implied by another skill on the resume
  (e.g., if React is present, HTML/CSS/JS are implied — do not flag them)
- DO NOT hallucinate gaps that the resume or analysis data does not support
- DO NOT be unnecessarily harsh — inferred skills count as present
- DO identify genuine gaps in core fundamentals with clear business impact

=== JOB DESCRIPTION ===
{jd}

=== RESUME TEXT ===
{resume}

=== ATS ANALYSIS RESULTS ===
- ATS Score         : {ats_score}/100
- Match Level       : {match_level}
- Matched Skills    : {matched_skills}
- Implied Skills    : {inferred_skills}
- Critical Missing  : {missing_skills}
- Weak Areas        : {weak_areas}

IMPORTANT: "Implied Skills" above are already credited — do NOT list them as missing.

Respond with EXACTLY this plain-text format (no JSON, no markdown fences):

STRENGTHS:
- [genuine domain-relevant strength backed by resume evidence]
- [use phrases like "demonstrates", "shows evidence of", "proficient in"]
- [if fewer than 2 real strengths exist, state that honestly]

WEAKNESSES:
- [each critical missing fundamental with its real hiring impact]
- [use phrases like "not explicitly demonstrated", "depth can be improved",
   "advanced concepts not evidenced" — NOT "no knowledge" or "not qualified"
   unless truly warranted by the data]
- [do NOT list implied skills as weaknesses]

FINAL VERDICT:
[2-3 sentences. State readiness honestly. Use language like:
"Strong technical foundation, but...", "Partially aligned with...",
"Lacks evidence of..." where gaps are real.
End with: Needs improvement / Partially ready / Interview ready]
""".strip()


def generate_ai_insight(
    resume_text:    str,
    jd_text:        str,
    ats_score:      float,
    match_level:    str,
    domain:         str,
    matched_skills: list,
    missing_skills: list,
    weak_areas:     list   = (),
    inferred_skills: dict  = (),   # ← NEW: {skill: reason} dict or empty
) -> str:
    """
    Call Ollama to generate strict, context-aware ATS insight.
    Falls back gracefully if Ollama is unavailable.
    """
    inferred_list = (
        ", ".join(f"{k} (implied by resume)" for k in sorted(inferred_skills)[:8])
        if inferred_skills else "none"
    )

    prompt = _PROMPT_TEMPLATE.format(
        jd=jd_text[:1500],
        resume=resume_text[:1500],
        ats_score=round(ats_score, 1),
        match_level=match_level,
        domain=domain,
        matched_skills=", ".join(matched_skills[:20]) or "none",
        missing_skills=", ".join(missing_skills[:10]) or "none",
        weak_areas="; ".join(weak_areas[:4]) if weak_areas else "none identified",
        inferred_skills=inferred_list,
    )

    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        # No "format": "json" — prompt now requests plain-text structured output
    }

    logger.info(f"Calling Ollama at {OLLAMA_URL} with model={MODEL_NAME}")

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "")
            logger.info("Ollama responded successfully.")
            return _parse_plain_text_response(raw, ats_score, match_level, matched_skills, missing_skills)
    except httpx.ConnectError:
        logger.warning("Ollama not reachable – returning fallback insight.")
        return _fallback_insight(ats_score, match_level, matched_skills, missing_skills)
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out – returning fallback insight.")
        return _fallback_insight(ats_score, match_level, matched_skills, missing_skills)
    except Exception as exc:
        logger.error(f"LLM service error: {exc}")
        return _fallback_insight(ats_score, match_level, matched_skills, missing_skills)


def _parse_plain_text_response(
    raw: str,
    ats_score: float,
    match_level: str,
    matched_skills: list,
    missing_skills: list,
) -> str:
    """
    Parse the plain-text STRENGTHS / WEAKNESSES / FINAL VERDICT response
    from the domain-strict prompt.  Adds emoji section headers for the UI.
    Falls back gracefully if the response is empty or malformed.
    """
    if not raw.strip():
        return _fallback_insight(ats_score, match_level, matched_skills, missing_skills)

    # Normalise section headers and add emoji markers the UI expects
    formatted = raw.strip()
    formatted = formatted.replace("STRENGTHS:",    "🔍 STRENGTHS:")
    formatted = formatted.replace("WEAKNESSES:",   "🛑 WEAKNESSES:")
    formatted = formatted.replace("FINAL VERDICT:","📋 FINAL VERDICT:")

    # Sanity check — if none of our expected headers appear the LLM went off-script
    if "🔍 STRENGTHS:" not in formatted and "🛑 WEAKNESSES:" not in formatted:
        logger.warning("Plain-text response missing expected headers — returning raw.")
        return raw.strip()

    return formatted


def _parse_ollama_response(
    raw: str,
    ats_score: float,
    match_level: str,
    matched_skills: list,
    missing_skills: list,
) -> str:
    """Parse JSON Ollama response (kept for the comparison insight endpoint)."""
    try:
        data = json.loads(raw)
        strengths = "\n".join(f"  ✅ {s}" for s in data.get("strengths", []))
        weaknesses = "\n".join(f"  ⚠️ {w}" for w in data.get("weaknesses", []))
        verdict = data.get("final_verdict", "")
        return (
            f"🔍 STRENGTHS:\n{strengths}\n\n"
            f"🛑 WEAKNESSES:\n{weaknesses}\n\n"
            f"📋 FINAL VERDICT:\n  {verdict}"
        )
    except Exception:
        return raw if raw.strip() else _fallback_insight(
            ats_score, match_level, matched_skills, missing_skills
        )


def _fallback_insight(
    ats_score: float,
    match_level: str,
    matched_skills: list,
    missing_skills: list,
) -> str:
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)

    if ats_score >= 75:
        verdict = (
            "Technically aligned but profile depth needs verification in a live interview. "
            "Strong tool coverage noted; core CS fundamentals should be confirmed."
        )
    elif ats_score >= 55:
        verdict = (
            "Partially qualified. Lacks sufficient coverage of role-critical skills "
            "to pass automated screening without manual review. "
            "Address the skill gaps before applying to competitive positions."
        )
    elif ats_score >= 40:
        verdict = (
            "Insufficient alignment with role requirements. "
            "Multiple core skill gaps will trigger automatic rejection in strict ATS filters. "
            "Significant profile rebuild required."
        )
    else:
        verdict = (
            "Not ready for this role. Resume does not meet minimum ATS thresholds. "
            "Foundational gaps in core technical skills must be addressed before re-applying."
        )

    strengths = []
    if matched_count > 0:
        strengths.append(
            f"Matches {matched_count} required skill(s) — partial technical alignment confirmed."
        )
    if not strengths:
        strengths.append("No significant matched skills detected against role requirements.")

    weaknesses = []
    if missing_count > 0:
        top_missing = ", ".join(missing_skills[:5])
        weaknesses.append(
            f"Missing {missing_count} role-critical skill(s): {top_missing}. "
            f"These gaps will cause automatic rejection in strict ATS pipelines."
        )
    if ats_score < 65:
        weaknesses.append(
            "Overall semantic alignment is below industry threshold for shortlisting."
        )
    if not weaknesses:
        weaknesses.append(
            "No hard gaps detected — verify depth and recency of claimed skills in interview."
        )

    str_block = "\n".join(f"  ✅ {s}" for s in strengths)
    wk_block  = "\n".join(f"  ⚠️ {w}" for w in weaknesses)

    return (
        f"🔍 STRENGTHS:\n{str_block}\n\n"
        f"🛑 WEAKNESSES:\n{wk_block}\n\n"
        f"📋 FINAL VERDICT:\n  {verdict}\n\n"
        f"ℹ️  Note: AI-powered insight unavailable (Ollama not running). "
        f"Rule-based strict evaluation applied."
    )


# ── NEW: Multi-resume comparison insight ─────────────────────────────────────

_COMPARISON_TEMPLATE = """
You are a senior technical recruiter and HR expert conducting a comparative evaluation.

=== JOB DESCRIPTION ===
{jd}

=== CANDIDATE SUMMARIES (ranked by ATS score) ===
{candidates_summary}

Provide a structured hiring comparison with EXACTLY the following JSON format (no extra text before or after):
{{
  "best_candidate_reason": "Why the top-ranked candidate is the best fit in 2-3 sentences.",
  "strengths_comparison": "Compare the key strengths across the top candidates in 2-3 sentences.",
  "weaknesses_comparison": "Compare the key gaps or weaknesses across the candidates in 2-3 sentences.",
  "final_recommendation": "Your final hiring recommendation in 2-3 sentences."
}}
""".strip()


def generate_comparison_insight(top_resumes: list, jd_text: str) -> dict:
    """
    Call Ollama to generate a comparative hiring insight across top resumes.
    Returns a dict with four keys. Falls back gracefully if Ollama is unavailable.
    """
    candidates_summary = ""
    for i, r in enumerate(top_resumes, 1):
        candidates_summary += (
            f"\nCandidate #{i} — {r['filename']}\n"
            f"  ATS Score : {r['ats_score']}/100  |  Match Level: {r['match_level']}\n"
            f"  Matched Skills : {', '.join(r.get('matched_skills', [])[:15]) or 'none'}\n"
            f"  Missing Skills : {', '.join(r.get('missing_skills', [])[:15]) or 'none'}\n"
        )

    prompt = _COMPARISON_TEMPLATE.format(
        jd=jd_text[:2000],
        candidates_summary=candidates_summary.strip(),
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    logger.info(f"Calling Ollama for comparison insight across {len(top_resumes)} candidate(s).")

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "")
            logger.info("Ollama comparison insight succeeded.")
            return _parse_comparison_response(raw, top_resumes)
    except httpx.ConnectError:
        logger.warning("Ollama not reachable – returning fallback comparison insight.")
        return _fallback_comparison(top_resumes)
    except httpx.TimeoutException:
        logger.warning("Ollama timed out – returning fallback comparison insight.")
        return _fallback_comparison(top_resumes)
    except Exception as exc:
        logger.error(f"Comparison LLM error: {exc}")
        return _fallback_comparison(top_resumes)


def _parse_comparison_response(raw: str, top_resumes: list) -> dict:
    try:
        data = json.loads(raw)
        return {
            "best_candidate_reason": data.get("best_candidate_reason", ""),
            "strengths_comparison":  data.get("strengths_comparison", ""),
            "weaknesses_comparison": data.get("weaknesses_comparison", ""),
            "final_recommendation":  data.get("final_recommendation", ""),
        }
    except Exception:
        logger.warning("Comparison response JSON parse failed – using fallback.")
        return _fallback_comparison(top_resumes)


def _fallback_comparison(top_resumes: list) -> dict:
    top  = top_resumes[0] if top_resumes else {}
    name = top.get("filename", "The top candidate")
    score = top.get("ats_score", 0)
    matched = len(top.get("matched_skills", []))

    reason = (
        f"{name} ranks highest with an ATS score of {score}/100 and {matched} matched skill(s), "
        f"demonstrating the strongest overall alignment with the job description."
    )

    if len(top_resumes) > 1:
        scores_str = ", ".join(
            f"{r['filename']} ({r['ats_score']}%)" for r in top_resumes
        )
        strengths = (
            f"Ranking by overall fit: {scores_str}. "
            f"The top candidate leads on semantic similarity and skill coverage."
        )
        weaknesses = (
            "Lower-ranked candidates show notable gaps in required skills and "
            "lower semantic alignment with the job description."
        )
    else:
        strengths  = f"{name} demonstrates the strongest skill alignment among the reviewed candidates."
        weaknesses = "Insufficient candidates submitted for a full comparative weakness analysis."

    recommendation = (
        f"Proceed with {name} to the next interview stage. "
        f"Lower-ranked candidates may be considered as backups if the primary pick is unavailable.\n\n"
        f"ℹ️ Note: AI-powered insights unavailable (Ollama not running). This is a rule-based summary."
    )

    return {
        "best_candidate_reason": reason,
        "strengths_comparison":  strengths,
        "weaknesses_comparison": weaknesses,
        "final_recommendation":  recommendation,
    }