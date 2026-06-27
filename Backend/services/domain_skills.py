"""
services/domain_skills.py
──────────────────────────────────────────────────────────────────────────────
Hybrid ATS skill engine — 4 layers:

  1. Domain Knowledge  — DOMAINS dict with core / optional tiers per role
  2. Synonym Mapping   — SYNONYMS + normalize() collapse abbreviations
  3. Skill Classification — classify_skills() with exact + semantic matching
  4. Smart Suggestions — smart_suggestions() produces 3 distinct suggestion
                         types, always non-empty

Public API (backward-compatible):
  normalize(skill)            → str
  filter_domain_skills(...)   → List[str]
  split_skills(...)           → Tuple[List, List]
  classify_skills(...)        → ClassifiedSkills
  smart_suggestions(...)      → List[str]
  semantic_skill_match(s1,s2) → float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. UNIFIED DOMAIN KNOWLEDGE
# ══════════════════════════════════════════════════════════════════════════════
# Design rules
# ────────────
# • core     = 5-8 must-have skills; missing one measurably hurts the score
# • optional = 5-10 nice-to-have; contribute to score, never block candidate
# • Keep lists SHORT — quality over quantity
# • Keys must match strings from services.domain_detector.detect_domain()

DOMAINS: Dict[str, Dict[str, List[str]]] = {
    "software_engineer": {
        "core":     ["data structures", "algorithms", "system design", "backend", "api"],
        "optional": ["git", "testing", "database", "docker", "cloud"],
    },
    "frontend": {
        "core":     ["javascript", "react", "html", "css", "git"],
        "optional": ["typescript", "next.js", "redux", "testing", "accessibility"],
    },
    "backend": {
        "core":     ["api", "database", "server", "authentication", "python"],
        "optional": ["redis", "microservices", "message queue", "docker", "sql"],
    },
    "ai_ml": {
        "core":     ["machine learning", "python", "model training",
                     "deep learning", "pytorch", "large language models"],
        "optional": ["tensorflow", "nlp", "numpy", "pandas", "scikit-learn",
                     "semantic search", "transformers", "computer vision",
                     "hugging face"],
    },
    "data_analyst": {
        "core":     ["sql", "data analysis", "excel", "statistics", "python"],
        "optional": ["power bi", "tableau", "pandas", "numpy", "data visualisation"],
    },
    "devops_cloud": {
        # docker removed from core — it's a support tool, not a domain signal
        "core":     ["kubernetes", "ci/cd", "terraform", "monitoring", "ansible"],
        "optional": ["cloud", "linux", "aws", "helm", "prometheus"],
    },
    # Legacy keys — kept so existing detect_domain() output still resolves
    "web_dev": {
        "core":     ["javascript", "react", "html", "css", "git"],
        "optional": ["typescript", "node.js", "rest api", "docker", "testing"],
    },
    "data": {
        "core":     ["sql", "python", "data analysis", "statistics", "excel"],
        "optional": ["tableau", "power bi", "pandas", "numpy", "machine learning"],
    },
    "devops": {
        # docker/git moved to support tools — kubernetes/terraform are the real signals
        "core":     ["kubernetes", "ci/cd", "terraform", "ansible", "monitoring"],
        "optional": ["cloud", "linux", "aws", "helm", "prometheus"],
    },
    # ── STRICT TAXONOMY: distinct roles that must NEVER be auto-merged ────────
    # UI/UX is NOT frontend. AI Engineer is NOT Data Analyst.
    # These domains use role-specific skills with zero cross-contamination.
    "full_stack_engineer": {
        "core":     ["react", "api", "backend", "database", "authentication"],
        "optional": ["fastapi", "flask", "node.js", "typescript", "docker", "python"],
    },
    "ui_ux_designer": {
        "core":     ["figma", "wireframing", "prototyping",
                     "user research", "usability testing"],
        "optional": ["adobe xd", "design systems", "typography",
                     "interaction design", "user flows", "accessibility"],
    },
    "devops_engineer": {
        "core":     ["docker", "kubernetes", "ci/cd", "cloud", "linux"],
        "optional": ["kubernetes", "terraform", "monitoring", "ansible", "aws"],
    },
    "ai_engineer": {
        "core":     ["machine learning", "deep learning", "python",
                     "model training", "pytorch", "large language models"],
        "optional": ["tensorflow", "nlp", "computer vision",
                     "semantic search", "mlops", "data analysis"],
    },
    "product_manager": {
        "core":     ["product roadmap", "stakeholder management",
                     "user stories", "agile", "prioritization"],
        "optional": ["jira", "analytics", "a/b testing",
                     "wireframing", "sql"],
    },
    "mobile_developer": {
        "core":     ["react native", "swift", "kotlin",
                     "mobile ui", "api integration"],
        "optional": ["firebase", "push notifications", "app store",
                     "testing", "git"],
    },
}

# Flat allowlists (core ∪ optional) per domain — built once at import time
_DOMAIN_ALLOWLISTS: Dict[str, frozenset] = {
    domain: frozenset(s.lower() for tier in cfg.values() for s in tier)
    for domain, cfg in DOMAINS.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. SYNONYM NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

SYNONYMS: Dict[str, str] = {
    # ML / AI — extended with LLM and semantic search variants
    "ml":                        "machine learning",
    "dl":                        "deep learning",
    "tf":                        "tensorflow",
    "cv":                        "computer vision",
    "rl":                        "reinforcement learning",
    "llm":                       "large language models",
    "llms":                      "large language models",
    "large language model":      "large language models",
    "gpt":                       "large language models",
    "generative ai":             "large language models",
    "gen ai":                    "large language models",
    "bert":                      "transformers",
    "transformer model":         "transformers",
    "transformer":               "transformers",
    "semantic search":           "semantic search",
    "vector search":             "semantic search",
    "vector database":           "semantic search",
    "embedding":                 "semantic search",
    "embeddings":                "semantic search",
    "rag":                       "semantic search",
    "retrieval augmented generation": "semantic search",
    "fine tuning":               "model training",
    "fine-tuning":               "model training",
    "model fine tuning":         "model training",
    "finetuning":                "model training",
    "nlp":                       "nlp",
    "js":                "javascript",
    "ts":                "typescript",
    "reactjs":           "react",
    "react.js":          "react",
    "vuejs":             "vue",
    "vue.js":            "vue",
    "nodejs":            "node.js",
    "node":              "node.js",
    "nextjs":            "next.js",
    # Data / ML libs
    "sklearn":           "scikit-learn",
    "scikit learn":      "scikit-learn",
    "powerbi":           "power bi",
    # Cloud / Infra
    "k8s":               "kubernetes",
    "gke":               "kubernetes",
    "gh actions":        "github actions",
    "cicd":              "ci/cd",
    "iac":               "terraform",
    # Databases
    "postgres":          "postgresql",
    "mongo":             "mongodb",
    "mssql":             "sql server",
    # Languages
    "golang":            "go",
    "py":                "python",
    "c sharp":           "c#",
}


def normalize(skill: str) -> str:
    """
    Lowercase, strip whitespace, then apply the synonym map.
    Examples:
      'ReactJS' → 'react'
      'K8s'     → 'kubernetes'
      'ML'      → 'machine learning'
    """
    cleaned = skill.lower().strip()
    return SYNONYMS.get(cleaned, cleaned)


def normalize_list(skills: List[str]) -> List[str]:
    """Normalize every skill, deduplicate while preserving insertion order."""
    seen: set = set()
    result: List[str] = []
    for s in skills:
        n = normalize(s)
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# IMPLIED SKILL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
# Maps a skill the candidate HAS → the skills that are reasonably implied by it.
# Used to avoid marking skills as missing when context makes them obvious.
#
# Design rules:
# - Only strong, industry-accepted implications (not guesses)
# - Implications are one-directional: React implies JS, not the reverse
# - Each implied skill comes with a human-readable explanation for the UI

SKILL_IMPLICATIONS: Dict[str, List[tuple]] = {
    # Frontend frameworks → basics
    "react":       [("html",       "React components are built on HTML"),
                    ("css",        "Styling is inherent to React development"),
                    ("javascript", "React requires JavaScript knowledge")],
    "vue":         [("html",       "Vue templates use HTML syntax"),
                    ("css",        "Component styling is part of Vue workflow"),
                    ("javascript", "Vue is a JavaScript framework")],
    "angular":     [("html",       "Angular templates are HTML-based"),
                    ("css",        "Angular projects use CSS/SCSS"),
                    ("javascript", "Angular is built on TypeScript/JavaScript"),
                    ("typescript", "Angular is the primary TypeScript framework")],
    "next.js":     [("react",      "Next.js is built on top of React"),
                    ("javascript", "Next.js is a JavaScript framework"),
                    ("html",       "Next.js renders HTML pages")],

    # TypeScript → JavaScript
    "typescript":  [("javascript", "TypeScript is a superset of JavaScript")],

    # Backend frameworks → their language + API patterns
    "django":      [("python",  "Django is a Python web framework"),
                    ("sql",     "Django ORM wraps SQL databases"),
                    ("api",     "Django REST framework is commonly used")],
    "flask":       [("python",  "Flask is a Python microframework"),
                    ("api",     "Flask is predominantly used for REST APIs")],
    "fastapi":     [("python",  "FastAPI requires Python"),
                    ("api",     "FastAPI is an API-first framework"),
                    ("rest api","FastAPI generates REST endpoints by design")],
    "spring boot": [("java",    "Spring Boot is a Java framework"),
                    ("api",     "Spring Boot is used to build REST services")],
    "express":     [("node.js", "Express runs on Node.js"),
                    ("javascript","Node.js and Express use JavaScript"),
                    ("api",     "Express is typically used to build APIs")],

    # ML frameworks → their language + data libraries
    "pytorch":     [("python",          "PyTorch requires Python"),
                    ("machine learning","PyTorch is an ML framework"),
                    ("deep learning",   "PyTorch is the primary deep learning library")],
    "tensorflow":  [("python",          "TensorFlow requires Python"),
                    ("machine learning","TensorFlow is an ML framework"),
                    ("deep learning",   "TensorFlow is used for deep learning")],
    "scikit-learn":[("python",          "scikit-learn is a Python library"),
                    ("machine learning","scikit-learn implements ML algorithms"),
                    ("data analysis",   "scikit-learn usage implies data analysis")],

    # Data libraries → language
    "pandas":      [("python",        "Pandas is a Python library"),
                    ("data analysis", "Pandas is the primary data analysis tool")],
    "numpy":       [("python",        "NumPy is a Python library")],

    # Database tools → their query language
    "postgresql":  [("sql",      "PostgreSQL is an SQL database"),
                    ("database", "PostgreSQL is a relational database")],
    "mysql":       [("sql",      "MySQL is an SQL database"),
                    ("database", "MySQL is a relational database")],
    "mongodb":     [("database", "MongoDB is a NoSQL database")],
    "redis":       [("database", "Redis is an in-memory data store")],

    # ORM → database fundamentals
    "sqlalchemy":  [("sql",      "SQLAlchemy is an SQL ORM"),
                    ("database", "ORMs abstract database interactions"),
                    ("python",   "SQLAlchemy is a Python library")],

    # Cloud-specific tools → cloud platform
    "ec2":         [("aws",   "EC2 is an AWS compute service")],
    "s3":          [("aws",   "S3 is an AWS storage service")],
    "lambda":      [("aws",   "Lambda is an AWS serverless service")],

    # DevOps tooling → CI/CD
    "jenkins":        [("ci/cd", "Jenkins is a CI/CD tool")],
    "github actions": [("ci/cd", "GitHub Actions is a CI/CD platform"),
                       ("git",   "GitHub Actions is tied to GitHub/git")],
    "gitlab ci":      [("ci/cd", "GitLab CI is a CI/CD tool"),
                       ("git",   "GitLab CI is built into GitLab/git")],

    # Container orchestration → containers
    "kubernetes":  [("docker", "Kubernetes orchestrates Docker containers")],
    "helm":        [("kubernetes","Helm is a Kubernetes package manager"),
                    ("docker",   "Helm deploys containerised applications")],

    # Backend project work → API usage
    "microservices":   [("api", "Microservices communicate via APIs"),
                        ("docker","Microservices are commonly containerised")],
    "rest api":        [("api", "REST API implies API design knowledge")],
    "graphql":         [("api", "GraphQL is an alternative to REST APIs")],

    # Testing frameworks → testing knowledge
    "jest":    [("testing",     "Jest is a JavaScript testing framework"),
                ("javascript",  "Jest is a JS ecosystem tool")],
    "pytest":  [("testing",     "Pytest is the primary Python testing tool"),
                ("python",      "Pytest is a Python library")],
    "cypress": [("testing",     "Cypress is an end-to-end testing tool"),
                ("javascript",  "Cypress uses JavaScript")],
}

# ── Cross-role firewall: skills that must NEVER imply UX/design capabilities ─
# Prevents React → wireframing, CSS → design systems, JS → usability testing.
# infer_skills() consults this before adding any inference.
CROSS_ROLE_FIREWALL: Dict[str, set] = {
    # Frontend tech skills MUST NOT imply design/UX skills
    "react":       {"wireframing", "prototyping", "user research",
                    "usability testing", "design systems", "interaction design",
                    "figma", "adobe xd", "user flows", "typography"},
    "javascript":  {"wireframing", "prototyping", "user research",
                    "usability testing", "design systems", "interaction design",
                    "figma", "adobe xd"},
    "css":         {"design systems", "typography", "wireframing",
                    "prototyping", "user research", "figma"},
    "html":        {"wireframing", "prototyping", "user research",
                    "design systems", "figma", "adobe xd"},
    "typescript":  {"wireframing", "prototyping", "user research",
                    "design systems", "figma"},
    # DevOps tools MUST NOT imply ML/AI skills
    "docker":      {"machine learning", "deep learning", "nlp",
                    "model training", "data analysis"},
    "kubernetes":  {"machine learning", "deep learning", "nlp",
                    "model training"},
    # Backend MUST NOT imply UX
    "api":         {"wireframing", "prototyping", "user research",
                    "usability testing", "figma", "adobe xd"},
    "database":    {"wireframing", "prototyping", "user research",
                    "figma", "adobe xd"},
}


def infer_skills(
    resume_skills: List[str],
    jd_skills:     List[str],
) -> Dict[str, str]:
    """
    Return a dict of skills that are IMPLIED by the resume but not explicitly listed,
    filtered to only those that appear in jd_skills (so only relevant inferences).

    Returns
    ───────
    { normalized_skill: "explanation string" }
      deterministic — iteration in sorted(SKILL_IMPLICATIONS) order
    """
    resume_set = set(resume_skills)
    jd_set     = set(jd_skills)
    inferred:  Dict[str, str] = {}

    for source_skill in sorted(SKILL_IMPLICATIONS.keys()):
        if source_skill not in resume_set:
            continue   # candidate doesn't have the source skill — no inference
        blocked = CROSS_ROLE_FIREWALL.get(source_skill, set())
        for implied_skill, explanation in SKILL_IMPLICATIONS[source_skill]:
            implied_norm = normalize(implied_skill)
            if implied_norm in resume_set:
                continue   # already explicitly on resume — not an inference
            if implied_norm not in jd_set:
                continue   # JD doesn't care about this skill — skip
            if implied_norm in blocked:
                continue   # FIREWALL: cross-role inference explicitly forbidden
            if implied_norm not in inferred:
                inferred[implied_norm] = explanation

    logger.debug(
        f"infer_skills: {len(inferred)} implied skills from {len(resume_set)} resume skills"
    )
    return inferred


# ══════════════════════════════════════════════════════════════════════════════
# 3. SEMANTIC SKILL MATCHING
# ══════════════════════════════════════════════════════════════════════════════
# Reuses the SentenceTransformer already loaded by services.matcher.
# Zero extra memory cost — only one model instance across the process.

_SEMANTIC_THRESHOLD = 0.75
_embedding_model = None          # loaded lazily on first use


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from services.matcher import _MODEL   # already loaded at startup
            _embedding_model = _MODEL
            logger.debug("semantic_skill_match: reusing services.matcher._MODEL")
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("semantic_skill_match: loaded fresh SentenceTransformer")
            except ImportError:
                logger.warning("sentence-transformers unavailable — semantic skill match disabled")
    return _embedding_model


def semantic_skill_match(skill1: str, skill2: str) -> float:
    """
    Return cosine similarity (0–1) between two skill strings.
    Returns 0.0 if the model is unavailable (safe fallback).
    """
    model = _get_model()
    if model is None:
        return 0.0
    try:
        from sentence_transformers import util
        emb = model.encode([skill1, skill2], convert_to_tensor=True)
        return float(max(0.0, min(util.cos_sim(emb[0], emb[1]), 1.0)))
    except Exception as exc:
        logger.debug(f"semantic_skill_match error: {exc}")
        return 0.0


def _skills_match(jd_skill: str, resume_skill: str) -> bool:
    """
    True if jd_skill and resume_skill refer to the same technology.
    Checks (in order):  exact match → substring → semantic similarity.
    """
    if jd_skill == resume_skill:
        return True
    if jd_skill in resume_skill or resume_skill in jd_skill:
        return True
    return semantic_skill_match(jd_skill, resume_skill) >= _SEMANTIC_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# 4. SKILL CLASSIFICATION — 3 TIERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassifiedSkills:
    """
    Result of classify_skills().

    Three tiers, each split into matched / missing:
      CORE     — domain must-haves                    (weight × 3 in scoring)
      JD       — explicitly in JD but not domain list (weight × 2)
      OPTIONAL — domain nice-to-haves                (weight × 1)
    """
    matched_core:     List[str] = field(default_factory=list)
    matched_jd:       List[str] = field(default_factory=list)
    matched_optional: List[str] = field(default_factory=list)
    missing_core:     List[str] = field(default_factory=list)
    missing_jd:       List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)

    @property
    def all_matched(self) -> List[str]:
        return sorted(set(self.matched_core + self.matched_jd + self.matched_optional))

    @property
    def all_missing(self) -> List[str]:
        """Only surface core + JD missing — optional gaps are non-blocking."""
        return sorted(set(self.missing_core + self.missing_jd))

    def skill_score(self) -> float:
        """
        Weighted score 0–100:
          matched_core × 3  +  matched_jd × 2  +  matched_optional × 1
          ───────────────────────────────────────────────────────────────
          total_core × 3    +  total_jd × 2    +  total_optional × 1
        """
        total_weight = (
            (len(self.matched_core)     + len(self.missing_core))     * 3
            + (len(self.matched_jd)     + len(self.missing_jd))       * 2
            + (len(self.matched_optional) + len(self.missing_optional))
        )
        matched_weight = (
            len(self.matched_core) * 3
            + len(self.matched_jd) * 2
            + len(self.matched_optional)
        )
        return round((matched_weight / max(total_weight, 1)) * 100, 2)


def classify_skills(
    jd_skills:     List[str],
    resume_skills: List[str],
    domain:        str,
) -> ClassifiedSkills:
    """
    Classify JD skills into three tiers and match them against the resume.

    Algorithm
    ─────────
    For each skill in jd_skills (after normalisation):
      1. Check which domain tier it belongs to (core / optional / neither)
      2. Try to match against resume via exact + semantic search
      3. Place into the appropriate matched_* or missing_* bucket
      4. Mark as seen so it never appears in two buckets

    Parameters
    ──────────
    jd_skills     : skill list from the (expanded) JD
    resume_skills : skill list from the candidate's resume
    domain        : e.g. "ai_ml", "devops_cloud" — from detect_domain()
    """
    domain_cfg   = DOMAINS.get(domain, {})
    core_set     = {normalize(s) for s in domain_cfg.get("core",     [])}
    optional_set = {normalize(s) for s in domain_cfg.get("optional", [])}

    resume_norm  = normalize_list(resume_skills)
    jd_norm      = normalize_list(jd_skills)

    result    = ClassifiedSkills()
    seen_jd:  set = set()

    for skill in jd_norm:
        if skill in seen_jd:
            continue
        seen_jd.add(skill)

        # Match check — O(|resume|) with early-exit semantic fallback
        matched = any(_skills_match(skill, rs) for rs in resume_norm)

        if skill in core_set:
            (result.matched_core     if matched else result.missing_core).append(skill)
        elif skill in optional_set:
            (result.matched_optional if matched else result.missing_optional).append(skill)
        else:
            # JD-specific — recruiter explicitly listed it; medium priority
            (result.matched_jd       if matched else result.missing_jd).append(skill)

    logger.debug(
        f"classify_skills '{domain}': "
        f"core {len(result.matched_core)}/{len(result.matched_core)+len(result.missing_core)} "
        f"jd {len(result.matched_jd)}/{len(result.matched_jd)+len(result.missing_jd)} "
        f"opt {len(result.matched_optional)}/{len(result.matched_optional)+len(result.missing_optional)} "
        f"→ score={result.skill_score()}"
    )
    return result


# ── NEW: Hybrid multi-domain classification ───────────────────────────────────

@dataclass
class HybridClassifiedSkills(ClassifiedSkills):
    """
    Extends ClassifiedSkills with secondary-domain and inferred skill buckets.

    Scoring formula:
      primary_core × 3  +  primary_optional × 1  +  inferred × 2  +  secondary × 0.5
      ───────────────────────────────────────────────────────────────────────────────
      total_primary_core × 3  +  total_primary_opt × 1  +  total_inferred × 2
      + total_secondary × 0.5
    """
    secondary_matched: List[str]       = field(default_factory=list)
    secondary_missing: List[str]       = field(default_factory=list)
    inferred_skills:   Dict[str, str]  = field(default_factory=dict)  # ← NEW: {skill: reason}
    primary_domain:    str             = ""
    secondary_domain:  str             = ""

    @property
    def all_matched(self) -> List[str]:
        return sorted(set(
            self.matched_core + self.matched_jd
            + self.matched_optional + self.secondary_matched
        ))

    @property
    def all_inferred(self) -> List[str]:
        """Sorted list of inferred skill names."""
        return sorted(self.inferred_skills.keys())

    @property
    def all_missing(self) -> List[str]:
        """
        Core + JD gaps that are NEITHER explicitly matched NOR inferred.
        Inferred skills are removed from the missing list — never contradict context.
        """
        inferred_set = set(self.inferred_skills.keys())
        raw_missing  = set(self.missing_core + self.missing_jd)
        return sorted(raw_missing - inferred_set)

    def skill_score(self) -> float:
        """
        Hybrid weighted score 0–100.
        Inferred skills count at 2× (less than explicit core ×3, more than optional ×1).
        """
        n_inferred     = len(self.inferred_skills)
        total_weight   = (
            (len(self.matched_core)        + len(self.missing_core))      * 3.0
            + (len(self.matched_jd)        + len(self.missing_jd))        * 2.0
            + n_inferred                                                   * 2.0
            + (len(self.matched_optional)  + len(self.missing_optional))  * 1.0
            + (len(self.secondary_matched) + len(self.secondary_missing)) * 0.5
        )
        matched_weight = (
            len(self.matched_core)       * 3.0
            + len(self.matched_jd)       * 2.0
            + n_inferred                 * 2.0   # inferred count as matched
            + len(self.matched_optional) * 1.0
            + len(self.secondary_matched)* 0.5
        )
        return round((matched_weight / max(total_weight, 1)) * 100, 2)


def classify_hybrid_skills(
    jd_skills:        List[str],
    resume_skills:    List[str],
    primary_domain:   str,
    secondary_domain: Optional[str] = None,
) -> HybridClassifiedSkills:
    """
    3-tier classification across primary + optional secondary domain.

    Primary domain  → full weight (core ×3, optional ×1)
    Secondary domain → reduced weight (all secondary skills ×0.5)
    Overlap between domains de-duplicated — a skill only classified once.

    Parameters
    ──────────
    jd_skills        : extracted JD skills (normalized)
    resume_skills    : extracted resume skills (normalized)
    primary_domain   : highest-scoring domain (always used)
    secondary_domain : second domain (optional; None = single-domain mode)
    """
    # Step 1: classify against primary domain (reuse existing logic)
    primary_result = classify_skills(jd_skills, resume_skills, primary_domain)

    result = HybridClassifiedSkills(
        matched_core     = primary_result.matched_core,
        matched_jd       = primary_result.matched_jd,
        matched_optional = primary_result.matched_optional,
        missing_core     = primary_result.missing_core,
        missing_jd       = primary_result.missing_jd,
        missing_optional = primary_result.missing_optional,
        primary_domain   = primary_domain,
        secondary_domain = secondary_domain or "",
    )

    # ── Step 2: implied skill inference ──────────────────────────────────────
    # Inferred skills are removed from missing_* buckets — never contradict context.
    resume_norm = normalize_list(resume_skills)
    result.inferred_skills = infer_skills(resume_norm, jd_skills)

    inferred_set = set(result.inferred_skills.keys())
    result.missing_core     = sorted(s for s in result.missing_core     if s not in inferred_set)
    result.missing_jd       = sorted(s for s in result.missing_jd       if s not in inferred_set)
    result.missing_optional = sorted(s for s in result.missing_optional if s not in inferred_set)

    if not secondary_domain:
        logger.debug(
            f"classify_hybrid: primary='{primary_domain}' (single-domain) "
            f"inferred={list(inferred_set)} → score={result.skill_score()}"
        )
        return result

    # ── Step 3: classify secondary domain — only new skills ──────────────────
    already_seen = set(
        result.matched_core + result.matched_jd + result.matched_optional
        + result.missing_core + result.missing_jd + result.missing_optional
    ) | inferred_set   # inferred skills also excluded from secondary bucket

    sec_cfg  = DOMAINS.get(secondary_domain, {})
    sec_all  = [normalize(s) for s in
                sec_cfg.get("core", []) + sec_cfg.get("optional", [])]

    for skill in sec_all:
        if skill in already_seen:
            continue
        already_seen.add(skill)
        matched = any(_skills_match(skill, rs) for rs in resume_norm)
        (result.secondary_matched if matched else result.secondary_missing).append(skill)

    result.secondary_matched = sorted(result.secondary_matched)
    result.secondary_missing = sorted(result.secondary_missing)

    logger.debug(
        f"classify_hybrid: primary='{primary_domain}' secondary='{secondary_domain}' "
        f"inferred={sorted(inferred_set)} "
        f"sec_matched={len(result.secondary_matched)} "
        f"sec_missing={len(result.secondary_missing)} "
        f"→ score={result.skill_score()}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. SMART SUGGESTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def smart_suggestions(
    classified:       ClassifiedSkills,
    ats_score:        float,
    secondary_domain: Optional[str] = None,
) -> List[str]:
    """
    Generate up to 4 professional, action-verb suggestions.
    Guarantees at least 1 suggestion is always returned.

    Tier priority
    ─────────────
    1. Critical missing core → domain-specific imperative
    1b. All core matched     → highlight/depth suggestion (never empty)
    2. Hybrid secondary gaps → role-enhancing imperative
    3. JD-specific gaps      → ATS keyword alignment
    4. Optional gaps         → differentiation tip
    5. Score fallback        → score-band specific career advice
    """
    suggestions: List[str] = []

    # Resolve human-readable primary domain label
    primary_label = "this"
    if isinstance(classified, HybridClassifiedSkills) and classified.primary_domain:
        primary_label = classified.primary_domain.replace("_", " ").title()

    # ── Tier 1 — primary core gaps (highest urgency) ──────────────────────────
    if classified.missing_core:
        top = sorted(classified.missing_core)[:5]
        skills_str = ", ".join(top)
        suggestions.append(
            f"Develop hands-on experience with {skills_str} — "
            f"these are non-negotiable requirements for {primary_label} roles "
            f"and will have the highest impact on your ATS score."
        )
    else:
        # All core skills matched — push for depth, never leave this tier empty
        suggestions.append(
            f"All critical {primary_label} skills are present — "
            "strengthen your profile by quantifying impact: add metrics such as "
            "'reduced API latency by 35%', 'scaled service to 500K requests/day', "
            "or 'led migration that cut infrastructure cost by 40%'."
        )

    # ── Tier 2 — hybrid secondary gaps ───────────────────────────────────────
    if secondary_domain and isinstance(classified, HybridClassifiedSkills):
        sec_missing = sorted(classified.secondary_missing)
        if sec_missing:
            top        = sec_missing[:3]
            skills_str = ", ".join(top)
            sec_label  = secondary_domain.replace("_", " ").title()
            suggestions.append(
                f"Enhance your {sec_label} capabilities by adding "
                f"{skills_str} to your resume — demonstrating cross-domain "
                f"proficiency significantly boosts hybrid role competitiveness."
            )

    # ── Tier 3 — JD-specific gaps (medium urgency) ───────────────────────────
    if classified.missing_jd:
        top        = sorted(classified.missing_jd)[:4]
        skills_str = ", ".join(top)
        suggestions.append(
            f"Tailor your resume to explicitly mention {skills_str} — "
            f"recruiters and ATS systems actively screen for these exact terms "
            f"in {primary_label} job descriptions."
        )

    # ── Tier 4 — optional gaps (low urgency) ─────────────────────────────────
    if classified.missing_optional:
        top        = sorted(classified.missing_optional)[:3]
        skills_str = ", ".join(top)
        suggestions.append(
            f"Differentiate yourself by gaining practical experience with "
            f"{skills_str} — these are increasingly expected in senior "
            f"{primary_label} positions and will broaden your opportunity pipeline."
        )

    # ── Tier 5 — score-band fallback (when no skill gaps exist at all) ────────
    if len(suggestions) <= 1 and not (classified.missing_core or classified.missing_jd
                                       or classified.missing_optional):
        if ats_score >= 80:
            suggestions.append(
                "Outstanding technical alignment — focus on leadership signals: "
                "add examples of mentoring, architecture decisions, or cross-team "
                "initiatives to position yourself for senior-level screening."
            )
        elif ats_score >= 60:
            suggestions.append(
                "Solid foundation — boost semantic alignment by mirroring the "
                "exact terminology and acronyms from the job description in your "
                "professional summary and project descriptions."
            )
        else:
            suggestions.append(
                "Rebuild your resume's opening section with a strong technical "
                "summary that front-loads domain keywords, then follow with "
                "quantified achievements and domain-relevant project descriptions."
            )

    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════
# These maintain the same call signature as previous versions so nothing else
# in the codebase needs to change.

def filter_domain_skills(skills: List[str], domain: str) -> List[str]:
    """
    Keep only skills relevant for *domain* (core ∪ optional allowlist).
    Unrecognised domain → return unchanged (safe passthrough).
    """
    allowlist = _DOMAIN_ALLOWLISTS.get(domain)
    if allowlist is None:
        logger.debug(f"filter_domain_skills: no allowlist for '{domain}' — passthrough.")
        return skills

    norm = normalize_list(skills)
    filtered = [s for s in norm if s in allowlist]
    logger.debug(
        f"filter_domain_skills '{domain}': {len(skills)} → {len(filtered)} "
        f"({len(skills)-len(filtered)} removed)."
    )
    return filtered


def split_skills(skills: List[str], domain: str) -> Tuple[List[str], List[str]]:
    """
    Split *skills* into (core, optional) for *domain*.
    Returns ([], []) for unrecognised domains → caller uses equal-weight fallback.
    """
    domain_cfg = DOMAINS.get(domain)
    if domain_cfg is None:
        logger.debug(f"split_skills: no tier table for '{domain}' — returning empty.")
        return [], []

    core_set     = {normalize(s) for s in domain_cfg.get("core",     [])}
    optional_set = {normalize(s) for s in domain_cfg.get("optional", [])}
    norm         = normalize_list(skills)

    return (
        [s for s in norm if s in core_set],
        [s for s in norm if s in optional_set],
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION ADDITIONS — Deterministic JD analysis + explainable scoring
# ══════════════════════════════════════════════════════════════════════════════

# ── Master skill vocabulary (used for deterministic JD extraction) ────────────
# Sorted alphabetically — guarantees same extraction order every run.
KNOWN_SKILLS: List[str] = sorted([
    # Languages
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "ruby", "swift", "kotlin", "scala", "r", "bash", "shell scripting",
    # Web / Frontend
    "html", "css", "react", "vue", "angular", "next.js", "svelte",
    "redux", "webpack", "vite", "sass", "tailwind", "api integration",
    # UI/UX Design — kept strictly separate from frontend
    "figma", "adobe xd", "wireframing", "prototyping", "user research",
    "usability testing", "design systems", "typography", "interaction design",
    "user flows", "accessibility", "sketch",
    # Backend / APIs
    "node.js", "django", "flask", "fastapi", "express", "spring boot",
    "rest api", "graphql", "grpc", "websockets", "authentication",
    "microservices", "server", "backend", "api", "message queue",
    "rabbitmq", "kafka", "caching", "database",
    # Databases
    "sql", "postgresql", "mysql", "mongodb", "redis", "sqlite",
    "elasticsearch", "cassandra", "dynamodb",
    # ML / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "pytorch", "tensorflow", "keras", "scikit-learn", "numpy", "pandas",
    "hugging face", "xgboost", "lightgbm", "reinforcement learning",
    "data analysis", "statistics", "feature engineering", "model training",
    "large language models", "semantic search", "transformers",    # ← NEW
    # Data Engineering
    "spark", "hadoop", "airflow", "dbt", "etl",
    "data pipeline", "data warehousing",
    # Visualisation / BI
    "tableau", "power bi", "matplotlib", "seaborn", "plotly",
    "excel", "google sheets",
    # Cloud
    "aws", "azure", "gcp", "cloud", "lambda", "ec2", "s3",
    # DevOps / Infra
    "docker", "kubernetes", "terraform", "ansible", "ci/cd",
    "jenkins", "github actions", "gitlab ci", "linux",
    "monitoring", "prometheus", "grafana", "helm",
    # CS Fundamentals
    "data structures", "algorithms", "system design",
    "object-oriented programming", "design patterns",
    # Mobile
    "react native", "mobile ui", "firebase",
    "push notifications", "app store",
    # Product
    "product roadmap", "stakeholder management", "user stories",
    "agile", "prioritization", "jira",
    # Process / Methodology
    "git", "testing", "scrum", "code review", "debugging",
    # Security
    "security", "authentication", "oauth", "jwt",
])

# Build a set for O(1) lookup
_KNOWN_SKILLS_SET: frozenset = frozenset(KNOWN_SKILLS)


def extract_jd_skills(jd_text: str) -> List[str]:
    """
    Deterministic JD skill extractor — no AI, no randomness.

    Algorithm
    ─────────
    1. Lowercase the full JD text
    2. Match against KNOWN_SKILLS (multi-word skills checked first to avoid
       partial matches — e.g. "machine learning" before "machine")
    3. Normalize matches through the SYNONYMS map
    4. Deduplicate and sort → same input always produces same sorted list

    Returns
    ───────
    Sorted list of normalized skill strings.
    """
    text = jd_text.lower()
    found: set = set()

    # Process multi-word skills before single-word to avoid partial shadowing
    multi  = sorted([s for s in KNOWN_SKILLS if " " in s], key=len, reverse=True)
    single = [s for s in KNOWN_SKILLS if " " not in s]

    for skill in multi + single:
        # Whole-word boundary check: skill must not be part of a larger word
        import re as _re
        pattern = r"(?<![a-z0-9\-])" + _re.escape(skill) + r"(?![a-z0-9\-])"
        if _re.search(pattern, text):
            found.add(normalize(skill))

    result = sorted(found)   # always sorted → deterministic
    logger.debug(f"extract_jd_skills: found {len(result)} skills from JD.")
    return result


def detect_domain_from_skills(
    jd_skills: List[str],
    fallback_domain: str = "software_engineer",
) -> tuple:
    """Thin wrapper — delegates to detect_multi_domain for backward compatibility."""
    primary, _secondary, confidence = detect_multi_domain(jd_skills, fallback_domain)
    return primary, confidence


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED DOMAIN DETECTION — fixes support-tool bias
# ══════════════════════════════════════════════════════════════════════════════
#
# ROOT CAUSE OF WRONG CLASSIFICATIONS:
#   Old system:  domain_score = core_hits × 2 + optional_hits × 1
#   Problem:     Docker (in every resume) scored the same as PyTorch
#   Result:      AI/ML resume → classified as DevOps because Docker appeared
#
# FIX: per-skill weights + support-tool suppression
#   - Domain-defining skills  → high weight (8-15)
#   - Domain-common skills    → medium weight (3-7)
#   - Universal support tools → suppressed weight (1) and excluded from
#                               primary domain evidence list

# ── Support tools: cross-domain utilities that must NEVER drive classification ─
# A skill on this list can only contribute a tiny weight boost.
# It can NEVER be the primary evidence for a domain.
SUPPORT_TOOLS: frozenset = frozenset({
    "docker", "git", "github", "gitlab", "linux", "bash",
    "shell scripting", "cloud", "aws", "azure", "gcp",
    "s3", "ec2", "lambda", "testing", "agile", "scrum",
    "jira", "slack", "vs code", "vim",
})

# ── Per-domain weighted skill tables ─────────────────────────────────────────
# weight = how strongly this skill signals membership in that domain
# Support tools intentionally absent — they get SUPPORT_TOOL_WEIGHT automatically
DOMAIN_SKILL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "ai_ml": {
        # Tier A — unmistakable AI/ML signals
        "large language models":   15.0,
        "llm":                     15.0,
        "semantic search":         15.0,
        "transformer":             13.0,
        "transformers":            13.0,
        "nlp":                     12.0,
        "natural language processing": 12.0,
        "deep learning":           12.0,
        "reinforcement learning":  12.0,
        # Tier B — strong frameworks
        "pytorch":                 11.0,
        "tensorflow":              11.0,
        "hugging face":            11.0,
        "model training":          10.0,
        "machine learning":        10.0,
        "computer vision":          9.0,
        # Tier C — common ML tools
        "scikit-learn":             8.0,
        "keras":                    8.0,
        "xgboost":                  7.0,
        "lightgbm":                 7.0,
        "feature engineering":      7.0,
        "data analysis":            5.0,
        "numpy":                    4.0,
        "pandas":                   4.0,
        "statistics":               4.0,
        "python":                   3.0,  # common — low weight
    },
    "ai_engineer": {
        "large language models":   15.0,
        "llm":                     15.0,
        "semantic search":         14.0,
        "nlp":                     12.0,
        "model training":          11.0,
        "deep learning":           11.0,
        "pytorch":                 10.0,
        "tensorflow":              10.0,
        "machine learning":         9.0,
        "python":                   3.0,
    },
    "devops_engineer": {
        # Tier A — dedicated infra skills (NOT shared tools)
        "kubernetes":              15.0,
        "terraform":               15.0,
        "infrastructure automation": 14.0,
        "jenkins":                 12.0,
        "ci/cd":                   12.0,
        "ansible":                 12.0,
        "helm":                    11.0,
        "prometheus":              10.0,
        "grafana":                 10.0,
        "site reliability":        13.0,
        "sre":                     13.0,
        # Tier B — infra-adjacent
        "monitoring":               8.0,
        "cloud infrastructure":     8.0,
        # Support tools at suppressed weight (handled below)
    },
    "devops": {
        "kubernetes":              15.0,
        "terraform":               14.0,
        "ci/cd":                   12.0,
        "ansible":                 12.0,
        "jenkins":                 11.0,
        "helm":                    10.0,
        "prometheus":              10.0,
        "monitoring":               7.0,
    },
    "devops_cloud": {
        "kubernetes":              15.0,
        "terraform":               14.0,
        "ci/cd":                   12.0,
        "ansible":                 11.0,
        "monitoring":               8.0,
    },
    "frontend_developer": {
        "react":                   13.0,
        "vue":                     13.0,
        "angular":                 13.0,
        "next.js":                 11.0,
        "javascript":              10.0,
        "typescript":              10.0,
        "css":                      8.0,
        "html":                     8.0,
        "redux":                    7.0,
        "webpack":                  6.0,
        "tailwind":                 6.0,
        "svelte":                   9.0,
    },
    "frontend": {
        "react":                   13.0,
        "javascript":              10.0,
        "typescript":              10.0,
        "css":                      8.0,
        "html":                     8.0,
        "next.js":                 11.0,
        "vue":                     13.0,
    },
    "backend": {
        "api":                     10.0,
        "rest api":                11.0,
        "graphql":                 10.0,
        "database":                 9.0,
        "authentication":          10.0,
        "microservices":           10.0,
        "server":                   8.0,
        "django":                  11.0,
        "flask":                   10.0,
        "fastapi":                 10.0,
        "spring boot":             11.0,
        "node.js":                 10.0,
        "redis":                    6.0,
        "postgresql":               6.0,
        "sql":                      6.0,
        "python":                   3.0,
        "java":                     5.0,
    },
    "software_engineer": {
        "data structures":         12.0,
        "algorithms":              12.0,
        "system design":           13.0,
        "backend":                  9.0,
        "api":                      9.0,
        "database":                 8.0,
        "object-oriented programming": 8.0,
        "design patterns":         10.0,
    },
    "data_analyst": {
        "sql":                     12.0,
        "data analysis":           13.0,
        "excel":                   10.0,
        "tableau":                 12.0,
        "power bi":                12.0,
        "statistics":              10.0,
        "python":                   5.0,
        "data visualisation":      11.0,
        "pandas":                   7.0,
        "numpy":                    5.0,
    },
    "data": {
        "sql":                     11.0,
        "data analysis":           12.0,
        "statistics":              10.0,
        "spark":                   11.0,
        "hadoop":                  10.0,
        "etl":                     11.0,
        "airflow":                 10.0,
        "data pipeline":           12.0,
        "dbt":                      9.0,
        "python":                   4.0,
    },
    "ui_ux_designer": {
        "figma":                   15.0,
        "wireframing":             14.0,
        "prototyping":             14.0,
        "user research":           14.0,
        "usability testing":       13.0,
        "interaction design":      13.0,
        "design systems":          12.0,
        "adobe xd":                12.0,
        "user flows":              11.0,
        "typography":              10.0,
        "accessibility":            9.0,
    },
    "mobile_developer": {
        "react native":            14.0,
        "swift":                   13.0,
        "kotlin":                  13.0,
        "mobile ui":               12.0,
        "app store":               11.0,
        "firebase":                 9.0,
        "push notifications":      10.0,
    },
    "product_manager": {
        "product roadmap":         14.0,
        "stakeholder management":  13.0,
        "user stories":            12.0,
        "agile":                   10.0,
        "prioritization":          12.0,
        "jira":                     8.0,
        "a/b testing":             10.0,
    },
    "web_dev": {
        "javascript":              10.0,
        "react":                   12.0,
        "html":                     8.0,
        "css":                      8.0,
        "node.js":                 10.0,
        "rest api":                10.0,
    },
    "full_stack_engineer": {
        "react":           12.0,
        "vue":             12.0,
        "angular":         11.0,
        "next.js":         10.0,
        "fastapi":         11.0,
        "flask":           10.0,
        "django":          11.0,
        "node.js":         10.0,
        "javascript":       8.0,
        "typescript":       8.0,
        "api":              9.0,
        "rest api":        10.0,
        "authentication":  10.0,
        "database":         9.0,
        "python":           4.0,
        "html":             5.0,
        "css":              5.0,
    },
}

# Weight given to support tools when they appear in a domain's optional list
_SUPPORT_TOOL_WEIGHT = 1.0
# Minimum weight given to any matched domain skill not in the weights table
_DEFAULT_SKILL_WEIGHT = 2.0
# Secondary domain threshold
_SECONDARY_THRESHOLD  = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# STACK COMBINATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# When BOTH sides of an engineering stack are present, apply a bonus to the
# correct domain.  This prevents React+FastAPI from being split across
# frontend(13) + backend(10) and losing to data_analyst(42).
#
# Each tuple: (required_skill_set, target_domain, bonus_score, label)
# ALL skills in required_set must be present for the bonus to fire.
# Processed in definition order — first matching combo sets the label.

_STACK_COMBINATIONS: List[tuple] = [
    # ── Full-stack detection (frontend framework + backend framework) ─────────
    ({"react",    "fastapi"},         "full_stack_engineer", 30.0, "React + FastAPI"),
    ({"react",    "flask"},           "full_stack_engineer", 28.0, "React + Flask"),
    ({"react",    "django"},          "full_stack_engineer", 28.0, "React + Django"),
    ({"react",    "node.js"},         "full_stack_engineer", 26.0, "React + Node.js"),
    ({"react",    "spring boot"},     "full_stack_engineer", 26.0, "React + Spring Boot"),
    ({"vue",      "fastapi"},         "full_stack_engineer", 28.0, "Vue + FastAPI"),
    ({"vue",      "flask"},           "full_stack_engineer", 26.0, "Vue + Flask"),
    ({"vue",      "node.js"},         "full_stack_engineer", 26.0, "Vue + Node.js"),
    ({"angular",  "node.js"},         "full_stack_engineer", 26.0, "Angular + Node.js"),
    ({"angular",  "spring boot"},     "full_stack_engineer", 28.0, "Angular + Spring Boot"),
    ({"angular",  "django"},          "full_stack_engineer", 26.0, "Angular + Django"),
    ({"next.js",  "fastapi"},         "full_stack_engineer", 28.0, "Next.js + FastAPI"),
    ({"next.js",  "node.js"},         "full_stack_engineer", 26.0, "Next.js + Node.js"),
    # ── AI/ML + application framework (AI engineer, not pure data analyst) ───
    ({"large language models", "fastapi"},  "ai_engineer",   28.0, "LLM + FastAPI"),
    ({"large language models", "flask"},    "ai_engineer",   26.0, "LLM + Flask"),
    ({"large language models", "react"},    "ai_engineer",   22.0, "LLM + React"),
    ({"large language models", "django"},   "ai_engineer",   24.0, "LLM + Django"),
    ({"semantic search",       "fastapi"},  "ai_engineer",   24.0, "SemanticSearch + FastAPI"),
    ({"nlp",                   "fastapi"},  "ai_engineer",   20.0, "NLP + FastAPI"),
    ({"nlp",                   "flask"},    "ai_engineer",   18.0, "NLP + Flask"),
    ({"pytorch",               "fastapi"},  "ai_engineer",   18.0, "PyTorch + FastAPI"),
    ({"tensorflow",            "flask"},    "ai_engineer",   18.0, "TF + Flask"),
    # ── DevOps stacks ─────────────────────────────────────────────────────────
    ({"kubernetes", "terraform"},     "devops_engineer",     24.0, "K8s + Terraform"),
    ({"kubernetes", "ci/cd"},         "devops_engineer",     22.0, "K8s + CI/CD"),
    ({"terraform",  "ansible"},       "devops_engineer",     22.0, "Terraform + Ansible"),
    ({"kubernetes", "ansible"},       "devops_engineer",     20.0, "K8s + Ansible"),
    # ── Backend-only stacks ───────────────────────────────────────────────────
    ({"fastapi", "authentication"},   "backend",             12.0, "FastAPI + Auth"),
    ({"flask",   "authentication"},   "backend",             10.0, "Flask + Auth"),
    ({"django",  "authentication"},   "backend",             10.0, "Django + Auth"),
    ({"fastapi", "database"},         "backend",             10.0, "FastAPI + DB"),
]

# ── Data analyst isolation guard ──────────────────────────────────────────────
# These skills are BI-exclusive — only meaningful as data analyst signals
# when they appear WITHOUT a strong engineering framework stack.
_BI_EXCLUSIVE_SKILLS: frozenset = frozenset({
    "tableau", "power bi", "excel", "data visualisation",
    "google sheets", "looker", "business intelligence", "reporting",
})

# data_analyst can only be primary domain when its BI-exclusive score
# exceeds this threshold (ensures pandas+sql alone can't win)
_DATA_ANALYST_BI_MIN_SCORE: float = 12.0

# Engineering domain keys — when any of these scores strongly, data_analyst
# is demoted regardless of its raw score from pandas/sql/statistics
_ENGINEERING_DOMAINS: frozenset = frozenset({
    "backend", "frontend", "frontend_developer", "software_engineer",
    "full_stack_engineer", "ai_ml", "ai_engineer",
    "devops_engineer", "devops", "devops_cloud", "web_dev",
})


def _score_domain_weighted(
    skill_set: frozenset,
    domain:    str,
) -> tuple:   # (score: float, evidence: List[str])
    """
    Compute a weighted domain score for the given normalised skill set.

    Returns (total_score, evidence_list)
    Evidence = skills that contributed a meaningful weight (> _SUPPORT_TOOL_WEIGHT).
    """
    weight_table = DOMAIN_SKILL_WEIGHTS.get(domain, {})
    cfg          = DOMAINS.get(domain, {})
    domain_skills = frozenset(
        normalize(s)
        for tier in cfg.values()
        for s in tier
    )

    total    = 0.0
    evidence = []

    # Score every skill that appears in both the input set and this domain's vocab
    for skill in sorted(skill_set):   # sorted → deterministic
        if skill not in domain_skills:
            continue
        # Look up weight: weights table → support-tool suppression → default
        if skill in SUPPORT_TOOLS:
            w = _SUPPORT_TOOL_WEIGHT
        else:
            w = weight_table.get(skill, _DEFAULT_SKILL_WEIGHT)

        total += w
        if w > _SUPPORT_TOOL_WEIGHT:   # only meaningful signals count as evidence
            evidence.append(skill)

    return round(total, 2), sorted(evidence)


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT CONTEXT EXTRACTION — reads raw resume text, not just skill tokens
# ══════════════════════════════════════════════════════════════════════════════
# This is the fix for: "AI Resume Scanner using LLMs" → ai_engineer
# Project descriptions contain domain signals that skill extraction misses.

import re as _re

# Phrases that introduce project descriptions in resumes
_PROJECT_INTRO_RE = _re.compile(
    r'(?:project|built|developed|created|designed|implemented|deployed'
    r'|worked on|contributed to|engineered)\b[:\-–]?\s*(.{10,120})',
    _re.IGNORECASE,
)

# Domain-specific free-text signals found in project descriptions
# Each key is a domain; values are phrase fragments scanned with `in text_lower`
_PROJECT_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "ai_engineer": [
        "llm", "large language model", "gpt", "bert", "nlp",
        "semantic search", "vector search", "rag",
        "retrieval augmented", "chatbot", "ai resume",
        "ai scanner", "recommendation system", "classification model",
        "deep learning model", "neural network", "transformer model",
        "hugging face", "fine-tun", "generative ai", "sentiment",
        "text classification", "named entity", "question answering",
    ],
    "ai_ml": [
        "machine learning", "model training", "prediction model",
        "regression", "clustering", "reinforcement", "tensorflow",
        "pytorch", "scikit", "xgboost", "feature engineering",
        "data science project", "kaggle",
    ],
    "full_stack_engineer": [
        "full stack", "web application", "web app", "react app",
        "frontend backend", "rest api", "crud", "authentication system",
        "login system", "dashboard", "portal", "e-commerce", "booking system",
        "fastapi", "flask api", "django app", "node app",
        "user interface", "responsive", "api integration",
    ],
    "backend": [
        "backend service", "api service", "rest service", "microservice",
        "server-side", "database design", "api development",
        "authentication", "authorization", "orm", "query optimisation",
    ],
    "frontend_developer": [
        "react", "vue", "angular", "ui component", "user interface design",
        "frontend app", "single page", "spa", "responsive design",
        "css animation", "tailwind", "bootstrap",
    ],
    "devops_engineer": [
        "kubernetes", "terraform", "ci/cd", "infrastructure", "helm",
        "ansible", "pipeline automation", "container orchestration",
        "cloud infrastructure", "deployment pipeline",
    ],
    "data_analyst": [
        "dashboard", "business intelligence", "power bi", "tableau",
        "data visualization", "reporting", "kpi", "metrics dashboard",
        "excel model", "data analysis project", "analytics platform",
    ],
    "data": [
        "data pipeline", "etl", "data warehouse", "spark job",
        "airflow dag", "data ingestion", "batch processing",
        "stream processing", "dbt model", "data mart",
    ],
}

# Weight applied per matching project keyword hit
_PROJECT_KEYWORD_WEIGHT = 8.0
# Cap per domain from project signals (prevents one huge project dominating)
_PROJECT_SIGNAL_CAP     = 40.0


def extract_project_signals(resume_text: str) -> Dict[str, float]:
    """
    Scan raw resume text for project descriptions and return domain boost scores.

    Algorithm
    ─────────
    1. Extract lines that look like project descriptions (via _PROJECT_INTRO_RE
       plus a sliding window of lines under "Projects" / "Experience" headings)
    2. For each line match _PROJECT_DOMAIN_KEYWORDS — each hit adds
       _PROJECT_KEYWORD_WEIGHT to that domain's boost
    3. Cap per domain at _PROJECT_SIGNAL_CAP

    Returns
    ───────
    Dict[domain_key, boost_score]  — domains with zero signal are absent.
    Deterministic: same text → same output.
    """
    if not resume_text:
        return {}

    text_lower = resume_text.lower()
    boosts: Dict[str, float] = {}

    # ── Pass 1: scan regex-matched project lines ──────────────────────────────
    for match in _PROJECT_INTRO_RE.finditer(text_lower):
        line = match.group(0)
        for domain, keywords in sorted(_PROJECT_DOMAIN_KEYWORDS.items()):
            for kw in keywords:
                if kw in line:
                    boosts[domain] = boosts.get(domain, 0.0) + _PROJECT_KEYWORD_WEIGHT

    # ── Pass 2: scan ALL lines for project-context keywords ───────────────────
    # Covers bullet points that don't start with an intro verb
    for line in text_lower.split("\n"):
        line = line.strip()
        if len(line) < 15:
            continue   # skip very short lines (dates, labels, etc.)
        for domain, keywords in sorted(_PROJECT_DOMAIN_KEYWORDS.items()):
            for kw in keywords:
                if kw in line:
                    boosts[domain] = boosts.get(domain, 0.0) + (_PROJECT_KEYWORD_WEIGHT * 0.5)

    # ── Cap each domain and round ─────────────────────────────────────────────
    capped = {
        d: round(min(s, _PROJECT_SIGNAL_CAP), 2)
        for d, s in boosts.items()
        if s > 0
    }

    if capped:
        logger.debug(
            f"[project_signals] {len(capped)} domain(s) signalled from resume text: "
            + ", ".join(f"{d}={s}" for d, s in sorted(capped.items(), key=lambda x: -x[1]))
        )
    return capped


def build_domain_evidence(
    primary_domain:  str,
    skill_evidence:  List[str],
    combo_hits:      List[str],
    project_signals: Dict[str, float],
    resume_skills:   List[str],
) -> dict:
    """
    Build a structured domain evidence object for the UI/response.

    Returns
    ───────
    {
      "primary_domain": str,
      "skill_signals":  [...],   # skills that scored for this domain
      "stack_combos":   [...],   # engineering stacks detected
      "project_signals":[...],   # project-level domain signals
      "support_tools":  [...],   # tools present but excluded from scoring
      "summary":        str,     # human-readable explanation
    }
    """
    support_on_resume = sorted(
        s for s in resume_skills
        if normalize(s) in SUPPORT_TOOLS
    )
    project_hits = sorted(
        kw for kw in _PROJECT_DOMAIN_KEYWORDS.get(primary_domain, [])
        if kw in " ".join(resume_skills).lower()
    )

    summary_parts = []
    if skill_evidence:
        summary_parts.append(f"Core skills: {', '.join(skill_evidence[:5])}")
    if combo_hits:
        summary_parts.append(f"Stack combos: {', '.join(combo_hits[:3])}")
    if project_signals.get(primary_domain):
        summary_parts.append(
            f"Project signals: {project_signals[primary_domain]:.0f}pt boost"
        )

    return {
        "primary_domain":  primary_domain,
        "skill_signals":   skill_evidence[:8],
        "stack_combos":    combo_hits[:4],
        "project_signals": project_hits[:6],
        "support_tools":   support_on_resume[:6],
        "summary":         " | ".join(summary_parts) or "No strong domain signals detected.",
    }


def detect_multi_domain(
    jd_skills:       List[str],
    fallback_domain: str = "software_engineer",
    resume_text:     str = "",    # ← NEW: enables project-context scoring
) -> tuple:   # (primary: str, secondary: str | None, confidence: str)
    """
    Project-context-aware, stack-combination-boosted domain detection.

    Algorithm
    ─────────
    Step 1 — Base weighted scoring per domain (skill tokens)
    Step 2a — Stack combination bonuses (React+FastAPI → full_stack_engineer)
    Step 2b — Project context boost from raw resume text (NEW)
              "AI Resume Scanner using LLMs" → ai_engineer +40
    Step 3 — Data analyst isolation guard
    Step 4 — Rank and select primary / secondary

    Deterministic: sorted() everywhere → same input = same output every run.
    """
    skill_set = frozenset(normalize(s) for s in jd_skills)

    # ── Step 1: base weighted scores per domain ───────────────────────────────
    domain_scores: Dict[str, tuple] = {}
    for domain in sorted(DOMAINS.keys()):
        score, evidence = _score_domain_weighted(skill_set, domain)
        domain_scores[domain] = (score, evidence)

    if "full_stack_engineer" not in domain_scores:
        fs_score, fs_evidence = _score_domain_weighted(skill_set, "full_stack_engineer")
        domain_scores["full_stack_engineer"] = (fs_score, fs_evidence)

    # ── Step 2a: stack combination bonuses ────────────────────────────────────
    combo_hits: Dict[str, List[str]] = {}

    for required, target_domain, bonus, label in _STACK_COMBINATIONS:
        if required.issubset(skill_set):
            prev_score, prev_evidence = domain_scores.get(target_domain, (0.0, []))
            new_score    = prev_score + bonus
            new_evidence = sorted(set(prev_evidence) | required)
            domain_scores[target_domain] = (round(new_score, 2), new_evidence)
            combo_hits.setdefault(target_domain, []).append(label)
            logger.debug(f"[stack_combo] +{bonus} to '{target_domain}' via '{label}'")

    if combo_hits:
        logger.info(f"[stack_combos] matched: {combo_hits}")

    # ── Step 2b: project context boost from raw resume text ───────────────────
    project_signals: Dict[str, float] = {}
    if resume_text:
        project_signals = extract_project_signals(resume_text)
        for domain, boost in project_signals.items():
            prev_score, prev_evidence = domain_scores.get(domain, (0.0, []))
            domain_scores[domain] = (round(prev_score + boost, 2), prev_evidence)
            logger.info(
                f"[project_boost] '{domain}' +{boost:.1f}pts from project context"
            )

    # ── Step 3: data analyst isolation guard ─────────────────────────────────
    da_score, da_evidence = domain_scores.get("data_analyst", (0.0, []))
    if da_score > 0:
        bi_score = sum(
            DOMAIN_SKILL_WEIGHTS.get("data_analyst", {}).get(s, 0)
            for s in skill_set
            if s in _BI_EXCLUSIVE_SKILLS
        )
        best_eng_score = max(
            (domain_scores.get(d, (0.0, []))[0] for d in _ENGINEERING_DOMAINS),
            default=0.0
        )
        if bi_score < _DATA_ANALYST_BI_MIN_SCORE and best_eng_score >= da_score:
            demotion_factor = 0.4
            demoted_score   = round(da_score * demotion_factor, 2)
            domain_scores["data_analyst"] = (demoted_score, da_evidence)
            logger.info(
                f"[da_guard] data_analyst demoted {da_score} → {demoted_score} "
                f"(bi_score={bi_score}, best_eng={best_eng_score})"
            )

    # ── Step 4: rank ──────────────────────────────────────────────────────────
    ranked = sorted(
        domain_scores.items(),
        key=lambda kv: (-kv[1][0], kv[0])
    )

    if not ranked:
        return fallback_domain, None, "low"

    primary_dom,   (primary_score,   primary_evidence)   = ranked[0]
    secondary_dom, (secondary_score, _secondary_evidence) = (
        ranked[1] if len(ranked) > 1 else (None, (0.0, []))
    )

    if primary_score == 0:
        primary_dom      = fallback_domain
        primary_evidence = []

    secondary = None
    if (
        secondary_dom
        and secondary_dom != primary_dom
        and primary_score > 0
        and secondary_score >= _SECONDARY_THRESHOLD * primary_score
    ):
        secondary = secondary_dom

    evidence_depth = len(primary_evidence) + len(combo_hits.get(primary_dom, []))
    if project_signals.get(primary_dom, 0) > 0:
        evidence_depth += 2   # project context is strong corroborating evidence
    if evidence_depth >= 3:
        confidence = "high"
    elif evidence_depth >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    logger.info(
        f"[detect_domain] primary='{primary_dom}' score={primary_score} "
        f"evidence={primary_evidence[:6]} combos={combo_hits.get(primary_dom, [])} "
        f"project_boost={project_signals.get(primary_dom, 0):.1f} | "
        f"secondary={secondary!r} score={secondary_score} | confidence={confidence}"
    )
    return primary_dom, secondary, confidence


def compute_role_confidence(
    skills:           List[str],
    primary_domain:   str,
    secondary_domain: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convert raw weighted scores into 0–100 confidence percentages.

    Algorithm
    ─────────
    1. Score every domain against the skill set (same weights as detect_multi_domain)
    2. Find the maximum score across all domains (the "ceiling")
    3. Express primary and secondary as a fraction of that ceiling
    4. Cap primary at 95 and secondary at 85 — never claim 100% certainty

    The percentage reflects how much of the available domain-specific
    evidence the candidate actually has, not just whether they matched.

    Returns
    ───────
    {"primary": int, "secondary": int}  — both 0-100
    """
    skill_set = frozenset(normalize(s) for s in skills)

    # Score all domains (same logic as detect_multi_domain step 1)
    all_scores: Dict[str, float] = {}
    for domain in sorted(DOMAINS.keys()):
        s, _ = _score_domain_weighted(skill_set, domain)
        all_scores[domain] = s
    # Include full_stack_engineer even if not in DOMAINS
    if "full_stack_engineer" not in all_scores:
        s, _ = _score_domain_weighted(skill_set, "full_stack_engineer")
        all_scores["full_stack_engineer"] = s

    max_score = max(all_scores.values(), default=1.0)
    if max_score == 0:
        return {"primary": 0, "secondary": 0}

    primary_raw   = all_scores.get(primary_domain,   0.0)
    secondary_raw = all_scores.get(secondary_domain, 0.0) if secondary_domain else 0.0

    primary_pct   = min(95, round(primary_raw   / max_score * 100))
    secondary_pct = min(85, round(secondary_raw / max_score * 100)) if secondary_domain else 0

    logger.debug(
        f"compute_role_confidence: primary='{primary_domain}' {primary_pct}% "
        f"secondary={secondary_domain!r} {secondary_pct}%"
    )
    return {"primary": primary_pct, "secondary": secondary_pct}
    """
    Explainable, deterministic score breakdown.
    Handles both ClassifiedSkills (single-domain) and HybridClassifiedSkills.
    """
    total_core     = len(classified.matched_core)     + len(classified.missing_core)
    total_jd       = len(classified.matched_jd)       + len(classified.missing_jd)
    total_optional = len(classified.matched_optional) + len(classified.missing_optional)

    core_weight     = len(classified.matched_core) * 3
    jd_weight       = len(classified.matched_jd)   * 2
    optional_weight = len(classified.matched_optional)

    explanation = [
        "Core skills contribute 3× weight to the score",
        "JD-specific skills contribute 2× weight",
        "Implied/inferred skills contribute 2× weight (present but not explicitly stated)",
        "Optional skills contribute 1× weight",
        f"Matched {len(classified.matched_core)}/{total_core} core skills",
        f"Matched {len(classified.matched_jd)}/{total_jd} JD-specific skills",
        f"Matched {len(classified.matched_optional)}/{total_optional} optional skills",
    ]

    inferred_weight  = 0.0
    secondary_weight = 0.0
    total_secondary  = 0

    # Inferred skills — context-aware credit (×2, same as JD-specific)
    if isinstance(classified, HybridClassifiedSkills):
        n_inferred    = len(classified.inferred_skills)
        inferred_weight = n_inferred * 2.0
        if n_inferred:
            explanation.append(
                f"{n_inferred} skill(s) credited as implied "
                f"({', '.join(sorted(classified.inferred_skills)[:4])})"
            )

    # Secondary domain — 0.5× weight
    if isinstance(classified, HybridClassifiedSkills) and classified.secondary_domain:
        total_secondary  = (len(classified.secondary_matched)
                            + len(classified.secondary_missing))
        secondary_weight = len(classified.secondary_matched) * 0.5
        explanation.append(
            f"Secondary domain '{classified.secondary_domain}' contributes 0.5× weight"
        )
        explanation.append(
            f"Matched {len(classified.secondary_matched)}/{total_secondary} "
            f"secondary domain skills"
        )

    total_possible = (
        total_core * 3 + total_jd * 2 + inferred_weight
        + total_optional + total_secondary * 0.5
    )
    matched_weight = core_weight + jd_weight + inferred_weight + optional_weight + secondary_weight

    return {
        "core_weight":      core_weight,
        "jd_weight":        jd_weight,
        "inferred_weight":  round(inferred_weight, 2),     # ← NEW
        "optional_weight":  optional_weight,
        "secondary_weight": round(secondary_weight, 2),
        "total_possible":   round(total_possible, 2),
        "matched_weight":   round(matched_weight, 2),
        "explanation":      sorted(explanation),
    }


def build_score_breakdown(classified: "ClassifiedSkills") -> dict:
    """
    Explainable, deterministic score breakdown for the UI.
    Handles both ClassifiedSkills (single-domain) and HybridClassifiedSkills.

    Returns a dict showing exactly how the skill score was computed,
    including inferred and secondary-domain contributions where present.
    """
    total_core     = len(classified.matched_core)     + len(classified.missing_core)
    total_jd       = len(classified.matched_jd)       + len(classified.missing_jd)
    total_optional = len(classified.matched_optional) + len(classified.missing_optional)

    core_weight     = len(classified.matched_core) * 3
    jd_weight       = len(classified.matched_jd)   * 2
    optional_weight = len(classified.matched_optional)

    explanation = [
        "Core skills contribute 3× weight to the score",
        "JD-specific skills contribute 2× weight",
        "Implied/inferred skills contribute 2× weight",
        "Optional skills contribute 1× weight",
        f"Matched {len(classified.matched_core)}/{total_core} core skills",
        f"Matched {len(classified.matched_jd)}/{total_jd} JD-specific skills",
        f"Matched {len(classified.matched_optional)}/{total_optional} optional skills",
    ]

    inferred_weight  = 0.0
    secondary_weight = 0.0
    total_secondary  = 0

    if isinstance(classified, HybridClassifiedSkills):
        n_inferred = len(classified.inferred_skills)
        inferred_weight = n_inferred * 2.0
        if n_inferred:
            explanation.append(
                f"{n_inferred} skill(s) credited as implied "
                f"({', '.join(sorted(classified.inferred_skills)[:4])})"
            )
        if classified.secondary_domain:
            total_secondary  = (len(classified.secondary_matched)
                                + len(classified.secondary_missing))
            secondary_weight = len(classified.secondary_matched) * 0.5
            explanation.append(
                f"Secondary domain '{classified.secondary_domain}' "
                f"contributes 0.5× weight"
            )
            explanation.append(
                f"Matched {len(classified.secondary_matched)}/{total_secondary} "
                f"secondary domain skills"
            )

    total_possible = (
        total_core * 3 + total_jd * 2 + inferred_weight
        + total_optional + total_secondary * 0.5
    )
    matched_weight = (
        core_weight + jd_weight + inferred_weight
        + optional_weight + secondary_weight
    )

    return {
        "core_weight":      core_weight,
        "jd_weight":        jd_weight,
        "inferred_weight":  round(inferred_weight, 2),
        "optional_weight":  optional_weight,
        "secondary_weight": round(secondary_weight, 2),
        "total_possible":   round(total_possible, 2),
        "matched_weight":   round(matched_weight, 2),
        "explanation":      sorted(explanation),
    }


def build_jd_analysis(
    detected_skills:  List[str],
    domain:           str,
    confidence:       str,
    secondary_domain: Optional[str] = None,
) -> dict:
    """
    UI-ready JD analysis summary. Deterministic given same inputs.
    """
    return {
        "detected_skills":  sorted(detected_skills),
        "detected_domain":  domain,
        "secondary_domain": secondary_domain,
        "confidence":       confidence,
        "skill_count":      len(detected_skills),
        "is_hybrid":        secondary_domain is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORE IMPROVEMENT SIMULATOR (deterministic, no AI)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_score_improvement(
    classified:          "HybridClassifiedSkills",
    jd_skills:           List[str],
    resume_skills:       List[str],
    domain:              str,
    secondary_domain:    Optional[str],
    current_ats_score:   float,
    current_skill_score: float,
    weight_skills:       float,
) -> List[dict]:
    """
    Simulate the ATS score impact of adding each missing skill individually.

    Algorithm
    ─────────
    For each candidate skill (up to 5, priority: core → jd → secondary → optional):
      1. Add the skill to a copy of resume_skills
      2. Re-run classify_hybrid_skills (pure skill-layer change)
      3. Compute new_skill_score (deterministic)
      4. Estimate new_ats using fixed non-skill contribution:
           fixed = current_ats - weight_skills × current_skill_score
           new_ats = fixed + weight_skills × new_skill_score
      5. Record delta

    Returns
    ───────
    List of dicts, sorted by new_score DESC, then skill name ASC.
    Each dict:
      { "skill": str, "new_score": float, "score_gain": float }

    Guarantees
    ──────────
    • Deterministic — same input → same output
    • All outputs sorted
    • Empty list returned when no improvement candidates exist
    """
    # Build candidate pool: core → jd → secondary → optional (deduplicated, sorted within tier)
    candidates: List[str] = []
    seen_candidates: set = set()

    for skill in (
        sorted(classified.missing_core)
        + sorted(classified.missing_jd)
        + sorted(classified.secondary_missing if isinstance(classified, HybridClassifiedSkills) else [])
        + sorted(classified.missing_optional)
    ):
        if skill not in seen_candidates:
            seen_candidates.add(skill)
            candidates.append(skill)
        if len(candidates) >= 5:   # cap computation at 5 simulations
            break

    if not candidates:
        return []

    # Non-skill ATS contribution is fixed for this resume (semantic + exp + edu)
    fixed_contribution = current_ats_score - weight_skills * current_skill_score

    simulations: List[dict] = []
    resume_norm = normalize_list(resume_skills)

    for skill in candidates:
        augmented_resume = normalize_list(resume_norm + [skill])
        new_classified   = classify_hybrid_skills(
            jd_skills, augmented_resume, domain, secondary_domain
        )
        new_skill_score = new_classified.skill_score()
        new_ats         = round(
            min(100.0, fixed_contribution + weight_skills * new_skill_score), 1
        )
        score_gain = round(new_ats - current_ats_score, 1)

        simulations.append({
            "skill":      skill,
            "new_score":  new_ats,
            "score_gain": score_gain,
        })

    # Sort: highest gain first, then alphabetically (deterministic tie-break)
    simulations.sort(key=lambda s: (-s["score_gain"], s["skill"]))
    logger.debug(
        f"simulate_score_improvement: {len(simulations)} simulations computed "
        f"(best: +{simulations[0]['score_gain']}% from '{simulations[0]['skill']}')"
        if simulations else "simulate_score_improvement: no candidates"
    )
    return simulations


# ══════════════════════════════════════════════════════════════════════════════
# STRICT ATS EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# Checks CS fundamentals regardless of what JD says.
# Even a strong tools-heavy resume gets penalised for missing these.

DOMAIN_FUNDAMENTALS: Dict[str, List[str]] = {
    "software_engineer": [
        "data structures", "algorithms", "system design", "database", "api",
    ],
    "backend": [
        "database", "api", "authentication", "server", "system design",
    ],
    "frontend": [
        "javascript", "html", "css", "react", "git",
    ],
    "full_stack_engineer": [           # ← NEW
        "api", "database", "authentication", "javascript", "backend",
    ],
    "frontend_developer": [            # ← NEW
        "javascript", "html", "css", "react", "git",
    ],
    "ai_ml": [
        "machine learning", "python", "statistics", "model training", "data analysis",
    ],
    "ai_engineer": [                   # ← NEW
        "machine learning", "python", "model training", "deep learning", "pytorch",
    ],
    "data_analyst": [
        "sql", "data analysis", "statistics", "excel", "python",
    ],
    "devops_cloud": [
        "docker", "ci/cd", "linux", "git", "cloud",
    ],
    "devops_engineer": [               # ← NEW
        "docker", "kubernetes", "ci/cd", "linux", "git",
    ],
    # Legacy aliases
    "web_dev":     ["javascript", "html", "css", "react", "git"],
    "data":        ["sql", "python", "data analysis", "statistics", "excel"],
    "devops":      ["docker", "ci/cd", "linux", "git", "cloud"],
}

# Weak-area detection patterns — maps a symptom to a diagnostic message
_WEAK_AREA_RULES: List[tuple] = [
    # (set_of_trigger_skills, weak_area_message)
    (
        {"data structures", "algorithms"},
        "No evidence of CS fundamentals (data structures / algorithms) — "
        "will fail whiteboard screening rounds at most product companies.",
    ),
    (
        {"system design"},
        "System design absent — critical for mid-senior roles; "
        "indicates limited experience with scalable architecture.",
    ),
    (
        {"database", "sql", "postgresql", "mysql"},
        "Database fundamentals missing — unexplained for any backend or full-stack role.",
    ),
    (
        {"testing"},
        "No mention of testing practices — signals potential code quality risk.",
    ),
    (
        {"git"},
        "Version control (git) not referenced — unusual gap for any engineering role.",
    ),
    (
        {"docker"},
        "Containerisation (Docker) absent — below market expectation for "
        "backend/cloud-adjacent roles in 2025.",
    ),
]


def get_weak_areas(
    resume_skills: List[str],
    domain:        str,
) -> List[str]:
    """
    Identify weak areas by checking critical skills that are NOT on the resume,
    regardless of JD content.

    Returns a sorted list of diagnostic strings.
    Each message explains the BUSINESS IMPACT of the gap, not just the name.
    """
    resume_set      = {normalize(s) for s in resume_skills}
    fundamentals    = DOMAIN_FUNDAMENTALS.get(domain, [])
    fund_set        = {normalize(s) for s in fundamentals}
    missing_fund    = fund_set - resume_set

    weak: List[str] = []

    for trigger_set, message in _WEAK_AREA_RULES:
        # Fire rule when ANY trigger skill is missing AND is relevant to this domain
        relevant_triggers = trigger_set & fund_set   # only rules applicable to domain
        if not relevant_triggers:
            continue
        if relevant_triggers & missing_fund:         # at least one trigger is missing
            weak.append(message)

    return sorted(weak)


def get_readiness_level(
    ats_score:      float,
    critical_count: int,
) -> str:
    """
    Derive a hiring-readiness label from ATS score + count of critical missing skills.

    Levels (strict — not career-coach):
      Not ready       → < 45 OR ≥ 3 critical missing
      Partially ready → 45-64 OR 1-2 critical missing
      Interview ready → ≥ 65 AND 0 critical missing
    """
    if ats_score < 45 or critical_count >= 3:
        return "Not ready"
    if ats_score < 65 or critical_count >= 1:
        return "Partially ready"
    return "Interview ready"


def evaluate_strict_ats(
    resume_skills:  List[str],
    classified:     HybridClassifiedSkills,
    domain:         str,
    ats_score:      float,
) -> dict:
    """
    Run the strict ATS evaluation layer.

    Returns
    ───────
    {
      "critical_missing": [...],   # fundamentals missing regardless of JD
      "weak_areas":       [...],   # diagnostic sentences per gap
      "readiness_level":  "...",   # Not ready / Partially ready / Interview ready
    }

    Design contract
    ───────────────
    • critical_missing is ALWAYS non-empty unless the resume is genuinely
      complete on every fundamental — real ATS always finds improvement areas.
    • When all fundamentals are present, at least one optional fundamental
      upgrade is injected so the list never returns empty.
    """
    resume_norm  = normalize_list(resume_skills)
    resume_set   = set(resume_norm)
    fundamentals = [normalize(s) for s in DOMAIN_FUNDAMENTALS.get(domain, [])]

    critical_missing = sorted([s for s in fundamentals if s not in resume_set])

    # Contract: NEVER return empty critical_missing — real ATS always surfaces gaps
    if not critical_missing:
        # Pick best improvement from optional skills not on resume
        optional_skills = [
            normalize(s)
            for s in DOMAINS.get(domain, {}).get("optional", [])
            if normalize(s) not in resume_set
        ]
        if optional_skills:
            critical_missing = sorted(optional_skills[:2])
        else:
            # Absolute fallback — depth signals
            critical_missing = sorted([
                "quantified project impact",
                "system scalability evidence",
            ])

    weak_areas     = get_weak_areas(resume_skills, domain)
    readiness      = get_readiness_level(ats_score, len(
        # Count genuine fundamental gaps (before fallback injection)
        [s for s in [normalize(f) for f in DOMAIN_FUNDAMENTALS.get(domain, [])]
         if s not in resume_set]
    ))

    logger.debug(
        f"strict_ats '{domain}': critical_missing={critical_missing} "
        f"weak_areas={len(weak_areas)} readiness='{readiness}'"
    )

    return {
        "critical_missing": critical_missing,
        "weak_areas":       weak_areas,
        "readiness_level":  readiness,
    }