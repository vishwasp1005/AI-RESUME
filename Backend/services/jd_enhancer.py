"""
services/jd_enhancer.py
──────────────────────────────────────────────────────────────────────────────
Domain-safe Job Description Expander — 4-layer architecture:

  Layer 1 — Role classification    : determines category BEFORE any LLM call
  Layer 2 — Constrained prompt     : role-specific, forbidden-skill lists baked in
  Layer 3 — Post-expansion guard   : strips hallucinated skills line-by-line
  Layer 4 — Deterministic fallback : safe hand-curated template when LLM unavailable

Key guarantee:
  "software engineer" → NEVER produces Kubernetes / TensorFlow / Terraform
  Only explicit specialist titles trigger specialist expansions.
"""

import httpx

from config import MODEL_NAME, OLLAMA_URL, TIMEOUT
from utils.logger import get_logger

logger = get_logger(__name__)

_WORD_THRESHOLD = 20   # JDs shorter than this are expanded


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROLE TAXONOMY — classification signals
# ══════════════════════════════════════════════════════════════════════════════
# GENERAL_SOFTWARE_ROLES is checked FIRST.  These roles must never drift
# into DevOps / AI-ML / Data regardless of what an LLM wants to add.

GENERAL_SOFTWARE_ROLES: frozenset = frozenset({
    "software engineer",       "software developer",    "software dev",
    "swe",                     "engineer",              "developer",
    "backend developer",       "backend engineer",      "backend dev",
    "back end developer",      "back-end developer",
    "full stack developer",    "fullstack developer",   "full stack engineer",
    "fullstack engineer",      "full stack",            "fullstack",
    "web developer",           "web engineer",
    "application developer",   "app developer",
    "junior developer",        "senior developer",
    "staff engineer",          "principal engineer",
})

_DEVOPS_SIGNALS: frozenset = frozenset({
    "devops", "site reliability", "sre", "cloud engineer",
    "infrastructure engineer", "platform engineer",
    "kubernetes", "terraform", "ci/cd", "ansible",
})

_AI_ML_SIGNALS: frozenset = frozenset({
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "deep learning", "llm", "ai/ml", "ai ml",
    "ml researcher", "research engineer", "computer vision engineer",
})

_FRONTEND_SIGNALS: frozenset = frozenset({
    "frontend", "front-end", "front end",
    "ui developer", "ux developer", "react developer",
    "vue developer", "angular developer", "javascript developer",
})

_DATA_SIGNALS: frozenset = frozenset({
    "data analyst", "data engineer", "data scientist", "bi developer",
    "etl developer", "analytics engineer",
})

_UI_UX_SIGNALS: frozenset = frozenset({
    "ui designer", "ux designer", "ui/ux", "product designer",
    "interaction designer", "visual designer",
})


# ══════════════════════════════════════════════════════════════════════════════
# 2. FORBIDDEN SKILL GUARDS — per role category
# ══════════════════════════════════════════════════════════════════════════════
# Any line in the LLM output that contains a forbidden term is stripped.
# Only general_software needs strict guarding — the other categories are
# already narrowly prompted.

_FORBIDDEN: dict = {
    "general_software": frozenset({
        # DevOps / infra — most common hallucinations for generic software roles
        "kubernetes", "k8s", "terraform", "ansible", "helm",
        "prometheus", "grafana", "jenkins", "ci/cd pipeline",
        "infrastructure as code", "iac", "infrastructure automation",
        "cloud infrastructure", "cloud architect", "site reliability", "sre",
        # AI / ML
        "pytorch", "tensorflow", "keras", "scikit-learn", "scikit learn",
        "machine learning", "deep learning", "nlp", "natural language processing",
        "large language model", "llm", "neural network", "model training",
        "reinforcement learning", "computer vision", "generative ai",
        # Data engineering
        "apache spark", "hadoop", "airflow", "etl pipeline", "data pipeline",
        "data warehouse", "dbt", "kafka",
        # Design
        "figma", "adobe xd", "wireframing", "prototyping", "ux research",
    }),
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROLE-CONSTRAINED PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_PROMPTS: dict = {
    "general_software": (
        "You are a conservative technical recruiter writing a GENERAL software engineering "
        "job description for a non-specialist role.\n\n"
        "STRICT NON-NEGOTIABLE RULES:\n"
        "- Include ONLY: programming fundamentals, REST APIs, SQL/databases, OOP, "
        "system design basics, debugging, testing, Git, backend/frontend basics\n"
        "- DO NOT mention: Kubernetes, Terraform, Ansible, Helm, any IaC or DevOps tooling\n"
        "- DO NOT mention: PyTorch, TensorFlow, machine learning, deep learning, NLP, "
        "LLMs, neural networks, or any AI/ML tools\n"
        "- DO NOT mention: Spark, Airflow, Hadoop, ETL pipelines, data warehousing\n"
        "- DO NOT mention: Figma, wireframing, UX research, or design tools\n"
        "- Keep Docker optional at most — do NOT list it as a core requirement\n"
        "- The candidate is a GENERALIST software engineer, not a DevOps or ML specialist\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
    "devops": (
        "You are a technical recruiter writing a DevOps / Site Reliability Engineering "
        "job description.\n"
        "Include: Docker, Kubernetes, Terraform, CI/CD, monitoring, cloud infrastructure, "
        "IaC, Linux, scripting.\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
    "ai_ml": (
        "You are a technical recruiter writing an AI/ML engineering job description.\n"
        "Include: machine learning, deep learning, Python, PyTorch or TensorFlow, "
        "model training, data analysis, NLP where relevant.\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
    "frontend": (
        "You are a technical recruiter writing a frontend engineering job description.\n"
        "Include: JavaScript, TypeScript, React/Vue/Angular, HTML, CSS, "
        "responsive design, REST API integration, testing.\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
    "data": (
        "You are a technical recruiter writing a data engineering or analytics "
        "job description.\n"
        "Include: SQL, Python, ETL, data pipelines, data warehousing, "
        "Spark/Airflow where appropriate, analytics tools.\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
    "ui_ux": (
        "You are a technical recruiter writing a UI/UX design job description.\n"
        "Include: Figma, wireframing, prototyping, user research, usability testing, "
        "interaction design, design systems.\n"
        "DO NOT include backend programming, CI/CD, or ML tools.\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
    "unknown": (
        "You are an expert technical recruiter. Convert the following job title into a "
        "detailed professional job description. Stay strictly within the domain implied "
        "by the title. DO NOT add unrelated technical skills from other specialisations.\n\n"
        "Job Title: {jd_text}\n\n"
        "Return ONLY the job description text. No preamble. No markdown fences."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 4. SAFE DETERMINISTIC TEMPLATES (LLM-free fallback)
# ══════════════════════════════════════════════════════════════════════════════
# Used when Ollama is unavailable OR when post-sanitization strips too much.
# Hand-curated — guaranteed to be domain-correct with zero hallucination risk.

_SAFE_TEMPLATES: dict = {
    "general_software": """Role Overview:
We are seeking a Software Engineer to design, develop, and maintain scalable software systems.
The ideal candidate writes clean, well-tested code and collaborates effectively across the team.

Key Responsibilities:
- Design and implement RESTful APIs and backend services
- Write unit, integration, and end-to-end tests to ensure code quality
- Debug, troubleshoot, and resolve application issues across the stack
- Participate in code reviews, architecture discussions, and sprint planning
- Collaborate with product and design teams to deliver features on time
- Maintain and improve technical documentation

Required Skills:
- Proficiency in at least one backend language: Python, Java, JavaScript (Node.js), or Go
- RESTful API design and development
- Relational databases: SQL, PostgreSQL, or MySQL
- Object-oriented programming and software design patterns
- Version control with Git and collaborative code-review workflows
- Understanding of software architecture and system design principles
- Testing practices: unit testing, integration testing, or TDD

Nice-to-have:
- Familiarity with containerisation (Docker)
- Basic experience with a cloud platform (AWS, Azure, or GCP)
- Knowledge of agile / scrum methodologies

Experience: 2–5 years of software development experience.
Education: Bachelor's degree in Computer Science, Engineering, or equivalent practical experience.""",

    "devops": """Role Overview:
We are seeking a DevOps / Site Reliability Engineer to automate, scale, and maintain our infrastructure.

Key Responsibilities:
- Design and maintain CI/CD pipelines using Jenkins, GitHub Actions, or GitLab CI
- Manage Kubernetes clusters and containerised workloads with Docker
- Write infrastructure-as-code using Terraform or Ansible
- Implement monitoring, alerting, and incident response with Prometheus and Grafana
- Ensure high availability, security, and scalability of cloud environments

Required Skills:
- Docker and Kubernetes (container orchestration)
- Terraform or Ansible (Infrastructure as Code)
- CI/CD pipeline design and maintenance
- Cloud platform: AWS, Azure, or GCP
- Linux administration and shell scripting
- Monitoring: Prometheus, Grafana, or Datadog

Experience: 3+ years in DevOps, SRE, or infrastructure engineering.""",

    "ai_ml": """Role Overview:
We are hiring an AI/ML Engineer to build, train, and deploy machine learning models at scale.

Key Responsibilities:
- Design and implement ML models for production use cases
- Fine-tune large language models and deep learning architectures
- Build data pipelines for model training, evaluation, and deployment
- Collaborate with data scientists and product teams

Required Skills:
- Python (NumPy, Pandas, scikit-learn)
- PyTorch or TensorFlow / Keras
- Model training, evaluation, and deployment (MLOps)
- NLP, computer vision, or LLM experience
- SQL and data manipulation

Experience: 2+ years in ML engineering or applied data science.""",

    "frontend": """Role Overview:
We are looking for a Frontend Engineer to build responsive, accessible user interfaces.

Key Responsibilities:
- Build and maintain UI components in React, Vue, or Angular
- Integrate with REST APIs and GraphQL endpoints
- Ensure cross-browser compatibility and responsive design
- Write unit and end-to-end tests for frontend components

Required Skills:
- JavaScript (ES6+) and TypeScript
- React, Vue, or Angular
- HTML5, CSS3, responsive and accessible design
- REST API integration and state management
- Git and agile collaboration

Experience: 2–4 years of frontend development.""",
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def _classify_role(jd_text: str) -> str:
    """
    Classify a short JD into a role category BEFORE any LLM call.

    Returns: 'general_software' | 'devops' | 'ai_ml' | 'frontend' |
             'data' | 'ui_ux' | 'unknown'

    GENERAL_SOFTWARE checked first — must never drift to specialist domains.
    Signals checked longest-first to avoid substring ambiguity.
    """
    text = jd_text.lower().strip()

    # Priority 1 — general software (must match before any specialisation)
    for role in sorted(GENERAL_SOFTWARE_ROLES, key=len, reverse=True):
        if role in text:
            logger.info(f"[classify] '{jd_text}' → general_software (signal: '{role}')")
            return "general_software"

    # Priority 2 — specialisations (only reached if no general match)
    checks = [
        (_DEVOPS_SIGNALS,   "devops"),
        (_AI_ML_SIGNALS,    "ai_ml"),
        (_FRONTEND_SIGNALS, "frontend"),
        (_DATA_SIGNALS,     "data"),
        (_UI_UX_SIGNALS,    "ui_ux"),
    ]
    for signal_set, category in checks:
        for signal in sorted(signal_set, key=len, reverse=True):
            if signal in text:
                logger.info(f"[classify] '{jd_text}' → {category} (signal: '{signal}')")
                return category

    logger.info(f"[classify] '{jd_text}' → unknown")
    return "unknown"


def _sanitize_expanded_jd(expanded: str, role_category: str) -> str:
    """
    Strip lines that contain forbidden skills for this role category.

    Conservative: only removes lines that contain a confirmed forbidden token.
    Does not rewrite or paraphrase — just drops offending lines.
    """
    forbidden = _FORBIDDEN.get(role_category)
    if not forbidden:
        return expanded   # no rules for this category — passthrough

    lines         = expanded.split("\n")
    clean_lines   = []
    removed_count = 0

    for line in lines:
        line_lower = line.lower()
        triggered  = [t for t in forbidden if t in line_lower]
        if triggered:
            removed_count += 1
            logger.debug(
                f"[sanitize/{role_category}] removed: {line.strip()[:70]!r} "
                f"(forbidden: {triggered[:3]})"
            )
        else:
            clean_lines.append(line)

    if removed_count:
        logger.warning(
            f"[sanitize] Removed {removed_count} hallucinated line(s) from "
            f"'{role_category}' expansion (LLM ignored constraints)."
        )

    return "\n".join(clean_lines).strip()


def _fallback_expansion(jd_text: str, role_category: str) -> str:
    """Return a safe, deterministic template — zero LLM involvement."""
    template = _SAFE_TEMPLATES.get(role_category)
    if template:
        logger.info(f"[fallback] Using deterministic template for '{role_category}'.")
        return template
    logger.info(f"[fallback] No template for '{role_category}' — returning original text.")
    return jd_text


def expand_job_description(jd_text: str) -> tuple:   # (str, bool)
    """
    Domain-safe JD expansion.

    Pipeline
    ────────
    1. Length check   — if ≥ _WORD_THRESHOLD: return as-is (unchanged)
    2. Classify role  — determines prompt and sanitizer before any LLM call
    3. LLM call       — role-constrained prompt prevents hallucination at source
    4. Sanitize       — strips any forbidden skills the LLM still managed to add
    5. Length check   — if sanitization stripped too much: use deterministic template
    6. Fallback       — Ollama offline / timeout → deterministic template always available

    Returns
    ───────
    (jd, was_expanded) — same interface as the original function
    """
    word_count = len(jd_text.split())

    if word_count >= _WORD_THRESHOLD:
        logger.debug(f"JD is {word_count} words — no expansion needed.")
        return jd_text, False

    role_category = _classify_role(jd_text)
    logger.info(
        f"[expand] Short JD ({word_count}w): '{jd_text[:60]}' | "
        f"category='{role_category}'"
    )

    prompt  = _PROMPTS[role_category].format(jd_text=jd_text.strip())
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

        if not raw or len(raw.split()) < _WORD_THRESHOLD:
            logger.warning(
                f"[expand] LLM returned {len(raw.split())} words — using fallback."
            )
            return _fallback_expansion(jd_text, role_category), True

        # Post-expansion domain guard
        sanitized = _sanitize_expanded_jd(raw, role_category)

        if len(sanitized.split()) < _WORD_THRESHOLD:
            logger.warning(
                "[expand] LLM hallucinated extensively — sanitizer stripped too much. "
                "Switching to deterministic safe template."
            )
            return _fallback_expansion(jd_text, role_category), True

        logger.info(
            f"[expand] {word_count}w → {len(sanitized.split())}w "
            f"(category='{role_category}')"
        )
        return sanitized, True

    except httpx.ConnectError:
        logger.warning("[expand] Ollama not reachable — using deterministic fallback.")
        return _fallback_expansion(jd_text, role_category), True
    except httpx.TimeoutException:
        logger.warning("[expand] Ollama timed out — using deterministic fallback.")
        return _fallback_expansion(jd_text, role_category), True
    except Exception as exc:
        logger.error(f"[expand] Unexpected error: {exc}")
        return _fallback_expansion(jd_text, role_category), True