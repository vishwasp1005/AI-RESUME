from typing import Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_DOMAIN_KEYWORDS = {
    "ai_ml": [
        "machine learning", "deep learning", "neural network", "nlp",
        "computer vision", "tensorflow", "pytorch", "data science",
        "llm", "transformers", "hugging face", "reinforcement learning",
        "mlops", "model training", "ai engineer", "data scientist"
    ],
    "finance": [
        "financial modeling", "investment banking", "equity research",
        "portfolio management", "risk management", "accounting", "audit",
        "fintech", "bloomberg", "trading", "dcf", "lbo", "valuation",
        "asset management", "hedge fund", "private equity", "quantitative"
    ],
    "marketing": [
        "digital marketing", "seo", "sem", "google ads", "content marketing",
        "brand strategy", "social media", "email marketing", "growth hacking",
        "hubspot", "salesforce", "influencer", "performance marketing",
        "market research", "crm", "lead generation", "copywriting"
    ],
}


def detect_domain(jd_text: str) -> str:
    """
    Detect the primary domain from the job description.
    Returns one of: 'ai_ml', 'finance', 'marketing', 'general'.
    """
    text_lower = jd_text.lower()
    scores = {domain: 0 for domain in _DOMAIN_KEYWORDS}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += 1

    best_domain, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score == 0:
        result = "general"
    else:
        result = best_domain

    logger.info(f"Detected domain: {result}  (scores={scores})")
    return result