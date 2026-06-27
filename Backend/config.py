import os

# ─── Ollama / LLM ───────────────────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
TIMEOUT: int = int(os.getenv("TIMEOUT", "60"))

# ─── Sentence Transformer ────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─── Scoring Weights ─────────────────────────────────────────────────────────────
WEIGHT_SEMANTIC: float = 0.40
WEIGHT_SKILLS: float = 0.30
WEIGHT_EXPERIENCE: float = 0.20
WEIGHT_EDUCATION: float = 0.10

# ─── Match Level Thresholds ──────────────────────────────────────────────────────
MATCH_EXCELLENT: int = 80
MATCH_GOOD: int = 60
MATCH_AVERAGE: int = 40