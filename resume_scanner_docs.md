# AI Resume Screening System — Complete Documentation

## Project Structure

```
Resume Scanner Project/
├── Backend/                        ← FastAPI backend
│   ├── main.py                     ← App entry point, CORS, routers
│   ├── config.py                   ← MODEL_NAME, OLLAMA_URL, TIMEOUT, weights
│   ├── requirements.txt            ← Python deps
│   ├── routes/
│   │   └── analyze.py              ← POST /api/v1/analyze
│   ├── services/
│   │   ├── parser.py               ← PyMuPDF PDF text extraction
│   │   ├── matcher.py              ← Semantic + skills + exp + edu scoring
│   │   ├── skill_extractor.py      ← Keyword-based skill extraction
│   │   ├── llm_service.py          ← Ollama integration + fallback
│   │   ├── domain_detector.py      ← AI/Finance/Marketing/General detection
│   │   └── suggestions.py         ← Resume improvement suggestions engine
│   ├── utils/
│   │   └── logger.py               ← Dual console+file logger
│   ├── models/
│   │   └── schema.py               ← Pydantic response schemas
│   └── data/
│       └── skills.json             ← 200+ skills across 10 domains
│
└── src/                            ← React/Vite frontend
    ├── App.jsx                     ← Main UI component
    ├── App.css                     ← Full dark-theme design system
    ├── index.css                   ← Global reset
    └── main.jsx                   ← React entry point
```

---

## API Response Format

```json
{
  "status": "success",
  "data": {
    "filename": "resume.pdf",
    "ats_score": 74.5,
    "match_level": "Good",
    "matched_skills": ["python", "fastapi", "docker"],
    "missing_skills": ["kubernetes", "terraform"],
    "section_scores": {
      "semantic": 82.0,
      "skills": 66.7,
      "experience": 60.0,
      "education": 48.0
    },
    "suggestions": ["Add missing skills: kubernetes, terraform", "..."],
    "ai_insight": "🔍 STRENGTHS:\n...\n\n🛑 WEAKNESSES:\n...\n\n📋 FINAL VERDICT:\n...",
    "domain": "ai_ml"
  },
  "error": null
}
```

---

## Hybrid ATS Scoring Formula

```
ATS Score = 0.40 × semantic + 0.30 × skills + 0.20 × experience + 0.10 × education
```

| Component | Weight | Method |
|-----------|--------|--------|
| Semantic  | 40%    | Cosine similarity via `all-MiniLM-L6-v2` |
| Skills    | 30%    | Keyword match against `skills.json` |
| Experience| 20%    | Year-range regex + keyword heuristic |
| Education | 10%    | Keyword presence heuristic |

---

## How to Run

### 1. Backend

```bash
cd "Resume Scanner Project/Backend"

# Install dependencies
pip install -r requirements.txt

# Start backend (runs on :8000)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> Backend docs available at: http://localhost:8000/docs

### 2. Frontend

```bash
cd "Resume Scanner Project"

# Install frontend deps (if not already)
npm install

# Start frontend dev server (runs on :5173 or :5174)
npm run dev
```

> Open: http://localhost:5174/

### 3. Optional — Ollama (AI Insights)

```bash
# Install from https://ollama.com/download
ollama pull llama3
ollama serve   # runs on :11434
```

> If Ollama is not running, the system gracefully falls back to rule-based insights.

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| PDF Parsing | PyMuPDF (fitz) with text cleaning |
| Semantic Matching | `sentence-transformers/all-MiniLM-L6-v2` |
| Skill Detection | Keyword regex across 200+ skills in 10 domains |
| Domain Detection | ai_ml / finance / marketing / general |
| LLM Insights | Ollama llama3 with JSON-formatted prompt |
| Fallback Mode | Full rule-based insight when Ollama is offline |
| Error Handling | Never crashes — always returns structured JSON |
| Logging | Dual file+console logger with timestamps |
| Suggestions | 6-8 personalized improvement recommendations |

---

## UI Components

| Component | Location |
|-----------|----------|
| Animated score ring (SVG) | `ScoreRing` in App.jsx |
| Segmented progress bars | `ProgressBar` in App.jsx |
| Skill badges (green/red) | `BadgeList` in App.jsx |
| Suggestion cards | `SuggestionList` in App.jsx |
| Drag-and-drop upload | `dropzone` in App.jsx |
| AI insight panel | `ai-insight-box` in App.jsx |
| Domain chip | inline in results section |
| Quick stats 2x2 grid | inline in right column |

