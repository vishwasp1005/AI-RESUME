from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.analyze import router as analyze_router
from utils.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="AI Resume Screening System",
    description="Analyze resumes against job descriptions using semantic AI and keyword matching.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])


@app.get("/", tags=["Health"])
async def root():
    logger.info("Health check hit.")
    return {"status": "running", "service": "AI Resume Screening System"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}