from pydantic import BaseModel, Field
from typing import List, Optional


class SectionScores(BaseModel):
    semantic: float = Field(..., ge=0, le=100)
    skills: float = Field(..., ge=0, le=100)
    experience: float = Field(..., ge=0, le=100)
    education: float = Field(..., ge=0, le=100)


class AnalysisData(BaseModel):
    filename: str
    ats_score: float = Field(..., ge=0, le=100)
    match_level: str
    matched_skills: List[str]
    missing_skills: List[str]
    section_scores: SectionScores
    suggestions: List[str]
    ai_insight: str
    domain: Optional[str] = "general"


class AnalysisResponse(BaseModel):
    status: str
    data: Optional[AnalysisData] = None
    error: Optional[str] = None