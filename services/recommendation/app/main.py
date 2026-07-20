"""
HTTP API микросервиса рекомендаций (Задача 3).

Запуск из services/recommendation/:
    uvicorn app.main:app --reload --port 8003
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from core.predictor import PredictError
from app.recommender import recommend_markers


app = FastAPI(title="TaxFacto Recommendation Service", version="1.0.0")


class SimilarCase(BaseModel):
    text: str = Field(..., min_length=1)
    outcome: Literal[0, 1]
    case_id: str | None = None
    tags: list[str] | None = None


class PredictionInput(BaseModel):
    probability: int = Field(..., ge=0, le=100)
    details: list[dict[str, Any]] | None = None


class RecommendRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    similar_cases: list[SimilarCase] = Field(..., min_length=1)
    prediction: PredictionInput


class RecommendationItem(BaseModel):
    tag: str
    gain: int
    simulated_probability: int


class ExtractedTagsItem(BaseModel):
    case_id: str | None = None
    outcome: int
    tags: list[str]
    tag_source: Literal["provided", "extracted"]


class RecommendResponse(BaseModel):
    baseline_probability: int
    extracted_tags: list[ExtractedTagsItem]
    recommendations: list[RecommendationItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> dict[str, Any]:
    try:
        return recommend_markers(
            query_text=request.query_text,
            similar_cases=[case.model_dump() for case in request.similar_cases],
            prediction=request.prediction.model_dump(),
        )
    except PredictError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": exc.code, "message": str(exc)},
        ) from exc

