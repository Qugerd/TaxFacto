"""
HTTP API микросервиса рекомендаций (Задача 3).

Запуск из services/recommendation/:
    uvicorn app.main:app --reload --port 8003
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.logging_config import configure_logging
from core.predictor import PredictError
from app.recommender import recommend_markers


configure_logging()
logger = logging.getLogger("taxfacto.recommendation")
ui_logger = logging.getLogger("taxfacto.recommendation.ui")

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


def _request_id(request: Request) -> str:
    raw = request.headers.get("X-Request-ID", "").strip()
    return raw or str(uuid.uuid4())


def _format_recommendations(items: list[dict[str, Any]]) -> str:
    if not items:
        return "-"
    return ",".join(f"{item['tag']}(+{item['gain']})" for item in items)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/recommend", response_model=RecommendResponse)
def recommend(request_body: RecommendRequest, request: Request) -> dict[str, Any]:
    req_id = _request_id(request)
    started = time.perf_counter()
    case_ids = [c.case_id for c in request_body.similar_cases if c.case_id]
    case_ids_str = ",".join(case_ids) if case_ids else "-"
    n_cases = len(request_body.similar_cases)
    n_success = sum(1 for c in request_body.similar_cases if c.outcome == 1)
    n_refusal = n_cases - n_success
    baseline = request_body.prediction.probability

    logger.info(
        "recommend_request request_id=%s n_cases=%s baseline_probability=%s "
        "query_len=%s case_ids=%s",
        req_id,
        n_cases,
        baseline,
        len(request_body.query_text),
        case_ids_str,
    )
    ui_logger.info("[%s] Загрузка запроса", req_id)
    ui_logger.info("[%s] Вероятность удовлетворения запроса: %s%%", req_id, baseline)
    ui_logger.info(
        "[%s] Релевантных дел: %s (удовлетворённых: %s, отказов: %s)",
        req_id,
        n_cases,
        n_success,
        n_refusal,
    )

    try:
        result = recommend_markers(
            query_text=request_body.query_text,
            similar_cases=[case.model_dump() for case in request_body.similar_cases],
            prediction=request_body.prediction.model_dump(),
        )
    except PredictError as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        logger.warning(
            "recommend_error request_id=%s error_code=%s message=%s duration_ms=%s",
            req_id,
            exc.code,
            str(exc),
            duration_ms,
        )
        ui_logger.info("[%s] Ошибка: %s - %s", req_id, exc.code, exc)
        raise HTTPException(
            status_code=400,
            detail={"error": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        logger.exception(
            "recommend_unhandled request_id=%s exc_type=%s duration_ms=%s",
            req_id,
            type(exc).__name__,
            duration_ms,
        )
        ui_logger.info("[%s] Внутренняя ошибка (%s)", req_id, type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Внутренняя ошибка рекомендаций",
            },
        ) from exc

    recommendations = result.get("recommendations", [])
    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "recommend_ok request_id=%s baseline_probability=%s n_recommendations=%s "
        "recommendations=%s duration_ms=%s",
        req_id,
        result["baseline_probability"],
        len(recommendations),
        _format_recommendations(recommendations),
        duration_ms,
    )

    rec_summary = ", ".join(
        f"{item['tag']}: +{item['gain']}%"
        for item in recommendations
    ) or "-"
    ui_logger.info("[%s] Поиск рекомендаций..", req_id)
    ui_logger.info("[%s] Найдено рекомендаций: %s", req_id, len(recommendations))
    ui_logger.info("[%s] Рекомендации: [%s]", req_id, rec_summary)
    return result
