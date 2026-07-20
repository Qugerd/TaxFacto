"""
HTTP API микросервиса прогноза успеха (Задача 2)

Запуск из services/prediction/:
    uvicorn app.main:app --reload --port 8002
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from core.logging_config import configure_logging
from core.predictor import PredictError, predict_success, tokenize


configure_logging()
logger = logging.getLogger("taxfacto.prediction")


app = FastAPI(title="Task 2 Predictor", version="1.0.0")


class SimilarCase(BaseModel):
    text: str = Field(..., min_length=1)
    outcome: Literal[0, 1]
    case_id: str | None = None


class PredictRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    similar_cases: list[SimilarCase] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    probability: int
    details: list[dict[str, Any]]


def _request_id(request: Request) -> str:
    raw = request.headers.get("X-Request-ID", "").strip()
    return raw or str(uuid.uuid4())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(request_body: PredictRequest, request: Request) -> dict[str, Any]:
    req_id = _request_id(request)
    started = time.perf_counter() 
    case_ids = [c.case_id for c in request_body.similar_cases if c.case_id]
    case_ids_str = ",".join(case_ids) if case_ids else "-"
    query_word_count = len(tokenize(request_body.query_text))

    logger.info(
        "predict_request request_id=%s n_cases=%s query_len=%s query_word_count=%s case_ids=%s",
        req_id,
        len(request_body.similar_cases),
        len(request_body.query_text),
        query_word_count,
        case_ids_str,
    )

    try:
        result = predict_success(
            query_text=request_body.query_text,
            similar_cases=[case.model_dump() for case in request_body.similar_cases],
        )
    except PredictError as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        logger.warning(
            "predict_error request_id=%s error_code=%s message=%s duration_ms=%s",
            req_id,
            exc.code,
            str(exc),
            duration_ms,
        )
        raise HTTPException(
            status_code=400,
            detail={"error": exc.code, "message": str(exc)},
        ) from exc
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000)
        logger.exception(
            "predict_unhandled request_id=%s exc_type=%s duration_ms=%s",
            req_id,
            type(exc).__name__,
            duration_ms,
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": "Внутренняя ошибка прогноза"},
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "predict_ok request_id=%s probability=%s n_scored=%s duration_ms=%s",
        req_id,
        result["probability"],
        len(result.get("details", [])),
        duration_ms,
    )
    return result
