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
ui_logger = logging.getLogger("taxfacto.prediction.ui")


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
    case_ids_str = ", ".join(case_ids) if case_ids else "-"
    n_cases = len(request_body.similar_cases)
    query_word_count = len(tokenize(request_body.query_text))
    query_len = len(request_body.query_text)

    logger.info(
        "predict_request request_id=%s n_cases=%s query_len=%s query_word_count=%s case_ids=%s",
        req_id,
        n_cases,
        query_len,
        query_word_count,
        case_ids_str,
    )
    ui_logger.info("[%s] Загрузка дела", req_id)
    ui_logger.info("[%s] Обработка текста запроса..", req_id)
    ui_logger.info("[%s] Запрос: %s слов, %s символов", req_id, query_word_count, query_len)

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
        ui_logger.info("[%s] Ошибка: %s - %s", req_id, exc.code, exc)
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
        ui_logger.info("[%s] Внутренняя ошибка (%s)", req_id, type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail={"error": "INTERNAL_ERROR", "message": "Внутренняя ошибка прогноза"},
        ) from exc

    details = result.get("details", [])
    n_scored = len(details)
    n_success = sum(1 for d in details if d["outcome"] == 1)
    n_refusal = n_scored - n_success
    duration_ms = round((time.perf_counter() - started) * 1000)

    logger.info(
        "predict_ok request_id=%s probability=%s n_scored=%s duration_ms=%s",
        req_id,
        result["probability"],
        n_scored,
        duration_ms,
    )
    # details уже отсортированы по убыванию схожести (predictor.py)
    scored_summary = ", ".join(
        f"{d.get('case_id') or 'unknown'}: {d['similarity_pct']}%"
        for d in details
    )
    ui_logger.info(
        "[%s] Загружено релевантных дел: %s (удовлетворённых: %s, отказов: %s)",
        req_id,
        n_scored,
        n_success,
        n_refusal,
    )
    ui_logger.info("[%s] Расчёт схожести..", req_id)
    ui_logger.info("[%s] Схожесть по делам: [%s]", req_id, scored_summary)
    ui_logger.info(
        "[%s] Вероятность удовлетворения запроса: %s%%",
        req_id,
        result["probability"],
    )
    return result
