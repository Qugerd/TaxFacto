"""Рекомендации маркеров для повышения прогноза (Задача 3)"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from core.predictor import PredictError, predict_success
from app.patterns import TAG_APPEND_SEP, extract_case_tags, resolve_case_tags, resolve_case_tags_source

MAX_CANDIDATES = 12


@contextmanager
def _quiet_predictor() -> Iterator[None]:
    """Не дублировать детальные логи Задачи 2 при симуляциях кандидатов."""
    pred_logger = logging.getLogger("taxfacto.predictor")
    previous = pred_logger.disabled
    pred_logger.disabled = True
    try:
        yield
    finally:
        pred_logger.disabled = previous


def _collect_candidates(query_text: str, similar_cases: list[dict]) -> list[str]:
    """Маркеры из выигранных дел, которых ещё нет в query_text"""
    
    query_lower = query_text.lower()
    query_tags = set(extract_case_tags(query_text))
    seen: set[str] = set()
    candidates: list[str] = []

    for case in similar_cases:
        if int(case["outcome"]) != 1:
            continue
        for tag in resolve_case_tags(case):
            if tag in query_tags or tag.lower() in query_lower or tag in seen:
                continue
            seen.add(tag)
            candidates.append(tag)

    candidates.sort()
    return candidates


def recommend_markers(
    query_text: str,
    similar_cases: list[dict],
    prediction: dict[str, Any],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    """
    Рекомендует маркеры, увеличивающие прогноз
    Args:
        query_text: текст анализируемого дела
        similar_cases: [{ "text", "outcome", "case_id"?, "tags"? }, ...]
        prediction: { "probability": int } — выход Задачи 2
    Returns:
        baseline_probability, extracted_tags, recommendations, n_candidates
    """


    if not str(query_text).strip():
        raise PredictError("VALIDATION_ERROR", "query_text не может быть пустым")

    if not similar_cases:
        raise PredictError("VALIDATION_ERROR", "similar_cases не может быть пустым")

    if "probability" not in prediction:
        raise PredictError(
            "VALIDATION_ERROR",
            "prediction: отсутствует поле 'probability'",
        )


    p0 = int(prediction["probability"])


    extracted_tags: list[dict[str, Any]] = []
    for case in similar_cases:
        extracted_tags.append(
            {
                "case_id": case.get("case_id"),
                "outcome": int(case["outcome"]),
                "tags": resolve_case_tags(case),
                "tag_source": resolve_case_tags_source(case),
            }
        )

    candidates = _collect_candidates(query_text, similar_cases)[:max_candidates]
    n_candidates = len(candidates)

    recommendations: list[dict[str, Any]] = []
    with _quiet_predictor():
        for tag in candidates:
            try:
                result = predict_success(
                    query_text + TAG_APPEND_SEP + tag,
                    similar_cases,
                )
            except PredictError:
                continue

            p1 = int(result["probability"])
            if p1 > p0:
                recommendations.append(
                    {
                        "tag": tag,
                        "gain": p1 - p0,
                        "simulated_probability": p1,
                    }
                )

    recommendations.sort(key=lambda item: item["gain"], reverse=True)

    return {
        "baseline_probability": p0,
        "extracted_tags": extracted_tags,
        "recommendations": recommendations,
        "n_candidates": n_candidates,
    }
