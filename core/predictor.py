"""
Прогноз вероятности успеха судебного дела
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any


MIN_WORD_LEN = 4
SMOOTHING = 0.1


logger = logging.getLogger("taxfacto.predictor")


class PredictError(ValueError):
    """Ошибка расчёта прогноза"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def tokenize(text: str, min_len: int = 4) -> list[str]:
    """Извлекает из текста русские слова длиной min_len и более"""
    return re.findall(r"[а-я]{%d,}" % min_len, text.lower())


def compute_similarity(query_words: list[str], case_words: set[str]) -> float:
    """Схожесть: доля слов запроса, найденных в тексте дела"""
    if not query_words:
        return 0.0
    hits = sum(1 for word in query_words if word in case_words) # количество слов запроса, найденных в тексте дела
    return hits / len(query_words) # доля слов запроса, найденных в тексте дела


def _outcome_label(outcome: int) -> str:
    return "успех" if outcome == 1 else "отказ"


def _validate_similar_case(case: dict, index: int) -> None:
    if "text" not in case or not str(case["text"]).strip():
        raise PredictError(
            "VALIDATION_ERROR",
            f"similar_cases[{index}]: обязательное поле 'text' пустое",
        )
    if "outcome" not in case:
        raise PredictError(
            "VALIDATION_ERROR",
            f"similar_cases[{index}]: отсутствует поле 'outcome'",
        )
    outcome = case["outcome"]
    if outcome not in (0, 1):
        raise PredictError(
            "VALIDATION_ERROR",
            f"similar_cases[{index}]: outcome должен быть 0 или 1, получено {outcome!r}",
        )


def _case_log_key(case_id: Any, index: int) -> str:
    if case_id is not None and str(case_id).strip():
        return str(case_id).strip()
    return f"unknown_{index}"


def _log_scored_cases(scored: list[dict[str, Any]], query_word_count: int) -> None:
    n_success = sum(1 for item in scored if item["outcome"] == 1)
    n_refusal = len(scored) - n_success
    logger.info(
        "predict_relevant_cases n_relevant=%s n_success=%s n_refusal=%s",
        len(scored),
        n_success,
        n_refusal,
    )
    cases_payload = {
        _case_log_key(item.get("case_id"), index): {
            "status": item["outcome_label"],
            "matched_words": item["matched_words"],
            "query_words": query_word_count,
            "case_words": item["case_words"],
            "similarity_pct": item["similarity_pct"],
        }
        for index, item in enumerate(scored)
    }
    logger.info(
        "predict_cases_detail %s",
        json.dumps(cases_payload, ensure_ascii=False, sort_keys=True),
    )


def _fail(code: str, message: str, **ctx: Any) -> PredictError:
    parts = " ".join(f"{k}={v}" for k, v in ctx.items())
    extra = f" {parts}" if parts else ""
    logger.warning("predict_fail error_code=%s message=%s%s", code, message, extra)
    return PredictError(code, message)


def predict_success(
    query_text: str,
    similar_cases: list[dict],
    *,
    min_word_len: int = MIN_WORD_LEN,
    smoothing: float = SMOOTHING,
) -> dict[str, Any]:
    """
    Рассчитывает вероятность успеха (%)
    Args:
        query_text: текст анализируемого дела
        similar_cases: [{ "text": str, "outcome": 0|1, "case_id": str? }, ...]
    Returns:
        { "probability": int, "details": [...] }
    """
    n_cases = len(similar_cases) if similar_cases is not None else 0 # сколько дел пришло

    if not str(query_text).strip():
        raise _fail(
            "VALIDATION_ERROR",
            "query_text не может быть пустым",
            n_cases=n_cases,
        )

    if not similar_cases:
        raise _fail(
            "VALIDATION_ERROR",
            "similar_cases не может быть пустым",
            n_cases=0,
        )

    # валидация дел
    for i, case in enumerate(similar_cases):
        try:
            _validate_similar_case(case, i)
        except PredictError as exc:
            raise _fail(exc.code, str(exc), n_cases=n_cases) from exc

    query_words = tokenize(query_text, min_len=min_word_len) # токенезация слов запроса
    query_word_count = len(query_words) # количество слов в запросе
    if not query_words: # если нет слов в запросе, то выбрасываем ошибку
        raise _fail(
            "NO_QUERY_WORDS",
            f"В query_text нет слов из кириллицы длиной ≥ {min_word_len}",
            n_cases=n_cases,
            query_word_count=0,
        )

    scored: list[dict[str, Any]] = [] 
    n_skipped = 0 # сколько дел отброшено из-за схожести
    for case in similar_cases:
        case_tokens = tokenize(str(case["text"]), min_len=min_word_len)
        case_words = set(case_tokens)
        case_word_count = len(case_tokens)
        matched_words = sum(1 for word in query_words if word in case_words)
        if matched_words <= 0:
            n_skipped += 1
            continue
        similarity = matched_words / len(query_words)
        outcome = int(case["outcome"]) # результат дела
        weight = similarity + smoothing # вес дела
        contribution = outcome * weight # вклад дела в прогноз
        scored.append(
            {
                "case_id": case.get("case_id"),
                "similarity": similarity,
                "similarity_pct": round(similarity * 100),
                "matched_words": matched_words,
                "case_words": case_word_count,
                "outcome": outcome,
                "outcome_label": _outcome_label(outcome),
                "weight": weight,
                "contribution": contribution,
            }
        )

    if not scored:
        raise _fail(
            "NO_SIMILARITY",
            "Нет пересечения слов запроса ни с одним делом практики",
            n_cases=n_cases,
            query_word_count=query_word_count,
            n_skipped=n_skipped,
        )

    scored.sort(key=lambda item: item["similarity"], reverse=True) # сортировка дел по схожести

    total_contribution = sum(item["contribution"] for item in scored) # сумма вкладов дел в прогноз
    total_weight = sum(item["weight"] for item in scored) # сумма весов дел
    probability = round(total_contribution / total_weight * 100) # вероятность успеха

    _log_scored_cases(scored, query_word_count)

    logger.info(
        "predict_success probability=%s n_input=%s n_scored=%s n_skipped=%s "
        "query_word_count=%s",
        probability,
        n_cases,
        len(scored),
        n_skipped,
        query_word_count,
    )
    
    if logger.isEnabledFor(logging.DEBUG):
        for item in scored:
            logger.debug(
                "predict_detail case_id=%s similarity=%.4f outcome=%s",
                item.get("case_id"),
                item["similarity"],
                item["outcome"],
            )

    return {
        "probability": probability,
        "details": scored,
    }
