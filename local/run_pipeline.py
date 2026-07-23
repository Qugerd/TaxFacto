"""
Сквозной пайплайн Задача 2 → Задача 3.

Запуск из корня репозитория:
    python local/run_pipeline.py

Режимы (переменные ниже):
  USE_DEMO = True  — демонстрационный кейс с рекомендациями
  USE_DEMO = False — реальные text_test.csv / text_reference.csv
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path


USE_DEMO = False




SAMPLE_SIZE = 3

TOP_N = 15


os.environ.setdefault("TAXFACTO_LOG_LEVEL", "OFF")
os.environ.setdefault("TAXFACTO_UI_LOG_LEVEL", "INFO")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "local" / "ML"
REC_ROOT = REPO_ROOT / "services" / "recommendation"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REC_ROOT))

import logging

from core.logging_config import configure_logging
from core.predictor import PredictError, compute_similarity, predict_success, tokenize
from app.recommender import recommend_markers

configure_logging()
pred_ui = logging.getLogger("taxfacto.prediction.ui")
rec_ui = logging.getLogger("taxfacto.recommendation.ui")


def _banner(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def demo_case() -> tuple[str, list[dict]]:
    """
    Демо: TOP_N релевантных дел с UUID.
    Много отказов с высокой схожестью → ниже база → несколько рекомендаций.
    """
    query_text = (
        "налоговая проверка документов компании и требований кредиторов"
    )
    base = "налоговая проверка документов компании требований кредиторов"

    # Выигрыши с маркерами (после добавления тега в запрос вероятность растёт)
    wins: list[tuple[str, list[str]]] = [
        ("вывод активов имущества без оснований сокрытие средств", ["Вывод активов"]),
        ("номинальный директор формальный статус не управлял", ["Номинальное руководство"]),
        ("аффилированность родственные связи заинтересованное лицо", ["Аффилированность / Родство"]),
        ("нерыночные условия заниженная стоимость отклонение рынка", ["Неравноценные сделки"]),
        ("перевод бизнеса зеркальная компания миграция активов", ["Перевод бизнеса / Зеркало"]),
        ("фиктивный договор мнимость сделки безденежность", ["Мнимые / Фиктивные сделки"]),
    ]

    similar_cases: list[dict] = []
    for extra, tags in wins:
        similar_cases.append(
            {
                "case_id": str(uuid.uuid4()),
                "outcome": 1,
                "text": f"{base} {extra}",
                "tags": tags,
            }
        )

    # Отказы: те же слова запроса → высокая схожесть, тянут вероятность вниз
    n_refusals = max(TOP_N - len(similar_cases), 0)
    for i in range(n_refusals):
        similar_cases.append(
            {
                "case_id": str(uuid.uuid4()),
                "outcome": 0,
                "text": (
                    f"{base} отказ в удовлетворении требований "
                    f"кредиторов компании номер {i + 1}"
                ),
                "tags": [],
            }
        )

    return query_text, similar_cases[:TOP_N]


def find_similar_cases(
    query_text: str,
    reference_rows: list[dict],
    top_n: int = TOP_N,
) -> list[dict]:
    query_words = tokenize(query_text)
    scored: list[dict] = []
    for row in reference_rows:
        sim = compute_similarity(query_words, row["_words"])
        if sim <= 0:
            continue
        scored.append(
            {
                "case_id": row["case_id"],
                "text": row["text"],
                "outcome": row["outcome"],
                "similarity": sim,
            }
        )
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:top_n]


def load_reference_rows() -> list[dict]:
    import pandas as pd

    path = ML_DIR / "text_reference.csv"
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, row in df.iterrows():
        text = str(row["document_text"])
        rows.append(
            {
                "case_id": str(row["card_id"]),
                "text": text,
                "outcome": 1 if row["result_code"] == "satisfied" else 0,
                "_words": set(tokenize(text)),
            }
        )
    return rows


def load_test_queries(sample_size: int) -> list[dict]:
    import pandas as pd

    path = ML_DIR / "text_test.csv"
    df = pd.read_csv(path)
    total = len(df)
    if sample_size < 1:
        raise SystemExit("Количество дел должно быть >= 1")
    if sample_size > total:
        sample_size = total

    df = df.head(sample_size)
    return [
        {
            "case_id": str(row["card_id"]),
            "query_text": str(row["document_text"]),
        }
        for _, row in df.iterrows()
    ]


def run_predict_stage(req_id: str, query_text: str, similar_cases: list[dict]) -> dict:
    query_word_count = len(tokenize(query_text))
    query_len = len(query_text)

    _banner("\n1. Расчёт вероятности удовлетворения запроса\n")
    pred_ui.info("[%s] Загрузка дела", req_id)
    pred_ui.info("[%s] Обработка текста запроса..", req_id)
    pred_ui.info("[%s] Запрос: %s слов, %s символов", req_id, query_word_count, query_len)

    result = predict_success(query_text=query_text, similar_cases=similar_cases)

    details = result.get("details", [])
    n_scored = len(details)
    n_success = sum(1 for d in details if d["outcome"] == 1)
    n_refusal = n_scored - n_success
    scored_summary = ", ".join(
        f"{d.get('case_id') or 'unknown'}: {d['similarity_pct']}%"
        for d in details
    )

    pred_ui.info(
        "[%s] Загружено релевантных дел: %s (удовлетворённых: %s, отказов: %s)",
        req_id,
        n_scored,
        n_success,
        n_refusal,
    )
    pred_ui.info("[%s] Расчёт схожести..", req_id)
    pred_ui.info("[%s] Схожесть по делам: [%s]", req_id, scored_summary)
    pred_ui.info(
        "[%s] Вероятность удовлетворения запроса: %s%%",
        req_id,
        result["probability"],
    )
    return result


def run_recommend_stage(
    req_id: str,
    query_text: str,
    similar_cases: list[dict],
    prediction: dict,
) -> dict:
    n_cases = len(similar_cases)
    n_success = sum(1 for c in similar_cases if int(c["outcome"]) == 1)
    n_refusal = n_cases - n_success
    baseline = int(prediction["probability"])

    _banner("\n2. Формирование рекомендаций по увеличению вероятности удовлетворения запроса\n")
    rec_ui.info("[%s] Загрузка запроса", req_id)
    rec_ui.info("[%s] Вероятность удовлетворения запроса: %s%%", req_id, baseline)
    rec_ui.info(
        "[%s] Релевантных дел: %s (удовлетворённых: %s, отказов: %s)",
        req_id,
        n_cases,
        n_success,
        n_refusal,
    )

    result = recommend_markers(
        query_text=query_text,
        similar_cases=similar_cases,
        prediction={
            "probability": baseline,
            "details": prediction.get("details"),
        },
    )

    recommendations = result.get("recommendations", [])
    rec_summary = ", ".join(
        f"{item['tag']}: +{item['gain']}%"
        for item in recommendations
    ) or "-"

    rec_ui.info("[%s] Поиск рекомендаций..", req_id)
    rec_ui.info("[%s] Найдено рекомендаций: %s", req_id, len(recommendations))
    rec_ui.info("[%s] Рекомендации: [%s]", req_id, rec_summary)
    return result


def run_one(query_text: str, similar_cases: list[dict]) -> None:
    req_id = str(uuid.uuid4())

    try:
        prediction = run_predict_stage(req_id, query_text, similar_cases)
    except PredictError as exc:
        pred_ui.info("[%s] Ошибка: %s - %s", req_id, exc.code, exc)
        return

    try:
        run_recommend_stage(req_id, query_text, similar_cases, prediction)
    except PredictError as exc:
        rec_ui.info("[%s] Ошибка: %s - %s", req_id, exc.code, exc)


def main() -> None:
    if USE_DEMO:
        _banner("Режим: DEMO (15 релевантных дел, UUID сгенерированы)")
        query_text, similar_cases = demo_case()
        _banner(f"Релевантных дел: {len(similar_cases)}")
        run_one(query_text, similar_cases)
        return

    _banner(f"Загрузка reference: {ML_DIR / 'text_reference.csv'}")
    reference = load_reference_rows()
    _banner(f"Загрузка test: {ML_DIR / 'text_test.csv'}")
    queries = load_test_queries(SAMPLE_SIZE)
    _banner(f"Тестовых дел: {len(queries)}")

    for item in queries:
        similar = find_similar_cases(item["query_text"], reference, top_n=TOP_N)
        if not similar:
            continue
        run_one(item["query_text"], similar)


if __name__ == "__main__":
    main()
