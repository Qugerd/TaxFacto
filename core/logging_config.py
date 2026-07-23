"""Общая настройка логирования TaxFacto

Управление через переменные окружения:
    TAXFACTO_LOG_LEVEL     — уровень технических логов (по умолчанию INFO).
                             Чтобы скрыть технические логи: WARNING или OFF.
    TAXFACTO_UI_LOG_LEVEL  — уровень UI-логов (по умолчанию INFO).
                             Чтобы скрыть UI-логи: WARNING или OFF.

Примеры:
    Только UI-логи:       TAXFACTO_LOG_LEVEL=OFF
    Только технические:   TAXFACTO_UI_LOG_LEVEL=OFF
    Всё:                  (не задавать, оба INFO по умолчанию)
    Ничего:               TAXFACTO_LOG_LEVEL=OFF  TAXFACTO_UI_LOG_LEVEL=OFF
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"
UI_LOG_FORMAT = "%(asctime)s %(message)s"
UI_DATE_FORMAT = "%H:%M:%S"

UI_LOGGERS = (
    "taxfacto.prediction.ui",
    "taxfacto.recommendation.ui",
)

TECH_LOGGERS = (
    "taxfacto.prediction",
    "taxfacto.predictor",
    "taxfacto.recommendation",
)


def _parse_level(raw: str) -> int | None:
    """Возвращает числовой уровень логирования или None если OFF"""
    normalized = raw.strip().upper()
    if normalized == "OFF":
        return None
    return getattr(logging, normalized, logging.INFO)


def configure_logging(level: str | None = None) -> None:
    """
    Идемпотентная настройка root-логгеров taxfacto

    Уровни задаются независимо:
        level / TAXFACTO_LOG_LEVEL    — для технических логеров
        TAXFACTO_UI_LOG_LEVEL         — для UI-логеров
    Значение OFF полностью отключает соответствующий слой.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    raw_tech = (level or os.getenv("TAXFACTO_LOG_LEVEL") or DEFAULT_LEVEL)
    raw_ui = os.getenv("TAXFACTO_UI_LOG_LEVEL") or DEFAULT_LEVEL

    tech_level = _parse_level(raw_tech)
    ui_level = _parse_level(raw_ui)

    tech_handler = logging.StreamHandler()
    tech_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    ui_handler = logging.StreamHandler()
    ui_handler.setFormatter(logging.Formatter(UI_LOG_FORMAT, datefmt=UI_DATE_FORMAT))

    for name in TECH_LOGGERS:
        lg = logging.getLogger(name)
        lg.propagate = False
        if tech_level is None:
            lg.disabled = True
        else:
            lg.setLevel(tech_level)
            if not lg.handlers:
                lg.addHandler(tech_handler)

    for name in UI_LOGGERS:
        lg = logging.getLogger(name)
        lg.propagate = False
        if ui_level is None:
            lg.disabled = True
        else:
            lg.setLevel(ui_level)
            if not lg.handlers:
                lg.addHandler(ui_handler)

    _CONFIGURED = True
