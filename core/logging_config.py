"""Общая настройка логирования TaxFacto"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False

DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(levelname)s %(name)s %(message)s"


def configure_logging(level: str | None = None) -> None:
    """
    Идемпотентная настройка root-логгеров taxfacto.
    Уровень: аргумент level, иначе env TAXFACTO_LOG_LEVEL, иначе INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    raw = (level or os.getenv("TAXFACTO_LOG_LEVEL") or DEFAULT_LEVEL).upper()
    log_level = getattr(logging, raw, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    for name in ("taxfacto.prediction", "taxfacto.predictor"):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False

    _CONFIGURED = True
