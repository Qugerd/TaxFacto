"""Извлечение маркеров (схем) из текста дела по словарю PATTERNS"""

from __future__ import annotations
import re
from app.patterns_data import PATTERNS

TAG_APPEND_SEP = " "
MAX_CANDIDATES = 12


def find_regex_matches(mapping: dict[str, str], text: str) -> list[str]:
    """Возвращает отсортированный список названий схем, чей regex найден в text"""
    found: list[str] = []
    text_clean = " ".join(str(text).split())
    for name, pattern in mapping.items():
        if re.search(pattern, text_clean, flags=re.IGNORECASE | re.DOTALL):
            found.append(name)
    return sorted(set(found))


def extract_case_tags(text: str) -> list[str]:
    """Извлекает маркеры схем из текста дела"""
    return find_regex_matches(PATTERNS, text)


def normalize_tags(tags: list[str]) -> list[str]:
    """Оставляет только известные имена схем из PATTERNS"""
    known = set(PATTERNS)
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        name = str(tag).strip()
        if name in known and name not in seen:
            seen.add(name)
            result.append(name)
    return sorted(result)


def resolve_case_tags(case: dict) -> list[str]:
    """
    если у дела есть непустой tags вызываем normalize_tags(tags)
    иначе вызываем extract_case_tags(text)
    """
    provided = case.get("tags")
    if provided is not None and len(provided) > 0:
        return normalize_tags(provided)
    return extract_case_tags(str(case.get("text", "")))


def resolve_case_tags_source(case: dict) -> str:
    """Источник тегов для ответа API: ``provided`` | ``extracted``."""
    provided = case.get("tags")
    if provided is not None and len(provided) > 0:
        return "provided"
    return "extracted"
