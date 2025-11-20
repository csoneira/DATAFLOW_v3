"""Utilities for selecting the newest DATAFLOW artifacts by basename."""

from __future__ import annotations

import re
from typing import Iterable, Optional

_ORDER_SUFFIXES: tuple[str, ...] = (
    ".hld.tar.gz",
    ".hld-tar-gz",
    ".tar.gz",
    ".hld",
    ".dat",
    ".root",
    ".list",
    ".lis",
    ".fit",
    ".corr",
)


def _strip_order_suffixes(name: str) -> str:
    """Remove known multi-part extensions so comparisons focus on the basename."""
    lowered = name.lower()
    for suffix in _ORDER_SUFFIXES:
        if lowered.endswith(suffix):
            return _strip_order_suffixes(name[: -len(suffix)])
    return name


def _normalize_prefix(name: str) -> str:
    """Normalize minI* prefixes back to mi01* for comparisons."""
    lowered = name.lower()
    if lowered.startswith("mini"):
        return "mi01" + lowered[4:]
    return lowered


def _station_prefix(station: str) -> str:
    try:
        station_int = int(str(station))
    except (TypeError, ValueError):
        return f"mi0{station}".lower()
    return f"mi0{station_int}".lower()


def newest_order_key(file_name: str, station: str) -> str:
    """Compute a comparison key that ignores the mi0X prefix when possible."""
    base = _strip_order_suffixes(file_name)
    normalized = _normalize_prefix(base)
    prefix = _station_prefix(station)
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    match = re.search(r"(\d{11})$", normalized)
    if match:
        return match.group(1)
    return normalized


def select_latest_candidate(files: Iterable[str], station: str) -> Optional[str]:
    """Return the lexicographically-last artifact after normalizing its prefix."""
    candidates = [name for name in files if name]
    if not candidates:
        return None
    return max(candidates, key=lambda name: newest_order_key(name, station))
