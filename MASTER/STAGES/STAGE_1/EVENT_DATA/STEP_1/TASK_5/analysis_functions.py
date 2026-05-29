from __future__ import annotations

import re
from ast import literal_eval
from typing import List

import numpy as np
import pandas as pd


def _charge_series_is_usable(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return False
    return bool((numeric > 0).any())


def _coerce_numeric_sequence(raw_value, caster):
    """Return a list of numbers parsed from *raw_value*."""
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        result: List[float] = []
        for item in raw_value:
            result.extend(_coerce_numeric_sequence(item, caster))
        return result
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if not cleaned:
            return []
        try:
            parsed = literal_eval(cleaned)
        except (ValueError, SyntaxError):
            cleaned = cleaned.replace("[", " ").replace("]", " ")
            tokens = [tok for tok in re.split(r"[;,\s]+", cleaned) if tok]
            result = []
            for tok in tokens:
                try:
                    result.append(caster(tok))
                except (ValueError, TypeError):
                    continue
            return result
        else:
            return _coerce_numeric_sequence(parsed, caster)
    if np.isscalar(raw_value):
        try:
            return [caster(raw_value)]
        except (ValueError, TypeError):
            return []
    return []


def _task5_parse_optional_top_n(raw_value: object, default: int | None) -> int | None:
    if raw_value in (None, "", "null", "None"):
        return None
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return default
