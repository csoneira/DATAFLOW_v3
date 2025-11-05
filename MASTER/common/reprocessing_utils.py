from __future__ import annotations

"""Helpers for safely retrieving values from reprocessing parameter tables."""

from typing import Any

import numpy as np
import pandas as pd


def get_reprocessing_value(
    params: pd.DataFrame | pd.Series | dict[str, object] | None, key: str
) -> Any:
    """
    Return the scalar reprocessing value associated with *key*.

    The helper accepts the raw DataFrame produced by the metadata tables (one row
    per file), as well as already-extracted Series or dict representations. It
    guards against missing keys, all-NaN columns, and NumPy containers that would
    otherwise raise “ambiguous truth value” errors when inspected in conditionals.
    """

    if params is None:
        return None

    if isinstance(params, pd.DataFrame):
        if params.empty or key not in params.columns:
            return None
        value = params.iloc[0][key]
    elif isinstance(params, pd.Series):
        if key not in params.index:
            return None
        value = params[key]
    elif isinstance(params, dict):
        value = params.get(key)
    else:
        return None

    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.squeeze()

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None

    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    else:
        if isinstance(missing, (np.ndarray, pd.Series)):
            missing = bool(np.all(missing))

    if missing:
        return None

    return value

