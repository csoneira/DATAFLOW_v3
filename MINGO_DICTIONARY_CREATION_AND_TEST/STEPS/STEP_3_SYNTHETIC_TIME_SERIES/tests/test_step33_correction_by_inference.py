#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_STEP33_DIR = (
    Path(__file__).resolve().parent.parent / "STEP_3_3_CORRECTION"
)
sys.path.insert(0, str(_STEP33_DIR))

from correction_by_inference import _resolve_tt_rate_breakdown_entries


def test_resolve_tt_rate_breakdown_entries_prefers_tt_specific_rate_columns() -> None:
    df = pd.DataFrame(
        {
            "post_tt_123_rate_hz": [1.1, 1.0],
            "efficiency_product_123": [0.8, 0.7],
            "events_per_second_global_rate": [9.0, 8.5],
        }
    )

    entries = _resolve_tt_rate_breakdown_entries(df)
    entry_by_label = {label: (rate_col, effprod_col, rate_source) for label, rate_col, effprod_col, _, rate_source in entries}

    assert entry_by_label["123"] == ("post_tt_123_rate_hz", "efficiency_product_123", "tt_specific")


def test_resolve_tt_rate_breakdown_entries_falls_back_to_shared_global_rate() -> None:
    df = pd.DataFrame(
        {
            "events_per_second_global_rate": [9.0, 8.5],
            "efficiency_product_123": [0.8, 0.7],
            "efficiency_product_34": [0.6, 0.5],
        }
    )

    entries = _resolve_tt_rate_breakdown_entries(df)
    entry_by_label = {label: (rate_col, effprod_col, rate_source) for label, rate_col, effprod_col, _, rate_source in entries}

    assert entry_by_label["123"] == ("events_per_second_global_rate", "efficiency_product_123", "shared_global")
    assert entry_by_label["34"] == ("events_per_second_global_rate", "efficiency_product_34", "shared_global")
