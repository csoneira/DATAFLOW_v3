#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_STEP14_DIR = (
    Path(__file__).resolve().parent.parent / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
)
sys.path.insert(0, str(_STEP14_DIR))

import ensure_continuity_dictionary as step14  # noqa: E402
from ensure_continuity_dictionary import (  # noqa: E402
    _plot_grouped_ball_convex_fallback,
    _plot_grouped_neighborhood_fallback,
)


def _grouped_flags_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flux_cm2_min": [1.0, 1.5, 2.2, 3.1, 4.5, 5.2],
            "keep_by_continuity": [True, True, True, False, True, False],
            "events_per_second_0_rate_hz": [0.62, 0.60, 0.56, 0.52, 0.48, 0.46],
            "events_per_second_1_rate_hz": [0.38, 0.40, 0.44, 0.48, 0.52, 0.54],
            "efficiency_vector_p1_x_bin_0_center_mm": [-50.0, -50.0, -50.0, -50.0, -50.0, -50.0],
            "efficiency_vector_p1_x_bin_1_center_mm": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            "efficiency_vector_p1_x_bin_0_eff": [0.81, 0.82, 0.83, 0.84, 0.85, 0.86],
            "efficiency_vector_p1_x_bin_1_eff": [0.87, 0.88, 0.89, 0.90, 0.91, 0.92],
            "efficiency_vector_p1_x_bin_0_unc": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            "efficiency_vector_p1_x_bin_1_unc": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        }
    )


def test_grouped_neighborhood_fallback_handles_1d_parameter_space(
    tmp_path: Path,
) -> None:
    step14._FIGURE_COUNTER = 0
    summary = _plot_grouped_neighborhood_fallback(
        flags=_grouped_flags_df(),
        feature_cols=[
            "events_per_second_0_rate_hz",
            "events_per_second_1_rate_hz",
        ],
        param_cols=["flux_cm2_min"],
        out_path=tmp_path / "neighborhood_grouped.png",
    )

    written = list(tmp_path.glob("1_4_*_neighborhood_grouped.png"))
    assert written
    assert summary["status"] == "grouped_feature_fallback"
    assert summary["parameter_order_axis"] == "flux_cm2_min"
    assert int(summary["rate_histogram_bins_used"]) == 2
    assert int(summary["efficiency_vector_groups_used"]) == 1
    assert summary["efficiency_vector_axes_used"] == ["x"]


def test_grouped_ball_convex_fallback_handles_1d_parameter_space(
    tmp_path: Path,
) -> None:
    step14._FIGURE_COUNTER = 0
    summary = _plot_grouped_ball_convex_fallback(
        flags=_grouped_flags_df(),
        feature_cols=[
            "events_per_second_0_rate_hz",
            "events_per_second_1_rate_hz",
        ],
        param_cols=["flux_cm2_min"],
        random_seed=1234,
        radius_fraction=0.30,
        out_path=tmp_path / "ball_convex_grouped.png",
    )

    written = list(tmp_path.glob("1_4_*_ball_convex_grouped.png"))
    assert written
    assert summary["status"] == "grouped_feature_fallback"
    assert summary["parameter_order_axis"] == "flux_cm2_min"
    assert int(summary["selected_count"]) >= 1
    assert int(summary["rate_histogram_bins_used"]) == 2
    assert int(summary["efficiency_vector_groups_used"]) == 1
