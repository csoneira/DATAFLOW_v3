from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from step_2_build_lut_single_z import _build_reference_curve


def test_distance_asymptote_reference_curve_uses_intercept_at_zero_distance() -> None:
    aggregated_cells = pd.DataFrame(
        {
            "emp_eff_1": [0.90, 0.85, 0.80, 0.95],
            "emp_eff_2": [0.90, 0.85, 0.80, 0.95],
            "emp_eff_3": [0.90, 0.85, 0.80, 0.95],
            "emp_eff_4": [0.90, 0.85, 0.80, 0.95],
            "flux_bin_index": [0, 0, 0, 1],
            "flux_bin_center": [1.0, 1.0, 1.0, 2.0],
            "rate_median": [10.0, 8.0, 6.0, 7.5],
            "eff_mean": [0.90, 0.85, 0.80, 0.95],
            "eff_span": [0.0, 0.0, 0.0, 0.0],
            "distance_to_perfect": [0.10, 0.20, 0.30, 0.05],
        }
    )

    reference_cells, reference_curve = _build_reference_curve(
        aggregated_cells,
        reference_curve_mode="distance_asymptote",
        top_k_closest_bins=10,
    )

    assert len(reference_cells) == 4
    flux0 = reference_curve.loc[reference_curve["flux_bin_index"] == 0].iloc[0]
    flux1 = reference_curve.loc[reference_curve["flux_bin_index"] == 1].iloc[0]

    assert flux0["reference_rate_median"] == pytest.approx(12.0)
    assert flux0["reference_fit_intercept"] == pytest.approx(12.0)
    assert flux0["reference_fit_slope"] == pytest.approx(-20.0)
    assert flux0["reference_cell_count"] == 3

    assert flux1["reference_rate_median"] == pytest.approx(7.5)
    assert flux1["reference_fit_intercept"] == pytest.approx(7.5)
    assert pd.isna(flux1["reference_fit_slope"])
    assert flux1["reference_cell_count"] == 1


def test_median_top_k_reference_curve_preserves_legacy_behavior() -> None:
    aggregated_cells = pd.DataFrame(
        {
            "emp_eff_1": [0.90, 0.85, 0.80],
            "emp_eff_2": [0.90, 0.85, 0.80],
            "emp_eff_3": [0.90, 0.85, 0.80],
            "emp_eff_4": [0.90, 0.85, 0.80],
            "flux_bin_index": [0, 0, 0],
            "flux_bin_center": [1.0, 1.0, 1.0],
            "rate_median": [10.0, 8.0, 6.0],
            "eff_mean": [0.90, 0.85, 0.80],
            "eff_span": [0.0, 0.0, 0.0],
            "distance_to_perfect": [0.10, 0.20, 0.30],
        }
    )

    _, reference_curve = _build_reference_curve(
        aggregated_cells,
        reference_curve_mode="median_top_k_closest",
        top_k_closest_bins=3,
    )

    flux0 = reference_curve.loc[reference_curve["flux_bin_index"] == 0].iloc[0]
    assert flux0["reference_rate_median"] == pytest.approx(8.0)
    assert "reference_fit_intercept" not in reference_curve.columns
