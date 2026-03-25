#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_STEP42_DIR = (
    Path(__file__).resolve().parent.parent / "STEP_4_2_ANALYZE"
)
sys.path.insert(0, str(_STEP42_DIR))

from analyze import (  # noqa: E402
    _mask_sim_eff_within_tolerance_band,
    _plot_distance_term_dominance,
    _plot_estimated_curve_flux_vs_eff,
    _plot_parameter_estimate_series,
    _resolve_estimation_parameter_columns,
)


def test_resolve_estimation_parameter_columns_uses_default_parameter_space_order() -> None:
    dictionary_df = pd.DataFrame(columns=["flux_only", "cos_n", "eff_sim_3"])

    resolved = _resolve_estimation_parameter_columns(
        dictionary_df=dictionary_df,
        configured_columns=None,
        default_columns=["flux_only", "missing_col", "cos_n"],
    )

    assert resolved == ["flux_only", "cos_n"]


def test_mask_sim_eff_within_tolerance_band_uses_available_eff_columns() -> None:
    df = pd.DataFrame(
        {
            "eff_sim_1": [0.80, 0.80],
            "eff_sim_2": [0.81, 0.95],
            "eff_sim_3": [0.79, 0.96],
        }
    )

    mask = _mask_sim_eff_within_tolerance_band(df, tolerance_pct=5.0)

    assert mask.tolist() == [True, False]


def test_plot_estimated_curve_flux_vs_eff_supports_generic_parameter_columns(
    tmp_path: Path,
) -> None:
    real_df = pd.DataFrame(
        {
            "est_flux_only": [1.0, 1.5, 2.0],
            "est_cos_n": [0.2, 0.4, 0.6],
        }
    )
    dict_df = pd.DataFrame(
        {
            "flux_only": [0.8, 1.4, 2.2],
            "cos_n": [0.1, 0.5, 0.7],
        }
    )
    out_path = tmp_path / "parameter_space_curve.png"

    n_real, n_dict = _plot_estimated_curve_flux_vs_eff(
        real_df=real_df,
        dict_df=dict_df,
        parameter_columns=["flux_only", "cos_n"],
        out_path=out_path,
    )

    assert n_real == 3
    assert n_dict == 3
    assert out_path.exists()


def test_plot_parameter_estimate_series_supports_generic_parameter_columns(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "est_flux_only": [1.0, 1.4, 1.8],
            "unc_flux_only_abs": [0.1, 0.1, 0.2],
            "est_cos_n": [2.0, 2.1, 1.9],
            "unc_cos_n_abs": [0.05, 0.04, 0.05],
        }
    )
    out_path = tmp_path / "parameter_series.png"

    n_panels = _plot_parameter_estimate_series(
        x=pd.Series([0.0, 1.0, 2.0]),
        has_time_axis=False,
        xlabel="Row index",
        df=df,
        parameter_columns=["flux_only", "cos_n"],
        out_path=out_path,
    )

    assert n_panels == 2
    assert out_path.exists()


def test_plot_distance_term_dominance_supports_dynamic_terms_and_scalar_features(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "best_distance": [1.0, 1.2, 0.8],
            "best_distance_term_shares_json": [
                "{\"rate_histogram\": 0.7, \"efficiency_vectors\": 0.3}",
                "{\"scalar_base\": 0.2, \"rate_histogram\": 0.5, \"efficiency_vectors\": 0.3}",
                "{\"scalar_base\": 0.6, \"efficiency_vectors\": 0.4}",
            ],
            "best_distance_scalar_feature_shares_json": [
                None,
                "{\"post_tt_123_rate_hz\": 0.12, \"eff_empirical_2\": 0.08}",
                "{\"post_tt_123_rate_hz\": 0.35, \"eff_empirical_2\": 0.25}",
            ],
            "best_distance_group_component_shares_json": [
                "{\"rate_histogram::events_per_second_0_rate_hz\": 0.45, \"rate_histogram::events_per_second_1_rate_hz\": 0.25, \"efficiency_vectors::p1_x\": 0.30}",
                "{\"scalar_base\": 0.0, \"rate_histogram::events_per_second_0_rate_hz\": 0.20, \"efficiency_vectors::p1_y\": 0.30}",
                "{\"efficiency_vectors::p1_x\": 0.10, \"efficiency_vectors::p2_theta\": 0.30}",
            ],
            "best_distance_group_component_within_term_shares_json": [
                "{\"rate_histogram::events_per_second_0_rate_hz\": 0.64, \"rate_histogram::events_per_second_1_rate_hz\": 0.36, \"efficiency_vectors::p1_x\": 1.0}",
                "{\"rate_histogram::events_per_second_0_rate_hz\": 1.0, \"efficiency_vectors::p1_y\": 1.0}",
                "{\"efficiency_vectors::p1_x\": 0.25, \"efficiency_vectors::p2_theta\": 0.75}",
            ],
        }
    )
    out_path = tmp_path / "distance_term_dominance.png"

    summary = _plot_distance_term_dominance(
        x=pd.Series([0.0, 1.0, 2.0]),
        has_time_axis=False,
        xlabel="Row index",
        df=df,
        out_path=out_path,
    )

    assert out_path.exists()
    assert summary["plot_available"] is True
    assert int(summary["n_rows"]) == 3
    assert "rate_histogram" in summary["term_median_share"]
    assert "efficiency_vectors" in summary["term_dominant_fraction"]
    assert "post_tt_123_rate_hz" in summary["top_scalar_feature_median_share"]
    assert "rate_histogram::events_per_second_0_rate_hz" in summary["top_group_component_median_total_share"]
    assert "efficiency_vectors::p2_theta" in summary["top_group_component_median_within_term_share"]
