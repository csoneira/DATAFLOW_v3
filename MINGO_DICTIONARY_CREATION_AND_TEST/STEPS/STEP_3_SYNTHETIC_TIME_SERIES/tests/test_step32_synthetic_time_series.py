#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STEP32_DIR = (
    Path(__file__).resolve().parent.parent / "STEP_3_2_SYNTHETIC_TIME_SERIES"
)
sys.path.insert(0, str(_STEP32_DIR))

from synthetic_time_series import (
    _build_event_mask,
    _build_weights,
    _enforce_rate_consistency_constraints,
    _make_synthetic_dataset,
    _rebuild_weighted_step12_helper_columns,
    _resolve_basis_source,
    _resolve_step31_curve_data_mode,
    _weighted_numeric_columns,
)


def test_closest_point_mode_copies_nearest_basis_row() -> None:
    dictionary_df = pd.DataFrame(
        {
            "flux_cm2_min": [1.0, 2.0],
            "eff_sim_1": [0.70, 0.90],
            "events_per_second_global_rate": [11.0, 22.0],
            "feature_linear": [100.0, 200.0],
            "n_events": [1000, 2000],
            "count_rate_denominator_seconds": [100.0, 100.0],
        }
    )
    template_df = dictionary_df.copy()
    time_df = pd.DataFrame(
        {
            "flux_cm2_min": [1.8],
            "eff_sim_1": [0.86],
            "duration_seconds": [60.0],
            "n_events": [900],
        }
    )

    basis_params = dictionary_df[["flux_cm2_min", "eff_sim_1"]].to_numpy(dtype=float)
    target_params = time_df[["flux_cm2_min", "eff_sim_1"]].to_numpy(dtype=float)
    weights = _build_weights(
        dict_param_matrix=basis_params,
        target_param_matrix=target_params,
        method="closest_point",
        top_k=None,
        distance_hardness=2.0,
        event_mask=np.ones((1, len(dictionary_df)), dtype=bool),
    )

    out, dominant_idx = _make_synthetic_dataset(
        dictionary_df=dictionary_df,
        template_df=template_df,
        time_df=time_df,
        weights=weights,
        basis_param_matrix=basis_params,
        target_param_matrix=target_params,
        feature_generation_mode="closest_point",
        flux_col="flux_cm2_min",
        eff_col="eff_sim_1",
        time_rate_col=None,
        time_events_col="n_events",
        time_duration_col="duration_seconds",
        interpolation_aggregation="local_linear",
    )

    assert int(dominant_idx[0]) == 1
    assert float(out.loc[0, "feature_linear"]) == 200.0
    assert float(out.loc[0, "events_per_second_global_rate"]) == 22.0
    assert float(out.loc[0, "flux_cm2_min"]) == 2.0
    assert float(out.loc[0, "eff_sim_1"]) == 0.90
    assert int(out.loc[0, "n_events"]) == 2000
    assert str(out.loc[0, "step32_feature_generation_mode"]) == "closest_point"


def test_local_linear_weighting_applies_to_non_efficiency_features() -> None:
    dict_df = pd.DataFrame(
        {
            "feature_linear": [10.0, 12.0, 14.0],
        }
    )
    basis_params = np.array([[0.0], [1.0], [2.0]], dtype=float)
    target_params = np.array([[1.5]], dtype=float)
    weights = _build_weights(
        dict_param_matrix=basis_params,
        target_param_matrix=target_params,
        method="inverse_distance",
        top_k=None,
        distance_hardness=2.0,
    )

    out = _weighted_numeric_columns(
        weights=weights,
        dict_df=dict_df,
        columns=["feature_linear"],
        interpolation_aggregation="local_linear",
        basis_param_matrix=basis_params,
        target_param_matrix=target_params,
    )

    assert np.isclose(out["feature_linear"][0], 13.0, atol=1e-9)


def test_rate_consistency_clips_negative_rate_like_columns_even_without_blend() -> None:
    out_df = pd.DataFrame(
        {
            "post_tt_12_rate_hz": [-1.0, 3.0],
            "events_per_second_0_rate_hz": [-5.0, 2.0],
            "events_per_second_1_rate_hz": [1.0, -2.0],
        }
    )

    info = _enforce_rate_consistency_constraints(
        out_df,
        target_rate=np.array([1.0, 1.0], dtype=float),
        tt_rate_columns=["post_tt_12_rate_hz"],
        histogram_rate_columns=[
            "events_per_second_0_rate_hz",
            "events_per_second_1_rate_hz",
        ],
    )

    assert info["tt_negative_entries_clipped"] == 1
    assert info["hist_negative_entries_clipped"] == 2
    assert float(out_df.loc[0, "post_tt_12_rate_hz"]) == 0.0
    assert float(out_df.loc[0, "events_per_second_0_rate_hz"]) == 0.0
    assert float(out_df.loc[1, "events_per_second_1_rate_hz"]) == 0.0


def test_event_filter_can_pad_with_parameter_nearest_rows() -> None:
    basis_events = np.array([100.0, 1000.0, 1100.0], dtype=float)
    target_events = np.array([100.0], dtype=float)
    basis_params = np.array([[0.0], [1.0], [2.0]], dtype=float)
    target_params = np.array([[1.8]], dtype=float)

    mask, info = _build_event_mask(
        basis_events=basis_events,
        target_events=target_events,
        tolerance_pct=10.0,
        min_rows=2,
        filter_mode="tolerance_then_param_nearest",
        basis_param_matrix=basis_params,
        target_param_matrix=target_params,
    )

    assert mask is not None
    assert mask.shape == (1, 3)
    assert bool(mask[0, 0])
    assert bool(mask[0, 2])
    assert int(mask[0].sum()) == 2
    assert info["param_nearest_padding_points_count"] == 1


def test_basis_source_auto_uses_dataset_for_closest_point_and_dictionary_for_weighted() -> None:
    resolved_closest, auto_closest = _resolve_basis_source(
        "auto",
        weighting_method="closest_point",
    )
    resolved_weighted, auto_weighted = _resolve_basis_source(
        "auto",
        weighting_method="inverse_distance",
    )

    assert auto_closest is True
    assert auto_weighted is True
    assert resolved_closest == "dataset"
    assert resolved_weighted == "dictionary"


def test_step31_curve_data_mode_is_read_from_time_series() -> None:
    mode, explicit = _resolve_step31_curve_data_mode(
        pd.DataFrame({"step31_curve_data_mode": ["dataset_data_curve", "dataset_data_curve"]})
    )

    assert explicit is True
    assert mode == "dataset_data_curve"


def test_dataset_data_curve_mode_copies_exact_basis_row() -> None:
    dictionary_df = pd.DataFrame(
        {
            "flux_cm2_min": [1.0, 2.0],
            "eff_sim_1": [0.70, 0.90],
            "events_per_second_global_rate": [11.0, 22.0],
            "feature_linear": [100.0, 200.0],
            "n_events": [1000, 2000],
            "count_rate_denominator_seconds": [100.0, 100.0],
        }
    )
    template_df = dictionary_df.copy()
    time_df = pd.DataFrame(
        {
            "flux_cm2_min": [2.0],
            "eff_sim_1": [0.90],
            "duration_seconds": [60.0],
            "n_events": [2000],
            "step31_curve_data_mode": ["dataset_data_curve"],
        }
    )
    basis_params = dictionary_df[["flux_cm2_min", "eff_sim_1"]].to_numpy(dtype=float)
    target_params = time_df[["flux_cm2_min", "eff_sim_1"]].to_numpy(dtype=float)
    weights = np.array([[0.0, 1.0]], dtype=float)

    out, dominant_idx = _make_synthetic_dataset(
        dictionary_df=dictionary_df,
        template_df=template_df,
        time_df=time_df,
        weights=weights,
        basis_param_matrix=basis_params,
        target_param_matrix=target_params,
        feature_generation_mode="dataset_data_curve",
        flux_col="flux_cm2_min",
        eff_col="eff_sim_1",
        time_rate_col=None,
        time_events_col="n_events",
        time_duration_col="duration_seconds",
        interpolation_aggregation="local_linear",
    )

    assert int(dominant_idx[0]) == 1
    assert float(out.loc[0, "feature_linear"]) == 200.0
    assert float(out.loc[0, "events_per_second_global_rate"]) == 22.0
    assert float(out.loc[0, "flux_cm2_min"]) == 2.0
    assert float(out.loc[0, "eff_sim_1"]) == 0.90
    assert str(out.loc[0, "step32_feature_generation_mode"]) == "dataset_data_curve"


def test_weighted_mode_rebuilds_step12_helper_columns_from_final_targets() -> None:
    dictionary_df = pd.DataFrame(
        {
            "flux_cm2_min": [1.0, 2.0],
            "eff_sim_1": [0.70, 0.90],
            "eff_sim_2": [0.60, 0.80],
            "eff_sim_3": [0.50, 0.70],
            "eff_sim_4": [0.40, 0.60],
            "eff_p1": [0.70, 0.90],
            "eff_p2": [0.60, 0.80],
            "eff_p3": [0.50, 0.70],
            "eff_p4": [0.40, 0.60],
            "events_per_second_global_rate": [10.0, 20.0],
            "efficiency_product_4planes": [0.084, 0.3024],
            "efficiency_product_123": [0.21, 0.504],
            "efficiency_product_234": [0.12, 0.336],
            "efficiency_product_12": [0.42, 0.72],
            "efficiency_product_34": [0.20, 0.42],
            "flux_proxy_rate_div_effprod": [119.0476190476, 66.1375661376],
            "flux_proxy_rate_div_effprod_123": [47.6190476190, 39.6825396825],
            "flux_proxy_rate_div_effprod_234": [83.3333333333, 59.5238095238],
            "flux_proxy_rate_div_effprod_12": [23.8095238095, 27.7777777778],
            "flux_proxy_rate_div_effprod_34": [50.0, 47.6190476190],
            "n_events": [1000, 2000],
            "count_rate_denominator_seconds": [100.0, 100.0],
        }
    )
    template_df = dictionary_df.copy()
    time_df = pd.DataFrame(
        {
            "flux_cm2_min": [1.3],
            "eff_sim_1": [0.82],
            "eff_sim_2": [0.73],
            "eff_sim_3": [0.64],
            "eff_sim_4": [0.55],
            "duration_seconds": [60.0],
            "n_events": [900],
        }
    )

    basis_params = dictionary_df[
        ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]
    ].to_numpy(dtype=float)
    target_params = time_df[
        ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]
    ].to_numpy(dtype=float)
    weights = _build_weights(
        dict_param_matrix=basis_params,
        target_param_matrix=target_params,
        method="inverse_distance",
        top_k=None,
        distance_hardness=2.0,
    )

    out, _ = _make_synthetic_dataset(
        dictionary_df=dictionary_df,
        template_df=template_df,
        time_df=time_df,
        weights=weights,
        basis_param_matrix=basis_params,
        target_param_matrix=target_params,
        feature_generation_mode="weighted_interpolation",
        flux_col="flux_cm2_min",
        eff_col="eff_sim_1",
        time_rate_col=None,
        time_events_col="n_events",
        time_duration_col="duration_seconds",
        interpolation_aggregation="local_linear",
    )

    for col in ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]:
        out[col] = pd.to_numeric(time_df[col], errors="coerce").to_numpy(dtype=float)
    info = _rebuild_weighted_step12_helper_columns(out)

    assert info["applied"] is True
    assert np.isclose(float(out.loc[0, "eff_p1"]), 0.82)
    assert np.isclose(float(out.loc[0, "eff_p2"]), 0.73)
    assert np.isclose(float(out.loc[0, "eff_p3"]), 0.64)
    assert np.isclose(float(out.loc[0, "eff_p4"]), 0.55)

    prod4 = 0.82 * 0.73 * 0.64 * 0.55
    prod123 = 0.82 * 0.73 * 0.64
    prod234 = 0.73 * 0.64 * 0.55
    prod12 = 0.82 * 0.73
    prod34 = 0.64 * 0.55
    rate = float(out.loc[0, "events_per_second_global_rate"])
    assert np.isclose(float(out.loc[0, "efficiency_product_4planes"]), prod4)
    assert np.isclose(float(out.loc[0, "efficiency_product_123"]), prod123)
    assert np.isclose(float(out.loc[0, "efficiency_product_234"]), prod234)
    assert np.isclose(float(out.loc[0, "efficiency_product_12"]), prod12)
    assert np.isclose(float(out.loc[0, "efficiency_product_34"]), prod34)
    assert np.isclose(float(out.loc[0, "flux_proxy_rate_div_effprod"]), rate / prod4)
    assert np.isclose(float(out.loc[0, "flux_proxy_rate_div_effprod_123"]), rate / prod123)
    assert np.isclose(float(out.loc[0, "flux_proxy_rate_div_effprod_234"]), rate / prod234)
    assert np.isclose(float(out.loc[0, "flux_proxy_rate_div_effprod_12"]), rate / prod12)
    assert np.isclose(float(out.loc[0, "flux_proxy_rate_div_effprod_34"]), rate / prod34)
