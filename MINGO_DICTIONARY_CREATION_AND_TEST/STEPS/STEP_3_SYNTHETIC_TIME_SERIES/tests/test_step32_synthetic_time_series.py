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
    _build_weights,
    _enforce_rate_consistency_constraints,
    _make_synthetic_dataset,
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
