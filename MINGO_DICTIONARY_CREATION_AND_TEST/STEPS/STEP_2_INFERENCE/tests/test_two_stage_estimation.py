#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_INFERENCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_INFERENCE_DIR))

from estimate_parameters import estimate_from_dataframes


def _toy_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z_plane_1": [0.0, 0.0, 0.0],
            "z_plane_2": [65.0, 65.0, 65.0],
            "z_plane_3": [130.0, 130.0, 130.0],
            "z_plane_4": [195.0, 195.0, 195.0],
            "flux_cm2_min": [1.0, 1.8, 2.5],
            "cos_n": [2.0, 2.0, 2.0],
            "eff_sim_1": [0.90, 0.75, 0.60],
            "eff_sim_2": [0.89, 0.74, 0.59],
            "eff_sim_3": [0.88, 0.73, 0.58],
            "eff_sim_4": [0.87, 0.72, 0.57],
            "eff_empirical_1": [0.88, 0.73, 0.58],
            "eff_empirical_2": [0.87, 0.72, 0.57],
            "eff_empirical_3": [0.86, 0.71, 0.56],
            "eff_empirical_4": [0.85, 0.70, 0.55],
            "__derived_tt_global_rate_hz": [90.0, 100.0, 110.0],
            "events_per_second_0_rate_hz": [40.0, 55.0, 70.0],
            "events_per_second_1_rate_hz": [30.0, 26.0, 22.0],
            "events_per_second_2_rate_hz": [10.0, 19.0, 28.0],
        }
    )


def test_two_stage_mode_populates_stage_diagnostics_and_flux() -> None:
    dict_df = _toy_dictionary()
    data_df = pd.DataFrame(
        {
            "z_plane_1": [0.0],
            "z_plane_2": [65.0],
            "z_plane_3": [130.0],
            "z_plane_4": [195.0],
            "eff_empirical_1": [0.79],
            "eff_empirical_2": [0.78],
            "eff_empirical_3": [0.77],
            "eff_empirical_4": [0.76],
            "__derived_tt_global_rate_hz": [103.0],
            "events_per_second_0_rate_hz": [56.0],
            "events_per_second_1_rate_hz": [25.0],
            "events_per_second_2_rate_hz": [19.0],
            "flux_cm2_min": [1.8],
            "cos_n": [2.0],
            "eff_sim_1": [0.75],
            "eff_sim_2": [0.74],
            "eff_sim_3": [0.73],
            "eff_sim_4": [0.72],
        }
    )
    features = [
        "eff_empirical_1",
        "eff_empirical_2",
        "eff_empirical_3",
        "eff_empirical_4",
        "__derived_tt_global_rate_hz",
    ]

    out = estimate_from_dataframes(
        dict_df=dict_df,
        data_df=data_df,
        feature_columns=features,
        distance_metric="l2_zscore",
        interpolation_k=2,
        include_global_rate=False,
        exclude_same_file=False,
        inverse_mapping_cfg={
            "estimation_mode": "two_stage_eff_address_then_flux",
            "neighbor_selection": "knn",
            "neighbor_count": 2,
            "weighting": "inverse_distance",
            "aggregation": "weighted_mean",
            "stage2_efficiency_conditioning_weight": 1.0,
            "stage2_efficiency_gate_max": 0.20,
            "stage2_efficiency_gate_min_candidates": 1,
            "stage2_use_rate_histogram": True,
            "stage2_histogram_distance_weight": 1.0,
        },
    )

    assert "estimation_mode_used" in out.columns
    assert str(out.loc[0, "estimation_mode_used"]) == "two_stage_eff_address_then_flux"
    assert int(out.loc[0, "n_neighbors_stage1"]) > 0
    assert int(out.loc[0, "n_neighbors_stage2"]) > 0
    assert int(out.loc[0, "stage2_candidates_before_gate"]) > 0
    assert int(out.loc[0, "stage2_candidates_after_gate"]) > 0
    assert int(out.loc[0, "stage2_candidates_after_gate"]) <= int(out.loc[0, "stage2_candidates_before_gate"])
    assert int(out.loc[0, "stage2_histogram_bins_used"]) >= 3
    assert np.isfinite(float(out.loc[0, "stage1_best_distance_eff"]))
    assert np.isfinite(float(out.loc[0, "stage2_best_distance_rate"]))
    flux_est = float(out.loc[0, "est_flux_cm2_min"])
    assert np.isfinite(flux_est)
    assert 1.0 <= flux_est <= 2.5
