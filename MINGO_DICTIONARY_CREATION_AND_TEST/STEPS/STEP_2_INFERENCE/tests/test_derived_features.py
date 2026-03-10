#!/usr/bin/env python3
"""
Regression tests for the derived-feature inverse-mapping fix.

These tests verify that using derived features (empirical efficiencies +
global rate) instead of raw TT rates breaks the flux-efficiency degeneracy
and produces meaningful flux estimates (R² > 0).

The root cause: raw TT rate features are all proportional to
flux × f(efficiency), so efficiency variation dominates the feature space
and flux cannot be discriminated by nearest-neighbor lookup.  Derived
features (eff_empirical_i, global_rate) separate these two signals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the parent estimation module is importable.
_INFERENCE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_INFERENCE_DIR))

from estimate_parameters import (
    DERIVED_EFF_PRODUCT_COL,
    DERIVED_EFFICIENCY_COLUMNS,
    DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL,
    DERIVED_RATE_COLUMN,
    DERIVED_TT_GLOBAL_RATE_COL,
    _append_derived_physics_feature_columns,
    _append_derived_tt_global_rate_column,
    _auto_feature_columns,
    _derived_feature_columns,
    _normalize_derived_physics_features,
    estimate_from_dataframes,
)

# ── Data paths ──────────────────────────────────────────────────────────
_STEPS_DIR = _INFERENCE_DIR.parent
_DICT_PATH = (
    _STEPS_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)
_DATASET_PATH = (
    _STEPS_DIR / "STEP_1_SETUP" / "STEP_1_3_ENLARGE_DATASET"
    / "OUTPUTS" / "FILES" / "enlarged_dataset.csv"
)

_DATA_AVAILABLE = _DICT_PATH.exists() and _DATASET_PATH.exists()
_SKIP_MSG = "Dictionary/dataset CSVs not available"


# ── Helpers ─────────────────────────────────────────────────────────────

def _r2(true: np.ndarray, est: np.ndarray) -> float:
    mask = np.isfinite(true) & np.isfinite(est)
    t, e = true[mask], est[mask]
    ss_res = np.sum((t - e) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


@pytest.fixture(scope="module")
def data():
    if not _DATA_AVAILABLE:
        pytest.skip(_SKIP_MSG)
    dict_df = pd.read_csv(_DICT_PATH, low_memory=False)
    data_df = pd.read_csv(_DATASET_PATH, low_memory=False)
    return dict_df, data_df


# ── Unit tests ──────────────────────────────────────────────────────────

class TestDerivedFeatureColumns:
    """Tests for _derived_feature_columns selection."""

    def test_returns_expected_columns(self, data):
        dict_df, _ = data
        cols = _derived_feature_columns(dict_df)
        assert set(DERIVED_EFFICIENCY_COLUMNS).issubset(set(cols))
        assert DERIVED_RATE_COLUMN in cols

    def test_length(self, data):
        dict_df, _ = data
        cols = _derived_feature_columns(dict_df)
        assert len(cols) == 5

    def test_columns_are_numeric(self, data):
        dict_df, _ = data
        cols = _derived_feature_columns(dict_df)
        for c in cols:
            vals = pd.to_numeric(dict_df[c], errors="coerce")
            assert vals.notna().sum() > 0, f"Column {c} has no numeric values"

    def test_tt_sum_global_rate_column_is_constructed(self, data):
        dict_df, data_df = data
        d2, x2, derived_col, sources = _append_derived_tt_global_rate_column(dict_df, data_df)
        assert derived_col == DERIVED_TT_GLOBAL_RATE_COL
        assert len(sources) > 0
        assert derived_col in d2.columns
        assert derived_col in x2.columns
        expected = d2[sources].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        assert np.allclose(
            pd.to_numeric(d2[derived_col], errors="coerce").values,
            pd.to_numeric(expected, errors="coerce").values,
            equal_nan=True,
        )

    def test_physics_feature_alias_normalization(self):
        feats = _normalize_derived_physics_features(
            ["log_tt_rate_over_eff_product", "eff_emp_product", "unknown"]
        )
        assert feats == ["log_rate_over_eff_product", "eff_product"]

    def test_physics_features_are_constructed(self, data):
        dict_df, data_df = data
        d2, x2, derived_col, _ = _append_derived_tt_global_rate_column(dict_df, data_df)
        assert derived_col == DERIVED_TT_GLOBAL_RATE_COL
        d3, x3, added = _append_derived_physics_feature_columns(
            dict_df=d2,
            data_df=x2,
            rate_column=derived_col,
            physics_features=["log_rate_over_eff_product", "eff_product"],
        )
        assert DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL in added
        assert DERIVED_EFF_PRODUCT_COL in added
        assert DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL in d3.columns
        assert DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL in x3.columns
        assert DERIVED_EFF_PRODUCT_COL in d3.columns
        assert DERIVED_EFF_PRODUCT_COL in x3.columns
        assert (
            pd.to_numeric(d3[DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL], errors="coerce")
            .notna()
            .sum()
            > 0
        )

    def test_derived_feature_columns_include_physics_columns(self, data):
        dict_df, data_df = data
        d2, x2, derived_col, _ = _append_derived_tt_global_rate_column(dict_df, data_df)
        d3, x3, added = _append_derived_physics_feature_columns(
            dict_df=d2,
            data_df=x2,
            rate_column=derived_col,
            physics_features=True,
        )
        cols_d = _derived_feature_columns(
            d3,
            rate_column=derived_col,
            physics_feature_columns=added,
        )
        cols_x = _derived_feature_columns(
            x3,
            rate_column=derived_col,
            physics_feature_columns=added,
        )
        assert DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL in cols_d
        assert DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL in cols_x


class TestDerivedModeEstimation:
    """Integration tests: derived mode produces valid estimates."""

    @pytest.fixture(scope="class")
    def result(self, data):
        dict_df, data_df = data
        return estimate_from_dataframes(
            dict_df=dict_df,
            data_df=data_df,
            feature_columns="derived",
            distance_metric="l2_zscore",
            include_global_rate=False,
            exclude_same_file=True,
        ), data_df

    def test_result_shape(self, result):
        res, data_df = result
        assert len(res) == len(data_df)

    def test_eff_empirical_not_estimated(self, result):
        """eff_empirical_* must not appear as estimated parameters when used as features."""
        res, _ = result
        est_cols = [c for c in res.columns if c.startswith("est_eff_empirical")]
        assert est_cols == [], (
            f"eff_empirical columns should be excluded from estimation "
            f"when used as features, but found: {est_cols}"
        )

    def test_flux_r2_positive(self, result):
        """Flux R² must be > 0 with derived features (was negative with raw features)."""
        res, data_df = result
        true = data_df["flux_cm2_min"].values[: len(res)]
        est = res["est_flux_cm2_min"].values
        r2 = _r2(true, est)
        assert r2 > 0.0, (
            f"Flux R² = {r2:.4f} is not positive. "
            "Derived features should break the flux-efficiency degeneracy."
        )

    def test_flux_r2_above_threshold(self, result):
        """Flux R² should be at least 0.25 with the current dataset."""
        res, data_df = result
        true = data_df["flux_cm2_min"].values[: len(res)]
        est = res["est_flux_cm2_min"].values
        r2 = _r2(true, est)
        assert r2 > 0.25, f"Flux R² = {r2:.4f} below 0.25 threshold."

    def test_efficiency_r2_high(self, result):
        """All efficiency R² values should exceed 0.85."""
        res, data_df = result
        for i in range(1, 5):
            col = f"eff_sim_{i}"
            if col not in data_df.columns:
                continue
            true = data_df[col].values[: len(res)]
            est = res[f"est_{col}"].values
            r2 = _r2(true, est)
            assert r2 > 0.85, f"{col} R² = {r2:.4f} below 0.85 threshold."

    def test_derived_mode_does_not_require_literal_global_rate_column(self, data):
        dict_df, data_df = data
        dict_wo = dict_df.drop(columns=[DERIVED_RATE_COLUMN], errors="ignore")
        data_wo = data_df.drop(columns=[DERIVED_RATE_COLUMN], errors="ignore")
        res = estimate_from_dataframes(
            dict_df=dict_wo,
            data_df=data_wo,
            feature_columns="derived",
            distance_metric="l2_zscore",
            include_global_rate=False,
            exclude_same_file=True,
        )
        assert "est_flux_cm2_min" in res.columns
        assert np.isfinite(pd.to_numeric(res["best_distance"], errors="coerce")).sum() > 0


class TestRawFeaturesRegression:
    """Regression guard: raw features must NOT achieve good flux estimation.

    This documents the known degeneracy and ensures we don't accidentally
    revert to a state where raw features appear to work (which would mean
    something else changed in the data).
    """

    @pytest.fixture(scope="class")
    def raw_result(self, data):
        dict_df, data_df = data
        raw = estimate_from_dataframes(
            dict_df=dict_df,
            data_df=data_df,
            feature_columns="auto",
            distance_metric="l2_zscore",
            include_global_rate=False,
            exclude_same_file=True,
        )
        derived = estimate_from_dataframes(
            dict_df=dict_df,
            data_df=data_df,
            feature_columns="derived",
            distance_metric="l2_zscore",
            include_global_rate=False,
            exclude_same_file=True,
        )
        return raw, derived, data_df

    def test_raw_flux_r2_not_better_than_derived(self, raw_result):
        """Raw features should not outperform derived mode on flux recovery."""
        raw_res, derived_res, data_df = raw_result
        true = data_df["flux_cm2_min"].values[: len(raw_res)]
        r2_raw = _r2(true, raw_res["est_flux_cm2_min"].values)
        r2_derived = _r2(true, derived_res["est_flux_cm2_min"].values)
        assert r2_raw <= (r2_derived + 0.02), (
            f"Raw-feature flux R² = {r2_raw:.4f} unexpectedly exceeds derived "
            f"flux R² = {r2_derived:.4f} by >0.02."
        )
