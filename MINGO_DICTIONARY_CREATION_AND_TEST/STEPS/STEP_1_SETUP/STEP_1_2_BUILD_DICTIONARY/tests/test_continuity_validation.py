#!/usr/bin/env python3
"""Tests for dictionary continuity validation checks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STEP12_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STEP12_DIR))

from build_dictionary import _apply_local_continuity_filter, _validate_dictionary_continuity


def _make_grid_dictionary(
    n_flux: int = 20,
    n_eff: int = 20,
    flux_range: tuple[float, float] = (0.8, 1.2),
    eff_range: tuple[float, float] = (0.5, 0.95),
) -> pd.DataFrame:
    """Create a uniform grid dictionary with consistent empirical efficiencies."""
    flux_vals = np.linspace(flux_range[0], flux_range[1], n_flux)
    eff_vals = np.linspace(eff_range[0], eff_range[1], n_eff)
    ff, ee = np.meshgrid(flux_vals, eff_vals)
    n = ff.size
    df = pd.DataFrame({
        "flux_cm2_min": ff.ravel(),
        "cos_n": np.full(n, 0.5),
        "eff_sim_1": ee.ravel(),
        "eff_sim_2": ee.ravel() * 0.95,
        "eff_sim_3": ee.ravel() * 0.90,
        "eff_sim_4": ee.ravel() * 0.85,
        "eff_empirical_1": ee.ravel() * 0.98,
        "eff_empirical_2": ee.ravel() * 0.93,
        "eff_empirical_3": ee.ravel() * 0.88,
        "eff_empirical_4": ee.ravel() * 0.83,
    })
    return df


def _default_isotonic() -> dict[int, dict]:
    """Return a valid isotonic calibration dict for all 4 planes."""
    iso = {}
    for p in (1, 2, 3, 4):
        iso[p] = {
            "x_knots": np.array([0.10, 0.40, 0.70], dtype=float),
            "y_knots": np.array([0.15, 0.55, 0.90], dtype=float),
            "x_min": 0.10,
            "x_max": 0.70,
            "slope_lo": 0.8,
            "slope_hi": 1.0,
        }
    return iso


PARAM_COLS = ["flux_cm2_min", "cos_n", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]
DEFAULT_CFG = {
    "enabled": True,
    "local_continuity_k": 10,
    "local_continuity_cv_p95_max": 0.50,
    "topology_k": 12,
    "topology_overlap_p10_min": 0.20,
    "topology_overlap_median_min": 0.30,
    "topology_forward_expansion_p95_max": 8.0,
    "topology_backward_expansion_p95_max": 8.0,
    "topology_bad_fraction_max": 0.20,
    "injectivity_enabled": True,
    "injectivity_k": 10,
    "injectivity_span_fraction_p95_max": 0.45,
    "injectivity_flux_span_fraction_p95_max": 0.45,
    "injectivity_point_span_fraction_max": 0.60,
    "injectivity_bad_fraction_max": 0.20,
    "out_of_support_max_fraction": 0.30,
    "density_grid_bins": 5,
    "density_ratio_max": 20.0,
}


# ── Test 1: Coverage passes for uniform grid ─────────────────────

def test_coverage_passes_uniform() -> None:
    dictionary = _make_grid_dictionary()
    dataset = dictionary.copy()
    _, metrics, _ = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, DEFAULT_CFG, _default_isotonic(),
    )
    c1 = metrics["checks"]["param_space_coverage"]
    assert c1["status"] == "PASS"
    assert c1["nn_p95_pct"] < 25.0


# ── Test 2: Coverage warns when large void exists ─────────────────

def test_coverage_warns_void() -> None:
    # Sparse dictionary with a large void — only 4 entries at the corners
    df = pd.DataFrame({
        "flux_cm2_min": [0.8, 0.8, 1.2, 1.2],
        "cos_n": [0.5, 0.5, 0.5, 0.5],
        "eff_sim_1": [0.5, 0.95, 0.5, 0.95],
        "eff_sim_2": [0.48, 0.90, 0.48, 0.90],
        "eff_sim_3": [0.45, 0.86, 0.45, 0.86],
        "eff_sim_4": [0.43, 0.81, 0.43, 0.81],
        "eff_empirical_1": [0.49, 0.93, 0.49, 0.93],
        "eff_empirical_2": [0.46, 0.88, 0.46, 0.88],
        "eff_empirical_3": [0.43, 0.83, 0.43, 0.83],
        "eff_empirical_4": [0.40, 0.78, 0.40, 0.78],
    })
    dataset = _make_grid_dictionary()
    _, metrics, _ = _validate_dictionary_continuity(
        df, dataset, PARAM_COLS, DEFAULT_CFG, _default_isotonic(),
    )
    c1 = metrics["checks"]["param_space_coverage"]
    # Only 4 corner points → very large NN distances
    assert c1["status"] == "WARN"
    assert c1["nn_p95_pct"] > 25.0


# ── Test 3: Local continuity passes for smooth mapping ────────────

def test_local_continuity_passes_smooth() -> None:
    dictionary = _make_grid_dictionary(n_flux=15, n_eff=15)
    dataset = dictionary.copy()
    _, metrics, _ = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, DEFAULT_CFG, _default_isotonic(),
    )
    c2 = metrics["checks"]["local_continuity"]
    assert c2["status"] == "PASS"
    assert c2["discontinuous_fraction"] < 0.10


# ── Test 4: Local continuity warns for degenerate mapping ─────────

def test_local_continuity_warns_degenerate() -> None:
    n = 100
    rng = np.random.RandomState(42)
    # Same empirical efficiencies but wildly different sim values
    emp = rng.uniform(0.5, 0.7, n)
    df = pd.DataFrame({
        "flux_cm2_min": rng.uniform(0.8, 1.2, n),
        "cos_n": np.full(n, 0.5),
        "eff_sim_1": rng.uniform(0.3, 0.95, n),  # random → high CV
        "eff_sim_2": rng.uniform(0.3, 0.95, n),
        "eff_sim_3": rng.uniform(0.3, 0.95, n),
        "eff_sim_4": rng.uniform(0.3, 0.95, n),
        "eff_empirical_1": emp,
        "eff_empirical_2": emp * 0.95,
        "eff_empirical_3": emp * 0.90,
        "eff_empirical_4": emp * 0.85,
    })
    dataset = df.copy()
    cfg = {**DEFAULT_CFG, "local_continuity_k": 10, "local_continuity_cv_p95_max": 0.20}
    _, metrics, _ = _validate_dictionary_continuity(
        df, dataset, PARAM_COLS, cfg, _default_isotonic(),
    )
    c2 = metrics["checks"]["local_continuity"]
    assert c2["status"] == "WARN"
    assert c2["n_discontinuous"] > 0


# ── Test 5: Isotonic bounds fail on negative slope ────────────────

def test_topology_bidirectional_passes_smooth() -> None:
    dictionary = _make_grid_dictionary(n_flux=18, n_eff=18)
    # Ensure an injective smooth mapping from parameter -> feature so
    # neighborhoods are preserved in both directions.
    flux = dictionary["flux_cm2_min"].to_numpy(dtype=float)
    flux_norm = (flux - np.min(flux)) / max(np.max(flux) - np.min(flux), 1e-12)
    dictionary["eff_empirical_1"] = 0.82 * dictionary["eff_sim_1"] + 0.06 * flux_norm
    dictionary["eff_empirical_2"] = 0.80 * dictionary["eff_sim_2"] + 0.05 * flux_norm
    dictionary["eff_empirical_3"] = 0.78 * dictionary["eff_sim_3"] + 0.04 * flux_norm
    dictionary["eff_empirical_4"] = 0.76 * dictionary["eff_sim_4"] + 0.03 * flux_norm
    dataset = dictionary.copy()
    _, metrics, _ = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, DEFAULT_CFG, _default_isotonic(),
    )
    ctopo = metrics["checks"]["topology_bidirectional_continuity"]
    assert ctopo["status"] == "PASS"
    assert ctopo["overlap_median"] >= 0.30
    assert ctopo["bad_fraction"] <= 0.20


def test_topology_bidirectional_warns_scrambled_mapping() -> None:
    dictionary = _make_grid_dictionary(n_flux=15, n_eff=15)
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(dictionary))
    # Break the row-wise parameter -> feature correspondence while keeping
    # feature-space values valid.
    for p in (1, 2, 3, 4):
        col = f"eff_empirical_{p}"
        dictionary[col] = dictionary[col].to_numpy()[perm]
    dataset = dictionary.copy()
    cfg = {
        **DEFAULT_CFG,
        "topology_k": 10,
        "topology_overlap_p10_min": 0.25,
        "topology_overlap_median_min": 0.40,
        "topology_forward_expansion_p95_max": 2.0,
        "topology_backward_expansion_p95_max": 2.0,
        "topology_bad_fraction_max": 0.15,
    }
    _, metrics, _ = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, cfg, _default_isotonic(),
    )
    ctopo = metrics["checks"]["topology_bidirectional_continuity"]
    assert ctopo["status"] == "WARN"
    assert (ctopo["bad_fraction"] > 0.15) or (ctopo["overlap_p10"] < 0.25)


def test_local_injectivity_fails_scrambled_mapping() -> None:
    dictionary = _make_grid_dictionary(n_flux=14, n_eff=14)
    rng = np.random.RandomState(11)
    perm = rng.permutation(len(dictionary))
    for p in (1, 2, 3, 4):
        col = f"eff_empirical_{p}"
        dictionary[col] = dictionary[col].to_numpy()[perm]
    dataset = dictionary.copy()
    cfg = {
        **DEFAULT_CFG,
        "injectivity_k": 10,
        "injectivity_span_fraction_p95_max": 0.30,
        "injectivity_flux_span_fraction_p95_max": 0.20,
        "injectivity_point_span_fraction_max": 0.35,
        "injectivity_bad_fraction_max": 0.10,
    }
    _, metrics, _ = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, cfg, _default_isotonic(),
    )
    cinj = metrics["checks"]["local_injectivity"]
    assert cinj["status"] == "FAIL"
    assert cinj["flux_span_fraction_p95"] > 0.20
    assert cinj["bad_fraction"] > 0.10


# ── Test 5: Isotonic bounds fail on negative slope ────────────────

def test_isotonic_bounds_fail_negative_slope() -> None:
    dictionary = _make_grid_dictionary(n_flux=5, n_eff=5)
    dataset = dictionary.copy()
    iso = _default_isotonic()
    iso[1]["slope_lo"] = -0.5  # violates bounds

    _, metrics, messages = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, DEFAULT_CFG, iso,
    )
    c3 = metrics["checks"]["isotonic_bounds"]
    assert c3["status"] == "FAIL"
    assert len(c3["violations"]) > 0
    assert metrics["status"] == "FAIL"


# ── Test 6: Support adequacy warns when dict range is narrow ──────

def test_support_adequacy_warns_narrow() -> None:
    # Dictionary has narrow eff range
    dictionary = _make_grid_dictionary(n_flux=10, n_eff=10, eff_range=(0.80, 0.90))
    # Dataset has wider range
    dataset = _make_grid_dictionary(n_flux=10, n_eff=20, eff_range=(0.50, 0.95))

    _, metrics, messages = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, DEFAULT_CFG, _default_isotonic(),
    )
    c4 = metrics["checks"]["support_adequacy"]
    assert c4["status"] == "WARN"
    # At least one plane should have high OOS fraction
    oos = c4["out_of_support_fraction_by_plane"]
    assert any(v > 0.30 for v in oos.values())


# ── Test 7: Density warns when entries are clustered ──────────────

def test_density_warns_cluster() -> None:
    n = 200
    rng = np.random.RandomState(123)
    # 90% in one corner, 10% spread
    n_corner = int(0.9 * n)
    n_spread = n - n_corner
    flux = np.concatenate([
        rng.uniform(0.80, 0.82, n_corner),
        rng.uniform(0.80, 1.20, n_spread),
    ])
    eff = np.concatenate([
        rng.uniform(0.50, 0.52, n_corner),
        rng.uniform(0.50, 0.95, n_spread),
    ])
    df = pd.DataFrame({
        "flux_cm2_min": flux,
        "cos_n": np.full(n, 0.5),
        "eff_sim_1": eff,
        "eff_sim_2": eff * 0.95,
        "eff_sim_3": eff * 0.90,
        "eff_sim_4": eff * 0.85,
        "eff_empirical_1": eff * 0.98,
        "eff_empirical_2": eff * 0.93,
        "eff_empirical_3": eff * 0.88,
        "eff_empirical_4": eff * 0.83,
    })
    dataset = df.copy()
    cfg = {**DEFAULT_CFG, "density_ratio_max": 5.0}

    _, metrics, _ = _validate_dictionary_continuity(
        df, dataset, PARAM_COLS, cfg, _default_isotonic(),
    )
    c5 = metrics["checks"]["density_uniformity"]
    assert c5["status"] == "WARN"
    assert c5["density_ratio"] > 5.0


# ── Test 8: Empty isotonic gracefully skips ───────────────────────

def test_disabled_returns_skipped() -> None:
    # Enabled/disabled behavior is handled in main(); here we verify
    # the function gracefully skips isotonic-bounds validation when no
    # isotonic calibration payload is provided.
    dictionary = _make_grid_dictionary(n_flux=5, n_eff=5)
    dataset = dictionary.copy()
    cfg = {**DEFAULT_CFG, "injectivity_enabled": False}
    overall, metrics, _ = _validate_dictionary_continuity(
        dictionary, dataset, PARAM_COLS, cfg, {},  # no isotonic data
    )
    c3 = metrics["checks"]["isotonic_bounds"]
    assert c3["status"] == "SKIPPED"
    assert overall in {"PASS", "WARN"}


def test_filter_caps_to_budget_when_flagged_exceeds_max_drop() -> None:
    dictionary = pd.DataFrame({"x": np.arange(10, dtype=float)})
    cv_metrics = {
        "checks": {
            "local_continuity": {
                "status": "WARN",
                "row_indices": list(range(10)),
                "worst_cv_per_entry": [0.1, 0.8, 1.0, 1.2, 0.7, 0.9, 1.4, 0.6, 0.2, 0.3],
                "cv_threshold": 0.5,
            },
            "topology_bidirectional_continuity": {"status": "SKIPPED"},
        }
    }
    cfg = {
        "filter_enabled": True,
        "filter_include_topology": False,
        "local_continuity_cv_p95_max": 0.5,
        "filter_max_drop_fraction": 0.30,
        "filter_allow_large_drop": False,
    }
    filtered, report = _apply_local_continuity_filter(dictionary, cv_metrics, cfg)
    assert report["applied"] is True
    assert report["capped_by_max_drop_fraction"] is True
    assert report["rows_flagged_before_cap"] == 7
    assert report["rows_removed"] == 3
    assert len(filtered) == 7


def test_filter_allows_large_drop_when_enabled() -> None:
    dictionary = pd.DataFrame({"x": np.arange(10, dtype=float)})
    cv_metrics = {
        "checks": {
            "local_continuity": {
                "status": "WARN",
                "row_indices": list(range(10)),
                "worst_cv_per_entry": [0.1, 0.8, 1.0, 1.2, 0.7, 0.9, 1.4, 0.6, 0.2, 0.3],
                "cv_threshold": 0.5,
            },
            "topology_bidirectional_continuity": {"status": "SKIPPED"},
        }
    }
    cfg = {
        "filter_enabled": True,
        "filter_include_topology": False,
        "local_continuity_cv_p95_max": 0.5,
        "filter_max_drop_fraction": 0.30,
        "filter_allow_large_drop": True,
    }
    filtered, report = _apply_local_continuity_filter(dictionary, cv_metrics, cfg)
    assert report["applied"] is True
    assert report["capped_by_max_drop_fraction"] is False
    assert report["rows_removed"] == 7
    assert len(filtered) == 3


def test_filter_includes_injectivity_flags() -> None:
    dictionary = pd.DataFrame({"x": np.arange(10, dtype=float)})
    cv_metrics = {
        "checks": {
            "local_continuity": {
                "status": "PASS",
                "row_indices": list(range(10)),
                "worst_cv_per_entry": [0.1] * 10,
                "cv_threshold": 0.5,
            },
            "topology_bidirectional_continuity": {"status": "SKIPPED"},
            "local_injectivity": {
                "status": "FAIL",
                "row_indices": list(range(10)),
                "bad_point_mask": [False, True, True, False, False, False, False, False, False, False],
                "max_span_fraction_per_entry": [0.1, 0.7, 0.65, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "thresholds": {
                    "point_span_fraction_max_by_param": {"flux_cm2_min": 0.45},
                    "bad_fraction_max": 0.1,
                },
            },
        }
    }
    cfg = {
        "filter_enabled": True,
        "filter_include_topology": False,
        "filter_include_injectivity": True,
        "local_continuity_cv_p95_max": 0.5,
        "filter_max_drop_fraction": 0.50,
        "filter_allow_large_drop": True,
    }
    filtered, report = _apply_local_continuity_filter(dictionary, cv_metrics, cfg)
    assert report["applied"] is True
    assert report["rows_flagged_injectivity"] == 2
    assert report["rows_removed"] == 2
    assert len(filtered) == 8
