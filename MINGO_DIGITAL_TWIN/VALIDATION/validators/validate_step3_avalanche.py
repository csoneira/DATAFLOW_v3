#!/usr/bin/env python3
"""Validator for STEP 3 avalanche outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_io import StepArtifact, load_frame
from .common_report import RESULT_COLUMNS, ResultBuilder


def _norm_tt(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().replace(".0", "")
    if text in {"", "nan", "None", "<NA>"}:
        return ""
    return "".join(ch for ch in text if ch in "1234")


def _eff_status(delta: float, sigma: float) -> tuple[str, float]:
    if sigma <= 0:
        zscore = np.inf if abs(delta) > 0 else 0.0
    else:
        zscore = abs(delta) / sigma
    if abs(delta) <= 0.01 or zscore <= 3:
        return "PASS", float(zscore)
    if abs(delta) <= 0.02 or zscore <= 5:
        return "WARN", float(zscore)
    return "FAIL", float(zscore)


def _plot(df: pd.DataFrame, cfg_eff: list[float], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    obs_vals = []
    for i in range(1, 5):
        t_col = f"T_sum_{i}_ns"
        e_col = f"avalanche_exists_{i}"
        if t_col in df.columns and e_col in df.columns:
            denom = df[t_col].notna()
            if denom.any():
                obs_vals.append(float(df.loc[denom, e_col].astype(bool).mean()))
            else:
                obs_vals.append(np.nan)
        else:
            obs_vals.append(np.nan)

    x = np.arange(1, 5)
    axes[0].bar(x - 0.2, cfg_eff, width=0.4, label="configured", color="slateblue", alpha=0.8)
    axes[0].bar(x + 0.2, obs_vals, width=0.4, label="observed", color="seagreen", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Efficiency closure")
    axes[0].legend()

    size_cols = [f"avalanche_size_electrons_{i}" for i in range(1, 5) if f"avalanche_size_electrons_{i}" in df.columns]
    vals = df[size_cols].to_numpy(dtype=float).ravel() if size_cols else np.array([])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        axes[1].hist(np.log10(vals), bins=80, color="darkorange", alpha=0.8)
    axes[1].set_title("log10(avalanche size)")

    fig.tight_layout()
    fig.savefig(plot_dir / "step3_efficiency_and_size.png", dpi=140)
    plt.close(fig)


def run(
    artifacts: dict[str, StepArtifact],
    run_timestamp: str,
    output_dir: Path,
    make_plots: bool = False,
) -> pd.DataFrame:
    art = artifacts.get("3")
    rb = ResultBuilder(
        run_timestamp=run_timestamp,
        validator="validate_step3_avalanche",
        step="3",
        sim_run=art.sim_run if art else None,
        config_hash=art.config_hash if art else None,
        upstream_hash=art.upstream_hash if art else None,
        n_rows_in=art.row_count if art else None,
        n_rows_out=art.row_count if art else None,
    )

    if art is None or art.data_path is None or not art.data_path.exists():
        rb.add(
            test_id="step3_exists",
            test_name="STEP 3 output exists",
            metric_name="step3_exists",
            metric_value=0,
            status="SKIP",
            notes="No STEP 3 dataset found",
        )
        return rb.to_frame().reindex(columns=RESULT_COLUMNS)

    cols = ["event_id", "tt_avalanche"]
    for i in range(1, 5):
        cols.extend(
            [
                f"T_sum_{i}_ns",
                f"avalanche_ion_{i}",
                f"avalanche_exists_{i}",
                f"avalanche_x_{i}",
                f"avalanche_y_{i}",
                f"avalanche_size_electrons_{i}",
            ]
        )

    try:
        df = load_frame(art.data_path, columns=cols)
    except Exception as exc:
        rb.add_exception(test_id="step3_read", test_name="Read STEP 3", exc=exc)
        return rb.to_frame().reindex(columns=RESULT_COLUMNS)

    if df.empty:
        rb.add(
            test_id="step3_non_empty",
            test_name="STEP 3 has rows",
            metric_name="rows",
            metric_value=0,
            status="FAIL",
            notes="No rows read from STEP 3",
        )
        return rb.to_frame().reindex(columns=RESULT_COLUMNS)

    rb.add(
        test_id="step3_rows",
        test_name="STEP 3 row count",
        metric_name="rows",
        metric_value=len(df),
        status="PASS",
    )

    cfg = art.metadata.get("config") or {}
    eff_cfg = cfg.get("efficiencies") if isinstance(cfg.get("efficiencies"), list) else [np.nan] * 4
    if len(eff_cfg) != 4:
        eff_cfg = [np.nan] * 4

    # Consistency of per-plane flags and sizes.
    exists_mismatch = 0
    neg_size = 0
    tt_mismatch = 0
    for _, row in df.iterrows():
        tt_expected = ""
        for i in range(1, 5):
            ion = row.get(f"avalanche_ion_{i}")
            exists = bool(row.get(f"avalanche_exists_{i}")) if pd.notna(row.get(f"avalanche_exists_{i}")) else False
            size = row.get(f"avalanche_size_electrons_{i}")
            if pd.notna(size) and float(size) < 0:
                neg_size += 1
            ion_positive = pd.notna(ion) and float(ion) > 0
            size_positive = pd.notna(size) and float(size) > 0
            if exists != ion_positive:
                exists_mismatch += 1
            if exists:
                tt_expected += str(i)
            if exists and not size_positive:
                exists_mismatch += 1

        if tt_expected != _norm_tt(row.get("tt_avalanche")):
            tt_mismatch += 1

    rb.add(
        test_id="step3_exists_consistency",
        test_name="avalanche_exists consistent with ion/size",
        metric_name="inconsistent_values",
        metric_value=exists_mismatch,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if exists_mismatch == 0 else "FAIL",
    )

    rb.add(
        test_id="step3_tt_consistency",
        test_name="tt_avalanche matches per-plane existence",
        metric_name="mismatch_rows",
        metric_value=tt_mismatch,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if tt_mismatch == 0 else "FAIL",
    )

    rb.add(
        test_id="step3_nonnegative_size",
        test_name="avalanche size is non-negative",
        metric_name="negative_values",
        metric_value=neg_size,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if neg_size == 0 else "FAIL",
    )

    # Efficiency closure per plane.
    for i in range(1, 5):
        t_col = f"T_sum_{i}_ns"
        e_col = f"avalanche_exists_{i}"
        if t_col not in df.columns or e_col not in df.columns:
            rb.add(
                test_id=f"step3_eff_plane{i}",
                test_name=f"Plane {i} efficiency closure",
                metric_name="status",
                metric_value=np.nan,
                status="SKIP",
                notes="Missing required columns",
            )
            continue

        denom = df[t_col].notna()
        n = int(denom.sum())
        if n == 0:
            rb.add(
                test_id=f"step3_eff_plane{i}",
                test_name=f"Plane {i} efficiency closure",
                metric_name="n_events",
                metric_value=0,
                status="SKIP",
                notes="No valid crossings for this plane",
            )
            continue

        obs = float(df.loc[denom, e_col].astype(bool).mean())
        exp = float(eff_cfg[i - 1]) if np.isfinite(eff_cfg[i - 1]) else np.nan
        if not np.isfinite(exp):
            rb.add(
                test_id=f"step3_eff_plane{i}",
                test_name=f"Plane {i} efficiency closure",
                metric_name="observed_eff",
                metric_value=obs,
                status="SKIP",
                notes="Configured efficiency unavailable",
            )
            continue

        sigma = float(np.sqrt(max(exp * (1.0 - exp), 0.0) / n))
        delta = obs - exp
        status, zscore = _eff_status(delta, sigma)
        rb.add(
            test_id=f"step3_eff_plane{i}",
            test_name=f"Plane {i} efficiency closure",
            metric_name="delta_eff",
            metric_value=delta,
            expected_value=0.0,
            threshold_low=-3 * sigma,
            threshold_high=3 * sigma,
            status=status,
            notes=f"obs={obs:.5f}, exp={exp:.5f}, N={n}, z={zscore:.2f}",
        )

    # Outlier sanity.
    size_cols = [f"avalanche_size_electrons_{i}" for i in range(1, 5) if f"avalanche_size_electrons_{i}" in df.columns]
    sizes = df[size_cols].to_numpy(dtype=float).ravel() if size_cols else np.array([])
    sizes = sizes[np.isfinite(sizes) & (sizes > 0)]
    if sizes.size > 50:
        p999 = float(np.quantile(sizes, 0.999))
        med = float(np.median(sizes))
        ratio = p999 / med if med > 0 else np.inf
        status = "PASS" if ratio <= 100 else ("WARN" if ratio <= 500 else "FAIL")
        rb.add(
            test_id="step3_size_tail",
            test_name="Avalanche size tail sanity",
            metric_name="p999_over_median",
            metric_value=ratio,
            expected_value=None,
            threshold_low=0,
            threshold_high=100,
            status=status,
            notes=f"median={med:.3e}, p999={p999:.3e}",
        )

    if make_plots:
        _plot(df, [float(x) if np.isfinite(x) else np.nan for x in eff_cfg], output_dir / "plots" / "validate_step3_avalanche")

    return rb.to_frame().reindex(columns=RESULT_COLUMNS)
