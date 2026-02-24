#!/usr/bin/env python3
"""Validator for STEP 4 induced strip observables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common_io import StepArtifact, load_frame
from .common_report import RESULT_COLUMNS, ResultBuilder, normalize_tt_series


def _plot(df4: pd.DataFrame, ratio_map: dict[int, np.ndarray], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    counts = df4["tt_hit"].astype("string").fillna("").value_counts().sort_index()
    axes[0, 0].bar(counts.index.astype(str), counts.values, color="steelblue", alpha=0.8)
    axes[0, 0].set_title("tt_hit")

    y_cols = [f"Y_mea_{i}_s{j}" for i in range(1, 5) for j in range(1, 5) if f"Y_mea_{i}_s{j}" in df4.columns]
    yv = df4[y_cols].to_numpy(dtype=float).ravel() if y_cols else np.array([])
    yv = yv[yv > 0]
    axes[0, 1].hist(np.log10(yv), bins=80, color="seagreen", alpha=0.8)
    axes[0, 1].set_title("log10(Y_mea)")

    for p, ax in zip(range(1, 5), [axes[1, 0], axes[1, 1], axes[1, 0], axes[1, 1]]):
        plane_cols = [f"Y_mea_{p}_s{j}" for j in range(1, 5) if f"Y_mea_{p}_s{j}" in df4.columns]
        if not plane_cols:
            continue
        arr = df4[plane_cols].to_numpy(dtype=float)
        mult = np.count_nonzero(arr > 0, axis=1)
        ax.hist(
            mult[mult > 0],
            bins=np.arange(0.5, 5.5, 1.0),
            histtype="step",
            linewidth=1.5,
            label=f"Plane {p}",
        )

    axes[1, 0].set_title("Strip multiplicity (planes 1,3)")
    axes[1, 1].set_title("Strip multiplicity (planes 2,4)")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel("n strips with Y_mea > 0")
        ax.set_ylabel("events")
        ax.set_xticks([1, 2, 3, 4])
        if ax.has_data():
            ax.legend()

    fig.tight_layout()
    fig.savefig(plot_dir / "step4_hit_overview.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        ratios = ratio_map.get(plane_idx)
        if ratios is None or ratios.size == 0:
            ax.set_title(f"Plane {plane_idx}: no ratio data")
            ax.axis("off")
            continue
        ax.hist(ratios, bins=80, color="darkorange", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} qsum/avalanche")
        ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(plot_dir / "step4_charge_conservation_ratios.png", dpi=140)
    plt.close(fig)


def run(
    artifacts: dict[str, StepArtifact],
    run_timestamp: str,
    output_dir: Path,
    make_plots: bool = False,
) -> pd.DataFrame:
    art4 = artifacts.get("4")
    art3 = artifacts.get("3")
    rb = ResultBuilder(
        run_timestamp=run_timestamp,
        validator="validate_step4_induction",
        step="4",
        sim_run=art4.sim_run if art4 else None,
        config_hash=art4.config_hash if art4 else None,
        upstream_hash=art4.upstream_hash if art4 else None,
        n_rows_in=art4.row_count if art4 else None,
        n_rows_out=art4.row_count if art4 else None,
    )

    if art4 is None or art4.data_path is None or not art4.data_path.exists():
        rb.add(
            test_id="step4_exists",
            test_name="STEP 4 output exists",
            metric_name="step4_exists",
            metric_value=0,
            status="SKIP",
            notes="No STEP 4 dataset found",
        )
        return rb.to_frame().reindex(columns=RESULT_COLUMNS)

    cols4 = ["event_id", "tt_hit"]
    for i in range(1, 5):
        for j in range(1, 5):
            cols4.extend([f"Y_mea_{i}_s{j}", f"X_mea_{i}_s{j}", f"T_sum_meas_{i}_s{j}"])

    try:
        df4 = load_frame(art4.data_path, columns=cols4)
    except Exception as exc:
        rb.add_exception(test_id="step4_read", test_name="Read STEP 4", exc=exc)
        return rb.to_frame().reindex(columns=RESULT_COLUMNS)

    if df4.empty:
        rb.add(
            test_id="step4_non_empty",
            test_name="STEP 4 has rows",
            metric_name="rows",
            metric_value=0,
            status="FAIL",
            notes="No rows read from STEP 4",
        )
        return rb.to_frame().reindex(columns=RESULT_COLUMNS)

    rb.add(
        test_id="step4_rows",
        test_name="STEP 4 row count",
        metric_name="rows",
        metric_value=len(df4),
        status="PASS",
    )

    # Non-negativity and internal consistency.
    neg_q = 0
    non_nan_x_when_zero = 0
    non_nan_t_when_zero = 0
    nan_x_when_positive = 0
    plane_hit = np.zeros((len(df4), 4), dtype=bool)
    for i in range(1, 5):
        strip_positive = []
        for j in range(1, 5):
            y_col = f"Y_mea_{i}_s{j}"
            x_col = f"X_mea_{i}_s{j}"
            t_col = f"T_sum_meas_{i}_s{j}"

            if y_col in df4.columns:
                yv = pd.to_numeric(df4[y_col], errors="coerce").to_numpy(dtype=float)
            else:
                yv = np.full(len(df4), np.nan, dtype=float)
            y_safe = np.nan_to_num(yv, nan=0.0)

            neg_q += int(np.count_nonzero(y_safe < 0))
            zero_or_less = y_safe <= 0
            positive = y_safe > 0
            strip_positive.append(positive)

            if x_col in df4.columns:
                x_notna = df4[x_col].notna().to_numpy(dtype=bool)
                non_nan_x_when_zero += int(np.count_nonzero(zero_or_less & x_notna))
                nan_x_when_positive += int(np.count_nonzero(positive & ~x_notna))
            else:
                nan_x_when_positive += int(np.count_nonzero(positive))

            if t_col in df4.columns:
                t_notna = df4[t_col].notna().to_numpy(dtype=bool)
                non_nan_t_when_zero += int(np.count_nonzero(zero_or_less & t_notna))

        if strip_positive:
            plane_hit[:, i - 1] = np.column_stack(strip_positive).any(axis=1)

    tt_expected = np.full(len(df4), "", dtype=object)
    for i in range(1, 5):
        tt_expected = np.where(plane_hit[:, i - 1], tt_expected + str(i), tt_expected)

    if "tt_hit" in df4.columns:
        tt_actual = normalize_tt_series(df4["tt_hit"]).to_numpy(dtype=str)
    else:
        tt_actual = np.full(len(df4), "", dtype=str)
    tt_mismatch = int(np.count_nonzero(tt_expected != tt_actual))

    rb.add(
        test_id="step4_q_nonnegative",
        test_name="Y_mea is non-negative",
        metric_name="negative_values",
        metric_value=neg_q,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if neg_q == 0 else "FAIL",
    )

    rb.add(
        test_id="step4_zero_charge_x_nan",
        test_name="X_mea is NaN when Y_mea <= 0",
        metric_name="violations",
        metric_value=non_nan_x_when_zero,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if non_nan_x_when_zero == 0 else "WARN",
    )

    rb.add(
        test_id="step4_zero_charge_t_nan",
        test_name="T_sum_meas is NaN when Y_mea <= 0",
        metric_name="violations",
        metric_value=non_nan_t_when_zero,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if non_nan_t_when_zero == 0 else "WARN",
    )

    rb.add(
        test_id="step4_positive_charge_x_present",
        test_name="X_mea present when Y_mea > 0",
        metric_name="missing_values",
        metric_value=nan_x_when_positive,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if nan_x_when_positive == 0 else "WARN",
    )

    rb.add(
        test_id="step4_tt_consistency",
        test_name="tt_hit matches per-plane hits",
        metric_name="mismatch_rows",
        metric_value=tt_mismatch,
        expected_value=0,
        threshold_low=0,
        threshold_high=0,
        status="PASS" if tt_mismatch == 0 else "FAIL",
    )

    # Charge conservation-like closure using STEP 3 avalanche sizes.
    ratio_map: dict[int, np.ndarray] = {}
    if art3 is None or art3.data_path is None or not art3.data_path.exists():
        rb.add(
            test_id="step4_charge_closure_inputs",
            test_name="STEP 3 inputs available for closure",
            metric_name="step3_available",
            metric_value=0,
            status="SKIP",
            notes="STEP 3 dataset unavailable",
        )
    else:
        cols3 = ["event_id"] + [f"avalanche_size_electrons_{i}" for i in range(1, 5)]
        try:
            df3 = load_frame(art3.data_path, columns=cols3)
            merged = df4.merge(df3, on="event_id", how="inner", suffixes=("", "_s3"))
        except Exception as exc:
            rb.add_exception(test_id="step4_charge_closure_read", test_name="Read STEP 3 for closure", exc=exc)
            merged = pd.DataFrame()

        if merged.empty:
            rb.add(
                test_id="step4_charge_closure_merge",
                test_name="Merge STEP 4 with STEP 3",
                metric_name="merged_rows",
                metric_value=0,
                status="SKIP",
                notes="No common event_id rows",
            )
        else:
            for i in range(1, 5):
                y_cols = [f"Y_mea_{i}_s{j}" for j in range(1, 5) if f"Y_mea_{i}_s{j}" in merged.columns]
                aval_col = f"avalanche_size_electrons_{i}"
                if not y_cols or aval_col not in merged.columns:
                    rb.add(
                        test_id=f"step4_charge_closure_plane{i}",
                        test_name=f"Plane {i} qsum vs avalanche size",
                        metric_name="status",
                        metric_value=np.nan,
                        status="SKIP",
                        notes="Missing required columns",
                    )
                    continue

                qsum = merged[y_cols].to_numpy(dtype=float).sum(axis=1)
                aval = merged[aval_col].to_numpy(dtype=float)
                mask = np.isfinite(aval) & (aval > 0)
                if not mask.any():
                    rb.add(
                        test_id=f"step4_charge_closure_plane{i}",
                        test_name=f"Plane {i} qsum vs avalanche size",
                        metric_name="n_rows",
                        metric_value=0,
                        status="SKIP",
                        notes="No avalanches in plane",
                    )
                    continue

                ratio = qsum[mask] / aval[mask]
                ratio = ratio[np.isfinite(ratio)]
                ratio_map[i] = ratio
                if ratio.size == 0:
                    rb.add(
                        test_id=f"step4_charge_closure_plane{i}",
                        test_name=f"Plane {i} qsum vs avalanche size",
                        metric_name="n_rows",
                        metric_value=0,
                        status="SKIP",
                        notes="No finite ratio values",
                    )
                    continue

                med = float(np.median(ratio))
                status = "PASS" if 0.8 <= med <= 1.2 else ("WARN" if 0.6 <= med <= 1.4 else "FAIL")
                rb.add(
                    test_id=f"step4_charge_closure_plane{i}",
                    test_name=f"Plane {i} qsum vs avalanche size",
                    metric_name="median_ratio",
                    metric_value=med,
                    expected_value=1.0,
                    threshold_low=0.8,
                    threshold_high=1.2,
                    status=status,
                    notes=f"N={ratio.size}",
                )

                hit_mult = []
                for j in range(1, 5):
                    col = f"Y_mea_{i}_s{j}"
                    if col in merged.columns:
                        hit_mult.append(merged[col].to_numpy(dtype=float) > 0)
                if hit_mult:
                    mult = np.sum(np.column_stack(hit_mult), axis=1)
                    mult = mult[mask]
                    mean_mult = float(np.mean(mult)) if mult.size else np.nan
                    rb.add(
                        test_id=f"step4_strip_multiplicity_plane{i}",
                        test_name=f"Plane {i} strip multiplicity",
                        metric_name="mean_multiplicity",
                        metric_value=mean_mult,
                        threshold_low=0,
                        threshold_high=4,
                        status="PASS" if np.isfinite(mean_mult) and 0 <= mean_mult <= 4 else "WARN",
                    )

    if make_plots:
        _plot(df4, ratio_map, output_dir / "plots" / "validate_step4_induction")

    return rb.to_frame().reindex(columns=RESULT_COLUMNS)
