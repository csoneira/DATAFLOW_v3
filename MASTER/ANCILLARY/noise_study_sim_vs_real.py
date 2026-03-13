#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/ANCILLARY/noise_study_sim_vs_real.py
Purpose: Comparative noise study of simulated (MINGO00) vs real (MINGO01) detector metadata.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-13
Runtime: python3
Usage: python3 MASTER/ANCILLARY/noise_study_sim_vs_real.py [--no-plots]
Inputs: task_*_metadata_specific.csv from MINGO00 and MINGO01 STAGE_1/STEP_1/TASK_{1-4}
Outputs: noise_study_report.txt, noise_study_ks_results.csv, PNG/PDF plots.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, gaussian_kde

# -- Paths --------------------------------------------------------------------
# MASTER/ANCILLARY/noise_study_sim_vs_real.py  →  parent = ANCILLARY, parent.parent = MASTER
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # DATAFLOW_v3

FILES_DIR = SCRIPT_DIR / "OUTPUTS" / "noise_study" / "FILES"
PLOTS_DIR = SCRIPT_DIR / "OUTPUTS" / "noise_study" / "PLOTS"

STATIONS = {0: "MINGO00", 1: "MINGO01"}
TASK_IDS = [1, 2, 3, 4]

COLOR_SIM = "#1f77b4"
COLOR_REAL = "#d62728"

_FILE_TS_RE = re.compile(r"(\d{11})$")

LOG = logging.getLogger("NOISE_STUDY")

# =============================================================================
# Data loading
# =============================================================================

def _task_metadata_path(station_id: int, task_id: int) -> Path:
    return (
        REPO_ROOT
        / "STATIONS"
        / f"MINGO{station_id:02d}"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
        / f"task_{task_id}_metadata_specific.csv"
    )


def _parse_filename_base_ts(value: object) -> pd.Timestamp:
    """Parse file time encoded in filename_base (mi0XYYDDDHHMMSS)."""
    text = str(value).strip()
    if not text:
        return pd.NaT
    stem = Path(text).stem.strip().lower()
    if stem.startswith("mini"):
        stem = "mi01" + stem[4:]
    match = _FILE_TS_RE.search(stem)
    if match is None:
        return pd.NaT
    stamp = match.group(1)
    try:
        yy = int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
    except ValueError:
        return pd.NaT
    if not (1 <= day_of_year <= 366):
        return pd.NaT
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return pd.NaT
    try:
        dt = datetime(2000 + yy, 1, 1) + timedelta(
            days=day_of_year - 1, hours=hour, minutes=minute, seconds=second
        )
    except ValueError:
        return pd.NaT
    return pd.Timestamp(dt, tz="UTC")


def _parse_exec_ts(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    parsed = pd.to_datetime(s, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s[missing], errors="coerce", utc=True)
    return parsed


def load_all_data() -> dict[str, pd.DataFrame]:
    data = {}
    for sid, sname in STATIONS.items():
        for tid in TASK_IDS:
            path = _task_metadata_path(sid, tid)
            key = f"{sname}_TASK_{tid}"
            if not path.exists():
                LOG.warning("Missing %s", path)
                continue
            df = pd.read_csv(path, low_memory=False)
            df["_exec_ts"] = _parse_exec_ts(df["execution_timestamp"])
            df["_data_ts"] = df["filename_base"].apply(_parse_filename_base_ts)
            # Deduplicate: keep latest execution_timestamp per filename_base
            if df["filename_base"].duplicated().any():
                df = df.sort_values("_exec_ts").drop_duplicates(
                    subset="filename_base", keep="last"
                )
            df = df.reset_index(drop=True)
            data[key] = df
            LOG.info("Loaded %s: %d rows x %d cols", key, len(df), len(df.columns))
    return data


def _shared_numeric_cols(df0: pd.DataFrame, df1: pd.DataFrame) -> list[str]:
    skip = {"filename_base", "execution_timestamp", "param_hash", "_exec_ts", "_data_ts"}
    cols0 = set(df0.select_dtypes(include="number").columns) - skip
    cols1 = set(df1.select_dtypes(include="number").columns) - skip
    shared = sorted(cols0 & cols1)
    return shared


# =============================================================================
# Report writer
# =============================================================================

class ReportWriter:
    def __init__(self):
        self._sections: list[str] = []

    def add_section(self, title: str, body: str) -> None:
        self._sections.append(f"\n{'='*80}\n{title}\n{'='*80}\n{body}\n")

    def add_table(self, title: str, headers: list[str], rows: list[list]) -> None:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines = [title, fmt.format(*headers), "-" * sum(widths + [2 * (len(widths) - 1)])]
        for row in rows:
            lines.append(fmt.format(*[str(c) for c in row]))
        self._sections.append("\n".join(lines) + "\n")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._sections), encoding="utf-8")
        LOG.info("Report saved to %s", path)


# =============================================================================
# Section 1: Data Quality Overview
# =============================================================================

def analyze_data_quality(data: dict, report: ReportWriter) -> list[plt.Figure]:
    lines = []
    rows = []
    for tid in TASK_IDS:
        for sid, sname in STATIONS.items():
            key = f"{sname}_TASK_{tid}"
            if key not in data:
                continue
            df = data[key]
            n_rows = len(df)
            n_unique = df["filename_base"].nunique()
            n_dupes = n_rows - n_unique  # after dedup this should be 0
            n_cols = len(df.columns)
            n_nan = int(df.select_dtypes(include="number").isna().sum().sum())
            n_zero_cols = int((df.select_dtypes(include="number") == 0).all().sum())
            rows.append([f"TASK_{tid}", sname, n_rows, n_unique, n_dupes, n_cols, n_nan, n_zero_cols])

    headers = ["Task", "Station", "Rows", "Unique files", "Dupes(pre-dedup)", "Cols", "NaN cells", "All-zero cols"]
    report.add_table("DATA QUALITY OVERVIEW", headers, rows)

    # Column presence comparison
    for tid in TASK_IDS:
        k0 = f"MINGO00_TASK_{tid}"
        k1 = f"MINGO01_TASK_{tid}"
        if k0 not in data or k1 not in data:
            continue
        cols0 = set(data[k0].columns)
        cols1 = set(data[k1].columns)
        only0 = sorted(cols0 - cols1 - {"_exec_ts", "_data_ts"})
        only1 = sorted(cols1 - cols0 - {"_exec_ts", "_data_ts"})
        if only0 or only1:
            lines.append(f"\nTASK_{tid} column differences:")
            if only0:
                lines.append(f"  Only in MINGO00 ({len(only0)}): {', '.join(only0[:20])}")
            if only1:
                lines.append(f"  Only in MINGO01 ({len(only1)}): {', '.join(only1[:20])}")

    # All-zero column analysis
    for tid in TASK_IDS:
        k0 = f"MINGO00_TASK_{tid}"
        k1 = f"MINGO01_TASK_{tid}"
        if k0 not in data or k1 not in data:
            continue
        shared = _shared_numeric_cols(data[k0], data[k1])
        zero_in_sim = [c for c in shared if (data[k0][c] == 0).all() and not (data[k1][c] == 0).all()]
        zero_in_real = [c for c in shared if (data[k1][c] == 0).all() and not (data[k0][c] == 0).all()]
        if zero_in_sim:
            lines.append(f"\nTASK_{tid}: {len(zero_in_sim)} cols all-zero in SIM but NOT in REAL:")
            lines.append(f"  {', '.join(zero_in_sim[:15])}")
        if zero_in_real:
            lines.append(f"\nTASK_{tid}: {len(zero_in_real)} cols all-zero in REAL but NOT in SIM:")
            lines.append(f"  {', '.join(zero_in_real[:15])}")

    report.add_section("DATA QUALITY DETAILS", "\n".join(lines))

    # Summary figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    col_labels = headers
    cell_text = [[str(c) for c in r] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax.set_title("Data Quality Overview: MINGO00 (sim) vs MINGO01 (real)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return [fig]


# =============================================================================
# Section 2: Distribution Divergence (KS Tests)
# =============================================================================

def analyze_distributions(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []
    all_ks_rows = []

    for tid in TASK_IDS:
        k0 = f"MINGO00_TASK_{tid}"
        k1 = f"MINGO01_TASK_{tid}"
        if k0 not in data or k1 not in data:
            continue
        df0, df1 = data[k0], data[k1]
        shared = _shared_numeric_cols(df0, df1)

        ks_results = []
        for col in shared:
            v0 = df0[col].dropna().values
            v1 = df1[col].dropna().values
            if len(v0) < 5 or len(v1) < 5:
                continue
            if np.std(v0) == 0 and np.std(v1) == 0:
                continue
            stat, pval = ks_2samp(v0, v1)
            ks_results.append((col, stat, pval))
            all_ks_rows.append([f"TASK_{tid}", col, f"{stat:.4f}", f"{pval:.2e}"])

        ks_results.sort(key=lambda x: -x[1])
        top = ks_results[:20]

        lines = [f"\nTASK_{tid} - Top 20 most divergent columns (KS test):"]
        lines.append(f"  {'Column':<60s} {'KS stat':>8s} {'p-value':>10s}")
        lines.append(f"  {'-'*60} {'-'*8} {'-'*10}")
        for col, stat, pval in top:
            lines.append(f"  {col:<60s} {stat:8.4f} {pval:10.2e}")
        report.add_section(f"DISTRIBUTION DIVERGENCE - TASK_{tid}", "\n".join(lines))

        # Plot top 10
        plot_top = ks_results[:10]
        if not plot_top:
            continue
        n_plots = len(plot_top)
        n_rows_fig = (n_plots + 1) // 2
        fig, axes = plt.subplots(n_rows_fig, 2, figsize=(14, 3 * n_rows_fig))
        axes = np.atleast_2d(axes)
        for idx, (col, stat, pval) in enumerate(plot_top):
            ax = axes[idx // 2, idx % 2]
            v0 = df0[col].dropna().values
            v1 = df1[col].dropna().values
            lo = min(np.nanmin(v0), np.nanmin(v1))
            hi = max(np.nanmax(v0), np.nanmax(v1))
            if lo == hi:
                continue
            bins = np.linspace(lo, hi, 40)
            ax.hist(v0, bins=bins, alpha=0.5, color=COLOR_SIM, density=True, label="SIM")
            ax.hist(v1, bins=bins, alpha=0.5, color=COLOR_REAL, density=True, label="REAL")
            try:
                if np.std(v0) > 0:
                    kde0 = gaussian_kde(v0)
                    xs = np.linspace(lo, hi, 200)
                    ax.plot(xs, kde0(xs), color=COLOR_SIM, lw=1.5)
                if np.std(v1) > 0:
                    kde1 = gaussian_kde(v1)
                    xs = np.linspace(lo, hi, 200)
                    ax.plot(xs, kde1(xs), color=COLOR_REAL, lw=1.5)
            except Exception:
                pass
            ax.set_title(f"{col}\nKS={stat:.3f} p={pval:.1e}", fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
        # hide unused axes
        for idx in range(n_plots, n_rows_fig * 2):
            axes[idx // 2, idx % 2].set_visible(False)
        fig.suptitle(f"TASK_{tid}: Most divergent distributions (SIM vs REAL)", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append(fig)

    # Save KS results CSV
    if all_ks_rows:
        ks_df = pd.DataFrame(all_ks_rows, columns=["task", "column", "ks_stat", "p_value"])
        ks_path = FILES_DIR / "noise_study_ks_results.csv"
        ks_df.to_csv(ks_path, index=False)
        LOG.info("KS results saved to %s", ks_path)

    return figs


# =============================================================================
# Section 3: Correlation & PCA Analysis
# =============================================================================

def _get_rate_cols(df: pd.DataFrame, pattern: str) -> list[str]:
    return sorted([c for c in df.columns if re.search(pattern, c)])


def analyze_correlations(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []

    # Use TASK_1 final_rate columns
    k0 = "MINGO00_TASK_1"
    k1 = "MINGO01_TASK_1"
    if k0 not in data or k1 not in data:
        return figs

    df0, df1 = data[k0], data[k1]
    rate_cols = _get_rate_cols(df0, r"_final_rate_hz$")
    rate_cols = [c for c in rate_cols if c in df1.columns]

    if len(rate_cols) < 4:
        return figs

    # Correlation difference heatmap
    corr0 = df0[rate_cols].corr().values
    corr1 = df1[rate_cols].corr().values
    diff = corr1 - corr0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, mat, title in zip(
        axes,
        [corr0, corr1, diff],
        ["MINGO00 (SIM) correlations", "MINGO01 (REAL) correlations", "Difference (REAL - SIM)"],
    ):
        vmax = 1.0 if "Difference" not in title else max(0.3, np.nanmax(np.abs(diff)))
        vmin = -vmax if "Difference" in title else -1.0
        cmap = "RdBu_r" if "Difference" in title else "coolwarm"
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(0, len(rate_cols), max(1, len(rate_cols) // 8)))
        ax.set_xticklabels([rate_cols[i][:12] for i in range(0, len(rate_cols), max(1, len(rate_cols) // 8))],
                           rotation=90, fontsize=5)
        ax.set_yticks(range(0, len(rate_cols), max(1, len(rate_cols) // 8)))
        ax.set_yticklabels([rate_cols[i][:12] for i in range(0, len(rate_cols), max(1, len(rate_cols) // 8))],
                           fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("TASK_1 Final Rate Correlation Matrices", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figs.append(fig)

    # Top correlation differences
    n = len(rate_cols)
    diffs_list = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs_list.append((rate_cols[i], rate_cols[j], diff[i, j]))
    diffs_list.sort(key=lambda x: -abs(x[2]))
    lines = ["Top 15 correlation pairs with largest SIM-vs-REAL difference:"]
    lines.append(f"  {'Col A':<30s} {'Col B':<30s} {'Diff':>8s}")
    for a, b, d in diffs_list[:15]:
        lines.append(f"  {a:<30s} {b:<30s} {d:+8.4f}")
    report.add_section("CORRELATION ANALYSIS - TASK_1", "\n".join(lines))

    # PCA via SVD
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (key, label, color) in enumerate([(k0, "SIM", COLOR_SIM), (k1, "REAL", COLOR_REAL)]):
        df = data[key]
        X = df[rate_cols].dropna().values
        if len(X) < 5:
            continue
        X_centered = X - X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1.0
        X_std = X_centered / stds
        try:
            U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        var_ratio = (S ** 2) / (S ** 2).sum()
        cum_var = np.cumsum(var_ratio)

        # Variance plot
        ax = axes2[0]
        ax.plot(range(1, len(var_ratio) + 1), cum_var, marker="o", ms=3, color=color, label=label)
        ax.axhline(0.9, color="gray", ls="--", lw=0.7)
        n90 = int(np.searchsorted(cum_var, 0.9) + 1)
        ax.axvline(n90, color=color, ls=":", lw=0.7)
        ax.set_xlabel("Component")
        ax.set_ylabel("Cumulative variance explained")
        ax.set_title("PCA variance explained")
        ax.legend()

        # Scatter of first 2 PCs
        ax2 = axes2[1]
        pc = X_std @ Vt[:2, :].T
        ax2.scatter(pc[:, 0], pc[:, 1], alpha=0.3, s=5, color=color, label=label)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("First 2 principal components")
        ax2.legend()

    fig2.suptitle("TASK_1 Rate PCA: SIM vs REAL", fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    figs.append(fig2)

    return figs


# =============================================================================
# Section 4: Strip & Plane Rate Analysis
# =============================================================================

def analyze_strips_planes(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []
    k0 = "MINGO00_TASK_1"
    k1 = "MINGO01_TASK_1"
    if k0 not in data or k1 not in data:
        return figs

    df0, df1 = data[k0], data[k1]

    # -- Zeroed percentage by strip --
    zp_cols = sorted([c for c in df0.columns if c.startswith("zeroed_percentage_")])
    zp_cols = [c for c in zp_cols if c in df1.columns]
    if zp_cols:
        means0 = [df0[c].mean() for c in zp_cols]
        means1 = [df1[c].mean() for c in zp_cols]
        stds1 = [df1[c].std() for c in zp_cols]
        labels = [c.replace("zeroed_percentage_", "") for c in zp_cols]

        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, means0, w, color=COLOR_SIM, label="SIM (MINGO00)", alpha=0.8)
        ax.bar(x + w / 2, means1, w, color=COLOR_REAL, label="REAL (MINGO01)", alpha=0.8,
               yerr=stds1, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean zeroed percentage (%)")
        ax.set_title("Zeroed Percentage by Strip (SIM vs REAL)")
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

        lines = ["Mean zeroed percentage by strip:"]
        lines.append(f"  {'Strip':<10s} {'SIM':>8s} {'REAL':>8s} {'REAL std':>8s}")
        for lbl, m0, m1, s1 in zip(labels, means0, means1, stds1):
            lines.append(f"  {lbl:<10s} {m0:8.3f} {m1:8.3f} {s1:8.3f}")
        report.add_section("ZEROED PERCENTAGE BY STRIP", "\n".join(lines))

    # -- Original vs final rate gap (MINGO01 only) --
    orig_cols = sorted([c for c in df1.columns if "_original_rate_hz" in c])
    final_cols = sorted([c for c in df1.columns if "_final_rate_hz" in c])

    if orig_cols and final_cols:
        # Match original/final pairs
        pairs = []
        for oc in orig_cols:
            fc = oc.replace("_original_", "_final_")
            if fc in final_cols:
                pairs.append((oc, fc))

        if pairs:
            gap_means = []
            gap_labels = []
            for oc, fc in pairs:
                orig_vals = df1[oc].values
                final_vals = df1[fc].values
                mask = orig_vals > 0
                if mask.sum() == 0:
                    continue
                gap = (orig_vals[mask] - final_vals[mask]) / orig_vals[mask] * 100
                gap_means.append(np.nanmean(gap))
                short = oc.replace("_entries_original_rate_hz", "")
                gap_labels.append(short)

            if gap_means:
                fig, ax = plt.subplots(figsize=(16, 5))
                colors = [COLOR_REAL if g > 5 else "#888888" for g in gap_means]
                ax.bar(range(len(gap_means)), gap_means, color=colors, alpha=0.8)
                ax.set_xticks(range(len(gap_labels)))
                ax.set_xticklabels(gap_labels, rotation=90, fontsize=6)
                ax.set_ylabel("Rate reduction (%)")
                ax.set_title("MINGO01: Rate reduction from original to final (% cleaned away)")
                ax.axhline(0, color="black", lw=0.5)
                fig.tight_layout()
                figs.append(fig)

                lines = ["Rate reduction (original -> final) for MINGO01:"]
                for lbl, g in sorted(zip(gap_labels, gap_means), key=lambda x: -x[1])[:20]:
                    lines.append(f"  {lbl:<25s} {g:+8.2f}%")
                report.add_section("RATE CLEANING GAP (MINGO01)", "\n".join(lines))

    # -- Per-plane average rates --
    planes = ["T1", "T2", "T3", "T4", "Q1", "Q2", "Q3", "Q4"]
    final_rate_cols = _get_rate_cols(df0, r"_final_rate_hz$")
    final_rate_cols = [c for c in final_rate_cols if c in df1.columns]

    plane_means_sim = []
    plane_means_real = []
    plane_labels = []
    for p in planes:
        pcols = [c for c in final_rate_cols if c.startswith(f"{p}_")]
        if not pcols:
            continue
        plane_labels.append(p)
        plane_means_sim.append(np.nanmean([df0[c].mean() for c in pcols]))
        plane_means_real.append(np.nanmean([df1[c].mean() for c in pcols]))

    if plane_labels:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(plane_labels))
        w = 0.35
        ax.bar(x - w / 2, plane_means_sim, w, color=COLOR_SIM, label="SIM", alpha=0.8)
        ax.bar(x + w / 2, plane_means_real, w, color=COLOR_REAL, label="REAL", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(plane_labels)
        ax.set_ylabel("Mean final rate (Hz)")
        ax.set_title("Per-plane average final rates (SIM vs REAL)")
        ax.legend()
        fig.tight_layout()
        figs.append(fig)

    # -- CRT timing (TASK_2) --
    k0_t2 = "MINGO00_TASK_2"
    k1_t2 = "MINGO01_TASK_2"
    if k0_t2 in data and k1_t2 in data:
        df0_t2, df1_t2 = data[k0_t2], data[k1_t2]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for idx, col in enumerate(["CRT_avg", "CRT_std"]):
            ax = axes[idx]
            if col in df0_t2.columns:
                v0 = df0_t2[col].dropna().values
                if len(v0) > 2:
                    ax.hist(v0, bins=30, alpha=0.5, color=COLOR_SIM, density=True, label=f"SIM (n={len(v0)})")
            if col in df1_t2.columns:
                v1 = df1_t2[col].dropna().values
                if len(v1) > 2:
                    ax.hist(v1, bins=30, alpha=0.5, color=COLOR_REAL, density=True, label=f"REAL (n={len(v1)})")
            ax.set_title(col)
            ax.set_xlabel("ns" if "avg" in col else "ns")
            ax.legend(fontsize=8)
        fig.suptitle("CRT Timing Distributions (TASK_2)", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        figs.append(fig)

        lines = ["CRT timing comparison (TASK_2):"]
        for col in ["CRT_avg", "CRT_std"]:
            for key, label in [(k0_t2, "SIM"), (k1_t2, "REAL")]:
                if col in data[key].columns:
                    vals = data[key][col].dropna()
                    lines.append(f"  {label} {col}: n={len(vals)}, mean={vals.mean():.2f}, std={vals.std():.2f}, "
                                 f"min={vals.min():.2f}, max={vals.max():.2f}")
        report.add_section("CRT TIMING ANALYSIS", "\n".join(lines))

    return figs


# =============================================================================
# Section 5: Temporal Stability
# =============================================================================

def analyze_temporal(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []
    representative_cols = {
        1: r"_final_rate_hz$",
        2: r"^P\d_s\d_entries_original_rate_hz$",
        3: r"_1111_rate_hz$",
        4: r"sigmoid_amplitude_",
    }

    for tid in TASK_IDS:
        k0 = f"MINGO00_TASK_{tid}"
        k1 = f"MINGO01_TASK_{tid}"
        if k0 not in data or k1 not in data:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
        pattern = representative_cols.get(tid, r"_rate_hz$")

        for idx, (key, label, color) in enumerate([(k0, "SIM", COLOR_SIM), (k1, "REAL", COLOR_REAL)]):
            df = data[key].copy()
            ax = axes[idx]
            ts_col = "_exec_ts"
            df = df.dropna(subset=[ts_col]).sort_values(ts_col)
            if len(df) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                continue

            rate_cols = _get_rate_cols(df, pattern)
            if not rate_cols:
                rate_cols = _get_rate_cols(df, r"_rate_hz$")
            if not rate_cols:
                continue

            avg_rate = df[rate_cols].mean(axis=1).values
            times = df[ts_col].values

            win = min(20, max(3, len(df) // 5))
            rolling_mean = pd.Series(avg_rate).rolling(win, center=True, min_periods=1).mean().values
            rolling_std = pd.Series(avg_rate).rolling(win, center=True, min_periods=1).std().values
            rolling_std = np.nan_to_num(rolling_std, nan=0.0)

            ax.plot(times, avg_rate, ".", ms=2, color=color, alpha=0.5)
            ax.plot(times, rolling_mean, "-", color=color, lw=1.5, label=f"{label} rolling mean")
            ax.fill_between(times, rolling_mean - rolling_std, rolling_mean + rolling_std,
                            alpha=0.15, color=color)

            # Flag anomalies (>3sigma)
            if np.any(rolling_std > 0):
                anomaly = np.abs(avg_rate - rolling_mean) > 3 * rolling_std
                n_anom = anomaly.sum()
                if n_anom > 0:
                    ax.plot(np.array(times)[anomaly], avg_rate[anomaly], "x", color="black", ms=5,
                            label=f"Anomalies ({n_anom})")

            ax.set_ylabel("Mean rate (Hz)")
            ax.set_title(f"{label} - TASK_{tid}")
            ax.legend(fontsize=7)
            ax.tick_params(axis="x", rotation=30, labelsize=7)

        fig.suptitle(f"TASK_{tid}: Temporal Rate Stability", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        figs.append(fig)

    # Autocorrelation for MINGO01
    fig_ac, axes_ac = plt.subplots(2, 2, figsize=(14, 8))
    for tidx, tid in enumerate(TASK_IDS):
        k1 = f"MINGO01_TASK_{tid}"
        ax = axes_ac[tidx // 2, tidx % 2]
        if k1 not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        df = data[k1]
        pattern = representative_cols.get(tid, r"_rate_hz$")
        rate_cols = _get_rate_cols(df, pattern)
        if not rate_cols:
            rate_cols = _get_rate_cols(df, r"_rate_hz$")
        if not rate_cols or len(df) < 5:
            continue
        avg_rate = df[rate_cols].mean(axis=1).dropna().values
        if len(avg_rate) < 5:
            continue
        avg_rate = avg_rate - avg_rate.mean()
        norm = np.sum(avg_rate ** 2)
        if norm == 0:
            continue
        acorr = np.correlate(avg_rate, avg_rate, mode="full")
        acorr = acorr[len(acorr) // 2:]
        acorr = acorr / norm
        max_lag = min(len(acorr), 50)
        ax.bar(range(max_lag), acorr[:max_lag], color=COLOR_REAL, alpha=0.7, width=1.0)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(2 / np.sqrt(len(avg_rate)), color="gray", ls="--", lw=0.7)
        ax.axhline(-2 / np.sqrt(len(avg_rate)), color="gray", ls="--", lw=0.7)
        ax.set_title(f"TASK_{tid}", fontsize=9)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
    fig_ac.suptitle("MINGO01 Rate Autocorrelation", fontsize=12, fontweight="bold")
    fig_ac.tight_layout(rect=[0, 0, 1, 0.94])
    figs.append(fig_ac)

    return figs


# =============================================================================
# Section 6: Active Strip Pattern Analysis (TASK_3)
# =============================================================================

def analyze_strip_patterns(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []
    k0 = "MINGO00_TASK_3"
    k1 = "MINGO01_TASK_3"
    if k0 not in data or k1 not in data:
        return figs

    df0, df1 = data[k0], data[k1]

    single_patterns = ["1000", "0100", "0010", "0001"]
    pair_patterns = ["1100", "0110", "0011", "1010", "1001", "0101"]
    triple_patterns = ["1110", "1101", "1011", "0111"]
    quad_patterns = ["1111"]
    all_patterns = single_patterns + pair_patterns + triple_patterns + quad_patterns

    planes = ["P1", "P2", "P3", "P4"]

    # Multiplicity fractions per plane
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    lines = ["Strip activation multiplicity fractions:\n"]
    lines.append(f"  {'Plane':<6s} {'Station':<8s} {'Single':>8s} {'Pair':>8s} {'Triple':>8s} {'Quad':>8s} {'S/(S+P+T+Q)':>12s}")

    for pidx, plane in enumerate(planes):
        ax = axes[pidx]
        for key, label, color in [(k0, "SIM", COLOR_SIM), (k1, "REAL", COLOR_REAL)]:
            df = data[key]
            groups = {"Single": 0.0, "Pair": 0.0, "Triple": 0.0, "Quad": 0.0}
            for pat in single_patterns:
                col = f"active_strips_{plane}_{pat}_rate_hz"
                if col in df.columns:
                    groups["Single"] += df[col].mean()
            for pat in pair_patterns:
                col = f"active_strips_{plane}_{pat}_rate_hz"
                if col in df.columns:
                    groups["Pair"] += df[col].mean()
            for pat in triple_patterns:
                col = f"active_strips_{plane}_{pat}_rate_hz"
                if col in df.columns:
                    groups["Triple"] += df[col].mean()
            for pat in quad_patterns:
                col = f"active_strips_{plane}_{pat}_rate_hz"
                if col in df.columns:
                    groups["Quad"] += df[col].mean()

            total = sum(groups.values())
            if total == 0:
                continue
            fracs = {k: v / total for k, v in groups.items()}
            noise_ratio = fracs["Single"]

            lines.append(
                f"  {plane:<6s} {label:<8s} {fracs['Single']:8.4f} {fracs['Pair']:8.4f} "
                f"{fracs['Triple']:8.4f} {fracs['Quad']:8.4f} {noise_ratio:12.4f}"
            )

            bottom = 0.0
            colors_mult = {"Single": "#e74c3c", "Pair": "#f39c12", "Triple": "#2ecc71", "Quad": "#3498db"}
            offset = -0.2 if label == "SIM" else 0.2
            for gname in ["Single", "Pair", "Triple", "Quad"]:
                ax.bar(offset, fracs[gname], width=0.35, bottom=bottom,
                       color=colors_mult[gname], alpha=0.8 if label == "REAL" else 0.5,
                       edgecolor="black" if label == "REAL" else "none", linewidth=0.5)
                bottom += fracs[gname]

        ax.set_title(plane, fontsize=10)
        ax.set_xticks([-0.2, 0.2])
        ax.set_xticklabels(["SIM", "REAL"], fontsize=8)
        if pidx == 0:
            ax.set_ylabel("Fraction")

    # Add manual legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=n) for n, c in
                      {"Single": "#e74c3c", "Pair": "#f39c12", "Triple": "#2ecc71", "Quad": "#3498db"}.items()]
    axes[-1].legend(handles=legend_handles, fontsize=7, loc="upper right")

    fig.suptitle("TASK_3: Strip Activation Multiplicity (SIM vs REAL)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figs.append(fig)
    report.add_section("STRIP PATTERN ANALYSIS", "\n".join(lines))

    # Per-pattern rate comparison for P1
    p1_patterns = [f"active_strips_P1_{p}_rate_hz" for p in all_patterns]
    p1_patterns = [c for c in p1_patterns if c in df0.columns and c in df1.columns]
    if p1_patterns:
        fig2, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(p1_patterns))
        w = 0.35
        m0 = [df0[c].mean() for c in p1_patterns]
        m1 = [df1[c].mean() for c in p1_patterns]
        ax.bar(x - w / 2, m0, w, color=COLOR_SIM, label="SIM", alpha=0.8)
        ax.bar(x + w / 2, m1, w, color=COLOR_REAL, label="REAL", alpha=0.8)
        labels_pat = [c.replace("active_strips_P1_", "").replace("_rate_hz", "") for c in p1_patterns]
        ax.set_xticks(x)
        ax.set_xticklabels(labels_pat, rotation=45, ha="right")
        ax.set_ylabel("Mean rate (Hz)")
        ax.set_title("TASK_3: P1 strip pattern rates (SIM vs REAL)")
        ax.legend()
        fig2.tight_layout()
        figs.append(fig2)

    return figs


# =============================================================================
# Section 7: Fit Quality Analysis (TASK_4)
# =============================================================================

def analyze_fit_quality(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []
    k0 = "MINGO00_TASK_4"
    k1 = "MINGO01_TASK_4"
    if k0 not in data or k1 not in data:
        return figs

    df0, df1 = data[k0], data[k1]

    # Identify trigger combinations from sigmoid columns
    sig_cols0 = sorted([c for c in df0.columns if c.startswith("sigmoid_amplitude_")])
    combos = sorted(set(c.replace("sigmoid_amplitude_", "") for c in sig_cols0))

    # Report missing columns
    cols0_set = set(df0.columns)
    cols1_set = set(df1.columns)
    missing_in_real = sorted(cols0_set - cols1_set - {"_exec_ts", "_data_ts"})
    if missing_in_real:
        lines = [f"Columns in SIM but missing from REAL ({len(missing_in_real)}):"]
        for c in missing_in_real:
            lines.append(f"  {c}")
        report.add_section("TASK_4 MISSING COLUMNS IN REAL DATA", "\n".join(lines))

    # Sigmoid amplitude comparison
    shared_combos = [c for c in combos if f"sigmoid_amplitude_{c}" in df1.columns]
    if shared_combos:
        fig, ax = plt.subplots(figsize=(14, 5))
        positions_sim = []
        positions_real = []
        bp_data_sim = []
        bp_data_real = []
        labels_combo = []
        for i, combo in enumerate(shared_combos):
            col = f"sigmoid_amplitude_{combo}"
            v0 = df0[col].dropna().values
            v1 = df1[col].dropna().values
            if len(v0) > 0:
                bp_data_sim.append(v0)
                positions_sim.append(i * 3)
            if len(v1) > 0:
                bp_data_real.append(v1)
                positions_real.append(i * 3 + 1)
            labels_combo.append(combo)

        if bp_data_sim:
            bp1 = ax.boxplot(bp_data_sim, positions=positions_sim, widths=0.8,
                             patch_artist=True, showfliers=False)
            for patch in bp1["boxes"]:
                patch.set_facecolor(COLOR_SIM)
                patch.set_alpha(0.6)
        if bp_data_real:
            bp2 = ax.boxplot(bp_data_real, positions=positions_real, widths=0.8,
                             patch_artist=True, showfliers=False)
            for patch in bp2["boxes"]:
                patch.set_facecolor(COLOR_REAL)
                patch.set_alpha(0.6)

        ax.set_xticks([i * 3 + 0.5 for i in range(len(labels_combo))])
        ax.set_xticklabels(labels_combo, rotation=45, ha="right")
        ax.set_ylabel("Sigmoid amplitude")
        ax.set_title("TASK_4: Sigmoid Amplitude by Trigger Combination (SIM vs REAL)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=COLOR_SIM, alpha=0.6, label="SIM"),
                           Patch(color=COLOR_REAL, alpha=0.6, label="REAL")])
        fig.tight_layout()
        figs.append(fig)

    # Convergence failure analysis
    fit_params = ["sigmoid_width", "sigmoid_amplitude", "sigmoid_center",
                  "background_slope", "fit_normalization"]
    lines = ["Near-zero value counts (|val| < 1e-10) per trigger combination:\n"]
    lines.append(f"  {'Combo':<8s} {'Param':<25s} {'SIM':>6s} {'REAL':>6s}")
    convergence_data = []
    for combo in shared_combos:
        for param in fit_params:
            col = f"{param}_{combo}"
            if col not in df0.columns or col not in df1.columns:
                continue
            n0 = int((df0[col].abs() < 1e-10).sum())
            n1 = int((df1[col].abs() < 1e-10).sum())
            if n0 > 0 or n1 > 0:
                lines.append(f"  {combo:<8s} {param:<25s} {n0:6d} {n1:6d}")
                convergence_data.append((combo, param, n0, n1))
    report.add_section("FIT CONVERGENCE ANALYSIS", "\n".join(lines))

    # Resolution comparison
    res_cols_shared = sorted([c for c in df0.columns if c.startswith("res_") and c.endswith("_sigma") and c in df1.columns])
    if res_cols_shared:
        n_res = min(len(res_cols_shared), 12)
        res_to_plot = res_cols_shared[:n_res]
        n_rows_fig = (n_res + 2) // 3
        fig, axes = plt.subplots(n_rows_fig, 3, figsize=(15, 4 * n_rows_fig))
        axes = np.atleast_2d(axes)
        for idx, col in enumerate(res_to_plot):
            ax = axes[idx // 3, idx % 3]
            v0 = df0[col].dropna().values
            v1 = df1[col].dropna().values
            if len(v0) > 0:
                ax.hist(v0, bins=30, alpha=0.5, color=COLOR_SIM, density=True, label="SIM")
            if len(v1) > 0:
                ax.hist(v1, bins=30, alpha=0.5, color=COLOR_REAL, density=True, label="REAL")
            short_name = col.replace("res_", "").replace("_sigma", "")
            ax.set_title(short_name, fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
        for idx in range(n_res, n_rows_fig * 3):
            axes[idx // 3, idx % 3].set_visible(False)
        fig.suptitle("TASK_4: Resolution Comparison (SIM vs REAL)", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        figs.append(fig)

    return figs


# =============================================================================
# Section 8: Composite Noise Summary
# =============================================================================

def analyze_noise_summary(data: dict, report: ReportWriter) -> list[plt.Figure]:
    figs = []
    planes = ["P1", "P2", "P3", "P4"]
    strips = ["s1", "s2", "s3", "s4"]
    noise_scores = np.zeros((4, 4))  # planes x strips

    k0_t1 = "MINGO00_TASK_1"
    k1_t1 = "MINGO01_TASK_1"
    k0_t3 = "MINGO00_TASK_3"
    k1_t3 = "MINGO01_TASK_3"

    # Component 1: zeroed_percentage (TASK_1)
    if k1_t1 in data:
        df1 = data[k1_t1]
        for pi, plane in enumerate(planes):
            for si, strip in enumerate(strips):
                col = f"zeroed_percentage_{plane}{strip}"
                if col in df1.columns:
                    noise_scores[pi, si] += df1[col].mean()

    # Component 2: single-strip noise ratio (TASK_3)
    if k0_t3 in data and k1_t3 in data:
        single_patterns = ["1000", "0100", "0010", "0001"]
        all_patterns_list = (
            single_patterns
            + ["1100", "0110", "0011", "1010", "1001", "0101"]
            + ["1110", "1101", "1011", "0111"]
            + ["1111"]
        )
        for pi, plane in enumerate(planes):
            sim_single = 0.0
            sim_total = 0.0
            real_single = 0.0
            real_total = 0.0
            for pat in all_patterns_list:
                col = f"active_strips_{plane}_{pat}_rate_hz"
                if col in data[k0_t3].columns:
                    sim_total += data[k0_t3][col].mean()
                    if pat in single_patterns:
                        sim_single += data[k0_t3][col].mean()
                if col in data[k1_t3].columns:
                    real_total += data[k1_t3][col].mean()
                    if pat in single_patterns:
                        real_single += data[k1_t3][col].mean()
            sim_ratio = sim_single / sim_total if sim_total > 0 else 0
            real_ratio = real_single / real_total if real_total > 0 else 0
            excess = max(0, (real_ratio - sim_ratio)) * 100
            for si in range(4):
                noise_scores[pi, si] += excess

    # Component 3: rate deviation from simulation (TASK_1)
    if k0_t1 in data and k1_t1 in data:
        df0 = data[k0_t1]
        df1 = data[k1_t1]
        for pi, plane in enumerate(planes):
            for si, strip in enumerate(strips):
                t_layer = f"T{pi+1}"
                q_layer = f"Q{pi+1}"
                for layer in [t_layer, q_layer]:
                    for side in ["F", "B"]:
                        col = f"{layer}_{side}_{si+1}_entries_final_rate_hz"
                        if col in df0.columns and col in df1.columns:
                            sim_mean = df0[col].mean()
                            real_mean = df1[col].mean()
                            if sim_mean > 0:
                                deviation = abs(real_mean - sim_mean) / sim_mean * 100
                                noise_scores[pi, si] += deviation * 0.1

    # Normalize scores
    if noise_scores.max() > 0:
        noise_scores_norm = noise_scores / noise_scores.max()
    else:
        noise_scores_norm = noise_scores

    # Noise heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(noise_scores_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(strips)
    ax.set_yticks(range(4))
    ax.set_yticklabels(planes)
    ax.set_xlabel("Strip")
    ax.set_ylabel("Plane")
    ax.set_title("Composite Noise Score Heatmap (MINGO01 vs MINGO00)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{noise_scores[i, j]:.1f}", ha="center", va="center",
                    fontsize=10, color="white" if noise_scores_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Normalized noise score")
    fig.tight_layout()
    figs.append(fig)

    # Ranked bar chart
    elements = []
    for pi, plane in enumerate(planes):
        for si, strip in enumerate(strips):
            elements.append((f"{plane}{strip}", noise_scores[pi, si]))
    elements.sort(key=lambda x: -x[1])

    fig2, ax = plt.subplots(figsize=(12, 5))
    names = [e[0] for e in elements]
    scores = [e[1] for e in elements]
    colors = [COLOR_REAL if s > np.median(scores) else "#888888" for s in scores]
    ax.bar(range(len(names)), scores, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Noise score")
    ax.set_title("Noise Ranking by Detector Element (MINGO01)")
    fig2.tight_layout()
    figs.append(fig2)

    # Report
    lines = ["COMPOSITE NOISE RANKING (highest = noisiest):\n"]
    lines.append(f"  {'Element':<10s} {'Score':>8s}")
    for name, score in elements:
        lines.append(f"  {name:<10s} {score:8.2f}")

    lines.append("\n\nRECOMMENDATIONS:")
    lines.append("=" * 40)
    top3 = elements[:3]
    for name, score in top3:
        lines.append(f"  - {name}: Score {score:.1f}. Investigate hardware, apply stricter cuts.")

    if k1_t1 in data:
        df1 = data[k1_t1]
        zp_p1s4 = df1.get("zeroed_percentage_P1s4")
        if zp_p1s4 is not None and zp_p1s4.mean() > 5:
            lines.append(f"  - P1s4 has {zp_p1s4.mean():.1f}% mean zeroed channels — possible HV or gas issue.")

    lines.append("  - Single-strip activations without coincidence may indicate electronic noise or cross-talk.")
    lines.append("  - Monitor CRT timing drift for calibration stability.")
    lines.append("  - Columns all-zero in simulation but non-zero in real data indicate")
    lines.append("    phenomena not modeled by the digital twin (e.g., dark counts, afterpulses).")

    report.add_section("COMPOSITE NOISE SUMMARY & RECOMMENDATIONS", "\n".join(lines))

    return figs


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Noise study: SIM vs REAL metadata comparison")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading data from MINGO00 and MINGO01...")
    data = load_all_data()

    if not data:
        LOG.error("No data loaded — check file paths.")
        sys.exit(1)

    report = ReportWriter()
    report.add_section(
        "NOISE STUDY: SIMULATED (MINGO00) vs REAL (MINGO01)",
        "Comparing task_*_metadata_specific.csv across TASK 1-4\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    all_figs = []
    fig_names = []

    # Section 1: Data Quality
    LOG.info("Section 1: Data quality overview...")
    figs = analyze_data_quality(data, report)
    for f in figs:
        all_figs.append(f)
        fig_names.append("noise_study_01_data_quality.png")

    # Section 2: Distribution Divergence
    LOG.info("Section 2: Distribution divergence (KS tests)...")
    figs = analyze_distributions(data, report)
    for i, f in enumerate(figs):
        all_figs.append(f)
        fig_names.append(f"noise_study_02_ks_distributions_task{i+1}.png")

    # Section 3: Correlations & PCA
    LOG.info("Section 3: Correlation & PCA analysis...")
    figs = analyze_correlations(data, report)
    names_corr = ["noise_study_03_correlation_diff.png", "noise_study_03_pca.png"]
    for i, f in enumerate(figs):
        all_figs.append(f)
        fig_names.append(names_corr[i] if i < len(names_corr) else f"noise_study_03_extra_{i}.png")

    # Section 4: Strip & Plane rates
    LOG.info("Section 4: Strip & plane rate analysis...")
    figs = analyze_strips_planes(data, report)
    names_strip = [
        "noise_study_04_zeroed_percentage.png",
        "noise_study_04_rate_gap.png",
        "noise_study_04_plane_rates.png",
        "noise_study_04_crt.png",
    ]
    for i, f in enumerate(figs):
        all_figs.append(f)
        fig_names.append(names_strip[i] if i < len(names_strip) else f"noise_study_04_extra_{i}.png")

    # Section 5: Temporal stability
    LOG.info("Section 5: Temporal stability...")
    figs = analyze_temporal(data, report)
    for i, f in enumerate(figs):
        all_figs.append(f)
        if i < 4:
            fig_names.append(f"noise_study_05_temporal_task{i+1}.png")
        else:
            fig_names.append("noise_study_05_autocorrelation.png")

    # Section 6: Strip patterns
    LOG.info("Section 6: Active strip pattern analysis...")
    figs = analyze_strip_patterns(data, report)
    names_pat = ["noise_study_06_strip_multiplicity.png", "noise_study_06_pattern_rates_p1.png"]
    for i, f in enumerate(figs):
        all_figs.append(f)
        fig_names.append(names_pat[i] if i < len(names_pat) else f"noise_study_06_extra_{i}.png")

    # Section 7: Fit quality
    LOG.info("Section 7: Fit quality analysis...")
    figs = analyze_fit_quality(data, report)
    names_fit = ["noise_study_07_fit_quality.png", "noise_study_07_resolution.png"]
    for i, f in enumerate(figs):
        all_figs.append(f)
        fig_names.append(names_fit[i] if i < len(names_fit) else f"noise_study_07_extra_{i}.png")

    # Section 8: Composite noise summary
    LOG.info("Section 8: Composite noise summary...")
    figs = analyze_noise_summary(data, report)
    names_sum = ["noise_study_08_noise_heatmap.png", "noise_study_08_noise_ranking.png"]
    for i, f in enumerate(figs):
        all_figs.append(f)
        fig_names.append(names_sum[i] if i < len(names_sum) else f"noise_study_08_extra_{i}.png")

    # Save outputs
    if not args.no_plots:
        # Save individual PNGs
        for fig, name in zip(all_figs, fig_names):
            path = PLOTS_DIR / name
            fig.savefig(path, dpi=150, bbox_inches="tight")
            LOG.info("Saved %s", path.name)

        # Save multi-page PDF
        pdf_path = PLOTS_DIR / "noise_study_all.pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in all_figs:
                pdf.savefig(fig, bbox_inches="tight")
        LOG.info("Saved %s", pdf_path.name)

    # Save report
    report_path = FILES_DIR / "noise_study_report.txt"
    report.save(report_path)

    plt.close("all")
    LOG.info("Noise study complete. %d figures generated.", len(all_figs))


if __name__ == "__main__":
    main()
