#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/ANCILLARY/task3_efficiency_noise_decomposition.py
Purpose: Count-level decomposition of efficiency vs threshold to separate signal from noise.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-17
Runtime: python3
Usage: python3 MASTER/ANCILLARY/task3_efficiency_noise_decomposition.py
Inputs: Task 3 OUTPUT_FILES listed_*.parquet from MINGO00 (sim) and MINGO01 (real).
Outputs: PNG plots in MASTER/ANCILLARY/OUTPUTS/efficiency_decomposition/
Notes: Quick ancillary study — loads first available parquet from each station.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

OUT_DIR = SCRIPT_DIR / "OUTPUTS" / "efficiency_decomposition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_STATION = "MINGO00"
REAL_STATION = "MINGO01"

TASK3_REL = Path("STAGE_1/EVENT_DATA/STEP_1/TASK_3/OUTPUT_FILES")
# Fallback: listed files may have been consumed downstream by Task 4
TASK4_INPUT_REL = Path("STAGE_1/EVENT_DATA/STEP_1/TASK_4/INPUT_FILES/COMPLETED_DIRECTORY")


def find_parquets(station: str) -> list[Path]:
    """Search Task 3 OUTPUT_FILES first; fall back to Task 4 INPUT_FILES/COMPLETED_DIRECTORY."""
    for rel_path in (TASK3_REL, TASK4_INPUT_REL):
        station_dir = REPO_ROOT / "STATIONS" / station / rel_path
        if station_dir.exists():
            matches = sorted(
                station_dir.glob("listed_*.parquet"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return matches
    return []


def load_first_parquet(station: str) -> pd.DataFrame | None:
    paths = find_parquets(station)
    if not paths:
        print(f"  No Task 3 parquets found for {station}")
        return None
    path = paths[0]
    print(f"  Loading {station}: {path.name}")
    df = pd.read_parquet(path)
    print(f"    Events: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Core: compute N(1234) and N(missing_k) as a function of Q threshold
# ---------------------------------------------------------------------------
Q_COLS = [f"P{p}_Q_sum_final" for p in range(1, 5)]
MISSING_TT = {1: 234, 2: 134, 3: 124, 4: 123}


def compute_threshold_tt(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Recompute TT using only planes with Q_sum > threshold."""
    tt_str = pd.Series("", index=df.index, dtype="object")
    for p in range(1, 5):
        col = f"P{p}_Q_sum_final"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        active = vals.gt(float(threshold))
        tt_str = tt_str.where(~active, tt_str + str(p))
    return tt_str.replace("", "0").astype(int)


def build_count_curves(df: pd.DataFrame, thresholds: np.ndarray):
    """Return N(1234, Q) and N(missing_k, Q) for each plane k, at each threshold."""
    n4 = np.zeros(len(thresholds))
    n3 = {k: np.zeros(len(thresholds)) for k in range(1, 5)}

    for i, thr in enumerate(thresholds):
        tt = compute_threshold_tt(df, thr)
        counts = tt.value_counts()
        n4[i] = counts.get(1234, 0)
        for k in range(1, 5):
            n3[k][i] = counts.get(MISSING_TT[k], 0)

    return n4, n3


def compute_efficiency_user(n4: np.ndarray, n3k: np.ndarray) -> np.ndarray:
    """ε = 1 - N(3-plane missing k) / N(1234)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = 1.0 - np.divide(n3k, n4, out=np.full_like(n4, np.nan, dtype=float), where=n4 > 0)
    return eff


def compute_efficiency_standard(n4: np.ndarray, n3k: np.ndarray) -> np.ndarray:
    """ε_std = N(1234) / (N(1234) + N(missing k))."""
    denom = n4 + n3k
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.divide(n4, denom, out=np.full_like(n4, np.nan, dtype=float), where=denom > 0)
    return eff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cleaned output directory: {OUT_DIR}")

    print("Loading data...")
    df_sim = load_first_parquet(SIM_STATION)
    df_real = load_first_parquet(REAL_STATION)

    if df_sim is None or df_real is None:
        print("Cannot proceed without data from both stations.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Build QUANTILE-based thresholds so charge scales are matched.
    # For each dataset, collect all positive Q_sum values, then use the
    # same percentile grid to determine station-specific absolute thresholds.
    # -----------------------------------------------------------------------
    percentiles = np.linspace(0, 95, 30)

    def pooled_positive_charges(df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in Q_COLS:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
                parts.append(v[v > 0])
        return np.concatenate(parts) if parts else np.array([])

    q_sim_pool = pooled_positive_charges(df_sim)
    q_real_pool = pooled_positive_charges(df_real)

    thr_sim = np.percentile(q_sim_pool, percentiles) if len(q_sim_pool) else percentiles
    thr_real = np.percentile(q_real_pool, percentiles) if len(q_real_pool) else percentiles

    print(f"Quantile grid: {len(percentiles)} points from p0 to p95")
    print(f"  Sim  absolute range: {thr_sim[0]:.2f} – {thr_sim[-1]:.2f}")
    print(f"  Real absolute range: {thr_real[0]:.2f} – {thr_real[-1]:.2f}")

    # Compute count curves — each dataset uses its OWN absolute thresholds
    print("Computing count curves for simulation...")
    n4_sim, n3_sim = build_count_curves(df_sim, thr_sim)
    print("Computing count curves for real data...")
    n4_real, n3_real = build_count_curves(df_real, thr_real)

    # Also compute with a shared absolute grid for the direct overlay
    q95_union = max(float(np.percentile(q_sim_pool, 95)), float(np.percentile(q_real_pool, 95)))
    thresholds_abs = np.linspace(0, q95_union, 30)
    print(f"Also computing absolute grid: 0 – {q95_union:.1f}")
    n4_sim_abs, n3_sim_abs = build_count_curves(df_sim, thresholds_abs)
    n4_real_abs, n3_real_abs = build_count_curves(df_real, thresholds_abs)

    plane_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    sim_file_name = find_parquets(SIM_STATION)[0].name
    real_file_name = find_parquets(REAL_STATION)[0].name

    # -----------------------------------------------------------------------
    # FIGURE 1: QUANTILE-NORMALIZED count survival and efficiency
    # x-axis = charge percentile (matched scale)
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # Top-left: N(1234) survival
    ax = axes1[0, 0]
    if n4_sim[0] > 0:
        ax.plot(percentiles, n4_sim / n4_sim[0], "o-", color="steelblue", ms=4, lw=1.5,
                label=f"{SIM_STATION} (sim, N₀={int(n4_sim[0])})")
    if n4_real[0] > 0:
        ax.plot(percentiles, n4_real / n4_real[0], "s-", color="firebrick", ms=4, lw=1.5,
                label=f"{REAL_STATION} (real, N₀={int(n4_real[0])})")
    ax.set_ylabel("N(1234, Q) / N(1234, 0)")
    ax.set_title("4-plane count survival fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: N(missing k) survival
    ax = axes1[0, 1]
    for k in range(1, 5):
        if n3_sim[k][0] > 0:
            ax.plot(percentiles, n3_sim[k] / n3_sim[k][0], "o-", color=plane_colors[k],
                    ms=3, lw=1, alpha=0.6, label=f"sim miss P{k} (N₀={int(n3_sim[k][0])})")
        if n3_real[k][0] > 0:
            ax.plot(percentiles, n3_real[k] / n3_real[k][0], "s--", color=plane_colors[k],
                    ms=3, lw=1, alpha=0.6, label=f"real miss P{k} (N₀={int(n3_real[k][0])})")
    ax.set_ylabel("N(missing k, Q) / N(missing k, 0)")
    ax.set_title("3-plane count survival (missing plane k)")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Survival ratio data/sim for N(1234)
    ax = axes1[1, 0]
    if n4_sim[0] > 0 and n4_real[0] > 0:
        f_sim = n4_sim / n4_sim[0]
        f_real = n4_real / n4_real[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(f_sim > 0.001, f_real / f_sim, np.nan)
        ax.plot(percentiles, ratio, "ko-", ms=4, lw=1.5)
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Charge percentile")
    ax.set_ylabel("f_real(p) / f_sim(p)")
    ax.set_title("Survival ratio data/sim for N(1234)")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Standard efficiency (quantile-normalized)
    ax = axes1[1, 1]
    for k in range(1, 5):
        eff_sim = compute_efficiency_standard(n4_sim, n3_sim[k])
        eff_real = compute_efficiency_standard(n4_real, n3_real[k])
        ax.plot(percentiles, eff_sim, "o-", color=plane_colors[k], ms=3, lw=1, alpha=0.6,
                label=f"sim P{k}")
        ax.plot(percentiles, eff_real, "s--", color=plane_colors[k], ms=3, lw=1, alpha=0.9,
                label=f"real P{k}")
    ax.set_xlabel("Charge percentile")
    ax.set_ylabel("ε_std = N(1234) / (N(1234) + N(miss k))")
    ax.set_title("Standard efficiency per plane")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig1.suptitle(
        f"QUANTILE-NORMALIZED decomposition: {SIM_STATION} vs {REAL_STATION}\n"
        f"x-axis = charge percentile (matches charge scales)  |  "
        f"Sim: {sim_file_name}  |  Real: {real_file_name}",
        fontsize=10,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = OUT_DIR / "01_quantile_normalized_decomposition.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Saved: {p1}")

    # -----------------------------------------------------------------------
    # FIGURE 2: Noise excess — Δf on quantile axis
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    if n4_sim[0] > 0 and n4_real[0] > 0:
        delta_f4 = (n4_real / n4_real[0]) - (n4_sim / n4_sim[0])
        ax.plot(percentiles, delta_f4, "ko-", ms=4, lw=1.5)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.fill_between(percentiles, 0, delta_f4, where=delta_f4 > 0, alpha=0.2, color="red",
                         label="Real retains more (noise excess?)")
        ax.fill_between(percentiles, 0, delta_f4, where=delta_f4 < 0, alpha=0.2, color="blue",
                         label="Sim retains more")
    ax.set_xlabel("Charge percentile")
    ax.set_ylabel("Δf = f_real − f_sim")
    ax.set_title("Survival fraction Δ for N(1234)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for k in range(1, 5):
        if n3_sim[k][0] > 0 and n3_real[k][0] > 0:
            delta_f3 = (n3_real[k] / n3_real[k][0]) - (n3_sim[k] / n3_sim[k][0])
            ax.plot(percentiles, delta_f3, "o-", color=plane_colors[k], ms=3, lw=1,
                    label=f"missing P{k}")
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Charge percentile")
    ax.set_ylabel("Δf = f_real − f_sim")
    ax.set_title("Survival fraction Δ for N(missing k)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.suptitle("Survival fraction differences (quantile-normalized)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = OUT_DIR / "02_noise_excess_delta_quantile.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {p2}")

    # -----------------------------------------------------------------------
    # FIGURE 3: Efficiency — ABSOLUTE thresholds (both on same Q axis)
    # This shows the raw shape difference including charge-scale effects.
    # -----------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    ax = axes3[0]
    for k in range(1, 5):
        eff_sim = compute_efficiency_standard(n4_sim_abs, n3_sim_abs[k])
        eff_real = compute_efficiency_standard(n4_real_abs, n3_real_abs[k])
        ax.plot(thresholds_abs, eff_sim, "o-", color=plane_colors[k], ms=3, lw=1, alpha=0.5,
                label=f"sim P{k}")
        ax.plot(thresholds_abs, eff_real, "s--", color=plane_colors[k], ms=3, lw=1.3,
                label=f"real P{k}")
    ax.set_xlabel("Absolute charge threshold Q")
    ax.set_ylabel("ε_std")
    ax.set_title("Standard efficiency — absolute Q scale")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes3[1]
    for k in range(1, 5):
        eff_sim = compute_efficiency_user(n4_sim_abs, n3_sim_abs[k])
        eff_real = compute_efficiency_user(n4_real_abs, n3_real_abs[k])
        ax.plot(thresholds_abs, eff_sim, "o-", color=plane_colors[k], ms=3, lw=1, alpha=0.5,
                label=f"sim P{k}")
        ax.plot(thresholds_abs, eff_real, "s--", color=plane_colors[k], ms=3, lw=1.3,
                label=f"real P{k}")
    ax.set_xlabel("Absolute charge threshold Q")
    ax.set_ylabel("ε = 1 − N(miss k)/N(1234)")
    ax.set_title("User formula — absolute Q scale")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig3.suptitle("Efficiency on ABSOLUTE charge scale (unmatched — shows scale difference)", fontsize=12)
    fig3.tight_layout(rect=[0, 0, 1, 0.92])
    p3 = OUT_DIR / "03_efficiency_absolute_scale.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    print(f"Saved: {p3}")

    # -----------------------------------------------------------------------
    # FIGURE 4: Weakest-plane charge for 4-plane events
    # -----------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, df, color) in zip(
        axes4, [(SIM_STATION, df_sim, "steelblue"), (REAL_STATION, df_real, "firebrick")]
    ):
        tt = compute_threshold_tt(df, 0.0)
        mask_4 = tt == 1234
        if mask_4.sum() == 0:
            ax.text(0.5, 0.5, "No 4-plane events", transform=ax.transAxes, ha="center")
            continue
        q_arr = df.loc[mask_4, Q_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        q_min = np.nanmin(q_arr, axis=1)
        q_min = q_min[np.isfinite(q_min)]
        q99 = float(np.percentile(q_min, 99))
        ax.hist(q_min, bins=80, range=(0, q99), color=color, alpha=0.7, edgecolor="none",
                density=True, label=f"N={len(q_min)}")
        ax.set_xlabel("Min Q_sum across 4 planes")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} — weakest plane charge in 4-plane events")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Mark percentiles
        for pct in (5, 10, 25):
            val = float(np.percentile(q_min, pct))
            ax.axvline(val, color="black", ls=":", lw=0.7, alpha=0.5)
            ax.text(val, ax.get_ylim()[1] * 0.9, f"p{pct}={val:.1f}", fontsize=7, rotation=90,
                    va="top", ha="right")

    fig4.suptitle(
        "Weakest-plane charge for 4-plane events\n"
        "Sim: single population (signal). Real: if bimodal, low-charge bump = noise/cross-talk.",
        fontsize=11,
    )
    fig4.tight_layout(rect=[0, 0, 1, 0.90])
    p4 = OUT_DIR / "04_weakest_plane_charge_4plane.png"
    fig4.savefig(p4, dpi=150)
    plt.close(fig4)
    print(f"Saved: {p4}")

    # -----------------------------------------------------------------------
    # FIGURE 5: Overlay weakest-plane charge (both on same axes, normalized)
    # -----------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    for label, df, color, ls in [
        (SIM_STATION, df_sim, "steelblue", "-"),
        (REAL_STATION, df_real, "firebrick", "--"),
    ]:
        tt = compute_threshold_tt(df, 0.0)
        mask_4 = tt == 1234
        if mask_4.sum() == 0:
            continue
        q_arr = df.loc[mask_4, Q_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        q_min = np.nanmin(q_arr, axis=1)
        q_min = q_min[np.isfinite(q_min)]
        ax5.hist(q_min, bins=100, range=(0, float(np.percentile(q_min, 99.5))),
                 color=color, alpha=0.5, edgecolor="none", density=True,
                 label=f"{label} (N={len(q_min)})", histtype="stepfilled", ls=ls)

    ax5.set_xlabel("Min Q_sum across 4 planes")
    ax5.set_ylabel("Density (normalized)")
    ax5.set_title("Weakest-plane charge overlay — sim vs real\nExcess at low charge = noise / cross-talk population")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    p5 = OUT_DIR / "05_weakest_plane_overlay.png"
    fig5.savefig(p5, dpi=150)
    plt.close(fig5)
    print(f"Saved: {p5}")

    # -----------------------------------------------------------------------
    # FIGURE 6: Which plane is weakest in 4-plane events — sim vs real
    # -----------------------------------------------------------------------
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, df, color) in zip(
        axes6, [(SIM_STATION, df_sim, "steelblue"), (REAL_STATION, df_real, "firebrick")]
    ):
        tt = compute_threshold_tt(df, 0.0)
        mask_4 = tt == 1234
        if mask_4.sum() == 0:
            continue
        q_arr = df.loc[mask_4, Q_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        weakest = np.nanargmin(q_arr, axis=1) + 1  # plane 1-4
        counts = np.bincount(weakest, minlength=5)[1:5]
        frac = counts / counts.sum()
        bars = ax.bar([1, 2, 3, 4], frac, color=[plane_colors[k] for k in range(1, 5)],
                      edgecolor="black", linewidth=0.5)
        for b, f in zip(bars, frac):
            ax.text(b.get_x() + b.get_width() / 2, f + 0.01, f"{f:.3f}",
                    ha="center", fontsize=9)
        ax.set_xlabel("Plane")
        ax.set_ylabel("Fraction of 4-plane events")
        ax.set_title(f"{label} — which plane is weakest")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(0, max(frac) * 1.2)
        ax.grid(True, alpha=0.3, axis="y")

    fig6.suptitle(
        "Which plane has the minimum charge in 4-plane events\n"
        "Asymmetry in real data suggests a specific plane receives more cross-talk",
        fontsize=11,
    )
    fig6.tight_layout(rect=[0, 0, 1, 0.90])
    p6 = OUT_DIR / "06_weakest_plane_identity.png"
    fig6.savefig(p6, dpi=150)
    plt.close(fig6)
    print(f"Saved: {p6}")

    # -----------------------------------------------------------------------
    # FIGURE 7: Noise fraction from quantile-normalized survival ratio
    # -----------------------------------------------------------------------
    fig7, ax7 = plt.subplots(figsize=(10, 6))

    if n4_sim[0] > 0 and n4_real[0] > 0:
        valid = (n4_sim > 10) & (n4_real > 10)
        f_sim = n4_sim / n4_sim[0]
        f_real = n4_real / n4_real[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            noise_frac = np.where(
                (f_real > 0.001) & valid,
                1.0 - f_sim / f_real,
                np.nan,
            )
        ax7.plot(percentiles, noise_frac, "ko-", ms=4, lw=1.5, label="Noise fraction in N(1234)")
        ax7.axhline(0, color="grey", ls="--", lw=0.8)
        ax7.fill_between(percentiles, 0, np.nan_to_num(noise_frac, 0),
                         where=np.nan_to_num(noise_frac, 0) > 0,
                         alpha=0.2, color="red")

        # Find the peak noise fraction and annotate it
        finite_mask = np.isfinite(noise_frac)
        if np.any(finite_mask):
            peak_idx = np.nanargmax(noise_frac)
            peak_pct = percentiles[peak_idx]
            peak_val = noise_frac[peak_idx]
            thr_sim_at_peak = thr_sim[peak_idx]
            thr_real_at_peak = thr_real[peak_idx]
            ax7.annotate(
                f"Peak noise frac: {peak_val:.3f} ({peak_val*100:.1f}%)\n"
                f"at p{peak_pct:.0f} (sim Q={thr_sim_at_peak:.1f}, real Q={thr_real_at_peak:.1f})",
                xy=(peak_pct, peak_val),
                xytext=(peak_pct + 15, peak_val),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
            )
            print(f"\n=== NOISE ESTIMATE (quantile-normalized) ===")
            print(f"Peak noise fraction: {peak_val:.4f} ({peak_val*100:.1f}%)")
            print(f"  at percentile {peak_pct:.0f} (sim Q={thr_sim_at_peak:.1f}, real Q={thr_real_at_peak:.1f})")
            print(f"  N(1234)_real at p0: {int(n4_real[0])}")
            print(f"  N(1234)_sim at p0: {int(n4_sim[0])}")

    ax7.set_xlabel("Charge percentile")
    ax7.set_ylabel("Estimated noise fraction")
    ax7.set_title(
        "Noise fraction in N(1234): quantile-normalized\n"
        "noise_frac(p) = 1 − f_sim(p)/f_real(p)  —  positive = real retains more events"
    )
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    fig7.tight_layout()
    p7 = OUT_DIR / "07_noise_fraction_quantile.png"
    fig7.savefig(p7, dpi=150)
    plt.close(fig7)
    print(f"Saved: {p7}")

    # -----------------------------------------------------------------------
    # FIGURE 8: Full topology distribution vs percentile
    # At each threshold, what fraction of events is 4-plane, 3-plane, 2-plane, etc.?
    # -----------------------------------------------------------------------
    def build_topology_fractions(df, thresholds_arr):
        """Return dict: multiplicity -> array of fractions at each threshold."""
        fracs = {m: np.zeros(len(thresholds_arr)) for m in range(5)}  # 0,1,2,3,4 planes
        for i, thr in enumerate(thresholds_arr):
            tt = compute_threshold_tt(df, thr)
            total = len(tt)
            if total == 0:
                continue
            for _, val in tt.value_counts().items():
                pass  # need multiplicity
            # Count by number of digits in tt (= number of active planes)
            n_active = tt.astype(str).str.replace("0", "", regex=False).str.len()
            # tt=0 means 0 planes
            n_active = n_active.where(tt > 0, 0)
            for m in range(5):
                fracs[m][i] = (n_active == m).sum() / total
        return fracs

    print("Computing topology fractions...")
    topo_sim = build_topology_fractions(df_sim, thr_sim)
    topo_real = build_topology_fractions(df_real, thr_real)

    fig8, axes8 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    mult_colors = {4: "tab:blue", 3: "tab:orange", 2: "tab:green", 1: "tab:red", 0: "grey"}
    mult_labels = {4: "4-plane", 3: "3-plane", 2: "2-plane", 1: "1-plane", 0: "0-plane"}

    for ax, (label, topo) in zip(axes8, [(SIM_STATION, topo_sim), (REAL_STATION, topo_real)]):
        bottom = np.zeros(len(percentiles))
        for m in [4, 3, 2, 1, 0]:
            ax.fill_between(percentiles, bottom, bottom + topo[m],
                            color=mult_colors[m], alpha=0.7, label=mult_labels[m])
            bottom += topo[m]
        ax.set_xlabel("Charge percentile")
        ax.set_ylabel("Fraction of events")
        ax.set_title(f"{label}")
        ax.legend(loc="center right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1)

    fig8.suptitle(
        "Full topology composition vs charge percentile threshold\n"
        "How event multiplicity redistributes as threshold rises",
        fontsize=12,
    )
    fig8.tight_layout(rect=[0, 0, 1, 0.92])
    p8 = OUT_DIR / "08_topology_composition_stacked.png"
    fig8.savefig(p8, dpi=150)
    plt.close(fig8)
    print(f"Saved: {p8}")

    # -----------------------------------------------------------------------
    # FIGURE 9: 3-plane ABSOLUTE counts (not survival fraction)
    # Shows the actual flow: do 3-plane counts increase before decreasing?
    # -----------------------------------------------------------------------
    fig9, axes9 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, n4, n3, thr_arr) in zip(
        axes9,
        [(SIM_STATION, n4_sim, n3_sim, thr_sim), (REAL_STATION, n4_real, n3_real, thr_real)],
    ):
        ax2 = ax.twinx()
        ax.plot(percentiles, n4, "k-", lw=2, label="N(1234)")
        ax2_lines = []
        for k in range(1, 5):
            ln, = ax2.plot(percentiles, n3[k], "--", color=plane_colors[k], lw=1.2,
                           label=f"N(miss P{k})")
            ax2_lines.append(ln)
        ax.set_xlabel("Charge percentile")
        ax.set_ylabel("N(1234) — black, left axis")
        ax2.set_ylabel("N(missing k) — colored, right axis")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.2)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig9.suptitle(
        "Absolute counts: N(1234) and N(missing k) vs charge percentile\n"
        "3-plane counts can INCREASE before decreasing (4→3 plane demotions)",
        fontsize=11,
    )
    fig9.tight_layout(rect=[0, 0, 1, 0.91])
    p9 = OUT_DIR / "09_absolute_counts_4plane_3plane.png"
    fig9.savefig(p9, dpi=150)
    plt.close(fig9)
    print(f"Saved: {p9}")

    # -----------------------------------------------------------------------
    # FIGURE 10: 4→3 transition rate
    # At each threshold step, how many events that were 4-plane at Q(p-1)
    # become 3-plane at Q(p)? Which plane do they lose?
    # -----------------------------------------------------------------------
    print("Computing 4→3 plane transition rates...")
    fig10, axes10 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, df, thr_arr) in zip(
        axes10,
        [(SIM_STATION, df_sim, thr_sim), (REAL_STATION, df_real, thr_real)],
    ):
        # Compute TT at each threshold
        tt_at_thr = []
        for thr in thr_arr:
            tt_at_thr.append(compute_threshold_tt(df, thr).to_numpy())

        lost_plane_counts = {k: np.zeros(len(thr_arr) - 1) for k in range(1, 5)}
        for i in range(1, len(thr_arr)):
            was_4 = tt_at_thr[i - 1] == 1234
            now_tt = tt_at_thr[i]
            for k in range(1, 5):
                lost_plane_counts[k][i - 1] = np.sum(was_4 & (now_tt == MISSING_TT[k]))

        mid_pct = 0.5 * (percentiles[:-1] + percentiles[1:])
        for k in range(1, 5):
            ax.plot(mid_pct, lost_plane_counts[k], "o-", color=plane_colors[k],
                    ms=3, lw=1.2, label=f"lost P{k}")
        ax.set_xlabel("Charge percentile (midpoint)")
        ax.set_ylabel("Events transitioning 4→3 plane")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig10.suptitle(
        "4→3 plane transitions per threshold step\n"
        "Which plane is lost first? Noise-promoted planes lost at low percentiles.",
        fontsize=11,
    )
    fig10.tight_layout(rect=[0, 0, 1, 0.91])
    p10 = OUT_DIR / "10_four_to_three_transitions.png"
    fig10.savefig(p10, dpi=150)
    plt.close(fig10)
    print(f"Saved: {p10}")

    # -----------------------------------------------------------------------
    # FIGURE 11: Per-plane charge distributions for 4-plane events
    # Each plane's Q_sum in 4-plane events — sim vs real overlay
    # -----------------------------------------------------------------------
    fig11, axes11 = plt.subplots(2, 2, figsize=(12, 10))

    for k, ax in zip(range(1, 5), axes11.flat):
        col = f"P{k}_Q_sum_final"
        for label, df, color, ls in [
            (SIM_STATION, df_sim, "steelblue", "-"),
            (REAL_STATION, df_real, "firebrick", "--"),
        ]:
            tt = compute_threshold_tt(df, 0.0)
            mask_4 = tt == 1234
            if mask_4.sum() == 0 or col not in df.columns:
                continue
            vals = pd.to_numeric(df.loc[mask_4, col], errors="coerce").dropna().to_numpy()
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            q99 = float(np.percentile(vals, 99.5))
            ax.hist(vals, bins=80, range=(0, q99), color=color, alpha=0.45,
                    density=True, label=f"{label} (N={len(vals)})", histtype="stepfilled")
        ax.set_xlabel(f"P{k} Q_sum_final")
        ax.set_ylabel("Density")
        ax.set_title(f"Plane {k} charge in 4-plane events")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig11.suptitle(
        "Per-plane charge distributions for 4-plane events — sim vs real\n"
        "Charge scale difference visible; shape differences indicate noise/streamers",
        fontsize=11,
    )
    fig11.tight_layout(rect=[0, 0, 1, 0.93])
    p11 = OUT_DIR / "11_perplane_charge_4plane_overlay.png"
    fig11.savefig(p11, dpi=150)
    plt.close(fig11)
    print(f"Saved: {p11}")

    # -----------------------------------------------------------------------
    # FIGURE 12: Efficiency on quantile axis — zoom into the dip-recovery region
    # Standard formula, with shaded band showing sim as reference
    # -----------------------------------------------------------------------
    fig12, ax12 = plt.subplots(figsize=(12, 7))

    for k in range(1, 5):
        eff_sim = compute_efficiency_standard(n4_sim, n3_sim[k])
        eff_real = compute_efficiency_standard(n4_real, n3_real[k])
        ax12.plot(percentiles, eff_sim, "o-", color=plane_colors[k], ms=4, lw=1.2,
                  alpha=0.4, label=f"sim P{k}")
        ax12.plot(percentiles, eff_real, "s-", color=plane_colors[k], ms=5, lw=1.8,
                  label=f"real P{k}")

    ax12.set_xlabel("Charge percentile", fontsize=12)
    ax12.set_ylabel("ε_std = N(1234) / (N(1234) + N(miss k))", fontsize=12)
    ax12.set_title(
        "Standard efficiency — quantile-normalized\n"
        "Sim: gently declining (pure signal). Real: dip-recovery (noise + cross-talk).",
        fontsize=12,
    )
    ax12.legend(fontsize=9, ncol=2)
    ax12.grid(True, alpha=0.3)
    ax12.set_ylim(0, 1.05)

    # Add secondary x-axis showing absolute Q values for real data
    ax12b = ax12.twiny()
    tick_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    tick_q_real = np.interp(tick_pcts, percentiles, thr_real)
    tick_q_sim = np.interp(tick_pcts, percentiles, thr_sim)
    ax12b.set_xlim(ax12.get_xlim())
    ax12b.set_xticks(tick_pcts)
    ax12b.set_xticklabels([f"R:{r:.0f}\nS:{s:.0f}" for r, s in zip(tick_q_real, tick_q_sim)],
                           fontsize=6)
    ax12b.set_xlabel("Absolute Q (R=real, S=sim)", fontsize=9)

    fig12.tight_layout()
    p12 = OUT_DIR / "12_efficiency_quantile_zoom.png"
    fig12.savefig(p12, dpi=150)
    plt.close(fig12)
    print(f"Saved: {p12}")

    # -----------------------------------------------------------------------
    # FIGURE 13: FROZEN-POPULATION efficiency
    # Classify events at Q=0 into native groups:
    #   - Group A: 4-plane at Q=0  →  track how many still have 4 planes at Q
    #   - Group B_k: 3-plane-missing-k at Q=0  →  track how many still have
    #     their 3 reference planes above Q
    # Efficiency = A_surviving(Q) / (A_surviving(Q) + B_k_surviving(Q))
    # No migration: a 4-plane event that loses a plane just leaves group A,
    # it does NOT enter group B_k.
    # -----------------------------------------------------------------------
    print("Computing frozen-population efficiency...")

    def build_frozen_efficiency(df, thr_arr):
        """Frozen-population: classify at Q=0, track survival without migration."""
        tt0 = compute_threshold_tt(df, 0.0)
        # Native 4-plane events
        mask_native4 = (tt0 == 1234).to_numpy()
        # Native 3-plane-missing-k events
        mask_native3 = {}
        for k in range(1, 5):
            mask_native3[k] = (tt0 == MISSING_TT[k]).to_numpy()

        n_native4_surviving = np.zeros(len(thr_arr))
        n_native3_surviving = {k: np.zeros(len(thr_arr)) for k in range(1, 5)}

        # For each native-3 group, define which planes must pass
        ref_planes = {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}

        # Pre-extract charge arrays once
        q_arrays = {}
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            q_arrays[p] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        for i, thr in enumerate(thr_arr):
            # Native-4 surviving: all 4 planes still above threshold
            all_above = np.ones(len(df), dtype=bool)
            for p in range(1, 5):
                all_above &= q_arrays[p] > float(thr)
            n_native4_surviving[i] = np.sum(mask_native4 & all_above)

            # Native-3 surviving: their 3 reference planes still above threshold
            for k in range(1, 5):
                ref_above = np.ones(len(df), dtype=bool)
                for p in ref_planes[k]:
                    ref_above &= q_arrays[p] > float(thr)
                n_native3_surviving[k][i] = np.sum(mask_native3[k] & ref_above)

        return n_native4_surviving, n_native3_surviving

    frozen4_sim, frozen3_sim = build_frozen_efficiency(df_sim, thr_sim)
    frozen4_real, frozen3_real = build_frozen_efficiency(df_real, thr_real)

    fig13, axes13 = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (label, f4, f3) in zip(
        axes13,
        [(SIM_STATION, frozen4_sim, frozen3_sim),
         (REAL_STATION, frozen4_real, frozen3_real)],
    ):
        for k in range(1, 5):
            denom = f4 + f3[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                eff = np.divide(f4, denom,
                                out=np.full(len(f4), np.nan), where=denom > 0)
            ax.plot(percentiles, eff, "o-", color=plane_colors[k], ms=4, lw=1.5,
                    label=f"P{k} (native N4={int(f4[0])}, N3={int(f3[k][0])})")
        ax.set_xlabel("Charge percentile")
        ax.set_ylabel("ε_frozen = N4_surv / (N4_surv + N3k_surv)")
        ax.set_title(f"{label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig13.suptitle(
        "FROZEN-POPULATION efficiency: no migration between groups\n"
        "Events classified once at Q=0. 4-plane demotions leave the count, "
        "they do NOT inflate N(3-plane).",
        fontsize=11,
    )
    fig13.tight_layout(rect=[0, 0, 1, 0.90])
    p13 = OUT_DIR / "13_frozen_population_efficiency.png"
    fig13.savefig(p13, dpi=150)
    plt.close(fig13)
    print(f"Saved: {p13}")

    # -----------------------------------------------------------------------
    # FIGURE 14: Frozen-population — survival curves for each native group
    # Shows the raw counts of native-4 and native-3 populations vs threshold
    # -----------------------------------------------------------------------
    fig14, axes14 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, f4, f3) in zip(
        axes14,
        [(SIM_STATION, frozen4_sim, frozen3_sim),
         (REAL_STATION, frozen4_real, frozen3_real)],
    ):
        if f4[0] > 0:
            ax.plot(percentiles, f4 / f4[0], "k-", lw=2.5,
                    label=f"Native 4-plane (N₀={int(f4[0])})")
        for k in range(1, 5):
            if f3[k][0] > 0:
                ax.plot(percentiles, f3[k] / f3[k][0], "--", color=plane_colors[k],
                        lw=1.2, label=f"Native 3-miss-P{k} (N₀={int(f3[k][0])})")
        ax.set_xlabel("Charge percentile")
        ax.set_ylabel("Survival fraction (normalized to Q=0)")
        ax.set_title(f"{label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig14.suptitle(
        "Frozen-population survival: how fast does each native group lose events?\n"
        "Native-4 should drop faster if it contains noise-promoted events.",
        fontsize=11,
    )
    fig14.tight_layout(rect=[0, 0, 1, 0.91])
    p14 = OUT_DIR / "14_frozen_population_survival.png"
    fig14.savefig(p14, dpi=150)
    plt.close(fig14)
    print(f"Saved: {p14}")

    # -----------------------------------------------------------------------
    # FIGURE 15: Overlay frozen vs standard efficiency — real data only
    # Direct comparison: how much does the migration artifact distort ε?
    # -----------------------------------------------------------------------
    fig15, ax15 = plt.subplots(figsize=(12, 7))

    for k in range(1, 5):
        # Standard (with migration)
        eff_std = compute_efficiency_standard(n4_real, n3_real[k])
        ax15.plot(percentiles, eff_std, "s--", color=plane_colors[k], ms=4, lw=1,
                  alpha=0.5, label=f"standard P{k}")
        # Frozen (no migration)
        denom = frozen4_real + frozen3_real[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_frz = np.divide(frozen4_real, denom,
                                out=np.full(len(frozen4_real), np.nan), where=denom > 0)
        ax15.plot(percentiles, eff_frz, "o-", color=plane_colors[k], ms=5, lw=2,
                  label=f"frozen P{k}")

    ax15.set_xlabel("Charge percentile", fontsize=12)
    ax15.set_ylabel("Efficiency", fontsize=12)
    ax15.set_title(
        f"{REAL_STATION}: Standard vs Frozen-population efficiency\n"
        "Dashed = standard (4→3 migration inflates denominator). "
        "Solid = frozen (no migration).",
        fontsize=11,
    )
    ax15.legend(fontsize=8, ncol=2)
    ax15.grid(True, alpha=0.3)
    ax15.set_ylim(0, 1.05)
    fig15.tight_layout()
    p15 = OUT_DIR / "15_frozen_vs_standard_efficiency_real.png"
    fig15.savefig(p15, dpi=150)
    plt.close(fig15)
    print(f"Saved: {p15}")

    # -----------------------------------------------------------------------
    # FIGURE 16: Noise-promoted fraction in native-4 — sim vs real survival
    # The sim native-4 survival is pure signal. The real native-4 survival
    # drops faster at low percentiles because it contains noise-promoted
    # events (3-plane events that got a fake 4th plane from cross-talk).
    # noise_promoted_frac(p) = 1 - f4_real(p) / f4_sim(p)
    # where f4 = native4_surviving / native4_at_Q0
    # -----------------------------------------------------------------------
    print("Computing noise-promoted fraction in native-4 events...")
    fig16, axes16 = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: overlay the native-4 survival curves (normalized)
    ax = axes16[0]
    if frozen4_sim[0] > 0:
        f4s = frozen4_sim / frozen4_sim[0]
        ax.plot(percentiles, f4s, "o-", color="steelblue", ms=4, lw=1.8,
                label=f"{SIM_STATION} native-4 (N₀={int(frozen4_sim[0])})")
    if frozen4_real[0] > 0:
        f4r = frozen4_real / frozen4_real[0]
        ax.plot(percentiles, f4r, "s-", color="firebrick", ms=4, lw=1.8,
                label=f"{REAL_STATION} native-4 (N₀={int(frozen4_real[0])})")
    ax.set_xlabel("Charge percentile")
    ax.set_ylabel("Survival fraction (native-4)")
    ax.set_title("Native-4 survival: sim (pure signal) vs real (signal + noise)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: noise-promoted fraction
    ax = axes16[1]
    if frozen4_sim[0] > 0 and frozen4_real[0] > 0:
        f4s = frozen4_sim / frozen4_sim[0]
        f4r = frozen4_real / frozen4_real[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            # If real drops faster than sim, the excess loss = noise being cleaned
            noise_prom = np.where(
                (f4s > 0.01) & (f4r > 0.001),
                1.0 - f4r / f4s,
                np.nan,
            )
        ax.plot(percentiles, noise_prom, "ko-", ms=4, lw=1.5,
                label="noise_promoted_frac = 1 − f4_real/f4_sim")
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.fill_between(percentiles, 0, np.nan_to_num(noise_prom, 0),
                         where=np.nan_to_num(noise_prom, 0) > 0,
                         alpha=0.25, color="red", label="Real loses faster (noise cleaned)")
        ax.fill_between(percentiles, 0, np.nan_to_num(noise_prom, 0),
                         where=np.nan_to_num(noise_prom, 0) < 0,
                         alpha=0.25, color="blue", label="Sim loses faster")

        finite = np.isfinite(noise_prom)
        if np.any(finite):
            peak_idx = np.nanargmax(noise_prom)
            peak_pct = percentiles[peak_idx]
            peak_val = noise_prom[peak_idx]
            ax.annotate(
                f"Peak: {peak_val:.3f} ({peak_val*100:.1f}%)\nat p{peak_pct:.0f}",
                xy=(peak_pct, peak_val),
                xytext=(peak_pct + 12, peak_val + 0.02),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
            )
            print(f"\n=== NOISE-PROMOTED FRACTION (frozen native-4) ===")
            print(f"Peak: {peak_val:.4f} ({peak_val*100:.1f}%) at percentile {peak_pct:.0f}")
            print(f"  Native-4 at Q=0: sim={int(frozen4_sim[0])}, real={int(frozen4_real[0])}")

    ax.set_xlabel("Charge percentile")
    ax.set_ylabel("Noise-promoted fraction")
    ax.set_title("Fraction of native-4 events that are noise-promoted\n(excess survival loss in real vs sim)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig16.suptitle(
        "Noise-promoted fraction in native-4 population\n"
        "Compares frozen native-4 survival: sim (pure signal) vs real (signal + cross-talk noise)",
        fontsize=11,
    )
    fig16.tight_layout(rect=[0, 0, 1, 0.90])
    p16 = OUT_DIR / "16_noise_promoted_fraction_native4.png"
    fig16.savefig(p16, dpi=150)
    plt.close(fig16)
    print(f"Saved: {p16}")

    # -----------------------------------------------------------------------
    # FIGURE 17: Which plane is the fake 4th in noise-promoted events?
    # For native-4 events in REAL data that lose a plane between consecutive
    # thresholds, track WHICH plane they lose. Compare with sim.
    # The plane lost preferentially at low thresholds in real (but not sim)
    # is the cross-talk plane.
    # -----------------------------------------------------------------------
    print("Computing per-plane loss profile for native-4 events...")
    fig17, axes17 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, df, thr_arr, mask_n4_0) in zip(
        axes17,
        [
            (SIM_STATION, df_sim, thr_sim, (compute_threshold_tt(df_sim, 0.0) == 1234).to_numpy()),
            (REAL_STATION, df_real, thr_real, (compute_threshold_tt(df_real, 0.0) == 1234).to_numpy()),
        ],
    ):
        # Pre-extract charges for native-4 events
        q_n4 = {}
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            q_n4[p] = pd.to_numeric(df.loc[mask_n4_0, col], errors="coerce").to_numpy(dtype=float)

        lost_by_plane = {k: np.zeros(len(thr_arr) - 1) for k in range(1, 5)}
        for i in range(1, len(thr_arr)):
            thr_prev = float(thr_arr[i - 1])
            thr_curr = float(thr_arr[i])
            # Events that had all 4 planes above prev threshold
            had4 = np.ones(mask_n4_0.sum(), dtype=bool)
            for p in range(1, 5):
                had4 &= q_n4[p] > thr_prev
            for k in range(1, 5):
                # Now plane k falls below current threshold but other 3 still above
                lost_k = had4 & (q_n4[k] <= thr_curr)
                others_ok = np.ones(mask_n4_0.sum(), dtype=bool)
                for p in range(1, 5):
                    if p != k:
                        others_ok &= q_n4[p] > thr_curr
                lost_by_plane[k][i - 1] = np.sum(lost_k & others_ok)

        mid_pct = 0.5 * (percentiles[:-1] + percentiles[1:])
        total_lost = sum(lost_by_plane[k] for k in range(1, 5))
        for k in range(1, 5):
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(total_lost > 5, lost_by_plane[k] / total_lost, np.nan)
            ax.plot(mid_pct, frac, "o-", color=plane_colors[k], ms=4, lw=1.5,
                    label=f"P{k}")
        ax.axhline(0.25, color="grey", ls=":", lw=0.8, label="Uniform (25%)")
        ax.set_xlabel("Charge percentile (midpoint)")
        ax.set_ylabel("Fraction of native-4 losses attributed to plane k")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.8)

    fig17.suptitle(
        "Which plane is lost first in native-4 events?\n"
        "In real data at low percentiles, the cross-talk plane dominates losses. Sim should be ~uniform.",
        fontsize=11,
    )
    fig17.tight_layout(rect=[0, 0, 1, 0.90])
    p17 = OUT_DIR / "17_native4_plane_loss_identity.png"
    fig17.savefig(p17, dpi=150)
    plt.close(fig17)
    print(f"Saved: {p17}")

    # -----------------------------------------------------------------------
    # FIGURE 18: Cumulative noise-promoted events — absolute count estimate
    # At each percentile, how many native-4 real events have been lost
    # IN EXCESS of what the sim predicts? This excess = noise-promoted events
    # being cleaned out.
    # -----------------------------------------------------------------------
    print("Computing cumulative noise-promoted event count...")
    fig18, ax18 = plt.subplots(figsize=(10, 6))

    if frozen4_sim[0] > 0 and frozen4_real[0] > 0:
        # Expected real native-4 survival if it were pure signal (like sim)
        f4s = frozen4_sim / frozen4_sim[0]
        expected_real_n4 = frozen4_real[0] * f4s
        excess_lost = expected_real_n4 - frozen4_real
        # Cumulative noise-promoted events removed up to each percentile
        ax18.plot(percentiles, excess_lost, "ro-", ms=4, lw=1.5,
                  label="Excess native-4 events lost (real − expected)")
        ax18.axhline(0, color="grey", ls="--", lw=0.8)
        ax18.fill_between(percentiles, 0, excess_lost,
                           where=excess_lost > 0, alpha=0.2, color="red")

        # Also show as fraction of initial real native-4
        ax18b = ax18.twinx()
        frac_excess = excess_lost / frozen4_real[0]
        ax18b.plot(percentiles, frac_excess * 100, "b--", lw=1, alpha=0.6,
                   label="As % of real native-4")
        ax18b.set_ylabel("% of initial real native-4 events", color="blue", fontsize=10)
        ax18b.tick_params(axis="y", labelcolor="blue")

        # Summary stats
        finite_mask = np.isfinite(excess_lost)
        if np.any(finite_mask):
            max_excess = np.nanmax(excess_lost)
            max_frac = max_excess / frozen4_real[0]
            print(f"\n=== CUMULATIVE NOISE-PROMOTED COUNT ===")
            print(f"Max excess lost: {max_excess:.0f} events ({max_frac*100:.1f}% of native-4)")
            print(f"  out of {int(frozen4_real[0])} native-4 events at Q=0")
            ax18.annotate(
                f"Max excess: {max_excess:.0f} events\n({max_frac*100:.1f}% of native-4)",
                xy=(percentiles[np.nanargmax(excess_lost)], max_excess),
                xytext=(percentiles[np.nanargmax(excess_lost)] - 20, max_excess * 0.8),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
            )

    ax18.set_xlabel("Charge percentile")
    ax18.set_ylabel("Excess native-4 events lost (count)")
    ax18.set_title(
        "Cumulative noise-promoted events removed from native-4 population\n"
        "Expected = real_N4(Q=0) × sim_survival_fraction(p). Excess = noise cleaned out.",
    )
    lines1, labels1 = ax18.get_legend_handles_labels()
    lines2, labels2 = ax18b.get_legend_handles_labels() if frozen4_sim[0] > 0 else ([], [])
    ax18.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax18.grid(True, alpha=0.3)
    fig18.tight_layout()
    p18 = OUT_DIR / "18_cumulative_noise_promoted_count.png"
    fig18.savefig(p18, dpi=150)
    plt.close(fig18)
    print(f"Saved: {p18}")

    # -----------------------------------------------------------------------
    # FIGURE 19: Summary table — noise decomposition numbers
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NOISE DECOMPOSITION SUMMARY")
    print("=" * 70)

    if frozen4_sim[0] > 0 and frozen4_real[0] > 0:
        f4s = frozen4_sim / frozen4_sim[0]
        f4r = frozen4_real / frozen4_real[0]

        # Find the percentile where noise-promoted events are mostly cleaned
        # (where the survival ratio stabilizes — derivative of noise_prom ≈ 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            noise_prom = np.where(f4s > 0.01, 1.0 - f4r / f4s, np.nan)

        # Estimate at p50 (a reasonable "cleaned" point)
        p50_idx = np.argmin(np.abs(percentiles - 50))
        noise_at_p50 = noise_prom[p50_idx] if np.isfinite(noise_prom[p50_idx]) else 0.0

        # Total native-4 and native-3 at Q=0
        total_native4_real = int(frozen4_real[0])
        total_native3_real = {k: int(frozen3_real[k][0]) for k in range(1, 5)}

        # Estimated noise-promoted events
        n_noise_promoted = total_native4_real * noise_at_p50

        print(f"\nReal data ({REAL_STATION}):")
        print(f"  Native 4-plane events at Q=0: {total_native4_real}")
        for k in range(1, 5):
            print(f"  Native 3-plane-miss-P{k} at Q=0: {total_native3_real[k]}")
        print(f"\n  Estimated noise-promoted fraction at p50: {noise_at_p50:.4f} ({noise_at_p50*100:.1f}%)")
        print(f"  Estimated noise-promoted count: ~{n_noise_promoted:.0f} events")
        print(f"  These are originally 3-plane events promoted to 4-plane by cross-talk.")
        print(f"\nSim data ({SIM_STATION}):")
        print(f"  Native 4-plane events at Q=0: {int(frozen4_sim[0])}")
        for k in range(1, 5):
            print(f"  Native 3-plane-miss-P{k} at Q=0: {int(frozen3_sim[k][0])}")
        print(f"  (Sim is pure signal — no noise-promoted events expected)")

    print(f"\nAll plots saved to: {OUT_DIR}")

    # -----------------------------------------------------------------------
    # FIGURE 20: Charge-ratio distribution for native-4 events
    # ratio = Q_min / Q_mean_other3
    # Noise-promoted events have a low ratio (fake 4th plane has anomalously
    # low charge). Sim (pure signal) should have a unimodal distribution at
    # high ratios; real data should show an excess at low ratios.
    # -----------------------------------------------------------------------
    print("\nComputing charge-ratio distribution for native-4 events...")

    def compute_charge_ratio_native4(df):
        """For each native-4 event at Q=0, compute Q_min / Q_mean_other3."""
        tt0 = compute_threshold_tt(df, 0.0)
        mask = (tt0 == 1234).to_numpy()
        q = {}
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            q[p] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        # Stack: shape (n_native4, 4)
        charges = np.column_stack([q[p][mask] for p in range(1, 5)])
        q_min = charges.min(axis=1)
        q_sum = charges.sum(axis=1)
        q_mean_other3 = (q_sum - q_min) / 3.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(q_mean_other3 > 0, q_min / q_mean_other3, np.nan)
        return ratio, charges, mask

    ratio_sim, charges_sim, mask_n4_sim = compute_charge_ratio_native4(df_sim)
    ratio_real, charges_real, mask_n4_real = compute_charge_ratio_native4(df_real)

    ratio_cuts = [0.0, 0.1, 0.2, 0.3, 0.4]  # thresholds to test
    ratio_cut_colors = ["grey", "steelblue", "darkorange", "seagreen", "firebrick"]

    fig20, axes20 = plt.subplots(1, 2, figsize=(14, 6))
    bins = np.linspace(0, 1.5, 60)
    for ax, (label, ratio) in zip(
        axes20,
        [(SIM_STATION, ratio_sim), (REAL_STATION, ratio_real)],
    ):
        r = ratio[np.isfinite(ratio)]
        ax.hist(r, bins=bins, density=True, alpha=0.7,
                color="steelblue" if "00" in label else "firebrick",
                label=f"{label} (N={len(r)})")
        for cut, col in zip(ratio_cuts[1:], ratio_cut_colors[1:]):
            frac = np.mean(r >= cut)
            ax.axvline(cut, color=col, ls="--", lw=1.2,
                       label=f"cut={cut:.1f} → {frac*100:.1f}% pass")
        ax.set_xlabel("Q_min / Q_mean_other3")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}: charge-ratio of native-4 events")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig20.suptitle(
        "Charge-ratio distribution in native-4 events\n"
        "ratio = Q_min / Q_mean(other 3 planes). "
        "Noise-promoted events have low ratio (weak fake plane).",
        fontsize=11,
    )
    fig20.tight_layout(rect=[0, 0, 1, 0.90])
    p20 = OUT_DIR / "20_charge_ratio_distribution_native4.png"
    fig20.savefig(p20, dpi=150)
    plt.close(fig20)
    print(f"Saved: {p20}")

    # -----------------------------------------------------------------------
    # FIGURE 21: Charge-ratio efficiency — efficiency using only "clean-4"
    # subset of native-4 events (those passing Q_min/Q_mean_other3 > cut).
    # Compare across multiple ratio cuts and against the standard formula.
    # -----------------------------------------------------------------------
    print("Computing charge-ratio cut efficiency...")

    def build_ratio_cut_efficiency(df, thr_arr, ratio_cut):
        """Frozen efficiency restricted to clean-4 events (ratio > ratio_cut at Q=0)."""
        tt0 = compute_threshold_tt(df, 0.0)
        q = {}
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            q[p] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        # Classify at Q=0
        mask_native4 = (tt0 == 1234).to_numpy()
        mask_native3 = {k: (tt0 == MISSING_TT[k]).to_numpy() for k in range(1, 5)}

        # Within native-4, keep only clean events
        charges = np.column_stack([q[p] for p in range(1, 5)])
        q_min = np.where(mask_native4, charges.min(axis=1), np.nan)
        q_sum = np.where(mask_native4, charges.sum(axis=1), np.nan)
        q_mean_other3 = (q_sum - q_min) / 3.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(q_mean_other3 > 0, q_min / q_mean_other3, np.nan)
        mask_clean4 = mask_native4 & (np.nan_to_num(ratio, nan=0.0) > ratio_cut)

        ref_planes = {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}

        n_clean4 = np.zeros(len(thr_arr))
        n_native3 = {k: np.zeros(len(thr_arr)) for k in range(1, 5)}

        for i, thr in enumerate(thr_arr):
            all_above = np.ones(len(df), dtype=bool)
            for p in range(1, 5):
                all_above &= q[p] > float(thr)
            n_clean4[i] = np.sum(mask_clean4 & all_above)

            for k in range(1, 5):
                ref_above = np.ones(len(df), dtype=bool)
                for p in ref_planes[k]:
                    ref_above &= q[p] > float(thr)
                n_native3[k][i] = np.sum(mask_native3[k] & ref_above)

        return n_clean4, n_native3

    fig21, axes21 = plt.subplots(2, 2, figsize=(14, 11), sharex=True)

    for ax, k in zip(axes21.flat, range(1, 5)):
        # Standard efficiency for reference
        eff_std_real = compute_efficiency_standard(n4_real, n3_real[k])
        ax.plot(percentiles, eff_std_real, "k:", lw=1.5, alpha=0.6,
                label="standard (real)")

        # Frozen (no cut) as baseline
        denom_frz = frozen4_real + frozen3_real[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_frz = np.divide(frozen4_real, denom_frz,
                                out=np.full(len(frozen4_real), np.nan),
                                where=denom_frz > 0)
        ax.plot(percentiles, eff_frz, "k--", lw=1.5, alpha=0.8,
                label="frozen (no cut, real)")

        # Sim frozen for ground truth
        denom_sim = frozen4_sim + frozen3_sim[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_sim = np.divide(frozen4_sim, denom_sim,
                                out=np.full(len(frozen4_sim), np.nan),
                                where=denom_sim > 0)
        ax.plot(percentiles, eff_sim, "b-", lw=2, alpha=0.5,
                label="frozen sim (ground truth)")

        # Ratio-cut efficiencies
        for cut, col in zip(ratio_cuts[1:], ratio_cut_colors[1:]):
            nc4, nn3 = build_ratio_cut_efficiency(df_real, thr_real, cut)
            denom = nc4 + nn3[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                eff_cut = np.divide(nc4, denom,
                                    out=np.full(len(nc4), np.nan), where=denom > 0)
            n0 = int(nc4[0])
            ax.plot(percentiles, eff_cut, "o-", color=col, ms=4, lw=1.5,
                    label=f"ratio>{cut:.1f} (N4={n0})")

        ax.set_ylabel("Efficiency")
        ax.set_title(f"P{k} efficiency")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    for ax in axes21[1]:
        ax.set_xlabel("Charge percentile")

    fig21.suptitle(
        "Charge-ratio cut efficiency — real data\n"
        "ratio = Q_min/Q_mean_other3 > cut selects 'clean-4' events (no noise-promoted).\n"
        "Dotted = standard, dashed = frozen no-cut, solid blue = sim (ground truth).",
        fontsize=11,
    )
    fig21.tight_layout(rect=[0, 0, 1, 0.88])
    p21 = OUT_DIR / "21_charge_ratio_cut_efficiency.png"
    fig21.savefig(p21, dpi=150)
    plt.close(fig21)
    print(f"Saved: {p21}")

    # -----------------------------------------------------------------------
    # FIGURE 22: Ratio-cut efficiency flatness — how much does the dip
    # reduce as we tighten the ratio cut?
    # Quantify: for each cut, compute max − min of efficiency over p5–p60
    # (the dip region). A smaller range = flatter = less noise contamination.
    # -----------------------------------------------------------------------
    print("Quantifying dip flatness vs ratio cut...")
    fig22, axes22 = plt.subplots(1, 2, figsize=(12, 6))

    # Left: efficiency range (max-min) in the dip region vs cut, per plane
    ax = axes22[0]
    dip_mask = (percentiles >= 5) & (percentiles <= 60)

    all_cuts_range = {k: [] for k in range(1, 5)}
    cuts_tested = ratio_cuts

    for cut in cuts_tested:
        nc4, nn3 = build_ratio_cut_efficiency(df_real, thr_real, cut)
        for k in range(1, 5):
            denom = nc4 + nn3[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                eff = np.divide(nc4, denom,
                                out=np.full(len(nc4), np.nan), where=denom > 0)
            eff_dip = eff[dip_mask]
            finite = eff_dip[np.isfinite(eff_dip)]
            rng = float(np.max(finite) - np.min(finite)) if len(finite) > 1 else np.nan
            all_cuts_range[k].append(rng)

    for k in range(1, 5):
        ax.plot(cuts_tested, all_cuts_range[k], "o-", color=plane_colors[k],
                lw=1.5, ms=6, label=f"P{k}")

    ax.set_xlabel("Ratio cut threshold (Q_min/Q_mean_other3 > cut)")
    ax.set_ylabel("Efficiency range (max−min) in dip region [p5–p60]")
    ax.set_title("Dip amplitude vs charge-ratio cut\n(smaller = flatter curve = less noise)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: N(clean-4) surviving at Q=0 vs cut (shows cost of tightening cut)
    ax = axes22[1]
    n_clean4_at0_real = []
    n_clean4_at0_sim = []
    for cut in cuts_tested:
        nc4_r, _ = build_ratio_cut_efficiency(df_real, thr_real, cut)
        nc4_s, _ = build_ratio_cut_efficiency(df_sim, thr_sim, cut)
        n_clean4_at0_real.append(nc4_r[0])
        n_clean4_at0_sim.append(nc4_s[0])

    ax2b = ax.twinx()
    ax.plot(cuts_tested, n_clean4_at0_real, "rs-", lw=1.5, ms=6,
            label=f"{REAL_STATION} N(clean-4)")
    ax.plot(cuts_tested, n_clean4_at0_sim, "bs-", lw=1.5, ms=6,
            label=f"{SIM_STATION} N(clean-4)")
    # Fraction retained
    if frozen4_real[0] > 0:
        frac_r = [n / frozen4_real[0] for n in n_clean4_at0_real]
        ax2b.plot(cuts_tested, [f * 100 for f in frac_r], "r--", lw=1, alpha=0.5,
                  label="real % retained")
    if frozen4_sim[0] > 0:
        frac_s = [n / frozen4_sim[0] for n in n_clean4_at0_sim]
        ax2b.plot(cuts_tested, [f * 100 for f in frac_s], "b--", lw=1, alpha=0.5,
                  label="sim % retained")
    ax.set_xlabel("Ratio cut threshold")
    ax.set_ylabel("N(clean-4) at Q=0", color="black")
    ax2b.set_ylabel("% of original native-4 retained")
    ax.set_title("Sample size cost of tightening the ratio cut")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    fig22.suptitle(
        "Trade-off: ratio cut stringency vs efficiency flatness\n"
        "Left: how much the dip shrinks. Right: statistical cost (events lost).",
        fontsize=11,
    )
    fig22.tight_layout(rect=[0, 0, 1, 0.91])
    p22 = OUT_DIR / "22_ratio_cut_dip_flatness_tradeoff.png"
    fig22.savefig(p22, dpi=150)
    plt.close(fig22)
    print(f"Saved: {p22}")

    # -----------------------------------------------------------------------
    # FIGURE 23: Per-plane charge comparison — native-4 vs native-3 events
    # The dip hypothesis: native-4 events spread charge over 4 planes, so
    # each plane individually has less charge than in native-3 events (which
    # concentrate the same total muon signal over only 3 planes).
    # If this is true, native-3-miss-k events are inherently harder and
    # survive threshold increases better → ε(Q) = N4/(N4+N3k) dips
    # even for pure signal, because the denominator is more robust.
    # Test: compare per-plane charge distributions between populations.
    # -----------------------------------------------------------------------
    print("Computing per-plane charge comparison (native-4 vs native-3)...")

    fig23, axes23 = plt.subplots(2, 4, figsize=(20, 10))

    for row_idx, (label, df) in enumerate([(SIM_STATION, df_sim), (REAL_STATION, df_real)]):
        tt0 = compute_threshold_tt(df, 0.0)
        mask_n4 = (tt0 == 1234).to_numpy()

        q_all = {}
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            q_all[p] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        for col_idx, k in enumerate(range(1, 5)):
            ax = axes23[row_idx, col_idx]
            mask_n3k = (tt0 == MISSING_TT[k]).to_numpy()

            # Collect all active-plane charges for each group
            # For native-4: all 4 planes; for native-3-miss-k: the 3 present planes
            ref_planes_k = [p for p in range(1, 5) if p != k]

            q_n4_planes = np.concatenate([q_all[p][mask_n4] for p in range(1, 5)])
            q_n3k_planes = np.concatenate([q_all[p][mask_n3k] for p in ref_planes_k])

            # Keep only positive charges
            q_n4_planes = q_n4_planes[q_n4_planes > 0]
            q_n3k_planes = q_n3k_planes[q_n3k_planes > 0]

            # Shared bin range up to 95th percentile of combined
            if len(q_n4_planes) > 0 and len(q_n3k_planes) > 0:
                cap = np.percentile(np.concatenate([q_n4_planes, q_n3k_planes]), 97)
                bins = np.linspace(0, cap, 50)

                ax.hist(q_n4_planes, bins=bins, density=True, alpha=0.55,
                        color="steelblue", label=f"native-4 (N_ev={mask_n4.sum()})")
                ax.hist(q_n3k_planes, bins=bins, density=True, alpha=0.55,
                        color="firebrick", label=f"native-3-miss-P{k} (N_ev={mask_n3k.sum()})")

                med_n4 = np.median(q_n4_planes)
                med_n3k = np.median(q_n3k_planes)
                ax.axvline(med_n4, color="steelblue", ls="--", lw=1.5,
                           label=f"med4={med_n4:.1f}")
                ax.axvline(med_n3k, color="firebrick", ls="--", lw=1.5,
                           label=f"med3k={med_n3k:.1f}")

                ax.set_title(f"{label}: missing P{k}\nmed shift = {med_n3k - med_n4:+.1f}")
            ax.set_xlabel("Q_sum per active plane")
            ax.set_ylabel("Density")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig23.suptitle(
        "Per-plane charge: native-4 vs native-3-miss-k events\n"
        "Hypothesis: native-3 events concentrate muon charge over 3 planes → higher per-plane Q\n"
        "→ native-3 survives threshold scan better → denominator of ε(Q) drops slower → dip",
        fontsize=11,
    )
    fig23.tight_layout(rect=[0, 0, 1, 0.90])
    p23 = OUT_DIR / "23_perplane_charge_n4_vs_n3k.png"
    fig23.savefig(p23, dpi=150)
    plt.close(fig23)
    print(f"Saved: {p23}")

    # -----------------------------------------------------------------------
    # FIGURE 24: Median per-plane charge — native-4 vs native-3 across planes
    # Compact summary: median Q per active plane for native-4 and each
    # native-3-miss-k group. If native-3 medians are consistently higher,
    # the structural bias hypothesis is confirmed.
    # -----------------------------------------------------------------------
    fig24, axes24 = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (label, df) in zip(axes24, [(SIM_STATION, df_sim), (REAL_STATION, df_real)]):
        tt0 = compute_threshold_tt(df, 0.0)
        mask_n4 = (tt0 == 1234).to_numpy()
        q_all = {}
        for p in range(1, 5):
            col = f"P{p}_Q_sum_final"
            q_all[p] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        # Median per-plane Q for native-4
        med_n4 = np.median(
            np.concatenate([q_all[p][mask_n4 & (q_all[p] > 0)] for p in range(1, 5)])
        )

        groups = ["native-4"] + [f"native-3\nmiss-P{k}" for k in range(1, 5)]
        medians = [med_n4]
        n_events = [mask_n4.sum()]

        for k in range(1, 5):
            mask_n3k = (tt0 == MISSING_TT[k]).to_numpy()
            ref = [p for p in range(1, 5) if p != k]
            q_pool = np.concatenate([q_all[p][mask_n3k & (q_all[p] > 0)] for p in ref])
            medians.append(np.median(q_pool) if len(q_pool) > 0 else np.nan)
            n_events.append(mask_n3k.sum())

        colors = ["steelblue"] + [plane_colors[k] for k in range(1, 5)]
        bars = ax.bar(range(5), medians, color=colors, alpha=0.75, edgecolor="k", lw=0.8)
        ax.axhline(med_n4, color="steelblue", ls="--", lw=1, alpha=0.6)

        for i, (bar, n, m) in enumerate(zip(bars, n_events, medians)):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.3,
                    f"N={n}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(range(5))
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylabel("Median per-plane Q_sum")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.3, axis="y")

    fig24.suptitle(
        "Median per-plane charge by topology group\n"
        "If native-3-miss-Pk bars are above the native-4 bar (blue dashed), "
        "the structural denominator-robustness bias is confirmed.",
        fontsize=11,
    )
    fig24.tight_layout(rect=[0, 0, 1, 0.88])
    p24 = OUT_DIR / "24_median_charge_by_topology.png"
    fig24.savefig(p24, dpi=150)
    plt.close(fig24)
    print(f"Saved: {p24}")

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
