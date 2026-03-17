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


def find_parquets(station: str) -> list[Path]:
    station_dir = REPO_ROOT / "STATIONS" / station / TASK3_REL
    if not station_dir.exists():
        return []
    return sorted(station_dir.glob("listed_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)


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

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
