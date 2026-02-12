#!/usr/bin/env python3
"""STEP 1.2 — Dictionary and dataset creation.

Takes the collected_data.csv from STEP 1.1 and:
1. Computes empirical efficiencies (eff_2, eff_3) from trigger topology counts.
2. Filters outliers where eff_2 or eff_3 fall outside configured bounds.
3. Splits the remaining table into:
   - **dataset**: the full clean table (all rows).
   - **dictionary**: a carefully chosen subsample satisfying:
     a) relative error of eff_2 and eff_3 below a threshold,
     b) event count above a minimum,
     c) one entry per unique parameter set (the one with the largest count).
4. Produces diagnostic plots showing dictionary quality and coverage.

Output
------
OUTPUTS/FILES/dataset.csv          — full cleaned dataset
OUTPUTS/FILES/dictionary.csv       — selected dictionary entries
OUTPUTS/FILES/build_summary.json   — summary stats
OUTPUTS/PLOTS/                     — diagnostic plots
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = STEP_DIR.parents[1]  # INFERENCE_DICTIONARY_VALIDATION
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"
DEFAULT_INPUT = (
    STEP_DIR.parent / "STEP_1_1_COLLECT_DATA" / "OUTPUTS" / "FILES" / "collected_data.csv"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="[%(levelname)s] STEP_1.2 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_1.2")


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _parse_efficiencies(value: object) -> list[float]:
    """Parse stringified [e1, e2, e3, e4] into four floats."""
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[i]) for i in range(4)]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [np.nan] * 4
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return [float(parsed[i]) for i in range(4)]
    return [np.nan] * 4


def _compute_eff(n_four: pd.Series, n_three_missing: pd.Series) -> pd.Series:
    """Efficiency = N_four / (N_four + N_three_missing)."""
    denom = n_four + n_three_missing
    return n_four / denom.replace({0: np.nan})


def _find_count_prefix(df: pd.DataFrame) -> str:
    """Detect which trigger-topology prefix is available (raw_tt_, clean_tt_, etc.)."""
    for prefix in ("raw_tt_", "clean_tt_", "cal_tt_", "list_tt_", "fit_tt_"):
        if f"{prefix}1234_count" in df.columns:
            return prefix
    raise KeyError("No trigger-topology count columns found (e.g. raw_tt_1234_count).")


# ── Dictionary coverage helpers ──────────────────────────────────────────

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull (2-D)."""
    if len(points) <= 1:
        return points.copy()
    pts = np.unique(points, axis=0)
    if len(pts) <= 1:
        return pts
    pts_list = sorted((float(x), float(y)) for x, y in pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_list:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula for simple polygon area."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _nearest_neighbor_distances(points: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Per-point nearest-neighbour distances (brute force, chunked)."""
    n = len(points)
    if n < 2:
        return np.full(n, np.nan)
    out = np.full(n, np.nan, dtype=float)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        diff = points[s:e, None, :] - points[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        ri = np.arange(e - s)
        ci = np.arange(s, e)
        d2[ri, ci] = np.inf
        out[s:e] = np.sqrt(np.min(d2, axis=1))
    return out


def _min_distance_to_points(queries: np.ndarray, points: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """For each query, distance to nearest point in *points*."""
    if len(points) == 0:
        return np.full(len(queries), np.nan)
    out = np.full(len(queries), np.nan, dtype=float)
    for s in range(0, len(queries), chunk):
        e = min(len(queries), s + chunk)
        diff = queries[s:e, None, :] - points[None, :, :]
        out[s:e] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def _plot_dictionary_coverage(dictionary: pd.DataFrame, path) -> None:
    """NN distance histogram + coverage-vs-radius curve, matching old STEP_5 style."""
    flux_col, eff_col = "flux_cm2_min", "eff_sim_1"
    if flux_col not in dictionary.columns or eff_col not in dictionary.columns:
        log.warning("Cannot plot dictionary coverage: missing %s or %s", flux_col, eff_col)
        return
    flux = pd.to_numeric(dictionary[flux_col], errors="coerce").dropna().to_numpy(dtype=float)
    eff = pd.to_numeric(dictionary[eff_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(flux) < 3 or len(eff) < 3:
        log.warning("Too few dictionary points for coverage plot")
        return

    # Normalize to [0, 1] bounding box
    f_min, f_max = flux.min(), flux.max()
    e_min, e_max = eff.min(), eff.max()
    f_span = f_max - f_min if f_max > f_min else 1.0
    e_span = e_max - e_min if e_max > e_min else 1.0
    xy = np.column_stack([(flux - f_min) / f_span, (eff - e_min) / e_span])
    unique_xy = np.unique(xy, axis=0)

    # NN distances
    nn_d = _nearest_neighbor_distances(unique_xy)
    nn_valid = nn_d[np.isfinite(nn_d)]

    # MC coverage by radius
    rng = np.random.default_rng(42)
    q = rng.random((5000, 2))
    q_min_d = _min_distance_to_points(q, unique_xy)
    radii = np.linspace(0.005, 0.15, 30)
    coverage_pct = [float(100.0 * np.mean(q_min_d <= r)) for r in radii]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: NN distance histogram
    # The "fraction of bbox diagonal" unit means: distances are measured after
    # mapping flux to [0,1] and eff to [0,1], so 0.10 = 10% of the bounding-box
    # diagonal in that normalised (flux, eff) space.
    nn_pct = nn_valid * 100.0  # convert to % of bbox side
    if len(nn_pct) > 0:
        axes[0].hist(nn_pct, bins=50, color="#F58518", alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("NN distance [% of bbox side in (flux, eff) space]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dictionary nearest-neighbor spacing")
    axes[0].grid(True, alpha=0.2)

    # Right: coverage curve — extend radii to reach 100 %
    radii_extended = np.linspace(0.005, 0.60, 60)
    coverage_pct = [float(100.0 * np.mean(q_min_d <= r)) for r in radii_extended]
    radii_pct = radii_extended * 100.0  # convert to %
    axes[1].plot(radii_pct, coverage_pct, "o-", color="#4C78A8", markersize=3)
    axes[1].set_xlim(0, radii_pct[-1])
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Coverage radius [% of bbox side in (flux, eff) space]")
    axes[1].set_ylabel("Covered random points [%]")
    axes[1].set_title("Dictionary filling by distance-based coverage")
    axes[1].grid(True, alpha=0.2)

    # Annotate key metrics
    hull = _convex_hull(unique_xy)
    hull_pct = _polygon_area(hull) * 100.0
    info = (f"Unique pts: {len(unique_xy)} | "
            f"Hull area: {hull_pct:.1f}% of bbox | "
            f"NN median: {np.median(nn_pct):.1f}% of bbox side")
    fig.suptitle(info, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_eff_sim_vs_empirical(dataset: pd.DataFrame, dictionary: pd.DataFrame,
                                path) -> None:
    """2×2 scatter of simulated vs empirical efficiency for all 4 planes."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    min_vals, max_vals = [], []
    for plane in range(1, 5):
        sim_col = f"eff_sim_{plane}"
        emp_col = f"eff_empirical_{plane}"
        for col in (sim_col, emp_col):
            if col in dataset.columns:
                s = pd.to_numeric(dataset[col], errors="coerce").dropna()
                if not s.empty:
                    min_vals.append(float(s.min()))
                    max_vals.append(float(s.max()))
    xmin = min(min_vals) if min_vals else 0.0
    xmax = max(max_vals) if max_vals else 1.0
    pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.01

    for plane in range(1, 5):
        ax = axes[(plane - 1) // 2, (plane - 1) % 2]
        sim_col = f"eff_sim_{plane}"
        emp_col = f"eff_empirical_{plane}"
        sim = pd.to_numeric(dataset.get(sim_col), errors="coerce")
        emp = pd.to_numeric(dataset.get(emp_col), errors="coerce")
        m = sim.notna() & emp.notna()
        if m.any():
            ax.scatter(sim[m], emp[m], s=12, alpha=0.4, color="#AAAAAA", zorder=2)
        # Dictionary points
        if not dictionary.empty:
            ds = pd.to_numeric(dictionary.get(sim_col), errors="coerce")
            de = pd.to_numeric(dictionary.get(emp_col), errors="coerce")
            dm = ds.notna() & de.notna()
            if dm.any():
                ax.scatter(ds[dm], de[dm], s=25, alpha=0.8, marker="x",
                           color="#E45756", linewidths=1.0, zorder=3, label="Dictionary")
        ax.plot([xmin - pad, xmax + pad], [xmin - pad, xmax + pad], "k--", linewidth=1)
        ax.set_title(f"Plane {plane}")
        ax.set_xlabel("Simulated efficiency")
        ax.set_ylabel("Empirical efficiency")
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(xmin - pad, xmax + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Simulated vs Empirical Efficiency per Plane (grey=data, red×=dict)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_iso_rate(dictionary: pd.DataFrame, path) -> None:
    """Scatter + iso-contours of global rate in the flux–eff plane (dictionary only)."""
    import ast as _ast
    from matplotlib.tri import Triangulation, LinearTriInterpolator

    if "efficiencies" not in dictionary.columns:
        log.warning("No 'efficiencies' column — skipping iso-rate plot.")
        return
    gr_col = "events_per_second_global_rate"
    if gr_col not in dictionary.columns:
        log.warning("No global rate column — skipping iso-rate plot.")
        return

    try:
        effs = dictionary["efficiencies"].apply(_ast.literal_eval)
        eff = effs.apply(lambda x: float(x[0]))
    except Exception:
        log.warning("Could not parse efficiencies — skipping iso-rate plot.")
        return

    flux = pd.to_numeric(dictionary["flux_cm2_min"], errors="coerce")
    rate = pd.to_numeric(dictionary[gr_col], errors="coerce")
    xm = flux.to_numpy(dtype=float)
    ym = eff.to_numpy(dtype=float)
    zm = rate.to_numpy(dtype=float)
    mask = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(zm)
    if mask.sum() < 10:
        log.warning("Too few valid points for iso-rate plot.")
        return
    xm, ym, zm = xm[mask], ym[mask], zm[mask]

    levels = np.arange(np.floor(zm.min()), np.ceil(zm.max()) + 1, 1.0)
    if len(levels) < 2:
        levels = np.linspace(zm.min(), zm.max(), 8)

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    sc = ax.scatter(xm, ym, c=zm, cmap=cmap, s=22, alpha=0.75,
                    edgecolors="0.3", linewidths=0.3, zorder=3,
                    vmin=levels.min(), vmax=levels.max())
    try:
        tri = Triangulation(xm, ym)
        interp = LinearTriInterpolator(tri, zm)
        xi = np.linspace(xm.min(), xm.max(), 300)
        yi = np.linspace(ym.min(), ym.max(), 300)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = interp(Xi, Yi)
        cs = ax.contour(Xi, Yi, Zi, levels=levels, cmap=cmap,
                        linewidths=1.4, alpha=0.9, zorder=2,
                        vmin=levels.min(), vmax=levels.max())
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f Hz")
    except Exception as exc:
        log.warning("Contour interpolation failed: %s", exc)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Global rate [Hz]", fontsize=10)
    ax.set_xlabel("Flux [cm⁻² min⁻¹]", fontsize=11)
    ax.set_ylabel("Efficiency (plane 1)", fontsize=11)
    ax.set_title(f"Iso-global-rate contours ({mask.sum()} dictionary entries)", fontsize=12)
    ax.grid(True, alpha=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_relerr_report(
    dataset: pd.DataFrame,
    dictionary: pd.DataFrame,
    path,
    relerr_cut_by_plane: dict[int, float] | None = None,
    min_events_cut: float | None = None,
) -> None:
    """Comprehensive 2×3 report of relative errors for eff planes 2 and 3.

    Layout (rows = plane 2, plane 3):
      col 0: relerr histogram (signed, filtered to ±3 %)
      col 1: relerr vs n_events scatter (signed)
      col 2: relerr vs simulated efficiency scatter (signed)
    """
    BASE_HIST_CLIP_PCT = 3.0  # minimum histogram clip
    BASE_SCATTER_Y_CLIP_PCT = 5.0  # minimum scatter y-range clip
    CUT_COLOR = "#2A9D8F"
    planes = [2, 3]
    fig, axes = plt.subplots(len(planes), 3, figsize=(16, 5 * len(planes)))

    for row, plane in enumerate(planes):
        re_col = f"relerr_eff_{plane}_pct"
        sim_col = f"eff_sim_{plane}"
        ev_col = "n_events"
        relerr_cut = None
        if relerr_cut_by_plane and plane in relerr_cut_by_plane:
            try:
                relerr_cut = abs(float(relerr_cut_by_plane[plane]))
            except (TypeError, ValueError):
                relerr_cut = None

        hist_clip_pct = BASE_HIST_CLIP_PCT
        scatter_y_clip_pct = BASE_SCATTER_Y_CLIP_PCT
        if relerr_cut is not None and np.isfinite(relerr_cut):
            hist_clip_pct = max(BASE_HIST_CLIP_PCT, relerr_cut)
            scatter_y_clip_pct = max(BASE_SCATTER_Y_CLIP_PCT, relerr_cut)

        hist_x_plot_lim = hist_clip_pct * 1.05
        scatter_y_plot_lim = scatter_y_clip_pct * 1.05

        re = pd.to_numeric(dataset.get(re_col), errors="coerce")
        sim_e = pd.to_numeric(dataset.get(sim_col), errors="coerce")
        n_ev = pd.to_numeric(dataset.get(ev_col), errors="coerce")
        is_dict = dataset.get("is_dictionary_entry", pd.Series(dtype=bool)).astype(bool)

        # ── Col 0: histogram (signed, filtered to ±hist_clip_pct) ──
        ax = axes[row, 0]
        filt = re.notna() & (re.abs() <= hist_clip_pct)
        data_vals = re[filt & (~is_dict if is_dict.any() else True)].dropna()
        dict_vals = re[filt & is_dict].dropna() if is_dict.any() else pd.Series(dtype=float)
        if not data_vals.empty:
            ax.hist(data_vals, bins=50, alpha=0.5, color="#4C78A8",
                    edgecolor="white", label="Dataset")
        if not dict_vals.empty:
            ax.hist(dict_vals, bins=50, alpha=0.6, color="#E45756",
                    edgecolor="white", label="Dictionary")
        all_filt = re[filt].dropna()
        med = all_filt.median() if not all_filt.empty else 0
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(med, color="#E45756", linestyle="--", linewidth=1,
                   label=f"median = {med:.2f}%")
        if relerr_cut is not None and np.isfinite(relerr_cut):
            ax.axvspan(-relerr_cut, relerr_cut, color=CUT_COLOR, alpha=0.07, zorder=0)
            ax.axvline(relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5,
                       label=f"dict cut ±{relerr_cut:.2f}%")
            ax.axvline(-relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5)
        ax.set_xlabel(f"Rel. error eff {plane} [%]")
        ax.set_ylabel("Count")
        ax.set_title(f"Plane {plane} — Rel. error dist. (|re| ≤ {hist_clip_pct:.1f}%)")
        ax.set_xlim(-hist_x_plot_lim, hist_x_plot_lim)
        ax.legend(fontsize=7)

        # ── Col 1: relerr vs n_events ──
        ax = axes[row, 1]
        m = re.notna() & n_ev.notna() & (re.abs() <= scatter_y_clip_pct)
        if m.sum() > 0:
            off = m & ~is_dict if is_dict.any() else m
            on = m & is_dict if is_dict.any() else pd.Series(False, index=m.index)
            if off.sum() > 0:
                ax.scatter(n_ev[off], re[off], s=10, alpha=0.4,
                           color="#AAAAAA", zorder=2, label="Dataset")
            if on.sum() > 0:
                ax.scatter(n_ev[on], re[on], s=25, alpha=0.8, marker="x",
                           color="#E45756", linewidths=1.0, zorder=3, label="Dict")
        ax.axhline(0, color="black", linewidth=0.8)
        if relerr_cut is not None and np.isfinite(relerr_cut):
            ax.axhline(relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5,
                       label=f"relerr cut ±{relerr_cut:.2f}%")
            ax.axhline(-relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5)
        if min_events_cut is not None:
            try:
                min_ev = float(min_events_cut)
                if np.isfinite(min_ev):
                    ax.axvline(min_ev, color="#F4A261", linestyle="-.", linewidth=1.5,
                               label=f"events cut {min_ev:,.0f}")
            except (TypeError, ValueError):
                pass
        ax.set_ylim(-scatter_y_plot_lim, scatter_y_plot_lim)
        ax.set_xlabel("Number of events")
        ax.set_ylabel(f"Rel. error eff {plane} [%]")
        ax.set_title(f"Plane {plane} — Rel. error vs Events (|re| ≤ {scatter_y_clip_pct:.1f}%)")
        ax.legend(fontsize=7)

        # ── Col 2: relerr vs simulated efficiency ──
        ax = axes[row, 2]
        m = re.notna() & sim_e.notna() & (re.abs() <= scatter_y_clip_pct)
        if m.sum() > 0:
            off = m & ~is_dict if is_dict.any() else m
            on = m & is_dict if is_dict.any() else pd.Series(False, index=m.index)
            if off.sum() > 0:
                ax.scatter(sim_e[off], re[off], s=10, alpha=0.4,
                           color="#AAAAAA", zorder=2, label="Dataset")
            if on.sum() > 0:
                ax.scatter(sim_e[on], re[on], s=25, alpha=0.8, marker="x",
                           color="#E45756", linewidths=1.0, zorder=3, label="Dict")
        ax.axhline(0, color="black", linewidth=0.8)
        if relerr_cut is not None and np.isfinite(relerr_cut):
            ax.axhline(relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5,
                       label=f"relerr cut ±{relerr_cut:.2f}%")
            ax.axhline(-relerr_cut, color=CUT_COLOR, linestyle="-.", linewidth=1.5)
        ax.set_ylim(-scatter_y_plot_lim, scatter_y_plot_lim)
        ax.set_xlabel(f"Simulated eff {plane}")
        ax.set_ylabel(f"Rel. error eff {plane} [%]")
        ax.set_title(f"Plane {plane} — Rel. error vs Sim. eff (|re| ≤ {scatter_y_clip_pct:.1f}%)")
        ax.legend(fontsize=7)

    fig.suptitle("Relative Error Report — Efficiency Planes 2 & 3", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1.2: Build dictionary and dataset from collected data."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-csv", default=None,
                        help="Override input collected_data.csv path.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_12 = config.get("step_1_2", {})

    input_path = Path(args.input_csv) if args.input_csv else DEFAULT_INPUT

    eff2_range = cfg_12.get("outlier_eff_2_range", [0.5, 1.0])
    eff3_range = cfg_12.get("outlier_eff_3_range", [0.5, 1.0])
    dict_relerr_eff2_max = cfg_12.get("dictionary_relerr_eff_2_max_pct", 5.0)
    dict_relerr_eff3_max = cfg_12.get("dictionary_relerr_eff_3_max_pct", 5.0)
    dict_min_events = cfg_12.get("dictionary_min_events", 20000)
    plot_params = cfg_12.get("plot_parameters", None)  # None = use all param_cols

    # ── Load ─────────────────────────────────────────────────────────
    if not input_path.exists():
        log.error("Input CSV not found: %s", input_path)
        return 1

    log.info("Loading collected data: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    log.info("  Rows loaded: %d", len(df))

    # ── Compute empirical efficiencies ───────────────────────────────
    prefix = _find_count_prefix(df)
    log.info("  Using count prefix: %s", prefix)

    four_col = f"{prefix}1234_count"
    miss_cols = {
        1: f"{prefix}234_count",
        2: f"{prefix}134_count",
        3: f"{prefix}124_count",
        4: f"{prefix}123_count",
    }

    # Coerce to numeric
    for col in [four_col, *miss_cols.values()]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for plane, miss_col in miss_cols.items():
        df[f"eff_empirical_{plane}"] = _compute_eff(df[four_col], df[miss_col])

    # Parse simulated efficiencies from the 'efficiencies' column
    if "efficiencies" in df.columns:
        effs = df["efficiencies"].apply(_parse_efficiencies)
        for i in range(1, 5):
            df[f"eff_sim_{i}"] = effs.apply(lambda x, idx=i - 1: x[idx])

    # Compute relative errors for eff 2 and 3
    for plane in (2, 3):
        emp_col = f"eff_empirical_{plane}"
        sim_col = f"eff_sim_{plane}"
        if sim_col in df.columns:
            df[f"relerr_eff_{plane}_pct"] = (
                (df[emp_col] - df[sim_col]) / df[sim_col].replace({0: np.nan}) * 100.0
            )
            df[f"abs_relerr_eff_{plane}_pct"] = df[f"relerr_eff_{plane}_pct"].abs()
        else:
            df[f"relerr_eff_{plane}_pct"] = np.nan
            df[f"abs_relerr_eff_{plane}_pct"] = np.nan

    # Determine event count column
    event_col = None
    for candidate in ("selected_rows", "requested_rows", "generated_events_count",
                       "num_events", "event_count"):
        if candidate in df.columns:
            event_col = candidate
            break
    if event_col:
        df["n_events"] = pd.to_numeric(df[event_col], errors="coerce")
    else:
        log.warning("No event count column found; setting n_events = NaN.")
        df["n_events"] = np.nan

    # ── Outlier removal ──────────────────────────────────────────────
    n_before = len(df)
    mask = np.ones(len(df), dtype=bool)
    mask &= df["eff_empirical_2"].between(eff2_range[0], eff2_range[1])
    mask &= df["eff_empirical_3"].between(eff3_range[0], eff3_range[1])
    # Also remove rows where eff_empirical values are NaN
    mask &= df["eff_empirical_2"].notna()
    mask &= df["eff_empirical_3"].notna()

    df_clean = df.loc[mask].copy().reset_index(drop=True)
    n_outliers = n_before - len(df_clean)
    log.info("  Outlier removal: %d outliers dropped, %d rows remain.", n_outliers, len(df_clean))

    # ── Dataset = the full clean table ───────────────────────────────
    dataset = df_clean.copy()

    # ── Dictionary selection ─────────────────────────────────────────
    # Criteria:
    # 1. abs relative error of eff_2 and eff_3 < threshold
    # 2. n_events >= minimum
    # 3. One entry per unique parameter set: keep the one with most events
    dict_mask = np.ones(len(df_clean), dtype=bool)
    dict_mask &= df_clean["abs_relerr_eff_2_pct"] < dict_relerr_eff2_max
    dict_mask &= df_clean["abs_relerr_eff_3_pct"] < dict_relerr_eff3_max
    dict_mask &= df_clean["n_events"] >= dict_min_events

    dict_candidates = df_clean.loc[dict_mask].copy()
    log.info("  Dictionary candidates (pass quality): %d / %d",
             len(dict_candidates), len(df_clean))

    # Unique parameter set = (flux_cm2_min, cos_n, eff_sim_1, eff_sim_2, eff_sim_3, eff_sim_4)
    # plus z-planes (already filtered to one config)
    param_cols = ["flux_cm2_min", "cos_n"]
    for i in range(1, 5):
        if f"eff_sim_{i}" in dict_candidates.columns:
            param_cols.append(f"eff_sim_{i}")

    if not dict_candidates.empty:
        # Sort by n_events descending, then keep first per unique param set
        dict_candidates = dict_candidates.sort_values("n_events", ascending=False)
        dictionary = dict_candidates.drop_duplicates(subset=param_cols, keep="first").copy()
        dictionary = dictionary.reset_index(drop=True)
    else:
        dictionary = pd.DataFrame(columns=df_clean.columns)

    log.info("  Dictionary entries (unique param sets): %d", len(dictionary))

    # Mark dictionary membership in dataset for downstream awareness
    if "filename_base" in dataset.columns and "filename_base" in dictionary.columns:
        dict_ids = set(dictionary["filename_base"].dropna().astype(str))
        dataset["is_dictionary_entry"] = (
            dataset["filename_base"].astype(str).isin(dict_ids)
        )
    else:
        dataset["is_dictionary_entry"] = False

    # ── Save ─────────────────────────────────────────────────────────
    dataset_path = FILES_DIR / "dataset.csv"
    dictionary_path = FILES_DIR / "dictionary.csv"
    dataset.to_csv(dataset_path, index=False)
    dictionary.to_csv(dictionary_path, index=False)
    log.info("Wrote dataset:    %s (%d rows)", dataset_path, len(dataset))
    log.info("Wrote dictionary: %s (%d rows)", dictionary_path, len(dictionary))

    summary = {
        "input_rows": n_before,
        "outliers_removed": n_outliers,
        "dataset_rows": len(dataset),
        "dictionary_candidates": int(dict_mask.sum()),
        "dictionary_rows": len(dictionary),
        "eff2_range": eff2_range,
        "eff3_range": eff3_range,
        "dict_relerr_eff2_max_pct": dict_relerr_eff2_max,
        "dict_relerr_eff3_max_pct": dict_relerr_eff3_max,
        "dict_min_events": dict_min_events,
    }
    with open(FILES_DIR / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostic plots ─────────────────────────────────────────────
    _make_plots(
        dataset,
        dictionary,
        param_cols,
        dict_relerr_eff2_max=dict_relerr_eff2_max,
        dict_relerr_eff3_max=dict_relerr_eff3_max,
        dict_min_events=dict_min_events,
        plot_params=plot_params,
    )

    log.info("Done.")
    return 0


def _make_plots(
    dataset: pd.DataFrame,
    dictionary: pd.DataFrame,
    param_cols: list[str],
    dict_relerr_eff2_max: float,
    dict_relerr_eff3_max: float,
    dict_min_events: float,
    plot_params: list[str] | None = None,
) -> None:
    """Generate concise diagnostic plots.

    *plot_params* selects which parameters appear in histograms, scatter
    matrix, and coverage plots.  If None, all *param_cols* are used.
    """
    # Resolve which params to plot
    if plot_params:
        plot_cols = [c for c in plot_params if c in dataset.columns]
    else:
        plot_cols = [c for c in param_cols if c in dataset.columns]
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if dataset.empty:
        log.warning("Dataset is empty — skipping plots.")
        return

    # ── 1. Dictionary coverage: flux vs eff scatter ──────────────────
    flux_col = "flux_cm2_min"
    eff_col = "eff_sim_1"  # simulated efficiency plane 1
    if flux_col in dataset.columns and eff_col in dataset.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            dataset[flux_col], dataset[eff_col],
            s=10, alpha=0.5, color="#AAAAAA", label="Dataset", zorder=2,
        )
        if not dictionary.empty:
            ax.scatter(
                dictionary[flux_col], dictionary[eff_col],
                s=30, alpha=0.8, marker="x", color="#E45756",
                label="Dictionary", zorder=3, linewidths=1.2,
            )
        ax.set_xlabel("Flux [cm⁻² min⁻¹]")
        ax.set_ylabel("Simulated efficiency (plane 1)")
        ax.set_title("Dictionary coverage in flux–efficiency plane")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "dictionary_coverage_flux_eff.png")
        plt.close(fig)

    # ── 2. Parameter histograms: data vs dictionary ──────────────────
    hist_cols = [c for c in plot_cols if c in dataset.columns]
    if hist_cols:
        n_cols = len(hist_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        for ax, col in zip(axes, hist_cols):
            d_vals = pd.to_numeric(dataset[col], errors="coerce").dropna()
            if not d_vals.empty:
                ax.hist(d_vals, bins=30, alpha=0.5, color="#4C78A8",
                        label="Dataset", density=True)
            if not dictionary.empty:
                dict_vals = pd.to_numeric(dictionary[col], errors="coerce").dropna()
                if not dict_vals.empty:
                    ax.hist(dict_vals, bins=30, alpha=0.6, color="#E45756",
                            label="Dictionary", density=True)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(col)
            ax.legend(fontsize=7)
        fig.suptitle("Parameter distributions: dataset vs dictionary", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "parameter_histograms.png")
        plt.close(fig)

    # ── 3. Scatter matrix of key params (data vs dictionary) ─────────
    scatter_cols = [c for c in plot_cols if c in dataset.columns]
    n_sc = len(scatter_cols)
    if n_sc >= 2:
        fig, axes = plt.subplots(n_sc, n_sc, figsize=(3 * n_sc, 3 * n_sc))
        for i, cy in enumerate(scatter_cols):
            for j, cx in enumerate(scatter_cols):
                ax = axes[i][j] if n_sc > 1 else axes
                if j < i:
                    # Below diagonal: hide
                    ax.set_visible(False)
                elif i == j:
                    # Diagonal: histogram with log scale
                    d_vals = pd.to_numeric(dataset[cx], errors="coerce").dropna()
                    if not d_vals.empty:
                        ax.hist(d_vals, bins=25, alpha=0.5, color="#4C78A8")
                    if not dictionary.empty:
                        dict_vals = pd.to_numeric(dictionary[cx], errors="coerce").dropna()
                        if not dict_vals.empty:
                            ax.hist(dict_vals, bins=25, alpha=0.6, color="#E45756")
                    ax.set_yscale("log")
                    ax.set_xlabel(cx, fontsize=7)
                else:
                    # Above diagonal: scatter (keeps first plot parameter on y-axis)
                    dx = pd.to_numeric(dataset[cx], errors="coerce")
                    dy = pd.to_numeric(dataset[cy], errors="coerce")
                    m = dx.notna() & dy.notna()
                    if m.sum() > 0:
                        ax.scatter(dx[m], dy[m], s=5, alpha=0.3, color="#AAAAAA")
                    if not dictionary.empty:
                        ddx = pd.to_numeric(dictionary[cx], errors="coerce")
                        ddy = pd.to_numeric(dictionary[cy], errors="coerce")
                        dm = ddx.notna() & ddy.notna()
                        if dm.sum() > 0:
                            ax.scatter(ddx[dm], ddy[dm], s=15, alpha=0.7,
                                       marker="x", color="#E45756", linewidths=0.8)
                    ax.set_xlabel(cx, fontsize=7)
                    ax.set_ylabel(cy, fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle("Parameter scatter matrix (grey=data, red×=dictionary)", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "parameter_scatter_matrix.png")
        plt.close(fig)

    # ── 4. Comprehensive relative error report for eff 2 and 3 ────
    _plot_relerr_report(
        dataset,
        dictionary,
        PLOTS_DIR / "relerr_eff_report.png",
        relerr_cut_by_plane={
            2: dict_relerr_eff2_max,
            3: dict_relerr_eff3_max,
        },
        min_events_cut=dict_min_events,
    )

    # ── 5. Events distribution ───────────────────────────────────────
    if "n_events" in dataset.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        d_ev = dataset["n_events"].dropna()
        if not d_ev.empty:
            ax.hist(d_ev, bins=40, alpha=0.5, color="#4C78A8", label="Dataset")
        if not dictionary.empty and "n_events" in dictionary.columns:
            dd_ev = dictionary["n_events"].dropna()
            if not dd_ev.empty:
                ax.hist(dd_ev, bins=40, alpha=0.6, color="#E45756", label="Dictionary")
        ax.set_xlabel("Number of events")
        ax.set_ylabel("Count")
        ax.set_title("Event count distribution: dataset vs dictionary")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "event_count_histogram.png")
        plt.close(fig)

    # ── 6. Dictionary coverage (NN spacing + radius-based filling) ──
    _plot_dictionary_coverage(dictionary, PLOTS_DIR / "dictionary_coverage.png")

    # ── 7. Efficiency sim vs empirical (2×2 scatter) ─────────────────
    _plot_eff_sim_vs_empirical(dataset, dictionary, PLOTS_DIR / "scatter_eff_sim_vs_estimated.png")

    # ── 8. Iso-rate contour in flux–eff space ────────────────────────
    _plot_iso_rate(dictionary, PLOTS_DIR / "iso_rate_global_rate.png")


if __name__ == "__main__":
    raise SystemExit(main())
