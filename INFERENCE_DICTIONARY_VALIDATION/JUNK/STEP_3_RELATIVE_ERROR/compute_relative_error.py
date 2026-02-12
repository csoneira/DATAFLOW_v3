#!/usr/bin/env python3
"""STEP_3: Compute relative error for eff_p2 and eff_p3 and filter reference rows.

Loads the dictionary CSV, builds a validation table (estimated vs simulated
efficiencies), applies quality cuts on relative error and minimum event count,
and writes filtered/unfiltered outputs plus diagnostic plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared utilities --------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    apply_clean_style,
    build_validation_table,
    load_config,
    plot_bar_counts,
    plot_histogram,
    plot_histogram_overlay,
    plot_scatter,
    plot_scatter_overlay,
    resolve_param,
    setup_logger,
    setup_output_dirs,
)

log = setup_logger("STEP_3")
apply_clean_style()

DEFAULT_DICT = (
    REPO_ROOT / "STEP_1_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_OUT = STEP_DIR
DEFAULT_CONFIG = STEP_DIR / "config.json"


# -------------------------------------------------------------------------
# Iso-rate contours on eff vs flux
# -------------------------------------------------------------------------

def _plot_iso_rate_contours(
    df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Scatter eff vs flux with iso-contour lines of constant global rate.

    Produces a single figure with the dictionary points in (flux, eff) space,
    coloured by global rate, with contour lines at integer Hz values
    (10, 11, 12, …, 18 Hz).
    """
    import ast as _ast
    from matplotlib.tri import Triangulation, LinearTriInterpolator

    # --- Extract efficiency from the 'efficiencies' string column ---
    if "efficiencies" not in df.columns:
        log.warning("No 'efficiencies' column — skipping iso-rate plot.")
        return
    try:
        effs = df["efficiencies"].apply(_ast.literal_eval)
        eff = effs.apply(lambda x: float(x[0]))
    except Exception:
        log.warning("Could not parse efficiencies — skipping iso-rate plot.")
        return

    gr_col = "events_per_second_global_rate"
    if gr_col not in df.columns:
        log.warning("No global rate column — skipping iso-rate plot.")
        return

    flux = pd.to_numeric(df["flux_cm2_min"], errors="coerce")
    rate = pd.to_numeric(df[gr_col], errors="coerce")

    x_arr = flux.to_numpy(dtype=float)
    y_arr = eff.to_numpy(dtype=float)
    z_arr = rate.to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
    if mask.sum() < 10:
        log.warning("Too few valid points for iso-rate plot.")
        return

    xm, ym, zm = x_arr[mask], y_arr[mask], z_arr[mask]

    # Fixed contour levels at integer Hz
    levels = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=float)

    cmap = plt.cm.coolwarm

    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")

    # Scatter: points coloured by global rate
    sc = ax.scatter(xm, ym, c=zm, cmap=cmap, s=22, alpha=0.75,
                    edgecolors="0.3", linewidths=0.3, zorder=3,
                    vmin=levels.min(), vmax=levels.max())

    # Triangulated iso-contours at the fixed levels
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
        ax.clabel(cs, inline=True, fontsize=9, fmt="%d Hz")
    except Exception as exc:
        log.warning("Contour interpolation failed: %s", exc)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Global rate [Hz]", fontsize=10)
    cbar.set_ticks(levels)

    ax.set_xlabel("Flux [cm$^{-2}$ min$^{-1}$]", fontsize=11)
    ax.set_ylabel("Efficiency", fontsize=11)
    ax.set_title(
        "Iso-global-rate contours\n"
        f"({mask.sum()} dictionary entries after deduplication)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.15)
    fig.savefig(plot_dir / "iso_rate_global_rate.png", dpi=150)
    plt.close(fig)
    log.info("Iso-rate contour plot saved to %s", plot_dir)


# -------------------------------------------------------------------------
# Efficiency sim-vs-est scatter (all 4 planes in one 2×2 figure)
# -------------------------------------------------------------------------

def _plot_eff_sim_vs_est_all(df: pd.DataFrame, plot_path: Path) -> None:
    """2×2 scatter of simulated vs calculated efficiency for all planes.

    For planes 1 & 4 (outer planes), the estimated efficiency includes an
    acceptance factor that does NOT cancel in the coincidence-ratio estimator.
    This means eff_est_p1 / eff_est_p4 are NOT directly comparable with the
    simulated efficiency.  The systematic offset visible on those planes is
    expected.  These estimates CAN, however, be compared between dictionary
    entries and test samples (both carry the same acceptance bias).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, plane in enumerate(range(1, 5)):
        ax = axes[idx // 2, idx % 2]
        sim_col, est_col = f"eff_sim_p{plane}", f"eff_est_p{plane}"
        if sim_col not in df.columns or est_col not in df.columns:
            ax.set_visible(False)
            continue
        sim = pd.to_numeric(df[sim_col], errors="coerce")
        est = pd.to_numeric(df[est_col], errors="coerce")
        mask = sim.notna() & est.notna()
        if mask.sum() >= 2:
            ax.scatter(sim[mask], est[mask], s=14, alpha=0.6, color="#4C78A8")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel(sim_col)
        ax.set_ylabel(est_col)
        if plane in (1, 4):
            ax.set_title(
                f"Plane {plane}: sim vs est\n"
                "(acceptance offset expected)",
                fontsize=10,
            )
        else:
            ax.set_title(f"Plane {plane}: sim vs est")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------------
# Validation bridge
# -------------------------------------------------------------------------

def _build_validation_table(
    df: pd.DataFrame, prefix: str, eff_method: str
) -> pd.DataFrame:
    return build_validation_table(df, prefix=prefix, eff_method=eff_method)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute relative error for eff2/eff3 and filter rows."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dictionary-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument(
        "--eff-method", default=None,
        choices=["four_over_three_plus_four", "one_minus_three_over_four"],
    )
    ap.add_argument("--relerr-threshold", type=float, default=None,
                    help="Absolute relative error threshold for eff2/eff3.")
    ap.add_argument("--min-events", type=int, default=None,
                    help="Minimum generated events required to keep an entry.")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip diagnostic plots.")
    ap.add_argument("--compare-methods", action="store_true",
                    help="Run both eff methods side-by-side and produce "
                         "comparison plots (to_do.md §3.2).")
    args = ap.parse_args()

    config = load_config(Path(args.config))

    dictionary_csv = resolve_param(
        args.dictionary_csv, config, "dictionary_csv", str(DEFAULT_DICT))
    out_base = Path(resolve_param(
        args.out_dir, config, "out_dir", str(DEFAULT_OUT)))
    prefix = resolve_param(args.prefix, config, "prefix", "raw")
    eff_method = resolve_param(
        args.eff_method, config, "eff_method", "four_over_three_plus_four")
    relerr_threshold = resolve_param(
        args.relerr_threshold, config, "relerr_threshold", 0.01, float)
    min_events = resolve_param(
        args.min_events, config, "min_events", 50000, int)

    files_dir, plot_dir = setup_output_dirs(out_base)
    if plot_dir.exists():
        for p in plot_dir.glob("*.png"):
            p.unlink()

    log.info("Loading dictionary: %s", dictionary_csv)
    df = pd.read_csv(dictionary_csv, low_memory=False)

    log.info("Building validation table (prefix=%s, method=%s)", prefix, eff_method)
    validation = _build_validation_table(df, prefix=prefix, eff_method=eff_method)

    # --- Join validation columns back to dictionary rows ---
    join_col = None
    for cand in ("file_name", "filename_base"):
        if cand in df.columns and cand in validation.columns:
            join_col = cand
            break
    if join_col is None:
        raise KeyError(
            "No shared file_name/filename_base column for joining "
            "validation results."
        )

    rel_cols = ["eff_rel_err_p1", "eff_rel_err_p2",
                 "eff_rel_err_p3", "eff_rel_err_p4"]
    # Only include columns that actually exist in validation output
    rel_cols = [c for c in rel_cols if c in validation.columns]
    rel_frame = validation[
        [join_col, "generated_events_count", *rel_cols]
    ].copy()
    merged = df.merge(rel_frame, on=join_col, how="left")

    # --- Apply filtering ---
    rel2 = merged["eff_rel_err_p2"].abs()
    rel3 = merged["eff_rel_err_p3"].abs()
    if "generated_events_count" not in merged.columns:
        raise KeyError("generated_events_count missing from validation table.")
    event_mask = merged["generated_events_count"].ge(min_events)
    mask = rel2.le(relerr_threshold) & rel3.le(relerr_threshold) & event_mask
    filtered = merged.loc[mask].copy()

    # --- Deduplicate: keep only the highest-statistics file per unique
    #     physical parameter set (flux, cos_n, z-planes).  Multiple files
    #     sharing the same parameters but with fewer events are statistical
    #     subsamples of the same simulation; keeping only the best one
    #     avoids redundancy and improves downstream matching (STEP 4). ---
    dedup_cols = ["flux_cm2_min", "cos_n",
                  "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
    dedup_existing = [c for c in dedup_cols if c in filtered.columns]
    events_col = "selected_rows" if "selected_rows" in filtered.columns else None
    n_before_dedup = len(filtered)
    if dedup_existing and events_col:
        filtered = (
            filtered
            .sort_values(events_col, ascending=False)
            .drop_duplicates(subset=dedup_existing, keep="first")
            .sort_index()
        )
        log.info("Deduplication: %d → %d rows (keeping max %s per [%s]).",
                 n_before_dedup, len(filtered), events_col,
                 ", ".join(dedup_existing))

    merged["used_in_reference"] = mask
    used_entries = merged.loc[mask].copy()
    unused_entries = merged.loc[~mask].copy()

    # --- Write outputs ---
    validation_csv = files_dir / "validation_table.csv"
    filtered_csv = files_dir / "filtered_reference.csv"
    used_csv = files_dir / "used_dictionary_entries.csv"
    unused_csv = files_dir / "unused_dictionary_entries.csv"
    validation.to_csv(validation_csv, index=False)
    filtered.to_csv(filtered_csv, index=False)
    used_entries.to_csv(used_csv, index=False)
    unused_entries.to_csv(unused_csv, index=False)

    log.info("Validation table: %s", validation_csv)
    log.info("Filtered reference: %s (rows=%d)", filtered_csv, len(filtered))
    log.info("Used entries: %s (rows=%d)", used_csv, len(used_entries))
    log.info("Unused entries: %s (rows=%d)", unused_csv, len(unused_entries))

    # --- Diagnostic plots ---
    if not args.no_plots:
        _write_plots(
            merged, validation, used_entries, unused_entries,
            filtered, plot_dir,
        )

    # --- Dual efficiency-method comparison (to_do.md §3.2) ---
    if args.compare_methods:
        _compare_eff_methods(df, prefix, plot_dir)

    return 0


def _compare_eff_methods(
    df: pd.DataFrame, prefix: str, plot_dir: Path,
) -> None:
    """Run both efficiency methods and compare their estimates (to_do.md §3.2).

    Produces per-plane comparison scatters and residual-difference histograms
    so disagreements flag estimator-model mismatch.
    """
    import matplotlib.pyplot as plt  # noqa: local for optional path
    from msv_utils import build_validation_table as _bvt

    methods = ("four_over_three_plus_four", "one_minus_three_over_four")
    tables = {}
    for m in methods:
        tables[m] = _bvt(df, prefix=prefix, eff_method=m)

    comp_dir = plot_dir / "method_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    t1 = tables[methods[0]]
    t2 = tables[methods[1]]

    for plane in range(1, 5):
        est1_col = f"eff_est_p{plane}"
        est2_col = f"eff_est_p{plane}"
        rel1_col = f"eff_rel_err_p{plane}"
        rel2_col = f"eff_rel_err_p{plane}"

        if est1_col not in t1.columns or est2_col not in t2.columns:
            continue

        est1 = pd.to_numeric(t1[est1_col], errors="coerce")
        est2 = pd.to_numeric(t2[est2_col], errors="coerce")
        mask = est1.notna() & est2.notna()
        if mask.sum() < 3:
            continue

        # Scatter: method A vs method B estimates
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(est1[mask], est2[mask], s=12, alpha=0.5, color="#4C78A8")
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax1.set_xlabel(f"eff_est_p{plane} ({methods[0][:12]}…)")
        ax1.set_ylabel(f"eff_est_p{plane} ({methods[1][:12]}…)")
        ax1.set_title(f"Plane {plane}: estimator comparison")
        ax1.grid(True, alpha=0.3)

        # Histogram: difference between methods
        diff = est1[mask] - est2[mask]
        ax2.hist(diff, bins=50, color="#F58518", alpha=0.85, edgecolor="white")
        ax2.axvline(0.0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel(f"Δeff_p{plane} (method₁ − method₂)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Plane {plane}: method difference")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(comp_dir / f"compare_methods_plane_{plane}.png", dpi=140)
        plt.close(fig)

        # Rel-error comparison if available
        if rel1_col in t1.columns and rel2_col in t2.columns:
            r1 = pd.to_numeric(t1[rel1_col], errors="coerce").abs()
            r2 = pd.to_numeric(t2[rel2_col], errors="coerce").abs()
            rmask = r1.notna() & r2.notna()
            if rmask.sum() > 3:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(r1[rmask] * 100, r2[rmask] * 100, s=12,
                           alpha=0.5, color="#54A24B")
                lim = max(float(r1[rmask].quantile(0.99) * 100),
                          float(r2[rmask].quantile(0.99) * 100))
                ax.plot([0, lim], [0, lim], "k--", linewidth=1)
                ax.set_xlabel(f"|rel_err_p{plane}| [%] ({methods[0][:12]}…)")
                ax.set_ylabel(f"|rel_err_p{plane}| [%] ({methods[1][:12]}…)")
                ax.set_title(f"Plane {plane}: abs rel-error comparison")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(comp_dir / f"compare_relerr_plane_{plane}.png",
                            dpi=140)
                plt.close(fig)

    log.info("Method comparison plots saved to %s", comp_dir)


def _write_plots(
    merged: pd.DataFrame,
    validation: pd.DataFrame,
    used: pd.DataFrame,
    unused: pd.DataFrame,
    filtered: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Generate all diagnostic plots for STEP 2 (consolidated).

    Output files:
    - hist_used_vs_unused_relerr.png     : overlay p2+p3 rel-err (2-panel)
    - scatter_used_vs_unused_relerr_p2_vs_p3.png : used/unused rel-err scatter
    - hist_used_vs_unused_flux_cosn.png  : flux+cos_n overlays (1×2)
    - counts_summary.png                 : grouped bar chart
    - selection_bias_diagnostics.png     : multi-panel selection-bias histograms
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Used/unused rel-err overlay for p2+p3 (1×2) ---
    # (Replaces hist_eff_rel_err_p2/p3 + their used/unused overlays)
    relerr_cols = [c for c in ("eff_rel_err_p2", "eff_rel_err_p3")
                   if c in used.columns and c in unused.columns]
    if relerr_cols:
        fig, axes = plt.subplots(1, len(relerr_cols),
                                 figsize=(7 * len(relerr_cols), 5))
        if len(relerr_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, relerr_cols):
            u = pd.to_numeric(used.get(col), errors="coerce").dropna()
            un = pd.to_numeric(unused.get(col), errors="coerce").dropna()
            if not u.empty:
                ax.hist(u, bins=40, density=True, color="#E45756",
                        alpha=0.65, label="used")
            if not un.empty:
                ax.hist(un, bins=40, density=True, color="#54A24B",
                        alpha=0.55, label="unused")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(f"{col} (used vs unused)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(plot_dir / "hist_used_vs_unused_relerr.png", dpi=140)
        plt.close(fig)

    # --- 2. Rel-err scatter used/unused ---
    x_used = pd.to_numeric(used.get("eff_rel_err_p2"), errors="coerce")
    y_used = pd.to_numeric(used.get("eff_rel_err_p3"), errors="coerce")
    x_unused = pd.to_numeric(unused.get("eff_rel_err_p2"), errors="coerce")
    y_unused = pd.to_numeric(unused.get("eff_rel_err_p3"), errors="coerce")
    mask_used = x_used.notna() & y_used.notna()
    mask_unused = x_unused.notna() & y_unused.notna()
    if (mask_used.sum() + mask_unused.sum()) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        if mask_used.any():
            ax.scatter(
                x_used[mask_used], y_used[mask_used],
                s=12, alpha=0.6, color="#E45756", label="used",
            )
        if mask_unused.any():
            ax.scatter(
                x_unused[mask_unused], y_unused[mask_unused],
                s=12, alpha=0.45, color="#54A24B", label="unused",
            )
        ax.set_xlabel("eff_rel_err_p2")
        ax.set_ylabel("eff_rel_err_p3")
        ax.set_title("Rel. error p2 vs p3 (used vs unused)")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(plot_dir / "scatter_used_vs_unused_relerr_p2_vs_p3.png", dpi=140)
        plt.close(fig)

    # --- 3. Flux + cos_n used/unused overlays (1×2) ---
    fc_cols = [c for c in ("flux_cm2_min", "cos_n")
               if c in used.columns and c in unused.columns]
    if fc_cols:
        fig, axes = plt.subplots(1, len(fc_cols),
                                 figsize=(7 * len(fc_cols), 5))
        if len(fc_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, fc_cols):
            u = pd.to_numeric(used.get(col), errors="coerce").dropna()
            un = pd.to_numeric(unused.get(col), errors="coerce").dropna()
            if not u.empty:
                ax.hist(u, bins=40, density=True, color="#54A24B",
                        alpha=0.65, label="used")
            if not un.empty:
                ax.hist(un, bins=40, density=True, color="#E45756",
                        alpha=0.55, label="unused")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(f"{col} (used vs unused)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(plot_dir / "hist_used_vs_unused_flux_cosn.png", dpi=140)
        plt.close(fig)

    # --- 4. Counts bar chart (single grouped figure) ---
    labels = ["total", "filtered", "used", "unused"]
    values = [len(merged), len(filtered), len(used), len(unused)]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rows")
    ax.set_title("Reference filtering summary")
    ax.grid(True, axis="y", alpha=0.2)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "counts_summary.png", dpi=140)
    plt.close(fig)

    # --- 5. Selection-bias diagnostics (multi-panel) ---
    # p1/p4 rel-err + generated_events_count
    bias_cols = []
    for col in ("eff_rel_err_p1", "eff_rel_err_p4"):
        if col in used.columns and col in unused.columns:
            bias_cols.append(col)
    for col in ("generated_events_count",):
        if col in used.columns and col in unused.columns:
            bias_cols.append(col)

    if bias_cols:
        ncols = min(3, len(bias_cols))
        nrows = (len(bias_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 4.5 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, col in enumerate(bias_cols):
            ax = axes_flat[idx]
            u = pd.to_numeric(used.get(col), errors="coerce").dropna()
            un = pd.to_numeric(unused.get(col), errors="coerce").dropna()
            if not u.empty:
                ax.hist(u, bins=40, density=True, color="#54A24B",
                        alpha=0.65, label="used")
            if not un.empty:
                ax.hist(un, bins=40, density=True, color="#E45756",
                        alpha=0.55, label="unused")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.set_title(col, fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, frameon=True, framealpha=0.9)
        for idx in range(len(bias_cols), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle("Selection-bias diagnostics (used vs unused)", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_dir / "selection_bias_diagnostics.png", dpi=140)
        plt.close(fig)

    # --- 6. Iso-rate contours (eff vs flux with constant-rate lines) ---
    _plot_iso_rate_contours(filtered, plot_dir)

    log.info("Plots saved to %s", plot_dir)


if __name__ == "__main__":
    raise SystemExit(main())
