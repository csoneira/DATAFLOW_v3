#!/usr/bin/env python3
"""STEP_2: Compute relative error for eff_p2 and eff_p3 and filter reference rows.

Loads the dictionary CSV, builds a validation table (estimated vs simulated
efficiencies), applies quality cuts on relative error and minimum event count,
and writes filtered/unfiltered outputs plus diagnostic plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Shared utilities --------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    load_config,
    plot_bar_counts,
    plot_histogram,
    plot_histogram_overlay,
    plot_scatter,
    plot_scatter_overlay,
    resolve_param,
    setup_logger,
)

log = setup_logger("STEP_2")

DEFAULT_DICT = (
    REPO_ROOT / "STEP_1_DICTIONARY" / "output" / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_OUT = STEP_DIR / "output"
DEFAULT_CONFIG = STEP_DIR / "config.json"


# -------------------------------------------------------------------------
# Efficiency sim-vs-est scatter
# -------------------------------------------------------------------------

def _plot_eff_sim_vs_est(df: pd.DataFrame, plane: int, plot_path: Path) -> None:
    sim_col, est_col = f"eff_sim_p{plane}", f"eff_est_p{plane}"
    if sim_col not in df.columns or est_col not in df.columns:
        return
    sim = pd.to_numeric(df[sim_col], errors="coerce")
    est = pd.to_numeric(df[est_col], errors="coerce")
    mask = sim.notna() & est.notna()
    if mask.sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sim[mask], est[mask], s=14, alpha=0.6, color="#4C78A8")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel(sim_col)
    ax.set_ylabel(est_col)
    ax.set_title(f"Eff {plane}: simulated vs calculated")
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
    from validate_simulation_vs_parameters import build_validation_table
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
    out_dir = Path(resolve_param(
        args.out_dir, config, "out_dir", str(DEFAULT_OUT)))
    prefix = resolve_param(args.prefix, config, "prefix", "raw")
    eff_method = resolve_param(
        args.eff_method, config, "eff_method", "four_over_three_plus_four")
    relerr_threshold = resolve_param(
        args.relerr_threshold, config, "relerr_threshold", 0.01, float)
    min_events = resolve_param(
        args.min_events, config, "min_events", 50000, int)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
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

    merged["used_in_reference"] = mask
    used_entries = merged.loc[mask].copy()
    unused_entries = merged.loc[~mask].copy()

    # --- Write outputs ---
    validation_csv = out_dir / "validation_table.csv"
    filtered_csv = out_dir / "filtered_reference.csv"
    used_csv = out_dir / "used_dictionary_entries.csv"
    unused_csv = out_dir / "unused_dictionary_entries.csv"
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
    from validate_simulation_vs_parameters import build_validation_table

    methods = ("four_over_three_plus_four", "one_minus_three_over_four")
    tables = {}
    for m in methods:
        tables[m] = build_validation_table(df, prefix=prefix, eff_method=m)

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
    """Generate all diagnostic plots for STEP 2."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_histogram(
        merged, "eff_rel_err_p2",
        plot_dir / "hist_eff_rel_err_p2.png",
        density=True, log_y=True, color="#54A24B",
    )
    plot_histogram(
        merged, "eff_rel_err_p3",
        plot_dir / "hist_eff_rel_err_p3.png",
        density=True, log_y=True, color="#54A24B",
    )
    plot_scatter(
        merged, "eff_rel_err_p2", "eff_rel_err_p3",
        plot_dir / "scatter_relerr_p2_vs_p3.png",
        color="#E45756",
    )
    _plot_eff_sim_vs_est(validation, 2,
                         plot_dir / "scatter_eff2_sim_vs_est.png")
    _plot_eff_sim_vs_est(validation, 3,
                         plot_dir / "scatter_eff3_sim_vs_est.png")

    # Overlay plots
    plot_histogram_overlay(
        used, unused, "eff_rel_err_p2",
        plot_dir / "hist_used_vs_unused_eff_rel_err_p2.png",
        "eff_rel_err_p2 (used vs unused)",
    )
    plot_histogram_overlay(
        used, unused, "eff_rel_err_p3",
        plot_dir / "hist_used_vs_unused_eff_rel_err_p3.png",
        "eff_rel_err_p3 (used vs unused)",
    )
    plot_scatter_overlay(
        used, unused, "eff_rel_err_p2", "eff_rel_err_p3",
        plot_dir / "scatter_used_vs_unused_relerr_p2_vs_p3.png",
        "Rel. error p2 vs p3 (used vs unused)",
    )
    plot_histogram_overlay(
        used, unused, "flux_cm2_min",
        plot_dir / "hist_used_vs_unused_flux.png",
        "flux_cm2_min (used vs unused)",
    )
    plot_histogram_overlay(
        used, unused, "cos_n",
        plot_dir / "hist_used_vs_unused_cos_n.png",
        "cos_n (used vs unused)",
    )

    # Count bar charts
    plot_bar_counts(
        ["total", "filtered"], [len(merged), len(filtered)],
        ["#4C78A8", "#F58518"],
        "Reference filtering by rel. error",
        plot_dir / "counts_total_vs_filtered.png",
    )
    plot_bar_counts(
        ["used", "unused"], [len(used), len(unused)],
        ["#54A24B", "#E45756"],
        "Dictionary entries used vs unused",
        plot_dir / "counts_used_vs_unused.png",
    )

    # --- Selection-bias diagnostics (to_do.md §3.3) ---
    # Compare used vs unused distributions for parameters NOT in the cut
    # to detect whether filtering on p2/p3 creates bias in p1/p4.
    for col in ("eff_rel_err_p1", "eff_rel_err_p4"):
        if col in used.columns and col in unused.columns:
            plot_histogram_overlay(
                used, unused, col,
                plot_dir / f"hist_used_vs_unused_{col}.png",
                f"{col} (used vs unused)",
            )

    # Geometry and efficiency distributions — selection bias check
    for col in ("z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"):
        if col in used.columns and col in unused.columns:
            plot_histogram_overlay(
                used, unused, col,
                plot_dir / f"hist_used_vs_unused_{col}.png",
                f"{col} (used vs unused)",
            )
    for col in ("generated_events_count",):
        if col in used.columns and col in unused.columns:
            plot_histogram_overlay(
                used, unused, col,
                plot_dir / f"hist_used_vs_unused_{col}.png",
                f"{col} (used vs unused)",
            )

    # Scatter: used vs unused in (flux, cos_n) space
    plot_scatter_overlay(
        used, unused, "flux_cm2_min", "cos_n",
        plot_dir / "scatter_used_vs_unused_flux_cosn.png",
        "Flux vs cos_n (used vs unused)",
    )

    # Scatter: p1/p4 errors for used entries — check they stay well-behaved
    for p in (1, 4):
        rcol = f"eff_rel_err_p{p}"
        if rcol in used.columns:
            plot_scatter(
                used, "generated_events_count", rcol,
                plot_dir / f"scatter_used_events_vs_{rcol}.png",
                color="#54A24B",
            )

    log.info("Plots saved to %s", plot_dir)


if __name__ == "__main__":
    raise SystemExit(main())
