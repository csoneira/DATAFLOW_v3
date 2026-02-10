#!/usr/bin/env python3
"""STEP_2: Compute relative error for eff_p2 and eff_p3 and filter reference rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
DEFAULT_DICT = (
    REPO_ROOT
    / "STEP_1_DICTIONARY"
    / "output"
    / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_OUT = REPO_ROOT / "STEP_2_SIM_VALIDATION" / "output"
DEFAULT_CONFIG = REPO_ROOT / "STEP_2_SIM_VALIDATION" / "config.json"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_histogram(df: pd.DataFrame, column: str, plot_path: Path) -> None:
    series = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(series, bins=40, density=True, color="#54A24B", alpha=0.85, edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {column}")
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, plot_path: Path) -> None:
    x = pd.to_numeric(df.get(x_col), errors="coerce")
    y = pd.to_numeric(df.get(y_col), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[mask], y[mask], s=12, alpha=0.6, color="#E45756")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_histogram_overlay(
    used_df: pd.DataFrame,
    unused_df: pd.DataFrame,
    column: str,
    plot_path: Path,
    title: str,
) -> None:
    used = pd.to_numeric(used_df.get(column), errors="coerce").dropna()
    unused = pd.to_numeric(unused_df.get(column), errors="coerce").dropna()
    if used.empty and unused.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(used, bins=40, density=True, color="#54A24B", alpha=0.65, label="used")
    ax.hist(unused, bins=40, density=True, color="#E45756", alpha=0.55, label="unused")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_scatter_overlay(
    used_df: pd.DataFrame,
    unused_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_path: Path,
    title: str,
) -> None:
    used_x = pd.to_numeric(used_df.get(x_col), errors="coerce")
    used_y = pd.to_numeric(used_df.get(y_col), errors="coerce")
    unused_x = pd.to_numeric(unused_df.get(x_col), errors="coerce")
    unused_y = pd.to_numeric(unused_df.get(y_col), errors="coerce")
    used_mask = used_x.notna() & used_y.notna()
    unused_mask = unused_x.notna() & unused_y.notna()
    if used_mask.sum() + unused_mask.sum() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(used_x[used_mask], used_y[used_mask], s=12, alpha=0.6, color="#54A24B", label="used")
    ax.scatter(
        unused_x[unused_mask],
        unused_y[unused_mask],
        s=12,
        alpha=0.45,
        color="#E45756",
        label="unused",
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_eff_sim_vs_est(df: pd.DataFrame, plane: int, plot_path: Path) -> None:
    sim_col = f"eff_sim_p{plane}"
    est_col = f"eff_est_p{plane}"
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
    ax.set_title(f"Eff {plane} simulated vs calculated")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _build_validation_table(df: pd.DataFrame, prefix: str, eff_method: str) -> pd.DataFrame:
    from validate_simulation_vs_parameters import build_validation_table

    return build_validation_table(df, prefix=prefix, eff_method=eff_method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute relative error for eff2/eff3.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--eff-method",
        default=None,
        choices=["four_over_three_plus_four", "one_minus_three_over_four"],
    )
    parser.add_argument(
        "--relerr-threshold",
        type=float,
        default=None,
        help="Absolute relative error threshold for eff2 and eff3.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="Minimum generated events required to keep an entry.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip quick sanity plots for validation outputs.",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    if args.dictionary_csv is None:
        args.dictionary_csv = str(config.get("dictionary_csv", DEFAULT_DICT))
    if args.out_dir is None:
        args.out_dir = str(config.get("out_dir", DEFAULT_OUT))
    if args.prefix is None:
        args.prefix = str(config.get("prefix", "raw"))
    if args.eff_method is None:
        args.eff_method = str(config.get("eff_method", "four_over_three_plus_four"))
    if args.relerr_threshold is None:
        args.relerr_threshold = float(config.get("relerr_threshold", 0.01))
    if args.min_events is None:
        args.min_events = int(config.get("min_events", 50000))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    if plot_dir.exists():
        for path in plot_dir.glob("*.png"):
            path.unlink()

    df = pd.read_csv(args.dictionary_csv, low_memory=False)
    validation = _build_validation_table(df, prefix=args.prefix, eff_method=args.eff_method)

    join_col = None
    for candidate in ("file_name", "filename_base"):
        if candidate in df.columns and candidate in validation.columns:
            join_col = candidate
            break
    if join_col is None:
        raise KeyError("No shared file_name/filename_base column for joining validation results.")

    rel_cols = ["eff_rel_err_p2", "eff_rel_err_p3"]
    rel_frame = validation[[join_col, "generated_events_count", *rel_cols]].copy()
    merged = df.merge(rel_frame, on=join_col, how="left")

    rel2 = merged["eff_rel_err_p2"].abs()
    rel3 = merged["eff_rel_err_p3"].abs()
    if "generated_events_count" not in merged.columns:
        raise KeyError("generated_events_count missing from validation table.")
    event_mask = merged["generated_events_count"].ge(args.min_events)
    mask = rel2.le(args.relerr_threshold) & rel3.le(args.relerr_threshold) & event_mask
    filtered = merged.loc[mask].copy()

    merged["used_in_reference"] = mask
    used_entries = merged.loc[mask].copy()
    unused_entries = merged.loc[~mask].copy()

    validation_csv = out_dir / "validation_table.csv"
    filtered_csv = out_dir / "filtered_reference.csv"
    used_csv = out_dir / "used_dictionary_entries.csv"
    unused_csv = out_dir / "unused_dictionary_entries.csv"
    validation.to_csv(validation_csv, index=False)
    filtered.to_csv(filtered_csv, index=False)
    used_entries.to_csv(used_csv, index=False)
    unused_entries.to_csv(unused_csv, index=False)

    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_histogram(merged, "eff_rel_err_p2", plot_dir / "hist_eff_rel_err_p2.png")
        _plot_histogram(merged, "eff_rel_err_p3", plot_dir / "hist_eff_rel_err_p3.png")
        _plot_scatter(merged, "eff_rel_err_p2", "eff_rel_err_p3", plot_dir / "scatter_relerr_p2_vs_p3.png")
        _plot_eff_sim_vs_est(validation, 2, plot_dir / "scatter_eff2_sim_vs_est.png")
        _plot_eff_sim_vs_est(validation, 3, plot_dir / "scatter_eff3_sim_vs_est.png")
        _plot_histogram_overlay(
            used_entries,
            unused_entries,
            "eff_rel_err_p2",
            plot_dir / "hist_used_vs_unused_eff_rel_err_p2.png",
            "eff_rel_err_p2 (used vs unused)",
        )
        _plot_histogram_overlay(
            used_entries,
            unused_entries,
            "eff_rel_err_p3",
            plot_dir / "hist_used_vs_unused_eff_rel_err_p3.png",
            "eff_rel_err_p3 (used vs unused)",
        )
        _plot_scatter_overlay(
            used_entries,
            unused_entries,
            "eff_rel_err_p2",
            "eff_rel_err_p3",
            plot_dir / "scatter_used_vs_unused_relerr_p2_vs_p3.png",
            "Rel. error p2 vs p3 (used vs unused)",
        )
        _plot_histogram_overlay(
            used_entries,
            unused_entries,
            "flux_cm2_min",
            plot_dir / "hist_used_vs_unused_flux.png",
            "flux_cm2_min (used vs unused)",
        )
        _plot_histogram_overlay(
            used_entries,
            unused_entries,
            "cos_n",
            plot_dir / "hist_used_vs_unused_cos_n.png",
            "cos_n (used vs unused)",
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["total", "filtered"], [len(merged), len(filtered)], color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Rows")
        ax.set_title("Reference filtering by rel. error")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "counts_total_vs_filtered.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["used", "unused"],
            [len(used_entries), len(unused_entries)],
            color=["#54A24B", "#E45756"],
        )
        ax.set_ylabel("Rows")
        ax.set_title("Dictionary entries used vs unused")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "counts_used_vs_unused.png", dpi=140)
        plt.close(fig)

    print(f"Wrote validation table: {validation_csv}")
    print(f"Wrote filtered reference: {filtered_csv} (rows={len(filtered)})")
    print(f"Wrote used dictionary entries: {used_csv} (rows={len(used_entries)})")
    print(f"Wrote unused dictionary entries: {unused_csv} (rows={len(unused_entries)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
