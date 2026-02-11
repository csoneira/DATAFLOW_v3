#!/usr/bin/env python3
"""STEP_1: Build the simulated dataset CSV for a given task.

This wrapper delegates to the builder script in ``STEP_1_BUILD/`` and
optionally produces quick-look plots to explore the dataset.
The output is a broad dataset of simulation runs; downstream steps
(especially Step 2) will filter it to select reference entries that
form the actual dictionary.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Shared utilities --------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    apply_clean_style,
    plot_histogram,
    plot_scatter,
    setup_logger,
    setup_output_dirs,
)

log = setup_logger("STEP_1")
apply_clean_style()

DEFAULT_CONFIG = STEP_DIR / "config" / "pipeline_config.json"
DEFAULT_OUT = STEP_DIR


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _generate_plots(csv_path: Path, plot_dir: Path) -> None:
    """Produce quick-look histograms and scatters from the dictionary CSV.

    Consolidated figures:
    - hist_flux_cos_n.png          : flux + cos_n histograms (1×2)
    - hist_z_planes.png            : z-plane geometry diagnostics (2×2)
    - scatter_flux_vs_cos_n.png    : single scatter
    - scatter_flux_vs_rates.png    : flux vs rate columns (multi-panel)
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Clean old plots
    if plot_dir.exists():
        for path in plot_dir.glob("*.png"):
            path.unlink()
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- flux + cos_n histograms (1×2) ---
    flux_cos_cols = [c for c in ("flux_cm2_min", "cos_n") if c in df.columns]
    if flux_cos_cols:
        fig, axes = plt.subplots(1, len(flux_cos_cols),
                                 figsize=(7 * len(flux_cos_cols), 5))
        if len(flux_cos_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, flux_cos_cols):
            series = pd.to_numeric(df.get(col), errors="coerce").dropna()
            if not series.empty:
                ax.hist(series, bins=40, color="#4C78A8", alpha=0.85,
                        edgecolor="white")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col}")
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "hist_flux_cos_n.png", dpi=140)
        plt.close(fig)

    # --- z-plane geometry diagnostics (2×2) ---
    z_cols = [f"z_plane_{i}" for i in range(1, 5) if f"z_plane_{i}" in df.columns]
    if z_cols:
        z_df = df[z_cols].apply(pd.to_numeric, errors="coerce")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel A: compact comparison of each z-plane distribution
        ax = axes[0, 0]
        box_data = [z_df[col].dropna().to_numpy(dtype=float) for col in z_cols]
        if any(len(arr) > 0 for arr in box_data):
            bp = ax.boxplot(
                box_data,
                labels=z_cols,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#4C78A8")
                patch.set_alpha(0.45)
        ax.set_title("z-plane distributions (boxplot)")
        ax.set_ylabel("z value")
        ax.grid(True, alpha=0.2)

        # Panel B: total geometry span (max(z)-min(z))
        ax = axes[0, 1]
        span = (z_df.max(axis=1) - z_df.min(axis=1)).dropna()
        if not span.empty:
            ax.hist(span, bins=40, color="#54A24B", alpha=0.85, edgecolor="white")
        ax.set_title("Geometry span: max(z)-min(z)")
        ax.set_xlabel("span")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

        # Panel C: nearest-neighbor gaps along the stack
        ax = axes[1, 0]
        sorted_z = np.sort(z_df.to_numpy(dtype=float), axis=1)
        if sorted_z.shape[1] >= 2:
            gaps = np.diff(sorted_z, axis=1)
            colors = ["#F58518", "#E45756", "#72B7B2"]
            for i in range(gaps.shape[1]):
                g = pd.Series(gaps[:, i]).dropna()
                if not g.empty:
                    ax.hist(
                        g,
                        bins=35,
                        alpha=0.45,
                        label=f"gap_{i+1}",
                        color=colors[i % len(colors)],
                        edgecolor="white",
                    )
            ax.legend(frameon=True, framealpha=0.9, fontsize=8)
        ax.set_title("Neighbor gaps in sorted z planes")
        ax.set_xlabel("gap")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

        # Panel D: stack center vs span (colored by cos_n if available)
        ax = axes[1, 1]
        center = z_df.mean(axis=1)
        mask = center.notna() & span.reindex(center.index).notna()
        if mask.sum() >= 2:
            y = span.reindex(center.index)[mask]
            if "cos_n" in df.columns:
                c = pd.to_numeric(df["cos_n"], errors="coerce").reindex(center.index)[mask]
                sc = ax.scatter(center[mask], y, c=c, s=14, alpha=0.65, cmap="viridis")
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("cos_n")
            else:
                ax.scatter(center[mask], y, s=14, alpha=0.65, color="#4C78A8")
        ax.set_title("Geometry center vs span")
        ax.set_xlabel("mean(z_planes)")
        ax.set_ylabel("span")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(plot_dir / "hist_z_planes.png", dpi=140)
        plt.close(fig)

    # --- flux vs cos_n scatter (single) ---
    plot_scatter(df, "flux_cm2_min", "cos_n",
                 plot_dir / "scatter_flux_vs_cos_n.png")

    # --- Rate-vs-flux multi-panel (§2.3) ---
    rate_cols = [c for c in df.columns
                 if "rate" in c.lower() or c.startswith("raw_tt_")]
    rate_cols = rate_cols[:8]
    if rate_cols and "flux_cm2_min" in df.columns:
        ncols = min(4, len(rate_cols))
        nrows = (len(rate_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, rc in enumerate(rate_cols):
            ax = axes_flat[idx]
            x = pd.to_numeric(df.get("flux_cm2_min"), errors="coerce")
            y = pd.to_numeric(df.get(rc), errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 2:
                ax.scatter(x[mask], y[mask], s=10, alpha=0.5, color="#F58518")
            short_name = rc.replace("_rate_hz", "").replace("raw_tt_", "tt_")
            ax.set_xlabel("flux_cm2_min")
            ax.set_ylabel(short_name)
            ax.set_title(f"{short_name} vs flux", fontsize=9)
            ax.grid(True, alpha=0.2)
        for idx in range(len(rate_cols), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_dir / "scatter_flux_vs_rates.png", dpi=140)
        plt.close(fig)

    log.info("Plots saved to %s", plot_dir)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build dictionary CSV for a task."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip quick sanity plots for the output CSV.",
    )
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    files_dir, plots_dir = setup_output_dirs(base_dir)
    out_csv = files_dir / f"task_{args.task_id:02d}" / "param_metadata_dictionary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    builder_script = STEP_DIR / "STEP_1_BUILD" / "build_param_metadata_dictionary.py"
    if not builder_script.exists():
        raise FileNotFoundError(
            f"Builder script not found: {builder_script}\n"
            "Expected at INFERENCE_DICTIONARY_VALIDATION/STEP_1_BUILD_DICTIONARY/STEP_1_BUILD/."
        )

    cmd = [
        sys.executable,
        str(builder_script),
        "--config", str(args.config),
        "--task-id", str(args.task_id),
        "--out", str(out_csv),
    ]
    log.info("Running builder: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    log.info("Dictionary written to %s", out_csv)

    if out_csv.exists() and not args.no_plots:
        _generate_plots(out_csv, plots_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
