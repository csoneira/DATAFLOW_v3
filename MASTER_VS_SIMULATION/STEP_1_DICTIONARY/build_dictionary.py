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
    plot_histogram,
    plot_scatter,
    setup_logger,
)

log = setup_logger("STEP_1")

DEFAULT_CONFIG = STEP_DIR / "config" / "pipeline_config.json"
DEFAULT_OUT = STEP_DIR / "output"


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _generate_plots(csv_path: Path) -> None:
    """Produce quick-look histograms and scatters from the dictionary CSV.

    Consolidated figures:
    - hist_flux_cos_n.png          : flux + cos_n histograms (1×2)
    - hist_z_planes.png            : z_plane_1..4 histograms (2×2)
    - scatter_flux_vs_cos_n.png    : single scatter
    - scatter_flux_vs_z_planes.png : flux vs z_plane_1..4 (2×2)
    - scatter_z_plane_pairs.png    : all z-plane pair combos (corner plot)
    - scatter_flux_vs_rates.png    : flux vs rate columns (multi-panel)
    """
    df = pd.read_csv(csv_path, low_memory=False)
    plot_dir = csv_path.parent / "plots"

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

    # --- z-plane histograms (2×2) ---
    z_cols = [f"z_plane_{i}" for i in range(1, 5) if f"z_plane_{i}" in df.columns]
    if z_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, col in enumerate(z_cols):
            ax = axes[idx // 2, idx % 2]
            series = pd.to_numeric(df.get(col), errors="coerce").dropna()
            if not series.empty:
                ax.hist(series, bins=40, color="#4C78A8", alpha=0.85,
                        edgecolor="white")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col}")
            ax.grid(True, alpha=0.2)
        # Hide unused subplots
        for idx in range(len(z_cols), 4):
            axes[idx // 2, idx % 2].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_dir / "hist_z_planes.png", dpi=140)
        plt.close(fig)

    # --- flux vs cos_n scatter (single) ---
    plot_scatter(df, "flux_cm2_min", "cos_n",
                 plot_dir / "scatter_flux_vs_cos_n.png")

    # --- flux vs z-planes (2×2) ---
    if z_cols and "flux_cm2_min" in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, zc in enumerate(z_cols):
            ax = axes[idx // 2, idx % 2]
            x = pd.to_numeric(df.get("flux_cm2_min"), errors="coerce")
            y = pd.to_numeric(df.get(zc), errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 2:
                ax.scatter(x[mask], y[mask], s=12, alpha=0.6, color="#F58518")
            ax.set_xlabel("flux_cm2_min")
            ax.set_ylabel(zc)
            ax.set_title(f"{zc} vs flux")
            ax.grid(True, alpha=0.2)
        for idx in range(len(z_cols), 4):
            axes[idx // 2, idx % 2].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_dir / "scatter_flux_vs_z_planes.png", dpi=140)
        plt.close(fig)

    # --- z-plane pair corner plot ---
    import itertools
    z_pairs = list(itertools.combinations(z_cols, 2))
    if z_pairs:
        n_pairs = len(z_pairs)
        ncols = min(3, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, (ci, cj) in enumerate(z_pairs):
            ax = axes_flat[idx]
            x = pd.to_numeric(df.get(ci), errors="coerce")
            y = pd.to_numeric(df.get(cj), errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 2:
                ax.scatter(x[mask], y[mask], s=12, alpha=0.6, color="#F58518")
            ax.set_xlabel(ci)
            ax.set_ylabel(cj)
            ax.set_title(f"{cj} vs {ci}")
            ax.grid(True, alpha=0.2)
        for idx in range(len(z_pairs), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.tight_layout()
        fig.savefig(plot_dir / "scatter_z_plane_pairs.png", dpi=140)
        plt.close(fig)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"task_{args.task_id:02d}" / "param_metadata_dictionary.csv"

    builder_script = STEP_DIR / "STEP_1_BUILD" / "build_param_metadata_dictionary.py"
    if not builder_script.exists():
        raise FileNotFoundError(
            f"Builder script not found: {builder_script}\n"
            "Expected at MASTER_VS_SIMULATION/STEP_1_DICTIONARY/STEP_1_BUILD/."
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
        _generate_plots(out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
