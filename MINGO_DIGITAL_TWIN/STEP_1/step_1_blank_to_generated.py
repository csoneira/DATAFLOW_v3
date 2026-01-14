#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from STEP_SHARED.sim_utils import (
    ensure_dir,
    find_sim_run,
    load_with_metadata,
    now_iso,
    resolve_sim_run,
    reset_dir,
    save_with_metadata,
)


def generate_muon_sample(
    n_tracks: int,
    xlim: float,
    ylim: float,
    z_plane: float,
    cos_n: float,
    seed: Optional[int],
    batch_size: int = 200_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    exponent = 1.0 / (cos_n + 1.0)

    x_out = np.empty(n_tracks)
    y_out = np.empty(n_tracks)
    z_out = np.full(n_tracks, z_plane)
    phi_out = np.empty(n_tracks)
    theta_out = np.empty(n_tracks)

    filled = 0
    while filled < n_tracks:
        n_batch = min(batch_size, n_tracks - filled)
        print(f"Generating batch: {filled}/{n_tracks} -> {filled + n_batch}/{n_tracks}")
        rand = rng.random((n_batch, 5))
        x_out[filled:filled + n_batch] = (rand[:, 0] * 2 - 1) * xlim
        y_out[filled:filled + n_batch] = (rand[:, 1] * 2 - 1) * ylim
        phi_out[filled:filled + n_batch] = rand[:, 3] * (2 * np.pi) - np.pi
        theta_out[filled:filled + n_batch] = np.arccos(rand[:, 4] ** exponent)
        filled += n_batch

    return pd.DataFrame({
        "X_gen": x_out,
        "Y_gen": y_out,
        "Z_gen": z_out,
        "Theta_gen": theta_out,
        "Phi_gen": phi_out,
        "T0_ns": np.zeros(n_tracks, dtype=float),
    })


def plot_muon_sample(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        _, _, patches = axes[0, 0].hist(df["X_gen"], bins=100, color="steelblue", alpha=0.8)
        for patch in patches:
            patch.set_rasterized(True)
        axes[0, 0].set_title("X_gen distribution")
        axes[0, 0].set_xlabel("X (mm)")
        axes[0, 0].set_ylabel("Counts")

        _, _, patches = axes[0, 1].hist(df["Y_gen"], bins=100, color="seagreen", alpha=0.8)
        for patch in patches:
            patch.set_rasterized(True)
        axes[0, 1].set_title("Y_gen distribution")
        axes[0, 1].set_xlabel("Y (mm)")

        _, _, patches = axes[1, 0].hist(df["Theta_gen"], bins=100, color="darkorange", alpha=0.8)
        for patch in patches:
            patch.set_rasterized(True)
        axes[1, 0].set_title("Theta_gen distribution")
        axes[1, 0].set_xlabel("Theta (rad)")

        _, _, patches = axes[1, 1].hist(df["Phi_gen"], bins=100, color="slateblue", alpha=0.8)
        for patch in patches:
            patch.set_rasterized(True)
        axes[1, 1].set_title("Phi_gen distribution")
        axes[1, 1].set_xlabel("Phi (rad)")

        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["Theta_gen"], df["Phi_gen"], s=1, alpha=0.2, rasterized=True)
        ax.set_title("Theta vs Phi")
        ax.set_xlabel("Theta (rad)")
        ax.set_ylabel("Phi (rad)")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def plot_step1_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].hist(df["X_gen"], bins=80, color="steelblue", alpha=0.8)
        axes[0, 0].set_title("X_gen")
        axes[0, 1].hist(df["Y_gen"], bins=80, color="seagreen", alpha=0.8)
        axes[0, 1].set_title("Y_gen")
        axes[1, 0].hist(df["Theta_gen"], bins=80, color="darkorange", alpha=0.8)
        axes[1, 0].set_title("Theta_gen")
        axes[1, 1].hist(df["Phi_gen"], bins=80, color="slateblue", alpha=0.8)
        axes[1, 1].set_title("Phi_gen")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: generate muon sample (x,y,z,theta,phi).")
    parser.add_argument("--config", default="config_step_1.yaml", help="Path to step config YAML")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing output")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    output_format = str(cfg.get("output_format", "pkl")).lower()
    output_name = f"{cfg.get('output_basename', 'muon_sample')}_{int(cfg['n_tracks'])}.{output_format}"
    if args.plot_only:
        sim_run = find_sim_run(output_dir, cfg, None)
        if sim_run is None:
            raise FileNotFoundError("No matching SIM_RUN found for this config.")
        sim_run_dir = output_dir / sim_run
        output_path = sim_run_dir / output_name
        config_hash = None
        upstream_hash = None
    else:
        sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
            output_dir, "STEP_1", config_path, cfg, None
        )
        reset_dir(sim_run_dir)
        output_path = sim_run_dir / output_name
    print(
        f"Step 1 config: n_tracks={cfg['n_tracks']}, xlim={cfg['xlim_mm']}, "
        f"ylim={cfg['ylim_mm']}, z_plane={cfg['z_plane_mm']}, c={cfg.get('c_mm_per_ns', 299.792458)}"
    )
    if args.plot_only:
        df, _ = load_with_metadata(output_path)
    else:
        df = generate_muon_sample(
            n_tracks=int(cfg["n_tracks"]),
            xlim=float(cfg["xlim_mm"]),
            ylim=float(cfg["ylim_mm"]),
            z_plane=float(cfg["z_plane_mm"]),
            cos_n=float(cfg["cos_n"]),
            seed=cfg.get("seed"),
        )
        metadata = {
            "created_at": now_iso(),
            "step": "STEP_1",
            "config": cfg,
            "sim_run": sim_run,
            "config_hash": config_hash,
            "upstream_hash": upstream_hash,
        }
        save_with_metadata(df, output_path, metadata, output_format)

    plot_dir = output_path.parent / "plots"
    ensure_dir(plot_dir)
    plot_path = plot_dir / f"{output_path.stem}_summary.pdf"
    plot_muon_sample(df, plot_path)
    single_path = plot_dir / f"{output_path.stem}_single.pdf"
    plot_step1_summary(df, single_path)

    if args.plot_only:
        print(f"Plotted {output_path.stem} -> {plot_path}")
    else:
        print(f"Generated {len(df)} muons -> {output_path}")


if __name__ == "__main__":
    main()
