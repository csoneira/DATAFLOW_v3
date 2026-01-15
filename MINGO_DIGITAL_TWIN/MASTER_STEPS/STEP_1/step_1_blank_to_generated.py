#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

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


def generate_muon_batches(
    n_tracks: int,
    xlim: float,
    ylim: float,
    z_plane: float,
    cos_n: float,
    seed: Optional[int],
    batch_size: int,
) -> Iterable[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    exponent = 1.0 / (cos_n + 1.0)

    filled = 0
    while filled < n_tracks:
        n_batch = min(batch_size, n_tracks - filled)
        print(f"Generating batch: {filled}/{n_tracks} -> {filled + n_batch}/{n_tracks}")
        rand = rng.random((n_batch, 5))
        x_out = (rand[:, 0] * 2 - 1) * xlim
        y_out = (rand[:, 1] * 2 - 1) * ylim
        phi_out = rand[:, 3] * (2 * np.pi) - np.pi
        theta_out = np.arccos(rand[:, 4] ** exponent)
        yield pd.DataFrame({
            "X_gen": x_out,
            "Y_gen": y_out,
            "Z_gen": np.full(n_batch, z_plane),
            "Theta_gen": theta_out,
            "Phi_gen": phi_out,
            "T0_ns": np.zeros(n_batch, dtype=float),
        })
        filled += n_batch


def plot_muon_sample(df: pd.DataFrame, pdf: PdfPages) -> None:
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


def plot_step1_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
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
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
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
    chunk_rows = cfg.get("chunk_rows")
    plot_sample_rows = cfg.get("plot_sample_rows")
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
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        plot_sample_rows = cfg.get("plot_sample_rows")
        chunk_manifest = output_path.with_suffix(".chunks.json")
        if chunk_manifest.exists():
            manifest = json.loads(chunk_manifest.read_text())
            chunks = manifest.get("chunks", [])
            if not chunks:
                raise FileNotFoundError(f"No chunks listed in {chunk_manifest}")
            last_chunk = Path(chunks[-1])
            if last_chunk.suffix == ".csv":
                df = pd.read_csv(last_chunk)
            else:
                df = pd.read_pickle(last_chunk)
            if plot_sample_rows:
                sample_n = len(df) if plot_sample_rows is True else int(plot_sample_rows)
                sample_n = min(sample_n, len(df))
                df = df.sample(n=sample_n, random_state=0)
        else:
            df, _ = load_with_metadata(output_path)
    else:
        stream_csv = bool(chunk_rows) and output_format == "csv"
        stream_pkl = bool(chunk_rows) and output_format == "pkl"
        metadata = {
            "created_at": now_iso(),
            "step": "STEP_1",
            "config": cfg,
            "sim_run": sim_run,
            "config_hash": config_hash,
            "upstream_hash": upstream_hash,
        }
        generated_rows = None
        generated_path = output_path
        if stream_csv:
            total_rows = 0
            header_written = False
            last_full_batch = None
            for batch in generate_muon_batches(
                n_tracks=int(cfg["n_tracks"]),
                xlim=float(cfg["xlim_mm"]),
                ylim=float(cfg["ylim_mm"]),
                z_plane=float(cfg["z_plane_mm"]),
                cos_n=float(cfg["cos_n"]),
                seed=cfg.get("seed"),
                batch_size=int(chunk_rows),
            ):
                if len(batch) < int(chunk_rows) and int(cfg["n_tracks"]) >= int(chunk_rows):
                    continue
                total_rows += len(batch)
                batch.to_csv(output_path, mode="a", index=False, header=not header_written)
                header_written = True
                last_full_batch = batch
            metadata["row_count"] = total_rows
            meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
            meta_path.write_text(json.dumps(metadata, indent=2))
            df = None
            generated_rows = total_rows
            if not args.no_plots:
                if plot_sample_rows and last_full_batch is not None:
                    sample_n = len(last_full_batch) if plot_sample_rows is True else int(plot_sample_rows)
                    sample_n = min(sample_n, len(last_full_batch))
                    sample_df = last_full_batch.sample(n=sample_n, random_state=0)
                    plot_dir = output_path.parent / "plots"
                    ensure_dir(plot_dir)
                    plot_path = plot_dir / f"{output_path.stem}_plots.pdf"
                    with PdfPages(plot_path) as pdf:
                        plot_muon_sample(sample_df, pdf)
                        plot_step1_summary(sample_df, pdf)
                else:
                    print("Chunked CSV mode enabled; skipping plots to limit memory usage.")
        elif stream_pkl:
            chunks_dir = output_path.with_suffix("") / "chunks"
            ensure_dir(chunks_dir)
            chunks = []
            total_rows = 0
            last_full_batch = None
            for idx, batch in enumerate(
                generate_muon_batches(
                    n_tracks=int(cfg["n_tracks"]),
                    xlim=float(cfg["xlim_mm"]),
                    ylim=float(cfg["ylim_mm"]),
                    z_plane=float(cfg["z_plane_mm"]),
                    cos_n=float(cfg["cos_n"]),
                    seed=cfg.get("seed"),
                    batch_size=int(chunk_rows),
                )
            ):
                if len(batch) < int(chunk_rows) and int(cfg["n_tracks"]) >= int(chunk_rows):
                    continue
                chunk_path = chunks_dir / f"part_{idx:04d}.pkl"
                batch.to_pickle(chunk_path)
                chunks.append(str(chunk_path))
                total_rows += len(batch)
                last_full_batch = batch
            manifest = {
                "version": 1,
                "chunks": chunks,
                "row_count": total_rows,
                "metadata": metadata,
            }
            manifest_path = output_path.with_suffix(".chunks.json")
            manifest_path.write_text(json.dumps(manifest, indent=2))
            df = None
            metadata["row_count"] = total_rows
            generated_rows = total_rows
            generated_path = manifest_path
            if not args.no_plots:
                if plot_sample_rows and last_full_batch is not None:
                    sample_n = len(last_full_batch) if plot_sample_rows is True else int(plot_sample_rows)
                    sample_n = min(sample_n, len(last_full_batch))
                    sample_df = last_full_batch.sample(n=sample_n, random_state=0)
                    plot_dir = output_path.parent / "plots"
                    ensure_dir(plot_dir)
                    plot_path = plot_dir / f"{output_path.stem}_plots.pdf"
                    with PdfPages(plot_path) as pdf:
                        plot_muon_sample(sample_df, pdf)
                        plot_step1_summary(sample_df, pdf)
                else:
                    print("Chunked PKL mode enabled; skipping plots to limit memory usage.")
        else:
            df = generate_muon_sample(
                n_tracks=int(cfg["n_tracks"]),
                xlim=float(cfg["xlim_mm"]),
                ylim=float(cfg["ylim_mm"]),
                z_plane=float(cfg["z_plane_mm"]),
                cos_n=float(cfg["cos_n"]),
                seed=cfg.get("seed"),
            )
            save_with_metadata(df, output_path, metadata, output_format)
            generated_rows = len(df)

    if not args.no_plots and df is not None:
        plot_dir = output_path.parent / "plots"
        ensure_dir(plot_dir)
        plot_path = plot_dir / f"{output_path.stem}_plots.pdf"
        with PdfPages(plot_path) as pdf:
            plot_muon_sample(df, pdf)
            plot_step1_summary(df, pdf)

    if args.plot_only and not args.no_plots:
        print(f"Plotted {output_path.stem} -> {plot_path}")
    else:
        if df is None:
            print(f"Generated {generated_rows or 0} muons -> {generated_path}")
        else:
            print(f"Generated {generated_rows or len(df)} muons -> {output_path}")


if __name__ == "__main__":
    main()
