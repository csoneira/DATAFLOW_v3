#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

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
    iter_input_frames,
    latest_sim_run,
    load_with_metadata,
    now_iso,
    resolve_sim_run,
    reset_dir,
    save_with_metadata,
    write_chunked_output,
)


def compute_tdiff_qdiff(df: pd.DataFrame, c_mm_per_ns: float, qdiff_frac: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    x_to_time_factor = 3.0 / (2.0 * c_mm_per_ns)

    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            x_col = f"X_mea_{plane_idx}_s{strip_idx}"
            qsum_col = f"Y_mea_{plane_idx}_s{strip_idx}"
            tsum_col = f"T_sum_meas_{plane_idx}_s{strip_idx}"
            if x_col not in out.columns or qsum_col not in out.columns:
                continue

            x_vals = out[x_col].to_numpy(dtype=float)
            qsum_vals = out[qsum_col].to_numpy(dtype=float)
            tdiff = x_vals * x_to_time_factor
            qdiff = np.zeros(len(out), dtype=float)
            hit_mask = qsum_vals > 0
            if hit_mask.any():
                qdiff[hit_mask] = rng.normal(0.0, qdiff_frac * qsum_vals[hit_mask], size=hit_mask.sum())

            out[f"T_diff_{plane_idx}_s{strip_idx}"] = tdiff
            out[f"q_diff_{plane_idx}_s{strip_idx}"] = qdiff

            if tsum_col in out.columns:
                out[tsum_col] = out[tsum_col]

    return out


def plot_signal_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        qsum_cols = [c for c in df.columns if c.startswith("Y_mea_") and "_s" in c]
        qsum_vals = df[qsum_cols].to_numpy(dtype=float).ravel() if qsum_cols else np.array([])
        qsum_vals = qsum_vals[qsum_vals > 0]
        axes[0].hist(qsum_vals, bins=60, color="steelblue", alpha=0.8)
        axes[0].set_title("qsum (all strips)")
        axes[0].set_xlabel("qsum")

        qdiff_cols = [c for c in df.columns if c.startswith("q_diff_")]
        qdiff_vals = df[qdiff_cols].to_numpy(dtype=float).ravel() if qdiff_cols else np.array([])
        qdiff_vals = qdiff_vals[qdiff_vals != 0]
        axes[1].hist(qdiff_vals, bins=60, color="seagreen", alpha=0.8)
        axes[1].set_title("q_diff (all strips)")
        axes[1].set_xlabel("q_diff")

        tdiff_cols = [c for c in df.columns if c.startswith("T_diff_")]
        tdiff_vals = df[tdiff_cols].to_numpy(dtype=float).ravel() if tdiff_cols else np.array([])
        tdiff_vals = tdiff_vals[~np.isnan(tdiff_vals)]
        axes[2].hist(tdiff_vals, bins=60, color="darkorange", alpha=0.8)
        axes[2].set_title("T_diff (all strips)")
        axes[2].set_xlabel("T_diff (ns)")

        tsum_cols = [c for c in df.columns if c.startswith("T_sum_meas_")]
        tsum_vals = df[tsum_cols].to_numpy(dtype=float).ravel() if tsum_cols else np.array([])
        tsum_vals = tsum_vals[~np.isnan(tsum_vals)]
        axes[3].hist(tsum_vals, bins=60, color="slateblue", alpha=0.8)
        axes[3].set_title("T_sum_meas (all strips)")
        axes[3].set_xlabel("T_sum_meas (ns)")

        for ax in axes:
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"T_diff_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                ax.hist(vals, bins=80, color="darkorange", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("T_diff (ns)")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"T_sum_meas_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                ax.hist(vals, bins=80, color="slateblue", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("T_sum_meas (ns)")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"Y_mea_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[vals > 0]
                ax.hist(vals, bins=80, color="seagreen", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("qsum")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                col = f"q_diff_{plane_idx}_s{strip_idx}"
                if col not in df.columns:
                    ax.axis("off")
                    continue
                vals = df[col].to_numpy(dtype=float)
                vals = vals[vals != 0]
                ax.hist(vals, bins=80, color="steelblue", alpha=0.8)
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("q_diff")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: compute T_diff and q_diff.")
    parser.add_argument("--config", default="config_step_5.yaml", help="Path to step config YAML")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing outputs")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    input_dir = Path(cfg["input_dir"])
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parent / input_dir
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    output_format = str(cfg.get("output_format", "pkl")).lower()
    chunk_rows = cfg.get("chunk_rows")
    plot_sample_rows = cfg.get("plot_sample_rows")
    c_mm_per_ns = float(cfg.get("c_mm_per_ns", 299.792458))
    qdiff_frac = float(cfg.get("qdiff_frac", 0.01))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*_hit.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is not None and str(geometry_id).lower() != "auto":
        geometry_id = int(geometry_id)
    else:
        geometry_id = None
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 5 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        for out_file in sorted(output_dir.rglob(f"SIM_RUN_*/geom_*_signal.{output_format}")):
            df, _ = load_with_metadata(out_file)
            plot_path = out_file.with_name(f"{out_file.stem}_plots.pdf")
            plot_signal_summary(df, plot_path)
            print(f"Saved {plot_path}")
        for manifest_path in sorted(output_dir.rglob("SIM_RUN_*/geom_*_signal.chunks.json")):
            manifest = json.loads(manifest_path.read_text())
            chunks = manifest.get("chunks", [])
            if not chunks:
                continue
            last_chunk = Path(chunks[-1])
            if last_chunk.suffix == ".csv":
                df = pd.read_csv(last_chunk)
            else:
                df = pd.read_pickle(last_chunk)
            plot_path = manifest_path.with_name(f"{manifest_path.stem}_plots.pdf")
            plot_signal_summary(df, plot_path)
            print(f"Saved {plot_path}")
        return

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)

    input_run_dir = input_dir / str(input_sim_run)
    if "**" in input_glob:
        input_paths = sorted(input_run_dir.rglob(input_glob.replace("**/", "")))
    else:
        input_paths = sorted(input_run_dir.glob(input_glob))
    def normalize_stem(path: Path) -> str:
        name = path.name
        if name.endswith(".chunks.json"):
            name = name[: -len(".chunks.json")]
        stem = Path(name).stem
        return stem.replace(".chunks", "")

    if geometry_id is not None:
        geom_key = f"geom_{geometry_id}"
        input_paths = [
            p for p in input_paths if normalize_stem(p) == f"{geom_key}_hit"
        ]
        if not input_paths:
            fallback_path = input_run_dir / f"{geom_key}_hit.chunks.json"
            if fallback_path.exists():
                input_paths = [fallback_path]
    elif not input_paths:
        input_paths = sorted(input_run_dir.glob("geom_*_hit.chunks.json"))
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    normalized_stem = normalize_stem(input_path)
    if geometry_id is None:
        parts = normalized_stem.split("_")
        if len(parts) < 2 or parts[0] != "geom":
            raise ValueError(f"Unable to infer geometry_id from {input_path.stem}")
        geometry_id = int(parts[1])
    print(f"Processing: {input_path}")
    input_iter, upstream_meta, chunked_input = iter_input_frames(input_path, chunk_rows)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_5", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_stem = normalized_stem.replace("_hit", "") + "_signal"
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_5",
        "config": cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
    }
    if chunk_rows:
        def _iter_out() -> Iterable[pd.DataFrame]:
            for chunk in input_iter:
                yield compute_tdiff_qdiff(chunk, c_mm_per_ns, qdiff_frac, rng)

        manifest_path, last_chunk, row_count = write_chunked_output(
            _iter_out(),
            sim_run_dir,
            out_stem,
            output_format,
            int(chunk_rows),
            metadata,
        )
        plot_df = last_chunk
        if plot_sample_rows and plot_df is not None:
            sample_n = len(plot_df) if plot_sample_rows is True else int(plot_sample_rows)
            sample_n = min(sample_n, len(plot_df))
            plot_df = plot_df.sample(n=sample_n, random_state=cfg.get("seed"))
        if not args.no_plots and plot_df is not None:
            plot_path = sim_run_dir / f"{out_stem}_plots.pdf"
            plot_signal_summary(plot_df, plot_path)
        print(f"Saved {manifest_path}")
    else:
        df, upstream_meta = load_with_metadata(input_path)
        out = compute_tdiff_qdiff(df, c_mm_per_ns, qdiff_frac, rng)
        out_path = sim_run_dir / f"{out_stem}.{output_format}"
        save_with_metadata(out, out_path, metadata, output_format)
        if not args.no_plots:
            plot_path = sim_run_dir / f"{out_path.stem}_plots.pdf"
            plot_signal_summary(out, plot_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
