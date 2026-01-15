#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable
from typing import List

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


def apply_offsets(values: np.ndarray, offsets: List[List[float]], plane_idx: int, strip_idx: int) -> np.ndarray:
    return values + float(offsets[plane_idx - 1][strip_idx - 1])


def apply_calibration(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    tfront_offsets = cfg.get("tfront_offsets", [[0, 0, 0, 0]] * 4)
    tback_offsets = cfg.get("tback_offsets", [[0, 0, 0, 0]] * 4)

    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            tf_col = f"T_front_{plane_idx}_s{strip_idx}"
            tb_col = f"T_back_{plane_idx}_s{strip_idx}"
            qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
            qb_col = f"Q_back_{plane_idx}_s{strip_idx}"

            if tf_col in out.columns:
                vals = out[tf_col].to_numpy(dtype=float)
                mask = (vals != 0) & ~np.isnan(vals)
                if mask.any():
                    vals[mask] = apply_offsets(vals[mask], tfront_offsets, plane_idx, strip_idx)
                out[tf_col] = vals
            if tb_col in out.columns:
                vals = out[tb_col].to_numpy(dtype=float)
                mask = (vals != 0) & ~np.isnan(vals)
                if mask.any():
                    vals[mask] = apply_offsets(vals[mask], tback_offsets, plane_idx, strip_idx)
                out[tb_col] = vals

    return out


def plot_calibrated_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                ax = axes[plane_idx - 1, strip_idx - 1]
                tf_col = f"T_front_{plane_idx}_s{strip_idx}"
                tb_col = f"T_back_{plane_idx}_s{strip_idx}"
                if tf_col not in df.columns and tb_col not in df.columns:
                    ax.axis("off")
                    continue
                if tf_col in df.columns:
                    vals = df[tf_col].to_numpy(dtype=float)
                    vals = vals[(~np.isnan(vals)) & (vals != 0)]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if tb_col in df.columns:
                    vals = df[tb_col].to_numpy(dtype=float)
                    vals = vals[(~np.isnan(vals)) & (vals != 0)]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("time (ns)")
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
                qf_col = f"Q_front_{plane_idx}_s{strip_idx}"
                qb_col = f"Q_back_{plane_idx}_s{strip_idx}"
                if qf_col not in df.columns and qb_col not in df.columns:
                    ax.axis("off")
                    continue
                if qf_col in df.columns:
                    vals = df[qf_col].to_numpy(dtype=float)
                    vals = vals[vals != 0]
                    ax.hist(vals, bins=80, color="steelblue", alpha=0.6, label="front")
                if qb_col in df.columns:
                    vals = df[qb_col].to_numpy(dtype=float)
                    vals = vals[vals != 0]
                    ax.hist(vals, bins=80, color="darkorange", alpha=0.6, label="back")
                ax.set_title(f"P{plane_idx} S{strip_idx}")
                ax.set_xlabel("charge")
        for ax in axes.flatten():
            for patch in ax.patches:
                patch.set_rasterized(True)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: apply offsets to front/back times and charges.")
    parser.add_argument("--config", default="config_step_7.yaml", help="Path to step config YAML")
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

    input_glob = cfg.get("input_glob", "**/geom_*_frontback.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is not None and str(geometry_id).lower() != "auto":
        geometry_id = int(geometry_id)
    else:
        geometry_id = None
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 7 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")

    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        for out_file in sorted(output_dir.rglob(f"SIM_RUN_*/geom_*_calibrated.{output_format}")):
            df, _ = load_with_metadata(out_file)
            plot_path = out_file.with_name(f"{out_file.stem}_plots.pdf")
            plot_calibrated_summary(df, plot_path)
            print(f"Saved {plot_path}")
        for manifest_path in sorted(output_dir.rglob("SIM_RUN_*/geom_*_calibrated.chunks.json")):
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
            plot_calibrated_summary(df, plot_path)
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
            p for p in input_paths if normalize_stem(p) == f"{geom_key}_frontback"
        ]
        if not input_paths:
            fallback_path = input_run_dir / f"{geom_key}_frontback.chunks.json"
            if fallback_path.exists():
                input_paths = [fallback_path]
    elif not input_paths:
        input_paths = sorted(input_run_dir.glob("geom_*_frontback.chunks.json"))
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
        output_dir, "STEP_7", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)

    out_stem = normalized_stem.replace("_frontback", "") + "_calibrated"
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_7",
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
                yield apply_calibration(chunk, cfg)

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
            plot_calibrated_summary(plot_df, plot_path)
        print(f"Saved {manifest_path}")
    else:
        df, upstream_meta = load_with_metadata(input_path)
        out = apply_calibration(df, cfg)
        out_path = sim_run_dir / f"{out_stem}.{output_format}"
        save_with_metadata(out, out_path, metadata, output_format)
        if not args.no_plots:
            plot_path = sim_run_dir / f"{out_path.stem}_plots.pdf"
            plot_calibrated_summary(out, plot_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
