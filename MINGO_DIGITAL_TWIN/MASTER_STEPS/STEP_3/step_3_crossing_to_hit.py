#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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


def normalize_tt_series(series: pd.Series) -> pd.Series:
    tt = series.astype("string").fillna("")
    tt = tt.str.strip()
    tt = tt.str.replace(r"\.0$", "", regex=True)
    tt = tt.replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
    return tt


def build_avalanche(
    df: pd.DataFrame,
    efficiencies: list[float],
    gain: float,
    townsend_alpha: float,
    gap_mm: float,
    electron_sigma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    tt_array = np.full(n, "", dtype=object)

    for plane_idx in range(1, 5):
        x_col = f"X_gen_{plane_idx}"
        y_col = f"Y_gen_{plane_idx}"
        if x_col not in out.columns or y_col not in out.columns:
            continue

        x_vals = out[x_col].to_numpy(dtype=float)
        y_vals = out[y_col].to_numpy(dtype=float)
        hit_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)

        eff = float(efficiencies[plane_idx - 1])
        if not (0.0 < eff < 1.0):
            raise ValueError(f"Efficiency must be in (0,1) for plane {plane_idx}, got {eff}")
        ion_lambda = -np.log(1.0 - eff)
        ions = np.zeros(n, dtype=int)
        ions[hit_mask] = rng.poisson(ion_lambda, size=hit_mask.sum())
        avalanche_exists = ions > 0

        avalanche_x = np.where(avalanche_exists, x_vals, np.nan)
        avalanche_y = np.where(avalanche_exists, y_vals, np.nan)
        avalanche_qsum = np.where(avalanche_exists, ions * gain, 0.0)
        avalanche_size = np.zeros(n, dtype=float)
        if avalanche_exists.any():
            gain_factor = np.exp(townsend_alpha * gap_mm)
            smear = rng.lognormal(mean=0.0, sigma=electron_sigma, size=avalanche_exists.sum())
            avalanche_size[avalanche_exists] = ions[avalanche_exists] * gain_factor * smear

        out[f"avalanche_ion_{plane_idx}"] = ions
        out[f"avalanche_exists_{plane_idx}"] = avalanche_exists
        out[f"avalanche_x_{plane_idx}"] = avalanche_x
        out[f"avalanche_y_{plane_idx}"] = avalanche_y
        out[f"avalanche_qsum_{plane_idx}"] = avalanche_qsum
        out[f"avalanche_size_electrons_{plane_idx}"] = avalanche_size

        tt_array[avalanche_exists] = tt_array[avalanche_exists] + str(plane_idx)

    tt_series = pd.Series(tt_array, dtype="string").replace("", pd.NA)
    out["tt_avalanche"] = tt_series
    return out


def plot_avalanche_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = normalize_tt_series(df["tt_avalanche"]).value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
    for patch in bars:
        patch.set_rasterized(True)
    ax.set_title("tt_avalanche counts")
    ax.set_xlabel("tt_avalanche")
    ax.set_ylabel("Counts")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        col = f"avalanche_ion_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=60, color="slateblue", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche_ion")
        ax.set_xlim(left=0)
    axes[-1].set_xlabel("ionizations")
    for ax in axes:
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for plane_idx, ax in enumerate(axes, start=1):
        col = f"avalanche_size_electrons_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=120, color="darkorange", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche size")
        ax.set_xlim(left=0)
    axes[-1].set_xlabel("avalanche size (electrons)")
    for ax in axes:
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def plot_step3_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))

    counts = normalize_tt_series(df["tt_avalanche"]).value_counts().sort_index()
    bars = axes[0].bar(counts.index, counts.values, color="steelblue", alpha=0.8)
    for patch in bars:
        patch.set_rasterized(True)
    axes[0].set_title("tt_avalanche")

    for plane_idx in range(1, 5):
        col = f"avalanche_size_electrons_{plane_idx}"
        if col not in df.columns:
            axes[plane_idx].axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        axes[plane_idx].hist(vals, bins=120, color="darkorange", alpha=0.8)
        axes[plane_idx].set_title(f"Plane {plane_idx} avalanche size")
        if plane_idx > 1:
            axes[plane_idx].sharex(axes[1])
        axes[plane_idx].set_xlim(left=0)
        for patch in axes[plane_idx].patches:
            patch.set_rasterized(True)
    axes[-1].set_xlabel("avalanche size (electrons)")

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: crossing -> avalanche (ionizations + centers).")
    parser.add_argument("--config", default="config_step_3.yaml", help="Path to step config YAML")
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
    efficiencies = [float(x) for x in cfg.get("efficiencies", [0.9, 0.9, 0.9, 0.9])]
    gain = float(cfg.get("avalanche_gain", 1.0))
    townsend_alpha = float(cfg.get("townsend_alpha_per_mm", 0.1))
    gap_mm = float(cfg.get("avalanche_gap_mm", 1.0))
    electron_sigma = float(cfg.get("avalanche_electron_sigma", 0.2))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/geom_*.pkl")
    geometry_id = cfg.get("geometry_id")
    if geometry_id is not None and str(geometry_id).lower() != "auto":
        geometry_id = int(geometry_id)
    else:
        geometry_id = None
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("Step 3 starting...")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"efficiencies: {efficiencies}")
    print(f"geometry_id: {geometry_id}")
    print(f"input_sim_run: {input_sim_run}")

    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        for out_file in sorted(output_dir.rglob(f"SIM_RUN_*/geom_*_avalanche.{output_format}")):
            df, _ = load_with_metadata(out_file)
            plot_path = out_file.with_name(f"{out_file.stem}_plots.pdf")
            with PdfPages(plot_path) as pdf:
                plot_avalanche_summary(df, pdf)
                plot_step3_summary(df, pdf)
            print(f"Saved {plot_path}")

        for manifest_path in sorted(output_dir.rglob("SIM_RUN_*/geom_*_avalanche.chunks.json")):
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
            with PdfPages(plot_path) as pdf:
                plot_avalanche_summary(df, pdf)
                plot_step3_summary(df, pdf)
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
        input_paths = [p for p in input_paths if normalize_stem(p) == geom_key]
        if not input_paths:
            fallback_path = input_run_dir / f"{geom_key}.chunks.json"
            if fallback_path.exists():
                input_paths = [fallback_path]
    else:
        input_paths = sorted(input_run_dir.glob("geom_*.pkl"))
        if not input_paths:
            input_paths = sorted(input_run_dir.glob("geom_*.chunks.json"))
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input for geometry {geometry_id}, found {len(input_paths)}.")

    input_path = input_paths[0]
    normalized_stem = normalize_stem(input_path)
    if geometry_id is None:
        parts = normalized_stem.split("_")
        if len(parts) < 2 or parts[0] != "geom":
            raise ValueError(f"Unable to infer geometry_id from {input_path.name}")
        geometry_id = int(parts[1])
    print(f"Processing: {input_path}")
    input_iter, upstream_meta, chunked_input = iter_input_frames(input_path, chunk_rows)

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_3", config_path, cfg, upstream_meta
    )
    print(f"Resolved output sim_run: {sim_run}")
    reset_dir(sim_run_dir)
    print(f"Output dir reset: {sim_run_dir}")

    out_stem = f"{normalized_stem}_avalanche"
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_3",
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
                yield build_avalanche(chunk, efficiencies, gain, townsend_alpha, gap_mm, electron_sigma, rng)

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
        print(f"Saved data: {manifest_path}")
        if not args.no_plots and plot_df is not None:
            plot_path = sim_run_dir / f"{out_stem}_plots.pdf"
            with PdfPages(plot_path) as pdf:
                plot_avalanche_summary(plot_df, pdf)
                plot_step3_summary(plot_df, pdf)
            print(f"Saved plots: {plot_path}")
    else:
        df, upstream_meta = load_with_metadata(input_path)
        print(f"Loaded {len(df):,} rows from {input_path.name}")
        out = build_avalanche(df, efficiencies, gain, townsend_alpha, gap_mm, electron_sigma, rng)
        print("Avalanche build complete.")
        out_name = f"{out_stem}.{output_format}"
        out_path = sim_run_dir / out_name
        save_with_metadata(out, out_path, metadata, output_format)
        print(f"Saved data: {out_path}")
        plot_sample_size = cfg.get("plot_sample_size")
        plot_df = out
        if plot_sample_size:
            plot_sample_size = int(plot_sample_size)
            if 0 < plot_sample_size < len(plot_df):
                plot_df = plot_df.sample(n=plot_sample_size, random_state=cfg.get("seed"))
                print(f"Plotting with sample size: {len(plot_df):,}")
        del df
        del out
        gc.collect()

        if not args.no_plots:
            plot_path = sim_run_dir / f"{out_path.stem}_plots.pdf"
            with PdfPages(plot_path) as pdf:
                plot_avalanche_summary(plot_df, pdf)
                plot_step3_summary(plot_df, pdf)
            print(f"Saved plots: {plot_path}")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
