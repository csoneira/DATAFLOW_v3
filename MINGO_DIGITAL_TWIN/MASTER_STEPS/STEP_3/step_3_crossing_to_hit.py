#!/usr/bin/env python3
"""Step 3: apply efficiencies and avalanche model to crossings.

Inputs: Step 2 output.
Outputs: step_3.(pkl|csv) or step_3_chunks.chunks.json with avalanche size/position and metadata.
"""

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
    find_latest_data_path,
    find_sim_run,
    find_sim_run_dir,
    load_step_configs,
    iter_input_frames,
    latest_sim_run,
    random_sim_run,
    load_with_metadata,
    now_iso,
    resolve_param_mesh,
    resolve_sim_run,
    reset_dir,
    save_with_metadata,
    select_param_row,
    extract_param_set,
    write_chunked_output,
)


def normalize_tt_series(series: pd.Series) -> pd.Series:
    tt = series.astype("string").fillna("")
    tt = tt.str.strip()
    tt = tt.str.replace(r"\.0$", "", regex=True)
    tt = tt.replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
    return tt


def normalize_efficiency_vectors(value: object) -> list[list[float]]:
    if value is None:
        return [[0.9, 0.9, 0.9, 0.9]]
    if not isinstance(value, list):
        raise ValueError("efficiencies must be a list or list of lists.")
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [[float(v) for v in value]]
    if all(isinstance(v, list) for v in value):
        vectors: list[list[float]] = []
        for vec in value:
            if len(vec) != 4 or not all(isinstance(v, (int, float)) for v in vec):
                raise ValueError("Each efficiencies vector must contain 4 numeric values.")
            vectors.append([float(v) for v in vec])
        return vectors
    raise ValueError("efficiencies must be a 4-value list or a list of 4-value lists.")


def is_random_value(value: object) -> bool:
    return isinstance(value, str) and value.lower() == "random"




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
        if not (0.0 < eff <= 1.0):
            raise ValueError(f"Efficiency must be in (0,1] for plane {plane_idx}, got {eff}")
        if eff >= 1.0:
            eff = 1.0 - 1e-6
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


def prune_step3(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"event_id", "T_thick_s", "tt_avalanche"}
    for plane_idx in range(1, 5):
        keep.add(f"T_sum_{plane_idx}_ns")
        keep.add(f"avalanche_ion_{plane_idx}")
        keep.add(f"avalanche_exists_{plane_idx}")
        keep.add(f"avalanche_x_{plane_idx}")
        keep.add(f"avalanche_y_{plane_idx}")
        keep.add(f"avalanche_size_electrons_{plane_idx}")
    keep_cols = [col for col in df.columns if col in keep]
    return df[keep_cols]


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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        col = f"avalanche_size_electrons_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=120, color="darkorange", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche size")
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.suptitle("Avalanche size per plane (log scale)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        col = f"avalanche_ion_{plane_idx}"
        if col not in df.columns:
            ax.axis("off")
            continue
        vals = df[col].to_numpy(dtype=float)
        vals = vals[vals > 0]
        ax.hist(vals, bins=80, color="slateblue", alpha=0.8)
        ax.set_title(f"Plane {plane_idx} avalanche_ion")
        ax.set_xlim(left=0)
        ax.set_yscale("log")
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.suptitle("Ionizations per plane (log scale)")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        x_col = f"avalanche_x_{plane_idx}"
        y_col = f"avalanche_y_{plane_idx}"
        if x_col not in df.columns or y_col not in df.columns:
            ax.axis("off")
            continue
        x_vals = df[x_col].to_numpy(dtype=float)
        y_vals = df[y_col].to_numpy(dtype=float)
        mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        ax.hist2d(x_vals[mask], y_vals[mask], bins=60, cmap="viridis")
        ax.set_title(f"Plane {plane_idx} avalanche center")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.suptitle("Avalanche center positions")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for plane_idx, ax in enumerate(axes.flatten(), start=1):
        size_col = f"avalanche_size_electrons_{plane_idx}"
        ion_col = f"avalanche_ion_{plane_idx}"
        if size_col not in df.columns or ion_col not in df.columns:
            ax.axis("off")
            continue
        size_vals = df[size_col].to_numpy(dtype=float)
        ion_vals = df[ion_col].to_numpy(dtype=float)
        mask = (size_vals > 0) & (ion_vals > 0)
        ax.scatter(ion_vals[mask], size_vals[mask], s=2, alpha=0.2, rasterized=True)
        ax.set_title(f"Plane {plane_idx} size vs ion")
        ax.set_xlabel("ionizations")
        ax.set_ylabel("avalanche size")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    fig.suptitle("Avalanche size vs ionizations")
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)




def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: crossing -> avalanche (ionizations + centers).")
    parser.add_argument("--config", default="config_step_3_physics.yaml", help="Path to step physics config YAML")
    parser.add_argument(
        "--runtime-config",
        default=None,
        help="Path to step runtime config YAML (defaults to *_runtime.yaml)",
    )
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing outputs")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--force", action="store_true", help="Recompute even if sim_run exists")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    runtime_path = Path(args.runtime_config) if args.runtime_config else None
    if runtime_path is not None and not runtime_path.is_absolute():
        runtime_path = Path(__file__).resolve().parent / runtime_path

    physics_cfg, runtime_cfg, cfg, runtime_path = load_step_configs(config_path, runtime_path)

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
    gain = float(cfg.get("avalanche_gain", 1.0))
    townsend_alpha = float(cfg.get("townsend_alpha_per_mm", 0.1))
    gap_mm = float(cfg.get("avalanche_gap_mm", 1.0))
    electron_sigma = float(cfg.get("avalanche_electron_sigma", 0.2))
    rng = np.random.default_rng(cfg.get("seed"))

    input_glob = cfg.get("input_glob", "**/step_2_chunks.chunks.json")
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("\n-----\nStep 3 starting...\n-----")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"input_sim_run: {input_sim_run}")

    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        latest_path = find_latest_data_path(output_dir)
        if latest_path is None:
            raise FileNotFoundError(f"No existing outputs found in {output_dir} for plot-only.")
        df, _ = load_with_metadata(latest_path)
        sim_run_dir = find_sim_run_dir(latest_path)
        plot_dir = (sim_run_dir or latest_path.parent) / "PLOTS"
        ensure_dir(plot_dir)
        plot_path = plot_dir / f"{latest_path.stem}_plots.pdf"
        with PdfPages(plot_path) as pdf:
            plot_avalanche_summary(df, pdf)
        print(f"Saved {plot_path}")
        return

    picked_random = False
    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)
    elif input_sim_run == "random":
        input_sim_run = random_sim_run(input_dir, cfg.get("seed"))
        picked_random = True

    def normalize_stem(path: Path) -> str:
        name = path.name
        if name.endswith(".chunks.json"):
            name = name[: -len(".chunks.json")]
        stem = Path(name).stem
        return stem.replace(".chunks", "")

    def collect_input_paths(run_dir: Path) -> list[Path]:
        if "**" in input_glob:
            input_paths = sorted(run_dir.rglob(input_glob.replace("**/", "")))
        else:
            input_paths = sorted(run_dir.glob(input_glob))
        return input_paths

    input_run_dir = input_dir / str(input_sim_run)
    input_paths = collect_input_paths(input_run_dir)
    if not input_paths and picked_random:
        fallback_sim_run = latest_sim_run(input_dir)
        if fallback_sim_run != input_sim_run:
            print(
                f"Warning: no inputs found in {input_run_dir}; "
                f"falling back to latest SIM_RUN {fallback_sim_run}."
            )
            input_sim_run = fallback_sim_run
            input_run_dir = input_dir / str(input_sim_run)
            input_paths = collect_input_paths(input_run_dir)
    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input, found {len(input_paths)}.")

    input_path = input_paths[0]
    print(f"Processing: {input_path}")
    input_iter, upstream_meta, chunked_input = iter_input_frames(input_path, chunk_rows)
    param_set_id, param_date = extract_param_set(upstream_meta)
    param_mesh_path = None
    eff_idx = None
    efficiency_vectors = None
    if is_random_value(cfg.get("efficiencies")):
        mesh_dir = Path(cfg.get("param_mesh_dir", "../../INTERSTEPS/STEP_0_TO_1"))
        if not mesh_dir.is_absolute():
            mesh_dir = Path(__file__).resolve().parent / mesh_dir
        mesh, mesh_path = resolve_param_mesh(mesh_dir, cfg.get("param_mesh_sim_run", "latest"), cfg.get("seed"))
        param_row = select_param_row(mesh, rng, param_set_id)
        if "param_set_id" in param_row.index and pd.notna(param_row["param_set_id"]):
            param_set_id = int(param_row["param_set_id"])
        if "param_date" in param_row:
            param_date = str(param_row["param_date"])
        param_mesh_path = mesh_path
        required = ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]
        if not all(col in param_row.index for col in required):
            raise ValueError("param_mesh.csv is missing eff_p1..eff_p4 required for efficiencies.")
        efficiencies = [float(param_row[col]) for col in required]
    else:
        efficiency_vectors = normalize_efficiency_vectors(cfg.get("efficiencies"))
        eff_idx = int(rng.integers(0, len(efficiency_vectors)))
        efficiencies = efficiency_vectors[eff_idx]

    physics_cfg_run = dict(physics_cfg)
    physics_cfg_run["efficiencies"] = efficiencies
    if efficiency_vectors is not None and len(efficiency_vectors) > 1:
        print(f"efficiencies candidates: {len(efficiency_vectors)} (selected index {eff_idx})")
    print(f"efficiencies: {efficiencies}")
    if param_set_id is not None:
        print(f"param_set_id: {param_set_id} (date {param_date})")
    if not args.force:
        existing = find_sim_run(output_dir, physics_cfg_run, upstream_meta)
        if existing:
            print(f"SIM_RUN {existing} already exists; skipping (use --force to regenerate).")
            return

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_3", config_path, physics_cfg_run, upstream_meta
    )
    print(f"Resolved output sim_run: {sim_run}")
    reset_dir(sim_run_dir)
    print(f"Output dir reset: {sim_run_dir}")

    out_stem_base = "step_3"
    out_stem = f"{out_stem_base}_chunks" if chunk_rows else out_stem_base
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_3",
        "config": physics_cfg_run,
        "runtime_config": runtime_cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
        "param_set_id": param_set_id,
        "param_date": param_date,
        "param_mesh_path": str(param_mesh_path) if param_mesh_path else None,
    }
    if chunk_rows:
        def _iter_out() -> Iterable[pd.DataFrame]:
            for chunk in input_iter:
                yield prune_step3(build_avalanche(chunk, efficiencies, gain, townsend_alpha, gap_mm, electron_sigma, rng))

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
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_stem_base}_plots.pdf"
            with PdfPages(plot_path) as pdf:
                plot_avalanche_summary(plot_df, pdf)
            print(f"Saved plots: {plot_path}")
    else:
        df, upstream_meta = load_with_metadata(input_path)
        print(f"Loaded {len(df):,} rows from {input_path.name}")
        out = prune_step3(build_avalanche(df, efficiencies, gain, townsend_alpha, gap_mm, electron_sigma, rng))
        print("Avalanche build complete.")
        out_name = f"{out_stem_base}.{output_format}"
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
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_path.stem}_plots.pdf"
            with PdfPages(plot_path) as pdf:
                plot_avalanche_summary(plot_df, pdf)
            print(f"Saved plots: {plot_path}")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
