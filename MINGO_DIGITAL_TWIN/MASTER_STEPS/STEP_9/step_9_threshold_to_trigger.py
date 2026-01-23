#!/usr/bin/env python3
"""Step 9: evaluate trigger combinations and retain passing events.

Inputs: Step 8 output.
Outputs: step_9.(pkl|csv) or step_9_chunks.chunks.json with trigger tags.
"""

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
    find_latest_data_path,
    find_sim_run,
    find_sim_run_dir,
    iter_input_frames,
    latest_sim_run,
    random_sim_run,
    load_step_configs,
    load_with_metadata,
    now_iso,
    build_sim_run_name,
    register_sim_run,
    extract_step_id_chain,
    select_next_step_id,
    resolve_sim_run,
    reset_dir,
    save_with_metadata,
    write_chunked_output,
)


def normalize_tt(series: pd.Series) -> pd.Series:
    tt = series.astype("string").fillna("")
    tt = tt.str.strip()
    tt = tt.str.replace(r"\.0$", "", regex=True)
    tt = tt.replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
    return tt


def passes_trigger(tt_value: str, triggers: List[str]) -> bool:
    for trig in triggers:
        if all(ch in tt_value for ch in trig):
            return True
    return False


def apply_trigger(df: pd.DataFrame, triggers: List[str]) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    tt_array = ["" for _ in range(n)]

    for plane_idx in range(1, 5):
        plane_active = pd.Series(False, index=out.index)
        for strip_idx in range(1, 5):
            qf = out.get(f"Q_front_{plane_idx}_s{strip_idx}")
            qb = out.get(f"Q_back_{plane_idx}_s{strip_idx}")
            if qf is None and qb is None:
                continue
            active = pd.Series(False, index=out.index)
            if qf is not None:
                active |= qf.to_numpy(dtype=float) > 0
            if qb is not None:
                active |= qb.to_numpy(dtype=float) > 0
            plane_active |= active
        tt_array = [tt + str(plane_idx) if active else tt for tt, active in zip(tt_array, plane_active)]

    tt_series = normalize_tt(pd.Series(tt_array))
    keep_mask = tt_series.apply(lambda val: passes_trigger(val, triggers))
    filtered = out[keep_mask].copy()
    filtered["tt_trigger"] = tt_series[keep_mask].values
    return filtered


def prune_step9(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"event_id", "T_thick_s", "tt_trigger"}
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            keep.add(f"T_front_{plane_idx}_s{strip_idx}")
            keep.add(f"T_back_{plane_idx}_s{strip_idx}")
            keep.add(f"Q_front_{plane_idx}_s{strip_idx}")
            keep.add(f"Q_back_{plane_idx}_s{strip_idx}")
    keep_cols = [col for col in df.columns if col in keep]
    return df[keep_cols]


def plot_trigger_summary(df: pd.DataFrame, output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = normalize_tt(df["tt_trigger"]).value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color="steelblue", alpha=0.8)
        for patch in bars:
            patch.set_rasterized(True)
        ax.set_title("tt_trigger counts")
        ax.set_xlabel("tt_trigger")
        ax.set_ylabel("Counts")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

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
    parser = argparse.ArgumentParser(description="Step 9: apply trigger combinations based on channel activity.")
    parser.add_argument("--config", default="config_step_9_physics.yaml", help="Path to step physics config YAML")
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
    triggers = [str(t) for t in cfg.get("trigger_combinations", [])]

    input_glob = cfg.get("input_glob", "**/step_8_chunks.chunks.json")
    input_sim_run = cfg.get("input_sim_run", "latest")

    print("\n-----\nStep 9 starting...\n-----")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Triggers: {triggers}")

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
        plot_trigger_summary(df, plot_path)
        print(f"Saved {plot_path}")
        return

    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)
    elif input_sim_run == "random":
        input_sim_run = random_sim_run(input_dir, cfg.get("seed"))

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

    if len(input_paths) != 1:
        raise FileNotFoundError(f"Expected 1 input, found {len(input_paths)}.")

    input_path = input_paths[0]
    normalized_stem = normalize_stem(input_path)
    print(f"Processing: {input_path}")
    input_iter, upstream_meta, chunked_input = iter_input_frames(input_path, chunk_rows)
    step_chain = extract_step_id_chain(upstream_meta)
    if not step_chain:
        raise ValueError("No step IDs found in upstream metadata.")
    mesh_dir = Path(cfg.get("param_mesh_dir", "../../INTERSTEPS/STEP_0_TO_1"))
    if not mesh_dir.is_absolute():
        mesh_dir = Path(__file__).resolve().parent / mesh_dir
    step_9_id = select_next_step_id(
        output_dir,
        mesh_dir,
        cfg.get("param_mesh_sim_run", "none"),
        "step_9_id",
        step_chain,
        cfg.get("seed"),
        physics_cfg.get("step_9_id"),
    )
    if step_9_id is None:
        print("Skipping STEP_9: all step_9_id combinations already exist.")
        return
    sim_run = build_sim_run_name(step_chain + [step_9_id])
    sim_run_dir = output_dir / sim_run
    if not args.force and sim_run_dir.exists():
        print(f"SIM_RUN {sim_run} already exists; skipping (use --force to regenerate).")
        return

    physics_cfg["step_9_id"] = step_9_id
    sim_run, sim_run_dir, config_hash, upstream_hash, _ = register_sim_run(
        output_dir, "STEP_9", config_path, physics_cfg, upstream_meta, sim_run
    )
    reset_dir(sim_run_dir)

    out_stem_base = "step_9"
    out_stem = f"{out_stem_base}_chunks" if chunk_rows else out_stem_base
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_9",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "source_dataset": str(input_path),
        "upstream": upstream_meta,
        "step_9_id": step_9_id,
    }
    if chunk_rows:
        def _iter_out() -> Iterable[pd.DataFrame]:
            for chunk in input_iter:
                yield prune_step9(apply_trigger(chunk, triggers))

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
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_stem_base}_plots.pdf"
            plot_trigger_summary(plot_df, plot_path)
        print(f"Saved {manifest_path}")
    else:
        df, upstream_meta = load_with_metadata(input_path)
        filtered = prune_step9(apply_trigger(df, triggers))
        out_path = sim_run_dir / f"{out_stem}.{output_format}"
        save_with_metadata(filtered, out_path, metadata, output_format)
        if not args.no_plots:
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_path.stem}_plots.pdf"
            plot_trigger_summary(filtered, plot_path)
        print(f"Saved {out_path} (kept {len(filtered):,}/{len(df):,})")


if __name__ == "__main__":
    main()
