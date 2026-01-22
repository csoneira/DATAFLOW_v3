#!/usr/bin/env python3
"""Step 2: propagate muons through station geometry and compute plane crossings.

Inputs: muon_sample from Step 1.
Outputs: step_2.(pkl|csv) with crossing coordinates/times and metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

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
    DEFAULT_BOUNDS,
    DetectorBounds,
    ensure_dir,
    extract_param_set,
    find_latest_data_path,
    find_sim_run,
    find_sim_run_dir,
    load_step_configs,
    latest_sim_run,
    random_sim_run,
    load_with_metadata,
    now_iso,
    resolve_param_mesh,
    resolve_sim_run,
    reset_dir,
    save_with_metadata,
    select_param_row,
)


def calculate_intersections(
    df: pd.DataFrame,
    z_positions: Iterable[float],
    bounds: DetectorBounds,
    c_mm_per_ns: float,
) -> pd.DataFrame:
    z_positions = list(z_positions)
    out = df.copy()
    crossing_array = np.full(len(out), "", dtype=object)

    theta = out["Theta_gen"].to_numpy(dtype=float)
    tan_theta = np.tan(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(out["Phi_gen"].to_numpy(dtype=float))
    sin_phi = np.sin(out["Phi_gen"].to_numpy(dtype=float))

    for plane_idx, z in enumerate(z_positions, start=1):
        dz = z + out["Z_gen"].to_numpy(dtype=float)
        x_proj = out["X_gen"].to_numpy(dtype=float) + dz * tan_theta * cos_phi
        y_proj = out["Y_gen"].to_numpy(dtype=float) + dz * tan_theta * sin_phi
        t_sum = dz / (c_mm_per_ns * cos_theta)

        in_bounds = (
            (x_proj >= bounds.x_min)
            & (x_proj <= bounds.x_max)
            & (y_proj >= bounds.y_min)
            & (y_proj <= bounds.y_max)
        )

        out[f"X_gen_{plane_idx}"] = np.where(in_bounds, x_proj, np.nan)
        out[f"Y_gen_{plane_idx}"] = np.where(in_bounds, y_proj, np.nan)
        out[f"Z_gen_{plane_idx}"] = np.where(in_bounds, z, np.nan)
        out[f"T_sum_{plane_idx}_ns"] = np.where(in_bounds, t_sum, np.nan)

        crossing_array[in_bounds] = crossing_array[in_bounds] + str(plane_idx)

    t_sum_cols = [f"T_sum_{idx}_ns" for idx in range(1, len(z_positions) + 1)]
    t_sum_matrix = out[t_sum_cols].to_numpy(dtype=float)
    valid_mask = ~np.isnan(t_sum_matrix)
    min_tsum = np.where(valid_mask, t_sum_matrix, np.inf).min(axis=1)
    min_tsum[~np.isfinite(min_tsum)] = 0.0
    out[t_sum_cols] = t_sum_matrix - min_tsum[:, None]

    crossing_series = pd.Series(crossing_array, dtype="string")
    crossing_series = crossing_series.replace("", pd.NA)
    out["tt_crossing"] = crossing_series
    return out


def prune_step2(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"event_id", "T_thick_s", "tt_crossing"}
    for plane_idx in range(1, 5):
        keep.add(f"X_gen_{plane_idx}")
        keep.add(f"Y_gen_{plane_idx}")
        keep.add(f"Z_gen_{plane_idx}")
        keep.add(f"T_sum_{plane_idx}_ns")
    keep_cols = [col for col in df.columns if col in keep]
    return df[keep_cols]


def normalize_positions(z_positions: Tuple[float, float, float, float], normalize: bool) -> np.ndarray:
    z_array = np.array(z_positions, dtype=float)
    if normalize:
        z_array = z_array - z_array[0]
    return z_array


def plot_geometry_summary(df: pd.DataFrame, pdf: PdfPages, title: str) -> None:
    required = {"Theta_gen", "Phi_gen", "tt_crossing"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        print(f"Skipping geometry summary plot (missing columns: {missing})")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plane_cols = [(f"X_gen_{i}", f"Y_gen_{i}") for i in range(1, 5)]
    for ax, (x_col, y_col) in zip(axes.flatten(), plane_cols):
        if x_col not in df or y_col not in df:
            ax.axis("off")
            continue
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        ax.scatter(x[mask], y[mask], s=1, alpha=0.2, rasterized=True)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    crossing_values = pd.Series(df["tt_crossing"]).dropna().astype(str)
    for ct in sorted(crossing_values.unique()):
        ct_df = df[df["tt_crossing"] == ct]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(ct_df["Theta_gen"], ct_df["Phi_gen"], s=6, alpha=0.25, rasterized=True)
        axes[0].set_title(f"Theta vs Phi (tt_crossing={ct})")
        axes[0].set_xlabel("Theta (rad)")
        axes[0].set_ylabel("Phi (rad)")
        axes[0].set_xlim(0, np.pi / 2)
        axes[0].set_ylim(-np.pi, np.pi)

        axes[1].hist2d(ct_df["Theta_gen"], ct_df["Phi_gen"], bins=60, cmap="magma")
        axes[1].set_title(f"Theta vs Phi density (tt_crossing={ct})")
        axes[1].set_xlabel("Theta (rad)")
        axes[1].set_ylabel("Phi (rad)")
        axes[1].set_xlim(0, np.pi / 2)
        axes[1].set_ylim(-np.pi, np.pi)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def plot_step2_summary(df: pd.DataFrame, pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    x_cols = [c for c in df.columns if c.startswith("X_gen_")]
    x_vals = df[x_cols].to_numpy(dtype=float).ravel() if x_cols else np.array([])
    x_vals = x_vals[~np.isnan(x_vals)]
    axes[0].hist(x_vals, bins=80, color="steelblue", alpha=0.8)
    axes[0].set_title("X_gen_i")

    y_cols = [c for c in df.columns if c.startswith("Y_gen_")]
    y_vals = df[y_cols].to_numpy(dtype=float).ravel() if y_cols else np.array([])
    y_vals = y_vals[~np.isnan(y_vals)]
    axes[1].hist(y_vals, bins=80, color="seagreen", alpha=0.8)
    axes[1].set_title("Y_gen_i")

    t_cols = [c for c in df.columns if c.startswith("T_sum_") and c.endswith("_ns")]
    t_vals = df[t_cols].to_numpy(dtype=float).ravel() if t_cols else np.array([])
    t_vals = t_vals[~np.isnan(t_vals)]
    t_vals = t_vals[t_vals != 0]
    axes[2].hist(t_vals, bins=80, color="darkorange", alpha=0.8)
    axes[2].set_title("T_sum_i_ns")

    counts = df["tt_crossing"].value_counts().sort_index()
    bars = axes[3].bar(counts.index.astype(str), counts.values, color="slateblue", alpha=0.8)
    for patch in bars:
        patch.set_rasterized(True)
    axes[3].set_title("tt_crossing")

    for ax in axes:
        for patch in ax.patches:
            patch.set_rasterized(True)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: expand muon sample for each station geometry.")
    parser.add_argument("--config", default="config_step_2_physics.yaml", help="Path to step physics config YAML")
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

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    output_format = str(cfg.get("output_format", "pkl")).lower()

    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        latest_path = find_latest_data_path(output_dir)
        if latest_path is None:
            raise FileNotFoundError(f"No existing outputs found in {output_dir} for plot-only.")
        print(f"Plot-only: {latest_path}")
        df, _ = load_with_metadata(latest_path)
        if "tt_crossing" in df.columns:
            df = df[df["tt_crossing"].notna()].reset_index(drop=True)
        sim_run_dir = find_sim_run_dir(latest_path)
        plot_dir = (sim_run_dir or latest_path.parent) / "PLOTS"
        ensure_dir(plot_dir)
        plot_path = plot_dir / f"{latest_path.stem}_plots.pdf"
        with PdfPages(plot_path) as pdf:
            plot_geometry_summary(df, pdf, latest_path.stem)
            plot_step2_summary(df, pdf)
        print(f"Saved {plot_path}")
        return

    input_path_cfg = cfg.get("input_muon_sample")
    if input_path_cfg:
        input_path = Path(input_path_cfg).expanduser()
        if not input_path.is_absolute():
            input_path = Path(__file__).resolve().parent / input_path
    else:
        input_dir = Path(cfg["input_dir"])
        if not input_dir.is_absolute():
            input_dir = Path(__file__).resolve().parent / input_dir
        input_sim_run = cfg.get("input_sim_run", "latest")
    if input_sim_run == "latest":
        input_sim_run = latest_sim_run(input_dir)
    elif input_sim_run == "random":
        input_sim_run = random_sim_run(input_dir, cfg.get("seed"))
        input_run_dir = input_dir / str(input_sim_run)
        input_basename = cfg.get("input_basename")
        if input_basename:
            input_path = input_run_dir / input_basename
            if not input_path.exists():
                manifest_path = input_path.with_suffix(".chunks.json")
                if manifest_path.exists():
                    print(f"Using chunk manifest: {manifest_path}")
                else:
                    print(f"Warning: input_basename not found: {input_path}")
                    input_path = None
        else:
            input_path = None

        if input_path is None:
            candidates = sorted(input_run_dir.glob("muon_sample_*.pkl")) + sorted(
                input_run_dir.glob("muon_sample_*.csv")
            )
            if len(candidates) != 1:
                manifest_candidates = sorted(input_run_dir.glob("muon_sample_*.chunks.json"))
                if not manifest_candidates:
                    manifest_candidates = sorted(input_run_dir.glob("muon_sample_*/chunks"))
                if len(manifest_candidates) != 1:
                    raise FileNotFoundError(
                        f"Expected 1 muon_sample file in {input_run_dir}, found {len(candidates)}."
                    )
                manifest_path = manifest_candidates[0]
                if manifest_path.name == "chunks":
                    input_path = manifest_path.parent
                else:
                    input_path = Path(str(manifest_path)[: -len(".chunks.json")] + ".pkl")
            else:
                input_path = candidates[0]
    bounds_cfg = cfg.get("bounds_mm", {})
    bounds = DetectorBounds(
        x_min=float(bounds_cfg.get("x_min", DEFAULT_BOUNDS.x_min)),
        x_max=float(bounds_cfg.get("x_max", DEFAULT_BOUNDS.x_max)),
        y_min=float(bounds_cfg.get("y_min", DEFAULT_BOUNDS.y_min)),
        y_max=float(bounds_cfg.get("y_max", DEFAULT_BOUNDS.y_max)),
    )

    normalize = bool(runtime_cfg.get("normalize_to_first_plane", True))
    chunk_rows = cfg.get("chunk_rows")
    rng = np.random.default_rng(cfg.get("seed"))
    plot_sample_rows = cfg.get("plot_sample_rows")
    stream_csv = bool(chunk_rows) and input_path.suffix == ".csv" and output_format == "csv"
    chunk_manifest = input_path.with_suffix(".chunks.json")
    stream_chunks = chunk_manifest.exists()

    print("\n-----\nStep 2 starting...\n-----")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    if args.plot_only:
        if args.no_plots:
            print("Plot-only requested with --no-plots; skipping plots.")
            return
        latest_path = find_latest_data_path(output_dir)
        if latest_path is None:
            raise FileNotFoundError(f"No existing outputs found in {output_dir} for plot-only.")
        print(f"Plot-only: {latest_path}")
        df, _ = load_with_metadata(latest_path)
        if "tt_crossing" in df.columns:
            df = df[df["tt_crossing"].notna()].reset_index(drop=True)
        sim_run_dir = find_sim_run_dir(latest_path)
        plot_dir = (sim_run_dir or latest_path.parent) / "PLOTS"
        ensure_dir(plot_dir)
        plot_path = plot_dir / f"{latest_path.stem}_plots.pdf"
        with PdfPages(plot_path) as pdf:
            plot_geometry_summary(df, pdf, latest_path.stem)
            plot_step2_summary(df, pdf)
        print(f"Saved {plot_path}")
        return

    if stream_csv:
        meta_path = input_path.with_suffix(input_path.suffix + ".meta.json")
        upstream_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    elif stream_chunks:
        manifest = json.loads(chunk_manifest.read_text())
        upstream_meta = manifest.get("metadata", {})
    else:
        muon_df, upstream_meta = load_with_metadata(input_path)
    if not args.force:
        existing = find_sim_run(output_dir, physics_cfg, upstream_meta)
        if existing:
            print(f"SIM_RUN {existing} already exists; skipping (use --force to regenerate).")
            return
    c_mm_per_ns = float(cfg.get("c_mm_per_ns", upstream_meta.get("config", {}).get("c_mm_per_ns", 299.792458)))
    print(f"c_mm_per_ns: {c_mm_per_ns}")
    z_positions_cfg = cfg.get("z_positions")
    param_set_id, param_date = extract_param_set(upstream_meta)
    param_mesh_path = None
    if z_positions_cfg is None:
        raise ValueError("config z_positions is required for Step 2.")
    if isinstance(z_positions_cfg, str) and z_positions_cfg.lower() == "random":
        mesh_dir = Path(cfg.get("param_mesh_dir", "../../INTERSTEPS/STEP_0_TO_1"))
        if not mesh_dir.is_absolute():
            mesh_dir = Path(__file__).resolve().parent / mesh_dir
        mesh, mesh_path = resolve_param_mesh(mesh_dir, cfg.get("param_mesh_sim_run", "none"), cfg.get("seed"))
        param_row = select_param_row(mesh, rng, param_set_id)
        if "param_set_id" in param_row.index and pd.notna(param_row["param_set_id"]):
            param_set_id = int(param_row["param_set_id"])
        if "param_date" in param_row:
            param_date = str(param_row["param_date"])
        param_mesh_path = mesh_path
        required_cols = ["z_p1", "z_p2", "z_p3", "z_p4"]
        if not all(col in param_row.index for col in required_cols):
            raise ValueError("param_mesh.csv is missing z_p1..z_p4 required for geometry selection.")
        z_values = [float(param_row[col]) for col in required_cols]
        physics_cfg["z_positions"] = z_values
        cfg["z_positions"] = z_values
        print(f"Selected z_positions from mesh (param_set_id={param_set_id})")
    else:
        if (
            isinstance(z_positions_cfg, (list, tuple))
            and len(z_positions_cfg) == 4
        ):
            z_values = [float(v) for v in z_positions_cfg]
            physics_cfg["z_positions"] = z_values
            cfg["z_positions"] = z_values
            param_set_id = None
            param_date = None
        else:
            raise ValueError("z_positions must be 'random' or a 4-value list [z1, z2, z3, z4].")
    z_positions = normalize_positions(tuple(z_values), normalize)
    print(f"z_positions={z_positions.tolist()}")

    sim_run, sim_run_dir, config_hash, upstream_hash, _ = resolve_sim_run(
        output_dir, "STEP_2", config_path, cfg, upstream_meta
    )
    reset_dir(sim_run_dir)
    out_stem_base = "step_2"
    out_name = f"{out_stem_base}.{output_format}"
    out_path = sim_run_dir / out_name
    metadata = {
        "created_at": now_iso(),
        "step": "STEP_2",
        "config": physics_cfg,
        "runtime_config": runtime_cfg,
        "sim_run": sim_run,
        "config_hash": config_hash,
        "upstream_hash": upstream_hash,
        "z_positions_mm": [float(z) for z in z_positions],
        "z_positions_raw_mm": [float(v) for v in z_values],
        "param_set_id": param_set_id,
        "param_date": param_date,
        "param_mesh_path": str(param_mesh_path) if param_mesh_path else None,
        "upstream": upstream_meta,
    }

    if stream_chunks or stream_csv:
        if output_format not in ("csv", "pkl"):
            raise ValueError("Chunked input requires output_format=csv or pkl.")

        out_stem = f"{out_stem_base}_chunks"
        chunks_dir = sim_run_dir / out_stem
        ensure_dir(chunks_dir)
        chunk_paths = []
        buffer = []
        buffered_rows = 0
        total_rows = 0
        full_chunks = 0
        last_full_chunk = None

        def flush_chunk(chunk_df: pd.DataFrame) -> None:
            nonlocal full_chunks, last_full_chunk
            chunk_path = chunks_dir / f"part_{full_chunks:04d}.{output_format}"
            if output_format == "csv":
                chunk_df.to_csv(chunk_path, index=False)
            else:
                chunk_df.to_pickle(chunk_path)
            chunk_paths.append(str(chunk_path))
            full_chunks += 1
            last_full_chunk = chunk_df

        def maybe_flush_buffer() -> None:
            nonlocal buffer, buffered_rows
            while buffered_rows >= int(chunk_rows):
                chunk_df = pd.concat(buffer, ignore_index=True)
                out_df = chunk_df.iloc[: int(chunk_rows)].copy()
                remainder = chunk_df.iloc[int(chunk_rows):].copy()
                flush_chunk(out_df)
                buffer = [remainder] if not remainder.empty else []
                buffered_rows = len(remainder)

        if stream_csv:
            chunk_iter = (pd.read_csv(input_path, chunksize=int(chunk_rows)), None)
        else:
            chunk_iter = (manifest.get("chunks", []), "pkl")

        processed_chunks = 0
        for item in chunk_iter[0]:
            processed_chunks += 1
            if chunk_iter[1] == "pkl":
                print(f"Processing chunk {processed_chunks}/{len(manifest.get('chunks', []))}")
            else:
                print(f"Processing chunk {processed_chunks}")
            if chunk_iter[1] == "pkl":
                chunk_df = pd.read_pickle(item)
            else:
                chunk_df = item
            geom_chunk = calculate_intersections(chunk_df, z_positions, bounds, c_mm_per_ns)
            if "tt_crossing" in geom_chunk.columns:
                geom_chunk = geom_chunk[geom_chunk["tt_crossing"].notna()].reset_index(drop=True)
            geom_chunk = prune_step2(geom_chunk)
            total_rows += len(geom_chunk)
            if not geom_chunk.empty:
                buffer.append(geom_chunk)
                buffered_rows += len(geom_chunk)
                maybe_flush_buffer()
            if processed_chunks % 10 == 0:
                print(f"Chunks processed: {processed_chunks}, rows kept so far: {total_rows:,}")

        if full_chunks == 0 and buffered_rows > 0:
            flush_chunk(pd.concat(buffer, ignore_index=True))
            buffered_rows = 0
            buffer = []
        else:
            buffered_rows = 0
            buffer = []

        metadata["row_count"] = full_chunks * int(chunk_rows) + buffered_rows
        manifest_out = {
            "version": 1,
            "chunks": chunk_paths,
            "row_count": metadata["row_count"],
            "metadata": metadata,
        }
        manifest_path = sim_run_dir / f"{out_stem}.chunks.json"
        manifest_path.write_text(json.dumps(manifest_out, indent=2))

        if not args.no_plots:
            if plot_sample_rows and last_full_chunk is not None:
                sample_n = len(last_full_chunk) if plot_sample_rows is True else int(plot_sample_rows)
                sample_n = min(sample_n, len(last_full_chunk))
                sample_df = last_full_chunk.sample(n=sample_n, random_state=0)
                plot_dir = sim_run_dir / "PLOTS"
                ensure_dir(plot_dir)
                plot_path = plot_dir / f"{out_stem_base}_plots.pdf"
                with PdfPages(plot_path) as pdf:
                    plot_geometry_summary(sample_df, pdf, f"Step 2 (sample)")
                    plot_step2_summary(sample_df, pdf)
            else:
                print("Chunked mode enabled; skipping plots to limit memory usage.")
    else:
        geom_df = calculate_intersections(muon_df, z_positions, bounds, c_mm_per_ns)
        if "tt_crossing" in geom_df.columns:
            geom_df = geom_df[geom_df["tt_crossing"].notna()].reset_index(drop=True)
        geom_df = prune_step2(geom_df)

        save_with_metadata(geom_df, out_path, metadata, output_format)
        if not args.no_plots:
            plot_dir = sim_run_dir / "PLOTS"
            ensure_dir(plot_dir)
            plot_path = plot_dir / f"{out_stem_base}_plots.pdf"
            with PdfPages(plot_path) as pdf:
                plot_geometry_summary(geom_df, pdf, "Step 2")
                plot_step2_summary(geom_df, pdf)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
