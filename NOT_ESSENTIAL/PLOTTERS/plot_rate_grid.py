#!/usr/bin/env python3
"""Plot total event rate per 10 minutes for flux/efficiency combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from MASTER_STEPS.STEP_SHARED.sim_utils import iter_input_frames


def normalize_efficiency_vectors(value: object) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [[float(v) for v in value]]
    if isinstance(value, list) and all(isinstance(v, list) for v in value):
        vectors = []
        for vec in value:
            if len(vec) != 4 or not all(isinstance(v, (int, float)) for v in vec):
                raise ValueError("Each efficiencies vector must contain 4 numeric values.")
            vectors.append([float(v) for v in vec])
        return vectors
    raise ValueError("efficiencies must be a 4-value list or a list of 4-value lists.")


def normalize_flux_values(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        if not all(isinstance(v, (int, float)) for v in value):
            raise ValueError("flux_cm2_min must be a number or list of numbers.")
        return [float(v) for v in value]
    raise ValueError("flux_cm2_min must be a number or list of numbers.")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def load_input_meta(path: Path) -> dict:
    if path.name.endswith(".chunks.json"):
        return json.loads(path.read_text()).get("metadata", {})
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def parse_geometry_id(meta: dict) -> int | None:
    if "geometry_id" in meta:
        try:
            return int(meta["geometry_id"])
        except (TypeError, ValueError):
            return None
    meta_cfg = meta.get("config", {})
    geom_val = meta_cfg.get("geometry_id")
    if geom_val is None or str(geom_val).lower() == "auto":
        return None
    try:
        return int(geom_val)
    except (TypeError, ValueError):
        return None


def find_in_meta(meta: dict, key: str) -> object | None:
    if not isinstance(meta, dict):
        return None
    if isinstance(meta.get("config"), dict) and key in meta["config"]:
        return meta["config"][key]
    upstream = meta.get("upstream")
    if isinstance(upstream, dict):
        return find_in_meta(upstream, key)
    return None


def vector_key(vec: list[float]) -> tuple[float, float, float, float]:
    return tuple(round(float(v), 4) for v in vec)


def list_input_paths(base_dir: Path, input_glob: str) -> list[Path]:
    paths: list[Path] = []
    for sim_run_dir in sorted(base_dir.glob("SIM_RUN_*")):
        if "**" in input_glob:
            paths.extend(sim_run_dir.rglob(input_glob.replace("**/", "")))
        else:
            paths.extend(sim_run_dir.glob(input_glob))
    return sorted(paths)


def accumulate_counts(counts: dict[int, int], bins: np.ndarray) -> None:
    for idx in bins:
        counts[idx] = counts.get(idx, 0) + 1


def build_series(counts: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if not counts:
        return np.array([]), np.array([])
    min_idx = min(counts)
    max_idx = max(counts)
    x = np.arange(min_idx, max_idx + 1, dtype=int)
    y = np.array([counts.get(i, 0) for i in x], dtype=int)
    x = x - min_idx
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot event rate grid for flux/efficiency combinations.")
    parser.add_argument("--geometry-id", type=int, default=0, help="Geometry id to plot")
    parser.add_argument(
        "--input-dir",
        default="INTERSTEPS/STEP_10_TO_FINAL",
        help="Directory with STEP_10 outputs",
    )
    parser.add_argument(
        "--input-glob",
        default="**/geom_*_daq.chunks.json",
        help="Glob for STEP_10 output manifests",
    )
    parser.add_argument("--bin-seconds", type=int, default=1, help="Bin size in seconds")
    parser.add_argument(
        "--step1-config",
        default="MASTER_STEPS/STEP_1/config_step_1_physics.yaml",
        help="Path to STEP_1 physics config",
    )
    parser.add_argument(
        "--step3-config",
        default="MASTER_STEPS/STEP_3/config_step_3_physics.yaml",
        help="Path to STEP_3 physics config",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output plot path (default: PLOTTERS/rate_grid_geom_<id>.pdf)",
    )
    parser.add_argument("--debug", action="store_true", help="Print matching diagnostics")
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir
    input_glob = args.input_glob

    step1_path = Path(args.step1_config)
    if not step1_path.is_absolute():
        step1_path = ROOT_DIR / step1_path
    step3_path = Path(args.step3_config)
    if not step3_path.is_absolute():
        step3_path = ROOT_DIR / step3_path
    step1_cfg = load_yaml(step1_path)
    step3_cfg = load_yaml(step3_path)
    flux_values = normalize_flux_values(step1_cfg.get("flux_cm2_min"))
    eff_vectors = normalize_efficiency_vectors(step3_cfg.get("efficiencies"))

    if not flux_values or not eff_vectors:
        raise ValueError(
            f"Flux values or efficiency vectors are missing in configs: {step1_path}, {step3_path}."
        )

    flux_values = [float(v) for v in flux_values]
    eff_vectors = [list(map(float, vec)) for vec in eff_vectors]

    eff_map = {vector_key(vec): vec for vec in eff_vectors}
    flux_map = {round(val, 4): val for val in flux_values}

    counts_map: dict[tuple[float, tuple[float, float, float, float]], dict[int, int]] = {}
    paths = list_input_paths(base_dir, input_glob)
    if not paths:
        raise FileNotFoundError(f"No inputs found in {base_dir} with glob {input_glob}.")

    bin_seconds = int(args.bin_seconds)
    seen_flux = {}
    seen_eff = {}
    seen_pairs = set()
    for path in paths:
        meta = load_input_meta(path)
        if parse_geometry_id(meta) != args.geometry_id:
            continue
        flux_val = find_in_meta(meta, "flux_cm2_min")
        eff_val = find_in_meta(meta, "efficiencies")
        if flux_val is None or eff_val is None:
            continue
        flux_key = round(float(flux_val), 4)
        eff_key = vector_key([float(v) for v in eff_val])
        seen_flux[flux_key] = seen_flux.get(flux_key, 0) + 1
        seen_eff[eff_key] = seen_eff.get(eff_key, 0) + 1
        seen_pairs.add((flux_key, eff_key))
        if flux_key not in flux_map or eff_key not in eff_map:
            continue
        key = (flux_map[flux_key], eff_key)
        counts = counts_map.setdefault(key, {})
        df_iter, _, _ = iter_input_frames(path, None)
        for df in df_iter:
            if "T_thick_s" not in df.columns:
                continue
            t0 = df["T_thick_s"].to_numpy(dtype=float)
            t0 = t0[np.isfinite(t0)]
            if t0.size == 0:
                continue
            bins = (t0 // bin_seconds).astype(int)
            accumulate_counts(counts, bins)

    if args.debug:
        print(f"Found geometry {args.geometry_id} in {len(seen_pairs)} flux/eff pairs.")
        print(f"Seen flux values: {sorted(seen_flux)}")
        print(f"Seen efficiencies: {sorted(seen_eff)}")
        print(f"Expected flux values: {sorted(flux_map)}")
        print(f"Expected efficiencies: {sorted(eff_map)}")

    n_rows = len(flux_values)
    n_cols = len(eff_vectors)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, flux in enumerate(flux_values):
        for j, eff in enumerate(eff_vectors):
            ax = axes[i, j]
            key = (flux, vector_key(eff))
            series = counts_map.get(key, {})
            x, y = build_series(series)
            if x.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(x * args.bin_seconds, y, linewidth=1.0, color="slateblue")
            ax.set_title(f"flux={flux:.2f}, eff={eff}")
            if i == n_rows - 1:
                ax.set_xlabel("Seconds since start")
            if j == 0:
                ax.set_ylabel("Events per bin")
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Geometry {args.geometry_id} event rate per {args.bin_seconds} seconds")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = args.output
    if output_path is None:
        output_path = f"PLOTTERS/rate_grid_geom_{args.geometry_id}.pdf"
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path
    out_path.parent.mkdir -p(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
