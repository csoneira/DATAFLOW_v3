#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_1_COLLECT_DATA/collect_data.py
Purpose: STEP 1.1 — Collect simulated data results and match with simulation parameters.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_1_COLLECT_DATA/collect_data.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
# Support both layouts:
#   - <pipeline>/STEP_1_SETUP/STEP_1_1_COLLECT_DATA
#   - <pipeline>/STEPS/STEP_1_SETUP/STEP_1_1_COLLECT_DATA
if STEP_DIR.parents[1].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[2]
else:
    PIPELINE_DIR = STEP_DIR.parents[1]
REPO_ROOT = PIPELINE_DIR.parent
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "1_1"


def _save_figure(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save figure with a per-script sequential numeric prefix."""
    global _FIGURE_COUNTER
    _FIGURE_COUNTER += 1
    out_path = Path(path)
    out_path = out_path.with_name(f"{FIGURE_STEP_PREFIX}_{_FIGURE_COUNTER}_{out_path.name}")
    fig.savefig(out_path, **kwargs)


_PLOT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".eps",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}


def _clear_plots_dir() -> None:
    """Remove previously generated plot files from the plots directory."""
    removed = 0
    for candidate in PLOTS_DIR.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in _PLOT_EXTENSIONS:
            try:
                candidate.unlink()
                removed += 1
            except OSError as exc:
                log.warning("Could not remove old plot file %s: %s", candidate, exc)
    log.info("Cleared %d plot file(s) from %s", removed, PLOTS_DIR)

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="[%(levelname)s] STEP_1.1 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_1.1")


# ── Helpers ──────────────────────────────────────────────────────────────

def _task_metadata_path(station_id: int, task_id: int) -> Path:
    """Return the path to the task-specific metadata CSV."""
    station = f"MINGO{station_id:02d}"
    return (
        REPO_ROOT / "STATIONS" / station / "STAGE_1" / "EVENT_DATA"
        / "STEP_1" / f"TASK_{task_id}" / "METADATA"
        / f"task_{task_id}_metadata_specific.csv"
    )


def _load_config(path: Path) -> dict:
    def _merge_dicts(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    runtime_path = path.with_name("config_runtime.json")
    if runtime_path.exists():
        runtime_cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, runtime_cfg)
        log.info("Loaded runtime overrides: %s", runtime_path)
    return cfg


def _resolve_z_config_rng(config: dict) -> tuple[np.random.Generator, int, bool]:
    """Return RNG for z-config selection and report whether seed came from config."""
    cfg_11 = config.get("step_1_1", {})
    seed_candidates = [
        cfg_11.get("z_config_random_seed"),
        cfg_11.get("random_seed"),
        config.get("z_config_random_seed"),
    ]
    for raw_seed in seed_candidates:
        if raw_seed in (None, "", "null", "None"):
            continue
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            log.warning("Invalid z-config random seed value %r; ignoring.", raw_seed)
            continue
        return np.random.default_rng(seed), seed, True

    # When not configured, generate and record a one-off seed for reproducibility.
    seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    return np.random.default_rng(seed), seed, False


def _aggregate_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest execution per filename_base."""
    if "execution_timestamp" in df.columns:
        dt = pd.to_datetime(
            df["execution_timestamp"], format="%Y-%m-%d_%H.%M.%S", errors="coerce"
        )
        df = df.assign(_exec_dt=dt)
        df = df.sort_values(
            ["filename_base", "_exec_dt"], na_position="last", kind="mergesort"
        )
        df = df.groupby("filename_base").tail(1)
        df = df.drop(columns=["_exec_dt"])
    else:
        df = df.groupby("filename_base").tail(1)
    return df


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1.1: Collect simulated data and match with simulation parameters."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()

    station_id = int(config.get("station_id", 0))
    task_ids = config.get("task_ids", [1])
    default_sim_params_path = (
        REPO_ROOT / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"
    )
    sim_params_cfg = config.get("simulation_params_csv", None)
    if sim_params_cfg in (None, "", "null", "None"):
        sim_params_path = default_sim_params_path
    else:
        sim_params_path = Path(str(sim_params_cfg)).expanduser()
    z_config = config.get("z_position_config", None)
    metadata_agg = config.get("metadata_agg", "latest")

    # ── Load simulation parameters ───────────────────────────────────
    if not sim_params_path.exists():
        log.error("Simulation params CSV not found: %s", sim_params_path)
        return 1

    log.info("Loading simulation parameters: %s", sim_params_path)
    params_df = pd.read_csv(sim_params_path, low_memory=False)
    params_df["filename_base"] = (
        params_df["file_name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    )
    # Drop the file_name column (user request)
    params_df = params_df.drop(columns=["file_name"], errors="ignore")
    log.info("  Simulation params rows: %d", len(params_df))

    # ── Determine z-position configuration ───────────────────────────
    z_cols = ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
    unique_z = params_df[z_cols].drop_duplicates().reset_index(drop=True)
    log.info("  Available z configurations (%d):", len(unique_z))
    for i, row in unique_z.iterrows():
        log.info("    [%d] %s", i, list(row.values))

    z_config_selection_seed: int | None = None
    z_config_seed_from_config = False
    z_config_selected_index: int | None = None

    if z_config is not None:
        z_config = [float(v) for v in z_config]
        log.info("  Selecting z config from config: %s", z_config)
    else:
        rng, z_config_selection_seed, z_config_seed_from_config = _resolve_z_config_rng(config)
        z_config_selected_index = int(rng.integers(0, len(unique_z)))
        z_config = unique_z.iloc[z_config_selected_index].tolist()
        if z_config_seed_from_config:
            log.info(
                "  No z config specified — selected [%d] using configured seed %d: %s",
                z_config_selected_index,
                z_config_selection_seed,
                z_config,
            )
        else:
            log.info(
                "  No z config specified — selected [%d] using generated seed %d: %s",
                z_config_selected_index,
                z_config_selection_seed,
                z_config,
            )

    # Apply z-position cut
    z_mask = np.ones(len(params_df), dtype=bool)
    for col, val in zip(z_cols, z_config):
        z_mask &= np.isclose(params_df[col].astype(float), val, atol=1e-6)
    params_df = params_df.loc[z_mask].reset_index(drop=True)
    log.info("  Rows after z-position cut: %d", len(params_df))

    if params_df.empty:
        log.error("No rows remain after z-position cut. Check your z_position_config.")
        return 1

    # ── Collect metadata for each task and merge ─────────────────────
    all_merged: list[pd.DataFrame] = []

    for task_id in task_ids:
        meta_path = _task_metadata_path(station_id, task_id)
        if not meta_path.exists():
            log.warning("Metadata CSV not found for task %d: %s — skipping.", task_id, meta_path)
            continue

        log.info("Loading metadata for task %d: %s", task_id, meta_path)
        meta_df = pd.read_csv(meta_path, low_memory=False)

        if "filename_base" not in meta_df.columns:
            log.error("  No 'filename_base' column in task %d metadata.", task_id)
            continue

        # Aggregate
        if metadata_agg == "latest":
            meta_df = _aggregate_latest(meta_df)
        log.info("  Metadata rows (after aggregation): %d", len(meta_df))

        # Inner join on filename_base: only keep rows present in BOTH
        merged = params_df.merge(meta_df, on="filename_base", how="inner")
        log.info("  Merged rows (task %d): %d", task_id, len(merged))

        if not merged.empty:
            merged["task_id"] = task_id
            all_merged.append(merged)

    if not all_merged:
        log.error("No data collected from any task. Check paths and data.")
        return 1

    collected = pd.concat(all_merged, ignore_index=True)
    log.info("Total collected rows (all tasks): %d", len(collected))

    # ── Save ─────────────────────────────────────────────────────────
    out_path = FILES_DIR / "collected_data.csv"
    collected.to_csv(out_path, index=False)
    log.info("Wrote collected data: %s", out_path)

    # Save the selected z configuration for downstream steps
    z_info = {
        "z_position_config": z_config,
        "z_config_selected_index": z_config_selected_index,
        "z_config_selection_seed": z_config_selection_seed,
        "z_config_seed_from_config": z_config_seed_from_config,
        "total_rows": len(collected),
        "task_ids_used": task_ids,
        "station_id": station_id,
    }
    z_info_path = FILES_DIR / "z_config_selected.json"
    with open(z_info_path, "w", encoding="utf-8") as f:
        json.dump(z_info, f, indent=2)
    log.info("Wrote z config info: %s", z_info_path)

    # ── Plot: event count histogram ──────────────────────────────────
    ev_col = "selected_rows" if "selected_rows" in collected.columns else "requested_rows"
    if ev_col in collected.columns:
        ev = pd.to_numeric(collected[ev_col], errors="coerce").dropna()
        if not ev.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            # Match the blue histogram alpha used elsewhere (was 0.8 here)
            ax.hist(ev, bins=40, alpha=0.5, color="#4C78A8", edgecolor="white")
            # Removed median vertical line / legend for a cleaner presentation
            ax.set_xlabel(f"Event count ({ev_col})")
            ax.set_ylabel("Number of files")
            ax.set_title(f"Event count distribution — {len(ev)} collected files")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            _save_figure(fig, PLOTS_DIR / "event_count_histogram.png", dpi=150)
            plt.close(fig)
            log.info("Wrote plot: %s", PLOTS_DIR / "event_count_histogram.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
