#!/usr/bin/env python3
"""STEP 1.1 — Collect simulated data results and match with simulation parameters.

For each task in config["task_ids"], loads the metadata CSV
(task_*_metadata_specific.csv) from the station and the simulation
parameters CSV (step_final_simulation_params.csv).  Rows are joined on
``filename_base``, keeping only files present in BOTH tables.  The
``file_name`` column is dropped (no longer needed after joining).

A z-position configuration cut is applied: only rows whose
(z_plane_1..4) match the selected configuration are kept.  If no
configuration is specified in the config, a random one is chosen from
the available options.

Output
------
OUTPUTS/FILES/collected_data.csv
    Full merged table with simulation parameters + metadata columns.
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
PIPELINE_DIR = STEP_DIR.parents[1]  # INFERENCE_DICTIONARY_VALIDATION
REPO_ROOT = PIPELINE_DIR.parent      # DATAFLOW_v3
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


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

    station_id = int(config.get("station_id", 0))
    task_ids = config.get("task_ids", [1])
    sim_params_path = Path(
        config.get(
            "simulation_params_csv",
            str(REPO_ROOT / "MINGO_DIGITAL_TWIN" / "SIMULATED_DATA" / "step_final_simulation_params.csv"),
        )
    )
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

    if z_config is not None:
        z_config = [float(v) for v in z_config]
        log.info("  Selecting z config from config: %s", z_config)
    else:
        idx = np.random.randint(0, len(unique_z))
        z_config = unique_z.iloc[idx].tolist()
        log.info("  No z config specified — randomly selected [%d]: %s", idx, z_config)

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
            ax.hist(ev, bins=40, alpha=0.8, color="#4C78A8", edgecolor="white")
            ax.axvline(ev.median(), color="#E45756", linestyle="--", linewidth=1.2,
                       label=f"median = {ev.median():.0f}")
            ax.set_xlabel(f"Event count ({ev_col})")
            ax.set_ylabel("Number of files")
            ax.set_title(f"Event count distribution — {len(ev)} collected files")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "event_count_histogram.png", dpi=150)
            plt.close(fig)
            log.info("Wrote plot: %s", PLOTS_DIR / "event_count_histogram.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
