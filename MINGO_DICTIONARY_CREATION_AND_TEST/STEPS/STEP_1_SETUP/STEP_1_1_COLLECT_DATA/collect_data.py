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
import re
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
DEFAULT_CONFIG = PIPELINE_DIR / "config_method.json"

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

CANONICAL_TT_LABELS = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "4",
        "12",
        "13",
        "14",
        "23",
        "24",
        "34",
        "123",
        "124",
        "134",
        "234",
        "1234",
    }
)
TT_RATE_COLUMN_RE = re.compile(r"^(?P<prefix>.+_tt)_(?P<label>[^_]+)_rate_hz$")
RATE_HIST_BIN_RE = re.compile(r"^events_per_second_(?P<bin>\d+)_rate_hz")


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

def _task_metadata_dir(station_id: int, task_id: int) -> Path:
    """Return the task metadata directory."""
    station = f"MINGO{station_id:02d}"
    return (
        REPO_ROOT / "STATIONS" / station / "STAGE_1" / "EVENT_DATA"
        / "STEP_1" / f"TASK_{task_id}" / "METADATA"
    )


def _task_specific_metadata_path(station_id: int, task_id: int) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_specific.csv"


def _task_trigger_type_metadata_path(station_id: int, task_id: int) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_trigger_type.csv"


def _task_rate_histogram_metadata_path(station_id: int, task_id: int) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_rate_histogram.csv"


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
    else:
        log.warning("Config file not found: %s", path)

    plots_path = path.with_name("config_plots.json")
    if plots_path != path and plots_path.exists():
        plots_cfg = json.loads(plots_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, plots_cfg)
        log.info("Loaded plot config: %s", plots_path)

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


def _normalize_tt_label(label: object) -> str:
    text = str(label).strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return text


def _safe_task_ids(raw: object) -> list[int]:
    if raw is None:
        return [1]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return [1]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [x.strip() for x in stripped.split(",") if x.strip()]
        raw = parsed
    out: list[int] = []
    if isinstance(raw, (list, tuple)):
        for value in raw:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(set(out)) or [1]


def _preferred_tt_prefixes_for_task_ids(task_ids: list[int]) -> tuple[str, ...]:
    """Preferred TT-rate prefixes based on the most advanced selected task."""
    max_task_id = max(task_ids) if task_ids else 1
    if max_task_id <= 1:
        return ("raw_tt", "clean_tt")
    if max_task_id == 2:
        return ("clean_tt", "raw_to_clean_tt", "raw_tt")
    if max_task_id == 3:
        return ("cal_tt", "clean_tt", "raw_to_clean_tt", "raw_tt")
    if max_task_id == 4:
        return ("list_tt", "list_to_fit_tt", "cal_tt", "clean_tt", "raw_to_clean_tt", "raw_tt")
    return (
        "post_tt",
        "fit_to_post_tt",
        "fit_tt",
        "list_to_fit_tt",
        "list_tt",
        "cal_tt",
        "clean_tt",
        "raw_to_clean_tt",
        "raw_tt",
        "corr_tt",
        "task5_to_corr_tt",
        "fit_to_corr_tt",
        "definitive_tt",
    )


def _resolve_metadata_prefix_override(raw: object) -> str | None:
    """Normalize optional metadata prefix override from config."""
    if raw in (None, "", "null", "None"):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() == "auto":
        return None
    return text.rstrip("_")


def _attach_tt_sum_global_rate(
    meta_df: pd.DataFrame,
    preferred_prefixes: tuple[str, ...],
) -> tuple[pd.DataFrame, str | None]:
    by_prefix: dict[str, list[str]] = {}
    for col in meta_df.columns:
        match = TT_RATE_COLUMN_RE.match(str(col))
        if match is None:
            continue
        prefix = str(match.group("prefix")).strip()
        label = _normalize_tt_label(match.group("label"))
        if label not in CANONICAL_TT_LABELS:
            continue
        by_prefix.setdefault(prefix, []).append(col)

    if not by_prefix:
        return meta_df, None

    selected_prefix: str | None = None
    for preferred in preferred_prefixes:
        if preferred in by_prefix and by_prefix[preferred]:
            selected_prefix = preferred
            break
    if selected_prefix is None:
        selected_prefix = min(by_prefix.keys(), key=lambda p: (-len(by_prefix[p]), p))

    cols = sorted(set(by_prefix[selected_prefix]))
    if not cols:
        return meta_df, None

    rate_sum = pd.Series(0.0, index=meta_df.index, dtype=float)
    valid_any = pd.Series(False, index=meta_df.index)
    for col in cols:
        numeric = pd.to_numeric(meta_df[col], errors="coerce")
        rate_sum = rate_sum + numeric.fillna(0.0)
        valid_any = valid_any | numeric.notna()

    out = meta_df.copy()
    out["events_per_second_global_rate"] = rate_sum.where(valid_any, np.nan)
    return out, selected_prefix


def _resolve_event_count_column(df: pd.DataFrame) -> str | None:
    """Return preferred event-count column name if available."""
    for candidate in (
        "selected_rows",
        "requested_rows",
        "generated_events_count",
        "num_events",
        "event_count",
    ):
        if candidate in df.columns:
            return candidate
    return None


def _load_rate_histogram_bins(
    *,
    station_id: int,
    task_id: int,
    metadata_agg: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load per-file rate-histogram bins for one task (if available)."""
    rate_hist_path = _task_rate_histogram_metadata_path(station_id, task_id)
    if not rate_hist_path.exists():
        log.info("Rate-histogram metadata CSV not found for task %d: %s", task_id, rate_hist_path)
        return (None, [])

    log.info("Loading metadata (rate_histogram) for task %d: %s", task_id, rate_hist_path)
    rate_hist_df = pd.read_csv(rate_hist_path, low_memory=False)
    if "filename_base" not in rate_hist_df.columns:
        log.warning(
            "  Task %d rate_histogram metadata has no 'filename_base'; skipping histogram merge.",
            task_id,
        )
        return (None, [])
    if metadata_agg == "latest":
        rate_hist_df = _aggregate_latest(rate_hist_df)

    hist_cols = [
        c for c in rate_hist_df.columns
        if RATE_HIST_BIN_RE.match(str(c)) is not None
    ]
    hist_cols.sort(key=lambda c: int(RATE_HIST_BIN_RE.match(str(c)).group("bin")))
    if not hist_cols:
        log.warning(
            "  Task %d rate_histogram metadata has no events_per_second_<bin>_rate_hz columns.",
            task_id,
        )
        return (None, [])

    out = rate_hist_df[["filename_base", *hist_cols]].copy()
    out = out.groupby("filename_base", sort=False).tail(1)
    log.info(
        "  Rate-histogram rows (after aggregation): %d with %d bin columns.",
        len(out),
        len(hist_cols),
    )
    return (out, hist_cols)


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
    task_ids = _safe_task_ids(config.get("task_ids", [1]))
    cfg_11 = config.get("step_1_1", {})
    min_rows_raw = cfg_11.get("min_rows_for_dataset", 0)
    try:
        min_rows_for_dataset = max(0, int(min_rows_raw))
    except (TypeError, ValueError):
        log.warning(
            "Invalid step_1_1.min_rows_for_dataset=%r; using 0 (disabled).",
            min_rows_raw,
        )
        min_rows_for_dataset = 0
    metadata_prefix_override = _resolve_metadata_prefix_override(
        cfg_11.get("metadata_prefix", "auto")
    )
    preferred_tt_prefixes = list(_preferred_tt_prefixes_for_task_ids(task_ids))
    if metadata_prefix_override is not None:
        preferred_tt_prefixes = [
            metadata_prefix_override,
            *[p for p in preferred_tt_prefixes if p != metadata_prefix_override],
        ]
    preferred_tt_prefixes_t = tuple(preferred_tt_prefixes)
    if metadata_prefix_override is None:
        log.info(
            "STEP 1.1 metadata_prefix=auto -> preferred TT prefixes from task_ids %s: %s",
            task_ids,
            preferred_tt_prefixes_t,
        )
    else:
        log.info(
            "STEP 1.1 metadata_prefix override '%s' -> preferred TT prefixes: %s",
            metadata_prefix_override,
            preferred_tt_prefixes_t,
        )

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
        specific_path = _task_specific_metadata_path(station_id, task_id)
        trigger_path = _task_trigger_type_metadata_path(station_id, task_id)

        meta_df: pd.DataFrame | None = None

        # Primary source: trigger_type metadata.
        if trigger_path.exists():
            log.info("Loading metadata (trigger_type) for task %d: %s", task_id, trigger_path)
            trigger_df = pd.read_csv(trigger_path, low_memory=False)
            if "filename_base" not in trigger_df.columns:
                log.error("  No 'filename_base' column in task %d trigger_type metadata.", task_id)
                trigger_df = None
            elif metadata_agg == "latest":
                trigger_df = _aggregate_latest(trigger_df)
            if trigger_df is not None:
                log.info("  Trigger-type rows (after aggregation): %d", len(trigger_df))
                meta_df = trigger_df
        else:
            log.info("Trigger-type metadata CSV not found for task %d: %s", task_id, trigger_path)

        # Fallback source: specific metadata only when trigger_type is unavailable.
        if meta_df is None:
            if specific_path.exists():
                log.warning(
                    "Using fallback metadata (specific) for task %d because trigger_type is unavailable: %s",
                    task_id,
                    specific_path,
                )
                specific_df = pd.read_csv(specific_path, low_memory=False)
                if "filename_base" not in specific_df.columns:
                    log.error("  No 'filename_base' column in task %d specific metadata.", task_id)
                else:
                    if metadata_agg == "latest":
                        specific_df = _aggregate_latest(specific_df)
                    log.info("  Specific rows (after aggregation): %d", len(specific_df))
                    meta_df = specific_df
            else:
                log.warning("Specific metadata CSV not found for task %d: %s", task_id, specific_path)

        if meta_df is None:
            log.warning("No usable metadata for task %d — skipping.", task_id)
            continue

        rate_hist_df, rate_hist_cols = _load_rate_histogram_bins(
            station_id=station_id,
            task_id=task_id,
            metadata_agg=metadata_agg,
        )
        if rate_hist_df is not None and rate_hist_cols:
            new_hist_cols = [c for c in rate_hist_cols if c not in meta_df.columns]
            if new_hist_cols:
                meta_df = meta_df.merge(
                    rate_hist_df[["filename_base", *new_hist_cols]],
                    on="filename_base",
                    how="left",
                )
                rows_with_hist = int(meta_df[new_hist_cols].notna().any(axis=1).sum())
                log.info(
                    "  Joined rate_histogram bins for task %d: %d columns, %d/%d rows with histogram values.",
                    task_id,
                    len(new_hist_cols),
                    rows_with_hist,
                    len(meta_df),
                )
            else:
                log.info(
                    "  Rate-histogram bin columns already present for task %d; merge skipped.",
                    task_id,
                )

        meta_df, rate_prefix = _attach_tt_sum_global_rate(
            meta_df,
            preferred_prefixes=preferred_tt_prefixes_t,
        )
        if rate_prefix is not None:
            log.info(
                "  Global rate for task %d set as TT-rate sum from prefix '%s'.",
                task_id,
                rate_prefix,
            )
        elif "events_per_second_global_rate" in meta_df.columns:
            log.info(
                "  Using existing global-rate column for task %d: events_per_second_global_rate",
                task_id,
            )
        else:
            log.warning(
                "  No TT-rate columns found to derive global rate for task %d.",
                task_id,
            )

        log.info("  Metadata rows (selected source): %d", len(meta_df))

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

    ev_col = _resolve_event_count_column(collected)
    removed_below_min_rows = 0
    if min_rows_for_dataset > 0:
        if ev_col is None:
            log.warning(
                "step_1_1.min_rows_for_dataset=%d configured, but no event-count column is available; skipping row filter.",
                min_rows_for_dataset,
            )
        else:
            ev_vals = pd.to_numeric(collected[ev_col], errors="coerce")
            keep_mask = ev_vals >= float(min_rows_for_dataset)
            removed_below_min_rows = int(np.count_nonzero(~keep_mask))
            collected = collected.loc[keep_mask].reset_index(drop=True)
            log.info(
                "Applied minimum event rows filter (%s >= %d): removed %d row(s), %d remain.",
                ev_col,
                min_rows_for_dataset,
                removed_below_min_rows,
                len(collected),
            )
            if collected.empty:
                log.error(
                    "No rows remain after applying step_1_1.min_rows_for_dataset=%d on column %s.",
                    min_rows_for_dataset,
                    ev_col,
                )
                return 1

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
        "metadata_prefix_override": metadata_prefix_override,
        "metadata_prefix_preferred_order": list(preferred_tt_prefixes_t),
        "min_rows_for_dataset": int(min_rows_for_dataset),
        "event_count_column_used": ev_col,
        "rows_removed_below_min_rows": int(removed_below_min_rows),
    }
    z_info_path = FILES_DIR / "z_config_selected.json"
    with open(z_info_path, "w", encoding="utf-8") as f:
        json.dump(z_info, f, indent=2)
    log.info("Wrote z config info: %s", z_info_path)

    # ── Plot: event count histogram ──────────────────────────────────
    if ev_col is not None and ev_col in collected.columns:
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
