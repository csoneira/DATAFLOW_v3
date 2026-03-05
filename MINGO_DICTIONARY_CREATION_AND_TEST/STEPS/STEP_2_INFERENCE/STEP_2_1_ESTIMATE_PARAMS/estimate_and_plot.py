#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_1_ESTIMATE_PARAMS/estimate_and_plot.py
Purpose: STEP 2.1 — Solution to the inverse problem.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_1_ESTIMATE_PARAMS/estimate_and_plot.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent           # STEP_2_INFERENCE
PIPELINE_DIR = INFERENCE_DIR.parent       # .../STEPS
PROJECT_DIR = PIPELINE_DIR.parent         # .../MINGO_DICTIONARY_CREATION_AND_TEST
DEFAULT_CONFIG = PROJECT_DIR / "config.json"

DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_DATASET = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_DATASET_ENLARGED = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_3_ENLARGE_DATASET"
    / "OUTPUTS" / "FILES" / "enlarged_dataset.csv"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "2_1"


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

# Import the self-contained estimation module
sys.path.insert(0, str(INFERENCE_DIR))
from estimate_parameters import (  # noqa: E402
    DISTANCE_FNS,
    _auto_feature_columns as _shared_auto_feature_columns,
    estimate_parameters,
)

logging.basicConfig(
    format="[%(levelname)s] STEP_2.1 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.1")


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


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_input_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_project = PROJECT_DIR / p
    if candidate_project.exists():
        return candidate_project
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_project


def _select_default_dataset_path(config: dict) -> Path:
    """Choose STEP 2 dataset source from STEP 1.3 enable state."""
    cfg_13 = config.get("step_1_3", {})
    enabled_13 = _as_bool(cfg_13.get("enabled", False), False)
    if not enabled_13:
        return DEFAULT_DATASET

    enlarged_cfg = cfg_13.get("enlarged_dataset_csv", None)
    enlarged_path = _resolve_input_path(enlarged_cfg) if enlarged_cfg else DEFAULT_DATASET_ENLARGED
    if enlarged_path.exists():
        log.info("STEP 1.3 selection: using enlarged dataset for STEP 2 (%s).", enlarged_path)
        return enlarged_path

    log.warning(
        "STEP 1.3 is enabled but enlarged dataset file is missing: %s. Falling back to STEP 1.2 dataset.",
        enlarged_path,
    )
    return DEFAULT_DATASET


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2.1: Estimate parameters using dictionary matching."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--dataset-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()
    cfg_21 = config.get("step_2_1", {})

    dict_path = _resolve_input_path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY
    data_path = _resolve_input_path(args.dataset_csv) if args.dataset_csv else _select_default_dataset_path(config)
    cfg_13 = config.get("step_1_3", {})
    if args.dataset_csv:
        dataset_mode = "cli_dataset_override"
    elif (
        _as_bool(cfg_13.get("enabled", False), False)
        and data_path.resolve() != DEFAULT_DATASET.resolve()
    ):
        dataset_mode = "step_1_3_enlarged"
    else:
        dataset_mode = "step_1_2_original"

    feature_columns = cfg_21.get("feature_columns", "auto")
    distance_metric = cfg_21.get("distance_metric", "l2_zscore")
    interpolation_k_cfg = cfg_21.get("interpolation_k", 5)
    interpolation_k = None if interpolation_k_cfg is None else int(interpolation_k_cfg)
    include_global_rate = cfg_21.get("include_global_rate", True)
    global_rate_col = cfg_21.get("global_rate_col", "events_per_second_global_rate")
    plot_params = cfg_21.get("plot_parameters", None)
    if plot_params is None:
        legacy_plot_params = config.get("step_2_2", {}).get("plot_parameters", None)
        if legacy_plot_params is not None:
            log.warning(
                "Deprecated config key step_2_2.plot_parameters detected; use step_2_1.plot_parameters."
            )
            plot_params = legacy_plot_params
    if plot_params is None:
        plot_params = config.get("step_1_2", {}).get("plot_parameters", None)

    if not dict_path.exists():
        log.error("Dictionary CSV not found: %s", dict_path)
        return 1
    if not data_path.exists():
        log.error("Dataset CSV not found: %s", data_path)
        return 1

    log.info("Dictionary: %s", dict_path)
    log.info("Dataset:    %s", data_path)
    log.info("Metric:     %s", distance_metric)
    log.info(
        "K:          %s",
        "all dictionary candidates" if interpolation_k is None else str(interpolation_k),
    )

    # ── Run estimation ───────────────────────────────────────────────
    result_df = estimate_parameters(
        dictionary_path=str(dict_path),
        dataset_path=str(data_path),
        feature_columns=feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
        exclude_same_file=True,
    )

    # ── Merge with dataset to have truth values alongside ────────────
    data_df = pd.read_csv(data_path, low_memory=False)

    # Attach truth columns needed for validation
    truth_cols = ["flux_cm2_min", "cos_n",
                  "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
                  "n_events", "is_dictionary_entry"]
    for col in truth_cols:
        if col in data_df.columns:
            result_df[f"true_{col}"] = data_df[col].values[:len(result_df)]

    # ── Save ─────────────────────────────────────────────────────────
    out_path = FILES_DIR / "estimated_params.csv"
    result_df.to_csv(out_path, index=False)
    log.info("Wrote estimated params: %s (%d rows)", out_path, len(result_df))

    n_ok = result_df["best_distance"].notna().sum()
    n_fail = result_df["best_distance"].isna().sum()

    summary = {
        "dictionary": str(dict_path),
        "dataset": str(data_path),
        "dataset_source_mode": dataset_mode,
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "feature_columns": feature_columns if isinstance(feature_columns, list) else "auto",
        "total_points": len(result_df),
        "successful_estimates": int(n_ok),
        "failed_estimates": int(n_fail),
    }
    with open(FILES_DIR / "estimation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostic plots ─────────────────────────────────────────────
    _make_plots(
        result_df=result_df,
        data_df=data_df,
        plot_params=plot_params,
        dict_path=dict_path,
        cfg_21=cfg_21,
    )

    log.info("Done.")
    return 0


def _auto_feature_columns(
    df: pd.DataFrame,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
) -> list[str]:
    return _shared_auto_feature_columns(
        df=df,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
    )


def _l2_distances(sample_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    valid = np.isfinite(candidates) & np.isfinite(sample_vec[np.newaxis, :])
    n_valid = valid.sum(axis=1)
    diff = np.where(valid, candidates - sample_vec[np.newaxis, :], 0.0)
    d = np.sqrt(np.sum(diff * diff, axis=1))
    d[n_valid < 2] = np.nan
    return d


def _axis_label_for_param(param_name: str) -> str:
    if param_name == "flux_cm2_min":
        return "Flux [cm⁻² min⁻¹]"
    if param_name.startswith("eff_"):
        return f"Efficiency ({param_name})"
    return param_name


def _sanitize_plot_token(token: str) -> str:
    out = []
    for char in str(token):
        if char.isalnum() or char in {"_", "-"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "param"


def _select_showcase_param_pairs(
    plot_params: object,
    result_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    if isinstance(plot_params, (list, tuple)):
        requested = [str(p) for p in plot_params]
    elif isinstance(plot_params, set):
        requested = sorted(str(p) for p in plot_params)
    else:
        requested = []

    if not requested:
        requested = []
        for col in result_df.columns:
            if not col.startswith("est_"):
                continue
            pname = col[4:]
            if f"true_{pname}" not in result_df.columns:
                continue
            if pname not in cand_df.columns:
                continue
            vals = pd.to_numeric(cand_df[pname], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            requested.append(pname)

    selected_params: list[str] = []
    seen: set[str] = set()
    for pname in requested:
        if pname in seen:
            continue
        if pname not in cand_df.columns:
            log.warning("Showcase parameter '%s' not found in dictionary; skipping.", pname)
            continue
        if f"est_{pname}" not in result_df.columns:
            log.warning("Showcase parameter '%s' has no est_%s column; skipping.", pname, pname)
            continue
        if f"true_{pname}" not in result_df.columns:
            log.warning("Showcase parameter '%s' has no true_%s column; skipping.", pname, pname)
            continue
        vals = pd.to_numeric(cand_df[pname], errors="coerce")
        if vals.notna().sum() == 0:
            log.warning("Showcase parameter '%s' has no finite dictionary values; skipping.", pname)
            continue
        selected_params.append(pname)
        seen.add(pname)

    return list(combinations(selected_params, 2))


def _select_showcase_matrix_parameters(
    cfg_21: dict,
    plot_params: object,
    result_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> list[str]:
    """Resolve ordered parameter list for showcase matrix cells."""
    requested: list[str] = []

    if isinstance(plot_params, str):
        requested = [x.strip() for x in plot_params.split(",") if x.strip()]
    elif isinstance(plot_params, (list, tuple, set)):
        requested = [str(p) for p in plot_params]
    else:
        requested = []

    # Backward compatibility with previous showcase-only knobs.
    if not requested:
        matrix_cfg = cfg_21.get("showcase_matrix_parameters", None)
        if isinstance(matrix_cfg, str):
            requested = [x.strip() for x in matrix_cfg.split(",") if x.strip()]
        elif isinstance(matrix_cfg, (list, tuple, set)):
            requested = [str(x) for x in matrix_cfg]
        if requested:
            log.warning(
                "Deprecated key step_2_1.showcase_matrix_parameters detected; use step_2_1.plot_parameters."
            )

    if not requested:
        for col in result_df.columns:
            if not col.startswith("est_"):
                continue
            pname = col[4:]
            if f"true_{pname}" in result_df.columns:
                requested.append(pname)

    selected: list[str] = []
    seen: set[str] = set()
    for pname in requested:
        if pname in seen:
            continue
        if pname not in cand_df.columns:
            log.warning("Showcase matrix parameter '%s' not found in dictionary candidates; skipping.", pname)
            continue
        if f"est_{pname}" not in result_df.columns:
            log.warning("Showcase matrix parameter '%s' has no est_%s column; skipping.", pname, pname)
            continue
        if f"true_{pname}" not in result_df.columns:
            log.warning("Showcase matrix parameter '%s' has no true_%s column; skipping.", pname, pname)
            continue
        vals = pd.to_numeric(cand_df[pname], errors="coerce")
        if vals.notna().sum() == 0:
            log.warning("Showcase matrix parameter '%s' has no finite candidate values; skipping.", pname)
            continue
        selected.append(pname)
        seen.add(pname)
    return selected


def _resolve_showcase_fixed_tolerance_pct(cfg_21: dict) -> float:
    """Global tolerance (%) used to relax fixed-parameter matching in showcase slices."""
    raw = (cfg_21 or {}).get("showcase_fixed_tolerance_pct", 5.0)
    if raw in (None, "", "null", "None"):
        return 5.0
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "Invalid step_2_1.showcase_fixed_tolerance_pct=%r; using default 5.0%%.",
            raw,
        )
        return 5.0
    if not np.isfinite(pct) or pct < 0.0:
        log.warning(
            "Invalid step_2_1.showcase_fixed_tolerance_pct=%r; using default 5.0%%.",
            raw,
        )
        return 5.0
    return pct


def _tolerance_band_from_unique_values(
    uniq_values: np.ndarray,
    target: float,
    nearest: float,
    tolerance_pct: float,
) -> tuple[float, float]:
    """Return (band, eps) for a percent-based tolerance around target."""
    pct = max(0.0, float(tolerance_pct))
    if pct <= 0.0:
        return (0.0, 0.0)
    span = float(np.max(uniq_values) - np.min(uniq_values)) if uniq_values.size >= 2 else 0.0
    scale = span if span > 0.0 else max(1.0, abs(target), abs(nearest))
    band = (pct / 100.0) * scale
    eps = max(1e-12, 1e-12 * scale)
    return (band, eps)


def _snap_fixed_value(
    values: np.ndarray,
    target: float,
    tolerance_pct: float,
) -> tuple[float | None, float | None, str | None]:
    uniq = np.unique(values[np.isfinite(values)])
    if uniq.size == 0:
        return (None, None, "no_finite_values")
    nearest = float(uniq[int(np.argmin(np.abs(uniq - target)))])
    if uniq.size >= 2:
        diffs = np.diff(np.sort(uniq))
        diffs = diffs[diffs > 0.0]
        min_step = float(diffs.min()) if diffs.size else np.nan
    else:
        min_step = np.nan
    if np.isfinite(min_step):
        atol = max(1e-9, 0.10 * min_step)
    else:
        atol = max(1e-9, 1e-6 * max(1.0, abs(nearest)))

    pct = max(0.0, float(tolerance_pct))
    if pct > 0.0:
        band, eps = _tolerance_band_from_unique_values(
            uniq_values=uniq,
            target=target,
            nearest=nearest,
            tolerance_pct=pct,
        )
        diff = abs(nearest - target)
        if diff > band + eps:
            return (
                None,
                None,
                f"outside_tolerance(diff={diff:.4g}>band={band:.4g},pct={pct:.4g})",
            )

    return nearest, atol, None


def _mask_within_fixed_tolerance(
    values: np.ndarray,
    target: float,
    nearest: float,
    atol: float,
    tolerance_pct: float,
) -> np.ndarray:
    """Mask values compatible with the fixed-parameter tolerance criterion."""
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values, dtype=bool)

    pct = max(0.0, float(tolerance_pct))
    if pct <= 0.0:
        return finite_mask & np.isclose(values, nearest, rtol=0.0, atol=atol)

    uniq = np.unique(values[finite_mask])
    band, eps = _tolerance_band_from_unique_values(
        uniq_values=uniq,
        target=target,
        nearest=nearest,
        tolerance_pct=pct,
    )
    mask = finite_mask & (np.abs(values - target) <= (band + eps))
    if np.any(mask):
        return mask

    # Numerical fallback: always retain the snapped closest value.
    return finite_mask & np.isclose(values, nearest, rtol=0.0, atol=atol)


def _format_fixed_params_note(fixed_params: dict[str, float]) -> str:
    if not fixed_params:
        return "fixed: (none)"
    chunks: list[str] = []
    items = [f"{k}={v:.4g}" for k, v in fixed_params.items()]
    for i in range(0, len(items), 3):
        chunks.append(", ".join(items[i:i + 3]))
    return "fixed: " + "\n       ".join(chunks)


def _is_axis_alias_param(
    cand_df: pd.DataFrame,
    fixed_param: str,
    axis_param: str,
) -> bool:
    if fixed_param not in cand_df.columns or axis_param not in cand_df.columns:
        return False
    a = pd.to_numeric(cand_df[fixed_param], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(cand_df[axis_param], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return False
    delta = np.abs(a[mask] - b[mask])
    scale = max(
        1.0,
        float(np.nanmax(np.abs(a[mask]))),
        float(np.nanmax(np.abs(b[mask]))),
    )
    return bool(np.nanmax(delta) <= 1e-9 * scale)


def _make_random_showcase_l2_contour(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    dict_path: Path,
    cfg_21: dict,
    plot_params: object = None,
) -> None:
    if not dict_path.exists():
        return

    required = ["dataset_index", "best_distance"]
    for col in required:
        if col not in result_df.columns:
            return

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return

    seed_raw = (cfg_21 or {}).get("showcase_seed", None)
    auto_seed = seed_raw in (None, "", "null", "None", "auto", "random")
    if auto_seed:
        showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        log.info("Random showcase seed (auto): %d", showcase_seed)
    else:
        try:
            showcase_seed = int(seed_raw)
        except (TypeError, ValueError):
            showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            log.warning(
                "Invalid step_2_1.showcase_seed=%r; using auto seed %d.",
                seed_raw,
                showcase_seed,
            )
        else:
            log.info("Random showcase seed (fixed): %d", showcase_seed)
    rng = np.random.default_rng(showcase_seed)
    valid_indices = result_df.index[valid_mask].to_numpy()
    chosen_idx = int(rng.choice(valid_indices))
    row = result_df.loc[chosen_idx]
    ds_idx = int(pd.to_numeric(pd.Series([row["dataset_index"]]), errors="coerce").iloc[0])
    log.info(
        "Random showcase selected dataset_index=%d (result row=%d).",
        ds_idx,
        chosen_idx,
    )
    if ds_idx < 0 or ds_idx >= len(data_df):
        return

    dict_df = pd.read_csv(dict_path, low_memory=False)

    include_global_rate = bool(cfg_21.get("include_global_rate", True))
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    feature_cfg = cfg_21.get("feature_columns", "auto")
    if isinstance(feature_cfg, str) and feature_cfg == "auto":
        feature_cols = sorted(
            set(_auto_feature_columns(dict_df, include_global_rate, global_rate_col))
            & set(_auto_feature_columns(data_df, include_global_rate, global_rate_col))
        )
    else:
        feature_cols = [
            str(c) for c in list(feature_cfg)
            if str(c) in dict_df.columns and str(c) in data_df.columns
        ]
    if not feature_cols:
        return

    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    sample_feat = pd.to_numeric(data_df.loc[ds_idx, feature_cols], errors="coerce").to_numpy(dtype=float)

    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    # short token for labels/filenames (e.g. 'l2' from 'l2_zscore', 'chi2' from 'chi2')
    metric_short = distance_metric.split("_")[0]
    metric_label = "L2" if metric_short == "l2" else metric_short

    if distance_metric == "l2_zscore":
        means = dict_feat.mean(axis=0, skipna=True)
        stds = dict_feat.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_mat = ((dict_feat - means) / stds).to_numpy(dtype=float)
        sample_vec = ((sample_feat - means.to_numpy(dtype=float)) / stds.to_numpy(dtype=float))
    else:
        dict_mat = dict_feat.to_numpy(dtype=float)
        sample_vec = sample_feat

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"] if c in dict_df.columns and c in data_df.columns]
    if z_cols:
        z_tol = float(cfg_21.get("z_tol", 1e-6))
        dict_z = dict_df[z_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sample_z = pd.to_numeric(data_df.loc[ds_idx, z_cols], errors="coerce").to_numpy(dtype=float)
        z_mask = np.all(np.abs(dict_z - sample_z[np.newaxis, :]) <= z_tol, axis=1)
    else:
        z_mask = np.ones(len(dict_df), dtype=bool)

    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    if join_col is not None:
        sample_id = str(data_df.loc[ds_idx, join_col])
        z_mask &= (dict_df[join_col].astype(str).to_numpy() != sample_id)

    cand_df = dict_df.loc[z_mask].copy()
    if cand_df.empty:
        return

    cand_mat = dict_mat[z_mask]

    param_pairs = _select_showcase_param_pairs(
        plot_params=plot_params,
        result_df=result_df,
        cand_df=cand_df,
    )
    showcase_pair_cfg = (cfg_21 or {}).get("showcase_param_pair", None)
    if isinstance(showcase_pair_cfg, (list, tuple)) and len(showcase_pair_cfg) == 2:
        pair_x = str(showcase_pair_cfg[0])
        pair_y = str(showcase_pair_cfg[1])
        if (pair_x, pair_y) in param_pairs:
            param_pairs = [(pair_x, pair_y)]
        elif (pair_y, pair_x) in param_pairs:
            param_pairs = [(pair_y, pair_x)]
        else:
            log.warning(
                "Configured showcase_param_pair=(%s, %s) is not available; using auto-selected pairs.",
                pair_x,
                pair_y,
            )
    if not param_pairs:
        log.warning("Random showcase: no valid parameter pairs selected for contour plotting.")
        return

    showcase_max_plots_raw = (cfg_21 or {}).get("showcase_max_plots", 1)
    try:
        showcase_max_plots = max(1, int(showcase_max_plots_raw))
    except (TypeError, ValueError):
        showcase_max_plots = 1
    fixed_tolerance_pct = _resolve_showcase_fixed_tolerance_pct(cfg_21)

    all_estimated_params: list[str] = []
    for col in result_df.columns:
        if not col.startswith("est_"):
            continue
        pname = col[4:]
        if pname not in cand_df.columns:
            continue
        if f"true_{pname}" not in result_df.columns:
            continue
        all_estimated_params.append(pname)
    all_estimated_params = sorted(set(all_estimated_params))

    # Use the same distance function used by the estimator so the plotted
    # quantity matches the reported `best_distance` (e.g. chi2, l2_zscore).
    dist_fn = DISTANCE_FNS.get(distance_metric, DISTANCE_FNS.get(metric_short, None))

    generated = 0
    for x_param, y_param in param_pairs:
        if generated >= showcase_max_plots:
            break
        true_x = float(pd.to_numeric(pd.Series([row.get(f"true_{x_param}")]), errors="coerce").iloc[0])
        true_y = float(pd.to_numeric(pd.Series([row.get(f"true_{y_param}")]), errors="coerce").iloc[0])
        est_x = float(pd.to_numeric(pd.Series([row.get(f"est_{x_param}")]), errors="coerce").iloc[0])
        est_y = float(pd.to_numeric(pd.Series([row.get(f"est_{y_param}")]), errors="coerce").iloc[0])
        if not (np.isfinite(true_x) and np.isfinite(true_y) and np.isfinite(est_x) and np.isfinite(est_y)):
            log.warning(
                "Random showcase pair (%s, %s): missing true/estimated values in chosen row; skipping.",
                x_param,
                y_param,
            )
            continue

        raw_fixed_params = [p for p in all_estimated_params if p not in {x_param, y_param}]
        fixed_params: list[str] = []
        for pname in raw_fixed_params:
            if _is_axis_alias_param(cand_df, pname, x_param) or _is_axis_alias_param(cand_df, pname, y_param):
                log.info(
                    "Random showcase pair (%s, %s): not fixing '%s' because it mirrors a plotted axis.",
                    x_param,
                    y_param,
                    pname,
                )
                continue
            fixed_params.append(pname)
        fixed_mask = np.ones(len(cand_df), dtype=bool)
        fixed_values: dict[str, float] = {}

        failed_fixed = False
        for pname in fixed_params:
            est_col = f"est_{pname}"
            target = float(pd.to_numeric(pd.Series([row.get(est_col)]), errors="coerce").iloc[0])
            if not np.isfinite(target):
                log.warning(
                    "Random showcase pair (%s, %s): fixed parameter %s has no finite estimate; skipping pair.",
                    x_param,
                    y_param,
                    pname,
                )
                failed_fixed = True
                break

            vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size == 0:
                log.warning(
                    "Random showcase pair (%s, %s): fixed parameter %s has no finite dictionary values; skipping pair.",
                    x_param,
                    y_param,
                    pname,
                )
                failed_fixed = True
                break

            snapped, atol, snap_reason = _snap_fixed_value(
                finite_vals,
                target,
                tolerance_pct=fixed_tolerance_pct,
            )
            if snapped is None or atol is None:
                log.warning(
                    "Random showcase pair (%s, %s): fixed parameter %s target %.6g not matched within tolerance %.4g%% (%s); skipping pair.",
                    x_param,
                    y_param,
                    pname,
                    target,
                    fixed_tolerance_pct,
                    snap_reason or "snap_failed",
                )
                failed_fixed = True
                break
            fixed_values[pname] = snapped
            fixed_mask &= np.isfinite(vals) & np.isclose(vals, snapped, rtol=0.0, atol=atol)

            if fixed_mask.sum() == 0:
                log.warning(
                    "Random showcase pair (%s, %s): no candidates remain after fixing %s=%.6g; skipping pair.",
                    x_param,
                    y_param,
                    pname,
                    snapped,
                )
                failed_fixed = True
                break

        pair_df: pd.DataFrame
        pair_mat: np.ndarray
        fallback_reason: str | None = None
        if failed_fixed:
            fallback_reason = "fixed_constraints_rejected"
            pair_df = cand_df.copy()
            pair_mat = cand_mat
            fixed_values = {}
        else:
            pair_df = cand_df.loc[fixed_mask].copy()
            pair_mat = cand_mat[fixed_mask]
            if pair_df.empty:
                fallback_reason = "fixed_constraints_empty_slice"
                pair_df = cand_df.copy()
                pair_mat = cand_mat
                fixed_values = {}

        if fallback_reason is not None:
            log.info(
                "Random showcase pair (%s, %s): fallback to unconstrained slice (%s).",
                x_param,
                y_param,
                fallback_reason,
            )

        def _build_pair_frame(work_df: pd.DataFrame, work_mat: np.ndarray) -> pd.DataFrame:
            tmp = work_df.copy()
            if dist_fn is None:
                z_vals = _l2_distances(sample_vec, work_mat)
                tmp["distance_value"] = z_vals
            else:
                z_list = [dist_fn(sample_vec, work_mat[i]) for i in range(work_mat.shape[0])]
                tmp["distance_value"] = np.array(z_list, dtype=float)
            tmp["x_for_plot"] = pd.to_numeric(tmp[x_param], errors="coerce")
            tmp["y_for_plot"] = pd.to_numeric(tmp[y_param], errors="coerce")
            tmp = tmp.dropna(subset=["x_for_plot", "y_for_plot", "distance_value"])
            if tmp.empty:
                return tmp
            return (
                tmp.groupby(["x_for_plot", "y_for_plot"], as_index=False, sort=True)["distance_value"]
                .min()
            )

        pair_df = _build_pair_frame(pair_df, pair_mat)
        if pair_df.empty:
            log.warning(
                "Random showcase pair (%s, %s): no finite points after numeric cleanup; skipping.",
                x_param,
                y_param,
            )
            continue

        if len(pair_df) < 3:
            if pair_mat.shape[0] < cand_mat.shape[0]:
                log.info(
                    "Random showcase pair (%s, %s): only %d unique points after fixed slicing; "
                    "fallback to unconstrained slice.",
                    x_param,
                    y_param,
                    len(pair_df),
                )
                fixed_values = {}
                pair_df = _build_pair_frame(cand_df, cand_mat)
            if pair_df.empty or len(pair_df) < 3:
                log.warning(
                    "Random showcase pair (%s, %s): only %d unique points; skipping.",
                    x_param,
                    y_param,
                    len(pair_df),
                )
                continue

        x = pair_df["x_for_plot"].to_numpy(dtype=float)
        y = pair_df["y_for_plot"].to_numpy(dtype=float)
        z = pair_df["distance_value"].to_numpy(dtype=float)
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min, z_max = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(8, 5.5))
        contour_ok = False
        try:
            tri = mtri.Triangulation(x, y)
            ctf = ax.tricontourf(
                tri, z, levels=24, cmap="viridis_r", alpha=0.75, vmin=z_min, vmax=z_max
            )
            ax.tricontour(
                tri, z, levels=12, colors="white", linewidths=0.35, alpha=0.30
            )
            contour_ok = True
        except Exception:
            contour_ok = False

        sc = ax.scatter(
            x, y, c=z, cmap="viridis_r", vmin=z_min, vmax=z_max,
            s=36, marker="o", alpha=0.93,
            edgecolors=(1.0, 1.0, 1.0, 0.75), linewidths=0.35, zorder=4
        )
        cb = fig.colorbar(ctf if contour_ok else sc, ax=ax, shrink=0.88)
        cb.set_label(f"{metric_label} distance in feature space")

        ax.scatter(
            [true_x], [true_y], s=170, marker="*", color="#E45756",
            edgecolors="black", linewidths=0.6, zorder=6, label="True point"
        )
        ax.scatter(
            [est_x], [est_y], s=140, marker="X", color="#F58518",
            edgecolors="black", linewidths=0.6, zorder=6, label="Estimated point"
        )

        ax.set_xlabel(_axis_label_for_param(x_param))
        ax.set_ylabel(_axis_label_for_param(y_param))
        ax.set_title(
            "Random showcase "
            f"{metric_label} distance map ({x_param} vs {y_param}, dataset_index={ds_idx}, "
            f"candidates={len(pair_df)})"
        )
        ax.legend(loc="best", fontsize=8)

        note = (
            f"true: {x_param}={true_x:.4g}, {y_param}={true_y:.4g}\n"
            f"est:  {x_param}={est_x:.4g}, {y_param}={est_y:.4g}\n"
            f"best_distance={float(row['best_distance']):.4g}\n"
            f"{_format_fixed_params_note(fixed_values)}"
        )
        ax.text(
            0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85)
        )

        fig.tight_layout()
        safe_x = _sanitize_plot_token(x_param)
        safe_y = _sanitize_plot_token(y_param)
        _save_figure(
            fig,
            PLOTS_DIR / f"random_showcase_distance_contour_{safe_x}__{safe_y}.png",
        )
        plt.close(fig)
        generated += 1

    if generated == 0:
        log.warning("Random showcase: no contour figure generated for any selected parameter pair.")


def _make_random_showcase_distance_matrix(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    dict_path: Path,
    cfg_21: dict,
    plot_params: object = None,
) -> None:
    """Build one n x n showcase matrix with pair maps + diagonal projections."""
    if not dict_path.exists():
        return

    required = ["dataset_index", "best_distance"]
    for col in required:
        if col not in result_df.columns:
            return

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return

    seed_raw = (cfg_21 or {}).get("showcase_seed", None)
    auto_seed = seed_raw in (None, "", "null", "None", "auto", "random")
    if auto_seed:
        showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        log.info("Showcase matrix seed (auto): %d", showcase_seed)
    else:
        try:
            showcase_seed = int(seed_raw)
        except (TypeError, ValueError):
            showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            log.warning(
                "Invalid step_2_1.showcase_seed=%r; using auto seed %d for showcase matrix.",
                seed_raw,
                showcase_seed,
            )
        else:
            log.info("Showcase matrix seed (fixed): %d", showcase_seed)

    rng = np.random.default_rng(showcase_seed)
    valid_indices = result_df.index[valid_mask].to_numpy()
    chosen_idx = int(rng.choice(valid_indices))
    row = result_df.loc[chosen_idx]
    ds_idx = int(pd.to_numeric(pd.Series([row["dataset_index"]]), errors="coerce").iloc[0])
    log.info(
        "Showcase matrix selected dataset_index=%d (result row=%d).",
        ds_idx,
        chosen_idx,
    )
    if ds_idx < 0 or ds_idx >= len(data_df):
        return

    dict_df = pd.read_csv(dict_path, low_memory=False)

    include_global_rate = bool(cfg_21.get("include_global_rate", True))
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    feature_cfg = cfg_21.get("feature_columns", "auto")
    if isinstance(feature_cfg, str) and feature_cfg == "auto":
        feature_cols = sorted(
            set(_auto_feature_columns(dict_df, include_global_rate, global_rate_col))
            & set(_auto_feature_columns(data_df, include_global_rate, global_rate_col))
        )
    else:
        feature_cols = [
            str(c) for c in list(feature_cfg)
            if str(c) in dict_df.columns and str(c) in data_df.columns
        ]
    if not feature_cols:
        return

    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    sample_feat = pd.to_numeric(data_df.loc[ds_idx, feature_cols], errors="coerce").to_numpy(dtype=float)

    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    metric_short = distance_metric.split("_")[0]
    metric_label = "L2" if metric_short == "l2" else metric_short

    if distance_metric == "l2_zscore":
        means = dict_feat.mean(axis=0, skipna=True)
        stds = dict_feat.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_mat = ((dict_feat - means) / stds).to_numpy(dtype=float)
        sample_vec = ((sample_feat - means.to_numpy(dtype=float)) / stds.to_numpy(dtype=float))
    else:
        dict_mat = dict_feat.to_numpy(dtype=float)
        sample_vec = sample_feat

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"] if c in dict_df.columns and c in data_df.columns]
    if z_cols:
        z_tol = float(cfg_21.get("z_tol", 1e-6))
        dict_z = dict_df[z_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sample_z = pd.to_numeric(data_df.loc[ds_idx, z_cols], errors="coerce").to_numpy(dtype=float)
        z_mask = np.all(np.abs(dict_z - sample_z[np.newaxis, :]) <= z_tol, axis=1)
    else:
        z_mask = np.ones(len(dict_df), dtype=bool)

    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    if join_col is not None:
        sample_id = str(data_df.loc[ds_idx, join_col])
        z_mask &= (dict_df[join_col].astype(str).to_numpy() != sample_id)

    cand_df = dict_df.loc[z_mask].copy()
    if cand_df.empty:
        return
    cand_mat = dict_mat[z_mask]

    matrix_params = _select_showcase_matrix_parameters(
        cfg_21=cfg_21,
        plot_params=plot_params,
        result_df=result_df,
        cand_df=cand_df,
    )
    if len(matrix_params) == 0:
        log.warning("Showcase matrix: no valid parameters selected.")
        return

    max_params_raw = cfg_21.get("showcase_matrix_max_params", None)
    if max_params_raw not in (None, "", "null", "None"):
        try:
            max_params = max(1, int(max_params_raw))
            if len(matrix_params) > max_params:
                matrix_params = matrix_params[:max_params]
        except (TypeError, ValueError):
            log.warning("Invalid step_2_1.showcase_matrix_max_params=%r; ignoring.", max_params_raw)
    fixed_tolerance_pct = _resolve_showcase_fixed_tolerance_pct(cfg_21)

    n_params = len(matrix_params)
    if n_params == 0:
        return

    dist_fn = DISTANCE_FNS.get(distance_metric, DISTANCE_FNS.get(metric_short, None))
    if dist_fn is None:
        cand_distance = _l2_distances(sample_vec, cand_mat)
    else:
        cand_distance = np.array([dist_fn(sample_vec, cand_mat[i]) for i in range(cand_mat.shape[0])], dtype=float)

    optimum_values: dict[str, float] = {}
    finite_dist = np.isfinite(cand_distance)
    if np.any(finite_dist):
        best_local_idx = int(np.nanargmin(cand_distance))
        best_row = cand_df.iloc[best_local_idx]
        for pname in matrix_params:
            v = float(pd.to_numeric(pd.Series([best_row.get(pname)]), errors="coerce").iloc[0])
            if np.isfinite(v):
                optimum_values[pname] = v

    def _limits_with_pad(values: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (0.0, 1.0)
        if np.isclose(lo, hi):
            pad = max(1e-6, 0.02 * max(1.0, abs(lo)))
            return (lo - pad, hi + pad)
        pad = (hi - lo) * pad_frac
        return (lo - pad, hi + pad)

    param_limits: dict[str, tuple[float, float]] = {}
    for pname in matrix_params:
        vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
        true_val = float(pd.to_numeric(pd.Series([row.get(f"true_{pname}")]), errors="coerce").iloc[0])
        est_val = float(pd.to_numeric(pd.Series([row.get(f"est_{pname}")]), errors="coerce").iloc[0])
        ext = vals
        if np.isfinite(true_val):
            ext = np.append(ext, true_val)
        if np.isfinite(est_val):
            ext = np.append(ext, est_val)
        param_limits[pname] = _limits_with_pad(ext)

    dist_limits = _limits_with_pad(cand_distance)
    n_color_levels = 14
    dist_levels = np.linspace(dist_limits[0], dist_limits[1], n_color_levels + 1)
    base_colors = plt.get_cmap("viridis_r")(np.linspace(0.06, 0.94, n_color_levels))
    pastel_mix = 0.32
    pastel_rgb = (1.0 - pastel_mix) * base_colors[:, :3] + pastel_mix * 1.0
    pastel_rgba = np.column_stack([pastel_rgb, np.full(n_color_levels, 0.96)])
    dist_cmap = mcolors.ListedColormap(pastel_rgba, name="viridis_r_pastel")
    dist_norm = mcolors.BoundaryNorm(dist_levels, dist_cmap.N, clip=True)

    def _slice_with_fixed(active_params: set[str]) -> tuple[pd.DataFrame, dict[str, float], str | None]:
        fixed_params = [p for p in matrix_params if p not in active_params]
        active_axis = next(iter(active_params)) if len(active_params) == 1 else None

        trial_tolerances = [float(fixed_tolerance_pct)]
        if fixed_tolerance_pct > 0.0:
            while trial_tolerances[-1] < 100.0:
                next_tol = min(100.0, trial_tolerances[-1] * 2.0)
                if np.isclose(next_tol, trial_tolerances[-1]):
                    break
                trial_tolerances.append(next_tol)

        best_out = pd.DataFrame()
        best_fixed_values: dict[str, float] = {}
        best_unique = -1
        best_tol = trial_tolerances[0]
        last_reason: str | None = None

        for tol_pct in trial_tolerances:
            fixed_mask = np.ones(len(cand_df), dtype=bool)
            fixed_values: dict[str, float] = {}
            failed_reason: str | None = None

            for pname in fixed_params:
                est_col = f"est_{pname}"
                target = optimum_values.get(pname, np.nan)
                if not np.isfinite(target):
                    target = float(pd.to_numeric(pd.Series([row.get(est_col)]), errors="coerce").iloc[0])
                if not np.isfinite(target):
                    failed_reason = f"missing_est_{pname}"
                    break
                vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                if finite_vals.size == 0:
                    failed_reason = f"no_values_{pname}"
                    break
                snapped, atol, snap_reason = _snap_fixed_value(
                    finite_vals,
                    target,
                    tolerance_pct=tol_pct,
                )
                if snapped is None or atol is None:
                    failed_reason = snap_reason or f"no_match_{pname}"
                    break
                fixed_values[pname] = snapped
                fixed_mask &= _mask_within_fixed_tolerance(
                    values=vals,
                    target=target,
                    nearest=snapped,
                    atol=atol,
                    tolerance_pct=tol_pct,
                )
                if fixed_mask.sum() == 0:
                    failed_reason = f"empty_after_fix_{pname}"
                    break

            if failed_reason is not None:
                last_reason = failed_reason
                continue

            out = cand_df.loc[fixed_mask].copy()
            out["distance_value"] = cand_distance[fixed_mask]
            out = out.replace([np.inf, -np.inf], np.nan)
            if out.empty:
                last_reason = "empty_after_all_fixes"
                continue

            unique_count = 1
            if active_axis is not None and active_axis in out.columns:
                unique_count = int(pd.to_numeric(out[active_axis], errors="coerce").dropna().nunique())

            if unique_count > best_unique:
                best_out = out
                best_fixed_values = fixed_values
                best_unique = unique_count
                best_tol = tol_pct

            if active_axis is None or unique_count >= 2:
                if np.isclose(tol_pct, fixed_tolerance_pct):
                    return (out, fixed_values, None)
                return (out, fixed_values, f"expanded_tol_pct={tol_pct:.4g}")

        if best_unique >= 1 and not best_out.empty:
            if np.isclose(best_tol, fixed_tolerance_pct):
                return (best_out, best_fixed_values, "single_point_slice")
            return (best_out, best_fixed_values, f"expanded_tol_pct={best_tol:.4g}")

        return (pd.DataFrame(), {}, last_reason or "empty_slice")

    fig_w = max(5.5, 3.2 * n_params)
    fig_h = max(5.0, 3.0 * n_params)
    fig, axes = plt.subplots(n_params, n_params, figsize=(fig_w, fig_h), squeeze=False)

    plotted_lower_any = False
    for i, y_param in enumerate(matrix_params):
        for j, x_param in enumerate(matrix_params):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            if i == j:
                diag_df, fixed_vals, reason = _slice_with_fixed({x_param})
                diag_df[x_param] = pd.to_numeric(diag_df.get(x_param), errors="coerce")
                diag_df["distance_value"] = pd.to_numeric(diag_df.get("distance_value"), errors="coerce")
                diag_df = diag_df.dropna(subset=[x_param, "distance_value"])
                if not diag_df.empty:
                    curve = (
                        diag_df.groupby(x_param, as_index=False, sort=True)["distance_value"]
                        .min()
                    )
                else:
                    curve = pd.DataFrame(columns=[x_param, "distance_value"])

                if len(curve) >= 1:
                    xv = curve[x_param].to_numpy(dtype=float)
                    dv = curve["distance_value"].to_numpy(dtype=float)
                    if len(curve) >= 2:
                        ax.plot(xv, dv, color="#4C78A8", linewidth=1.5)
                    ax.scatter(xv, dv, s=14, color="#4C78A8", alpha=0.9)
                    true_x = float(pd.to_numeric(pd.Series([row.get(f"true_{x_param}")]), errors="coerce").iloc[0])
                    est_x = float(pd.to_numeric(pd.Series([row.get(f"est_{x_param}")]), errors="coerce").iloc[0])
                    if np.isfinite(true_x):
                        ax.axvline(true_x, color="#E45756", linestyle="--", linewidth=1.0)
                    if np.isfinite(est_x):
                        ax.axvline(est_x, color="#F58518", linestyle="-.", linewidth=1.0)
                    if reason in {"single_point_slice"} or (isinstance(reason, str) and reason.startswith("expanded_tol_pct=")):
                        ax.text(
                            0.03,
                            0.97,
                            reason,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=6.3,
                            color="#555555",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", alpha=0.8),
                        )
                else:
                    message = "N/A"
                    if reason is not None:
                        message = f"N/A\n({reason})"
                    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=7, transform=ax.transAxes)

                ax.set_xlim(*param_limits[x_param])
                ax.set_ylim(*dist_limits)
                ax.set_title(f"{x_param}", fontsize=8)
            else:
                # Lower-triangle cells vary this pair while fixing all other
                # parameters with the same tolerance policy used on diagonals.
                pair_slice_df, _, pair_reason = _slice_with_fixed({x_param, y_param})
                pair_df = (
                    pair_slice_df[[x_param, y_param, "distance_value"]].copy()
                    if not pair_slice_df.empty
                    else pd.DataFrame(columns=[x_param, y_param, "distance_value"])
                )
                pair_df[x_param] = pd.to_numeric(pair_df[x_param], errors="coerce")
                pair_df[y_param] = pd.to_numeric(pair_df[y_param], errors="coerce")
                pair_df["distance_value"] = pd.to_numeric(pair_df["distance_value"], errors="coerce")
                pair_df = pair_df.dropna(subset=[x_param, y_param, "distance_value"])
                if not pair_df.empty:
                    pair_df = (
                        pair_df.groupby([x_param, y_param], as_index=False, sort=True)["distance_value"]
                        .min()
                    )

                plotted = False
                if len(pair_df) >= 3:
                    x = pair_df[x_param].to_numpy(dtype=float)
                    y = pair_df[y_param].to_numpy(dtype=float)
                    z = pair_df["distance_value"].to_numpy(dtype=float)
                    try:
                        tri = mtri.Triangulation(x, y)
                        ax.tricontourf(
                            tri,
                            z,
                            levels=dist_levels,
                            cmap=dist_cmap,
                            norm=dist_norm,
                            alpha=0.96,
                        )
                        ax.tricontour(
                            tri,
                            z,
                            levels=dist_levels[1:-1:2],
                            colors="white",
                            linewidths=0.22,
                            alpha=0.20,
                        )
                        plotted = True
                    except Exception:
                        plotted = False
                elif len(pair_df) > 0:
                    x = pair_df[x_param].to_numpy(dtype=float)
                    y = pair_df[y_param].to_numpy(dtype=float)
                    z = pair_df["distance_value"].to_numpy(dtype=float)
                    ax.scatter(
                        x,
                        y,
                        c=z,
                        s=18,
                        cmap=dist_cmap,
                        norm=dist_norm,
                        alpha=0.95,
                        linewidths=0.0,
                        zorder=2,
                    )
                    plotted = True

                if plotted:
                    plotted_lower_any = True
                    true_x = float(pd.to_numeric(pd.Series([row.get(f"true_{x_param}")]), errors="coerce").iloc[0])
                    true_y = float(pd.to_numeric(pd.Series([row.get(f"true_{y_param}")]), errors="coerce").iloc[0])
                    est_x = float(pd.to_numeric(pd.Series([row.get(f"est_{x_param}")]), errors="coerce").iloc[0])
                    est_y = float(pd.to_numeric(pd.Series([row.get(f"est_{y_param}")]), errors="coerce").iloc[0])
                    if np.isfinite(true_x) and np.isfinite(true_y):
                        ax.scatter([true_x], [true_y], s=46, marker="*", color="#E45756", edgecolors="black", linewidths=0.45, zorder=4)
                    if np.isfinite(est_x) and np.isfinite(est_y):
                        ax.scatter([est_x], [est_y], s=40, marker="X", color="#F58518", edgecolors="black", linewidths=0.45, zorder=4)
                    if isinstance(pair_reason, str) and pair_reason.startswith("expanded_tol_pct="):
                        ax.text(
                            0.03,
                            0.97,
                            pair_reason,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=6.3,
                            color="#555555",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", alpha=0.8),
                        )
                else:
                    message = "N/A"
                    if pair_reason is not None:
                        message = f"N/A\n({pair_reason})"
                    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=7, transform=ax.transAxes)

                ax.set_xlim(*param_limits[x_param])
                ax.set_ylim(*param_limits[y_param])

            if i < n_params - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(_axis_label_for_param(x_param))
            if j > 0:
                ax.set_yticklabels([])
            else:
                if i == j:
                    ax.set_ylabel(f"{metric_label} distance")
                else:
                    ax.set_ylabel(_axis_label_for_param(y_param))

    fig.suptitle(
        "Random showcase distance matrix\n"
        f"(dataset_index={ds_idx}, metric={distance_metric}, seed={showcase_seed}, fixed_tol={fixed_tolerance_pct:.4g}%)",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.88 if plotted_lower_any else 0.97,
        bottom=0.06,
        top=0.93,
        wspace=0.08,
        hspace=0.08,
    )
    if plotted_lower_any:
        sm = plt.cm.ScalarMappable(norm=dist_norm, cmap=dist_cmap)
        sm.set_array([])
        cax = fig.add_axes([0.905, 0.10, 0.018, 0.78])
        tick_step = max(1, int(np.ceil(n_color_levels / 6)))
        tick_values = dist_levels[::tick_step]
        if not np.isclose(tick_values[-1], dist_levels[-1]):
            tick_values = np.append(tick_values, dist_levels[-1])
        cbar = fig.colorbar(
            sm,
            cax=cax,
            boundaries=dist_levels,
            ticks=tick_values,
            spacing="proportional",
        )
        cbar.set_label(f"{metric_label} distance")

    _save_figure(fig, PLOTS_DIR / "random_showcase_distance_matrix.png")
    plt.close(fig)


def _make_plots(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    plot_params=None,
    dict_path: Path | None = None,
    cfg_21: dict | None = None,
) -> None:
    """Quick diagnostic plots for the estimation step."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    metric = str((cfg_21 or {}).get("distance_metric", "unknown"))
    k_cfg = (cfg_21 or {}).get("interpolation_k", None)
    k_label = "all" if k_cfg is None else str(k_cfg)

    # ── 1. Distance diagnostics (distribution + method relevance) ───
    distances = pd.to_numeric(result_df.get("best_distance"), errors="coerce").dropna()
    if not distances.empty:
        q1 = float(distances.quantile(0.25))
        q3 = float(distances.quantile(0.75))
        iqr = q3 - q1
        upper_fence = float(q3 + 1.5 * iqr) if np.isfinite(iqr) else float(distances.max())
        inlier_mask = distances <= upper_fence
        inliers = distances[inlier_mask]
        n_outliers = int((~inlier_mask).sum())

        q50 = float(distances.quantile(0.50))
        q90 = float(distances.quantile(0.90))
        q95 = float(distances.quantile(0.95))

        fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_cdf = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 0])
        ax_oper = fig.add_subplot(gs[1, 1])

        # 1A) Core histogram (IQR-clipped) with robust quantiles
        hist_values = inliers if len(inliers) >= 5 else distances
        ax_hist.hist(hist_values, bins=45, color="#4C78A8", alpha=0.82, edgecolor="white")
        ax_hist.axvline(q50, color="#E45756", linestyle="--", linewidth=1.6, label=f"p50 = {q50:.4g}")
        ax_hist.axvline(q90, color="#F58518", linestyle="-.", linewidth=1.4, label=f"p90 = {q90:.4g}")
        ax_hist.axvline(q95, color="#72B7B2", linestyle=":", linewidth=1.6, label=f"p95 = {q95:.4g}")
        if n_outliers:
            ax_hist.axvline(
                upper_fence, color="#B279A2", linestyle="-", linewidth=1.1,
                label=f"IQR upper fence = {upper_fence:.4g}",
            )
        ax_hist.set_xlabel("Best distance")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Core distance density (IQR-clipped)")
        ax_hist.legend(fontsize=7.5, loc="upper right")
        ax_hist.text(
            0.02,
            0.98,
            (
                f"N={len(distances)} | outliers={n_outliers} "
                f"({(100.0 * n_outliers / len(distances)):.1f}%)\n"
                f"median={q50:.3g}, p90={q90:.3g}, p95={q95:.3g}"
            ),
            transform=ax_hist.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
        )

        # 1B) Coverage view: fraction of rows below distance threshold
        d_sorted = np.sort(distances.to_numpy(dtype=float))
        cdf_y = np.arange(1, len(d_sorted) + 1, dtype=float) / len(d_sorted)
        ax_cdf.plot(d_sorted, cdf_y, color="#54A24B", linewidth=1.8)
        for value, color, label in [
            (q50, "#E45756", "p50"),
            (q90, "#F58518", "p90"),
            (q95, "#72B7B2", "p95"),
        ]:
            ax_cdf.axvline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.8, label=label)
        ax_cdf.set_xlabel("Best distance threshold")
        ax_cdf.set_ylabel("Fraction with best_distance <= threshold")
        ax_cdf.set_ylim(0.0, 1.02)
        ax_cdf.set_title("Coverage curve (all rows)")
        ax_cdf.legend(fontsize=7.5, loc="lower right")
        # Keep the CDF readable when very large outliers exist.
        cdf_xmax = float(distances.quantile(0.995))
        if np.isfinite(cdf_xmax) and cdf_xmax > 0 and distances.max() > 1.15 * cdf_xmax:
            ax_cdf.set_xlim(0.0, cdf_xmax)
            ax_cdf.text(
                0.02,
                0.03,
                f"Zoomed to 99.5% (max={float(distances.max()):.3g})",
                transform=ax_cdf.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="0.35",
            )

        # 1C) Distance-vs-error calibration: does distance track inference quality?
        selected_params: list[str] = []
        if isinstance(plot_params, (list, tuple, set)):
            selected_params = [
                str(p) for p in plot_params
                if f"true_{p}" in result_df.columns and f"est_{p}" in result_df.columns
            ]
        if not selected_params:
            for pname in ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4", "cos_n"]:
                if f"true_{pname}" in result_df.columns and f"est_{pname}" in result_df.columns:
                    selected_params.append(pname)

        relerr_cols = []
        for pname in selected_params:
            t = pd.to_numeric(result_df[f"true_{pname}"], errors="coerce")
            e = pd.to_numeric(result_df[f"est_{pname}"], errors="coerce")
            denom = np.maximum(np.abs(t), 1e-9)
            relerr_cols.append((((e - t).abs() / denom) * 100.0).rename(pname))

        if relerr_cols:
            relerr_df = pd.concat(relerr_cols, axis=1)
            row_relerr = relerr_df.median(axis=1, skipna=True)
            eval_df = pd.DataFrame({
                "distance": pd.to_numeric(result_df["best_distance"], errors="coerce"),
                "agg_relerr_pct": row_relerr,
            }).dropna()
        else:
            eval_df = pd.DataFrame(columns=["distance", "agg_relerr_pct"])

        if len(eval_df) >= 3:
            ax_err.scatter(
                eval_df["distance"], eval_df["agg_relerr_pct"],
                s=15, alpha=0.35, color="#72B7B2", edgecolors="none",
            )
            if len(eval_df) >= 20 and eval_df["distance"].nunique() >= 6:
                q_edges = np.unique(np.quantile(eval_df["distance"], np.linspace(0.0, 1.0, 9)))
                if len(q_edges) >= 3:
                    dist_bins = pd.cut(eval_df["distance"], bins=q_edges, include_lowest=True, duplicates="drop")
                    trend = (
                        eval_df.assign(dist_bin=dist_bins)
                        .groupby("dist_bin", observed=True)
                        .agg(
                            distance_mid=("distance", "median"),
                            relerr_median=("agg_relerr_pct", "median"),
                        )
                        .dropna()
                    )
                    if not trend.empty:
                        ax_err.plot(
                            trend["distance_mid"], trend["relerr_median"],
                            color="#E45756", linewidth=1.7, marker="o",
                            label="Median |rel.err| across distance quantiles",
                        )

            pearson = float(eval_df["distance"].corr(eval_df["agg_relerr_pct"], method="pearson"))
            spearman = float(eval_df["distance"].corr(eval_df["agg_relerr_pct"], method="spearman"))
            ptxt = f"{pearson:.2f}" if np.isfinite(pearson) else "nan"
            stxt = f"{spearman:.2f}" if np.isfinite(spearman) else "nan"
            shown_params = ", ".join(selected_params[:3]) + ("..." if len(selected_params) > 3 else "")
            ax_err.set_title("Distance vs estimation error")
            ax_err.set_xlabel("Best distance")
            ax_err.set_ylabel("Median |relative error| [%]")
            ax_err.text(
                0.02,
                0.98,
                f"Params: {shown_params}\nPearson={ptxt}, Spearman={stxt}",
                transform=ax_err.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
            )
            y_hi = float(eval_df["agg_relerr_pct"].quantile(0.99))
            if np.isfinite(y_hi) and y_hi > 0:
                ax_err.set_ylim(0.0, max(0.5, 1.15 * y_hi))
            x_hi = float(eval_df["distance"].quantile(0.99))
            if np.isfinite(x_hi) and x_hi > 0 and eval_df["distance"].max() > 1.15 * x_hi:
                ax_err.set_xlim(0.0, x_hi)
                ax_err.text(
                    0.98,
                    0.03,
                    f"Zoomed to p99 (max={float(eval_df['distance'].max()):.3g})",
                    transform=ax_err.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )
            if ax_err.get_legend_handles_labels()[0]:
                ax_err.legend(fontsize=7.5, loc="upper right")
        else:
            ax_err.text(
                0.5, 0.5,
                "Not enough true/estimated\nparameter overlap for\nerror-calibration panel",
                transform=ax_err.transAxes,
                ha="center", va="center", fontsize=9,
            )
            ax_err.set_title("Distance vs estimation error")
            ax_err.set_xlabel("Best distance")
            ax_err.set_ylabel("Median |relative error| [%]")

        # 1D) Distance operating curve: threshold trade-off for coverage vs quality
        if len(eval_df) >= 20 and eval_df["distance"].nunique() >= 8:
            thr = np.unique(np.quantile(eval_df["distance"], np.linspace(0.05, 0.95, 15)))
            if len(thr) >= 3:
                coverage_pct = []
                med_relerr = []
                p90_relerr = []
                n_eval = float(len(eval_df))
                for tval in thr:
                    subset = eval_df[eval_df["distance"] <= tval]
                    if subset.empty:
                        coverage_pct.append(np.nan)
                        med_relerr.append(np.nan)
                        p90_relerr.append(np.nan)
                    else:
                        coverage_pct.append(100.0 * len(subset) / n_eval)
                        med_relerr.append(float(subset["agg_relerr_pct"].median()))
                        p90_relerr.append(float(subset["agg_relerr_pct"].quantile(0.90)))

                ax_oper.plot(
                    thr, coverage_pct, color="#54A24B", linewidth=1.8, marker="o",
                    markersize=3.2, label="Coverage retained [%]",
                )
                ax_oper.set_xlabel("Distance threshold")
                ax_oper.set_ylabel("Coverage retained [%]", color="#2F6B2D")
                ax_oper.tick_params(axis="y", labelcolor="#2F6B2D")
                ax_oper.set_ylim(0.0, 101.0)

                ax_err2 = ax_oper.twinx()
                ax_err2.plot(
                    thr, med_relerr, color="#E45756", linewidth=1.6, marker="s",
                    markersize=3.0, label="Median |rel.err| [%]",
                )
                ax_err2.plot(
                    thr, p90_relerr, color="#F58518", linewidth=1.3, linestyle="--",
                    label="p90 |rel.err| [%]",
                )
                ax_err2.set_ylabel("Error among retained rows [%]", color="#A94D00")
                ax_err2.tick_params(axis="y", labelcolor="#A94D00")

                star_thr = q90
                star_subset = eval_df[eval_df["distance"] <= star_thr]
                if len(star_subset) > 0:
                    star_cov = 100.0 * len(star_subset) / len(eval_df)
                    star_med = float(star_subset["agg_relerr_pct"].median())
                    ax_oper.axvline(star_thr, color="0.45", linestyle=":", linewidth=1.0)
                    ax_oper.text(
                        0.02,
                        0.98,
                        (
                            f"At p90 threshold ({star_thr:.3g}):\n"
                            f"coverage={star_cov:.1f}%, median err={star_med:.2f}%"
                        ),
                        transform=ax_oper.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
                    )

                ax_oper.set_title("Operating curve: threshold trade-off")
                h1, l1 = ax_oper.get_legend_handles_labels()
                h2, l2 = ax_err2.get_legend_handles_labels()
                ax_oper.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
            else:
                ax_oper.text(
                    0.5, 0.5, "Insufficient distance spread\nfor operating-curve panel",
                    transform=ax_oper.transAxes, ha="center", va="center", fontsize=9,
                )
                ax_oper.set_title("Operating curve: threshold trade-off")
                ax_oper.set_xlabel("Distance threshold")
                ax_oper.set_ylabel("Coverage retained [%]")
        else:
            ax_oper.text(
                0.5, 0.5, "Not enough rows with error estimates\nfor operating-curve panel",
                transform=ax_oper.transAxes, ha="center", va="center", fontsize=9,
            )
            ax_oper.set_title("Operating curve: threshold trade-off")
            ax_oper.set_xlabel("Distance threshold")
            ax_oper.set_ylabel("Coverage retained [%]")

        fig.suptitle(
            f"Best-match distance diagnostics (metric={metric}, IDW K={k_label})",
            fontsize=11,
        )
        _save_figure(fig, PLOTS_DIR / "distance_distribution.png")
        plt.close(fig)

    # ── 2. True vs estimated scatter for available params ────────────
    # Build all possible pairs, then filter by plot_parameters if set
    all_param_pairs = []
    for col in result_df.columns:
        if col.startswith("est_"):
            pname = col[4:]  # strip "est_"
            true_col = f"true_{pname}"
            if true_col in result_df.columns:
                all_param_pairs.append((true_col, col, pname))
    if plot_params:
        all_param_pairs = [(t, e, l) for t, e, l in all_param_pairs
                           if l in plot_params]
    valid_pairs = all_param_pairs

    if valid_pairs:
        n_p = len(valid_pairs)
        fig, axes = plt.subplots(1, n_p, figsize=(5 * n_p, 5))
        if n_p == 1:
            axes = [axes]
        for ax, (true_col, est_col, label) in zip(axes, valid_pairs):
            t = pd.to_numeric(result_df[true_col], errors="coerce")
            e = pd.to_numeric(result_df[est_col], errors="coerce")
            m = t.notna() & e.notna()
            if m.sum() > 0:
                ax.scatter(t[m], e[m], s=12, alpha=0.5, color="#F58518")
                lo = min(t[m].min(), e[m].min())
                hi = max(t[m].max(), e[m].max())
                pad = 0.02 * (hi - lo) if hi > lo else 0.01
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        "k--", linewidth=1)
            ax.set_xlabel(f"True {label}")
            ax.set_ylabel(f"Estimated {label}")
            ax.set_title(f"True vs Est: {label}")
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"Parameter estimation: true vs estimated (metric={metric})", fontsize=11, y=0.98)
        # Leave extra room under the suptitle so subplot titles don't collide
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        _save_figure(fig, PLOTS_DIR / "true_vs_estimated.png")
        plt.close(fig)

    # ── 3. Random showcase matrix: lower-triangle 2D maps + diagonal projections ──
    if dict_path is not None:
        _make_random_showcase_distance_matrix(
            result_df=result_df,
            data_df=data_df,
            dict_path=dict_path,
            cfg_21=cfg_21 or {},
            plot_params=plot_params,
        )



if __name__ == "__main__":
    raise SystemExit(main())
