#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from MASTER.ANCILLARY.SIMULATION_TUNING.common import (
    date_range_mask,
    group_label,
    load_tuning_config,
    resolve_selection,
)
from MASTER.common.file_selection import extract_run_datetime_from_name


PLANE_ORDER = (1, 2, 3, 4)
AXIS_ORDER = ("x", "y", "theta")
GROUP_COLORS = {
    "SIM": "#d95f02",
    "REAL": "#1b9e77",
}
STATUS_COLORS = {
    "ok": "#2ca02c",
    "watch": "#ff7f0e",
    "retune": "#d62728",
    "excluded_sparse": "#7f7f7f",
    "no_data": "#bdbdbd",
}
AXIS_LABELS = {
    "x": "X [mm]",
    "y": "Y [mm]",
    "theta": "theta [deg]",
}
EFFICIENCY_VECTOR_PATTERN = re.compile(
    r"^efficiency_vector_p(?P<plane>[1-4])_(?P<axis>x|y|theta)_bin_(?P<bin>\d{3})_eff$"
)


@dataclass(frozen=True)
class EfficiencyVectorTuningConfig:
    min_valid_files_per_bin: int
    low_efficiency_threshold: float
    warning_abs_difference: float
    severe_abs_difference: float
    low_eff_warning_abs_difference: float
    low_eff_severe_abs_difference: float
    xy_edge_bin_fraction: float
    theta_high_bin_fraction: float
    baseline_x_abs_max_mm: float
    baseline_y_abs_max_mm: float
    baseline_theta_max_deg: float
    outside_shape_warning_rmse: float
    outside_shape_severe_rmse: float
    outside_shape_warning_max_abs: float
    outside_shape_severe_max_abs: float
    dictionary_match_top_k: int
    dictionary_match_weight_power: float
    dictionary_match_weight_epsilon: float
    dictionary_match_min_fiducial_bins: int
    dictionary_match_min_outside_bins: int


def output_dir() -> Path:
    return Path(__file__).resolve().parent / "OUTPUTS"


def load_efficiency_vector_tuning_config(config: dict) -> EfficiencyVectorTuningConfig:
    study_cfg = config.get("efficiency_vector_tuning", {})
    return EfficiencyVectorTuningConfig(
        min_valid_files_per_bin=max(1, int(study_cfg.get("min_valid_files_per_bin", 10))),
        low_efficiency_threshold=float(study_cfg.get("low_efficiency_threshold", 0.80)),
        warning_abs_difference=float(study_cfg.get("warning_abs_difference", 0.08)),
        severe_abs_difference=float(study_cfg.get("severe_abs_difference", 0.12)),
        low_eff_warning_abs_difference=float(
            study_cfg.get("low_eff_warning_abs_difference", 0.10)
        ),
        low_eff_severe_abs_difference=float(
            study_cfg.get("low_eff_severe_abs_difference", 0.18)
        ),
        xy_edge_bin_fraction=float(study_cfg.get("xy_edge_bin_fraction", 0.20)),
        theta_high_bin_fraction=float(study_cfg.get("theta_high_bin_fraction", 0.25)),
        baseline_x_abs_max_mm=float(study_cfg.get("baseline_x_abs_max_mm", 100.0)),
        baseline_y_abs_max_mm=float(study_cfg.get("baseline_y_abs_max_mm", 100.0)),
        baseline_theta_max_deg=float(study_cfg.get("baseline_theta_max_deg", 30.0)),
        outside_shape_warning_rmse=float(
            study_cfg.get("outside_shape_warning_rmse", 0.05)
        ),
        outside_shape_severe_rmse=float(
            study_cfg.get("outside_shape_severe_rmse", 0.08)
        ),
        outside_shape_warning_max_abs=float(
            study_cfg.get("outside_shape_warning_max_abs", 0.08)
        ),
        outside_shape_severe_max_abs=float(
            study_cfg.get("outside_shape_severe_max_abs", 0.12)
        ),
        dictionary_match_top_k=max(1, int(study_cfg.get("dictionary_match_top_k", 5))),
        dictionary_match_weight_power=float(
            study_cfg.get("dictionary_match_weight_power", 2.0)
        ),
        dictionary_match_weight_epsilon=float(
            study_cfg.get("dictionary_match_weight_epsilon", 1e-4)
        ),
        dictionary_match_min_fiducial_bins=max(
            2,
            int(study_cfg.get("dictionary_match_min_fiducial_bins", 4)),
        ),
        dictionary_match_min_outside_bins=max(
            1,
            int(study_cfg.get("dictionary_match_min_outside_bins", 2)),
        ),
    )


def _parse_exec_ts(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series.astype("string"),
        format="%Y-%m-%d_%H.%M.%S",
        errors="coerce",
    )
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series[missing], errors="coerce")
    return parsed


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def station_task4_efficiency_path(station_label: str) -> Path:
    return (
        ROOT_DIR
        / "STATIONS"
        / station_label
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_4"
        / "METADATA"
        / "task_4_metadata_efficiency.csv"
    )


def load_task4_efficiency_metadata(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = lambda col: (
        col in {"filename_base", "execution_timestamp"}
        or str(col).startswith("efficiency_metadata_")
        or str(col).startswith("efficiency_vector_")
    )
    for station_label in station_labels:
        csv_path = station_task4_efficiency_path(station_label)
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
        if frame.empty or "filename_base" not in frame.columns:
            continue
        frame = frame.copy()
        frame["station_label"] = station_label
        frame["_exec_ts"] = _parse_exec_ts(frame["execution_timestamp"])
        frame.sort_values(["filename_base", "_exec_ts"], inplace=True)
        frame = frame.drop_duplicates(subset=["filename_base"], keep="last")
        frame["datetime"] = frame["filename_base"].map(extract_run_datetime_from_name)
        if date_ranges:
            mask = date_range_mask(frame["datetime"], date_ranges)
            frame = frame.loc[mask].copy()
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _discover_vector_bins(frame: pd.DataFrame) -> dict[tuple[int, str], list[int]]:
    found: dict[tuple[int, str], set[int]] = {}
    for column in frame.columns:
        match = EFFICIENCY_VECTOR_PATTERN.match(str(column))
        if match is None:
            continue
        key = (int(match.group("plane")), match.group("axis"))
        found.setdefault(key, set()).add(int(match.group("bin")))
    return {key: sorted(indices) for key, indices in found.items()}


def build_group_bin_summary(
    frame: pd.DataFrame,
    *,
    group_kind: str,
    group_name: str,
    cfg: EfficiencyVectorTuningConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    available_mask = (
        _coerce_bool(frame["efficiency_metadata_available"])
        if "efficiency_metadata_available" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    usable = frame.loc[available_mask].copy()
    if usable.empty:
        return pd.DataFrame()

    vector_bins = _discover_vector_bins(usable)
    rows: list[dict[str, object]] = []
    for plane in PLANE_ORDER:
        for axis in AXIS_ORDER:
            bin_indices = vector_bins.get((plane, axis), [])
            center_suffix = "deg" if axis == "theta" else "mm"
            for bin_index in bin_indices:
                center_col = (
                    f"efficiency_vector_p{plane}_{axis}_bin_{bin_index:03d}_center_{center_suffix}"
                )
                eff_col = f"efficiency_vector_p{plane}_{axis}_bin_{bin_index:03d}_eff"
                unc_col = f"efficiency_vector_p{plane}_{axis}_bin_{bin_index:03d}_unc"
                if eff_col not in usable.columns or center_col not in usable.columns:
                    continue
                center_values = pd.to_numeric(usable[center_col], errors="coerce")
                eff_values = pd.to_numeric(usable[eff_col], errors="coerce")
                unc_values = (
                    pd.to_numeric(usable[unc_col], errors="coerce")
                    if unc_col in usable.columns
                    else pd.Series(np.nan, index=usable.index)
                )
                valid_mask = np.isfinite(center_values) & np.isfinite(eff_values)
                n_valid = int(np.count_nonzero(valid_mask))
                if n_valid <= 0:
                    continue
                eff_valid = eff_values.loc[valid_mask].to_numpy(dtype=float)
                unc_valid = unc_values.loc[valid_mask].to_numpy(dtype=float)
                center_valid = center_values.loc[valid_mask].to_numpy(dtype=float)
                rows.append(
                    {
                        "group_kind": group_kind,
                        "group_name": group_name,
                        "plane": plane,
                        "axis": axis,
                        "bin_index": bin_index,
                        "center_value": float(np.nanmedian(center_valid)),
                        "center_unit": center_suffix,
                        "n_valid_files": n_valid,
                        "median_eff": float(np.nanmedian(eff_valid)),
                        "p25_eff": float(np.nanquantile(eff_valid, 0.25)),
                        "p75_eff": float(np.nanquantile(eff_valid, 0.75)),
                        "median_unc": (
                            float(np.nanmedian(unc_valid[np.isfinite(unc_valid)]))
                            if np.isfinite(unc_valid).any()
                            else np.nan
                        ),
                        "included_for_comparison": n_valid >= int(cfg.min_valid_files_per_bin),
                    }
                )
    return pd.DataFrame(rows)


def build_file_curve_payload(
    frame: pd.DataFrame,
) -> dict[tuple[int, str], dict[str, object]]:
    if frame.empty:
        return {}
    available_mask = (
        _coerce_bool(frame["efficiency_metadata_available"])
        if "efficiency_metadata_available" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    usable = frame.loc[available_mask].copy()
    if usable.empty:
        return {}
    usable = usable.reset_index(drop=True)
    meta = usable.loc[
        :,
        [col for col in ("filename_base", "datetime", "execution_timestamp", "station_label") if col in usable.columns],
    ].copy()
    vector_bins = _discover_vector_bins(usable)
    payload: dict[tuple[int, str], dict[str, object]] = {}
    for plane in PLANE_ORDER:
        for axis in AXIS_ORDER:
            bin_indices = vector_bins.get((plane, axis), [])
            if not bin_indices:
                continue
            center_suffix = "deg" if axis == "theta" else "mm"
            eff_cols = []
            center_cols = []
            for bin_index in bin_indices:
                eff_col = f"efficiency_vector_p{plane}_{axis}_bin_{bin_index:03d}_eff"
                center_col = (
                    f"efficiency_vector_p{plane}_{axis}_bin_{bin_index:03d}_center_{center_suffix}"
                )
                if eff_col in usable.columns and center_col in usable.columns:
                    eff_cols.append(eff_col)
                    center_cols.append(center_col)
            if not eff_cols:
                continue
            eff_matrix = (
                usable.loc[:, eff_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            center_matrix = (
                usable.loc[:, center_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            centers = np.nanmedian(center_matrix, axis=0)
            payload[(plane, axis)] = {
                "meta": meta.copy(),
                "centers": centers,
                "eff_matrix": eff_matrix,
                "center_unit": center_suffix,
                "bin_indices": np.asarray(bin_indices, dtype=int),
            }
    return payload


def _edge_mask(size: int, axis_name: str, cfg: EfficiencyVectorTuningConfig) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=bool)
    if axis_name in {"x", "y"}:
        edge_count = max(1, int(math.ceil(cfg.xy_edge_bin_fraction * size)))
        mask = np.zeros(size, dtype=bool)
        mask[:edge_count] = True
        mask[-edge_count:] = True
        return mask
    edge_count = max(1, int(math.ceil(cfg.theta_high_bin_fraction * size)))
    mask = np.zeros(size, dtype=bool)
    mask[-edge_count:] = True
    return mask


def _fiducial_mask(
    center_values: np.ndarray,
    axis_name: str,
    cfg: EfficiencyVectorTuningConfig,
) -> np.ndarray:
    if axis_name == "x":
        return np.abs(center_values) <= float(cfg.baseline_x_abs_max_mm) + 1e-9
    if axis_name == "y":
        return np.abs(center_values) <= float(cfg.baseline_y_abs_max_mm) + 1e-9
    return center_values <= float(cfg.baseline_theta_max_deg) + 1e-9


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def _classify_outside_shape(
    outside_rmse: float,
    outside_max_abs: float,
    cfg: EfficiencyVectorTuningConfig,
) -> str:
    if not np.isfinite(outside_rmse) and not np.isfinite(outside_max_abs):
        return "no_data"
    if (
        (np.isfinite(outside_rmse) and outside_rmse >= float(cfg.outside_shape_severe_rmse))
        or (
            np.isfinite(outside_max_abs)
            and outside_max_abs >= float(cfg.outside_shape_severe_max_abs)
        )
    ):
        return "retune"
    if (
        (np.isfinite(outside_rmse) and outside_rmse >= float(cfg.outside_shape_warning_rmse))
        or (
            np.isfinite(outside_max_abs)
            and outside_max_abs >= float(cfg.outside_shape_warning_max_abs)
        )
    ):
        return "watch"
    return "ok"


def _fit_baseline_prediction(
    sim_values: np.ndarray,
    real_values: np.ndarray,
    fiducial_mask: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray | None, dict[str, float]]:
    sim_fid = sim_values[fiducial_mask]
    real_fid = real_values[fiducial_mask]
    if sim_fid.size == 0 or real_fid.size == 0:
        return None, {"scale": np.nan, "offset": np.nan}

    scale = np.nan
    offset = np.nan
    if model_name == "shift":
        scale = 1.0
        offset = float(np.nanmean(real_fid - sim_fid))
    elif model_name == "scale":
        denom = float(np.dot(sim_fid, sim_fid))
        if denom <= 0:
            return None, {"scale": np.nan, "offset": np.nan}
        scale = float(np.dot(sim_fid, real_fid) / denom)
        offset = 0.0
    elif model_name == "affine":
        if sim_fid.size < 2:
            return None, {"scale": np.nan, "offset": np.nan}
        design = np.column_stack([sim_fid, np.ones_like(sim_fid)])
        try:
            solution, *_ = np.linalg.lstsq(design, real_fid, rcond=None)
        except np.linalg.LinAlgError:
            return None, {"scale": np.nan, "offset": np.nan}
        scale = float(solution[0])
        offset = float(solution[1])
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    if not np.isfinite(scale) or not np.isfinite(offset):
        return None, {"scale": np.nan, "offset": np.nan}

    predicted = np.clip(scale * sim_values + offset, 0.0, 1.0)
    return predicted, {"scale": scale, "offset": offset}


def build_real_vs_sim_bin_summary(
    sim_summary: pd.DataFrame,
    real_summary: pd.DataFrame,
    cfg: EfficiencyVectorTuningConfig,
) -> pd.DataFrame:
    if sim_summary.empty or real_summary.empty:
        return pd.DataFrame()

    merge_keys = ["plane", "axis", "bin_index"]
    sim_cols = {
        "center_value": "center_value",
        "center_unit": "center_unit",
        "n_valid_files": "sim_n_valid_files",
        "median_eff": "sim_median_eff",
        "p25_eff": "sim_p25_eff",
        "p75_eff": "sim_p75_eff",
        "median_unc": "sim_median_unc",
        "included_for_comparison": "sim_included_for_comparison",
    }
    real_cols = {
        "n_valid_files": "real_n_valid_files",
        "median_eff": "real_median_eff",
        "p25_eff": "real_p25_eff",
        "p75_eff": "real_p75_eff",
        "median_unc": "real_median_unc",
        "included_for_comparison": "real_included_for_comparison",
    }
    merged = sim_summary.loc[:, merge_keys + list(sim_cols)].rename(columns=sim_cols).merge(
        real_summary.loc[:, merge_keys + list(real_cols)].rename(columns=real_cols),
        on=merge_keys,
        how="inner",
    )
    if merged.empty:
        return merged

    merged = merged.copy()
    merged["eligible_for_comparison"] = (
        merged["sim_included_for_comparison"].fillna(False)
        & merged["real_included_for_comparison"].fillna(False)
        & np.isfinite(merged["sim_median_eff"])
        & np.isfinite(merged["real_median_eff"])
    )
    merged["real_minus_sim_eff"] = merged["real_median_eff"] - merged["sim_median_eff"]
    merged["abs_eff_difference"] = merged["real_minus_sim_eff"].abs()
    merged["min_group_eff"] = merged[["sim_median_eff", "real_median_eff"]].min(axis=1)
    merged["low_efficiency_bin"] = merged["min_group_eff"] < float(cfg.low_efficiency_threshold)

    status = np.full(len(merged), "ok", dtype=object)
    sparse_mask = ~merged["eligible_for_comparison"]
    status[sparse_mask.to_numpy()] = "excluded_sparse"
    eligible = merged["eligible_for_comparison"].to_numpy()
    severe_mask = eligible & (
        (merged["abs_eff_difference"].to_numpy(dtype=float) >= float(cfg.severe_abs_difference))
        | (
            merged["low_efficiency_bin"].to_numpy(dtype=bool)
            & (
                merged["abs_eff_difference"].to_numpy(dtype=float)
                >= float(cfg.low_eff_severe_abs_difference)
            )
        )
    )
    warning_mask = eligible & ~severe_mask & (
        (merged["abs_eff_difference"].to_numpy(dtype=float) >= float(cfg.warning_abs_difference))
        | (
            merged["low_efficiency_bin"].to_numpy(dtype=bool)
            & (
                merged["abs_eff_difference"].to_numpy(dtype=float)
                >= float(cfg.low_eff_warning_abs_difference)
            )
        )
    )
    status[warning_mask] = "watch"
    status[severe_mask] = "retune"
    merged["bin_status"] = status

    merged["edge_bin"] = False
    for (_plane, axis_name), subset in merged.groupby(["plane", "axis"]):
        ordered_idx = subset.sort_values("bin_index").index.to_numpy()
        merged.loc[ordered_idx, "edge_bin"] = _edge_mask(len(ordered_idx), axis_name, cfg)
    return merged


def augment_with_baseline_models(
    compare_df: pd.DataFrame,
    cfg: EfficiencyVectorTuningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if compare_df.empty:
        return compare_df.copy(), pd.DataFrame()

    enhanced = compare_df.copy()
    for column_name, default_value in (
        ("fiducial_baseline_bin", False),
        ("raw_residual", np.nan),
        ("scale_baseline_pred_eff", np.nan),
        ("scale_baseline_residual", np.nan),
        ("best_baseline_pred_eff", np.nan),
        ("best_baseline_residual", np.nan),
        ("best_baseline_model", ""),
        ("global_baseline_pred_eff", np.nan),
        ("global_baseline_residual", np.nan),
        ("global_baseline_model", ""),
    ):
        enhanced[column_name] = default_value

    summary_rows: list[dict[str, object]] = []
    baseline_models = ("shift", "scale", "affine")
    for plane in PLANE_ORDER:
        for axis_name in AXIS_ORDER:
            subset = enhanced[
                (enhanced["plane"] == plane)
                & (enhanced["axis"] == axis_name)
                & (enhanced["eligible_for_comparison"])
            ].sort_values("bin_index")
            if subset.empty:
                summary_rows.append(
                    {
                        "plane": plane,
                        "axis": axis_name,
                        "n_bins_compared": 0,
                        "n_fiducial_bins": 0,
                        "n_outside_bins": 0,
                        "best_baseline_model": "none",
                        "best_baseline_scale": np.nan,
                        "best_baseline_offset": np.nan,
                        "best_baseline_fiducial_rmse": np.nan,
                        "best_baseline_outside_rmse": np.nan,
                        "best_baseline_outside_max_abs": np.nan,
                        "best_baseline_outside_mean_signed_residual": np.nan,
                        "best_baseline_status": "no_data",
                        "scale_baseline_scale": np.nan,
                        "scale_baseline_fiducial_rmse": np.nan,
                        "scale_baseline_outside_rmse": np.nan,
                        "scale_baseline_outside_max_abs": np.nan,
                        "scale_baseline_status": "no_data",
                        "raw_outside_rmse": np.nan,
                        "raw_outside_max_abs": np.nan,
                        "raw_outside_mean_signed_residual": np.nan,
                    }
                )
                continue

            centers = subset["center_value"].to_numpy(dtype=float)
            sim_values = subset["sim_median_eff"].to_numpy(dtype=float)
            real_values = subset["real_median_eff"].to_numpy(dtype=float)
            fid_mask = _fiducial_mask(centers, axis_name, cfg)
            outside_mask = ~fid_mask
            enhanced.loc[subset.index, "fiducial_baseline_bin"] = fid_mask
            raw_residual = real_values - sim_values
            enhanced.loc[subset.index, "raw_residual"] = raw_residual

            raw_outside = raw_residual[outside_mask]
            raw_outside_rmse = _rmse(raw_outside)
            raw_outside_max_abs = (
                float(np.nanmax(np.abs(raw_outside)))
                if raw_outside.size
                else np.nan
            )
            raw_outside_mean_signed = (
                float(np.nanmean(raw_outside))
                if raw_outside.size
                else np.nan
            )

            model_metrics: dict[str, dict[str, float | np.ndarray | str]] = {}
            for model_name in baseline_models:
                predicted, params = _fit_baseline_prediction(
                    sim_values,
                    real_values,
                    fid_mask,
                    model_name,
                )
                if predicted is None:
                    model_metrics[model_name] = {
                        "predicted": None,
                        "fiducial_rmse": np.nan,
                        "outside_rmse": np.nan,
                        "outside_max_abs": np.nan,
                        "outside_mean_signed_residual": np.nan,
                        "scale": np.nan,
                        "offset": np.nan,
                    }
                    continue
                residual = real_values - predicted
                fid_residual = residual[fid_mask]
                outside_residual = residual[outside_mask]
                model_metrics[model_name] = {
                    "predicted": predicted,
                    "fiducial_rmse": _rmse(fid_residual),
                    "outside_rmse": _rmse(outside_residual),
                    "outside_max_abs": (
                        float(np.nanmax(np.abs(outside_residual)))
                        if outside_residual.size
                        else np.nan
                    ),
                    "outside_mean_signed_residual": (
                        float(np.nanmean(outside_residual))
                        if outside_residual.size
                        else np.nan
                    ),
                    "scale": float(params["scale"]),
                    "offset": float(params["offset"]),
                }

            best_model_name = min(
                baseline_models,
                key=lambda name: (
                    np.inf
                    if not np.isfinite(model_metrics[name]["fiducial_rmse"])
                    else float(model_metrics[name]["fiducial_rmse"]),
                    np.inf
                    if not np.isfinite(model_metrics[name]["outside_rmse"])
                    else float(model_metrics[name]["outside_rmse"]),
                ),
            )
            best_metrics = model_metrics[best_model_name]
            global_model_name = min(
                baseline_models,
                key=lambda name: (
                    np.inf
                    if not np.isfinite(model_metrics[name]["outside_rmse"])
                    else float(model_metrics[name]["outside_rmse"]),
                    np.inf
                    if not np.isfinite(model_metrics[name]["outside_max_abs"])
                    else float(model_metrics[name]["outside_max_abs"]),
                    np.inf
                    if not np.isfinite(model_metrics[name]["fiducial_rmse"])
                    else float(model_metrics[name]["fiducial_rmse"]),
                ),
            )
            global_metrics = model_metrics[global_model_name]
            scale_metrics = model_metrics["scale"]
            if best_metrics["predicted"] is not None:
                enhanced.loc[subset.index, "best_baseline_pred_eff"] = best_metrics["predicted"]
                enhanced.loc[subset.index, "best_baseline_residual"] = (
                    real_values - np.asarray(best_metrics["predicted"], dtype=float)
                )
                enhanced.loc[subset.index, "best_baseline_model"] = best_model_name
            if global_metrics["predicted"] is not None:
                enhanced.loc[subset.index, "global_baseline_pred_eff"] = global_metrics["predicted"]
                enhanced.loc[subset.index, "global_baseline_residual"] = (
                    real_values - np.asarray(global_metrics["predicted"], dtype=float)
                )
                enhanced.loc[subset.index, "global_baseline_model"] = global_model_name
            if scale_metrics["predicted"] is not None:
                enhanced.loc[subset.index, "scale_baseline_pred_eff"] = scale_metrics["predicted"]
                enhanced.loc[subset.index, "scale_baseline_residual"] = (
                    real_values - np.asarray(scale_metrics["predicted"], dtype=float)
                )

            summary_rows.append(
                {
                    "plane": plane,
                    "axis": axis_name,
                    "n_bins_compared": int(len(subset)),
                    "n_fiducial_bins": int(np.count_nonzero(fid_mask)),
                    "n_outside_bins": int(np.count_nonzero(outside_mask)),
                    "best_baseline_model": best_model_name,
                    "best_baseline_scale": float(best_metrics["scale"]),
                    "best_baseline_offset": float(best_metrics["offset"]),
                    "best_baseline_fiducial_rmse": float(best_metrics["fiducial_rmse"]),
                    "best_baseline_outside_rmse": float(best_metrics["outside_rmse"]),
                    "best_baseline_outside_max_abs": float(best_metrics["outside_max_abs"]),
                    "best_baseline_outside_mean_signed_residual": float(
                        best_metrics["outside_mean_signed_residual"]
                    ),
                    "best_baseline_status": _classify_outside_shape(
                        float(best_metrics["outside_rmse"]),
                        float(best_metrics["outside_max_abs"]),
                        cfg,
                    ),
                    "global_baseline_model": global_model_name,
                    "global_baseline_scale": float(global_metrics["scale"]),
                    "global_baseline_offset": float(global_metrics["offset"]),
                    "global_baseline_fiducial_rmse": float(global_metrics["fiducial_rmse"]),
                    "global_baseline_outside_rmse": float(global_metrics["outside_rmse"]),
                    "global_baseline_outside_max_abs": float(global_metrics["outside_max_abs"]),
                    "global_baseline_outside_mean_signed_residual": float(
                        global_metrics["outside_mean_signed_residual"]
                    ),
                    "global_baseline_status": _classify_outside_shape(
                        float(global_metrics["outside_rmse"]),
                        float(global_metrics["outside_max_abs"]),
                        cfg,
                    ),
                    "scale_baseline_scale": float(scale_metrics["scale"]),
                    "scale_baseline_fiducial_rmse": float(scale_metrics["fiducial_rmse"]),
                    "scale_baseline_outside_rmse": float(scale_metrics["outside_rmse"]),
                    "scale_baseline_outside_max_abs": float(scale_metrics["outside_max_abs"]),
                    "scale_baseline_status": _classify_outside_shape(
                        float(scale_metrics["outside_rmse"]),
                        float(scale_metrics["outside_max_abs"]),
                        cfg,
                    ),
                    "raw_outside_rmse": raw_outside_rmse,
                    "raw_outside_max_abs": raw_outside_max_abs,
                    "raw_outside_mean_signed_residual": raw_outside_mean_signed,
                }
            )

    return enhanced, pd.DataFrame(summary_rows)


def _weighted_average_prediction(
    predictions: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    if not predictions:
        return np.asarray([], dtype=float)
    pred_stack = np.vstack(predictions)
    finite = np.isfinite(pred_stack)
    weighted = np.where(finite, pred_stack * weights[:, None], 0.0)
    weight_sum = np.where(finite, weights[:, None], 0.0).sum(axis=0)
    out = np.full(pred_stack.shape[1], np.nan, dtype=float)
    valid = weight_sum > 0
    out[valid] = weighted.sum(axis=0)[valid] / weight_sum[valid]
    return out


def match_real_file_to_sim_dictionary(
    *,
    sim_payload: dict[tuple[int, str], dict[str, object]],
    real_payload: dict[tuple[int, str], dict[str, object]],
    cfg: EfficiencyVectorTuningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    baseline_models = ("shift", "scale", "affine")

    for plane in PLANE_ORDER:
        for axis_name in AXIS_ORDER:
            sim_axis = sim_payload.get((plane, axis_name))
            real_axis = real_payload.get((plane, axis_name))
            if sim_axis is None or real_axis is None:
                continue

            sim_meta = sim_axis["meta"]
            sim_centers = np.asarray(sim_axis["centers"], dtype=float)
            sim_matrix = np.asarray(sim_axis["eff_matrix"], dtype=float)
            real_meta = real_axis["meta"]
            real_centers = np.asarray(real_axis["centers"], dtype=float)
            real_matrix = np.asarray(real_axis["eff_matrix"], dtype=float)

            if sim_matrix.size == 0 or real_matrix.size == 0:
                continue

            # The metadata vectors share the same binning by construction.
            if sim_centers.shape != real_centers.shape:
                continue
            centers = real_centers
            fid_mask_base = _fiducial_mask(centers, axis_name, cfg)
            outer_mask_base = ~fid_mask_base
            center_unit = str(real_axis["center_unit"])

            for real_idx in range(real_matrix.shape[0]):
                real_curve = real_matrix[real_idx]
                real_fid_mask = fid_mask_base & np.isfinite(real_curve)
                real_outer_mask = outer_mask_base & np.isfinite(real_curve)
                if int(np.count_nonzero(real_fid_mask)) < int(cfg.dictionary_match_min_fiducial_bins):
                    continue
                if int(np.count_nonzero(real_outer_mask)) < int(cfg.dictionary_match_min_outside_bins):
                    continue

                candidates: list[dict[str, object]] = []
                for sim_idx in range(sim_matrix.shape[0]):
                    sim_curve = sim_matrix[sim_idx]
                    shared_fid_mask = fid_mask_base & np.isfinite(real_curve) & np.isfinite(sim_curve)
                    if int(np.count_nonzero(shared_fid_mask)) < int(cfg.dictionary_match_min_fiducial_bins):
                        continue
                    shared_outer_mask = outer_mask_base & np.isfinite(real_curve) & np.isfinite(sim_curve)
                    if int(np.count_nonzero(shared_outer_mask)) < int(cfg.dictionary_match_min_outside_bins):
                        continue

                    best_model_name = None
                    best_fid_rmse = np.inf
                    best_pred = None
                    best_out_rmse = np.nan
                    best_out_max_abs = np.nan
                    for model_name in baseline_models:
                        pred, params = _fit_baseline_prediction(
                            sim_curve,
                            real_curve,
                            shared_fid_mask,
                            model_name,
                        )
                        if pred is None:
                            continue
                        residual = real_curve - pred
                        fid_rmse = _rmse(residual[shared_fid_mask])
                        out_residual = residual[shared_outer_mask]
                        out_rmse = _rmse(out_residual)
                        out_max_abs = (
                            float(np.nanmax(np.abs(out_residual)))
                            if out_residual.size
                            else np.nan
                        )
                        if np.isfinite(fid_rmse) and fid_rmse < best_fid_rmse:
                            best_fid_rmse = float(fid_rmse)
                            best_model_name = model_name
                            best_pred = pred
                            best_out_rmse = float(out_rmse)
                            best_out_max_abs = float(out_max_abs)
                            best_params = params
                    if best_model_name is None or best_pred is None:
                        continue

                    candidates.append(
                        {
                            "sim_idx": sim_idx,
                            "sim_filename_base": str(sim_meta.loc[sim_idx, "filename_base"]),
                            "sim_datetime": sim_meta.loc[sim_idx, "datetime"],
                            "model_name": str(best_model_name),
                            "fiducial_rmse": float(best_fid_rmse),
                            "outside_rmse": float(best_out_rmse),
                            "outside_max_abs": float(best_out_max_abs),
                            "scale": float(best_params["scale"]),
                            "offset": float(best_params["offset"]),
                            "prediction": np.asarray(best_pred, dtype=float),
                        }
                    )

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item["fiducial_rmse"], item["outside_rmse"]))
                top_k = candidates[: int(cfg.dictionary_match_top_k)]
                weights = np.asarray(
                    [
                        1.0
                        / (
                            float(item["fiducial_rmse"])
                            + float(cfg.dictionary_match_weight_epsilon)
                        )
                        ** float(cfg.dictionary_match_weight_power)
                        for item in top_k
                    ],
                    dtype=float,
                )
                weights = weights / np.sum(weights)
                weighted_prediction = _weighted_average_prediction(
                    [np.asarray(item["prediction"], dtype=float) for item in top_k],
                    weights,
                )
                valid_outer = outer_mask_base & np.isfinite(real_curve) & np.isfinite(weighted_prediction)
                valid_fid = fid_mask_base & np.isfinite(real_curve) & np.isfinite(weighted_prediction)
                if int(np.count_nonzero(valid_outer)) < int(cfg.dictionary_match_min_outside_bins):
                    continue

                weighted_residual = real_curve - weighted_prediction
                weighted_fid_rmse = _rmse(weighted_residual[valid_fid])
                weighted_out_rmse = _rmse(weighted_residual[valid_outer])
                weighted_out_max_abs = float(
                    np.nanmax(np.abs(weighted_residual[valid_outer]))
                )
                weighted_out_mean = float(np.nanmean(weighted_residual[valid_outer]))
                weighted_status = _classify_outside_shape(
                    weighted_out_rmse,
                    weighted_out_max_abs,
                    cfg,
                )
                best_single = top_k[0]
                fid_mean_real = float(np.nanmean(real_curve[real_fid_mask]))
                outer_mean_real = float(np.nanmean(real_curve[real_outer_mask]))
                fid_mean_match = (
                    float(np.nanmean(weighted_prediction[valid_fid]))
                    if np.any(valid_fid)
                    else np.nan
                )
                outer_mean_match = (
                    float(np.nanmean(weighted_prediction[valid_outer]))
                    if np.any(valid_outer)
                    else np.nan
                )
                time_order = real_idx + 1
                if "datetime" in real_meta.columns:
                    # Will be re-ranked globally after concatenation, but keep deterministic order now.
                    time_order = real_idx + 1

                detail_rows.append(
                    {
                        "real_filename_base": str(real_meta.loc[real_idx, "filename_base"]),
                        "real_datetime": real_meta.loc[real_idx, "datetime"],
                        "real_execution_timestamp": real_meta.loc[real_idx, "execution_timestamp"]
                        if "execution_timestamp" in real_meta.columns
                        else "",
                        "real_station_label": real_meta.loc[real_idx, "station_label"]
                        if "station_label" in real_meta.columns
                        else "",
                        "plane": plane,
                        "axis": axis_name,
                        "center_unit": center_unit,
                        "n_candidates": int(len(candidates)),
                        "top_k_used": int(len(top_k)),
                        "best_single_sim_filename_base": str(best_single["sim_filename_base"]),
                        "best_single_sim_datetime": best_single["sim_datetime"],
                        "best_single_model": str(best_single["model_name"]),
                        "best_single_fiducial_rmse": float(best_single["fiducial_rmse"]),
                        "best_single_outside_rmse": float(best_single["outside_rmse"]),
                        "best_single_outside_max_abs": float(best_single["outside_max_abs"]),
                        "weighted_match_fiducial_rmse": float(weighted_fid_rmse),
                        "weighted_match_outside_rmse": float(weighted_out_rmse),
                        "weighted_match_outside_max_abs": float(weighted_out_max_abs),
                        "weighted_match_outside_mean_signed_residual": float(weighted_out_mean),
                        "weighted_match_status": str(weighted_status),
                        "real_fiducial_mean_eff": float(fid_mean_real),
                        "real_outside_mean_eff": float(outer_mean_real),
                        "matched_fiducial_mean_eff": float(fid_mean_match),
                        "matched_outside_mean_eff": float(outer_mean_match),
                        "top_match_labels": " | ".join(
                            [
                                f"{item['sim_filename_base']}[{item['model_name']}]"
                                f":{float(item['fiducial_rmse']):.4f}"
                                for item in top_k
                            ]
                        ),
                    }
                )

            axis_detail = [row for row in detail_rows if row["plane"] == plane and row["axis"] == axis_name]
            if not axis_detail:
                continue
            axis_df = pd.DataFrame(axis_detail)
            summary_rows.append(
                {
                    "plane": plane,
                    "axis": axis_name,
                    "n_real_files_matched": int(len(axis_df)),
                    "median_weighted_outside_rmse": float(
                        np.nanmedian(axis_df["weighted_match_outside_rmse"].to_numpy(dtype=float))
                    ),
                    "max_weighted_outside_rmse": float(
                        np.nanmax(axis_df["weighted_match_outside_rmse"].to_numpy(dtype=float))
                    ),
                    "median_weighted_outside_max_abs": float(
                        np.nanmedian(axis_df["weighted_match_outside_max_abs"].to_numpy(dtype=float))
                    ),
                    "max_weighted_outside_max_abs": float(
                        np.nanmax(axis_df["weighted_match_outside_max_abs"].to_numpy(dtype=float))
                    ),
                    "median_real_fiducial_mean_eff": float(
                        np.nanmedian(axis_df["real_fiducial_mean_eff"].to_numpy(dtype=float))
                    ),
                    "median_real_outside_mean_eff": float(
                        np.nanmedian(axis_df["real_outside_mean_eff"].to_numpy(dtype=float))
                    ),
                    "n_watch_or_retune": int(
                        axis_df["weighted_match_status"].isin({"watch", "retune"}).sum()
                    ),
                    "n_retune": int(
                        axis_df["weighted_match_status"].eq("retune").sum()
                    ),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df["real_datetime"] = pd.to_datetime(detail_df["real_datetime"], errors="coerce")
        detail_df.sort_values(["plane", "axis", "real_datetime", "real_filename_base"], inplace=True)
        detail_df["time_order"] = (
            detail_df.groupby(["plane", "axis"]).cumcount() + 1
        )
    return detail_df, pd.DataFrame(summary_rows)


def build_axis_plane_summary(
    compare_df: pd.DataFrame,
    cfg: EfficiencyVectorTuningConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plane in PLANE_ORDER:
        for axis_name in AXIS_ORDER:
            subset = compare_df[
                (compare_df["plane"] == plane) & (compare_df["axis"] == axis_name)
            ].copy()
            eligible = subset[subset["eligible_for_comparison"]].copy()
            if eligible.empty:
                rows.append(
                    {
                        "plane": plane,
                        "axis": axis_name,
                        "n_bins_compared": 0,
                        "n_low_efficiency_bins": 0,
                        "median_abs_eff_difference": np.nan,
                        "max_abs_eff_difference": np.nan,
                        "low_efficiency_max_abs_difference": np.nan,
                        "edge_mean_abs_difference": np.nan,
                        "center_mean_abs_difference": np.nan,
                        "edge_mean_real_minus_sim": np.nan,
                        "center_mean_real_minus_sim": np.nan,
                        "worst_bin_center_value": np.nan,
                        "worst_bin_real_minus_sim": np.nan,
                        "worst_bin_status": "no_data",
                        "status": "no_data",
                    }
                )
                continue

            worst_row = eligible.sort_values(
                ["abs_eff_difference", "bin_index"],
                ascending=[False, True],
            ).iloc[0]
            low_eff = eligible[eligible["low_efficiency_bin"]]
            edge = eligible[eligible["edge_bin"]]
            center = eligible[~eligible["edge_bin"]]
            median_abs = float(np.nanmedian(eligible["abs_eff_difference"].to_numpy(dtype=float)))
            max_abs = float(np.nanmax(eligible["abs_eff_difference"].to_numpy(dtype=float)))
            low_eff_max = (
                float(np.nanmax(low_eff["abs_eff_difference"].to_numpy(dtype=float)))
                if not low_eff.empty
                else np.nan
            )
            if (
                np.isfinite(low_eff_max)
                and low_eff_max >= float(cfg.low_eff_severe_abs_difference)
            ) or median_abs >= float(cfg.severe_abs_difference):
                status = "retune"
            elif (
                max_abs >= float(cfg.warning_abs_difference)
                or (
                    np.isfinite(low_eff_max)
                    and low_eff_max >= float(cfg.low_eff_warning_abs_difference)
                )
            ):
                status = "watch"
            else:
                status = "ok"

            rows.append(
                {
                    "plane": plane,
                    "axis": axis_name,
                    "n_bins_compared": int(len(eligible)),
                    "n_low_efficiency_bins": int(len(low_eff)),
                    "median_abs_eff_difference": median_abs,
                    "max_abs_eff_difference": max_abs,
                    "low_efficiency_max_abs_difference": low_eff_max,
                    "edge_mean_abs_difference": (
                        float(np.nanmean(edge["abs_eff_difference"].to_numpy(dtype=float)))
                        if not edge.empty
                        else np.nan
                    ),
                    "center_mean_abs_difference": (
                        float(np.nanmean(center["abs_eff_difference"].to_numpy(dtype=float)))
                        if not center.empty
                        else np.nan
                    ),
                    "edge_mean_real_minus_sim": (
                        float(np.nanmean(edge["real_minus_sim_eff"].to_numpy(dtype=float)))
                        if not edge.empty
                        else np.nan
                    ),
                    "center_mean_real_minus_sim": (
                        float(np.nanmean(center["real_minus_sim_eff"].to_numpy(dtype=float)))
                        if not center.empty
                        else np.nan
                    ),
                    "worst_bin_center_value": float(worst_row["center_value"]),
                    "worst_bin_real_minus_sim": float(worst_row["real_minus_sim_eff"]),
                    "worst_bin_status": str(worst_row["bin_status"]),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _save_dual_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=170, bbox_inches="tight")
    return [png_path, pdf_path]


def plot_efficiency_curves(
    compare_df: pd.DataFrame,
    axis_plane_summary: pd.DataFrame,
    *,
    out_dir: Path,
    sim_label: str,
    real_label: str,
    cfg: EfficiencyVectorTuningConfig,
) -> None:
    fig, axes = plt.subplots(len(PLANE_ORDER), len(AXIS_ORDER), figsize=(15.5, 12.5), sharey=True)
    for row_idx, plane in enumerate(PLANE_ORDER):
        for col_idx, axis_name in enumerate(AXIS_ORDER):
            ax = axes[row_idx, col_idx]
            subset = compare_df[
                (compare_df["plane"] == plane)
                & (compare_df["axis"] == axis_name)
                & (compare_df["eligible_for_comparison"])
            ].sort_values("bin_index")
            summary_row = axis_plane_summary[
                (axis_plane_summary["plane"] == plane)
                & (axis_plane_summary["axis"] == axis_name)
            ]
            status = (
                str(summary_row["status"].iloc[0])
                if not summary_row.empty
                else "no_data"
            )
            if subset.empty:
                ax.set_axis_off()
                continue
            x = subset["center_value"].to_numpy(dtype=float)
            sim_med = subset["sim_median_eff"].to_numpy(dtype=float)
            sim_lo = subset["sim_p25_eff"].to_numpy(dtype=float)
            sim_hi = subset["sim_p75_eff"].to_numpy(dtype=float)
            real_med = subset["real_median_eff"].to_numpy(dtype=float)
            real_lo = subset["real_p25_eff"].to_numpy(dtype=float)
            real_hi = subset["real_p75_eff"].to_numpy(dtype=float)

            ax.fill_between(x, sim_lo, sim_hi, color=GROUP_COLORS["SIM"], alpha=0.18)
            ax.plot(x, sim_med, color=GROUP_COLORS["SIM"], linewidth=1.8, label=sim_label)
            ax.fill_between(x, real_lo, real_hi, color=GROUP_COLORS["REAL"], alpha=0.18)
            ax.plot(x, real_med, color=GROUP_COLORS["REAL"], linewidth=1.8, label=real_label)

            flagged = subset["bin_status"].isin({"watch", "retune"}).to_numpy()
            low_eff = subset["low_efficiency_bin"].to_numpy(dtype=bool)
            if np.any(flagged):
                ax.scatter(
                    x[flagged],
                    real_med[flagged],
                    s=28,
                    facecolor="white",
                    edgecolor="#d62728",
                    linewidth=1.2,
                    zorder=5,
                )
            if np.any(low_eff):
                ax.scatter(
                    x[low_eff],
                    np.minimum(real_med[low_eff], sim_med[low_eff]),
                    s=14,
                    color="#555555",
                    alpha=0.75,
                    zorder=4,
                )

            ax.axhline(float(cfg.low_efficiency_threshold), color="#9e9e9e", linestyle=":", linewidth=1.0)
            ax.set_ylim(0.0, 1.05)
            ax.grid(alpha=0.20)
            status_color = STATUS_COLORS.get(status, "black")
            title_suffix = ""
            if not summary_row.empty and np.isfinite(summary_row["max_abs_eff_difference"].iloc[0]):
                title_suffix = (
                    f" | max|Δ|={float(summary_row['max_abs_eff_difference'].iloc[0]):.3f}"
                )
            ax.set_title(
                f"P{plane} vs {axis_name} [{status}]{title_suffix}",
                color=status_color,
                fontsize=10,
            )
            if row_idx == len(PLANE_ORDER) - 1:
                ax.set_xlabel(AXIS_LABELS[axis_name], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("efficiency", fontsize=9)
            ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D([0], [0], color=GROUP_COLORS["SIM"], lw=2, label=sim_label),
        plt.Line2D([0], [0], color=GROUP_COLORS["REAL"], lw=2, label=real_label),
        plt.Line2D([0], [0], color="#9e9e9e", lw=1, ls=":", label=f"low-eff threshold = {cfg.low_efficiency_threshold:.2f}"),
        plt.Line2D([0], [0], marker="o", color="#d62728", markerfacecolor="white", lw=0, label="flagged bin"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Task 4 efficiency vectors: real vs simulation\nMedian with interquartile band",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    _save_dual_figure(fig, out_dir, "efficiency_vector_curves_real_vs_sim")
    plt.close(fig)


def plot_efficiency_differences(
    compare_df: pd.DataFrame,
    axis_plane_summary: pd.DataFrame,
    *,
    out_dir: Path,
    cfg: EfficiencyVectorTuningConfig,
) -> None:
    fig, axes = plt.subplots(len(PLANE_ORDER), len(AXIS_ORDER), figsize=(15.5, 12.5), sharey=True)
    finite = compare_df.loc[
        compare_df["eligible_for_comparison"],
        "abs_eff_difference",
    ].to_numpy(dtype=float)
    y_abs_max = float(np.nanmax(finite)) if finite.size else float(cfg.severe_abs_difference)
    y_limit = max(float(cfg.severe_abs_difference), y_abs_max) * 1.15
    for row_idx, plane in enumerate(PLANE_ORDER):
        for col_idx, axis_name in enumerate(AXIS_ORDER):
            ax = axes[row_idx, col_idx]
            subset = compare_df[
                (compare_df["plane"] == plane)
                & (compare_df["axis"] == axis_name)
                & (compare_df["eligible_for_comparison"])
            ].sort_values("bin_index")
            summary_row = axis_plane_summary[
                (axis_plane_summary["plane"] == plane)
                & (axis_plane_summary["axis"] == axis_name)
            ]
            status = (
                str(summary_row["status"].iloc[0])
                if not summary_row.empty
                else "no_data"
            )
            if subset.empty:
                ax.set_axis_off()
                continue

            x = subset["center_value"].to_numpy(dtype=float)
            diff = subset["real_minus_sim_eff"].to_numpy(dtype=float)
            flagged = subset["bin_status"].isin({"watch", "retune"}).to_numpy()
            low_eff = subset["low_efficiency_bin"].to_numpy(dtype=bool)
            edge = subset["edge_bin"].to_numpy(dtype=bool)

            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.axhline(float(cfg.warning_abs_difference), color="#9e9e9e", linestyle="--", linewidth=0.9)
            ax.axhline(-float(cfg.warning_abs_difference), color="#9e9e9e", linestyle="--", linewidth=0.9)
            ax.axhline(float(cfg.severe_abs_difference), color="#d62728", linestyle=":", linewidth=1.0)
            ax.axhline(-float(cfg.severe_abs_difference), color="#d62728", linestyle=":", linewidth=1.0)
            ax.plot(x, diff, color="#4c78a8", linewidth=1.7)
            if np.any(edge):
                ax.scatter(x[edge], diff[edge], s=18, color="#7f7f7f", alpha=0.65, zorder=4)
            if np.any(low_eff):
                ax.scatter(x[low_eff], diff[low_eff], s=30, color="#111111", alpha=0.85, zorder=5)
            if np.any(flagged):
                ax.scatter(
                    x[flagged],
                    diff[flagged],
                    s=40,
                    facecolor="white",
                    edgecolor="#d62728",
                    linewidth=1.2,
                    zorder=6,
                )
            ax.set_ylim(-y_limit, y_limit)
            ax.grid(alpha=0.20)
            status_color = STATUS_COLORS.get(status, "black")
            ax.set_title(f"P{plane} vs {axis_name} [{status}]", color=status_color, fontsize=10)
            if row_idx == len(PLANE_ORDER) - 1:
                ax.set_xlabel(AXIS_LABELS[axis_name], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("REAL - SIM", fontsize=9)
            ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D([0], [0], color="#4c78a8", lw=2, label="REAL - SIM"),
        plt.Line2D([0], [0], color="#9e9e9e", lw=1, ls="--", label=f"warning = ±{cfg.warning_abs_difference:.2f}"),
        plt.Line2D([0], [0], color="#d62728", lw=1, ls=":", label=f"severe = ±{cfg.severe_abs_difference:.2f}"),
        plt.Line2D([0], [0], marker="o", color="#111111", lw=0, label="low-eff bin"),
        plt.Line2D([0], [0], marker="o", color="#7f7f7f", lw=0, label="edge bin"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        "Task 4 efficiency vector discrepancy by plane and axis",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    _save_dual_figure(fig, out_dir, "efficiency_vector_real_minus_sim")
    plt.close(fig)


def plot_summary_heatmap(
    axis_plane_summary: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    if axis_plane_summary.empty:
        return

    metrics = [
        ("max_abs_eff_difference", "max |REAL-SIM|"),
        ("low_efficiency_max_abs_difference", "low-eff max |REAL-SIM|"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for ax, (metric, title) in zip(axes, metrics):
        heatmap = np.full((len(PLANE_ORDER), len(AXIS_ORDER)), np.nan, dtype=float)
        labels = np.full((len(PLANE_ORDER), len(AXIS_ORDER)), "", dtype=object)
        for row_idx, plane in enumerate(PLANE_ORDER):
            for col_idx, axis_name in enumerate(AXIS_ORDER):
                row = axis_plane_summary[
                    (axis_plane_summary["plane"] == plane)
                    & (axis_plane_summary["axis"] == axis_name)
                ]
                if row.empty:
                    continue
                value = pd.to_numeric(row[metric], errors="coerce").iloc[0]
                heatmap[row_idx, col_idx] = float(value) if np.isfinite(value) else np.nan
                labels[row_idx, col_idx] = str(row["status"].iloc[0])
        finite = heatmap[np.isfinite(heatmap)]
        vmax = float(np.nanmax(finite)) if finite.size else 0.20
        image = ax.imshow(
            heatmap,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=max(vmax, 0.12),
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(AXIS_ORDER)))
        ax.set_xticklabels(list(AXIS_ORDER))
        ax.set_yticks(np.arange(len(PLANE_ORDER)))
        ax.set_yticklabels([f"P{plane}" for plane in PLANE_ORDER])
        ax.set_title(title, fontsize=11)
        for row_idx, plane in enumerate(PLANE_ORDER):
            for col_idx, axis_name in enumerate(AXIS_ORDER):
                value = heatmap[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}\n{labels[row_idx, col_idx]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("absolute efficiency difference")

    fig.suptitle("Efficiency-vector discrepancy summary", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_dual_figure(fig, out_dir, "efficiency_vector_discrepancy_summary")
    plt.close(fig)


def _fiducial_span(axis_name: str, cfg: EfficiencyVectorTuningConfig) -> tuple[float, float]:
    if axis_name == "x":
        limit = float(cfg.baseline_x_abs_max_mm)
        return -limit, limit
    if axis_name == "y":
        limit = float(cfg.baseline_y_abs_max_mm)
        return -limit, limit
    return 0.0, float(cfg.baseline_theta_max_deg)


def plot_baseline_matched_curves(
    compare_df: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    *,
    out_dir: Path,
    sim_label: str,
    real_label: str,
    cfg: EfficiencyVectorTuningConfig,
) -> None:
    fig, axes = plt.subplots(len(PLANE_ORDER), len(AXIS_ORDER), figsize=(16.0, 12.5), sharey=True)
    for row_idx, plane in enumerate(PLANE_ORDER):
        for col_idx, axis_name in enumerate(AXIS_ORDER):
            ax = axes[row_idx, col_idx]
            subset = compare_df[
                (compare_df["plane"] == plane)
                & (compare_df["axis"] == axis_name)
                & (compare_df["eligible_for_comparison"])
            ].sort_values("bin_index")
            summary_row = baseline_summary[
                (baseline_summary["plane"] == plane)
                & (baseline_summary["axis"] == axis_name)
            ]
            if subset.empty or summary_row.empty:
                ax.set_axis_off()
                continue

            x = subset["center_value"].to_numpy(dtype=float)
            real_med = subset["real_median_eff"].to_numpy(dtype=float)
            sim_med = subset["sim_median_eff"].to_numpy(dtype=float)
            scale_pred = subset["scale_baseline_pred_eff"].to_numpy(dtype=float)
            best_pred = subset["global_baseline_pred_eff"].to_numpy(dtype=float)
            span_left, span_right = _fiducial_span(axis_name, cfg)
            ax.axvspan(span_left, span_right, color="#d9d9d9", alpha=0.25, zorder=0)
            ax.plot(x, sim_med, color=GROUP_COLORS["SIM"], linewidth=1.3, alpha=0.45, label=f"{sim_label} raw")
            ax.plot(x, scale_pred, color="#6a3d9a", linewidth=1.6, linestyle="--", label="scale fiducial match")
            ax.plot(x, best_pred, color="#1f78b4", linewidth=1.7, label="best global baseline")
            ax.plot(x, real_med, color=GROUP_COLORS["REAL"], linewidth=1.8, label=real_label)
            ax.set_ylim(0.0, 1.05)
            ax.grid(alpha=0.20)
            best_status = str(summary_row["global_baseline_status"].iloc[0])
            status_color = STATUS_COLORS.get(best_status, "black")
            ax.set_title(
                f"P{plane} {axis_name} [{best_status}] | best={summary_row['global_baseline_model'].iloc[0]} | "
                f"out max={float(summary_row['global_baseline_outside_max_abs'].iloc[0]):.3f}",
                fontsize=9,
                color=status_color,
            )
            if row_idx == len(PLANE_ORDER) - 1:
                ax.set_xlabel(AXIS_LABELS[axis_name], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("efficiency", fontsize=9)
            ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D([0], [0], color=GROUP_COLORS["SIM"], lw=1.5, alpha=0.45, label=f"{sim_label} raw"),
        plt.Line2D([0], [0], color="#6a3d9a", lw=1.8, ls="--", label="scale fiducial match"),
        plt.Line2D([0], [0], color="#1f78b4", lw=1.8, label="best global baseline"),
        plt.Line2D([0], [0], color=GROUP_COLORS["REAL"], lw=1.8, label=real_label),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", alpha=0.25, label="fiducial baseline region"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        "Baseline-matched efficiency vectors\nBest global non-position-dependent baseline fitted in the fiducial region",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    _save_dual_figure(fig, out_dir, "efficiency_vector_baseline_matched_curves")
    plt.close(fig)


def plot_baseline_residuals(
    compare_df: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    *,
    out_dir: Path,
    cfg: EfficiencyVectorTuningConfig,
) -> None:
    fig, axes = plt.subplots(len(PLANE_ORDER), len(AXIS_ORDER), figsize=(16.0, 12.5), sharey=True)
    finite = np.concatenate(
        [
            compare_df["scale_baseline_residual"].to_numpy(dtype=float),
            compare_df["best_baseline_residual"].to_numpy(dtype=float),
        ]
    )
    finite = finite[np.isfinite(finite)]
    y_limit = max(
        float(cfg.outside_shape_severe_max_abs),
        float(np.nanmax(np.abs(finite))) if finite.size else 0.15,
    ) * 1.18
    for row_idx, plane in enumerate(PLANE_ORDER):
        for col_idx, axis_name in enumerate(AXIS_ORDER):
            ax = axes[row_idx, col_idx]
            subset = compare_df[
                (compare_df["plane"] == plane)
                & (compare_df["axis"] == axis_name)
                & (compare_df["eligible_for_comparison"])
            ].sort_values("bin_index")
            summary_row = baseline_summary[
                (baseline_summary["plane"] == plane)
                & (baseline_summary["axis"] == axis_name)
            ]
            if subset.empty or summary_row.empty:
                ax.set_axis_off()
                continue

            x = subset["center_value"].to_numpy(dtype=float)
            raw_residual = subset["raw_residual"].to_numpy(dtype=float)
            scale_residual = subset["scale_baseline_residual"].to_numpy(dtype=float)
            best_residual = subset["global_baseline_residual"].to_numpy(dtype=float)
            span_left, span_right = _fiducial_span(axis_name, cfg)
            ax.axvspan(span_left, span_right, color="#d9d9d9", alpha=0.25, zorder=0)
            ax.axhline(0.0, color="black", linewidth=1.0)
            ax.axhline(float(cfg.outside_shape_warning_max_abs), color="#9e9e9e", linestyle="--", linewidth=0.9)
            ax.axhline(-float(cfg.outside_shape_warning_max_abs), color="#9e9e9e", linestyle="--", linewidth=0.9)
            ax.axhline(float(cfg.outside_shape_severe_max_abs), color="#d62728", linestyle=":", linewidth=1.0)
            ax.axhline(-float(cfg.outside_shape_severe_max_abs), color="#d62728", linestyle=":", linewidth=1.0)
            ax.plot(x, raw_residual, color="#9e9e9e", linewidth=1.2, alpha=0.9, label="raw residual")
            ax.plot(x, scale_residual, color="#6a3d9a", linewidth=1.5, linestyle="--", label="scale residual")
            ax.plot(x, best_residual, color="#1f78b4", linewidth=1.7, label="best-global residual")
            ax.set_ylim(-y_limit, y_limit)
            ax.grid(alpha=0.20)
            best_status = str(summary_row["global_baseline_status"].iloc[0])
            scale_status = str(summary_row["scale_baseline_status"].iloc[0])
            status_color = STATUS_COLORS.get(best_status, "black")
            ax.set_title(
                f"P{plane} {axis_name} | best={best_status} | scale={scale_status}",
                fontsize=9,
                color=status_color,
            )
            if row_idx == len(PLANE_ORDER) - 1:
                ax.set_xlabel(AXIS_LABELS[axis_name], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("REAL - matched SIM", fontsize=9)
            ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D([0], [0], color="#9e9e9e", lw=1.5, label="raw residual"),
        plt.Line2D([0], [0], color="#6a3d9a", lw=1.5, ls="--", label="scale residual"),
        plt.Line2D([0], [0], color="#1f78b4", lw=1.8, label="best-global residual"),
        plt.Line2D([0], [0], color="#9e9e9e", lw=1, ls="--", label=f"warning = ±{cfg.outside_shape_warning_max_abs:.2f}"),
        plt.Line2D([0], [0], color="#d62728", lw=1, ls=":", label=f"severe = ±{cfg.outside_shape_severe_max_abs:.2f}"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", alpha=0.25, label="fiducial baseline region"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(
        "Residual shape after fiducial baseline matching",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    _save_dual_figure(fig, out_dir, "efficiency_vector_baseline_residuals")
    plt.close(fig)


def plot_baseline_summary_heatmap(
    baseline_summary: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    if baseline_summary.empty:
        return

    metrics = [
        ("global_baseline_outside_max_abs", "best-global outside max |res|", "global_baseline_status"),
        ("global_baseline_outside_rmse", "best-global outside RMSE", "global_baseline_status"),
        ("scale_baseline_outside_max_abs", "scale-only outside max |res|", "scale_baseline_status"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, (metric, title, status_col) in zip(axes, metrics):
        heatmap = np.full((len(PLANE_ORDER), len(AXIS_ORDER)), np.nan, dtype=float)
        labels = np.full((len(PLANE_ORDER), len(AXIS_ORDER)), "", dtype=object)
        for row_idx, plane in enumerate(PLANE_ORDER):
            for col_idx, axis_name in enumerate(AXIS_ORDER):
                row = baseline_summary[
                    (baseline_summary["plane"] == plane)
                    & (baseline_summary["axis"] == axis_name)
                ]
                if row.empty:
                    continue
                value = pd.to_numeric(row[metric], errors="coerce").iloc[0]
                heatmap[row_idx, col_idx] = float(value) if np.isfinite(value) else np.nan
                if status_col == "global_baseline_status":
                    labels[row_idx, col_idx] = (
                        f"{row['global_baseline_model'].iloc[0]}\n{row[status_col].iloc[0]}"
                    )
                else:
                    labels[row_idx, col_idx] = str(row[status_col].iloc[0])
        finite = heatmap[np.isfinite(heatmap)]
        vmax = float(np.nanmax(finite)) if finite.size else 0.15
        image = ax.imshow(
            heatmap,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=max(vmax, 0.12),
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(AXIS_ORDER)))
        ax.set_xticklabels(list(AXIS_ORDER))
        ax.set_yticks(np.arange(len(PLANE_ORDER)))
        ax.set_yticklabels([f"P{plane}" for plane in PLANE_ORDER])
        ax.set_title(title, fontsize=10)
        for row_idx, plane in enumerate(PLANE_ORDER):
            for col_idx, axis_name in enumerate(AXIS_ORDER):
                value = heatmap[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}\n{labels[row_idx, col_idx]}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black",
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("outside residual")

    fig.suptitle("Baseline-match summary for the uniform-vs-dependent question", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_dual_figure(fig, out_dir, "efficiency_vector_baseline_summary")
    plt.close(fig)


def plot_dictionary_match_timeseries(
    match_detail: pd.DataFrame,
    *,
    out_dir: Path,
    cfg: EfficiencyVectorTuningConfig,
) -> None:
    if match_detail.empty:
        return

    fig, axes = plt.subplots(len(PLANE_ORDER), len(AXIS_ORDER), figsize=(16.5, 12.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for row_idx, plane in enumerate(PLANE_ORDER):
        for col_idx, axis_name in enumerate(AXIS_ORDER):
            ax = axes[row_idx, col_idx]
            subset = match_detail[
                (match_detail["plane"] == plane)
                & (match_detail["axis"] == axis_name)
            ].sort_values("real_datetime")
            if subset.empty:
                ax.set_axis_off()
                continue
            times = pd.to_datetime(subset["real_datetime"], errors="coerce")
            residual = subset["weighted_match_outside_max_abs"].to_numpy(dtype=float)
            eff_value = subset["real_fiducial_mean_eff"].to_numpy(dtype=float)
            colors = cmap(np.clip(eff_value, 0.0, 1.0))
            ax.plot(times, residual, color="#7f7f7f", linewidth=0.9, alpha=0.6)
            ax.scatter(
                times,
                residual,
                c=eff_value,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=18,
                edgecolors="none",
                zorder=4,
            )
            retune_mask = subset["weighted_match_status"].eq("retune").to_numpy()
            if np.any(retune_mask):
                ax.scatter(
                    times[retune_mask],
                    residual[retune_mask],
                    s=42,
                    facecolor="white",
                    edgecolor="#d62728",
                    linewidth=1.2,
                    zorder=5,
                )
            ax.axhline(float(cfg.outside_shape_warning_max_abs), color="#9e9e9e", linestyle="--", linewidth=0.9)
            ax.axhline(float(cfg.outside_shape_severe_max_abs), color="#d62728", linestyle=":", linewidth=1.0)
            ax.grid(alpha=0.20)
            n_bad = int(subset["weighted_match_status"].isin({"watch", "retune"}).sum())
            ax.set_title(
                f"P{plane} {axis_name} | bad={n_bad}/{len(subset)}",
                fontsize=9,
                color=STATUS_COLORS.get(
                    "retune" if subset["weighted_match_status"].eq("retune").any() else "ok",
                    "black",
                ),
            )
            if col_idx == 0:
                ax.set_ylabel("outer max |res|", fontsize=9)
            if row_idx == len(PLANE_ORDER) - 1:
                ax.set_xlabel("datetime", fontsize=9)
            ax.tick_params(labelsize=8)

    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    cbar.set_label("real fiducial mean efficiency")
    fig.suptitle(
        "Per-file dictionary match quality over time\nColor = real fiducial mean efficiency; open red markers = severe mismatch",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 0.985, 0.955])
    _save_dual_figure(fig, out_dir, "efficiency_vector_dictionary_match_timeseries")
    plt.close(fig)


def plot_dictionary_efficiency_timeseries(
    match_detail: pd.DataFrame,
    *,
    out_dir: Path,
    cfg: EfficiencyVectorTuningConfig,
) -> None:
    if match_detail.empty:
        return

    fig, axes = plt.subplots(len(PLANE_ORDER), len(AXIS_ORDER), figsize=(16.5, 12.5), sharex=True, sharey=False)
    cmap = plt.get_cmap("magma_r")
    for row_idx, plane in enumerate(PLANE_ORDER):
        for col_idx, axis_name in enumerate(AXIS_ORDER):
            ax = axes[row_idx, col_idx]
            subset = match_detail[
                (match_detail["plane"] == plane)
                & (match_detail["axis"] == axis_name)
            ].sort_values("real_datetime")
            if subset.empty:
                ax.set_axis_off()
                continue
            times = pd.to_datetime(subset["real_datetime"], errors="coerce")
            eff_value = subset["real_fiducial_mean_eff"].to_numpy(dtype=float)
            quality = subset["weighted_match_outside_max_abs"].to_numpy(dtype=float)
            ax.plot(times, eff_value, color="#7f7f7f", linewidth=0.9, alpha=0.6)
            ax.scatter(
                times,
                eff_value,
                c=quality,
                cmap="magma_r",
                vmin=0.0,
                vmax=max(float(cfg.outside_shape_severe_max_abs), float(np.nanmax(quality))),
                s=18,
                edgecolors="none",
                zorder=4,
            )
            retune_mask = subset["weighted_match_status"].eq("retune").to_numpy()
            if np.any(retune_mask):
                ax.scatter(
                    times[retune_mask],
                    eff_value[retune_mask],
                    s=42,
                    facecolor="white",
                    edgecolor="#d62728",
                    linewidth=1.2,
                    zorder=5,
                )
            ax.axhline(float(cfg.low_efficiency_threshold), color="#9e9e9e", linestyle=":", linewidth=0.9)
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.20)
            ax.set_title(f"P{plane} {axis_name}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("real fiducial eff", fontsize=9)
            if row_idx == len(PLANE_ORDER) - 1:
                ax.set_xlabel("datetime", fontsize=9)
            ax.tick_params(labelsize=8)

    norm = plt.Normalize(
        vmin=0.0,
        vmax=max(
            float(cfg.outside_shape_severe_max_abs),
            float(np.nanmax(match_detail["weighted_match_outside_max_abs"].to_numpy(dtype=float))),
        ),
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
    cbar.set_label("outer max |res| after dictionary match")
    fig.suptitle(
        "Real fiducial efficiency over time\nColor = outside residual after fiducial dictionary match",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 0.985, 0.955])
    _save_dual_figure(fig, out_dir, "efficiency_vector_real_fiducial_eff_timeseries")
    plt.close(fig)


def build_text_report(
    *,
    sim_label: str,
    real_label: str,
    selection,
    cfg: EfficiencyVectorTuningConfig,
    sim_frame: pd.DataFrame,
    real_frame: pd.DataFrame,
    compare_df: pd.DataFrame,
    axis_plane_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    match_detail: pd.DataFrame,
    match_summary: pd.DataFrame,
) -> str:
    eligible = compare_df[compare_df["eligible_for_comparison"]].copy()
    low_eff = eligible[eligible["low_efficiency_bin"]].copy()
    non_low_eff = eligible[~eligible["low_efficiency_bin"]].copy()
    retune_rows = axis_plane_summary[axis_plane_summary["status"] == "retune"].copy()
    watch_rows = axis_plane_summary[axis_plane_summary["status"] == "watch"].copy()
    overall_status = (
        "tuning required"
        if not retune_rows.empty
        else "watch"
        if not watch_rows.empty
        else "ok"
    )

    lines = [
        "Efficiency-vector tuning study",
        "",
        f"Simulation group: {sim_label}",
        f"Real-data group: {real_label}",
        f"Simulation stations: {', '.join(selection.simulation_stations)}",
        f"Real stations: {', '.join(selection.real_stations)}",
        f"Simulation files used: {int(sim_frame['filename_base'].nunique()) if not sim_frame.empty else 0}",
        f"Real files used: {int(real_frame['filename_base'].nunique()) if not real_frame.empty else 0}",
        f"Minimum valid files per bin in each group: {cfg.min_valid_files_per_bin}",
        f"Low-efficiency threshold: {cfg.low_efficiency_threshold:.2f}",
        f"Warning |REAL-SIM| threshold: {cfg.warning_abs_difference:.2f}",
        f"Severe |REAL-SIM| threshold: {cfg.severe_abs_difference:.2f}",
        f"Low-efficiency severe threshold: {cfg.low_eff_severe_abs_difference:.2f}",
        "",
        f"Overall verdict: {overall_status}",
        "",
        "Baseline-matched question",
        f"- fiducial baseline x window: |x| <= {cfg.baseline_x_abs_max_mm:.1f} mm",
        f"- fiducial baseline y window: |y| <= {cfg.baseline_y_abs_max_mm:.1f} mm",
        f"- fiducial baseline theta window: theta <= {cfg.baseline_theta_max_deg:.1f} deg",
        "- baseline set: shift, scale, affine",
    ]

    if not eligible.empty:
        lines.extend(
            [
                "",
                "Global comparison",
                f"- eligible bins compared: {len(eligible)}",
                f"- median |REAL-SIM| across eligible bins: {float(np.nanmedian(eligible['abs_eff_difference'])):.4f}",
                f"- max |REAL-SIM| across eligible bins: {float(np.nanmax(eligible['abs_eff_difference'])):.4f}",
                f"- low-efficiency bins compared: {len(low_eff)}",
                (
                    f"- median |REAL-SIM| in low-efficiency bins: "
                    f"{float(np.nanmedian(low_eff['abs_eff_difference'])):.4f}"
                    if not low_eff.empty
                    else "- median |REAL-SIM| in low-efficiency bins: NaN"
                ),
                (
                    f"- max |REAL-SIM| in low-efficiency bins: "
                    f"{float(np.nanmax(low_eff['abs_eff_difference'])):.4f}"
                    if not low_eff.empty
                    else "- max |REAL-SIM| in low-efficiency bins: NaN"
                ),
                (
                    f"- median |REAL-SIM| outside low-efficiency bins: "
                    f"{float(np.nanmedian(non_low_eff['abs_eff_difference'])):.4f}"
                    if not non_low_eff.empty
                    else "- median |REAL-SIM| outside low-efficiency bins: NaN"
                ),
            ]
        )

    if not baseline_summary.empty:
        lines.append("")
        lines.append("Uniform-vs-dependent conclusion by axis")
        for axis_name in AXIS_ORDER:
            subset = baseline_summary[baseline_summary["axis"] == axis_name].copy()
            if subset.empty:
                lines.append(f"- {axis_name}: no data")
                continue
            best_fail = subset["global_baseline_status"].eq("retune").any()
            best_watch = subset["global_baseline_status"].isin({"retune", "watch"}).any()
            scale_fail = subset["scale_baseline_status"].eq("retune").any()
            scale_watch = subset["scale_baseline_status"].isin({"retune", "watch"}).any()
            if best_fail:
                best_text = "explicit dependence required"
            elif best_watch:
                best_text = "watch"
            else:
                best_text = "global baseline likely sufficient"
            if scale_fail:
                scale_text = "uniform scale not sufficient"
            elif scale_watch:
                scale_text = "uniform scale questionable"
            else:
                scale_text = "uniform scale acceptable"
            lines.append(
                f"- {axis_name}: best-baseline verdict = {best_text}; scale-only verdict = {scale_text}"
            )

        lines.append("")
        lines.append("Baseline-matched plane-axis summary")
        for _, row in baseline_summary.sort_values(
            ["global_baseline_status", "global_baseline_outside_max_abs"],
            ascending=[True, False],
        ).iterrows():
            lines.append(
                f"- P{int(row['plane'])} {row['axis']}: "
                f"best-global={row['global_baseline_model']} "
                f"(fid RMSE={float(row['global_baseline_fiducial_rmse']):.4f}, "
                f"out RMSE={float(row['global_baseline_outside_rmse']):.4f}, "
                f"out max={float(row['global_baseline_outside_max_abs']):.4f}, "
                f"status={row['global_baseline_status']}), "
                f"best-fid={row['best_baseline_model']} "
                f"(out max={float(row['best_baseline_outside_max_abs']):.4f}, "
                f"status={row['best_baseline_status']}), "
                f"scale-only out max={float(row['scale_baseline_outside_max_abs']):.4f} "
                f"(status={row['scale_baseline_status']})"
            )

    if not match_summary.empty:
        lines.append("")
        lines.append("Per-file dictionary match summary")
        for axis_name in AXIS_ORDER:
            subset = match_summary[match_summary["axis"] == axis_name].copy()
            if subset.empty:
                continue
            n_bad = int(subset["n_watch_or_retune"].sum())
            n_total = int(subset["n_real_files_matched"].sum())
            worst = subset.sort_values("max_weighted_outside_max_abs", ascending=False).iloc[0]
            lines.append(
                f"- {axis_name}: bad matches {n_bad}/{n_total}, "
                f"worst plane P{int(worst['plane'])} max outside |res|={float(worst['max_weighted_outside_max_abs']):.4f}"
            )

    if not match_detail.empty:
        lines.append("")
        lines.append("Worst matched times")
        worst_rows = match_detail.sort_values(
            ["weighted_match_outside_max_abs", "real_datetime"],
            ascending=[False, True],
        ).head(12)
        for _, row in worst_rows.iterrows():
            lines.append(
                f"- {pd.Timestamp(row['real_datetime']).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['real_datetime']) else row['real_filename_base']} "
                f"| P{int(row['plane'])} {row['axis']} "
                f"| real fid eff={float(row['real_fiducial_mean_eff']):.4f} "
                f"| outer max |res|={float(row['weighted_match_outside_max_abs']):.4f} "
                f"| status={row['weighted_match_status']} "
                f"| best single={row['best_single_sim_filename_base']}[{row['best_single_model']}]"
            )

    if not retune_rows.empty:
        lines.append("")
        lines.append("Plane-axis combinations that need tuning attention")
        for _, row in retune_rows.sort_values(
            ["max_abs_eff_difference", "median_abs_eff_difference"],
            ascending=[False, False],
        ).iterrows():
            low_eff_max = row["low_efficiency_max_abs_difference"]
            low_eff_text = f", low-eff max={low_eff_max:.4f}" if np.isfinite(low_eff_max) else ""
            lines.append(
                f"- P{int(row['plane'])} {row['axis']}: "
                f"median |Δ|={float(row['median_abs_eff_difference']):.4f}, "
                f"max |Δ|={float(row['max_abs_eff_difference']):.4f}"
                f"{low_eff_text}, "
                f"edge mean Δ={float(row['edge_mean_real_minus_sim']):.4f}, "
                f"center mean Δ={float(row['center_mean_real_minus_sim']):.4f}"
            )

    if not eligible.empty:
        lines.append("")
        lines.append("Worst eligible bins")
        for _, row in eligible.sort_values(
            ["abs_eff_difference", "plane", "axis", "bin_index"],
            ascending=[False, True, True, True],
        ).head(10).iterrows():
            lines.append(
                f"- P{int(row['plane'])} {row['axis']} bin {int(row['bin_index'])} "
                f"at {float(row['center_value']):.3f} {row['center_unit']}: "
                f"SIM={float(row['sim_median_eff']):.4f}, "
                f"REAL={float(row['real_median_eff']):.4f}, "
                f"Δ={float(row['real_minus_sim_eff']):+.4f}, "
                f"status={row['bin_status']}, "
                f"low_eff={bool(row['low_efficiency_bin'])}, "
                f"edge={bool(row['edge_bin'])}"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Task 4 efficiency vectors between simulation and real data."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to the shared simulation-tuning YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_tuning_config(args.config)
    selection = resolve_selection(config)
    cfg = load_efficiency_vector_tuning_config(config)

    sim_label = group_label(selection.simulation_stations, fallback="SIM")
    real_label = group_label(selection.real_stations, fallback="REAL")

    sim_frame = load_task4_efficiency_metadata(
        selection.simulation_stations,
        selection.simulation_date_ranges,
    )
    real_frame = load_task4_efficiency_metadata(
        selection.real_stations,
        selection.real_date_ranges,
    )
    if sim_frame.empty:
        raise SystemExit("No Task 4 efficiency metadata found for the selected simulation stations.")
    if real_frame.empty:
        raise SystemExit("No Task 4 efficiency metadata found for the selected real-data stations.")

    sim_summary = build_group_bin_summary(
        sim_frame,
        group_kind="SIM",
        group_name=sim_label,
        cfg=cfg,
    )
    real_summary = build_group_bin_summary(
        real_frame,
        group_kind="REAL",
        group_name=real_label,
        cfg=cfg,
    )
    compare_df = build_real_vs_sim_bin_summary(sim_summary, real_summary, cfg)
    axis_plane_summary = build_axis_plane_summary(compare_df, cfg)
    compare_df, baseline_summary = augment_with_baseline_models(compare_df, cfg)
    sim_file_payload = build_file_curve_payload(sim_frame)
    real_file_payload = build_file_curve_payload(real_frame)
    match_detail, match_summary = match_real_file_to_sim_dictionary(
        sim_payload=sim_file_payload,
        real_payload=real_file_payload,
        cfg=cfg,
    )

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_summary.to_csv(out_dir / "efficiency_vector_group_bin_summary.csv", index=False)
    compare_df.to_csv(out_dir / "efficiency_vector_real_vs_sim_bin_summary.csv", index=False)
    axis_plane_summary.to_csv(out_dir / "efficiency_vector_axis_plane_summary.csv", index=False)
    baseline_summary.to_csv(out_dir / "efficiency_vector_baseline_summary.csv", index=False)
    match_detail.to_csv(out_dir / "efficiency_vector_dictionary_match_timeseries.csv", index=False)
    match_summary.to_csv(out_dir / "efficiency_vector_dictionary_match_summary.csv", index=False)

    plot_efficiency_curves(
        compare_df,
        axis_plane_summary,
        out_dir=out_dir,
        sim_label=sim_label,
        real_label=real_label,
        cfg=cfg,
    )
    plot_efficiency_differences(
        compare_df,
        axis_plane_summary,
        out_dir=out_dir,
        cfg=cfg,
    )
    plot_summary_heatmap(
        axis_plane_summary,
        out_dir=out_dir,
    )
    plot_baseline_matched_curves(
        compare_df,
        baseline_summary,
        out_dir=out_dir,
        sim_label=sim_label,
        real_label=real_label,
        cfg=cfg,
    )
    plot_baseline_residuals(
        compare_df,
        baseline_summary,
        out_dir=out_dir,
        cfg=cfg,
    )
    plot_baseline_summary_heatmap(
        baseline_summary,
        out_dir=out_dir,
    )
    plot_dictionary_match_timeseries(
        match_detail,
        out_dir=out_dir,
        cfg=cfg,
    )
    plot_dictionary_efficiency_timeseries(
        match_detail,
        out_dir=out_dir,
        cfg=cfg,
    )

    report = build_text_report(
        sim_label=sim_label,
        real_label=real_label,
        selection=selection,
        cfg=cfg,
        sim_frame=sim_frame,
        real_frame=real_frame,
        compare_df=compare_df,
        axis_plane_summary=axis_plane_summary,
        baseline_summary=baseline_summary,
        match_detail=match_detail,
        match_summary=match_summary,
    )
    (out_dir / "efficiency_vector_tuning_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
