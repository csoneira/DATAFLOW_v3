#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
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
    collect_calibrated_file_entries,
    date_range_mask,
    filter_frame_by_datetime,
    group_label,
    load_tuning_config,
    resolve_selection,
)
from MASTER.common.file_selection import extract_run_datetime_from_name


TT_ORDER = ("123", "124", "134", "234", "1234")
GROUP_COLORS = {
    "SIM": "#d95f02",
    "REAL": "#1b9e77",
    "TASK2": "#4c78a8",
}
PLANE_QSUM_COLUMNS = {
    plane: [f"Q{plane}_Q_sum_{strip}" for strip in range(1, 5)]
    for plane in range(1, 5)
}
PLANE_TDIF_COLUMNS = {
    plane: [f"T{plane}_T_dif_{strip}" for strip in range(1, 5)]
    for plane in range(1, 5)
}


@dataclass(frozen=True)
class ProjectionAxisMismatchConfig:
    focus_definitive_tt: tuple[str, ...]
    task4_min_points_per_tt: int
    task2_abs_quantile: float
    strip_balance_mode: str
    scale_agreement_tolerance: float
    strip_balance_jsd_warning_threshold: float
    timeseries_reference_quantile_low: float
    timeseries_reference_quantile_high: float


@dataclass(frozen=True)
class CalibratedAxisPayload:
    files: list[tuple[str, Path]]
    tdif_by_plane_strip: dict[tuple[int, int], np.ndarray]
    single_strip_counts: dict[tuple[int, int], int]
    single_strip_total_by_plane: dict[int, int]


def output_dir() -> Path:
    return Path(__file__).resolve().parent / "OUTPUTS"


def load_projection_axis_config(config: dict) -> ProjectionAxisMismatchConfig:
    study_cfg = config.get("projection_axis_mismatch", {})
    focus_raw = study_cfg.get("focus_definitive_tt", list(TT_ORDER))
    if not isinstance(focus_raw, (list, tuple)) or not focus_raw:
        raise ValueError("projection_axis_mismatch.focus_definitive_tt must be a non-empty list.")
    focus_definitive_tt = tuple(str(item).strip() for item in focus_raw)
    for tt_label in focus_definitive_tt:
        if tt_label not in TT_ORDER:
            raise ValueError(
                "projection_axis_mismatch.focus_definitive_tt contains unsupported "
                f"definitive_tt={tt_label!r}."
            )

    strip_balance_mode = str(
        study_cfg.get("strip_balance_mode", "single_strip_only")
    ).strip().lower()
    if strip_balance_mode not in {"single_strip_only", "all_active"}:
        raise ValueError(
            "projection_axis_mismatch.strip_balance_mode must be "
            "'single_strip_only' or 'all_active'."
        )

    return ProjectionAxisMismatchConfig(
        focus_definitive_tt=focus_definitive_tt,
        task4_min_points_per_tt=int(study_cfg.get("task4_min_points_per_tt", 1000)),
        task2_abs_quantile=float(study_cfg.get("task2_abs_quantile", 0.95)),
        strip_balance_mode=strip_balance_mode,
        scale_agreement_tolerance=float(study_cfg.get("scale_agreement_tolerance", 0.08)),
        strip_balance_jsd_warning_threshold=float(
            study_cfg.get("strip_balance_jsd_warning_threshold", 0.08)
        ),
        timeseries_reference_quantile_low=float(
            study_cfg.get("timeseries_reference_quantile_low", 0.25)
        ),
        timeseries_reference_quantile_high=float(
            study_cfg.get("timeseries_reference_quantile_high", 0.75)
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


def _safe_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _safe_quantile_abs(values: np.ndarray, quantile: float) -> float:
    clean = np.abs(values[np.isfinite(values)])
    if clean.size == 0:
        return np.nan
    return float(np.quantile(clean, quantile))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def station_task4_specific_path(station_label: str) -> Path:
    return (
        ROOT_DIR
        / "STATIONS"
        / station_label
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_4"
        / "METADATA"
        / "task_4_metadata_specific.csv"
    )


def load_task4_projection_metadata(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    needed_prefixes = (
        "timtrack_projection_ellipse_tt_",
        "timtrack_projection_scaling_",
    )
    needed_columns = {
        "filename_base",
        "execution_timestamp",
        "timtrack_projection_ellipse_available",
        "timtrack_projection_ellipse_reason",
    }
    frames: list[pd.DataFrame] = []
    for station_label in station_labels:
        csv_path = station_task4_specific_path(station_label)
        if not csv_path.exists():
            continue
        frame = pd.read_csv(
            csv_path,
            usecols=lambda col: (
                col in needed_columns
                or any(col.startswith(prefix) for prefix in needed_prefixes)
            ),
            low_memory=False,
        )
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


def summarize_task4_projection_metrics(
    frame: pd.DataFrame,
    *,
    group_kind: str,
    group_name: str,
    cfg: ProjectionAxisMismatchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tt_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    for tt_label in cfg.focus_definitive_tt:
        prefix = f"timtrack_projection_ellipse_tt_{tt_label}"
        available_col = f"{prefix}_available"
        n_points_col = f"{prefix}_n_points"
        ratio_col = f"{prefix}_fwhm_halfwidth_x_over_y"
        scale_col = f"{prefix}_xproj_scaling_factor_to_match_y"
        if available_col not in frame.columns:
            continue
        available = _coerce_bool(frame[available_col])
        n_points = pd.to_numeric(frame.get(n_points_col), errors="coerce")
        ratio = pd.to_numeric(frame.get(ratio_col), errors="coerce")
        scale = pd.to_numeric(frame.get(scale_col), errors="coerce")
        mask = (
            available
            & np.isfinite(n_points)
            & np.isfinite(ratio)
            & np.isfinite(scale)
            & (n_points >= int(cfg.task4_min_points_per_tt))
        )
        if not mask.any():
            continue
        n_valid = n_points.loc[mask].to_numpy(dtype=float)
        ratio_valid = ratio.loc[mask].to_numpy(dtype=float)
        scale_valid = scale.loc[mask].to_numpy(dtype=float)
        tt_rows.append(
            {
                "group_kind": group_kind,
                "group_name": group_name,
                "tt": tt_label,
                "n_files": int(mask.sum()),
                "total_points": float(np.nansum(n_valid)),
                "median_fwhm_halfwidth_x_over_y": float(np.nanmedian(ratio_valid)),
                "p25_fwhm_halfwidth_x_over_y": float(np.nanquantile(ratio_valid, 0.25)),
                "p75_fwhm_halfwidth_x_over_y": float(np.nanquantile(ratio_valid, 0.75)),
                "weighted_mean_fwhm_halfwidth_x_over_y": _safe_weighted_mean(
                    ratio_valid, n_valid
                ),
                "median_xproj_scaling_factor_to_match_y": float(np.nanmedian(scale_valid)),
                "p25_xproj_scaling_factor_to_match_y": float(np.nanquantile(scale_valid, 0.25)),
                "p75_xproj_scaling_factor_to_match_y": float(np.nanquantile(scale_valid, 0.75)),
                "weighted_mean_xproj_scaling_factor_to_match_y": _safe_weighted_mean(
                    scale_valid, n_valid
                ),
            }
        )

    global_scale = pd.to_numeric(
        frame.get("timtrack_projection_scaling_factor_xproj_global"),
        errors="coerce",
    )
    global_weight = pd.to_numeric(
        frame.get("timtrack_projection_scaling_global_total_weight"),
        errors="coerce",
    )
    global_mask = np.isfinite(global_scale)
    if global_mask.any():
        global_rows.append(
            {
                "group_kind": group_kind,
                "group_name": group_name,
                "n_files": int(global_mask.sum()),
                "median_global_xproj_scaling_factor": float(
                    np.nanmedian(global_scale.loc[global_mask].to_numpy(dtype=float))
                ),
                "weighted_mean_global_xproj_scaling_factor": _safe_weighted_mean(
                    global_scale.loc[global_mask].to_numpy(dtype=float),
                    global_weight.loc[global_mask].fillna(1.0).to_numpy(dtype=float),
                ),
            }
        )

    return pd.DataFrame(tt_rows), pd.DataFrame(global_rows)


def collect_calibrated_axis_payload(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
    *,
    strip_balance_mode: str,
) -> CalibratedAxisPayload:
    file_entries = collect_calibrated_file_entries(station_labels)
    tdif_chunks: dict[tuple[int, int], list[np.ndarray]] = {
        (plane, strip): []
        for plane in range(1, 5)
        for strip in range(1, 5)
    }
    single_strip_counts = {
        (plane, strip): 0
        for plane in range(1, 5)
        for strip in range(1, 5)
    }
    single_strip_total_by_plane = {plane: 0 for plane in range(1, 5)}
    used_entries: list[tuple[str, Path]] = []
    columns = ["datetime"]
    for plane in range(1, 5):
        columns.extend(PLANE_TDIF_COLUMNS[plane])
        columns.extend(PLANE_QSUM_COLUMNS[plane])

    for station_label, parquet_path in file_entries:
        if not parquet_path.exists():
            continue
        try:
            frame = pd.read_parquet(parquet_path, columns=columns)
        except FileNotFoundError:
            # Cron can move files between discovery and read while the study runs.
            continue
        frame = filter_frame_by_datetime(frame, date_ranges)
        if frame.empty:
            continue
        used_entries.append((station_label, parquet_path))

        for plane in range(1, 5):
            for strip, column in enumerate(PLANE_TDIF_COLUMNS[plane], start=1):
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values) & (values != 0)]
                if values.size:
                    tdif_chunks[(plane, strip)].append(values)

            qsum_values = frame[PLANE_QSUM_COLUMNS[plane]].to_numpy(dtype=float)
            hit_mask = np.isfinite(qsum_values) & (qsum_values > 0)
            if strip_balance_mode == "single_strip_only":
                row_mask = hit_mask.sum(axis=1) == 1
                if not np.any(row_mask):
                    continue
                strip_indices = np.argmax(hit_mask[row_mask], axis=1)
                single_strip_total_by_plane[plane] += int(strip_indices.size)
                for strip_idx in strip_indices:
                    single_strip_counts[(plane, int(strip_idx) + 1)] += 1
            else:
                single_strip_total_by_plane[plane] += int(hit_mask.sum())
                active_rows, active_cols = np.where(hit_mask)
                del active_rows
                for strip_idx in active_cols:
                    single_strip_counts[(plane, int(strip_idx) + 1)] += 1

    finalized_tdif = {
        key: (np.concatenate(chunks) if chunks else np.array([], dtype=float))
        for key, chunks in tdif_chunks.items()
    }
    return CalibratedAxisPayload(
        files=used_entries,
        tdif_by_plane_strip=finalized_tdif,
        single_strip_counts=single_strip_counts,
        single_strip_total_by_plane=single_strip_total_by_plane,
    )


def summarize_task2_tdif_mismatch(
    sim_payload: CalibratedAxisPayload,
    real_payload: CalibratedAxisPayload,
    *,
    abs_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _append_row(level: str, plane: int | str, strip: int | str, sim_values: np.ndarray, real_values: np.ndarray) -> None:
        sim_q = _safe_quantile_abs(sim_values, abs_quantile)
        real_q = _safe_quantile_abs(real_values, abs_quantile)
        ratio = float(real_q / sim_q) if np.isfinite(sim_q) and sim_q > 0 and np.isfinite(real_q) else np.nan
        implied_scale = float(1.0 / ratio) if np.isfinite(ratio) and ratio > 0 else np.nan
        rows.append(
            {
                "level": level,
                "plane": plane,
                "strip": strip,
                "sim_count": int(np.isfinite(sim_values).sum()),
                "real_count": int(np.isfinite(real_values).sum()),
                f"sim_q{int(round(abs_quantile * 100))}_abs_tdif_ns": sim_q,
                f"real_q{int(round(abs_quantile * 100))}_abs_tdif_ns": real_q,
                f"real_over_sim_q{int(round(abs_quantile * 100))}_abs_tdif": ratio,
                "implied_real_x_scale_from_tdif": implied_scale,
            }
        )

    all_sim_chunks: list[np.ndarray] = []
    all_real_chunks: list[np.ndarray] = []
    for plane in range(1, 5):
        plane_sim_chunks: list[np.ndarray] = []
        plane_real_chunks: list[np.ndarray] = []
        for strip in range(1, 5):
            sim_values = sim_payload.tdif_by_plane_strip[(plane, strip)]
            real_values = real_payload.tdif_by_plane_strip[(plane, strip)]
            _append_row("plane_strip", plane, strip, sim_values, real_values)
            if sim_values.size:
                plane_sim_chunks.append(sim_values)
                all_sim_chunks.append(sim_values)
            if real_values.size:
                plane_real_chunks.append(real_values)
                all_real_chunks.append(real_values)
        plane_sim = np.concatenate(plane_sim_chunks) if plane_sim_chunks else np.array([], dtype=float)
        plane_real = np.concatenate(plane_real_chunks) if plane_real_chunks else np.array([], dtype=float)
        _append_row("plane", plane, "all", plane_sim, plane_real)

    overall_sim = np.concatenate(all_sim_chunks) if all_sim_chunks else np.array([], dtype=float)
    overall_real = np.concatenate(all_real_chunks) if all_real_chunks else np.array([], dtype=float)
    _append_row("overall", "all", "all", overall_sim, overall_real)
    return pd.DataFrame(rows)


def summarize_strip_balance(
    sim_payload: CalibratedAxisPayload,
    real_payload: CalibratedAxisPayload,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    plane_rows: list[dict[str, object]] = []
    for plane in range(1, 5):
        sim_counts = np.array(
            [sim_payload.single_strip_counts[(plane, strip)] for strip in range(1, 5)],
            dtype=float,
        )
        real_counts = np.array(
            [real_payload.single_strip_counts[(plane, strip)] for strip in range(1, 5)],
            dtype=float,
        )
        sim_total = float(sim_counts.sum())
        real_total = float(real_counts.sum())
        sim_share = sim_counts / sim_total if sim_total > 0 else np.full(4, np.nan)
        real_share = real_counts / real_total if real_total > 0 else np.full(4, np.nan)
        jsd = _js_divergence(sim_counts, real_counts)
        l1_distance = (
            float(np.nansum(np.abs(real_share - sim_share)))
            if np.isfinite(sim_share).all() and np.isfinite(real_share).all()
            else np.nan
        )
        mean_strip_sim = (
            float(np.dot(np.arange(1, 5, dtype=float), sim_share))
            if np.isfinite(sim_share).all()
            else np.nan
        )
        mean_strip_real = (
            float(np.dot(np.arange(1, 5, dtype=float), real_share))
            if np.isfinite(real_share).all()
            else np.nan
        )
        plane_rows.append(
            {
                "plane": plane,
                "sim_total": int(sim_total),
                "real_total": int(real_total),
                "js_divergence": jsd,
                "l1_distance": l1_distance,
                "mean_strip_sim": mean_strip_sim,
                "mean_strip_real": mean_strip_real,
                "mean_strip_delta_real_minus_sim": (
                    mean_strip_real - mean_strip_sim
                    if np.isfinite(mean_strip_real) and np.isfinite(mean_strip_sim)
                    else np.nan
                ),
            }
        )
        for strip in range(1, 5):
            detail_rows.append(
                {
                    "plane": plane,
                    "strip": strip,
                    "sim_count": int(sim_counts[strip - 1]),
                    "real_count": int(real_counts[strip - 1]),
                    "sim_share": sim_share[strip - 1],
                    "real_share": real_share[strip - 1],
                    "share_delta_real_minus_sim": (
                        real_share[strip - 1] - sim_share[strip - 1]
                        if np.isfinite(real_share[strip - 1]) and np.isfinite(sim_share[strip - 1])
                        else np.nan
                    ),
                    "plane_js_divergence": jsd,
                    "plane_l1_distance": l1_distance,
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(plane_rows)


def build_task4_timeseries_tables(
    sim_frame: pd.DataFrame,
    real_frame: pd.DataFrame,
    *,
    cfg: ProjectionAxisMismatchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_specs: list[tuple[str, str, str | None, str | None, str | None]] = [
        (
            "global_xproj_scaling_factor",
            "Global x scale factor",
            "timtrack_projection_scaling_factor_xproj_global",
            None,
            "timtrack_projection_scaling_global_total_weight",
        )
    ]
    for tt_label in cfg.focus_definitive_tt:
        prefix = f"timtrack_projection_ellipse_tt_{tt_label}"
        metric_specs.append(
            (
                f"tt_{tt_label}_fwhm_halfwidth_x_over_y",
                f"TT {tt_label} FWHM x/y",
                f"{prefix}_fwhm_halfwidth_x_over_y",
                f"{prefix}_available",
                f"{prefix}_n_points",
            )
        )

    reference_rows: list[dict[str, object]] = []
    ts_frame = real_frame.copy()
    if ts_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "datetime" in ts_frame.columns:
        ts_frame = ts_frame.sort_values(["datetime", "filename_base"], na_position="last").copy()
    else:
        ts_frame = ts_frame.sort_values(["filename_base"]).copy()

    ts_frame["time_order"] = np.arange(1, len(ts_frame) + 1, dtype=int)
    base_columns = [
        "filename_base",
        "station_label",
        "datetime",
        "execution_timestamp",
        "time_order",
    ]
    time_series = ts_frame[base_columns].copy()

    for metric_key, metric_label, value_col, available_col, weight_col in metric_specs:
        sim_values = pd.to_numeric(sim_frame.get(value_col), errors="coerce")
        if available_col and available_col in sim_frame.columns:
            sim_available = _coerce_bool(sim_frame[available_col])
        else:
            sim_available = pd.Series(True, index=sim_frame.index, dtype=bool)
        if weight_col and weight_col in sim_frame.columns:
            sim_weight = pd.to_numeric(sim_frame[weight_col], errors="coerce")
        else:
            sim_weight = pd.Series(np.nan, index=sim_frame.index, dtype=float)
        if available_col and weight_col:
            sim_mask = (
                sim_available
                & np.isfinite(sim_values)
                & np.isfinite(sim_weight)
                & (sim_weight >= cfg.task4_min_points_per_tt)
            )
        else:
            sim_mask = np.isfinite(sim_values)
        sim_valid = sim_values.loc[sim_mask].to_numpy(dtype=float)
        if sim_valid.size == 0:
            continue
        sim_median = float(np.nanmedian(sim_valid))
        sim_q_low = float(
            np.nanquantile(sim_valid, cfg.timeseries_reference_quantile_low)
        )
        sim_q_high = float(
            np.nanquantile(sim_valid, cfg.timeseries_reference_quantile_high)
        )
        reference_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": metric_label,
                "sim_n_files": int(sim_valid.size),
                "sim_median": sim_median,
                "sim_q_low": sim_q_low,
                "sim_q_high": sim_q_high,
            }
        )

        real_values = pd.to_numeric(ts_frame.get(value_col), errors="coerce")
        if available_col and available_col in ts_frame.columns:
            real_available = _coerce_bool(ts_frame[available_col])
        else:
            real_available = pd.Series(True, index=ts_frame.index, dtype=bool)
        if weight_col and weight_col in ts_frame.columns:
            real_weight = pd.to_numeric(ts_frame[weight_col], errors="coerce")
        else:
            real_weight = pd.Series(np.nan, index=ts_frame.index, dtype=float)
        if available_col and weight_col:
            real_mask = (
                real_available
                & np.isfinite(real_values)
                & np.isfinite(real_weight)
                & (real_weight >= cfg.task4_min_points_per_tt)
            )
        else:
            real_mask = np.isfinite(real_values)
        series_value = real_values.where(real_mask, np.nan)
        ratio_to_sim = (
            series_value / sim_median
            if np.isfinite(sim_median) and sim_median != 0
            else np.nan
        )
        time_series[metric_key] = series_value
        time_series[f"{metric_key}_over_sim_median"] = ratio_to_sim

    reference_df = pd.DataFrame(reference_rows)
    summary_rows: list[dict[str, object]] = []
    if not reference_df.empty:
        for _, ref_row in reference_df.iterrows():
            metric_key = str(ref_row["metric_key"])
            values = pd.to_numeric(time_series.get(metric_key), errors="coerce")
            ratio_values = pd.to_numeric(
                time_series.get(f"{metric_key}_over_sim_median"),
                errors="coerce",
            )
            valid_mask = np.isfinite(values)
            if not valid_mask.any():
                continue
            valid_values = values.loc[valid_mask].to_numpy(dtype=float)
            valid_ratios = ratio_values.loc[valid_mask].to_numpy(dtype=float)
            valid_frame = time_series.loc[valid_mask, ["datetime", "filename_base"]].copy()
            valid_frame["ratio"] = valid_ratios
            if valid_frame["datetime"].notna().any():
                valid_frame.sort_values(["datetime", "filename_base"], inplace=True)
            else:
                valid_frame.sort_values(["filename_base"], inplace=True)
            split_index = max(1, len(valid_frame) // 2)
            first_half = valid_frame["ratio"].iloc[:split_index].to_numpy(dtype=float)
            last_half = valid_frame["ratio"].iloc[-split_index:].to_numpy(dtype=float)
            sim_q_low = float(ref_row["sim_q_low"])
            sim_q_high = float(ref_row["sim_q_high"])
            above = int(np.sum(valid_values > sim_q_high))
            below = int(np.sum(valid_values < sim_q_low))
            in_band = int(np.sum((valid_values >= sim_q_low) & (valid_values <= sim_q_high)))
            dominant_outside = max(above, below)
            dominant_direction = "above" if above >= below else "below"
            summary_rows.append(
                {
                    "metric_key": metric_key,
                    "metric_label": ref_row["metric_label"],
                    "n_real_files": int(valid_values.size),
                    "n_in_sim_iqr_band": in_band,
                    "n_above_sim_iqr_band": above,
                    "n_below_sim_iqr_band": below,
                    "dominant_outside_side": dominant_direction,
                    "dominant_outside_fraction": float(dominant_outside / valid_values.size),
                    "median_ratio_to_sim_median": float(np.nanmedian(valid_ratios)),
                    "p10_ratio_to_sim_median": float(np.nanquantile(valid_ratios, 0.10)),
                    "p90_ratio_to_sim_median": float(np.nanquantile(valid_ratios, 0.90)),
                    "first_half_median_ratio_to_sim_median": float(np.nanmedian(first_half)),
                    "last_half_median_ratio_to_sim_median": float(np.nanmedian(last_half)),
                    "half_to_half_ratio_shift": float(np.nanmedian(last_half) - np.nanmedian(first_half)),
                }
            )

    return time_series, reference_df, pd.DataFrame(summary_rows)


def save_dual_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_task4_projection_timeseries(
    time_series: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    out_dir: Path,
    real_label: str,
) -> None:
    if time_series.empty or reference_df.empty:
        return
    metric_keys = ["global_xproj_scaling_factor"] + [
        f"tt_{tt_label}_fwhm_halfwidth_x_over_y"
        for tt_label in TT_ORDER
        if f"tt_{tt_label}_fwhm_halfwidth_x_over_y" in set(reference_df["metric_key"])
    ]
    if not metric_keys:
        return

    ncols = 2
    nrows = int(np.ceil(len(metric_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    plot_frame = time_series.copy()
    has_datetime = plot_frame["datetime"].notna().any()
    if has_datetime:
        x_values = pd.to_datetime(plot_frame["datetime"], errors="coerce")
        x_label = "run datetime"
    else:
        x_values = plot_frame["time_order"].to_numpy(dtype=float)
        x_label = "file order"

    for ax, metric_key in zip(axes.flat, metric_keys):
        ref_row = reference_df[reference_df["metric_key"] == metric_key]
        if ref_row.empty:
            ax.set_visible(False)
            continue
        ref = ref_row.iloc[0]
        ratio_col = f"{metric_key}_over_sim_median"
        series = pd.to_numeric(plot_frame.get(ratio_col), errors="coerce")
        valid_mask = np.isfinite(series)
        if not valid_mask.any():
            ax.set_visible(False)
            continue
        band_low = float(ref["sim_q_low"] / ref["sim_median"]) if ref["sim_median"] else np.nan
        band_high = float(ref["sim_q_high"] / ref["sim_median"]) if ref["sim_median"] else np.nan
        if np.isfinite(band_low) and np.isfinite(band_high):
            ax.axhspan(band_low, band_high, color="#bdbdbd", alpha=0.35, label="SIM IQR / median")
        ax.axhline(1.0, color="black", lw=1.0, ls="--", label="SIM median")
        ax.plot(
            x_values[valid_mask],
            series[valid_mask].to_numpy(dtype=float),
            color=GROUP_COLORS["REAL"],
            lw=1.2,
            marker="o",
            ms=4.0,
            label=real_label,
        )
        metric_label = str(ref["metric_label"])
        ax.set_title(metric_label)
        ax.set_ylabel("REAL / SIM median")
        ax.grid(alpha=0.25, axis="y")

    for ax in axes.flat[len(metric_keys):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(x_label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
    fig.suptitle(
        "Task 4 per-file time series relative to simulation reference\n"
        "Checks whether the mismatch is systematic across files or concentrated in a few runs",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_dual_figure(fig, out_dir, "task4_projection_ratio_timeseries")


def plot_task4_projection_summary(
    tt_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    tdif_summary: pd.DataFrame,
    *,
    out_dir: Path,
    sim_label: str,
    real_label: str,
) -> None:
    if tt_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    width = 0.36
    x = np.arange(len(TT_ORDER), dtype=float)

    overall_row = tdif_summary[tdif_summary["level"] == "overall"]
    implied_x_scale = (
        float(overall_row["implied_real_x_scale_from_tdif"].iloc[0])
        if not overall_row.empty
        else np.nan
    )
    real_global_row = global_summary[global_summary["group_kind"] == "REAL"]
    real_global_scale = (
        float(real_global_row["weighted_mean_global_xproj_scaling_factor"].iloc[0])
        if not real_global_row.empty
        else np.nan
    )

    for idx, group_kind in enumerate(("SIM", "REAL")):
        subset = (
            tt_summary[tt_summary["group_kind"] == group_kind]
            .set_index("tt")
            .reindex(TT_ORDER)
        )
        centers = x + (idx - 0.5) * width
        color = GROUP_COLORS[group_kind]

        median_ratio = subset["median_fwhm_halfwidth_x_over_y"].to_numpy(dtype=float)
        ratio_p25 = subset["p25_fwhm_halfwidth_x_over_y"].to_numpy(dtype=float)
        ratio_p75 = subset["p75_fwhm_halfwidth_x_over_y"].to_numpy(dtype=float)
        ratio_yerr = np.vstack([median_ratio - ratio_p25, ratio_p75 - median_ratio])
        axes[0].bar(centers, median_ratio, width=width, color=color, alpha=0.85, label=group_kind)
        axes[0].errorbar(
            centers,
            median_ratio,
            yerr=ratio_yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
        )

        median_scale = subset["median_xproj_scaling_factor_to_match_y"].to_numpy(dtype=float)
        scale_p25 = subset["p25_xproj_scaling_factor_to_match_y"].to_numpy(dtype=float)
        scale_p75 = subset["p75_xproj_scaling_factor_to_match_y"].to_numpy(dtype=float)
        scale_yerr = np.vstack([median_scale - scale_p25, scale_p75 - median_scale])
        axes[1].bar(centers, median_scale, width=width, color=color, alpha=0.85, label=group_kind)
        axes[1].errorbar(
            centers,
            median_scale,
            yerr=scale_yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
        )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(TT_ORDER)
        ax.grid(alpha=0.25, axis="y")
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
    axes[0].set_title("Task 4 ellipse width ratio")
    axes[0].set_ylabel("median FWHM x/y")
    axes[1].set_title("Task 4 x scale needed to match y")
    axes[1].set_ylabel("median xproj scale factor")
    if np.isfinite(implied_x_scale):
        axes[1].axhline(
            implied_x_scale,
            color=GROUP_COLORS["TASK2"],
            lw=1.6,
            ls=":",
            label="Task 2 implied x scale",
        )
    if np.isfinite(real_global_scale):
        axes[1].text(
            0.02,
            0.98,
            f"REAL global weighted mean = {real_global_scale:.4f}",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
        )
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "Projection-axis mismatch bridge\n"
        f"Simulation={sim_label}  Real={real_label}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_dual_figure(fig, out_dir, "task4_projection_ratio_scale_by_tt")


def plot_task2_tdif_summary(
    tdif_summary: pd.DataFrame,
    *,
    out_dir: Path,
    abs_quantile: float,
) -> None:
    if tdif_summary.empty:
        return
    plane_df = tdif_summary[tdif_summary["level"] == "plane"].sort_values("plane")
    overall_df = tdif_summary[tdif_summary["level"] == "overall"]
    strip_df = tdif_summary[tdif_summary["level"] == "plane_strip"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    labels = ["ALL", "P1", "P2", "P3", "P4"]
    values = []
    if not overall_df.empty:
        values.append(float(overall_df["implied_real_x_scale_from_tdif"].iloc[0]))
    else:
        values.append(np.nan)
    values.extend(plane_df["implied_real_x_scale_from_tdif"].to_list())
    axes[0].bar(
        np.arange(len(labels)),
        values,
        color="#4c78a8",
        alpha=0.9,
    )
    axes[0].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("implied real-data x scale")
    axes[0].set_title(
        f"From 1 / real-over-sim q{int(round(abs_quantile * 100))}(|T_dif|)"
    )
    axes[0].grid(alpha=0.25, axis="y")

    heatmap = np.full((4, 4), np.nan, dtype=float)
    for _, row in strip_df.iterrows():
        plane = int(row["plane"])
        strip = int(row["strip"])
        heatmap[plane - 1, strip - 1] = float(row["implied_real_x_scale_from_tdif"])
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size:
        span = max(float(np.nanmax(np.abs(finite - 1.0))), 0.03)
        vmin = 1.0 - span
        vmax = 1.0 + span
    else:
        vmin, vmax = 0.9, 1.1
    image = axes[1].imshow(
        heatmap,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels([1, 2, 3, 4])
    axes[1].set_yticks(np.arange(4))
    axes[1].set_yticklabels([1, 2, 3, 4])
    axes[1].set_xlabel("strip")
    axes[1].set_ylabel("plane")
    axes[1].set_title("Per-plane / per-strip implied x scale")
    for plane in range(4):
        for strip in range(4):
            value = heatmap[plane, strip]
            if np.isfinite(value):
                axes[1].text(
                    strip,
                    plane,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("implied real-data x scale")

    fig.suptitle("Task 2 calibrated T_dif evidence for the x-axis scale", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_dual_figure(fig, out_dir, "task2_tdif_implied_x_scale")


def plot_strip_balance_summary(
    strip_balance_detail: pd.DataFrame,
    strip_balance_plane: pd.DataFrame,
    *,
    out_dir: Path,
    balance_mode: str,
) -> None:
    if strip_balance_detail.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharey=True)
    axes = axes.flatten()
    x = np.arange(4, dtype=float)
    width = 0.36
    for plane in range(1, 5):
        ax = axes[plane - 1]
        detail = (
            strip_balance_detail[strip_balance_detail["plane"] == plane]
            .sort_values("strip")
        )
        plane_summary = strip_balance_plane[strip_balance_plane["plane"] == plane]
        sim_share = detail["sim_share"].to_numpy(dtype=float)
        real_share = detail["real_share"].to_numpy(dtype=float)
        ax.bar(x - 0.5 * width, sim_share, width=width, color=GROUP_COLORS["SIM"], alpha=0.85, label="SIM")
        ax.bar(x + 0.5 * width, real_share, width=width, color=GROUP_COLORS["REAL"], alpha=0.85, label="REAL")
        ax.set_xticks(x)
        ax.set_xticklabels([1, 2, 3, 4])
        ax.set_ylim(0, max(0.35, float(np.nanmax([sim_share, real_share])) * 1.18))
        ax.grid(alpha=0.2, axis="y")
        jsd = float(plane_summary["js_divergence"].iloc[0]) if not plane_summary.empty else np.nan
        ax.set_title(f"Plane {plane}\nJSD={jsd:.4f}" if np.isfinite(jsd) else f"Plane {plane}")
        ax.set_xlabel("strip")
        if plane in (1, 3):
            ax.set_ylabel("share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)
    fig.suptitle(
        "Task 3-side strip balance sanity check\n"
        f"Mode={balance_mode}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_dual_figure(fig, out_dir, "task3_strip_balance_sanity")


def plot_bridge_summary(
    global_summary: pd.DataFrame,
    tdif_summary: pd.DataFrame,
    strip_balance_plane: pd.DataFrame,
    *,
    out_dir: Path,
    tolerance: float,
    jsd_warning_threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    sim_row = global_summary[global_summary["group_kind"] == "SIM"]
    real_row = global_summary[global_summary["group_kind"] == "REAL"]
    overall_row = tdif_summary[tdif_summary["level"] == "overall"]
    bars = [
        (
            "SIM Task4\nweighted global",
            float(sim_row["weighted_mean_global_xproj_scaling_factor"].iloc[0]) if not sim_row.empty else np.nan,
            GROUP_COLORS["SIM"],
        ),
        (
            "REAL Task4\nweighted global",
            float(real_row["weighted_mean_global_xproj_scaling_factor"].iloc[0]) if not real_row.empty else np.nan,
            GROUP_COLORS["REAL"],
        ),
        (
            "REAL Task2\nimplied x scale",
            float(overall_row["implied_real_x_scale_from_tdif"].iloc[0]) if not overall_row.empty else np.nan,
            GROUP_COLORS["TASK2"],
        ),
    ]
    axes[0].bar(
        np.arange(len(bars)),
        [item[1] for item in bars],
        color=[item[2] for item in bars],
        alpha=0.9,
    )
    axes[0].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[0].set_xticks(np.arange(len(bars)))
    axes[0].set_xticklabels([item[0] for item in bars])
    axes[0].set_ylabel("x scale factor")
    axes[0].set_title("Task 4 vs Task 2 bridge")
    axes[0].grid(alpha=0.25, axis="y")

    plane_summary = strip_balance_plane.sort_values("plane")
    axes[1].bar(
        plane_summary["plane"].astype(str),
        plane_summary["js_divergence"].to_numpy(dtype=float),
        color="#7f7f7f",
        alpha=0.9,
    )
    axes[1].axhline(jsd_warning_threshold, color="#d62728", lw=1.2, ls=":")
    axes[1].set_ylabel("Jensen-Shannon divergence")
    axes[1].set_title("Strip-balance mismatch by plane")
    axes[1].grid(alpha=0.25, axis="y")

    if not real_row.empty and not overall_row.empty:
        agreement = abs(
            float(real_row["weighted_mean_global_xproj_scaling_factor"].iloc[0])
            - float(overall_row["implied_real_x_scale_from_tdif"].iloc[0])
        )
        axes[0].text(
            0.98,
            0.98,
            f"|Task4-Task2| = {agreement:.4f}\nthreshold = {tolerance:.4f}",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    fig.suptitle("Projection-axis mismatch decision summary", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_dual_figure(fig, out_dir, "projection_axis_bridge_summary")


def build_text_report(
    *,
    sim_label: str,
    real_label: str,
    selection,
    cfg: ProjectionAxisMismatchConfig,
    task4_tt_summary: pd.DataFrame,
    task4_global_summary: pd.DataFrame,
    task4_timeseries_summary: pd.DataFrame,
    tdif_summary: pd.DataFrame,
    strip_balance_plane: pd.DataFrame,
) -> str:
    real_task4_global = task4_global_summary[
        task4_global_summary["group_kind"] == "REAL"
    ]
    sim_task4_global = task4_global_summary[
        task4_global_summary["group_kind"] == "SIM"
    ]
    overall_tdif = tdif_summary[tdif_summary["level"] == "overall"]
    plane_tdif = tdif_summary[tdif_summary["level"] == "plane"].sort_values("plane")

    real_global_scale = (
        float(real_task4_global["weighted_mean_global_xproj_scaling_factor"].iloc[0])
        if not real_task4_global.empty
        else np.nan
    )
    sim_global_scale = (
        float(sim_task4_global["weighted_mean_global_xproj_scaling_factor"].iloc[0])
        if not sim_task4_global.empty
        else np.nan
    )
    task2_implied_scale = (
        float(overall_tdif["implied_real_x_scale_from_tdif"].iloc[0])
        if not overall_tdif.empty
        else np.nan
    )
    agreement = (
        abs(real_global_scale - task2_implied_scale)
        if np.isfinite(real_global_scale) and np.isfinite(task2_implied_scale)
        else np.nan
    )
    plane_jsd_values = strip_balance_plane["js_divergence"].to_numpy(dtype=float)
    median_jsd = float(np.nanmedian(plane_jsd_values)) if plane_jsd_values.size else np.nan
    max_jsd = float(np.nanmax(plane_jsd_values)) if plane_jsd_values.size else np.nan

    strongest_tt = None
    real_tt = task4_tt_summary[task4_tt_summary["group_kind"] == "REAL"]
    if not real_tt.empty:
        strongest_row = real_tt.sort_values(
            "weighted_mean_xproj_scaling_factor_to_match_y",
            ascending=False,
        ).iloc[0]
        strongest_tt = (
            str(strongest_row["tt"]),
            float(strongest_row["weighted_mean_xproj_scaling_factor_to_match_y"]),
        )

    strongest_plane = None
    if not plane_tdif.empty:
        plane_row = plane_tdif.sort_values(
            "implied_real_x_scale_from_tdif",
            ascending=False,
        ).iloc[0]
        strongest_plane = (
            int(plane_row["plane"]),
            float(plane_row["implied_real_x_scale_from_tdif"]),
        )

    global_ts = task4_timeseries_summary[
        task4_timeseries_summary["metric_key"] == "global_xproj_scaling_factor"
    ]
    tt1234_ts = task4_timeseries_summary[
        task4_timeseries_summary["metric_key"] == "tt_1234_fwhm_halfwidth_x_over_y"
    ]
    global_ts_row = global_ts.iloc[0] if not global_ts.empty else None
    tt1234_ts_row = tt1234_ts.iloc[0] if not tt1234_ts.empty else None

    lines = [
        "Projection-axis mismatch study",
        "",
        f"Simulation group: {sim_label}",
        f"Real-data group: {real_label}",
        f"Simulation stations: {', '.join(selection.simulation_stations)}",
        f"Real stations: {', '.join(selection.real_stations)}",
        f"Task 4 minimum fitted points per TT: {cfg.task4_min_points_per_tt}",
        f"Task 2 absolute quantile: q{int(round(cfg.task2_abs_quantile * 100))}(|T_dif|)",
        f"Strip-balance mode: {cfg.strip_balance_mode}",
        "",
        "Task 4 evidence",
        f"- SIM weighted global x scale: {sim_global_scale:.6f}" if np.isfinite(sim_global_scale) else "- SIM weighted global x scale: NaN",
        f"- REAL weighted global x scale: {real_global_scale:.6f}" if np.isfinite(real_global_scale) else "- REAL weighted global x scale: NaN",
        f"- strongest REAL TT mismatch: TT {strongest_tt[0]} -> {strongest_tt[1]:.6f}" if strongest_tt else "- strongest REAL TT mismatch: NaN",
        "",
        "Task 4 time-series consistency",
        (
            f"- global x scale outside SIM IQR on the {global_ts_row['dominant_outside_side']} side: "
            f"{int(global_ts_row['n_above_sim_iqr_band']) if global_ts_row['dominant_outside_side'] == 'above' else int(global_ts_row['n_below_sim_iqr_band'])}/"
            f"{int(global_ts_row['n_real_files'])} files"
        ) if global_ts_row is not None else "- global x scale outside SIM IQR: NaN",
        (
            f"- global x scale median REAL/SIM ratio: {float(global_ts_row['median_ratio_to_sim_median']):.6f} "
            f"(first half {float(global_ts_row['first_half_median_ratio_to_sim_median']):.6f}, "
            f"last half {float(global_ts_row['last_half_median_ratio_to_sim_median']):.6f})"
        ) if global_ts_row is not None else "- global x scale REAL/SIM ratio: NaN",
        (
            f"- TT1234 x/y outside SIM IQR on the {tt1234_ts_row['dominant_outside_side']} side: "
            f"{int(tt1234_ts_row['n_above_sim_iqr_band']) if tt1234_ts_row['dominant_outside_side'] == 'above' else int(tt1234_ts_row['n_below_sim_iqr_band'])}/"
            f"{int(tt1234_ts_row['n_real_files'])} files"
        ) if tt1234_ts_row is not None else "- TT1234 x/y outside SIM IQR: NaN",
        (
            f"- TT1234 x/y median REAL/SIM ratio: {float(tt1234_ts_row['median_ratio_to_sim_median']):.6f} "
            f"(first half {float(tt1234_ts_row['first_half_median_ratio_to_sim_median']):.6f}, "
            f"last half {float(tt1234_ts_row['last_half_median_ratio_to_sim_median']):.6f})"
        ) if tt1234_ts_row is not None else "- TT1234 x/y REAL/SIM ratio: NaN",
        "",
        "Task 2 bridge",
        f"- implied REAL x scale from T_dif: {task2_implied_scale:.6f}" if np.isfinite(task2_implied_scale) else "- implied REAL x scale from T_dif: NaN",
        f"- |Task4 global - Task2 implied|: {agreement:.6f}" if np.isfinite(agreement) else "- |Task4 global - Task2 implied|: NaN",
        f"- strongest plane-level implied x scale: plane {strongest_plane[0]} -> {strongest_plane[1]:.6f}" if strongest_plane else "- strongest plane-level implied x scale: NaN",
        "",
        "Task 3-side strip-balance sanity",
        f"- median plane JSD: {median_jsd:.6f}" if np.isfinite(median_jsd) else "- median plane JSD: NaN",
        f"- max plane JSD: {max_jsd:.6f}" if np.isfinite(max_jsd) else "- max plane JSD: NaN",
        "",
        "Interpretation",
    ]

    if (
        np.isfinite(agreement)
        and np.isfinite(median_jsd)
        and agreement <= cfg.scale_agreement_tolerance
        and median_jsd <= cfg.strip_balance_jsd_warning_threshold
    ):
        lines.extend(
            [
                "- The Task 4 x-rescaling needed in real data agrees with the Task 2 T_dif-based prediction within tolerance.",
                "- The y-side strip occupancy mismatch is comparatively small.",
                "- Current evidence points more strongly to the x / T_dif / active-length side than to the y / strip-assignment side.",
            ]
        )
        if (
            global_ts_row is not None
            and tt1234_ts_row is not None
            and float(global_ts_row["dominant_outside_fraction"]) >= 0.75
            and float(tt1234_ts_row["dominant_outside_fraction"]) >= 0.75
        ):
            lines.append(
                "- The time series also looks systematic rather than file-localized: most real files sit on the same side of the simulation band."
            )
        elif global_ts_row is not None and tt1234_ts_row is not None:
            lines.append(
                "- The time series is more mixed, so the mismatch may be amplified by a subset of files and should not be corrected blindly."
            )
    else:
        lines.extend(
            [
                "- The bridge is not yet clean enough to isolate the issue to x only.",
                "- Either the Task 4 and Task 2 scale estimates disagree beyond tolerance, or the strip-balance mismatch is not small.",
                "- Recheck per-plane T_dif calibration and plane-combination dependence before changing geometry globally.",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge the Task 4 projection-ellipse asymmetry to Task 2 calibrated "
            "T_dif widths and a Task 3-side strip-balance sanity check."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "MASTER" / "ANCILLARY" / "SIMULATION_TUNING" / "config_simulation_tuning.yaml"),
        help="Path to the shared simulation-tuning YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_tuning_config(args.config)
    selection = resolve_selection(config)
    study_cfg = load_projection_axis_config(config)

    sim_group_name = group_label(selection.simulation_stations, fallback="SIMULATION")
    real_group_name = group_label(selection.real_stations, fallback="REAL_AGG")

    out_dir = Path(args.output_dir) if args.output_dir else output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    task4_sim = load_task4_projection_metadata(
        selection.simulation_stations,
        selection.simulation_date_ranges,
    )
    task4_real = load_task4_projection_metadata(
        selection.real_stations,
        selection.real_date_ranges,
    )
    task4_tt_sim, task4_global_sim = summarize_task4_projection_metrics(
        task4_sim,
        group_kind="SIM",
        group_name=sim_group_name,
        cfg=study_cfg,
    )
    task4_tt_real, task4_global_real = summarize_task4_projection_metrics(
        task4_real,
        group_kind="REAL",
        group_name=real_group_name,
        cfg=study_cfg,
    )
    task4_tt_summary = pd.concat(
        [task4_tt_sim, task4_tt_real],
        ignore_index=True,
    ) if (not task4_tt_sim.empty or not task4_tt_real.empty) else pd.DataFrame()
    task4_global_summary = pd.concat(
        [task4_global_sim, task4_global_real],
        ignore_index=True,
    ) if (not task4_global_sim.empty or not task4_global_real.empty) else pd.DataFrame()
    task4_time_series, task4_reference_summary, task4_timeseries_summary = build_task4_timeseries_tables(
        task4_sim,
        task4_real,
        cfg=study_cfg,
    )

    sim_axis_payload = collect_calibrated_axis_payload(
        selection.simulation_stations,
        selection.simulation_date_ranges,
        strip_balance_mode=study_cfg.strip_balance_mode,
    )
    real_axis_payload = collect_calibrated_axis_payload(
        selection.real_stations,
        selection.real_date_ranges,
        strip_balance_mode=study_cfg.strip_balance_mode,
    )
    tdif_summary = summarize_task2_tdif_mismatch(
        sim_axis_payload,
        real_axis_payload,
        abs_quantile=study_cfg.task2_abs_quantile,
    )
    strip_balance_detail, strip_balance_plane = summarize_strip_balance(
        sim_axis_payload,
        real_axis_payload,
    )

    if not task4_tt_summary.empty:
        task4_tt_summary.to_csv(out_dir / "task4_projection_ratio_summary.csv", index=False)
    if not task4_global_summary.empty:
        task4_global_summary.to_csv(out_dir / "task4_projection_global_summary.csv", index=False)
    if not task4_time_series.empty:
        task4_time_series.to_csv(out_dir / "task4_projection_timeseries_real_vs_sim.csv", index=False)
    if not task4_reference_summary.empty:
        task4_reference_summary.to_csv(out_dir / "task4_projection_timeseries_reference_summary.csv", index=False)
    if not task4_timeseries_summary.empty:
        task4_timeseries_summary.to_csv(out_dir / "task4_projection_timeseries_stability_summary.csv", index=False)
    tdif_summary.to_csv(out_dir / "task2_tdif_projection_bridge_summary.csv", index=False)
    strip_balance_detail.to_csv(out_dir / "task3_strip_balance_detail.csv", index=False)
    strip_balance_plane.to_csv(out_dir / "task3_strip_balance_plane_summary.csv", index=False)

    plot_task4_projection_summary(
        task4_tt_summary,
        task4_global_summary,
        tdif_summary,
        out_dir=out_dir,
        sim_label=sim_group_name,
        real_label=real_group_name,
    )
    plot_task4_projection_timeseries(
        task4_time_series,
        task4_reference_summary,
        out_dir=out_dir,
        real_label=real_group_name,
    )
    plot_task2_tdif_summary(
        tdif_summary,
        out_dir=out_dir,
        abs_quantile=study_cfg.task2_abs_quantile,
    )
    plot_strip_balance_summary(
        strip_balance_detail,
        strip_balance_plane,
        out_dir=out_dir,
        balance_mode=study_cfg.strip_balance_mode,
    )
    plot_bridge_summary(
        task4_global_summary,
        tdif_summary,
        strip_balance_plane,
        out_dir=out_dir,
        tolerance=study_cfg.scale_agreement_tolerance,
        jsd_warning_threshold=study_cfg.strip_balance_jsd_warning_threshold,
    )

    report = build_text_report(
        sim_label=sim_group_name,
        real_label=real_group_name,
        selection=selection,
        cfg=study_cfg,
        task4_tt_summary=task4_tt_summary,
        task4_global_summary=task4_global_summary,
        task4_timeseries_summary=task4_timeseries_summary,
        tdif_summary=tdif_summary,
        strip_balance_plane=strip_balance_plane,
    )
    (out_dir / "projection_axis_mismatch_report.txt").write_text(report, encoding="utf-8")

    print(f"Projection-axis mismatch study complete. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
