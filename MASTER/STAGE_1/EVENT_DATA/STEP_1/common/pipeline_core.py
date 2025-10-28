#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared stage logic for STEP_1 RAW→LIST conversion.

The functions defined here are refactored and trimmed versions of the logic
originally present in the monolithic raw_to_list.py script. They operate on
plain pandas DataFrames and configuration dictionaries so each TASK script can
remain lightweight and focused on orchestration (queue management, file I/O,
plot emission).
"""

from __future__ import annotations

import builtins
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation

# =============================================================================
# Configuration containers
# =============================================================================


@dataclass
class ChargeCalibrationConfig:
    translate_charge_cal: float
    percentile: float
    rel_th: float
    rel_th_cal: float
    abs_th: float
    q_quantile: float
    thr_factor: float
    thr_factor_2: float
    q_left: float
    q_right: float
    pedestal_left: float
    pedestal_right: float


@dataclass
class TimeCalibrationConfig:
    T_F_left: float
    T_F_right: float
    T_B_left: float
    T_B_right: float
    T_F_left_ST: float
    T_F_right_ST: float
    T_B_left_ST: float
    T_B_right_ST: float
    T_sum_left_pre: float
    T_sum_right_pre: float
    T_diff_threshold: float
    coincidence_window_precal_ns: float


@dataclass
class PlotConfig:
    save_plots: bool
    show_plots: bool
    article_format: bool
    distance_sum_charges_plot: float
    distance_sum_charges_left_fit: float
    distance_sum_charges_right_fit: float
    distance_diff_charges_up_fit: float
    distance_diff_charges_low_fit: float
    front_back_fit_threshold: float
    scatter_xlim_left: float
    scatter_xlim_right: float
    scatter_ylim_bottom: float
    scatter_ylim_top: float
    num_bins: int
    log_scale: bool


@dataclass
class StageMetadata:
    station: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    execution_time: str
    date_execution: str


# =============================================================================
# Utility helpers (trimmed versions of monolithic functions)
# =============================================================================


def _apply_bounds(frame: pd.DataFrame, column_names: Iterable[str], lower: float, upper: float) -> None:
    cols = tuple(column_names)
    if not cols:
        return
    subset = frame.loc[:, cols]
    frame.loc[:, cols] = subset.where((subset >= lower) & (subset <= upper), 0)


def _collect_columns(columns: Iterable[str], pattern: str) -> List[str]:
    import re

    regex = re.compile(pattern)
    return [name for name in columns if regex.match(name)]


def load_itineraries_from_file(file_path: Path, required: bool = True) -> List[List[str]]:
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []
    itineraries: List[List[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        for raw_line in itinerary_file:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")
    return itineraries


def write_itineraries_to_file(file_path: Path, itineraries: Iterable[Iterable[str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique: Dict[Tuple[str, ...], None] = {}
    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if itinerary_tuple:
            unique.setdefault(itinerary_tuple, None)
    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")


def calibrate_strip_Q_pedestal(
    Q_ch: np.ndarray,
    T_ch: np.ndarray,
    Q_other: np.ndarray,
    cfg: ChargeCalibrationConfig,
    time_cfg: TimeCalibrationConfig,
    self_trigger_mode: bool = False,
) -> float:
    if self_trigger_mode:
        T_left_side = time_cfg.T_F_left_ST
        T_right_side = time_cfg.T_F_right_ST
    else:
        T_left_side = time_cfg.T_F_left
        T_right_side = time_cfg.T_F_right

    cond = (T_ch != 0) & (T_ch > T_left_side) & (T_ch < T_right_side)
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    Q_other = Q_other[cond]

    Q_dif = Q_ch - Q_other
    cond = (
        (Q_dif > np.percentile(Q_dif, cfg.percentile))
        & (Q_dif < np.percentile(Q_dif, 100 - cfg.percentile))
    )
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]

    counts, bin_edges = np.histogram(T_ch, bins="auto")
    max_counts = np.max(counts)
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return 0.0
    min_counts = np.min(nonzero)
    threshold = max_counts / cfg.thr_factor

    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]
    else:
        threshold = (min_counts + max_counts) / cfg.thr_factor_2
        indices_above_threshold = np.where(counts > threshold)[0]
        if indices_above_threshold.size == 0:
            return 0.0
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]

    Q_ch = Q_ch[(T_ch > min_bin_edge) & (T_ch < max_bin_edge)]

    Q_ch = Q_ch[Q_ch != 0]
    Q_ch = Q_ch[(Q_ch > cfg.q_left) & (Q_ch < cfg.q_right)]
    Q_ch = Q_ch[Q_ch > np.percentile(Q_ch, cfg.q_quantile)]
    if Q_ch.size == 0:
        return 0.0

    counts, bin_edges = np.histogram(Q_ch, bins="auto")
    max_counts = np.max(counts)
    counts = counts[counts < max_counts]
    if counts.size == 0:
        return np.percentile(Q_ch, 50)
    max_counts = np.max(counts)
    non_empty_bins = counts >= max(cfg.rel_th * max_counts, cfg.abs_th)

    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
        else:
            current_length = 0

    offset = bin_edges[start_index]

    Q_ch_cal = Q_ch - offset
    Q_ch_cal = Q_ch_cal[(Q_ch_cal > cfg.pedestal_left) & (Q_ch_cal < cfg.pedestal_right)]
    if Q_ch_cal.size == 0:
        return offset

    counts, bin_edges = np.histogram(Q_ch_cal, bins="auto")
    max_counts = np.max(counts)
    max_bin_index = np.argmax(counts)
    threshold = cfg.rel_th_cal * max_counts

    offset_bin_index = max_bin_index
    while offset_bin_index > 0 and counts[offset_bin_index] >= threshold:
        offset_bin_index -= 1

    offset_cal = bin_edges[offset_bin_index]
    pedestal = offset + offset_cal
    return pedestal - cfg.translate_charge_cal


def calibrate_strip_T_diff(T_F: np.ndarray, T_B: np.ndarray, cfg: TimeCalibrationConfig, self_trigger_mode: bool = False) -> float:
    if self_trigger_mode:
        left = cfg.T_F_left_ST
        right = cfg.T_F_right_ST
    else:
        left = cfg.T_F_left
        right = cfg.T_F_right

    cond = (T_F != 0) & (T_F > left) & (T_F < right) & (T_B != 0)
    if not np.any(cond):
        return 0.0

    values = ((T_B[cond] - T_F[cond]) / 2.0)
    if values.size == 0:
        return 0.0
    return np.median(values)


def create_original_tt(df: pd.DataFrame) -> pd.Series:
    def original_tt_from_row(row: pd.Series) -> int:
        name = ""
        for plane in range(1, 5):
            this_plane = False
            for strip in range(1, 5):
                q_sum_col = f"Q_P{plane}s{strip}"
                if row.get(q_sum_col, 0) != 0:
                    this_plane = True
            if this_plane:
                name += str(plane)
        return int(name) if name else 0

    return df.apply(original_tt_from_row, axis=1)


def compute_preprocessed_tt(row: pd.Series) -> int:
    name = ""
    for plane in range(1, 5):
        this_plane = False
        for strip in range(1, 5):
            col = f"T{plane}_T_sum_{strip}"
            if row.get(col, 0) != 0:
                this_plane = True
        if this_plane:
            name += str(plane)
    return int(name) if name else 0


def compute_posfiltered_tt(row: pd.Series) -> int:
    name = ""
    for plane in range(1, 5):
        this_plane = False
        for strip in range(1, 5):
            col = f"T{plane}_T_sum_{strip}"
            if row.get(col, 0) != 0:
                this_plane = True
        if this_plane:
            name += str(plane)
    return int(name) if name else 0


def zero_outlier_tsum(row: pd.Series, threshold: float) -> pd.Series:
    t_sum_cols = [col for col in row.index if "_T_sum_" in col]
    t_sum_vals = row[t_sum_cols].copy()
    nonzero_vals = t_sum_vals[t_sum_vals != 0]
    if len(nonzero_vals) < 2:
        return row
    center = np.median(nonzero_vals)
    deviations = np.abs(nonzero_vals - center)
    outliers = deviations > threshold / 2
    for col in outliers.index[outliers]:
        row[col] = 0.0
    return row


def interpolate_fast_charge(width: np.ndarray, cfg: dict) -> CubicSpline:
    values = width[(width != 0) & (width > cfg["Q_clip_min"]) & (width < cfg["Q_clip_max"])]
    hist, bin_edges = np.histogram(values, bins=cfg["num_bins"])
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cumulative = np.cumsum(hist)
    cumulative = cumulative / np.max(cumulative)
    return CubicSpline(centers, cumulative, extrapolate=True)


# =============================================================================
# Stage 1 – raw_to_clean and Stage 2 – clean_to_cal
# =============================================================================


def stage1_clean_raw(
    coincidence_df: pd.DataFrame,
    self_trigger_df: Optional[pd.DataFrame],
    station: str,
    cfg: dict,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    coincidence_df = coincidence_df.reset_index(drop=True)
    if self_trigger_df is not None:
        self_trigger_df = self_trigger_df.reset_index(drop=True)
    return coincidence_df, self_trigger_df


def stage2_clean_to_cal(
    working_df: pd.DataFrame,
    config: dict,
    metadata: StageMetadata,
) -> pd.DataFrame:
    charge_cfg = ChargeCalibrationConfig(
        translate_charge_cal=config["calibrate_strip_Q_pedestal_translate_charge_cal"],
        percentile=config["calibrate_strip_Q_pedestal_percentile"],
        rel_th=config["calibrate_strip_Q_pedestal_rel_th"],
        rel_th_cal=config["calibrate_strip_Q_pedestal_rel_th_cal"],
        abs_th=config["calibrate_strip_Q_pedestal_abs_th"],
        q_quantile=config["calibrate_strip_Q_pedestal_q_quantile"],
        thr_factor=config["calibrate_strip_Q_pedestal_thr_factor"],
        thr_factor_2=config["calibrate_strip_Q_pedestal_thr_factor_2"],
        q_left=config["Q_left_pre_cal"],
        q_right=config["Q_right_pre_cal"],
        pedestal_left=config["pedestal_left"],
        pedestal_right=config["pedestal_right"],
    )
    time_cfg = TimeCalibrationConfig(
        T_F_left=config["T_side_left_pre_cal_default"],
        T_F_right=config["T_side_right_pre_cal_default"],
        T_B_left=config["T_side_left_pre_cal_default"],
        T_B_right=config["T_side_right_pre_cal_default"],
        T_F_left_ST=config["T_side_left_pre_cal_ST"],
        T_F_right_ST=config["T_side_right_pre_cal_ST"],
        T_B_left_ST=config["T_side_left_pre_cal_ST"],
        T_B_right_ST=config["T_side_right_pre_cal_ST"],
        T_sum_left_pre=config["T_sum_left_pre_cal"],
        T_sum_right_pre=config["T_sum_right_pre_cal"],
        T_diff_threshold=config["T_diff_cal_threshold"],
        coincidence_window_precal_ns=config["coincidence_window_precal_ns"],
    )

    working_df = working_df.copy()
    working_df.fillna(0, inplace=True)

    T_F_cols = _collect_columns(working_df.columns, r"^T\d+_F_\d+$")
    T_B_cols = _collect_columns(working_df.columns, r"^T\d+_B_\d+$")
    Q_F_cols = _collect_columns(working_df.columns, r"^Q\d+_F_\d+$")
    Q_B_cols = _collect_columns(working_df.columns, r"^Q\d+_B_\d+$")

    _apply_bounds(working_df, T_F_cols, time_cfg.T_F_left, time_cfg.T_F_right)
    _apply_bounds(working_df, T_B_cols, time_cfg.T_B_left, time_cfg.T_B_right)
    _apply_bounds(working_df, Q_F_cols, charge_cfg.q_left, charge_cfg.q_right)
    _apply_bounds(working_df, Q_B_cols, charge_cfg.q_left, charge_cfg.q_right)

    charge_test = working_df.copy()
    for idx, plane in enumerate(["1", "2", "3", "4"]):
        Q_F_plane = np.stack([working_df[f"Q{plane}_F_{i+1}"].values for i in range(4)], axis=1)
        Q_B_plane = np.stack([working_df[f"Q{plane}_B_{i+1}"].values for i in range(4)], axis=1)
        T_F_plane = np.stack([working_df[f"T{plane}_F_{i+1}"].values for i in range(4)], axis=1)
        T_B_plane = np.stack([working_df[f"T{plane}_B_{i+1}"].values for i in range(4)], axis=1)

        for strip in range(4):
            pedestal_F = calibrate_strip_Q_pedestal(
                Q_F_plane[:, strip],
                T_F_plane[:, strip],
                Q_B_plane[:, strip],
                charge_cfg,
                time_cfg,
            )
            mask_F = charge_test[f"Q{plane}_F_{strip+1}"] != 0
            charge_test.loc[mask_F, f"Q{plane}_F_{strip+1}"] -= pedestal_F

            pedestal_B = calibrate_strip_Q_pedestal(
                Q_B_plane[:, strip],
                T_B_plane[:, strip],
                Q_F_plane[:, strip],
                charge_cfg,
                time_cfg,
            )
            mask_B = charge_test[f"Q{plane}_B_{strip+1}"] != 0
            charge_test.loc[mask_B, f"Q{plane}_B_{strip+1}"] -= pedestal_B

    working_df = charge_test

    pos_test = working_df.copy()
    for i, plane in enumerate(["1", "2", "3", "4"]):
        for strip in range(4):
            pos_test[f"T{plane}_diff_{strip+1}"] = (pos_test[f"T{plane}_B_{strip+1}"] - pos_test[f"T{plane}_F_{strip+1}"]) / 2

    for i, plane in enumerate(["1", "2", "3", "4"]):
        T_F_plane = np.stack([working_df[f"T{plane}_F_{strip+1}"].values for strip in range(4)], axis=1)
        T_B_plane = np.stack([working_df[f"T{plane}_B_{strip+1}"].values for strip in range(4)], axis=1)
        offsets = [
            calibrate_strip_T_diff(
                T_F_plane[:, strip],
                T_B_plane[:, strip],
                time_cfg,
            )
            for strip in range(4)
        ]
        for strip, offset in enumerate(offsets):
            mask = pos_test[f"T{plane}_diff_{strip+1}"] != 0
            pos_test.loc[mask, f"T{plane}_diff_{strip+1}"] -= offset

    working_df = working_df.copy()
    working_df["original_tt"] = create_original_tt(working_df)
    working_df["preprocessed_tt"] = working_df.apply(
        compute_preprocessed_tt,
        axis=1,
    )
    working_df["posfiltered_tt"] = working_df.apply(
        compute_posfiltered_tt,
        axis=1,
    )
    working_df = working_df.apply(
        zero_outlier_tsum,
        axis=1,
        threshold=time_cfg.coincidence_window_precal_ns,
    )
    return working_df


# =============================================================================
# Stage 3 and 4 placeholders (these will be filled in later)
# =============================================================================


def stage3_cal_to_list(
    calibrated_df: pd.DataFrame,
    config: dict,
    metadata: StageMetadata,
) -> pd.DataFrame:
    # TODO: implement based on monolithic logic (e.g., y position, list conversion)
    return calibrated_df.copy()


def stage4_list_to_fit(
    listed_df: pd.DataFrame,
    config: dict,
    metadata: StageMetadata,
) -> pd.DataFrame:
    # TODO: implement based on monolithic logic (e.g., fitting, final outputs)
    return listed_df.copy()
