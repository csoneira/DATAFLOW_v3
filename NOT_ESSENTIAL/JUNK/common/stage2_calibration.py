#!/usr/bin/env python3
"""Stage 2 calibration logic ported from the legacy raw_to_list.py script."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import poisson

from .pipeline_core import (
    calibrate_strip_Q_pedestal,
    calibrate_strip_T_diff,
    ChargeCalibrationConfig,
    TimeCalibrationConfig,
    zero_outlier_tsum,
    interpolate_fast_charge,
)


@dataclass
class Stage2CalibrationResult:
    dataframe: pd.DataFrame
    plots: List[str]
    metadata: Dict[str, object]


def _stage1_flag(config: dict, key: str, default: bool = False) -> bool:
    stage_key = f"stage1_{key}"
    if stage_key in config:
        return bool(config[stage_key])
    return bool(config.get(key, default))


def _finalize_plot(fig: plt.Figure, *, save_path: Optional[Path], show: bool) -> Optional[str]:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return str(save_path) if save_path is not None else None


def _build_y_positions(config: dict) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    wide_strip = config["wide_strip"]
    narrow_strip = config["narrow_strip"]

    y_widths = [
        np.array([wide_strip, wide_strip, wide_strip, narrow_strip], dtype=np.float64),
        np.array([narrow_strip, wide_strip, wide_strip, wide_strip], dtype=np.float64),
    ]

    def y_pos(width):
        return np.cumsum(width) - (np.sum(width) + width) / 2

    y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
    y_lookup = {1: y_pos_T[0], 2: y_pos_T[1], 3: y_pos_T[0], 4: y_pos_T[1]}
    return y_pos_T[0], y_pos_T[1], y_lookup


def _perform_charge_pedestal_calibration(
    working_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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

    charge_test = working_df.copy()
    charge_test_copy = charge_test.copy()

    QF_pedestal: List[List[float]] = []
    QB_pedestal: List[List[float]] = []
    for key in ["1", "2", "3", "4"]:
        Q_F_cols = [f"Q{key}_F_{i+1}" for i in range(4)]
        Q_B_cols = [f"Q{key}_B_{i+1}" for i in range(4)]
        T_F_cols = [f"T{key}_F_{i+1}" for i in range(4)]
        T_B_cols = [f"T{key}_B_{i+1}" for i in range(4)]

        Q_F = working_df[Q_F_cols].values
        Q_B = working_df[Q_B_cols].values
        T_F = working_df[T_F_cols].values
        T_B = working_df[T_B_cols].values

        QF_pedestal.append(
            [
                calibrate_strip_Q_pedestal(Q_F[:, i], T_F[:, i], Q_B[:, i], charge_cfg, time_cfg)
                for i in range(4)
            ]
        )
        QB_pedestal.append(
            [
                calibrate_strip_Q_pedestal(Q_B[:, i], T_B[:, i], Q_F[:, i], charge_cfg, time_cfg)
                for i in range(4)
            ]
        )

    QF_arr = np.array(QF_pedestal)
    QB_arr = np.array(QB_pedestal)

    for i, key in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        for j in range(4):
            mask = charge_test_copy[f"{key}_F_{j+1}"] != 0
            charge_test.loc[mask, f"{key}_F_{j+1}"] -= QF_arr[i, j]

    for i, key in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        for j in range(4):
            mask = charge_test_copy[f"{key}_B_{j+1}"] != 0
            charge_test.loc[mask, f"{key}_B_{j+1}"] -= QB_arr[i, j]

    return charge_test, QF_arr, QB_arr


def _perform_time_diff_calibration(
    working_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, np.ndarray]:
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

    pos_test = working_df.copy()
    for plane, key in enumerate(["T1", "T2", "T3", "T4"], start=1):
        for strip in range(4):
            pos_test[f"{key}_diff_{strip+1}"] = (
                pos_test[f"{key}_B_{strip+1}"] - pos_test[f"{key}_F_{strip+1}"]
            ) / 2

    pos_test_copy = pos_test.copy()
    Tdiff_cal = []
    for key in ["1", "2", "3", "4"]:
        T_F_cols = [f"T{key}_F_{i+1}" for i in range(4)]
        T_B_cols = [f"T{key}_B_{i+1}" for i in range(4)]
        T_F = working_df[T_F_cols].values
        T_B = working_df[T_B_cols].values
        Tdiff_cal.append(
            [
                calibrate_strip_T_diff(T_F[:, i], T_B[:, i], time_cfg)
                for i in range(4)
            ]
        )
    Tdiff_cal = np.array(Tdiff_cal)

    for i, key in enumerate(["T1", "T2", "T3", "T4"]):
        for j in range(4):
            mask = pos_test_copy[f"{key}_diff_{j+1}"] != 0
            pos_test.loc[mask, f"{key}_diff_{j+1}"] -= Tdiff_cal[i, j]

    return pos_test, Tdiff_cal


def run_stage2_calibration(
    working_df: pd.DataFrame,
    config: dict,
    plots_dir: Path,
    output_name: str,
    *,
    station: str,
    start_time: Optional[str],
) -> Stage2CalibrationResult:
    show_plots = _stage1_flag(config, "show_plots", False)
    save_plots = _stage1_flag(config, "save_plots", True)

    plot_root = plots_dir / output_name
    plot_paths: List[str] = []

    charge_test, QF_pedestal, QB_pedestal = _perform_charge_pedestal_calibration(working_df, config)

    q_min_settings = _resolve_charge_settings(config, debug=bool(config.get("debug_mode", False)))
    title = f"Pedestal-subtracted charge — Station {station}"
    if start_time:
        title += f"\nStart {start_time}"

    if save_plots:
        save_path = plot_root / "calibration_q_pedestal.png"
    else:
        save_path = None
    path = _plot_charge_grid(
        charge_test,
        q_min=q_min_settings["q_clip_min"],
        q_max=q_min_settings["q_clip_max"],
        num_bins=q_min_settings["num_bins"],
        log_scale=q_min_settings["log_scale"],
        title=title,
        save_path=save_path,
        show_plots=show_plots,
    )
    if path:
        plot_paths.append(path)

    pedestal_left = config.get("pedestal_left", -5)
    pedestal_right = config.get("pedestal_right", 5)

    if save_plots:
        save_path = plot_root / "calibration_q_pedestal_zoom.png"
    else:
        save_path = None
    path = _plot_charge_grid(
        charge_test,
        q_min=pedestal_left,
        q_max=pedestal_right,
        num_bins=q_min_settings["num_bins"],
        log_scale=q_min_settings["log_scale"],
        title=f"Pedestal-subtracted charge (zoom) — Station {station}",
        save_path=save_path,
        show_plots=show_plots,
        add_zero_line=True,
    )
    if path:
        plot_paths.append(path)

    if save_plots:
        save_path = plot_root / "calibration_q_pedestal_less_zoom.png"
    else:
        save_path = None
    path = _plot_charge_grid(
        charge_test,
        q_min=pedestal_left * 2,
        q_max=pedestal_right * 12,
        num_bins=q_min_settings["num_bins"],
        log_scale=q_min_settings["log_scale"],
        title=f"Pedestal-subtracted charge (wide) — Station {station}",
        save_path=save_path,
        show_plots=show_plots,
        add_zero_line=True,
    )
    if path:
        plot_paths.append(path)

    if save_plots:
        save_path = plot_root / "calibration_q_fc.png"
    else:
        save_path = None
    if config.get("calibrate_charge_ns_to_fc", False):
        path = _plot_charge_fc_histograms(
            charge_test,
            config,
            save_path=save_path,
            show_plots=show_plots,
        )
        if path:
            plot_paths.append(path)

    pos_test, Tdiff_cal = _perform_time_diff_calibration(working_df, config)

    if save_plots:
        save_path = plot_root / "calibration_t_diff.png"
    else:
        save_path = None
    path = _plot_time_diff_histograms(
        pos_test,
        num_bins=q_min_settings["num_bins"],
        diff_threshold=config.get("T_diff_cal_threshold", 5.0),
        save_path=save_path,
        show_plots=show_plots,
    )
    if path:
        plot_paths.append(path)

    augmented = _augment_semisums(working_df)

    if save_plots:
        save_path = plot_root / "calibration_t_sum.png"
    else:
        save_path = None
    path = _plot_time_sum_histograms(
        augmented,
        num_bins=q_min_settings["num_bins"],
        sum_left=config.get("T_sum_left_cal", -5.0),
        sum_right=config.get("T_sum_right_cal", 5.0),
        save_path=save_path,
        show_plots=show_plots,
    )
    if path:
        plot_paths.append(path)

    if save_plots:
        save_path = plot_root / "calibration_qdiff_vs_qsum.png"
    else:
        save_path = None
    path = _plot_qsum_qdiff_hexbin(
        augmented,
        save_path=save_path,
        show_plots=show_plots,
        qsum_limits=(config.get("Q_sum_left_cal", 0.0), config.get("Q_sum_right_cal", 120.0)),
        qdiff_limits=(-config.get("Q_diff_cal_threshold", 40.0), config.get("Q_diff_cal_threshold", 40.0)),
    )
    if path:
        plot_paths.append(path)

    return Stage2CalibrationResult(
        dataframe=augmented,
        plots=plot_paths,
        metadata={
            "QF_pedestal": QF_pedestal.tolist(),
            "QB_pedestal": QB_pedestal.tolist(),
            "Tdiff_cal": Tdiff_cal.tolist(),
        },
    )
