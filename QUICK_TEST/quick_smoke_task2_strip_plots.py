#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TT_COUNT_VALUES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234
)
TT_COLOR_LABELS: tuple[str, ...] = tuple(str(tt_value) for tt_value in TT_COUNT_VALUES)
TT_COLOR_CMAP = plt.get_cmap("tab10")
_palette = sns.color_palette("tab10", n_colors=10)
TT_COLOR_MAP: dict[str, tuple[float, float, float, float]] = {}
_multi_idx = 0
for tt_label in TT_COLOR_LABELS:
    if len(tt_label) == 1:
        TT_COLOR_MAP[tt_label] = (0.60, 0.60, 0.60, 1.0)
    else:
        rgb = _palette[_multi_idx % len(_palette)]
        TT_COLOR_MAP[tt_label] = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        _multi_idx += 1
TT_COLOR_DEFAULT = (0.45, 0.45, 0.45, 1.0)
TASK2_STRIP_VAR_ORDER: tuple[str, ...] = ("Q_sum", "Q_dif", "T_sum", "T_dif")
SMOKE_STRIP_COMBINATION_Q_SUM_SUM_LIMITS: tuple[float, float] = (2.0, 10.0)
SMOKE_STRIP_COMBINATION_Q_SUM_DIF_THRESHOLD: float = 3.5
SMOKE_STRIP_COMBINATION_Q_DIF_SUM_THRESHOLD: float = 6.0
SMOKE_STRIP_COMBINATION_Q_DIF_DIF_THRESHOLD: float = 6.0
SMOKE_STRIP_COMBINATION_T_SUM_SUM_LIMITS: tuple[float, float] = (-3.0, 3.0)
SMOKE_STRIP_COMBINATION_T_SUM_DIF_THRESHOLD: float = 2.0
SMOKE_STRIP_COMBINATION_T_DIF_SUM_THRESHOLD: float = 1.0
SMOKE_STRIP_COMBINATION_T_DIF_DIF_THRESHOLD: float = 1.0
TASK2_PLANE_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


def normalize_tt_label(value: object) -> str:
    if pd.isna(value):
        return "0"
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def get_tt_color(tt_value: object) -> tuple[float, float, float, float]:
    return TT_COLOR_MAP.get(normalize_tt_label(tt_value), TT_COLOR_DEFAULT)


def strip_column_name(plane: int, strip: int, variable_name: str) -> str:
    prefix = "Q" if variable_name.startswith("Q") else "T"
    return f"{prefix}{plane}_{variable_name}_{strip}"


def series_to_numpy(df: pd.DataFrame, column_name: str) -> np.ndarray:
    if column_name not in df.columns:
        return np.zeros(len(df), dtype=float)
    return pd.to_numeric(df[column_name], errors="coerce").fillna(0).to_numpy(dtype=float)


def original_series(
    df: pd.DataFrame,
    column_name: str,
    original_columns_store: dict[str, pd.Series],
) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    if column_name in original_columns_store:
        return pd.to_numeric(
            original_columns_store[column_name].reindex(df.index),
            errors="coerce",
        ).fillna(0.0)
    return pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)


def original_numpy(
    df: pd.DataFrame,
    column_name: str,
    original_columns_store: dict[str, pd.Series],
) -> np.ndarray:
    return original_series(df, column_name, original_columns_store).to_numpy(dtype=float)


def touched_mask(
    df: pd.DataFrame,
    column_name: str,
    original_columns_store: dict[str, pd.Series],
) -> np.ndarray:
    if column_name not in df.columns or column_name not in original_columns_store:
        return np.zeros(len(df), dtype=bool)
    current = pd.to_numeric(df[column_name], errors="coerce")
    original = pd.to_numeric(
        original_columns_store[column_name].reindex(df.index),
        errors="coerce",
    )
    current_arr = current.to_numpy(dtype=float)
    original_arr = original.to_numpy(dtype=float)
    current_nan = np.isnan(current_arr)
    original_nan = np.isnan(original_arr)
    return (~(current_nan & original_nan)) & ~np.isclose(
        current_arr,
        original_arr,
        equal_nan=True,
    )


def strip_valid_mask(df: pd.DataFrame, plane: int, strip: int) -> np.ndarray:
    q_values = series_to_numpy(df, strip_column_name(plane, strip, "Q_sum"))
    t_values = series_to_numpy(df, strip_column_name(plane, strip, "T_sum"))
    return np.isfinite(q_values) & np.isfinite(t_values) & (q_values != 0) & (t_values != 0)


def snapshot_original_columns_once(
    frame: pd.DataFrame,
    column_names: list[str],
    original_columns_store: dict[str, pd.Series],
    enabled: bool,
) -> None:
    if not enabled:
        return
    for col in column_names:
        if col in frame.columns and col not in original_columns_store:
            original_columns_store[col] = frame[col].copy()


def snapshot_column_if_changed(
    frame: pd.DataFrame,
    column_name: str,
    change_mask: np.ndarray | pd.Series | None,
    original_columns_store: dict[str, pd.Series],
    enabled: bool,
) -> None:
    if not enabled or column_name not in frame.columns:
        return
    if change_mask is None or np.asarray(change_mask, dtype=bool).any():
        snapshot_original_columns_once(frame, [column_name], original_columns_store, enabled)


def restore_original_values(
    rows: pd.DataFrame,
    original_columns_store: dict[str, pd.Series],
    enabled: bool,
) -> pd.DataFrame:
    if not enabled or rows.empty:
        return rows
    restored_rows = rows.copy()
    for col, original_series in original_columns_store.items():
        if col in restored_rows.columns:
            restored_rows.loc[:, col] = original_series.reindex(restored_rows.index)
    return restored_rows


def append_removed_rows_from_mask(
    frame: pd.DataFrame,
    removed_mask: pd.Series,
    removed_rows_df: pd.DataFrame,
    original_columns_store: dict[str, pd.Series],
    enabled: bool,
) -> pd.DataFrame:
    if not enabled or frame.empty:
        return removed_rows_df
    aligned_mask = removed_mask.reindex(frame.index, fill_value=False)
    if not aligned_mask.any():
        return removed_rows_df
    rows_to_add = restore_original_values(frame.loc[aligned_mask].copy(), original_columns_store, enabled)
    if not removed_rows_df.empty:
        rows_to_add = rows_to_add.loc[~rows_to_add.index.isin(removed_rows_df.index)]
        if rows_to_add.empty:
            return removed_rows_df
    return pd.concat([removed_rows_df, rows_to_add], ignore_index=False, sort=False)


def build_original_columns_frame(
    base_index: pd.Index,
    original_columns_store: dict[str, pd.Series],
) -> pd.DataFrame:
    if not original_columns_store:
        return pd.DataFrame(index=base_index)
    return pd.DataFrame(
        {col: series.reindex(base_index) for col, series in original_columns_store.items()},
        index=base_index,
    )


def strip_component_columns_map(df: pd.DataFrame) -> dict[tuple[int, int], dict[str, str]]:
    strip_map: dict[tuple[int, int], dict[str, str]] = {}
    for plane in range(1, 5):
        for strip in range(1, 5):
            cols = {
                "Q_sum": strip_column_name(plane, strip, "Q_sum"),
                "Q_dif": strip_column_name(plane, strip, "Q_dif"),
                "T_sum": strip_column_name(plane, strip, "T_sum"),
                "T_dif": strip_column_name(plane, strip, "T_dif"),
            }
            if all(col in df.columns for col in cols.values()):
                strip_map[(plane, strip)] = cols
    return strip_map


def apply_strip_combination_filter(
    df_input: pd.DataFrame,
    original_columns_store: dict[str, pd.Series],
    enabled: bool,
) -> dict[str, int]:
    strip_map = strip_component_columns_map(df_input)
    if len(strip_map) < 2:
        return {
            "tracked_strip_count": len(strip_map),
            "valid_pair_observations": 0,
            "failed_pair_any": 0,
            "failed_pair_q_sum_sum": 0,
            "failed_pair_q_sum_dif": 0,
            "failed_pair_q_dif_sum": 0,
            "failed_pair_q_dif_dif": 0,
            "failed_pair_t_sum_sum": 0,
            "failed_pair_t_sum_dif": 0,
            "failed_pair_t_dif_sum": 0,
            "failed_pair_t_dif_dif": 0,
            "rows_affected": 0,
            "values_zeroed": 0,
        }

    strip_fail_masks = {strip_key: np.zeros(len(df_input), dtype=bool) for strip_key in strip_map}
    summary = {
        "tracked_strip_count": len(strip_map),
        "valid_pair_observations": 0,
        "failed_pair_any": 0,
        "failed_pair_q_sum_sum": 0,
        "failed_pair_q_sum_dif": 0,
        "failed_pair_q_dif_sum": 0,
        "failed_pair_q_dif_dif": 0,
        "failed_pair_t_sum_sum": 0,
        "failed_pair_t_sum_dif": 0,
        "failed_pair_t_dif_sum": 0,
        "failed_pair_t_dif_dif": 0,
        "rows_affected": 0,
        "values_zeroed": 0,
    }

    for strip_a, strip_b in combinations(sorted(strip_map), 2):
        cols_a = strip_map[strip_a]
        cols_b = strip_map[strip_b]
        q_sum_a = series_to_numpy(df_input, cols_a["Q_sum"])
        q_sum_b = series_to_numpy(df_input, cols_b["Q_sum"])
        q_dif_a = series_to_numpy(df_input, cols_a["Q_dif"])
        q_dif_b = series_to_numpy(df_input, cols_b["Q_dif"])
        t_sum_a = series_to_numpy(df_input, cols_a["T_sum"])
        t_sum_b = series_to_numpy(df_input, cols_b["T_sum"])
        t_dif_a = series_to_numpy(df_input, cols_a["T_dif"])
        t_dif_b = series_to_numpy(df_input, cols_b["T_dif"])

        valid_mask = (
            np.isfinite(q_sum_a)
            & np.isfinite(q_sum_b)
            & np.isfinite(q_dif_a)
            & np.isfinite(q_dif_b)
            & np.isfinite(t_sum_a)
            & np.isfinite(t_sum_b)
            & np.isfinite(t_dif_a)
            & np.isfinite(t_dif_b)
            & (q_sum_a != 0)
            & (q_sum_b != 0)
            & (q_dif_a != 0)
            & (q_dif_b != 0)
            & (t_sum_a != 0)
            & (t_sum_b != 0)
            & (t_dif_a != 0)
            & (t_dif_b != 0)
        )
        if not np.any(valid_mask):
            continue

        summary["valid_pair_observations"] += int(np.count_nonzero(valid_mask))
        pair_q_sum_sum = 0.5 * (q_sum_a + q_sum_b)
        pair_q_sum_dif = 0.5 * (q_sum_a - q_sum_b)
        pair_q_dif_sum = 0.5 * (q_dif_a + q_dif_b)
        pair_q_dif_dif = 0.5 * (q_dif_a - q_dif_b)
        pair_t_sum_sum = 0.5 * (t_sum_a + t_sum_b)
        pair_t_sum_dif = 0.5 * (t_sum_a - t_sum_b)
        pair_t_dif_sum = 0.5 * (t_dif_a + t_dif_b)
        pair_t_dif_dif = 0.5 * (t_dif_a - t_dif_b)

        fail_q_sum_sum = valid_mask & (
            (pair_q_sum_sum < SMOKE_STRIP_COMBINATION_Q_SUM_SUM_LIMITS[0])
            | (pair_q_sum_sum > SMOKE_STRIP_COMBINATION_Q_SUM_SUM_LIMITS[1])
        )
        fail_q_sum_dif = valid_mask & (np.abs(pair_q_sum_dif) > SMOKE_STRIP_COMBINATION_Q_SUM_DIF_THRESHOLD)
        fail_q_dif_sum = valid_mask & (np.abs(pair_q_dif_sum) > SMOKE_STRIP_COMBINATION_Q_DIF_SUM_THRESHOLD)
        fail_q_dif_dif = valid_mask & (np.abs(pair_q_dif_dif) > SMOKE_STRIP_COMBINATION_Q_DIF_DIF_THRESHOLD)
        fail_t_sum_sum = valid_mask & (
            (pair_t_sum_sum < SMOKE_STRIP_COMBINATION_T_SUM_SUM_LIMITS[0])
            | (pair_t_sum_sum > SMOKE_STRIP_COMBINATION_T_SUM_SUM_LIMITS[1])
        )
        fail_t_sum_dif = valid_mask & (np.abs(pair_t_sum_dif) > SMOKE_STRIP_COMBINATION_T_SUM_DIF_THRESHOLD)
        fail_t_dif_sum = valid_mask & (np.abs(pair_t_dif_sum) > SMOKE_STRIP_COMBINATION_T_DIF_SUM_THRESHOLD)
        fail_t_dif_dif = valid_mask & (np.abs(pair_t_dif_dif) > SMOKE_STRIP_COMBINATION_T_DIF_DIF_THRESHOLD)
        fail_any = (
            fail_q_sum_sum
            | fail_q_sum_dif
            | fail_q_dif_sum
            | fail_q_dif_dif
            | fail_t_sum_sum
            | fail_t_sum_dif
            | fail_t_dif_sum
            | fail_t_dif_dif
        )

        summary["failed_pair_q_sum_sum"] += int(np.count_nonzero(fail_q_sum_sum))
        summary["failed_pair_q_sum_dif"] += int(np.count_nonzero(fail_q_sum_dif))
        summary["failed_pair_q_dif_sum"] += int(np.count_nonzero(fail_q_dif_sum))
        summary["failed_pair_q_dif_dif"] += int(np.count_nonzero(fail_q_dif_dif))
        summary["failed_pair_t_sum_sum"] += int(np.count_nonzero(fail_t_sum_sum))
        summary["failed_pair_t_sum_dif"] += int(np.count_nonzero(fail_t_sum_dif))
        summary["failed_pair_t_dif_sum"] += int(np.count_nonzero(fail_t_dif_sum))
        summary["failed_pair_t_dif_dif"] += int(np.count_nonzero(fail_t_dif_dif))
        summary["failed_pair_any"] += int(np.count_nonzero(fail_any))

        if np.any(fail_any):
            strip_fail_masks[strip_a] |= fail_any
            strip_fail_masks[strip_b] |= fail_any

    any_row_affected = np.zeros(len(df_input), dtype=bool)
    for strip_key, fail_mask in strip_fail_masks.items():
        if not np.any(fail_mask):
            continue
        cols = strip_map[strip_key]
        any_row_affected |= fail_mask
        snapshot_original_columns_once(df_input, list(cols.values()), original_columns_store, enabled)
        for variable_name in TASK2_STRIP_VAR_ORDER:
            values = series_to_numpy(df_input, cols[variable_name])
            summary["values_zeroed"] += int(np.count_nonzero(values[fail_mask] != 0))
        df_input.loc[fail_mask, list(cols.values())] = 0.0

    summary["rows_affected"] = int(np.count_nonzero(any_row_affected))
    return summary


def compute_synthetic_cal_tt(df: pd.DataFrame) -> pd.Series:
    plane_active: list[np.ndarray] = []
    for plane in range(1, 5):
        strip_masks = [strip_valid_mask(df, plane, strip) for strip in range(1, 5)]
        plane_active.append(np.column_stack(strip_masks).any(axis=1))
    plane_count = np.column_stack(plane_active).sum(axis=1)
    return pd.Series(
        np.where(plane_count >= 2, plane_count * 10, plane_count),
        index=df.index,
        dtype=float,
    )


def limit_range(variable_name: str) -> tuple[float, float]:
    if variable_name == "Q_sum":
        return 1.0, 12.0
    if variable_name == "Q_dif":
        return -8.0, 8.0
    if variable_name == "T_sum":
        return -3.5, 3.5
    if variable_name == "T_dif":
        return -1.5, 1.5
    return 0.0, 1.0


def compute_variable_range(
    arrays: list[np.ndarray],
    default_limits: tuple[float, float],
) -> tuple[float, float]:
    finite_parts: list[np.ndarray] = []
    for values in arrays:
        if values.size == 0:
            continue
        finite_values = values[np.isfinite(values) & (values != 0)]
        if finite_values.size:
            finite_parts.append(finite_values)
    if finite_parts:
        merged = np.concatenate(finite_parts)
        lo, hi = np.nanpercentile(merged, [1.0, 99.0])
        if lo == hi:
            pad = abs(lo) * 0.05 + 1e-6
            lo -= pad
            hi += pad
    else:
        lo, hi = default_limits
    lo = min(float(lo), float(default_limits[0]))
    hi = max(float(hi), float(default_limits[1]))
    if lo == hi:
        hi = lo + 1.0
    pad = max(1e-3, 0.03 * (hi - lo))
    return lo - pad, hi + pad


def plot_strip_pair_matrices(
    retained_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    original_columns_store: dict[str, pd.Series],
    out_dir: Path,
    basename: str,
    strip_pair_min_events: int,
    removed_marker: str,
    removed_marker_size: int,
    removed_marker_alpha: float,
    touched_marker: str,
    touched_marker_size: int,
    touched_marker_alpha: float,
    figsize: tuple[float, float],
    max_points: int,
    max_figures: int,
) -> list[Path]:
    png_paths: list[Path] = []
    retained_tt_all = retained_df["clean_tt"].apply(normalize_tt_label).astype(str)
    removed_tt_all = removed_df["clean_tt"].apply(normalize_tt_label).astype(str) if "clean_tt" in removed_df.columns else pd.Series(dtype=str)

    for plane_i, plane_j in TASK2_PLANE_PAIRS:
        retained_pair_mask = retained_tt_all.str.contains(str(plane_i)) & retained_tt_all.str.contains(str(plane_j))
        retained_pair_df = retained_df.loc[retained_pair_mask]
        retained_pair_tt = retained_tt_all.loc[retained_pair_mask]

        if "clean_tt" in removed_df.columns:
            removed_pair_mask = removed_tt_all.str.contains(str(plane_i)) & removed_tt_all.str.contains(str(plane_j))
            removed_pair_df = removed_df.loc[removed_pair_mask]
            removed_pair_tt = removed_tt_all.loc[removed_pair_mask]
        else:
            removed_pair_df = removed_df.iloc[0:0].copy()
            removed_pair_tt = pd.Series(dtype=str)

        if retained_pair_df.empty and removed_pair_df.empty:
            continue

        retained_pair_sample = (
            retained_pair_df.sample(n=min(len(retained_pair_df), max_points), random_state=42)
            if len(retained_pair_df) > max_points
            else retained_pair_df.copy()
        )
        retained_pair_sample_tt = retained_pair_tt.reindex(retained_pair_sample.index)
        unique_tts = sorted(set(retained_pair_sample_tt.unique()).union(set(removed_pair_tt.unique())))
        if not unique_tts:
            unique_tts = [f"{plane_i}{plane_j}"]
        tt_color_map = {tt_label: get_tt_color(tt_label) for tt_label in unique_tts}
        retained_row_colors = np.array([tt_color_map[str(tt)] for tt in retained_pair_sample_tt], dtype=object)
        removed_row_colors = np.array([tt_color_map.get(str(tt), TT_COLOR_DEFAULT) for tt in removed_pair_tt], dtype=object)

        for strip_i in range(1, 5):
            for strip_j in range(1, 5):
                same_strip_case = plane_i == plane_j and strip_i == strip_j
                current_valid_mask = (
                    strip_valid_mask(retained_pair_df, plane_i, strip_i)
                    & strip_valid_mask(retained_pair_df, plane_j, strip_j)
                )
                touched_q_i = original_numpy(retained_pair_df, strip_column_name(plane_i, strip_i, "Q_sum"), original_columns_store)
                touched_t_i = original_numpy(retained_pair_df, strip_column_name(plane_i, strip_i, "T_sum"), original_columns_store)
                touched_q_j = original_numpy(retained_pair_df, strip_column_name(plane_j, strip_j, "Q_sum"), original_columns_store)
                touched_t_j = original_numpy(retained_pair_df, strip_column_name(plane_j, strip_j, "T_sum"), original_columns_store)
                original_valid_mask = (
                    np.isfinite(touched_q_i)
                    & np.isfinite(touched_t_i)
                    & np.isfinite(touched_q_j)
                    & np.isfinite(touched_t_j)
                    & (touched_q_i != 0)
                    & (touched_t_i != 0)
                    & (touched_q_j != 0)
                    & (touched_t_j != 0)
                )
                effective_events = int((current_valid_mask | original_valid_mask).sum())
                if effective_events < strip_pair_min_events:
                    continue

                retained_arrays: dict[tuple[str, str], np.ndarray] = {}
                touched_arrays: dict[tuple[str, str], np.ndarray] = {}
                touched_masks: dict[tuple[str, str], np.ndarray] = {}
                removed_arrays: dict[tuple[str, str], np.ndarray] = {}
                ranges: dict[str, tuple[float, float]] = {}
                for variable_name in TASK2_STRIP_VAR_ORDER:
                    col_name = strip_column_name(plane_i, strip_i, variable_name)
                    row_name = strip_column_name(plane_j, strip_j, variable_name)
                    retained_arrays[("col", variable_name)] = series_to_numpy(retained_pair_sample, col_name)
                    retained_arrays[("row", variable_name)] = series_to_numpy(retained_pair_sample, row_name)
                    touched_arrays[("col", variable_name)] = original_numpy(retained_pair_sample, col_name, original_columns_store)
                    touched_arrays[("row", variable_name)] = original_numpy(retained_pair_sample, row_name, original_columns_store)
                    touched_masks[("col", variable_name)] = touched_mask(retained_pair_sample, col_name, original_columns_store)
                    touched_masks[("row", variable_name)] = touched_mask(retained_pair_sample, row_name, original_columns_store)
                    removed_arrays[("col", variable_name)] = series_to_numpy(removed_pair_df, col_name)
                    removed_arrays[("row", variable_name)] = series_to_numpy(removed_pair_df, row_name)
                    ranges[variable_name] = compute_variable_range(
                        [
                            retained_arrays[("col", variable_name)],
                            retained_arrays[("row", variable_name)],
                            touched_arrays[("col", variable_name)],
                            touched_arrays[("row", variable_name)],
                            removed_arrays[("col", variable_name)],
                            removed_arrays[("row", variable_name)],
                        ],
                        limit_range(variable_name),
                    )

                fig, axes = plt.subplots(4, 4, figsize=figsize, squeeze=False, sharex="col", sharey="row")
                for row_idx, row_var in enumerate(TASK2_STRIP_VAR_ORDER):
                    for col_idx, col_var in enumerate(TASK2_STRIP_VAR_ORDER):
                        ax = axes[row_idx][col_idx]
                        ax.tick_params(labelsize=6)
                        for spine in ax.spines.values():
                            spine.set_linewidth(0.4)

                        x_values = retained_arrays[("col", col_var)]
                        y_values = retained_arrays[("row", row_var)]
                        touched_x = touched_arrays[("col", col_var)]
                        touched_y = touched_arrays[("row", row_var)]
                        touched_x_mask = touched_masks[("col", col_var)]
                        touched_y_mask = touched_masks[("row", row_var)]
                        removed_x = removed_arrays[("col", col_var)]
                        removed_y = removed_arrays[("row", row_var)]
                        x_limits = ranges[col_var]
                        y_limits = ranges[row_var]

                        if same_strip_case and col_idx > row_idx:
                            ax.set_visible(False)
                            continue

                        if same_strip_case and row_var == col_var:
                            ax.set_xlim(x_limits)
                            ax.set_ylim(y_limits)
                            limit_lo, limit_hi = limit_range(col_var)
                            if col_var.startswith("Q"):
                                hist_ax = ax.twinx()
                                for tt_label in unique_tts:
                                    tt_mask = retained_pair_sample_tt.to_numpy() == tt_label
                                    hist_values = np.concatenate((x_values[tt_mask], y_values[tt_mask]))
                                    hist_values = hist_values[np.isfinite(hist_values) & (hist_values != 0)]
                                    if hist_values.size > 0:
                                        hist_ax.hist(hist_values, bins=30, histtype="step", color=tt_color_map[tt_label], linewidth=1.2, log=True)
                                removed_hist_values = np.concatenate((removed_x, removed_y))
                                removed_hist_values = removed_hist_values[np.isfinite(removed_hist_values) & (removed_hist_values != 0)]
                                if removed_hist_values.size > 0:
                                    hist_ax.hist(removed_hist_values, bins=30, histtype="step", color="lightgrey", linewidth=1.6, linestyle="--", log=True)
                                hist_ax.set_yticks([])
                                hist_ax.tick_params(
                                    axis="both",
                                    which="both",
                                    labelbottom=False,
                                    labeltop=False,
                                    labelleft=False,
                                    labelright=False,
                                    bottom=False,
                                    top=False,
                                    left=False,
                                    right=False,
                                )
                                ax.axvline(limit_lo, color="lightgrey", linestyle="--", linewidth=0.8)
                                ax.axvline(limit_hi, color="lightgrey", linestyle="--", linewidth=0.8)
                            else:
                                hist_ax = ax.twiny()
                                for tt_label in unique_tts:
                                    tt_mask = retained_pair_sample_tt.to_numpy() == tt_label
                                    hist_values = np.concatenate((x_values[tt_mask], y_values[tt_mask]))
                                    hist_values = hist_values[np.isfinite(hist_values) & (hist_values != 0)]
                                    if hist_values.size > 0:
                                        hist_ax.hist(
                                            hist_values,
                                            bins=30,
                                            histtype="step",
                                            color=tt_color_map[tt_label],
                                            linewidth=1.2,
                                            log=True,
                                            orientation="horizontal",
                                        )
                                removed_hist_values = np.concatenate((removed_x, removed_y))
                                removed_hist_values = removed_hist_values[np.isfinite(removed_hist_values) & (removed_hist_values != 0)]
                                if removed_hist_values.size > 0:
                                    hist_ax.hist(
                                        removed_hist_values,
                                        bins=30,
                                        histtype="step",
                                        color="lightgrey",
                                        linewidth=1.6,
                                        linestyle="--",
                                        log=True,
                                        orientation="horizontal",
                                    )
                                hist_ax.set_xticks([])
                                hist_ax.tick_params(
                                    axis="both",
                                    which="both",
                                    labelbottom=False,
                                    labeltop=False,
                                    labelleft=False,
                                    labelright=False,
                                    bottom=False,
                                    top=False,
                                    left=False,
                                    right=False,
                                )
                                ax.axhline(limit_lo, color="lightgrey", linestyle="--", linewidth=0.8)
                                ax.axhline(limit_hi, color="lightgrey", linestyle="--", linewidth=0.8)
                            ax.tick_params(
                                axis="both",
                                which="both",
                                labelbottom=False,
                                labelleft=False,
                                bottom=False,
                                left=False,
                            )
                        else:
                            point_mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values != 0) & (y_values != 0)
                            if np.any(point_mask):
                                ax.scatter(
                                    x_values[point_mask],
                                    y_values[point_mask],
                                    s=12,
                                    alpha=0.75,
                                    linewidths=0,
                                    c=retained_row_colors[point_mask].tolist(),
                                    edgecolors="none",
                                    rasterized=True,
                                )
                            touched_panel_mask = (
                                np.isfinite(touched_x)
                                & np.isfinite(touched_y)
                                & (touched_x != 0)
                                & (touched_y != 0)
                                & (touched_x_mask | touched_y_mask)
                            )
                            if np.any(touched_panel_mask):
                                ax.scatter(
                                    touched_x[touched_panel_mask],
                                    touched_y[touched_panel_mask],
                                    s=touched_marker_size,
                                    marker=touched_marker,
                                    alpha=touched_marker_alpha,
                                    linewidths=1.0,
                                    c=retained_row_colors[touched_panel_mask].tolist(),
                                    rasterized=True,
                                    zorder=2.5,
                                )
                            removed_mask = np.isfinite(removed_x) & np.isfinite(removed_y) & (removed_x != 0) & (removed_y != 0)
                            if np.any(removed_mask):
                                ax.scatter(
                                    removed_x[removed_mask],
                                    removed_y[removed_mask],
                                    s=removed_marker_size,
                                    marker=removed_marker,
                                    alpha=removed_marker_alpha,
                                    linewidths=1.0,
                                    c=removed_row_colors[removed_mask].tolist(),
                                    rasterized=True,
                                    zorder=3,
                                )
                            ax.set_xlim(x_limits)
                            ax.set_ylim(y_limits)
                            x_lo, x_hi = limit_range(col_var)
                            y_lo, y_hi = limit_range(row_var)
                            ax.axvline(x_lo, color="lightgrey", linestyle="--", linewidth=0.8)
                            ax.axvline(x_hi, color="lightgrey", linestyle="--", linewidth=0.8)
                            ax.axhline(y_lo, color="lightgrey", linestyle="--", linewidth=0.8)
                            ax.axhline(y_hi, color="lightgrey", linestyle="--", linewidth=0.8)

                        if row_idx == len(TASK2_STRIP_VAR_ORDER) - 1:
                            ax.set_xlabel(f"P{plane_i} s{strip_i} {col_var}", fontsize=7)
                        else:
                            ax.tick_params(labelbottom=False)
                        if col_idx == 0:
                            ax.set_ylabel(f"P{plane_j} s{strip_j} {row_var}", fontsize=7)
                        else:
                            ax.tick_params(labelleft=False)

                fig.suptitle(
                    f"Smoke Task 2 strip-pair: P{plane_i}s{strip_i} vs P{plane_j}s{strip_j} · {basename}",
                    fontsize=9,
                )
                plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.10, right=0.97, top=0.92, bottom=0.09)
                png_path = out_dir / f"{len(png_paths) + 1:03d}_{basename}_P{plane_i}P{plane_j}_s{strip_i}_s{strip_j}.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                png_paths.append(png_path)
                plt.close(fig)

                if len(png_paths) >= max_figures:
                    return png_paths
    return png_paths


def build_synthetic_df(n_events: int, rng: np.random.Generator) -> pd.DataFrame:
    tt_choices = np.array([12, 13, 14, 23, 24, 34, 123, 124, 134, 234, 1234], dtype=int)
    clean_tt = rng.choice(tt_choices, size=n_events, replace=True)
    df = pd.DataFrame(index=pd.RangeIndex(n_events))
    df["clean_tt"] = clean_tt
    df["raw_tt"] = clean_tt

    for plane in range(1, 5):
        plane_active = np.array([str(plane) in str(tt_value) for tt_value in clean_tt], dtype=bool)
        for strip in range(1, 5):
            hit_mask = plane_active & (rng.random(n_events) < (0.72 + 0.04 * strip))
            q_sum = np.where(hit_mask, rng.normal(7.5 + plane * 0.45 + strip * 0.3, 1.2, n_events), 0.0)
            q_dif = np.where(hit_mask, rng.normal(0.0, 3.2, n_events), 0.0)
            t_sum = np.where(hit_mask, rng.normal((strip - 2.5) * 0.45 + (plane - 2.5) * 0.2, 0.55, n_events), 0.0)
            t_dif = np.where(hit_mask, rng.normal(0.0, 0.22, n_events), 0.0)
            df[strip_column_name(plane, strip, "Q_sum")] = q_sum.round(4)
            df[strip_column_name(plane, strip, "Q_dif")] = q_dif.round(4)
            df[strip_column_name(plane, strip, "T_sum")] = t_sum.round(4)
            df[strip_column_name(plane, strip, "T_dif")] = t_dif.round(4)

    anomaly_idx = df.index[3::37]
    if len(anomaly_idx) > 0:
        df.loc[anomaly_idx, ["clean_tt", "raw_tt"]] = 1234
        anomaly_strip_values = {
            (1, 1): (9.4, 5.9, 2.9, 0.95),
            (2, 2): (9.2, 5.7, 2.8, 0.90),
            (3, 1): (2.5, -5.9, -2.9, -0.95),
            (4, 2): (2.3, -5.7, -2.8, -0.90),
        }
        for (plane, strip), (q_sum, q_dif, t_sum, t_dif) in anomaly_strip_values.items():
            df.loc[anomaly_idx, strip_column_name(plane, strip, "Q_sum")] = q_sum
            df.loc[anomaly_idx, strip_column_name(plane, strip, "Q_dif")] = q_dif
            df.loc[anomaly_idx, strip_column_name(plane, strip, "T_sum")] = t_sum
            df.loc[anomaly_idx, strip_column_name(plane, strip, "T_dif")] = t_dif

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke test for Task 2 strip-pair diagnostic plots.")
    parser.add_argument("--basename", default="smoke_task2_strip_pairs")
    parser.add_argument("--n-events", type=int, default=180)
    parser.add_argument("--max-figures", type=int, default=12)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    figure_dir = repo_root / "QUICK_TEST" / "FIGURES"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    working_df = build_synthetic_df(args.n_events, rng)
    removed_rows_df = working_df.iloc[0:0].copy()
    tracking_base_index = working_df.index.copy()
    original_columns_store: dict[str, pd.Series] = {}
    track_removed_rows_task2 = True

    q_sum_columns = [col for col in working_df.columns if "_Q_sum_" in col]
    q_dif_columns = [col for col in working_df.columns if "_Q_dif_" in col]
    t_sum_columns = [col for col in working_df.columns if "_T_sum_" in col]

    # Preserve pre-modification values once before each in-place calibration/zeroing step.
    for col in q_sum_columns:
        change_mask = working_df[col] > 12.0
        snapshot_column_if_changed(working_df, col, change_mask, original_columns_store, track_removed_rows_task2)
        working_df.loc[change_mask, col] = 0.0
    for col in q_dif_columns:
        change_mask = np.abs(working_df[col]) > 7.5
        snapshot_column_if_changed(working_df, col, change_mask, original_columns_store, track_removed_rows_task2)
        working_df.loc[change_mask, col] = 0.0
    for col in t_sum_columns:
        nonzero_mask = working_df[col] != 0
        change_mask = nonzero_mask & (np.abs(working_df[col]) > 3.4)
        snapshot_column_if_changed(working_df, col, change_mask, original_columns_store, track_removed_rows_task2)
        working_df.loc[change_mask, col] = 0.0

    strip_combination_summary = apply_strip_combination_filter(
        working_df,
        original_columns_store,
        track_removed_rows_task2,
    )

    # Force a few fully removed rows for q_sum/all-zero/cal_tt-like filters.
    qsum_drop_idx = working_df.index[::17]
    all_zero_drop_idx = working_df.index[5::29]
    low_cal_idx = working_df.index[9::23]
    working_df.loc[qsum_drop_idx, q_sum_columns] = 0.0
    component_columns = [col for col in working_df.columns if any(tag in col for tag in ("_Q_sum_", "_Q_dif_", "_T_sum_", "_T_dif_"))]
    working_df.loc[all_zero_drop_idx, component_columns] = 0.0
    for plane in (3, 4):
        for strip in range(1, 5):
            for variable_name in TASK2_STRIP_VAR_ORDER:
                col = strip_column_name(plane, strip, variable_name)
                working_df.loc[low_cal_idx, col] = 0.0

    qsum_mask = (working_df[q_sum_columns] != 0).any(axis=1)
    removed_rows_df = append_removed_rows_from_mask(
        working_df,
        ~qsum_mask,
        removed_rows_df,
        original_columns_store,
        track_removed_rows_task2,
    )
    working_df = working_df.loc[qsum_mask].copy()

    component_data = working_df[component_columns].fillna(0)
    all_zero_mask = (component_data == 0).all(axis=1)
    removed_rows_df = append_removed_rows_from_mask(
        working_df,
        all_zero_mask,
        removed_rows_df,
        original_columns_store,
        track_removed_rows_task2,
    )
    working_df = working_df.loc[~all_zero_mask].copy()

    working_df["cal_tt"] = compute_synthetic_cal_tt(working_df)
    working_df["clean_to_cal_tt"] = (
        working_df["clean_tt"].apply(normalize_tt_label)
        + "_"
        + working_df["cal_tt"].fillna(0).astype(int).astype(str)
    )
    cal_tt_mask = working_df["cal_tt"].notna() & (working_df["cal_tt"] >= 10)
    removed_rows_df = append_removed_rows_from_mask(
        working_df,
        ~cal_tt_mask,
        removed_rows_df,
        original_columns_store,
        track_removed_rows_task2,
    )
    working_df = working_df.loc[cal_tt_mask].copy()

    png_paths = plot_strip_pair_matrices(
        working_df,
        removed_rows_df,
        original_columns_store,
        figure_dir,
        args.basename,
        strip_pair_min_events=10,
        removed_marker="x",
        removed_marker_size=30,
        removed_marker_alpha=0.9,
        touched_marker="+",
        touched_marker_size=24,
        touched_marker_alpha=0.95,
        figsize=(7.0, 7.0),
        max_points=1000,
        max_figures=args.max_figures,
    )

    removed_rows_path = figure_dir / f"removed_rows_{args.basename}.parquet"
    removed_rows_csv_path = figure_dir / f"removed_rows_{args.basename}.csv"
    original_cols_path = figure_dir / f"original_cols_{args.basename}.parquet"

    removed_rows_df.to_parquet(removed_rows_path, engine="pyarrow", compression="zstd", index=True)
    removed_rows_df.to_csv(removed_rows_csv_path, index=True)
    build_original_columns_frame(tracking_base_index, original_columns_store).to_parquet(
        original_cols_path,
        engine="pyarrow",
        compression="zstd",
        index=True,
    )

    print(f"Retained rows: {len(working_df)}")
    print(f"Removed rows: {len(removed_rows_df)}")
    print(
        "Strip-combination filter: "
        f"valid={strip_combination_summary['valid_pair_observations']} "
        f"failed_any={strip_combination_summary['failed_pair_any']} "
        f"rows_affected={strip_combination_summary['rows_affected']}"
    )
    print(f"PNG files written: {len(png_paths)}")
    for png_path in png_paths[:3]:
        print(f"Example PNG: {png_path}")
    print(f"Removed rows parquet: {removed_rows_path}")
    print(f"Removed rows CSV: {removed_rows_csv_path}")
    print(f"Original columns parquet: {original_cols_path}")


if __name__ == "__main__":
    main()
