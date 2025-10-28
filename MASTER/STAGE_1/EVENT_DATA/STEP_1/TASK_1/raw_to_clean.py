#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_ROOT = SCRIPT_DIR.parent
if str(STEP_ROOT) not in sys.path:
    sys.path.append(str(STEP_ROOT))

REPO_ROOT = next((p for p in STEP_ROOT.parents if (p / "MASTER").is_dir()), STEP_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from common.pipeline_common import (  # noqa: E402
    acquire_file,
    complete_file,
    ensure_task_layout,
    fail_file,
    load_config,
    resolve_station_paths,
    save_dataframe_h5,
    sync_sources_to_queue,
    write_metadata,
)
from MASTER.common.execution_logger import set_station, start_timer  # noqa: E402
from MASTER.common.status_csv import append_status_row, mark_status_complete  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STEP 1 · TASK 1 — convert raw .dat files into cleaned DataFrames."
    )
    parser.add_argument("station", help="Station number (1-4)")
    parser.add_argument(
        "--strategy",
        choices=("oldest", "random"),
        default="oldest",
        help="File selection strategy when multiple inputs are pending.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional override to process a specific file (bypasses queue).",
    )
    return parser.parse_args()


def build_work_dirs(task_root: Path) -> dict:
    layout = ensure_task_layout(task_root)
    work_root = task_root / "WORK"
    empty_dir = work_root / "EMPTY_FILES"
    rejected_dir = work_root / "REJECTED_FILES"
    temp_dir = work_root / "TEMP_FILES"
    plots_root = layout["plots"]
    for directory in (work_root, empty_dir, rejected_dir, temp_dir, plots_root):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "queues": layout,
        "empty": empty_dir,
        "rejected": rejected_dir,
        "temp": temp_dir,
        "plots": plots_root,
    }


def _resolve_plot_settings(config: dict, *, self_trigger: bool = False) -> dict:
    """Pick clip ranges and thresholds based on execution mode."""
    debug_mode = bool(config.get("debug_mode", False))

    if debug_mode:
        t_clip_min = config.get("T_clip_min_debug", -500)
        t_clip_max = config.get("T_clip_max_debug", 500)
        q_clip_min = config.get("Q_clip_min_debug", -500)
        q_clip_max = config.get("Q_clip_max_debug", 500)
        num_bins = config.get("num_bins_debug", 100)
        t_left = config.get("T_side_left_pre_cal_debug", -500)
        t_right = config.get("T_side_right_pre_cal_debug", 500)
        q_left = config.get("Q_side_left_pre_cal_debug", -500)
        q_right = config.get("Q_side_right_pre_cal_debug", 500)
    else:
        t_clip_min = config.get("T_clip_min_default", -300)
        t_clip_max = config.get("T_clip_max_default", 100)
        q_clip_min = config.get("Q_clip_min_default", 0)
        q_clip_max = config.get("Q_clip_max_default", 500)
        num_bins = config.get("num_bins_default", 100)
        t_left = config.get("T_side_left_pre_cal_default", -200)
        t_right = config.get("T_side_right_pre_cal_default", -100)
        q_left = config.get("Q_side_left_pre_cal_default", 80)
        q_right = config.get("Q_side_right_pre_cal_default", 300)

    if self_trigger:
        t_clip_min = config.get("T_clip_min_ST", t_clip_min)
        t_clip_max = config.get("T_clip_max_ST", t_clip_max)
        q_clip_min = config.get("Q_clip_min_ST", q_clip_min)
        q_clip_max = config.get("Q_clip_max_ST", q_clip_max)
        t_left = config.get("T_side_left_pre_cal_ST", t_left)
        t_right = config.get("T_side_right_pre_cal_ST", t_right)
        q_left = config.get("Q_side_left_pre_cal_ST", q_left)
        q_right = config.get("Q_side_right_pre_cal_ST", q_right)

    return {
        "t_clip_min": t_clip_min,
        "t_clip_max": t_clip_max,
        "q_clip_min": q_clip_min,
        "q_clip_max": q_clip_max,
        "num_bins": num_bins,
        "t_left": t_left,
        "t_right": t_right,
        "q_left": q_left,
        "q_right": q_right,
        "log_scale": bool(config.get("log_scale", False)),
    }


def _finalize_plot(fig: plt.Figure, *, save_path: Optional[Path], show: bool) -> None:
    """Persist/close a matplotlib figure respecting CLI flags."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_time_histograms(
    df: pd.DataFrame,
    settings: dict,
    *,
    save_dir: Optional[Path],
    dataset_tag: str,
    show: bool,
) -> list[Path]:
    saved: list[Path] = []
    if df.empty:
        return saved

    fig, axes = plt.subplots(4, 4, figsize=(20, 10))
    axes = axes.flatten()
    t_min = settings["t_clip_min"]
    t_max = settings["t_clip_max"]
    num_bins = settings["num_bins"]
    log_scale = settings["log_scale"]

    for plane in range(1, 5):
        for strip in range(1, 5):
            idx = (plane - 1) * 4 + (strip - 1)
            ax = axes[idx]
            col_front = f"T{plane}_F_{strip}"
            col_back = f"T{plane}_B_{strip}"
            if col_front not in df or col_back not in df:
                ax.axis("off")
                continue
            values_front = df[col_front]
            values_back = df[col_back]
            mask_front = (values_front != 0) & (values_front >= t_min) & (values_front <= t_max)
            mask_back = (values_back != 0) & (values_back >= t_min) & (values_back <= t_max)
            ax.hist(values_front[mask_front], bins=num_bins, alpha=0.5, label="Front")
            ax.hist(values_back[mask_back], bins=num_bins, alpha=0.5, label="Back")
            ax.set_title(f"T{plane} strip {strip}")
            if log_scale:
                ax.set_yscale("log")
            ax.legend(fontsize="small")

    plt.tight_layout()
    save_path = None
    if save_dir is not None:
        save_path = save_dir / f"{dataset_tag}_time_hist.png"
        saved.append(save_path)
    _finalize_plot(fig, save_path=save_path, show=show)
    return saved


def _plot_charge_histograms(
    df: pd.DataFrame,
    settings: dict,
    *,
    save_dir: Optional[Path],
    dataset_tag: str,
    show: bool,
) -> list[Path]:
    saved: list[Path] = []
    if df.empty:
        return saved

    fig, axes = plt.subplots(4, 4, figsize=(20, 10))
    axes = axes.flatten()
    q_min = settings["q_clip_min"]
    q_max = settings["q_clip_max"]
    num_bins = settings["num_bins"]
    log_scale = settings["log_scale"]

    for plane in range(1, 5):
        for strip in range(1, 5):
            idx = (plane - 1) * 4 + (strip - 1)
            ax = axes[idx]
            col_front = f"Q{plane}_F_{strip}"
            col_back = f"Q{plane}_B_{strip}"
            if col_front not in df or col_back not in df:
                ax.axis("off")
                continue
            values_front = df[col_front]
            values_back = df[col_back]
            mask_front = (values_front != 0) & (values_front >= q_min) & (values_front <= q_max)
            mask_back = (values_back != 0) & (values_back >= q_min) & (values_back <= q_max)
            ax.hist(values_front[mask_front], bins=num_bins, alpha=0.5, label="Front")
            ax.hist(values_back[mask_back], bins=num_bins, alpha=0.5, label="Back")
            ax.set_title(f"Q{plane} strip {strip}")
            if log_scale:
                ax.set_yscale("log")
            ax.legend(fontsize="small")

    plt.tight_layout()
    save_path = None
    if save_dir is not None:
        save_path = save_dir / f"{dataset_tag}_charge_hist.png"
        saved.append(save_path)
    _finalize_plot(fig, save_path=save_path, show=show)
    return saved


def _plot_time_charge_scatter(
    df: pd.DataFrame,
    settings: dict,
    *,
    save_dir: Optional[Path],
    dataset_tag: str,
    show: bool,
) -> list[Path]:
    saved: list[Path] = []
    if df.empty:
        return saved

    fig, axes = plt.subplots(4, 4, figsize=(20, 10))
    axes = axes.flatten()
    t_min = settings["t_clip_min"]
    t_max = settings["t_clip_max"]
    q_min = settings["q_clip_min"]
    q_max = settings["q_clip_max"]
    t_left = settings["t_left"]
    t_right = settings["t_right"]
    q_left = settings["q_left"]
    q_right = settings["q_right"]

    for plane in range(1, 5):
        for strip in range(1, 5):
            idx = (plane - 1) * 4 + (strip - 1)
            ax = axes[idx]
            t_front_col = f"T{plane}_F_{strip}"
            t_back_col = f"T{plane}_B_{strip}"
            q_front_col = f"Q{plane}_F_{strip}"
            q_back_col = f"Q{plane}_B_{strip}"
            if any(col not in df for col in (t_front_col, t_back_col, q_front_col, q_back_col)):
                ax.axis("off")
                continue

            t_front = df[t_front_col]
            t_back = df[t_back_col]
            q_front = df[q_front_col]
            q_back = df[q_back_col]

            mask_front = (
                (t_front != 0)
                & (t_front >= t_min)
                & (t_front <= t_max)
                & (q_front >= q_min)
                & (q_front <= q_max)
            )
            mask_back = (
                (t_back != 0)
                & (t_back >= t_min)
                & (t_back <= t_max)
                & (q_back >= q_min)
                & (q_back <= q_max)
            )

            ax.scatter(q_front[mask_front], t_front[mask_front], s=1, alpha=0.4, label="Front")
            ax.scatter(q_back[mask_back], t_back[mask_back], s=1, alpha=0.4, label="Back")

            ax.axhline(y=t_left, color="red", linestyle="--", linewidth=0.8)
            ax.axhline(y=t_right, color="blue", linestyle="--", linewidth=0.8)
            ax.axvline(x=q_left, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(x=q_right, color="blue", linestyle="--", linewidth=0.8)
            ax.set_title(f"T{plane} strip {strip}")
            ax.legend(fontsize="x-small")
            ax.set_xlim(q_min, q_max)
            ax.set_ylim(t_min, t_max)

    plt.tight_layout()
    save_path = None
    if save_dir is not None:
        save_path = save_dir / f"{dataset_tag}_time_charge_scatter.png"
        saved.append(save_path)
    _finalize_plot(fig, save_path=save_path, show=show)
    return saved


def generate_task1_plots(
    coincidence_df: pd.DataFrame,
    self_trigger_df: Optional[pd.DataFrame],
    config: dict,
    plots_dir: Path,
    output_stub: Path,
    *,
    station: str,
    start_time: Optional[str],
) -> list[Path]:
    """Mirror the most relevant legacy plots for STEP_1 · TASK_1."""
    create_plots = bool(config.get("create_plots", False))
    create_essential = bool(config.get("create_essential_plots", False))
    if not (create_plots or create_essential):
        return []

    save_plots = bool(config.get("save_plots", True))
    show_plots = bool(config.get("show_plots", False))
    render_histograms = create_plots or create_essential
    render_scatter = create_plots

    save_dir = plots_dir / output_stub.name if save_plots else None
    saved_paths: list[Path] = []

    timestamp_label = ""
    if start_time:
        try:
            ts = pd.Timestamp(start_time)
            timestamp_label = ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            timestamp_label = start_time

    if render_histograms or render_scatter:
        settings = _resolve_plot_settings(config, self_trigger=False)
        dataset_tag = f"station{station}_coincidence"

        if render_histograms:
            saved_paths.extend(
                _plot_time_histograms(
                    coincidence_df,
                    settings,
                    save_dir=save_dir,
                    dataset_tag=dataset_tag,
                    show=show_plots,
                )
            )
            saved_paths.extend(
                _plot_charge_histograms(
                    coincidence_df,
                    settings,
                    save_dir=save_dir,
                    dataset_tag=dataset_tag,
                    show=show_plots,
                )
            )
        if render_scatter:
            saved_paths.extend(
                _plot_time_charge_scatter(
                    coincidence_df,
                    settings,
                    save_dir=save_dir,
                    dataset_tag=dataset_tag,
                    show=show_plots,
                )
            )

    if self_trigger_df is not None and not self_trigger_df.empty and (render_histograms or render_scatter):
        settings_st = _resolve_plot_settings(config, self_trigger=True)
        dataset_tag = f"station{station}_self_trigger"
        if render_histograms:
            saved_paths.extend(
                _plot_time_histograms(
                    self_trigger_df,
                    settings_st,
                    save_dir=save_dir,
                    dataset_tag=dataset_tag,
                    show=show_plots,
                )
            )
            saved_paths.extend(
                _plot_charge_histograms(
                    self_trigger_df,
                    settings_st,
                    save_dir=save_dir,
                    dataset_tag=dataset_tag,
                    show=show_plots,
                )
            )
        if render_scatter:
            saved_paths.extend(
                _plot_time_charge_scatter(
                    self_trigger_df,
                    settings_st,
                    save_dir=save_dir,
                    dataset_tag=dataset_tag,
                    show=show_plots,
                )
            )

    if save_dir is not None and timestamp_label:
        (save_dir / "README.txt").write_text(
            f"Plots for station {station} — start {timestamp_label}\nGenerated from {output_stub.name}\n",
            encoding="utf-8",
        )

    return saved_paths

def process_raw_file(
    dat_path: Path,
    config: dict,
    output_stub: Path,
    work_dirs: dict,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict]:
    temp_dir: Path = work_dirs["temp"]
    rejected_dir: Path = work_dirs["rejected"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    temp_file = temp_dir / f"{dat_path.stem}_{timestamp}.csv"
    rejected_file = rejected_dir / f"{dat_path.stem}_{timestamp}_rejected.txt"

    expected_columns = config["EXPECTED_COLUMNS_config"]
    zero_token_pattern = re.compile(r"0000\.0000")
    leading_zero_pattern = re.compile(r"\b0+([0-9]+)")
    multi_space_pattern = re.compile(r" +")
    xyear_pattern = re.compile(r"X(20\d{2})")
    neg_gap_pattern = re.compile(r"(\w)-(\d)")
    malformed_number_pattern = re.compile(r"-?\d+\.\d+\.\d+")

    read_lines = 0
    written_lines = 0

    def process_line(line: str) -> str:
        line = zero_token_pattern.sub("0", line)
        line = leading_zero_pattern.sub(r"\1", line)
        line = multi_space_pattern.sub(",", line.strip())
        line = xyear_pattern.sub(r"X\n\1", line)
        line = neg_gap_pattern.sub(r"\1 -\2", line)
        return line

    def is_valid_date(values: list[str]) -> bool:
        try:
            year, month, day = int(values[0]), int(values[1]), int(values[2])
        except ValueError:
            return False
        if not (2000 <= year <= 2100):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        return True

    with dat_path.open("r") as infile, temp_file.open("w") as cleaned, rejected_file.open("w") as rejected:
        for i, line in enumerate(infile, start=1):
            read_lines += 1
            cleaned_line = process_line(line)
            values = cleaned_line.split(",")

            if len(values) < 3 or not is_valid_date(values[:3]):
                rejected.write(f"Line {i} (Invalid date): {line.strip()}\n")
                continue

            if malformed_number_pattern.search(cleaned_line):
                rejected.write(f"Line {i} (Malformed number): {line.strip()}\n")
                continue

            if len(values) == expected_columns:
                cleaned.write(cleaned_line + "\n")
            else:
                rejected.write(f"Line {i} (Wrong column count): {line.strip()}\n")
                continue

            written_lines += 1

    if written_lines == 0:
        raise ValueError("No valid rows found in raw file.")

    read_df = pd.read_csv(
        temp_file,
        header=None,
        engine="c",
        dtype=np.float64,
        na_values=["", " "],
        keep_default_na=True,
    )

    read_df.columns = ["year", "month", "day", "hour", "minute", "second"] + [
        f"column_{i}" for i in range(6, 71)
    ]
    time_columns = ["year", "month", "day", "hour", "minute", "second"]
    for col in time_columns:
        read_df[col] = read_df[col].round().astype("Int64")

    read_df["datetime"] = pd.to_datetime(read_df[time_columns])
    if "column_6" in read_df.columns:
        read_df["column_6"] = read_df["column_6"].round().astype("Int64")

    left_limit_time = pd.to_datetime("2000-01-01")
    right_limit_time = pd.to_datetime("2100-01-01")
    selected_df = read_df.loc[
        read_df["datetime"].between(left_limit_time, right_limit_time)
    ].copy()

    self_trigger_df = selected_df[selected_df["column_6"] == 2].copy()
    coincidence_df = selected_df[selected_df["column_6"] == 1].copy()

    if coincidence_df.empty and self_trigger_df.empty:
        raise ValueError("No valid coincidence or self-trigger events found.")

    start_time = coincidence_df["datetime"].iloc[0] if not coincidence_df.empty else self_trigger_df["datetime"].iloc[0]
    end_time = coincidence_df["datetime"].iloc[-1] if not coincidence_df.empty else self_trigger_df["datetime"].iloc[-1]

    metadata = {
        "source_file": dat_path.name,
        "rows_total": int(read_lines),
        "rows_valid": int(written_lines),
        "valid_ratio": float(written_lines) / float(read_lines),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "self_trigger": not self_trigger_df.empty,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_stub": output_stub.name,
    }

    temp_file.unlink(missing_ok=True)
    rejected_file.unlink(missing_ok=True)

    column_indices = {
        'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
        'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
        'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
        'T4_F': range(7, 11),  'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23),
    }

    def build_working_frame(df: pd.DataFrame) -> pd.DataFrame:
        data = {'datetime': df['datetime'].values}
        for key, idx_range in column_indices.items():
            for i, col_idx in enumerate(idx_range):
                column_name = f'{key}_{i+1}'
                data[column_name] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
        working = pd.DataFrame(data)
        return working

    coincidence_df = build_working_frame(coincidence_df)
    self_trigger_out = None
    if not self_trigger_df.empty:
        self_trigger_out = build_working_frame(self_trigger_df)

    return coincidence_df.reset_index(drop=True), (
        self_trigger_out.reset_index(drop=True) if self_trigger_out is not None else None
    ), metadata




def main() -> None:
    args = parse_args()
    station = args.station
    set_station(station)
    start_timer(__file__)

    config = load_config()
    station_paths = resolve_station_paths(station)
    task_root = station_paths.task_dirs["TASK_1"]
    work_dirs = build_work_dirs(task_root)
    queues = work_dirs["queues"]
    status_csv = (task_root / "LOGS" / "raw_to_clean_status.csv")
    status_timestamp = append_status_row(status_csv)

    upstream = station_paths.raw_dir
    sync_sources_to_queue(upstream, queues["unprocessed"])

    source_path: Optional[Path]
    if args.source:
        source_path = args.source
        if not source_path.exists():
            print(f"[raw_to_clean] Provided source file does not exist: {source_path}")
            return
        processing_target = queues["processing"] / source_path.name
        source_path.rename(processing_target)
        candidate_ctx = processing_target
        borrowed_source = True
    else:
        borrowed_source = False
        with acquire_file(
            queues["unprocessed"],
            queues["processing"],
            strategy=args.strategy,
        ) as candidate:
            if candidate is None:
                print("[raw_to_clean] No files to process.")
                mark_status_complete(status_csv, status_timestamp)
                return
            candidate_ctx = candidate

    assert candidate_ctx is not None
    processing_path = candidate_ctx
    basename = processing_path.stem
    start_wall = time.time()

    try:
        output_stub = Path(f"cleaned_{basename}")
        coincidence_df, self_trigger_df, metadata = process_raw_file(
            processing_path,
            config,
            output_stub,
            work_dirs,
        )

        generated_plots = generate_task1_plots(
            coincidence_df,
            self_trigger_df,
            config,
            work_dirs["plots"],
            output_stub,
            station=station,
            start_time=metadata.get("start_time"),
        )

        stage2_dirs = ensure_task_layout(station_paths.task_dirs["TASK_2"])
        output_h5 = stage2_dirs["unprocessed"] / f"{output_stub.name}.h5"
        save_dataframe_h5(coincidence_df, output_h5)

        if self_trigger_df is not None:
            save_dataframe_h5(self_trigger_df, output_h5.with_suffix(".self_trigger.h5"))

        metadata.update(
            {
                "station": station,
                "wall_time_sec": round(time.time() - start_wall, 2),
                "stage": "raw_to_clean",
            }
        )
        if generated_plots:
            metadata["plots"] = [str(path) for path in generated_plots]
        write_metadata(metadata, output_h5.with_suffix(".json"))
        complete_file(processing_path, queues["completed"])
        mark_status_complete(status_csv, status_timestamp)
        print(f"[raw_to_clean] Generated {output_h5.name}")
    except Exception as exc:  # pragma: no cover - pipeline robustness
        fail_file(processing_path, queues["error"])
        print(f"[raw_to_clean] Failed to process {processing_path.name}: {exc}")
        raise
    finally:
        if borrowed_source and processing_path.exists():
            processing_path.unlink()


if __name__ == "__main__":
    main()
