#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 Task 0 (ACQ->RAW) driver.

Ingests one acquisition text file from STAGE_0_TO_1, preserves the existing
Task 1 raw-line parsing/rejection behavior, splits by trigger type, and writes
raw parquet files for Task 1.
"""

from __future__ import annotations

import builtins
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = None
for parent in CURRENT_PATH.parents:
    if parent.name == "MASTER":
        REPO_ROOT = parent.parent
        break
if REPO_ROOT is None:
    REPO_ROOT = CURRENT_PATH.parents[-1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.file_selection import (
    file_name_in_any_date_range,
    load_date_ranges_from_config,
    newest_order_key,
    select_latest_candidate,
    sync_unprocessed_with_date_range,
)
from MASTER.common.input_file_config import select_input_file_configuration
from MASTER.common.path_config import get_repo_root, resolve_home_path_from_config
from MASTER.common.plot_utils import pdf_save_rasterized_page
from MASTER.common.selection_config import load_selection_for_paths, station_is_selected
from MASTER.common.simulated_data_utils import resolve_simulated_z_positions
from MASTER.common.status_csv import (
    initialize_status_row,
    rename_status_row,
    update_status_progress,
)
from MASTER.common.step1_shared import (
    build_step1_cli_parser,
    build_step1_filtered_print,
    build_step1_raw_input_dataframe,
    load_step1_task_plot_catalog,
    load_step1_task_config_bundle,
    resolve_step1_effective_task_config,
    resolve_step1_plot_options,
    save_metadata,
    step1_task_plot_enabled,
    validate_step1_input_file_args,
)
from analysis_functions import (
    compute_acq_tt,
    datetime_bounds,
    duration_seconds,
    rate_hz,
    raw_channel_rename_map,
    station_matches_file,
)
from plotting_functions import (
    plot_acquisition_rate_vs_time_by_task_tt_with_histograms,
    plot_acquisition_rate_vs_time_by_trigger_type,
)


task_number = 0
STATION_CHOICES = ("0", "1", "2", "3", "4")
TASK0_PLOT_ALIASES: tuple[str, ...] = (
    "acquisition_rate_vs_time_by_trigger_type",
    "acquisition_rate_vs_time_by_task_tt_with_histograms",
)
task0_plot_status_by_alias: dict[str, str] = {}


def safe_move(source_path: str | Path, dest_path: str | Path) -> str:
    return shutil.move(str(source_path), str(dest_path))


def _select_candidate(file_names: list[str], station: str) -> str | None:
    if not file_names:
        return None
    order = os.environ.get("DATAFLOW_STEP1_SELECTION_ORDER", "latest").strip().lower()
    if order == "oldest":
        return min(file_names, key=lambda name: newest_order_key(name, station))
    return select_latest_candidate(file_names, station)


def task0_plot_enabled(alias: str) -> bool:
    if not task0_plot_status_by_alias:
        return True
    return step1_task_plot_enabled(alias, task0_plot_status_by_alias, plot_mode)


def _build_temp_pdf_path(target_path: Path) -> Path:
    base = Path(f"{target_path}.tmp.{os.getpid()}")
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = Path(f"{base}.{counter}")
        counter += 1
    return candidate


def _write_task0_pdf_from_pngs(plot_paths: list[Path], pdf_path: Path, figure_directory: Path) -> None:
    valid_plot_paths = [path for path in plot_paths if path.exists()]
    if not valid_plot_paths:
        return

    import matplotlib.pyplot as plt

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temp_pdf_path = _build_temp_pdf_path(pdf_path)
    try:
        with PdfPages(temp_pdf_path) as pdf:
            for png_path in valid_plot_paths:
                img = Image.open(png_path)
                fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
                ax.imshow(img)
                ax.axis("off")
                pdf_save_rasterized_page(pdf, fig, bbox_inches="tight")
                plt.close(fig)
        os.replace(temp_pdf_path, pdf_path)
    finally:
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()

    for png_path in valid_plot_paths:
        try:
            png_path.unlink()
        except OSError as exc:
            print(f"Error removing temporary Task 0 PNG '{png_path}': {exc}", force=True)
    if figure_directory.exists():
        shutil.rmtree(figure_directory)


def _resolve_station_conf_value(
    *,
    config_root: Path,
    station: str,
    start_time: object,
    end_time: object,
) -> float | None:
    input_file_config_path = (
        config_root
        / "STAGE_0"
        / "ONLINE_RUN_DICTIONARY"
        / f"STATION_{station}"
        / f"input_file_mingo0{station}.csv"
    )
    if not input_file_config_path.exists():
        print(f"Task 0 input configuration file does not exist: {input_file_config_path}", force=True)
        return None
    try:
        input_file = pd.read_csv(input_file_config_path, skiprows=1)
    except pd.errors.EmptyDataError:
        print(f"Task 0 input configuration file is empty: {input_file_config_path}", force=True)
        return None
    if input_file.empty:
        print(f"Task 0 input configuration file has no rows: {input_file_config_path}", force=True)
        return None

    selection_result = select_input_file_configuration(
        input_file,
        start_time=start_time,
        end_time=end_time,
    )
    selected_conf = selection_result.selected_conf
    if selected_conf is None:
        print("Task 0 found no selectable input configuration row for channel ordering.", force=True)
        return None
    try:
        return float(selected_conf.get("conf"))
    except (TypeError, ValueError):
        return None


def _apply_task0_raw_channel_ordering(
    frame: pd.DataFrame,
    *,
    station: str,
    conf_value: float | None,
) -> bool:
    if station != "2" or conf_value is None or conf_value >= 2:
        return False
    print("Task 0 applying station 2 configuration<2 Plane 4 channel swap.", force=True)
    plane4_keys = ("T4_F", "T4_B", "Q4_F", "Q4_B")
    swapped = False
    for key in plane4_keys:
        col3 = f"{key}_3"
        col4 = f"{key}_4"
        if col3 not in frame.columns or col4 not in frame.columns:
            print(f"Warning: Task 0 cannot swap missing Plane 4 columns: {col3}, {col4}", force=True)
            continue
        frame.loc[:, [col3, col4]] = frame.loc[:, [col4, col3]].to_numpy()
        swapped = True
    return swapped


CLI_PARSER = build_step1_cli_parser("Run Stage 1 STEP_1 TASK_0 (ACQ->RAW).", STATION_CHOICES)
CLI_ARGS = CLI_PARSER.parse_args()
validate_step1_input_file_args(CLI_PARSER, CLI_ARGS)

VERBOSE = bool(os.environ.get("DATAFLOW_VERBOSE")) or CLI_ARGS.verbose
print = build_step1_filtered_print(
    verbose=VERBOSE,
    debug_mode_getter=lambda: False,
    raw_print=builtins.print,
)

start_execution_time_counting = datetime.now()
start_timer(__file__)

task_config_bundle = load_step1_task_config_bundle(
    task_number,
    include_filter_parameter_config=False,
    log_fn=print,
)
config_root = task_config_bundle["config_root"]
config_file_path = task_config_bundle["config_file_path"]
plot_catalog_file_path = task_config_bundle["plot_catalog_file_path"]
parameter_config_file_path = task_config_bundle["parameter_config_file_path"]
plot_parameter_config_file_path = task_config_bundle["plot_parameter_config_file_path"]
fallback_parameter_config_file_path = task_config_bundle["fallback_parameter_config_file_path"]
config = task_config_bundle["config"]

if CLI_ARGS.station is None:
    CLI_PARSER.error("No station provided. Pass <station>.")
station = str(CLI_ARGS.station)
set_station(station)

config = resolve_step1_effective_task_config(
    config,
    station_id=station,
    task_number=task_number,
    config_root=config_root,
    parameter_config_file_path=parameter_config_file_path,
    fallback_parameter_config_file_path=fallback_parameter_config_file_path,
    plot_parameter_config_file_path=plot_parameter_config_file_path,
    use_filter_parameter_config=False,
    log_fn=print,
)
task0_plot_status_by_alias = load_step1_task_plot_catalog(
    plot_catalog_file_path,
    TASK0_PLOT_ALIASES,
    "Task 0",
    log_fn=print,
)
(
    plot_mode,
    create_plots,
    create_essential_plots,
    save_plots,
    create_pdf,
    show_plots,
    create_debug_plots,
) = resolve_step1_plot_options(config)

selection_config = load_selection_for_paths([config_file_path], master_config_root=config_root)
if not station_is_selected(station, selection_config.stations):
    print(f"Station {station} skipped by selection.stations.", force=True)
    sys.exit(0)

repo_root = get_repo_root()
home_path = str(resolve_home_path_from_config(config))
station_directory = repo_root / "STATIONS" / f"MINGO0{station}"
base_directory = station_directory / "STAGE_1" / "EVENT_DATA"
task_directory = base_directory / "STEP_1" / f"TASK_{task_number}"
raw_directory = station_directory / "STAGE_0_TO_1"

directories = {
    "unprocessed": task_directory / "INPUT_FILES" / "UNPROCESSED_DIRECTORY",
    "out_of_date": task_directory / "INPUT_FILES" / "OUT_OF_DATE_DIRECTORY",
    "processing": task_directory / "INPUT_FILES" / "PROCESSING_DIRECTORY",
    "completed": task_directory / "INPUT_FILES" / "COMPLETED_DIRECTORY",
    "error": task_directory / "INPUT_FILES" / "ERROR_DIRECTORY",
    "empty": task_directory / "ANCILLARY" / "EMPTY_FILES",
    "rejected": task_directory / "ANCILLARY" / "REJECTED_FILES",
    "metadata": task_directory / "METADATA",
    "output": task_directory / "OUTPUT_FILES",
    "pdf": task_directory / "PLOTS" / "PDF_DIRECTORY",
}
for directory in directories.values():
    directory.mkdir(parents=True, exist_ok=True)

csv_path = directories["metadata"] / f"task_{task_number}_metadata_execution.csv"
csv_path_status = directories["metadata"] / f"task_{task_number}_metadata_status.csv"

selected_input_file = CLI_ARGS.input_file_flag or CLI_ARGS.input_file
user_file_selection = bool(selected_input_file)
status_filename_base = Path(selected_input_file).stem if user_file_selection else f"__task{task_number}_startup_station_{station}__"
status_execution_date = initialize_status_row(
    csv_path_status,
    filename_base=status_filename_base,
    completion_fraction=0.0,
)

if not raw_directory.exists():
    raise FileNotFoundError(f"Task 0 raw acquisition directory does not exist: {raw_directory}")

if not user_file_selection:
    known_files = set()
    for key in ("unprocessed", "processing", "completed"):
        known_files.update(p.name for p in directories[key].iterdir() if p.is_file())

    for src in raw_directory.iterdir():
        if not src.is_file() or not src.name.lower().endswith(".dat"):
            continue
        if not station_matches_file(src.name, station):
            continue
        if src.stat().st_size == 0:
            dst = directories["empty"] / src.name
            if dst.exists():
                dst.unlink()
            safe_move(src, dst)
            continue
        if src.name in known_files:
            continue
        dst = directories["unprocessed"] / src.name
        safe_move(src, dst)
        os.utime(dst, None)
        print(f"Moved acquisition file to Task 0 UNPROCESSED: {src.name}")

    sync_unprocessed_with_date_range(
        config=config,
        unprocessed_directory=str(directories["unprocessed"]),
        out_of_date_directory=str(directories["out_of_date"]),
        log_fn=print,
        station_id=station,
        master_config_root=config_root,
    )

    date_ranges = load_date_ranges_from_config(config, station_id=station, master_config_root=config_root)
    candidates_by_source: list[tuple[str, Path, Path | None]] = []
    for source_key in ("unprocessed", "processing"):
        names = [p.name for p in directories[source_key].iterdir() if p.is_file() and p.name.lower().endswith(".dat")]
        if date_ranges:
            names = [name for name in names if file_name_in_any_date_range(name, date_ranges)]
        selected = _select_candidate(names, station)
        if selected:
            src = directories[source_key] / selected
            completed = directories["completed"] / selected
            candidates_by_source.append((selected, src, completed))
            break
    if not candidates_by_source and bool(config.get("complete_reanalysis", False)):
        names = [p.name for p in directories["completed"].iterdir() if p.is_file() and p.name.lower().endswith(".dat")]
        if date_ranges:
            names = [name for name in names if file_name_in_any_date_range(name, date_ranges)]
        selected = _select_candidate(names, station)
        if selected:
            src = directories["completed"] / selected
            completed = directories["completed"] / selected
            candidates_by_source.append((selected, src, completed))

    if not candidates_by_source:
        print("No Task 0 acquisition files to process.", force=True)
        no_file_status_basename = f"__task{task_number}_no_files_station_{station}__"
        if status_filename_base != no_file_status_basename:
            renamed = rename_status_row(
                csv_path_status,
                filename_base=status_filename_base,
                execution_date=status_execution_date,
                new_filename_base=no_file_status_basename,
            )
            if renamed:
                status_filename_base = no_file_status_basename
        update_status_progress(
            csv_path_status,
            filename_base=status_filename_base,
            execution_date=status_execution_date,
            completion_fraction=-1.0,
        )
        sys.exit(0)

    file_name, source_path, completed_file_path = candidates_by_source[0]
    processing_file_path = directories["processing"] / file_name
    if source_path != processing_file_path:
        if processing_file_path.exists():
            processing_file_path.unlink()
        safe_move(source_path, processing_file_path)
else:
    processing_file_path = Path(selected_input_file).expanduser()
    file_name = processing_file_path.name
    completed_file_path = None

if not station_matches_file(file_name, station):
    error_file_path = directories["error"] / file_name
    if processing_file_path.exists() and not user_file_selection:
        safe_move(processing_file_path, error_file_path)
    raise SystemExit(f"File '{file_name}' does not belong to station {station}.")

basename_no_ext = Path(file_name).stem
if status_filename_base != basename_no_ext:
    renamed = rename_status_row(
        csv_path_status,
        filename_base=status_filename_base,
        execution_date=status_execution_date,
        new_filename_base=basename_no_ext,
    )
    if renamed:
        status_filename_base = basename_no_ext
    else:
        print(
            "Warning: unable to rename Task 0 startup status row "
            f"from {status_filename_base} to {basename_no_ext}.",
            force=True,
        )
execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")
rejected_file = directories["rejected"] / f"rejected_{basename_no_ext}_{date_execution}.csv"

expected_columns = int(config.get("EXPECTED_COLUMNS_config", 71))
limit_number = config.get("limit_number")
limit_rows = int(limit_number) if config.get("last_file_test", False) and limit_number is not None else None
try:
    acquisition_rate_accumulation_window_seconds = int(
        config.get("acquisition_rate_accumulation_window_seconds", 60)
    )
except (TypeError, ValueError):
    acquisition_rate_accumulation_window_seconds = 60
if acquisition_rate_accumulation_window_seconds <= 0:
    acquisition_rate_accumulation_window_seconds = 60
try:
    acquisition_rate_task_tt_histogram_bins = int(
        config.get("acquisition_rate_task_tt_histogram_bins", 80)
    )
except (TypeError, ValueError):
    acquisition_rate_task_tt_histogram_bins = 80
if acquisition_rate_task_tt_histogram_bins <= 0:
    acquisition_rate_task_tt_histogram_bins = 80

update_status_progress(
    csv_path_status,
    filename_base=status_filename_base,
    execution_date=status_execution_date,
    completion_fraction=0.25,
)

read_df, read_lines, written_lines = build_step1_raw_input_dataframe(
    processing_file_path,
    rejected_file,
    expected_columns,
    limit_rows=limit_rows,
)
if read_df.empty:
    if not user_file_selection and processing_file_path.exists():
        safe_move(processing_file_path, directories["error"] / file_name)
    raise SystemExit("No valid acquisition rows parsed; moved file to ERROR.")

left_limit_time = pd.to_datetime("1-1-2000", format="%d-%m-%Y")
right_limit_time = pd.to_datetime("1-1-9999", format="%d-%m-%Y")
read_df = read_df.loc[read_df["datetime"].between(left_limit_time, right_limit_time)].copy()
read_df = read_df.rename(columns=raw_channel_rename_map())
read_df = read_df.rename(columns={"column_6": "acquisition_type"})
_, simulated_param_hash = resolve_simulated_z_positions(
    basename_no_ext,
    repo_root / "STATIONS" / f"MINGO0{station}" / "STAGE_1" / "EVENT_DATA",
    dat_path=processing_file_path,
)
read_df.loc[:, "param_hash"] = str(simulated_param_hash) if simulated_param_hash else ""

coincidence_mask = read_df["acquisition_type"] == 1
self_trigger_mask = read_df["acquisition_type"] == 2
other_trigger_mask = ~(coincidence_mask | self_trigger_mask)
conf_time_source = read_df.loc[coincidence_mask, "datetime"]
if conf_time_source.dropna().empty:
    conf_time_source = read_df["datetime"]
conf_time_source = pd.to_datetime(conf_time_source, errors="coerce").dropna()
task0_conf_value = None
task0_channel_swap_applied = False
if not conf_time_source.empty:
    task0_conf_value = _resolve_station_conf_value(
        config_root=config_root,
        station=station,
        start_time=conf_time_source.iloc[0],
        end_time=conf_time_source.iloc[-1],
    )
    task0_channel_swap_applied = _apply_task0_raw_channel_ordering(
        read_df,
        station=station,
        conf_value=task0_conf_value,
    )
else:
    print("Task 0 channel ordering lookup skipped: no valid acquisition timestamps.", force=True)
read_df = compute_acq_tt(read_df)

saved_plot_paths: list[Path] = []
figure_directory: Path | None = None
if save_plots and task0_plot_enabled("acquisition_rate_vs_time_by_trigger_type"):
    figure_directory = (
        task_directory
        / "PLOTS"
        / "FIGURE_DIRECTORY"
        / f"FIGURES_EXEC_ON_{date_execution}"
    )
    plot_path = figure_directory / f"acquisition_rate_vs_time_by_trigger_type_{basename_no_ext}.png"
    plotted = plot_acquisition_rate_vs_time_by_trigger_type(
        read_df,
        plot_path,
        title=f"Task 0 acquisition rate by trigger type, {basename_no_ext}",
        accumulation_window_seconds=acquisition_rate_accumulation_window_seconds,
    )
    if plotted:
        print(f"Task 0 acquisition-rate plot saved: {plot_path}", force=True)
        saved_plot_paths.append(plot_path)
    else:
        print("Task 0 acquisition-rate plot skipped: no valid datetime rows.", force=True)

if save_plots and task0_plot_enabled("acquisition_rate_vs_time_by_task_tt_with_histograms"):
    if figure_directory is None:
        figure_directory = (
            task_directory
            / "PLOTS"
            / "FIGURE_DIRECTORY"
            / f"FIGURES_EXEC_ON_{date_execution}"
        )
    plot_path = figure_directory / f"acquisition_rate_vs_time_by_task_tt_with_histograms_{basename_no_ext}.png"
    plotted = plot_acquisition_rate_vs_time_by_task_tt_with_histograms(
        read_df,
        plot_path,
        title=f"Task 0 acquisition rate by acq_tt, {basename_no_ext}",
        tt_column="acq_tt",
        accumulation_window_seconds=acquisition_rate_accumulation_window_seconds,
        rate_histogram_bins=acquisition_rate_task_tt_histogram_bins,
        y_limit_left=config.get("acquisition_rate_task_tt_ylim_left", 0),
        y_limit_right=config.get("acquisition_rate_task_tt_ylim_right", 4),
    )
    if plotted:
        print(f"Task 0 acquisition-rate-by-task-tt plot saved: {plot_path}", force=True)
        saved_plot_paths.append(plot_path)
    else:
        print("Task 0 acquisition-rate-by-task-tt plot skipped: no valid acq_tt/datetime rows.", force=True)

if create_pdf and saved_plot_paths and figure_directory is not None:
    save_pdf_filename = f"mingo{str(station).zfill(2)}_task0_{basename_no_ext}_{date_execution}.pdf"
    save_pdf_path = directories["pdf"] / save_pdf_filename
    print(f"Creating Task 0 PDF with all plots in {save_pdf_path}", force=True)
    _write_task0_pdf_from_pngs(saved_plot_paths, save_pdf_path, figure_directory)

coincidence_df = read_df.loc[coincidence_mask].copy()
self_trigger_df = read_df.loc[self_trigger_mask].copy()
coincidence_df = compute_acq_tt(coincidence_df, "raw_tt")
if not self_trigger_df.empty:
    self_trigger_df = compute_acq_tt(self_trigger_df, "raw_tt")

raw_output_path = directories["output"] / f"raw_{basename_no_ext}.parquet"
selftrigger_output_path = directories["output"] / f"selftrigger_raw_{basename_no_ext}.parquet"

coincidence_df.to_parquet(raw_output_path, engine="pyarrow", compression="zstd", index=False)
print(f"Task 0 coincidence raw parquet saved: {raw_output_path} rows={len(coincidence_df)}", force=True)

if self_trigger_df.empty:
    print("Task 0 self-trigger raw parquet skipped: no self-trigger rows.", force=True)
else:
    self_trigger_df.to_parquet(selftrigger_output_path, engine="pyarrow", compression="zstd", index=False)
    print(f"Task 0 self-trigger raw parquet saved: {selftrigger_output_path} rows={len(self_trigger_df)}", force=True)

coincidence_first, coincidence_last = datetime_bounds(coincidence_df)
self_first, self_last = datetime_bounds(self_trigger_df)
total_first, total_last = datetime_bounds(read_df)
total_duration_seconds = duration_seconds(read_df)
coincidence_duration_seconds = duration_seconds(coincidence_df)
self_trigger_duration_seconds = duration_seconds(self_trigger_df)
execution_time_minutes = (datetime.now() - start_execution_time_counting).total_seconds() / 60.0
metadata_row = {
    "filename_base": basename_no_ext,
    "execution_timestamp": execution_timestamp,
    "original_input_filename": file_name,
    "acquisition_basename": basename_no_ext,
    "param_hash": str(simulated_param_hash) if simulated_param_hash else "",
    "coincidence_raw_output_path": str(raw_output_path),
    "selftrigger_raw_output_path": "" if self_trigger_df.empty else str(selftrigger_output_path),
    "total_parsed_rows": int(len(read_df)),
    "input_read_lines": int(read_lines),
    "input_valid_lines": int(written_lines),
    "coincidence_rows": int(len(coincidence_df)),
    "self_trigger_rows": int(len(self_trigger_df)),
    "other_trigger_rows": int(other_trigger_mask.sum()),
    "total_first_datetime": total_first,
    "total_last_datetime": total_last,
    "coincidence_first_datetime": coincidence_first,
    "coincidence_last_datetime": coincidence_last,
    "self_trigger_first_datetime": self_first,
    "self_trigger_last_datetime": self_last,
    "total_duration_seconds": int(total_duration_seconds),
    "coincidence_duration_seconds": int(coincidence_duration_seconds),
    "self_trigger_duration_seconds": int(self_trigger_duration_seconds),
    "total_event_rate_hz": rate_hz(len(read_df), total_duration_seconds),
    "coincidence_event_rate_hz": rate_hz(len(coincidence_df), coincidence_duration_seconds),
    "self_trigger_event_rate_hz": rate_hz(len(self_trigger_df), self_trigger_duration_seconds),
    "input_valid_line_rate_hz": rate_hz(int(written_lines), total_duration_seconds),
    "total_execution_time_minutes": round(float(execution_time_minutes), 4),
    "raw_channel_order_conf": "" if task0_conf_value is None else task0_conf_value,
    "raw_channel_swap_plane4_3_4_applied": bool(task0_channel_swap_applied),
}
for acq_tt_value, count in read_df["acq_tt"].value_counts(dropna=False).sort_index().items():
    metadata_row[f"acq_tt_{int(acq_tt_value)}_count"] = int(count)
for raw_tt_value, count in coincidence_df["raw_tt"].value_counts(dropna=False).sort_index().items():
    metadata_row[f"raw_tt_{int(raw_tt_value)}_count"] = int(count)
save_metadata(
    csv_path,
    metadata_row,
    preferred_fieldnames=tuple(metadata_row.keys()),
    replace_existing_basename=True,
)

if not user_file_selection and completed_file_path is not None and processing_file_path.exists():
    if completed_file_path.exists():
        completed_file_path.unlink()
    safe_move(processing_file_path, completed_file_path)
    now = time.time()
    os.utime(completed_file_path, (now, now))
    print(f"Task 0 acquisition moved to COMPLETED: {completed_file_path}")

update_status_progress(
    csv_path_status,
    filename_base=status_filename_base,
    execution_date=status_execution_date,
    completion_fraction=1.0,
)
