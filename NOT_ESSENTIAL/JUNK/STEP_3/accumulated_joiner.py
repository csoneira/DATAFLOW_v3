#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import builtins
import os
import random
import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Set
import csv
import warnings

# Third-party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import poisson
from tqdm import tqdm




import yaml

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
from MASTER.common.status_csv import append_status_row, mark_status_complete

start_timer(__file__)
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]



load_big_event_file = config["load_big_event_file"]



# Load the config once at the top of your script
with open(f"{home_path}/DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml") as f:
    config = yaml.safe_load(f)

SIG_DIGITS = config["significant_digits"]

suffixes = config["suffixes"]
print(suffixes)


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Running big_event_file_joiner.py -------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Check if the script has an argument
parser = argparse.ArgumentParser(
    description="Aggregate accumulated CSV files for a given station."
)
parser.add_argument("station", help="Station identifier (e.g. 1, 2, 3, 4)")
parser.add_argument(
    "--requeue",
    action="store_true",
    help="Return all files from COMPLETED to UNPROCESSED before processing.",
)
args = parser.parse_args()

station = args.station
requeue = args.requeue

print(f"Station: {station}")
set_station(station)

PIPELINE_CSV_HEADERS = [
    'basename',
    'start_date',
    'hld_remote_add_date',
    'hld_local_add_date',
    'dat_add_date',
    'list_ev_name',
    'list_ev_add_date',
    'acc_name',
    'acc_add_date',
    'merge_add_date',
]

station_dir = Path.home() / 'DATAFLOW_v3' / 'STATIONS' / f'MINGO0{station}'
pipeline_csv_path = station_dir / f'database_status_{station}.csv'
pipeline_csv_path.parent.mkdir -p(parents=True, exist_ok=True)
if not pipeline_csv_path.exists():
    with pipeline_csv_path.open('w', newline='') as handle:
        csv.writer(handle).writerow(PIPELINE_CSV_HEADERS)


def ensure_start_value(row):
    base_name = row.get('basename', '')
    digits = base_name[-11:]
    if len(digits) == 11 and digits.isdigit() and not row.get('start_date'):
        yy = int(digits[:2])
        doy = int(digits[2:5])
        hh = int(digits[5:7])
        mm = int(digits[7:9])
        ss = int(digits[9:11])
        year = 2000 + yy
        try:
            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
            row['start_date'] = dt.strftime('%Y-%m-%d_%H.%M.%S')
        except ValueError:
            pass


def load_pipeline_rows():
    rows = []
    with pipeline_csv_path.open('r', newline='') as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def store_pipeline_rows(rows):
    with pipeline_csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=PIPELINE_CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def update_merge_timestamp(acc_filename: str, timestamp: str) -> None:
    rows = load_pipeline_rows()
    updated = False
    for row in rows:
        if row.get('acc_name') == acc_filename:
            ensure_start_value(row)
            row['merge_add_date'] = timestamp
            updated = True
    if updated:
        store_pipeline_rows(rows)


merge_excluded_acc = {row.get('acc_name') for row in load_pipeline_rows() if row.get('acc_name') and row.get('merge_add_date')}

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

fig_idx = 0
plot_list = []

station_directory_path = Path.home() / "DATAFLOW_v3" / "STATIONS" / f"MINGO0{station}"
station_directory = str(station_directory_path)
config_file_directory = str(Path.home() / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "ONLINE_RUN_DICTIONARY" / f"STATION_{station}")
working_directory_path = station_directory_path / "STAGE_1" / "EVENT_DATA" / "STEP_3"
working_directory = str(working_directory_path)
acc_working_directory = working_directory
source_directory = str(station_directory_path / "STAGE_1" / "EVENT_DATA" / "STEP_2_TO_3_OUTPUT")

event_data_level = station_directory_path / "STAGE_1" / "EVENT_DATA"

# status_csv_path = os.path.join(working_directory, "big_event_file_joiner_status.csv")
# status_timestamp = append_status_row(status_csv_path)

# Define subdirectories relative to the working directory
base_directories = {
    "source_directory": source_directory,
    "base_plots_directory": os.path.join(acc_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(acc_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(acc_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(acc_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "unprocessed_directory": os.path.join(acc_working_directory, "INPUT_FILES/UNPROCESSED"),
    "processing_directory": os.path.join(acc_working_directory, "INPUT_FILES/PROCESSING"),
    "error_directory": os.path.join(acc_working_directory, "INPUT_FILES/ERROR"),
    "completed_directory": os.path.join(acc_working_directory, "INPUT_FILES/COMPLETED"),
    
    "acc_events_directory": os.path.join(event_data_level, "STEP_3_TO_4_OUTPUT"),
    "acc_rejected_directory": os.path.join(acc_working_directory, "INPUT_FILES/REJECTED"),
}

# Create ALL directories if they don't already exist
directories_to_create = {
    key: path
    for key, path in base_directories.items()
    if key not in {"source_directory"}
}
for directory in directories_to_create.values():
    os.makedirs(directory, exist_ok=True)

# Path to big_event_data.csv
big_event_file = os.path.join(base_directories["acc_events_directory"], "big_event_data.csv")

# Erase all files in the figure_directory
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory) if os.path.exists(figure_directory) else []

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))


def normalise_basename(name: str) -> str:
    base = name
    if base.startswith(("cleaned_", "calibrated_", "fitted_", "corrected_", "accumulated_", "listed_")):
        base = base.split("_", 1)[1]
    base = base.replace(".parquet", "").replace(".csv", "")
    return base

def load_step2_metadata(metadata_csv: Path) -> Tuple[pd.DataFrame, Set[str]]:
    if metadata_csv.exists():
        try:
            df = pd.read_csv(metadata_csv)
        except Exception as exc:
            print(f"Warning: unable to read {metadata_csv}: {exc}")
            df = pd.DataFrame(columns=['filename_base'])
    else:
        df = pd.DataFrame(columns=['filename_base'])

    if 'filename_base' not in df.columns:
        df['filename_base'] = ""
    if 'acc_join_timestamp' not in df.columns:
        df['acc_join_timestamp'] = ""

    processed: Set[str] = set()
    mask = df['acc_join_timestamp'].astype(str).str.strip() != ""
    for raw_base in df.loc[mask, 'filename_base'].dropna():
        base_str = str(raw_base).strip()
        if not base_str:
            continue
        processed.add(f"accumulated_{normalise_basename(base_str)}.csv")

    return df, processed

# --------------------------------------------------------------------------------------------
# Move accumulated files from STEP_2_TO_3_OUTPUT into STEP_3 pipeline directories ------------
# --------------------------------------------------------------------------------------------

source_directory = base_directories["source_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
processing_directory = base_directories["processing_directory"]
error_directory = base_directories["error_directory"]
completed_directory = base_directories["completed_directory"]

step2_metadata_path = station_directory_path / 'STAGE_1' / 'EVENT_DATA' / 'STEP_2' / 'METADATA' / 'step_2_metadata_execution.csv'
step2_metadata_df, processed_files = load_step2_metadata(step2_metadata_path)

if os.path.isdir(source_directory):
    source_files = {f for f in os.listdir(source_directory) if f.lower().endswith('.csv')}
else:
    source_files = set()

unprocessed_files = {f for f in os.listdir(unprocessed_directory) if f.lower().endswith('.csv')}
processing_files_existing = {f for f in os.listdir(processing_directory) if f.lower().endswith('.csv')}
completed_files = {f for f in os.listdir(completed_directory) if f.lower().endswith('.csv')}

files_to_move = source_files - unprocessed_files - processing_files_existing - completed_files

print("Source directory:", source_directory)
print("Files in source_directory are: ", sorted(source_files))

# If in source_files but not in files_to_move, then erase the source file
for file_name in source_files - files_to_move:
    src_path = os.path.join(source_directory, file_name)
    try:
        os.remove(src_path)
        print(f"Removed duplicate source file {file_name}.")
    except Exception as exc:
        print(f"Failed to remove duplicate source file {file_name}: {exc}")

for file_name in sorted(files_to_move):
    src_path = os.path.join(source_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        shutil.move(src_path, dest_path)
        print(f"Moved {file_name} to UNPROCESSED directory.")
    except Exception as exc:
        print(f"Failed to move {file_name} to UNPROCESSED: {exc}")

if requeue:
    requeued_bases: Set[str] = set()

    def requeue_from_directory(directory: str, file_name: str, label: str, *, respect_processing: bool) -> None:
        src_path = Path(directory) / file_name
        dest_path = Path(unprocessed_directory) / file_name
        processing_path = Path(processing_directory) / file_name
        if dest_path.exists():
            print(f"Skipping requeue of {file_name}; already present in UNPROCESSED.")
            return
        if respect_processing and processing_path.exists():
            print(f"Skipping requeue of {file_name}; already in PROCESSING queue.")
            return
        try:
            shutil.move(str(src_path), str(dest_path))
            print(f"Requeued {file_name} from {label} to UNPROCESSED.")
            processed_files.discard(file_name)
            requeued_bases.add(normalise_basename(file_name.replace('accumulated_', '')))
        except Exception as exc:
            print(f"Failed to requeue {file_name} from {label}: {exc}")

    completed_csvs = sorted(
        f for f in os.listdir(completed_directory) if f.lower().endswith('.csv')
    )
    for file_name in completed_csvs:
        requeue_from_directory(completed_directory, file_name, "COMPLETED", respect_processing=True)

    processing_csvs = sorted(
        f for f in os.listdir(processing_directory) if f.lower().endswith('.csv')
    )
    for file_name in processing_csvs:
        requeue_from_directory(processing_directory, file_name, "PROCESSING", respect_processing=False)

    if requeued_bases and not step2_metadata_df.empty:
        mask = step2_metadata_df['filename_base'].astype(str).str.strip().isin(requeued_bases)
        if mask.any():
            step2_metadata_df.loc[mask, 'acc_join_timestamp'] = ""
            try:
                step2_metadata_df.to_csv(step2_metadata_path, index=False)
                print("Cleared acc_join_timestamp for requeued basenames.")
            except Exception as exc:
                print(f"Warning: unable to update {step2_metadata_path}: {exc}")

for file_name in sorted(os.listdir(unprocessed_directory)):
    if not file_name.lower().endswith('.csv'):
        continue
    src_path = os.path.join(unprocessed_directory, file_name)
    dest_path = os.path.join(processing_directory, file_name)
    if os.path.exists(dest_path):
        continue
    try:
        shutil.move(src_path, dest_path)
        print(f"Moved {file_name} to PROCESSING directory.")
    except Exception as exc:
        print(f"Failed to move {file_name} to PROCESSING: {exc}")

processing_files = sorted(Path(processing_directory).glob('*.csv'))
if not processing_files:
    print("No accumulated CSV files available for processing.")
    sys.exit(0)


def round_to_significant_digits(x):
    if isinstance(x, float):
        return float(f"{x:.{SIG_DIGITS}g}")
    return x

SOURCE_COLUMN = "acc_source_basenames"
METADATA_COLUMNS = {SOURCE_COLUMN}




def combine_duplicates(group):
    # print("-----------------------------------------------------------------")
    
    # Drop fully NaN rows (except 'Time')
    group = group.dropna(
        subset=[col for col in group.columns if col not in ["Time", "execution_date"] and col not in METADATA_COLUMNS],
        how='all'
    )

    if group.empty:
        return group

    # Ensure metadata column exists
    if SOURCE_COLUMN not in group.columns:
        group = group.assign(**{SOURCE_COLUMN: ""})

    group = group.copy()
    group[SOURCE_COLUMN] = group[SOURCE_COLUMN].fillna("")

    def source_key(value: str) -> Tuple[str, ...]:
        tokens = [normalise_basename(token.strip()) for token in value.split(";") if token.strip()]
        return tuple(sorted(tokens))

    group["__source_key"] = group[SOURCE_COLUMN].apply(source_key)
    group = group.sort_values("execution_date", ascending=False)

    earliest_time = group["Time"].min()
    mismatch = group[group["Time"] != earliest_time]
    if not mismatch.empty:
        warnings.warn(
            "combine_duplicates received rows with different Time values; keeping latest row.",
            RuntimeWarning,
        )
        latest_row = group.iloc[[0]].drop(columns="__source_key", errors='ignore')
        return latest_row

    grouped_by_source = group.groupby("__source_key", dropna=False, sort=False)

    newest_rows = []
    for _, subset in grouped_by_source:
        newest_rows.append(subset.iloc[0])

    if not newest_rows:
        return group.drop(columns="__source_key", errors='ignore').iloc[0:0]

    deduped = pd.DataFrame(newest_rows)
    if deduped.empty:
        return deduped

    deduped = deduped.reset_index(drop=True)

    if len(deduped) == 1:
        return deduped.drop(columns="__source_key", errors='ignore').iloc[[0]]

    result: dict[str, object] = {}
    result["Time"] = deduped.iloc[0]["Time"]
    result["execution_date"] = deduped["execution_date"].max()

    tokens: list[str] = []
    for value in deduped[SOURCE_COLUMN]:
        for token in value.split(";"):
            token = normalise_basename(token.strip())
            if token and token not in tokens:
                tokens.append(token)
    result[SOURCE_COLUMN] = ";".join(tokens)

    for col in deduped.columns:
        if col in {"Time", "execution_date", SOURCE_COLUMN, "__source_key"}:
            continue
        numeric = pd.to_numeric(deduped[col], errors='coerce')
        if not numeric.isna().all():
            total = numeric.fillna(0).sum()
            if pd.notna(total) and float(total).is_integer():
                result[col] = int(total)
            else:
                result[col] = float(total)
        else:
            non_null = deduped[col].dropna()
            result[col] = non_null.iloc[0] if not non_null.empty else np.nan

    return pd.DataFrame([pd.Series(result)])


big_event_df = pd.DataFrame()
loaded_existing_file = False

if load_big_event_file and os.path.exists(big_event_file):
    big_event_df = pd.read_csv(big_event_file, sep=',', parse_dates=['Time'])
    loaded_existing_file = True
    print(f"Loaded existing big_event_data.csv with {len(big_event_df)} rows.")
elif load_big_event_file:
    print("Existing big_event_data.csv not found. Creating a new one from available ACC files.")
else:
    print("Configuration requests fresh big_event_data.csv creation.")

acc_directory = base_directories["processing_directory"]
acc_files = sorted([f for f in os.listdir(acc_directory) if f.lower().endswith('.csv')])

files_to_process = [f for f in acc_files if f not in processed_files and f not in merge_excluded_acc]
processed_any_new_file = False
successful_files: list[Tuple[str, str]] = []
successful_updates: list[Tuple[str, pd.Timestamp]] = []

if files_to_process:
    iterator = tqdm(files_to_process, total=len(files_to_process), desc="Joining ACC CSV files")
else:
    iterator = []

for acc_file in iterator:
    acc_path = os.path.join(acc_directory, acc_file)
    if not os.path.isfile(acc_path):
        print(f"File {acc_path} not found; skipping.")
        continue

    source_list: list[str] = []
    execution_stamp: Optional[str] = None

    try:
        with open(acc_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    break
                payload = stripped[1:].strip()
                if payload.startswith("source_basenames="):
                    raw_sources = payload.split("=", 1)[1].strip()
                    if raw_sources:
                        source_list = [normalise_basename(token.strip()) for token in raw_sources.split(";") if token.strip()]
                elif payload.startswith("execution_date="):
                    execution_stamp = payload.split("=", 1)[1].strip()

        new_data = pd.read_csv(acc_path, comment="#", sep=',', parse_dates=['Time'])
        # Ensure Time column uses datetime and floor to minute
        new_data['Time'] = pd.to_datetime(new_data['Time'], errors='coerce').dt.floor('1min')
        # Attach metadata columns
        if not source_list:
            source_list = [normalise_basename(Path(acc_file).stem)]
        # Preserve order + deduplicate
        seen_sources = []
        for source in source_list:
            if source and source not in seen_sources:
                seen_sources.append(source)
        new_data[SOURCE_COLUMN] = ";".join(seen_sources)

        if execution_stamp:
            try:
                exec_dt = pd.to_datetime(execution_stamp, format="%Y-%m-%d_%H.%M.%S")
            except ValueError:
                exec_dt = pd.to_datetime(execution_stamp, errors='coerce')
        else:
            exec_dt = pd.to_datetime(os.path.getmtime(acc_path), unit='s')

        if isinstance(exec_dt, pd.Timestamp) and pd.isna(exec_dt):
            exec_dt = pd.to_datetime(os.path.getmtime(acc_path), unit='s')

        new_data = new_data.replace(0, np.nan).copy()
        new_data["execution_date"] = exec_dt
        big_event_df = pd.concat([big_event_df, new_data], ignore_index=True)
        processed_files.add(acc_file)
        processed_any_new_file = True
        update_merge_timestamp(acc_file, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        successful_files.append((acc_file, acc_path))
        file_base = normalise_basename(acc_file.replace("accumulated_", ""))
        successful_updates.append((file_base, exec_dt))
    except Exception as exc:
        print(f"Error processing {acc_file}: {exc}")
        error_target = os.path.join(error_directory, acc_file)
        try:
            shutil.move(acc_path, error_target)
            print(f"Moved {acc_file} to ERROR directory.")
        except Exception as move_exc:
            print(f"Failed to move {acc_file} to ERROR directory: {move_exc}")
        continue

if not files_to_process:
    if loaded_existing_file and acc_files:
        print("No new ACC CSV files found to append.")
    elif not acc_files:
        print("No ACC CSV files found in the processing directory. Nothing to process.")

needs_aggregation = processed_any_new_file or not loaded_existing_file

if big_event_df.empty:
    print("No data available to aggregate or save.")
elif needs_aggregation:
    if "Time" not in big_event_df.columns:
        print("Column 'Time' missing; cannot aggregate big_event_df.")
    else:
        print("Grouping the ACC files by 'Time' and combining duplicates...")
        print("Columns in big_event_df:", big_event_df.columns.to_list())
        tqdm.pandas()
        big_event_df = big_event_df.groupby('Time', as_index=False).progress_apply(combine_duplicates).reset_index(drop=True)

        if not isinstance(big_event_df, pd.DataFrame):
            print("Warning: big_event_df is not a DataFrame. Converting it...")
            big_event_df = big_event_df.to_frame()

        big_event_df = big_event_df.sort_values(by="Time")
        print("Replacing 0s with NaNs...")
        big_event_df = big_event_df.replace(0, np.nan)

        big_event_df.to_csv(big_event_file, sep=',', index=False)
        print(big_event_df.columns.to_list())
        print(f"Saved big_event_data.csv with {len(big_event_df)} rows.")
else:
    print("No new ACC files to append. Existing big_event_data.csv left unchanged.")

if successful_files:
    if needs_aggregation and not big_event_df.empty:
        for acc_file, original_path in successful_files:
            processing_path = original_path if os.path.exists(original_path) else os.path.join(acc_directory, acc_file)
            if not os.path.exists(processing_path):
                continue
            completed_target = os.path.join(completed_directory, acc_file)
            try:
                shutil.move(processing_path, completed_target)
                print(f"Moved {acc_file} to COMPLETED directory.")
            except Exception as move_exc:
                print(f"Warning: unable to move {acc_file} to COMPLETED directory: {move_exc}")

        if not step2_metadata_df.empty and successful_updates:
            for base, exec_dt in successful_updates:
                mask = step2_metadata_df['filename_base'].astype(str).str.strip() == base
                if not mask.any():
                    continue
                if pd.notna(exec_dt):
                    ts_str = pd.to_datetime(exec_dt).strftime("%Y-%m-%d_%H.%M.%S")
                else:
                    ts_str = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
                step2_metadata_df.loc[mask, 'acc_join_timestamp'] = ts_str
            try:
                step2_metadata_df.to_csv(step2_metadata_path, index=False)
            except Exception as exc:
                print(f"Warning: unable to update {step2_metadata_path}: {exc}")
    else:
        print("Deferred moving processed files to COMPLETED because aggregation did not execute.")

# mark_status_complete(status_csv_path, status_timestamp)
