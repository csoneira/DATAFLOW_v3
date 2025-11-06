#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

try:
    from MASTER.common.execution_logger import set_station, start_timer
except ImportError:
    def set_station(_: str) -> None:
        pass

    def start_timer(_: str) -> None:
        pass


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    ensure_directory(path.parent)


def normalise_basename(name: str) -> str:
    base = Path(name).stem
    prefixes = (
        "cleaned_",
        "calibrated_",
        "fitted_",
        "corrected_",
        "accumulated_",
        "listed_",
    )
    for prefix in prefixes:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    return base


def initialise_metadata(path: Path, headers: Iterable[str]) -> None:
    if path.exists():
        return
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))


def append_metadata(path: Path, row: Dict[str, object]) -> None:
    ensure_parent(path)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_accumulated_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns:
        raise ValueError(f"Column 'Time' missing in {csv_path}")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    if df["Time"].isna().all():
        raise ValueError(f"No valid datetime values in 'Time' column for {csv_path}")
    df = df.sort_values("Time").reset_index(drop=True)
    numeric_cols = []
    for col in df.columns:
        if col == "Time":
            continue
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.notna().any():
            df[col] = numeric_series
            numeric_cols.append(col)
        else:
            df[col] = df[col]
    return df


def plot_time_series(df: pd.DataFrame, columns: List[str], output_pdf: Path) -> None:
    ensure_parent(output_pdf)
    with PdfPages(output_pdf) as pdf:
        for column in columns:
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().sum() == 0:
                continue

            plt.figure(figsize=(11.69, 4.0))  # A4 width in landscape / single row
            plt.plot(df["Time"], series, linewidth=0.9)
            plt.title(column)
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            plt.gcf().autofmt_xdate()
            pdf.savefig()
            plt.close()


def move_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        ensure_parent(dest)
        shutil.move(str(src), str(dest))


def main() -> None:
    start_timer(__file__)

    if len(sys.argv) < 2:
        print("Usage: accumulated_check.py <station>")
        sys.exit(1)

    station = sys.argv[1]
    set_station(station)

    station_path = Path.home() / "DATAFLOW_v3" / "STATIONS" / f"MINGO0{station}"
    stage1_event_dir = station_path / "STAGE_1" / "EVENT_DATA"
    step3_output_dir = stage1_event_dir / "STEP_3_TO_4_OUTPUT"
    step3_output_file = step3_output_dir / "big_event_data.csv"

    step4_base = stage1_event_dir / "STEP_4"
    input_base = step4_base / "INPUT_FILES"
    unprocessed_dir = input_base / "UNPROCESSED"
    processing_dir = input_base / "PROCESSING"
    error_dir = input_base / "ERROR_DIRECTORY"
    completed_dir = input_base / "COMPLETED"
    plots_dir = step4_base / "PLOTS"
    metadata_dir = step4_base / "METADATA"
    output_stage_dir = station_path / "STAGE_1_to_2"
    pdf_output_dir = plots_dir / "PDF"

    for directory in (
        unprocessed_dir,
        processing_dir,
        error_dir,
        completed_dir,
        pdf_output_dir,
        metadata_dir,
        output_stage_dir,
    ):
        ensure_directory(directory)

    step3_exports = sorted(p for p in step3_output_dir.glob("*.csv"))
    for export_file in step3_exports:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staged_name = f"{export_file.stem}_{timestamp}{export_file.suffix}"
        destination = unprocessed_dir / staged_name
        shutil.move(str(export_file), str(destination))
        print(f"Moved {export_file} to {destination}")

    unprocessed_files = sorted(p for p in unprocessed_dir.glob("*.csv"))
    for pending_file in unprocessed_files:
        target_in_processing = processing_dir / pending_file.name
        move_if_exists(pending_file, target_in_processing)

    processing_files = sorted(p for p in processing_dir.glob("*.csv"))
    if not processing_files:
        print("No accumulated CSV files available for processing.")
        sys.exit(0)

    execution_metadata_path = metadata_dir / "step_4_metadata_execution.csv"
    specific_metadata_path = metadata_dir / "step_4_metadata_specific.csv"

    execution_headers = [
        "filename_base",
        "execution_timestamp",
        "rows_processed",
        "columns_processed",
        "output_filename",
    ]
    specific_headers = [
        "filename_base",
        "metric",
        "value",
    ]

    initialise_metadata(execution_metadata_path, execution_headers)
    initialise_metadata(specific_metadata_path, specific_headers)

    processed_successfully: List[Tuple[Path, Path]] = []

    for csv_path in processing_files:
        print(f"Processing {csv_path.name} ...")
        try:
            df = read_accumulated_csv(csv_path)
        except Exception as exc:
            print(f"Failed reading {csv_path.name}: {exc}")
            move_if_exists(csv_path, error_dir / csv_path.name)
            continue

        base_name = normalise_basename(csv_path.name)
        execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        numeric_columns = [col for col in df.columns if col != "Time" and pd.api.types.is_numeric_dtype(df[col])]
        pdf_path = pdf_output_dir / f"{base_name}_timeseries_{execution_timestamp}.pdf"
        try:
            plot_time_series(df, numeric_columns, pdf_path)
        except Exception as exc:
            warnings.warn(f"Unable to generate plots for {csv_path.name}: {exc}", RuntimeWarning)

        output_path = output_stage_dir / "event_data_table.parquet"
        df.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        print(f"Written processed Parquet to {output_path}")

        execution_row = {
            "filename_base": base_name,
            "execution_timestamp": execution_timestamp,
            "rows_processed": len(df),
            "columns_processed": len(df.columns),
            "output_filename": output_path.name,
        }
        append_metadata(execution_metadata_path, execution_row)

        summary_metrics = {
            "first_time": df["Time"].min(),
            "last_time": df["Time"].max(),
            "minutes_covered": df["Time"].nunique(),
            "numeric_columns": len(numeric_columns),
        }
        for metric, value in summary_metrics.items():
            append_metadata(
                specific_metadata_path,
                {
                    "filename_base": base_name,
                    "metric": metric,
                    "value": value,
                },
            )

        processed_successfully.append((csv_path, output_path))

    for processing_file, _ in processed_successfully:
        destination = completed_dir / processing_file.name
        move_if_exists(processing_file, destination)
        print(f"Moved {processing_file.name} to COMPLETED directory.")

    print("accumulated_check.py finished.")


if __name__ == "__main__":
    main()
