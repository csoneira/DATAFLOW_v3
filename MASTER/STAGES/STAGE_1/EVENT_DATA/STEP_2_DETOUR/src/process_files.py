from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import sys

import pandas as pd

from src.gates import GateDefinition, apply_gates, load_gate_config
from src.rates import accumulate_rate_tables, build_chunk_rates, finalize_rate_table, write_rate_table
from src.select_files import FileSelectionResult, SelectedFile, SelectionConfig, discover_selected_files, load_selection_config


@dataclass(frozen=True)
class ProcessResult:
    selection: SelectionConfig
    gates: list[GateDefinition]
    file_selection: FileSelectionResult
    labelled_outputs: list[Path]
    rate_output_path: Path | None


def load_event_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file format: {path}")


def write_event_file(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output file format: {path}")


def process_selection(
    selection_config_path: Path,
    gate_config_path: Path,
) -> ProcessResult:
    selection = load_selection_config(selection_config_path)
    gates = load_gate_config(gate_config_path)
    file_selection = discover_selected_files(selection)

    if file_selection.invalid_files:
        for invalid_file in file_selection.invalid_files:
            print(
                f"Skipping malformed filename without parseable timestamp: {invalid_file}",
                file=sys.stderr,
            )

    if not file_selection.selected_files:
        raise FileNotFoundError("No input files matched the configured station and date selection.")

    labelled_dir = selection.output_dir / "labelled_events"
    labelled_dir.mkdir(parents=True, exist_ok=True)

    rate_accumulator: pd.DataFrame | None = None
    labelled_outputs: list[Path] = []

    for selected_file in file_selection.selected_files:
        df = load_event_file(selected_file.path)
        gate_mask = apply_gates(df, gates)
        df["gate_mask"] = gate_mask

        labelled_output_path = build_labelled_output_path(labelled_dir, selected_file)
        write_event_file(df, labelled_output_path)
        labelled_outputs.append(labelled_output_path)

        if selection.rate_output.enabled:
            chunk_rates = build_chunk_rates(
                df=df,
                gate_mask=gate_mask,
                gates=gates,
                time_column=selection.rate_output.time_column,
                bin_size=selection.rate_output.bin_size,
                selection_start=selection.start_datetime,
                selection_end=selection.end_datetime,
            )
            rate_accumulator = accumulate_rate_tables(rate_accumulator, chunk_rates)

        del df
        del gate_mask
        gc.collect()

    rate_output_path: Path | None = None
    if selection.rate_output.enabled:
        if rate_accumulator is None:
            raise RuntimeError("Rate output was enabled but no rate table was generated.")

        rate_table = finalize_rate_table(rate_accumulator, gates)
        resolved_rate_output = selection.resolve_output_path(selection.rate_output.output_file)
        rate_output_path = write_rate_table(rate_table, resolved_rate_output)

    return ProcessResult(
        selection=selection,
        gates=gates,
        file_selection=file_selection,
        labelled_outputs=labelled_outputs,
        rate_output_path=rate_output_path,
    )


def build_labelled_output_path(labelled_dir: Path, selected_file: SelectedFile) -> Path:
    return labelled_dir / f"{selected_file.station}_{selected_file.path.name}"
