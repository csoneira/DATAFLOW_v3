#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.step1_shared import canonicalize_step1_columns


OUTPUT_PATH = Path(
    "/home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/columns_review.md"
)


@dataclass(frozen=True)
class Section:
    title: str
    display_pattern: str
    glob_pattern: str
    exclude_prefixes: tuple[str, ...] = ()


SECTIONS = [
    Section(
        title="TASK_0 -> TASK_1",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_1/INPUT_FILES/COMPLETED_DIRECTORY/raw_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_1/INPUT_FILES/COMPLETED_DIRECTORY/raw_*.parquet"
        ),
        exclude_prefixes=("selftrigger_",),
    ),
    Section(
        title="TASK_0 -> SELFTRIGGER",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_0/OUTPUT_FILES/selftrigger_raw_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_0/OUTPUT_FILES/selftrigger_raw_*.parquet"
        ),
    ),
    Section(
        title="TASK_1 -> TASK_2",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_2/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_2/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
    ),
    Section(
        title="TASK_2 -> TASK_3",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_3/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_3/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
    ),
    Section(
        title="TASK_3 -> TASK_4",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_4/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_4/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
    ),
    Section(
        title="TASK_4 -> TASK_5",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_5/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
            "TASK_5/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet"
        ),
    ),
    Section(
        title="TASK_5 -> OUT",
        display_pattern=(
            "~/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_2/"
            "INPUT_FILES/COMPLETED/*_*.parquet"
        ),
        glob_pattern=(
            "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_2/"
            "INPUT_FILES/COMPLETED/*_*.parquet"
        ),
    ),
]


def get_first_file(section: Section) -> Path | None:
    pattern = section.glob_pattern
    matches = [Path(match) for match in glob(pattern)]
    if section.exclude_prefixes:
        matches = [
            match
            for match in matches
            if not match.name.startswith(section.exclude_prefixes)
        ]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def get_columns(parquet_path: Path) -> list[str]:
    parquet_file = pq.ParquetFile(parquet_path)
    try:
        batch = next(parquet_file.iter_batches(batch_size=5))
        columns = list(batch.schema.names)
    except StopIteration:
        columns = list(parquet_file.schema_arrow.names)
    schema_df = pd.DataFrame(columns=columns)
    return list(canonicalize_step1_columns(schema_df).columns)


def build_document() -> tuple[str, list[str]]:
    lines = ["# Columns in pipeline files", ""]
    sample_files: list[str] = []

    for section in SECTIONS:
        lines.append(f"## {section.title}")
        lines.append(f"### from the {section.display_pattern}")
        lines.append("")

        sample_file = get_first_file(section)
        if sample_file is None:
            lines.append("No parquet files found")
        else:
            sample_files.append(str(sample_file))
            lines.extend(sorted(get_columns(sample_file), key=str.casefold))

        lines.append("")
        lines.append("")

    return "\n".join(lines), sample_files


def main() -> int:
    document, sample_files = build_document()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(document, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for sample_file in sample_files:
        print(f"Sampled: {sample_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
