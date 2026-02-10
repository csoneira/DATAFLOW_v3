#!/usr/bin/env python3
"""Build a parameter-to-metadata dictionary by matching file_name to filename_base."""
# See config/COLUMN_REFERENCE_TASKS.md for task-specific column lists by type.

# COLUMN REFERENCE (from TASK_1 metadata headers; expected same structure in TASK_2-5)
# RAW_TT_COUNT:
#   raw_tt_1234_count, raw_tt_123_count, raw_tt_124_count, raw_tt_12_count, raw_tt_134_count,
#   raw_tt_13_count, raw_tt_14_count, raw_tt_1_count, raw_tt_234_count, raw_tt_23_count,
#   raw_tt_24_count, raw_tt_2_count, raw_tt_34_count, raw_tt_3_count, raw_tt_4_count
# CLEAN_TT_COUNT:
#   clean_tt_1234_count, clean_tt_123_count, clean_tt_124_count, clean_tt_12_count,
#   clean_tt_134_count, clean_tt_13_count, clean_tt_14_count, clean_tt_1_count,
#   clean_tt_234_count, clean_tt_23_count, clean_tt_24_count, clean_tt_2_count,
#   clean_tt_34_count, clean_tt_3_count, clean_tt_4_count
# RAW_TO_CLEAN_TT_COUNT:
#   raw_to_clean_tt_1234_1234_count, raw_to_clean_tt_1234_123_count, raw_to_clean_tt_1234_124_count,
#   raw_to_clean_tt_1234_12_count, raw_to_clean_tt_1234_134_count, raw_to_clean_tt_1234_13_count,
#   raw_to_clean_tt_1234_14_count, raw_to_clean_tt_1234_1_count, raw_to_clean_tt_1234_234_count,
#   raw_to_clean_tt_1234_23_count, raw_to_clean_tt_1234_24_count, raw_to_clean_tt_1234_2_count,
#   raw_to_clean_tt_1234_34_count, raw_to_clean_tt_1234_3_count, raw_to_clean_tt_1234_4_count,
#   raw_to_clean_tt_123_123_count, raw_to_clean_tt_123_12_count, raw_to_clean_tt_123_13_count,
#   raw_to_clean_tt_123_1_count, raw_to_clean_tt_123_23_count, raw_to_clean_tt_123_2_count,
#   raw_to_clean_tt_123_3_count, raw_to_clean_tt_124_124_count, raw_to_clean_tt_124_12_count,
#   raw_to_clean_tt_124_14_count, raw_to_clean_tt_124_1_count, raw_to_clean_tt_124_24_count,
#   raw_to_clean_tt_124_2_count, raw_to_clean_tt_124_4_count, raw_to_clean_tt_12_12_count,
#   raw_to_clean_tt_12_1_count, raw_to_clean_tt_12_2_count, raw_to_clean_tt_134_134_count,
#   raw_to_clean_tt_134_13_count, raw_to_clean_tt_134_14_count, raw_to_clean_tt_134_1_count,
#   raw_to_clean_tt_134_34_count, raw_to_clean_tt_134_3_count, raw_to_clean_tt_134_4_count,
#   raw_to_clean_tt_13_13_count, raw_to_clean_tt_13_1_count, raw_to_clean_tt_13_3_count,
#   raw_to_clean_tt_14_14_count, raw_to_clean_tt_14_1_count, raw_to_clean_tt_14_4_count,
#   raw_to_clean_tt_1_1_count, raw_to_clean_tt_234_234_count, raw_to_clean_tt_234_23_count,
#   raw_to_clean_tt_234_24_count, raw_to_clean_tt_234_2_count, raw_to_clean_tt_234_34_count,
#   raw_to_clean_tt_234_3_count, raw_to_clean_tt_234_4_count, raw_to_clean_tt_23_23_count,
#   raw_to_clean_tt_23_2_count, raw_to_clean_tt_23_3_count, raw_to_clean_tt_24_24_count,
#   raw_to_clean_tt_24_2_count, raw_to_clean_tt_24_4_count, raw_to_clean_tt_2_2_count,
#   raw_to_clean_tt_34_34_count, raw_to_clean_tt_34_3_count, raw_to_clean_tt_34_4_count,
#   raw_to_clean_tt_3_3_count
# Q_ENTRIES_ORIGINAL:
#   Q1_B_1_entries_original, Q1_B_2_entries_original, Q1_B_3_entries_original, Q1_B_4_entries_original,
#   Q1_F_1_entries_original, Q1_F_2_entries_original, Q1_F_3_entries_original, Q1_F_4_entries_original,
#   Q2_B_1_entries_original, Q2_B_2_entries_original, Q2_B_3_entries_original, Q2_B_4_entries_original,
#   Q2_F_1_entries_original, Q2_F_2_entries_original, Q2_F_3_entries_original, Q2_F_4_entries_original,
#   Q3_B_1_entries_original, Q3_B_2_entries_original, Q3_B_3_entries_original, Q3_B_4_entries_original,
#   Q3_F_1_entries_original, Q3_F_2_entries_original, Q3_F_3_entries_original, Q3_F_4_entries_original,
#   Q4_B_1_entries_original, Q4_B_2_entries_original, Q4_B_3_entries_original, Q4_B_4_entries_original,
#   Q4_F_1_entries_original, Q4_F_2_entries_original, Q4_F_3_entries_original, Q4_F_4_entries_original
# Q_ENTRIES_FINAL:
#   Q1_B_1_entries_final, Q1_B_2_entries_final, Q1_B_3_entries_final, Q1_B_4_entries_final,
#   Q1_F_1_entries_final, Q1_F_2_entries_final, Q1_F_3_entries_final, Q1_F_4_entries_final,
#   Q2_B_1_entries_final, Q2_B_2_entries_final, Q2_B_3_entries_final, Q2_B_4_entries_final,
#   Q2_F_1_entries_final, Q2_F_2_entries_final, Q2_F_3_entries_final, Q2_F_4_entries_final,
#   Q3_B_1_entries_final, Q3_B_2_entries_final, Q3_B_3_entries_final, Q3_B_4_entries_final,
#   Q3_F_1_entries_final, Q3_F_2_entries_final, Q3_F_3_entries_final, Q3_F_4_entries_final,
#   Q4_B_1_entries_final, Q4_B_2_entries_final, Q4_B_3_entries_final, Q4_B_4_entries_final,
#   Q4_F_1_entries_final, Q4_F_2_entries_final, Q4_F_3_entries_final, Q4_F_4_entries_final
# T_ENTRIES_ORIGINAL:
#   T1_B_1_entries_original, T1_B_2_entries_original, T1_B_3_entries_original, T1_B_4_entries_original,
#   T1_F_1_entries_original, T1_F_2_entries_original, T1_F_3_entries_original, T1_F_4_entries_original,
#   T2_B_1_entries_original, T2_B_2_entries_original, T2_B_3_entries_original, T2_B_4_entries_original,
#   T2_F_1_entries_original, T2_F_2_entries_original, T2_F_3_entries_original, T2_F_4_entries_original,
#   T3_B_1_entries_original, T3_B_2_entries_original, T3_B_3_entries_original, T3_B_4_entries_original,
#   T3_F_1_entries_original, T3_F_2_entries_original, T3_F_3_entries_original, T3_F_4_entries_original,
#   T4_B_1_entries_original, T4_B_2_entries_original, T4_B_3_entries_original, T4_B_4_entries_original,
#   T4_F_1_entries_original, T4_F_2_entries_original, T4_F_3_entries_original, T4_F_4_entries_original
# T_ENTRIES_FINAL:
#   T1_B_1_entries_final, T1_B_2_entries_final, T1_B_3_entries_final, T1_B_4_entries_final,
#   T1_F_1_entries_final, T1_F_2_entries_final, T1_F_3_entries_final, T1_F_4_entries_final,
#   T2_B_1_entries_final, T2_B_2_entries_final, T2_B_3_entries_final, T2_B_4_entries_final,
#   T2_F_1_entries_final, T2_F_2_entries_final, T2_F_3_entries_final, T2_F_4_entries_final,
#   T3_B_1_entries_final, T3_B_2_entries_final, T3_B_3_entries_final, T3_B_4_entries_final,
#   T3_F_1_entries_final, T3_F_2_entries_final, T3_F_3_entries_final, T3_F_4_entries_final,
#   T4_B_1_entries_final, T4_B_2_entries_final, T4_B_3_entries_final, T4_B_4_entries_final,
#   T4_F_1_entries_final, T4_F_2_entries_final, T4_F_3_entries_final, T4_F_4_entries_final
# OTHER:
#   analysis_mode, execution_timestamp, filename_base, valid_lines_in_binary_file_percentage,
#   zeroed_percentage_P1s1, zeroed_percentage_P1s2, zeroed_percentage_P1s3, zeroed_percentage_P1s4,
#   zeroed_percentage_P2s1, zeroed_percentage_P2s2, zeroed_percentage_P2s3, zeroed_percentage_P2s4,
#   zeroed_percentage_P3s1, zeroed_percentage_P3s2, zeroed_percentage_P3s3, zeroed_percentage_P3s4,
#   zeroed_percentage_P4s1, zeroed_percentage_P4s2, zeroed_percentage_P4s3, zeroed_percentage_P4s4

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "config/pipeline_config.json"
DEFAULT_METADATA = (
    "/home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/"
    "TASK_1/METADATA/task_1_metadata_specific.csv"
)
DEFAULT_PARAMS = (
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/"
    "step_final_simulation_params.csv"
)
DEFAULT_OUT = str(BASE_DIR / "STEP_1_BUILD/output/task_01/param_metadata_dictionary.csv")


def _task_metadata_path(station_id: int, task_id: int) -> Path:
    station = f"MINGO{station_id:02d}"
    return Path(
        "/home/mingo/DATAFLOW_v3/STATIONS"
        f"/{station}/STAGE_1/EVENT_DATA/STEP_1/TASK_{task_id}/METADATA/"
        f"task_{task_id}_metadata_specific.csv"
    )


def _load_config(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def _apply_task_overrides(config: dict, task_id: int) -> dict:
    overrides = (config.get("task_settings") or {}).get(str(task_id))
    if not overrides:
        return config
    merged = dict(config)
    merged.update(overrides)
    return merged


def _find_params_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Params path not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Params path is not a file or directory: {path}")

    candidates: list[Path] = []
    for csv_path in path.rglob("*.csv"):
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        if "file_name" in cols and "param_set_id" in cols:
            candidates.append(csv_path)
    if not candidates and path.name == "tools":
        fallback = path.parent / "SIMULATED_DATA"
        if fallback.exists():
            for csv_path in fallback.rglob("*.csv"):
                try:
                    cols = pd.read_csv(csv_path, nrows=0).columns
                except Exception:
                    continue
                if "file_name" in cols and "param_set_id" in cols:
                    candidates.append(csv_path)

    if not candidates:
        raise FileNotFoundError(
            "No params CSV with columns 'file_name' and 'param_set_id' found under "
            f"{path}"
        )
    if len(candidates) > 1:
        candidates_str = "\n".join(str(p) for p in candidates)
        raise ValueError(
            "Multiple candidate params CSVs found; please pass one explicitly:\n"
            f"{candidates_str}"
        )
    return candidates[0]


def _select_metadata_columns(df: pd.DataFrame, cols: str | None, prefix: str | None) -> list[str]:
    if cols:
        if cols.strip().lower() == "all":
            return list(df.columns)
        return [c.strip() for c in cols.split(",") if c.strip()]
    if prefix:
        return [c for c in df.columns if c.startswith(prefix)]
    return list(df.columns)


def _aggregate_metadata(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if "filename_base" not in df.columns:
        raise KeyError("Metadata CSV must contain 'filename_base' column.")

    if method in ("latest", "earliest"):
        if "execution_timestamp" in df.columns:
            dt = pd.to_datetime(
                df["execution_timestamp"],
                format="%Y-%m-%d_%H.%M.%S",
                errors="coerce",
            )
            df = df.assign(_exec_dt=dt)
            df = df.sort_values(
                ["filename_base", "_exec_dt"],
                na_position="last",
                kind="mergesort",
            )
            if method == "latest":
                df = df.groupby("filename_base").tail(1)
            else:
                df = df.groupby("filename_base").head(1)
            df = df.drop(columns=["_exec_dt"])
        else:
            df = df.sort_index()
            if method == "latest":
                df = df.groupby("filename_base").tail(1)
            else:
                df = df.groupby("filename_base").head(1)
        return df

    if method in ("mean", "median"):
        num_cols = df.select_dtypes(include="number").columns
        if method == "mean":
            agg_df = df.groupby("filename_base")[num_cols].mean()
        else:
            agg_df = df.groupby("filename_base")[num_cols].median()
        agg_df = agg_df.reset_index()
        return agg_df

    if method == "none":
        return df

    raise ValueError(f"Unknown aggregation method: {method}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Match simulation parameter sets to metadata rows by file base name "
            "and output a dictionary CSV."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--station-id", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--metadata-agg",
        default="latest",
        choices=["latest", "earliest", "mean", "median", "none"],
        help="How to reduce multiple metadata rows per filename_base.",
    )
    parser.add_argument(
        "--metadata-cols",
        default=None,
        help="Comma-separated metadata columns to keep, or 'all'.",
    )
    parser.add_argument(
        "--metadata-prefix",
        default="raw_tt_",
        help="Prefix for metadata columns to keep if --metadata-cols is not set.",
    )
    parser.add_argument(
        "--param-file-col",
        default="file_name",
        help="Column in params CSV containing file name.",
    )
    parser.add_argument(
        "--metadata-file-col",
        default="filename_base",
        help="Column in metadata CSV containing base name.",
    )

    args = parser.parse_args()

    config = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            config = _load_config(cfg_path)
    station_id = args.station_id or int(config.get("station_id", 0))
    task_id = args.task_id or int(config.get("task_id", 1))
    dictionary_station_id = int(config.get("dictionary_station_id", station_id))
    dictionary_task_id = int(config.get("dictionary_task_id", task_id))
    config = _apply_task_overrides(config, task_id)

    metadata_path = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else _task_metadata_path(dictionary_station_id, dictionary_task_id)
    )
    params_path = _find_params_file(Path(args.params))

    out_template = config.get(
        "dictionary_csv",
        str(BASE_DIR / "STEP_1_BUILD/output/task_{task_id:02d}/param_metadata_dictionary.csv"),
    )
    if "{task_id" in out_template:
        out_default = out_template.format(task_id=task_id)
    else:
        out_default = out_template
    out_path = Path(args.out or out_default)

    if not metadata_path.exists():
        print(f"Metadata CSV not found: {metadata_path}", file=sys.stderr)
        return 1

    print(f"Using params CSV: {params_path}")
    metadata_df = pd.read_csv(metadata_path, low_memory=False)
    metadata_df = _aggregate_metadata(metadata_df, config.get("metadata_agg", args.metadata_agg))

    if args.metadata_file_col not in metadata_df.columns:
        print(
            f"Metadata file column '{args.metadata_file_col}' not found in "
            f"{metadata_path}",
            file=sys.stderr,
        )
        return 1

    metadata_cols = _select_metadata_columns(
        metadata_df,
        config.get("metadata_cols", args.metadata_cols),
        config.get("metadata_prefix", args.metadata_prefix),
    )
    if args.metadata_file_col not in metadata_cols:
        metadata_cols.insert(0, args.metadata_file_col)
    if len(metadata_cols) == 1:
        print(
            "Warning: no metadata columns matched selection; only filename_base will be kept.",
            file=sys.stderr,
        )
    metadata_df = metadata_df.loc[:, list(dict.fromkeys(metadata_cols))]

    params_df = pd.read_csv(params_path, low_memory=False)
    if args.param_file_col not in params_df.columns:
        print(
            f"Params file column '{args.param_file_col}' not found in {params_path}",
            file=sys.stderr,
        )
        return 1

    params_df = params_df.copy()
    params_df["filename_base"] = (
        params_df[args.param_file_col]
        .astype(str)
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    metadata_df = metadata_df.rename(columns={args.metadata_file_col: "filename_base"})

    dictionary_df = params_df.merge(metadata_df, on="filename_base", how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_csv(out_path, index=False)

    total = len(dictionary_df)
    if len(metadata_cols) > 1:
        present_cols = [col for col in metadata_cols[1:] if col in dictionary_df.columns]
        if present_cols:
            missing = dictionary_df[present_cols].isna().all(axis=1).sum()
        else:
            missing = 0
    else:
        missing = 0
    print(
        "Wrote dictionary:",
        out_path,
        f"(rows={total}, unmatched_metadata={missing})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
