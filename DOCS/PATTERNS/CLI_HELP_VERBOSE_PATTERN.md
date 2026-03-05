---
title: CLI Help and Verbose Pattern
description: Standard CLI contract for long-running pipeline scripts.
last_updated: 2026-02-24
status: active
---

# CLI Help and Verbose Pattern

## Goal
Use a consistent CLI contract for scripts that historically parsed `sys.argv` manually.

Required behavior:
- `-h/--help` usage text
- positional `station` argument
- optional positional `input_file`
- optional named `--input-file`
- `-v/--verbose` plus `DATAFLOW_VERBOSE` env support

## Standard CLI shape

```bash
python3 script_x.py [-h] [--input-file INPUT_FILE] [-v] [{0,1,2,3,4}] [input_file]
```

Rules:
- `station` choices: `0,1,2,3,4`
- positional and named input-file forms are mutually exclusive
- verbose flag should resolve with env var fallback

```python
VERBOSE = bool(os.environ.get("DATAFLOW_VERBOSE")) or CLI_ARGS.verbose
```

## Template

```python
import argparse

STATION_CHOICES = ("0", "1", "2", "3", "4")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run <stage/step/task description>.",
    )
    parser.add_argument(
        "station",
        nargs="?",
        choices=STATION_CHOICES,
        help="Station identifier (0, 1, 2, 3, or 4).",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional input file path to process instead of auto-selecting.",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file_flag",
        help="Optional input file path (named form).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose execution logging.",
    )
    return parser


CLI_PARSER = _build_cli_parser()
CLI_ARGS = CLI_PARSER.parse_args()
if CLI_ARGS.input_file and CLI_ARGS.input_file_flag:
    CLI_PARSER.error("Use either positional input_file or --input-file, not both.")
```

Station resolution with notebook fallback:

```python
run_jupyter_notebook = bool(config.get("run_jupyter_notebook", False))
if CLI_ARGS.station is not None:
    station = CLI_ARGS.station
elif run_jupyter_notebook:
    station = str(config.get("jupyter_station_default_task_X", "2"))
else:
    CLI_PARSER.error(
        "No station provided. Pass <station> or enable run_jupyter_notebook in TASK_<N>/config_task_<N>.yaml."
    )

if station not in STATION_CHOICES:
    CLI_PARSER.error("Invalid station. Choose one of: 0, 1, 2, 3, 4.")
```

Input-file resolution:

```python
selected_input_file = CLI_ARGS.input_file_flag or CLI_ARGS.input_file
if selected_input_file:
    user_file_path = selected_input_file
    user_file_selection = True
else:
    user_file_selection = False
```

## Migration checklist
- add `argparse` parser near script entrypoint
- replace raw `sys.argv` station handling
- replace raw `sys.argv` input-file handling
- preserve existing notebook fallback behavior
- keep log/filter semantics unchanged

## Validation commands

```bash
python3 -m py_compile path/to/script.py
python3 path/to/script.py -h
python3 path/to/script.py 2 --verbose
python3 path/to/script.py 2 --input-file /tmp/example.csv
```

## Current references
Implemented in:
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/script_2_clean_to_cal.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_3/script_3_cal_to_list.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_4/script_4_list_to_fit.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_5/script_5_fit_to_post.py`
