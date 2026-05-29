from __future__ import annotations

import argparse
from pathlib import Path

from src.process_files import process_selection


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label STEP_2 event files with gate bitmasks and compute per-minute gate rates.",
    )
    parser.add_argument(
        "--selection-config",
        required=True,
        help="Path to the YAML file describing station and date selection.",
    )
    parser.add_argument(
        "--gate-config",
        required=True,
        help="Path to the YAML file describing the gate definitions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    result = process_selection(
        selection_config_path=Path(args.selection_config),
        gate_config_path=Path(args.gate_config),
    )

    print(f"Processed {len(result.file_selection.selected_files)} input files.")
    print(f"Wrote {len(result.labelled_outputs)} labelled files to {result.selection.output_dir / 'labelled_events'}")
    if result.rate_output_path is not None:
        print(f"Wrote rate table to {result.rate_output_path}")
    if result.file_selection.invalid_files:
        print(f"Skipped {len(result.file_selection.invalid_files)} malformed filenames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
