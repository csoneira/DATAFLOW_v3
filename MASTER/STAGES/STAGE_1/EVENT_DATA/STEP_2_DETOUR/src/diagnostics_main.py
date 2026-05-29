from __future__ import annotations

import argparse
from pathlib import Path

from src.diagnostics import run_diagnostics


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute gate efficiencies and create combined diagnostic plots from the aggregated rate table.",
    )
    parser.add_argument(
        "--selection-config",
        required=True,
        help="Path to the YAML file describing station and output selection.",
    )
    parser.add_argument(
        "--gate-config",
        required=True,
        help="Path to the YAML file describing the gate definitions.",
    )
    parser.add_argument(
        "--rate-file",
        default=None,
        help="Optional override for the aggregated rate/count file produced by src.main.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    result = run_diagnostics(
        selection_config_path=Path(args.selection_config),
        gate_config_path=Path(args.gate_config),
        rate_file=Path(args.rate_file) if args.rate_file else None,
    )

    print(f"Read aggregated rate table from {result.input_rate_path}")
    print(f"Wrote diagnostic table to {result.diagnostic_output_path}")
    if result.plot_output_paths:
        print(f"Wrote {len(result.plot_output_paths)} diagnostic plots to {result.plot_output_paths[0].parent}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
