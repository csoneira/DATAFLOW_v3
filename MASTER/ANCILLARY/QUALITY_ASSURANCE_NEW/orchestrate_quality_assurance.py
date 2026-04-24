#!/usr/bin/env python3
"""Run configured QUALITY_ASSURANCE_NEW steps and rebuild TOTAL_SUMMARY."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Any

QA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QA_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.common import load_yaml_mapping, normalize_station_name
from MASTER.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.runner import rebuild_step_summaries, run_step
from MASTER.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.total_summary import build_total_summary


def _load_pipeline_steps(qa_root: Path) -> list[dict[str, Any]]:
    config = load_yaml_mapping(qa_root / "config_pipeline.yaml", required=True)
    raw_steps = config.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("config_pipeline.yaml must define a 'steps' list.")

    steps: list[dict[str, Any]] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        step_name = str(item.get("step_name", "")).strip()
        if not step_name:
            continue
        steps.append(
            {
                "order": int(item.get("order", 0)),
                "step_name": step_name,
                "display_name": str(item.get("display_name", step_name)).strip() or step_name,
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return sorted(steps, key=lambda step: (step["order"], step["step_name"]))


def _parse_station_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [normalize_station_name(value) for value in values]


def _parse_step_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="*", help="Optional station list like: MINGO01 MINGO02")
    parser.add_argument("--steps", nargs="*", help="Optional step-name filter like: STEP_1_CALIBRATIONS")
    parser.add_argument(
        "--mode",
        choices=("often", "plot"),
        default="plot",
        help="Cron-friendly run mode: 'often' updates tables only; 'plot' also regenerates plots.",
    )
    parser.add_argument("--aggregate-only", action="store_true", help="Only rebuild TOTAL_SUMMARY from existing outputs.")
    parser.add_argument("--skip-total-summary", action="store_true", help="Run steps only.")
    args = parser.parse_args(argv)

    root_config = load_yaml_mapping(QA_ROOT / "config.yaml", required=True)
    pipeline_steps = _load_pipeline_steps(QA_ROOT)
    step_filter = _parse_step_filter(args.steps)
    stations_override = _parse_station_list(args.stations)
    generate_plots = args.mode == "plot"
    print(f"QUALITY_ASSURANCE_NEW mode={args.mode} generate_plots={generate_plots}")

    if not args.aggregate_only:
        for step in pipeline_steps:
            if not step["enabled"]:
                continue
            if step_filter is not None and step["step_name"] not in step_filter:
                continue
            step_dir = QA_ROOT / "STEPS" / str(step["step_name"])
            run_step(
                step_dir,
                root_config=root_config,
                stations_override=stations_override,
                generate_plots=generate_plots,
            )
    else:
        for step in pipeline_steps:
            if not step["enabled"]:
                continue
            if step_filter is not None and step["step_name"] not in step_filter:
                continue
            step_dir = QA_ROOT / "STEPS" / str(step["step_name"])
            rebuild_step_summaries(
                step_dir,
                root_config=root_config,
                stations_override=stations_override,
            )

    if not args.skip_total_summary:
        build_total_summary(
            QA_ROOT,
            root_config=root_config,
            pipeline_steps=[step for step in pipeline_steps if step_filter is None or step["step_name"] in step_filter],
            stations_override=stations_override,
            generate_plots=generate_plots,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
