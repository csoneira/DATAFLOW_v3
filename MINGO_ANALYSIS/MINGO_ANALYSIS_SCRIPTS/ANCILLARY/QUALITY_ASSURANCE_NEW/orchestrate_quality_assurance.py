#!/usr/bin/env python3
"""Run configured QUALITY_ASSURANCE_NEW steps and rebuild TOTAL_SUMMARY."""

from __future__ import annotations

from pathlib import Path
import argparse
import fcntl
import os
import shutil
import sys
import tempfile
from typing import Any

QA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QA_ROOT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MINGO_ANALYSIS.MINGO_ANALYSIS_SCRIPTS.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.common import load_yaml_mapping, normalize_station_name
from MINGO_ANALYSIS.MINGO_ANALYSIS_SCRIPTS.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.runner import rebuild_step_summaries, run_step
from MINGO_ANALYSIS.MINGO_ANALYSIS_SCRIPTS.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.total_summary import build_total_summary
from MINGO_ANALYSIS.MINGO_ANALYSIS_SCRIPTS.ANCILLARY.QUALITY_ASSURANCE_NEW.build_problematic_basename_lists import build_from_file


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


def _publish_stage1_product_metadata(
    repo_root: Path,
    root_config: dict[str, Any],
    stations_override: list[str] | None,
) -> tuple[int, int]:
    """Atomically refresh the Stage 1 product metadata used by QA."""

    raw_stations = stations_override or root_config.get("stations") or []
    station_names = [normalize_station_name(value) for value in raw_stations]
    copied_files = 0
    copied_bytes = 0
    for station_name in station_names:
        station_root = repo_root / "MINGO_ANALYSIS" / "MINGO_ANALYSIS_STATIONS" / station_name
        for task_id in range(6):
            source_dir = station_root / "STAGE_1" / "EVENT_DATA" / "STEP_1" / f"TASK_{task_id}" / "METADATA"
            product_dir = station_root / "STAGE_1_PRODUCTS" / "EVENT_DATA" / "METADATA" / f"TASK_{task_id}"
            if not source_dir.is_dir():
                continue
            product_dir.mkdir(parents=True, exist_ok=True)
            for source_path in sorted(source_dir.glob(f"task_{task_id}_metadata_*.csv")):
                if not source_path.is_file():
                    continue
                lock_path = source_dir / "OPERATION" / f"{source_path.name}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                destination_path = product_dir / source_path.name
                temporary_name: str | None = None
                with lock_path.open("a", encoding="utf-8") as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
                    source_stat = source_path.stat()
                    if destination_path.exists():
                        destination_stat = destination_path.stat()
                        if (
                            destination_stat.st_size == source_stat.st_size
                            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
                        ):
                            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                            continue
                    try:
                        with source_path.open("rb") as source_handle:
                            with tempfile.NamedTemporaryFile(
                                mode="wb",
                                dir=product_dir,
                                prefix=f".{source_path.name}.",
                                suffix=".tmp",
                                delete=False,
                            ) as temporary_handle:
                                temporary_name = temporary_handle.name
                                shutil.copyfileobj(source_handle, temporary_handle)
                                temporary_handle.flush()
                                os.fsync(temporary_handle.fileno())
                        os.replace(temporary_name, destination_path)
                        shutil.copystat(source_path, destination_path)
                        temporary_name = None
                    finally:
                        if temporary_name is not None:
                            Path(temporary_name).unlink(missing_ok=True)
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                copied_files += 1
                copied_bytes += destination_path.stat().st_size
    return copied_files, copied_bytes


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
        copied_files, copied_bytes = _publish_stage1_product_metadata(
            REPO_ROOT, root_config, stations_override
        )
        print(
            "Published Stage 1 product metadata for QA: "
            f"files={copied_files} bytes={copied_bytes}"
        )

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
        manifest_path, manifest_df = build_from_file()
        print(f"QA problematic-basename manifest: {manifest_path} ({len(manifest_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
