#!/usr/bin/env python3
"""Run ordered QUALITY_ASSURANCE steps and build cross-step summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess
import sys
from typing import Any

import pandas as pd

QA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QA_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.status_plots import plot_column_status_grid, plot_step_score_grid
from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.task_setup import (
    ensure_task_station_tree,
    load_yaml_mapping,
    normalize_station_name,
    parse_filename_timestamp_series,
)


@dataclass(frozen=True)
class PipelineStep:
    order: int
    step_name: str
    display_name: str
    enabled: bool
    runner: Path


def _load_pipeline_steps(qa_root: Path) -> list[PipelineStep]:
    config = load_yaml_mapping(qa_root / "config_pipeline.yaml", required=True)
    raw_steps = config.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("config_pipeline.yaml must define a 'steps' list.")

    steps: list[PipelineStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        step_name = str(item.get("step_name", "")).strip()
        runner_rel = str(item.get("runner", "")).strip()
        if not step_name or not runner_rel:
            continue
        steps.append(
            PipelineStep(
                order=int(item.get("order", 0)),
                step_name=step_name,
                display_name=str(item.get("display_name", step_name)).strip() or step_name,
                enabled=bool(item.get("enabled", True)),
                runner=(qa_root / runner_rel).resolve(),
            )
        )
    return sorted(steps, key=lambda step: (step.order, step.step_name))


def _task_id_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("TASK_"):
            suffix = part.removeprefix("TASK_")
            if suffix.isdigit():
                return int(suffix)
    return None


def _step_station_output_dirs(step_dir: Path, station_name: str) -> tuple[Path, Path]:
    files_dir = step_dir / "STATIONS" / station_name / "OUTPUTS" / "FILES"
    plots_dir = step_dir / "STATIONS" / station_name / "OUTPUTS" / "PLOTS"
    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return files_dir, plots_dir


def _first_timestamp(series: pd.Series) -> pd.Timestamp | None:
    parsed = pd.to_datetime(series, errors="coerce")
    parsed = parsed.dropna()
    return None if parsed.empty else parsed.iloc[0]


def _summarize_step_outputs(step: PipelineStep, station_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    step_dir = QA_ROOT / step.step_name
    pass_paths = sorted(step_dir.glob(f"TASK_*/STATIONS/{station_name}/OUTPUTS/FILES/*_pass.csv"))
    eval_paths = sorted(step_dir.glob(f"TASK_*/STATIONS/{station_name}/OUTPUTS/FILES/*_column_evaluations.csv"))

    pass_frames: list[pd.DataFrame] = []
    for path in pass_paths:
        task_id = _task_id_from_path(path)
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["task_id"] = task_id
        frame["step_name"] = step.step_name
        frame["step_display_name"] = step.display_name
        pass_frames.append(frame)

    eval_frames: list[pd.DataFrame] = []
    for path in eval_paths:
        task_id = _task_id_from_path(path)
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["task_id"] = task_id
        frame["step_name"] = step.step_name
        frame["step_display_name"] = step.display_name
        eval_frames.append(frame)

    pass_df = pd.concat(pass_frames, ignore_index=True) if pass_frames else pd.DataFrame()
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()

    if pass_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "qa_timestamp" not in pass_df.columns:
        pass_df["qa_timestamp"] = parse_filename_timestamp_series(pass_df["filename_base"])
    else:
        pass_df["qa_timestamp"] = pd.to_datetime(pass_df["qa_timestamp"], errors="coerce")

    if not eval_df.empty:
        eval_df = eval_df.merge(
            pass_df[["filename_base", "task_id", "qa_timestamp"]].drop_duplicates(subset=["filename_base", "task_id"]),
            on=["filename_base", "task_id"],
            how="left",
        )
        eval_df["row_label"] = eval_df["task_id"].map(lambda value: f"TASK_{int(value)}" if pd.notna(value) else "TASK_?") + "::" + eval_df["evaluation_column"].astype(str)

    summary_base = (
        pass_df.groupby("filename_base", dropna=False)
        .agg(
            qa_timestamp=("qa_timestamp", _first_timestamp),
            qa_status_source=("qa_status", lambda values: ";".join(sorted(set(values.astype(str))))),
            qa_status_task_count=("task_id", "nunique"),
        )
        .reset_index()
    )

    if not eval_df.empty:
        step_agg = (
            eval_df.groupby("filename_base", dropna=False)
            .agg(
                qa_evaluated_columns=("status", lambda values: int(sum(item in {"pass", "fail"} for item in values))),
                qa_passed_columns=("status", lambda values: int(sum(item == "pass" for item in values))),
                qa_failed_columns=("status", lambda values: int(sum(item == "fail" for item in values))),
                qa_warning_columns=("status", lambda values: int(sum(item not in {"pass", "fail"} for item in values))),
            )
            .reset_index()
        )
        failed_observables = (
            eval_df.loc[eval_df["status"] == "fail", ["filename_base", "task_id", "evaluation_column"]]
            .assign(
                failed_label=lambda frame: frame["task_id"].map(lambda value: f"TASK_{int(value)}" if pd.notna(value) else "TASK_?")
                + "::"
                + frame["evaluation_column"].astype(str)
            )
            .groupby("filename_base")["failed_label"]
            .apply(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("qa_failed_observables")
        )
        warning_reasons = (
            eval_df.loc[eval_df["status"] != "pass", ["filename_base", "reason"]]
            .groupby("filename_base")["reason"]
            .apply(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("qa_warning_reasons")
        )
        summary_df = summary_base.merge(step_agg, on="filename_base", how="left")
        summary_df = summary_df.merge(failed_observables.reset_index(), on="filename_base", how="left")
        summary_df = summary_df.merge(warning_reasons.reset_index(), on="filename_base", how="left")
    else:
        summary_df = summary_base.copy()
        summary_df["qa_evaluated_columns"] = 0
        summary_df["qa_passed_columns"] = 0
        summary_df["qa_failed_columns"] = 0
        summary_df["qa_warning_columns"] = 0
        summary_df["qa_failed_observables"] = ""
        summary_df["qa_warning_reasons"] = ""

    for column_name in (
        "qa_evaluated_columns",
        "qa_passed_columns",
        "qa_failed_columns",
        "qa_warning_columns",
    ):
        summary_df[column_name] = pd.to_numeric(summary_df[column_name], errors="coerce").fillna(0).astype(int)

    summary_df["qa_pass_fraction"] = pd.Series(pd.NA, index=summary_df.index, dtype="Float64")
    valid_mask = summary_df["qa_evaluated_columns"] > 0
    summary_df.loc[valid_mask, "qa_pass_fraction"] = (
        summary_df.loc[valid_mask, "qa_passed_columns"] / summary_df.loc[valid_mask, "qa_evaluated_columns"]
    ).astype(float)

    def _step_status(row: pd.Series) -> str:
        source_tokens = set(str(row.get("qa_status_source", "")).split(";"))
        source_tokens.discard("")
        if row.get("qa_failed_columns", 0) > 0 or "fail" in source_tokens:
            return "fail"
        if row.get("qa_warning_columns", 0) > 0 or "warn" in source_tokens:
            return "warn"
        if "no_epoch_match" in source_tokens and row.get("qa_evaluated_columns", 0) == 0:
            return "no_epoch_match"
        if row.get("qa_evaluated_columns", 0) > 0 and row.get("qa_failed_columns", 0) == 0:
            return "pass"
        if source_tokens == {"out_of_scope"}:
            return "out_of_scope"
        if source_tokens == {"invalid_timestamp"}:
            return "invalid_timestamp"
        return "not_evaluated"

    summary_df["qa_status"] = summary_df.apply(_step_status, axis=1)
    summary_df["step_name"] = step.step_name
    summary_df["step_display_name"] = step.display_name
    summary_df = summary_df.sort_values("qa_timestamp", na_position="last").reset_index(drop=True)
    return summary_df, eval_df


def _write_step_outputs(step: PipelineStep, station_name: str, summary_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    step_dir = QA_ROOT / step.step_name
    files_dir, plots_dir = _step_station_output_dirs(step_dir, station_name)

    summary_path = files_dir / f"{station_name}_{step.display_name}_step_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not eval_df.empty:
        detail_path = files_dir / f"{station_name}_{step.display_name}_column_status.csv"
        eval_df.to_csv(detail_path, index=False)
        plot_column_status_grid(
            df=eval_df,
            x_column="qa_timestamp",
            y_column="row_label",
            status_column="status",
            out_path=plots_dir / f"{station_name}_{step.display_name}_quality_by_column.png",
            title=f"{station_name} {step.display_name} quality by column",
            max_rows_per_plot=120,
        )


def _build_final_outputs(steps: list[PipelineStep], station_name: str) -> None:
    step_summaries: list[pd.DataFrame] = []
    step_details: list[pd.DataFrame] = []
    for step in steps:
        summary_df, eval_df = _summarize_step_outputs(step, station_name)
        if summary_df.empty:
            continue
        _write_step_outputs(step, station_name, summary_df, eval_df)
        step_summaries.append(summary_df)
        if not eval_df.empty:
            step_details.append(eval_df)

    if not step_summaries:
        return

    final_dir = QA_ROOT / "STEP_FINAL_AGGREGATE"
    base_config = load_yaml_mapping(QA_ROOT / "config.yaml", required=True)
    ensure_task_station_tree(final_dir, base_config)
    files_dir, plots_dir = _step_station_output_dirs(final_dir, station_name)

    long_df = pd.concat(step_summaries, ignore_index=True)
    long_df = long_df.sort_values("qa_timestamp", na_position="last").reset_index(drop=True)

    wide_df = long_df[["filename_base", "qa_timestamp"]].drop_duplicates(subset=["filename_base"]).copy()
    for step in steps:
        summary_df = long_df.loc[long_df["step_name"] == step.step_name].copy()
        if summary_df.empty:
            continue
        renamed = summary_df[
            [
                "filename_base",
                "qa_status",
                "qa_pass_fraction",
                "qa_failed_observables",
                "qa_warning_reasons",
            ]
        ].rename(
            columns={
                "qa_status": f"{step.step_name}__qa_status",
                "qa_pass_fraction": f"{step.step_name}__qa_pass_fraction",
                "qa_failed_observables": f"{step.step_name}__qa_failed_observables",
                "qa_warning_reasons": f"{step.step_name}__qa_warning_reasons",
            }
        )
        wide_df = wide_df.merge(renamed, on="filename_base", how="left")

    status_columns = [f"{step.step_name}__qa_status" for step in steps if f"{step.step_name}__qa_status" in wide_df.columns]

    def _overall_status(row: pd.Series) -> str:
        statuses = {str(row[column]) for column in status_columns if pd.notna(row[column])}
        statuses.discard("")
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        if "no_epoch_match" in statuses:
            return "no_epoch_match"
        if statuses and statuses <= {"pass"}:
            return "pass"
        if statuses and statuses <= {"out_of_scope"}:
            return "out_of_scope"
        return "not_evaluated"

    wide_df["overall_status"] = wide_df.apply(_overall_status, axis=1)
    wide_df = wide_df.sort_values("qa_timestamp", na_position="last").reset_index(drop=True)
    wide_df.to_csv(files_dir / f"{station_name}_final_step_summary.csv", index=False)

    if step_details:
        final_detail_df = pd.concat(step_details, ignore_index=True).sort_values("qa_timestamp", na_position="last")
        final_detail_df.to_csv(files_dir / f"{station_name}_final_column_status.csv", index=False)

    plot_step_score_grid(
        df=long_df,
        x_column="qa_timestamp",
        y_column="step_display_name",
        score_column="qa_pass_fraction",
        y_order=[step.display_name for step in steps],
        out_path=plots_dir / f"{station_name}_final_step_scores.png",
        title=f"{station_name} final QA step scores",
    )


def run_pipeline(*, run_steps: bool) -> int:
    steps = [step for step in _load_pipeline_steps(QA_ROOT) if step.enabled]
    if run_steps:
        for step in steps:
            print(f"[qa] Running {step.step_name}")
            subprocess.run(["bash", str(step.runner)], check=True, cwd=QA_ROOT)

    base_config = load_yaml_mapping(QA_ROOT / "config.yaml", required=True)
    stations = [normalize_station_name(station) for station in base_config.get("stations", [0, 1, 2, 3, 4])]
    for station_name in stations:
        print(f"[qa] Aggregating {station_name}")
        _build_final_outputs(steps, station_name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true", help="Skip step runners and rebuild summaries/plots only.")
    args = parser.parse_args(argv)
    return run_pipeline(run_steps=not args.aggregate_only)


if __name__ == "__main__":
    raise SystemExit(main())
