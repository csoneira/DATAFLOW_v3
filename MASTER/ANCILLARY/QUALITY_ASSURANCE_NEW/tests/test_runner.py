from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.common import deduplicate_metadata_rows_with_report  # noqa: E402
from MASTER.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.runner import (  # noqa: E402
    _collect_step_outputs,
    _generate_station_plots,
    _write_step_outputs,
)


class RunnerTests(unittest.TestCase):
    def test_deduplicate_metadata_rows_reports_overwritten_duplicate_basenames(self) -> None:
        meta_df = pd.DataFrame(
            {
                "filename_base": ["mi0124074013648", "mi0124074013648", "mi0124074030910"],
                "execution_timestamp": [
                    "2026-04-22_10.00.00",
                    "2026-04-22_10.05.00",
                    "2026-04-22_10.02.00",
                ],
                "metric": [1.0, 2.0, 3.0],
            }
        )

        deduped_df, overwritten_df = deduplicate_metadata_rows_with_report(meta_df)

        self.assertEqual(deduped_df["filename_base"].tolist(), ["mi0124074013648", "mi0124074030910"])
        self.assertEqual(deduped_df.loc[deduped_df["filename_base"] == "mi0124074013648", "metric"].iloc[0], 2.0)
        self.assertEqual(len(overwritten_df), 1)
        self.assertEqual(overwritten_df.loc[0, "overwritten_status"], "overwritten")
        self.assertEqual(overwritten_df.loc[0, "filename_base"], "mi0124074013648")
        self.assertEqual(int(overwritten_df.loc[0, "source_row_index"]), 0)
        self.assertEqual(int(overwritten_df.loc[0, "kept_source_row_index"]), 1)

    def test_collect_step_outputs_reads_task_id_from_task_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_dir = Path(tmp_dir) / "STEP_1_SAMPLE"
            files_dir = step_dir / "TASK_7" / "STATIONS" / "MINGO01" / "OUTPUTS" / "FILES"
            files_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "filename_base": ["mi0124074013648"],
                    "qa_status": ["pass"],
                    "qa_timestamp": ["2024-03-14 01:36:48"],
                }
            ).to_csv(files_dir / "MINGO01_task_7_sample_pass.csv", index=False)

            pd.DataFrame(
                {
                    "filename_base": ["mi0124074013648"],
                    "evaluation_column": ["metric_a"],
                    "status": ["pass"],
                    "reason": [""],
                }
            ).to_csv(files_dir / "sample_column_evaluations.csv", index=False)

            summary_df, eval_df = _collect_step_outputs(step_dir, "sample", "MINGO01")

            self.assertEqual(int(eval_df.loc[0, "task_id"]), 7)
            self.assertEqual(str(summary_df.loc[0, "step_name"]), "STEP_1_SAMPLE")
            self.assertEqual(str(summary_df.loc[0, "step_display_name"]), "sample")
            self.assertEqual(int(summary_df.loc[0, "qa_evaluated_columns"]), 1)

    def test_collect_step_outputs_ignores_stale_eval_when_manifest_has_no_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_dir = Path(tmp_dir) / "STEP_1_SAMPLE"
            files_dir = step_dir / "TASK_7" / "STATIONS" / "MINGO01" / "OUTPUTS" / "FILES"
            files_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "filename_base": ["mi0124074013648"],
                    "sample_pass": [1.0],
                    "qa_timestamp": ["2024-03-14 01:36:48"],
                }
            ).to_csv(files_dir / "MINGO01_task_7_sample_pass.csv", index=False)

            pd.DataFrame(
                {
                    "filename_base": ["mi0124074013648"],
                    "evaluation_column": ["metric_a"],
                    "source_column": ["metric_a"],
                    "status": ["pass"],
                    "reason": [""],
                }
            ).to_csv(files_dir / "sample_column_evaluations.csv", index=False)

            pd.DataFrame(
                {
                    "column_name": ["metric_a"],
                    "requested_category": ["plot_only"],
                    "effective_plot": [1],
                    "effective_quality": [0],
                }
            ).to_csv(files_dir / "MINGO01_task_7_sample_column_manifest.csv", index=False)

            summary_df, eval_df = _collect_step_outputs(step_dir, "sample", "MINGO01")

            self.assertTrue(eval_df.empty)
            self.assertEqual(int(summary_df.loc[0, "qa_evaluated_columns"]), 0)
            self.assertTrue(pd.isna(summary_df.loc[0, "qa_pass_fraction"]))

    def test_write_step_outputs_uses_step_output_tree_not_step_stations_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            step_dir = Path(tmp_dir) / "STEP_1_SAMPLE"
            summary_df = pd.DataFrame(
                {
                    "filename_base": ["mi0124074013648"],
                    "qa_status": ["pass"],
                    "qa_evaluated_columns": [1],
                    "qa_passed_columns": [1],
                    "qa_failed_columns": [0],
                    "qa_warning_columns": [0],
                    "qa_pass_fraction": [1.0],
                    "step_name": ["STEP_1_SAMPLE"],
                    "step_display_name": ["sample"],
                }
            )
            eval_df = pd.DataFrame(
                {
                    "filename_base": ["mi0124074013648"],
                    "task_id": [7],
                    "source_column": ["metric_a"],
                    "evaluation_column": ["metric_a"],
                    "status": ["pass"],
                    "plot_timestamp": ["2024-03-14 01:36:48"],
                }
            )

            _write_step_outputs(step_dir, "sample", "MINGO01", summary_df, eval_df)

            self.assertTrue((step_dir / "OUTPUTS" / "MINGO01" / "FILES" / "MINGO01_sample_step_summary.csv").exists())
            self.assertFalse((step_dir / "STATIONS").exists())

    def test_generate_station_plots_special_group_applies_sharey_and_ylim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_output_dir = Path(tmp_dir) / "STEP_1_SAMPLE" / "TASK_1"
            analyzed_df = pd.DataFrame(
                {
                    "plot_x": [1, 2, 3],
                    "metric_a": [2.0, 4.0, 6.0],
                    "metric_b": [20.0, 40.0, 60.0],
                }
            )
            created_figures: dict[str, object] = {}

            def fake_savefig(fig: object, fname: str | Path, *args: object, **kwargs: object) -> None:
                created_figures[str(fname)] = fig
                Path(fname).touch()

            with patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
                created_paths = _generate_station_plots(
                    task_output_dir=task_output_dir,
                    station_name="MINGO01",
                    task_id=1,
                    metadata_type="sample",
                    analyzed_df=analyzed_df,
                    plot_columns=["metric_a", "metric_b"],
                    config={
                        "x_axis": {"mode": "column", "column": "plot_x"},
                        "plots": {"format": "png"},
                    },
                    plot_config={
                        "default": {"default_ncols": 2},
                        "special": [
                            {
                                "name": "paired_metrics",
                                "mode": "columns",
                                "columns": ["metric_*"],
                                "ncols": 2,
                                "sharey": True,
                                "ylim": [0, 100],
                            }
                        ],
                    },
                )

            self.assertEqual(len(created_paths), 1)
            self.assertTrue(created_paths[0].exists())

            figure = created_figures[str(created_paths[0])]
            axes = figure.axes
            self.assertEqual(len(axes), 2)
            self.assertTrue(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
            self.assertEqual(tuple(round(value, 5) for value in axes[0].get_ylim()), (0.0, 100.0))
            self.assertEqual(tuple(round(value, 5) for value in axes[1].get_ylim()), (0.0, 100.0))

    def test_generate_station_plots_panels_group_overlays_explicit_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_output_dir = Path(tmp_dir) / "STEP_1_SAMPLE" / "TASK_1"
            analyzed_df = pd.DataFrame(
                {
                    "plot_x": [1, 2, 3],
                    "overall_metric": [2.0, 4.0, 6.0],
                    "self_metric": [1.0, 2.0, 3.0],
                    "overall_other": [5.0, 10.0, 15.0],
                    "self_other": [2.5, 5.0, 7.5],
                }
            )
            created_figures: dict[str, object] = {}

            def fake_savefig(fig: object, fname: str | Path, *args: object, **kwargs: object) -> None:
                created_figures[str(fname)] = fig
                Path(fname).touch()

            with patch("matplotlib.figure.Figure.savefig", new=fake_savefig):
                created_paths = _generate_station_plots(
                    task_output_dir=task_output_dir,
                    station_name="MINGO01",
                    task_id=1,
                    metadata_type="sample",
                    analyzed_df=analyzed_df,
                    plot_columns=[
                        "overall_metric",
                        "self_metric",
                        "overall_other",
                        "self_other",
                    ],
                    config={
                        "x_axis": {"mode": "column", "column": "plot_x"},
                        "plots": {"format": "png"},
                    },
                    plot_config={
                        "default": {"default_ncols": 2},
                        "special": [
                            {
                                "name": "explicit_panels",
                                "mode": "panels",
                                "ncols": 2,
                                "sharey": True,
                                "ylim": [0, 20],
                                "series_labels": ["overall", "self"],
                                "panels": [
                                    {
                                        "title": "metric",
                                        "columns": ["overall_metric", "self_metric"],
                                    },
                                    {
                                        "title": "other",
                                        "columns": ["overall_other", "self_other"],
                                    },
                                ],
                            }
                        ],
                    },
                )

            self.assertEqual(len(created_paths), 1)
            self.assertTrue(created_paths[0].exists())

            figure = created_figures[str(created_paths[0])]
            axes = figure.axes
            self.assertEqual(len(axes), 2)
            self.assertEqual(axes[0].get_title(), "metric")
            self.assertEqual(axes[1].get_title(), "other")
            self.assertTrue(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
            self.assertEqual(tuple(round(value, 5) for value in axes[0].get_ylim()), (0.0, 20.0))
            self.assertEqual(len(axes[0].lines), 2)
            self.assertEqual(len(axes[1].lines), 2)


if __name__ == "__main__":
    unittest.main()
