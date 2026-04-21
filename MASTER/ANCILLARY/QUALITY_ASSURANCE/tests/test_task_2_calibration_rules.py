from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = (
    REPO_ROOT
    / "MASTER"
    / "ANCILLARY"
    / "QUALITY_ASSURANCE"
    / "STEP_1_CALIBRATIONS"
    / "TASK_2"
    / "task_2_calibrations.py"
)
MODULE_SPEC = spec_from_file_location("qa_task_2_calibrations", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
TASK2 = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(TASK2)


def _write_rule_table(tmp_dir: str) -> Path:
    csv_path = Path(tmp_dir) / "rules.csv"
    csv_path.write_text(
        "\n".join(
            [
                "pattern,figure,plot_enabled,quality_enabled,center_method,tolerance_mode,tolerance_value,min_samples,notes",
                "*_crstlk_pedestal,crstlk,true,true,mean,zscore,3.0,8,crosstalk pedestal",
                "*_crstlk_limit,crstlk,true,true,mean,zscore,3.0,8,crosstalk limit",
                "*_Q_B,q_b,true,true,mean,zscore,2.5,8,back charge",
                "*_Q_FB_coeffs,q_fb_coeffs,true,true,mean,zscore,3.5,8,charge coeffs",
                "*,unexpected,true,false,mean,zscore,3.0,8,default fallback",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path


class Task2CalibrationRuleTests(unittest.TestCase):
    def test_component_rule_inherits_source_column_rule(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = _write_rule_table(tmp_dir)
            config = {
                "column_rule_table_csv": str(csv_path),
                "plot_ignore_patterns_common": [],
                "plot_ignore_patterns_extra": [],
            }

            resolved = TASK2._resolve_effective_column_rule(
                config,
                "P1_s1_Q_FB_coeffs__0",
                source_column="P1_s1_Q_FB_coeffs",
            )
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.pattern, "*_Q_FB_coeffs")

            threshold_rule = TASK2._quality_rule_for_column(
                config,
                "P1_s1_Q_FB_coeffs__0",
                source_column="P1_s1_Q_FB_coeffs",
            )
            self.assertEqual(threshold_rule.center_method, "mean")
            self.assertEqual(threshold_rule.tolerance_mode, "zscore")
            self.assertEqual(threshold_rule.tolerance_value, 3.5)

    def test_quality_columns_use_csv_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = _write_rule_table(tmp_dir)
            config = {
                "column_rule_table_csv": str(csv_path),
                "plot_ignore_patterns_common": ["param_hash"],
                "plot_ignore_patterns_extra": [],
                "quality": {"enabled": True},
            }

            quality_columns = TASK2._quality_columns_from_config(
                config=config,
                groups=[],
                available_columns=["param_hash", "P1_s1_Q_B", "P1_s1_Q_FB_coeffs", "new_numeric"],
            )
            self.assertEqual(quality_columns, ["P1_s1_Q_B", "P1_s1_Q_FB_coeffs"])

    def test_runtime_plot_groups_add_unexpected_numeric_columns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = _write_rule_table(tmp_dir)
            config = {
                "column_rule_table_csv": str(csv_path),
                "plot_ignore_patterns_common": ["param_hash", "date"],
                "plot_ignore_patterns_extra": ["skip_*"],
                "unexpected_plot_group_name": "unexpected",
                "unexpected_group_max_columns": 4,
                "plots": {
                    "plot_groups": [
                        {
                            "name": "crstlk",
                            "columns": ["*_crstlk_pedestal", "*_crstlk_limit"],
                            "pair_suffixes": ["crstlk_pedestal", "crstlk_limit"],
                        }
                    ]
                },
            }
            df = pd.DataFrame(
                {
                    "param_hash": ["a", "b"],
                    "date": ["2026-04-20", "2026-04-21"],
                    "skip_debug": [1.0, 2.0],
                    "P1_s1_crstlk_pedestal": [1.0, 1.1],
                    "P1_s1_crstlk_limit": [2.0, 2.1],
                    "P1_s1_Q_B": [10.0, 10.2],
                    "new_numeric": [5.0, 5.1],
                }
            )

            groups = TASK2._runtime_plot_groups(config, df)
            group_names = [group["name"] for group in groups]

            self.assertIn("crstlk", group_names)
            self.assertIn("q_b", group_names)
            self.assertIn("unexpected", group_names)

            q_b_group = next(group for group in groups if group["name"] == "q_b")
            unexpected_group = next(group for group in groups if group["name"] == "unexpected")
            self.assertEqual(q_b_group["columns"], ["P1_s1_Q_B"])
            self.assertEqual(unexpected_group["columns"], ["new_numeric"])


if __name__ == "__main__":
    unittest.main()
