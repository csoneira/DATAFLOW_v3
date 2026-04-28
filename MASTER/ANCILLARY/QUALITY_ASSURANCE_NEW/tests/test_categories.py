from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE_NEW.qa_core.categories import build_column_manifest  # noqa: E402


class CategoryTests(unittest.TestCase):
    def test_default_is_plot_only_and_ignore_wins(self) -> None:
        df = pd.DataFrame(
            {
                "metric_a": [1.0, 2.0],
                "metric_b": [3.0, 4.0],
                "param_hash": ["x", "y"],
            }
        )

        manifest_df = build_column_manifest(
            df,
            {
                "quality_and_plot": ["metric_b"],
                "ignore": ["param_hash"],
            },
        )
        by_column = manifest_df.set_index("column_name")

        self.assertEqual(by_column.loc["metric_a", "requested_category"], "plot_only")
        self.assertEqual(int(by_column.loc["metric_a", "effective_plot"]), 1)
        self.assertEqual(int(by_column.loc["metric_a", "effective_quality"]), 0)

        self.assertEqual(by_column.loc["metric_b", "requested_category"], "quality_and_plot")
        self.assertEqual(int(by_column.loc["metric_b", "effective_plot"]), 1)
        self.assertEqual(int(by_column.loc["metric_b", "effective_quality"]), 1)

        self.assertEqual(by_column.loc["param_hash", "requested_category"], "ignore")
        self.assertEqual(int(by_column.loc["param_hash", "effective_plot"]), 0)
        self.assertEqual(int(by_column.loc["param_hash", "effective_quality"]), 0)

    def test_non_numeric_columns_do_not_plot_or_enter_quality(self) -> None:
        df = pd.DataFrame(
            {
                "tt_loo_mode": ["leave_one_out", "full"],
                "correct_angle": [1, 0],
            }
        )

        manifest_df = build_column_manifest(
            df,
            {
                "quality_and_plot": ["*"],
            },
        )
        by_column = manifest_df.set_index("column_name")

        self.assertEqual(by_column.loc["tt_loo_mode", "requested_category"], "quality_and_plot")
        self.assertEqual(int(by_column.loc["tt_loo_mode", "effective_plot"]), 0)
        self.assertEqual(int(by_column.loc["tt_loo_mode", "effective_quality"]), 0)
        self.assertEqual(by_column.loc["tt_loo_mode", "note"], "non_numeric")

        self.assertEqual(int(by_column.loc["correct_angle", "effective_plot"]), 1)
        self.assertEqual(int(by_column.loc["correct_angle", "effective_quality"]), 1)

    def test_shorthand_quality_rule_promotes_matching_limit_column(self) -> None:
        df = pd.DataFrame(
            {
                "plane_combination_plane_q_dif_sum_low_limit": [1.0, 1.1],
                "plane_combination_plane_q_dif_sum_low_rows_failed_pct": [2.0, 3.0],
            }
        )

        manifest_df = build_column_manifest(
            df,
            {
                "quality_and_plot": ["q_dif_sum_low"],
                "plot_only": ["*_rows_failed_pct", "*_limit"],
            },
        )
        by_column = manifest_df.set_index("column_name")

        self.assertEqual(
            by_column.loc["plane_combination_plane_q_dif_sum_low_limit", "requested_category"],
            "quality_and_plot",
        )
        self.assertEqual(
            int(by_column.loc["plane_combination_plane_q_dif_sum_low_limit", "effective_quality"]),
            1,
        )
        self.assertEqual(
            by_column.loc["plane_combination_plane_q_dif_sum_low_rows_failed_pct", "requested_category"],
            "plot_only",
        )


if __name__ == "__main__":
    unittest.main()
