from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.epoch_quality import (  # noqa: E402
    build_epoch_reference_table,
    build_scalar_value_frame,
    evaluate_scalar_frame,
)


class EpochQualityTests(unittest.TestCase):
    def test_build_scalar_value_frame_expands_vector_columns(self) -> None:
        df = pd.DataFrame(
            {
                "filename_base": ["f1", "f2"],
                "scalar_col": [10.0, 12.0],
                "vector_col": ["[1.0, 0.1]", "[0.9, -0.1]"],
            }
        )

        value_frame, specs_df = build_scalar_value_frame(df, ["scalar_col", "vector_col"])

        self.assertIn("scalar_col", value_frame.columns)
        self.assertIn("vector_col__0", value_frame.columns)
        self.assertIn("vector_col__1", value_frame.columns)
        self.assertEqual(set(specs_df["source_column"]), {"scalar_col", "vector_col"})

    def test_evaluate_scalar_frame_uses_epoch_references(self) -> None:
        df = pd.DataFrame(
            {
                "filename_base": ["f1", "f2", "f3"],
                "epoch_id": ["epoch_a", "epoch_a", "epoch_a"],
                "scalar_col": [100.0, 102.0, 98.0],
                "vector_col": ["[1.0, 0.1]", "[1.1, 0.2]", "[0.9, 0.0]"],
            }
        )

        value_frame, specs_df = build_scalar_value_frame(df, ["scalar_col", "vector_col"])
        reference_df = build_epoch_reference_table(value_frame, specs_df, df["epoch_id"])
        eval_df = evaluate_scalar_frame(
            df[["filename_base", "epoch_id"]],
            value_frame,
            reference_df,
            defaults={"tolerance_mode": "relative_pct", "tolerance_value": 0.05, "min_samples": 2},
            column_rules={"vector_col__1": {"tolerance_mode": "absolute", "tolerance_value": 0.15}},
        )

        self.assertFalse(eval_df.empty)
        self.assertTrue({"pass", "fail"} >= set(eval_df["status"]))
        scalar_eval = eval_df[eval_df["evaluation_column"] == "scalar_col"]
        self.assertTrue((scalar_eval["status"] == "pass").all())

    def test_evaluate_scalar_frame_supports_zscore_thresholds(self) -> None:
        df = pd.DataFrame(
            {
                "filename_base": ["f1", "f2", "f3", "f4"],
                "epoch_id": ["epoch_a", "epoch_a", "epoch_a", "epoch_a"],
                "scalar_col": [10.0, 11.0, 9.0, 25.0],
            }
        )

        value_frame, specs_df = build_scalar_value_frame(df, ["scalar_col"])
        reference_df = build_epoch_reference_table(value_frame, specs_df, df["epoch_id"])
        self.assertIn("scale_std", reference_df.columns)

        eval_df = evaluate_scalar_frame(
            df[["filename_base", "epoch_id"]],
            value_frame,
            reference_df,
            defaults={"center_method": "mean", "tolerance_mode": "zscore", "tolerance_value": 1.0, "min_samples": 2},
        )

        self.assertFalse(eval_df.empty)
        status_by_file = dict(zip(eval_df["filename_base"], eval_df["status"]))
        self.assertEqual(status_by_file["f4"], "fail")


if __name__ == "__main__":
    unittest.main()
