from __future__ import annotations

from pathlib import Path
import csv
import sys
from tempfile import TemporaryDirectory
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.common.step1_shared import _normalize_metadata_row, normalize_metadata_file_schema  # noqa: E402


class Step1MetadataNormalizationTests(unittest.TestCase):
    def test_normalize_metadata_row_flattens_vectors_and_bools(self) -> None:
        row = _normalize_metadata_row(
            {
                "filename_base": "mi0124074013648",
                "P1_s1_Q_FB_coeffs": "[1.0, -2.5, 0.25]",
                "correct_angle": "True",
                "timtrack_projection_ellipse_tt_1234_available": False,
                "timtrack_projection_ellipse_contour_fractions": "0.25,0.50,0.75",
            }
        )

        self.assertNotIn("P1_s1_Q_FB_coeffs", row)
        self.assertEqual(row["P1_s1_Q_FB_coeffs__0"], 1.0)
        self.assertEqual(row["P1_s1_Q_FB_coeffs__1"], -2.5)
        self.assertEqual(row["P1_s1_Q_FB_coeffs__2"], 0.25)
        self.assertEqual(row["correct_angle"], 1)
        self.assertEqual(row["timtrack_projection_ellipse_tt_1234_available"], 0)
        self.assertEqual(row["timtrack_projection_ellipse_contour_fractions__0"], 0.25)
        self.assertEqual(row["timtrack_projection_ellipse_contour_fractions__1"], 0.50)
        self.assertEqual(row["timtrack_projection_ellipse_contour_fractions__2"], 0.75)

    def test_normalize_metadata_file_schema_rewrites_old_columns_in_place(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "task_2_metadata_calibration.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "filename_base",
                        "execution_timestamp",
                        "P1_s1_Q_FB_coeffs",
                        "correct_angle",
                        "timtrack_projection_ellipse_contour_fractions",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "filename_base": "mi0124074013648",
                        "execution_timestamp": "2026-04-17_14.50.27",
                        "P1_s1_Q_FB_coeffs": "[1.0, 2.0, 3.0]",
                        "correct_angle": "False",
                        "timtrack_projection_ellipse_contour_fractions": "0.25,0.50,0.75",
                    }
                )

            normalize_metadata_file_schema(csv_path)

            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertIsNotNone(reader.fieldnames)
                header = set(reader.fieldnames or [])
                self.assertNotIn("P1_s1_Q_FB_coeffs", header)
                self.assertNotIn("timtrack_projection_ellipse_contour_fractions", header)
                self.assertIn("P1_s1_Q_FB_coeffs__0", header)
                self.assertIn("P1_s1_Q_FB_coeffs__1", header)
                self.assertIn("P1_s1_Q_FB_coeffs__2", header)
                self.assertIn("timtrack_projection_ellipse_contour_fractions__2", header)
                row = next(reader)

            self.assertEqual(row["P1_s1_Q_FB_coeffs__0"], "1.0")
            self.assertEqual(row["P1_s1_Q_FB_coeffs__1"], "2.0")
            self.assertEqual(row["P1_s1_Q_FB_coeffs__2"], "3.0")
            self.assertEqual(row["correct_angle"], "0")
            self.assertEqual(row["timtrack_projection_ellipse_contour_fractions__0"], "0.25")
            self.assertEqual(row["timtrack_projection_ellipse_contour_fractions__1"], "0.5")
            self.assertEqual(row["timtrack_projection_ellipse_contour_fractions__2"], "0.75")


if __name__ == "__main__":
    unittest.main()
