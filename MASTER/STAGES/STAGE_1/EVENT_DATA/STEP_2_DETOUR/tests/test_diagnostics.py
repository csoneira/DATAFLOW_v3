from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.diagnostics import build_diagnostic_table
from src.gates import GateDefinition
from src.plotting import (
    RATE_AND_EFFICIENCY_FIGURE,
    TOPOLOGY_FIGURE,
    apply_moving_average,
    apply_rate_mode,
    plot_diagnostic_table,
)
from src.rates import gate_efficiency_column


class TestDiagnostics(unittest.TestCase):
    def test_build_diagnostic_table_computes_empirical_efficiencies(self) -> None:
        gates = [GateDefinition(name="gate_a", bit=0, expression="(value > 0)")]
        rate_table = pd.DataFrame(
            {
                "datetime_minute": pd.to_datetime(["2026-05-01 00:00:00"]),
                "live_time_seconds": [60.0],
                "total_events_count": [20],
                "total_events_hz": [20.0 / 60.0],
                "gate_a_count": [10],
                "gate_a_hz": [10.0 / 60.0],
                "gate_a_fraction": [0.5],
                "gate_a_percent": [50.0],
                "gate_a_post_tt_1234_count": [4],
                "gate_a_post_tt_1234_hz": [4.0 / 60.0],
                "gate_a_post_tt_123_count": [1],
                "gate_a_post_tt_123_hz": [1.0 / 60.0],
                "gate_a_post_tt_124_count": [2],
                "gate_a_post_tt_124_hz": [2.0 / 60.0],
                "gate_a_post_tt_134_count": [3],
                "gate_a_post_tt_134_hz": [3.0 / 60.0],
                "gate_a_post_tt_234_count": [5],
                "gate_a_post_tt_234_hz": [5.0 / 60.0],
            }
        )

        diagnostic = build_diagnostic_table(rate_table, gates)

        self.assertAlmostEqual(diagnostic.at[0, gate_efficiency_column("gate_a", 1)], 4.0 / 9.0)
        self.assertAlmostEqual(diagnostic.at[0, gate_efficiency_column("gate_a", 2)], 4.0 / 7.0)
        self.assertAlmostEqual(diagnostic.at[0, gate_efficiency_column("gate_a", 3)], 4.0 / 6.0)
        self.assertAlmostEqual(diagnostic.at[0, gate_efficiency_column("gate_a", 4)], 4.0 / 5.0)

    def test_apply_moving_average_uses_centered_kernel(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        smoothed = apply_moving_average(series, 3)
        expected = pd.Series([1.5, 2.0, 3.0, 3.5])
        pd.testing.assert_series_equal(smoothed, expected)

    def test_apply_rate_mode_zscores_only_changes_hz_columns(self) -> None:
        diagnostic_table = pd.DataFrame(
            {
                "gate_a_hz": [1.0, 2.0, 3.0],
                "total_events_hz": [2.0, 4.0, 6.0],
                "gate_a_eff_1": [0.8, 0.7, 0.9],
                "gate_a_percent": [25.0, 20.0, 24.0],
            }
        )

        transformed = apply_rate_mode(diagnostic_table, "zscores")

        expected_zscores = pd.Series([-1.224744871391589, 0.0, 1.224744871391589], dtype="float64")
        pd.testing.assert_series_equal(transformed["gate_a_hz"], expected_zscores, check_names=False)
        pd.testing.assert_series_equal(transformed["total_events_hz"], expected_zscores, check_names=False)
        pd.testing.assert_series_equal(transformed["gate_a_eff_1"], diagnostic_table["gate_a_eff_1"], check_names=False)
        pd.testing.assert_series_equal(transformed["gate_a_percent"], diagnostic_table["gate_a_percent"], check_names=False)

    def test_plot_diagnostic_table_creates_combined_pngs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "plots"
            diagnostic_table = pd.DataFrame(
                {
                    "datetime_minute": pd.to_datetime(
                        [
                            "2026-05-01 00:00:00",
                            "2026-05-01 00:01:00",
                            "2026-05-01 00:02:00",
                        ]
                    ),
                    "gate_a_hz": [0.5, 0.4, 0.6],
                    "gate_b_hz": [0.3, 0.5, 0.4],
                    "gate_a_eff_1": [0.8, 0.7, 0.9],
                    "gate_b_eff_1": [0.6, 0.65, 0.7],
                    "gate_a_eff_2": [0.7, 0.75, 0.8],
                    "gate_b_eff_2": [0.5, 0.55, 0.6],
                    "gate_a_eff_3": [0.85, 0.8, 0.82],
                    "gate_b_eff_3": [0.66, 0.62, 0.64],
                    "gate_a_eff_4": [0.9, 0.88, 0.91],
                    "gate_b_eff_4": [0.71, 0.72, 0.73],
                    "gate_a_percent": [25.0, 20.0, 24.0],
                    "gate_b_percent": [15.0, 17.0, 16.0],
                    "gate_a_post_tt_1234_hz": [0.2, 0.18, 0.22],
                    "gate_b_post_tt_1234_hz": [0.1, 0.11, 0.12],
                    "gate_a_post_tt_123_hz": [0.05, 0.04, 0.03],
                    "gate_b_post_tt_123_hz": [0.03, 0.02, 0.01],
                    "gate_a_post_tt_124_hz": [0.02, 0.03, 0.04],
                    "gate_b_post_tt_124_hz": [0.01, 0.01, 0.02],
                    "gate_a_post_tt_134_hz": [0.03, 0.02, 0.03],
                    "gate_b_post_tt_134_hz": [0.02, 0.03, 0.02],
                    "gate_a_post_tt_234_hz": [0.04, 0.05, 0.04],
                    "gate_b_post_tt_234_hz": [0.03, 0.04, 0.03],
                }
            )

            output_paths = plot_diagnostic_table(
                diagnostic_table=diagnostic_table,
                output_path=output_dir,
                gate_names=["gate_a", "gate_b"],
                moving_average_minutes=3,
                rate_mode="zscores",
                title_context="Stations: MINGO00 | 2026-05-01",
            )

            self.assertEqual(len(output_paths), 2)
            self.assertTrue((output_dir / RATE_AND_EFFICIENCY_FIGURE).exists())
            self.assertTrue((output_dir / TOPOLOGY_FIGURE).exists())
