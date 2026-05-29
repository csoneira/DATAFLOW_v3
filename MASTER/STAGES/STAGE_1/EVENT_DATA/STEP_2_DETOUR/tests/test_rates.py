from __future__ import annotations

from datetime import datetime
import unittest

import numpy as np
import pandas as pd

from src.gates import GateDefinition
from src.rates import (
    LIVE_TIME_SECONDS_COLUMN,
    TOTAL_EVENTS_COUNT_COLUMN,
    TOTAL_EVENTS_HZ_COLUMN,
    accumulate_rate_tables,
    build_chunk_rates,
    finalize_rate_table,
    gate_count_column,
    gate_hz_column,
    gate_percent_column,
    gate_topology_count_column,
    gate_topology_hz_column,
)


class TestRates(unittest.TestCase):
    def test_full_minute_uses_sixty_second_live_time_for_interior_bin(self) -> None:
        gates = [GateDefinition(name="gate_a", bit=0, expression="(value > 0)")]
        df = pd.DataFrame(
            {
                "event_time": pd.to_datetime(
                    [
                        "2026-05-01 23:59:50",
                        "2026-05-02 00:00:05",
                        "2026-05-02 00:00:55",
                        "2026-05-02 00:01:10",
                    ]
                ),
                "post_tt": [1234, 1234, 134, 123],
            }
        )
        gate_mask = np.array([0, 1, 0, 0], dtype=np.uint64)

        chunk = build_chunk_rates(
            df=df,
            gate_mask=gate_mask,
            gates=gates,
            time_column="event_time",
            bin_size="1min",
            selection_start=datetime(2026, 5, 1, 23, 59, 0),
            selection_end=datetime(2026, 5, 2, 0, 2, 0),
        )
        final = finalize_rate_table(chunk, gates)

        minute = pd.Timestamp("2026-05-02 00:00:00")
        self.assertEqual(final.at[minute, LIVE_TIME_SECONDS_COLUMN], 60.0)
        self.assertEqual(final.at[minute, TOTAL_EVENTS_COUNT_COLUMN], 2)
        self.assertAlmostEqual(final.at[minute, TOTAL_EVENTS_HZ_COLUMN], 2.0 / 60.0)

    def test_partial_minute_uses_observed_coverage_and_hz(self) -> None:
        gates = [GateDefinition(name="gate_a", bit=0, expression="(value > 0)")]
        df = pd.DataFrame(
            {
                "event_time": pd.to_datetime(
                    [
                        "2026-05-01 00:00:05",
                        "2026-05-01 00:00:30",
                    ]
                ),
                "post_tt": [1234, 134],
            }
        )
        gate_mask = np.array([1, 0], dtype=np.uint64)

        chunk = build_chunk_rates(
            df=df,
            gate_mask=gate_mask,
            gates=gates,
            time_column="event_time",
            bin_size="1min",
            selection_start=datetime(2026, 5, 1, 0, 0, 0),
            selection_end=datetime(2026, 5, 1, 0, 1, 0),
        )
        final = finalize_rate_table(chunk, gates)

        minute = pd.Timestamp("2026-05-01 00:00:00")
        self.assertEqual(final.at[minute, LIVE_TIME_SECONDS_COLUMN], 25.0)
        self.assertEqual(final.at[minute, TOTAL_EVENTS_COUNT_COLUMN], 2)
        self.assertAlmostEqual(final.at[minute, TOTAL_EVENTS_HZ_COLUMN], 2.0 / 25.0)
        self.assertAlmostEqual(final.at[minute, gate_hz_column("gate_a")], 1.0 / 25.0)
        self.assertEqual(final.at[minute, gate_topology_count_column("gate_a", 1234)], 1)
        self.assertEqual(final.at[minute, gate_topology_count_column("gate_a", 134)], 0)
        self.assertAlmostEqual(final.at[minute, gate_topology_hz_column("gate_a", 1234)], 1.0 / 25.0)

    def test_gate_percentages_are_computed_from_total_counts(self) -> None:
        gates = [GateDefinition(name="gate_a", bit=0, expression="(value > 0)")]
        df = pd.DataFrame(
            {
                "event_time": pd.to_datetime(
                    [
                        "2026-05-01 00:00:00",
                        "2026-05-01 00:00:10",
                        "2026-05-01 00:00:20",
                        "2026-05-01 00:00:30",
                    ]
                ),
                "post_tt": [1234, 134, 124, 123],
            }
        )
        gate_mask = np.array([1, 0, 0, 0], dtype=np.uint64)

        chunk = build_chunk_rates(
            df=df,
            gate_mask=gate_mask,
            gates=gates,
            time_column="event_time",
            bin_size="1min",
            selection_start=datetime(2026, 5, 1, 0, 0, 0),
            selection_end=datetime(2026, 5, 1, 0, 1, 0),
        )
        final = finalize_rate_table(chunk, gates)

        minute = pd.Timestamp("2026-05-01 00:00:00")
        self.assertEqual(final.at[minute, gate_count_column("gate_a")], 1)
        self.assertAlmostEqual(final.at[minute, gate_percent_column("gate_a")], 25.0)

    def test_accumulate_rate_tables_sums_counts_before_final_hz(self) -> None:
        gates = [GateDefinition(name="gate_a", bit=0, expression="(value > 0)")]

        df_one = pd.DataFrame(
            {
                "event_time": pd.to_datetime(["2026-05-01 00:00:05", "2026-05-01 00:00:10"]),
                "post_tt": [1234, 134],
            }
        )
        df_two = pd.DataFrame(
            {
                "event_time": pd.to_datetime(["2026-05-01 00:00:20", "2026-05-01 00:00:30"]),
                "post_tt": [1234, 1234],
            }
        )
        gate_mask_one = np.array([1, 0], dtype=np.uint64)
        gate_mask_two = np.array([1, 1], dtype=np.uint64)

        chunk_one = build_chunk_rates(
            df=df_one,
            gate_mask=gate_mask_one,
            gates=gates,
            time_column="event_time",
            bin_size="1min",
            selection_start=datetime(2026, 5, 1, 0, 0, 0),
            selection_end=datetime(2026, 5, 1, 0, 1, 0),
        )
        chunk_two = build_chunk_rates(
            df=df_two,
            gate_mask=gate_mask_two,
            gates=gates,
            time_column="event_time",
            bin_size="1min",
            selection_start=datetime(2026, 5, 1, 0, 0, 0),
            selection_end=datetime(2026, 5, 1, 0, 1, 0),
        )

        merged = accumulate_rate_tables(chunk_one, chunk_two)
        final = finalize_rate_table(merged, gates)

        minute = pd.Timestamp("2026-05-01 00:00:00")
        self.assertEqual(final.at[minute, TOTAL_EVENTS_COUNT_COLUMN], 4)
        self.assertEqual(final.at[minute, gate_count_column("gate_a")], 3)
        self.assertEqual(final.at[minute, gate_topology_count_column("gate_a", 1234)], 3)
        self.assertEqual(final.at[minute, LIVE_TIME_SECONDS_COLUMN], 25.0)
        self.assertAlmostEqual(final.at[minute, TOTAL_EVENTS_HZ_COLUMN], 4.0 / 25.0)
