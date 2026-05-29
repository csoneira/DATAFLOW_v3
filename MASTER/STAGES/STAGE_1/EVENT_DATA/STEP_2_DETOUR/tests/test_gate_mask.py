from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.gates import GateDefinition, GateEvaluationError, apply_gates


class TestGateMask(unittest.TestCase):
    def test_apply_gates_sets_expected_bits(self) -> None:
        df = pd.DataFrame(
            {
                "s": [0.0030, 0.0005, 0.0035],
                "raw_tt": [1000, 1234, 1234],
                "Q": [15, 5, 12],
                "cs0": [0, 1, 1],
                "cs1": [0, 1, 1],
                "cs2": [0, 1, 1],
                "cs3": [0, 1, 1],
            }
        )
        gates = [
            GateDefinition(
                name="charge_or_trigger_gate",
                bit=0,
                description="Large s or specific raw_tt, with Q above threshold",
                expression="((s > 0.002) | (raw_tt == 1234)) & (Q > 10)",
            ),
            GateDefinition(
                name="clean_four_plane_event",
                bit=1,
                description="One cluster in each detector plane",
                expression="(cs0 == 1) & (cs1 == 1) & (cs2 == 1) & (cs3 == 1)",
            ),
        ]

        gate_mask = apply_gates(df, gates)
        expected = np.array([1, 2, 3], dtype=np.uint64)

        np.testing.assert_array_equal(gate_mask, expected)

    def test_apply_gates_reports_missing_columns(self) -> None:
        df = pd.DataFrame({"value": [1, 2, 3]})
        gates = [GateDefinition(name="broken", bit=0, expression="(missing_col > 0)")]

        with self.assertRaises(GateEvaluationError):
            apply_gates(df, gates)
