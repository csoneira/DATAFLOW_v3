from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import CANONICAL_EFF_COLUMNS
from step_2_build_lut import _build_lut_ascii_export


def test_lut_ascii_export_uses_quantized_bin_coordinates() -> None:
    lut = pd.DataFrame(
        {
            "emp_eff_1": [0.300000, 0.425000],
            "emp_eff_2": [0.325000, 0.450000],
            "emp_eff_3": [0.350000, 0.475000],
            "emp_eff_4": [0.375000, 0.500000],
            "eff_empirical_1": [0.301234, 0.423456],
            "eff_empirical_2": [0.326789, 0.451234],
            "eff_empirical_3": [0.349876, 0.476543],
            "eff_empirical_4": [0.374321, 0.498765],
            "scale_factor": [1.25, 0.85],
        }
    )

    exported = _build_lut_ascii_export(lut)

    assert list(exported.columns) == [*CANONICAL_EFF_COLUMNS, "scale_factor"]
    pd.testing.assert_frame_equal(
        exported.reset_index(drop=True),
        lut[[*CANONICAL_EFF_COLUMNS, "scale_factor"]].reset_index(drop=True),
    )
