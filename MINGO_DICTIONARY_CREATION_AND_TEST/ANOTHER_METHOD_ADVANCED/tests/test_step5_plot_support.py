from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import CANONICAL_EFF_COLUMNS
from step_5_apply_lut_to_real_data import _select_lut_empirical_support_for_plot


def test_plot_support_prefers_lut_empirical_columns_when_available() -> None:
    lut_support = pd.DataFrame(
        {
            "emp_eff_1": [0.300000, 0.325000],
            "emp_eff_2": [0.350000, 0.375000],
            "emp_eff_3": [0.400000, 0.425000],
            "emp_eff_4": [0.450000, 0.475000],
            "eff_empirical_1": [0.311111, 0.322222],
            "eff_empirical_2": [0.355555, 0.366666],
            "eff_empirical_3": [0.411111, 0.422222],
            "eff_empirical_4": [0.455555, 0.466666],
        }
    )

    selected = _select_lut_empirical_support_for_plot(lut_support)

    assert list(selected.columns) == CANONICAL_EFF_COLUMNS
    pd.testing.assert_frame_equal(
        selected.reset_index(drop=True),
        lut_support[[f"eff_empirical_{idx}" for idx in range(1, 5)]]
        .rename(columns={f"eff_empirical_{idx}": column for idx, column in enumerate(CANONICAL_EFF_COLUMNS, start=1)})
        .reset_index(drop=True),
    )
