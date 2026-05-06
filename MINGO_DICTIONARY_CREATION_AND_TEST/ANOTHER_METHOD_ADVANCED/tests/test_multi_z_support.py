from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import CANONICAL_EFF_COLUMNS, CANONICAL_Z_COLUMNS, apply_lut_fallback_matches
from multi_z_support import (
    apply_rate_to_flux_lines,
    build_rate_to_flux_lines,
    resolve_active_z_vectors_from_config,
)


def test_apply_lut_fallback_matches_respects_z_groups() -> None:
    dataframe = pd.DataFrame(
        {
            "z_pos_1": [0.0, 0.0],
            "z_pos_2": [100.0, 145.0],
            "z_pos_3": [200.0, 290.0],
            "z_pos_4": [400.0, 435.0],
            "query_emp_eff_1": [0.405, 0.405],
            "query_emp_eff_2": [0.405, 0.405],
            "query_emp_eff_3": [0.405, 0.405],
            "query_emp_eff_4": [0.405, 0.405],
            "emp_eff_1": [0.405, 0.405],
            "emp_eff_2": [0.405, 0.405],
            "emp_eff_3": [0.405, 0.405],
            "emp_eff_4": [0.405, 0.405],
            "lut_scale_factor": [pd.NA, pd.NA],
            "lut_match_method": [pd.NA, pd.NA],
        }
    )
    lut = pd.DataFrame(
        {
            "z_pos_1": [0.0, 0.0],
            "z_pos_2": [100.0, 145.0],
            "z_pos_3": [200.0, 290.0],
            "z_pos_4": [400.0, 435.0],
            "emp_eff_1": [0.40, 0.41],
            "emp_eff_2": [0.40, 0.41],
            "emp_eff_3": [0.40, 0.41],
            "emp_eff_4": [0.40, 0.41],
            "scale_factor": [1.25, 2.50],
        }
    )

    filled = apply_lut_fallback_matches(
        dataframe,
        lut,
        query_columns=[f"query_{column}" for column in CANONICAL_EFF_COLUMNS],
        raw_columns=CANONICAL_EFF_COLUMNS,
        match_mode="nearest",
        group_columns=CANONICAL_Z_COLUMNS,
    )

    assert filled.loc[0, "lut_scale_factor"] == pytest.approx(1.25)
    assert filled.loc[1, "lut_scale_factor"] == pytest.approx(2.50)
    assert filled.loc[0, "lut_match_method"] == "nearest"
    assert filled.loc[1, "lut_match_method"] == "nearest"


def test_apply_rate_to_flux_lines_uses_matching_z_linear_fit() -> None:
    reference_table = pd.DataFrame(
        {
            "z_pos_1": [0.0, 0.0, 0.0, 0.0],
            "z_pos_2": [100.0, 100.0, 145.0, 145.0],
            "z_pos_3": [200.0, 200.0, 290.0, 290.0],
            "z_pos_4": [400.0, 400.0, 435.0, 435.0],
            "flux_bin_index": [0, 1, 0, 1],
            "flux_bin_lo": [1.0, 2.0, 10.0, 20.0],
            "flux_bin_hi": [2.0, 3.0, 20.0, 30.0],
            "flux_bin_center": [1.5, 2.5, 15.0, 25.0],
            "reference_rate_median": [10.0, 20.0, 100.0, 200.0],
        }
    )
    row_z = pd.DataFrame(
        {
            "z_pos_1": [0.0, 0.0],
            "z_pos_2": [100.0, 145.0],
            "z_pos_3": [200.0, 290.0],
            "z_pos_4": [400.0, 435.0],
        }
    )
    corrected_rate = pd.Series([15.0, 150.0], dtype=float)
    line_table = build_rate_to_flux_lines(reference_table)

    flux, method = apply_rate_to_flux_lines(
        corrected_rate,
        row_z_frame=row_z,
        line_table=line_table,
    )

    assert flux.iloc[0] == pytest.approx(2.0)
    assert flux.iloc[1] == pytest.approx(20.0)
    assert method.iloc[0] == "linear_fit"
    assert method.iloc[1] == "linear_fit"


def test_step5_window_selects_latest_active_schedule_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = pd.DataFrame(
        {
            "start_utc": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"],
                utc=True,
            ),
            "end_utc": pd.to_datetime(
                ["2026-01-31T23:59:59Z", "2026-02-28T23:59:59Z", "2026-03-31T23:59:59Z"],
                utc=True,
            ),
            "z_tuple": [
                (0.0, 65.0, 130.0, 195.0),
                (0.0, 100.0, 200.0, 400.0),
                (0.0, 106.0, 270.0, 460.0),
            ],
        }
    )

    monkeypatch.setattr(
        "multi_z_support.load_online_schedule",
        lambda station_id: (schedule, Path("/tmp/input_file_mingo02.csv")),
    )

    config = {
        "step1": {"z_selection_mode": "step5_window"},
        "step5": {
            "station": "MINGO02",
            "date_from": "2026-01-15",
            "date_to": "2026-03-15",
        },
    }

    selected, meta = resolve_active_z_vectors_from_config(config)

    assert selected == [(0.0, 106.0, 270.0, 460.0)]
    assert meta["window_candidate_geometry_count"] == 3
    assert meta["window_ignored_geometry_count"] == 2
    assert "latest active schedule row" in str(meta["window_selected_reason"])
    assert meta["window_selected_schedule_start_utc"].startswith("2026-03-01")


def test_step5_window_falls_back_to_configured_vector_when_window_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_schedule = pd.DataFrame(columns=["start_utc", "end_utc", "z_tuple"])
    monkeypatch.setattr(
        "multi_z_support.load_online_schedule",
        lambda station_id: (empty_schedule, Path("/tmp/input_file_mingo01.csv")),
    )

    config = {
        "step1": {
            "z_selection_mode": "step5_window",
            "z_position_vector": [0.0, 100.0, 200.0, 400.0],
        },
        "step5": {
            "station": "MINGO01",
            "date_from": "2026-01-01",
            "date_to": "2026-01-02",
        },
    }

    selected, meta = resolve_active_z_vectors_from_config(config)

    assert selected == [(0.0, 100.0, 200.0, 400.0)]
    assert meta["window_candidate_geometry_count"] == 0
    assert meta["window_ignored_geometry_count"] == 0
    assert "fallback to configured_vector" in str(meta["window_selected_reason"])
