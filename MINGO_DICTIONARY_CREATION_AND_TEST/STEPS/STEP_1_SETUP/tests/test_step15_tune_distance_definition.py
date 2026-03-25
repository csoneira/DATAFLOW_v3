#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


STEP15_DIR = (
    Path(__file__).resolve().parents[1]
    / "STEP_1_5_TUNE_DISTANCE_DEFINITION"
)
if str(STEP15_DIR) not in sys.path:
    sys.path.insert(0, str(STEP15_DIR))

from tune_distance_definition import _load_parameter_space_columns


def test_parameter_columns_resolve_effsim_alias_to_effp_when_needed(tmp_path: Path) -> None:
    payload = {
        "parameter_space_columns_downstream_preferred": [
            "flux_cm2_min",
            "eff_sim_1",
            "eff_sim_2",
            "eff_sim_3",
            "eff_sim_4",
        ],
        "parameter_space_column_aliases": {
            "eff_p1": "eff_sim_1",
            "eff_sim_1": "eff_sim_1",
            "eff_p2": "eff_sim_2",
            "eff_sim_2": "eff_sim_2",
            "eff_p3": "eff_sim_3",
            "eff_sim_3": "eff_sim_3",
            "eff_p4": "eff_sim_4",
            "eff_sim_4": "eff_sim_4",
        },
    }
    p = tmp_path / "parameter_space_columns.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    resolved = _load_parameter_space_columns(
        p,
        available_columns=["flux_cm2_min", "eff_p1", "eff_p2", "eff_p3", "eff_p4"],
    )
    assert resolved == [
        "flux_cm2_min",
        "eff_p1",
        "eff_p2",
        "eff_p3",
        "eff_p4",
    ]
