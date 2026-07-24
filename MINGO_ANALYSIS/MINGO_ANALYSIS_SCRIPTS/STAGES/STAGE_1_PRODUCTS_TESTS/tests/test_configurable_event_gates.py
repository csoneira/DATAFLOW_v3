from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from test_3_configurable_event_gates import (  # noqa: E402
    assign_gates,
    parse_gates,
    write_enabled_gate_comparison,
)


CONFIG = SCRIPT_DIR / "config_test_3_event_gates.yaml"


def enabled_gates():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return parse_gates(config["gates"])


def topology_frame(patterns: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        f"p{plane}_strip_topology": [row[plane - 1] for row in patterns]
        for plane in range(1, 5)
    })


def test_config_enables_only_requested_topology_gates() -> None:
    gates = enabled_gates()

    assert [gate.short_label for gate in gates] == ["0100_3of4", "0010_3of4", "MID_3of4"]

    frame = topology_frame([
        ("0100", "0100", "0100", "0100"),
        ("0100", "0000", "0100", "0100"),
        ("0100", "0000", "0000", "0100"),
        ("0010", "0010", "0010", "0010"),
        ("0010", "0010", "0000", "0010"),
        ("0100", "0110", "0000", "0010"),
        ("0100", "0110", "0010", "0001"),
    ])
    masks = assign_gates(frame, gates)

    assert masks[gates[0].code].tolist() == [True, True, False, False, False, False, False]
    assert masks[gates[1].code].tolist() == [False, False, False, True, True, False, False]
    assert masks[gates[2].code].tolist() == [True, True, False, True, True, True, False]


def test_enabled_gate_comparison_writes_both_requested_figures(tmp_path: Path) -> None:
    gates = enabled_gates()
    patterns = [
        ("0100", "0100", "0100", "0100"),
        ("0010", "0010", "0010", "0010"),
        ("0100", "0110", "0010", "0100"),
    ] * 5
    frame = topology_frame(patterns)
    frame["datetime"] = pd.date_range("2026-01-01", periods=len(frame), freq="s")
    frame["tt_task3_list"] = [1234, 234, 134, 124, 123] * 3
    masks = assign_gates(frame, gates)

    efficiency_path, metrics_path, csv_path, plot_count = write_enabled_gate_comparison(
        frame,
        gates,
        masks,
        {
            "time_column": "datetime",
            "topology_column": "tt_task3_list",
            "window": pd.Timedelta("10min"),
        },
        tmp_path,
        "Synthetic gate comparison",
    )

    assert plot_count == 2
    assert efficiency_path.is_file()
    assert metrics_path.is_file()
    assert csv_path.is_file()
    summary = pd.read_csv(csv_path, dtype={"gate_code": str})
    assert set(summary["gate_code"]) == {gate.code for gate in gates}
    assert {
        "plane_1_efficiency",
        "plane_4_efficiency",
        "efficiency_product",
        "total_gate_rate_hz",
        "topology_1234_rate_hz",
        "corrected_1234_rate_hz",
    }.issubset(summary.columns)
    finite = summary.loc[summary["efficiency_product"].gt(0)].iloc[0]
    assert np.isclose(
        finite["corrected_1234_rate_hz"],
        finite["topology_1234_rate_hz"] / finite["efficiency_product"],
    )
