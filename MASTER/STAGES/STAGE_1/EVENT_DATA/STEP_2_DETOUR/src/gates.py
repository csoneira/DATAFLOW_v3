from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class GateConfigError(ValueError):
    """Raised when the gate configuration is malformed."""


class GateEvaluationError(ValueError):
    """Raised when a gate expression cannot be evaluated on a dataframe."""


@dataclass(frozen=True)
class GateDefinition:
    name: str
    bit: int
    expression: str
    description: str = ""

    @property
    def bit_value(self) -> np.uint64:
        return np.uint64(1 << self.bit)


def load_gate_config(config_path: str | Path) -> list[GateDefinition]:
    raw_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    gates_section = raw_config.get("gates")
    if not isinstance(gates_section, dict) or not gates_section:
        raise GateConfigError("Gate config must define a non-empty top-level 'gates' mapping.")

    seen_bits: set[int] = set()
    gate_definitions: list[GateDefinition] = []

    for gate_name, gate_config in gates_section.items():
        if not isinstance(gate_config, dict):
            raise GateConfigError(f"Gate '{gate_name}' must be defined as a mapping.")

        if "bit" not in gate_config:
            raise GateConfigError(f"Gate '{gate_name}' is missing its bit index.")
        if "expression" not in gate_config or not str(gate_config["expression"]).strip():
            raise GateConfigError(f"Gate '{gate_name}' is missing its expression.")

        bit = int(gate_config["bit"])
        if bit < 0 or bit > 63:
            raise GateConfigError(f"Gate '{gate_name}' uses bit {bit}, but valid bits are 0 through 63.")
        if bit in seen_bits:
            raise GateConfigError(f"Gate '{gate_name}' reuses bit {bit}. Gate bit indices must be unique.")

        seen_bits.add(bit)
        gate_definitions.append(
            GateDefinition(
                name=str(gate_name),
                bit=bit,
                description=str(gate_config.get("description", "") or ""),
                expression=str(gate_config["expression"]).strip(),
            )
        )

    return gate_definitions


def apply_gates(df: pd.DataFrame, gates: list[GateDefinition]) -> np.ndarray:
    gate_mask = np.zeros(len(df), dtype=np.uint64)

    for gate in gates:
        event_mask = evaluate_gate_expression(df, gate)
        gate_mask[event_mask] |= gate.bit_value

    return gate_mask


def evaluate_gate_expression(df: pd.DataFrame, gate: GateDefinition) -> np.ndarray:
    try:
        result = df.eval(gate.expression, engine="python")
    except Exception as exc:
        raise GateEvaluationError(
            f"Failed to evaluate gate '{gate.name}' with expression '{gate.expression}': {exc}"
        ) from exc

    if not isinstance(result, pd.Series):
        raise GateEvaluationError(
            f"Gate '{gate.name}' did not return a pandas Series. "
            "Make sure the expression is a vectorized boolean expression."
        )

    if len(result) != len(df):
        raise GateEvaluationError(
            f"Gate '{gate.name}' returned {len(result)} rows, expected {len(df)}."
        )

    return result.fillna(False).to_numpy(dtype=bool)
