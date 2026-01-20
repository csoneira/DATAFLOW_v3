#!/usr/bin/env python3
"""Print a compact, single-event causal chain report from STEP 10 outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import iter_input_frames, latest_sim_run, random_sim_run


def is_finite(val: object) -> bool:
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def fmt(val: object, digits: int = 4) -> str:
    if not is_finite(val):
        return "NA"
    return f"{float(val):.{digits}f}"


def rows_match(base: pd.Series, other: pd.Series, tol: float) -> bool:
    keys = ("X_gen", "Y_gen", "Z_gen", "Theta_gen", "Phi_gen", "T0_ns", "T_thick_s")
    matched = 0
    compared = 0
    for key in keys:
        if key not in base or key not in other:
            continue
        a = base.get(key)
        b = other.get(key)
        if not is_finite(a) and not is_finite(b):
            compared += 1
            matched += 1
            continue
        if not is_finite(a) or not is_finite(b):
            compared += 1
            continue
        compared += 1
        if abs(float(a) - float(b)) <= tol:
            matched += 1
    if compared == 0:
        return False
    return matched == compared


def find_geom_file(run_dir: Path, geom_id: int, stem_suffix: str) -> Path:
    candidates = [
        run_dir / f"geom_{geom_id}_{stem_suffix}.chunks.json",
        run_dir / f"geom_{geom_id}_{stem_suffix}.pkl",
        run_dir / f"geom_{geom_id}_{stem_suffix}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing geom_{geom_id}_{stem_suffix} in {run_dir}")


def load_metadata(path: Path) -> dict:
    if path.name.endswith(".chunks.json"):
        return json.loads(path.read_text()).get("metadata", {})
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def resolve_upstream_chain(step10_path: Path) -> dict:
    chain = {"step10": step10_path}
    meta10 = load_metadata(step10_path)
    step9 = meta10.get("source_dataset")
    if step9:
        chain["step9"] = Path(step9)
        meta9 = load_metadata(chain["step9"])
        step8 = meta9.get("source_dataset")
        if step8:
            chain["step8"] = Path(step8)
            meta8 = load_metadata(chain["step8"])
            step7 = meta8.get("source_dataset")
            if step7:
                chain["step7"] = Path(step7)
                meta7 = load_metadata(chain["step7"])
                step6 = meta7.get("source_dataset")
                if step6:
                    chain["step6"] = Path(step6)
                    meta6 = load_metadata(chain["step6"])
                    step5 = meta6.get("source_dataset")
                    if step5:
                        chain["step5"] = Path(step5)
                        meta5 = load_metadata(chain["step5"])
                        step4 = meta5.get("source_dataset")
                        if step4:
                            chain["step4"] = Path(step4)
                            meta4 = load_metadata(chain["step4"])
                            step3 = meta4.get("source_dataset")
                            if step3:
                                chain["step3"] = Path(step3)
                                meta3 = load_metadata(chain["step3"])
                                step2 = meta3.get("source_dataset")
                                if step2:
                                    chain["step2"] = Path(step2)
    return chain


def locate_row(
    path: Path,
    index_value: int | None,
    row_number: int | None,
    allow_row_number: bool,
) -> tuple[pd.Series | None, int | None, int | None]:
    offset = 0
    event_id_seen = False
    frames, _, _ = iter_input_frames(path, None)
    for df in frames:
        if df.empty:
            offset += len(df)
            continue
        if index_value is not None and "event_id" in df.columns:
            event_id_seen = True
            mask = pd.to_numeric(df["event_id"], errors="coerce") == index_value
            if mask.any():
                row = df.loc[mask].iloc[0]
                pos = int(np.flatnonzero(mask.to_numpy())[0])
                index_label = row.name
                return (
                    row,
                    int(index_label) if np.isscalar(index_label) else None,
                    offset + pos,
                )
        if index_value is not None and index_value in df.index:
            row = df.loc[index_value]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            loc = df.index.get_loc(index_value)
            if isinstance(loc, slice):
                loc = loc.start
            elif isinstance(loc, np.ndarray):
                loc = int(loc[0])
            return row, int(index_value), offset + int(loc)
        if allow_row_number and row_number is not None and offset <= row_number < offset + len(df):
            row = df.iloc[row_number - offset]
            index_label = df.index[row_number - offset]
            return row, int(index_label) if np.isscalar(index_label) else None, row_number
        offset += len(df)
    if event_id_seen:
        return None, None, None
    return None, None, None


def find_closest_event_id(path: Path, target: int) -> int | None:
    best_id: int | None = None
    best_delta: float | None = None
    for df in iter_input_frames(path, None)[0]:
        if df.empty or "event_id" not in df.columns:
            continue
        ids = pd.to_numeric(df["event_id"], errors="coerce").dropna().astype(int)
        if ids.empty:
            continue
        deltas = (ids - target).abs()
        idx = int(deltas.idxmin())
        candidate = int(ids.loc[idx])
        delta = abs(candidate - target)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_id = candidate
    return best_id


def plane_activity(row: pd.Series) -> str:
    planes = []
    for plane_idx in range(1, 5):
        active = False
        for strip_idx in range(1, 5):
            for prefix in ("Q_front", "Q_back"):
                val = row.get(f"{prefix}_{plane_idx}_s{strip_idx}")
                if val is not None and is_finite(val) and float(val) > 0:
                    active = True
        if active:
            planes.append(str(plane_idx))
    return "".join(planes)


def list_active_strips(row: pd.Series, prefix: str) -> list[tuple[int, int]]:
    active = []
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            val = row.get(f"{prefix}_{plane_idx}_s{strip_idx}")
            if val is not None and is_finite(val) and float(val) > 0:
                active.append((plane_idx, strip_idx))
    return active


def describe_generation(row: pd.Series) -> list[str]:
    return [
        "Muon generation",
        f"X_gen={fmt(row.get('X_gen'))} mm, Y_gen={fmt(row.get('Y_gen'))} mm, Z_gen={fmt(row.get('Z_gen'))} mm",
        f"Theta_gen={fmt(row.get('Theta_gen'), 5)} rad, Phi_gen={fmt(row.get('Phi_gen'), 5)} rad",
        f"T0_ns={fmt(row.get('T0_ns'))} ns, T_thick_s={fmt(row.get('T_thick_s'), 6)} s",
    ]


def describe_crossings(row: pd.Series) -> list[str]:
    lines = ["Plane incidence (STEP 2)"]
    for plane_idx in range(1, 5):
        x = row.get(f"X_gen_{plane_idx}")
        y = row.get(f"Y_gen_{plane_idx}")
        z = row.get(f"Z_gen_{plane_idx}")
        t = row.get(f"T_sum_{plane_idx}_ns")
        if is_finite(x) or is_finite(y) or is_finite(t):
            lines.append(
                f"P{plane_idx}: X={fmt(x)} mm, Y={fmt(y)} mm, Z={fmt(z)} mm, T_sum={fmt(t)} ns"
            )
    lines.append(f"tt_crossing={row.get('tt_crossing', 'NA')}")
    return lines


def describe_avalanche(row: pd.Series) -> list[str]:
    lines = ["Avalanche (STEP 3)"]
    for plane_idx in range(1, 5):
        exists = row.get(f"avalanche_exists_{plane_idx}")
        if bool(exists):
            ions = row.get(f"avalanche_ion_{plane_idx}")
            size = row.get(f"avalanche_size_electrons_{plane_idx}")
            x = row.get(f"avalanche_x_{plane_idx}")
            y = row.get(f"avalanche_y_{plane_idx}")
            lines.append(
                f"P{plane_idx}: ions={fmt(ions, 0)}, size_e={fmt(size, 1)}, x={fmt(x)} mm, y={fmt(y)} mm"
            )
    lines.append(f"tt_avalanche={row.get('tt_avalanche', 'NA')}")
    return lines


def describe_induced_strips(row: pd.Series) -> list[str]:
    lines = ["Induced strips + charges (STEP 4)"]
    found = False
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            q = row.get(f"Y_mea_{plane_idx}_s{strip_idx}")
            if q is None or not is_finite(q) or float(q) <= 0:
                continue
            x = row.get(f"X_mea_{plane_idx}_s{strip_idx}")
            t = row.get(f"T_sum_meas_{plane_idx}_s{strip_idx}")
            lines.append(
                f"P{plane_idx} S{strip_idx}: Q={fmt(q, 2)}, X={fmt(x)} mm, T_sum_meas={fmt(t)} ns"
            )
            found = True
    if not found:
        lines.append("No active strips in STEP 4")
    lines.append(f"tt_hit={row.get('tt_hit', 'NA')}")
    return lines


def describe_diff(row: pd.Series) -> list[str]:
    lines = ["T_diff / q_diff (STEP 5)"]
    found = False
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            q = row.get(f"Y_mea_{plane_idx}_s{strip_idx}")
            if q is None or not is_finite(q) or float(q) <= 0:
                continue
            tdiff = row.get(f"T_diff_{plane_idx}_s{strip_idx}")
            qdiff = row.get(f"q_diff_{plane_idx}_s{strip_idx}")
            lines.append(
                f"P{plane_idx} S{strip_idx}: T_diff={fmt(tdiff)} ns, q_diff={fmt(qdiff, 2)}"
            )
            found = True
    if not found:
        lines.append("No active strips in STEP 5")
    return lines


def describe_frontback(row: pd.Series, label: str, source: str) -> list[str]:
    lines = [f"{label} (source={source})"]
    active = list_active_strips(row, "Q_front")
    if not active:
        active = list_active_strips(row, "Q_back")
    if not active:
        lines.append("No active strips")
        return lines
    for plane_idx, strip_idx in active:
        tf = row.get(f"T_front_{plane_idx}_s{strip_idx}")
        tb = row.get(f"T_back_{plane_idx}_s{strip_idx}")
        qf = row.get(f"Q_front_{plane_idx}_s{strip_idx}")
        qb = row.get(f"Q_back_{plane_idx}_s{strip_idx}")
        lines.append(
            f"P{plane_idx} S{strip_idx}: T_front={fmt(tf)} ns, T_back={fmt(tb)} ns,"
            f" Q_front={fmt(qf, 2)}, Q_back={fmt(qb, 2)}"
        )
    return lines


def describe_offsets(step6: pd.Series | None, step7: pd.Series | None, cfg7: dict | None) -> list[str]:
    lines = ["Connector offsets (STEP 7 deltas)"]
    if step6 is not None and step7 is not None:
        active = list_active_strips(step6, "Q_front")
        if not active:
            active = list_active_strips(step6, "Q_back")
        if not active:
            lines.append("No active strips in STEP 6")
            return lines
        for plane_idx, strip_idx in active:
            tf6 = step6.get(f"T_front_{plane_idx}_s{strip_idx}")
            tb6 = step6.get(f"T_back_{plane_idx}_s{strip_idx}")
            tf7 = step7.get(f"T_front_{plane_idx}_s{strip_idx}")
            tb7 = step7.get(f"T_back_{plane_idx}_s{strip_idx}")
            d_tf = float(tf7) - float(tf6) if is_finite(tf6) and is_finite(tf7) else float("nan")
            d_tb = float(tb7) - float(tb6) if is_finite(tb6) and is_finite(tb7) else float("nan")
            lines.append(
                f"P{plane_idx} S{strip_idx}: dT_front={fmt(d_tf)} ns, dT_back={fmt(d_tb)} ns"
            )
        return lines
    if cfg7 is not None:
        tfront = cfg7.get("tfront_offsets", [[0, 0, 0, 0]] * 4)
        tback = cfg7.get("tback_offsets", [[0, 0, 0, 0]] * 4)
        lines.append("Upstream rows missing; showing configured offsets")
        for plane_idx in range(1, 5):
            vals = []
            for strip_idx in range(1, 5):
                vals.append(
                    f"S{strip_idx}: {fmt(tfront[plane_idx - 1][strip_idx - 1])}/{fmt(tback[plane_idx - 1][strip_idx - 1])}"
                )
            lines.append(f"P{plane_idx}: " + ", ".join(vals))
    else:
        lines.append("Offsets unavailable (missing upstream data and config)")
    return lines


def describe_threshold(row: pd.Series | None, source: str) -> list[str]:
    lines = [f"Threshold + time-walk (STEP 8, source={source})"]
    if row is None:
        lines.append("STEP 8 row unavailable")
        return lines
    active = list_active_strips(row, "Q_front")
    if not active:
        active = list_active_strips(row, "Q_back")
    if not active:
        lines.append("No active strips after threshold")
        return lines
    for plane_idx, strip_idx in active:
        qf = row.get(f"Q_front_{plane_idx}_s{strip_idx}")
        qb = row.get(f"Q_back_{plane_idx}_s{strip_idx}")
        tf = row.get(f"T_front_{plane_idx}_s{strip_idx}")
        tb = row.get(f"T_back_{plane_idx}_s{strip_idx}")
        lines.append(
            f"P{plane_idx} S{strip_idx}: Q_front={fmt(qf, 2)}, Q_back={fmt(qb, 2)},"
            f" T_front={fmt(tf)} ns, T_back={fmt(tb)} ns"
        )
    if source != "step8":
        lines.append("Note: T_front/T_back include downstream timing effects.")
    return lines


def describe_trigger(row: pd.Series | None, cfg9: dict | None, source: str) -> list[str]:
    lines = [f"Trigger decision (STEP 9, source={source})"]
    if row is None:
        lines.append("Trigger row unavailable")
        return lines
    tt_trigger = row.get("tt_trigger", "NA")
    active_planes = plane_activity(row)
    triggers = cfg9.get("trigger_combinations", []) if cfg9 is not None else []
    match = False
    for trig in triggers:
        if all(ch in active_planes for ch in str(trig)):
            match = True
    lines.append(f"active_planes={active_planes}, tt_trigger={tt_trigger}")
    if triggers:
        lines.append(f"trigger_match={match} vs {triggers}")
    return lines


def describe_tdc(step9: pd.Series | None, step10: pd.Series) -> list[str]:
    lines = ["TDC + jitter (STEP 10)"]
    jitter = step10.get("daq_jitter_ns", "NA")
    lines.append(f"daq_jitter_ns={fmt(jitter)}")
    active = list_active_strips(step10, "Q_front")
    if not active:
        active = list_active_strips(step10, "Q_back")
    if not active:
        lines.append("No active strips")
        return lines
    for plane_idx, strip_idx in active:
        tf10 = step10.get(f"T_front_{plane_idx}_s{strip_idx}")
        tb10 = step10.get(f"T_back_{plane_idx}_s{strip_idx}")
        if step9 is None:
            lines.append(
                f"P{plane_idx} S{strip_idx}: T_front={fmt(tf10)} ns, T_back={fmt(tb10)} ns"
            )
            continue
        tf9 = step9.get(f"T_front_{plane_idx}_s{strip_idx}")
        tb9 = step9.get(f"T_back_{plane_idx}_s{strip_idx}")
        d_tf = float(tf10) - float(tf9) if is_finite(tf10) and is_finite(tf9) else float("nan")
        d_tb = float(tb10) - float(tb9) if is_finite(tb10) and is_finite(tb9) else float("nan")
        lines.append(
            f"P{plane_idx} S{strip_idx}: T_front={fmt(tf10)} ns (d={fmt(d_tf)}),"
            f" T_back={fmt(tb10)} ns (d={fmt(d_tb)})"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-event microscope report from STEP 10 outputs.")
    parser.add_argument("--sim-run", default="latest", help="SIM_RUN id, or latest/random")
    parser.add_argument("--geometry-id", type=int, default=1, help="Geometry id to inspect")
    parser.add_argument(
        "--event-id",
        type=int,
        required=True,
        help="Event id (uses event_id column if present, else index label or 0-based row)",
    )
    parser.add_argument("--input-path", default=None, help="Override STEP 10 input path")
    args = parser.parse_args()

    step10_dir = ROOT_DIR / "INTERSTEPS/STEP_10_TO_FINAL"
    sim_run = args.sim_run
    if args.input_path:
        step10_path = Path(args.input_path)
        sim_run = "custom"
    else:
        if sim_run == "latest":
            sim_run = latest_sim_run(step10_dir)
        elif sim_run == "random":
            sim_run = random_sim_run(step10_dir, None)
        run_dir = step10_dir / sim_run
        step10_path = find_geom_file(run_dir, args.geometry_id, "daq")

    requested_event_id = args.event_id
    row10, index_label, global_row = locate_row(step10_path, requested_event_id, requested_event_id, True)
    selected_event_id = requested_event_id
    if row10 is None:
        closest = find_closest_event_id(step10_path, requested_event_id)
        if closest is not None:
            row10, index_label, global_row = locate_row(step10_path, closest, None, False)
            selected_event_id = closest
        if row10 is None:
            raise FileNotFoundError(f"Event {requested_event_id} not found in {step10_path}")
    if row10 is not None and "event_id" in row10 and is_finite(row10.get("event_id")):
        selected_event_id = int(row10.get("event_id"))

    chain = resolve_upstream_chain(step10_path)

    def row_for_step(step_key: str, mismatch_notes: list[str]) -> pd.Series | None:
        step_path = chain.get(step_key)
        if step_path is None:
            return None
        row, _, _ = locate_row(step_path, selected_event_id, None, False)
        if row is None:
            mismatch_notes.append(f"{step_key} event_id {selected_event_id} not found")
            return None
        if not rows_match(row10, row, tol=1e-3):
            mismatch_notes.append(f"{step_key} row mismatch; using STEP 10 context only")
            return None
        return row

    mismatch_notes: list[str] = []
    row9 = row_for_step("step9", mismatch_notes)
    row8 = row_for_step("step8", mismatch_notes)
    row7 = row_for_step("step7", mismatch_notes)
    row6 = row_for_step("step6", mismatch_notes)
    row5 = row_for_step("step5", mismatch_notes)
    row4 = row_for_step("step4", mismatch_notes)
    row3 = row_for_step("step3", mismatch_notes)
    row2 = row_for_step("step2", mismatch_notes)

    cfg7 = None
    cfg9 = None
    cfg7_path = ROOT_DIR / "MASTER_STEPS/STEP_7/config_step_7_physics.yaml"
    if cfg7_path.exists():
        cfg7 = yaml.safe_load(cfg7_path.read_text()) or {}
    cfg9_path = ROOT_DIR / "MASTER_STEPS/STEP_9/config_step_9_physics.yaml"
    if cfg9_path.exists():
        cfg9 = yaml.safe_load(cfg9_path.read_text()) or {}

    header = [
        "Event microscope report",
        f"sim_run={sim_run}, geometry_id={args.geometry_id}, event_id={selected_event_id}",
        f"source={step10_path}",
        f"lookup=index_label={index_label}, row={global_row}",
    ]
    if selected_event_id != requested_event_id:
        header.append(f"requested_event_id={requested_event_id} -> selected_event_id={selected_event_id}")
    if mismatch_notes:
        header.append("upstream_warnings=" + "; ".join(mismatch_notes))

    sections = []
    gen_row = row2 if row2 is not None else row10
    cross_row = row2 if row2 is not None else row10
    avalanche_row = row3 if row3 is not None else row10
    induced_row = row4 if row4 is not None else row10
    diff_row = row5 if row5 is not None else row10
    sections.extend(describe_generation(gen_row))
    sections.extend(describe_crossings(cross_row))
    sections.extend(describe_avalanche(avalanche_row))
    sections.extend(describe_induced_strips(induced_row))
    sections.extend(describe_diff(diff_row))
    step6_row = row6 if row6 is not None else row10
    step6_source = "step6" if row6 is not None else "step10"
    sections.extend(describe_frontback(step6_row, "Front/back", step6_source))
    sections.extend(describe_offsets(row6, row7, cfg7))
    step8_row = row8 if row8 is not None else row10
    step8_source = "step8" if row8 is not None else "step10"
    sections.extend(describe_threshold(step8_row, step8_source))
    trigger_source = "step9" if row9 is not None else "step10"
    trigger_row = row9 if row9 is not None else row10
    sections.extend(describe_trigger(trigger_row, cfg9, trigger_source))
    sections.extend(describe_tdc(row9, row10))

    for line in header:
        print(line)
    print("")
    for line in sections:
        print(line)


if __name__ == "__main__":
    main()
