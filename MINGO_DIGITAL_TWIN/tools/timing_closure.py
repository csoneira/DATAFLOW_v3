#!/usr/bin/env python3
"""Timing closure checks using event_id alignment across steps."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import iter_input_frames, latest_sim_run, random_sim_run


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_sq += value * value

    def add_array(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.total += float(finite.sum())
        self.total_sq += float((finite ** 2).sum())

    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count

    def rms(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.mean()
        return math.sqrt(max(self.total_sq / self.count - mean * mean, 0.0))


def is_finite(val: object) -> bool:
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


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


def resolve_chain(step10_path: Path) -> dict:
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


def reservoir_sample(values: list[int], max_samples: int, rng: random.Random) -> list[int]:
    if max_samples <= 0:
        return []
    samples: list[int] = []
    seen = 0
    for val in values:
        seen += 1
        if len(samples) < max_samples:
            samples.append(val)
        else:
            idx = rng.randint(0, seen - 1)
            if idx < max_samples:
                samples[idx] = val
    return samples


def sample_event_ids(path: Path, max_events: int, seed: int | None) -> tuple[list[int], bool]:
    rng = random.Random(seed)
    event_ids: list[int] = []
    event_id_seen = False
    for df in iter_input_frames(path, None)[0]:
        if df.empty:
            continue
        if "event_id" in df.columns:
            ids = pd.to_numeric(df["event_id"], errors="coerce").dropna().astype(int).tolist()
            event_id_seen = True
        else:
            ids = [int(idx) for idx in df.index.to_list() if pd.notna(idx)]
        event_ids.extend(ids)
        if len(event_ids) >= max_events * 5:
            break
    sampled = reservoir_sample(event_ids, max_events, rng)
    return sampled, event_id_seen


def collect_rows_by_event_id(path: Path, event_ids: set[int], columns: list[str]) -> dict[int, pd.Series]:
    rows: dict[int, pd.Series] = {}
    for df in iter_input_frames(path, None)[0]:
        if df.empty:
            continue
        if "event_id" in df.columns:
            ids = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
            mask = ids.isin(event_ids)
            subset = df.loc[mask]
            if subset.empty:
                continue
            cols = [col for col in columns if col in subset.columns]
            if "event_id" not in cols:
                cols = ["event_id"] + cols
            subset = subset[cols]
            for _, row in subset.iterrows():
                event_id = int(row.get("event_id"))
                rows.setdefault(event_id, row)
        else:
            for idx, row in df.iterrows():
                if idx in event_ids:
                    cols = [col for col in columns if col in row.index]
                    rows.setdefault(int(idx), row[cols])
    return rows


def collect_front_back(step5: dict[int, pd.Series], step6: dict[int, pd.Series]) -> RunningStats:
    stats = RunningStats()
    for event_id in sorted(step5.keys() & step6.keys()):
        row5 = step5[event_id]
        row6 = step6[event_id]
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                tdiff = row5.get(f"T_diff_{plane_idx}_s{strip_idx}")
                tf = row6.get(f"T_front_{plane_idx}_s{strip_idx}")
                tb = row6.get(f"T_back_{plane_idx}_s{strip_idx}")
                if not (is_finite(tdiff) and is_finite(tf) and is_finite(tb)):
                    continue
                residual = (float(tf) - float(tb)) + 2.0 * float(tdiff)
                stats.add(residual)
    return stats


def collect_connector_delta(
    step6: dict[int, pd.Series],
    step7: dict[int, pd.Series],
    offsets: dict[str, list[list[float]]] | None,
    apply_offsets: bool,
) -> tuple[RunningStats, RunningStats]:
    stats_front = RunningStats()
    stats_back = RunningStats()
    tfront_offsets = offsets.get("tfront_offsets") if offsets else None
    tback_offsets = offsets.get("tback_offsets") if offsets else None

    for event_id in sorted(step6.keys() & step7.keys()):
        row6 = step6[event_id]
        row7 = step7[event_id]
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                tf6 = row6.get(f"T_front_{plane_idx}_s{strip_idx}")
                tf7 = row7.get(f"T_front_{plane_idx}_s{strip_idx}")
                tb6 = row6.get(f"T_back_{plane_idx}_s{strip_idx}")
                tb7 = row7.get(f"T_back_{plane_idx}_s{strip_idx}")
                if is_finite(tf6) and is_finite(tf7):
                    delta = float(tf7) - float(tf6)
                    if apply_offsets and tfront_offsets is not None:
                        delta -= float(tfront_offsets[plane_idx - 1][strip_idx - 1])
                    stats_front.add(delta)
                if is_finite(tb6) and is_finite(tb7):
                    delta = float(tb7) - float(tb6)
                    if apply_offsets and tback_offsets is not None:
                        delta -= float(tback_offsets[plane_idx - 1][strip_idx - 1])
                    stats_back.add(delta)
    return stats_front, stats_back


def collect_tdc_delta(step9: dict[int, pd.Series], step10: dict[int, pd.Series]) -> tuple[RunningStats, RunningStats]:
    stats_front = RunningStats()
    stats_back = RunningStats()
    for event_id in sorted(step9.keys() & step10.keys()):
        row9 = step9[event_id]
        row10 = step10[event_id]
        for plane_idx in range(1, 5):
            for strip_idx in range(1, 5):
                qf = row9.get(f"Q_front_{plane_idx}_s{strip_idx}")
                qb = row9.get(f"Q_back_{plane_idx}_s{strip_idx}")
                active = (is_finite(qf) and float(qf) > 0) or (is_finite(qb) and float(qb) > 0)
                if not active:
                    continue
                tf9 = row9.get(f"T_front_{plane_idx}_s{strip_idx}")
                tb9 = row9.get(f"T_back_{plane_idx}_s{strip_idx}")
                tf10 = row10.get(f"T_front_{plane_idx}_s{strip_idx}")
                tb10 = row10.get(f"T_back_{plane_idx}_s{strip_idx}")
                if is_finite(tf9) and is_finite(tf10):
                    stats_front.add(float(tf10) - float(tf9))
                if is_finite(tb9) and is_finite(tb10):
                    stats_back.add(float(tb10) - float(tb9))
    return stats_front, stats_back


def collect_absolute_sum(step2: dict[int, pd.Series], step10: dict[int, pd.Series]) -> RunningStats:
    stats = RunningStats()
    for event_id in sorted(step2.keys() & step10.keys()):
        row2 = step2[event_id]
        row10 = step10[event_id]
        for plane_idx in range(1, 5):
            tsum = row2.get(f"T_sum_{plane_idx}_ns")
            if not is_finite(tsum):
                continue
            for strip_idx in range(1, 5):
                tf = row10.get(f"T_front_{plane_idx}_s{strip_idx}")
                tb = row10.get(f"T_back_{plane_idx}_s{strip_idx}")
                if not (is_finite(tf) and is_finite(tb)):
                    continue
                residual = 0.5 * (float(tf) + float(tb)) - float(tsum)
                stats.add(residual)
    return stats


def stats_payload(stats: RunningStats) -> dict:
    return {"count": stats.count, "mean": stats.mean(), "rms": stats.rms()}


def check_stats(
    stats: RunningStats,
    expected_rms: float | None,
    mean_sigma: float,
    rms_ratio_min: float,
    rms_ratio_max: float,
    abs_tol: float,
) -> tuple[bool, dict]:
    if stats.count == 0:
        return False, {"count": 0, "reason": "no samples"}
    mean_val = stats.mean()
    rms_val = stats.rms()
    mean_ok = True
    rms_ok = True
    if expected_rms is not None and math.isfinite(expected_rms):
        threshold = max(abs_tol, mean_sigma * expected_rms)
        mean_ok = abs(mean_val) <= threshold
        if expected_rms > 0:
            ratio = rms_val / expected_rms if math.isfinite(rms_val) else float("inf")
            rms_ok = rms_ratio_min <= ratio <= rms_ratio_max
        else:
            rms_ok = rms_val <= abs_tol
    ok = mean_ok and rms_ok
    return ok, {
        "count": stats.count,
        "mean": mean_val,
        "rms": rms_val,
        "expected_rms": expected_rms,
        "mean_ok": mean_ok,
        "rms_ok": rms_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Timing closure checks with event_id alignment.")
    parser.add_argument("--sim-run", default="latest", help="SIM_RUN id, or latest/random")
    parser.add_argument("--geometry-id", type=int, default=1, help="Geometry id to inspect")
    parser.add_argument("--max-events", type=int, default=200000, help="Maximum events to sample from STEP 10")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--input-path", default=None, help="Override STEP 10 input path")
    parser.add_argument("--include-absolute", action="store_true", help="Include STEP 2 -> STEP 10 sum-time check")
    parser.add_argument("--apply-offsets", action="store_true", help="Subtract configured offsets in connector delta")
    parser.add_argument("--output-json", default=None, help="Write summary JSON to this path")
    parser.add_argument("--fail-on-bad", action="store_true", help="Exit non-zero if checks fail")
    parser.add_argument("--mean-sigma", type=float, default=5.0, help="Mean tolerance in units of expected RMS")
    parser.add_argument("--rms-ratio-min", type=float, default=0.5, help="Min allowed RMS/expected RMS ratio")
    parser.add_argument("--rms-ratio-max", type=float, default=2.0, help="Max allowed RMS/expected RMS ratio")
    parser.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute tolerance for zero-expected checks")
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
            sim_run = random_sim_run(step10_dir, args.seed)
        run_dir = step10_dir / sim_run
        step10_path = find_geom_file(run_dir, args.geometry_id, "daq")

    event_ids, event_id_seen = sample_event_ids(step10_path, args.max_events, args.seed)
    if not event_ids:
        raise FileNotFoundError(f"No events found in {step10_path}")
    if not event_id_seen:
        print("Warning: event_id column not found; using row index alignment.")

    chain = resolve_chain(step10_path)

    cols_step10 = ["event_id"] + [
        f"T_front_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ] + [
        f"T_back_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ] + [
        f"Q_front_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ] + [
        f"Q_back_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ]

    cols_step9 = cols_step10
    cols_step7 = ["event_id"] + [
        f"T_front_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ] + [
        f"T_back_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ]
    cols_step6 = cols_step10
    cols_step5 = ["event_id"] + [
        f"T_diff_{i}_s{j}" for i in range(1, 5) for j in range(1, 5)
    ]
    cols_step2 = ["event_id"] + [f"T_sum_{i}_ns" for i in range(1, 5)]

    event_id_set = set(event_ids)
    step10_rows = collect_rows_by_event_id(step10_path, event_id_set, cols_step10)
    step9_rows = collect_rows_by_event_id(chain.get("step9", Path("")), event_id_set, cols_step9) if "step9" in chain else {}
    step7_rows = collect_rows_by_event_id(chain.get("step7", Path("")), event_id_set, cols_step7) if "step7" in chain else {}
    step6_rows = collect_rows_by_event_id(chain.get("step6", Path("")), event_id_set, cols_step6) if "step6" in chain else {}
    step5_rows = collect_rows_by_event_id(chain.get("step5", Path("")), event_id_set, cols_step5) if "step5" in chain else {}
    step2_rows = collect_rows_by_event_id(chain.get("step2", Path("")), event_id_set, cols_step2) if "step2" in chain else {}

    cfg7 = {}
    cfg10 = {}
    cfg7_path = ROOT_DIR / "MASTER_STEPS/STEP_7/config_step_7_physics.yaml"
    if cfg7_path.exists():
        cfg7 = yaml.safe_load(cfg7_path.read_text()) or {}
    cfg10_path = ROOT_DIR / "MASTER_STEPS/STEP_10/config_step_10_physics.yaml"
    if cfg10_path.exists():
        cfg10 = yaml.safe_load(cfg10_path.read_text()) or {}

    closure = {
        "front_back": stats_payload(collect_front_back(step5_rows, step6_rows)) if step5_rows and step6_rows else None,
        "connector_front": None,
        "connector_back": None,
        "tdc_front": None,
        "tdc_back": None,
        "absolute_sum": None,
    }
    checks = {}

    if step6_rows and step7_rows:
        stats_front, stats_back = collect_connector_delta(
            step6_rows,
            step7_rows,
            {"tfront_offsets": cfg7.get("tfront_offsets"), "tback_offsets": cfg7.get("tback_offsets")},
            args.apply_offsets,
        )
        closure["connector_front"] = stats_payload(stats_front)
        closure["connector_back"] = stats_payload(stats_back)

    if step9_rows and step10_rows:
        stats_front, stats_back = collect_tdc_delta(step9_rows, step10_rows)
        closure["tdc_front"] = stats_payload(stats_front)
        closure["tdc_back"] = stats_payload(stats_back)
        tdc_sigma = float(cfg10.get("tdc_sigma_ns", 0.0))
        jitter_width = float(cfg10.get("jitter_width_ns", 0.0))
        closure["tdc_expected_rms"] = math.sqrt(tdc_sigma ** 2 + (jitter_width ** 2) / 12.0)

    if args.include_absolute and step2_rows and step10_rows:
        closure["absolute_sum"] = stats_payload(collect_absolute_sum(step2_rows, step10_rows))
        closure["absolute_note"] = "Offsets and downstream jitter are not removed in this check."

    if step5_rows and step6_rows:
        stats = collect_front_back(step5_rows, step6_rows)
        ok, details = check_stats(
            stats,
            expected_rms=0.0,
            mean_sigma=args.mean_sigma,
            rms_ratio_min=args.rms_ratio_min,
            rms_ratio_max=args.rms_ratio_max,
            abs_tol=args.abs_tol,
        )
        checks["front_back"] = {"ok": ok, **details}

    if step9_rows and step10_rows:
        stats_front, stats_back = collect_tdc_delta(step9_rows, step10_rows)
        expected = closure.get("tdc_expected_rms")
        ok_f, details_f = check_stats(
            stats_front,
            expected_rms=expected,
            mean_sigma=args.mean_sigma,
            rms_ratio_min=args.rms_ratio_min,
            rms_ratio_max=args.rms_ratio_max,
            abs_tol=args.abs_tol,
        )
        ok_b, details_b = check_stats(
            stats_back,
            expected_rms=expected,
            mean_sigma=args.mean_sigma,
            rms_ratio_min=args.rms_ratio_min,
            rms_ratio_max=args.rms_ratio_max,
            abs_tol=args.abs_tol,
        )
        checks["tdc_front"] = {"ok": ok_f, **details_f}
        checks["tdc_back"] = {"ok": ok_b, **details_b}

    summary = {
        "sim_run": sim_run,
        "geometry_id": args.geometry_id,
        "sample_events": len(event_ids),
        "paths": {key: str(val) for key, val in chain.items()},
        "closures": closure,
        "checks": checks,
        "options": {
            "include_absolute": args.include_absolute,
            "apply_offsets": args.apply_offsets,
            "max_events": args.max_events,
        },
    }

    print(f"Timing closure summary (sim_run={sim_run}, geometry_id={args.geometry_id})")
    print(f"Sample events: {len(event_ids)}")
    if closure["front_back"]:
        print("Front/back vs T_diff (STEP 5->6):", closure["front_back"])
    if closure["connector_front"]:
        label = "Connector delta (STEP 6->7, offsets subtracted)" if args.apply_offsets else "Connector delta (STEP 6->7)"
        print(label + " front:", closure["connector_front"])
        print(label + " back:", closure["connector_back"])
    if closure["tdc_front"]:
        print("TDC delta (STEP 9->10) front:", closure["tdc_front"])
        print("TDC delta (STEP 9->10) back:", closure["tdc_back"])
        print("TDC expected RMS:", closure.get("tdc_expected_rms"))
    if closure["absolute_sum"]:
        print("Absolute sum-time (STEP 2->10):", closure["absolute_sum"])
        print(closure.get("absolute_note", ""))

    if checks:
        failing = [name for name, result in checks.items() if not result.get("ok", True)]
        if failing:
            print("Check failures:", ", ".join(failing))
        else:
            print("All checks passed.")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved {out_path}")

    if args.fail_on_bad:
        failing = [name for name, result in checks.items() if not result.get("ok", True)]
        if failing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
