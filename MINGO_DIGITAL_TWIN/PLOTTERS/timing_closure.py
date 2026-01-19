#!/usr/bin/env python3
"""Timing closure tests across STEP_6/7/8/9/10 for a geometry."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "MASTER_STEPS"))

from STEP_SHARED.sim_utils import (
    iter_input_frames,
    latest_sim_run,
    random_sim_run,
)


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float((values ** 2).sum())

    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.total / self.count

    def rms(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.mean()
        return math.sqrt(max(self.total_sq / self.count - mean * mean, 0.0))

    def to_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean(), "rms": self.rms()}


class Reservoir:
    def __init__(self, max_samples: int, seed: int | None = None) -> None:
        self.max_samples = max_samples
        self.samples: list[float] = []
        self.rng = random.Random(seed)
        self.seen = 0

    def add(self, values: np.ndarray) -> None:
        for val in values.tolist():
            self.seen += 1
            if len(self.samples) < self.max_samples:
                self.samples.append(float(val))
            else:
                idx = self.rng.randint(0, self.seen - 1)
                if idx < self.max_samples:
                    self.samples[idx] = float(val)

    def array(self) -> np.ndarray:
        return np.asarray(self.samples, dtype=float)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def resolve_sim_run(base_dir: Path, sim_run: str, seed: int | None) -> str:
    if sim_run == "latest":
        return latest_sim_run(base_dir)
    if sim_run == "random":
        return random_sim_run(base_dir, seed)
    return sim_run


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


def iterate_pairs(path_a: Path, path_b: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    iter_a, _, _ = iter_input_frames(path_a, None)
    iter_b, _, _ = iter_input_frames(path_b, None)
    frames_a = list(iter_a)
    frames_b = list(iter_b)
    return frames_a, frames_b


def collect_step6_closure(df: pd.DataFrame, stats: RunningStats, samples: Reservoir) -> None:
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            tf = f"T_front_{plane_idx}_s{strip_idx}"
            tb = f"T_back_{plane_idx}_s{strip_idx}"
            td = f"T_diff_{plane_idx}_s{strip_idx}"
            if tf not in df.columns or tb not in df.columns or td not in df.columns:
                continue
            front = df[tf].to_numpy(dtype=float)
            back = df[tb].to_numpy(dtype=float)
            tdiff = df[td].to_numpy(dtype=float)
            mask = np.isfinite(front) & np.isfinite(back) & np.isfinite(tdiff)
            if not mask.any():
                continue
            residual = (front - back) + 2.0 * tdiff
            residual = residual[mask]
            stats.add(residual)
            samples.add(residual)


def collect_delta(
    df_after: pd.DataFrame,
    df_before: pd.DataFrame,
    stats: RunningStats,
    samples: Reservoir,
    offsets: list[list[float]] | None = None,
) -> None:
    for plane_idx in range(1, 5):
        for strip_idx in range(1, 5):
            for side, offset_map in (("front", offsets), ("back", offsets)):
                col = f"T_{side}_{plane_idx}_s{strip_idx}"
                if col not in df_after.columns or col not in df_before.columns:
                    continue
                after = df_after[col].to_numpy(dtype=float)
                before = df_before[col].to_numpy(dtype=float)
                n = min(len(after), len(before))
                if n == 0:
                    continue
                after = after[:n]
                before = before[:n]
                mask = np.isfinite(after) & np.isfinite(before) & (after != 0) & (before != 0)
                if not mask.any():
                    continue
                residual = after - before
                if offset_map is not None:
                    expected = float(offset_map[plane_idx - 1][strip_idx - 1])
                    residual = residual - expected
                residual = residual[mask]
                stats.add(residual)
                samples.add(residual)


def plot_hist(ax: plt.Axes, data: np.ndarray, title: str, expected_rms: float | None = None) -> None:
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(data, bins=80, color="slateblue", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Residual (ns)")
    ax.set_ylabel("Counts")
    if expected_rms is not None and np.isfinite(expected_rms):
        ax.axvline(expected_rms, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(-expected_rms, color="black", linestyle="--", linewidth=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Timing closure tests.")
    parser.add_argument("--geometry-id", type=int, default=1)
    parser.add_argument("--sim-run", default="latest")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sim_seed = args.seed

    step6_dir = ROOT_DIR / "INTERSTEPS/STEP_6_TO_7"
    step7_dir = ROOT_DIR / "INTERSTEPS/STEP_7_TO_8"
    step8_dir = ROOT_DIR / "INTERSTEPS/STEP_8_TO_9"
    step9_dir = ROOT_DIR / "INTERSTEPS/STEP_9_TO_10"
    step10_dir = ROOT_DIR / "INTERSTEPS/STEP_10_TO_FINAL"

    sim_run = resolve_sim_run(step6_dir, args.sim_run, sim_seed)
    run6 = step6_dir / sim_run
    run7 = step7_dir / sim_run
    run8 = step8_dir / sim_run
    run9 = step9_dir / sim_run
    run10 = step10_dir / sim_run

    path6 = find_geom_file(run6, args.geometry_id, "frontback")
    path7 = find_geom_file(run7, args.geometry_id, "calibrated")
    path8 = find_geom_file(run8, args.geometry_id, "threshold")
    path9 = find_geom_file(run9, args.geometry_id, "triggered")
    path10 = find_geom_file(run10, args.geometry_id, "daq")

    cfg7 = load_yaml(ROOT_DIR / "MASTER_STEPS/STEP_7/config_step_7_physics.yaml")
    cfg8 = load_yaml(ROOT_DIR / "MASTER_STEPS/STEP_8/config_step_8_physics.yaml")
    cfg10 = load_yaml(ROOT_DIR / "MASTER_STEPS/STEP_10/config_step_10_physics.yaml")

    tfront_offsets = cfg7.get("tfront_offsets", [[0, 0, 0, 0]] * 4)
    tback_offsets = cfg7.get("tback_offsets", [[0, 0, 0, 0]] * 4)
    t_fee_sigma = float(cfg8.get("t_fee_sigma_ns", 0.0))
    tdc_sigma = float(cfg10.get("tdc_sigma_ns", 0.0))
    jitter_width = float(cfg10.get("jitter_width_ns", 0.0))
    expected_tdc_rms = math.sqrt(tdc_sigma ** 2 + (jitter_width ** 2) / 12.0)

    stats = {
        "step6_front_back": RunningStats(),
        "step7_front_delta": RunningStats(),
        "step7_back_delta": RunningStats(),
        "step8_front_delta": RunningStats(),
        "step8_back_delta": RunningStats(),
        "step10_front_delta": RunningStats(),
        "step10_back_delta": RunningStats(),
    }
    samples = {key: Reservoir(args.max_samples, seed=sim_seed) for key in stats}

    frames6, frames7 = iterate_pairs(path6, path7)
    frames7b, frames8 = iterate_pairs(path7, path8)
    frames9, frames10 = iterate_pairs(path9, path10)

    for df6 in frames6:
        collect_step6_closure(df6, stats["step6_front_back"], samples["step6_front_back"])

    for df7, df6 in zip_longest(frames7, frames6):
        if df7 is None or df6 is None:
            break
        collect_delta(df7, df6, stats["step7_front_delta"], samples["step7_front_delta"], tfront_offsets)
        collect_delta(df7, df6, stats["step7_back_delta"], samples["step7_back_delta"], tback_offsets)

    for df8, df7 in zip_longest(frames8, frames7b):
        if df8 is None or df7 is None:
            break
        collect_delta(df8, df7, stats["step8_front_delta"], samples["step8_front_delta"])
        collect_delta(df8, df7, stats["step8_back_delta"], samples["step8_back_delta"])

    for df10, df9 in zip_longest(frames10, frames9):
        if df10 is None or df9 is None:
            break
        collect_delta(df10, df9, stats["step10_front_delta"], samples["step10_front_delta"])
        collect_delta(df10, df9, stats["step10_back_delta"], samples["step10_back_delta"])

    output_base = args.output
    if output_base is None:
        output_base = f"timing_closure_geom_{args.geometry_id}_{sim_run}"
    out_json = ROOT_DIR / "PLOTTERS" / f"{output_base}.json"
    out_pdf = ROOT_DIR / "PLOTTERS" / f"{output_base}.pdf"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "geometry_id": args.geometry_id,
        "sim_run": sim_run,
        "expected": {
            "t_fee_sigma_ns": t_fee_sigma,
            "tdc_sigma_ns": tdc_sigma,
            "jitter_width_ns": jitter_width,
            "expected_tdc_rms": expected_tdc_rms,
        },
        "stats": {key: value.to_dict() for key, value in stats.items()},
    }
    out_json.write_text(json.dumps(payload, indent=2))

    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_hist(ax, samples["step6_front_back"].array(), "STEP 6: (T_front - T_back) + 2*T_diff")
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_hist(
            axes[0],
            samples["step7_front_delta"].array(),
            "STEP 7: T_front delta - offset",
        )
        plot_hist(
            axes[1],
            samples["step7_back_delta"].array(),
            "STEP 7: T_back delta - offset",
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_hist(
            axes[0],
            samples["step8_front_delta"].array(),
            "STEP 8: T_front delta (FEE noise)",
            expected_rms=t_fee_sigma,
        )
        plot_hist(
            axes[1],
            samples["step8_back_delta"].array(),
            "STEP 8: T_back delta (FEE noise)",
            expected_rms=t_fee_sigma,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_hist(
            axes[0],
            samples["step10_front_delta"].array(),
            "STEP 10: T_front delta (TDC + jitter)",
            expected_rms=expected_tdc_rms,
        )
        plot_hist(
            axes[1],
            samples["step10_back_delta"].array(),
            "STEP 10: T_back delta (TDC + jitter)",
            expected_rms=expected_tdc_rms,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"Saved {out_json}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
