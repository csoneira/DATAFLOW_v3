#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quick test: apply affine flux formula already calibrated in PURELY_LINEAR.

Important: this script does NOT calibrate a matrix.
It only:
1) extracts the affine formula from PURELY_LINEAR summary,
2) computes global_rate_hz, eff_2, eff_3, reference_efficiency on real metadata,
3) detects active station z-positions (P1..P4),
4) applies the extracted formula to estimate flux,
5) writes outputs + quick plots.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


ROOT = Path("/home/mingo/DATAFLOW_v3")
QUICK_TEST_DIR = ROOT / "QUICK_TEST"

DEFAULT_INPUT = ROOT / "STATIONS/MINGO01/STAGE_1/EVENT_DATA/STEP_1/TASK_1/METADATA/task_1_metadata_specific.csv"
DEFAULT_FORMULA_SUMMARY = (
    ROOT
    / "INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/PURELY_LINEAR/PLOTS/00_linear_summary.txt"
)
DEFAULT_ONLINE_RUN_DICT = ROOT / "MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY"
DEFAULT_OUTPUT_CSV = QUICK_TEST_DIR / "quick_test_flux_output.csv"
DEFAULT_OUTPUT_INFO = QUICK_TEST_DIR / "quick_test_apply_info.txt"
DEFAULT_PLOTS_DIR = QUICK_TEST_DIR / "PLOTS"

MI_FILENAME_PATTERN = re.compile(r"mi(?P<station>\d{2})(?P<digits>\d{11})$", re.IGNORECASE)
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_time_from_basename(value: object) -> pd.Timestamp:
    if not isinstance(value, str):
        return pd.NaT

    stem = Path(value).stem

    # Alternate explicit format used elsewhere.
    try:
        return pd.Timestamp(datetime.strptime(stem, "%Y-%m-%d_%H.%M.%S"))
    except ValueError:
        pass

    match = MI_FILENAME_PATTERN.search(stem)
    if not match:
        return pd.NaT

    digits = match.group("digits")
    try:
        year = 2000 + int(digits[0:2])
        day_of_year = int(digits[2:5])
        hour = int(digits[5:7])
        minute = int(digits[7:9])
        second = int(digits[9:11])
        base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    except ValueError:
        return pd.NaT

    return pd.Timestamp(base_date.replace(hour=hour, minute=minute, second=second))


def infer_station_number(input_path: Path, filename_base_series: pd.Series) -> int:
    path_match = re.search(r"MINGO(\d+)", str(input_path), flags=re.IGNORECASE)
    if path_match:
        return int(path_match.group(1))

    for value in filename_base_series.dropna():
        match = MI_FILENAME_PATTERN.search(Path(str(value)).stem)
        if match:
            return int(match.group("station"))

    raise RuntimeError("Could not infer station number from path or filename_base.")


def station_config_path(station_number: int, online_run_dict_root: Path) -> Path:
    return online_run_dict_root / f"STATION_{station_number}" / f"input_file_mingo{station_number:02d}.csv"


def load_station_config(config_csv: Path) -> pd.DataFrame:
    if not config_csv.exists():
        raise FileNotFoundError(f"Station config not found: {config_csv}")

    cfg = pd.read_csv(config_csv, skiprows=1)
    needed = ["station", "conf", "start", "end", "P1", "P2", "P3", "P4"]
    missing = [c for c in needed if c not in cfg.columns]
    if missing:
        raise KeyError(f"Missing columns in {config_csv}: {missing}")

    cfg = cfg.copy()
    cfg["start"] = pd.to_datetime(cfg["start"], errors="coerce")
    cfg["end"] = pd.to_datetime(cfg["end"], errors="coerce")
    for c in ["station", "conf", "P1", "P2", "P3", "P4"]:
        cfg[c] = pd.to_numeric(cfg[c], errors="coerce")
    cfg = cfg.dropna(subset=["station", "start", "P1", "P2", "P3", "P4"])
    return cfg


def select_active_configuration(cfg: pd.DataFrame, station_number: int, reference_time: pd.Timestamp) -> pd.Series:
    cfg_s = cfg[cfg["station"] == station_number].copy()
    if cfg_s.empty:
        raise RuntimeError(f"No station={station_number} rows in station config.")

    active = cfg_s[
        (cfg_s["start"] <= reference_time)
        & (cfg_s["end"].isna() | (reference_time <= cfg_s["end"]))
    ].copy()

    if active.empty:
        past = cfg_s[cfg_s["start"] <= reference_time].copy()
        if past.empty:
            raise RuntimeError(f"No active config for station={station_number} at {reference_time}.")
        active = past.sort_values("start").tail(1)

    return active.sort_values("start").tail(1).iloc[0]


def extract_formula_coefficients(summary_txt: Path) -> tuple[float, float, float, str]:
    """Extract m11, m12, t1 from a line like:
       flux_est = m11*global_rate + m12*eff + t1
    """
    if not summary_txt.exists():
        raise FileNotFoundError(f"Formula summary not found: {summary_txt}")

    text = summary_txt.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    candidate_lines = [ln.strip() for ln in lines if "flux_est" in ln and "=" in ln and "*" in ln]
    if not candidate_lines:
        raise RuntimeError(
            f"Could not find formula line with 'flux_est =' in {summary_txt}."
        )

    formula_line = candidate_lines[-1]
    nums = FLOAT_PATTERN.findall(formula_line)
    if len(nums) < 3:
        raise RuntimeError(f"Could not parse 3 coefficients from formula line: {formula_line}")

    m11, m12, t1 = float(nums[0]), float(nums[1]), float(nums[2])
    return m11, m12, t1, formula_line


def make_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    t = pd.to_datetime(df["time"], errors="coerce")
    if t.notna().sum() == 0:
        t = pd.date_range("2000-01-01", periods=len(df), freq="min")

    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)

    fig, (ax_rate, ax_eff) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    ax_rate.plot(t, df["global_rate_hz"], color="C0", lw=1.2, label="global_rate_hz")
    ax_eff.plot(t, df["eff_2"], color="C1", lw=1.0, alpha=0.85, label="eff_2")
    ax_eff.plot(t, df["eff_3"], color="C2", lw=1.0, alpha=0.85, label="eff_3")
    ax_eff.plot(t, df["reference_efficiency"], color="C3", lw=1.2, alpha=0.95, label="reference_eff")

    ax_rate.set_title("Quick test inputs over time")
    ax_rate.set_ylabel("rate [Hz]")
    ax_eff.set_ylabel("efficiency")
    ax_eff.set_xlabel("time")
    ax_rate.grid(True, alpha=0.2)
    ax_eff.grid(True, alpha=0.2)

    ax_eff.xaxis.set_major_locator(loc)
    ax_eff.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    plt.setp(ax_eff.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    ax_rate.legend(loc="best", fontsize=8)
    ax_eff.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "01_inputs_over_time.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t, df["flux_est"], color="C4", lw=1.2)
    ax.set_title("Estimated flux over time (quick test)")
    ax.set_xlabel("time")
    ax.set_ylabel("flux_est")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "02_flux_est_over_time.png", dpi=170)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quick test: apply affine formula from PURELY_LINEAR to real metadata."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="TASK_1 metadata CSV")
    p.add_argument(
        "--formula-summary",
        type=Path,
        default=DEFAULT_FORMULA_SUMMARY,
        help="Path to PURELY_LINEAR/PLOTS/00_linear_summary.txt",
    )
    p.add_argument(
        "--online-run-dict-root",
        type=Path,
        default=DEFAULT_ONLINE_RUN_DICT,
        help="Root containing STATION_X/input_file_mingoXX.csv",
    )
    p.add_argument("--station", type=int, default=None, help="Optional station override")
    p.add_argument("--station-config", type=Path, default=None, help="Optional explicit station config CSV")
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    p.add_argument("--output-info", type=Path, default=DEFAULT_OUTPUT_INFO, help="Output info TXT path")
    p.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Directory for PNG plots")
    p.add_argument("--clip-eff", action="store_true", help="Clip efficiencies to [0,1]")
    return p


def main() -> None:
    args = build_parser().parse_args()

    meta = pd.read_csv(args.input)
    needed = ["filename_base", "raw_tt_1234_rate_hz", "raw_tt_134_rate_hz", "raw_tt_124_rate_hz"]
    missing = [c for c in needed if c not in meta.columns]
    if missing:
        raise KeyError(f"Missing columns in {args.input}: {missing}")

    data = meta.dropna(subset=needed).copy()
    data["time"] = data["filename_base"].map(parse_time_from_basename)
    data = data.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if data.empty:
        raise RuntimeError("No valid rows after parsing metadata times/rates.")

    data["rate_1234_hz"] = pd.to_numeric(data["raw_tt_1234_rate_hz"], errors="coerce")

    if "events_per_second_global_rate" in data.columns and data["events_per_second_global_rate"].notna().sum() > 0:
        rate_source = "events_per_second_global_rate (fallback to raw_tt_1234_rate_hz when NaN)"
        global_rate = pd.to_numeric(data["events_per_second_global_rate"], errors="coerce")
        data["global_rate_hz"] = global_rate.where(global_rate.notna(), data["rate_1234_hz"])
    else:
        rate_source = "raw_tt_1234_rate_hz"
        data["global_rate_hz"] = data["rate_1234_hz"]

    # Efficiencies are defined from telescope rates, not from global rate.
    denom = data["rate_1234_hz"].replace(0, np.nan)
    data["eff_2"] = 1.0 - pd.to_numeric(data["raw_tt_134_rate_hz"], errors="coerce") / denom
    data["eff_3"] = 1.0 - pd.to_numeric(data["raw_tt_124_rate_hz"], errors="coerce") / denom
    data["reference_efficiency"] = 0.5 * (data["eff_2"] + data["eff_3"])
    eff_definition = "eff_2=1-raw_tt_134_rate_hz/raw_tt_1234_rate_hz; eff_3=1-raw_tt_124_rate_hz/raw_tt_1234_rate_hz"

    if args.clip_eff:
        data["eff_2"] = data["eff_2"].clip(0, 1)
        data["eff_3"] = data["eff_3"].clip(0, 1)
        data["reference_efficiency"] = data["reference_efficiency"].clip(0, 1)

    station_number = args.station if args.station is not None else infer_station_number(args.input, data["filename_base"])
    cfg_path = args.station_config if args.station_config is not None else station_config_path(station_number, args.online_run_dict_root)
    cfg = load_station_config(cfg_path)
    ref_time = pd.Timestamp(data["time"].median())
    active = select_active_configuration(cfg, station_number=station_number, reference_time=ref_time)
    z_positions = (float(active["P1"]), float(active["P2"]), float(active["P3"]), float(active["P4"]))

    m11, m12, t1, formula_line = extract_formula_coefficients(args.formula_summary)
    eff_term_used = "reference_efficiency = (eff_2 + eff_3)/2"
    data["flux_est"] = m11 * data["global_rate_hz"] + m12 * data["reference_efficiency"] + t1

    out_cols = [
        "time",
        "filename_base",
        "global_rate_hz",
        "rate_1234_hz",
        "eff_2",
        "eff_3",
        "reference_efficiency",
        "flux_est",
    ]
    out = data[out_cols].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    make_plots(out, args.plots_dir)

    info_lines = [
        f"Input metadata: {args.input}",
        f"Formula summary source: {args.formula_summary}",
        f"Extracted formula line: {formula_line}",
        f"Applied eff term in this quick test: {eff_term_used}",
        f"Output CSV: {args.output_csv}",
        f"Plots directory: {args.plots_dir}",
        f"Global rate source in metadata: {rate_source}",
        f"Efficiency definition: {eff_definition}",
        f"Station: MINGO{station_number:02d}",
        f"Station config used: {cfg_path}",
        (
            "Active config row: "
            f"conf={int(active['conf']) if pd.notna(active['conf']) else 'NA'}, "
            f"start={str(active['start'])[:10]}, "
            f"end={str(active['end'])[:10] if pd.notna(active['end']) else 'open'}"
        ),
        (
            "z positions selected (P1,P2,P3,P4) = "
            f"({z_positions[0]:.0f}, {z_positions[1]:.0f}, {z_positions[2]:.0f}, {z_positions[3]:.0f})"
        ),
        "",
        "Applied coefficients:",
        f"  m11 = {m11:.12f}",
        f"  m12 = {m12:.12f}",
        f"  t1  = {t1:.12f}",
        "",
        "Applied row-wise formula:",
        "  flux_est = m11*global_rate_hz + m12*reference_efficiency + t1",
    ]
    args.output_info.parent.mkdir(parents=True, exist_ok=True)
    args.output_info.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    print(f"Input metadata: {args.input}")
    print(f"Formula summary source: {args.formula_summary}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output info TXT: {args.output_info}")
    print(f"Plots directory: {args.plots_dir}")
    print(f"Global rate source: {rate_source}")
    print(f"Efficiency definition: {eff_definition}")
    print(f"Applied eff term: {eff_term_used}")
    print(f"Station: MINGO{station_number:02d}")
    print(f"Station config used: {cfg_path}")
    print(
        f"z positions selected (P1,P2,P3,P4) = "
        f"({z_positions[0]:.0f}, {z_positions[1]:.0f}, {z_positions[2]:.0f}, {z_positions[3]:.0f})"
    )
    print("\nExtracted coefficients:")
    print(f"  m11={m11:.12f}, m12={m12:.12f}, t1={t1:.12f}")
    print("Applied formula:")
    print("  flux_est = m11*global_rate_hz + m12*reference_efficiency + t1")


if __name__ == "__main__":
    main()
