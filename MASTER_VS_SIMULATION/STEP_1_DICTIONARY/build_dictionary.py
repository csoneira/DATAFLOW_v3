#!/usr/bin/env python3
"""STEP_1: Build ``param_metadata_dictionary.csv`` for a given task.

This wrapper delegates to the builder script in ``STEP_1_BUILD/`` and
optionally produces quick sanity-check plots from the resulting CSV,
plus schema-level and distribution-level validations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared utilities --------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    parse_efficiencies,
    plot_histogram,
    plot_scatter,
    setup_logger,
)

log = setup_logger("STEP_1")

DEFAULT_CONFIG = STEP_DIR / "config" / "pipeline_config.json"
DEFAULT_OUT = STEP_DIR / "output"


# -------------------------------------------------------------------------
# Dictionary validation (to_do.md §2)
# -------------------------------------------------------------------------

_REQUIRED_COLS = [
    "file_name", "flux_cm2_min", "cos_n",
    "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4",
]
_EVENT_CANDIDATES = ["selected_rows", "generated_events_count"]


def _validate_dictionary(df: pd.DataFrame, out_dir: Path) -> dict:
    """Run schema, distribution and coherence checks (to_do.md §2).

    Returns a dict of validation results, also written as JSON.
    """
    issues: list[str] = []
    info: dict[str, object] = {"n_rows": int(len(df))}

    # --- 2.1 Schema & join-key integrity ---
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    info["missing_required_columns"] = missing

    # Join key uniqueness
    for key in ("file_name", "filename_base"):
        if key in df.columns:
            n_dup = int(df[key].duplicated().sum())
            info[f"{key}_duplicates"] = n_dup
            if n_dup > 0:
                issues.append(
                    f"{key} has {n_dup} duplicate values "
                    f"({n_dup / len(df) * 100:.1f}%)."
                )

    # Event-count field
    event_col = None
    for cand in _EVENT_CANDIDATES:
        if cand in df.columns:
            event_col = cand
            break
    info["event_count_column"] = event_col
    if event_col is None:
        issues.append(
            f"No event-count column found (tried {_EVENT_CANDIDATES})."
        )
    else:
        events = pd.to_numeric(df[event_col], errors="coerce")
        n_missing = int(events.isna().sum())
        if n_missing > 0:
            issues.append(f"{event_col}: {n_missing} non-numeric / NaN values.")
        info[f"{event_col}_stats"] = {
            "min": float(events.min()) if events.notna().any() else None,
            "max": float(events.max()) if events.notna().any() else None,
            "median": float(events.median()) if events.notna().any() else None,
            "n_missing": n_missing,
        }

    # Efficiency column parsability
    if "efficiencies" in df.columns:
        n_parse_fail = 0
        for val in df["efficiencies"]:
            parsed = parse_efficiencies(val)
            if parsed is None or len(parsed) < 4:
                n_parse_fail += 1
        info["efficiencies_parse_failures"] = n_parse_fail
        if n_parse_fail > 0:
            issues.append(
                f"efficiencies: {n_parse_fail} rows failed to parse into ≥4 floats."
            )

    # --- 2.2 Distribution / range sanity ---
    dist_cols = ["flux_cm2_min", "cos_n",
                 "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
    dist_summary: dict[str, dict] = {}
    for col in dist_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        vals = s.dropna()
        n_uniq = int(vals.nunique())
        dist_summary[col] = {
            "count": int(len(vals)),
            "n_unique": n_uniq,
            "min": float(vals.min()) if not vals.empty else None,
            "max": float(vals.max()) if not vals.empty else None,
            "mean": float(vals.mean()) if not vals.empty else None,
            "std": float(vals.std()) if not vals.empty else None,
        }
        if n_uniq < 3 and len(vals) > 10:
            issues.append(
                f"{col}: only {n_uniq} unique values — possible coverage gap."
            )
    info["distributions"] = dist_summary

    # Efficiency range check
    for i in range(1, 5):
        eff_col = f"eff_{i}" if f"eff_{i}" in df.columns else None
        if eff_col is None and "efficiencies" in df.columns:
            # Already checked parsability above; extract if present
            continue
        if eff_col and eff_col in df.columns:
            eff_vals = pd.to_numeric(df[eff_col], errors="coerce").dropna()
            if not eff_vals.empty:
                lo, hi = float(eff_vals.min()), float(eff_vals.max())
                if lo < 0 or hi > 1:
                    issues.append(
                        f"{eff_col} out of [0, 1] range: [{lo:.4f}, {hi:.4f}]."
                    )

    # --- 2.3 Forward coherence: rate-vs-flux linearity ---
    rate_cols = [c for c in df.columns
                 if "rate" in c.lower() or c.startswith("raw_tt_")]
    coherence: dict[str, float] = {}
    if "flux_cm2_min" in df.columns and rate_cols:
        flux = pd.to_numeric(df["flux_cm2_min"], errors="coerce")
        for rc in rate_cols[:6]:  # limit to first 6
            rate = pd.to_numeric(df[rc], errors="coerce")
            mask = flux.notna() & rate.notna() & flux.gt(0) & rate.gt(0)
            if mask.sum() > 10:
                r = np.corrcoef(
                    np.log(flux[mask].to_numpy(dtype=float)),
                    np.log(rate[mask].to_numpy(dtype=float)),
                )[0, 1]
                coherence[rc] = float(r)
                if abs(r) < 0.5:
                    issues.append(
                        f"Weak log-log correlation between flux and {rc}: "
                        f"r={r:.3f} (expected ≈1 for linear scaling)."
                    )
    info["rate_flux_log_correlation"] = coherence

    # --- Summary ---
    info["n_issues"] = len(issues)
    info["issues"] = issues

    for iss in issues:
        log.warning("VALIDATION: %s", iss)
    if not issues:
        log.info("Dictionary validation passed — no issues found.")

    report_path = out_dir / "dictionary_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log.info("Validation report: %s", report_path)
    return info


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def _generate_plots(csv_path: Path) -> None:
    """Produce quick-look histograms and scatters from the dictionary CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    plot_dir = csv_path.parent / "plots"

    # Clean old plots
    if plot_dir.exists():
        for path in plot_dir.glob("*.png"):
            path.unlink()
    plot_dir.mkdir(parents=True, exist_ok=True)

    hist_cols = ("flux_cm2_min", "cos_n",
                 "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4")
    for col in hist_cols:
        if col in df.columns:
            plot_histogram(df, col, plot_dir / f"hist_{col}.png")

    # Scatters for parameter space coverage (§2.2)
    plot_scatter(df, "flux_cm2_min", "cos_n",
                 plot_dir / "scatter_flux_vs_cos_n.png")
    for i in range(1, 5):
        zc = f"z_plane_{i}"
        if zc in df.columns:
            plot_scatter(df, "flux_cm2_min", zc,
                         plot_dir / f"scatter_flux_vs_z{i}.png")

    # Geometry coverage: all z-plane pairs (§2.2)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            ci, cj = f"z_plane_{i}", f"z_plane_{j}"
            if ci in df.columns and cj in df.columns:
                plot_scatter(df, ci, cj,
                             plot_dir / f"scatter_z{i}_vs_z{j}.png")

    # Rate-vs-flux scatters for forward coherence (§2.3)
    rate_cols = [c for c in df.columns
                 if "rate" in c.lower() or c.startswith("raw_tt_")]
    if "flux_cm2_min" in df.columns:
        for rc in rate_cols[:8]:
            plot_scatter(df, "flux_cm2_min", rc,
                         plot_dir / f"scatter_flux_vs_{rc}.png")

    log.info("Plots saved to %s", plot_dir)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build dictionary CSV for a task."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip quick sanity plots for the output CSV.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema/distribution validation of the dictionary.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"task_{args.task_id:02d}" / "param_metadata_dictionary.csv"

    builder_script = STEP_DIR / "STEP_1_BUILD" / "build_param_metadata_dictionary.py"
    if not builder_script.exists():
        raise FileNotFoundError(
            f"Builder script not found: {builder_script}\n"
            "Expected at MASTER_VS_SIMULATION/STEP_1_DICTIONARY/STEP_1_BUILD/."
        )

    cmd = [
        sys.executable,
        str(builder_script),
        "--config", str(args.config),
        "--task-id", str(args.task_id),
        "--out", str(out_csv),
    ]
    log.info("Running builder: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    log.info("Dictionary written to %s", out_csv)

    if out_csv.exists():
        if not args.no_validate:
            _validate_dictionary(
                pd.read_csv(out_csv, low_memory=False), out_dir,
            )
        if not args.no_plots:
            _generate_plots(out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
