#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_2_VALIDATION/validate_solution.py
Purpose: STEP 2.2 — Validation of the inverse-problem solution.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_2_VALIDATION/validate_solution.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent
PIPELINE_DIR = INFERENCE_DIR.parent               # .../STEPS
PROJECT_DIR = PIPELINE_DIR.parent                 # .../MINGO_DICTIONARY_CREATION_AND_TEST
DEFAULT_CONFIG = PROJECT_DIR / "config.json"

DEFAULT_ESTIMATED = (
    INFERENCE_DIR / "STEP_2_1_ESTIMATE_PARAMS"
    / "OUTPUTS" / "FILES" / "estimated_params.csv"
)
DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_DATASET = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_DATASET_ENLARGED = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_3_ENLARGE_DATASET"
    / "OUTPUTS" / "FILES" / "enlarged_dataset.csv"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "2_2"


def _save_figure(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save figure with a per-script sequential numeric prefix."""
    global _FIGURE_COUNTER
    _FIGURE_COUNTER += 1
    out_path = Path(path)
    out_path = out_path.with_name(f"{FIGURE_STEP_PREFIX}_{_FIGURE_COUNTER}_{out_path.name}")
    fig.savefig(out_path, **kwargs)


_PLOT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".eps",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}


def _clear_plots_dir() -> None:
    """Remove previously generated plot files from the plots directory."""
    removed = 0
    for candidate in PLOTS_DIR.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in _PLOT_EXTENSIONS:
            try:
                candidate.unlink()
                removed += 1
            except OSError as exc:
                log.warning("Could not remove old plot file %s: %s", candidate, exc)
    log.info("Cleared %d plot file(s) from %s", removed, PLOTS_DIR)

logging.basicConfig(
    format="[%(levelname)s] STEP_2.2 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.2")


def _load_config(path: Path) -> dict:
    def _merge_dicts(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    runtime_path = path.with_name("config_runtime.json")
    if runtime_path.exists():
        runtime_cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, runtime_cfg)
        log.info("Loaded runtime overrides: %s", runtime_path)
    return cfg


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_input_path(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_project = PROJECT_DIR / p
    if candidate_project.exists():
        return candidate_project
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_project


def _select_default_dataset_path(config: dict) -> Path:
    cfg_13 = config.get("step_1_3", {})
    enabled_13 = _as_bool(cfg_13.get("enabled", False), False)
    if not enabled_13:
        return DEFAULT_DATASET

    enlarged_cfg = cfg_13.get("enlarged_dataset_csv", None)
    enlarged_path = _resolve_input_path(enlarged_cfg) if enlarged_cfg else DEFAULT_DATASET_ENLARGED
    if enlarged_path.exists():
        log.info("STEP 1.3 selection: using enlarged dataset for STEP 2 (%s).", enlarged_path)
        return enlarged_path

    log.warning(
        "STEP 1.3 is enabled but enlarged dataset file is missing: %s. Falling back to STEP 1.2 dataset.",
        enlarged_path,
    )
    return DEFAULT_DATASET


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2.2: Validate the inverse-problem solution."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--estimated-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--dataset-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()
    cfg_22 = config.get("step_2_2", {})
    cfg_21 = config.get("step_2_1", {})

    est_path = _resolve_input_path(args.estimated_csv) if args.estimated_csv else DEFAULT_ESTIMATED
    dict_path = _resolve_input_path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY
    data_path = _resolve_input_path(args.dataset_csv) if args.dataset_csv else _select_default_dataset_path(config)
    cfg_13 = config.get("step_1_3", {})
    if args.dataset_csv:
        dataset_mode = "cli_dataset_override"
    elif (
        _as_bool(cfg_13.get("enabled", False), False)
        and data_path.resolve() != DEFAULT_DATASET.resolve()
    ):
        dataset_mode = "step_1_3_enlarged"
    else:
        dataset_mode = "step_1_2_original"

    relerr_clip = float(cfg_22.get("relerr_threshold_pct", 50.0))

    # Plot parameters: use step_2_1 as the shared source for STEP 2.x.
    plot_params = cfg_21.get("plot_parameters", None)
    if plot_params is None:
        legacy_plot_params = cfg_22.get("plot_parameters", None)
        if legacy_plot_params is not None:
            log.warning(
                "Deprecated config key step_2_2.plot_parameters detected; use step_2_1.plot_parameters."
            )
            plot_params = legacy_plot_params
    if plot_params is None:
        plot_params = config.get("step_1_2", {}).get("plot_parameters", None)

    relerr_plot_limits_cfg = cfg_22.get("relerr_plot_limits_pct", [-5.0, 5.0])
    relerr_plot_limits = (-5.0, 5.0)
    if isinstance(relerr_plot_limits_cfg, (list, tuple)) and len(relerr_plot_limits_cfg) == 2:
        try:
            lo = float(relerr_plot_limits_cfg[0])
            hi = float(relerr_plot_limits_cfg[1])
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                if lo > hi:
                    lo, hi = hi, lo
                relerr_plot_limits = (lo, hi)
        except (TypeError, ValueError):
            log.warning(
                "Invalid step_2_2.relerr_plot_limits_pct=%r; using default [-5, 5].",
                relerr_plot_limits_cfg,
            )
    signed_relerr_cmap_name = str(cfg_22.get("signed_relerr_cmap", "PiYG")).strip() or "PiYG"

    for label, path in [("Estimated", est_path), ("Dictionary", dict_path), ("Dataset", data_path)]:
        if not path.exists():
            log.error("%s CSV not found: %s", label, path)
            return 1

    log.info("Loading estimated: %s", est_path)
    est_df = pd.read_csv(est_path, low_memory=False)
    log.info("Loading dictionary: %s", dict_path)
    dict_df = pd.read_csv(dict_path, low_memory=False)
    log.info("Loading dataset: %s", data_path)
    data_df = pd.read_csv(data_path, low_memory=False)

    # ── Build validation table ───────────────────────────────────────
    # Identify parameter columns (true_<x> and est_<x>)
    param_names = []
    for col in est_df.columns:
        if col.startswith("est_"):
            base = col[4:]  # strip "est_"
            true_col = f"true_{base}"
            if true_col in est_df.columns:
                param_names.append(base)

    log.info("Validating parameters: %s", param_names)
    if plot_params:
        log.info("Configured plot parameters: %s", plot_params)
    else:
        log.info("Configured plot parameters: all validated parameters")
    log.info(
        "Configured relative-error plotting window: [%.2f, %.2f]%%",
        relerr_plot_limits[0], relerr_plot_limits[1],
    )

    val = est_df.copy()
    for pname in param_names:
        true_col = f"true_{pname}"
        est_col = f"est_{pname}"
        t = pd.to_numeric(val[true_col], errors="coerce")
        e = pd.to_numeric(val[est_col], errors="coerce")
        err = (e - t)
        err = err.where(err.abs() > 1e-10, 0.0)
        val[f"error_{pname}"] = err
        relerr = (e - t) / t.replace({0: np.nan}) * 100.0
        relerr = relerr.where(relerr.abs() > 1e-10, 0.0)
        val[f"relerr_{pname}_pct"] = relerr
        val[f"abs_relerr_{pname}_pct"] = relerr.abs()

    # Add n_events from dataset
    if "true_n_events" in val.columns:
        val["n_events"] = pd.to_numeric(val["true_n_events"], errors="coerce")
    elif "n_events" in data_df.columns:
        val["n_events"] = pd.to_numeric(data_df["n_events"].values[:len(val)], errors="coerce")

    # ── Save ─────────────────────────────────────────────────────────
    val_path = FILES_DIR / "validation_results.csv"
    val.to_csv(val_path, index=False)
    log.info("Wrote validation results: %s (%d rows)", val_path, len(val))

    # Summary
    summary: dict = {
        "estimated_csv": str(est_path),
        "dictionary_csv": str(dict_path),
        "dataset_csv": str(data_path),
        "dataset_source_mode": dataset_mode,
        "total_points": len(val),
        "parameters_validated": param_names,
    }
    for pname in param_names:
        col = f"abs_relerr_{pname}_pct"
        if col in val.columns:
            s = val[col].dropna()
            if not s.empty:
                summary[f"median_abs_relerr_{pname}_pct"] = float(s.median())
                summary[f"p68_abs_relerr_{pname}_pct"] = float(s.quantile(0.68))
                summary[f"p95_abs_relerr_{pname}_pct"] = float(s.quantile(0.95))

    with open(FILES_DIR / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────
    _make_plots(
        val,
        dict_df,
        data_df,
        param_names,
        relerr_clip,
        plot_params,
        relerr_plot_limits,
        signed_relerr_cmap_name,
    )

    log.info("Done.")
    return 0


def _make_plots(
    val: pd.DataFrame,
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    param_names: list[str],
    relerr_clip: float,
    plot_params: list[str] | None = None,
    relerr_plot_limits: tuple[float, float] = (-5.0, 5.0),
    signed_relerr_cmap_name: str = "PiYG",
) -> None:
    """Generate validation plots."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    signed_relerr_cmap = None
    cmap_candidates = [
        str(signed_relerr_cmap_name).strip(),
        f"cmc.{str(signed_relerr_cmap_name).strip()}",
        "PiYG",
        "PiYG_r",
        "RdYlBu_r",
    ]
    for cmap_name in cmap_candidates:
        if not cmap_name:
            continue
        try:
            signed_relerr_cmap = plt.get_cmap(cmap_name)
            break
        except ValueError:
            continue
    if signed_relerr_cmap is None:
        signed_relerr_cmap = plt.get_cmap("RdYlBu_r")

    selected_params = param_names
    if plot_params:
        selected_params = [p for p in plot_params if p in param_names]
        missing = [p for p in plot_params if p not in param_names]
        if missing:
            log.warning(
                "Ignoring plot parameters not present in validation output: %s",
                missing,
            )
    if not selected_params:
        selected_params = param_names
        log.warning(
            "No valid configured plot parameters found; using all validated parameters."
        )

    relerr_plot_min, relerr_plot_max = relerr_plot_limits
    if relerr_plot_min > relerr_plot_max:
        relerr_plot_min, relerr_plot_max = relerr_plot_max, relerr_plot_min
    if relerr_plot_min == relerr_plot_max:
        relerr_plot_min -= 1.0
        relerr_plot_max += 1.0
    relerr_clip = abs(float(relerr_clip)) if np.isfinite(relerr_clip) else 50.0

    def _rows_with_dictionary_parameter_set() -> pd.Series:
        """Flag validation rows whose parameter set exists in dictionary.

        Priority:
        1. `param_hash_x` mapped from dataset via `dataset_index` (robust and exact).
        2. Fallback tuple match on true parameter columns.
        """
        mask = pd.Series(False, index=val.index, dtype=bool)

        # Preferred: exact identifier from dataset row.
        if (
            "dataset_index" in val.columns
            and "param_hash_x" in data_df.columns
            and "param_hash_x" in dict_df.columns
        ):
            idx = pd.to_numeric(val["dataset_index"], errors="coerce")
            idx_ok = idx.notna() & (idx >= 0) & (idx < len(data_df))
            if idx_ok.any():
                row_keys = pd.Series(index=val.index, dtype=object)
                mapped = data_df.iloc[idx[idx_ok].astype(int).to_numpy()]["param_hash_x"].astype(str).to_numpy()
                row_keys.loc[idx_ok] = mapped
                dict_keys = set(dict_df["param_hash_x"].astype(str).dropna().tolist())
                return row_keys.isin(dict_keys).fillna(False)

        # Fallback: tuple over true physical params (and z planes when available).
        base_cols = [
            "flux_cm2_min", "cos_n",
            "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
            "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4",
        ]
        dict_cols = [c for c in base_cols if c in dict_df.columns and f"true_{c}" in val.columns]
        if not dict_cols:
            return mask

        dict_num = dict_df[dict_cols].apply(pd.to_numeric, errors="coerce")
        val_num = val[[f"true_{c}" for c in dict_cols]].apply(pd.to_numeric, errors="coerce")
        # Quantize to suppress tiny float noise in merges/computations.
        dict_keys = {
            tuple(np.round(r, 12)) for r in dict_num.to_numpy(dtype=float)
            if np.all(np.isfinite(r))
        }
        if not dict_keys:
            return mask

        val_keys = [
            tuple(np.round(r, 12)) if np.all(np.isfinite(r)) else None
            for r in val_num.to_numpy(dtype=float)
        ]
        return pd.Series([k in dict_keys if k is not None else False for k in val_keys], index=val.index)

    # Remove stale per-parameter figures so outputs reflect current config only.
    for pattern in (
        "validation_*.png",
        "error_vs_events_*.png",
        "contour_relerr_*.png",
        "dict_vs_offdict_relerr_*.png",
    ):
        for old_plot in PLOTS_DIR.glob(pattern):
            try:
                old_plot.unlink()
            except OSError as exc:
                log.warning("Could not remove old plot %s: %s", old_plot, exc)

    # Precompute dictionary/off-dictionary split once for combined validation plots.
    dict_compare_available = False
    in_df = pd.DataFrame()
    out_df = pd.DataFrame()
    if "true_is_dictionary_entry" in val.columns:
        is_dict = val["true_is_dictionary_entry"].astype(str).str.lower().isin(
            ("true", "1", "yes")
        )
        same_paramset_as_dict = _rows_with_dictionary_parameter_set()
        overlap_rows = (~is_dict) & same_paramset_as_dict
        strict_offdict = (~is_dict) & (~same_paramset_as_dict)
        if overlap_rows.any():
            log.info(
                "Excluding %d off-dict rows with dictionary-equivalent parameter set from dict-vs-offdict overlays.",
                int(overlap_rows.sum()),
            )
        in_df = val.loc[is_dict]
        out_df = val.loc[strict_offdict]
        dict_compare_available = True

    # ── 1. Estimated vs Simulated with signed relative error ─────────
    for pname in selected_params:
        true_col = f"true_{pname}"
        est_col = f"est_{pname}"
        relerr_col = f"relerr_{pname}_pct"
        if true_col not in val.columns or est_col not in val.columns:
            continue
        t = pd.to_numeric(val[true_col], errors="coerce")
        e = pd.to_numeric(val[est_col], errors="coerce")
        r = pd.to_numeric(val.get(relerr_col), errors="coerce")
        m = t.notna() & e.notna() & r.notna() & (r.abs() <= relerr_clip)
        if m.sum() < 3:
            continue

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(13.5, 5.5),
            gridspec_kw={"width_ratios": [1.2, 1.0]},
        )

        # Left: scatter true vs estimated, coloured by signed relative error
        tv, ev, rv = t[m].values, e[m].values, r[m].values
        if relerr_plot_min < 0 < relerr_plot_max:
            norm = mcolors.TwoSlopeNorm(
                vmin=relerr_plot_min, vcenter=0.0, vmax=relerr_plot_max
            )
        else:
            norm = mcolors.Normalize(vmin=relerr_plot_min, vmax=relerr_plot_max)
        sc = ax1.scatter(
            tv,
            ev,
            c=rv,
            cmap=signed_relerr_cmap,
            s=28,
            alpha=0.92,
            norm=norm,
            edgecolors="#1A1A1A",
            linewidths=0.42,
        )
        lo = min(tv.min(), ev.min())
        hi = max(tv.max(), ev.max())
        pad = 0.02 * (hi - lo) if hi > lo else 0.01
        ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
        ax1.set_xlabel(f"True {pname}")
        ax1.set_ylabel(f"Estimated {pname}")
        ax1.set_title(f"Estimated vs True: {pname}")
        ax1.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax1, label="Rel. error [%] (clipped)", shrink=0.85)

        # Right: in-dictionary vs strict off-dictionary relative-error overlay.
        if dict_compare_available:
            in_raw = pd.to_numeric(in_df.get(relerr_col), errors="coerce").dropna()
            out_raw = pd.to_numeric(out_df.get(relerr_col), errors="coerce").dropna()
            in_raw = in_raw[in_raw.abs() <= relerr_clip]
            out_raw = out_raw[out_raw.abs() <= relerr_clip]
            in_vals = in_raw[(in_raw >= relerr_plot_min) & (in_raw <= relerr_plot_max)]
            out_vals = out_raw[(out_raw >= relerr_plot_min) & (out_raw <= relerr_plot_max)]

            density_flag = False if pname in ("eff_sim_1", "flux_cm2_min") else True
            y_label = "Count" if not density_flag else "Density"
            overlay_plotted = False
            if not in_vals.empty:
                ax2.hist(
                    in_vals,
                    bins=40,
                    histtype="step",
                    linewidth=1.8,
                    color="#2ca02c",
                    label=f"In-dict (n={len(in_vals)})",
                    density=density_flag,
                )
                overlay_plotted = True
            if not out_vals.empty:
                ax2.hist(
                    out_vals,
                    bins=40,
                    histtype="step",
                    linewidth=1.8,
                    color="#d62728",
                    label=f"Off-dict strict (n={len(out_vals)})",
                    density=density_flag,
                )
                overlay_plotted = True
            ax2.set_xlabel(f"Rel. error {pname} [%]")
            ax2.set_ylabel(y_label)
            ax2.set_title("In-dict vs off-dict strict")
            ax2.set_xlim(relerr_plot_min, relerr_plot_max)
            if pname in ("eff_sim_1", "flux_cm2_min"):
                ax2.set_yscale("log")
            if overlay_plotted:
                ax2.legend(fontsize=8)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No in-dict/off-dict\npoints in window",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                    fontsize=8,
                    color="#555555",
                )
        else:
            ax2.set_title("In-dict vs off-dict strict")
            ax2.text(
                0.5,
                0.5,
                "true_is_dictionary_entry\nnot available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=8,
                color="#555555",
            )
            ax2.set_xlabel(f"Rel. error {pname} [%]")
            ax2.set_ylabel("Count")
            ax2.set_xlim(relerr_plot_min, relerr_plot_max)

        fig.suptitle(f"Validation: {pname}", fontsize=11)
        fig.tight_layout()
        _save_figure(fig, PLOTS_DIR / f"validation_{pname}.png")
        plt.close(fig)

    # ── 2. Contour plots in flux-eff space (one per plot-relevant param) ──
    flux_true = "true_flux_cm2_min"
    contour_params = selected_params
    for pname in contour_params:
        relerr_col = f"relerr_{pname}_pct"
        if flux_true not in val.columns or relerr_col not in val.columns:
            continue
        # Pick the best available eff column for the y-axis
        eff_col = None
        for candidate in ["true_eff_sim_1", "true_eff_sim_2",
                          "true_eff_empirical_1", "true_eff_empirical_2"]:
            if candidate in val.columns and candidate != f"true_{pname}":
                eff_col = candidate
                break
        if eff_col is None:
            for c in val.columns:
                if c.startswith("true_") and c not in (flux_true, f"true_{pname}"):
                    eff_col = c
                    break
        if eff_col is None:
            continue
        eff_label = eff_col.replace("true_", "")

        fx = pd.to_numeric(val[flux_true], errors="coerce")
        ey = pd.to_numeric(val[eff_col], errors="coerce")
        er = pd.to_numeric(val[relerr_col], errors="coerce")  # signed
        m = fx.notna() & ey.notna() & er.notna() & (er.abs() <= relerr_clip)
        if m.sum() < 10:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        # Dictionary crosses BEHIND, semi-transparent
        dict_flux = pd.to_numeric(dict_df.get("flux_cm2_min"), errors="coerce")
        dict_eff = pd.to_numeric(dict_df.get(eff_label), errors="coerce")
        dm = dict_flux.notna() & dict_eff.notna()
        if dm.sum() > 0:
            ax.scatter(
                dict_flux[dm], dict_eff[dm],
                s=50, marker="x", color="grey", linewidths=0.8,
                alpha=0.25, zorder=1, label="Dictionary entries",
            )

        # Data points ON TOP, signed relative error
        er_vals = er[m].values
        if relerr_plot_min < 0 < relerr_plot_max:
            norm = mcolors.TwoSlopeNorm(
                vmin=relerr_plot_min, vcenter=0.0, vmax=relerr_plot_max
            )
        else:
            norm = mcolors.Normalize(vmin=relerr_plot_min, vmax=relerr_plot_max)
        sc = ax.scatter(
            fx[m], ey[m], c=er_vals,
            cmap=signed_relerr_cmap,
            s=28,
            alpha=0.95,
            norm=norm,
            zorder=3,
            edgecolors="#1A1A1A",
            linewidths=0.38,
        )
        fig.colorbar(sc, ax=ax, label=f"Rel. error {pname} [%] (clipped)", shrink=0.85)

        if dm.sum() > 0:
            ax.legend(fontsize=8)

        ax.set_xlabel("True flux [cm⁻² min⁻¹]")
        ax.set_ylabel(f"True {eff_label}")
        ax.set_title(f"Rel. error of {pname} in flux–efficiency plane")
        fig.tight_layout()
        _save_figure(fig, PLOTS_DIR / f"contour_relerr_{pname}.png")
        plt.close(fig)

    # ── 3. Error vs event count ──────────────────────────────────────
    if "n_events" in val.columns:
        for pname in selected_params:
            relerr_col = f"relerr_{pname}_pct"
            if relerr_col not in val.columns:
                continue
            ne = pd.to_numeric(val["n_events"], errors="coerce")
            er = pd.to_numeric(val[relerr_col], errors="coerce")
            m_all = ne.notna() & er.notna() & (ne > 0) & (er.abs() <= relerr_clip)
            m = m_all & (er >= relerr_plot_min) & (er <= relerr_plot_max)
            if m.sum() < 5:
                continue
            n_outside = int(m_all.sum() - m.sum())

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(ne[m], er[m], s=10, alpha=0.4, color="#4C78A8")
            ax.axhline(0, color="black", linewidth=0.8)

            # Theoretical ±1/√N guide converted to percent: 100/√N (matches rel. error [%])
            nv = ne[m].values.astype(float)
            n_line = np.linspace(max(1.0, nv.min()), nv.max(), 300)
            # muted grey dashed line (thin) — theoretical expectation in percent
            ax.plot(n_line, 100.0 / np.sqrt(n_line), color="#6e6e6e", linestyle="--",
                    linewidth=1.0, label=r"$100/\sqrt{N}$")
            ax.plot(n_line, -100.0 / np.sqrt(n_line), color="#6e6e6e", linestyle="--",
                    linewidth=1.0)
            ax.legend(fontsize=8)

            ax.set_xlabel("Number of events")
            ax.set_ylabel(f"Rel. error {pname} [%]")
            title = f"Rel. error vs events: {pname}"
            if n_outside > 0:
                title += f" ({n_outside} outside range omitted)"
            ax.set_title(title)
            ax.set_ylim(relerr_plot_min, relerr_plot_max)
            # Set x-axis to linear and force integer ticks
            ax.set_xscale("linear")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune=None))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"{int(x):,}" if x >= 1 else f"{x:g}")
            )
            fig.tight_layout()
            _save_figure(fig, PLOTS_DIR / f"error_vs_events_{pname}.png")
            plt.close(fig)

    # Note: dict-vs-offdict histogram overlays are now embedded in validation_*.png
    # to keep one validation figure per parameter.


if __name__ == "__main__":
    raise SystemExit(main())
