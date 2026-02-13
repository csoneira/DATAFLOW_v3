#!/usr/bin/env python3
"""STEP 2.2 — Validation of the inverse-problem solution.

Takes the estimated parameters from STEP 2.1 and the dataset from STEP 1.2
and produces validation plots and tables:

1. Estimated vs simulated parameters (scatter + identity line, with
   relative error in colour or as a secondary panel).
2. Contour plots in flux–eff space of the relative error, with dictionary
   points marked.
3. Relative error statistics saved for the next step (STEP 3.1).

Output
------
OUTPUTS/FILES/validation_results.csv  — per-point errors and estimates
OUTPUTS/FILES/validation_summary.json
OUTPUTS/PLOTS/                        — validation plots
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
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent
PIPELINE_DIR = INFERENCE_DIR.parent
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

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

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="[%(levelname)s] STEP_2.2 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.2")


def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


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
    cfg_22 = config.get("step_2_2", {})

    est_path = Path(args.estimated_csv) if args.estimated_csv else DEFAULT_ESTIMATED
    dict_path = Path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY
    data_path = Path(args.dataset_csv) if args.dataset_csv else DEFAULT_DATASET

    relerr_clip = float(cfg_22.get("relerr_threshold_pct", 50.0))

    # Plot parameters: prefer step_2_2, fallback to shared step_1_2 selection.
    plot_params = cfg_22.get("plot_parameters", None)
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
) -> None:
    """Generate validation plots."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    signed_relerr_cmap = mcolors.LinearSegmentedColormap.from_list(
        "signed_relerr_contrast",
        ["#2166AC", "#67A9CF", "#222222", "#EF8A62", "#B2182B"],
        N=256,
    )

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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Left: scatter true vs estimated, coloured by signed relative error
        tv, ev, rv = t[m].values, e[m].values, r[m].values
        if relerr_plot_min < 0 < relerr_plot_max:
            norm = mcolors.TwoSlopeNorm(
                vmin=relerr_plot_min, vcenter=0.0, vmax=relerr_plot_max
            )
        else:
            norm = mcolors.Normalize(vmin=relerr_plot_min, vmax=relerr_plot_max)
        sc = ax1.scatter(tv, ev, c=rv, cmap=signed_relerr_cmap, s=14, alpha=0.9,
                         norm=norm)
        lo = min(tv.min(), ev.min())
        hi = max(tv.max(), ev.max())
        pad = 0.02 * (hi - lo) if hi > lo else 0.01
        ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
        ax1.set_xlabel(f"True {pname}")
        ax1.set_ylabel(f"Estimated {pname}")
        ax1.set_title(f"Estimated vs True: {pname}")
        ax1.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax1, label="Rel. error [%] (clipped)", shrink=0.85)

        # Right: histogram of signed relative error (configured plotting window)
        rv_finite = rv[np.isfinite(rv)]
        rv_filtered = rv_finite[
            (rv_finite >= relerr_plot_min) & (rv_finite <= relerr_plot_max)
        ]
        n_outside = len(rv_finite) - len(rv_filtered)
        if len(rv_filtered) > 0:
            ax2.hist(rv_filtered, bins=50, color="#4C78A8", alpha=0.8, edgecolor="white")
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.axvline(np.median(rv_filtered), color="#E45756", linestyle="--",
                        label=f"median = {np.median(rv_filtered):.2f}%")
            title2 = f"Relative error distribution: {pname}"
            if n_outside > 0:
                title2 += (
                    f"  ({n_outside} outside [{relerr_plot_min:.1f}, "
                    f"{relerr_plot_max:.1f}]% omitted)"
                )
            ax2.set_xlabel(f"Rel. error {pname} [%]")
            ax2.set_ylabel("Count")
            ax2.set_title(title2)
            ax2.set_xlim(relerr_plot_min, relerr_plot_max)
            ax2.legend(fontsize=8)

        fig.suptitle(f"Validation: {pname}", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"validation_{pname}.png")
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

        # Data points ON TOP, signed relative error, RdYlBu_r
        er_vals = er[m].values
        if relerr_plot_min < 0 < relerr_plot_max:
            norm = mcolors.TwoSlopeNorm(
                vmin=relerr_plot_min, vcenter=0.0, vmax=relerr_plot_max
            )
        else:
            norm = mcolors.Normalize(vmin=relerr_plot_min, vmax=relerr_plot_max)
        sc = ax.scatter(
            fx[m], ey[m], c=er_vals,
            cmap=signed_relerr_cmap, s=18, alpha=0.95, norm=norm, zorder=3,
        )
        fig.colorbar(sc, ax=ax, label=f"Rel. error {pname} [%] (clipped)", shrink=0.85)

        if dm.sum() > 0:
            ax.legend(fontsize=8)

        ax.set_xlabel("True flux [cm⁻² min⁻¹]")
        ax.set_ylabel(f"True {eff_label}")
        ax.set_title(f"Rel. error of {pname} in flux–efficiency plane")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"contour_relerr_{pname}.png")
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

            # ±1/sqrt(N) guide
            nv = ne[m].values.astype(float)
            ev = np.abs(er[m].values.astype(float))
            med_err = float(np.nanmedian(ev))
            med_n = float(np.nanmedian(nv))
            if med_n > 0 and med_err > 0:
                scale = med_err * np.sqrt(med_n)
                n_line = np.linspace(nv.min(), nv.max(), 300)
                ax.plot(n_line, scale / np.sqrt(n_line), "r--",
                        linewidth=1.5, label=r"$\propto 1/\sqrt{N}$")
                ax.plot(n_line, -scale / np.sqrt(n_line), "r--",
                        linewidth=1.5)
                ax.legend(fontsize=8)

            ax.set_xlabel("Number of events")
            ax.set_ylabel(f"Rel. error {pname} [%]")
            title = f"Rel. error vs events: {pname}"
            if n_outside > 0:
                title += f" ({n_outside} outside range omitted)"
            ax.set_title(title)
            ax.set_ylim(relerr_plot_min, relerr_plot_max)
            ax.set_xscale("log")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"error_vs_events_{pname}.png")
            plt.close(fig)

    # ── 4. In-dictionary vs out-of-dictionary comparison ─────────────
    if "true_is_dictionary_entry" in val.columns:
        is_dict = val["true_is_dictionary_entry"].astype(str).str.lower().isin(
            ("true", "1", "yes")
        )
        same_paramset_as_dict = _rows_with_dictionary_parameter_set()
        overlap_rows = (~is_dict) & same_paramset_as_dict
        strict_offdict = (~is_dict) & (~same_paramset_as_dict)
        if overlap_rows.any():
            log.info(
                "Excluding %d off-dict rows with dictionary-equivalent parameter set from dict-vs-offdict plots.",
                int(overlap_rows.sum()),
            )

        in_df = val.loc[is_dict]
        out_df = val.loc[strict_offdict]

        dict_plot_params = selected_params
        for pname in dict_plot_params:
            relerr_col = f"relerr_{pname}_pct"
            if relerr_col not in val.columns:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            in_raw = pd.to_numeric(in_df[relerr_col], errors="coerce").dropna()
            out_raw = pd.to_numeric(out_df[relerr_col], errors="coerce").dropna()
            in_raw = in_raw[in_raw.abs() <= relerr_clip]
            out_raw = out_raw[out_raw.abs() <= relerr_clip]
            in_vals = in_raw[
                (in_raw >= relerr_plot_min) & (in_raw <= relerr_plot_max)
            ]
            out_vals = out_raw[
                (out_raw >= relerr_plot_min) & (out_raw <= relerr_plot_max)
            ]
            if not in_vals.empty:
                ax.hist(in_vals, bins=40, alpha=0.6, color="#2ca02c",
                        label=f"In-dict (n={len(in_vals)})", density=True)
            if not out_vals.empty:
                ax.hist(out_vals, bins=40, alpha=0.6, color="#d62728",
                        label=f"Off-dict strict (n={len(out_vals)})", density=True)
            ax.set_xlabel(f"Rel. error {pname} [%]")
            ax.set_ylabel("Density")
            ax.set_title(
                f"In-dict vs off-dict strict: rel. error {pname} "
                f"([{relerr_plot_min:.1f}, {relerr_plot_max:.1f}]%)"
            )
            ax.set_xlim(relerr_plot_min, relerr_plot_max)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"dict_vs_offdict_relerr_{pname}.png")
            plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
