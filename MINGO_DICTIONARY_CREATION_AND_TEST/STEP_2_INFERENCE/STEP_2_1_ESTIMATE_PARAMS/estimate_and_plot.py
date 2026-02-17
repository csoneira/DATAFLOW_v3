#!/usr/bin/env python3
"""STEP 2.1 — Solution to the inverse problem.

Uses the self-contained `estimate_parameters` module to reconstruct
physical parameters (flux, cos_n, efficiencies) for each dataset entry
by matching its rate fingerprint against the dictionary.

Produces the estimation results CSV and diagnostic plots.

Output
------
OUTPUTS/FILES/estimated_params.csv   — estimated parameters for each data point
OUTPUTS/FILES/estimation_summary.json
OUTPUTS/PLOTS/                       — diagnostic plots
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
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent           # STEP_2_INFERENCE
PIPELINE_DIR = INFERENCE_DIR.parent       # INFERENCE_DICTIONARY_VALIDATION
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

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

# Import the self-contained estimation module
sys.path.insert(0, str(INFERENCE_DIR))
from estimate_parameters import estimate_parameters, DISTANCE_FNS  # noqa: E402

logging.basicConfig(
    format="[%(levelname)s] STEP_2.1 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.1")


def _load_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2.1: Estimate parameters using dictionary matching."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--dataset-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_21 = config.get("step_2_1", {})

    dict_path = Path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY
    data_path = Path(args.dataset_csv) if args.dataset_csv else DEFAULT_DATASET

    feature_columns = cfg_21.get("feature_columns", "auto")
    distance_metric = cfg_21.get("distance_metric", "l2_zscore")
    interpolation_k_cfg = cfg_21.get("interpolation_k", 5)
    interpolation_k = None if interpolation_k_cfg is None else int(interpolation_k_cfg)
    include_global_rate = cfg_21.get("include_global_rate", True)
    global_rate_col = cfg_21.get("global_rate_col", "events_per_second_global_rate")
    plot_params = config.get("step_1_2", {}).get("plot_parameters", None)

    if not dict_path.exists():
        log.error("Dictionary CSV not found: %s", dict_path)
        return 1
    if not data_path.exists():
        log.error("Dataset CSV not found: %s", data_path)
        return 1

    log.info("Dictionary: %s", dict_path)
    log.info("Dataset:    %s", data_path)
    log.info("Metric:     %s", distance_metric)
    log.info(
        "K:          %s",
        "all dictionary candidates" if interpolation_k is None else str(interpolation_k),
    )

    # ── Run estimation ───────────────────────────────────────────────
    result_df = estimate_parameters(
        dictionary_path=str(dict_path),
        dataset_path=str(data_path),
        feature_columns=feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
        exclude_same_file=True,
    )

    # ── Merge with dataset to have truth values alongside ────────────
    data_df = pd.read_csv(data_path, low_memory=False)

    # Attach truth columns needed for validation
    truth_cols = ["flux_cm2_min", "cos_n",
                  "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
                  "n_events", "is_dictionary_entry"]
    for col in truth_cols:
        if col in data_df.columns:
            result_df[f"true_{col}"] = data_df[col].values[:len(result_df)]

    # ── Save ─────────────────────────────────────────────────────────
    out_path = FILES_DIR / "estimated_params.csv"
    result_df.to_csv(out_path, index=False)
    log.info("Wrote estimated params: %s (%d rows)", out_path, len(result_df))

    n_ok = result_df["best_distance"].notna().sum()
    n_fail = result_df["best_distance"].isna().sum()

    summary = {
        "dictionary": str(dict_path),
        "dataset": str(data_path),
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "feature_columns": feature_columns if isinstance(feature_columns, list) else "auto",
        "total_points": len(result_df),
        "successful_estimates": int(n_ok),
        "failed_estimates": int(n_fail),
    }
    with open(FILES_DIR / "estimation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostic plots ─────────────────────────────────────────────
    _make_plots(
        result_df=result_df,
        data_df=data_df,
        plot_params=plot_params,
        dict_path=dict_path,
        cfg_21=cfg_21,
    )

    log.info("Done.")
    return 0


def _auto_feature_columns(
    df: pd.DataFrame,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
) -> list[str]:
    cols = sorted([
        c for c in df.columns
        if c.startswith("raw_tt_") and c.endswith("_rate_hz")
    ])
    if not cols:
        cols = sorted([c for c in df.columns if c.endswith("_rate_hz")])
    if include_global_rate and global_rate_col in df.columns and global_rate_col not in cols:
        cols.append(global_rate_col)
    return cols


def _l2_distances(sample_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    valid = np.isfinite(candidates) & np.isfinite(sample_vec[np.newaxis, :])
    n_valid = valid.sum(axis=1)
    diff = np.where(valid, candidates - sample_vec[np.newaxis, :], 0.0)
    d = np.sqrt(np.sum(diff * diff, axis=1))
    d[n_valid < 2] = np.nan
    return d


def _make_random_showcase_l2_contour(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    dict_path: Path,
    cfg_21: dict,
) -> None:
    if not dict_path.exists():
        return

    required = ["dataset_index", "true_flux_cm2_min", "est_flux_cm2_min"]
    for col in required:
        if col not in result_df.columns:
            return

    eff_true_col = "true_eff_sim_1" if "true_eff_sim_1" in result_df.columns else None
    eff_est_col = "est_eff_sim_1" if "est_eff_sim_1" in result_df.columns else None
    if eff_true_col is None or eff_est_col is None:
        return

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & pd.to_numeric(result_df["true_flux_cm2_min"], errors="coerce").notna()
        & pd.to_numeric(result_df["est_flux_cm2_min"], errors="coerce").notna()
        & pd.to_numeric(result_df[eff_true_col], errors="coerce").notna()
        & pd.to_numeric(result_df[eff_est_col], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return

    rng = np.random.default_rng(int(cfg_21.get("showcase_seed", 42)))
    valid_indices = result_df.index[valid_mask].to_numpy()
    chosen_idx = int(rng.choice(valid_indices))
    row = result_df.loc[chosen_idx]
    ds_idx = int(pd.to_numeric(pd.Series([row["dataset_index"]]), errors="coerce").iloc[0])
    if ds_idx < 0 or ds_idx >= len(data_df):
        return

    dict_df = pd.read_csv(dict_path, low_memory=False)

    include_global_rate = bool(cfg_21.get("include_global_rate", True))
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    feature_cfg = cfg_21.get("feature_columns", "auto")
    if isinstance(feature_cfg, str) and feature_cfg == "auto":
        feature_cols = sorted(
            set(_auto_feature_columns(dict_df, include_global_rate, global_rate_col))
            & set(_auto_feature_columns(data_df, include_global_rate, global_rate_col))
        )
    else:
        feature_cols = [
            str(c) for c in list(feature_cfg)
            if str(c) in dict_df.columns and str(c) in data_df.columns
        ]
    if not feature_cols:
        return

    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    sample_feat = pd.to_numeric(data_df.loc[ds_idx, feature_cols], errors="coerce").to_numpy(dtype=float)

    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    # short token for labels/filenames (e.g. 'l2' from 'l2_zscore', 'chi2' from 'chi2')
    metric_short = distance_metric.split("_")[0]
    metric_label = "L2" if metric_short == "l2" else metric_short

    if distance_metric == "l2_zscore":
        means = dict_feat.mean(axis=0, skipna=True)
        stds = dict_feat.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_mat = ((dict_feat - means) / stds).to_numpy(dtype=float)
        sample_vec = ((sample_feat - means.to_numpy(dtype=float)) / stds.to_numpy(dtype=float))
    else:
        dict_mat = dict_feat.to_numpy(dtype=float)
        sample_vec = sample_feat

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"] if c in dict_df.columns and c in data_df.columns]
    if z_cols:
        z_tol = float(cfg_21.get("z_tol", 1e-6))
        dict_z = dict_df[z_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sample_z = pd.to_numeric(data_df.loc[ds_idx, z_cols], errors="coerce").to_numpy(dtype=float)
        z_mask = np.all(np.abs(dict_z - sample_z[np.newaxis, :]) <= z_tol, axis=1)
    else:
        z_mask = np.ones(len(dict_df), dtype=bool)

    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    if join_col is not None:
        sample_id = str(data_df.loc[ds_idx, join_col])
        z_mask &= (dict_df[join_col].astype(str).to_numpy() != sample_id)

    cand_df = dict_df.loc[z_mask].copy()
    if cand_df.empty:
        return

    cand_mat = dict_mat[z_mask]

    # Use the same distance function used by the estimator so the plotted
    # quantity matches the reported `best_distance` (e.g. chi2, l2_zscore).
    dist_fn = DISTANCE_FNS.get(distance_metric, DISTANCE_FNS.get(metric_short, None))
    if dist_fn is None:
        # Fallback to the original L2 helper if nothing found
        z_vals = _l2_distances(sample_vec, cand_mat)
        cand_df["distance_value"] = z_vals
    else:
        # compute per-candidate scalar distances using the estimator's funcs
        z_list = [dist_fn(sample_vec, cand_mat[i]) for i in range(cand_mat.shape[0])]
        cand_df["distance_value"] = np.array(z_list, dtype=float)

    cand_df["flux_for_plot"] = pd.to_numeric(cand_df.get("flux_cm2_min"), errors="coerce")
    cand_df["eff_for_plot"] = pd.to_numeric(cand_df.get("eff_sim_1"), errors="coerce")
    cand_df = cand_df.dropna(subset=["flux_for_plot", "eff_for_plot", "distance_value"])
    if len(cand_df) < 3:
        return

    x = cand_df["flux_for_plot"].to_numpy(dtype=float)
    y = cand_df["eff_for_plot"].to_numpy(dtype=float)
    z = cand_df["distance_value"].to_numpy(dtype=float)
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
        z_min, z_max = 0.0, 1.0

    true_flux = float(pd.to_numeric(pd.Series([row["true_flux_cm2_min"]]), errors="coerce").iloc[0])
    est_flux = float(pd.to_numeric(pd.Series([row["est_flux_cm2_min"]]), errors="coerce").iloc[0])
    true_eff = float(pd.to_numeric(pd.Series([row[eff_true_col]]), errors="coerce").iloc[0])
    est_eff = float(pd.to_numeric(pd.Series([row[eff_est_col]]), errors="coerce").iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    contour_ok = False
    try:
        tri = mtri.Triangulation(x, y)
        ctf = ax.tricontourf(
            tri, z, levels=24, cmap="viridis_r", alpha=0.55, vmin=z_min, vmax=z_max
        )
        ax.tricontour(
            tri, z, levels=12, colors="white", linewidths=0.35, alpha=0.30
        )
        contour_ok = True
    except Exception:
        contour_ok = False

    sc = ax.scatter(
        x, y, c=z, cmap="viridis_r", vmin=z_min, vmax=z_max,
        s=36, marker="o", alpha=0.93,
        edgecolors=(1.0, 1.0, 1.0, 0.75), linewidths=0.35, zorder=4
    )
    cb = fig.colorbar(ctf if contour_ok else sc, ax=ax, shrink=0.88)
    cb.set_label(f"{metric_label} distance in feature space")

    ax.scatter(
        [true_flux], [true_eff], s=170, marker="*", color="#E45756",
        edgecolors="black", linewidths=0.6, zorder=6, label="True point"
    )
    ax.scatter(
        [est_flux], [est_eff], s=140, marker="X", color="#F58518",
        edgecolors="black", linewidths=0.6, zorder=6, label="Estimated point"
    )

    ax.set_xlabel("Flux [cm⁻² min⁻¹]")
    ax.set_ylabel("Efficiency (eff_sim_1)")
    ax.set_title(f"Random showcase {metric_label} distance map (dataset_index={ds_idx}, candidates={len(cand_df)})")
    ax.legend(loc="best", fontsize=8)

    note = (
        f"true: flux={true_flux:.4g}, eff={true_eff:.4g}\n"
        f"est:  flux={est_flux:.4g}, eff={est_eff:.4g}\n"
        f"best_distance={float(row['best_distance']):.4g}"
    )
    ax.text(
        0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85)
    )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "random_showcase_distance_contour_flux_eff.png")
    plt.close(fig)


def _make_plots(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    plot_params=None,
    dict_path: Path | None = None,
    cfg_21: dict | None = None,
) -> None:
    """Quick diagnostic plots for the estimation step."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ── 1. Distance diagnostics (distribution + method relevance) ───
    distances = pd.to_numeric(result_df.get("best_distance"), errors="coerce").dropna()
    if not distances.empty:
        q1 = float(distances.quantile(0.25))
        q3 = float(distances.quantile(0.75))
        iqr = q3 - q1
        upper_fence = float(q3 + 1.5 * iqr) if np.isfinite(iqr) else float(distances.max())
        inlier_mask = distances <= upper_fence
        inliers = distances[inlier_mask]
        n_outliers = int((~inlier_mask).sum())

        q50 = float(distances.quantile(0.50))
        q90 = float(distances.quantile(0.90))
        q95 = float(distances.quantile(0.95))

        fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_cdf = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 0])
        ax_oper = fig.add_subplot(gs[1, 1])

        # 1A) Core histogram (IQR-clipped) with robust quantiles
        hist_values = inliers if len(inliers) >= 5 else distances
        ax_hist.hist(hist_values, bins=45, color="#4C78A8", alpha=0.82, edgecolor="white")
        ax_hist.axvline(q50, color="#E45756", linestyle="--", linewidth=1.6, label=f"p50 = {q50:.4g}")
        ax_hist.axvline(q90, color="#F58518", linestyle="-.", linewidth=1.4, label=f"p90 = {q90:.4g}")
        ax_hist.axvline(q95, color="#72B7B2", linestyle=":", linewidth=1.6, label=f"p95 = {q95:.4g}")
        if n_outliers:
            ax_hist.axvline(
                upper_fence, color="#B279A2", linestyle="-", linewidth=1.1,
                label=f"IQR upper fence = {upper_fence:.4g}",
            )
        ax_hist.set_xlabel("Best distance")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Core distance density (IQR-clipped)")
        ax_hist.legend(fontsize=7.5, loc="upper right")
        ax_hist.text(
            0.02,
            0.98,
            (
                f"N={len(distances)} | outliers={n_outliers} "
                f"({(100.0 * n_outliers / len(distances)):.1f}%)\n"
                f"median={q50:.3g}, p90={q90:.3g}, p95={q95:.3g}"
            ),
            transform=ax_hist.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
        )

        # 1B) Coverage view: fraction of rows below distance threshold
        d_sorted = np.sort(distances.to_numpy(dtype=float))
        cdf_y = np.arange(1, len(d_sorted) + 1, dtype=float) / len(d_sorted)
        ax_cdf.plot(d_sorted, cdf_y, color="#54A24B", linewidth=1.8)
        for value, color, label in [
            (q50, "#E45756", "p50"),
            (q90, "#F58518", "p90"),
            (q95, "#72B7B2", "p95"),
        ]:
            ax_cdf.axvline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.8, label=label)
        ax_cdf.set_xlabel("Best distance threshold")
        ax_cdf.set_ylabel("Fraction with best_distance <= threshold")
        ax_cdf.set_ylim(0.0, 1.02)
        ax_cdf.set_title("Coverage curve (all rows)")
        ax_cdf.legend(fontsize=7.5, loc="lower right")
        # Keep the CDF readable when very large outliers exist.
        cdf_xmax = float(distances.quantile(0.995))
        if np.isfinite(cdf_xmax) and cdf_xmax > 0 and distances.max() > 1.15 * cdf_xmax:
            ax_cdf.set_xlim(0.0, cdf_xmax)
            ax_cdf.text(
                0.02,
                0.03,
                f"Zoomed to 99.5% (max={float(distances.max()):.3g})",
                transform=ax_cdf.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="0.35",
            )

        # 1C) Distance-vs-error calibration: does distance track inference quality?
        selected_params: list[str] = []
        if isinstance(plot_params, (list, tuple, set)):
            selected_params = [
                str(p) for p in plot_params
                if f"true_{p}" in result_df.columns and f"est_{p}" in result_df.columns
            ]
        if not selected_params:
            for pname in ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4", "cos_n"]:
                if f"true_{pname}" in result_df.columns and f"est_{pname}" in result_df.columns:
                    selected_params.append(pname)

        relerr_cols = []
        for pname in selected_params:
            t = pd.to_numeric(result_df[f"true_{pname}"], errors="coerce")
            e = pd.to_numeric(result_df[f"est_{pname}"], errors="coerce")
            denom = np.maximum(np.abs(t), 1e-9)
            relerr_cols.append((((e - t).abs() / denom) * 100.0).rename(pname))

        if relerr_cols:
            relerr_df = pd.concat(relerr_cols, axis=1)
            row_relerr = relerr_df.median(axis=1, skipna=True)
            eval_df = pd.DataFrame({
                "distance": pd.to_numeric(result_df["best_distance"], errors="coerce"),
                "agg_relerr_pct": row_relerr,
            }).dropna()
        else:
            eval_df = pd.DataFrame(columns=["distance", "agg_relerr_pct"])

        if len(eval_df) >= 3:
            ax_err.scatter(
                eval_df["distance"], eval_df["agg_relerr_pct"],
                s=15, alpha=0.35, color="#72B7B2", edgecolors="none",
            )
            if len(eval_df) >= 20 and eval_df["distance"].nunique() >= 6:
                q_edges = np.unique(np.quantile(eval_df["distance"], np.linspace(0.0, 1.0, 9)))
                if len(q_edges) >= 3:
                    dist_bins = pd.cut(eval_df["distance"], bins=q_edges, include_lowest=True, duplicates="drop")
                    trend = (
                        eval_df.assign(dist_bin=dist_bins)
                        .groupby("dist_bin", observed=True)
                        .agg(
                            distance_mid=("distance", "median"),
                            relerr_median=("agg_relerr_pct", "median"),
                        )
                        .dropna()
                    )
                    if not trend.empty:
                        ax_err.plot(
                            trend["distance_mid"], trend["relerr_median"],
                            color="#E45756", linewidth=1.7, marker="o",
                            label="Median |rel.err| across distance quantiles",
                        )

            pearson = float(eval_df["distance"].corr(eval_df["agg_relerr_pct"], method="pearson"))
            spearman = float(eval_df["distance"].corr(eval_df["agg_relerr_pct"], method="spearman"))
            ptxt = f"{pearson:.2f}" if np.isfinite(pearson) else "nan"
            stxt = f"{spearman:.2f}" if np.isfinite(spearman) else "nan"
            shown_params = ", ".join(selected_params[:3]) + ("..." if len(selected_params) > 3 else "")
            ax_err.set_title("Distance vs estimation error")
            ax_err.set_xlabel("Best distance")
            ax_err.set_ylabel("Median |relative error| [%]")
            ax_err.text(
                0.02,
                0.98,
                f"Params: {shown_params}\nPearson={ptxt}, Spearman={stxt}",
                transform=ax_err.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
            )
            y_hi = float(eval_df["agg_relerr_pct"].quantile(0.99))
            if np.isfinite(y_hi) and y_hi > 0:
                ax_err.set_ylim(0.0, max(0.5, 1.15 * y_hi))
            x_hi = float(eval_df["distance"].quantile(0.99))
            if np.isfinite(x_hi) and x_hi > 0 and eval_df["distance"].max() > 1.15 * x_hi:
                ax_err.set_xlim(0.0, x_hi)
                ax_err.text(
                    0.98,
                    0.03,
                    f"Zoomed to p99 (max={float(eval_df['distance'].max()):.3g})",
                    transform=ax_err.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )
            if ax_err.get_legend_handles_labels()[0]:
                ax_err.legend(fontsize=7.5, loc="upper right")
        else:
            ax_err.text(
                0.5, 0.5,
                "Not enough true/estimated\nparameter overlap for\nerror-calibration panel",
                transform=ax_err.transAxes,
                ha="center", va="center", fontsize=9,
            )
            ax_err.set_title("Distance vs estimation error")
            ax_err.set_xlabel("Best distance")
            ax_err.set_ylabel("Median |relative error| [%]")

        # 1D) Distance operating curve: threshold trade-off for coverage vs quality
        if len(eval_df) >= 20 and eval_df["distance"].nunique() >= 8:
            thr = np.unique(np.quantile(eval_df["distance"], np.linspace(0.05, 0.95, 15)))
            if len(thr) >= 3:
                coverage_pct = []
                med_relerr = []
                p90_relerr = []
                n_eval = float(len(eval_df))
                for tval in thr:
                    subset = eval_df[eval_df["distance"] <= tval]
                    if subset.empty:
                        coverage_pct.append(np.nan)
                        med_relerr.append(np.nan)
                        p90_relerr.append(np.nan)
                    else:
                        coverage_pct.append(100.0 * len(subset) / n_eval)
                        med_relerr.append(float(subset["agg_relerr_pct"].median()))
                        p90_relerr.append(float(subset["agg_relerr_pct"].quantile(0.90)))

                ax_oper.plot(
                    thr, coverage_pct, color="#54A24B", linewidth=1.8, marker="o",
                    markersize=3.2, label="Coverage retained [%]",
                )
                ax_oper.set_xlabel("Distance threshold")
                ax_oper.set_ylabel("Coverage retained [%]", color="#2F6B2D")
                ax_oper.tick_params(axis="y", labelcolor="#2F6B2D")
                ax_oper.set_ylim(0.0, 101.0)

                ax_err2 = ax_oper.twinx()
                ax_err2.plot(
                    thr, med_relerr, color="#E45756", linewidth=1.6, marker="s",
                    markersize=3.0, label="Median |rel.err| [%]",
                )
                ax_err2.plot(
                    thr, p90_relerr, color="#F58518", linewidth=1.3, linestyle="--",
                    label="p90 |rel.err| [%]",
                )
                ax_err2.set_ylabel("Error among retained rows [%]", color="#A94D00")
                ax_err2.tick_params(axis="y", labelcolor="#A94D00")

                star_thr = q90
                star_subset = eval_df[eval_df["distance"] <= star_thr]
                if len(star_subset) > 0:
                    star_cov = 100.0 * len(star_subset) / len(eval_df)
                    star_med = float(star_subset["agg_relerr_pct"].median())
                    ax_oper.axvline(star_thr, color="0.45", linestyle=":", linewidth=1.0)
                    ax_oper.text(
                        0.02,
                        0.98,
                        (
                            f"At p90 threshold ({star_thr:.3g}):\n"
                            f"coverage={star_cov:.1f}%, median err={star_med:.2f}%"
                        ),
                        transform=ax_oper.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
                    )

                ax_oper.set_title("Operating curve: threshold trade-off")
                h1, l1 = ax_oper.get_legend_handles_labels()
                h2, l2 = ax_err2.get_legend_handles_labels()
                ax_oper.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
            else:
                ax_oper.text(
                    0.5, 0.5, "Insufficient distance spread\nfor operating-curve panel",
                    transform=ax_oper.transAxes, ha="center", va="center", fontsize=9,
                )
                ax_oper.set_title("Operating curve: threshold trade-off")
                ax_oper.set_xlabel("Distance threshold")
                ax_oper.set_ylabel("Coverage retained [%]")
        else:
            ax_oper.text(
                0.5, 0.5, "Not enough rows with error estimates\nfor operating-curve panel",
                transform=ax_oper.transAxes, ha="center", va="center", fontsize=9,
            )
            ax_oper.set_title("Operating curve: threshold trade-off")
            ax_oper.set_xlabel("Distance threshold")
            ax_oper.set_ylabel("Coverage retained [%]")

        metric = str((cfg_21 or {}).get("distance_metric", "unknown"))
        k_cfg = (cfg_21 or {}).get("interpolation_k", None)
        k_label = "all" if k_cfg is None else str(k_cfg)
        fig.suptitle(
            f"Best-match distance diagnostics (metric={metric}, IDW K={k_label})",
            fontsize=11,
        )
        fig.savefig(PLOTS_DIR / "distance_distribution.png")
        plt.close(fig)

    # ── 2. True vs estimated scatter for available params ────────────
    # Build all possible pairs, then filter by plot_parameters if set
    all_param_pairs = []
    for col in result_df.columns:
        if col.startswith("est_"):
            pname = col[4:]  # strip "est_"
            true_col = f"true_{pname}"
            if true_col in result_df.columns:
                all_param_pairs.append((true_col, col, pname))
    if plot_params:
        all_param_pairs = [(t, e, l) for t, e, l in all_param_pairs
                           if l in plot_params]
    valid_pairs = all_param_pairs

    if valid_pairs:
        n_p = len(valid_pairs)
        fig, axes = plt.subplots(1, n_p, figsize=(5 * n_p, 5))
        if n_p == 1:
            axes = [axes]
        for ax, (true_col, est_col, label) in zip(axes, valid_pairs):
            t = pd.to_numeric(result_df[true_col], errors="coerce")
            e = pd.to_numeric(result_df[est_col], errors="coerce")
            m = t.notna() & e.notna()
            if m.sum() > 0:
                ax.scatter(t[m], e[m], s=12, alpha=0.5, color="#F58518")
                lo = min(t[m].min(), e[m].min())
                hi = max(t[m].max(), e[m].max())
                pad = 0.02 * (hi - lo) if hi > lo else 0.01
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        "k--", linewidth=1)
            ax.set_xlabel(f"True {label}")
            ax.set_ylabel(f"Estimated {label}")
            ax.set_title(f"True vs Est: {label}")
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"Parameter estimation: true vs estimated (metric={metric})", fontsize=11, y=0.98)
        # Leave extra room under the suptitle so subplot titles don't collide
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(PLOTS_DIR / "true_vs_estimated.png")
        plt.close(fig)

    # ── 3. Random showcase with L2 contour in flux-eff space ────────
    if dict_path is not None:
        _make_random_showcase_l2_contour(
            result_df=result_df,
            data_df=data_df,
            dict_path=dict_path,
            cfg_21=cfg_21 or {},
        )



if __name__ == "__main__":
    raise SystemExit(main())
