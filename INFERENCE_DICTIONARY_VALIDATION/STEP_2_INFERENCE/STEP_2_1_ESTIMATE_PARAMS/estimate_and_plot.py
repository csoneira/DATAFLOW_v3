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
from estimate_parameters import estimate_parameters  # noqa: E402

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
    interpolation_k = int(cfg_21.get("interpolation_k", 5))
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
    log.info("K:          %d", interpolation_k)

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
    l2 = _l2_distances(sample_vec, cand_mat)
    cand_df["l2_distance"] = l2
    cand_df["flux_for_plot"] = pd.to_numeric(cand_df.get("flux_cm2_min"), errors="coerce")
    cand_df["eff_for_plot"] = pd.to_numeric(cand_df.get("eff_sim_1"), errors="coerce")
    cand_df = cand_df.dropna(subset=["flux_for_plot", "eff_for_plot", "l2_distance"])
    if len(cand_df) < 3:
        return

    x = cand_df["flux_for_plot"].to_numpy(dtype=float)
    y = cand_df["eff_for_plot"].to_numpy(dtype=float)
    z = cand_df["l2_distance"].to_numpy(dtype=float)
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
    cb.set_label("L2 distance in feature space")

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
    ax.set_title(f"Random showcase L2 map (dataset_index={ds_idx}, candidates={len(cand_df)})")
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
    fig.savefig(PLOTS_DIR / "random_showcase_l2_contour_flux_eff.png")
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

    # ── 1. Distance distribution (IQR-clipped) ─────────────────────
    distances = result_df["best_distance"].dropna()
    if not distances.empty:
        q1, q3 = distances.quantile(0.25), distances.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        inliers = distances[distances <= upper_fence]
        n_outliers = len(distances) - len(inliers)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(inliers, bins=50, color="#4C78A8", alpha=0.8, edgecolor="white")
        ax.axvline(inliers.median(), color="#E45756", linestyle="--",
                   label=f"median = {inliers.median():.4g}")
        ax.set_xlabel("Best distance")
        ax.set_ylabel("Count")
        title = "Distribution of best-match distances"
        if n_outliers:
            title += f"  ({n_outliers} outlier{'s' if n_outliers > 1 else ''}"
            title += f" above {upper_fence:.3g} removed)"
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
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
        fig.suptitle("Parameter estimation: true vs estimated", fontsize=11)
        fig.tight_layout()
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
