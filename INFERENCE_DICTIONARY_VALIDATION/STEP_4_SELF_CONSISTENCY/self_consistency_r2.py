#!/usr/bin/env python3
"""STEP_3: Self-consistency analysis — compare rate fingerprints to recover
the (flux, efficiency) parameters that generated a data file.

Approach
--------
Each simulated file produces a unique "fingerprint" of coincidence rates
(raw_tt_1234, raw_tt_234, raw_tt_134, raw_tt_124, raw_tt_123, …) that
depends on flux, cos_n, z-positions and detector efficiencies.  By comparing
the rate fingerprint of a test file against all candidates sharing the same
z-plane geometry, the candidate with the closest fingerprint should have the
most similar physical parameters.

Metric modes
~~~~~~~~~~~~
* ``raw_tt_rates`` (DEFAULT, recommended) — uses all raw trigger-topology
  rate columns (+ optional global rate) as features.  Apples-to-apples
  comparison of actual observables.
* ``eff_global`` — uses parsed eff_1–4 + global rate from the reference CSV.
* ``dict_eff_global`` — uses dictionary eff_1–4 + global rate.  The sample
  vector uses empirical efficiency estimates from count ratios (**unreliable
  for asymmetric geometries — use at own risk**).

Execution modes
~~~~~~~~~~~~~~~
* ``--single`` (legacy behavior) — evaluate one sample and produce detailed
  candidate-level diagnostics.
* ``--all`` — evaluate all samples, store true vs estimated parameter pairs,
  and produce aggregate error-vs-sample-size diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

# Optional smooth interpolation
try:
    from scipy.interpolate import griddata as _griddata
except ImportError:
    _griddata = None

# Shared utilities --------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msv_utils import (  # noqa: E402
    LOWER_IS_BETTER,
    SCORE_FNS,
    as_float,
    build_empirical_eff,
    build_uncertainty_table,
    compute_efficiency,
    extract_eff_columns,
    find_join_col,
    load_config,
    maybe_log_x,
    resolve_param,
    safe_rel_error_pct,
    setup_logger,
)

log = setup_logger("STEP_4")

DEFAULT_REF = REPO_ROOT / "STEP_3_RELATIVE_ERROR" / "output" / "filtered_reference.csv"
DEFAULT_OUT = STEP_DIR / "output"
DEFAULT_DICT = (
    REPO_ROOT / "STEP_1_BUILD_DICTIONARY" / "output" / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_CONFIG = STEP_DIR / "config.json"


# ---------------------------------------------------------------------------
# Step-3 specific helpers
# ---------------------------------------------------------------------------

def _select_metric_cols(df: pd.DataFrame, suffix: str, prefix: str | None) -> list[str]:
    cols = [col for col in df.columns if col.endswith(suffix)]
    if prefix:
        cols = [col for col in cols if col.startswith(prefix)]
    return sorted(cols)



def _scale_metrics(
    df: pd.DataFrame, method: str
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    if method == "zscore":
        means = df.mean(axis=0, skipna=True)
        stds = df.std(axis=0, skipna=True).replace({0.0: np.nan})
        return (df - means) / stds, means, stds
    return df, None, None


def _scale_vector(
    vec: pd.Series,
    method: str,
    means: pd.Series | None,
    stds: pd.Series | None,
) -> np.ndarray:
    if method == "zscore" and means is not None and stds is not None:
        return ((vec - means) / stds).to_numpy(dtype=float)
    return vec.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_contour(
    candidates: pd.DataFrame,
    sample_flux: float,
    sample_eff: float,
    best_flux,
    best_eff,
    score_metric: str,
    metric_mode: str,
    best_score,
    truth_rank: int,
    n_candidates: int,
    plot_path: Path,
    events_label: str,
    sample_label: str,
    top_n: int = 0,
) -> None:
    """Create a contour/heatmap of scores in (flux, efficiency) space.

    Also annotates top-N ranked candidates inline (replaces the old
    standalone top_n_param_space plot).
    """
    x = candidates["dict_flux_cm2_min"].to_numpy(dtype=float)
    y = candidates["dict_eff_1"].to_numpy(dtype=float)
    z = candidates["score_value"].to_numpy(dtype=float)

    # Use log scale for distance-like metrics (avoid log(0))
    lower_is_better = score_metric in LOWER_IS_BETTER
    if lower_is_better:
        z_display = np.log10(np.maximum(z, 1e-12))
        color_label = "log10(" + score_metric + ")"
        cmap = "viridis_r"
    else:
        z_display = z
        color_label = score_metric.upper()
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- smooth background contour (if scipy available & enough points) ---
    if _griddata is not None and len(x) >= 10:
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        Xi, Yi = np.meshgrid(xi, yi)
        try:
            Zi = _griddata((x, y), z_display, (Xi, Yi), method="cubic")
            ax.contourf(Xi, Yi, Zi, levels=20, cmap=cmap, alpha=0.45)
        except Exception:
            pass  # fall back to scatter only
    elif len(x) >= 3:
        try:
            triang = mtri.Triangulation(x, y)
            ax.tricontourf(triang, z_display, levels=12, cmap=cmap, alpha=0.45)
        except Exception:
            pass

    # --- scatter all candidates ---
    sc = ax.scatter(
        x, y,
        c=z_display,
        cmap=cmap,
        s=50,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, label=color_label, shrink=0.85)

    # --- mark true sample ---
    if np.isfinite(sample_flux) and np.isfinite(sample_eff):
        ax.scatter(
            [sample_flux], [sample_eff],
            s=220, marker="*", color="#E45756",
            edgecolors="black", linewidths=0.8, zorder=6,
            label="True sample",
        )

    # --- mark best candidate ---
    if best_flux is not None and best_eff is not None:
        ax.scatter(
            [best_flux], [best_eff],
            s=180, marker="X", color="#F58518",
            edgecolors="black", linewidths=0.8, zorder=6,
            label="Best candidate",
        )

    # --- annotations ---
    best_lbl = "n/a" if best_score is None else f"{score_metric}={best_score:.4g}"
    tr_str = f"{truth_rank}/{n_candidates}" if truth_rank > 0 else "self"
    ax.set_title(
        f"Self-consistency score map ({metric_mode}, {score_metric})\n"
        f"Sample={sample_label}  events={events_label}  best {best_lbl}  "
        f"truth_rank={tr_str}",
        fontsize=11,
    )
    ax.set_xlabel("Flux [cm^-2 min^-1]")
    ax.set_ylabel("Efficiency (eff_1)")

    if best_flux is not None and best_eff is not None:
        info_text = (
            f"sample flux = {sample_flux:.4g}\n"
            f"sample eff = {sample_eff:.4g}\n"
            f"delta_flux = {best_flux - sample_flux:.4g}\n"
            f"delta_eff = {best_eff - sample_eff:.4g}"
        )
    else:
        info_text = f"sample flux = {sample_flux:.4g}\nsample eff = {sample_eff:.4g}"
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#D9D9D9"),
    )
    # --- top-N rank annotations (integrated from former top_n_param_space) ---
    if top_n > 0:
        ascending = score_metric in LOWER_IS_BETTER
        df_sorted = candidates.dropna(subset=["dict_flux_cm2_min", "dict_eff_1", "score_value"]) \
            .sort_values("score_value", ascending=ascending)
        top = df_sorted.head(top_n)
        rank_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, min(top_n, len(top))))
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            if rank == 1:
                continue  # best candidate already marked with X
            ax.annotate(
                f"#{rank}",
                (row["dict_flux_cm2_min"], row["dict_eff_1"]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7, fontweight="bold",
                color=rank_colors[rank - 1],
            )

    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _plot_score_vs_param_distance(
    candidates: pd.DataFrame,
    sample_flux: float,
    sample_eff: float,
    score_metric: str,
    plot_path: Path,
) -> None:
    """Scatter of score vs Euclidean distance in (flux, eff) space."""
    df = candidates.dropna(subset=["dict_flux_cm2_min", "dict_eff_1", "score_value"])
    if len(df) < 3:
        return
    flux_norm = (df["dict_flux_cm2_min"].astype(float) - sample_flux)
    eff_norm = (df["dict_eff_1"].astype(float) - sample_eff)
    param_dist = np.sqrt(flux_norm ** 2 + eff_norm ** 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(param_dist, df["score_value"], s=18, alpha=0.6, color="#4C78A8")
    ax.set_xlabel("Parameter distance sqrt(d_flux^2 + d_eff^2)")
    ax.set_ylabel(f"Score ({score_metric})")
    ax.set_title(f"Score vs parameter distance  (n={len(df)})")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_score_distribution(
    scores: pd.Series,
    score_metric: str,
    best_score: float,
    plot_path: Path,
) -> None:
    """Combined histogram + CDF of scores (2-panel figure)."""
    scores = scores.dropna()
    if scores.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Histogram
    ax1.hist(scores, bins=40, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax1.axvline(best_score, color="#E45756", linestyle="--", linewidth=1.2,
               label=f"Best = {best_score:.4g}")
    ax1.set_xlabel(f"Score ({score_metric})")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Score distribution  (n={len(scores)})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    # CDF
    sorted_scores = scores.sort_values()
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.step(sorted_scores.values, cdf, where="post", color="#4C78A8", linewidth=1.5)
    ax2.axvline(best_score, color="#E45756", linestyle="--", linewidth=1.2,
               label=f"Best = {best_score:.4g}")
    ax2.fill_between(sorted_scores.values, 0, cdf, alpha=0.12, color="#4C78A8", step="post")
    ax2.set_xlabel(f"Score ({score_metric})")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title(f"Score CDF  (n={len(scores)})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_feature_diagnostics(
    sample_vec: np.ndarray,
    best_vec: np.ndarray,
    sample_vec_scaled: np.ndarray,
    best_vec_scaled: np.ndarray,
    feature_names: list,
    plot_path: Path,
    title: str,
) -> None:
    """2×2 multi-panel: feature bars, scatter, residuals, L2 contribution."""
    mask_all = np.isfinite(sample_vec) & np.isfinite(best_vec)
    if mask_all.sum() < 1:
        return

    short_names = [n.replace("_rate_hz", "").replace("raw_tt_", "tt_")
                   .replace("events_per_second_", "") for n in feature_names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Top-left: grouped bar chart ---
    ax = axes[0, 0]
    idx = np.where(mask_all)[0]
    names = [short_names[i] for i in idx]
    sv = sample_vec[idx]
    bv = best_vec[idx]
    x_pos = np.arange(len(names))
    width = 0.38
    ax.bar(x_pos - width / 2, sv, width, label="Test sample", color="#4C78A8", alpha=0.85)
    ax.bar(x_pos + width / 2, bv, width, label="Best candidate", color="#F58518", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rate [Hz]")
    ax.set_title("Feature comparison", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # --- Top-right: sample vs best scatter ---
    ax = axes[0, 1]
    x, y = best_vec[mask_all], sample_vec[mask_all]
    ax.scatter(x, y, s=24, alpha=0.7, color="#72B7B2")
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    pad = 0.02 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
    ax.set_xlabel("Best-fit candidate rate")
    ax.set_ylabel("Test sample rate")
    ax.set_title("Sample vs best-fit", fontsize=10)
    ax.grid(True, alpha=0.2)

    # --- Bottom-left: residuals bar chart ---
    ax = axes[1, 0]
    mask_resid = mask_all & (np.abs(sample_vec) > 1e-12)
    idx_r = np.where(mask_resid)[0]
    if len(idx_r) > 0:
        residuals = (best_vec[idx_r] - sample_vec[idx_r]) / np.abs(sample_vec[idx_r]) * 100
        r_names = [short_names[i] for i in idx_r]
        colors_r = ["#E45756" if r > 0 else "#4C78A8" for r in residuals]
        ax.bar(np.arange(len(r_names)), residuals, color=colors_r, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(r_names)))
        ax.set_xticklabels(r_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Relative residual [%]")
    ax.set_title("Per-feature residual: (best - sample) / sample", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    # --- Bottom-right: L2 contribution ---
    ax = axes[1, 1]
    mask_l2 = np.isfinite(sample_vec_scaled) & np.isfinite(best_vec_scaled)
    idx_l = np.where(mask_l2)[0]
    if len(idx_l) > 0:
        contrib = (sample_vec_scaled[idx_l] - best_vec_scaled[idx_l]) ** 2
        l_names = [short_names[i] for i in idx_l]
        order = np.argsort(contrib)[::-1]
        y_pos = np.arange(len(l_names))
        ax.barh(y_pos, contrib[order], color="#4C78A8", alpha=0.85, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([l_names[i] for i in order], fontsize=8)
        ax.invert_yaxis()
    ax.set_xlabel("Squared difference (scaled)")
    ax.set_title("Per-feature contribution to L2 distance", fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _resolve_sample_events_count(sample_row: pd.Series, dict_sample: pd.Series) -> float:
    for col in ("generated_events_count", "selected_rows", "requested_rows"):
        if col in sample_row.index:
            val = as_float(sample_row.get(col))
            if np.isfinite(val):
                return val
    for col in ("selected_rows", "requested_rows"):
        val = as_float(dict_sample.get(col))
        if np.isfinite(val):
            return val
    return np.nan


def _resolve_truth_value(sample_row: pd.Series, dict_sample: pd.Series, col: str) -> float:
    """Pick truth values from dictionary when available, else from reference row."""
    val = as_float(dict_sample.get(col))
    if np.isfinite(val):
        return val
    return as_float(sample_row.get(col))


def _pick_single_sample_index(
    sample_index_arg: int | None,
    ref_df: pd.DataFrame,
    ref_with_dict: pd.DataFrame,
    cos_n_target: float,
    eff_tol: float,
    rng: np.random.Generator,
) -> int:
    if sample_index_arg is not None:
        return int(sample_index_arg)

    eligible_mask = (
        ref_with_dict["dict_cos_n"].notna()
        & (ref_with_dict["dict_cos_n"] - cos_n_target).abs().le(1e-9)
        & ref_with_dict["dict_eff_1"].notna()
        & ref_with_dict["dict_eff_2"].notna()
        & ref_with_dict["dict_eff_3"].notna()
        & ref_with_dict["dict_eff_4"].notna()
        & (ref_with_dict["dict_eff_2"] - ref_with_dict["dict_eff_1"]).abs().le(eff_tol)
        & (ref_with_dict["dict_eff_3"] - ref_with_dict["dict_eff_1"]).abs().le(eff_tol)
        & (ref_with_dict["dict_eff_4"] - ref_with_dict["dict_eff_1"]).abs().le(eff_tol)
    )
    eligible_pos = np.flatnonzero(eligible_mask.to_numpy())
    if len(eligible_pos) == 0:
        log.warning("No eligible samples with exact cos_n and equal effs; "
                    "relaxing to cos_n +/-0.05 tolerance.")
        eligible_mask = (
            ref_with_dict["dict_cos_n"].notna()
            & (ref_with_dict["dict_cos_n"] - cos_n_target).abs().le(0.05)
            & ref_with_dict["dict_eff_1"].notna()
        )
        eligible_pos = np.flatnonzero(eligible_mask.to_numpy())
    if len(eligible_pos) == 0:
        raise ValueError("No eligible samples found even with relaxed criteria.")
    log.info("Eligible sample rows: %d", len(eligible_pos))
    return int(rng.choice(eligible_pos))


def _evaluate_sample(
    sample_idx: int,
    *,
    ref_df: pd.DataFrame,
    dict_lookup: pd.DataFrame,
    dict_subset: pd.DataFrame,
    metric_df: pd.DataFrame,
    metric_df_scaled: pd.DataFrame,
    metric_cols: list[str],
    metric_mode: str,
    metric_scale: str,
    score_metric: str,
    rate_prefix: str,
    eff_method: str,
    global_rate_col: str,
    z_tol: float,
    flux_tol: float,
    eff_match_tol: float,
    exclude_self: bool,
    join_col: str,
    scale_means: pd.Series | None,
    scale_stds: pd.Series | None,
    emp_eff: pd.DataFrame | None,
    interpolation_k: int = 1,
) -> dict:
    if not (0 <= sample_idx < len(ref_df)):
        raise IndexError(f"Sample index {sample_idx} out of range (0..{len(ref_df) - 1}).")

    sample_row = ref_df.iloc[sample_idx]
    sample_id = sample_row.get(join_col)
    sample_in_dictionary = sample_id in dict_lookup.index
    if sample_in_dictionary:
        dict_sample = dict_lookup.loc[sample_id]
        if isinstance(dict_sample, pd.DataFrame):
            dict_sample = dict_sample.iloc[0]
    else:
        dict_sample = sample_row

    if metric_mode == "dict_eff_global":
        if emp_eff is None:
            raise ValueError("Empirical efficiencies are required for dict_eff_global mode.")
        sample_vec_series = pd.Series(
            [
                emp_eff.iloc[sample_idx]["eff_est_p1"],
                emp_eff.iloc[sample_idx]["eff_est_p2"],
                emp_eff.iloc[sample_idx]["eff_est_p3"],
                emp_eff.iloc[sample_idx]["eff_est_p4"],
                ref_df.iloc[sample_idx][global_rate_col],
            ],
            index=metric_cols,
        )
    else:
        sample_vec_series = metric_df.iloc[sample_idx]

    sample_flux = _resolve_truth_value(sample_row, dict_sample, "flux_cm2_min")
    sample_eff = _resolve_truth_value(sample_row, dict_sample, "eff_1")
    sample_cosn = _resolve_truth_value(sample_row, dict_sample, "cos_n")
    sample_eff_2 = _resolve_truth_value(sample_row, dict_sample, "eff_2")
    sample_eff_3 = _resolve_truth_value(sample_row, dict_sample, "eff_3")
    sample_eff_4 = _resolve_truth_value(sample_row, dict_sample, "eff_4")
    sample_events = _resolve_sample_events_count(sample_row, dict_sample)
    events_label = "n/a" if not np.isfinite(sample_events) else f"{int(sample_events)}"
    sample_label = (
        dict_sample.get("filename_base")
        or dict_sample.get("file_name")
        or sample_row.get("filename_base")
        or sample_row.get("file_name")
        or "sample"
    )

    z_cols = ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
    for col in z_cols:
        if col not in ref_df.columns:
            raise KeyError(f"Missing {col} in reference CSV.")
    z_vals = sample_row[z_cols].astype(float).to_numpy()
    z_mask = np.ones(len(ref_df), dtype=bool)
    for idx_z, col in enumerate(z_cols):
        z_mask &= np.isclose(ref_df[col].astype(float), z_vals[idx_z], atol=z_tol)
    if exclude_self:
        z_mask[sample_idx] = False
        # Also exclude rows sharing the same file identifier as the sample.
        if join_col in ref_df.columns and pd.notna(sample_id):
            same_id = ref_df[join_col].astype(str).eq(str(sample_id)).to_numpy()
            z_mask &= ~same_id
        # Exclude candidates with identical physical parameters
        # (flux, cos_n, eff_1..4).  Without this, duplicate parameter
        # sets in the reference trivially match, giving ~0 error and
        # producing optimistic uncertainty estimates downstream.
        param_match_cols = [
            ("flux_cm2_min", sample_flux),
            ("cos_n", sample_cosn),
            ("eff_1", sample_eff),
            ("eff_2", sample_eff_2),
            ("eff_3", sample_eff_3),
            ("eff_4", sample_eff_4),
        ]
        same_params = np.ones(len(ref_df), dtype=bool)
        for col_name, val in param_match_cols:
            if col_name in ref_df.columns and np.isfinite(val):
                same_params &= np.isclose(
                    pd.to_numeric(ref_df[col_name], errors="coerce").to_numpy(dtype=float),
                    val, atol=flux_tol if col_name == "flux_cm2_min" else eff_match_tol,
                )
        z_mask &= ~same_params

    candidates = ref_df.loc[z_mask].copy()
    if candidates.empty:
        raise ValueError("No candidates after z-position filtering.")
    orig_ref_indices = candidates.index.to_numpy().copy()
    candidates = candidates.merge(dict_subset, on=join_col, how="left")
    # For non-dictionary files, fill candidate truth columns from reference CSV.
    fallback_cols = [
        ("dict_flux_cm2_min", "flux_cm2_min"),
        ("dict_cos_n", "cos_n"),
        ("dict_eff_1", "eff_1"),
        ("dict_eff_2", "eff_2"),
        ("dict_eff_3", "eff_3"),
        ("dict_eff_4", "eff_4"),
        ("dict_selected_rows", "selected_rows"),
        ("dict_requested_rows", "requested_rows"),
    ]
    for dict_col, ref_col in fallback_cols:
        if dict_col not in candidates.columns:
            candidates[dict_col] = np.nan
        if ref_col in candidates.columns:
            dict_vals = pd.to_numeric(candidates[dict_col], errors="coerce")
            ref_vals = pd.to_numeric(candidates[ref_col], errors="coerce")
            candidates[dict_col] = dict_vals.where(dict_vals.notna(), ref_vals)
    candidates["_orig_ref_idx"] = orig_ref_indices
    if candidates["dict_flux_cm2_min"].isna().all():
        raise ValueError("Flux truth values missing for all candidates.")

    sample_vec = _scale_vector(sample_vec_series, metric_scale, scale_means, scale_stds)
    if np.isfinite(sample_vec).sum() < 2:
        raise ValueError("Sample metric vector has fewer than 2 finite values.")
    score_fn = SCORE_FNS.get(score_metric)
    if score_fn is None:
        raise ValueError(f"Unknown score_metric: {score_metric}")

    score_values = []
    for idx in candidates.index:
        orig_idx = int(candidates.loc[idx, "_orig_ref_idx"])
        cand_vec = metric_df_scaled.loc[orig_idx].to_numpy(dtype=float)
        score_values.append(score_fn(sample_vec, cand_vec))
    candidates["score_value"] = score_values

    scores_valid = candidates["score_value"].dropna()
    if scores_valid.empty:
        raise ValueError("All candidates have NaN scores.")
    best_idx = scores_valid.idxmin() if score_metric in LOWER_IS_BETTER else scores_valid.idxmax()
    best_row = candidates.loc[best_idx]
    best_orig_idx = int(best_row["_orig_ref_idx"])
    best_vec_raw = metric_df.loc[best_orig_idx].to_numpy(dtype=float)
    best_score = float(scores_valid.loc[best_idx])

    # --- IDW interpolation across K nearest neighbours ----------------
    if interpolation_k > 1 and len(candidates) >= 2:
        k_use = min(interpolation_k, len(candidates))
        if score_metric in LOWER_IS_BETTER:
            top_k = candidates.nsmallest(k_use, "score_value")
        else:
            top_k = candidates.nlargest(k_use, "score_value")
        top_k_scores = top_k["score_value"].to_numpy(dtype=float)
        # For lower-is-better metrics, distance = score; for higher-is-better
        # metrics, invert so that highest score gets highest weight.
        if score_metric in LOWER_IS_BETTER:
            dists = np.clip(top_k_scores, 1e-15, None)
        else:
            dists = np.clip(1.0 / (top_k_scores + 1e-15), 1e-15, None)
        # If the best distance is exactly 0, give it all the weight.
        if dists[0] < 1e-12:
            weights = np.zeros_like(dists)
            weights[0] = 1.0
        else:
            weights = 1.0 / dists ** 2
            weights /= weights.sum()
        top_k_flux = pd.to_numeric(
            top_k["dict_flux_cm2_min"], errors="coerce"
        ).to_numpy(dtype=float)
        top_k_eff = pd.to_numeric(
            top_k["dict_eff_1"], errors="coerce"
        ).to_numpy(dtype=float)
        # Fall back to single-best if flux/eff have NaNs
        flux_finite = np.isfinite(top_k_flux)
        eff_finite = np.isfinite(top_k_eff)
        if flux_finite.sum() >= 2 and eff_finite.sum() >= 2:
            w_f = weights.copy(); w_f[~flux_finite] = 0.0
            if w_f.sum() > 0: w_f /= w_f.sum()
            w_e = weights.copy(); w_e[~eff_finite] = 0.0
            if w_e.sum() > 0: w_e /= w_e.sum()
            best_flux = float(np.nansum(w_f * top_k_flux))
            best_eff = float(np.nansum(w_e * top_k_eff))
        else:
            best_flux = as_float(best_row.get("dict_flux_cm2_min"))
            best_eff = as_float(best_row.get("dict_eff_1"))
    else:
        best_flux = as_float(best_row.get("dict_flux_cm2_min"))
        best_eff = as_float(best_row.get("dict_eff_1"))

    flux_error = best_flux - sample_flux if np.isfinite(best_flux) and np.isfinite(sample_flux) else np.nan
    eff_error = best_eff - sample_eff if np.isfinite(best_eff) and np.isfinite(sample_eff) else np.nan
    flux_rel_error_pct = safe_rel_error_pct(flux_error, sample_flux)
    eff_rel_error_pct = safe_rel_error_pct(eff_error, sample_eff)

    candidates["flux_error"] = candidates["dict_flux_cm2_min"].astype(float) - sample_flux
    candidates["eff_error"] = candidates["dict_eff_1"].astype(float) - sample_eff
    candidates["flux_rel_error"] = np.where(
        sample_flux != 0,
        candidates["flux_error"] / sample_flux,
        np.nan,
    )
    candidates["eff_rel_error"] = np.where(
        sample_eff != 0,
        candidates["eff_error"] / sample_eff,
        np.nan,
    )

    truth_rank = -1
    if np.isfinite(sample_flux) and np.isfinite(sample_eff):
        flux_match = (candidates["dict_flux_cm2_min"].astype(float) - sample_flux).abs().le(flux_tol)
        eff_match = (candidates["dict_eff_1"].astype(float) - sample_eff).abs().le(eff_match_tol)
        truth_scores = candidates.loc[flux_match & eff_match, "score_value"].dropna()
        if not truth_scores.empty:
            truth_score = float(truth_scores.iloc[0])
            if score_metric in LOWER_IS_BETTER:
                truth_rank = int((candidates["score_value"] < truth_score).sum()) + 1
            else:
                truth_rank = int((candidates["score_value"] > truth_score).sum()) + 1

    # --- Top-N candidate spread (to_do.md §4.3) ---
    n_top = min(10, len(candidates))
    if score_metric in LOWER_IS_BETTER:
        top_n = candidates.nsmallest(n_top, "score_value")
    else:
        top_n = candidates.nlargest(n_top, "score_value")
    top_flux = pd.to_numeric(top_n.get("dict_flux_cm2_min"), errors="coerce").dropna()
    top_eff = pd.to_numeric(top_n.get("dict_eff_1"), errors="coerce").dropna()
    top_n_flux_std = float(top_flux.std()) if len(top_flux) > 1 else 0.0
    top_n_eff_std = float(top_eff.std()) if len(top_eff) > 1 else 0.0
    top_n_flux_range = float(top_flux.max() - top_flux.min()) if len(top_flux) > 1 else 0.0
    top_n_eff_range = float(top_eff.max() - top_eff.min()) if len(top_eff) > 1 else 0.0

    return {
        "sample_idx": int(sample_idx),
        "sample_id": sample_id,
        "sample_in_dictionary": bool(sample_in_dictionary),
        "sample_label": sample_label,
        "sample_flux": sample_flux,
        "sample_eff": sample_eff,
        "sample_cosn": sample_cosn,
        "sample_eff_2": sample_eff_2,
        "sample_eff_3": sample_eff_3,
        "sample_eff_4": sample_eff_4,
        "sample_events_count": sample_events,
        "events_label": events_label,
        "sample_vec_series": sample_vec_series,
        "sample_vec_scaled": sample_vec,
        "candidates": candidates,
        "best_idx": int(best_idx),
        "best_row": best_row,
        "best_orig_idx": best_orig_idx,
        "best_vec_raw": best_vec_raw,
        "best_flux": best_flux,
        "best_eff": best_eff,
        "best_score": best_score,
        "best_file": best_row.get(join_col),
        "best_in_dictionary": bool(best_row.get(join_col) in dict_lookup.index),
        "flux_error": flux_error,
        "eff_error": eff_error,
        "flux_rel_error_pct": flux_rel_error_pct,
        "eff_rel_error_pct": eff_rel_error_pct,
        "truth_rank": int(truth_rank),
        "truth_rank_total": int(len(candidates)),
        "top_n_flux_std": top_n_flux_std,
        "top_n_eff_std": top_n_eff_std,
        "top_n_flux_range": top_n_flux_range,
        "top_n_eff_range": top_n_eff_range,
    }


def _plot_all_true_vs_est_combined(
    df: pd.DataFrame,
    plot_path: Path,
) -> None:
    """True vs estimated: flux and eff side-by-side (1×2)."""
    pairs = [
        ("true_flux_cm2_min", "estimated_flux_cm2_min",
         "flux [cm^-2 min^-1]", "True vs estimated flux"),
        ("true_eff_1", "estimated_eff_1",
         "eff_1", "True vs estimated efficiency"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (true_col, est_col, axis_label, title) in zip(axes, pairs):
        true_vals = pd.to_numeric(df.get(true_col), errors="coerce")
        est_vals = pd.to_numeric(df.get(est_col), errors="coerce")
        mask = true_vals.notna() & est_vals.notna()
        if mask.sum() < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        x = true_vals[mask].to_numpy(dtype=float)
        y = est_vals[mask].to_numpy(dtype=float)
        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        pad = 0.02 * (hi - lo) if hi > lo else 0.1
        ax.scatter(x, y, s=20, alpha=0.65, color="#F58518")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
        ax.set_xlabel(f"True {axis_label}")
        ax.set_ylabel(f"Estimated {axis_label}")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    fig.suptitle("ALL mode: true vs estimated parameters", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# All-mode stratification helpers (to_do.md §4.1, §4.3)
# ---------------------------------------------------------------------------

def _stratified_error_summary(ok_df: pd.DataFrame) -> dict:
    """Compute error medians/quantiles stratified by in-dict vs off-dict."""
    if "sample_in_dictionary" not in ok_df.columns:
        return {}
    from msv_utils import coerce_bool_series
    membership = coerce_bool_series(ok_df["sample_in_dictionary"])
    strata: dict[str, object] = {}
    for label, flag in (("in_dictionary", True), ("out_dictionary", False)):
        part = ok_df.loc[membership == flag]
        if part.empty:
            strata[label] = {"n_samples": 0}
            continue
        strata[label] = {
            "n_samples": int(len(part)),
            "median_abs_flux_rel_error_pct": float(part["abs_flux_rel_error_pct"].median()),
            "p68_abs_flux_rel_error_pct": float(part["abs_flux_rel_error_pct"].quantile(0.68)),
            "p95_abs_flux_rel_error_pct": float(part["abs_flux_rel_error_pct"].quantile(0.95)),
            "median_abs_eff_rel_error_pct": float(part["abs_eff_rel_error_pct"].median()),
            "p68_abs_eff_rel_error_pct": float(part["abs_eff_rel_error_pct"].quantile(0.68)),
            "p95_abs_eff_rel_error_pct": float(part["abs_eff_rel_error_pct"].quantile(0.95)),
        }
    return strata


def _plot_stratified_errors(ok_df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    """Overlay in-dict vs off-dict error distributions (to_do.md §4.1)."""
    if "sample_in_dictionary" not in ok_df.columns:
        return
    from msv_utils import coerce_bool_series
    membership = coerce_bool_series(ok_df["sample_in_dictionary"])
    in_df = ok_df.loc[membership == True].copy()
    out_df = ok_df.loc[membership == False].copy()
    if in_df.empty and out_df.empty:
        return

    for error_col, label in (
        ("abs_flux_rel_error_pct", "|Flux rel. error| [%]"),
        ("abs_eff_rel_error_pct", "|Eff rel. error| [%]"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        vals_in = pd.to_numeric(in_df.get(error_col), errors="coerce").dropna()
        vals_out = pd.to_numeric(out_df.get(error_col), errors="coerce").dropna()
        bins = 50
        if not vals_in.empty:
            ax.hist(vals_in, bins=bins, alpha=0.6, label=f"in-dict (n={len(vals_in)})",
                    color="#4C78A8", edgecolor="white")
        if not vals_out.empty:
            ax.hist(vals_out, bins=bins, alpha=0.6, label=f"off-dict (n={len(vals_out)})",
                    color="#E45756", edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"In-dict vs Off-dict: {label}")
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"all_stratified_{error_col}_{tag}.png", dpi=150)
        plt.close(fig)

    # Scatter: events vs error, colored by membership, with 1/√N guide
    for error_col, ylabel in (
        ("abs_flux_rel_error_pct", "|Flux rel. error| [%]"),
        ("abs_eff_rel_error_pct", "|Eff rel. error| [%]"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        all_x, all_y = [], []
        for subset_df, lbl, color in (
            (in_df, "in-dict", "#4C78A8"),
            (out_df, "off-dict", "#E45756"),
        ):
            x = pd.to_numeric(subset_df.get("sample_events_count"), errors="coerce")
            y = pd.to_numeric(subset_df.get(error_col), errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() > 0:
                ax.scatter(x[mask], y[mask], s=16, alpha=0.5,
                           color=color, label=f"{lbl} (n={mask.sum()})")
                all_x.extend(x[mask].tolist())
                all_y.extend(y[mask].tolist())
        # 1/√N guide curve
        if len(all_x) >= 3:
            ax_arr = np.asarray(all_x, dtype=float)
            ay_arr = np.asarray(all_y, dtype=float)
            pos = ax_arr > 0
            if pos.sum() >= 3:
                med_err = float(np.nanmedian(ay_arr[pos]))
                med_ev = float(np.nanmedian(ax_arr[pos]))
                if med_ev > 0 and med_err > 0:
                    scale = med_err * np.sqrt(med_ev)
                    ev_line = np.linspace(float(ax_arr[pos].min()),
                                         float(ax_arr[pos].max()), 300)
                    ax.plot(ev_line, scale / np.sqrt(ev_line), "r--",
                            linewidth=1.5, label=r"expected $\propto 1/\sqrt{N}$")
        ax.set_xlabel("Sample events count")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Error vs events — in-dict vs off-dict")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        maybe_log_x(ax, pd.to_numeric(ok_df.get("sample_events_count"), errors="coerce"))
        fig.tight_layout()
        fig.savefig(out_dir / f"all_stratified_events_vs_{error_col}_{tag}.png", dpi=150)
        plt.close(fig)




# _build_uncertainty_table — replaced by msv_utils.build_uncertainty_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Self-consistency analysis.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metric-suffix", default=None)
    parser.add_argument("--metric-prefix", default=None,
                        help="Prefix filter for metric columns.")
    parser.add_argument("--metric-mode", default=None,
                        choices=["raw_tt_rates", "eff_global", "dict_eff_global"],
                        help="Metric set to use for scoring.")
    parser.add_argument("--global-rate-col", default=None)
    parser.add_argument("--include-global-rate", default=None,
                        help="Add global rate to raw_tt_rates feature vector (true/false).")
    parser.add_argument("--eff-method", default=None,
                        choices=["four_over_three_plus_four", "one_minus_three_over_four"])
    parser.add_argument("--rate-prefix", default=None)
    parser.add_argument("--metric-scale", default=None,
                        choices=["none", "zscore"])
    parser.add_argument("--score-metric", default=None,
                        choices=["l2", "chi2", "poisson", "r2"])
    parser.add_argument("--cos-n", type=float, default=None)
    parser.add_argument("--eff-tol", type=float, default=None)
    parser.add_argument("--eff-match-tol", type=float, default=None)
    parser.add_argument("--flux-tol", type=float, default=None)
    parser.add_argument("--z-tol", type=float, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--max-sample-tries", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--interpolation-k", type=int, default=None,
                        help="Number of nearest neighbours for IDW "
                             "interpolation (1 = nearest-only, default).")
    parser.add_argument("--exclude-self", action="store_true",
                        help="Exclude the sample itself from candidates.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-s", "--single", action="store_true",
                            help="Run one-file mode (legacy behavior).")
    mode_group.add_argument("-a", "--all", dest="run_all", action="store_true",
                            help="Run all-files validation mode.")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    reference_csv = resolve_param(args.reference_csv, config, "reference_csv", str(DEFAULT_REF))
    dictionary_csv = resolve_param(args.dictionary_csv, config, "dictionary_csv", str(DEFAULT_DICT))
    out_dir_str = resolve_param(args.out_dir, config, "out_dir", str(DEFAULT_OUT))
    seed_raw = resolve_param(args.seed, config, "seed", 123, str)
    metric_suffix = resolve_param(args.metric_suffix, config, "metric_suffix", "_rate_hz")
    metric_prefix = resolve_param(args.metric_prefix, config, "metric_prefix", "raw_tt_")
    metric_mode = resolve_param(args.metric_mode, config, "metric_mode", "raw_tt_rates")
    global_rate_col = resolve_param(args.global_rate_col, config, "global_rate_col", "events_per_second_global_rate")
    include_global = str(resolve_param(args.include_global_rate, config, "include_global_rate", "true")).lower() in ("true", "1", "yes")
    eff_method = resolve_param(args.eff_method, config, "eff_method", "four_over_three_plus_four")
    rate_prefix = resolve_param(args.rate_prefix, config, "rate_prefix", "raw")
    metric_scale = resolve_param(args.metric_scale, config, "metric_scale", "zscore")
    score_metric = resolve_param(args.score_metric, config, "score_metric", "l2")
    cos_n_target = resolve_param(args.cos_n, config, "cos_n", 2.0, float)
    eff_tol = resolve_param(args.eff_tol, config, "eff_tol", 1e-9, float)
    eff_match_tol = resolve_param(args.eff_match_tol, config, "eff_match_tol", 1e-9, float)
    flux_tol = resolve_param(args.flux_tol, config, "flux_tol", 1e-9, float)
    z_tol = resolve_param(args.z_tol, config, "z_tol", 1e-6, float)
    max_sample_tries = resolve_param(args.max_sample_tries, config, "max_sample_tries", 1000, int)  # noqa: F841
    top_n = resolve_param(args.top_n, config, "top_n", 10, int)
    interpolation_k = int(resolve_param(
        args.interpolation_k, config, "interpolation_k", 1, int))
    exclude_self = args.exclude_self or config.get("exclude_self", False)
    sample_index_arg = args.sample_index

    run_mode_cfg = str(config.get("run_mode", "single")).strip().lower()
    if args.run_all:
        run_mode = "all"
    elif args.single:
        run_mode = "single"
    elif run_mode_cfg in ("single", "all"):
        run_mode = run_mode_cfg
    else:
        raise ValueError(f"Unsupported run_mode in config: {run_mode_cfg}")
    if run_mode == "all" and sample_index_arg is not None:
        raise ValueError("--sample-index is only valid in single mode.")

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.png"):
        path.unlink()

    log.info("Run mode: %s", run_mode)
    log.info("Interpolation K: %d%s", interpolation_k,
             " (nearest-only)" if interpolation_k <= 1
             else f" (IDW over {interpolation_k} neighbours)")
    log.info("Loading reference: %s", reference_csv)
    ref_df = pd.read_csv(reference_csv, low_memory=False)
    ref_df = extract_eff_columns(ref_df)

    log.info("Loading dictionary: %s", dictionary_csv)
    dict_df = pd.read_csv(dictionary_csv, low_memory=False)
    dict_df = extract_eff_columns(dict_df)

    join_col = find_join_col(ref_df, dict_df)
    if join_col is None:
        raise KeyError("No shared file_name/filename_base column for dictionary join.")
    dict_lookup = dict_df.drop_duplicates(subset=[join_col], keep="first").set_index(join_col, drop=False)

    dict_cols = [
        join_col, "flux_cm2_min", "cos_n",
        "eff_1", "eff_2", "eff_3", "eff_4",
        "selected_rows", "requested_rows",
    ]
    dict_cols = [col for col in dict_cols if col in dict_df.columns]
    dict_subset = dict_df[dict_cols].copy()
    dict_subset = dict_subset.rename(columns={
        "flux_cm2_min": "dict_flux_cm2_min",
        "cos_n": "dict_cos_n",
        "eff_1": "dict_eff_1",
        "eff_2": "dict_eff_2",
        "eff_3": "dict_eff_3",
        "eff_4": "dict_eff_4",
        "selected_rows": "dict_selected_rows",
        "requested_rows": "dict_requested_rows",
    })
    ref_with_dict = ref_df[[join_col]].join(dict_subset.set_index(join_col), on=join_col)

    log.info("Metric mode: %s", metric_mode)
    if metric_mode == "raw_tt_rates":
        metric_cols = _select_metric_cols(ref_df, metric_suffix, metric_prefix or None)
        if not metric_cols:
            raise ValueError(
                f"No metric columns found with suffix '{metric_suffix}'"
                f" and prefix '{metric_prefix}'."
            )
        if include_global and global_rate_col in ref_df.columns:
            metric_cols.append(global_rate_col)
        metric_df = ref_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    elif metric_mode == "eff_global":
        metric_cols = ["eff_1", "eff_2", "eff_3", "eff_4", global_rate_col]
        missing = [c for c in metric_cols if c not in ref_df.columns]
        if missing:
            raise KeyError(f"Missing columns for eff_global: {missing}")
        metric_df = ref_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    elif metric_mode == "dict_eff_global":
        metric_cols = ["dict_eff_1", "dict_eff_2", "dict_eff_3", "dict_eff_4", global_rate_col]
        missing = [c for c in metric_cols[:-1] if c not in ref_with_dict.columns]
        if missing:
            raise KeyError(f"Missing columns for dict_eff_global: {missing}")
        metric_df = pd.concat(
            [ref_with_dict[metric_cols[:-1]], ref_df[[global_rate_col]]],
            axis=1,
        ).apply(pd.to_numeric, errors="coerce")
    else:
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    log.info("Feature columns (%d): %s", len(metric_cols), metric_cols)
    if metric_df.isna().all(axis=None):
        raise ValueError("All metric columns are NaN after numeric coercion.")

    metric_df_scaled, scale_means, scale_stds = _scale_metrics(metric_df, metric_scale)
    emp_eff = build_empirical_eff(ref_df, rate_prefix, eff_method) if metric_mode == "dict_eff_global" else None

    if isinstance(seed_raw, str) and seed_raw.strip().lower() == "random":
        seed = None
    else:
        seed = int(seed_raw)
    rng = np.random.default_rng(seed)

    tag = f"{metric_mode}_{score_metric}"

    if run_mode == "single":
        sample_idx = _pick_single_sample_index(
            sample_index_arg=sample_index_arg,
            ref_df=ref_df,
            ref_with_dict=ref_with_dict,
            cos_n_target=cos_n_target,
            eff_tol=eff_tol,
            rng=rng,
        )
        log.info("Sample index: %d", sample_idx)
        result = _evaluate_sample(
            sample_idx,
            ref_df=ref_df,
            dict_lookup=dict_lookup,
            dict_subset=dict_subset,
            metric_df=metric_df,
            metric_df_scaled=metric_df_scaled,
            metric_cols=metric_cols,
            metric_mode=metric_mode,
            metric_scale=metric_scale,
            score_metric=score_metric,
            rate_prefix=rate_prefix,
            eff_method=eff_method,
            global_rate_col=global_rate_col,
            z_tol=z_tol,
            flux_tol=flux_tol,
            eff_match_tol=eff_match_tol,
            exclude_self=exclude_self,
            join_col=join_col,
            scale_means=scale_means,
            scale_stds=scale_stds,
            emp_eff=emp_eff,
            interpolation_k=interpolation_k,
        )

        if metric_mode == "dict_eff_global":
            vals = result["sample_vec_series"].to_numpy(dtype=float)[:4]
            log.info("  Sample empirical effs: %s", [f'{v:.4f}' for v in vals])
        log.info("  Sample in dictionary: %s", result['sample_in_dictionary'])
        log.info("  True flux = %.4g", result['sample_flux'])
        log.info(
            "  True eff  = [%.4f, %.4f, %.4f, %.4f]",
            result['sample_eff'], result['sample_eff_2'],
            result['sample_eff_3'], result['sample_eff_4'],
        )
        log.info("  cos_n     = %.4f", result['sample_cosn'])
        log.info("Candidates (same z-planes): %d", result['truth_rank_total'])
        log.info("  Best candidate: flux=%.4g, eff=%.4g", result['best_flux'], result['best_eff'])
        log.info("  Best in dictionary: %s", result['best_in_dictionary'])
        log.info("  Best score (%s): %.4g", score_metric, result['best_score'])
        log.info(
            "  Flux error:  %.4g  (%.2f%%)",
            result['flux_error'], result['flux_rel_error_pct'],
        )
        log.info(
            "  Eff error:   %.4g  (%.2f%%)",
            result['eff_error'], result['eff_rel_error_pct'],
        )
        if result["truth_rank"] > 0:
            log.info("  Truth rank: %d / %d", result['truth_rank'], result['truth_rank_total'])
        else:
            log.info("  Truth row not found among candidates (sample may be excluded).")

        candidates = result["candidates"]
        candidates_csv = out_dir / "r2_candidates.csv"
        candidates.to_csv(candidates_csv, index=False)
        ascending = score_metric in LOWER_IS_BETTER
        if top_n > 0:
            top_df = candidates.sort_values("score_value", ascending=ascending).head(top_n)
            top_df.to_csv(out_dir / "top_candidates.csv", index=False)

        def _none_if_nan(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        summary = {
            "run_mode": "single",
            "sample_index": int(result["sample_idx"]),
            "sample_file": result["sample_id"],
            "sample_in_dictionary": bool(result["sample_in_dictionary"]),
            "sample_flux": _none_if_nan(result["sample_flux"]),
            "sample_eff": _none_if_nan(result["sample_eff"]),
            "sample_cos_n": _none_if_nan(result["sample_cosn"]),
            "sample_events_count": _none_if_nan(result["sample_events_count"]),
            "candidate_rows": int(result["truth_rank_total"]),
            "metric_mode": metric_mode,
            "metric_scale": metric_scale,
            "score_metric": score_metric,
            "metric_columns": metric_cols,
            "exclude_self": bool(exclude_self),
            "best_index": int(result["best_idx"]),
            "best_score": float(result["best_score"]),
            "best_file": result["best_file"],
            "best_in_dictionary": bool(result["best_in_dictionary"]),
            "best_flux_cm2_min": _none_if_nan(result["best_flux"]),
            "best_eff": _none_if_nan(result["best_eff"]),
            "flux_error": _none_if_nan(result["flux_error"]),
            "flux_rel_error_pct": _none_if_nan(result["flux_rel_error_pct"]),
            "eff_error": _none_if_nan(result["eff_error"]),
            "eff_rel_error_pct": _none_if_nan(result["eff_rel_error_pct"]),
            "truth_rank": int(result["truth_rank"]),
            "truth_rank_total": int(result["truth_rank_total"]),
        }
        with (out_dir / "r2_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        plot_cands = candidates.dropna(subset=["dict_flux_cm2_min", "dict_eff_1", "score_value"])
        if len(plot_cands) >= 3:
            _plot_contour(
                plot_cands,
                float(result["sample_flux"]),
                float(result["sample_eff"]),
                float(result["best_flux"]) if np.isfinite(result["best_flux"]) else None,
                float(result["best_eff"]) if np.isfinite(result["best_eff"]) else None,
                score_metric,
                metric_mode,
                result["best_score"],
                result["truth_rank"],
                len(candidates),
                out_dir / f"contour_{tag}.png",
                result["events_label"],
                result["sample_label"],
                top_n=top_n,
            )
        else:
            log.warning("Too few candidates for contour plot.")

        sample_vec_series = result["sample_vec_series"]
        best_orig_idx = int(result["best_orig_idx"])
        _plot_feature_diagnostics(
            sample_vec_series.to_numpy(dtype=float),
            metric_df.loc[best_orig_idx].to_numpy(dtype=float),
            result["sample_vec_scaled"],
            metric_df_scaled.loc[best_orig_idx].to_numpy(dtype=float),
            metric_cols,
            out_dir / f"feature_diagnostics_{tag}.png",
            f"Feature diagnostics: sample vs best ({tag})",
        )
        _plot_score_distribution(
            candidates["score_value"], score_metric,
            result["best_score"],
            out_dir / f"score_distribution_{tag}.png",
        )
        _plot_score_vs_param_distance(
            candidates,
            float(result["sample_flux"]),
            float(result["sample_eff"]),
            score_metric,
            out_dir / f"score_vs_param_dist_{tag}.png",
        )
        log.info("Wrote candidates: %s", candidates_csv)
        log.info("Wrote plots to: %s", out_dir)
        return 0

    if not exclude_self:
        log.warning("exclude_self is False in ALL mode; trivial self-matches may dominate.")
    log.info("Running ALL mode on %d samples.", len(ref_df))
    all_rows: list[dict[str, object]] = []
    for pos, sample_idx in enumerate(range(len(ref_df)), 1):
        try:
            result = _evaluate_sample(
                sample_idx,
                ref_df=ref_df,
                dict_lookup=dict_lookup,
                dict_subset=dict_subset,
                metric_df=metric_df,
                metric_df_scaled=metric_df_scaled,
                metric_cols=metric_cols,
                metric_mode=metric_mode,
                metric_scale=metric_scale,
                score_metric=score_metric,
                rate_prefix=rate_prefix,
                eff_method=eff_method,
                global_rate_col=global_rate_col,
                z_tol=z_tol,
                flux_tol=flux_tol,
                eff_match_tol=eff_match_tol,
                exclude_self=exclude_self,
                join_col=join_col,
                scale_means=scale_means,
                scale_stds=scale_stds,
                emp_eff=emp_eff,
                interpolation_k=interpolation_k,
            )
            all_rows.append(
                {
                    "status": "ok",
                    "error_message": "",
                    "sample_index": int(result["sample_idx"]),
                    "sample_file": result["sample_id"],
                    "sample_in_dictionary": bool(result["sample_in_dictionary"]),
                    "sample_events_count": result["sample_events_count"],
                    "true_flux_cm2_min": result["sample_flux"],
                    "true_eff_1": result["sample_eff"],
                    "estimated_flux_cm2_min": result["best_flux"],
                    "estimated_eff_1": result["best_eff"],
                    "flux_error": result["flux_error"],
                    "eff_error": result["eff_error"],
                    "flux_rel_error_pct": result["flux_rel_error_pct"],
                    "eff_rel_error_pct": result["eff_rel_error_pct"],
                    "abs_flux_rel_error_pct": abs(result["flux_rel_error_pct"])
                        if np.isfinite(result["flux_rel_error_pct"]) else np.nan,
                    "abs_eff_rel_error_pct": abs(result["eff_rel_error_pct"])
                        if np.isfinite(result["eff_rel_error_pct"]) else np.nan,
                    "best_score": result["best_score"],
                    "best_file": result["best_file"],
                    "best_in_dictionary": bool(result["best_in_dictionary"]),
                    "candidate_rows": result["truth_rank_total"],
                    "truth_rank": result["truth_rank"],
                    "truth_rank_total": result["truth_rank_total"],
                    "top_n_flux_std": result["top_n_flux_std"],
                    "top_n_eff_std": result["top_n_eff_std"],
                    "top_n_flux_range": result["top_n_flux_range"],
                    "top_n_eff_range": result["top_n_eff_range"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            sample_file = ref_df.iloc[sample_idx].get(join_col, f"idx_{sample_idx}")
            all_rows.append(
                {
                    "status": "error",
                    "error_message": str(exc),
                    "sample_index": sample_idx,
                    "sample_file": sample_file,
                    "sample_in_dictionary": np.nan,
                    "sample_events_count": np.nan,
                    "true_flux_cm2_min": np.nan,
                    "true_eff_1": np.nan,
                    "estimated_flux_cm2_min": np.nan,
                    "estimated_eff_1": np.nan,
                    "flux_error": np.nan,
                    "eff_error": np.nan,
                    "flux_rel_error_pct": np.nan,
                    "eff_rel_error_pct": np.nan,
                    "abs_flux_rel_error_pct": np.nan,
                    "abs_eff_rel_error_pct": np.nan,
                    "best_score": np.nan,
                    "best_file": "",
                    "best_in_dictionary": np.nan,
                    "candidate_rows": np.nan,
                    "truth_rank": -1,
                    "truth_rank_total": np.nan,
                    "top_n_flux_std": np.nan,
                    "top_n_eff_std": np.nan,
                    "top_n_flux_range": np.nan,
                    "top_n_eff_range": np.nan,
                }
            )

        if pos % 50 == 0 or pos == len(ref_df):
            log.info("Processed %d/%d samples...", pos, len(ref_df))

    all_df = pd.DataFrame(all_rows)
    all_csv = out_dir / "all_samples_results.csv"
    all_df.to_csv(all_csv, index=False)
    ok_df = all_df.loc[all_df["status"] == "ok"].copy()
    failed_df = all_df.loc[all_df["status"] != "ok"].copy()
    ok_csv = out_dir / "all_samples_success.csv"
    failed_csv = out_dir / "all_samples_failed.csv"
    ok_df.to_csv(ok_csv, index=False)
    failed_df.to_csv(failed_csv, index=False)

    summary = {
        "run_mode": "all",
        "metric_mode": metric_mode,
        "metric_scale": metric_scale,
        "score_metric": score_metric,
        "exclude_self": bool(exclude_self),
        "total_samples": int(len(all_df)),
        "successful_samples": int(len(ok_df)),
        "failed_samples": int(len(failed_df)),
        "successful_samples_in_dictionary": int((ok_df["sample_in_dictionary"] == True).sum())
            if "sample_in_dictionary" in ok_df.columns else None,
        "successful_samples_out_dictionary": int((ok_df["sample_in_dictionary"] == False).sum())
            if "sample_in_dictionary" in ok_df.columns else None,
        "median_abs_flux_rel_error_pct": float(ok_df["abs_flux_rel_error_pct"].median())
            if not ok_df.empty else None,
        "median_abs_eff_rel_error_pct": float(ok_df["abs_eff_rel_error_pct"].median())
            if not ok_df.empty else None,
        "p68_abs_flux_rel_error_pct": float(ok_df["abs_flux_rel_error_pct"].quantile(0.68))
            if not ok_df.empty else None,
        "p68_abs_eff_rel_error_pct": float(ok_df["abs_eff_rel_error_pct"].quantile(0.68))
            if not ok_df.empty else None,
        "mean_truth_rank": float(ok_df.loc[ok_df["truth_rank"] > 0, "truth_rank"].mean())
            if (ok_df["truth_rank"] > 0).any() else None,
        "top_n_flux_std_median": float(ok_df["top_n_flux_std"].median())
            if "top_n_flux_std" in ok_df.columns and not ok_df["top_n_flux_std"].isna().all() else None,
        "top_n_eff_std_median": float(ok_df["top_n_eff_std"].median())
            if "top_n_eff_std" in ok_df.columns and not ok_df["top_n_eff_std"].isna().all() else None,
        "stratified_errors": _stratified_error_summary(ok_df),
    }
    with (out_dir / "all_samples_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not ok_df.empty:
        # True vs estimated (combined 1×2 figure)
        _plot_all_true_vs_est_combined(
            ok_df,
            plot_path=out_dir / f"all_true_vs_est_{tag}.png",
        )
        # Stratified plots already include events-vs-error and error histograms
        uncertainty_df = build_uncertainty_table(ok_df, n_bins=10)
        if not uncertainty_df.empty:
            uncertainty_csv = out_dir / f"all_uncertainty_by_events_{tag}.csv"
            uncertainty_df.to_csv(uncertainty_csv, index=False)
            log.info("Wrote uncertainty table: %s", uncertainty_csv)

        # Stratified diagnostics (to_do.md §4.1)
        _plot_stratified_errors(ok_df, out_dir, tag)

        # ── Showcase: single-sample diagnostic plots for one representative ──
        # Pick the sample whose flux error is closest to the median (most
        # representative of the overall distribution) and re-evaluate it
        # to produce the rich contour / feature / score plots from single mode.
        median_flux_err = float(ok_df["abs_flux_rel_error_pct"].median())
        dist_to_median = (ok_df["abs_flux_rel_error_pct"] - median_flux_err).abs()
        showcase_row = ok_df.loc[dist_to_median.idxmin()]
        showcase_idx = int(showcase_row["sample_index"])
        log.info("Generating single-sample diagnostics for showcase "
                 "sample %d (flux err %.2f%%, closest to median %.2f%%).",
                 showcase_idx,
                 float(showcase_row["abs_flux_rel_error_pct"]),
                 median_flux_err)
        try:
            sc_result = _evaluate_sample(
                showcase_idx,
                ref_df=ref_df,
                dict_lookup=dict_lookup,
                dict_subset=dict_subset,
                metric_df=metric_df,
                metric_df_scaled=metric_df_scaled,
                metric_cols=metric_cols,
                metric_mode=metric_mode,
                metric_scale=metric_scale,
                score_metric=score_metric,
                rate_prefix=rate_prefix,
                eff_method=eff_method,
                global_rate_col=global_rate_col,
                z_tol=z_tol,
                flux_tol=flux_tol,
                eff_match_tol=eff_match_tol,
                exclude_self=exclude_self,
                join_col=join_col,
                scale_means=scale_means,
                scale_stds=scale_stds,
                emp_eff=emp_eff,
                interpolation_k=interpolation_k,
            )
            sc_cands = sc_result["candidates"]
            plot_cands = sc_cands.dropna(
                subset=["dict_flux_cm2_min", "dict_eff_1", "score_value"])
            if len(plot_cands) >= 3:
                _plot_contour(
                    plot_cands,
                    float(sc_result["sample_flux"]),
                    float(sc_result["sample_eff"]),
                    float(sc_result["best_flux"])
                        if np.isfinite(sc_result["best_flux"]) else None,
                    float(sc_result["best_eff"])
                        if np.isfinite(sc_result["best_eff"]) else None,
                    score_metric,
                    metric_mode,
                    sc_result["best_score"],
                    sc_result["truth_rank"],
                    len(sc_cands),
                    out_dir / f"all_showcase_contour_{tag}.png",
                    sc_result["events_label"],
                    sc_result["sample_label"],
                    top_n=top_n,
                )
            sv = sc_result["sample_vec_series"]
            bo = int(sc_result["best_orig_idx"])
            _plot_feature_diagnostics(
                sv.to_numpy(dtype=float),
                metric_df.loc[bo].to_numpy(dtype=float),
                sc_result["sample_vec_scaled"],
                metric_df_scaled.loc[bo].to_numpy(dtype=float),
                metric_cols,
                out_dir / f"all_showcase_feature_diagnostics_{tag}.png",
                f"Showcase sample #{showcase_idx} — feature diagnostics ({tag})",
            )
            _plot_score_distribution(
                sc_cands["score_value"], score_metric,
                sc_result["best_score"],
                out_dir / f"all_showcase_score_distribution_{tag}.png",
            )
            _plot_score_vs_param_distance(
                sc_cands,
                float(sc_result["sample_flux"]),
                float(sc_result["sample_eff"]),
                score_metric,
                out_dir / f"all_showcase_score_vs_param_dist_{tag}.png",
            )
            log.info("Showcase plots written for sample %d.", showcase_idx)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not generate showcase plots: %s", exc)

    log.info("Wrote all-mode results: %s", all_csv)
    log.info("Wrote successful rows: %s", ok_csv)
    log.info("Wrote failed rows: %s", failed_csv)
    log.info("Wrote summary JSON: %s", out_dir / 'all_samples_summary.json')
    log.info("Wrote plots to: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
