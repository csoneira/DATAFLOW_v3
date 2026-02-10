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
"""

from __future__ import annotations

import argparse
import ast
import json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = REPO_ROOT / "STEP_2_SIM_VALIDATION" / "output" / "filtered_reference.csv"
DEFAULT_OUT = REPO_ROOT / "STEP_3_SELF_CONSISTENCY" / "output"
DEFAULT_DICT = (
    REPO_ROOT
    / "STEP_1_DICTIONARY"
    / "output"
    / "task_01"
    / "param_metadata_dictionary.csv"
)
DEFAULT_CONFIG = REPO_ROOT / "STEP_3_SELF_CONSISTENCY" / "config.json"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_efficiencies(value: object) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return [float(parsed[0]), float(parsed[1]), float(parsed[2]), float(parsed[3])]
    return None


def _extract_eff_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "efficiencies" not in df.columns:
        return df
    effs = df["efficiencies"].apply(_parse_efficiencies)
    df["eff_1"] = effs.apply(lambda x: x[0] if x else np.nan)
    df["eff_2"] = effs.apply(lambda x: x[1] if x else np.nan)
    df["eff_3"] = effs.apply(lambda x: x[2] if x else np.nan)
    df["eff_4"] = effs.apply(lambda x: x[3] if x else np.nan)
    return df


# --- scoring functions ---

def _l2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.sum(diff ** 2)))


def _chi2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    y_t = y_true[mask]
    y_p = y_pred[mask]
    sigma2 = np.maximum(y_t, 1.0)
    return float(np.sum((y_t - y_p) ** 2 / sigma2))


def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    y_t = np.maximum(y_true[mask], 0.0)
    y_p = np.maximum(y_pred[mask], 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y_t > 0, y_t / y_p, 1.0)
        term = np.where(y_t > 0, y_t * np.log(ratio) - (y_t - y_p), y_p)
    return float(2.0 * np.sum(term))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    y_t = y_true[mask]
    y_p = y_pred[mask]
    denom = np.sum((y_t - y_t.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - float(np.sum((y_t - y_p) ** 2) / denom)


SCORE_FNS = {
    "l2": _l2_score,
    "chi2": _chi2_score,
    "poisson": _poisson_deviance,
    "r2": _r2_score,
}

# Lower is better for these metrics
LOWER_IS_BETTER = {"l2", "chi2", "poisson"}


def _select_metric_cols(df: pd.DataFrame, suffix: str, prefix: str | None) -> list[str]:
    cols = [col for col in df.columns if col.endswith(suffix)]
    if prefix:
        cols = [col for col in cols if col.startswith(prefix)]
    return sorted(cols)


def _compute_efficiency(
    n_four: pd.Series,
    n_three_missing: pd.Series,
    method: str,
) -> pd.Series:
    if method == "four_over_three_plus_four":
        return n_four / (n_four + n_three_missing)
    if method == "one_minus_three_over_four":
        return 1.0 - (n_three_missing / n_four.replace({0: np.nan}))
    raise ValueError(f"Unsupported efficiency method: {method}")


def _build_empirical_eff(df: pd.DataFrame, prefix: str, eff_method: str) -> pd.DataFrame:
    four_col = f"{prefix}_tt_1234_count"
    miss_cols = {
        1: f"{prefix}_tt_234_count",
        2: f"{prefix}_tt_134_count",
        3: f"{prefix}_tt_124_count",
        4: f"{prefix}_tt_123_count",
    }
    needed = [four_col, *miss_cols.values()]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for empirical efficiency: {missing}")
    out = pd.DataFrame(index=df.index)
    out["eff_est_p1"] = _compute_efficiency(df[four_col], df[miss_cols[1]], eff_method)
    out["eff_est_p2"] = _compute_efficiency(df[four_col], df[miss_cols[2]], eff_method)
    out["eff_est_p3"] = _compute_efficiency(df[four_col], df[miss_cols[3]], eff_method)
    out["eff_est_p4"] = _compute_efficiency(df[four_col], df[miss_cols[4]], eff_method)
    return out


def _find_join_col(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    for candidate in ("file_name", "filename_base"):
        if candidate in left.columns and candidate in right.columns:
            return candidate
    return None


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
) -> None:
    """Create a contour/heatmap of scores in (flux, efficiency) space."""
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
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _plot_feature_comparison(
    sample_vec: np.ndarray,
    best_vec: np.ndarray,
    feature_names: list,
    plot_path: Path,
    title: str,
) -> None:
    """Bar chart comparing sample vs best-candidate feature values."""
    mask = np.isfinite(sample_vec) & np.isfinite(best_vec)
    if mask.sum() < 1:
        return
    idx = np.where(mask)[0]
    names = [feature_names[i] for i in idx]
    sv = sample_vec[idx]
    bv = best_vec[idx]

    # Shorten long column names for display
    short_names = [n.replace("_rate_hz", "").replace("raw_tt_", "tt_")
                   .replace("events_per_second_", "") for n in names]

    x_pos = np.arange(len(short_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(short_names) * 0.8), 5))
    ax.bar(x_pos - width / 2, sv, width, label="Test sample", color="#4C78A8", alpha=0.85)
    ax.bar(x_pos + width / 2, bv, width, label="Best candidate", color="#F58518", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rate [Hz]")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
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


def _plot_score_histogram(
    scores: pd.Series,
    score_metric: str,
    plot_path: Path,
) -> None:
    scores = scores.dropna()
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores, bins=40, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.set_xlabel(f"Score ({score_metric})")
    ax.set_ylabel("Count")
    ax.set_title(f"Score distribution  (n={len(scores)})")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_scatter_sample_vs_best(
    sample_vec: np.ndarray,
    best_vec: np.ndarray,
    plot_path: Path,
    title: str,
) -> None:
    mask = np.isfinite(sample_vec) & np.isfinite(best_vec)
    if mask.sum() < 2:
        return
    x = best_vec[mask]
    y = sample_vec[mask]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=24, alpha=0.7, color="#72B7B2")
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    pad = 0.02 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
    ax.set_xlabel("Best-fit candidate rate")
    ax.set_ylabel("Test sample rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_top_n_param_space(
    candidates: pd.DataFrame,
    sample_flux: float,
    sample_eff: float,
    score_metric: str,
    top_n: int,
    plot_path: Path,
) -> None:
    """Show top-N candidates highlighted with rank labels on the (flux, eff) plane."""
    df = candidates.dropna(subset=["dict_flux_cm2_min", "dict_eff_1", "score_value"])
    if len(df) < 3:
        return
    ascending = score_metric in LOWER_IS_BETTER
    df_sorted = df.sort_values("score_value", ascending=ascending)
    top = df_sorted.head(top_n)
    rest = df_sorted.iloc[top_n:]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        rest["dict_flux_cm2_min"], rest["dict_eff_1"],
        s=20, alpha=0.35, color="#AAAAAA", edgecolors="none", zorder=2,
        label="Other candidates",
    )
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        ax.scatter(
            [row["dict_flux_cm2_min"]], [row["dict_eff_1"]],
            s=120, color=colors[rank - 1], edgecolors="black", linewidths=0.7,
            zorder=5,
        )
        ax.annotate(
            f"#{rank}", (row["dict_flux_cm2_min"], row["dict_eff_1"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8, fontweight="bold",
            color=colors[rank - 1],
        )
    ax.scatter(
        [sample_flux], [sample_eff], s=250, marker="*", color="#E45756",
        edgecolors="black", linewidths=0.8, zorder=6, label="True sample",
    )
    ax.set_xlabel("Flux [cm$^{-2}$ min$^{-1}$]")
    ax.set_ylabel("Efficiency (eff_1)")
    ax.set_title(f"Top-{top_n} candidates in parameter space ({score_metric})")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _plot_feature_residuals(
    sample_vec: np.ndarray,
    best_vec: np.ndarray,
    feature_names: list,
    plot_path: Path,
) -> None:
    """Bar chart of relative residuals (best - sample) / sample per feature."""
    mask = np.isfinite(sample_vec) & np.isfinite(best_vec) & (np.abs(sample_vec) > 1e-12)
    if mask.sum() < 1:
        return
    idx = np.where(mask)[0]
    residuals = (best_vec[idx] - sample_vec[idx]) / np.abs(sample_vec[idx]) * 100
    short = [feature_names[i].replace("_rate_hz", "").replace("raw_tt_", "tt_")
             .replace("events_per_second_", "") for i in idx]

    fig, ax = plt.subplots(figsize=(max(8, len(short) * 0.8), 5))
    colors = ["#E45756" if r > 0 else "#4C78A8" for r in residuals]
    ax.bar(np.arange(len(short)), residuals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative residual [%]")
    ax.set_title("Per-feature residual: (best − sample) / sample")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _plot_l2_contribution(
    sample_vec: np.ndarray,
    best_vec: np.ndarray,
    feature_names: list,
    plot_path: Path,
) -> None:
    """Horizontal bar chart showing each feature's contribution to the L2 score."""
    mask = np.isfinite(sample_vec) & np.isfinite(best_vec)
    if mask.sum() < 1:
        return
    idx = np.where(mask)[0]
    contrib = (sample_vec[idx] - best_vec[idx]) ** 2
    short = [feature_names[i].replace("_rate_hz", "").replace("raw_tt_", "tt_")
             .replace("events_per_second_", "") for i in idx]
    order = np.argsort(contrib)[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(short) * 0.35)))
    y_pos = np.arange(len(short))
    ax.barh(y_pos, contrib[order], color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short[i] for i in order], fontsize=9)
    ax.set_xlabel("Squared difference (scaled)")
    ax.set_title("Per-feature contribution to L2 distance")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _plot_top_n_profiles(
    metric_df: pd.DataFrame,
    candidates: pd.DataFrame,
    sample_vec: np.ndarray,
    feature_names: list,
    score_metric: str,
    top_n: int,
    plot_path: Path,
) -> None:
    """Overlay the rate profiles of the top-N candidates vs the sample."""
    df = candidates.dropna(subset=["score_value"])
    if len(df) < 2:
        return
    ascending = score_metric in LOWER_IS_BETTER
    top = df.sort_values("score_value", ascending=ascending).head(top_n)

    short = [n.replace("_rate_hz", "").replace("raw_tt_", "tt_")
             .replace("events_per_second_", "") for n in feature_names]
    x_pos = np.arange(len(short))

    fig, ax = plt.subplots(figsize=(max(9, len(short) * 0.9), 6))
    cmap = plt.cm.Blues(np.linspace(0.3, 0.85, len(top)))
    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        orig_idx = int(row["_orig_ref_idx"])
        vals = metric_df.loc[orig_idx].to_numpy(dtype=float)
        ax.plot(x_pos, vals, "o-", color=cmap[rank - 1], markersize=4,
                linewidth=1.2, alpha=0.7, label=f"#{rank} (s={row['score_value']:.3g})")
    ax.plot(x_pos, sample_vec, "s-", color="#E45756", markersize=6,
            linewidth=2, alpha=0.9, label="Test sample", zorder=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rate [Hz]")
    ax.set_title(f"Rate profiles: sample vs top-{len(top)} candidates")
    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _plot_score_cdf(
    scores: pd.Series,
    score_metric: str,
    best_score: float,
    plot_path: Path,
) -> None:
    """Cumulative distribution function of scores with best-score marker."""
    scores = scores.dropna().sort_values()
    if scores.empty:
        return
    cdf = np.arange(1, len(scores) + 1) / len(scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(scores.values, cdf, where="post", color="#4C78A8", linewidth=1.5)
    ax.axvline(best_score, color="#E45756", linestyle="--", linewidth=1.2,
               label=f"Best = {best_score:.4g}")
    ax.fill_between(scores.values, 0, cdf, alpha=0.12, color="#4C78A8", step="post")
    ax.set_xlabel(f"Score ({score_metric})")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"Score CDF  (n={len(scores)})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _plot_flux_eff_errors(
    candidates: pd.DataFrame,
    score_metric: str,
    plot_path: Path,
) -> None:
    """Scatter of flux error vs eff error for all candidates, colored by score."""
    df = candidates.dropna(subset=["flux_error", "eff_error", "score_value"])
    if len(df) < 3:
        return
    lower_is_better = score_metric in LOWER_IS_BETTER
    z = df["score_value"].to_numpy(dtype=float)
    cmap = "viridis_r" if lower_is_better else "viridis"

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["flux_error"] * 100, df["eff_error"] * 100,
        c=z, cmap=cmap, s=30, alpha=0.75, edgecolors="black", linewidths=0.3,
    )
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    fig.colorbar(sc, ax=ax, label=f"Score ({score_metric})", shrink=0.85)
    ax.set_xlabel("Flux error [%]")
    ax.set_ylabel("Efficiency error [%]")
    ax.set_title("Parameter errors colored by score")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


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
    parser.add_argument("--exclude-self", action="store_true",
                        help="Exclude the sample itself from candidates.")
    args = parser.parse_args()

    # ---- Merge config file + CLI ------------------------------------------
    config = _load_config(Path(args.config))

    def _cfg(attr, key, default, cast=str):
        val = getattr(args, attr, None)
        if val is None:
            val = config.get(key, default)
        return cast(val)

    reference_csv  = _cfg("reference_csv",  "reference_csv",  str(DEFAULT_REF))
    dictionary_csv = _cfg("dictionary_csv", "dictionary_csv", str(DEFAULT_DICT))
    out_dir_str    = _cfg("out_dir",        "out_dir",        str(DEFAULT_OUT))
    seed_raw       = _cfg("seed",           "seed",           123,   str)
    metric_suffix  = _cfg("metric_suffix",  "metric_suffix",  "_rate_hz")
    metric_prefix  = _cfg("metric_prefix",  "metric_prefix",  "raw_tt_")
    metric_mode    = _cfg("metric_mode",    "metric_mode",    "raw_tt_rates")
    global_rate_col = _cfg("global_rate_col", "global_rate_col", "events_per_second_global_rate")
    include_global = str(_cfg("include_global_rate", "include_global_rate", "true")).lower() in ("true", "1", "yes")
    eff_method     = _cfg("eff_method",     "eff_method",     "four_over_three_plus_four")
    rate_prefix    = _cfg("rate_prefix",    "rate_prefix",    "raw")
    metric_scale   = _cfg("metric_scale",   "metric_scale",   "zscore")
    score_metric   = _cfg("score_metric",   "score_metric",   "l2")
    cos_n_target   = _cfg("cos_n",          "cos_n",          2.0,  float)
    eff_tol        = _cfg("eff_tol",        "eff_tol",        1e-9, float)
    eff_match_tol  = _cfg("eff_match_tol",  "eff_match_tol",  1e-9, float)
    flux_tol       = _cfg("flux_tol",       "flux_tol",       1e-9, float)
    z_tol          = _cfg("z_tol",          "z_tol",          1e-6, float)
    max_sample_tries = _cfg("max_sample_tries", "max_sample_tries", 1000, int)
    top_n          = _cfg("top_n",          "top_n",          10, int)
    exclude_self   = args.exclude_self or config.get("exclude_self", False)
    sample_index_arg = args.sample_index  # None if not given

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean old plots
    for path in out_dir.glob("*.png"):
        path.unlink()

    # ---- Load data --------------------------------------------------------
    print(f"Loading reference: {reference_csv}")
    ref_df = pd.read_csv(reference_csv, low_memory=False)
    ref_df = _extract_eff_columns(ref_df)

    print(f"Loading dictionary: {dictionary_csv}")
    dict_df = pd.read_csv(dictionary_csv, low_memory=False)
    dict_df = _extract_eff_columns(dict_df)

    join_col = _find_join_col(ref_df, dict_df)
    if join_col is None:
        raise KeyError("No shared file_name/filename_base column for dictionary join.")

    # Bring dictionary parameters into reference
    dict_cols = [
        join_col, "flux_cm2_min", "cos_n",
        "eff_1", "eff_2", "eff_3", "eff_4",
        "selected_rows", "requested_rows",
    ]
    dict_cols = [col for col in dict_cols if col in dict_df.columns]
    dict_subset = dict_df[dict_cols].copy()
    dict_subset = dict_subset.rename(columns={
        "flux_cm2_min": "dict_flux_cm2_min",
        "cos_n":        "dict_cos_n",
        "eff_1":        "dict_eff_1",
        "eff_2":        "dict_eff_2",
        "eff_3":        "dict_eff_3",
        "eff_4":        "dict_eff_4",
        "selected_rows":  "dict_selected_rows",
        "requested_rows": "dict_requested_rows",
    })
    ref_with_dict = ref_df[[join_col]].join(
        dict_subset.set_index(join_col), on=join_col
    )

    # ---- Build metric_df (feature matrix for scoring) ----------------------
    print(f"Metric mode: {metric_mode}")
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
        # WARNING: this mode is unreliable because the sample's empirical
        # efficiency estimates are poor for asymmetric detector geometries.
        metric_cols = [
            "dict_eff_1", "dict_eff_2", "dict_eff_3", "dict_eff_4",
            global_rate_col,
        ]
        missing = [c for c in metric_cols[:-1] if c not in ref_with_dict.columns]
        if missing:
            raise KeyError(f"Missing columns for dict_eff_global: {missing}")
        metric_df = pd.concat(
            [ref_with_dict[metric_cols[:-1]], ref_df[[global_rate_col]]],
            axis=1,
        ).apply(pd.to_numeric, errors="coerce")
    else:
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    print(f"Feature columns ({len(metric_cols)}): {metric_cols}")

    if metric_df.isna().all(axis=None):
        raise ValueError("All metric columns are NaN after numeric coercion.")

    # ---- Scale features ---------------------------------------------------
    metric_df_scaled, scale_means, scale_stds = _scale_metrics(metric_df, metric_scale)

    # ---- Select sample row ------------------------------------------------
    if isinstance(seed_raw, str) and seed_raw.strip().lower() == "random":
        seed = None
    else:
        seed = int(seed_raw)
    rng = np.random.default_rng(seed)

    if sample_index_arg is not None:
        sample_idx = int(sample_index_arg)
    else:
        # Pick a sample with cos_n close to target and equal efficiencies
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
        eligible_idx = ref_with_dict.index[eligible_mask].to_numpy()
        if len(eligible_idx) == 0:
            print("WARNING: No eligible samples with exact cos_n and equal effs.")
            print("         Relaxing to cos_n +/-0.05 tolerance...")
            eligible_mask = (
                ref_with_dict["dict_cos_n"].notna()
                & (ref_with_dict["dict_cos_n"] - cos_n_target).abs().le(0.05)
                & ref_with_dict["dict_eff_1"].notna()
            )
            eligible_idx = ref_with_dict.index[eligible_mask].to_numpy()
        if len(eligible_idx) == 0:
            raise ValueError("No eligible samples found even with relaxed criteria.")
        sample_idx = int(rng.choice(eligible_idx))
        print(f"Eligible sample rows: {len(eligible_idx)}")

    if not (0 <= sample_idx < len(ref_df)):
        raise IndexError(f"Sample index {sample_idx} out of range (0..{len(ref_df) - 1}).")
    print(f"Sample index: {sample_idx}")

    # ---- Build sample feature vector (AFTER sample_idx is known) ----------
    if metric_mode == "dict_eff_global":
        # Use empirical efficiencies for the sample (this is the whole idea
        # of dict_eff_global mode, but beware: empirical eff estimation can
        # be very inaccurate)
        emp_eff = _build_empirical_eff(ref_df, rate_prefix, eff_method)
        sample_vec_series = pd.Series(
            [
                emp_eff.loc[sample_idx, "eff_est_p1"],
                emp_eff.loc[sample_idx, "eff_est_p2"],
                emp_eff.loc[sample_idx, "eff_est_p3"],
                emp_eff.loc[sample_idx, "eff_est_p4"],
                ref_df.loc[sample_idx, global_rate_col],
            ],
            index=metric_cols,
        )
        print(f"  Sample empirical effs: {[f'{v:.4f}' for v in sample_vec_series.values[:4]]}")
    else:
        # Apples-to-apples: sample vector from the same metric_df
        sample_vec_series = metric_df.loc[sample_idx]

    # ---- Get sample's true parameters from dictionary ---------------------
    sample_row = ref_df.iloc[sample_idx]
    sample_id = sample_row[join_col]
    dict_match = dict_df.loc[dict_df[join_col] == sample_id]
    if dict_match.empty:
        raise ValueError(f"No dictionary row found for sample {sample_id}.")
    dict_sample = dict_match.iloc[0]
    sample_flux = pd.to_numeric(dict_sample.get("flux_cm2_min"), errors="coerce")
    sample_eff = pd.to_numeric(dict_sample.get("eff_1"), errors="coerce")
    sample_cosn = pd.to_numeric(dict_sample.get("cos_n"), errors="coerce")
    sample_eff_2 = pd.to_numeric(dict_sample.get("eff_2"), errors="coerce")
    sample_eff_3 = pd.to_numeric(dict_sample.get("eff_3"), errors="coerce")
    sample_eff_4 = pd.to_numeric(dict_sample.get("eff_4"), errors="coerce")

    events_used = dict_sample.get("selected_rows")
    if pd.isna(events_used):
        events_used = dict_sample.get("requested_rows")
    events_label = "n/a" if pd.isna(events_used) else f"{int(events_used)}"
    sample_label = dict_sample.get("filename_base", dict_sample.get("file_name", "sample"))

    print(f"  True flux = {sample_flux:.4g}")
    print(f"  True eff  = [{sample_eff:.4f}, {sample_eff_2:.4f}, {sample_eff_3:.4f}, {sample_eff_4:.4f}]")
    print(f"  cos_n     = {sample_cosn:.4f}")

    # ---- Filter candidates by z-positions ---------------------------------
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

    candidates = ref_df.loc[z_mask].copy()
    if candidates.empty:
        raise ValueError("No candidates after z-position filtering.")
    print(f"Candidates (same z-planes): {len(candidates)}")

    # Save original ref_df indices BEFORE merge (merge resets them)
    orig_ref_indices = candidates.index.to_numpy().copy()

    # Attach dictionary parameters
    candidates = candidates.merge(dict_subset, on=join_col, how="left")

    # Store the original ref_df index as a column so we can look up
    # the correct row in metric_df_scaled during scoring
    candidates["_orig_ref_idx"] = orig_ref_indices
    if candidates["dict_flux_cm2_min"].isna().all():
        raise ValueError("Dictionary flux values missing for all candidates.")

    # ---- Score all candidates ---------------------------------------------
    sample_vec = _scale_vector(sample_vec_series, metric_scale, scale_means, scale_stds)
    if np.isfinite(sample_vec).sum() < 2:
        raise ValueError("Sample metric vector has fewer than 2 finite values.")

    score_fn = SCORE_FNS.get(score_metric)
    if score_fn is None:
        raise ValueError(f"Unknown score_metric: {score_metric}")

    score_values = []
    for i, idx in enumerate(candidates.index):
        orig_idx = candidates.loc[idx, "_orig_ref_idx"]
        cand_vec = metric_df_scaled.loc[orig_idx].to_numpy(dtype=float)
        score_values.append(score_fn(sample_vec, cand_vec))
    candidates["score_value"] = score_values

    # ---- Find best candidate ----------------------------------------------
    scores_valid = candidates["score_value"].dropna()
    if scores_valid.empty:
        raise ValueError("All candidates have NaN scores.")

    if score_metric in LOWER_IS_BETTER:
        best_idx = scores_valid.idxmin()
    else:
        best_idx = scores_valid.idxmax()

    best_row = candidates.loc[best_idx]
    best_orig_idx = int(best_row["_orig_ref_idx"])
    best_vec_raw = metric_df.loc[best_orig_idx].to_numpy(dtype=float)
    best_flux = pd.to_numeric(best_row.get("dict_flux_cm2_min"), errors="coerce")
    best_eff = pd.to_numeric(best_row.get("dict_eff_1"), errors="coerce")
    best_score = float(scores_valid.loc[best_idx])

    # ---- Parameter error diagnostics --------------------------------------
    candidates["flux_error"] = candidates["dict_flux_cm2_min"].astype(float) - float(sample_flux)
    candidates["eff_error"] = candidates["dict_eff_1"].astype(float) - float(sample_eff)
    if float(sample_flux) != 0:
        candidates["flux_rel_error"] = candidates["flux_error"] / float(sample_flux)
    else:
        candidates["flux_rel_error"] = np.nan
    if float(sample_eff) != 0:
        candidates["eff_rel_error"] = candidates["eff_error"] / float(sample_eff)
    else:
        candidates["eff_rel_error"] = np.nan

    # ---- Rank of truth (how well does the method recover ground truth?) ----
    flux_match = (candidates["dict_flux_cm2_min"].astype(float) - float(sample_flux)).abs().le(flux_tol)
    eff_match_mask = (candidates["dict_eff_1"].astype(float) - float(sample_eff)).abs().le(eff_match_tol)
    truth_mask = flux_match & eff_match_mask
    truth_rank = -1
    if truth_mask.any():
        truth_score = candidates.loc[truth_mask, "score_value"].iloc[0]
        if score_metric in LOWER_IS_BETTER:
            truth_rank = int((candidates["score_value"] < truth_score).sum()) + 1
        else:
            truth_rank = int((candidates["score_value"] > truth_score).sum()) + 1
    print(f"\n  Best candidate: flux={best_flux:.4g}, eff={best_eff:.4g}")
    print(f"  Best score ({score_metric}): {best_score:.4g}")
    print(f"  Flux error:  {best_flux - sample_flux:.4g}  ({(best_flux - sample_flux) / sample_flux * 100:.2f}%)")
    print(f"  Eff error:   {best_eff - sample_eff:.4g}  ({(best_eff - sample_eff) / sample_eff * 100:.2f}%)")
    if truth_rank > 0:
        print(f"  Truth rank: {truth_rank} / {len(candidates)}")
    else:
        print("  Truth row not found among candidates (sample may be excluded).")

    # ---- Export CSVs ------------------------------------------------------
    candidates_csv = out_dir / "r2_candidates.csv"
    candidates.to_csv(candidates_csv, index=False)

    ascending = score_metric in LOWER_IS_BETTER
    if top_n > 0:
        top_df = candidates.sort_values("score_value", ascending=ascending).head(top_n)
        top_df.to_csv(out_dir / "top_candidates.csv", index=False)

    # ---- Summary JSON -----------------------------------------------------
    summary = {
        "sample_index": sample_idx,
        "sample_file": sample_id,
        "sample_flux": float(sample_flux),
        "sample_eff": float(sample_eff),
        "sample_cos_n": float(sample_cosn),
        "candidate_rows": int(len(candidates)),
        "metric_mode": metric_mode,
        "metric_scale": metric_scale,
        "score_metric": score_metric,
        "metric_columns": metric_cols,
        "exclude_self": bool(exclude_self),
        "best_index": int(best_idx),
        "best_score": best_score,
        "best_file": best_row.get(join_col),
        "best_flux_cm2_min": float(best_flux) if np.isfinite(best_flux) else None,
        "best_eff": float(best_eff) if np.isfinite(best_eff) else None,
        "flux_error": float(best_flux - sample_flux),
        "flux_rel_error_pct": float((best_flux - sample_flux) / sample_flux * 100)
            if sample_flux != 0 else None,
        "eff_error": float(best_eff - sample_eff),
        "eff_rel_error_pct": float((best_eff - sample_eff) / sample_eff * 100)
            if sample_eff != 0 else None,
        "truth_rank": truth_rank,
        "truth_rank_total": int(len(candidates)),
    }
    with (out_dir / "r2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # ---- Plots ------------------------------------------------------------
    tag = f"{metric_mode}_{score_metric}"

    # 1. Contour/heatmap in (flux, eff) space
    plot_cands = candidates.dropna(subset=["dict_flux_cm2_min", "dict_eff_1", "score_value"])
    if len(plot_cands) >= 3:
        _plot_contour(
            plot_cands,
            float(sample_flux), float(sample_eff),
            float(best_flux) if np.isfinite(best_flux) else None,
            float(best_eff) if np.isfinite(best_eff) else None,
            score_metric, metric_mode, best_score,
            truth_rank, len(candidates),
            out_dir / f"contour_{tag}.png",
            events_label, sample_label,
        )
    else:
        print("WARNING: Too few candidates for contour plot.")

    # 2. Feature comparison bar chart (sample vs best)
    _plot_feature_comparison(
        sample_vec_series.to_numpy(dtype=float),
        metric_df.loc[best_orig_idx].to_numpy(dtype=float),
        metric_cols,
        out_dir / f"features_{tag}.png",
        f"Feature comparison: sample vs best ({tag})",
    )

    # 3. Scatter: sample rate vs best-candidate rate
    _plot_scatter_sample_vs_best(
        sample_vec_series.to_numpy(dtype=float),
        best_vec_raw,
        out_dir / f"scatter_sample_vs_best_{tag}.png",
        f"Test sample vs best-fit ({tag})",
    )

    # 4. Score histogram
    _plot_score_histogram(
        candidates["score_value"],
        score_metric,
        out_dir / f"hist_score_{tag}.png",
    )

    # 5. Score vs parameter distance
    _plot_score_vs_param_distance(
        candidates, float(sample_flux), float(sample_eff),
        score_metric,
        out_dir / f"score_vs_param_dist_{tag}.png",
    )

    # 6. Top-N candidates highlighted in parameter space
    _plot_top_n_param_space(
        candidates, float(sample_flux), float(sample_eff),
        score_metric, top_n,
        out_dir / f"top_n_param_space_{tag}.png",
    )

    # 7. Per-feature relative residual (best vs sample)
    _plot_feature_residuals(
        sample_vec_series.to_numpy(dtype=float),
        metric_df.loc[best_orig_idx].to_numpy(dtype=float),
        metric_cols,
        out_dir / f"residuals_{tag}.png",
    )

    # 8. Per-feature L2 contribution (scaled space)
    _plot_l2_contribution(
        sample_vec,  # already scaled
        metric_df_scaled.loc[best_orig_idx].to_numpy(dtype=float),
        metric_cols,
        out_dir / f"l2_contribution_{tag}.png",
    )

    # 9. Rate profile overlay: sample vs top-N candidates
    _plot_top_n_profiles(
        metric_df, candidates,
        sample_vec_series.to_numpy(dtype=float),
        metric_cols, score_metric, top_n,
        out_dir / f"profiles_top_n_{tag}.png",
    )

    # 10. Score CDF with best-score marker
    _plot_score_cdf(
        candidates["score_value"], score_metric, best_score,
        out_dir / f"score_cdf_{tag}.png",
    )

    # 11. Flux error vs eff error scatter colored by score
    _plot_flux_eff_errors(
        candidates, score_metric,
        out_dir / f"flux_eff_errors_{tag}.png",
    )

    print(f"\nWrote candidates: {candidates_csv}")
    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
