#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/PURELY_LINEAR/try_purely_linear_transform.py
Purpose: Calibrate a compact affine inverse from (global_rate, eff) to flux.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/PURELY_LINEAR/try_purely_linear_transform.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.tri import LinearTriInterpolator, Triangulation


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[2]
PLOTS = HERE / "PLOTS"

# Keep this explicit and simple: dictionary_test currently matches the target
# synthetic series domain better than older dictionary snapshots.
TRAIN_CSV_CANDIDATES = [
    HERE.parent / "dictionary_test.csv",
    Path(
        str(PROJECT_DIR / "STEPS" / "STEP_1_SETUP" /
            "STEP_1_2_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "dictionary.csv")
    ),
    Path(
        "/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_1_SETUP/"
        "STEP_1_2_BUILD_DICTIONARY/OUTPUTS/FILES/dictionary.csv"
    ),  # legacy path
]

TARGET_CSV_CANDIDATES = [
    HERE.parents[1] / "the_simulated_file.csv",
    Path(
        str(PROJECT_DIR / "STEPS" / "STEP_3_SYNTHETIC_TIME_SERIES" /
            "STEP_3_1_TIME_SERIES_CREATION" / "OUTPUTS" / "FILES" / "complete_curve_time_series.csv")
    ),
    Path(
        str(PROJECT_DIR / "STEPS" / "STEP_3_SYNTHETIC_TIME_SERIES" /
            "STEP_3_2_SYNTHETIC_TIME_SERIES" / "OUTPUTS" / "FILES" / "synthetic_dataset.csv")
    ),
    Path(
        "/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_1_TIME_SERIES_CREATION/OUTPUTS/FILES/"
        "complete_curve_time_series.csv"
    ),  # legacy path
    # HERE / "dictionary_test.csv",
]

K_NEIGHBORS = 10
MAD_MULTIPLIER = 2.0
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit affine inverse mapping from (global_rate, eff) to flux."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Explicit training CSV path. If omitted, use built-in candidate search.",
    )
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=None,
        help="Explicit target CSV path. If omitted, use built-in candidate search.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS,
        help="Output directory for generated plots (default: script PLOTS/).",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=K_NEIGHBORS,
        help="Neighborhood size for local gradient estimation (default: 10).",
    )
    return parser.parse_args()

@dataclass
class AffineInverse:
    a_flux: float
    b_eff: float
    c_bias: float

    @property
    def m11(self) -> float:
        return 1.0 / self.a_flux

    @property
    def m12(self) -> float:
        return -self.b_eff / self.a_flux

    @property
    def t1(self) -> float:
        return -self.c_bias / self.a_flux

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.m11, self.m12], [0.0, 1.0]], dtype=float)

    @property
    def offset(self) -> np.ndarray:
        return np.array([self.t1, 0.0], dtype=float)

    def estimate_flux(self, rate: np.ndarray, eff: np.ndarray) -> np.ndarray:
        return self.m11 * rate + self.m12 * eff + self.t1


@dataclass
class CanonicalMeta:
    flux_col: str
    eff_col: str
    rate_col: str
    time_col: str


def prepare_output_dir() -> None:
    if PLOTS.exists():
        shutil.rmtree(PLOTS)
    PLOTS.mkdir(parents=True, exist_ok=True)


def first_existing(paths: list[Path], label: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No {label} CSV found. Checked:\n" + "\n".join(str(p) for p in paths))


def pick_flux(df: pd.DataFrame) -> tuple[str, pd.Series]:
    for col in ("flux", "flux_cm2_min"):
        if col in df.columns:
            return col, pd.to_numeric(df[col], errors="coerce")
    raise RuntimeError("Missing flux column (tried flux, flux_cm2_min).")


def pick_rate(df: pd.DataFrame) -> tuple[str, pd.Series]:
    for col in (
        "global_rate_hz",
        "events_per_second_global_rate",
        "clean_tt_1234_rate_hz",
        "raw_tt_1234_rate_hz",
    ):
        if col in df.columns:
            return col, pd.to_numeric(df[col], errors="coerce")
    raise RuntimeError("Missing global-rate column.")


def pick_eff(df: pd.DataFrame, prefer_sim: bool) -> tuple[str, pd.Series]:
    if prefer_sim:
        order = ("eff_sim_1", "eff", "eff_empirical_1")
    else:
        order = ("eff", "eff_sim_1", "eff_empirical_1")

    for col in order:
        if col in df.columns:
            return col, pd.to_numeric(df[col], errors="coerce")

    sim_cols = [c for c in df.columns if c.startswith("eff_sim_")]
    if sim_cols:
        return "eff_sim_mean", df[sim_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    emp_cols = [c for c in df.columns if c.startswith("eff_empirical_")]
    if emp_cols:
        return "eff_empirical_mean", df[emp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    raise RuntimeError("Missing efficiency column.")


def pick_time(df: pd.DataFrame) -> tuple[str, pd.Series]:
    for col in ("time_utc", "execution_timestamp", "param_date"):
        if col not in df.columns:
            continue
        raw = df[col]
        if col == "execution_timestamp":
            t = pd.to_datetime(raw, format="%Y-%m-%d_%H.%M.%S", errors="coerce")
            if t.notna().sum() > 0:
                return col, t
        if col == "param_date":
            t = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
            if t.notna().sum() > 0:
                return col, t
        return col, pd.to_datetime(raw, errors="coerce")

    return "synthetic_time", pd.Series(pd.date_range("2000-01-01", periods=len(df), freq="min"))


def canonicalize(csv_path: Path, prefer_sim_eff: bool) -> tuple[pd.DataFrame, CanonicalMeta]:
    df = pd.read_csv(csv_path)

    flux_col, flux = pick_flux(df)
    eff_col, eff = pick_eff(df, prefer_sim=prefer_sim_eff)
    rate_col, rate = pick_rate(df)
    time_col, time = pick_time(df)

    out = pd.DataFrame({"flux": flux, "eff": eff, "global_rate_hz": rate, "time_utc": time})
    mask = np.isfinite(out["flux"]) & np.isfinite(out["eff"]) & np.isfinite(out["global_rate_hz"])
    out = out.loc[mask].copy()

    if out["time_utc"].notna().sum() < max(3, int(0.5 * len(out))):
        out["time_utc"] = pd.date_range("2000-01-01", periods=len(out), freq="min")
    else:
        out["time_utc"] = out["time_utc"].ffill().bfill()

    out = out.sort_values("time_utc", kind="mergesort").reset_index(drop=True)
    return out, CanonicalMeta(flux_col=flux_col, eff_col=eff_col, rate_col=rate_col, time_col=time_col)


def local_gradients(df: pd.DataFrame, k_neighbors: int = K_NEIGHBORS) -> pd.DataFrame:
    x = df["flux"].to_numpy(dtype=float)
    y = df["eff"].to_numpy(dtype=float)
    z = df["global_rate_hz"].to_numpy(dtype=float)

    n = len(df)
    if n < 4:
        raise RuntimeError("Need at least 4 points for local gradient estimation.")

    k = max(4, min(k_neighbors, n))
    xy = np.column_stack([x, y])
    mu = np.nanmean(xy, axis=0)
    sd = np.nanstd(xy, axis=0)
    sd = np.where(np.abs(sd) < EPS, 1.0, sd)
    xy_n = (xy - mu) / sd

    gx = np.full(n, np.nan, dtype=float)
    gy = np.full(n, np.nan, dtype=float)

    for i in range(n):
        d2 = np.sum((xy_n - xy_n[i]) ** 2, axis=1)
        idx = np.argsort(d2)[:k]
        A = np.column_stack([x[idx], y[idx], np.ones(k)])
        a_i, b_i, _ = np.linalg.lstsq(A, z[idx], rcond=None)[0]
        gx[i] = a_i
        gy[i] = b_i

    return pd.DataFrame({"grad_drate_dflux": gx, "grad_drate_deff": gy})


def filter_gradients(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    tan = np.divide(gy, gx, out=np.full_like(gy, np.nan), where=np.abs(gx) > EPS)
    finite = np.isfinite(tan)
    if finite.sum() < 4:
        raise RuntimeError("Not enough finite tan(angle) values.")

    med = float(np.nanmedian(tan))
    mad = float(np.nanmedian(np.abs(tan[finite] - med)))
    if mad > EPS:
        thr = MAD_MULTIPLIER * mad
    else:
        thr = float(np.nanstd(tan[finite]))

    keep = finite & (np.abs(tan - med) <= thr)
    if keep.sum() < max(5, int(0.25 * finite.sum())):
        q1, q3 = np.nanpercentile(tan[finite], [25, 75])
        iqr = float(q3 - q1)
        thr = 1.5 * iqr if iqr > EPS else np.inf
        keep = finite & (np.abs(tan - med) <= thr)

    return tan, keep, med, mad, float(thr)


def fit_affine_inverse(df: pd.DataFrame, keep_mask: np.ndarray) -> AffineInverse:
    x = df["flux"].to_numpy(dtype=float)
    e = df["eff"].to_numpy(dtype=float)
    r = df["global_rate_hz"].to_numpy(dtype=float)

    A = np.column_stack([x[keep_mask], e[keep_mask], np.ones(int(keep_mask.sum()))])
    a, b, c = [float(v) for v in np.linalg.lstsq(A, r[keep_mask], rcond=None)[0]]
    if abs(a) < EPS:
        raise RuntimeError("Degenerate fit: coefficient a is too small for inversion.")
    return AffineInverse(a_flux=a, b_eff=b, c_bias=c)


def compute_metrics(flux_true: np.ndarray, flux_est: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(flux_true) & np.isfinite(flux_est)
    if valid.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    err = flux_est[valid] - flux_true[valid]
    corr = float(np.corrcoef(flux_true[valid], flux_est[valid])[0, 1])
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return corr, rmse, mae, bias


def format_time_axis(ax: plt.Axes) -> None:
    loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


def plot_tan_histogram(tan: np.ndarray, keep: np.ndarray, med: float, thr: float) -> None:
    finite = np.isfinite(tan)
    vals_keep = tan[finite & keep]
    vals_rej = tan[finite & ~keep]
    vals_all = tan[finite]

    # compute MAD for reporting and kept count
    mad = float(np.nanmedian(np.abs(vals_all - med))) if vals_all.size else float("nan")
    kept_count = int(np.count_nonzero(finite & keep))

    # use thinner histogram bins for more detail
    bins = np.linspace(float(np.nanmin(vals_all)), float(np.nanmax(vals_all)), 80)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if vals_rej.size:
        ax.hist(vals_rej, bins=bins, color="0.75", alpha=0.65, edgecolor="white", label="rejected")
    ax.hist(vals_keep, bins=bins, color="C2", alpha=0.8, edgecolor="white", label="kept")
    ax.axvline(med, color="k", ls="--", lw=1.2, label="median")
    ax.axvline(med - thr, color="C1", ls=":", lw=1.1, label="keep bounds")
    ax.axvline(med + thr, color="C1", ls=":", lw=1.1)

    # summary textbox with filter statistics
    txt = f"median={med:.3f}\nMAD={mad:.3f}\nthr={thr:.3f}\nkept={kept_count}"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.85))

    ax.set_title("Gradient-direction filter (tan(angle))")
    ax.set_xlabel("tan(angle) = (dRate/dEff)/(dRate/dFlux)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "04_tan_angle_histogram.png", dpi=170)
    plt.close(fig)


def plot_iso_quiver(df: pd.DataFrame, grads: pd.DataFrame, keep: np.ndarray) -> None:
    x = df["flux"].to_numpy(dtype=float)
    e = df["eff"].to_numpy(dtype=float)
    r = df["global_rate_hz"].to_numpy(dtype=float)
    gx = grads["grad_drate_dflux"].to_numpy(dtype=float)
    gy = grads["grad_drate_deff"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(e) & np.isfinite(r) & np.isfinite(gx) & np.isfinite(gy)
    if finite.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    tri = Triangulation(x[finite], e[finite])
    interp = LinearTriInterpolator(tri, r[finite])
    xi = np.linspace(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])), 240)
    yi = np.linspace(float(np.nanmin(e[finite])), float(np.nanmax(e[finite])), 240)
    Xi, Yi = np.meshgrid(xi, yi)
    Ri = np.asarray(interp(Xi, Yi), dtype=float)

    r_min = float(np.nanmin(r[finite]))
    r_max = float(np.nanmax(r[finite]))
    fine_levels = np.linspace(r_min, r_max, 16)
    cf = ax.contourf(Xi, Yi, Ri, levels=fine_levels, cmap="viridis", alpha=0.16)
    int_levels = np.arange(int(np.ceil(r_min)), int(np.floor(r_max)) + 1, 1, dtype=float)
    if int_levels.size >= 1:
        cs = ax.contour(Xi, Yi, Ri, levels=int_levels, colors="0.18", linewidths=1.2, alpha=0.95)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%d Hz")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Global rate [Hz]")

    # Faint local gradients
    norm = np.sqrt(gx[finite] ** 2 + gy[finite] ** 2)
    norm = np.where(norm < EPS, 1.0, norm)
    ax.quiver(
        x[finite],
        e[finite],
        gx[finite] / norm,
        gy[finite] / norm,
        angles="xy",
        scale_units="xy",
        scale=43,
        color="0.35",
        alpha=0.25,
        width=0.0028,
    )

    # Chosen mean gradient (kept points only)
    kept = finite & keep
    gmx = float("nan")
    gmy = float("nan")
    if kept.sum() >= 3:
        gmx = float(np.nanmean(gx[kept]))
        gmy = float(np.nanmean(gy[kept]))
        gnorm = float(np.hypot(gmx, gmy))
        if gnorm > EPS:
            ux, uy = gmx / gnorm, gmy / gnorm
            x0 = float(np.nanmedian(x[kept]))
            e0 = float(np.nanmedian(e[kept]))
            span = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), float(np.nanmax(e[finite]) - np.nanmin(e[finite])))
            a_scale = 0.18 * span
            ax.quiver([x0], [e0], [ux], [uy], angles="xy", scale_units="xy", scale=1.0 / a_scale, color="C0", width=0.008)
            ax.scatter([x0], [e0], s=26, color="C0", zorder=6)

    # add a small summary textbox with filter stats + mean local gradient (kept)
    try:
        tan = np.divide(gy, gx, out=np.full_like(gy, np.nan), where=np.abs(gx) > EPS)
        med_tan = float(np.nanmedian(tan[np.isfinite(tan)]))
        mad_tan = float(np.nanmedian(np.abs(tan[np.isfinite(tan)] - med_tan)))
    except Exception:
        med_tan = float("nan")
        mad_tan = float("nan")
    kept_count = int(np.count_nonzero(kept))
    if kept_count > 0:
        mean_local = f"({gmx:.3f},{gmy:.3f})"
    else:
        mean_local = "(nan,nan)"
    info = f"median tan={med_tan:.3f}\nMAD={mad_tan:.3f}\nkept={kept_count}\nmean local grad={mean_local}"
    ax.text(0.98, 0.03, info, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=0.85))

    ax.scatter(x[finite & ~keep], e[finite & ~keep], s=16, c="0.7", alpha=0.8, label="filtered")
    ax.scatter(x[finite & keep], e[finite & keep], s=20, c="white", edgecolors="0.25", linewidths=0.5, label="kept")
    ax.set_title("Iso-rate map + local gradients")
    ax.set_xlabel("Flux")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.2)

    # preserve geometric angles so gradient quivers are visually perpendicular
    ax.set_xlim(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])))
    ax.set_ylim(float(np.nanmin(e[finite])), float(np.nanmax(e[finite])))
    ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS / "02_local_gradients_quiver.png", dpi=170)
    plt.close(fig)


def plot_linearized_field(model: AffineInverse, df: pd.DataFrame, keep: np.ndarray) -> None:
    """Iso-rate map styled like `plot_iso_quiver` but overlaying the affine linearized field.

    - uses triangulation + contourf of the *actual* global-rate field for background
    - draws faint local gradient quivers (same style as `plot_iso_quiver`)
    - overlays straight, parallel affine iso-rate lines clipped to the plotting limits
    - keeps the same colourbar, limits and point styling as `02_local_gradients_quiver.png`
    """
    x = df["flux"].to_numpy(dtype=float)
    e = df["eff"].to_numpy(dtype=float)
    r = df["global_rate_hz"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(e) & np.isfinite(r)
    if finite.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    # --- triangulate & show actual iso-rate map (same style as plot_iso_quiver)
    tri = Triangulation(x[finite], e[finite])
    interp = LinearTriInterpolator(tri, r[finite])
    xi = np.linspace(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])), 240)
    yi = np.linspace(float(np.nanmin(e[finite])), float(np.nanmax(e[finite])), 240)
    Xi, Yi = np.meshgrid(xi, yi)
    Ri = np.asarray(interp(Xi, Yi), dtype=float)

    r_min = float(np.nanmin(r[finite]))
    r_max = float(np.nanmax(r[finite]))
    fine_levels = np.linspace(r_min, r_max, 16)
    cf = ax.contourf(Xi, Yi, Ri, levels=fine_levels, cmap="viridis", alpha=0.16)
    int_levels = np.arange(int(np.ceil(r_min)), int(np.floor(r_max)) + 1, 1, dtype=float)
    if int_levels.size >= 1:
        cs = ax.contour(Xi, Yi, Ri, levels=int_levels, colors="0.18", linewidths=1.2, alpha=0.95)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%d Hz")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Global rate [Hz]")

    # --- faint local gradients (recompute to match style)
    grads = local_gradients(df)
    gx = grads["grad_drate_dflux"].to_numpy(dtype=float)
    gy = grads["grad_drate_deff"].to_numpy(dtype=float)
    finite_grad = finite & np.isfinite(gx) & np.isfinite(gy)
    if finite_grad.sum() >= 3:
        norm = np.sqrt(gx[finite_grad] ** 2 + gy[finite_grad] ** 2)
        norm = np.where(norm < EPS, 1.0, norm)
        ax.quiver(
            x[finite_grad],
            e[finite_grad],
            gx[finite_grad] / norm,
            gy[finite_grad] / norm,
            angles="xy",
            scale_units="xy",
            scale=43,
            color="0.35",
            alpha=0.25,
            width=0.0028,
        )

    # --- blue gradient vector: use the *affine-model* gradient (a, b) so it is
    # guaranteed perpendicular to the affine iso-lines drawn above.
    # Keep 'kept' mask for placement, but derive direction from the model.
    kept = finite & keep
    if kept.sum() >= 1:
        gmx = float(model.a_flux)  # dRate/dFlux from fitted affine model
        gmy = float(model.b_eff)   # dRate/dEff from fitted affine model
        gnorm = float(np.hypot(gmx, gmy))
        if gnorm > EPS:
            ux, uy = gmx / gnorm, gmy / gnorm
            x0 = float(np.nanmedian(x[kept]))
            e0 = float(np.nanmedian(e[kept]))
            span = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), float(np.nanmax(e[finite]) - np.nanmin(e[finite])))
            a_scale = 0.18 * span
            ax.quiver([x0], [e0], [ux], [uy], angles="xy", scale_units="xy", scale=1.0 / a_scale, color="C0", width=0.008)
            ax.scatter([x0], [e0], s=26, color="C0", zorder=6)
            # add a small text note showing the model gradient components
            ax.text(x0, e0 - 0.035 * span, f"model grad=({gmx:.2f},{gmy:.2f})", color="C0", fontsize=8, ha="center")

    # --- affine-model iso-rate lines (linearized field) clipped to plotting limits
    # reuse the same `yi` vertical sampling used for the background
    eff_lin = yi
    # Prefer integer iso-lines starting at 9 Hz (9, 10, 11, ...) up to floor(r_max).
    # Do NOT remove integers below r_min — draw 9..floor(r_max) if possible so the
    # requested labelled lines (9,10,11,...) are always present when meaningful.
    if r_max - r_min < 1e-6:
        levels = np.array([r_min])
    else:
        if int(np.floor(r_max)) >= 9:
            levels = np.arange(9, int(np.floor(r_max)) + 1, dtype=float)
        else:
            # fallback: integer ticks that cover the observed span
            levels = np.arange(int(np.floor(r_min)), int(np.ceil(r_max)) + 1, dtype=float)
        levels = np.unique(levels)

    # compute flux positions for all integer iso-lines and determine plot x-limits
    flux_lines = [((rl - model.b_eff * eff_lin - model.c_bias) / model.a_flux) for rl in levels]
    # find finite extents of all computed lines
    finite_flux_vals = np.hstack([f[np.isfinite(f)] for f in flux_lines if np.isfinite(f).any()]) if flux_lines else np.array([])
    if finite_flux_vals.size:
        min_flux_line = float(np.nanmin(finite_flux_vals))
        max_flux_line = float(np.nanmax(finite_flux_vals))
    else:
        min_flux_line = float(np.nanmin(x[finite]))
        max_flux_line = float(np.nanmax(x[finite]))

    # expand x-limits to include the affine lines so they become visible
    x_data_min = float(np.nanmin(x[finite]))
    x_data_max = float(np.nanmax(x[finite]))
    x_min_plot = min(x_data_min, min_flux_line)
    x_max_plot = max(x_data_max, max_flux_line)
    # add a small margin
    xpad = max(1e-9, 0.03 * (x_max_plot - x_min_plot))
    x_min_plot -= xpad
    x_max_plot += xpad

    # plot each affine iso-line and add an individual label for every integer level
    first_line = True
    eff_span = float(np.nanmax(e[finite]) - np.nanmin(e[finite]))
    eff_positions = np.linspace(float(np.nanmin(e[finite])) + 0.1 * eff_span, float(np.nanmax(e[finite])) - 0.1 * eff_span, max(1, len(levels)))

    for i, rl in enumerate(levels):
        flux_line = flux_lines[i]
        # only plot finite points and within the extended plotting x-range
        mask = np.isfinite(flux_line) & (flux_line >= x_min_plot) & (flux_line <= x_max_plot)
        if mask.any():
            lbl = "affine iso-lines" if first_line else "_nolegend_"
            ax.plot(flux_line[mask], eff_lin[mask], color="C3", lw=1.4, alpha=0.95, label=lbl)
            first_line = False

        # place a small label for the integer level at a convenient efficiency
        eff_label = eff_positions[min(i, len(eff_positions) - 1)]
        flux_label = (rl - model.b_eff * eff_label - model.c_bias) / model.a_flux
        if x_min_plot <= flux_label <= x_max_plot and np.isfinite(flux_label):
            ax.text(
                flux_label,
                eff_label,
                f"{int(round(rl))} Hz",
                color="C3",
                fontsize=8,
                backgroundcolor=("white"),
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
            )

    # (no separate median-only annotation — each integer line is labelled)
    # highlight points (same style as iso_quiver)
    ax.scatter(x[finite & ~keep], e[finite & ~keep], s=16, c="0.7", alpha=0.8, label="filtered")
    ax.scatter(x[finite & keep], e[finite & keep], s=20, c="white", edgecolors="0.25", linewidths=0.5, label="kept")

    # compact summary box (evidence): kept count, tan stats, mean local gradient and model coeffs
    try:
        tan_all = np.divide(gy, gx, out=np.full_like(gy, np.nan), where=np.abs(gx) > EPS)
        med_tan = float(np.nanmedian(tan_all[np.isfinite(tan_all)]))
        mad_tan = float(np.nanmedian(np.abs(tan_all[np.isfinite(tan_all)] - med_tan)))
    except Exception:
        med_tan = float('nan')
        mad_tan = float('nan')
    kept_count = int(np.count_nonzero(finite & keep))
    try:
        mean_local_gx = float(np.nanmean(gx[finite & keep]))
        mean_local_gy = float(np.nanmean(gy[finite & keep]))
    except Exception:
        mean_local_gx = float('nan')
        mean_local_gy = float('nan')

    summary = (
        f"kept={kept_count}\nmedian tan={med_tan:.3f}\nMAD={mad_tan:.3f}\n"
        f"mean local grad=({mean_local_gx:.3f},{mean_local_gy:.3f})\n"
        f"model (a,b,c)=({model.a_flux:.3f},{model.b_eff:.3f},{model.c_bias:.3f})"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.9))

    ax.set_title("Iso-rate map + affine linearized field")
    ax.set_xlabel("Flux")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.2)

    # enforce the same axis limits as the interpolation grid
    ax.set_xlim(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])))
    ax.set_ylim(float(np.nanmin(e[finite])), float(np.nanmax(e[finite])))

    # preserve geometric angles so the model-gradient arrow is visually perpendicular
    ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS / "03_linearized_field_parallel_lines.png", dpi=170)
    plt.close(fig)


def plot_validation_scatter(df_target: pd.DataFrame, flux_est: np.ndarray) -> None:
    if "flux" not in df_target:
        return
    flux_true = df_target["flux"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(flux_true, flux_est, s=24, alpha=0.82, edgecolors="0.25", linewidths=0.25)
    lo = float(min(np.nanmin(flux_true), np.nanmin(flux_est)))
    hi = float(max(np.nanmax(flux_true), np.nanmax(flux_est)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="ideal")
    ax.set_xlabel("True flux")
    ax.set_ylabel("Estimated flux")
    ax.set_title("Flux reconstruction")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "05_validation_scatter_flux.png", dpi=170)
    plt.close(fig)


def plot_timeseries(df_target: pd.DataFrame, flux_est: np.ndarray) -> None:
    t = df_target["time_utc"]
    flux_true = df_target["flux"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.plot(t, flux_true, lw=1.2, label="True flux")
    ax.plot(t, flux_est, lw=1.1, label="Estimated flux")
    ax.set_title("Flux time series: true vs estimated")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.grid(True, alpha=0.2)
    ax.legend()
    format_time_axis(ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "06_flux_timeseries_validation.png", dpi=170)
    plt.close(fig)


def write_outputs(
    train_csv: Path,
    target_csv: Path,
    train_meta: CanonicalMeta,
    target_meta: CanonicalMeta,
    model: AffineInverse,
    df_train: pd.DataFrame,
    grads: pd.DataFrame,
    tan: np.ndarray,
    keep_mask: np.ndarray,
    med: float,
    mad: float,
    thr: float,
    df_target: pd.DataFrame,
    flux_est: np.ndarray,
    corr: float,
    rmse: float,
    mae: float,
    bias: float,
) -> None:
    out = df_target.copy()
    out["flux_est"] = flux_est
    out["flux_error"] = out["flux_est"] - out["flux"]
    out.to_csv(PLOTS / "08_linearized_predictions.csv", index=False)

    # human-readable summary with explicit evidence about filtering and gradients
    lines = []
    lines.append(f"Train CSV: {train_csv}")
    lines.append(f"Target CSV: {target_csv}")
    lines.append(f"Train columns used: flux={train_meta.flux_col}, eff={train_meta.eff_col}, rate={train_meta.rate_col}")
    lines.append(f"Target columns used: flux={target_meta.flux_col}, eff={target_meta.eff_col}, rate={target_meta.rate_col}")
    lines.append("")
    lines.append("Method justification:")
    lines.append("- We keep only gradient directions near the median tan(angle), assuming")
    lines.append("  local iso-rate lines are approximately parallel in that stable region.")
    lines.append("- In that region, an affine model rate = a*flux + b*eff + c is sufficient.")
    lines.append("- The inverse is analytic and gives a single fixed matrix.")

    lines.append("")
    lines.append("Gradient-filter diagnostics:")
    finite_tan = np.isfinite(tan)
    kept_count = int(np.count_nonzero(finite_tan & keep_mask))
    total_considered = int(np.count_nonzero(finite_tan))
    lines.append(f"  tan(angle) median = {med:.6f}")
    lines.append(f"  tan(angle) MAD    = {mad:.6f}")
    lines.append(f"  keep threshold     = {thr:.6f}")
    lines.append(f"  points kept        = {kept_count} / {total_considered}")

    # mean local gradient on the kept points (diagnostic)
    gxs = grads["grad_drate_dflux"].to_numpy(dtype=float)
    gys = grads["grad_drate_deff"].to_numpy(dtype=float)
    mask_kept = (np.isfinite(gxs) & np.isfinite(gys) & np.asarray(keep_mask, dtype=bool))
    if mask_kept.sum() > 0:
        mean_gx = float(np.nanmean(gxs[mask_kept]))
        mean_gy = float(np.nanmean(gys[mask_kept]))
    else:
        mean_gx = float("nan")
        mean_gy = float("nan")
    lines.append("")
    lines.append("Mean local gradient on kept points (dRate/dFlux, dRate/dEff):")
    lines.append(f"  ({mean_gx:.6f}, {mean_gy:.6f})")

    lines.append("")
    lines.append("Fitted forward model:")
    lines.append(f"  global_rate = {model.a_flux:.12f}*flux + {model.b_eff:.12f}*eff + {model.c_bias:.12f}")
    lines.append("")
    lines.append("Inverse matrix (global_rate, eff) -> (flux, eff):")
    lines.append(np.array2string(model.matrix, precision=12, suppress_small=False))
    lines.append("offset=" + np.array2string(model.offset, precision=12, suppress_small=False))
    lines.append("")
    lines.append("How to use (for each row):")
    lines.append(f"  flux_est = {model.m11:.12f}*global_rate + {model.m12:.12f}*eff + {model.t1:.12f}")
    lines.append("  eff_out   = eff")
    lines.append("")
    lines.append("Validation on target series:")
    lines.append(f"  corr={corr:.8f}")
    lines.append(f"  rmse={rmse:.8f}")
    lines.append(f"  mae={mae:.8f}")
    lines.append(f"  mean_flux_bias={bias:.8f}")

    (PLOTS / "00_linear_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global PLOTS, K_NEIGHBORS
    args = parse_args()
    if args.k_neighbors < 4:
        raise SystemExit("--k-neighbors must be >= 4")
    K_NEIGHBORS = int(args.k_neighbors)
    PLOTS = Path(args.plots_dir).resolve()

    prepare_output_dir()

    if args.train_csv is None:
        train_csv = first_existing(TRAIN_CSV_CANDIDATES, "training")
    else:
        train_csv = Path(args.train_csv).resolve()
        if not train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    if args.target_csv is None:
        target_csv = first_existing(TARGET_CSV_CANDIDATES, "target")
    else:
        target_csv = Path(args.target_csv).resolve()
        if not target_csv.exists():
            raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    # Prefer simulated efficiencies for this validation path (can still fall back automatically).
    df_train, train_meta = canonicalize(train_csv, prefer_sim_eff=True)
    df_target, target_meta = canonicalize(target_csv, prefer_sim_eff=True)

    grads = local_gradients(df_train, k_neighbors=K_NEIGHBORS)
    tan, keep_mask, med, mad, thr = filter_gradients(
        grads["grad_drate_dflux"].to_numpy(dtype=float),
        grads["grad_drate_deff"].to_numpy(dtype=float),
    )

    model = fit_affine_inverse(df_train, keep_mask)

    rate_t = df_target["global_rate_hz"].to_numpy(dtype=float)
    eff_t = df_target["eff"].to_numpy(dtype=float)
    flux_est = model.estimate_flux(rate_t, eff_t)

    flux_true = df_target["flux"].to_numpy(dtype=float)
    corr, rmse, mae, bias = compute_metrics(flux_true, flux_est)

    plot_iso_quiver(df_train, grads, keep_mask)
    plot_linearized_field(model, df_train, keep_mask)
    plot_tan_histogram(tan, keep_mask, med, thr)
    plot_validation_scatter(df_target, flux_est)
    plot_timeseries(df_target, flux_est)

    write_outputs(
        train_csv=train_csv,
        target_csv=target_csv,
        train_meta=train_meta,
        target_meta=target_meta,
        model=model,
        df_train=df_train,
        grads=grads,
        tan=tan,
        keep_mask=keep_mask,
        med=med,
        mad=mad,
        thr=thr,
        df_target=df_target,
        flux_est=flux_est,
        corr=corr,
        rmse=rmse,
        mae=mae,
        bias=bias,
    )

    print(f"Train CSV: {train_csv}")
    print(f"Target CSV: {target_csv}")
    print(f"Rows (train/target): {len(df_train)}/{len(df_target)}")
    print(f"tan(angle) median={med:.6f}, mad={mad:.6f}, threshold={thr:.6f}, kept={int(keep_mask.sum())}/{len(keep_mask)}")
    print("\nFinal affine inverse matrix M and offset t:")
    print(np.array2string(model.matrix, precision=12, suppress_small=False))
    print("t = " + np.array2string(model.offset, precision=12, suppress_small=False))
    print("\nApply to each row:")
    print(f"  flux_est = {model.m11:.12f}*global_rate + {model.m12:.12f}*eff + {model.t1:.12f}")
    print("\nValidation:")
    print(f"  corr={corr:.6f}, rmse={rmse:.6f}, mae={mae:.6f}, mean_bias={bias:.6f}")
    print(f"Outputs: {PLOTS}")


if __name__ == "__main__":
    main()
