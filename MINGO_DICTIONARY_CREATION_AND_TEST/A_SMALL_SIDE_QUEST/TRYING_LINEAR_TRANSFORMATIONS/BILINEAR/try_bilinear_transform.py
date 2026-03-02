#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/BILINEAR/try_bilinear_transform.py
Purpose: Calibrate an inverse mapping from (global_rate, eff) to flux using a bilinear lookup.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/BILINEAR/try_bilinear_transform.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

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
PLOTS = HERE / "PLOTS"

# Keep this explicit and simple: dictionary_test currently matches the target
# synthetic series domain better than older dictionary snapshots.
TRAIN_CSV_CANDIDATES = [
    # Path(
    #     "/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/"
    #     "TRYING_LINEAR_TRANSFORMATIONS/dictionary_test.csv"
    # ),
    Path(
        "/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_1_SETUP/"
        "STEP_1_2_BUILD_DICTIONARY/OUTPUTS/FILES/dictionary.csv"
    ),
]

TARGET_CSV_CANDIDATES = [
    # Path(
    #     "/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/"
    #     "the_simulated_file.csv"
    # ),
    Path(
        "/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_1_TIME_SERIES_CREATION/OUTPUTS/FILES/"
        "complete_curve_time_series.csv"
    ),
    # HERE / "dictionary_test.csv",
]

K_NEIGHBORS = 10
MAD_MULTIPLIER = 2.0
EPS = 1e-12

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
class BilinearInverse:
    """Bilinear lookup-table inverse from (global_rate, eff) -> flux.

    - `grid_rate` is 1D increasing array of rate grid points (x axis).
    - `grid_eff` is 1D increasing array of eff grid points (y axis).
    - `flux_grid` is a 2D array with shape (len(grid_eff), len(grid_rate)).
    - `a_flux`, `b_eff`, `c_bias` store a diagnostic global affine fit (rate ≈ a*flux + b*eff + c)
      so existing plotting/summary code can continue to use the linearization.
    """

    grid_rate: np.ndarray
    grid_eff: np.ndarray
    flux_grid: np.ndarray
    a_flux: float
    b_eff: float
    c_bias: float

    @property
    def m11(self) -> float:
        return 1.0 / self.a_flux if abs(self.a_flux) > EPS else float("nan")

    @property
    def m12(self) -> float:
        return -self.b_eff / self.a_flux if abs(self.a_flux) > EPS else float("nan")

    @property
    def t1(self) -> float:
        return -self.c_bias / self.a_flux if abs(self.a_flux) > EPS else float("nan")

    @property
    def matrix(self) -> np.ndarray:
        # present a compact linearization matrix (diagnostic only)
        return np.array([[self.m11, self.m12], [0.0, 1.0]], dtype=float)

    @property
    def offset(self) -> np.ndarray:
        return np.array([self.t1, 0.0], dtype=float)

    def estimate_flux(self, rate: np.ndarray, eff: np.ndarray) -> np.ndarray:
        """Bilinear interpolation on the precomputed grid. Extrapolates by clipping
        to the grid edges."""
        r = np.asarray(rate, dtype=float)
        e = np.asarray(eff, dtype=float)
        r_b, e_b = np.broadcast_arrays(r, e)
        flat_r = r_b.ravel()
        flat_e = e_b.ravel()

        # locate indices
        jr = np.searchsorted(self.grid_rate, flat_r)
        ie = np.searchsorted(self.grid_eff, flat_e)

        j0 = np.clip(jr - 1, 0, len(self.grid_rate) - 2)
        i0 = np.clip(ie - 1, 0, len(self.grid_eff) - 2)
        j1 = j0 + 1
        i1 = i0 + 1

        r0 = self.grid_rate[j0]
        r1 = self.grid_rate[j1]
        e0 = self.grid_eff[i0]
        e1 = self.grid_eff[i1]

        # avoid division by zero
        denom_r = np.where(r1 - r0 == 0.0, EPS, (r1 - r0))
        denom_e = np.where(e1 - e0 == 0.0, EPS, (e1 - e0))
        tx = np.clip((flat_r - r0) / denom_r, 0.0, 1.0)
        ty = np.clip((flat_e - e0) / denom_e, 0.0, 1.0)

        f00 = self.flux_grid[i0, j0]
        f10 = self.flux_grid[i0, j1]
        f01 = self.flux_grid[i1, j0]
        f11 = self.flux_grid[i1, j1]

        interp_flat = (1.0 - ty) * ((1.0 - tx) * f00 + tx * f10) + ty * ((1.0 - tx) * f01 + tx * f11)
        return interp_flat.reshape(r_b.shape)


@dataclass
class GlobalBilinear:
    """Compact global bilinear polynomial approximation:

    flux ≈ a00 + a10*rate + a01*eff + a11*rate*eff

    Represented as a symmetric 3×3 matrix M so
      flux = [rate, eff, 1]^T @ M @ [rate, eff, 1]
    (with zero r^2 and e^2 coefficients).
    """

    a00: float
    a10: float
    a01: float
    a11: float
    r_min: float
    r_max: float
    e_min: float
    e_max: float
    # diagnostic linearization (keeps compatibility with plotting code)
    a_flux: float = float('nan')
    b_eff: float = float('nan')
    c_bias: float = float('nan')

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [0.0, self.a11 / 2.0, self.a10 / 2.0],
                [self.a11 / 2.0, 0.0, self.a01 / 2.0],
                [self.a10 / 2.0, self.a01 / 2.0, self.a00],
            ],
            dtype=float,
        )

    def estimate_flux(self, rate: np.ndarray | float, eff: np.ndarray | float) -> np.ndarray:
        r = np.asarray(rate, dtype=float)
        e = np.asarray(eff, dtype=float)
        return (self.a00 + self.a10 * r + self.a01 * e + self.a11 * r * e)


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


def fit_bilinear_inverse(df: pd.DataFrame, keep_mask: np.ndarray, n_rate: int = 80, n_eff: int = 80) -> BilinearInverse:
    """Build a bilinear lookup-table inverse (global_rate, eff) -> flux.

    - interpolates flux from kept training points onto a regular (rate x eff) grid
      using triangulation; grid nodes outside the convex hull are filled by nearest
      training flux.
    - returns a BilinearInverse which exposes `estimate_flux(rate, eff)` for use.
    """
    flux = df["flux"].to_numpy(dtype=float)
    eff = df["eff"].to_numpy(dtype=float)
    rate = df["global_rate_hz"].to_numpy(dtype=float)

    mask = keep_mask & np.isfinite(flux) & np.isfinite(eff) & np.isfinite(rate)
    if mask.sum() < 6:
        raise RuntimeError("Not enough valid kept points to build bilinear inverse.")

    rate_k = rate[mask]
    eff_k = eff[mask]
    flux_k = flux[mask]

    r_min, r_max = float(np.nanmin(rate_k)), float(np.nanmax(rate_k))
    e_min, e_max = float(np.nanmin(eff_k)), float(np.nanmax(eff_k))
    if r_max - r_min < EPS or e_max - e_min < EPS:
        raise RuntimeError("Degenerate range for rate/eff to build grid.")

    n_rate = max(10, min(int(n_rate), 200))
    n_eff = max(10, min(int(n_eff), 200))
    grid_rate = np.linspace(r_min, r_max, n_rate)
    grid_eff = np.linspace(e_min, e_max, n_eff)

    # interpolate scattered (rate_k, eff_k) -> flux_k onto the rectangular grid
    tri = Triangulation(rate_k, eff_k)
    interp = LinearTriInterpolator(tri, flux_k)
    Rg, Eg = np.meshgrid(grid_rate, grid_eff)
    Fg = np.asarray(interp(Rg, Eg), dtype=float)

    # fill NaNs (outside convex hull) by nearest training-point flux
    nan_mask = ~np.isfinite(Fg)
    if nan_mask.any():
        qx = Rg[nan_mask].ravel()
        qy = Eg[nan_mask].ravel()
        # squared distances between training points and query points (vectorized)
        dx = rate_k[:, None] - qx[None, :]
        dy = eff_k[:, None] - qy[None, :]
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2, axis=0)
        Fg[nan_mask] = flux_k[idx]

    # diagnostic global affine fit (rate ≈ a*flux + b*eff + c)
    A = np.column_stack([flux_k, eff_k, np.ones(flux_k.shape[0])])
    a, b, c = [float(v) for v in np.linalg.lstsq(A, rate_k, rcond=None)[0]]

    return BilinearInverse(grid_rate=grid_rate, grid_eff=grid_eff, flux_grid=Fg, a_flux=a, b_eff=b, c_bias=c)


def fit_global_bilinear(df: pd.DataFrame, keep_mask: np.ndarray) -> GlobalBilinear:
    """Fit a single global bilinear polynomial to the *kept* training points.

    Model form:
      flux ≈ a00 + a10*rate + a01*eff + a11*rate*eff

    This is a least-squares fit (kept points only). The returned object
    exposes `.matrix` (3×3) so the mapping can be written as
      flux = [rate,eff,1]^T @ M @ [rate,eff,1]
    """
    flux = df["flux"].to_numpy(dtype=float)
    eff = df["eff"].to_numpy(dtype=float)
    rate = df["global_rate_hz"].to_numpy(dtype=float)

    mask = keep_mask & np.isfinite(flux) & np.isfinite(eff) & np.isfinite(rate)
    if mask.sum() < 4:
        raise RuntimeError("Not enough valid kept points to fit global bilinear polynomial.")

    r_k = rate[mask]
    e_k = eff[mask]
    f_k = flux[mask]

    # design: [1, r, e, r*e]
    A = np.column_stack([np.ones_like(r_k), r_k, e_k, r_k * e_k])
    a00, a10, a01, a11 = [float(v) for v in np.linalg.lstsq(A, f_k, rcond=None)[0]]

    r_min, r_max = float(np.nanmin(r_k)), float(np.nanmax(r_k))
    e_min, e_max = float(np.nanmin(e_k)), float(np.nanmax(e_k))

    # diagnostic forward affine (rate ≈ a*flux + b*eff + c) kept for plotting
    Af = np.column_stack([f_k, e_k, np.ones_like(f_k)])
    a_flux, b_eff, c_bias = [float(v) for v in np.linalg.lstsq(Af, r_k, rcond=None)[0]]

    return GlobalBilinear(a00=a00, a10=a10, a01=a01, a11=a11, r_min=r_min, r_max=r_max, e_min=e_min, e_max=e_max, a_flux=a_flux, b_eff=b_eff, c_bias=c_bias)


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


def plot_linearized_field(grid_model, global_model, df: pd.DataFrame, keep: np.ndarray) -> None:
    """Iso-rate map that overlays the piecewise bilinear iso-lines (grid_model)
    and the single global bilinear polynomial iso-lines (global_model) for direct comparison.

    - background: actual global-rate field (triangulation)
    - faint local gradient quivers
    - overlay: grid-based bilinear iso-lines (solid C3)
    - overlay: global polynomial iso-lines (dashed C4)
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
    Ri = np.asarray(interp(Rg, Eg), dtype=float) if False else np.asarray(interp(Xi, Yi), dtype=float)

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

    # --- blue gradient vector: use the diagnostic affine linearization (global_model)
    kept = finite & keep
    if kept.sum() >= 1:
        gmx = float(global_model.a_flux)
        gmy = float(global_model.b_eff)
        gnorm = float(np.hypot(gmx, gmy))
        if gnorm > EPS:
            ux, uy = gmx / gnorm, gmy / gnorm
            x0 = float(np.nanmedian(x[kept]))
            e0 = float(np.nanmedian(e[kept]))
            span = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), float(np.nanmax(e[finite]) - np.nanmin(e[finite])))
            a_scale = 0.18 * span
            ax.quiver([x0], [e0], [ux], [uy], angles="xy", scale_units="xy", scale=1.0 / a_scale, color="C0", width=0.008)
            ax.scatter([x0], [e0], s=26, color="C0", zorder=6)
            ax.text(x0, e0 - 0.035 * span, f"diagnostic grad=({gmx:.2f},{gmy:.2f})", color="C0", fontsize=8, ha="center")

    # --- compute integer iso-line levels
    eff_lin = yi
    if r_max - r_min < 1e-6:
        levels = np.array([r_min])
    else:
        if int(np.floor(r_max)) >= 9:
            levels = np.arange(9, int(np.floor(r_max)) + 1, dtype=float)
        else:
            levels = np.arange(int(np.floor(r_min)), int(np.ceil(r_max)) + 1, dtype=float)
        levels = np.unique(levels)

    # --- grid-model (piecewise bilinear) iso-lines
    grid_lines = [grid_model.estimate_flux(np.full_like(eff_lin, rl), eff_lin) for rl in levels]

    # --- global polynomial iso-lines
    poly_lines = [global_model.estimate_flux(np.full_like(eff_lin, rl), eff_lin) for rl in levels]

    # determine plotting x-limits that include both sets of lines
    all_vals = np.hstack([g[np.isfinite(g)] for g in grid_lines if np.isfinite(g).any()] + [p[np.isfinite(p)] for p in poly_lines if np.isfinite(p).any()]) if levels.size else np.array([])
    if all_vals.size:
        min_flux_line = float(np.nanmin(all_vals))
        max_flux_line = float(np.nanmax(all_vals))
    else:
        min_flux_line = float(np.nanmin(x[finite]))
        max_flux_line = float(np.nanmax(x[finite]))

    x_data_min = float(np.nanmin(x[finite]))
    x_data_max = float(np.nanmax(x[finite]))
    x_min_plot = min(x_data_min, min_flux_line)
    x_max_plot = max(x_data_max, max_flux_line)
    xpad = max(1e-9, 0.03 * (x_max_plot - x_min_plot))
    x_min_plot -= xpad
    x_max_plot += xpad

    # plot both sets of iso-lines and labels
    first_grid = True
    eff_span = float(np.nanmax(e[finite]) - np.nanmin(e[finite]))
    eff_positions = np.linspace(float(np.nanmin(e[finite])) + 0.1 * eff_span, float(np.nanmax(e[finite])) - 0.1 * eff_span, max(1, len(levels)))

    for i, rl in enumerate(levels):
        gline = grid_lines[i]
        pline = poly_lines[i]

        mask_g = np.isfinite(gline) & (gline >= x_min_plot) & (gline <= x_max_plot)
        mask_p = np.isfinite(pline) & (pline >= x_min_plot) & (pline <= x_max_plot)

        if mask_g.any():
            lbl = "grid bilinear iso-lines" if first_grid else "_nolegend_"
            ax.plot(gline[mask_g], eff_lin[mask_g], color="C3", lw=1.4, alpha=0.95, label=lbl)
            first_grid = False

        if mask_p.any():
            ax.plot(pline[mask_p], eff_lin[mask_p], color="C4", lw=1.0, ls="--", alpha=0.9, label="global bilinear poly" if i == 0 else "_nolegend_")

        eff_label = eff_positions[min(i, len(eff_positions) - 1)]
        flux_label_g = float(np.nanmedian(grid_model.estimate_flux(np.array([rl]), np.array([eff_label]))))
        flux_label_p = float(np.nanmedian(global_model.estimate_flux(np.array([rl]), np.array([eff_label]))))
        # prefer labeling grid placement if available
        flux_label = flux_label_g if np.isfinite(flux_label_g) else flux_label_p
        if x_min_plot <= flux_label <= x_max_plot and np.isfinite(flux_label):
            ax.text(
                flux_label,
                eff_label,
                f"{int(round(rl))} Hz",
                color="black",
                fontsize=8,
                backgroundcolor=("white"),
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1),
            )

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
        f"diag affine=(a,b,c)=({global_model.a_flux:.3f},{global_model.b_eff:.3f},{global_model.c_bias:.3f})"
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", alpha=0.9))

    ax.set_title("Iso-rate map — grid vs global bilinear")
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
    fig.savefig(PLOTS / "03_bilinear_vs_global_comparison.png", dpi=170)
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
    grid_model: BilinearInverse,
    global_model: GlobalBilinear,
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
    out.to_csv(PLOTS / "08_bilinear_predictions.csv", index=False)

    lines = []
    lines.append(f"Train CSV: {train_csv}")
    lines.append(f"Target CSV: {target_csv}")
    lines.append(f"Train columns used: flux={train_meta.flux_col}, eff={train_meta.eff_col}, rate={train_meta.rate_col}")
    lines.append(f"Target columns used: flux={target_meta.flux_col}, eff={target_meta.eff_col}, rate={target_meta.rate_col}")
    lines.append("")
    lines.append("Method justification:")
    lines.append("- We keep only gradient directions near the median tan(angle), assuming")
    lines.append("  local iso-rate lines are approximately parallel in that stable region.")
    lines.append("- We build a piecewise bilinear lookup (rate, eff) -> flux from the kept training points and use it for prediction.")
    lines.append("- A compact global bilinear polynomial (matrix M) is fitted for diagnostics and comparison.")
    lines.append("")
    lines.append("Inverse mapping (piecewise bilinear lookup):")
    lines.append(f"  grid shape: eff={len(grid_model.grid_eff)}, rate={len(grid_model.grid_rate)}")
    lines.append(f"  rate range: [{grid_model.grid_rate[0]:.6g}, {grid_model.grid_rate[-1]:.6g}]")
    lines.append(f"  eff range:  [{grid_model.grid_eff[0]:.6g}, {grid_model.grid_eff[-1]:.6g}]")
    lines.append("")
    lines.append("Fitted global bilinear polynomial (diagnostic):")
    lines.append("  flux ≈ a00 + a10*rate + a01*eff + a11*rate*eff")
    lines.append(f"  coefficients: a00={global_model.a00:.12f}, a10={global_model.a10:.12f}, a01={global_model.a01:.12f}, a11={global_model.a11:.12f}")
    lines.append("Matrix M (3×3) such that flux = [rate,eff,1]^T M [rate,eff,1]:")
    lines.append(np.array2string(global_model.matrix, precision=12, suppress_small=False))
    lines.append(f"  rate range: [{global_model.r_min:.6g}, {global_model.r_max:.6g}]")
    lines.append(f"  eff range:  [{global_model.e_min:.6g}, {global_model.e_max:.6g}]")
    lines.append("")
    lines.append("How to use (for each row):")
    lines.append("  flux_est = grid_model.estimate_flux(global_rate, eff)  # piecewise bilinear lookup (used for predictions)")
    lines.append("  eff_out   = eff")
    lines.append("")
    lines.append("Validation on target series (using piecewise bilinear lookup):")
    lines.append(f"  corr={corr:.8f}")
    lines.append(f"  rmse={rmse:.8f}")
    lines.append(f"  mae={mae:.8f}")
    lines.append(f"  mean_flux_bias={bias:.8f}")

    (PLOTS / "00_bilinear_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    prepare_output_dir()

    train_csv = first_existing(TRAIN_CSV_CANDIDATES, "training")
    target_csv = first_existing(TARGET_CSV_CANDIDATES, "target")

    # Prefer simulated efficiencies for this validation path (can still fall back automatically).
    df_train, train_meta = canonicalize(train_csv, prefer_sim_eff=True)
    df_target, target_meta = canonicalize(target_csv, prefer_sim_eff=True)

    grads = local_gradients(df_train, k_neighbors=K_NEIGHBORS)
    tan, keep_mask, med, mad, thr = filter_gradients(
        grads["grad_drate_dflux"].to_numpy(dtype=float),
        grads["grad_drate_deff"].to_numpy(dtype=float),
    )

    # build both models: flexible grid-based bilinear inverse (used for prediction)
    # and a compact global bilinear polynomial for diagnostics/comparison
    grid_model = fit_bilinear_inverse(df_train, keep_mask)
    global_model = fit_global_bilinear(df_train, keep_mask)

    rate_t = df_target["global_rate_hz"].to_numpy(dtype=float)
    eff_t = df_target["eff"].to_numpy(dtype=float)
    # use the piecewise bilinear lookup for predictions (more flexible / non-linear)
    flux_est = grid_model.estimate_flux(rate_t, eff_t)

    flux_true = df_target["flux"].to_numpy(dtype=float)
    corr, rmse, mae, bias = compute_metrics(flux_true, flux_est)

    plot_iso_quiver(df_train, grads, keep_mask)
    plot_linearized_field(grid_model, global_model, df_train, keep_mask)
    plot_tan_histogram(tan, keep_mask, med, thr)
    plot_validation_scatter(df_target, flux_est)
    plot_timeseries(df_target, flux_est)

    write_outputs(
        train_csv=train_csv,
        target_csv=target_csv,
        train_meta=train_meta,
        target_meta=target_meta,
        grid_model=grid_model,
        global_model=global_model,
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

    print("\nPrediction method: piecewise bilinear lookup (grid)")
    print(f"  grid shape: eff={len(grid_model.grid_eff)}, rate={len(grid_model.grid_rate)}")
    print(f"  rate range: [{grid_model.grid_rate[0]:.6g}, {grid_model.grid_rate[-1]:.6g}]")
    print(f"  eff range:  [{grid_model.grid_eff[0]:.6g}, {grid_model.grid_eff[-1]:.6g}]")

    print("\nDiagnostic: global bilinear polynomial (matrix M)")
    print(f"  coefficients: a00={global_model.a00:.6g}, a10={global_model.a10:.6g}, a01={global_model.a01:.6g}, a11={global_model.a11:.6g}")
    print(np.array2string(global_model.matrix, precision=12, suppress_small=False))
    print(f"  domain: rate=[{global_model.r_min:.6g},{global_model.r_max:.6g}], eff=[{global_model.e_min:.6g},{global_model.e_max:.6g}]")

    print("\nApply to each row (used for predictions):")
    print("  flux_est = grid_model.estimate_flux(global_rate, eff)  # piecewise bilinear lookup")

    print("\nValidation (piecewise bilinear lookup):")
    print(f"  corr={corr:.6f}, rmse={rmse:.6f}, mae={mae:.6f}, mean_bias={bias:.6f}")
    print(f"Outputs: {PLOTS}")


if __name__ == "__main__":
    main()
