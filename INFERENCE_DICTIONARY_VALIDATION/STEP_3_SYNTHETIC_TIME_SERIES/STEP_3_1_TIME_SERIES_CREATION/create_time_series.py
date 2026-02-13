#!/usr/bin/env python3
"""STEP 3.1 — Random complete trajectory + event-based discretization.

This step creates one complete random trajectory in `(flux_cm2_min, eff)` and
discretizes it into synthetic files using:

- total trajectory duration,
- global rate along the trajectory (interpolated from dictionary),
- target events per file.

Key idea
--------
If `rate(t)` is known, cumulative events are:
`N(t) = integral(rate(t) dt)`.
File boundaries are placed at `N = k * events_per_file`, producing variable
time widths and physically consistent sampling density.

Output
------
OUTPUTS/FILES/time_series.csv
    One row per synthetic file with timing, flux, efficiency, global rate and
    expected events.
OUTPUTS/FILES/complete_curve_time_series.csv
    Complete trajectory before discretization.
OUTPUTS/FILES/time_series_summary.json
    Run metadata including the used random seed.
OUTPUTS/PLOTS/curve_flux_vs_eff.png
    Trajectory over semitransparent global-rate contour map.
OUTPUTS/PLOTS/time_series_flux_eff.png
    Combined time-series view: complete curve and discretized points
    overlaid for flux, efficiency and global rate.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = STEP_DIR.parents[1]  # INFERENCE_DICTIONARY_VALIDATION
DEFAULT_CONFIG = PIPELINE_DIR / "config.json"

DEFAULT_DATASET = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="[%(levelname)s] STEP_3.1 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_3.1")


def _load_config(path: Path) -> dict:
    """Load pipeline configuration from JSON path."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _safe_float(value: object, default: float) -> float:
    """Convert value to float, with default fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_int(value: object, default: int, minimum: int | None = None) -> int:
    """Convert value to int, with optional lower bound."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


def _as_bool(value: object, default: bool = False) -> bool:
    """Parse booleans from config-like values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_input_path(path_like: str | Path) -> Path:
    """Resolve path relative to pipeline when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_pipeline


def _choose_eff_column(df: pd.DataFrame, preferred: str) -> str:
    """Return efficiency column to use from source table."""
    if preferred in df.columns:
        return preferred
    for candidate in ("eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
                      "eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No efficiency column found in source table.")


def _resolve_numeric_range(
    cfg_value: object,
    fallback_min: float,
    fallback_max: float,
) -> tuple[float, float]:
    """Resolve [min, max] range from config, or fallback to source bounds."""
    lo = float(fallback_min)
    hi = float(fallback_max)
    if isinstance(cfg_value, (list, tuple)) and len(cfg_value) == 2:
        lo = _safe_float(cfg_value[0], lo)
        hi = _safe_float(cfg_value[1], hi)
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _normalised_axis(n_points: int) -> np.ndarray:
    """Build normalized axis in [0, 1) with n_points samples."""
    n = max(2, int(n_points))
    return np.linspace(0.0, 1.0, n, endpoint=False, dtype=float)


def _smooth_with_moving_average(values: np.ndarray, window_points: int) -> np.ndarray:
    """Smooth series with edge-preserving moving average."""
    arr = np.asarray(values, dtype=float)
    w = max(1, int(window_points))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return arr.copy()
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(padded, kernel, mode="valid")


def _random_smooth_unit_series(
    u: np.ndarray,
    rng: np.random.Generator,
    n_harmonics: int,
    roughness: float,
    smoothing_window_points: int,
    smoothing_passes: int,
) -> np.ndarray:
    """Generate smooth random series normalized to [0, 1]."""
    y = np.zeros_like(u, dtype=float)
    two_pi = 2.0 * np.pi
    n_h = max(1, int(n_harmonics))
    rough = max(0.1, float(roughness))

    for k in range(1, n_h + 1):
        amp = rng.uniform(0.3, 1.0) / (k ** rough)
        freq_s = max(0.15, float(k) + rng.uniform(-0.35, 0.35))
        freq_c = max(0.15, float(k) + rng.uniform(-0.35, 0.35))
        phase_s = rng.uniform(0.0, two_pi)
        phase_c = rng.uniform(0.0, two_pi)
        y += amp * np.sin(two_pi * freq_s * u + phase_s)
        y += 0.5 * amp * np.cos(two_pi * freq_c * u + phase_c)

    # Add low-order drift to avoid near-periodic closure.
    du = u - 0.5
    y += rng.normal(0.0, 0.8) * du
    y += rng.normal(0.0, 0.4) * du * du

    for _ in range(max(0, int(smoothing_passes))):
        y = _smooth_with_moving_average(y, smoothing_window_points)

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    span = y_max - y_min
    if not np.isfinite(span) or span <= 1e-12:
        unit = np.full_like(y, 0.5, dtype=float)
    else:
        unit = (y - y_min) / span

    # Ensure start/end are not artificially identical.
    if len(unit) >= 2 and abs(float(unit[-1]) - float(unit[0])) < 0.03:
        delta = (0.03 - abs(float(unit[-1]) - float(unit[0])))
        if rng.random() < 0.5:
            delta *= -1.0
        unit = unit + np.linspace(0.0, delta, len(unit), dtype=float)
        umin = float(np.nanmin(unit))
        umax = float(np.nanmax(unit))
        uspan = umax - umin
        if np.isfinite(uspan) and uspan > 1e-12:
            unit = (unit - umin) / uspan
        else:
            unit = np.full_like(unit, 0.5, dtype=float)

    return np.clip(unit, 0.0, 1.0)


def _map_unit_to_range(unit: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """Map unit-interval values to numeric range, supporting constant ranges."""
    lo, hi = float(value_range[0]), float(value_range[1])
    if np.isclose(lo, hi, atol=0.0):
        return np.full_like(unit, lo, dtype=float)
    return lo + (hi - lo) * np.asarray(unit, dtype=float)


def _build_random_curve(
    u: np.ndarray,
    flux_range: tuple[float, float],
    eff_range: tuple[float, float],
    rng: np.random.Generator,
    cfg_31: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build smooth random flux and efficiency trajectories."""
    n_harmonics = _safe_int(cfg_31.get("n_harmonics", 4), 4, minimum=1)
    roughness = _safe_float(cfg_31.get("roughness", 1.6), 1.6)
    smoothing_window_points = _safe_int(
        cfg_31.get("smoothing_window_points", 11), 11, minimum=1
    )
    smoothing_passes = _safe_int(cfg_31.get("smoothing_passes", 2), 2, minimum=0)

    flux_u = _random_smooth_unit_series(
        u=u,
        rng=rng,
        n_harmonics=n_harmonics,
        roughness=roughness,
        smoothing_window_points=smoothing_window_points,
        smoothing_passes=smoothing_passes,
    )
    eff_u = _random_smooth_unit_series(
        u=u,
        rng=rng,
        n_harmonics=n_harmonics,
        roughness=roughness,
        smoothing_window_points=smoothing_window_points,
        smoothing_passes=smoothing_passes,
    )
    flux = _map_unit_to_range(flux_u, flux_range)
    eff = _map_unit_to_range(eff_u, eff_range)
    return flux.astype(float), eff.astype(float)


def _build_rate_model(
    rate_df: pd.DataFrame,
    flux_col: str,
    eff_col: str,
    rate_col: str,
) -> dict:
    """Build interpolation model for global rate in flux-eff plane."""
    x = pd.to_numeric(rate_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rate_df.get(eff_col), errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(rate_df.get(rate_col), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if len(x) < 3:
        raise ValueError("Not enough valid points to build global-rate interpolation.")

    tri = None
    interp = None
    try:
        tri = Triangulation(x, y)
        interp = LinearTriInterpolator(tri, z)
    except Exception as exc:
        log.warning("Triangulation/interpolation failed; using nearest fallback only: %s", exc)

    return {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": z.astype(float),
        "tri": tri,
        "interp": interp,
    }


def _predict_rate(
    model: dict,
    flux_values: np.ndarray,
    eff_values: np.ndarray,
    min_rate_hz: float = 1e-6,
) -> np.ndarray:
    """Predict global rate at query points with interpolation + nearest fallback."""
    xq = np.asarray(flux_values, dtype=float)
    yq = np.asarray(eff_values, dtype=float)
    shape = xq.shape
    qx = xq.ravel()
    qy = yq.ravel()

    zq = np.full(qx.shape, np.nan, dtype=float)
    interp = model.get("interp")
    if interp is not None:
        zi = interp(qx, qy)
        zq = np.asarray(np.ma.filled(zi, np.nan), dtype=float)

    missing = ~np.isfinite(zq)
    if missing.any():
        x = model["x"]
        y = model["y"]
        z = model["z"]
        qx_m = qx[missing]
        qy_m = qy[missing]
        out = np.empty(len(qx_m), dtype=float)
        chunk = 4096
        for s in range(0, len(qx_m), chunk):
            e = min(len(qx_m), s + chunk)
            dx = qx_m[s:e, None] - x[None, :]
            dy = qy_m[s:e, None] - y[None, :]
            idx = np.argmin(dx * dx + dy * dy, axis=1)
            out[s:e] = z[idx]
        zq[missing] = out

    zq = np.maximum(zq, float(max(min_rate_hz, 0.0)))
    return zq.reshape(shape)


def _cumulative_events(dense_time_s: np.ndarray, dense_rate_hz: np.ndarray) -> np.ndarray:
    """Cumulative expected counts along dense trajectory."""
    dt = np.diff(dense_time_s)
    seg_events = 0.5 * (dense_rate_hz[:-1] + dense_rate_hz[1:]) * dt
    seg_events = np.clip(seg_events, 0.0, None)
    return np.concatenate([[0.0], np.cumsum(seg_events)])


def _discretize_curve_by_events(
    dense_time_s: np.ndarray,
    dense_flux: np.ndarray,
    dense_eff: np.ndarray,
    dense_rate_hz: np.ndarray,
    events_per_file: int,
    include_partial_last_file: bool,
) -> tuple[pd.DataFrame, float]:
    """Split dense trajectory into variable-duration synthetic files by events."""
    cum_events = _cumulative_events(dense_time_s, dense_rate_hz)
    total_events = float(cum_events[-1])

    if total_events <= 0.0:
        raise ValueError("Total expected events is zero; cannot discretize trajectory.")

    thresholds = [0.0]
    full_files = int(total_events // float(events_per_file))
    for k in range(1, full_files + 1):
        thresholds.append(float(k * events_per_file))

    remainder = total_events - thresholds[-1]
    if include_partial_last_file and remainder > 1e-9:
        thresholds.append(total_events)
    elif len(thresholds) == 1:
        thresholds.append(total_events)

    thr = np.asarray(thresholds, dtype=float)
    idx = np.searchsorted(cum_events, thr, side="right") - 1
    idx = np.clip(idx, 0, len(dense_time_s) - 2)

    e0 = cum_events[idx]
    e1 = cum_events[idx + 1]
    frac = np.divide(
        thr - e0,
        e1 - e0,
        out=np.zeros_like(thr, dtype=float),
        where=(e1 - e0) > 0,
    )

    t_b = dense_time_s[idx] + frac * (dense_time_s[idx + 1] - dense_time_s[idx])

    t_start = t_b[:-1]
    t_end = t_b[1:]
    duration_s = np.maximum(t_end - t_start, 1e-12)
    events_expected = thr[1:] - thr[:-1]
    t_mid = 0.5 * (t_start + t_end)

    flux_mid = np.interp(t_mid, dense_time_s, dense_flux)
    eff_mid = np.interp(t_mid, dense_time_s, dense_eff)
    rate_mid = np.interp(t_mid, dense_time_s, dense_rate_hz)
    rate_mean = events_expected / duration_s

    out = pd.DataFrame({
        "file_index": np.arange(1, len(t_mid) + 1, dtype=int),
        "elapsed_seconds_start": t_start,
        "elapsed_seconds_end": t_end,
        "elapsed_seconds": t_mid,
        "elapsed_hours_start": t_start / 3600.0,
        "elapsed_hours_end": t_end / 3600.0,
        "elapsed_hours": t_mid / 3600.0,
        "duration_seconds": duration_s,
        "n_events_expected": events_expected,
        "n_events": np.rint(events_expected).astype(int),
        "target_events_per_file": int(events_per_file),
        "global_rate_hz_mid": rate_mid,
        "global_rate_hz_mean": rate_mean,
        "flux": flux_mid,
        "eff": eff_mid,
    })
    out["flux_cm2_min"] = out["flux"]
    out["eff_sim_1"] = out["eff"]
    return out, total_events


def _plot_curve_flux_vs_eff(
    source_df: pd.DataFrame,
    source_flux_col: str,
    source_eff_col: str,
    dense_flux: np.ndarray,
    dense_eff: np.ndarray,
    file_df: pd.DataFrame,
    rate_model: dict,
    events_per_file: int,
    contour_grid_points: int,
    path: Path,
) -> None:
    """Plot trajectory on semitransparent contour map of global rate."""
    fig, ax = plt.subplots(figsize=(9, 7))

    x_ref = np.asarray(rate_model["x"], dtype=float)
    y_ref = np.asarray(rate_model["y"], dtype=float)
    src_flux = pd.to_numeric(source_df.get(source_flux_col), errors="coerce").to_numpy(dtype=float)
    src_eff = pd.to_numeric(source_df.get(source_eff_col), errors="coerce").to_numpy(dtype=float)

    x_all = np.concatenate([
        x_ref[np.isfinite(x_ref)],
        np.asarray(dense_flux, dtype=float)[np.isfinite(dense_flux)],
        src_flux[np.isfinite(src_flux)],
    ])
    y_all = np.concatenate([
        y_ref[np.isfinite(y_ref)],
        np.asarray(dense_eff, dtype=float)[np.isfinite(dense_eff)],
        src_eff[np.isfinite(src_eff)],
    ])

    flux_lo = float(np.nanmin(x_all))
    flux_hi = float(np.nanmax(x_all))
    eff_lo = float(np.nanmin(y_all))
    eff_hi = float(np.nanmax(y_all))
    x_span = max(flux_hi - flux_lo, 1e-6)
    y_span = max(eff_hi - eff_lo, 1e-6)
    flux_lo -= 0.03 * x_span
    flux_hi += 0.03 * x_span
    eff_lo -= 0.03 * y_span
    eff_hi += 0.03 * y_span

    g = max(40, int(contour_grid_points))
    xi = np.linspace(flux_lo, flux_hi, g, dtype=float)
    yi = np.linspace(eff_lo, eff_hi, g, dtype=float)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _predict_rate(rate_model, Xi, Yi, min_rate_hz=1e-6)
    finite_z = Zi[np.isfinite(Zi)]
    if finite_z.size >= 2:
        levels = np.linspace(float(np.nanmin(finite_z)), float(np.nanmax(finite_z)), 16)
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", alpha=0.35, zorder=0)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")
        ax.contour(Xi, Yi, Zi, levels=levels[::2], colors="k", linewidths=0.35, alpha=0.25, zorder=1)

    src_flux_s = pd.Series(src_flux)
    src_eff_s = pd.Series(src_eff)
    src_mask = src_flux_s.notna() & src_eff_s.notna()
    if src_mask.any():
        ax.scatter(
            src_flux_s[src_mask],
            src_eff_s[src_mask],
            s=10,
            alpha=0.18,
            color="#606060",
            zorder=1,
            label="Reference points",
        )

    ax.plot(dense_flux, dense_eff, linewidth=1.8, color="#1f77b4", alpha=0.9, zorder=3, label="Complete trajectory")
    ax.scatter(
        file_df["flux"],
        file_df["eff"],
        s=26,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Discretized file points",
    )

    start_x = float(dense_flux[0])
    start_y = float(dense_eff[0])
    end_x = float(dense_flux[-1])
    end_y = float(dense_eff[-1])
    ax.scatter([start_x], [start_y], color="#2CA02C", marker="o", s=80, edgecolor="black", linewidth=0.8, zorder=5)
    ax.scatter([end_x], [end_y], color="#D62728", marker="X", s=95, edgecolor="black", linewidth=0.8, zorder=5)

    if len(dense_flux) >= 3:
        i = min(len(dense_flux) - 2, max(0, int(0.85 * len(dense_flux))))
        ax.annotate(
            "",
            xy=(dense_flux[i + 1], dense_eff[i + 1]),
            xytext=(dense_flux[i], dense_eff[i]),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            zorder=5,
        )

    ax.set_xlim(flux_lo, flux_hi)
    ax.set_ylim(eff_lo, eff_hi)
    ax.set_xlabel("flux_cm2_min")
    ax.set_ylabel("eff")
    ax.set_title(
        f"STEP 3.1 trajectory with global-rate contours\n"
        f"event discretization target: {events_per_file:,} events/file"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_time_series(
    dense_df: pd.DataFrame,
    file_df: pd.DataFrame,
    path: Path,
) -> None:
    """Plot complete and discretized time series in the same figure."""
    x_dense = dense_df["elapsed_hours"].to_numpy(dtype=float)
    x_disc = file_df["elapsed_hours"].to_numpy(dtype=float)

    def _apply_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
        """Apply stripes at 1%-of-mean increments, uniformly across the y-axis."""
        y_min, y_max = ax.get_ylim()
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        span = y_max - y_min
        if span <= 0.0:
            return

        valid = np.isfinite(y_vals)
        if not np.any(valid):
            return
        mean_val = float(np.mean(y_vals[valid]))

        band = abs(mean_val) * 0.01
        if not np.isfinite(band) or band <= 0.0:
            band = span * 0.01
        if band <= 0.0:
            return

        ax.set_facecolor("#FFFFFF")
        idx = int(np.floor((y_min - mean_val) / band))
        y0 = mean_val + idx * band
        while y0 < y_max:
            y1 = y0 + band
            lo = max(y0, y_min)
            hi = min(y1, y_max)
            color = "#FFFFFF" if (idx % 2 == 0) else "#D8DDE4"
            if hi > lo:
                ax.axhspan(lo, hi, facecolor=color, alpha=1.0, linewidth=0.0, zorder=0)
            y0 = y1
            idx += 1
        ax.set_ylim(y_min, y_max)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True)

    # Flux
    flux_dense = dense_df["flux"].to_numpy(dtype=float)
    axes[0].plot(
        x_dense,
        flux_dense,
        color="#1f77b4",
        linewidth=1.3,
        alpha=0.85,
        label="Complete curve",
    )
    axes[0].plot(
        x_disc,
        file_df["flux"].to_numpy(dtype=float),
        color="#1f77b4",
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
    )
    axes[0].scatter(
        x_disc,
        file_df["flux"].to_numpy(dtype=float),
        s=22,
        facecolor="white",
        edgecolor="#1f77b4",
        linewidth=0.7,
        label="Discretized points",
    )
    axes[0].set_ylabel("flux_cm2_min")
    axes[0].set_title("Flux: complete curve + discretized points")
    _apply_striped_background(axes[0], flux_dense)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    # Efficiency
    eff_dense = dense_df["eff"].to_numpy(dtype=float)
    axes[1].plot(
        x_dense,
        eff_dense,
        color="#FF7F0E",
        linewidth=1.3,
        alpha=0.85,
        label="Complete curve",
    )
    axes[1].plot(
        x_disc,
        file_df["eff"].to_numpy(dtype=float),
        color="#FF7F0E",
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
    )
    axes[1].scatter(
        x_disc,
        file_df["eff"].to_numpy(dtype=float),
        s=22,
        facecolor="white",
        edgecolor="#FF7F0E",
        linewidth=0.7,
        label="Discretized points",
    )
    axes[1].set_ylabel("eff")
    axes[1].set_title("Efficiency: complete curve + discretized points")
    _apply_striped_background(axes[1], eff_dense)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    # Global rate
    rate_dense = dense_df["global_rate_hz"].to_numpy(dtype=float)
    axes[2].plot(
        x_dense,
        rate_dense,
        color="#2CA02C",
        linewidth=1.3,
        alpha=0.85,
        label="Complete curve",
    )
    axes[2].plot(
        x_disc,
        file_df["global_rate_hz_mean"].to_numpy(dtype=float),
        color="#2CA02C",
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
    )
    axes[2].scatter(
        x_disc,
        file_df["global_rate_hz_mean"].to_numpy(dtype=float),
        s=22,
        facecolor="white",
        edgecolor="#2CA02C",
        linewidth=0.7,
        label="Discretized points",
    )
    axes[2].set_xlabel("Elapsed time [hours]")
    axes[2].set_ylabel("global rate [Hz]")
    axes[2].set_title("Global rate: complete curve + discretized points")
    _apply_striped_background(axes[2], rate_dense)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _resolve_seed(cfg_31: dict) -> int:
    """Resolve seed from config or generate one when missing/invalid."""
    raw = cfg_31.get("random_seed", None)
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return int(np.random.default_rng().integers(0, 2**32 - 1))


def main() -> int:
    """Run STEP 3.1 with event-based discretization."""
    parser = argparse.ArgumentParser(
        description="Step 3.1: Create random complete (flux, eff) time series."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_31 = config.get("step_3_1", {})

    source_table = str(cfg_31.get("source_table", "dataset")).strip().lower()
    if args.source_csv:
        source_path = _resolve_input_path(args.source_csv)
    elif cfg_31.get("source_csv"):
        source_path = _resolve_input_path(str(cfg_31.get("source_csv")))
    elif source_table == "dictionary":
        source_path = DEFAULT_DICTIONARY
    else:
        source_path = DEFAULT_DATASET
    if not source_path.exists():
        log.error("Source CSV not found: %s", source_path)
        return 1

    source_df = pd.read_csv(source_path, low_memory=False)
    if source_df.empty:
        log.error("Source CSV is empty: %s", source_path)
        return 1

    flux_col = str(cfg_31.get("flux_column", "flux_cm2_min"))
    if flux_col not in source_df.columns:
        log.error("Flux column '%s' not found in source table.", flux_col)
        return 1
    preferred_eff_col = str(cfg_31.get("eff_column", "eff_sim_1"))
    try:
        eff_col = _choose_eff_column(source_df, preferred_eff_col)
    except KeyError as exc:
        log.error("%s", exc)
        return 1

    src_flux = pd.to_numeric(source_df[flux_col], errors="coerce")
    src_eff = pd.to_numeric(source_df[eff_col], errors="coerce")
    valid = src_flux.notna() & src_eff.notna()
    if valid.sum() < 3:
        log.error("Not enough valid rows in source table for %s and %s.", flux_col, eff_col)
        return 1

    flux_range = _resolve_numeric_range(
        cfg_31.get("flux_range"),
        float(src_flux[valid].min()),
        float(src_flux[valid].max()),
    )
    eff_range = _resolve_numeric_range(
        cfg_31.get("eff_range"),
        float(src_eff[valid].min()),
        float(src_eff[valid].max()),
    )

    # Rate reference table (usually dictionary)
    if cfg_31.get("rate_dictionary_csv"):
        rate_path = _resolve_input_path(str(cfg_31.get("rate_dictionary_csv")))
    else:
        rate_path = DEFAULT_DICTIONARY
    if not rate_path.exists():
        log.error("Rate dictionary CSV not found: %s", rate_path)
        return 1
    rate_df = pd.read_csv(rate_path, low_memory=False)
    if rate_df.empty:
        log.error("Rate dictionary CSV is empty: %s", rate_path)
        return 1

    rate_flux_col = str(cfg_31.get("rate_flux_column", flux_col))
    if rate_flux_col not in rate_df.columns:
        log.warning("rate_flux_column '%s' missing; using '%s'.", rate_flux_col, flux_col)
        rate_flux_col = flux_col

    rate_eff_cfg = str(cfg_31.get("rate_eff_column", eff_col))
    if rate_eff_cfg in rate_df.columns:
        rate_eff_col = rate_eff_cfg
    else:
        try:
            rate_eff_col = _choose_eff_column(rate_df, rate_eff_cfg)
        except KeyError as exc:
            log.error("%s", exc)
            return 1

    rate_col = str(cfg_31.get("rate_column", "events_per_second_global_rate"))
    if rate_col not in rate_df.columns:
        log.error("Global-rate column '%s' not found in rate dictionary.", rate_col)
        return 1

    try:
        rate_model = _build_rate_model(rate_df, rate_flux_col, rate_eff_col, rate_col)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    # Counts per synthetic file
    default_events_per_file = 50000
    if "n_events" in source_df.columns:
        m = pd.to_numeric(source_df["n_events"], errors="coerce").dropna()
        if not m.empty:
            default_events_per_file = int(max(1.0, float(m.median())))
    events_per_file = _safe_int(
        cfg_31.get("events_per_file", default_events_per_file),
        default_events_per_file,
        minimum=1,
    )
    include_partial_last_file = _as_bool(cfg_31.get("include_partial_last_file", True), default=True)

    # Dense trajectory for robust event integration
    duration_hours = max(1e-6, _safe_float(cfg_31.get("duration_hours", 72.0), 72.0))
    duration_seconds = duration_hours * 3600.0
    dense_points_default = 4000
    if "n_points" in cfg_31:
        dense_points_default = max(dense_points_default, 20 * _safe_int(cfg_31.get("n_points"), 240, minimum=2))
    dense_points = _safe_int(cfg_31.get("dense_points", dense_points_default), dense_points_default, minimum=200)

    start_time_raw = str(cfg_31.get("start_time_utc", "2026-01-01T00:00:00Z"))
    start_time = pd.to_datetime(start_time_raw, utc=True, errors="coerce")
    if pd.isna(start_time):
        log.warning("Invalid start_time_utc '%s'; using 2026-01-01T00:00:00Z", start_time_raw)
        start_time = pd.Timestamp("2026-01-01T00:00:00Z")

    seed = _resolve_seed(cfg_31)
    rng = np.random.default_rng(seed)
    log.info("Using random seed: %d", seed)

    u = _normalised_axis(dense_points)
    dense_flux, dense_eff = _build_random_curve(
        u=u,
        flux_range=flux_range,
        eff_range=eff_range,
        rng=rng,
        cfg_31=cfg_31,
    )
    dense_time_s = np.linspace(0.0, duration_seconds, dense_points, dtype=float)
    dense_rate_hz = _predict_rate(rate_model, dense_flux, dense_eff, min_rate_hz=1e-6)
    dense_cum_events = _cumulative_events(dense_time_s, dense_rate_hz)

    dense_time = start_time + pd.to_timedelta(dense_time_s, unit="s")
    dense_df = pd.DataFrame({
        "curve_index": np.arange(len(dense_time_s), dtype=int),
        "time_utc": dense_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": dense_time_s,
        "elapsed_hours": dense_time_s / 3600.0,
        "flux": dense_flux,
        "eff": dense_eff,
        "flux_cm2_min": dense_flux,
        "eff_sim_1": dense_eff,
        "global_rate_hz": dense_rate_hz,
        "cumulative_events_expected": dense_cum_events,
    })

    out_dense_csv = FILES_DIR / "complete_curve_time_series.csv"
    dense_df.to_csv(out_dense_csv, index=False)
    log.info("Wrote complete curve CSV: %s (%d rows)", out_dense_csv, len(dense_df))

    stale_old_curve_csv = FILES_DIR / "original_curve_time_series.csv"
    if stale_old_curve_csv.exists():
        stale_old_curve_csv.unlink()
        log.info("Removed deprecated CSV: %s", stale_old_curve_csv)

    try:
        file_df, total_events = _discretize_curve_by_events(
            dense_time_s=dense_time_s,
            dense_flux=dense_flux,
            dense_eff=dense_eff,
            dense_rate_hz=dense_rate_hz,
            events_per_file=events_per_file,
            include_partial_last_file=include_partial_last_file,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    if len(file_df) < 3:
        log.warning(
            "Only %d synthetic file(s) produced; consider increasing duration_hours "
            "or decreasing events_per_file.",
            len(file_df),
        )

    time_start = start_time + pd.to_timedelta(file_df["elapsed_seconds_start"], unit="s")
    time_end = start_time + pd.to_timedelta(file_df["elapsed_seconds_end"], unit="s")
    time_mid = start_time + pd.to_timedelta(file_df["elapsed_seconds"], unit="s")
    file_df["time_start_utc"] = time_start.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    file_df["time_end_utc"] = time_end.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    file_df["time_utc"] = time_mid.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_cols = [
        "file_index",
        "time_start_utc", "time_end_utc", "time_utc",
        "elapsed_hours_start", "elapsed_hours_end", "elapsed_hours",
        "duration_seconds",
        "target_events_per_file", "n_events_expected", "n_events",
        "global_rate_hz_mid", "global_rate_hz_mean",
        "flux", "eff", "flux_cm2_min", "eff_sim_1",
    ]
    out_df = file_df[out_cols].copy()

    out_csv = FILES_DIR / "time_series.csv"
    out_df.to_csv(out_csv, index=False)
    log.info("Wrote time series CSV: %s (%d rows)", out_csv, len(out_df))

    out_summary = FILES_DIR / "time_series_summary.json"
    summary = {
        "source_csv": str(source_path),
        "source_rows": int(len(source_df)),
        "rate_dictionary_csv": str(rate_path),
        "flux_column": flux_col,
        "eff_column": eff_col,
        "rate_flux_column": rate_flux_col,
        "rate_eff_column": rate_eff_col,
        "rate_column": rate_col,
        "generator": "random_complete",
        "random_seed": int(seed),
        "complete_sampling_points": int(dense_points),
        "complete_curve_points": int(len(dense_df)),
        "duration_hours": float(duration_hours),
        "start_time_utc": str(out_df["time_start_utc"].iloc[0]),
        "end_time_utc": str(out_df["time_end_utc"].iloc[-1]),
        "flux_range_used": [float(flux_range[0]), float(flux_range[1])],
        "eff_range_used": [float(eff_range[0]), float(eff_range[1])],
        "events_per_file": int(events_per_file),
        "include_partial_last_file": bool(include_partial_last_file),
        "n_files": int(len(out_df)),
        "total_expected_events": float(total_events),
        "total_expected_events_complete_integral": float(dense_cum_events[-1]),
        "mean_file_duration_seconds": float(out_df["duration_seconds"].mean()),
        "min_file_duration_seconds": float(out_df["duration_seconds"].min()),
        "max_file_duration_seconds": float(out_df["duration_seconds"].max()),
        "flux_generated_range": [
            float(out_df["flux"].min()),
            float(out_df["flux"].max()),
        ],
        "eff_generated_range": [
            float(out_df["eff"].min()),
            float(out_df["eff"].max()),
        ],
        "global_rate_generated_range_hz": [
            float(out_df["global_rate_hz_mean"].min()),
            float(out_df["global_rate_hz_mean"].max()),
        ],
        "n_harmonics": _safe_int(cfg_31.get("n_harmonics", 4), 4, minimum=1),
        "roughness": float(_safe_float(cfg_31.get("roughness", 1.6), 1.6)),
        "smoothing_window_points": _safe_int(
            cfg_31.get("smoothing_window_points", 11), 11, minimum=1
        ),
        "smoothing_passes": _safe_int(cfg_31.get("smoothing_passes", 2), 2, minimum=0),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary JSON: %s", out_summary)

    out_ts_plot = PLOTS_DIR / "time_series_flux_eff.png"
    _plot_time_series(dense_df, out_df, out_ts_plot)
    log.info("Wrote plot: %s", out_ts_plot)

    stale_old_plot = PLOTS_DIR / "original_curve_time_series_flux_eff.png"
    if stale_old_plot.exists():
        stale_old_plot.unlink()
        log.info("Removed deprecated plot: %s", stale_old_plot)

    contour_grid_points = _safe_int(cfg_31.get("contour_grid_points", 180), 180, minimum=40)
    out_curve_plot = PLOTS_DIR / "curve_flux_vs_eff.png"
    _plot_curve_flux_vs_eff(
        source_df=source_df,
        source_flux_col=flux_col,
        source_eff_col=eff_col,
        dense_flux=dense_flux,
        dense_eff=dense_eff,
        file_df=out_df,
        rate_model=rate_model,
        events_per_file=events_per_file,
        contour_grid_points=contour_grid_points,
        path=out_curve_plot,
    )
    log.info("Wrote plot: %s", out_curve_plot)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
