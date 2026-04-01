"""
Compare calibrated Task 2 strip T_dif distributions between MINGO00 and MINGO01.

The stable Task 2 outputs are the calibrated parquet files already handed off to
Task 3 (STATIONS/MINGO0*/.../TASK_3/INPUT_FILES/COMPLETED_DIRECTORY).
This script reads those calibrated parquets, keeps only non-zero calibrated
T_dif values, and reports the width ratio real/sim as a proxy for the strip
length scaling that would align the simulation to data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - fallback path
    curve_fit = None


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from MASTER.ANCILLARY.SIMULATION_TUNING.common import (
    collect_calibrated_file_entries,
    filter_frame_by_datetime,
    group_label,
    load_tuning_config,
    resolve_selection,
)


TDIF_COLUMNS = [f"T{plane}_T_dif_{strip}" for plane in range(1, 5) for strip in range(1, 5)]
CURRENT_REFERENCE_STRIP_LENGTH_MM = 300.0
DEFAULT_BIN_COUNT = 241
SMOOTHING_WINDOW = 9
ALPHA_GRID = np.linspace(0.82, 1.03, 169)
STRIP_SIGNAL_SPEED_MM_NS = (2.0 / 3.0) * 299_792_458.0 / 1_000_000.0
EDGE_FIT_HALF_WIDTH_NS = 0.10
SUMMARY_STRIP_SELECTION_SEED = 20260331


@dataclass(frozen=True)
class StationConfig:
    label: str
    station_dir: str
    color: str


SIM_COLOR = "#d95f02"
REAL_COLOR = "#1b9e77"
SCALED_COLOR = "#7570b3"

SIM_LABEL = "MINGO00"
REAL_LABEL = "MINGO01"
SIMULATION_STATIONS = ["MINGO00"]
REAL_STATIONS = ["MINGO01"]
SIMULATION_DATE_RANGES: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None
REAL_DATE_RANGES: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None
OUTPUT_DIR_OVERRIDE: Path | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def calibrated_input_dirs(station_dir: str) -> list[Path]:
    base = (
        repo_root()
        / "STATIONS"
        / station_dir
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "INPUT_FILES"
    )
    return [
        base / "COMPLETED_DIRECTORY",
        base / "UNPROCESSED_DIRECTORY",
        base / "PROCESSING_DIRECTORY",
    ]


def output_dir() -> Path:
    if OUTPUT_DIR_OVERRIDE is not None:
        return OUTPUT_DIR_OVERRIDE
    return Path(__file__).resolve().parent / "OUTPUTS"


def active_station_configs() -> tuple[StationConfig, StationConfig]:
    return (
        StationConfig(label=SIM_LABEL, station_dir="SIMULATION", color=SIM_COLOR),
        StationConfig(label=REAL_LABEL, station_dir="REAL", color=REAL_COLOR),
    )


def collect_group_tdif_values(
    station_labels: list[str],
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> tuple[list[tuple[str, Path]], dict[str, np.ndarray], np.ndarray]:
    file_entries = collect_calibrated_file_entries(station_labels)
    by_column: dict[str, list[np.ndarray]] = {column: [] for column in TDIF_COLUMNS}
    overall_chunks: list[np.ndarray] = []
    used_entries: list[tuple[str, Path]] = []

    for station_label, parquet_path in file_entries:
        columns = list(TDIF_COLUMNS)
        if date_ranges is not None:
            columns = ["datetime", *columns]
        frame = pd.read_parquet(parquet_path, columns=columns)
        frame = filter_frame_by_datetime(frame, date_ranges)
        if frame.empty:
            continue
        used_entries.append((station_label, parquet_path))
        for column in TDIF_COLUMNS:
            values = frame[column].to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values != 0)]
            if values.size:
                by_column[column].append(values)
                overall_chunks.append(values)

    finalized_by_column = {
        column: np.concatenate(chunks) if chunks else np.array([], dtype=float)
        for column, chunks in by_column.items()
    }
    overall = np.concatenate(overall_chunks) if overall_chunks else np.array([], dtype=float)
    return used_entries, finalized_by_column, overall


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values)
    return {
        "count": float(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
        "mean_abs": float(abs_values.mean()),
        "q90_abs": float(np.quantile(abs_values, 0.90)),
        "q95_abs": float(np.quantile(abs_values, 0.95)),
        "q975_abs": float(np.quantile(abs_values, 0.975)),
        "q99_abs": float(np.quantile(abs_values, 0.99)),
    }


def _histogram_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _smooth_histogram(values: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _density_histogram(values: np.ndarray, edges: np.ndarray, *, absolute: bool = False) -> tuple[np.ndarray, np.ndarray]:
    sample = np.abs(values) if absolute else values
    density, _ = np.histogram(sample, bins=edges, density=True)
    return _histogram_centers(edges), density


def _span_cap(sim_values: np.ndarray, real_values: np.ndarray, *, absolute: bool) -> float:
    if absolute:
        sim_core = float(np.quantile(np.abs(sim_values), 0.999))
        real_core = float(np.quantile(np.abs(real_values), 0.999))
        return float(np.clip(max(sim_core, real_core) * 1.10, 0.85, 2.0))
    sim_core = float(np.quantile(np.abs(sim_values), 0.999))
    real_core = float(np.quantile(np.abs(real_values), 0.999))
    return float(np.clip(max(sim_core, real_core) * 1.10, 0.85, 2.0))


def _estimate_abs_edge_position(values: np.ndarray, centers: np.ndarray, smoothed_density: np.ndarray) -> float:
    abs_values = np.abs(values)
    derivative = np.gradient(smoothed_density, centers)
    low = float(np.quantile(abs_values, 0.75)) * 0.95
    high = min(float(np.quantile(abs_values, 0.999)) * 1.02, float(centers[-1]))
    search_mask = (centers >= low) & (centers <= high)
    if not np.any(search_mask):
        search_mask = np.ones_like(centers, dtype=bool)
    masked_derivative = np.where(search_mask, derivative, np.inf)
    return float(centers[int(np.argmin(masked_derivative))])


def _fit_histogram_scale_alpha(sim_values: np.ndarray, real_values: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    centers, real_density = _density_histogram(real_values, edges, absolute=True)
    real_smooth = _smooth_histogram(real_density)
    losses = []
    for alpha in ALPHA_GRID:
        _, scaled_sim_density = _density_histogram(alpha * sim_values, edges, absolute=True)
        scaled_sim_smooth = _smooth_histogram(scaled_sim_density)
        loss = float(np.mean((scaled_sim_smooth - real_smooth) ** 2))
        losses.append(loss)
    best_index = int(np.argmin(losses))
    return float(ALPHA_GRID[best_index]), float(losses[best_index])


def _gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gaussian_to_edge(
    centers: np.ndarray,
    derivative: np.ndarray,
    *,
    side: str,
) -> dict[str, np.ndarray | float]:
    if side == "left":
        search_mask = centers < 0
        search_values = np.where(search_mask, derivative, -np.inf)
        peak_index = int(np.argmax(search_values))
    else:
        search_mask = centers > 0
        search_values = np.where(search_mask, derivative, np.inf)
        peak_index = int(np.argmin(search_values))

    peak_x = float(centers[peak_index])
    peak_y = float(derivative[peak_index])
    if not np.isfinite(peak_x) or not np.isfinite(peak_y):
        return {
            "mu": np.nan,
            "sigma": np.nan,
            "amplitude": np.nan,
            "fit_x": np.array([], dtype=float),
            "fit_y": np.array([], dtype=float),
        }

    fit_mask = np.abs(centers - peak_x) <= EDGE_FIT_HALF_WIDTH_NS
    x_fit = centers[fit_mask]
    y_fit = derivative[fit_mask]
    if x_fit.size < 5:
        return {
            "mu": peak_x,
            "sigma": np.nan,
            "amplitude": peak_y,
            "fit_x": x_fit,
            "fit_y": np.array([], dtype=float),
        }

    sigma0 = max(float(np.std(x_fit)), 0.02)
    if curve_fit is not None:
        try:
            params, _ = curve_fit(
                _gaussian,
                x_fit,
                y_fit,
                p0=(peak_y, peak_x, sigma0),
                bounds=(
                    [peak_y * 4.0 if peak_y < 0 else peak_y * 0.25, peak_x - 0.08, 0.003],
                    [peak_y * 0.25 if peak_y < 0 else peak_y * 4.0, peak_x + 0.08, 0.20],
                ),
                maxfev=5000,
            )
            amp, mu, sigma = params
            fit_y = _gaussian(x_fit, amp, mu, sigma)
            return {
                "mu": float(mu),
                "sigma": abs(float(sigma)),
                "amplitude": float(amp),
                "fit_x": x_fit,
                "fit_y": fit_y,
            }
        except Exception:
            pass

    weights = np.abs(y_fit)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return {
            "mu": peak_x,
            "sigma": np.nan,
            "amplitude": peak_y,
            "fit_x": x_fit,
            "fit_y": np.array([], dtype=float),
        }
    mu = float(np.sum(weights * x_fit) / weight_sum)
    sigma = float(np.sqrt(np.sum(weights * (x_fit - mu) ** 2) / weight_sum))
    amplitude = peak_y
    fit_y = _gaussian(x_fit, amplitude, mu, max(sigma, 1e-6))
    return {
        "mu": mu,
        "sigma": sigma,
        "amplitude": amplitude,
        "fit_x": x_fit,
        "fit_y": fit_y,
    }


def compute_signed_edge_fit_diagnostics(values: np.ndarray) -> dict[str, float]:
    cap = _span_cap(values, values, absolute=False)
    edges = np.linspace(-cap, cap, DEFAULT_BIN_COUNT)
    centers, density = _density_histogram(values, edges, absolute=False)
    smooth = _smooth_histogram(density)
    derivative = np.gradient(smooth, centers)
    left_fit = _fit_gaussian_to_edge(centers, derivative, side="left")
    right_fit = _fit_gaussian_to_edge(centers, derivative, side="right")

    sigmas = [
        float(left_fit["sigma"]) if np.isfinite(left_fit["sigma"]) else np.nan,
        float(right_fit["sigma"]) if np.isfinite(right_fit["sigma"]) else np.nan,
    ]
    sigma_ns = float(np.nanmean(sigmas))
    sigma_x_mm = sigma_ns * STRIP_SIGNAL_SPEED_MM_NS if np.isfinite(sigma_ns) else np.nan
    return {
        "left_mu_ns": float(left_fit["mu"]),
        "right_mu_ns": float(right_fit["mu"]),
        "left_sigma_ns": float(left_fit["sigma"]),
        "right_sigma_ns": float(right_fit["sigma"]),
        "mean_sigma_ns": sigma_ns,
        "mean_sigma_x_mm": sigma_x_mm,
    }


def compute_shape_diagnostics(sim_values: np.ndarray, real_values: np.ndarray) -> dict[str, float]:
    abs_cap = _span_cap(sim_values, real_values, absolute=True)
    abs_edges = np.linspace(0.0, abs_cap, DEFAULT_BIN_COUNT)
    abs_centers, sim_abs_density = _density_histogram(sim_values, abs_edges, absolute=True)
    _, real_abs_density = _density_histogram(real_values, abs_edges, absolute=True)
    sim_abs_smooth = _smooth_histogram(sim_abs_density)
    real_abs_smooth = _smooth_histogram(real_abs_density)
    sim_edge = _estimate_abs_edge_position(sim_values, abs_centers, sim_abs_smooth)
    real_edge = _estimate_abs_edge_position(real_values, abs_centers, real_abs_smooth)
    best_alpha, fit_loss = _fit_histogram_scale_alpha(sim_values, real_values, abs_edges)
    return {
        "sim_edge_abs": sim_edge,
        "real_edge_abs": real_edge,
        "real_over_sim_edge_abs": real_edge / sim_edge,
        "best_hist_scale_alpha": best_alpha,
        "best_hist_scale_loss": fit_loss,
    }


def select_summary_columns(per_strip_df: pd.DataFrame) -> list[str]:
    rng = np.random.default_rng(SUMMARY_STRIP_SELECTION_SEED)
    selected: list[str] = []
    available = set(per_strip_df["column"].astype(str).tolist())
    for plane in range(1, 5):
        candidates = [f"T{plane}_T_dif_{strip}" for strip in range(1, 5) if f"T{plane}_T_dif_{strip}" in available]
        if not candidates:
            continue
        selected.append(str(rng.choice(candidates)))
    return selected


def build_outputs() -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_payload: dict[str, dict[str, object]] = {}

    for station in active_station_configs():
        station_list = SIMULATION_STATIONS if station.label == SIM_LABEL else REAL_STATIONS
        date_ranges = SIMULATION_DATE_RANGES if station.label == SIM_LABEL else REAL_DATE_RANGES
        files, by_column, overall = collect_group_tdif_values(station_list, date_ranges)
        station_payload[station.label] = {
            "files": files,
            "by_column": by_column,
            "overall": overall,
        }

    overall_rows = []
    for station in active_station_configs():
        overall_stats = summarize_distribution(station_payload[station.label]["overall"])  # type: ignore[index]
        overall_rows.append(
            {
                "station": station.label,
                "files": len(station_payload[station.label]["files"]),  # type: ignore[index]
                **overall_stats,
            }
        )
    overall_df = pd.DataFrame(overall_rows)

    per_strip_rows = []
    sim_columns: dict[str, np.ndarray] = station_payload[SIM_LABEL]["by_column"]  # type: ignore[assignment]
    real_columns: dict[str, np.ndarray] = station_payload[REAL_LABEL]["by_column"]  # type: ignore[assignment]
    for column in TDIF_COLUMNS:
        sim_values = sim_columns[column]
        real_values = real_columns[column]
        sim_stats = summarize_distribution(sim_values)
        real_stats = summarize_distribution(real_values)
        shape_stats = compute_shape_diagnostics(sim_values, real_values)
        sim_edge_fit = compute_signed_edge_fit_diagnostics(sim_values)
        real_edge_fit = compute_signed_edge_fit_diagnostics(real_values)
        scaled_sim_values = shape_stats["best_hist_scale_alpha"] * sim_values
        scaled_sim_edge_fit = compute_signed_edge_fit_diagnostics(scaled_sim_values)
        per_strip_rows.append(
            {
                "column": column,
                "sim_count": int(sim_stats["count"]),
                "real_count": int(real_stats["count"]),
                "sim_std": sim_stats["std"],
                "real_std": real_stats["std"],
                "real_over_sim_std": real_stats["std"] / sim_stats["std"],
                "sim_q90_abs": sim_stats["q90_abs"],
                "real_q90_abs": real_stats["q90_abs"],
                "real_over_sim_q90_abs": real_stats["q90_abs"] / sim_stats["q90_abs"],
                "sim_q95_abs": sim_stats["q95_abs"],
                "real_q95_abs": real_stats["q95_abs"],
                "real_over_sim_q95_abs": real_stats["q95_abs"] / sim_stats["q95_abs"],
                "sim_q975_abs": sim_stats["q975_abs"],
                "real_q975_abs": real_stats["q975_abs"],
                "real_over_sim_q975_abs": real_stats["q975_abs"] / sim_stats["q975_abs"],
                "sim_q99_abs": sim_stats["q99_abs"],
                "real_q99_abs": real_stats["q99_abs"],
                "real_over_sim_q99_abs": real_stats["q99_abs"] / sim_stats["q99_abs"],
                "sim_min": sim_stats["min"],
                "sim_max": sim_stats["max"],
                "real_min": real_stats["min"],
                "real_max": real_stats["max"],
                **shape_stats,
                "sim_edge_sigma_ns": sim_edge_fit["mean_sigma_ns"],
                "sim_edge_sigma_x_mm": sim_edge_fit["mean_sigma_x_mm"],
                "real_edge_sigma_ns": real_edge_fit["mean_sigma_ns"],
                "real_edge_sigma_x_mm": real_edge_fit["mean_sigma_x_mm"],
                "scaled_sim_edge_sigma_ns": scaled_sim_edge_fit["mean_sigma_ns"],
                "scaled_sim_edge_sigma_x_mm": scaled_sim_edge_fit["mean_sigma_x_mm"],
            }
        )
    per_strip_df = pd.DataFrame(per_strip_rows)

    recommended_factor = float(per_strip_df["real_over_sim_q95_abs"].median())
    recommended_length_mm = CURRENT_REFERENCE_STRIP_LENGTH_MM * recommended_factor
    summary_df = pd.DataFrame(
        [
            {
                "metric": "median_real_over_sim_std",
                "value": float(per_strip_df["real_over_sim_std"].median()),
            },
            {
                "metric": "median_real_over_sim_q90_abs",
                "value": float(per_strip_df["real_over_sim_q90_abs"].median()),
            },
            {
                "metric": "median_real_over_sim_q95_abs",
                "value": recommended_factor,
            },
            {
                "metric": "median_real_over_sim_q975_abs",
                "value": float(per_strip_df["real_over_sim_q975_abs"].median()),
            },
            {
                "metric": "median_real_over_sim_q99_abs",
                "value": float(per_strip_df["real_over_sim_q99_abs"].median()),
            },
            {
                "metric": "median_real_over_sim_edge_abs",
                "value": float(per_strip_df["real_over_sim_edge_abs"].median()),
            },
            {
                "metric": "median_best_hist_scale_alpha",
                "value": float(per_strip_df["best_hist_scale_alpha"].median()),
            },
            {
                "metric": "median_sim_edge_sigma_ns",
                "value": float(per_strip_df["sim_edge_sigma_ns"].median()),
            },
            {
                "metric": "median_real_edge_sigma_ns",
                "value": float(per_strip_df["real_edge_sigma_ns"].median()),
            },
            {
                "metric": "median_scaled_sim_edge_sigma_ns",
                "value": float(per_strip_df["scaled_sim_edge_sigma_ns"].median()),
            },
            {
                "metric": "median_sim_edge_sigma_x_mm",
                "value": float(per_strip_df["sim_edge_sigma_x_mm"].median()),
            },
            {
                "metric": "median_real_edge_sigma_x_mm",
                "value": float(per_strip_df["real_edge_sigma_x_mm"].median()),
            },
            {
                "metric": "median_scaled_sim_edge_sigma_x_mm",
                "value": float(per_strip_df["scaled_sim_edge_sigma_x_mm"].median()),
            },
            {
                "metric": "recommended_strip_length_scale_factor",
                "value": recommended_factor,
            },
            {
                "metric": "recommended_strip_length_mm_if_current_is_300",
                "value": recommended_length_mm,
            },
        ]
    )
    return station_payload, overall_df, per_strip_df, summary_df


def plot_summary_outputs(
    station_payload: dict[str, dict[str, object]],
    overall_df: pd.DataFrame,
    per_strip_df: pd.DataFrame,
) -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sim_overall = station_payload[SIM_LABEL]["overall"]  # type: ignore[index]
    real_overall = station_payload[REAL_LABEL]["overall"]  # type: ignore[index]
    overall_shape = compute_shape_diagnostics(sim_overall, real_overall)
    signed_cap = _span_cap(sim_overall, real_overall, absolute=False)
    bins = np.linspace(-signed_cap, signed_cap, DEFAULT_BIN_COUNT)
    for station in active_station_configs():
        values = station_payload[station.label]["overall"]  # type: ignore[index]
        axes[0, 0].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=station.color,
            label=station.label,
        )
        axes[0, 1].hist(
            np.abs(values),
            bins=np.linspace(0, _span_cap(sim_overall, real_overall, absolute=True), DEFAULT_BIN_COUNT),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=station.color,
            label=station.label,
        )
    axes[0, 0].hist(
        overall_shape["best_hist_scale_alpha"] * sim_overall,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        linestyle="--",
        color=SCALED_COLOR,
        label=f"{SIM_LABEL} scaled x {overall_shape['best_hist_scale_alpha']:.3f}",
    )
    axes[0, 1].hist(
        np.abs(overall_shape["best_hist_scale_alpha"] * sim_overall),
        bins=np.linspace(0, _span_cap(sim_overall, real_overall, absolute=True), DEFAULT_BIN_COUNT),
        density=True,
        histtype="step",
        linewidth=1.2,
        linestyle="--",
        color=SCALED_COLOR,
        label=f"{SIM_LABEL} scaled x {overall_shape['best_hist_scale_alpha']:.3f}",
    )

    axes[0, 0].set_title("Calibrated non-zero T_dif")
    axes[0, 0].set_xlabel("T_dif [ns]")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].set_title("Absolute calibrated non-zero T_dif")
    axes[0, 1].set_xlabel("|T_dif| [ns]")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    ratio_q95 = per_strip_df["real_over_sim_q95_abs"].to_numpy(dtype=float)
    ratio_alpha = per_strip_df["best_hist_scale_alpha"].to_numpy(dtype=float)
    labels = per_strip_df["column"].tolist()
    x = np.arange(len(labels))

    axes[1, 0].bar(x, ratio_q95, color="#4c78a8")
    axes[1, 0].axhline(np.median(ratio_q95), color="#d62728", linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("Per-strip scale factor from q95(|T_dif|)")
    axes[1, 0].set_ylabel("real / sim")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=90, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].bar(x, ratio_alpha, color="#59a14f")
    axes[1, 1].axhline(np.median(ratio_alpha), color="#d62728", linestyle="--", linewidth=1.5)
    axes[1, 1].set_title("Per-strip scale factor from abs-hist fit")
    axes[1, 1].set_ylabel("real / sim")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=90, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "tdif_span_comparison.png", dpi=180)
    fig.savefig(out_dir / "tdif_span_comparison.pdf")
    plt.close(fig)


def plot_scale_factor_summary(per_strip_df: pd.DataFrame) -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = per_strip_df["column"].tolist()
    x = np.arange(len(labels))
    metrics = [
        ("real_over_sim_q95_abs", "q95(|T_dif|)", "#4c78a8"),
        ("real_over_sim_edge_abs", "Derivative edge", "#f28e2b"),
        ("best_hist_scale_alpha", "Abs-hist fit alpha", "#59a14f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (column, title, color) in zip(axes, metrics, strict=True):
        values = per_strip_df[column].to_numpy(dtype=float)
        ax.bar(x, values, color=color)
        ax.axhline(np.median(values), color="#d62728", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel("real / sim")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "tdif_scale_factor_summary.png", dpi=180)
    fig.savefig(out_dir / "tdif_scale_factor_summary.pdf")
    plt.close(fig)


def plot_selected_strip_signed_overlay(
    station_payload: dict[str, dict[str, object]],
    per_strip_df: pd.DataFrame,
    selected_columns: list[str],
) -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_columns: dict[str, np.ndarray] = station_payload[SIM_LABEL]["by_column"]  # type: ignore[assignment]
    real_columns: dict[str, np.ndarray] = station_payload[REAL_LABEL]["by_column"]  # type: ignore[assignment]
    alpha_lookup = per_strip_df.set_index("column")["best_hist_scale_alpha"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    for idx, column in enumerate(selected_columns):
        ax = axes.flat[idx]
        sim_values = sim_columns[column]
        real_values = real_columns[column]
        alpha = float(alpha_lookup.loc[column])
        cap = _span_cap(sim_values, real_values, absolute=False)
        edges = np.linspace(-cap, cap, DEFAULT_BIN_COUNT)
        ax.hist(sim_values, bins=edges, density=True, histtype="step", linewidth=1.5, color=SIM_COLOR, label=SIM_LABEL)
        ax.hist(real_values, bins=edges, density=True, histtype="step", linewidth=1.5, color=REAL_COLOR, label=REAL_LABEL)
        ax.hist(alpha * sim_values, bins=edges, density=True, histtype="step", linewidth=1.2, linestyle="--", color=SCALED_COLOR, label=f"scaled sim {alpha:.3f}")
        ax.set_title(column, fontsize=10)
        ax.grid(alpha=0.2)
        if idx % 2 == 0:
            ax.set_ylabel("Density")
        if idx >= 2:
            ax.set_xlabel("T_dif [ns]")
        if idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Summary calibrated non-zero T_dif\n"
        f"One reproducible random strip per plane (seed={SUMMARY_STRIP_SELECTION_SEED})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "tdif_summary_selected_signed_overlay.png", dpi=180)
    fig.savefig(out_dir / "tdif_summary_selected_signed_overlay.pdf")
    plt.close(fig)


def plot_selected_strip_abs_overlay(
    station_payload: dict[str, dict[str, object]],
    per_strip_df: pd.DataFrame,
    selected_columns: list[str],
) -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_columns: dict[str, np.ndarray] = station_payload[SIM_LABEL]["by_column"]  # type: ignore[assignment]
    real_columns: dict[str, np.ndarray] = station_payload[REAL_LABEL]["by_column"]  # type: ignore[assignment]
    diagnostics_lookup = per_strip_df.set_index("column")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    for idx, column in enumerate(selected_columns):
        ax = axes.flat[idx]
        sim_values = sim_columns[column]
        real_values = real_columns[column]
        alpha = float(diagnostics_lookup.loc[column, "best_hist_scale_alpha"])
        sim_edge = float(diagnostics_lookup.loc[column, "sim_edge_abs"])
        real_edge = float(diagnostics_lookup.loc[column, "real_edge_abs"])
        cap = _span_cap(sim_values, real_values, absolute=True)
        edges = np.linspace(0.0, cap, DEFAULT_BIN_COUNT)
        ax.hist(np.abs(sim_values), bins=edges, density=True, histtype="step", linewidth=1.5, color=SIM_COLOR, label=SIM_LABEL)
        ax.hist(np.abs(real_values), bins=edges, density=True, histtype="step", linewidth=1.5, color=REAL_COLOR, label=REAL_LABEL)
        ax.hist(np.abs(alpha * sim_values), bins=edges, density=True, histtype="step", linewidth=1.2, linestyle="--", color=SCALED_COLOR, label=f"scaled sim {alpha:.3f}")
        ax.axvline(sim_edge, color=SIM_COLOR, linestyle=":", linewidth=1.2)
        ax.axvline(real_edge, color=REAL_COLOR, linestyle=":", linewidth=1.2)
        ax.set_title(f"{column} | alpha={alpha:.3f}", fontsize=10)
        ax.grid(alpha=0.2)
        if idx % 2 == 0:
            ax.set_ylabel("Density")
        if idx >= 2:
            ax.set_xlabel("|T_dif| [ns]")
        if idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Summary |T_dif| with original and scaled simulation\n"
        f"One reproducible random strip per plane (seed={SUMMARY_STRIP_SELECTION_SEED})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "tdif_summary_selected_abs_overlay.png", dpi=180)
    fig.savefig(out_dir / "tdif_summary_selected_abs_overlay.pdf")
    plt.close(fig)


def plot_selected_strip_signed_derivative_with_gaussian_fits(
    station_payload: dict[str, dict[str, object]],
    per_strip_df: pd.DataFrame,
    selected_columns: list[str],
) -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_columns: dict[str, np.ndarray] = station_payload[SIM_LABEL]["by_column"]  # type: ignore[assignment]
    real_columns: dict[str, np.ndarray] = station_payload[REAL_LABEL]["by_column"]  # type: ignore[assignment]
    alpha_lookup = per_strip_df.set_index("column")["best_hist_scale_alpha"]

    colors = {
        "sim": SIM_COLOR,
        "real": REAL_COLOR,
        "scaled": SCALED_COLOR,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    for idx, column in enumerate(selected_columns):
        ax = axes.flat[idx]
        sim_values = sim_columns[column]
        real_values = real_columns[column]
        alpha = float(alpha_lookup.loc[column])
        scaled_sim_values = alpha * sim_values
        cap = max(
            _span_cap(sim_values, real_values, absolute=False),
            _span_cap(scaled_sim_values, real_values, absolute=False),
        )
        edges = np.linspace(-cap, cap, DEFAULT_BIN_COUNT)

        datasets = [
            ("sim", SIM_LABEL, sim_values, "-"),
            ("real", REAL_LABEL, real_values, "-"),
            ("scaled", f"scaled sim {alpha:.3f}", scaled_sim_values, "--"),
        ]
        sigma_text_lines = []
        for key, label, values, linestyle in datasets:
            centers, density = _density_histogram(values, edges, absolute=False)
            smooth = _smooth_histogram(density)
            derivative = np.gradient(smooth, centers)
            ax.plot(centers, derivative, color=colors[key], linewidth=1.4, linestyle=linestyle, label=label)

            left_fit = _fit_gaussian_to_edge(centers, derivative, side="left")
            right_fit = _fit_gaussian_to_edge(centers, derivative, side="right")
            for fit in (left_fit, right_fit):
                fit_x = np.asarray(fit["fit_x"], dtype=float)
                fit_y = np.asarray(fit["fit_y"], dtype=float)
                if fit_x.size and fit_y.size:
                    ax.plot(fit_x, fit_y, color=colors[key], linewidth=1.0, linestyle=":")
            sigmas = [
                float(left_fit["sigma"]) if np.isfinite(left_fit["sigma"]) else np.nan,
                float(right_fit["sigma"]) if np.isfinite(right_fit["sigma"]) else np.nan,
            ]
            sigma_ns = float(np.nanmean(sigmas))
            sigma_x_mm = sigma_ns * STRIP_SIGNAL_SPEED_MM_NS if np.isfinite(sigma_ns) else np.nan
            short = {"sim": "S", "real": "R", "scaled": "Sx"}[key]
            sigma_text_lines.append(
                f"{short}: {sigma_ns:.3f} ns / {sigma_x_mm:.1f} mm"
                if np.isfinite(sigma_ns)
                else f"{short}: n/a"
            )

        ax.set_title(column, fontsize=10)
        ax.grid(alpha=0.2)
        if idx % 2 == 0:
            ax.set_ylabel("d density / dT_dif")
        if idx >= 2:
            ax.set_xlabel("T_dif [ns]")
        ax.text(
            0.03,
            0.97,
            "\n".join(sigma_text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Summary derivative of T_dif density with Gaussian edge fits\n"
        "Text box: sigma_T and sigma_X from fitted derivative extrema\n"
        f"One reproducible random strip per plane (seed={SUMMARY_STRIP_SELECTION_SEED})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "tdif_summary_selected_signed_derivative_gaussian_fit.png", dpi=180)
    fig.savefig(out_dir / "tdif_summary_selected_signed_derivative_gaussian_fit.pdf")
    plt.close(fig)


def write_report(overall_df: pd.DataFrame, summary_df: pd.DataFrame, selected_columns: list[str]) -> None:
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_lookup = overall_df.set_index("station")
    summary_lookup = summary_df.set_index("metric")["value"]
    report = f"""Task 2 calibrated T_dif span comparison
=================================

Source products:
- Simulation stations: {", ".join(SIMULATION_STATIONS)}
- Real-data stations: {", ".join(REAL_STATIONS)}

Comparison uses only non-zero calibrated strip T_dif values.

Overall counts:
- {SIM_LABEL} files: {int(overall_lookup.loc[SIM_LABEL, 'files'])}
- {REAL_LABEL} files: {int(overall_lookup.loc[REAL_LABEL, 'files'])}
- {SIM_LABEL} non-zero T_dif values: {int(overall_lookup.loc[SIM_LABEL, 'count'])}
- {REAL_LABEL} non-zero T_dif values: {int(overall_lookup.loc[REAL_LABEL, 'count'])}

Summary strips shown in the compact plots:
- {", ".join(selected_columns)}

Overall width indicators:
- std real/sim: {summary_lookup['median_real_over_sim_std']:.6f}
- q90(|T_dif|) real/sim: {summary_lookup['median_real_over_sim_q90_abs']:.6f}
- q95(|T_dif|) real/sim: {summary_lookup['median_real_over_sim_q95_abs']:.6f}
- q97.5(|T_dif|) real/sim: {summary_lookup['median_real_over_sim_q975_abs']:.6f}
- q99(|T_dif|) real/sim: {summary_lookup['median_real_over_sim_q99_abs']:.6f}
- derivative-edge real/sim: {summary_lookup['median_real_over_sim_edge_abs']:.6f}
- abs-hist fit alpha: {summary_lookup['median_best_hist_scale_alpha']:.6f}
- median sim edge sigma_T: {summary_lookup['median_sim_edge_sigma_ns']:.6f} ns
- median real edge sigma_T: {summary_lookup['median_real_edge_sigma_ns']:.6f} ns
- median scaled-sim edge sigma_T: {summary_lookup['median_scaled_sim_edge_sigma_ns']:.6f} ns
- median sim edge sigma_X: {summary_lookup['median_sim_edge_sigma_x_mm']:.3f} mm
- median real edge sigma_X: {summary_lookup['median_real_edge_sigma_x_mm']:.3f} mm
- median scaled-sim edge sigma_X: {summary_lookup['median_scaled_sim_edge_sigma_x_mm']:.3f} mm

Recommended scaling:
- Recommended strip-length scale factor: {summary_lookup['recommended_strip_length_scale_factor']:.6f}
- If the current strip length is {CURRENT_REFERENCE_STRIP_LENGTH_MM:.1f} mm, suggested effective length: {summary_lookup['recommended_strip_length_mm_if_current_is_300']:.2f} mm

Interpretation:
- The central calibrated T_dif distribution in real data is narrower than in simulation.
- Raw extrema are not reliable for scaling because real data has a few larger tails.
- The robust q95(|T_dif|) ratio is the recommended scale factor for reducing the simulated strip length.
- The derivative-edge and abs-hist-fit diagnostics are included to make the square-width interpretation easier to inspect by eye.
"""
    (out_dir / "tdif_span_report.txt").write_text(report, encoding="utf-8")


def configure_from_tuning_config(config_path: str | Path | None) -> None:
    global SIM_LABEL
    global REAL_LABEL
    global SIMULATION_STATIONS
    global REAL_STATIONS
    global SIMULATION_DATE_RANGES
    global REAL_DATE_RANGES

    if config_path is None:
        return

    config = load_tuning_config(config_path)
    selection = resolve_selection(config)
    SIMULATION_STATIONS = selection.simulation_stations
    REAL_STATIONS = selection.real_stations
    SIMULATION_DATE_RANGES = selection.simulation_date_ranges
    REAL_DATE_RANGES = selection.real_date_ranges
    SIM_LABEL = group_label(SIMULATION_STATIONS, fallback="SIM_SELECTED")
    REAL_LABEL = group_label(REAL_STATIONS, fallback="REAL_SELECTED")

    tdif_cfg = config.get("tdif_active_length", {})
    if "reference_strip_length_mm" in tdif_cfg:
        global CURRENT_REFERENCE_STRIP_LENGTH_MM
        CURRENT_REFERENCE_STRIP_LENGTH_MM = float(tdif_cfg["reference_strip_length_mm"])
    if "summary_strip_selection_seed" in tdif_cfg:
        global SUMMARY_STRIP_SELECTION_SEED
        SUMMARY_STRIP_SELECTION_SEED = int(tdif_cfg["summary_strip_selection_seed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare calibrated Task 2 T_dif widths between simulation and real data.")
    parser.add_argument(
        "--config",
        default=None,
        help="Shared simulation-tuning config file. If omitted, keep the legacy MINGO00 vs MINGO01 comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_from_tuning_config(args.config)
    if args.output_dir:
        global OUTPUT_DIR_OVERRIDE
        OUTPUT_DIR_OVERRIDE = Path(args.output_dir)

    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    station_payload, overall_df, per_strip_df, summary_df = build_outputs()
    selected_columns = select_summary_columns(per_strip_df)
    overall_df.to_csv(out_dir / "overall_tdif_summary.csv", index=False)
    per_strip_df.to_csv(out_dir / "per_strip_tdif_summary.csv", index=False)
    summary_df.to_csv(out_dir / "recommended_scale_summary.csv", index=False)
    pd.DataFrame(
        {
            "selected_column": selected_columns,
            "selection_seed": [SUMMARY_STRIP_SELECTION_SEED] * len(selected_columns),
        }
    ).to_csv(out_dir / "selected_summary_strips.csv", index=False)
    plot_summary_outputs(station_payload, overall_df, per_strip_df)
    plot_scale_factor_summary(per_strip_df)
    plot_selected_strip_signed_overlay(station_payload, per_strip_df, selected_columns)
    plot_selected_strip_abs_overlay(station_payload, per_strip_df, selected_columns)
    plot_selected_strip_signed_derivative_with_gaussian_fits(
        station_payload,
        per_strip_df,
        selected_columns,
    )
    write_report(overall_df, summary_df, selected_columns)


if __name__ == "__main__":
    main()
