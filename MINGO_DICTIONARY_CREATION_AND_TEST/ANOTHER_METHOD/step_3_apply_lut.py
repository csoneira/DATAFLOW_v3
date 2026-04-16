#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CANONICAL_EFF_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    apply_lut_fallback_matches,
    cfg_path,
    derive_trigger_rate_features,
    ensure_output_dirs,
    format_selected_rate_name,
    get_rate_column_name,
    load_config,
    quantize_efficiency_series,
    read_ascii_lut,
    write_json,
)

log = logging.getLogger("another_method.step3")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_3 - %(message)s", level=logging.INFO, force=True)


def _prepare_synthetic_dataframe(dataframe: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    z_columns = list(config["columns"]["z_positions"])
    work, trigger_info = derive_trigger_rate_features(
        dataframe,
        config,
        allow_plain_fallback=True,
    )
    rename_map = {
        "eff_empirical_1": CANONICAL_EFF_COLUMNS[0],
        "eff_empirical_2": CANONICAL_EFF_COLUMNS[1],
        "eff_empirical_3": CANONICAL_EFF_COLUMNS[2],
        "eff_empirical_4": CANONICAL_EFF_COLUMNS[3],
        z_columns[0]: "z_pos_1",
        z_columns[1]: "z_pos_2",
        z_columns[2]: "z_pos_3",
        z_columns[3]: "z_pos_4",
        str(config["columns"]["simulated_flux"]): "sim_flux_cm2_min",
    }
    return work.rename(columns=rename_map), trigger_info


def _resolve_lut_match_settings(config: dict) -> tuple[str, int | None, float]:
    step3_config = config.get("step3", {})
    if not isinstance(step3_config, dict):
        step3_config = {}

    raw_mode = step3_config.get("lut_match_mode")
    if raw_mode in (None, "", "null", "None"):
        allow_nearest = bool(step3_config.get("allow_nearest_lut_match", True))
        match_mode = "nearest" if allow_nearest else "exact"
    else:
        normalized = str(raw_mode).strip().lower()
        match_mode = {
            "idw": "interpolate",
            "interpolated": "interpolate",
            "interpolation": "interpolate",
        }.get(normalized, normalized)

    interpolation_k_raw = step3_config.get("lut_interpolation_k", 8)
    interpolation_k = None if interpolation_k_raw in (None, "", "null", "None") else int(interpolation_k_raw)

    interpolation_power_raw = step3_config.get("lut_interpolation_power", 2.0)
    interpolation_power = (
        2.0 if interpolation_power_raw in (None, "", "null", "None") else float(interpolation_power_raw)
    )
    return match_mode, interpolation_k, interpolation_power


def _resolve_time_axis(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    for candidate in ("time_utc", "time_start_utc", "time_end_utc"):
        if candidate not in dataframe.columns:
            continue
        parsed = pd.to_datetime(dataframe[candidate], errors="coerce", utc=True)
        if parsed.notna().any():
            order = np.argsort(parsed.fillna(parsed.min()))
            ordered = dataframe.iloc[order].reset_index(drop=True)
            ordered_time = parsed.iloc[order].reset_index(drop=True)
            return ordered, ordered_time, candidate

    ordered = dataframe.reset_index(drop=True).copy()
    return ordered, pd.Series(np.arange(len(ordered)), dtype=float), "synthetic_dataset_row"


def _plot_time_series_with_flux(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        height_ratios=[2.0, 1.0, 1.2],
    )

    axes[0].plot(x_values, ordered["rate_hz"], marker="o", linewidth=1.6, label="Observed rate")
    axes[0].plot(
        x_values,
        ordered["corrected_rate_to_perfect_hz"],
        marker="o",
        linewidth=1.6,
        label="LUT-corrected rate",
    )
    axes[0].set_ylabel("Rate [Hz]")
    axes[0].set_title(
        "Observed and corrected rate with flux time series\n"
        f"rate column: {rate_column_name}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x_values,
        ordered["sim_flux_cm2_min"],
        marker="o",
        linewidth=1.6,
        color="#B22222",
        label="Simulated flux",
    )
    axes[1].set_xlabel(x_label.replace("_", " "))
    axes[1].set_ylabel("Flux [cm^-2 min^-1]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plane_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(CANONICAL_EFF_COLUMNS):
        axes[2].plot(
            x_values,
            ordered[column],
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=plane_colors[idx],
            label=f"Plane {idx + 1} eff",
        )
    axes[2].set_xlabel(x_label.replace("_", " "))
    axes[2].set_ylabel("Empirical efficiency")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=2)

    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_flux_rate_comparison(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    def _plot_linear_fit(
        axis: plt.Axes,
        x_values: pd.Series,
        y_values: pd.Series,
        *,
        fit_color: str,
    ) -> None:
        fit_frame = pd.DataFrame(
            {
                "x": pd.to_numeric(x_values, errors="coerce"),
                "y": pd.to_numeric(y_values, errors="coerce"),
            }
        ).dropna()
        if len(fit_frame) < 2 or fit_frame["x"].nunique() < 2:
            return

        x_data = fit_frame["x"].to_numpy(dtype=float)
        y_data = fit_frame["y"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_data, y_data, deg=1)
        fit_values = slope * x_data + intercept
        ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
        ss_res = float(np.sum((y_data - fit_values) ** 2))
        r_squared = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

        x_line = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 200)
        y_line = slope * x_line + intercept
        r2_text = "n/a" if not np.isfinite(r_squared) else f"{r_squared:.3f}"
        axis.plot(
            x_line,
            y_line,
            color=fit_color,
            linewidth=2.0,
            label=f"Linear fit (R^2={r2_text})",
        )

    ordered, _, _ = _resolve_time_axis(dataframe)
    sequence = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    scatter_left = axes[0].scatter(
        ordered["sim_flux_cm2_min"],
        ordered["rate_hz"],
        c=sequence,
        cmap="viridis",
        s=44,
        alpha=0.85,
        label="Samples",
    )
    _plot_linear_fit(
        axes[0],
        ordered["sim_flux_cm2_min"],
        ordered["rate_hz"],
        fit_color="#B22222",
    )
    axes[0].set_title(f"Flux vs observed rate\nrate column: {rate_column_name}")
    axes[0].set_xlabel("Flux [cm^-2 min^-1]")
    axes[0].set_ylabel(f"Observed rate [Hz]\n({rate_column_name})")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(
        ordered["sim_flux_cm2_min"],
        ordered["corrected_rate_to_perfect_hz"],
        c=sequence,
        cmap="viridis",
        s=44,
        alpha=0.85,
        label="Samples",
    )
    _plot_linear_fit(
        axes[1],
        ordered["sim_flux_cm2_min"],
        ordered["corrected_rate_to_perfect_hz"],
        fit_color="#8B1E3F",
    )
    axes[1].set_title(f"Flux vs corrected rate\nrate column: {rate_column_name}")
    axes[1].set_xlabel("Flux [cm^-2 min^-1]")
    axes[1].set_ylabel(f"Corrected rate [Hz]\n(from {rate_column_name})")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    cbar = fig.colorbar(scatter_left, ax=axes, shrink=0.95)
    cbar.set_label("Time-series order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_flux_reference_curve(flux_cells_path: Path) -> pd.DataFrame:
    if not flux_cells_path.exists():
        raise FileNotFoundError(
            f"Step 2 flux-cell diagnostics not found: {flux_cells_path}. "
            "Run Step 2 before Step 3 to build the almost-perfect flux reference."
        )

    flux_cells = pd.read_csv(flux_cells_path)
    required_columns = [
        "flux_bin_index",
        "flux_bin_lo",
        "flux_bin_hi",
        "flux_bin_center",
        "reference_rate_median",
    ]
    missing = [column for column in required_columns if column not in flux_cells.columns]
    if missing:
        raise ValueError(
            "Step 2 flux-cell diagnostics are missing required columns: " + ", ".join(missing)
        )

    curve = (
        flux_cells.groupby("flux_bin_index", dropna=False)
        .agg(
            flux_bin_lo=("flux_bin_lo", "min"),
            flux_bin_hi=("flux_bin_hi", "max"),
            flux_bin_center=("flux_bin_center", "median"),
            reference_rate_median=("reference_rate_median", "median"),
        )
        .reset_index()
        .sort_values("flux_bin_index")
        .reset_index(drop=True)
    )
    curve = curve.dropna(subset=["flux_bin_lo", "flux_bin_hi", "reference_rate_median"])
    if curve.empty:
        raise ValueError("Step 2 flux-cell diagnostics contain no usable reference-rate rows.")
    return curve


def _map_reference_rate_by_flux(
    flux_values: pd.Series,
    reference_curve: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    flux_numeric = pd.to_numeric(flux_values, errors="coerce")
    bin_edges = [float(reference_curve["flux_bin_lo"].iloc[0])] + [
        float(value) for value in reference_curve["flux_bin_hi"].tolist()
    ]
    if len(bin_edges) < 2 or any(high <= low for low, high in zip(bin_edges[:-1], bin_edges[1:])):
        raise ValueError("Invalid Step 2 flux-bin edges while building almost-perfect reference mapping.")

    labels = [int(value) for value in reference_curve["flux_bin_index"].tolist()]
    binned = pd.cut(
        flux_numeric,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    reference_rate_by_bin = reference_curve.set_index("flux_bin_index")["reference_rate_median"]
    mapped = binned.map(reference_rate_by_bin)
    mapped = pd.to_numeric(mapped, errors="coerce")

    assignment_method = pd.Series("flux_bin_edges", index=flux_numeric.index, dtype="object")
    missing_mask = flux_numeric.notna() & mapped.isna()
    if missing_mask.any():
        centers = reference_curve["flux_bin_center"].to_numpy(dtype=float)
        reference_values = reference_curve["reference_rate_median"].to_numpy(dtype=float)
        flux_missing = flux_numeric.loc[missing_mask].to_numpy(dtype=float)
        nearest_indices = np.abs(flux_missing[:, None] - centers[None, :]).argmin(axis=1)
        mapped.loc[missing_mask] = reference_values[nearest_indices]
        assignment_method.loc[missing_mask] = "nearest_flux_bin_center"

    assignment_method.loc[flux_numeric.isna()] = "missing_flux"
    return mapped, assignment_method


def _plot_corrected_vs_almost_perfect_reference(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> dict[str, float | int | None]:
    ordered, _, _ = _resolve_time_axis(dataframe)
    compare = pd.DataFrame(
        {
            "reference": pd.to_numeric(ordered["reference_rate_almost_perfect_hz"], errors="coerce"),
            "corrected": pd.to_numeric(ordered["corrected_rate_to_perfect_hz"], errors="coerce"),
        }
    )
    valid_mask = compare["reference"].notna() & compare["corrected"].notna()
    valid = compare.loc[valid_mask].copy()

    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    metrics: dict[str, float | int | None] = {
        "rows_with_reference": int(valid_mask.sum()),
        "rows_without_reference": int((~valid_mask).sum()),
        "linear_fit_r_squared": None,
        "mean_absolute_error_hz": None,
        "rmse_hz": None,
        "mean_bias_hz": None,
    }

    if valid.empty:
        ax.text(
            0.5,
            0.5,
            "No valid points for corrected/reference comparison",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return metrics

    sequence = np.arange(len(ordered))
    scatter = ax.scatter(
        valid["reference"],
        valid["corrected"],
        c=sequence[valid_mask.to_numpy()],
        cmap="viridis",
        s=46,
        alpha=0.85,
        label="Samples",
    )

    line_min = float(np.nanmin(np.concatenate([valid["reference"].to_numpy(), valid["corrected"].to_numpy()])))
    line_max = float(np.nanmax(np.concatenate([valid["reference"].to_numpy(), valid["corrected"].to_numpy()])))
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.75,
        label="1:1 line",
    )

    if len(valid) >= 2 and valid["reference"].nunique() >= 2:
        x_data = valid["reference"].to_numpy(dtype=float)
        y_data = valid["corrected"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_data, y_data, deg=1)
        y_fit = slope * x_data + intercept
        ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
        ss_res = float(np.sum((y_data - y_fit) ** 2))
        r_squared = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
        x_line = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#8B1E3F",
            linewidth=2.0,
            label=("Linear fit (R^2=n/a)" if not np.isfinite(r_squared) else f"Linear fit (R^2={r_squared:.3f})"),
        )
        metrics["linear_fit_r_squared"] = None if not np.isfinite(r_squared) else float(r_squared)

    residual = valid["corrected"] - valid["reference"]
    metrics["mean_absolute_error_hz"] = float(residual.abs().mean())
    metrics["rmse_hz"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["mean_bias_hz"] = float(residual.mean())

    ax.set_title(
        "Corrected rate vs almost-perfect flux reference\n"
        f"rate column: {rate_column_name}"
    )
    ax.set_xlabel("Almost-perfect reference rate [Hz] (from Step 2 flux bins)")
    ax.set_ylabel(f"Corrected rate [Hz] (from {rate_column_name})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.94)
    cbar.set_label("Time-series order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return metrics


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    synthetic_input_path = cfg_path(config, "paths", "synthetic_dataset_csv")
    lut_path = cfg_path(config, "paths", "step2_lut_ascii")
    flux_cells_path = cfg_path(config, "paths", "step2_flux_cells_csv")
    lut_meta_path = cfg_path(config, "paths", "step2_meta_json")
    output_path = cfg_path(config, "paths", "step3_output_csv")
    meta_path = cfg_path(config, "paths", "step3_meta_json")

    synthetic_dataframe = pd.read_csv(synthetic_input_path)
    work, trigger_info = _prepare_synthetic_dataframe(synthetic_dataframe.copy(), config)
    rate_column_name = format_selected_rate_name(
        stage_prefix=str(trigger_info["stage_prefix"]),
        rate_family_column=str(trigger_info["rate_family_column"]),
        offender_threshold=trigger_info.get("used_offender_threshold"),
    )
    renamed_required_columns = [column for column in ["rate_hz", "sim_flux_cm2_min", *CANONICAL_EFF_COLUMNS] if column not in work.columns]
    if renamed_required_columns:
        raise ValueError("Synthetic dataset is missing derived trigger-feature columns: " + ", ".join(renamed_required_columns))
    lut_dataframe, lut_comments = read_ascii_lut(lut_path)

    lut_meta = {}
    if lut_meta_path.exists():
        lut_meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
    efficiency_bin_width = float(
        lut_meta.get("efficiency_bin_width", config.get("step2", {}).get("efficiency_bin_width", 0.02))
    )

    query_columns: list[str] = []
    for column in CANONICAL_EFF_COLUMNS:
        query_column = f"query_{column}"
        work[query_column] = quantize_efficiency_series(work[column], efficiency_bin_width)
        query_columns.append(query_column)

    lut_lookup = lut_dataframe.rename(columns={column: f"lut_{column}" for column in CANONICAL_EFF_COLUMNS})
    merged = work.merge(
        lut_lookup,
        how="left",
        left_on=query_columns,
        right_on=[f"lut_{column}" for column in CANONICAL_EFF_COLUMNS],
    )
    merged = merged.rename(columns={"scale_factor": "lut_scale_factor"})
    merged["lut_match_method"] = np.where(merged["lut_scale_factor"].notna(), "exact", pd.NA)
    merged["lut_match_distance"] = np.where(merged["lut_scale_factor"].notna(), 0.0, np.nan)

    match_mode, interpolation_k, interpolation_power = _resolve_lut_match_settings(config)
    merged = apply_lut_fallback_matches(
        merged,
        lut_dataframe,
        query_columns=query_columns,
        raw_columns=CANONICAL_EFF_COLUMNS,
        match_mode=match_mode,
        interpolation_k=interpolation_k,
        interpolation_power=interpolation_power,
    )

    merged["corrected_rate_to_perfect_hz"] = merged["rate_hz"] * merged["lut_scale_factor"]
    reference_curve = _load_flux_reference_curve(flux_cells_path)
    (
        merged["reference_rate_almost_perfect_hz"],
        merged["reference_rate_assignment_method"],
    ) = _map_reference_rate_by_flux(merged["sim_flux_cm2_min"], reference_curve)
    merged["corrected_minus_reference_hz"] = (
        pd.to_numeric(merged["corrected_rate_to_perfect_hz"], errors="coerce")
        - pd.to_numeric(merged["reference_rate_almost_perfect_hz"], errors="coerce")
    )
    merged["selected_z_vector_match"] = True
    selected_z_positions = lut_meta.get("selected_z_positions")
    if selected_z_positions:
        z_match = np.ones(len(merged), dtype=bool)
        for idx, z_value in enumerate(selected_z_positions, start=1):
            z_match &= np.isclose(merged[f"z_pos_{idx}"].astype(float), float(z_value))
        merged["selected_z_vector_match"] = z_match

    output_dataframe = synthetic_dataframe.copy()
    for column in [
        "two_plane_rate_hz",
        "three_plane_rate_hz",
        "four_plane_rate_hz",
        "three_and_four_plane_rate_hz",
        "two_and_three_plane_rate_hz",
        "total_rate_hz",
        "selected_rate_hz",
        "selected_rate_count",
        "eff_empirical_1",
        "eff_empirical_2",
        "eff_empirical_3",
        "eff_empirical_4",
    ]:
        if column in merged.columns:
            output_dataframe[column] = merged[column]
    for column in query_columns:
        output_dataframe[column] = merged[column]
    for column in CANONICAL_EFF_COLUMNS:
        output_dataframe[f"lut_{column}"] = merged[f"lut_{column}"]
    output_dataframe["lut_scale_factor"] = merged["lut_scale_factor"]
    output_dataframe["lut_match_method"] = merged["lut_match_method"]
    output_dataframe["lut_match_distance"] = merged["lut_match_distance"]
    output_dataframe["lut_neighbor_count"] = merged["lut_neighbor_count"]
    output_dataframe["lut_neighbor_min_distance"] = merged["lut_neighbor_min_distance"]
    output_dataframe["lut_neighbor_max_distance"] = merged["lut_neighbor_max_distance"]
    output_dataframe["corrected_rate_to_perfect_hz"] = merged["corrected_rate_to_perfect_hz"]
    output_dataframe["reference_rate_almost_perfect_hz"] = merged["reference_rate_almost_perfect_hz"]
    output_dataframe["reference_rate_assignment_method"] = merged["reference_rate_assignment_method"]
    output_dataframe["corrected_minus_reference_hz"] = merged["corrected_minus_reference_hz"]
    output_dataframe["selected_z_vector_match"] = merged["selected_z_vector_match"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, index=False)
    _plot_time_series_with_flux(
        merged,
        PLOTS_DIR / "step3_rate_correction.png",
        rate_column_name=rate_column_name,
    )
    _plot_flux_rate_comparison(
        merged,
        PLOTS_DIR / "step3_flux_rate_comparison.png",
        rate_column_name=rate_column_name,
    )
    corrected_vs_reference_metrics = _plot_corrected_vs_almost_perfect_reference(
        merged,
        PLOTS_DIR / "step3_corrected_vs_almost_perfect_reference.png",
        rate_column_name=rate_column_name,
    )

    metadata = {
        "source_file": str(synthetic_input_path),
        "lut_file": str(lut_path),
        "flux_reference_file": str(flux_cells_path),
        "lut_comments": lut_comments,
        "efficiency_bin_width": efficiency_bin_width,
        "lut_match_mode_requested": match_mode,
        "lut_interpolation_k": interpolation_k,
        "lut_interpolation_power": interpolation_power,
        "row_count": int(len(output_dataframe)),
        "exact_matches": int((output_dataframe["lut_match_method"] == "exact").sum()),
        "nearest_matches": int((output_dataframe["lut_match_method"] == "nearest").sum()),
        "interpolated_matches": int((output_dataframe["lut_match_method"] == "interpolated").sum()),
        "unmatched_rows": int(output_dataframe["lut_scale_factor"].isna().sum()),
        "rows_matching_selected_z_vector": int(output_dataframe["selected_z_vector_match"].sum()),
        "rate_input_column": rate_column_name,
        "trigger_type_selection": trigger_info,
        "almost_perfect_flux_reference": {
            "flux_bins": int(len(reference_curve)),
            "reference_flux_bin_centers": [float(value) for value in reference_curve["flux_bin_center"].tolist()],
            "reference_rates_hz": [float(value) for value in reference_curve["reference_rate_median"].tolist()],
            "assignment_method_counts": {
                str(key): int(value)
                for key, value in merged["reference_rate_assignment_method"].value_counts(dropna=False).to_dict().items()
            },
            "comparison_metrics": corrected_vs_reference_metrics,
        },
        "time_axis_column_used_for_plots": _resolve_time_axis(merged)[2],
    }
    write_json(meta_path, metadata)

    log.info("Wrote Step 3 output with LUT corrections to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the scale-factor LUT to the synthetic dataset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
