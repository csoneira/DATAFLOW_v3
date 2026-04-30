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
    CANONICAL_Z_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX,
    TASK_FINAL_STAGE_PREFIX,
    TT_FOUR_PLANE_LABEL,
    TT_RATE_LABELS,
    TT_THREE_PLANE_LABELS,
    TT_TWO_PLANE_LABELS,
    apply_lut_fallback_matches,
    cfg_path,
    derive_trigger_rate_features,
    ensure_output_dirs,
    format_selected_rate_name,
    get_rate_column_name,
    get_trigger_type_selection,
    load_config,
    quantize_efficiency_series,
    read_ascii_lut,
    write_json,
)
from multi_z_support import load_reference_curve_table, map_reference_rate_by_flux, reference_table_has_z, unique_z_vectors, z_mask_for_vector

log = logging.getLogger("another_method.step3")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_3 - %(message)s", level=logging.INFO, force=True)


def _resolve_step3_input_dataframe(config: dict) -> tuple[pd.DataFrame, Path, str]:
    step3_config = config.get("step3", {})
    if not isinstance(step3_config, dict):
        step3_config = {}

    raw_mode = str(step3_config.get("input_source", "training_dataset")).strip().lower()
    input_mode = {
        "training": "training_dataset",
        "training_dataset": "training_dataset",
        "step1": "training_dataset",
        "step1_filtered": "training_dataset",
        "step1_filtered_csv": "training_dataset",
        "synthetic": "legacy_synthetic_dataset",
        "synthetic_dataset": "legacy_synthetic_dataset",
        "legacy": "legacy_synthetic_dataset",
        "legacy_synthetic": "legacy_synthetic_dataset",
        "legacy_synthetic_dataset": "legacy_synthetic_dataset",
    }.get(raw_mode, raw_mode)

    if input_mode == "training_dataset":
        input_path = cfg_path(config, "paths", "step1_filtered_csv")
    elif input_mode == "legacy_synthetic_dataset":
        input_path = cfg_path(config, "paths", "synthetic_dataset_csv")
    else:
        raise ValueError(
            "Unsupported step3.input_source: "
            f"{step3_config.get('input_source')!r}. Supported values are "
            "'training_dataset' and 'legacy_synthetic_dataset'."
        )

    return pd.read_csv(input_path, low_memory=False), input_path, input_mode


def _derive_robust_synthetic_features(
    dataframe: pd.DataFrame,
    selection: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = dataframe.copy()
    empirical_columns = [f"eff_empirical_{idx}" for idx in range(1, 5)]
    selected_rate_column = str(selection["rate_family_column"])
    selected_source_rate_column = str(selection["selected_source_rate_column"])
    efficiency_variant = str(selection.get("robust_efficiency_variant", "default"))
    efficiency_suffix = ROBUST_EFFICIENCY_VARIANT_TO_SUFFIX[efficiency_variant]
    source_mode: str | None = None
    used_stage_prefix: str | None = None

    direct_robust_columns = [f"eff{idx}{efficiency_suffix}" for idx in range(1, 5)] + ["rate_1234_hz", "rate_total_hz"]
    if selected_source_rate_column not in direct_robust_columns:
        direct_robust_columns.append(selected_source_rate_column)
    if all(column in out.columns for column in direct_robust_columns):
        source_mode = "robust_efficiency_columns"
        source_rate_columns = {
            "four_plane": "rate_1234_hz",
            "four_plane_robust_hz": "four_plane_robust_hz" if "four_plane_robust_hz" in out.columns else None,
            "total": "rate_total_hz",
        }
        out["four_plane_rate_hz"] = pd.to_numeric(out["rate_1234_hz"], errors="coerce")
        out["four_plane_robust_hz"] = (
            pd.to_numeric(out["four_plane_robust_hz"], errors="coerce")
            if "four_plane_robust_hz" in out.columns
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
        out["total_rate_hz"] = pd.to_numeric(out["rate_total_hz"], errors="coerce")
        for plane_idx in range(1, 5):
            out[f"eff_empirical_{plane_idx}"] = pd.to_numeric(out[f"eff{plane_idx}{efficiency_suffix}"], errors="coerce")
    canonical_required_columns = empirical_columns + ["four_plane_rate_hz", "total_rate_hz"]
    if selected_rate_column not in canonical_required_columns:
        canonical_required_columns.append(selected_rate_column)
    if source_mode is None and all(column in out.columns for column in canonical_required_columns):
        source_mode = "canonical_training_columns"
        source_rate_columns = {
            "four_plane": "four_plane_rate_hz",
            "four_plane_robust_hz": "four_plane_robust_hz" if "four_plane_robust_hz" in out.columns else None,
            "total": "total_rate_hz",
        }
    elif source_mode is None:
        missing_empirical = [column for column in empirical_columns if column not in out.columns]
        if missing_empirical:
            raise KeyError(
                "Synthetic dataset is missing empirical-efficiency columns required for robust mode: "
                + ", ".join(missing_empirical)
            )

        stage_candidates = [
            TASK_FINAL_STAGE_PREFIX[task_id]
            for task_id in sorted(TASK_FINAL_STAGE_PREFIX.keys(), reverse=True)
        ]
        stage_sources: dict[str, str] | None = None
        for stage_prefix in stage_candidates:
            candidate_sources = {
                label: f"{stage_prefix}_{label}_rate_hz"
                for label in TT_RATE_LABELS
            }
            missing_candidate = [column for column in candidate_sources.values() if column not in out.columns]
            if missing_candidate:
                continue
            stage_sources = candidate_sources
            used_stage_prefix = stage_prefix
            break

        if stage_sources is None:
            raise KeyError(
                "Synthetic dataset cannot satisfy robust mode because it has neither robust-efficiency "
                "columns nor a full trigger-type stage such as post_tt_*_rate_hz."
            )

        if selected_rate_column == "four_plane_robust_hz":
            raise KeyError(
                "Synthetic dataset cannot satisfy robust mode for four_plane_robust_hz because it has "
                "no direct robust-efficiency rate column. Rebuild the Step 1 training dataset with "
                "trigger_type_selection.rate_family = 'four_plane_robust_hz'."
            )

        component_rates = {
            label: pd.to_numeric(out[column], errors="coerce")
            for label, column in stage_sources.items()
        }
        out["two_plane_rate_hz"] = pd.DataFrame(
            {label: component_rates[label] for label in TT_TWO_PLANE_LABELS}
        ).sum(axis=1, min_count=1)
        out["three_plane_rate_hz"] = pd.DataFrame(
            {label: component_rates[label] for label in TT_THREE_PLANE_LABELS}
        ).sum(axis=1, min_count=1)
        out["four_plane_rate_hz"] = component_rates[TT_FOUR_PLANE_LABEL]
        out["four_plane_robust_hz"] = pd.Series(np.nan, index=out.index, dtype=float)
        out["three_and_four_plane_rate_hz"] = out["three_plane_rate_hz"] + out["four_plane_rate_hz"]
        out["two_and_three_plane_rate_hz"] = out["two_plane_rate_hz"] + out["three_plane_rate_hz"]
        out["total_rate_hz"] = out["two_plane_rate_hz"] + out["three_plane_rate_hz"] + out["four_plane_rate_hz"]
        source_mode = "trigger_type_columns_fallback"
        source_rate_columns = {
            "four_plane": stage_sources[TT_FOUR_PLANE_LABEL],
            "four_plane_robust_hz": None,
            "total": "derived_total_rate_hz_from_trigger_type_columns",
        }

    for column in [
        "two_plane_rate_hz",
        "three_plane_rate_hz",
        "four_plane_rate_hz",
        "four_plane_robust_hz",
        "three_and_four_plane_rate_hz",
        "two_and_three_plane_rate_hz",
        "total_rate_hz",
    ]:
        if column not in out.columns:
            out[column] = pd.Series(np.nan, index=out.index, dtype=float)

    for column in [
        "two_plane_count",
        "three_plane_count",
        "four_plane_count",
        "four_plane_robust_count",
        "three_and_four_plane_count",
        "two_and_three_plane_count",
        "total_count",
    ]:
        if column not in out.columns:
            out[column] = pd.Series(np.nan, index=out.index, dtype=float)

    if str(selection.get("metadata_source", "trigger_type")) == "robust_efficiency":
        selected_count_column = {
            "four_plane_rate_hz": "four_plane_count",
            "four_plane_robust_hz": "four_plane_robust_count",
            "total_rate_hz": "total_count",
        }.get(selected_rate_column, selected_rate_column.replace("_rate_hz", "_count"))
    else:
        selected_count_column = selected_rate_column.replace("_rate_hz", "_count")
    out["selected_rate_hz"] = pd.to_numeric(out[selected_rate_column], errors="coerce")
    if selected_count_column in out.columns:
        out["selected_rate_count"] = pd.to_numeric(out[selected_count_column], errors="coerce")
    elif "selected_rate_count" not in out.columns:
        out["selected_rate_count"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["rate_hz"] = out["selected_rate_hz"]

    metadata = {
        "metadata_source": selection["metadata_source"],
        "source_name": selection["source_name"],
        "task_id": int(selection["task_id"]),
        "stage_prefix": used_stage_prefix,
        "requested_stage_prefix": None,
        "used_stage_prefix": used_stage_prefix,
        "stage_prefix_fallback_used": bool(used_stage_prefix is not None),
        "requested_offender_threshold": None,
        "used_offender_threshold": None,
        "rate_family": selection["rate_family"],
        "rate_family_column": selected_rate_column,
        "selected_source_rate_column": selection["selected_source_rate_column"],
        "robust_efficiency_variant": efficiency_variant,
        "plain_column_fallback_used": bool(source_mode != "robust_efficiency_columns"),
        "source_rate_columns": source_rate_columns,
        "source_count_columns": {
            "four_plane": None,
            "four_plane_robust_hz": None,
            "total": None,
        },
        "synthetic_source_mode": source_mode,
    }
    return out, metadata


def _prepare_synthetic_dataframe(dataframe: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    z_columns = list(config["columns"]["z_positions"])
    selection = get_trigger_type_selection(config)
    if str(selection.get("metadata_source", "trigger_type")) == "robust_efficiency":
        work, trigger_info = _derive_robust_synthetic_features(dataframe, selection)
    else:
        work, trigger_info = derive_trigger_rate_features(
            dataframe,
            config,
            allow_plain_fallback=True,
        )
    rename_candidates = {
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
    rename_map = {
        source: target
        for source, target in rename_candidates.items()
        if source in work.columns and source != target and target not in work.columns
    }
    return work.rename(columns=rename_map), trigger_info


def _resolve_lut_match_settings(config: dict) -> tuple[str, int | None, float]:
    step3_config = config.get("step3", {})
    if not isinstance(step3_config, dict):
        step3_config = {}

    raw_mode = step3_config.get("lut_match_mode")
    if raw_mode in (None, "", "null", "None"):
        match_mode = "nearest"
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
    def _build_linear_fit(
        x_values: pd.Series,
        y_values: pd.Series,
    ) -> dict[str, float | np.ndarray] | None:
        fit_frame = pd.DataFrame(
            {
                "x": pd.to_numeric(x_values, errors="coerce"),
                "y": pd.to_numeric(y_values, errors="coerce"),
            }
        ).dropna()
        if len(fit_frame) < 2 or fit_frame["x"].nunique() < 2:
            return None

        x_data = fit_frame["x"].to_numpy(dtype=float)
        y_data = fit_frame["y"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_data, y_data, deg=1)
        fit_values = slope * x_data + intercept
        ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
        ss_res = float(np.sum((y_data - fit_values) ** 2))
        r_squared = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
        return {
            "x_data": x_data,
            "y_data": y_data,
            "slope": float(slope),
            "intercept": float(intercept),
            "fit_values": fit_values,
            "r_squared": r_squared,
        }

    def _plot_linear_fit(
        axis: plt.Axes,
        x_values: pd.Series,
        y_values: pd.Series,
        *,
        fit_color: str,
    ) -> dict[str, float | np.ndarray] | None:
        fit_result = _build_linear_fit(x_values, y_values)
        if fit_result is None:
            return None

        x_data = np.asarray(fit_result["x_data"], dtype=float)
        slope = float(fit_result["slope"])
        intercept = float(fit_result["intercept"])
        r_squared = float(fit_result["r_squared"])
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
        return fit_result

    ordered, _, _ = _resolve_time_axis(dataframe)
    sequence = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
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
    corrected_fit = _plot_linear_fit(
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

    if corrected_fit is None:
        axes[2].text(
            0.5,
            0.5,
            "Insufficient data for fit-error histogram",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    else:
        corrected_rate = np.asarray(corrected_fit["y_data"], dtype=float)
        corrected_fit_values = np.asarray(corrected_fit["fit_values"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_fit_error = (corrected_fit_values - corrected_rate) / corrected_rate
        relative_fit_error = relative_fit_error[np.isfinite(relative_fit_error)]
        if relative_fit_error.size == 0:
            axes[2].text(
                0.5,
                0.5,
                "No finite relative fit errors",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
        else:
            axes[2].hist(
                relative_fit_error * 100.0,
                bins="auto",
                histtype="step",
                linewidth=2.0,
                color="#8B1E3F",
            )
            axes[2].axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[2].set_title("Corrected-rate linear-fit relative error")
    axes[2].set_xlabel("(fit - corrected) / corrected [%]")
    axes[2].set_ylabel("Count")
    axes[2].grid(alpha=0.25)

    cbar = fig.colorbar(scatter_left, ax=axes[:2], shrink=0.95)
    cbar.set_label("Time-series order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_flux_reference_curve(flux_cells_path: Path) -> pd.DataFrame:
    return load_reference_curve_table(flux_cells_path)


def _map_reference_rate_by_flux(
    flux_values: pd.Series,
    reference_curve: pd.DataFrame,
    *,
    row_z_frame: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    return map_reference_rate_by_flux(
        flux_values,
        row_z_frame=row_z_frame,
        reference_table=reference_curve,
    )


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

    synthetic_dataframe, synthetic_input_path, input_mode = _resolve_step3_input_dataframe(config)
    lut_path = cfg_path(config, "paths", "step2_lut_ascii")
    flux_cells_path = cfg_path(config, "paths", "step2_flux_cells_csv")
    lut_meta_path = cfg_path(config, "paths", "step2_meta_json")
    output_path = cfg_path(config, "paths", "step3_output_csv")
    meta_path = cfg_path(config, "paths", "step3_meta_json")

    work, trigger_info = _prepare_synthetic_dataframe(synthetic_dataframe.copy(), config)
    rate_column_name = format_selected_rate_name(
        stage_prefix=trigger_info.get("stage_prefix"),
        rate_family_column=str(trigger_info["rate_family_column"]),
        offender_threshold=trigger_info.get("used_offender_threshold"),
        metadata_source=str(trigger_info.get("metadata_source", "trigger_type")),
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

    lut_has_z = all(column in lut_dataframe.columns for column in CANONICAL_Z_COLUMNS)
    if lut_has_z:
        missing_z_columns = [column for column in CANONICAL_Z_COLUMNS if column not in work.columns]
        if missing_z_columns:
            raise ValueError(
                "The combined LUT requires z-position columns in the synthetic dataframe, but these are missing: "
                + ", ".join(missing_z_columns)
            )

    lut_lookup = lut_dataframe.rename(columns={column: f"lut_{column}" for column in CANONICAL_EFF_COLUMNS})
    left_on = list(query_columns)
    right_on = [f"lut_{column}" for column in CANONICAL_EFF_COLUMNS]
    if lut_has_z:
        left_on = [*CANONICAL_Z_COLUMNS, *left_on]
        right_on = [*CANONICAL_Z_COLUMNS, *right_on]
    merged = work.merge(
        lut_lookup,
        how="left",
        left_on=left_on,
        right_on=right_on,
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
        group_columns=CANONICAL_Z_COLUMNS if lut_has_z else None,
    )

    merged["corrected_rate_to_perfect_hz"] = merged["rate_hz"] * merged["lut_scale_factor"]
    reference_curve = _load_flux_reference_curve(flux_cells_path)
    (
        merged["reference_rate_almost_perfect_hz"],
        merged["reference_rate_assignment_method"],
    ) = _map_reference_rate_by_flux(
        merged["sim_flux_cm2_min"],
        reference_curve,
        row_z_frame=(merged[CANONICAL_Z_COLUMNS] if reference_table_has_z(reference_curve) else None),
    )
    merged["corrected_minus_reference_hz"] = (
        pd.to_numeric(merged["corrected_rate_to_perfect_hz"], errors="coerce")
        - pd.to_numeric(merged["reference_rate_almost_perfect_hz"], errors="coerce")
    )
    merged["selected_z_vector_match"] = True
    if lut_has_z:
        merged["selected_z_vector_match"] = False
        for z_vector in unique_z_vectors(lut_dataframe, z_columns=CANONICAL_Z_COLUMNS):
            merged.loc[z_mask_for_vector(merged, z_vector, z_columns=CANONICAL_Z_COLUMNS), "selected_z_vector_match"] = True

    output_dataframe = synthetic_dataframe.copy()
    for column in [
        "two_plane_rate_hz",
        "three_plane_rate_hz",
        "four_plane_rate_hz",
        "four_plane_robust_hz",
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
        "input_source_mode": input_mode,
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
        "trigger_rate_selection": trigger_info,
        "trigger_type_selection": trigger_info,
        "almost_perfect_flux_reference": {
            "flux_bins": int(len(reference_curve)),
            "z_configuration_count": (
                int(len(unique_z_vectors(reference_curve, z_columns=CANONICAL_Z_COLUMNS)))
                if reference_table_has_z(reference_curve)
                else 1
            ),
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
