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
    assign_efficiency_bins,
    assign_flux_bins,
    cfg_path,
    choose_reference_row,
    ensure_output_dirs,
    get_rate_column_name,
    load_config,
    q25,
    q75,
    write_ascii_lut,
    write_json,
)

log = logging.getLogger("another_method.step2")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_2 - %(message)s", level=logging.INFO, force=True)


def _same_eff_bin(dataframe: pd.DataFrame, values: tuple[float, float, float, float]) -> pd.Series:
    mask = np.ones(len(dataframe), dtype=bool)
    for column, value in zip(CANONICAL_EFF_COLUMNS, values):
        mask &= np.isclose(dataframe[column].astype(float), float(value))
    return pd.Series(mask, index=dataframe.index)


def _aggregate_flux_cells(dataframe: pd.DataFrame, flux_bin_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    work = dataframe.copy()
    work["flux_bin_index"], flux_edges = assign_flux_bins(work["sim_flux_cm2_min"], flux_bin_count)
    work = work.dropna(subset=["flux_bin_index"]).copy()
    work["flux_bin_index"] = work["flux_bin_index"].astype(int)

    group_columns = CANONICAL_EFF_COLUMNS + ["flux_bin_index"]
    aggregated = (
        work.groupby(group_columns, dropna=False)
        .agg(
            n_rows=("rate_hz", "size"),
            rate_median=("rate_hz", "median"),
            rate_q25=("rate_hz", q25),
            rate_q75=("rate_hz", q75),
            flux_median=("sim_flux_cm2_min", "median"),
            flux_q25=("sim_flux_cm2_min", q25),
            flux_q75=("sim_flux_cm2_min", q75),
            eff_empirical_1=("eff_empirical_1", "median"),
            eff_empirical_2=("eff_empirical_2", "median"),
            eff_empirical_3=("eff_empirical_3", "median"),
            eff_empirical_4=("eff_empirical_4", "median"),
        )
        .reset_index()
    )

    aggregated["flux_bin_lo"] = aggregated["flux_bin_index"].map(lambda idx: float(flux_edges[int(idx)]))
    aggregated["flux_bin_hi"] = aggregated["flux_bin_index"].map(lambda idx: float(flux_edges[int(idx) + 1]))
    aggregated["flux_bin_center"] = 0.5 * (aggregated["flux_bin_lo"] + aggregated["flux_bin_hi"])
    return aggregated.sort_values(CANONICAL_EFF_COLUMNS + ["flux_bin_center"]).reset_index(drop=True), flux_edges


def _distance_to_perfect(dataframe: pd.DataFrame) -> pd.Series:
    values = np.sqrt(np.sum((1.0 - dataframe[CANONICAL_EFF_COLUMNS].astype(float).to_numpy()) ** 2, axis=1))
    return pd.Series(values, index=dataframe.index)


def _add_efficiency_summary_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    out["eff_mean"] = out[CANONICAL_EFF_COLUMNS].mean(axis=1)
    out["eff_span"] = out[CANONICAL_EFF_COLUMNS].max(axis=1) - out[CANONICAL_EFF_COLUMNS].min(axis=1)
    out["distance_to_perfect"] = _distance_to_perfect(out)
    return out


def _exact_diagonal_mask(dataframe: pd.DataFrame) -> pd.Series:
    mask = np.isclose(dataframe[CANONICAL_EFF_COLUMNS[0]], dataframe[CANONICAL_EFF_COLUMNS[1]])
    mask &= np.isclose(dataframe[CANONICAL_EFF_COLUMNS[1]], dataframe[CANONICAL_EFF_COLUMNS[2]])
    mask &= np.isclose(dataframe[CANONICAL_EFF_COLUMNS[2]], dataframe[CANONICAL_EFF_COLUMNS[3]])
    return pd.Series(mask, index=dataframe.index)


def _quantize_efficiency(values: pd.Series, width: float) -> pd.Series:
    quantized = np.round(values.astype(float).to_numpy() / float(width)) * float(width)
    return pd.Series(np.clip(np.round(quantized, 6), 0.0, 1.0), index=values.index)


def _summarize_diagonal_band(
    aggregated_cells: pd.DataFrame,
    *,
    summary_bin_width: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagonal_cells = aggregated_cells.copy()
    diagonal_cells["diag_eff_bin"] = _quantize_efficiency(diagonal_cells["eff_mean"], summary_bin_width)

    diagonal_curves = (
        diagonal_cells.groupby(["diag_eff_bin", "flux_bin_index"], dropna=False)
        .agg(
            diag_eff=("eff_mean", "median"),
            n_cells=("rate_median", "size"),
            flux_bin_center=("flux_bin_center", "median"),
            rate_median=("rate_median", "median"),
            rate_q25=("rate_median", q25),
            rate_q75=("rate_median", q75),
            scale_factor=("cell_scale_factor", "median"),
            scale_factor_q25=("cell_scale_factor", q25),
            scale_factor_q75=("cell_scale_factor", q75),
        )
        .reset_index()
        .sort_values(["diag_eff_bin", "flux_bin_center"])
        .reset_index(drop=True)
    )

    diagonal_summary = (
        diagonal_cells.groupby("diag_eff_bin", dropna=False)
        .agg(
            diag_eff=("eff_mean", "median"),
            diag_support_cells=("eff_mean", "size"),
            n_flux_bins=("flux_bin_index", "nunique"),
            scale_factor=("cell_scale_factor", "median"),
            scale_factor_q25=("cell_scale_factor", q25),
            scale_factor_q75=("cell_scale_factor", q75),
            median_eff_span=("eff_span", "median"),
        )
        .reset_index()
        .sort_values("diag_eff_bin")
        .reset_index(drop=True)
    )
    return diagonal_curves, diagonal_summary


def _build_supported_diagonal_summary(
    aggregated_cells: pd.DataFrame,
    diagonal_tolerance: float,
    *,
    summary_bin_width: float,
    min_flux_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    exact_cells = aggregated_cells.loc[_exact_diagonal_mask(aggregated_cells)].copy()
    if not exact_cells.empty:
        exact_curves, exact_summary = _summarize_diagonal_band(
            exact_cells,
            summary_bin_width=summary_bin_width,
        )
        exact_summary = exact_summary[exact_summary["n_flux_bins"] >= int(min_flux_bins)].copy()
        if not exact_summary.empty:
            exact_curves = exact_curves[exact_curves["diag_eff_bin"].isin(exact_summary["diag_eff_bin"])].copy()
            return exact_curves, exact_summary, "exact_diagonal_summary"

    near_diagonal_cells = aggregated_cells.loc[
        aggregated_cells["eff_span"] <= float(diagonal_tolerance)
    ].copy()
    near_curves, near_summary = _summarize_diagonal_band(
        near_diagonal_cells,
        summary_bin_width=summary_bin_width,
    )
    near_summary = near_summary[near_summary["n_flux_bins"] >= int(min_flux_bins)].copy()
    near_curves = near_curves[near_curves["diag_eff_bin"].isin(near_summary["diag_eff_bin"])].copy()
    return near_curves, near_summary, "near_diagonal_band_summary"


def _build_reference_curve(
    aggregated_cells: pd.DataFrame,
    *,
    top_k_closest_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_cells = (
        aggregated_cells.sort_values(
            ["flux_bin_index", "distance_to_perfect", "eff_span", "eff_mean"],
            ascending=[True, True, True, False],
        )
        .groupby("flux_bin_index", dropna=False)
        .head(int(top_k_closest_bins))
        .copy()
    )
    reference_curve = (
        reference_cells.groupby("flux_bin_index", dropna=False)
        .agg(
            flux_bin_center=("flux_bin_center", "median"),
            reference_rate_median=("rate_median", "median"),
            reference_rate_q25=("rate_median", q25),
            reference_rate_q75=("rate_median", q75),
            reference_cell_count=("rate_median", "size"),
            reference_eff_mean=("eff_mean", "median"),
            reference_distance_to_perfect=("distance_to_perfect", "median"),
        )
        .reset_index()
        .sort_values("flux_bin_index")
        .reset_index(drop=True)
    )
    return reference_cells, reference_curve


def _select_curve_bins(diagonal_summary: pd.DataFrame, max_curves: int) -> list[float]:
    if diagonal_summary.empty:
        return []
    bins = diagonal_summary.sort_values("diag_eff_bin")["diag_eff_bin"].tolist()
    if len(bins) <= int(max_curves):
        return [float(value) for value in bins]
    positions = np.linspace(0, len(bins) - 1, num=int(max_curves), dtype=int)
    return [float(bins[index]) for index in np.unique(positions)]


def _plot_rate_vs_flux(
    aggregated_cells: pd.DataFrame,
    reference_curve: pd.DataFrame,
    reference_cells: pd.DataFrame,
    diagonal_curves: pd.DataFrame,
    selected_curve_bins: list[float],
    output_path: Path,
    *,
    reference_label: str,
    diagonal_mode: str,
    rate_column_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        aggregated_cells["flux_bin_center"],
        aggregated_cells["rate_median"],
        c=aggregated_cells["eff_mean"],
        cmap="viridis",
        s=24,
        alpha=0.30,
        edgecolors="none",
    )
    ax.plot(
        reference_curve["flux_bin_center"],
        reference_curve["reference_rate_median"],
        color="black",
        linewidth=2.2,
        marker="o",
        label=reference_label,
    )
    ax.fill_between(
        reference_curve["flux_bin_center"],
        reference_curve["reference_rate_q25"],
        reference_curve["reference_rate_q75"],
        color="black",
        alpha=0.15,
    )
    ax.scatter(
        reference_cells["flux_bin_center"],
        reference_cells["rate_median"],
        marker="x",
        color="black",
        s=42,
        alpha=0.55,
        label="reference-band cells",
    )

    cmap = plt.get_cmap("viridis")
    for idx, diag_eff_bin in enumerate(selected_curve_bins):
        subset = diagonal_curves.loc[
            np.isclose(diagonal_curves["diag_eff_bin"].astype(float), float(diag_eff_bin))
        ].sort_values("flux_bin_center")
        if subset.empty:
            continue
        color = cmap(idx / max(len(selected_curve_bins) - 1, 1))
        ax.plot(
            subset["flux_bin_center"],
            subset["rate_median"],
            marker="o",
            linewidth=1.5,
            color=color,
            label=f"mean eff ~ {float(subset['diag_eff'].median()):.2f}",
        )
        ax.fill_between(
            subset["flux_bin_center"],
            subset["rate_q25"],
            subset["rate_q75"],
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("Simulated flux [cm^-2 min^-1]")
    ax.set_ylabel("Rate [Hz]")
    ax.set_title(
        f"Rate vs flux with diagonal-band summaries ({diagonal_mode})\n"
        f"rate column: {rate_column_name}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean efficiency of 4D bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scale_factor_vs_eff(
    diagonal_summary: pd.DataFrame,
    output_path: Path,
    *,
    diagonal_mode: str,
    rate_column_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    yerr = np.vstack(
        [
            diagonal_summary["scale_factor"] - diagonal_summary["scale_factor_q25"],
            diagonal_summary["scale_factor_q75"] - diagonal_summary["scale_factor"],
        ]
    )
    ax.errorbar(
        diagonal_summary["diag_eff"],
        diagonal_summary["scale_factor"],
        yerr=yerr,
        fmt="o-",
        color="#B22222",
        ecolor="#F4A582",
        capsize=3,
    )
    ax.set_xlabel("Mean efficiency of diagonal-band summary bin")
    ax.set_ylabel("Scale factor to closest perfect-efficiency reference")
    ax.set_title(
        f"Scale factor vs efficiency ({diagonal_mode})\n"
        f"rate column: {rate_column_name}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scale_factor_vs_flux(
    aggregated_cells: pd.DataFrame,
    diagonal_curves: pd.DataFrame,
    selected_curve_bins: list[float],
    output_path: Path,
    *,
    diagonal_mode: str,
    rate_column_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        aggregated_cells["flux_bin_center"],
        aggregated_cells["cell_scale_factor"],
        c=aggregated_cells["eff_mean"],
        cmap="plasma",
        s=24,
        alpha=0.30,
        edgecolors="none",
    )
    cmap = plt.get_cmap("plasma")
    for idx, diag_eff_bin in enumerate(selected_curve_bins):
        subset = diagonal_curves.loc[
            np.isclose(diagonal_curves["diag_eff_bin"].astype(float), float(diag_eff_bin))
        ].sort_values("flux_bin_center")
        if subset.empty:
            continue
        color = cmap(idx / max(len(selected_curve_bins) - 1, 1))
        ax.plot(
            subset["flux_bin_center"],
            subset["scale_factor"],
            marker="o",
            linewidth=1.5,
            color=color,
            label=f"mean eff ~ {float(subset['diag_eff'].median()):.2f}",
        )
        ax.fill_between(
            subset["flux_bin_center"],
            subset["scale_factor_q25"],
            subset["scale_factor_q75"],
            color=color,
            alpha=0.10,
        )
    ax.set_xlabel("Simulated flux [cm^-2 min^-1]")
    ax.set_ylabel("Per-flux-bin scale factor")
    ax.set_title(f"Flux dependence check ({diagonal_mode})\nrate column: {rate_column_name}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean efficiency of 4D bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step1_input_path = cfg_path(config, "paths", "step1_filtered_csv")
    step1_meta_path = cfg_path(config, "paths", "step1_meta_json")
    flux_cells_path = cfg_path(config, "paths", "step2_flux_cells_csv")
    lut_diag_path = cfg_path(config, "paths", "step2_lut_diagnostics_csv")
    lut_ascii_path = cfg_path(config, "paths", "step2_lut_ascii")
    meta_path = cfg_path(config, "paths", "step2_meta_json")

    efficiency_bin_width = float(config.get("step2", {}).get("efficiency_bin_width", 0.02))
    flux_bin_count = int(config.get("step2", {}).get("flux_bin_count", 10))
    diagonal_tolerance = float(config.get("step2", {}).get("diagonal_tolerance", efficiency_bin_width))
    diagnostic_efficiency_summary_bin_width = float(
        config.get("step2", {}).get("diagnostic_efficiency_summary_bin_width", 0.05)
    )
    diagnostic_min_flux_bins = int(config.get("step2", {}).get("diagnostic_min_flux_bins", 3))
    diagonal_plot_max_curves = int(config.get("step2", {}).get("diagonal_plot_max_curves", 6))
    min_flux_bins_per_entry = int(config.get("step2", {}).get("min_flux_bins_per_lut_entry", 1))
    reference_top_k_per_flux_bin = int(config.get("step2", {}).get("reference_top_k_per_flux_bin", 5))

    dataframe = pd.read_csv(step1_input_path)
    step1_meta = {}
    if step1_meta_path.exists():
        step1_meta = json.loads(step1_meta_path.read_text(encoding="utf-8"))
    rate_column_name = str(
        step1_meta.get("input_columns", {}).get("rate", get_rate_column_name(config))
    )

    binned = assign_efficiency_bins(dataframe, CANONICAL_EFF_COLUMNS, efficiency_bin_width, suffix="_bin")
    for column in CANONICAL_EFF_COLUMNS:
        binned[column] = binned[f"{column}_bin"]
    binned = binned.drop(columns=[f"{column}_bin" for column in CANONICAL_EFF_COLUMNS])

    aggregated_cells, flux_edges = _aggregate_flux_cells(binned, flux_bin_count)
    aggregated_cells = _add_efficiency_summary_columns(aggregated_cells)

    support = (
        aggregated_cells.groupby(CANONICAL_EFF_COLUMNS, dropna=False)
        .agg(
            n_flux_bins=("flux_bin_index", "nunique"),
            support_rows=("n_rows", "sum"),
        )
        .reset_index()
    )
    closest_single_row = choose_reference_row(support, CANONICAL_EFF_COLUMNS)
    closest_single_key = tuple(float(closest_single_row[column]) for column in CANONICAL_EFF_COLUMNS)

    reference_cells, reference_curve = _build_reference_curve(
        aggregated_cells,
        top_k_closest_bins=reference_top_k_per_flux_bin,
    )
    if reference_curve.empty:
        raise ValueError("Could not build the flux-dependent reference curve.")

    aggregated_cells = aggregated_cells.merge(
        reference_curve[["flux_bin_index", "reference_rate_median"]],
        on="flux_bin_index",
        how="left",
    )
    aggregated_cells["cell_scale_factor"] = aggregated_cells["reference_rate_median"] / aggregated_cells["rate_median"]
    aggregated_cells["cell_scale_factor"] = aggregated_cells["cell_scale_factor"].replace([np.inf, -np.inf], np.nan)
    aggregated_cells.to_csv(flux_cells_path, index=False)

    lut = (
        aggregated_cells.groupby(CANONICAL_EFF_COLUMNS, dropna=False)
        .agg(
            scale_factor=("cell_scale_factor", "median"),
            scale_factor_q25=("cell_scale_factor", q25),
            scale_factor_q75=("cell_scale_factor", q75),
            n_flux_bins=("flux_bin_index", "nunique"),
            support_rows=("n_rows", "sum"),
            eff_empirical_1=("eff_empirical_1", "median"),
            eff_empirical_2=("eff_empirical_2", "median"),
            eff_empirical_3=("eff_empirical_3", "median"),
            eff_empirical_4=("eff_empirical_4", "median"),
        )
        .reset_index()
    )
    negative_eff_mask = (
        (lut["eff_empirical_2"].astype(float) >= 0.0)
        & (lut["eff_empirical_3"].astype(float) >= 0.0)
    )
    if len(lut) != int(negative_eff_mask.sum()):
        log.info(
            "Filtering %d LUT rows with negative eff_empirical_2 or eff_empirical_3 values.",
            len(lut) - int(negative_eff_mask.sum()),
        )
    lut = lut.loc[negative_eff_mask].copy()
    lut = lut[lut["n_flux_bins"] >= min_flux_bins_per_entry].copy()
    lut["relative_factor_iqr"] = (
        (lut["scale_factor_q75"] - lut["scale_factor_q25"]) / lut["scale_factor"].replace(0.0, np.nan)
    )
    lut = _add_efficiency_summary_columns(lut)
    lut["is_closest_single_bin"] = _same_eff_bin(lut, closest_single_key).astype(int)
    lut = lut.sort_values(CANONICAL_EFF_COLUMNS).reset_index(drop=True)
    lut.to_csv(lut_diag_path, index=False)

    lut_ascii = lut[[f"eff_empirical_{idx}" for idx in range(1, 5)] + ["scale_factor"]].copy()
    lut_ascii.columns = CANONICAL_EFF_COLUMNS + ["scale_factor"]
    exact_perfect_reference_available = bool(np.isclose(float(closest_single_row["distance_to_perfect"]), 0.0))
    closest_single_comment = (
        "closest_single_eff_bin: "
        + ", ".join(f"{float(closest_single_row[column]):.6f}" for column in CANONICAL_EFF_COLUMNS)
    )
    if step1_meta.get("selected_z_positions"):
        z_vector = tuple(float(value) for value in step1_meta["selected_z_positions"])
    else:
        z_vector = tuple(float(dataframe.iloc[0][column]) for column in ["z_pos_1", "z_pos_2", "z_pos_3", "z_pos_4"])
    write_ascii_lut(
        lut_ascii_path,
        z_vector,
        lut_ascii,
        trigger=step1_meta.get("selected_trigger"),
        rate_column_name=rate_column_name,
    )

    diagonal_curves, diagonal_summary, diagonal_mode = _build_supported_diagonal_summary(
        aggregated_cells,
        diagonal_tolerance,
        summary_bin_width=diagnostic_efficiency_summary_bin_width,
        min_flux_bins=diagnostic_min_flux_bins,
    )
    selected_curve_bins = _select_curve_bins(diagonal_summary, diagonal_plot_max_curves)
    if selected_curve_bins:
        _plot_rate_vs_flux(
            aggregated_cells,
            reference_curve,
            reference_cells,
            diagonal_curves,
            selected_curve_bins,
            PLOTS_DIR / "step2_rate_vs_flux.png",
            reference_label=f"reference median of top-{reference_top_k_per_flux_bin} closest bins / flux",
            diagonal_mode=diagonal_mode,
            rate_column_name=rate_column_name,
        )
        _plot_scale_factor_vs_eff(
            diagonal_summary,
            PLOTS_DIR / "step2_scale_factor_vs_diagonal_eff.png",
            diagonal_mode=diagonal_mode,
            rate_column_name=rate_column_name,
        )
        _plot_scale_factor_vs_flux(
            aggregated_cells,
            diagonal_curves,
            selected_curve_bins,
            PLOTS_DIR / "step2_scale_factor_vs_flux.png",
            diagonal_mode=diagonal_mode,
            rate_column_name=rate_column_name,
        )
    else:
        log.warning("No diagonal or near-diagonal bins were available for the requested diagnostic plots.")

    input_columns_meta = step1_meta.get("input_columns", {})
    rate_input_column = input_columns_meta.get("rate", input_columns_meta.get("global_rate"))

    metadata = {
        "source_file": str(step1_input_path),
        "selected_z_positions": step1_meta.get("selected_z_positions"),
        "selected_trigger": step1_meta.get("selected_trigger"),
        "trigger_values_in_filtered_data": step1_meta.get("trigger_values_in_filtered_data"),
        "rate_input_column": rate_input_column,
        "efficiency_bin_width": efficiency_bin_width,
        "flux_bin_count": flux_bin_count,
        "flux_bin_edges": [float(value) for value in flux_edges.tolist()],
        "closest_single_efficiency_bin": {
            column: float(closest_single_row[column]) for column in CANONICAL_EFF_COLUMNS
        },
        "closest_single_distance_to_perfect": float(closest_single_row["distance_to_perfect"]),
        "closest_single_is_exact_perfect_bin": exact_perfect_reference_available,
        "closest_single_efficiency_bin_comment": closest_single_comment,
        "reference_method": "median_of_top_closest_bins_per_flux_bin",
        "reference_top_k_per_flux_bin": reference_top_k_per_flux_bin,
        "reference_flux_bins": int(len(reference_curve)),
        "reference_median_distance_to_perfect": float(reference_curve["reference_distance_to_perfect"].median()),
        "lut_rows": int(len(lut)),
        "diagonal_mode": diagonal_mode,
        "diagnostic_diagonal_tolerance": diagonal_tolerance,
        "diagnostic_efficiency_summary_bin_width": diagnostic_efficiency_summary_bin_width,
        "diagnostic_min_flux_bins": diagnostic_min_flux_bins,
        "diagnostic_diagonal_summary_rows": int(len(diagonal_summary)),
        "diagnostic_curve_count": int(len(selected_curve_bins)),
        "exact_4d_bins_with_multiple_flux_bins": int((support["n_flux_bins"] > 1).sum()),
        "min_flux_bins_per_lut_entry": min_flux_bins_per_entry,
    }
    write_json(meta_path, metadata)

    log.info("Wrote LUT with %d rows to %s", len(lut), lut_ascii_path)
    log.info(
        "Closest single bin: %s (distance to perfect %.5f)",
        closest_single_key,
        float(closest_single_row["distance_to_perfect"]),
    )
    return lut_ascii_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the scale-factor LUT from filtered collected data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
