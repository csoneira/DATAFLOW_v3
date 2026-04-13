#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from common import (
    CANONICAL_EFF_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    cfg_path,
    ensure_output_dirs,
    get_rate_column_name,
    load_config,
    write_json,
)

log = logging.getLogger("another_method.step4")


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_4 - %(message)s", level=logging.INFO, force=True)


def _build_plane_slice(dataframe: pd.DataFrame, plane_index: int) -> pd.DataFrame:
    varied_column = CANONICAL_EFF_COLUMNS[plane_index]
    fixed_columns = [column for idx, column in enumerate(CANONICAL_EFF_COLUMNS) if idx != plane_index]

    work = dataframe.copy()
    work["plane_index"] = plane_index + 1
    work["varied_eff"] = work[varied_column].astype(float)
    work["fixed_distance_to_one"] = np.sqrt(
        np.sum((1.0 - work[fixed_columns].astype(float).to_numpy()) ** 2, axis=1)
    )
    work["fixed_eff_mean"] = work[fixed_columns].astype(float).mean(axis=1)
    work = work.sort_values(
        [
            "varied_eff",
            "fixed_distance_to_one",
            "distance_to_perfect",
            "relative_factor_iqr",
            "support_rows",
        ],
        ascending=[True, True, True, True, False],
    )
    work = work.groupby("varied_eff", as_index=False).head(1).copy()
    work["relative_rate_to_reference"] = 1.0 / work["scale_factor"]
    work["relative_rate_q25"] = 1.0 / work["scale_factor_q75"]
    work["relative_rate_q75"] = 1.0 / work["scale_factor_q25"]
    work["slice_label"] = f"vary eff_{plane_index + 1}"
    work = work.sort_values("varied_eff").reset_index(drop=True)
    return work


def _build_pair_slice(dataframe: pd.DataFrame, plane_indices: tuple[int, int]) -> pd.DataFrame:
    varied_columns = [CANONICAL_EFF_COLUMNS[plane_indices[0]], CANONICAL_EFF_COLUMNS[plane_indices[1]]]
    fixed_columns = [
        column for idx, column in enumerate(CANONICAL_EFF_COLUMNS) if idx not in set(plane_indices)
    ]

    work = dataframe.copy()
    work["pair_plane_a"] = plane_indices[0] + 1
    work["pair_plane_b"] = plane_indices[1] + 1
    work["varied_eff_a"] = work[varied_columns[0]].astype(float)
    work["varied_eff_b"] = work[varied_columns[1]].astype(float)
    work["fixed_distance_to_one"] = np.sqrt(
        np.sum((1.0 - work[fixed_columns].astype(float).to_numpy()) ** 2, axis=1)
    )
    work["fixed_eff_mean"] = work[fixed_columns].astype(float).mean(axis=1)
    work = work.sort_values(
        [
            "varied_eff_a",
            "varied_eff_b",
            "fixed_distance_to_one",
            "distance_to_perfect",
            "relative_factor_iqr",
            "support_rows",
        ],
        ascending=[True, True, True, True, True, False],
    )
    work = work.groupby(["varied_eff_a", "varied_eff_b"], as_index=False).head(1).copy()
    work["relative_rate_to_reference"] = 1.0 / work["scale_factor"]
    work["relative_rate_q25"] = 1.0 / work["scale_factor_q75"]
    work["relative_rate_q75"] = 1.0 / work["scale_factor_q25"]
    work["pair_label"] = f"vary eff_{plane_indices[0] + 1} and eff_{plane_indices[1] + 1}"
    work = work.sort_values(["varied_eff_a", "varied_eff_b"]).reset_index(drop=True)
    return work


def _contour_panel(
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    value_column: str,
    *,
    title: str,
    cmap: str,
    colorbar_label: str,
) -> None:
    triangulation = mtri.Triangulation(
        dataframe["varied_eff_a"].to_numpy(dtype=float),
        dataframe["varied_eff_b"].to_numpy(dtype=float),
    )
    contour = ax.tricontourf(
        triangulation,
        dataframe[value_column].to_numpy(dtype=float),
        levels=16,
        cmap=cmap,
    )
    ax.tricontour(
        triangulation,
        dataframe[value_column].to_numpy(dtype=float),
        levels=8,
        colors="white",
        linewidths=0.45,
        alpha=0.65,
    )
    ax.scatter(
        dataframe["varied_eff_a"],
        dataframe["varied_eff_b"],
        s=18 + 6 * np.sqrt(dataframe["support_rows"].clip(lower=1)),
        c="black",
        alpha=0.35,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel(f"eff_{int(dataframe['pair_plane_a'].iloc[0])}")
    ax.set_ylabel(f"eff_{int(dataframe['pair_plane_b'].iloc[0])}")
    ax.set_xlim(0.58, 1.02)
    ax.set_ylim(0.58, 1.02)
    ax.grid(alpha=0.15)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label)


def _plot_scale_factor_slices(
    slices: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for curve_index, plane_index in enumerate(sorted(slices["plane_index"].unique())):
        subset = slices[slices["plane_index"] == plane_index].copy()
        color = cmap((curve_index + 1) % 10)
        yerr = np.vstack(
            [
                subset["scale_factor"] - subset["scale_factor_q25"],
                subset["scale_factor_q75"] - subset["scale_factor"],
            ]
        )
        ax.errorbar(
            subset["varied_eff"],
            subset["scale_factor"],
            yerr=yerr,
            fmt="o-",
            linewidth=1.6,
            capsize=3,
            color=color,
            label=f"vary eff_{plane_index}",
        )

    ax.set_xlabel("Varied empirical efficiency")
    ax.set_ylabel("Scale factor")
    ax.set_title(
        "LUT scale factor when varying one efficiency at a time\n"
        f"rate column: {rate_column_name}"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_relative_rate_slices(
    slices: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for curve_index, plane_index in enumerate(sorted(slices["plane_index"].unique())):
        subset = slices[slices["plane_index"] == plane_index].copy()
        color = cmap((curve_index + 1) % 10)
        yerr = np.vstack(
            [
                subset["relative_rate_to_reference"] - subset["relative_rate_q25"],
                subset["relative_rate_q75"] - subset["relative_rate_to_reference"],
            ]
        )
        ax.errorbar(
            subset["varied_eff"],
            subset["relative_rate_to_reference"],
            yerr=yerr,
            fmt="o-",
            linewidth=1.6,
            capsize=3,
            color=color,
            label=f"R(slice) / R(reference), vary eff_{plane_index}",
        )

    ax.set_xlabel("Varied empirical efficiency")
    ax.set_ylabel("Relative rate to reference band")
    ax.set_title(
        "Reference-normalized rate when varying one efficiency at a time\n"
        f"rate column: {rate_column_name}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pair_surface(
    pair_slice: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _contour_panel(
        axes[0],
        pair_slice,
        "scale_factor",
        title="Scale factor surface",
        cmap="viridis",
        colorbar_label="Scale factor",
    )
    _contour_panel(
        axes[1],
        pair_slice,
        "relative_rate_to_reference",
        title="Relative rate surface",
        cmap="magma_r",
        colorbar_label="R(slice) / R(reference)",
    )
    fig.suptitle(
        "Two-plane LUT slice: "
        f"vary eff_{int(pair_slice['pair_plane_a'].iloc[0])} "
        f"and eff_{int(pair_slice['pair_plane_b'].iloc[0])}\n"
        f"rate column: {rate_column_name}",
        y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pair_quality(
    pair_slice: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _contour_panel(
        axes[0],
        pair_slice,
        "fixed_distance_to_one",
        title="Distance of fixed planes to 1",
        cmap="cividis_r",
        colorbar_label="Distance to 1 of fixed planes",
    )
    _contour_panel(
        axes[1],
        pair_slice,
        "support_rows",
        title="Support rows used by selected slice",
        cmap="plasma",
        colorbar_label="Support rows",
    )
    fig.suptitle(
        "Two-plane slice quality: "
        f"vary eff_{int(pair_slice['pair_plane_a'].iloc[0])} "
        f"and eff_{int(pair_slice['pair_plane_b'].iloc[0])}\n"
        f"LUT built from rate column: {rate_column_name}",
        y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)
    rate_column_name = get_rate_column_name(config)

    lut_diag_path = cfg_path(config, "paths", "step2_lut_diagnostics_csv")
    output_csv_path = cfg_path(config, "paths", "step4_axis_slice_csv")
    pair_csv_path = cfg_path(config, "paths", "step4_pair_slice_csv")
    meta_path = cfg_path(config, "paths", "step4_meta_json")

    pair_planes_raw = config.get("step4", {}).get("pair_planes", [2, 3])
    if not isinstance(pair_planes_raw, list) or len(pair_planes_raw) != 2:
        raise ValueError("step4.pair_planes must be a list with exactly two plane numbers.")
    pair_planes = tuple(int(value) - 1 for value in pair_planes_raw)
    if pair_planes[0] == pair_planes[1] or min(pair_planes) < 0 or max(pair_planes) >= len(CANONICAL_EFF_COLUMNS):
        raise ValueError("step4.pair_planes must contain two distinct plane numbers in the range [1, 4].")

    lut = pd.read_csv(lut_diag_path)
    slices = pd.concat(
        [_build_plane_slice(lut, plane_index) for plane_index in range(len(CANONICAL_EFF_COLUMNS))],
        ignore_index=True,
    )
    pair_slice = _build_pair_slice(lut, pair_planes)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    slices.to_csv(output_csv_path, index=False)
    pair_slice.to_csv(pair_csv_path, index=False)

    _plot_scale_factor_slices(
        slices,
        PLOTS_DIR / "step4_axis_slice_scale_factor.png",
        rate_column_name=rate_column_name,
    )
    _plot_relative_rate_slices(
        slices,
        PLOTS_DIR / "step4_axis_slice_relative_rate.png",
        rate_column_name=rate_column_name,
    )
    _plot_pair_surface(
        pair_slice,
        PLOTS_DIR / "step4_pair_slice_surface.png",
        rate_column_name=rate_column_name,
    )
    _plot_pair_quality(
        pair_slice,
        PLOTS_DIR / "step4_pair_slice_quality.png",
        rate_column_name=rate_column_name,
    )

    slice_summary: dict[str, dict[str, float | int]] = {}
    for plane_index in sorted(slices["plane_index"].unique()):
        subset = slices[slices["plane_index"] == plane_index]
        slice_summary[f"eff_{plane_index}"] = {
            "points": int(len(subset)),
            "min_varied_eff": float(subset["varied_eff"].min()),
            "max_varied_eff": float(subset["varied_eff"].max()),
            "median_fixed_distance_to_one": float(subset["fixed_distance_to_one"].median()),
            "max_fixed_distance_to_one": float(subset["fixed_distance_to_one"].max()),
        }

    metadata = {
        "source_file": str(lut_diag_path),
        "row_count": int(len(slices)),
        "pair_slice_row_count": int(len(pair_slice)),
        "planes": [1, 2, 3, 4],
        "selection_rule": (
            "For each varied efficiency and plane, choose the LUT row whose other three "
            "efficiencies minimize distance to 1, then break ties with smaller distance_to_perfect, "
            "smaller relative_factor_iqr, and larger support_rows."
        ),
        "pair_selection_rule": (
            "For each efficiency pair in the selected two-plane slice, choose the LUT row whose "
            "other two efficiencies minimize distance to 1, then break ties with smaller "
            "distance_to_perfect, smaller relative_factor_iqr, and larger support_rows."
        ),
        "pair_planes": [int(pair_planes[0] + 1), int(pair_planes[1] + 1)],
        "pair_slice_summary": {
            "unique_eff_a": int(pair_slice["varied_eff_a"].nunique()),
            "unique_eff_b": int(pair_slice["varied_eff_b"].nunique()),
            "median_fixed_distance_to_one": float(pair_slice["fixed_distance_to_one"].median()),
            "max_fixed_distance_to_one": float(pair_slice["fixed_distance_to_one"].max()),
            "median_support_rows": float(pair_slice["support_rows"].median()),
        },
        "slice_summary": slice_summary,
    }
    write_json(meta_path, metadata)

    log.info("Wrote Step 4 LUT study table to %s", output_csv_path)
    return output_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Study single-plane slices through the LUT.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
