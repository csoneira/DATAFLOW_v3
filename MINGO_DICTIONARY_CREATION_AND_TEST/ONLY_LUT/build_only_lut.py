#!/usr/bin/env python3
"""Build a compact four-plane simulation-efficiency LUT for poster use."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATION_CSV = REPO_ROOT / "MINGO_DIGITAL_TWIN/SIMULATED_DATA/step_final_simulation_params.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "OUTPUTS"
EFF_COLUMNS = [f"eff_{plane}" for plane in range(1, 5)]
Z_COLUMNS = [f"z_plane_{plane}" for plane in range(1, 5)]
RATE_COLUMN = "trigger_rate_hz"


def parse_efficiencies(value: object) -> list[float]:
    try:
        parsed = value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value))
        if len(parsed) != 4:
            raise ValueError
        return [float(item) for item in parsed]
    except (SyntaxError, ValueError, TypeError):
        return [np.nan] * 4


def load_selected_training(
    simulation_csv: Path,
    z_positions: tuple[float, float, float, float],
    min_selected_rows: int,
) -> pd.DataFrame:
    simulation = pd.read_csv(
        simulation_csv,
        usecols=["file_name", "param_hash", "flux_cm2_min", "selected_rows", "efficiencies", RATE_COLUMN, *Z_COLUMNS],
        low_memory=False,
    )
    efficiency_frame = simulation["efficiencies"].map(parse_efficiencies).apply(pd.Series)
    efficiency_frame.columns = EFF_COLUMNS
    simulation[EFF_COLUMNS] = efficiency_frame.apply(pd.to_numeric, errors="coerce")
    for column in [*Z_COLUMNS, "flux_cm2_min", "selected_rows", RATE_COLUMN]:
        simulation[column] = pd.to_numeric(simulation[column], errors="coerce")

    valid = np.isfinite(
        simulation[[*EFF_COLUMNS, *Z_COLUMNS, "flux_cm2_min", "selected_rows", RATE_COLUMN]]
    ).all(axis=1)
    valid &= simulation[RATE_COLUMN] > 0
    valid &= simulation["selected_rows"] >= min_selected_rows
    for column, expected in zip(Z_COLUMNS, z_positions):
        valid &= np.isclose(simulation[column], expected)
    selected = simulation.loc[valid].copy()
    selected["distance_to_perfect"] = np.sqrt(
        np.square(1.0 - selected[EFF_COLUMNS].to_numpy(dtype=float)).sum(axis=1)
    )
    return selected.sort_values([*EFF_COLUMNS, "flux_cm2_min"], kind="mergesort").reset_index(drop=True)


def build_lut(
    training: pd.DataFrame,
    *,
    efficiency_bin_width: float,
    flux_bin_count: int,
    reference_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = training.copy()
    for column in EFF_COLUMNS:
        work[f"{column}_bin"] = np.round(work[column] / efficiency_bin_width) * efficiency_bin_width
    work["flux_bin"], flux_edges = pd.qcut(
        work["flux_cm2_min"],
        q=min(flux_bin_count, work["flux_cm2_min"].nunique()),
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    work = work.dropna(subset=["flux_bin"]).copy()
    work["flux_bin"] = work["flux_bin"].astype(int)

    references = (
        work.sort_values(["flux_bin", "distance_to_perfect", RATE_COLUMN], ascending=[True, True, False])
        .groupby("flux_bin", sort=True)
        .head(reference_top_k)
        .groupby("flux_bin", sort=True)
        .agg(
            reference_rate_hz=(RATE_COLUMN, "median"),
            reference_support=(RATE_COLUMN, "size"),
            reference_distance=("distance_to_perfect", "median"),
        )
        .reset_index()
    )
    supported = work.merge(references, on="flux_bin", how="inner")
    supported["scale_factor"] = supported["reference_rate_hz"] / supported[RATE_COLUMN]
    supported = supported.loc[np.isfinite(supported["scale_factor"]) & (supported["scale_factor"] > 0)].copy()

    bin_columns = [f"{column}_bin" for column in EFF_COLUMNS]
    lut = (
        supported.groupby(bin_columns, dropna=False)
        .agg(
            scale_factor=("scale_factor", "median"),
            scale_factor_q25=("scale_factor", lambda values: values.quantile(0.25)),
            scale_factor_q75=("scale_factor", lambda values: values.quantile(0.75)),
            support_rows=("scale_factor", "size"),
            supported_flux_bins=("flux_bin", "nunique"),
            rate_hz_median=(RATE_COLUMN, "median"),
            flux_cm2_min_median=("flux_cm2_min", "median"),
        )
        .reset_index()
        .rename(columns={f"{column}_bin": column for column in EFF_COLUMNS})
        .sort_values(EFF_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )
    references["flux_bin_left"] = references["flux_bin"].map(lambda index: float(flux_edges[index]))
    references["flux_bin_right"] = references["flux_bin"].map(lambda index: float(flux_edges[index + 1]))
    return lut, references


def make_plots(training: pd.DataFrame, lut: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
    for plane, ax in enumerate(axes.flat, start=1):
        scatter = ax.scatter(
            training[f"eff_{plane}"],
            training[RATE_COLUMN],
            c=training["flux_cm2_min"],
            s=12,
            alpha=0.55,
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title(f"Plane {plane}")
        ax.set_xlabel("Simulated efficiency")
        ax.set_ylabel("Generated file trigger rate [Hz]")
        ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Simulated flux [cm$^{-2}$ min$^{-1}$]")
    path = output_dir / "training_rate_vs_efficiency.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    points = ax.scatter(
        lut["eff_2"],
        lut["scale_factor"],
        c=lut[["eff_1", "eff_3", "eff_4"]].mean(axis=1),
        s=np.clip(lut["support_rows"] * 5, 18, 130),
        alpha=0.75,
        cmap="viridis",
        rasterized=True,
    )
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("Plane-2 simulated efficiency")
    ax.set_ylabel("Generated trigger-rate scale factor")
    ax.set_title("Simulation trigger-rate scale-factor LUT")
    ax.grid(alpha=0.25)
    fig.colorbar(points, ax=ax, label="Mean simulated efficiency of planes 1, 3, 4")
    for extension in ("png", "pdf"):
        path = output_dir / f"poster_scale_factor_vs_plane2_efficiency.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the minimal simulation-only four-plane LUT.")
    parser.add_argument("--simulation-csv", type=Path, default=DEFAULT_SIMULATION_CSV)
    parser.add_argument("--z-positions", type=float, nargs=4, default=(0.0, 145.0, 290.0, 435.0))
    parser.add_argument("--min-selected-rows", type=int, default=10_000)
    parser.add_argument("--efficiency-bin-width", type=float, default=0.01)
    parser.add_argument("--flux-bin-count", type=int, default=3)
    parser.add_argument("--reference-top-k", type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    z_positions = tuple(args.z_positions)
    training = load_selected_training(args.simulation_csv, z_positions, args.min_selected_rows)
    if training.empty:
        raise SystemExit(f"No rows matched z_positions={z_positions}.")
    lut, references = build_lut(
        training,
        efficiency_bin_width=args.efficiency_bin_width,
        flux_bin_count=args.flux_bin_count,
        reference_top_k=args.reference_top_k,
    )
    training_path = OUTPUT_DIR / "selected_training_rows.csv"
    lut_path = OUTPUT_DIR / "four_plane_rate_scale_factor_lut.csv"
    reference_path = OUTPUT_DIR / "flux_bin_reference_rates.csv"
    metadata_path = OUTPUT_DIR / "run_summary.json"
    training.to_csv(training_path, index=False)
    lut.to_csv(lut_path, index=False)
    references.to_csv(reference_path, index=False)
    plot_paths = make_plots(training, lut, OUTPUT_DIR)
    metadata_path.write_text(
        json.dumps(
            {
                "z_positions": z_positions,
                "training_rows": len(training),
                "lut_rows": len(lut),
                "rate_column": RATE_COLUMN,
                "efficiency_bin_width": args.efficiency_bin_width,
                "flux_bin_count": args.flux_bin_count,
                "reference_top_k": args.reference_top_k,
                "outputs": [str(path) for path in [training_path, lut_path, reference_path, *plot_paths]],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Selected z positions: {z_positions}")
    print(f"Selected training rows: {len(training)}")
    print(f"LUT rows: {len(lut)}")
    print(f"LUT: {lut_path}")
    for path in plot_paths:
        print(f"Plot: {path}")


if __name__ == "__main__":
    main()
