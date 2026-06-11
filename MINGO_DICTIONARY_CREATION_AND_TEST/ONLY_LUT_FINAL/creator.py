#!/usr/bin/env python3
"""
Build final LUT diagnostic plots for 1234 and combined trigger combinations.

Curves included:

    1234:
        R_1234 = tt_task0_acq_1234_rate_hz

    combined:
        R_combined = R_1234 + R_134 + R_124

For each curve:

    A_combo = R_combo / Phi

where:
    R_combo is the TASK_0 trigger rate in Hz
    Phi is the flux in cts s^-1 cm^-2

Each curve gets its own extrapolated top effective area:

    A_combo,0 = A_combo(mean_efficiency = 1)

and its own scale factor:

    F_combo = A_combo / A_combo,0

The correction factor is:

    C_combo = 1 / F_combo

Independent-efficiency references:

    1234:
        eff1 * eff2 * eff3 * eff4

    combined:
        eff1 * eff2       * eff3       * eff4
        + eff1 * (1-eff2) * eff3       * eff4
        + eff1 * eff2     * (1-eff3)   * eff4

The script accepts both `tt_task0_acq_*_rate_hz` and
`tt_task0_acs_*_rate_hz` spellings.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RATE_INPUT = Path(
    "/home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/"
    "MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_0/METADATA/"
    "task_0_metadata_trigger_type.csv"
)

DEFAULT_SIMULATION_INPUT = Path(
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/"
    "step_final_simulation_params.csv"
)

OUTPUT_DIR = Path(
    "/home/mingo/DATAFLOW_v3/MINGO_DICTIONARY_CREATION_AND_TEST/"
    "ONLY_LUT_FINAL/PLOTS"
)

OUTPUT_PNG = OUTPUT_DIR / "final_lut_1234_and_combined_134_124_diagnostics.png"

PARAMETER_COLUMNS = ("flux_cm2_min", "efficiencies")

MERGE_KEY_CANDIDATES = (
    "file_name",
    "param_hash",
    "param_set_id",
    "param_date",
    "execution_time",
    "cos_n",
    "z_plane_1",
    "z_plane_2",
    "z_plane_3",
    "z_plane_4",
)

BASE_COMBOS = ("1234", "134", "124")
PLOT_COMBOS = ("1234", "combined")

RATE_COLUMN_ALIASES = {
    "1234": (
        "tt_task0_acq_1234_rate_hz",
        "tt_task0_acs_1234_rate_hz",
    ),
    "134": (
        "tt_task0_acq_134_rate_hz",
        "tt_task0_acs_134_rate_hz",
    ),
    "124": (
        "tt_task0_acq_124_rate_hz",
        "tt_task0_acs_124_rate_hz",
    ),
}

PLOT_COLOURS = {
    "1234": "tab:blue",
    "combined": "tab:red",
}

REFERENCE_LABELS = {
    "1234": r"$\epsilon_1\epsilon_2\epsilon_3\epsilon_4$",
    "combined": (
        r"$\epsilon_1\epsilon_2\epsilon_3\epsilon_4"
        r"+\epsilon_1(1-\epsilon_2)\epsilon_3\epsilon_4"
        r"+\epsilon_1\epsilon_2(1-\epsilon_3)\epsilon_4$"
    ),
}


def parse_efficiencies(value: object) -> list[float]:
    """Parse the efficiencies column, expected as a list-like string."""
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, (list, tuple)):
        return []

    try:
        return [float(x) for x in parsed]
    except (TypeError, ValueError):
        return []


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Return stable metadata keys available in both tables."""
    return [c for c in MERGE_KEY_CANDIDATES if c in left.columns and c in right.columns]


def require_columns(df: pd.DataFrame, columns: Iterable[str], csv_path: Path) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")


def resolve_rate_column(df: pd.DataFrame, combo: str, csv_path: Path) -> str:
    """Resolve the actual TASK_0 rate column for one trigger combination."""
    for col in RATE_COLUMN_ALIASES[combo]:
        if col in df.columns:
            return col

    raise ValueError(
        f"Missing rate column for trigger combination {combo} in {csv_path}. "
        f"Accepted names are: {list(RATE_COLUMN_ALIASES[combo])}"
    )


def attach_simulation_parameters(
    rate_df: pd.DataFrame,
    rate_csv_path: Path,
    simulation_csv_path: Path,
) -> pd.DataFrame:
    """
    Ensure `flux_cm2_min` and `efficiencies` are available.

    If they are absent from the TASK_0 metadata file, attach them from
    step_final_simulation_params.csv using stable common metadata keys.
    """
    missing_params = [c for c in PARAMETER_COLUMNS if c not in rate_df.columns]
    if not missing_params:
        return rate_df.copy()

    if not simulation_csv_path.exists():
        raise FileNotFoundError(
            "The TASK_0 metadata file does not contain "
            f"{missing_params}, and the simulation parameter file was not found: "
            f"{simulation_csv_path}"
        )

    sim_df = pd.read_csv(simulation_csv_path)
    require_columns(sim_df, PARAMETER_COLUMNS, simulation_csv_path)

    merge_keys = choose_merge_keys(rate_df, sim_df)
    if not merge_keys:
        raise ValueError(
            "Cannot attach flux/efficiencies because no common merge key was "
            f"found between {rate_csv_path} and {simulation_csv_path}. "
            f"Expected at least one of: {list(MERGE_KEY_CANDIDATES)}"
        )

    columns_to_take = merge_keys + missing_params

    sim_subset = (
        sim_df[columns_to_take]
        .drop_duplicates(subset=merge_keys)
        .copy()
    )

    merged = rate_df.merge(
        sim_subset,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    missing_after_merge = [c for c in PARAMETER_COLUMNS if c not in merged.columns]
    if missing_after_merge:
        raise ValueError(
            "The following required parameter columns are still absent after "
            f"the merge: {missing_after_merge}. "
            f"Available columns are: {list(merged.columns)}"
        )

    unmatched = merged[list(PARAMETER_COLUMNS)].isna().any(axis=1).sum()
    if unmatched:
        print(
            "Warning: "
            f"{unmatched} rows could not be matched to flux/efficiency parameters "
            f"using keys {merge_keys}. These rows will be excluded."
        )

    return merged


def fit_extrapolation(
    mean_efficiency: np.ndarray,
    effective_area_cm2: np.ndarray,
    fit_degree: int,
    combo: str,
) -> tuple[np.poly1d, float]:
    """Fit A_combo(mean_efficiency) and evaluate at mean_efficiency = 1."""
    if len(mean_efficiency) < fit_degree + 1:
        raise ValueError(
            f"Cannot perform degree-{fit_degree} fit for {combo} with only "
            f"{len(mean_efficiency)} valid points."
        )

    coeffs = np.polyfit(mean_efficiency, effective_area_cm2, deg=fit_degree)
    poly = np.poly1d(coeffs)
    effective_area_at_eff_1 = float(poly(1.0))

    if not np.isfinite(effective_area_at_eff_1) or effective_area_at_eff_1 <= 0:
        raise ValueError(
            f"The extrapolated effective area for {combo} at mean efficiency = 1 "
            f"is not positive and finite: {effective_area_at_eff_1}"
        )

    return poly, effective_area_at_eff_1


def load_data(
    rate_csv_path: Path,
    simulation_csv_path: Path,
) -> pd.DataFrame:
    rate_df = pd.read_csv(rate_csv_path)

    rate_columns = {
        combo: resolve_rate_column(rate_df, combo, rate_csv_path)
        for combo in BASE_COMBOS
    }

    df = attach_simulation_parameters(
        rate_df=rate_df,
        rate_csv_path=rate_csv_path,
        simulation_csv_path=simulation_csv_path,
    )

    require_columns(df, ["flux_cm2_min", "efficiencies"], rate_csv_path)

    df["eff_list"] = df["efficiencies"].apply(parse_efficiencies)

    df["eff1"] = df["eff_list"].apply(lambda xs: xs[0] if len(xs) == 4 else np.nan)
    df["eff2"] = df["eff_list"].apply(lambda xs: xs[1] if len(xs) == 4 else np.nan)
    df["eff3"] = df["eff_list"].apply(lambda xs: xs[2] if len(xs) == 4 else np.nan)
    df["eff4"] = df["eff_list"].apply(lambda xs: xs[3] if len(xs) == 4 else np.nan)

    df["mean_efficiency"] = df[["eff1", "eff2", "eff3", "eff4"]].mean(axis=1)

    df["flux_cm2_min"] = pd.to_numeric(df["flux_cm2_min"], errors="coerce")

    # CSV flux units:
    #   flux_cm2_min = cts min^-1 cm^-2
    # Convert to:
    #   flux_cm2_s = cts s^-1 cm^-2
    # because trigger rates are in Hz = cts s^-1.
    df["flux_cm2_s"] = df["flux_cm2_min"] / 60.0

    for combo, col in rate_columns.items():
        df[f"rate_{combo}_hz"] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------------------------
    # Curve 1: pure 1234.
    # -------------------------------------------------------------------------
    df["rate_1234_plot_hz"] = df["rate_1234_hz"]

    df["effective_area_1234_cm2"] = (
        df["rate_1234_plot_hz"] / df["flux_cm2_s"]
    )

    df["reference_efficiency_1234"] = (
        df["eff1"] * df["eff2"] * df["eff3"] * df["eff4"]
    )

    # -------------------------------------------------------------------------
    # Curve 2: combined = 1234 + 134 + 124.
    # -------------------------------------------------------------------------
    df["rate_combined_hz"] = (
        df["rate_1234_hz"]
        + df["rate_134_hz"]
        + df["rate_124_hz"]
    )

    df["effective_area_combined_cm2"] = (
        df["rate_combined_hz"] / df["flux_cm2_s"]
    )

    df["reference_efficiency_combined"] = (
        df["eff1"] * df["eff2"] * df["eff3"] * df["eff4"]
        + df["eff1"] * (1.0 - df["eff2"]) * df["eff3"] * df["eff4"]
        + df["eff1"] * df["eff2"] * (1.0 - df["eff3"]) * df["eff4"]
    )

    for combo in PLOT_COMBOS:
        df[f"reference_correction_{combo}"] = (
            1.0 / df[f"reference_efficiency_{combo}"]
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    required_numeric = [
        "mean_efficiency",
        "flux_cm2_min",
        "flux_cm2_s",
        "eff1",
        "eff2",
        "eff3",
        "eff4",
        "rate_1234_hz",
        "rate_134_hz",
        "rate_124_hz",
        "rate_1234_plot_hz",
        "rate_combined_hz",
        "effective_area_1234_cm2",
        "effective_area_combined_cm2",
        "reference_efficiency_1234",
        "reference_efficiency_combined",
        "reference_correction_1234",
        "reference_correction_combined",
    ]

    valid = df[required_numeric].notna().all(axis=1)

    valid &= df["flux_cm2_min"] > 0
    valid &= df["flux_cm2_s"] > 0

    valid &= df["rate_1234_hz"] >= 0
    valid &= df["rate_134_hz"] >= 0
    valid &= df["rate_124_hz"] >= 0
    valid &= df["rate_1234_plot_hz"] >= 0
    valid &= df["rate_combined_hz"] >= 0

    valid &= df["effective_area_1234_cm2"] >= 0
    valid &= df["effective_area_combined_cm2"] >= 0

    valid &= df["reference_efficiency_1234"] > 0
    valid &= df["reference_efficiency_combined"] > 0

    return df.loc[valid].copy()


def make_plot(
    df: pd.DataFrame,
    output_png: Path,
    fit_degree: int,
) -> None:
    x_mean = df["mean_efficiency"].to_numpy(dtype=float)

    fit_results: dict[str, dict[str, object]] = {}

    for combo in PLOT_COMBOS:
        y_area = df[f"effective_area_{combo}_cm2"].to_numpy(dtype=float)

        poly, area_at_1 = fit_extrapolation(
            mean_efficiency=x_mean,
            effective_area_cm2=y_area,
            fit_degree=fit_degree,
            combo=combo,
        )

        fit_results[combo] = {
            "poly": poly,
            "area_at_1": area_at_1,
        }

        df[f"scale_factor_{combo}"] = (
            df[f"effective_area_{combo}_cm2"] / area_at_1
        )

        df[f"correction_factor_{combo}"] = (
            1.0 / df[f"scale_factor_{combo}"]
        )

        df[f"correction_residual_{combo}"] = (
            df[f"correction_factor_{combo}"]
            - df[f"reference_correction_{combo}"]
        )

        df[f"correction_ratio_{combo}"] = (
            df[f"correction_factor_{combo}"]
            / df[f"reference_correction_{combo}"]
        )

        # ---------------------------------------------------------------------
        # Relative residual of the reference scale-factor curve with respect to
        # the measured LUT scale factor.
        #
        # Measured is used as the denominator, as requested:
        #
        #   relative_residual = (reference - measured) / measured
        #
        # In percent:
        #
        #   100 * (reference - measured) / measured
        # ---------------------------------------------------------------------
        df[f"relative_scale_residual_{combo}"] = (
            (
                df[f"reference_efficiency_{combo}"]
                - df[f"scale_factor_{combo}"]
            )
            / df[f"scale_factor_{combo}"]
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    x_grid = np.linspace(min(float(np.min(x_mean)), 1.0), 1.0, 300)

    fig, (
        ax_rate_flux,
        ax_area,
        ax_scale,
        ax_correction,
        ax_residual,
    ) = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(9.8, 17.0),
        sharex=False,
    )

    # -------------------------------------------------------------------------
    # Panel 1:
    # Trigger rate vs flux, coloured by mean-efficiency bins.
    # -------------------------------------------------------------------------
    n_eff_bins = 5

    eff_min = float(df["mean_efficiency"].min())
    eff_max = float(df["mean_efficiency"].max())

    if np.isclose(eff_min, eff_max):
        df["mean_efficiency_bin"] = "single efficiency bin"
        bin_labels = ["single efficiency bin"]
    else:
        bin_edges = np.linspace(eff_min, eff_max, n_eff_bins + 1)

        bin_labels = [
            f"{bin_edges[i]:.3f}–{bin_edges[i + 1]:.3f}"
            for i in range(n_eff_bins)
        ]

        df["mean_efficiency_bin"] = pd.cut(
            df["mean_efficiency"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )

    rate_flux_data = (
        df[
            [
                "flux_cm2_s",
                "mean_efficiency",
                "mean_efficiency_bin",
                "rate_1234_plot_hz",
                "rate_combined_hz",
            ]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    cmap = plt.get_cmap("viridis")
    n_bins_actual = max(len(bin_labels), 1)

    for i, bin_label in enumerate(bin_labels):
        subset = rate_flux_data[
            rate_flux_data["mean_efficiency_bin"].astype(str) == str(bin_label)
        ]

        if subset.empty:
            continue

        colour = cmap(i / max(n_bins_actual - 1, 1))

        ax_rate_flux.scatter(
            subset["flux_cm2_s"],
            subset["rate_1234_plot_hz"],
            s=34,
            alpha=0.78,
            color=colour,
            marker="o",
            label=rf"1234, $\bar{{\epsilon}}$ {bin_label}",
        )

        ax_rate_flux.scatter(
            subset["flux_cm2_s"],
            subset["rate_combined_hz"],
            s=38,
            alpha=0.78,
            color=colour,
            marker="^",
            label=rf"combined, $\bar{{\epsilon}}$ {bin_label}",
        )

    ax_rate_flux.set_ylim(bottom=0)
    ax_rate_flux.set_xlabel(r"Flux, $\Phi$ [cts s$^{-1}$ cm$^{-2}$]")
    ax_rate_flux.set_ylabel(r"Trigger rate, $R$ [Hz]")
    ax_rate_flux.set_title(
        r"Trigger rate vs flux, binned by mean detector efficiency"
    )
    ax_rate_flux.grid(True, alpha=0.3)
    ax_rate_flux.legend(fontsize=6.2, ncols=2)

    # -------------------------------------------------------------------------
    # Panel 2:
    # Effective area for 1234 and combined.
    # -------------------------------------------------------------------------
    for combo in PLOT_COMBOS:
        colour = PLOT_COLOURS[combo]
        area_col = f"effective_area_{combo}_cm2"

        data = df[["mean_efficiency", area_col]].dropna().sort_values(
            "mean_efficiency"
        )

        poly = fit_results[combo]["poly"]
        area_at_1 = float(fit_results[combo]["area_at_1"])
        y_fit_grid = poly(x_grid)

        label_name = "1234 + 134 + 124" if combo == "combined" else "1234"

        ax_area.scatter(
            data["mean_efficiency"],
            data[area_col],
            s=32,
            alpha=0.78,
            color=colour,
            label=rf"Measured {label_name}: $R/\Phi$",
        )

        ax_area.plot(
            x_grid,
            y_fit_grid,
            linewidth=2.0,
            color=colour,
            label=rf"Fit {label_name}, $A_0={area_at_1:.3g}$ cm$^2$",
        )

        ax_area.scatter(
            [1.0],
            [area_at_1],
            marker="x",
            s=85,
            linewidths=2,
            color=colour,
        )

    ax_area.axvline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_area.set_ylim(bottom=0)
    ax_area.set_xlabel(r"Mean detector efficiency, $\bar{\epsilon}$")
    ax_area.set_ylabel(r"Effective area, $R/\Phi$ [cm$^2$]")
    ax_area.set_title("Effective area: 1234 and combined = 1234 + 134 + 124")
    ax_area.grid(True, alpha=0.3)
    ax_area.legend(fontsize=7, ncols=2)

    # -------------------------------------------------------------------------
    # Panel 3:
    # Scale factor for 1234 and combined.
    # -------------------------------------------------------------------------
    for combo in PLOT_COMBOS:
        colour = PLOT_COLOURS[combo]
        scale_col = f"scale_factor_{combo}"
        ref_col = f"reference_efficiency_{combo}"

        data = df[["mean_efficiency", scale_col, ref_col]].dropna().sort_values(
            "mean_efficiency"
        )

        label_name = "1234 + 134 + 124" if combo == "combined" else "1234"

        ax_scale.scatter(
            data["mean_efficiency"],
            data[scale_col],
            s=32,
            alpha=0.78,
            color=colour,
            label=rf"Measured {label_name}: $F=A/A_0$",
        )

        ax_scale.plot(
            data["mean_efficiency"],
            data[ref_col],
            linestyle="--",
            linewidth=2.0,
            color=colour,
            label=rf"Reference {label_name}: {REFERENCE_LABELS[combo]}",
        )

    ax_scale.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_scale.axvline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_scale.set_ylim(bottom=0)
    ax_scale.set_xlabel(r"Mean detector efficiency, $\bar{\epsilon}$")
    ax_scale.set_ylabel(r"Scale factor, $F=A/A_0$")
    ax_scale.grid(True, alpha=0.3)
    ax_scale.legend(fontsize=7, ncols=2)

    # -------------------------------------------------------------------------
    # Panel 4:
    # Correction factor for 1234 and combined.
    # -------------------------------------------------------------------------
    for combo in PLOT_COMBOS:
        colour = PLOT_COLOURS[combo]
        corr_col = f"correction_factor_{combo}"
        ref_corr_col = f"reference_correction_{combo}"

        data = df[["mean_efficiency", corr_col, ref_corr_col]].dropna().sort_values(
            "mean_efficiency"
        )

        label_name = "1234 + 134 + 124" if combo == "combined" else "1234"

        ax_correction.scatter(
            data["mean_efficiency"],
            data[corr_col],
            s=32,
            alpha=0.78,
            color=colour,
            label=rf"Measured {label_name}: $C=1/F$",
        )

        ax_correction.plot(
            data["mean_efficiency"],
            data[ref_corr_col],
            linestyle="--",
            linewidth=2.0,
            color=colour,
            label=rf"Reference {label_name}: $1/$({REFERENCE_LABELS[combo]})",
        )

    ax_correction.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_correction.axvline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_correction.set_ylim(bottom=0)
    ax_correction.set_xlabel(r"Mean detector efficiency, $\bar{\epsilon}$")
    ax_correction.set_ylabel(r"Correction factor, $C=1/F$")
    ax_correction.grid(True, alpha=0.3)
    ax_correction.legend(fontsize=7, ncols=2)

    # -------------------------------------------------------------------------
    # Panel 5:
    # Relative residual of reference scale-factor curve with respect to measured.
    #
    #   residual [%] = 100 * (F_reference - F_measured) / F_measured
    #
    # Measured is the denominator.
    # -------------------------------------------------------------------------
    for combo in PLOT_COMBOS:
        colour = PLOT_COLOURS[combo]
        residual_col = f"relative_scale_residual_{combo}"

        data = df[["mean_efficiency", residual_col]].dropna().sort_values(
            "mean_efficiency"
        )

        label_name = "1234 + 134 + 124" if combo == "combined" else "1234"

        ax_residual.scatter(
            data["mean_efficiency"],
            100.0 * data[residual_col],
            s=32,
            alpha=0.78,
            color=colour,
            label=rf"{label_name}: $100(F_{{ref}}-F_{{meas}})/F_{{meas}}$",
        )

    ax_residual.axhline(0.0, linestyle="--", linewidth=1.5, color="black", alpha=0.7)
    ax_residual.axvline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax_residual.set_xlabel(r"Mean detector efficiency, $\bar{\epsilon}$")
    ax_residual.set_ylabel(r"Relative residual [%]")
    ax_residual.set_title(
        r"Reference-line residual relative to measured LUT scale factor"
    )
    ax_residual.grid(True, alpha=0.3)
    ax_residual.legend(fontsize=7, ncols=2)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    print("Fitted top effective areas at mean efficiency = 1:")
    for combo in PLOT_COMBOS:
        label_name = "1234 + 134 + 124" if combo == "combined" else "1234"
        print(
            f"  A_{label_name},0 = "
            f"{float(fit_results[combo]['area_at_1']):.8g} cm^2"
        )

    print()
    print("Scale factors:")
    print("  F_1234 = A_1234 / A_1234,0")
    print("  F_combined = A_combined / A_combined,0")

    print()
    print("Correction factors:")
    print("  C_1234 = 1 / F_1234")
    print("  C_combined = 1 / F_combined")

    print()
    print("Independent-efficiency references:")
    print("  1234     : eff1 * eff2 * eff3 * eff4")
    print(
        "  combined : "
        "eff1 * eff2 * eff3 * eff4 "
        "+ eff1 * (1 - eff2) * eff3 * eff4 "
        "+ eff1 * eff2 * (1 - eff3) * eff4"
    )

    print()
    print("Relative residual diagnostic:")
    print("  residual = 100 * (F_reference - F_measured) / F_measured [%]")
    print("  measured is used as the denominator")
    print()
    print("Saved plot:")
    print(f"  {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final LUT diagnostic plots for TASK_0 trigger combinations "
            "1234 and combined = 1234 + 134 + 124. Each curve gets its own "
            "effective area and is normalized by its own fitted value at "
            "mean efficiency = 1."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RATE_INPUT,
        help=f"TASK_0 trigger-type metadata CSV. Default: {DEFAULT_RATE_INPUT}",
    )

    parser.add_argument(
        "--simulation-input",
        type=Path,
        default=DEFAULT_SIMULATION_INPUT,
        help=(
            "Simulation parameter CSV used only if the input metadata does not "
            f"already contain {PARAMETER_COLUMNS}. Default: {DEFAULT_SIMULATION_INPUT}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PNG,
        help=f"Output PNG path. Default: {OUTPUT_PNG}",
    )

    parser.add_argument(
        "--fit-degree",
        type=int,
        default=2,
        help=(
            "Polynomial degree for extrapolating each effective area to "
            "mean efficiency = 1. Default: 2."
        ),
    )

    args = parser.parse_args()

    if args.fit_degree < 1:
        raise ValueError(
            "A fit is required because each curve needs A_0 at mean efficiency = 1. "
            "Use --fit-degree 1 or --fit-degree 2."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(
        rate_csv_path=args.input,
        simulation_csv_path=args.simulation_input,
    )

    if df.empty:
        raise RuntimeError(
            "No valid rows found. Check that the 1234, 134, and 124 TASK_0 "
            "rate columns are filled, flux_cm2_min is positive, and efficiencies "
            "are parseable as four plane efficiencies."
        )

    print(f"Input CSV: {args.input}")
    print(f"Simulation CSV for parameter completion: {args.simulation_input}")
    print(f"Valid rows: {len(df)}")
    print(f"Output PNG: {args.output}")
    print()

    make_plot(
        df=df,
        output_png=args.output,
        fit_degree=args.fit_degree,
    )


if __name__ == "__main__":
    main()