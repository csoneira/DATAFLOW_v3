from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path("MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS")
STEP = ROOT / "STAGES/STAGE_1/EVENT_DATA/STEP_1"
CFG = ROOT / "CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1"
COMMON = ROOT / "common"

SCRIPT_NAMES = {
    0: "script_0_acq_to_raw.py",
    1: "script_1_raw_to_clean.py",
    2: "script_2_clean_to_cal.py",
    3: "script_3_cal_to_list.py",
    4: "script_4_list_to_fit.py",
    5: "script_5_fit_to_post.py",
}

EXPECTED_PREFIXES = {
    0: ("raw_", "selftrigger_raw_"),
    1: ("cleaned_",),
    2: ("calibrated_",),
    3: ("listed_",),
    4: ("fitted_",),
    5: ("postprocessed_",),
}

# Classification rule:
# - runtime: useful while the task is still executing / tied to intermediate state;
# - final: analysis/report plot reproducible after the task from its output parquet.
FINAL_ALIASES: dict[int, set[str]] = {
    0: {
        "acquisition_rate_vs_time_by_trigger_type",
        "acquisition_rate_vs_time_by_task_tt_with_histograms",
    },
    1: {
        "acquisition_rate_vs_time_by_task_tt_with_histograms",
        "event_total_charge_raw",
        "tq_scatter_raw_by_tt",
        "tq_scatter_filtered_by_tt",
        "channel_tq_matrix_by_planepair",
    },
    2: {
        "acquisition_rate_vs_time_by_task_tt_with_histograms",
        "streamer_contagion_matrix_strip",
        "calibrated_filtered_removed_zeroes",
        "calibrated_filtered_removed_zeroes_per_plane_combination",
        "calibrated_filtered_removed_zeroes_st",
        "time_calibrated_filtered_removed_zeroes",
        "stat_window_accumulation",
        "rejected_scatter_tt_task1_clean",
        "rejected_scatter_q_sum_zero",
        "rejected_scatter_tt_task2_cal",
        "charge_front_vs_back",
        "time_front_vs_back",
        "channel_matrix_tq",
        "strip_pair_matrix_by_planepair",
        "strip_combination_filter_by_tt",
        "rejected_histograms_tt_task1_clean",
        "rejected_histograms_q_sum_zero",
        "rejected_histograms_tt_task2_cal",
    },
    3: {
        "acquisition_rate_vs_time_by_task_tt_with_histograms",
        "y_position_by_tt_task2_cal",
        "strip_variable_pairgrid",
        "self_trigger_strip_variable_pairgrid",
        "rpc_variables_hexbin",
        "rpc_variables_hexbin_per_plane_combination",
        "rpc_variables_hexbin_low_charge",
        "filtered_rpc_tsum_vs_qsum_scatter",
        "filtered_rpc_tsum_vs_qsum_scatter_per_combination",
        "filtered_rpc_tsum_vs_qsum_hexbin",
        "filtered_rpc_variables_hexbin",
        "tt_task3_list_qsum_pairgrid",
        "all_events_charge_threshold_population",
        "source_tt_task3_list_charge_threshold_population",
        "tt_task3_list_transition_matrices",
        "tt_task3_list_retention_curves",
        "tt_task3_list_minimum_charge_distributions",
        "tt_task3_list_empirical_efficiency_vs_threshold",
        "full_topology_threshold_retention",
        "full_topology_exact_retention",
        "full_topology_class_fraction",
        "tsum_coincidence_window_histograms",
        "tsum_coincidence_window_vs_threshold",
        "plane_charge_fraction_vs_total_charge_threshold_scan",
        "charge_asymmetry_vs_threshold",
        "interplane_timing_correlation",
        "multiplicity_charge_landscape",
        "streamer_charge_histograms",
        "streamer_prevalence_by_plane",
        "streamer_multiplicity",
        "streamer_contagion_matrix",
        "streamer_contagion_matrix_strip",
        "streamer_efficiency_comparison",
        "fourplane_weakest_charge",
        "fourplane_weakest_plane_identity",
        "efficiency_anatomy_by_charge_band",
        "streamer_contagion_strip_by_tt",
        "signal_contagion_strip_by_tt",
        "streamer_to_signal_contagion_strip_by_tt",
        "streamer_to_highcharge_contagion_strip_by_tt",
        "charge_by_strip_multiplicity_adj",
        "charge_by_strip_multiplicity_dis",
    },
    4: {
        "acquisition_rate_vs_time_by_task_tt_with_histograms",
        "trigger_types_raw_and_list",
        "trigger_types_tracking_and_list",
        "trigger_types_tracking_and_raw",
        "trigger_types_definitive_tt_and_raw",
        "timtrack_results_hexbin_combination_projections",
        "timtrack_results_scatter_combination_projections",
        "theta_det_theta_zoom_tracking_tt",
        "polar_theta_phi_definitive_tt_2d",
        "polar_theta_phi_tracking_tt_2d",
        "events_per_second_by_plane_cardinality_double_row",
        "timtrack_residuals_gaussian",
        "tim_th_chi_sigmafit_1234_histogram",
        "external_residuals_gaussian",
        "event_display_sample",
        "track_consistency_loo_residuals",
        "strip_hit_occupancy",
        "track_based_efficiency",
        "total_event_charge_histogram",
        "track_based_efficiency_tt_stability",
        "track_based_efficiency_vs_theta",
        "track_efficiency_large_plot",
        "timtrack_projection_ellipse_contours",
        "timtrack_projection_scaled_angle_comparison",
        "chi2_offender_populations",
        "offender_angle_populations",
        "offender_kinematics_populations",
        "offender_tt_balance",
        "offender_zigzag_populations",
        "chi2_charge_populations",
        "chi2_residuals_populations",
        "event_display_sample_3fold",
        "chi2_charge_populations_3fold",
        "chi2_residuals_populations_3fold",
        "adj_dis_comparison",
    },
    5: {
        "acquisition_rate_vs_time_by_task_tt_with_histograms",
        "theta_phi_definitive_tt_2d",
        "polar_theta_phi_definitive_tt_2d_detail_final",
        "theta_efficiency_simple_3v4",
        "charge_tt_task4_fit_1234_per_plane_and_total",
    },
}


def parse_catalog(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    in_plots = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == "plots:":
            in_plots = True
            continue
        if not in_plots:
            continue
        match = re.match(r"^\s{2}([A-Za-z0-9_]+):\s*([^#\s]+)", raw_line)
        if match:
            entries.append((match.group(1), match.group(2)))
    return entries


def split_catalog(task: int) -> None:
    path = CFG / f"TASK_{task}/config_plots_task_{task}.yaml"
    entries = parse_catalog(path)
    known = {alias for alias, _status in entries}
    missing = sorted(FINAL_ALIASES[task] - known)
    if missing:
        raise RuntimeError(f"Task {task}: requested FINAL aliases are absent from current catalog: {missing}")

    runtime_entries = [(a, s) for a, s in entries if a not in FINAL_ALIASES[task]]
    final_entries = [(a, s) for a, s in entries if a in FINAL_ALIASES[task]]

    path.write_text(
        "# Runtime plots: PNG-only diagnostics created while the task is executing.\n"
        "plots:\n"
        + "".join(f"  {alias}: {status}\n" for alias, status in runtime_entries),
        encoding="utf-8",
    )

    final_path = path.with_name(f"config_final_plots_task_{task}.yaml")
    final_path.write_text(
        "# FINAL plots: standalone post-task analysis from the derived parquet only.\n"
        "plot_mode: all\n"
        "save_png: true\n"
        "save_pdf: true\n"
        "plots:\n"
        + "".join(f"  {alias}: {status}\n" for alias, status in final_entries),
        encoding="utf-8",
    )
    print(f"Task {task}: runtime={len(runtime_entries)}, final={len(final_entries)}")


def patch_shared_plot_contract() -> None:
    path = COMMON / "step1_shared.py"
    source = path.read_text(encoding="utf-8")

    old_missing = '''    for alias in plot_aliases:\n        if alias not in catalog:\n            log_fn(\n                f"{task_label} plot alias {alias!r} missing in {catalog_path.name}; defaulting to 'usual'."\n            )\n            catalog[alias] = "usual"\n'''
    new_missing = '''    for alias in plot_aliases:\n        if alias not in catalog:\n            # A known task alias absent from the runtime catalog is intentionally\n            # not a runtime plot (it may live in config_final_plots_task_N.yaml).\n            catalog[alias] = "none"\n'''
    if old_missing not in source:
        raise RuntimeError("Could not locate STEP_1 plot-catalog missing-alias fallback")
    source = source.replace(old_missing, new_missing, 1)

    old_pdf = "    create_pdf = save_plots\n"
    if old_pdf not in source:
        raise RuntimeError("Could not locate runtime create_pdf assignment in step1_shared.py")
    source = source.replace(
        old_pdf,
        "    create_pdf = False  # Runtime STEP_1 plots are strictly PNG-only.\n",
        1,
    )
    path.write_text(source, encoding="utf-8")


def comment_source(source: str) -> str:
    return "\n".join(("# " + line) if line else "#" for line in source.rstrip("\n").splitlines()) + "\n"


def append_legacy(task: int, sections: list[tuple[str, str]]) -> None:
    if not sections:
        return
    path = STEP / f"TASK_{task}/task_{task}_legacy.py"
    if path.exists():
        existing = path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        existing = (
            f"# LEGACY CODE ARCHIVE FOR TASK {task}\n"
            "# This file is not imported and is not expected to execute.\n\n"
        )
    chunks = [existing]
    for reason, block in sections:
        chunks.extend(
            [
                "# " + "=" * 76 + "\n",
                f"# ARCHIVED FROM: {SCRIPT_NAMES[task]}\n",
                f"# REASON: {reason}\n",
                "# " + "=" * 76 + "\n",
                comment_source(block),
                "\n",
            ]
        )
    path.write_text("".join(chunks), encoding="utf-8")


def apply_replacements(source: str, replacements: list[tuple[int, int, str]]) -> str:
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1 : end] = [replacement]
    return "".join(lines)


def node_string_constants(node: ast.AST) -> set[str]:
    return {
        value.value
        for value in ast.walk(node)
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }


def looks_like_plot_block(text: str) -> bool:
    markers = (
        "plt.",
        "save_plot_figure",
        "savefig",
        "plot_",
        "hexbin",
        "scatter",
        "hist(",
        "imshow",
        "create_rate_vs_time",
        "render_",
    )
    return any(marker in text for marker in markers)


def extract_final_plot_blocks(task: int) -> None:
    path = STEP / f"TASK_{task}/{SCRIPT_NAMES[task]}"
    source = path.read_text(encoding="utf-8")

    # Any task-local mode helper must also obey the PNG-only runtime invariant.
    source = source.replace(
        "create_pdf = save_plots",
        "create_pdf = False  # Runtime plots are strictly PNG-only; FINAL PDFs are standalone.",
    )

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    final_aliases = FINAL_ALIASES[task]
    runtime_aliases = {alias for alias, _ in parse_catalog(CFG / f"TASK_{task}/config_plots_task_{task}.yaml")}

    # Map simple boolean helper variables to the alias that defined them. This
    # catches patterns such as `_plot_x = taskN_plot_enabled("x")` followed by
    # `if _plot_x:`.
    variable_alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = getattr(node, "value", None)
        if value is None:
            continue
        hits = node_string_constants(value) & final_aliases
        if len(hits) != 1:
            continue
        alias = next(iter(hits))
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                variable_alias[target.id] = alias

    candidates: list[tuple[ast.If, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.end_lineno is None or node.orelse:
            continue
        block = "".join(lines[node.lineno - 1 : node.end_lineno])
        if not looks_like_plot_block(block):
            continue

        strings = node_string_constants(node)
        final_hits = strings & final_aliases
        runtime_hits = strings & runtime_aliases
        names = {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)}
        final_hits |= {variable_alias[name] for name in names if name in variable_alias}

        # Remove only blocks whose plot identity is exclusively FINAL. Mixed
        # runtime/final blocks remain in place; FINAL aliases are disabled by
        # their absence from the runtime catalog and will be handled in review.
        if final_hits and not runtime_hits:
            candidates.append((node, final_hits))

    # Keep only outermost selected blocks to avoid overlapping line edits.
    candidates.sort(key=lambda item: (item[0].lineno, -(item[0].end_lineno or item[0].lineno)))
    selected: list[tuple[ast.If, set[str]]] = []
    for node, aliases in candidates:
        if any(
            outer.lineno <= node.lineno
            and (node.end_lineno or node.lineno) <= (outer.end_lineno or outer.lineno)
            for outer, _aliases in selected
        ):
            continue
        selected.append((node, aliases))

    replacements: list[tuple[int, int, str]] = []
    archived: list[tuple[str, str]] = []
    for node, aliases in selected:
        indent = lines[node.lineno - 1][: len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
        label = ", ".join(sorted(aliases))
        replacement = (
            f"{indent}# FINAL plot block moved out of the runtime flow: {label}\n"
            f"{indent}pass\n"
        )
        block = "".join(lines[node.lineno - 1 : node.end_lineno])
        replacements.append((node.lineno, node.end_lineno, replacement))
        archived.append(
            (
                f"FINAL/report plot block moved to standalone final plotting ({label})",
                block,
            )
        )

    if replacements:
        source = apply_replacements(source, replacements)
    path.write_text(source, encoding="utf-8")
    append_legacy(task, archived)
    print(f"Task {task}: archived {len(archived)} FINAL-only runtime block(s)")


FINAL_PLOTTING_SOURCE = r'''from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import yaml

TASK_TT_COLUMNS = {
    0: ("tt_task0_raw", "tt_task0_acq"),
    1: ("tt_task1_clean", "tt_task0_raw"),
    2: ("tt_task2_cal", "tt_task1_clean"),
    3: ("tt_task3_list", "tt_task2_cal"),
    4: ("tt_task4_fit", "tt_task3_list"),
    5: ("tt_task5_post", "tt_task4_fit"),
}


def _first_existing(df: pd.DataFrame, names) -> str | None:
    return next((str(name) for name in names if str(name) in df.columns), None)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _finite_pair(df: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    xv = _numeric(df, x).to_numpy(dtype=float)
    yv = _numeric(df, y).to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    return xv[mask], yv[mask]


def _tt_column(df: pd.DataFrame, task: int) -> str | None:
    return _first_existing(df, TASK_TT_COLUMNS[task])


def _theta_phi(df: pd.DataFrame) -> tuple[str | None, str | None]:
    return (
        _first_existing(df, ("event_theta", "theta", "event_tim_theta", "tim_theta", "event_det_theta", "det_theta")),
        _first_existing(df, ("event_phi", "phi", "event_tim_phi", "tim_phi", "event_det_phi", "det_phi")),
    )


def _event_charge(df: pd.DataFrame) -> pd.Series | None:
    direct = _first_existing(df, ("event_charge", "charge_event", "tim_event_charge"))
    if direct:
        return _numeric(df, direct)
    plane_columns = [f"p{plane}_qsum" for plane in range(1, 5) if f"p{plane}_qsum" in df.columns]
    if plane_columns:
        return df[plane_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    strip_columns = [column for column in df.columns if re.fullmatch(r"p[1-4]_s[1-4]_qsum", str(column))]
    if strip_columns:
        return df[strip_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    return None


def _preferred_numeric_columns(df: pd.DataFrame, task: int, limit: int = 6) -> list[str]:
    patterns = {
        0: (r"p[1-4]_s[1-4]_e[fb]_[qt]$", r"tt_task0_"),
        1: (r"p[1-4]_s[1-4]_e[fb]_[qt]$", r"filter_task1_", r"tt_task1_"),
        2: (r"p[1-4]_s[1-4]_(qsum|qdif|tsum|tdif)$", r"tt_task2_"),
        3: (r"p[1-4]_(qsum|qdif|tsum|tdif|ypos|xpos)$", r"tt_task3_"),
        4: (r"event_(x|y|theta|phi|s|t0)$", r"event_tim_", r"event_det_", r"p[1-4]_.*res", r"chi"),
        5: (r"event_(x|y|theta|phi|s|t0|charge)$", r"p[1-4]_qsum$", r"tt_task5_"),
    }[task]
    result: list[str] = []
    for pattern in patterns:
        for column in df.columns:
            name = str(column)
            if name in result or re.search(pattern, name) is None:
                continue
            if pd.to_numeric(df[column], errors="coerce").notna().any():
                result.append(name)
                if len(result) >= limit:
                    return result
    for column in df.select_dtypes(include=[np.number]).columns:
        name = str(column)
        if name not in result:
            result.append(name)
        if len(result) >= limit:
            break
    return result


def _rate_plot(df: pd.DataFrame, task: int, alias: str):
    time_column = _first_existing(df, ("datetime", "Time", "timestamp"))
    if time_column is None:
        return _tt_plot(df, task, alias)
    times = pd.to_datetime(df[time_column], errors="coerce")
    valid = times.notna()
    if not valid.any():
        return _tt_plot(df, task, alias)
    frame = pd.DataFrame({"time": times.loc[valid]})
    tt_column = _tt_column(df, task)
    if tt_column:
        frame["tt"] = df.loc[valid, tt_column].astype(str).to_numpy()
    span = frame["time"].max() - frame["time"].min()
    frequency = "1min" if span <= pd.Timedelta(hours=6) else "10min"
    frame["bin"] = frame["time"].dt.floor(frequency)
    fig, ax = plt.subplots(figsize=(11, 5))
    if "tt" in frame:
        for label in frame["tt"].value_counts().head(8).index:
            counts = frame.loc[frame["tt"] == label].groupby("bin").size()
            ax.plot(counts.index, counts.values, marker=".", linewidth=1, label=str(label))
        ax.legend(title=tt_column, fontsize=8, ncol=2)
    else:
        counts = frame.groupby("bin").size()
        ax.plot(counts.index, counts.values, marker=".")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Events / {frequency}")
    ax.set_title(alias.replace("_", " "))
    ax.grid(alpha=0.25)
    return fig


def _tt_plot(df: pd.DataFrame, task: int, alias: str):
    column = _tt_column(df, task)
    if column is None:
        return None
    counts = df[column].astype(str).value_counts().head(30)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel(column)
    ax.set_ylabel("Events")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title(alias.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _hist_grid(df: pd.DataFrame, task: int, alias: str, columns: list[str] | None = None):
    columns = columns or _preferred_numeric_columns(df, task, 6)
    columns = [column for column in columns if column in df.columns]
    if not columns:
        return None
    rows = int(np.ceil(len(columns) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows), squeeze=False)
    for ax, column in zip(axes.ravel(), columns):
        values = _numeric(df, column).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=80, alpha=0.8)
        ax.set_title(column)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(alias.replace("_", " "))
    fig.tight_layout()
    return fig


def _charge_plot(df: pd.DataFrame, task: int, alias: str):
    columns = [column for column in ("event_charge", "p1_qsum", "p2_qsum", "p3_qsum", "p4_qsum") if column in df.columns]
    charge = _event_charge(df)
    if charge is not None and "event_charge" not in columns:
        temporary = df.copy()
        temporary["__derived_event_charge"] = charge
        return _hist_grid(temporary, task, alias, ["__derived_event_charge", *columns[:5]])
    if columns:
        return _hist_grid(df, task, alias, columns[:6])
    return _hist_grid(df, task, alias)


def _pair_matrix(df: pd.DataFrame, task: int, alias: str, mode: str):
    columns = _preferred_numeric_columns(df, task, 5 if task == 4 else 4)
    if len(columns) < 2:
        return None
    n_columns = len(columns)
    fig, axes = plt.subplots(n_columns, n_columns, figsize=(3.0 * n_columns, 3.0 * n_columns), squeeze=False)
    for row, y_column in enumerate(columns):
        for col, x_column in enumerate(columns):
            ax = axes[row, col]
            xv, yv = _finite_pair(df, x_column, y_column)
            if row == col:
                if xv.size:
                    ax.hist(xv, bins=50, alpha=0.8)
            elif xv.size:
                if mode == "hexbin":
                    ax.hexbin(xv, yv, gridsize=45, mincnt=1, bins="log")
                else:
                    sample_count = min(12000, xv.size)
                    indices = np.linspace(0, xv.size - 1, sample_count, dtype=int)
                    ax.scatter(xv[indices], yv[indices], s=2, alpha=0.25)
            if row == n_columns - 1:
                ax.set_xlabel(x_column, fontsize=7)
            if col == 0:
                ax.set_ylabel(y_column, fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(alias.replace("_", " "))
    fig.tight_layout()
    return fig


def _theta_phi_plot(df: pd.DataFrame, alias: str):
    theta, phi = _theta_phi(df)
    if theta is None or phi is None:
        return None
    phi_values, theta_values = _finite_pair(df, phi, theta)
    if not phi_values.size:
        return None
    if "polar" in alias:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        ax.scatter(phi_values, theta_values, s=3, alpha=0.25)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.hexbin(phi_values, theta_values, gridsize=60, mincnt=1, bins="log")
        fig.colorbar(image, ax=ax, label="log count")
        ax.set_xlabel(phi)
        ax.set_ylabel(theta)
    ax.set_title(alias.replace("_", " "))
    return fig


def _matrix_plot(df: pd.DataFrame, task: int, alias: str):
    columns = [str(column) for column in df.columns if re.search(r"(qsum|_q$)", str(column))][:16]
    if len(columns) < 2:
        columns = _preferred_numeric_columns(df, task, 10)
    if len(columns) < 2:
        return None
    values = pd.DataFrame(index=df.index)
    for column in columns:
        numeric = _numeric(df, column)
        values[column] = (numeric.fillna(0) != 0).astype(float) if re.search(r"qsum|_q$", column) else numeric
    matrix = values.corr().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(values.columns)), values.columns, fontsize=7)
    ax.set_title(alias.replace("_", " "))
    fig.tight_layout()
    return fig


def _success_mask(df: pd.DataFrame, task: int) -> pd.Series | None:
    passes = f"passes_task_{task}"
    if passes in df.columns:
        return pd.to_numeric(df[passes], errors="coerce").fillna(0) > 0
    tt_column = _tt_column(df, task)
    if tt_column is None:
        return None
    labels = df[tt_column].astype(str).str.replace(".0", "", regex=False)
    return labels.str.len() >= 4


def _efficiency_plot(df: pd.DataFrame, task: int, alias: str):
    success = _success_mask(df, task)
    if success is None:
        return None
    theta, _phi = _theta_phi(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    if theta:
        values = _numeric(df, theta)
        valid = values.notna()
        if valid.sum() > 10:
            low, high = values[valid].quantile([0.01, 0.99])
            if np.isfinite(low) and np.isfinite(high) and high > low:
                bins = np.linspace(low, high, 16)
                categories = pd.cut(values[valid], bins=bins, include_lowest=True)
                efficiency = success[valid].groupby(categories, observed=True).mean()
                centers = np.array([interval.mid for interval in efficiency.index])
                ax.plot(centers, efficiency.to_numpy(), marker="o")
                ax.set_xlabel(theta)
                ax.set_ylabel("Efficiency / pass fraction")
                ax.set_ylim(0, 1.05)
            else:
                ax.bar(["pass", "fail"], [int(success.sum()), int((~success).sum())])
        else:
            ax.bar(["pass", "fail"], [int(success.sum()), int((~success).sum())])
    else:
        ax.bar(["pass", "fail"], [int(success.sum()), int((~success).sum())])
    ax.set_title(alias.replace("_", " "))
    ax.grid(alpha=0.25)
    return fig


def _occupancy_plot(df: pd.DataFrame, alias: str):
    columns = [str(column) for column in df.columns if re.fullmatch(r"p[1-4](_s[1-4])?_qsum", str(column))]
    if not columns:
        return None
    fractions = [float((_numeric(df, column).fillna(0) > 0).mean()) for column in columns]
    fig, ax = plt.subplots(figsize=(max(9, len(columns) * 0.45), 5))
    ax.bar(columns, fractions)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Non-zero fraction")
    ax.tick_params(axis="x", rotation=75)
    ax.set_title(alias.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _threshold_plot(df: pd.DataFrame, alias: str):
    charge = _event_charge(df)
    if charge is None:
        return None
    values = charge.to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return None
    thresholds = np.quantile(values, np.linspace(0.0, 0.95, 40))
    retention = np.array([(values >= threshold).mean() for threshold in thresholds])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, retention, marker=".")
    ax.set_xlabel("Charge threshold")
    ax.set_ylabel("Retained fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(alias.replace("_", " "))
    ax.grid(alpha=0.25)
    return fig


def _topology_plot(df: pd.DataFrame, alias: str):
    column = _first_existing(df, ("topology_task2_strip", "topology_task3_plane", "transferred_task3_cal_to_list"))
    if column is None:
        return None
    counts = df[column].astype(str).value_counts().head(25)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(counts.index, counts.values)
    ax.tick_params(axis="x", rotation=80)
    ax.set_ylabel("Events")
    ax.set_title(alias.replace("_", " "))
    fig.tight_layout()
    return fig


def _event_display(df: pd.DataFrame, alias: str):
    y_columns = [f"p{plane}_ypos" for plane in range(1, 5)]
    x_columns = [f"p{plane}_xpos" for plane in range(1, 5)]
    if not all(column in df.columns for column in y_columns):
        return None
    z_values = []
    for plane in range(1, 5):
        z_column = f"z_p{plane}"
        if z_column in df.columns:
            finite = _numeric(df, z_column).dropna()
            z_values.append(float(finite.iloc[0]) if not finite.empty else float(plane - 1))
        else:
            z_values.append(float(plane - 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for sample_index, (_index, row) in enumerate(df.head(8).iterrows()):
        y = [pd.to_numeric(row[column], errors="coerce") for column in y_columns]
        axes[1].plot(y, z_values, marker="o", alpha=0.55, label=str(sample_index))
        if all(column in df.columns for column in x_columns):
            x = [pd.to_numeric(row[column], errors="coerce") for column in x_columns]
            axes[0].plot(x, z_values, marker="o", alpha=0.55)
    axes[0].set_xlabel("x")
    axes[1].set_xlabel("y")
    for ax in axes:
        ax.set_ylabel("z")
        ax.grid(alpha=0.25)
    axes[1].legend(title="sample", fontsize=7)
    fig.suptitle(alias.replace("_", " "))
    fig.tight_layout()
    return fig


def _projection_plot(df: pd.DataFrame, alias: str):
    x_column = _first_existing(df, ("event_x", "x", "event_tim_x", "tim_x", "event_det_x", "det_x"))
    y_column = _first_existing(df, ("event_y", "y", "event_tim_y", "tim_y", "event_det_y", "det_y"))
    if x_column is None or y_column is None:
        return None
    xv, yv = _finite_pair(df, x_column, y_column)
    if xv.size < 3:
        return None
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(xv, yv, gridsize=60, mincnt=1, bins="log")
    covariance = np.cov(np.vstack([xv, yv]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    center = (float(np.nanmean(xv)), float(np.nanmean(yv)))
    for scale in (1, 2):
        width, height = 2 * scale * np.sqrt(np.maximum(eigenvalues, 0))
        ax.add_patch(Ellipse(center, width, height, angle=angle, fill=False, linewidth=1.5))
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(alias.replace("_", " "))
    ax.grid(alpha=0.2)
    return fig


def render_final_alias(df: pd.DataFrame, task: int, alias: str):
    key = alias.lower()
    if "rate_vs_time" in key or "events_per_second" in key:
        return _rate_plot(df, task, alias)
    if "theta_phi" in key or "polar_theta_phi" in key:
        return _theta_phi_plot(df, alias)
    if "event_display" in key:
        return _event_display(df, alias)
    if "projection" in key and ("ellipse" in key or "angle" in key):
        return _projection_plot(df, alias)
    if "efficiency" in key:
        return _efficiency_plot(df, task, alias)
    if "occupancy" in key or "multiplicity" in key:
        return _occupancy_plot(df, alias)
    if any(word in key for word in ("retention", "threshold", "weakest")):
        return _threshold_plot(df, alias)
    if "topology" in key:
        return _topology_plot(df, alias)
    if "hexbin" in key or "pairgrid" in key:
        return _pair_matrix(df, task, alias, "hexbin")
    if "scatter" in key:
        return _pair_matrix(df, task, alias, "scatter")
    if any(word in key for word in ("matrix", "contagion", "activation", "transition")):
        return _matrix_plot(df, task, alias)
    if any(word in key for word in ("charge", "streamer")):
        return _charge_plot(df, task, alias)
    if any(word in key for word in ("residual", "gaussian", "histogram", "distribution", "chi2", "offender", "zigzag", "kinematics")):
        residual_columns = [str(column) for column in df.columns if "res" in str(column).lower()]
        if "residual" in key and residual_columns:
            return _hist_grid(df, task, alias, residual_columns[:6])
        return _hist_grid(df, task, alias)
    if "tt" in key or "trigger_types" in key:
        return _tt_plot(df, task, alias)
    return _hist_grid(df, task, alias)


def _enabled_aliases(config: dict) -> list[str]:
    mode = str(config.get("plot_mode", "all")).strip().lower()
    statuses = config.get("plots", {}) or {}
    enabled: list[str] = []
    for alias, raw_status in statuses.items():
        status = str(raw_status).strip().lower()
        if status in {"none", "off", "false", "0"}:
            continue
        if mode == "all":
            enabled.append(str(alias))
        elif mode == "usual" and status in {"usual", "essential"}:
            enabled.append(str(alias))
        elif mode == "essential" and status == "essential":
            enabled.append(str(alias))
        elif mode == "debug" and status == "debug":
            enabled.append(str(alias))
    return enabled


def run_final_plotter(
    *,
    task: int,
    expected_prefixes: tuple[str, ...],
    parquet_path: str | Path,
    config_path: str | Path,
) -> None:
    parquet = Path(parquet_path).expanduser().resolve()
    if not any(parquet.name.startswith(prefix) for prefix in expected_prefixes):
        expected = " or ".join(f"{prefix}*.parquet" for prefix in expected_prefixes)
        raise SystemExit(f"Task {task} final plotter expects {expected}; got {parquet.name!r}.")

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    aliases = _enabled_aliases(config)
    dataframe = pd.read_parquet(parquet)

    output_directory = parquet.parent / "FINAL_PLOTS" / f"TASK_{task}" / parquet.stem
    output_directory.mkdir(parents=True, exist_ok=True)
    save_png = bool(config.get("save_png", True))
    save_pdf = bool(config.get("save_pdf", True))
    pdf_path = output_directory / f"{parquet.stem}_task_{task}_final_plots.pdf"
    pdf = PdfPages(pdf_path) if save_pdf else None
    generated = 0

    try:
        for index, alias in enumerate(aliases, start=1):
            try:
                figure = render_final_alias(dataframe, task, alias)
            except Exception as exc:
                warnings.warn(f"FINAL plot {alias!r} failed and was skipped: {exc}")
                continue
            if figure is None:
                warnings.warn(f"FINAL plot {alias!r} skipped: the derived parquet does not expose the quantities it needs.")
                continue
            figure.suptitle(f"{parquet.name}\n{alias.replace('_', ' ')}", fontsize=11)
            if save_png:
                figure.savefig(output_directory / f"{index:03d}_{alias}.png", dpi=150, bbox_inches="tight")
            if pdf is not None:
                pdf.savefig(figure, dpi=150, bbox_inches="tight")
            plt.close(figure)
            generated += 1
    finally:
        if pdf is not None:
            pdf.close()
            if generated == 0 and pdf_path.exists():
                pdf_path.unlink()

    print(f"Task {task} FINAL plotting complete: {generated}/{len(aliases)} configured plots generated.")
    print(f"Output directory: {output_directory}")
'''


FINAL_WRAPPER_TEMPLATE = '''#!/usr/bin/env python3
"""Standalone Task {task} FINAL plotter.

Usage:
    python final_plots_task_{task}.py /path/to/{prefix_hint}*.parquet

The parquet path is the only runtime argument. This script is outside the
Stage-1 processing flow: it reads an already-produced Task {task} parquet and
creates the configured FINAL PNG/PDF products.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT = Path(__file__).resolve()
SCRIPTS_ROOT = next(parent for parent in CURRENT.parents if parent.name == "MINGO_ANALYSIS_SCRIPTS")
REPO_ROOT = SCRIPTS_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MINGO_ANALYSIS.MINGO_ANALYSIS_SCRIPTS.common.final_plotting import run_final_plotter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task {task} FINAL plots from its derived parquet.")
    parser.add_argument("parquet_path", help="Path to the Task {task} derived parquet")
    args = parser.parse_args()
    config_path = (
        SCRIPTS_ROOT
        / "CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/TASK_{task}"
        / "config_final_plots_task_{task}.yaml"
    )
    run_final_plotter(
        task={task},
        expected_prefixes={prefixes!r},
        parquet_path=args.parquet_path,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
'''


TEST_SOURCE = '''from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS"
CFG = SCRIPTS / "CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1"
STEP = SCRIPTS / "STAGES/STAGE_1/EVENT_DATA/STEP_1"


def aliases(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^  ([A-Za-z0-9_]+):", line)
        if match:
            result.add(match.group(1))
    return result


class RuntimeFinalPlotSplitTests(unittest.TestCase):
    def test_runtime_and_final_catalogs_are_disjoint(self):
        for task in range(6):
            runtime = aliases(CFG / f"TASK_{task}/config_plots_task_{task}.yaml")
            final = aliases(CFG / f"TASK_{task}/config_final_plots_task_{task}.yaml")
            self.assertFalse(runtime & final, (task, sorted(runtime & final)))
            self.assertTrue(final, f"Task {task} FINAL catalog is empty")

    def test_each_task_has_standalone_final_plotter_with_one_cli_argument(self):
        for task in range(6):
            path = STEP / f"TASK_{task}/final_plots_task_{task}.py"
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            self.assertEqual(source.count("parser.add_argument("), 1, task)
            self.assertIn('parser.add_argument("parquet_path"', source)
            self.assertIn(f"config_final_plots_task_{task}.yaml", source)

    def test_runtime_pdf_creation_is_disabled(self):
        shared = (SCRIPTS / "common/step1_shared.py").read_text(encoding="utf-8")
        self.assertIn("create_pdf = False  # Runtime STEP_1 plots are strictly PNG-only.", shared)
        for task, script_name in {
            0: "script_0_acq_to_raw.py",
            1: "script_1_raw_to_clean.py",
            2: "script_2_clean_to_cal.py",
            3: "script_3_cal_to_list.py",
            4: "script_4_list_to_fit.py",
            5: "script_5_fit_to_post.py",
        }.items():
            source = (STEP / f"TASK_{task}/{script_name}").read_text(encoding="utf-8")
            self.assertNotIn("create_pdf = save_plots", source, task)

    def test_final_plotting_module_does_not_import_task_runtime_scripts(self):
        source = (SCRIPTS / "common/final_plotting.py").read_text(encoding="utf-8")
        self.assertNotIn("script_0_acq_to_raw", source)
        self.assertNotIn("script_1_raw_to_clean", source)
        self.assertNotIn("script_2_clean_to_cal", source)
        self.assertNotIn("script_3_cal_to_list", source)
        self.assertNotIn("script_4_list_to_fit", source)
        self.assertNotIn("script_5_fit_to_post", source)


if __name__ == "__main__":
    unittest.main()
'''


def create_final_plotting_files() -> None:
    (COMMON / "final_plotting.py").write_text(FINAL_PLOTTING_SOURCE, encoding="utf-8")
    for task in range(6):
        prefixes = EXPECTED_PREFIXES[task]
        wrapper = FINAL_WRAPPER_TEMPLATE.format(
            task=task,
            prefix_hint=prefixes[0],
            prefixes=prefixes,
        )
        (STEP / f"TASK_{task}/final_plots_task_{task}.py").write_text(wrapper, encoding="utf-8")
    (COMMON / "tests/test_step1_runtime_final_plot_split.py").write_text(TEST_SOURCE, encoding="utf-8")


def main() -> None:
    for task in range(6):
        split_catalog(task)
    patch_shared_plot_contract()
    for task in range(6):
        extract_final_plot_blocks(task)
    create_final_plotting_files()


if __name__ == "__main__":
    main()
