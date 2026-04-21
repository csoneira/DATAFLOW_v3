"""Shared engine for simple time-series QUALITY_ASSURANCE tasks."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .column_rule_table import (
    ColumnRule,
    load_column_rule_table,
    matches_any_pattern,
    resolve_column_rule,
    rule_to_threshold_mapping,
    split_default_rule,
)
from .epoch_quality import build_epoch_reference_table, build_scalar_value_frame, evaluate_scalar_frame
from .epochs import load_online_run_dictionary
from .task_setup import (
    apply_date_filter,
    bootstrap_task,
    deep_merge_dicts,
    ensure_task_station_tree,
    load_step_configs,
    load_task_configs,
    load_yaml_mapping,
    normalize_station_name,
    resolve_filter_timestamp,
    resolve_plot_x_axis,
    x_axis_config,
    get_station_date_range,
    parse_timestamp_series,
)
from .thresholds import ThresholdRule

DEFAULT_SKIP_COLUMNS = {"filename_base", "execution_timestamp", "param_hash"}


def _metadata_path(repo_root: Path, station_name: str, task_id: int, metadata_csv_filename: str) -> Path:
    return (
        repo_root
        / "STATIONS"
        / station_name
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
        / metadata_csv_filename
    )


def _outputs_root(task_dir: Path, station_name: str) -> Path:
    return task_dir / "STATIONS" / station_name / "OUTPUTS"


def _output_files_dir(task_dir: Path, station_name: str) -> Path:
    return _outputs_root(task_dir, station_name) / "FILES"


def _output_plots_dir(task_dir: Path, station_name: str) -> Path:
    return _outputs_root(task_dir, station_name) / "PLOTS"


def _build_pass_dataframe(meta_df: pd.DataFrame, pass_column: str, default_pass: float) -> pd.DataFrame:
    if "filename_base" not in meta_df.columns:
        return pd.DataFrame(columns=["filename_base", pass_column])
    out = pd.DataFrame()
    out["filename_base"] = meta_df["filename_base"].astype("string").fillna("").str.strip()
    out = out[out["filename_base"] != ""].drop_duplicates().reset_index(drop=True)
    out[pass_column] = float(default_pass)
    return out


def _load_metadata_frame(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame(columns=["filename_base"])


def _deduplicate_metadata_rows(meta_df: pd.DataFrame) -> pd.DataFrame:
    if meta_df.empty or "filename_base" not in meta_df.columns:
        return pd.DataFrame(columns=["filename_base"])

    df = meta_df.copy()
    df["filename_base"] = df["filename_base"].astype("string").fillna("").str.strip()
    df = df[df["filename_base"] != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=["filename_base"])

    if "execution_timestamp" in df.columns:
        parsed_exec = parse_timestamp_series(df["execution_timestamp"])
        df["_exec_ts"] = parsed_exec
        df = df.sort_values(["filename_base", "_exec_ts"], na_position="last")
        df = df.drop_duplicates(subset=["filename_base"], keep="last")
        df = df.drop(columns=["_exec_ts"])
    else:
        df = df.drop_duplicates(subset=["filename_base"], keep="last")

    return df.reset_index(drop=True)


def _column_rule_csv_path(base_dir: Path, config: dict[str, Any]) -> Path | None:
    raw_path = str(config.get("column_rule_table_csv", "")).strip()
    if not raw_path:
        return None
    csv_path = Path(raw_path)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    return csv_path


def _column_rule_table(base_dir: Path, config: dict[str, Any]) -> list[ColumnRule]:
    cache_key = "__column_rule_table__"
    cached = config.get(cache_key)
    if isinstance(cached, list):
        return cached
    csv_path = _column_rule_csv_path(base_dir, config)
    rules = load_column_rule_table(csv_path) if csv_path is not None and csv_path.exists() else []
    config[cache_key] = rules
    return rules


def _plot_ignore_patterns(config: dict[str, Any]) -> list[str]:
    patterns: list[str] = []
    for key in ("plot_ignore_patterns_common", "plot_ignore_patterns_extra"):
        values = config.get(key)
        if not isinstance(values, list):
            continue
        patterns.extend(str(value).strip() for value in values if str(value).strip())
    return _unique_preserve(patterns)


def _should_ignore_column(column_name: str, *, skip_columns: set[str], config: dict[str, Any]) -> bool:
    if column_name in skip_columns:
        return True
    return matches_any_pattern(column_name, _plot_ignore_patterns(config))


def _is_quality_candidate_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return True
    string_values = series.astype("string").fillna("").str.strip()
    return string_values.str.startswith("[").any()


def _resolve_effective_column_rule(
    base_dir: Path,
    config: dict[str, Any],
    column_name: str,
    *,
    source_column: str | None = None,
) -> ColumnRule | None:
    rules = _column_rule_table(base_dir, config)
    if not rules:
        return None

    direct_rule = resolve_column_rule(column_name, rules)
    if source_column is None or source_column == column_name:
        return direct_rule

    source_rule = resolve_column_rule(source_column, rules)
    if direct_rule is None:
        return source_rule
    if source_rule is None:
        return direct_rule
    if direct_rule.pattern == "*" and source_rule.pattern != "*":
        return source_rule
    return direct_rule


def _quality_columns_from_config(
    *,
    base_dir: Path,
    config: dict[str, Any],
    df: pd.DataFrame,
    skip_columns: set[str],
    groups: list[dict[str, Any]],
) -> list[str]:
    available_columns = list(df.columns)
    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    explicit_columns = quality_cfg.get("columns")
    if isinstance(explicit_columns, list):
        patterns = [str(item).strip() for item in explicit_columns if str(item).strip()]
        resolved = _resolve_columns(available_columns, patterns)
        return _unique_preserve(
            [
                col
                for col in resolved
                if col in available_columns
                and not _should_ignore_column(col, skip_columns=skip_columns, config=config)
                and _is_quality_candidate_series(df[col])
            ]
        )

    rules = _column_rule_table(base_dir, config)
    if rules:
        out: list[str] = []
        for column_name in available_columns:
            if _should_ignore_column(column_name, skip_columns=skip_columns, config=config):
                continue
            resolved_rule = _resolve_effective_column_rule(base_dir, config, column_name)
            if resolved_rule is None or not resolved_rule.quality_enabled:
                continue
            if not _is_quality_candidate_series(df[column_name]):
                continue
            out.append(column_name)
        return _unique_preserve(out)

    ordered: list[str] = []
    for group in groups:
        if group.get("mode") == "overlay":
            ordered.extend(_resolve_columns(available_columns, group.get("overlay_columns", [])))
        else:
            ordered.extend(_resolve_columns(available_columns, group.get("columns", [])))
    ordered = _unique_preserve(ordered)
    if ordered:
        return [
            col
            for col in ordered
            if not _should_ignore_column(col, skip_columns=skip_columns, config=config)
            and _is_quality_candidate_series(df[col])
        ]
    return [
        col
        for col in available_columns
        if not _should_ignore_column(col, skip_columns=skip_columns, config=config)
        and _is_quality_candidate_series(df[col])
    ]


def _quality_threshold_config_for_specs(
    *,
    base_dir: Path,
    config: dict[str, Any],
    specs_df: pd.DataFrame,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None]:
    rules = _column_rule_table(base_dir, config)
    if rules:
        default_rule, _ = split_default_rule(rules)
        defaults = rule_to_threshold_mapping(default_rule) if default_rule is not None else None
        column_rules: dict[str, dict[str, Any]] = {}
        for spec in specs_df.to_dict("records"):
            evaluation_column = str(spec["evaluation_column"])
            source_column = str(spec["source_column"])
            resolved_rule = _resolve_effective_column_rule(
                base_dir,
                config,
                evaluation_column,
                source_column=source_column,
            )
            if resolved_rule is None:
                continue
            column_rules[evaluation_column] = rule_to_threshold_mapping(resolved_rule)
        return defaults, column_rules or None

    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    rules_cfg = quality_cfg.get("rules") if isinstance(quality_cfg.get("rules"), dict) else {}
    return rules_cfg.get("defaults"), rules_cfg.get("column_rules")


def _attach_epoch_metadata(repo_root: Path, station_name: str, analyzed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if analyzed_df.empty or station_name == "MINGO00":
        df = analyzed_df.copy()
        df["epoch_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
        return df, pd.DataFrame()

    try:
        epochs_df = load_online_run_dictionary(repo_root, int(station_name.removeprefix("MINGO")))
    except (FileNotFoundError, ValueError):
        df = analyzed_df.copy()
        df["epoch_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
        return df, pd.DataFrame()

    df = analyzed_df.copy()
    epoch_records: list[str | None] = []
    for timestamp in df["__timestamp__"]:
        match = epochs_df[
            epochs_df["start_timestamp"].notna()
            & (epochs_df["start_timestamp"] <= timestamp)
            & (epochs_df["end_timestamp"].isna() | (timestamp <= epochs_df["end_timestamp"]))
        ]
        epoch_records.append(None if match.empty else str(match.iloc[-1]["epoch_id"]))
    df["epoch_id"] = pd.array(epoch_records, dtype="string")
    return df, epochs_df


def _enrich_reference_table(reference_df: pd.DataFrame, epochs_df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    if reference_df.empty:
        return reference_df

    epoch_columns = [
        column
        for column in (
            "epoch_id",
            "conf_number",
            "start_date",
            "end_date",
            "location",
            "comment",
            "boundary_overlap",
        )
        if column in epochs_df.columns
    ]
    epoch_lookup = epochs_df[epoch_columns].drop_duplicates(subset=["epoch_id"])
    out = reference_df.merge(epoch_lookup, on="epoch_id", how="left")
    out.insert(0, "station_name", station_name)
    return out


def _prepare_scope_dataframe(station_name: str, config: dict[str, Any], meta_df: pd.DataFrame) -> pd.DataFrame:
    df = _deduplicate_metadata_rows(meta_df)
    if df.empty:
        return df

    try:
        timestamps, timestamp_source = resolve_filter_timestamp(df, config)
        df["__timestamp__"] = timestamps
        df["qa_timestamp_source"] = timestamp_source
    except (KeyError, ValueError):
        df["__timestamp__"] = pd.NaT
        df["qa_timestamp_source"] = ""

    date_range = get_station_date_range(config=config, station=station_name)
    if date_range is None:
        df["qa_in_scope"] = df["__timestamp__"].notna()
    else:
        filtered = apply_date_filter(df.loc[df["__timestamp__"].notna()].copy(), date_range)
        df["qa_in_scope"] = df.index.isin(filtered.index)
    return df


def _prepare_analysis_dataframe(station_name: str, config: dict[str, Any], meta_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if meta_df.empty:
        return pd.DataFrame(), "empty_metadata"

    scope_df = _prepare_scope_dataframe(station_name=station_name, config=config, meta_df=meta_df)
    if scope_df.empty:
        return pd.DataFrame(), "empty_metadata"

    df = scope_df.loc[scope_df["qa_in_scope"]].copy()
    if df.empty:
        date_range = get_station_date_range(config=config, station=station_name)
        return pd.DataFrame(), f"all rows filtered out by date_range={date_range}"

    x_values, x_label, _ = resolve_plot_x_axis(df, config)
    df["__plot_x__"] = x_values
    df = df.loc[df["__plot_x__"].notna()].copy()
    if df.empty:
        return pd.DataFrame(), "no rows with valid x-axis values"

    sort_key = "__plot_x__" if x_label != "__timestamp__" else "__timestamp__"
    df = df.sort_values(sort_key).reset_index(drop=True)
    return df, None

def _unique_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_columns(available_columns: list[str], patterns: list[str]) -> list[str]:
    resolved: list[str] = []
    for pattern in patterns:
        pat = str(pattern).strip()
        if not pat:
            continue
        if any(token in pat for token in ("*", "?", "[", "]")):
            resolved.extend([col for col in available_columns if fnmatch(col, pat)])
        elif pat in available_columns:
            resolved.append(pat)
    return _unique_preserve(resolved)


def _split_single_star_pattern(pattern: str) -> tuple[str, str] | None:
    if pattern.count("*") != 1:
        return None
    prefix, suffix = pattern.split("*", 1)
    return prefix, suffix


def _resolve_overlay_panels(available_columns: list[str], overlay_patterns: list[str]) -> list[dict[str, Any]]:
    series_maps: list[dict[str, str]] = []
    for pattern in overlay_patterns:
        pat = str(pattern).strip()
        if not pat:
            series_maps.append({})
            continue

        star = _split_single_star_pattern(pat)
        if star is None:
            series_maps.append({pat: pat} if pat in available_columns else {})
            continue

        prefix, suffix = star
        mapping: dict[str, str] = {}
        for col in available_columns:
            if not col.startswith(prefix):
                continue
            if suffix and not col.endswith(suffix):
                continue
            middle = col[len(prefix) :]
            if suffix:
                middle = middle[: -len(suffix)]
            mapping[middle] = col
        series_maps.append(mapping)

    panel_keys = sorted({key for mapping in series_maps for key in mapping})
    return [{"title": key, "columns": [mapping.get(key) for mapping in series_maps]} for key in panel_keys]


def _column_default_plot_groups(config: dict[str, Any], skip_columns: set[str]) -> list[dict[str, Any]]:
    patterns = config.get("column_patterns")
    if not isinstance(patterns, list):
        patterns = []
    filtered = [
        str(p).strip()
        for p in patterns
        if str(p).strip() and str(p).strip() not in skip_columns
    ]
    if not filtered:
        filtered = ["*"]
    return [{"name": "default", "mode": "columns", "columns": filtered}]


def _plot_groups_from_config(
    config: dict[str, Any],
    *,
    default_plot_groups: list[dict[str, Any]] | None,
    skip_columns: set[str],
) -> list[dict[str, Any]]:
    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    groups_cfg = plot_cfg.get("plot_groups")
    if not isinstance(groups_cfg, list) or not groups_cfg:
        if default_plot_groups:
            return default_plot_groups
        return _column_default_plot_groups(config, skip_columns)

    groups: list[dict[str, Any]] = []
    for idx, item in enumerate(groups_cfg, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"group_{idx}")).strip() or f"group_{idx}"
        group: dict[str, Any] = {"name": name}
        if "ncols" in item:
            try:
                group["ncols"] = int(item.get("ncols"))
            except (TypeError, ValueError):
                pass

        overlay_columns = item.get("overlay_columns")
        if isinstance(overlay_columns, list):
            overlay_patterns = [str(c).strip() for c in overlay_columns if str(c).strip()]
            if overlay_patterns:
                labels_cfg = item.get("overlay_labels")
                labels = (
                    [str(lbl).strip() for lbl in labels_cfg if str(lbl).strip()]
                    if isinstance(labels_cfg, list)
                    else []
                )
                if len(labels) < len(overlay_patterns):
                    labels.extend(overlay_patterns[len(labels) :])
                group.update(
                    {
                        "mode": "overlay",
                        "overlay_columns": overlay_patterns,
                        "overlay_labels": labels,
                    }
                )
                groups.append(group)
                continue

        columns = item.get("columns")
        if isinstance(columns, list):
            patterns = [str(c).strip() for c in columns if str(c).strip()]
            if patterns:
                group.update({"mode": "columns", "columns": patterns})
                groups.append(group)

    if groups:
        return groups
    if default_plot_groups:
        return default_plot_groups
    return _column_default_plot_groups(config, skip_columns)


def _parse_migration_col(col: str) -> tuple[str, str, str, str] | None:
    if "_to_" not in col or "_tt_" not in col:
        return None
    try:
        prefix, tail = col.split("_tt_", 1)
    except ValueError:
        return None
    if "_to_" not in prefix:
        return None
    metric_parts = tail.split("_")
    if len(metric_parts) < 2:
        return None
    from_tt = metric_parts[0]
    to_tt = metric_parts[1]
    metric = "_".join(metric_parts[2:]) if len(metric_parts) > 2 else ""
    return prefix, from_tt, to_tt, metric


def _auto_grid(n_panels: int, ncols_hint: int | None) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    if ncols_hint is not None and ncols_hint > 0:
        ncols = min(n_panels, ncols_hint)
        nrows = int(math.ceil(n_panels / ncols))
        return nrows, ncols
    ncols = int(math.ceil(math.sqrt(n_panels)))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _plot_group(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    columns: list[str],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    task_id: int,
    metadata_type: str,
    skip_columns: set[str],
    allow_migration_matrix: bool,
    group_ncols: int | None = None,
) -> Path | None:
    plot_columns = [
        col
        for col in columns
        if col in df.columns and not _should_ignore_column(col, skip_columns=skip_columns, config=config)
    ]
    if not plot_columns:
        return None

    if allow_migration_matrix:
        migration_candidates = [_parse_migration_col(c) for c in plot_columns]
        if any(migration_candidates):
            try:
                return _plot_migration_matrix(
                    df=df,
                    x_values=x_values,
                    x_label=x_label,
                    x_is_datetime=x_is_datetime,
                    columns=plot_columns,
                    station_name=station_name,
                    group_name=group_name,
                    out_dir=out_dir,
                    config=config,
                    task_id=task_id,
                    metadata_type=metadata_type,
                )
            except Exception:
                pass

    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    ncols_hint = group_ncols if group_ncols is not None else plot_cfg.get("ncols")
    try:
        ncols_hint_int = int(ncols_hint) if ncols_hint is not None else None
    except (TypeError, ValueError):
        ncols_hint_int = None

    nrows, ncols = _auto_grid(len(plot_columns), ncols_hint_int)
    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))
    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    fig_width = float(plot_cfg.get("figsize_per_col", 4.2)) * ncols
    fig_height = float(plot_cfg.get("figsize_per_row", 2.8)) * max(nrows, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True)
    axes_flat = np.ravel(axes)

    for idx, col in enumerate(plot_columns):
        ax = axes_flat[idx]
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().any():
            ax.plot(x_values, y, linestyle="-", linewidth=line_width, marker="o", markersize=marker_size)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", fontsize=9)
        ax.set_title(col, fontsize=8)
        ax.grid(True, alpha=0.25)

    for idx in range(len(plot_columns), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"TASK_{task_id} {station_name} {metadata_type} - {group_name} (x={x_label})", fontsize=11)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_name}_task_{task_id}_{metadata_type}_{group_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_overlay_group(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    panels: list[dict[str, Any]],
    series_labels: list[str],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    task_id: int,
    metadata_type: str,
    group_ncols: int | None = None,
) -> Path | None:
    if not panels:
        return None

    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    ncols_hint = group_ncols if group_ncols is not None else plot_cfg.get("ncols")
    try:
        ncols_hint_int = int(ncols_hint) if ncols_hint is not None else None
    except (TypeError, ValueError):
        ncols_hint_int = None

    nrows, ncols = _auto_grid(len(panels), ncols_hint_int)
    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))
    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    fig_width = float(plot_cfg.get("figsize_per_col", 4.2)) * ncols
    fig_height = float(plot_cfg.get("figsize_per_row", 2.8)) * max(nrows, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), sharex=True)
    axes_flat = np.ravel(axes)

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        title = str(panel.get("title", f"panel_{idx + 1}"))
        columns = panel.get("columns", [])

        has_data = False
        missing_labels: list[str] = []
        for series_idx, col in enumerate(columns):
            label = series_labels[series_idx] if series_idx < len(series_labels) else f"series_{series_idx + 1}"
            if not col or col not in df.columns:
                missing_labels.append(label)
                continue

            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().any():
                ax.plot(
                    x_values,
                    y,
                    linestyle="-",
                    linewidth=line_width,
                    marker="o",
                    markersize=marker_size,
                    label=label,
                )
                has_data = True
            else:
                ax.plot([], [], linestyle="-", linewidth=line_width, marker="o", markersize=marker_size, label=f"{label} (no data)")

        if not has_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", fontsize=9)
        if missing_labels:
            ax.text(
                0.5,
                0.08,
                f"missing: {', '.join(missing_labels)}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
            )
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    for idx in range(len(panels), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"TASK_{task_id} {station_name} {metadata_type} - {group_name} (x={x_label})", fontsize=11)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_name}_task_{task_id}_{metadata_type}_{group_name}.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_migration_matrix(
    *,
    df: pd.DataFrame,
    x_values: pd.Series,
    x_label: str,
    x_is_datetime: bool,
    columns: list[str],
    station_name: str,
    group_name: str,
    out_dir: Path,
    config: dict[str, Any],
    task_id: int,
    metadata_type: str,
) -> Path | None:
    entries = [(_parse_migration_col(c), c) for c in columns]
    entries = [(p, c) for p, c in entries if p is not None]
    if not entries:
        return None

    raw_types = sorted({p[1] for p, _ in entries} | {p[2] for p, _ in entries})
    types = [t for t in raw_types if len(str(t)) > 1 and str(t).lower() != "rate"]
    if not types:
        return None

    def _plane_sort_key(t: str) -> tuple[int, int]:
        digits = "".join(ch for ch in str(t) if ch.isdigit())
        try:
            num = int(digits) if digits else 0
        except Exception:
            num = 0
        return (-len(str(t)), num)

    types = sorted(types, key=_plane_sort_key)
    n = len(types)
    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    dpi = int(plot_cfg.get("dpi", 150))
    marker_size = float(plot_cfg.get("marker_size", 2.0))
    line_width = float(plot_cfg.get("line_width", 0.9))
    image_format = str(plot_cfg.get("format", "png")).strip().lower()
    fig_width = float(plot_cfg.get("figsize_per_col", 3.0)) * n
    fig_height = float(plot_cfg.get("figsize_per_row", 2.6)) * n

    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    lookup: dict[tuple[str, str], str] = {}
    for (_, from_tt, to_tt, _), col in entries:
        lookup[(from_tt, to_tt)] = col

    y_mins: list[float] = []
    y_maxs: list[float] = []
    for col in lookup.values():
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().any():
            y_mins.append(float(y.min()))
            y_maxs.append(float(y.max()))
    y_min = min(y_mins) if y_mins else None
    y_max = max(y_maxs) if y_maxs else None

    for i, from_tt in enumerate(types):
        for j, to_tt in enumerate(types):
            ax = axes[i, j] if n > 1 else axes
            if i > j:
                ax.axis("off")
                continue

            col = lookup.get((from_tt, to_tt))
            if col and col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().any():
                    ax.plot(x_values, y, linestyle="-", linewidth=line_width, marker="o", markersize=marker_size)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", fontsize=8)
            else:
                ax.text(0.5, 0.5, "-", ha="center", va="center", color="lightgray", fontsize=10)

            if from_tt == to_tt:
                ax.set_facecolor("#f3f3f3")
                ax.text(0.02, 0.92, "no change", transform=ax.transAxes, fontsize=7, va="top")
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            if j == 0:
                ax.set_ylabel(from_tt)
            if i == 0:
                ax.set_title(to_tt, fontsize=8)
            ax.grid(True, alpha=0.25)

    fig.suptitle(f"TASK_{task_id} {station_name} {metadata_type} - {group_name} (x={x_label})", fontsize=11)
    if x_is_datetime:
        fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{station_name}_task_{task_id}_{metadata_type}_{group_name}_migration_matrix.{image_format}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _write_epoch_reference_table(
    *,
    task_dir: Path,
    base_dir: Path,
    station_name: str,
    metadata_type: str,
    analyzed_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    config: dict[str, Any],
    groups: list[dict[str, Any]],
    skip_columns: set[str],
) -> tuple[pd.DataFrame, Path | None]:
    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    if not quality_cfg.get("enabled", False):
        return pd.DataFrame(), None
    if analyzed_df.empty or epochs_df.empty:
        return pd.DataFrame(), None

    quality_columns = _quality_columns_from_config(
        base_dir=base_dir,
        config=config,
        df=analyzed_df,
        skip_columns=skip_columns,
        groups=groups,
    )
    value_frame, specs_df = build_scalar_value_frame(analyzed_df, quality_columns)
    reference_df = build_epoch_reference_table(value_frame, specs_df, analyzed_df["epoch_id"])
    if reference_df.empty:
        return pd.DataFrame(), None

    reference_df = _enrich_reference_table(reference_df, epochs_df, station_name)
    output_filename = str(
        quality_cfg.get("output_reference_filename", f"{metadata_type}_epoch_references.csv")
    ).strip() or f"{metadata_type}_epoch_references.csv"
    out_path = _output_files_dir(task_dir, station_name) / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(out_path, index=False)
    return reference_df, out_path


def _build_quality_pass_dataframe(
    *,
    task_dir: Path,
    base_dir: Path,
    station_name: str,
    metadata_type: str,
    scope_df: pd.DataFrame,
    analyzed_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    config: dict[str, Any],
    groups: list[dict[str, Any]],
    skip_columns: set[str],
    pass_column: str,
) -> tuple[pd.DataFrame, Path | None]:
    quality_cfg = config.get("quality") if isinstance(config.get("quality"), dict) else {}
    if not quality_cfg.get("enabled", False):
        default_pass = float(config.get("pass_default_value", 1.0))
        return _build_pass_dataframe(scope_df, pass_column, default_pass), None

    if scope_df.empty:
        return pd.DataFrame(columns=["filename_base", pass_column]), None

    file_df = scope_df[["filename_base", "__timestamp__", "qa_timestamp_source", "qa_in_scope"]].copy()
    file_df.rename(columns={"__timestamp__": "qa_timestamp"}, inplace=True)
    file_df["epoch_id"] = pd.Series(pd.NA, index=file_df.index, dtype="string")
    file_df["conf_number"] = pd.Series(pd.NA, index=file_df.index, dtype="Int64")

    if not analyzed_df.empty and "epoch_id" in analyzed_df.columns:
        file_epoch_df = analyzed_df[["filename_base", "epoch_id"]].drop_duplicates(subset=["filename_base"])
        if not reference_df.empty and {"epoch_id", "conf_number"} <= set(reference_df.columns):
            epoch_lookup = reference_df[["epoch_id", "conf_number"]].drop_duplicates(subset=["epoch_id"])
            file_epoch_df = file_epoch_df.merge(epoch_lookup, on="epoch_id", how="left")
        else:
            file_epoch_df["conf_number"] = pd.Series(pd.NA, index=file_epoch_df.index, dtype="Int64")
        file_df = file_df.merge(file_epoch_df, on="filename_base", how="left", suffixes=("", "_eval"))
        file_df["epoch_id"] = file_df["epoch_id_eval"].combine_first(file_df["epoch_id"])
        merged_conf_number = pd.to_numeric(file_df["conf_number_eval"], errors="coerce").astype("Float64")
        base_conf_number = pd.to_numeric(file_df["conf_number"], errors="coerce").astype("Float64")
        file_df["conf_number"] = merged_conf_number.fillna(base_conf_number).astype("Int64")
        file_df = file_df.drop(columns=["epoch_id_eval", "conf_number_eval"])

    column_eval_df = pd.DataFrame()
    if not analyzed_df.empty and not reference_df.empty:
        quality_columns = _quality_columns_from_config(
            base_dir=base_dir,
            config=config,
            df=analyzed_df,
            skip_columns=skip_columns,
            groups=groups,
        )
        value_frame, specs_df = build_scalar_value_frame(analyzed_df, quality_columns)
        defaults, column_rules = _quality_threshold_config_for_specs(
            base_dir=base_dir,
            config=config,
            specs_df=specs_df,
        )
        column_eval_df = evaluate_scalar_frame(
            analyzed_df,
            value_frame,
            reference_df,
            defaults=defaults,
            column_rules=column_rules,
        )

    if not column_eval_df.empty:
        grouped = column_eval_df.groupby("filename_base", dropna=False)
        agg_df = grouped.agg(
            qa_evaluated_columns=("status", lambda values: int(sum(item in {"pass", "fail"} for item in values))),
            qa_passed_columns=("status", lambda values: int(sum(item == "pass" for item in values))),
            qa_failed_columns=("status", lambda values: int(sum(item == "fail" for item in values))),
            qa_warning_columns=("status", lambda values: int(sum(item not in {"pass", "fail"} for item in values))),
        ).reset_index()
        failed_observables = (
            column_eval_df.loc[column_eval_df["status"] == "fail", ["filename_base", "evaluation_column"]]
            .groupby("filename_base")["evaluation_column"]
            .apply(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("qa_failed_observables")
        )
        warning_reasons = (
            column_eval_df.loc[column_eval_df["status"] != "pass", ["filename_base", "reason"]]
            .groupby("filename_base")["reason"]
            .apply(lambda values: ";".join(sorted(values.astype(str).unique())))
            .rename("qa_warning_reasons")
        )
        agg_df = agg_df.merge(failed_observables.reset_index(), on="filename_base", how="left")
        agg_df = agg_df.merge(warning_reasons.reset_index(), on="filename_base", how="left")
        file_df = file_df.merge(agg_df, on="filename_base", how="left")
    else:
        file_df["qa_evaluated_columns"] = 0
        file_df["qa_passed_columns"] = 0
        file_df["qa_failed_columns"] = 0
        file_df["qa_warning_columns"] = 0
        file_df["qa_failed_observables"] = ""
        file_df["qa_warning_reasons"] = ""

    for column_name in (
        "qa_evaluated_columns",
        "qa_passed_columns",
        "qa_failed_columns",
        "qa_warning_columns",
    ):
        file_df[column_name] = pd.to_numeric(file_df[column_name], errors="coerce").fillna(0).astype(int)
    for column_name in ("qa_failed_observables", "qa_warning_reasons"):
        file_df[column_name] = file_df[column_name].fillna("")

    file_df["qa_status"] = "not_evaluated"
    invalid_mask = file_df["qa_timestamp"].isna()
    out_of_scope_mask = (~invalid_mask) & (~file_df["qa_in_scope"].fillna(False))
    no_epoch_mask = file_df["qa_in_scope"].fillna(False) & file_df["epoch_id"].isna()
    fail_mask = file_df["qa_failed_columns"] > 0
    warn_mask = (~fail_mask) & (file_df["qa_warning_columns"] > 0)
    pass_mask = (
        file_df["qa_in_scope"].fillna(False)
        & file_df["epoch_id"].notna()
        & (file_df["qa_evaluated_columns"] > 0)
        & (file_df["qa_failed_columns"] == 0)
        & (file_df["qa_warning_columns"] == 0)
    )

    file_df.loc[invalid_mask, "qa_status"] = "invalid_timestamp"
    file_df.loc[out_of_scope_mask, "qa_status"] = "out_of_scope"
    file_df.loc[no_epoch_mask, "qa_status"] = "no_epoch_match"
    file_df.loc[warn_mask, "qa_status"] = "warn"
    file_df.loc[fail_mask, "qa_status"] = "fail"
    file_df.loc[pass_mask, "qa_status"] = "pass"
    file_df[pass_column] = file_df["qa_status"].eq("pass").astype(float)
    file_df["qa_pass_fraction"] = np.where(
        file_df["qa_evaluated_columns"] > 0,
        file_df["qa_passed_columns"] / file_df["qa_evaluated_columns"],
        np.nan,
    )

    output_filename = str(
        quality_cfg.get("output_column_evaluation_filename", f"{metadata_type}_column_evaluations.csv")
    ).strip() or f"{metadata_type}_column_evaluations.csv"
    out_path = None
    if not column_eval_df.empty:
        out_path = _output_files_dir(task_dir, station_name) / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        column_eval_df.to_csv(out_path, index=False)

    ordered_columns = [
        "filename_base",
        pass_column,
        "qa_status",
        "qa_timestamp",
        "qa_timestamp_source",
        "qa_in_scope",
        "epoch_id",
        "conf_number",
        "qa_evaluated_columns",
        "qa_passed_columns",
        "qa_failed_columns",
        "qa_warning_columns",
        "qa_pass_fraction",
        "qa_failed_observables",
        "qa_warning_reasons",
    ]
    available_columns = [column for column in ordered_columns if column in file_df.columns]
    return file_df[available_columns].copy(), out_path


def _chunk_columns(columns: list[str], chunk_size: int | None) -> list[list[str]]:
    if chunk_size is None or chunk_size <= 0 or len(columns) <= chunk_size:
        return [columns]
    return [columns[idx : idx + chunk_size] for idx in range(0, len(columns), chunk_size)]


def _generate_station_plots(
    *,
    task_dir: Path,
    station_name: str,
    task_id: int,
    metadata_type: str,
    config: dict[str, Any],
    analyzed_df: pd.DataFrame,
    default_plot_groups: list[dict[str, Any]] | None,
    skip_columns: set[str],
    allow_migration_matrix: bool,
) -> list[Path]:
    if analyzed_df.empty:
        return []

    df = analyzed_df.copy()
    x_values, x_label, x_is_datetime = resolve_plot_x_axis(df, config)
    df["__plot_x__"] = x_values
    sort_key = "__plot_x__" if x_label != "__timestamp__" else "__timestamp__"
    df = df.sort_values(sort_key).reset_index(drop=True)
    x_values = df["__plot_x__"]

    internal_skip = set(skip_columns) | {"__timestamp__", "__plot_x__"}
    groups = _plot_groups_from_config(config, default_plot_groups=default_plot_groups, skip_columns=internal_skip)
    plots_dir = _output_plots_dir(task_dir, station_name)
    plot_cfg = config.get("plots") if isinstance(config.get("plots"), dict) else {}
    image_format = str(plot_cfg.get("format", "png")).strip().lower() or "png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in plots_dir.glob(f"{station_name}_task_{task_id}_{metadata_type}_*.{image_format}"):
        stale_path.unlink()
    max_columns_per_group = int(plot_cfg.get("max_columns_per_group", 12))

    created: list[Path] = []
    for group in groups:
        group_name = str(group["name"])
        group_ncols = group.get("ncols")
        if group.get("mode") == "overlay":
            panels = _resolve_overlay_panels(list(df.columns), group.get("overlay_columns", []))
            out_path = _plot_overlay_group(
                df=df,
                x_values=x_values,
                x_label=x_label,
                x_is_datetime=x_is_datetime,
                panels=panels,
                series_labels=group.get("overlay_labels", []),
                station_name=station_name,
                group_name=group_name,
                out_dir=plots_dir,
                config=config,
                task_id=task_id,
                metadata_type=metadata_type,
                group_ncols=group_ncols,
            )
        else:
            columns = [
                col
                for col in _resolve_columns(list(df.columns), group.get("columns", []))
                if col in df.columns and not _should_ignore_column(col, skip_columns=internal_skip, config=config)
            ]
            for chunk_index, chunk in enumerate(_chunk_columns(columns, max_columns_per_group), start=1):
                chunk_name = group_name if len(columns) <= max_columns_per_group else f"{group_name}_{chunk_index}"
                out_path = _plot_group(
                    df=df,
                    x_values=x_values,
                    x_label=x_label,
                    x_is_datetime=x_is_datetime,
                    columns=chunk,
                    station_name=station_name,
                    group_name=chunk_name,
                    out_dir=plots_dir,
                    config=config,
                    task_id=task_id,
                    metadata_type=metadata_type,
                    skip_columns=internal_skip,
                    allow_migration_matrix=allow_migration_matrix,
                    group_ncols=group_ncols,
                )
                if out_path is not None:
                    created.append(out_path)
            continue
        if out_path is not None:
            created.append(out_path)

    return created


def _run_time_series_task_core(
    *,
    config: dict[str, Any],
    repo_root: Path,
    base_dir: Path,
    output_dir: Path,
    task_id: int,
    metadata_suffix: str,
    metadata_type: str,
    default_pass_column: str,
    default_plot_groups: list[dict[str, Any]] | None,
    skip_columns: set[str] | None,
    allow_migration_matrix: bool,
) -> int:
    ensure_task_station_tree(output_dir, config)

    configured_task_id = int(config.get("task_id", task_id))
    configured_metadata_type = str(config.get("metadata_type", metadata_type)).strip() or metadata_type
    configured_metadata_filename = str(
        config.get("metadata_csv_filename", f"task_{configured_task_id}_metadata_{metadata_suffix}.csv")
    ).strip()

    stations = config.get("stations", [0, 1, 2, 3, 4])
    pass_column = str(config.get("pass_column_name", default_pass_column))
    default_pass = float(config.get("pass_default_value", 1.0))
    effective_skip = set(DEFAULT_SKIP_COLUMNS if skip_columns is None else skip_columns)

    total_rows = 0
    written = 0
    total_plots = 0
    total_references = 0
    total_quality_tables = 0

    for station in stations:
        station_name = normalize_station_name(station)
        meta_path = _metadata_path(repo_root, station_name, configured_task_id, configured_metadata_filename)
        meta_df = _load_metadata_frame(meta_path)
        scope_df = _prepare_scope_dataframe(station_name=station_name, config=config, meta_df=meta_df)
        analyzed_df, analysis_reason = _prepare_analysis_dataframe(
            station_name=station_name,
            config=config,
            meta_df=meta_df,
        )
        analyzed_df, epochs_df = _attach_epoch_metadata(repo_root, station_name, analyzed_df)

        groups = _plot_groups_from_config(
            config,
            default_plot_groups=default_plot_groups,
            skip_columns=set(effective_skip) | {"__timestamp__", "__plot_x__"},
        )
        reference_df, reference_path = _write_epoch_reference_table(
            task_dir=output_dir,
            base_dir=base_dir,
            station_name=station_name,
            metadata_type=configured_metadata_type,
            analyzed_df=analyzed_df,
            epochs_df=epochs_df,
            config=config,
            groups=groups,
            skip_columns=effective_skip,
        )
        pass_df, quality_eval_path = _build_quality_pass_dataframe(
            task_dir=output_dir,
            base_dir=base_dir,
            station_name=station_name,
            metadata_type=configured_metadata_type,
            scope_df=scope_df,
            analyzed_df=analyzed_df,
            reference_df=reference_df,
            config=config,
            groups=groups,
            skip_columns=effective_skip,
            pass_column=pass_column,
        )
        if pass_df.empty and not scope_df.empty:
            pass_df = _build_pass_dataframe(scope_df, pass_column, default_pass)
        total_rows += len(pass_df)

        files_dir = _output_files_dir(output_dir, station_name)
        files_dir.mkdir(parents=True, exist_ok=True)
        out_path = files_dir / f"{station_name}_task_{configured_task_id}_{configured_metadata_type}_pass.csv"
        pass_df.to_csv(out_path, index=False)
        written += 1
        if reference_path is not None:
            total_references += 1
        if quality_eval_path is not None:
            total_quality_tables += 1

        try:
            created_plots = _generate_station_plots(
                task_dir=output_dir,
                station_name=station_name,
                task_id=configured_task_id,
                metadata_type=configured_metadata_type,
                config=config,
                analyzed_df=analyzed_df,
                default_plot_groups=default_plot_groups,
                skip_columns=effective_skip,
                allow_migration_matrix=allow_migration_matrix,
            )
            total_plots += len(created_plots)
            if created_plots or reference_path is not None or quality_eval_path is not None:
                print(
                    f"TASK_{configured_task_id} {station_name}: plots={len(created_plots)} "
                    f"epoch_ref={'yes' if reference_path is not None else 'no'} "
                    f"quality={'yes' if quality_eval_path is not None else 'no'}"
                )
            elif analysis_reason is not None:
                print(f"TASK_{configured_task_id} {station_name}: {analysis_reason}.")
        except (KeyError, ValueError) as exc:
            print(f"TASK_{configured_task_id} {station_name}: plotting skipped ({exc})")

    print(
        f"TASK_{configured_task_id} {configured_metadata_type} complete: "
        f"stations={written} total_rows={total_rows} pass_column={pass_column} "
        f"plots={total_plots} epoch_references={total_references} quality_tables={total_quality_tables}"
    )
    return 0


def run_time_series_task(
    task_dir: Path,
    *,
    task_id: int,
    metadata_suffix: str,
    metadata_type: str,
    default_pass_column: str,
    default_plot_groups: list[dict[str, Any]] | None = None,
    skip_columns: set[str] | None = None,
    allow_migration_matrix: bool = False,
) -> int:
    """Run a task-level time-series QA step with epoch-aware optional quality evaluation."""
    step_dir = task_dir.parent
    qa_root = step_dir.parent
    repo_root = qa_root.parents[2]

    config = load_task_configs(task_dir)
    bootstrap_task(task_dir)
    return _run_time_series_task_core(
        config=config,
        repo_root=repo_root,
        base_dir=task_dir,
        output_dir=task_dir,
        task_id=task_id,
        metadata_suffix=metadata_suffix,
        metadata_type=metadata_type,
        default_pass_column=default_pass_column,
        default_plot_groups=default_plot_groups,
        skip_columns=skip_columns,
        allow_migration_matrix=allow_migration_matrix,
    )


def run_time_series_family_step(
    step_dir: Path,
    *,
    task_ids: list[int],
    metadata_suffix: str,
    metadata_type: str,
    pass_column_template: str,
    default_plot_groups: list[dict[str, Any]] | None = None,
    skip_columns: set[str] | None = None,
    allow_migration_matrix: bool = False,
) -> int:
    """Run one metadata family step over multiple TASK_N inputs using one shared step config."""
    qa_root = step_dir.parent
    repo_root = qa_root.parents[2]
    step_config = load_step_configs(step_dir)
    ensure_task_station_tree(step_dir, step_config)

    exit_code = 0
    for task_id in task_ids:
        task_output_dir = step_dir / f"TASK_{task_id}"
        task_config = deep_merge_dicts(
            step_config,
            {
                "task_id": task_id,
                "metadata_type": metadata_type,
                "metadata_csv_filename": f"task_{task_id}_metadata_{metadata_suffix}.csv",
                "pass_column_name": pass_column_template.format(task_id=task_id),
            },
        )
        ensure_task_station_tree(task_output_dir, task_config)
        try:
            _run_time_series_task_core(
                config=task_config,
                repo_root=repo_root,
                base_dir=step_dir,
                output_dir=task_output_dir,
                task_id=task_id,
                metadata_suffix=metadata_suffix,
                metadata_type=metadata_type,
                default_pass_column=pass_column_template.format(task_id=task_id),
                default_plot_groups=default_plot_groups,
                skip_columns=skip_columns,
                allow_migration_matrix=allow_migration_matrix,
            )
        except Exception as exc:
            exit_code = 1
            print(f"[family-step] TASK_{task_id} {metadata_type} failed: {exc}")
    return exit_code
