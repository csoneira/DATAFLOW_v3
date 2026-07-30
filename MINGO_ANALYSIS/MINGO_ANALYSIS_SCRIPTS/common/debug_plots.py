"""
DATAFLOW_v3 Script Header v1
Script: MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/debug_plots.py
Purpose: Debug plots.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/debug_plots.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import math
import os
import re
from typing import Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return cleaned.strip("_") or "debug"


def _normalize_thresholds(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value if v is not None]
    return [float(value)]


def _histogram_values(series) -> tuple[np.ndarray, list[str] | None, str]:
    """Return finite plottable values for numeric, datetime, or categorical data."""
    raw = np.asarray(series)
    if np.issubdtype(raw.dtype, np.datetime64):
        nanoseconds = raw.astype("datetime64[ns]").astype(np.int64)
        missing = nanoseconds == np.iinfo(np.int64).min
        values = nanoseconds.astype(float) / 1_000_000_000.0
        values[missing] = np.nan
        return values[np.isfinite(values)], None, "Unix time [s]"

    try:
        values = np.asarray(series, dtype=float)
        return values, None, "value"
    except (TypeError, ValueError):
        categories: list[str] = []
        category_index: dict[str, int] = {}
        encoded: list[float] = []
        for value in raw:
            if value is None:
                continue
            try:
                if bool(np.asarray(value != value).item()):
                    continue
            except (TypeError, ValueError):
                pass
            label = str(value)
            if label not in category_index:
                category_index[label] = len(categories)
                categories.append(label)
            encoded.append(float(category_index[label]))
        return np.asarray(encoded, dtype=float), categories, "category"


def _filter_numeric_histogram_values(
    values: np.ndarray,
    *,
    exclude_zeros: bool,
    absolute_limit: float | None,
) -> tuple[np.ndarray, int, int]:
    """Filter finite numeric values and return distinct omission counts."""
    zero_mask = values == 0 if exclude_zeros else np.zeros(values.size, dtype=bool)
    outside_mask = (
        np.abs(values) > absolute_limit
        if absolute_limit is not None
        else np.zeros(values.size, dtype=bool)
    )
    keep = np.isfinite(values) & ~(zero_mask | outside_mask)
    return values[keep], int(np.count_nonzero(zero_mask)), int(np.count_nonzero(outside_mask))


def plot_debug_histograms(
    df,
    columns: Sequence[str],
    thresholds: Mapping[str, Iterable[float]] | None,
    title: str,
    out_dir: str,
    fig_idx: int = 1,
    *,
    bins: int = 80,
    max_cols_per_fig: int = 12,
    show: bool = False,
    y_scale: str = "log",
    exclude_zeros: bool = False,
    absolute_limit: float | None = None,
    annotate_omissions: bool = False,
) -> int:
    """Save debug histogram grids for *columns*, with optional threshold lines."""
    if df is None or not columns:
        return fig_idx

    if thresholds is None:
        thresholds = {}

    os.makedirs(out_dir, exist_ok=True)

    # Chunk columns to keep figures readable.
    for start in range(0, len(columns), max_cols_per_fig):
        chunk = columns[start : start + max_cols_per_fig]
        ncols = 3
        nrows = max(1, math.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, col in zip(axes, chunk):
            if col not in df.columns:
                ax.axis("off")
                continue
            values, category_labels, xlabel = _histogram_values(df[col])
            zeros_not_plotted = 0
            outside_window_not_plotted = 0
            if category_labels is None and xlabel == "value":
                (
                    values,
                    zeros_not_plotted,
                    outside_window_not_plotted,
                ) = _filter_numeric_histogram_values(
                    values,
                    exclude_zeros=exclude_zeros,
                    absolute_limit=absolute_limit,
                )
            title_text = col
            if annotate_omissions:
                window_label = (
                    f"|value| > {absolute_limit:g}"
                    if absolute_limit is not None
                    else "outside window"
                )
                title_text = (
                    f"{col}\nzeros not plotted: {zeros_not_plotted}; "
                    f"{window_label} not plotted: {outside_window_not_plotted}"
                )
            if values.size == 0:
                ax.text(0.5, 0.5, "No plottable data", ha="center", va="center")
                ax.set_title(title_text)
                if annotate_omissions:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                else:
                    ax.axis("off")
                continue

            histogram_bins = (
                np.arange(-0.5, len(category_labels) + 0.5, 1.0)
                if category_labels is not None
                else bins
            )
            ax.hist(values, bins=histogram_bins, color="C0", alpha=0.7)
            if y_scale in {"linear", "log"}:
                ax.set_yscale(y_scale)
            else:
                ax.set_yscale("log")
            ax.set_title(title_text)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count")
            if category_labels is not None and len(category_labels) <= 20:
                ax.set_xticks(range(len(category_labels)))
                ax.set_xticklabels(category_labels, rotation=45, ha="right", fontsize=7)

            line_values = _normalize_thresholds(thresholds.get(col))
            if line_values:
                ymin, ymax = ax.get_ylim()
                for value in line_values:
                    ax.axvline(value, color="red", linestyle="--", linewidth=1)
                    ax.text(
                        value,
                        ymax * 0.95,
                        f"{value:g}",
                        rotation=90,
                        color="red",
                        va="top",
                        ha="right",
                        fontsize=8,
                    )
                data_min = float(np.nanmin(values))
                data_max = float(np.nanmax(values))
                bound_min = min([data_min, *line_values])
                bound_max = max([data_max, *line_values])
                pad = 0.05 * (bound_max - bound_min) if bound_max > bound_min else 1.0
                ax.set_xlim(bound_min - pad, bound_max + pad)

        # Hide any unused axes.
        for ax in axes[len(chunk) :]:
            ax.axis("off")

        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"{fig_idx}_debug_{_slugify(title)}.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        fig_idx += 1

    return fig_idx
