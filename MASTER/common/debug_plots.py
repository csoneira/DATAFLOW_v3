from __future__ import annotations

import math
import os
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
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
            series = df[col]
            series = np.asarray(series, dtype=float)
            series = series[np.isfinite(series)]
            if series.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(col)
                ax.axis("off")
                continue

            ax.hist(series, bins=bins, color="C0", alpha=0.7)
            ax.set_yscale("log")
            ax.set_title(col)
            ax.set_xlabel("value")
            ax.set_ylabel("count")

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
                data_min = float(np.nanmin(series))
                data_max = float(np.nanmax(series))
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
