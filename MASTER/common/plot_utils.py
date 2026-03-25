#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MASTER/common/plot_utils.py
Purpose: Utilities to ensure matplotlib figures are rasterised before exporting to PDF.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/common/plot_utils.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend for batch jobs

from pathlib import Path
from typing import Iterable, Callable  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, VPacker  # noqa: E402
from matplotlib.spines import Spine  # noqa: E402

__all__ = [
    "collect_saved_plot_paths",
    "ensure_plot_state",
    "rasterize_figure",
    "pdf_save_rasterized_page",
    "save_rasterized_pdf",
]


UNSUPPORTED_RASTER_TYPES = (TextArea, DrawingArea, HPacker, VPacker, Spine)


def ensure_plot_state(
    namespace: dict[str, object],
    *,
    fig_key: str = "fig_idx",
    plot_list_key: str = "plot_list",
    start_index: int = 1,
) -> tuple[int, list[str]]:
    """
    Ensure plot bookkeeping exists exactly once within a task script namespace.

    This prevents repeated config/setup blocks from accidentally resetting the
    current figure counter or the list of generated plots.
    """

    fig_val = namespace.get(fig_key)
    if not isinstance(fig_val, int) or fig_val < int(start_index):
        fig_val = int(start_index)
        namespace[fig_key] = fig_val

    plot_val = namespace.get(plot_list_key)
    if not isinstance(plot_val, list):
        plot_val = []
        namespace[plot_list_key] = plot_val

    return fig_val, plot_val


def _plot_order_key(path_str: str) -> tuple[int, str]:
    name = Path(path_str).name
    stem = Path(path_str).stem
    prefix = stem.split("_", 1)[0]
    try:
        idx = int(prefix)
    except (TypeError, ValueError):
        idx = 10**12
    return idx, name.lower()


def collect_saved_plot_paths(
    plot_list: Iterable[str] | None,
    figure_directory: str | os.PathLike[str] | None,
    *,
    suffixes: Iterable[str] = (".png",),
) -> list[str]:
    """
    Return saved plot paths in stable order, using both in-memory bookkeeping
    and the on-disk figure directory as a fallback.
    """

    normalized_suffixes = {
        str(s).lower() if str(s).startswith(".") else f".{str(s).lower()}"
        for s in suffixes
    }
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str | os.PathLike[str] | None) -> None:
        if not candidate:
            return
        path_str = str(candidate)
        if path_str in seen or not os.path.exists(path_str):
            return
        if normalized_suffixes and Path(path_str).suffix.lower() not in normalized_suffixes:
            return
        seen.add(path_str)
        ordered.append(path_str)

    for candidate in plot_list or []:
        _add(candidate)

    if figure_directory and os.path.isdir(figure_directory):
        directory_paths = sorted(
            (str(Path(figure_directory) / name) for name in os.listdir(figure_directory)),
            key=_plot_order_key,
        )
        for candidate in directory_paths:
            _add(candidate)

    return ordered


def rasterize_figure(fig: plt.Figure, rasterized_predicate: Callable[[Artist], bool] | None = None) -> None:
    """
    Mark all relevant artists on *fig* as rasterised so exported PDFs embed bitmaps.

    Parameters
    ----------
    fig:
        Matplotlib figure to rasterise.
    rasterized_predicate:
        Optional predicate to decide which artists should be rasterised. When
        omitted, every artist with ``set_rasterized`` is rasterised.
    """

    if rasterized_predicate is None:
        rasterized_predicate = (
            lambda artist: hasattr(artist, "set_rasterized")
            and not isinstance(artist, UNSUPPORTED_RASTER_TYPES)
        )

    for artist in fig.findobj(rasterized_predicate):
        try:
            if isinstance(artist, UNSUPPORTED_RASTER_TYPES):
                continue
            artist.set_rasterized(True)
        except Exception:
            continue


def pdf_save_rasterized_page(pdf: PdfPages, fig: plt.Figure, dpi: int = 150, **savefig_kwargs) -> None:
    """
    Rasterise *fig* and append it to the provided ``PdfPages`` object.
    """

    rasterize_figure(fig)
    pdf.savefig(fig, dpi=dpi, **savefig_kwargs)


def save_rasterized_pdf(fig: plt.Figure, path: str, dpi: int = 150, **savefig_kwargs) -> None:
    """
    Rasterise *fig* and save it directly as a PDF to *path*.
    """

    rasterize_figure(fig)
    fig.savefig(path, dpi=dpi, format="pdf", **savefig_kwargs)
