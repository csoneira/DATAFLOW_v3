#!/usr/bin/env python3
"""
Utilities to ensure matplotlib figures are rasterised before exporting to PDF.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend for batch jobs

from typing import Iterable, Callable  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402

__all__ = [
    "rasterize_figure",
    "pdf_save_rasterized_page",
    "save_rasterized_pdf",
]


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
        rasterized_predicate = lambda artist: hasattr(artist, "set_rasterized")

    for artist in fig.findobj(rasterized_predicate):
        try:
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
