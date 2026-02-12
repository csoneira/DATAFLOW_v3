#!/usr/bin/env python3
"""Uncertainty LUT loader and interpolation module (pure empirical).

This module provides a single entry point for downstream code to translate
a **data-side** parameter estimate + event count into uncertainty values:

    from uncertainty_lut import UncertaintyLUT
    lut = UncertaintyLUT.load("STEP_6_UNCERTAINTY_LUT/output/lut")
    sigma_flux, sigma_eff = lut.query(est_flux=1.02, est_eff=0.92,
                                       n_events=40000)

Terminology
-----------
- **Dictionary**: the fixed reference lookup table of (flux, eff) →
  observables, built from simulations in Step 1.
- **Data**: the sample being analysed — may be a simulated validation
  sample *or* a real detector measurement.  The query inputs ``est_flux``,
  ``est_eff``, and ``n_events`` are always data-side quantities.

Query strategy
--------------
1. **Trilinear interpolation** on the dense 3-D grid
   (est_flux × est_eff × n_events) → (σ_flux_pct, σ_eff_pct).
   Values are clamped to the grid boundaries (no extrapolation).

2. **Global fall-back** — position-independent quantile of the entire
   validation dataset, used only if the grid is completely empty or
   something unexpected happens.

All returned uncertainties are in **percent relative error** (matching the
convention of ``abs_flux_rel_error_pct`` / ``abs_eff_rel_error_pct`` from
Step 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# ── module-level helpers ────────────────────────────────────────────────────

def _interp3d(
    x: float, y: float, z: float,
    xm: np.ndarray, ym: np.ndarray, zm: np.ndarray,
    grid: np.ndarray,
) -> float | None:
    """Trilinear interpolation on a 3-D rectilinear grid with clamp-to-edge.

    Parameters
    ----------
    x, y, z : float
        Query point (flux, eff, events).
    xm, ym, zm : 1-D sorted arrays
        Midpoints along each axis.
    grid : 3-D array  (len(xm), len(ym), len(zm))
        Values to interpolate (should be NaN-free).
    """
    if len(xm) < 2 or len(ym) < 2 or len(zm) < 2:
        # Grid too small for interpolation — return single value if available
        if grid.size == 1:
            v = grid.flat[0]
            return float(v) if np.isfinite(v) else None
        return None

    # Clamp to boundary
    x = float(np.clip(x, xm[0], xm[-1]))
    y = float(np.clip(y, ym[0], ym[-1]))
    z = float(np.clip(z, zm[0], zm[-1]))

    # Index helpers
    def _bracket(val, mids):
        i = int(np.searchsorted(mids, val, side="right")) - 1
        i = max(0, min(i, len(mids) - 2))
        lo, hi = mids[i], mids[i + 1]
        t = (val - lo) / (hi - lo) if hi != lo else 0.0
        return i, t

    ix, tx = _bracket(x, xm)
    iy, ty = _bracket(y, ym)
    iz, tz = _bracket(z, zm)

    # 8 corner values
    c000 = grid[ix,     iy,     iz]
    c100 = grid[ix + 1, iy,     iz]
    c010 = grid[ix,     iy + 1, iz]
    c110 = grid[ix + 1, iy + 1, iz]
    c001 = grid[ix,     iy,     iz + 1]
    c101 = grid[ix + 1, iy,     iz + 1]
    c011 = grid[ix,     iy + 1, iz + 1]
    c111 = grid[ix + 1, iy + 1, iz + 1]

    corners = np.array([c000, c100, c010, c110, c001, c101, c011, c111])
    if np.any(~np.isfinite(corners)):
        valid = corners[np.isfinite(corners)]
        return float(np.mean(valid)) if len(valid) else None

    # Trilinear
    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    val = c0 * (1 - tz) + c1 * tz
    return float(val)


@dataclass
class UncertaintyLUT:
    """Look-Up Table for inference uncertainty estimates (pure empirical).

    This LUT is trained on *validation data* (simulated data matched
    against the dictionary) but is applied to *any* data — simulated or
    real.  The query inputs are always data-side quantities.

    Attributes
    ----------
    lut : pd.DataFrame
        Dense 3-D empirical LUT with σ per (flux, eff, events) cell.
    meta : dict
        Full metadata including ranges, edges, global fallback.
    global_sigma_flux : float
        Global fallback σ_flux [%].
    global_sigma_eff : float
        Global fallback σ_eff [%].
    """

    lut: pd.DataFrame
    meta: dict
    global_sigma_flux: float
    global_sigma_eff: float

    # Derived 3-D grids (built at load time)
    _flux_mids: np.ndarray  = field(default_factory=lambda: np.array([]))
    _eff_mids: np.ndarray   = field(default_factory=lambda: np.array([]))
    _events_mids: np.ndarray = field(default_factory=lambda: np.array([]))
    _sigma_flux_grid: np.ndarray = field(
        default_factory=lambda: np.array([]))
    _sigma_eff_grid: np.ndarray  = field(
        default_factory=lambda: np.array([]))

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def load(cls, lut_dir: str | Path) -> "UncertaintyLUT":
        """Load a previously built LUT from disk.

        Parameters
        ----------
        lut_dir : path-like
            Directory containing ``uncertainty_lut_meta.json`` and
            ``uncertainty_lut.csv``.
        """
        lut_dir = Path(lut_dir)
        meta_path = lut_dir / "uncertainty_lut_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"LUT metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        lut_path = lut_dir / "uncertainty_lut.csv"
        if not lut_path.exists():
            raise FileNotFoundError(f"LUT CSV not found: {lut_path}")
        lut_df = pd.read_csv(lut_path)

        lut_meta = meta.get("lut", {})
        global_sf = float(lut_meta.get("global_sigma_flux_pct", 5.0))
        global_se = float(lut_meta.get("global_sigma_eff_pct",  4.0))

        obj = cls(
            lut=lut_df,
            meta=meta,
            global_sigma_flux=global_sf,
            global_sigma_eff=global_se,
        )
        obj._build_grids()
        return obj

    def _build_grids(self) -> None:
        """Pre-compute dense 3-D grids for trilinear interpolation."""
        df = self.lut
        if df.empty:
            return

        flux_mids = np.sort(df["flux_mid"].unique())
        eff_mids  = np.sort(df["eff_mid"].unique())
        evts_mids = np.sort(df["events_mid"].unique())

        self._flux_mids   = flux_mids
        self._eff_mids    = eff_mids
        self._events_mids = evts_mids

        nf = len(flux_mids)
        ne = len(eff_mids)
        nn = len(evts_mids)

        sf_grid = np.full((nf, ne, nn), np.nan)
        se_grid = np.full((nf, ne, nn), np.nan)

        flux_idx  = {v: i for i, v in enumerate(flux_mids)}
        eff_idx   = {v: i for i, v in enumerate(eff_mids)}
        events_idx = {v: i for i, v in enumerate(evts_mids)}

        for _, row in df.iterrows():
            fi = flux_idx.get(row["flux_mid"])
            ei = eff_idx.get(row["eff_mid"])
            ni = events_idx.get(row["events_mid"])
            if fi is None or ei is None or ni is None:
                continue
            sf = row.get("sigma_flux_pct")
            se = row.get("sigma_eff_pct")
            if sf is not None and np.isfinite(sf):
                sf_grid[fi, ei, ni] = sf
            if se is not None and np.isfinite(se):
                se_grid[fi, ei, ni] = se

        # Fill remaining NaN via nearest-neighbour in 3-D grid space
        _fill_nan_nearest_3d(sf_grid)
        _fill_nan_nearest_3d(se_grid)

        self._sigma_flux_grid = sf_grid
        self._sigma_eff_grid  = se_grid

    # ── query interface ─────────────────────────────────────────────────

    def query(
        self,
        est_flux: float,
        est_eff: float,
        n_events: float,
    ) -> tuple[float, float]:
        """Return (σ_flux_pct, σ_eff_pct) for the given data measurement.

        Parameters
        ----------
        est_flux : float
            Estimated flux from the dictionary match for this data sample.
        est_eff : float
            Estimated efficiency from the dictionary match.
        n_events : float
            Number of events in the data sample being analysed.

        Returns
        -------
        (sigma_flux_pct, sigma_eff_pct) : tuple[float, float]
            Uncertainty in percent relative error.  These represent the
            expected error between the dictionary-matched estimate and
            the true value, as calibrated from the validation dataset.
        """
        sf = _interp3d(
            est_flux, est_eff, n_events,
            self._flux_mids, self._eff_mids, self._events_mids,
            self._sigma_flux_grid,
        )
        se = _interp3d(
            est_flux, est_eff, n_events,
            self._flux_mids, self._eff_mids, self._events_mids,
            self._sigma_eff_grid,
        )
        if sf is None or se is None or not np.isfinite(sf) or not np.isfinite(se):
            return (self.global_sigma_flux, self.global_sigma_eff)
        return (max(sf, 0.0), max(se, 0.0))

    def query_batch(
        self,
        est_flux: np.ndarray,
        est_eff: np.ndarray,
        n_events: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised version of :meth:`query`.

        Returns arrays (sigma_flux_pct, sigma_eff_pct).
        """
        est_flux = np.asarray(est_flux, dtype=float)
        est_eff  = np.asarray(est_eff,  dtype=float)
        n_events = np.asarray(n_events, dtype=float)
        n = len(est_flux)
        sf = np.empty(n, dtype=float)
        se = np.empty(n, dtype=float)
        for i in range(n):
            sf[i], se[i] = self.query(est_flux[i], est_eff[i], n_events[i])
        return sf, se

    # ── utilities ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the loaded LUT."""
        m = self.meta
        lm = m.get("lut", {})
        lines = [
            "Uncertainty LUT Summary (empirical)",
            "=" * 44,
            f"Source:        {m.get('source_csv', '?')}",
            f"Valid rows:    {m.get('n_valid_rows', '?')}",
            f"Flux range:    {m.get('estimated_flux_range', '?')}",
            f"Eff range:     {m.get('estimated_eff_range', '?')}",
            f"Events range:  {m.get('events_range', '?')}",
            "",
            f"Grid shape:    {len(self._flux_mids)} × "
            f"{len(self._eff_mids)} × {len(self._events_mids)}",
            f"Total cells:   {lm.get('n_cells_total', '?')}",
            f"Direct cells:  {lm.get('n_cells_filled', '?')}",
            f"Infilled:      {lm.get('n_cells_infilled', '?')}",
            f"Outliers removed: {lm.get('n_outliers_removed', '?')}",
            f"Quantile:      p{int(lm.get('quantile', 0.68) * 100)}",
            f"IQR factor:    {lm.get('iqr_factor', '?')}",
            "",
            f"Global fallback: σ_flux={self.global_sigma_flux:.3f}%  "
            f"σ_eff={self.global_sigma_eff:.3f}%",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        shape = (len(self._flux_mids), len(self._eff_mids),
                 len(self._events_mids))
        return f"UncertaintyLUT(grid={shape[0]}×{shape[1]}×{shape[2]})"


# ── 3-D NaN infilling ──────────────────────────────────────────────────────

def _fill_nan_nearest_3d(grid: np.ndarray) -> None:
    """In-place fill NaN cells with nearest valid neighbour in 3-D."""
    nan_mask = ~np.isfinite(grid)
    if not nan_mask.any():
        return
    if not (~nan_mask).any():
        return

    try:
        from scipy.ndimage import distance_transform_edt
        _, nearest_idx = distance_transform_edt(
            nan_mask, return_distances=True, return_indices=True)
        grid[nan_mask] = grid[tuple(nearest_idx[:, nan_mask])]
    except ImportError:
        valid_ijk = np.argwhere(~nan_mask)
        nan_ijk   = np.argwhere(nan_mask)
        for ni, nj, nk in nan_ijk:
            dists = ((valid_ijk[:, 0] - ni) ** 2
                     + (valid_ijk[:, 1] - nj) ** 2
                     + (valid_ijk[:, 2] - nk) ** 2)
            best = valid_ijk[dists.argmin()]
            grid[ni, nj, nk] = grid[best[0], best[1], best[2]]


# ── convenience for scripts ─────────────────────────────────────────────────

def load_lut(lut_dir: str | Path) -> UncertaintyLUT:
    """Shortcut for ``UncertaintyLUT.load(lut_dir)``."""
    return UncertaintyLUT.load(lut_dir)
