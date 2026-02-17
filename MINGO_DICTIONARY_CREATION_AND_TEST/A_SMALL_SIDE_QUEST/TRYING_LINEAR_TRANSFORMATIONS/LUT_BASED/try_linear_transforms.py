#!/usr/bin/env python3
"""Try linear transformations of `eff` (scale only) and show results.

Produces:
 - metric vs eff-scale (corr between flux and decorrelated global rate)
 - time-series overlay: flux (z) vs best-scale decorrelated global rate (z)
 - scatter: flux vs best-scale decorrelated global rate

Saves PNGs to a local `PLOTS/` folder.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = '/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/dictionary_test.csv'
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, 'PLOTS')
DOWNSAMPLE = 10


def prepare():
    if os.path.exists(PLOTS):
        shutil.rmtree(PLOTS)
    os.makedirs(PLOTS, exist_ok=True)


def load():
    """Read CSV and create canonical columns used downstream:
    - `flux` (prefer `flux` else `flux_cm2_min`)
    - `eff` (prefer `eff` else `eff_empirical_1` else mean of eff_empirical_*)
    - `global_rate_hz` (prefer `events_per_second_global_rate`, else `clean_tt_1234_rate_hz`, else `raw_tt_1234_rate_hz`)
    - `time_utc` (prefer `time_utc`, else `execution_timestamp`, else fallback to integer index)
    """
    df = pd.read_csv(CSV_PATH)
    # downsample for plotting speed
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)

    # flux
    if 'flux' in df.columns:
        df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
    elif 'flux_cm2_min' in df.columns:
        df['flux'] = pd.to_numeric(df['flux_cm2_min'], errors='coerce')
    else:
        raise RuntimeError('No flux column found in CSV (expected flux or flux_cm2_min)')

    # efficiency (try several reasonable fallbacks)
    if 'eff' in df.columns:
        df['eff'] = pd.to_numeric(df['eff'], errors='coerce')
    else:
        # prefer empirical single-channel value, else average empirical across channels, else sim
        if 'eff_empirical_1' in df.columns:
            df['eff'] = pd.to_numeric(df['eff_empirical_1'], errors='coerce')
        else:
            emp_cols = [c for c in df.columns if c.startswith('eff_empirical_')]
            sim_cols = [c for c in df.columns if c.startswith('eff_sim_')]
            if emp_cols:
                df['eff'] = df[emp_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            elif sim_cols:
                df['eff'] = df[sim_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            else:
                raise RuntimeError('No efficiency column found (tried eff, eff_empirical_*, eff_sim_*)')

    # global rate (choose the most appropriate column available)
    if 'global_rate_hz' in df.columns:
        df['global_rate_hz'] = pd.to_numeric(df['global_rate_hz'], errors='coerce')
    else:
        for cand in ('events_per_second_global_rate', 'clean_tt_1234_rate_hz', 'raw_tt_1234_rate_hz'):
            if cand in df.columns:
                df['global_rate_hz'] = pd.to_numeric(df[cand], errors='coerce')
                break
        else:
            # as a last resort try any column with 'global' or 'rate' in the name
            candidates = [c for c in df.columns if 'global' in c or 'rate' in c]
            if candidates:
                df['global_rate_hz'] = pd.to_numeric(df[candidates[0]], errors='coerce')
            else:
                raise RuntimeError('No global rate column found (tried several fallbacks)')

    # time column
    if 'time_utc' in df.columns:
        df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
    elif 'execution_timestamp' in df.columns:
        # execution_timestamp looks like 2026-02-06_00.41.36
        ts = pd.to_datetime(df['execution_timestamp'], format='%Y-%m-%d_%H.%M.%S', errors='coerce')
        if ts.notna().sum() == 0:
            ts = pd.to_datetime(df['execution_timestamp'], errors='coerce')
        df['time_utc'] = ts
    elif 'param_date' in df.columns:
        ts = pd.to_datetime(df['param_date'], format='%Y-%m-%d', errors='coerce')
        if ts.notna().sum() == 0:
            ts = pd.to_datetime(df['param_date'], errors='coerce')
        df['time_utc'] = ts
    else:
        # create a monotonic time index when no timestamp column exists
        df['time_utc'] = pd.date_range('2000-01-01', periods=len(df), freq='min')

    return df


def robust_linear_decorrelation(x, y):
    # single-pass OLS here (we only need a stable linear removal for tuning)
    m, b = np.polyfit(x, y, 1)
    preds = m * x + b
    decor = (y - preds) + y.mean()
    return decor


def tune_scale(df, scales=np.linspace(0.5, 1.5, 41)):
    out = []
    for s in scales:
        eff_s = df['eff'].values * s
        dec = robust_linear_decorrelation(eff_s, df['global_rate_hz'].values)
        r = pd.Series(df['flux']).corr(pd.Series(dec))
        out.append((s, r, dec))
    return out


def format_time(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)


def main():
    prepare()
    df = load()

    results = tune_scale(df)
    scales = [r[0] for r in results]
    corrs = [r[1] for r in results]

    # best scale -> maximize corr(flux, decorrelated)
    best_scale, best_r, best_dec = max(results, key=lambda t: t[1])

    # metric plot
    plt.figure(figsize=(6, 3))
    plt.plot(scales, corrs, marker='o')
    plt.axvline(best_scale, color='red', ls='--', label=f'best={best_scale:.3f}')
    plt.xlabel('eff scale')
    plt.ylabel('corr(flux, decorrelated)')
    plt.title('Tuning eff scale')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, '01_effscale_metric.png'))
    plt.close()

    # time-series overlay (z-scored)
    flux_z = (df['flux'] - df['flux'].mean()) / df['flux'].std()
    best_dec_z = (best_dec - np.mean(best_dec)) / np.std(best_dec)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['time_utc'], flux_z, label='flux (z)', color='C0')
    ax.plot(df['time_utc'], best_dec_z, label=f'decorrelated (scale={best_scale:.3f})', color='C1')
    ax.set_ylabel('z-score')
    ax.set_xlabel('time_utc')
    ax.set_title('Flux vs decorrelated global rate (best eff scale)')
    ax.legend()
    ax.grid(True)
    format_time(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, '02_flux_vs_best_decor_timeseries.png'))
    plt.close()

    # scatter validation
    plt.figure(figsize=(6, 5))
    plt.scatter(df['flux'], best_dec, s=8, alpha=0.8)
    m, b = np.polyfit(df['flux'], best_dec, 1)
    xs = np.linspace(df['flux'].min(), df['flux'].max(), 100)
    plt.plot(xs, m * xs + b, color='k', lw=1)
    plt.xlabel('flux')
    plt.ylabel('global_rate_decorrelated')
    plt.title(f'Flux vs decorrelated (best scale={best_scale:.3f}, r={best_r:.3f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, '03_flux_vs_best_decor_scatter.png'))
    plt.close()

    # write summary CSV
    with open(os.path.join(PLOTS, '00_effscale_summary.csv'), 'w') as f:
        f.write('scale,corr_flux_decorrelated\n')
        for s, r, _ in results:
            f.write(f"{s:.6f},{r:.6f}\n")
        f.write(f"#best,{best_scale:.6f},{best_r:.6f}\n")

    # --- Build simple LUT (binned) mapping (eff, flux) -> mean global_rate ---
    eff_bins = 50
    flux_bins = 50
    eff_edges = np.linspace(df['eff'].min(), df['eff'].max(), eff_bins + 1)
    flux_edges = np.linspace(df['flux'].min(), df['flux'].max(), flux_bins + 1)
    # compute sums and counts in 2D bins
    inds_eff = np.digitize(df['eff'], eff_edges) - 1
    inds_flux = np.digitize(df['flux'], flux_edges) - 1
    lut_sum = np.zeros((eff_bins, flux_bins), dtype=float)
    lut_count = np.zeros((eff_bins, flux_bins), dtype=int)
    for ie, ifx, gr in zip(inds_eff, inds_flux, df['global_rate_hz']):
        if 0 <= ie < eff_bins and 0 <= ifx < flux_bins:
            lut_sum[ie, ifx] += gr
            lut_count[ie, ifx] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        lut_mean = np.where(lut_count > 0, lut_sum / lut_count, np.nan)

    eff_centers = 0.5 * (eff_edges[:-1] + eff_edges[1:])
    flux_centers = 0.5 * (flux_edges[:-1] + flux_edges[1:])

    # save LUT as CSV (rows=eff, columns=flux)
    lut_df = pd.DataFrame(lut_mean, index=np.round(eff_centers, 6), columns=np.round(flux_centers, 6))
    lut_df.to_csv(os.path.join(PLOTS, '04_lut_eff_flux_to_globalrate.csv'), na_rep='')

    # --- Use LUT to estimate flux from (eff, global_rate)
    # 1) Nearest-neighbour (existing simple approach)
    def estimate_flux_from_lut_nn(eff_val, gr_val):
        ie = np.argmin(np.abs(eff_centers - eff_val))
        row = lut_mean[ie, :]
        if np.all(np.isnan(row)):
            return np.nan
        idx = np.nanargmin(np.abs(row - gr_val))
        return flux_centers[idx]

    # 2) Interpolation-based inversion per eff-bin
    interpolators = [None] * eff_bins
    for ie in range(eff_bins):
        row = lut_mean[ie, :]
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            interpolators[ie] = None
            continue
        gr_vals = row[valid]
        fx_vals = flux_centers[valid]
        # sort by gr for monotonic interpolation; keep unique gr values
        order = np.argsort(gr_vals)
        gr_sorted = gr_vals[order]
        fx_sorted = fx_vals[order]
        # remove duplicates in gr_sorted
        unique_gr, uidx = np.unique(gr_sorted, return_index=True)
        fx_unique = fx_sorted[uidx]
        if unique_gr.size < 2:
            interpolators[ie] = None
            continue
        # store arrays for np.interp (gr -> flux)
        interpolators[ie] = (unique_gr, fx_unique)

    def estimate_flux_from_lut_interp(eff_val, gr_val):
        ie = np.argmin(np.abs(eff_centers - eff_val))
        interp = interpolators[ie]
        if interp is None:
            return estimate_flux_from_lut_nn(eff_val, gr_val)
        xgr, yfx = interp
        # if gr_val outside interpolation range, clip to min/max
        if gr_val <= xgr[0]:
            return yfx[0]
        if gr_val >= xgr[-1]:
            return yfx[-1]
        return float(np.interp(gr_val, xgr, yfx))

    # compute estimates using both methods
    est_nn = [estimate_flux_from_lut_nn(e, g) for e, g in zip(df['eff'].values, df['global_rate_hz'].values)]
    est_interp = [estimate_flux_from_lut_interp(e, g) for e, g in zip(df['eff'].values, df['global_rate_hz'].values)]
    df['est_flux_from_lut_nn'] = est_nn
    df['est_flux_from_lut_interp'] = est_interp

    # plot time series: original flux vs estimated flux (LUT, interpolation)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['time_utc'], df['flux'], label='flux (orig)', color='C0', linewidth=1)
    ax.plot(df['time_utc'], df['est_flux_from_lut_interp'], label='flux (estimated from LUT — interp)', color='C2', linewidth=1, alpha=0.9)
    ax.plot(df['time_utc'], df['est_flux_from_lut_nn'], label='flux (estimated from LUT — nn)', color='C1', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('time_utc')
    ax.set_ylabel('flux')
    ax.set_title('Original flux vs flux estimated from (eff, global_rate) LUT — interp vs nn')
    ax.legend()
    ax.grid(True)
    format_time(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, '05_flux_vs_estimated_from_lut_timeseries_interp.png'))

    # scatter: original flux vs estimated flux (interp)
    plt.figure(figsize=(6, 5))
    plt.scatter(df['flux'], df['est_flux_from_lut_interp'], s=8, alpha=0.8, label='interp')
    plt.scatter(df['flux'], df['est_flux_from_lut_nn'], s=8, alpha=0.4, label='nn')
    plt.xlabel('flux (orig)')
    plt.ylabel('flux (estimated from LUT)')
    plt.title('Original vs estimated flux (LUT interpolation vs NN)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, '06_flux_vs_estimated_from_lut_scatter_interp.png'))

    # Save interpolator existence map for debugging
    interp_exist = np.array([0 if v is None else 1 for v in interpolators])
    plt.figure(figsize=(6, 1))
    plt.imshow(interp_exist.reshape(1, -1), aspect='auto', cmap='Greens')
    plt.yticks([])
    plt.xlabel('eff bin')
    plt.title('Interpolation available per eff bin (1=yes)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, '07_interpolator_map.png'))

    # --- Iso-rate contour (flux vs eff colored by global_rate) ---
    xm = pd.to_numeric(df['flux'], errors='coerce').to_numpy(dtype=float)
    ym = pd.to_numeric(df['eff'], errors='coerce').to_numpy(dtype=float)
    zm = pd.to_numeric(df['global_rate_hz'], errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(zm)
    if mask.sum() >= 10:
        xm_m, ym_m, zm_m = xm[mask], ym[mask], zm[mask]
        levels = np.arange(np.floor(zm_m.min()), np.ceil(zm_m.max()) + 1, 1.0)
        if len(levels) < 2:
            levels = np.linspace(zm_m.min(), zm_m.max(), 8)
        cmap = plt.cm.viridis
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(xm_m, ym_m, c=zm_m, cmap=cmap, s=20, alpha=0.8,
                        edgecolors='0.3', linewidths=0.3, vmin=levels.min(), vmax=levels.max())
        try:
            from matplotlib.tri import Triangulation, LinearTriInterpolator
            tri = Triangulation(xm_m, ym_m)
            interp = LinearTriInterpolator(tri, zm_m)
            xi = np.linspace(xm_m.min(), xm_m.max(), 300)
            yi = np.linspace(ym_m.min(), ym_m.max(), 300)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = interp(Xi, Yi)
            cs = ax.contour(Xi, Yi, Zi, levels=levels, cmap=cmap, linewidths=1.2, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f Hz")
        except Exception as exc:
            # fall back to no contours
            pass
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Global rate [Hz]')
        ax.set_xlabel('Flux [cm^-2 min^-1]')
        ax.set_ylabel('Efficiency')
        ax.set_title('Iso-global-rate contours (data)')
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        plt.savefig(os.path.join(PLOTS, '08_iso_rate_global_rate.png'))
        plt.close(fig)

    print('Done — outputs in', PLOTS)


if __name__ == '__main__':
    main()
