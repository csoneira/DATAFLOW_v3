#!/usr/bin/env python3
"""Compact analysis: show original data (time series + scatters),
apply ONE decorrelation method (robust linear), and show resulting
scatter + time series for validation.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# file locations
HERE = os.path.dirname(os.path.abspath(__file__))
# Always read the CSV from the fixed absolute path as requested
CSV_PATH = '/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/the_simulated_file.csv'
# keep plots next to the script
PLOTS_DIR = os.path.join(HERE, 'PLOTS')

# configuration
DOWNSAMPLE_STEP = 10  # take every Nth row to speed plotting


def prepare_plots_dir():
    if os.path.exists(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    os.makedirs(PLOTS_DIR, exist_ok=True)


plot_counter = {'i': 1}

def save_prefixed(fname):
    out = os.path.join(PLOTS_DIR, f"{plot_counter['i']:02d}_" + fname)
    plt.savefig(out)
    plot_counter['i'] += 1
    return out


def load_and_prepare(path):
    df = pd.read_csv(path)
    df = df.iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    return df


def robust_linear_decorrelation(x, y, niter=6, sigma=2.5):
    """Iterative sigma-clipped linear fit of y ~ x and return decorrelated y.
    Decorrelated series = residuals + mean(y) (keeps original mean).
    Returns: decorrelated_y, slope, intercept
    """
    mask = np.ones(len(x), dtype=bool)
    slope = intercept = 0.0
    for _ in range(niter):
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        preds = intercept + slope * x
        resid = y - preds
        std = np.std(resid[mask])
        new_mask = np.abs(resid) < sigma * std
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    preds_final = intercept + slope * x
    decorrelated = (y - preds_final) + y.mean()
    return decorrelated, slope, intercept


def quadratic_decorrelation(x, y):
    """Remove quadratic dependence of y on x via degree-2 polynomial fit."""
    coeffs = np.polyfit(x, y, 2)
    preds = np.polyval(coeffs, x)
    decorrelated = (y - preds) + y.mean()
    return decorrelated, coeffs


def local_window_linear_decorrelation(x, y, window=31, min_points=10):
    """For each time point, fit a local linear model y~x within a time window and
    remove the local predicted component. Returns decorrelated series and
    (unused) arrays of local slopes/intercepts.
    """
    n = len(x)
    half = window // 2
    preds = np.zeros(n)
    slopes = np.zeros(n)
    inters = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < min_points:
            lo = max(0, i - min_points // 2)
            hi = min(n, lo + min_points)
        xs = x[lo:hi]
        ys = y[lo:hi]
        if len(xs) < 3:
            slopes[i] = 0.0
            inters[i] = np.mean(ys)
            preds[i] = inters[i]
            continue
        m_local, b_local = np.polyfit(xs, ys, 1)
        slopes[i] = m_local
        inters[i] = b_local
        preds[i] = b_local + m_local * x[i]
    decorrelated = (y - preds) + y.mean()
    return decorrelated, slopes, inters


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)


def plot_original(df):
    # time-series (three stacked subplots)
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(df['time_utc'], df['flux'], color='C0')
    ax[0].set_ylabel('flux')
    ax[0].grid(True)

    ax[1].plot(df['time_utc'], df['eff'], color='C2')
    ax[1].set_ylabel('eff')
    ax[1].grid(True)

    ax[2].plot(df['time_utc'], df['global_rate_hz'], color='C3')
    ax[2].set_ylabel('global_rate_hz')
    ax[2].set_xlabel('time_utc')
    ax[2].grid(True)

    format_time_axis(ax[2])
    plt.tight_layout()
    save_prefixed('original_timeseries.png')

    # scatter matrix (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(df['flux'], df['eff'], s=8)
    axes[0].set_xlabel('flux')
    axes[0].set_ylabel('eff')
    axes[0].grid(True)

    axes[1].scatter(df['flux'], df['global_rate_hz'], s=8)
    axes[1].set_xlabel('flux')
    axes[1].set_ylabel('global_rate_hz')
    axes[1].grid(True)

    axes[2].scatter(df['eff'], df['global_rate_hz'], s=8)
    axes[2].set_xlabel('eff')
    axes[2].set_ylabel('global_rate_hz')
    axes[2].grid(True)

    plt.tight_layout()
    save_prefixed('original_scatters.png')


def plot_after_decorrelation(df, decorrelated, slope, intercept):
    df = df.copy()
    df['global_rate_decorrelated'] = decorrelated

    # time-series comparison (original vs decorrelated)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['time_utc'], df['global_rate_hz'], label='original', alpha=0.6)
    ax.plot(df['time_utc'], df['global_rate_decorrelated'], label='decorrelated', alpha=0.9)
    ax.set_ylabel('global_rate (Hz)')
    ax.set_xlabel('time_utc')
    ax.legend()
    ax.grid(True)
    format_time_axis(ax)
    plt.tight_layout()
    # globalrate_original_vs_decorrelated_timeseries suppressed per user request

    # scatter: eff vs original and eff vs decorrelated (side-by-side)
    corr_before = df['eff'].corr(df['global_rate_hz'])
    corr_after = df['eff'].corr(df['global_rate_decorrelated'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df['eff'], df['global_rate_hz'], s=8)
    axes[0].set_title(f'before (r={corr_before:.3f})')
    axes[0].set_xlabel('eff')
    axes[0].set_ylabel('global_rate_hz')
    axes[0].grid(True)

    axes[1].scatter(df['eff'], df['global_rate_decorrelated'], s=8)
    axes[1].set_title(f'after (r={corr_after:.3f})')
    axes[1].set_xlabel('eff')
    axes[1].set_ylabel('global_rate_decorrelated')
    axes[1].grid(True)

    plt.tight_layout()
    save_prefixed('eff_vs_globalrate_decorrelated.png')

    # validation scatter: flux vs decorrelated global rate (fit shown, validation only)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df['flux'], df['global_rate_decorrelated'], s=8, alpha=0.8)
    m, b = np.polyfit(df['flux'], df['global_rate_decorrelated'], 1)
    xs = np.linspace(df['flux'].min(), df['flux'].max(), 100)
    ax.plot(xs, m * xs + b, color='k', lw=1, label=f'fit y={m:.3g}x+{b:.3g}')
    ax.set_xlabel('flux')
    ax.set_ylabel('global_rate_decorrelated')
    ax.set_title(f'flux vs decorrelated global rate (r={df["flux"].corr(df["global_rate_decorrelated"]):.3f})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # flux_vs_decorrelated_validation suppressed per user request

    # --- Compare flux time series with decorrelated global rate (z-scored overlay + cross-corr) ---
    # z-score both series for visual comparison
    flux_z = (df['flux'] - df['flux'].mean()) / df['flux'].std()
    gr_dec_z = (df['global_rate_decorrelated'] - df['global_rate_decorrelated'].mean()) / df['global_rate_decorrelated'].std()

    # Pearson correlation (visual/summary)
    corr_flux_decor = df['flux'].corr(df['global_rate_decorrelated'])
    print(f"corr(flux, global_rate_decorrelated) = {corr_flux_decor:.4f}")

    # Overlay time-series (z-scored)
    fig_ov, ax_ov = plt.subplots(figsize=(10, 4))
    ax_ov.plot(df['time_utc'], flux_z, label='flux (z)', color='C0', linewidth=1)
    ax_ov.plot(df['time_utc'], gr_dec_z, label='global_rate_decorrelated (z)', color='C1', linewidth=1, alpha=0.9)
    ax_ov.set_xlabel('time_utc')
    ax_ov.set_ylabel('z-score')
    ax_ov.set_title(f'Flux vs decorrelated global rate (z-scored) — r={corr_flux_decor:.3f}')
    ax_ov.legend()
    ax_ov.grid(True)
    format_time_axis(ax_ov)
    plt.tight_layout()
    save_prefixed('flux_vs_decorrelated_timeseries_overlay.png')

    # print summary
    print(f"decorrelation linear fit: slope={slope:.6g}, intercept={intercept:.6g}")
    print(f"corr(eff, global_rate) before={corr_before:.4f}, after={corr_after:.4f}")


def main():
    prepare_plots_dir()
    df = load_and_prepare(CSV_PATH)

    # quick validation of required columns
    for c in ('flux', 'eff', 'global_rate_hz'):
        if c not in df.columns:
            raise SystemExit(f"missing column: {c}")

    plot_original(df)

    # Try the three kept methods (OLS linear, robust linear, quadratic).
    # Exclude local-window method per your comment.
    methods = []

    # OLS linear (single-pass)
    deco_lin, sl_lin, ic_lin = robust_linear_decorrelation(df['eff'].values, df['global_rate_hz'].values, niter=1, sigma=1e9)
    methods.append(('linear_ols', deco_lin, {'slope': sl_lin, 'intercept': ic_lin}))

    # Robust linear (iterative sigma-clip)
    deco_rob, sl_rob, ic_rob = robust_linear_decorrelation(df['eff'].values, df['global_rate_hz'].values)
    methods.append(('linear_robust', deco_rob, {'slope': sl_rob, 'intercept': ic_rob}))

    # Quadratic
    deco_quad, coeffs_q = quadratic_decorrelation(df['eff'].values, df['global_rate_hz'].values)
    methods.append(('quadratic', deco_quad, {'coeffs': coeffs_q.tolist()}))

    # Prepare overlay colors/labels
    cmap = {'linear_ols': 'C1', 'linear_robust': 'C2', 'quadratic': 'C3'}

    # Produce the single combined time-series you requested: flux (z-scored)
    # with the ROBUST-decorrelated global rate (z-scored) overlaid.
    # (Do not create the other four files you listed.)
    # Select the robust-decorrelated series from methods
    deco_rob = None
    for name, deco_vals, meta in methods:
        if name == 'linear_robust':
            deco_rob = deco_vals
            break

    if deco_rob is None:
        raise RuntimeError('robust-decorrelated series not found')

    flux_z = (df['flux'] - df['flux'].mean()) / df['flux'].std()
    gr_dec_z = (deco_rob - np.mean(deco_rob)) / np.std(deco_rob)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['time_utc'], flux_z, label='flux (z)', color='C0', linewidth=1)
    ax.plot(df['time_utc'], gr_dec_z, label='global_rate_decorrelated (robust, z)', color='C1', linewidth=1, alpha=0.9)
    ax.set_xlabel('time_utc')
    ax.set_ylabel('z-score')
    ax.set_title('Flux vs robust-decorrelated global rate (z-scored)')
    ax.legend()
    ax.grid(True)
    format_time_axis(ax)
    plt.tight_layout()
    # remove duplicate single-overlay file (will produce the requested two-row comparison instead)
    # save_prefixed('flux_and_decorrelated_timeseries.png')
    # --- Combine the two requested panels into ONE FIGURE (replace the two files)
    fig_comb, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 10), sharex=False, gridspec_kw={'height_ratios': [1, 1]})

    # top: flux vs decorrelated global rate (scatter + fits for each method)
    ax_top.scatter(df['flux'], df['global_rate_hz'], s=8, alpha=0.4, label='original')
    for name, deco_vals, _ in methods:
        ax_top.scatter(df['flux'], deco_vals, s=6, alpha=0.8, label=name, color=cmap[name])
        m_fit, b_fit = np.polyfit(df['flux'], deco_vals, 1)
        xs = np.linspace(df['flux'].min(), df['flux'].max(), 100)
        ax_top.plot(xs, m_fit * xs + b_fit, color=cmap[name], lw=1)
    ax_top.set_xlabel('flux')
    ax_top.set_ylabel('global_rate_decorrelated')
    ax_top.set_title('flux vs decorrelated global rate (methods overlay)')
    ax_top.legend(fontsize=8)
    ax_top.grid(True)

    # bottom: eff vs decorrelated global rate (overlay methods)
    ax_bottom.scatter(df['eff'], df['global_rate_hz'], s=8, alpha=0.5, label='original')
    for name, deco_vals, _ in methods:
        ax_bottom.scatter(df['eff'], deco_vals, s=6, alpha=0.8, label=name, color=cmap[name])
    ax_bottom.set_xlabel('eff')
    ax_bottom.set_ylabel('global_rate_decorrelated')
    ax_bottom.set_title('eff vs global_rate (decorrelated methods overlay)')
    ax_bottom.legend(fontsize=8)
    ax_bottom.grid(True)

    plt.tight_layout()
    # Save the combined figure to replace both filenames the user listed
    combined_path = os.path.join(PLOTS_DIR, '03_eff_vs_globalrate_decorrelated_overlay.png')
    plt.savefig(combined_path)
    print('Wrote combined figure to', combined_path)

    # write a small summary file (keeps useful metrics)
    summary_lines = ['method,corr_eff_before,corr_eff_after,corr_flux_after']
    corr_eff_before = df['eff'].corr(df['global_rate_hz'])
    for name, deco_vals, meta in methods:
        corr_eff_after = pd.Series(df['eff']).corr(pd.Series(deco_vals))
        corr_flux_after = pd.Series(df['flux']).corr(pd.Series(deco_vals))
        summary_lines.append(f"{name},{corr_eff_before:.6f},{corr_eff_after:.6f},{corr_flux_after:.6f}")
    summary_path = os.path.join(PLOTS_DIR, f"{plot_counter['i']:02d}_decorrelation_summary.csv")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    plot_counter['i'] += 1

    print('Wrote summary to', summary_path)

    # --- EFF-SCALING SEARCH: tune eff multiplier to improve decorrelation ---
    def tune_eff_scaling(df, scales=np.linspace(0.5, 1.5, 41)):
        results = []
        for s in scales:
            eff_scaled = df['eff'].values * s
            deco_vals, _, _ = robust_linear_decorrelation(eff_scaled, df['global_rate_hz'].values)
            # metric: Pearson corr between flux and decorrelated global rate
            r = pd.Series(df['flux']).corr(pd.Series(deco_vals))
            results.append((s, r, deco_vals))
        return results

    scales = np.linspace(0.5, 1.5, 41)
    tune_results = tune_eff_scaling(df, scales=scales)
    # find best scale (max corr between flux and decorrelated global rate)
    best_scale, best_r, best_deco = max(tune_results, key=lambda x: x[1])
    print(f"Best eff scale = {best_scale:.3f} -> corr(flux, decorrelated) = {best_r:.4f}")

    # save metric vs scale plot
    scales_list = [t[0] for t in tune_results]
    r_list = [t[1] for t in tune_results]
    fig_ts, ax_ts = plt.subplots(figsize=(6, 3))
    ax_ts.plot(scales_list, r_list, marker='o')
    ax_ts.axvline(best_scale, color='red', linestyle='--', label=f'best={best_scale:.3f}')
    ax_ts.set_xlabel('eff scaling')
    ax_ts.set_ylabel('corr(flux, decorrelated)')
    ax_ts.set_title('Tuning eff scaling')
    ax_ts.grid(True)
    ax_ts.legend()
    plt.tight_layout()
    save_prefixed('eff_scaling_tuning_metric.png')

    # save best-scale time-series overlay (flux z vs decorrelated at best scale z)
    flux_z = (df['flux'] - df['flux'].mean()) / df['flux'].std()
    best_deco_z = (best_deco - np.mean(best_deco)) / np.std(best_deco)
    fig_best, ax_best = plt.subplots(figsize=(10, 4))
    ax_best.plot(df['time_utc'], flux_z, label='flux (z)', color='C0')
    ax_best.plot(df['time_utc'], best_deco_z, label=f'decorrelated (best scale={best_scale:.3f})', color='C1')
    ax_best.set_xlabel('time_utc')
    ax_best.set_ylabel('z-score')
    ax_best.set_title('Flux vs decorrelated global rate (best eff scaling)')
    ax_best.legend()
    ax_best.grid(True)
    format_time_axis(ax_best)
    plt.tight_layout()
    save_prefixed('flux_vs_decorrelated_best_scale_timeseries.png')

    # append tuning summary to the summary CSV
    with open(summary_path, 'a') as f:
        f.write(f"\n# eff scaling tuning: best_scale={best_scale:.6f}, best_corr_flux={best_r:.6f}\n")

    # --- FINAL: two-row figure requested by user (top: flux; bottom: all decorrelated methods) ---
    fig_final, axs_final = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    axs_final[0].plot(df['time_utc'], (df['flux'] - df['flux'].mean()) / df['flux'].std(), color='C0', lw=1)
    axs_final[0].set_ylabel('flux (z)')
    axs_final[0].grid(True)

    colors = {'linear_ols': 'C1', 'linear_robust': 'C2', 'quadratic': 'C3'}
    for name, deco_vals, _ in methods:
        dec_z = (deco_vals - np.mean(deco_vals)) / np.std(deco_vals)
        axs_final[1].plot(df['time_utc'], dec_z, label=name, color=colors.get(name, None), lw=0.9)
    # include best-scale decorrelated as well
    axs_final[1].plot(df['time_utc'], best_deco_z, label=f'best_scale={best_scale:.3f}', color='black', lw=1.2, alpha=0.8)
    axs_final[1].set_ylabel('global_rate (decorrelated, z)')
    axs_final[1].set_xlabel('time_utc')
    axs_final[1].legend(fontsize=9)
    axs_final[1].grid(True)
    format_time_axis(axs_final[1])
    plt.tight_layout()

    final_path = os.path.join(PLOTS_DIR, '03_flux_and_decorrelated_timeseries.png')
    plt.savefig(final_path)
    print('Wrote final comparison to', final_path)


if __name__ == '__main__':
    main()
