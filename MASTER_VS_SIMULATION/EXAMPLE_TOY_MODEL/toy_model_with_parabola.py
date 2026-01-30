#!/usr/bin/env python3
"""
Toy example: parameter estimation with chi-square scans using a precomputed "dictionary"
of noiseless templates (grid search), plus visualization.

Includes:
  1) 1-parameter model: y = a x^2
     - build dictionary over a_grid
     - generate one noisy dataset from a chosen "truth" template
     - scan chi2(a) over dictionary
     - plot data + selected templates
     - plot chi2(a) as discrete POINTS with 95% CI from Delta-chi2

  2) 2-parameter model: y = a x^2 + b x
     - build implicit dictionary over (a_grid, b_grid)
     - generate one noisy dataset from (a_true_grid, b_true_grid)
     - scan chi2(a,b) over grid
     - plot data + selected templates
     - plot Delta-chi2 heatmap + 68%/95% contours with:
         a) robust contrast scaling from zoom region
         b) larger zoom + robust scaling
         c) larger zoom + tight fixed color scale (vmax=25)

Assumptions:
  - Independent Gaussian measurement errors
  - Known sigma (constant or array-like)
  - Delta-chi2 confidence regions rely on Wilks' theorem:
      * 1 parameter: 95% -> Delta-chi2 = 3.84
      * 2 parameters: 68% -> 2.30, 95% -> 5.99
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Common utilities
# ----------------------------
def chi2(y_obs, y_pred, sigma):
    """Chi-square with independent errors; sigma can be scalar or array-like."""
    return np.sum(((y_obs - y_pred) / sigma) ** 2)


def snap_to_grid(val, grid):
    """Snap a scalar to the nearest point in a 1D grid."""
    return float(min(grid, key=lambda g: abs(g - val)))


# ============================================================
# 1-parameter case: y = a x^2
# ============================================================
def model_y_1p(a, x):
    return a * x**2


def fit_by_dictionary_1p(y_obs, templates, a_grid, sigma):
    """Scan chi2 over precomputed 1D dictionary templates."""
    chi2_vals = np.array([chi2(y_obs, templates[a], sigma) for a in a_grid])
    i_min = int(np.argmin(chi2_vals))
    return float(a_grid[i_min]), float(chi2_vals[i_min]), chi2_vals


def delta_chi2_interval_1p(a_grid, chi2_vals, delta=3.84):
    """Return [a_low, a_high] where chi2 <= chi2_min + delta."""
    chi2_min = float(np.min(chi2_vals))
    mask = chi2_vals <= (chi2_min + delta)
    if not np.any(mask):
        return None
    a_in = a_grid[mask]
    return float(a_in[0]), float(a_in[-1])


def run_one_parameter_demo(
    a_true=2.0,
    sigma=0.20,
    n_points=41,
    a_min=0.0,
    a_max=4.0,
    n_a=201,
    seed=123,
):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_points)

    a_grid = np.linspace(a_min, a_max, n_a)

    # Dictionary of noiseless templates
    templates = {a: model_y_1p(a, x) for a in a_grid}

    # Choose "truth" as nearest grid point
    a_true_grid = snap_to_grid(a_true, a_grid)
    y_true = templates[a_true_grid]

    # One observed dataset
    y_obs = y_true + rng.normal(0.0, sigma, size=x.shape)

    # Fit by scan
    a_hat, chi2_min, chi2_vals = fit_by_dictionary_1p(y_obs, templates, a_grid, sigma)
    ci95 = delta_chi2_interval_1p(a_grid, chi2_vals, delta=3.84)

    print("=== 1-parameter demo: y = a x^2 ===")
    print(f"Truth (grid): a_true_grid = {a_true_grid:.4f}")
    print(f"Best fit:     a_hat       = {a_hat:.4f}")
    print(f"chi2_min                = {chi2_min:.2f}")
    if ci95:
        print(f"Delta-chi2 95% CI       = [{ci95[0]:.4f}, {ci95[1]:.4f}]  (Delta chi2=3.84 for 1 parameter)")
    else:
        print("Delta-chi2 95% CI could not be determined (grid too narrow or too coarse).")

    # Plot observed data + selected templates
    plt.figure()
    plt.errorbar(x, y_obs, yerr=sigma, fmt="o", capsize=2, label="observed data")

    # Show a few dictionary entries (not all): truth, best-fit, and two offsets
    a_examples = [
        a_true_grid,
        a_hat,
        snap_to_grid(a_hat - 0.4, a_grid),
        snap_to_grid(a_hat + 0.4, a_grid),
    ]
    a_examples = sorted(set(a_examples))
    for ae in a_examples:
        plt.plot(x, templates[ae], label=f"template a={ae:.2f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Observed dataset and selected dictionary templates (1 parameter)")
    plt.legend()
    plt.show()

    # Plot chi2 scan as POINTS (discrete dictionary)
    plt.figure()
    plt.scatter(a_grid, chi2_vals, s=18)
    plt.scatter([a_hat], [chi2_min], s=60, label=f"best-fit a={a_hat:.3f}")

    if ci95:
        plt.axhline(chi2_min + 3.84)
        plt.axvline(ci95[0])
        plt.axvline(ci95[1])
        plt.legend()

    plt.xlabel("a")
    plt.ylabel(r"$\chi^2(a)$")
    plt.title(r"$\chi^2$ scan (discrete dictionary points) for $y=a x^2$")
    plt.show()

    return {
        "x": x,
        "sigma": sigma,
        "a_grid": a_grid,
        "templates": templates,
        "y_true": y_true,
        "y_obs": y_obs,
        "a_true_grid": a_true_grid,
        "a_hat": a_hat,
        "chi2_vals": chi2_vals,
        "chi2_min": chi2_min,
        "ci95_delta_chi2": ci95,
    }


# ============================================================
# 2-parameter case: y = a x^2 + b x
# ============================================================
def model_y_2p(a, b, x):
    return a * x**2 + b * x


def chi2_grid_scan_2p(y_obs, x, sigma, a_grid, b_grid):
    """
    Compute chi2 on a 2D grid.
      chi2_ab has shape (len(a_grid), len(b_grid)), indexed [ia, ib] = [a, b].
    """
    x2 = x**2
    y_pred = (
        a_grid[:, None, None] * x2[None, None, :] +
        b_grid[None, :, None] * x[None, None, :]
    )
    r = (y_obs[None, None, :] - y_pred) / sigma
    chi2_ab = np.sum(r**2, axis=2)

    ia, ib = np.unravel_index(np.argmin(chi2_ab), chi2_ab.shape)
    a_hat = float(a_grid[ia])
    b_hat = float(b_grid[ib])
    chi2_min = float(chi2_ab[ia, ib])
    return chi2_ab, (a_hat, b_hat), chi2_min


def plot_2p_data_and_templates(x, y_obs, sigma, a_true_grid, b_true_grid, a_hat, b_hat, a_grid, b_grid):
    """Observed data with error bars + a few selected templates."""
    plt.figure()
    plt.errorbar(x, y_obs, yerr=sigma, fmt="o", capsize=2, label="observed data")

    # Select a few templates: truth, best-fit, and two offset points (diagonal offsets)
    a_off_p = snap_to_grid(a_hat + 0.35, a_grid)
    b_off_p = snap_to_grid(b_hat + 0.35, b_grid)
    a_off_m = snap_to_grid(a_hat - 0.35, a_grid)
    b_off_m = snap_to_grid(b_hat - 0.35, b_grid)

    examples = [
        (a_true_grid, b_true_grid, "template (truth)"),
        (a_hat, b_hat, "template (best)"),
        (a_off_p, b_off_p, "template (offset +)"),
        (a_off_m, b_off_m, "template (offset -)"),
    ]

    for a_t, b_t, lab in examples:
        plt.plot(x, model_y_2p(a_t, b_t, x), label=f"{lab}: a={a_t:.2f}, b={b_t:.2f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Observed dataset and selected dictionary templates (2 parameters)")
    plt.legend()
    plt.show()


def plot_delta_chi2_zoomed(
    delta,
    a_grid,
    b_grid,
    a_hat,
    b_hat,
    a_true_grid,
    b_true_grid,
    levels,
    b_pad,
    a_pad,
    title,
    vmin=None,
    vmax=None,
    robust_from_zoom=True,
    robust_q=(1.0, 99.0),
):
    """
    Plot Delta-chi2 heatmap + contour levels, zoomed around (a_hat, b_hat).

    If vmin/vmax are None and robust_from_zoom=True, they are computed from
    the zoomed region using percentiles robust_q.
    """
    b_lo, b_hi = max(b_grid[0], b_hat - b_pad), min(b_grid[-1], b_hat + b_pad)
    a_lo, a_hi = max(a_grid[0], a_hat - a_pad), min(a_grid[-1], a_hat + a_pad)

    if vmin is None or vmax is None:
        if robust_from_zoom:
            j_lo = int(np.searchsorted(b_grid, b_lo, side="left"))
            j_hi = int(np.searchsorted(b_grid, b_hi, side="right"))
            i_lo = int(np.searchsorted(a_grid, a_lo, side="left"))
            i_hi = int(np.searchsorted(a_grid, a_hi, side="right"))
            dz = delta[i_lo:i_hi, j_lo:j_hi]

            q_low, q_high = robust_q
            vmin_auto = float(np.nanpercentile(dz, q_low))
            vmax_auto = float(np.nanpercentile(dz, q_high))

            # Ensure vmax has headroom above highest contour to avoid flattening near contours
            vmax_auto = max(vmax_auto, max(levels) * 2.0)

            vmin = vmin_auto if vmin is None else vmin
            vmax = vmax_auto if vmax is None else vmax
        else:
            # fallback to global scaling if requested
            vmin = float(np.min(delta)) if vmin is None else vmin
            vmax = float(np.max(delta)) if vmax is None else vmax

    plt.figure()
    plt.imshow(
        delta,
        origin="lower",
        extent=[b_grid[0], b_grid[-1], a_grid[0], a_grid[-1]],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=r"$\Delta\chi^2$")

    cs = plt.contour(b_grid, a_grid, delta, levels=levels)
    # Label contours with confidence tags if levels match Wilks thresholds
    fmt = {}
    for lev in levels:
        if abs(lev - 2.30) < 1e-6:
            fmt[lev] = "68%"
        elif abs(lev - 5.99) < 1e-6:
            fmt[lev] = "95%"
        else:
            fmt[lev] = f"{lev:g}"
    plt.clabel(cs, inline=True, fontsize=9, fmt=fmt)

    plt.scatter([b_true_grid], [a_true_grid], s=70, marker="x", label="truth (grid)")
    plt.scatter([b_hat], [a_hat], s=70, marker="o", label="best fit (grid)")

    plt.xlim(b_lo, b_hi)
    plt.ylim(a_lo, a_hi)

    plt.xlabel("b")
    plt.ylabel("a")
    plt.title(title)
    plt.legend()
    plt.show()


def run_two_parameter_demo(
    a_true=2.0,
    b_true=-0.6,
    sigma=0.20,
    n_points=41,
    seed=321,
    a_grid=np.linspace(0.5, 3.5, 161),
    b_grid=np.linspace(-1.5, 1.5, 161),
):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_points)

    # Snap truth to grid
    a_true_grid = snap_to_grid(a_true, a_grid)
    b_true_grid = snap_to_grid(b_true, b_grid)

    y_true = model_y_2p(a_true_grid, b_true_grid, x)
    y_obs = y_true + rng.normal(0.0, sigma, size=x.shape)

    chi2_ab, (a_hat, b_hat), chi2_min = chi2_grid_scan_2p(y_obs, x, sigma, a_grid, b_grid)
    delta = chi2_ab - chi2_min

    # Wilks thresholds for 2 parameters
    dchi2_68 = 2.30
    dchi2_95 = 5.99

    print("=== 2-parameter demo: y = a x^2 + b x ===")
    print(f"Truth (grid): a_true_grid = {a_true_grid:.4f}, b_true_grid = {b_true_grid:.4f}")
    print(f"Best fit:     a_hat       = {a_hat:.5f}, b_hat       = {b_hat:.5f}")
    print(f"chi2_min                  = {chi2_min:.2f}")
    print(f"Contours: 68% Δχ²={dchi2_68:.2f}, 95% Δχ²={dchi2_95:.2f}")

    # Plot data + selected templates
    plot_2p_data_and_templates(x, y_obs, sigma, a_true_grid, b_true_grid, a_hat, b_hat, a_grid, b_grid)

    # (A) Contrast-scaled colormap from zoom region (moderate zoom)
    plot_delta_chi2_zoomed(
        delta=delta,
        a_grid=a_grid,
        b_grid=b_grid,
        a_hat=a_hat,
        b_hat=b_hat,
        a_true_grid=a_true_grid,
        b_true_grid=b_true_grid,
        levels=[dchi2_68, dchi2_95],
        b_pad=0.6,
        a_pad=0.6,
        title=r"2D scan: $\Delta\chi^2(a,b)$ with contrast-scaled colormap (zoomed)",
        robust_from_zoom=True,
    )

    # (B) Larger zoom window + robust scaling
    plot_delta_chi2_zoomed(
        delta=delta,
        a_grid=a_grid,
        b_grid=b_grid,
        a_hat=a_hat,
        b_hat=b_hat,
        a_true_grid=a_true_grid,
        b_true_grid=b_true_grid,
        levels=[dchi2_68, dchi2_95],
        b_pad=0.9,
        a_pad=0.9,
        title=r"Larger zoom: $\Delta\chi^2(a,b)$ with 68% and 95% contours",
        robust_from_zoom=True,
    )

    # (C) Larger zoom window + tight fixed color scale to highlight the basin
    plot_delta_chi2_zoomed(
        delta=delta,
        a_grid=a_grid,
        b_grid=b_grid,
        a_hat=a_hat,
        b_hat=b_hat,
        a_true_grid=a_true_grid,
        b_true_grid=b_true_grid,
        levels=[dchi2_68, dchi2_95],
        b_pad=0.9,
        a_pad=0.9,
        title=r"Tight color scale: $\Delta\chi^2(a,b)$ (vmax=25) with 68% and 95% contours",
        vmin=0.0,
        vmax=25.0,
        robust_from_zoom=False,
    )

    return {
        "x": x,
        "sigma": sigma,
        "a_grid": a_grid,
        "b_grid": b_grid,
        "y_true": y_true,
        "y_obs": y_obs,
        "a_true_grid": a_true_grid,
        "b_true_grid": b_true_grid,
        "a_hat": a_hat,
        "b_hat": b_hat,
        "chi2_ab": chi2_ab,
        "chi2_min": chi2_min,
        "delta": delta,
        "delta_chi2_levels": {"68%": dchi2_68, "95%": dchi2_95},
    }


# ============================================================
# Run demos (edit sigma to study impact)
# ============================================================
if __name__ == "__main__":
    # 1-parameter
    res1 = run_one_parameter_demo(
        a_true=2.0,
        sigma=0.20,     # change this to see uncertainty scaling
        n_points=41,
        a_min=0.0,
        a_max=4.0,
        n_a=201,
        seed=123,
    )

    # 2-parameter
    res2 = run_two_parameter_demo(
        a_true=2.0,
        b_true=-0.6,
        sigma=0.20,     # change this to see the contour region expand/shrink
        n_points=41,
        seed=321,
        a_grid=np.linspace(0.5, 3.5, 161),
        b_grid=np.linspace(-1.5, 1.5, 161),
    )
