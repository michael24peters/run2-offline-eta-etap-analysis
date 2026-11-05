"""
fit_gauss.py

Fit a Gaussian signal + 2nd-order Chebyshev background to a distribution of
mass values, using iminuit ExtendedBinnedNLL.

This is a library module: it takes an array of numbers and fits it. Reading
ROOT files, applying selection, and reducing to one candidate per event all
happen upstream in driver_mass_fit.py via cut_utils.
"""

import os
import sys

import numpy as np

# Ensure src/utils is importable regardless of working directory
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics


def fit_gauss(values, xmin=500.0, xmax=600.0, nbins=100,
              mean_init=kinematics.M_ETA, sigma_init=5.0, fixed_params=None):
    """
    Run the Gaussian + background fit and return (result, m).

    Arguments:
    ----------
    values : 1-D numpy array of values to fit.
    xmin, xmax, nbins : float, float, int : The range and number of bins for the
             histogram.
    mean_init, sigma_init : float : Initial values for the Gaussian parameters.

    Returns:
    --------
    result : JSON-serialisable dict.
    m      : the final iminuit.Minuit object.
    """
    values = np.asarray(values, dtype=float)

    # Get histogram of values in the window.
    centers, edges, counts, errors = fit_utils.make_histogram(
        values, xmin=xmin, xmax=xmax, nbins=nbins)
    n_total = counts.sum()
    print(f'[INFO] {int(n_total)} entries in [{xmin}, {xmax}] across {nbins} bins',
          file=sys.stderr)

    # Fit
    cost = fit_utils.make_gauss_cost(counts, edges, xmin, xmax)

    init = dict(
        mean  = mean_init,
        sigma = sigma_init,
        n_sig = n_total * 0.5,
        n_bkg = n_total * 0.5,
        **fit_utils.BKG_INIT,
    )
    if fixed_params:
        init.update(fixed_params)
    limits = dict(
        n_sig=(0.0, None),
        n_bkg=(0.0, None),
    )

    # Stage 1: fit with the background params fixed at their initial values
    # (also anything in fixed_params).
    # Stage 2: float everything except fixed_params, starting from stage 1's
    # result.
    stage1_fixed = set(fit_utils.BKG_PARAMS) | set(fixed_params or {})
    stage1 = fit_utils.run_fit(cost, init, limits, fixed=list(stage1_fixed))
    init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
                 n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    m = fit_utils.run_fit(cost, init2, limits, fixed=list(fixed_params) if fixed_params else None)
    p, e = m.values, m.errors

    print(f"[INFO] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)

    # Bin predictions
    bw = (xmax - xmin) / nbins

    sig_cdf_edges = fit_utils.gauss_cdf(edges, xmin, xmax, p['mean'], p['sigma'])
    bkg_cdf_edges = fit_utils.cheb_cdf(edges, xmin, xmax, p['c0'], p['c1'])
    bin_sig = np.diff(sig_cdf_edges) * p['n_sig']
    bin_bkg = np.diff(bkg_cdf_edges) * p['n_bkg']
    bin_fit = bin_sig + bin_bkg

    # Goodness of fit, over the filled mass bins
    chi2 = fit_utils.neyman_chi2(counts, bin_fit)
    ndof = int((counts > 0).sum()) - m.nfit
    chi2_per_ndof = chi2 / ndof if ndof > 0 else float('nan')
    pulls = fit_utils.data_pulls(counts, bin_fit)

    print(f'[INFO] χ²/ndf = {chi2_per_ndof:.4f}  ({chi2:.2f} / {ndof})',
          file=sys.stderr)

    # Smooth curves for plotting
    x_curve   = np.linspace(xmin, xmax, 200)
    curve_sig = fit_utils.gauss_pdf(x_curve, xmin, xmax, p['mean'], p['sigma']) * p['n_sig'] * bw
    curve_bkg = fit_utils.cheb_bkg_pdf(x_curve, xmin, xmax, p['c0'], p['c1']) * p['n_bkg'] * bw
    curve_fit = curve_sig + curve_bkg

    # Build JSON
    result = {
        'meta': {
            'n_input': int(values.size),
            'model': 'gauss',
            'xmin': xmin,
            'xmax': xmax,
            'nbins': nbins,
        },
        'fit': {
            'converged': bool(m.valid),
            'valid': bool(m.valid),
            'fval': round(float(m.fval), 4),
            'chi2': round(chi2, 4),
            'ndof': ndof,
            'chi2_per_ndof': round(chi2_per_ndof, 6),
            'n_free_params': m.nfit,
            'parameters': {
                'mean':  {'value': round(float(p['mean']),  6),
                          'error': round(float(e['mean']),  6), 'unit': 'MeV'},
                'sigma': {'value': round(float(p['sigma']), 6),
                          'error': round(float(e['sigma']), 6), 'unit': 'MeV'},
                'n_sig': {'value': round(float(p['n_sig']), 2),
                          'error': round(float(e['n_sig']), 2)},
                'n_bkg': {'value': round(float(p['n_bkg']), 2),
                          'error': round(float(e['n_bkg']), 2)},
                **{name: {'value': round(float(p[name]), 6),
                          'error': round(float(e[name]), 6)}
                   for name in fit_utils.BKG_PARAMS},
            },
        },
        'histogram': {
            'bin_centers': centers.tolist(),
            'bin_counts':  counts.tolist(),
            'bin_errors':  errors.tolist(),
            'bin_fit':     bin_fit.tolist(),
            'bin_sig':     bin_sig.tolist(),
            'bin_bkg':     bin_bkg.tolist(),
            'bin_pulls':   pulls.tolist(),
        },
        'curve': {
            'x':   x_curve.tolist(),
            'fit': curve_fit.tolist(),
            'sig': curve_sig.tolist(),
            'bkg': curve_bkg.tolist(),
        },
    }
    return result, m
