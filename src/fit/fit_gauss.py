"""
fit_gauss.py

Fit a Gaussian signal + 2nd-order Chebyshev background to a distribution of
mass values, using iminuit ExtendedBinnedNLL.

This is a library module: it takes an array of numbers and fits it. Reading
ROOT files, applying selection, and reducing to one candidate per event all
happen upstream in driver_gauss.py via cut_utils.
"""

import sys
import os
import numpy as np
from iminuit.cost import NormalConstraint

# Ensure src/utils is importable regardless of working directory
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics


def fit_gauss(values, xmin=480.0, xmax=620.0, nbins=80,
              mean_init=kinematics.M_ETA, sigma_init=5.0,
              bkg_model='cheb2', bkg_init=None,
              mean_prior=None, sigma_prior=None, fixed_params=None):
    """
    Run the Gaussian + background fit and return (result, m).

    Arguments:
    ----------
    values : 1-D numpy array of values to fit.
    xmin, xmax, nbins : float, float, int : The range and number of bins for the
             histogram.
    mean_init, sigma_init : float : Initial values for the Gaussian parameters.
    bkg_model : one of fit_utils.BACKGROUND_MODELS's keys ('cheb2' (default,
             prior behavior), 'cheb_exp'). See fit_utils.py for what each
             shape is and why.
    bkg_init : optional dict overriding the chosen model's default initial
             values for its own parameters (e.g. {'c0': 0.3} for 'cheb2'),
             merged over fit_utils.BACKGROUND_MODELS[bkg_model].init. These
             still float in stage 2 -- for holding a background parameter
             fixed instead, use fixed_params.
    mean_prior, sigma_prior : optional (value, error). Adds
             NormalConstraint(name, value, error=error) to the cost. None
             (default) adds no constraint.

    Returns:
    --------
    result : JSON-serialisable dict.
    m      : the final iminuit.Minuit object.
    """
    values = np.asarray(values, dtype=float)
    model = fit_utils.BACKGROUND_MODELS[bkg_model]

    # Get histogram of values in the window.
    centers, edges, counts, errors = fit_utils.make_histogram(
        values, xmin=xmin, xmax=xmax, nbins=nbins)
    n_total = counts.sum()
    print(f'[INFO] {int(n_total)} entries in [{xmin}, {xmax}] across {nbins} bins',
          file=sys.stderr)

    # Fit
    cost = fit_utils.make_gauss_cost(counts, edges, xmin, xmax, bkg=bkg_model)
    if mean_prior is not None:
        cost = cost + NormalConstraint('mean', mean_prior[0], error=mean_prior[1])
    if sigma_prior is not None:
        cost = cost + NormalConstraint('sigma', sigma_prior[0], error=sigma_prior[1])

    init = dict(
        mean  = mean_init,
        sigma = sigma_init,
        n_sig = n_total * 0.5,
        n_bkg = n_total * 0.5,
        **{**model.init, **(bkg_init or {})},
    )
    if fixed_params:
        init.update(fixed_params)
    limits = dict(
        n_sig=(0.0, None),
        n_bkg=(0.0, None),
        **model.limits,
    )

    # Stage 1: fit with the background params fixed at their initial values.
    # Stage 2: float them, starting from stage 1's result.
    stage1 = fit_utils.run_fit(cost, init, limits, fixed=list(model.param_names))
    init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
                 n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    m = fit_utils.run_fit(cost, init2, limits)
    p, e = m.values, m.errors

    print(f"[INFO] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)

    # Bin predictions
    bw = (xmax - xmin) / nbins

    sig_cdf_edges = fit_utils.gauss_cdf(edges, xmin, xmax, p['mean'], p['sigma'])
    bkg_cdf_edges = model.cdf(edges, xmin, xmax, *(p[n] for n in model.param_names))
    bin_sig = np.diff(sig_cdf_edges) * p['n_sig']
    bin_bkg = np.diff(bkg_cdf_edges) * p['n_bkg']
    bin_fit = bin_sig + bin_bkg

    # Goodness of fit: Neyman (data-based) chi2, with the data as the variance
    # estimate. The denominator is the measurement uncertainty rather than a fit
    # uncertainty.
    mask = counts > 0
    chi2 = float(np.sum((counts[mask] - bin_fit[mask])**2 / counts[mask]))
    ndof = int(mask.sum()) - m.nfit
    chi2_per_ndof = chi2 / ndof if ndof > 0 else float('nan')

    # Data-based pulls: (d - f) / sqrt(d), 0 for empty bins.
    pulls = np.where(counts > 0, (counts - bin_fit) / np.sqrt(counts), 0.0)

    print(f'[INFO] χ²/ndf = {chi2_per_ndof:.4f}  ({chi2:.2f} / {ndof})',
          file=sys.stderr)

    # Smooth curves for plotting
    x_curve   = np.linspace(xmin, xmax, 200)
    curve_sig = fit_utils.gauss_pdf(x_curve, xmin, xmax, p['mean'], p['sigma']) * p['n_sig'] * bw
    curve_bkg = model.pdf(x_curve, xmin, xmax, *(p[n] for n in model.param_names)) * p['n_bkg'] * bw
    curve_fit = curve_sig + curve_bkg

    # Build JSON
    result = {
        'meta': {
            'n_input': int(values.size),
            'model': 'gauss',
            'background_model': bkg_model,
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
                   for name in model.param_names},
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
