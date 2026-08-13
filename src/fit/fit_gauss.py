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

# Ensure src/utils is importable regardless of working directory
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils


def fit_gauss(values, xmin=480.0, xmax=620.0, nbins=80,
              mean_init=547.9, sigma_init=10.0):
    """
    Run the Gaussian + Chebyshev fit and return the result dict.

    values : 1-D numpy array of masses [MeV], one per event. This function
             applies no selection of its own -- it fits whatever it is given.
    """
    values = np.asarray(values, dtype=float)

    centers, counts, errors = fit_utils.make_histogram(
        values, xmin=xmin, xmax=xmax, nbins=nbins)
    bin_edges = np.linspace(xmin, xmax, nbins + 1)
    N_CAND = counts.sum()
    print(f'[info] {int(N_CAND)} entries in [{xmin}, {xmax}] MeV across {nbins} bins',
          file=sys.stderr)

    # Fit
    cost = fit_utils.make_gauss_cost(counts, bin_edges, xmin, xmax)

    init = dict(
        mean  = mean_init,
        sigma = sigma_init,
        n_sig = N_CAND * 0.5,
        n_bkg = N_CAND * 0.5,
        c0    = 0.0,
        c1    = 0.0,
    )
    limits = dict(
        mean  = (xmin, xmax),
        sigma = (1.0, 50.0),
        n_sig = (0.0, N_CAND * 2),
        n_bkg = (0.0, N_CAND * 2),
        c0    = (-1.0, 1.0),
        c1    = (-1.4, 1.4),
    )

    m = fit_utils.run_fit(cost, init, limits)
    p, e = m.values, m.errors

    n_bkg = p['n_bkg']
    n_bkg_err = e['n_bkg']

    print(f"[fit] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)

    # Bin predictions
    bw = (xmax - xmin) / nbins

    sig_cdf_edges = fit_utils.gauss_cdf(bin_edges, p['mean'], p['sigma'], xmin, xmax)
    bkg_cdf_edges = fit_utils.cheb_cdf(bin_edges, xmin, xmax, p['c0'], p['c1'])
    bin_sig = np.diff(sig_cdf_edges) * p['n_sig']
    bin_bkg = np.diff(bkg_cdf_edges) * n_bkg
    bin_fit = bin_sig + bin_bkg

    # Goodness of fit
    mask = bin_fit > 0
    chi2 = float(np.sum((counts[mask] - bin_fit[mask])**2 / bin_fit[mask]))
    ndof = int(mask.sum()) - m.nfit
    chi2_per_ndof = chi2 / ndof if ndof > 0 else float('nan')

    # Data-based pulls: (d − f) / √d  (RooFit pull plots; 0 for empty bins)
    # pulls = np.where(counts > 0, (counts - bin_fit) / np.sqrt(counts), 0.0)
    
    # Per-bin Pearson pull: (d − f) / √f, model error in the denominator so that
    # Σ pull² == the Pearson chi2 reported above (0 for empty bins). Note the fit
    # itself minimizes a Poisson binned NLL, whose exact residual is the
    # Baker–Cousins deviance sign(d−f)·√(2[f − d + d·ln(d/f)]); at these per-bin
    # counts (1e4+) Pearson and deviance pulls are numerically identical, so this
    # simpler form is used.
    pulls = np.where(bin_fit > 0, (counts - bin_fit) / np.sqrt(bin_fit), 0.0)

    print(f'[fit] χ²/ndf = {chi2_per_ndof:.4f}  ({chi2:.2f} / {ndof})',
          file=sys.stderr)

    # Smooth curve
    x_curve   = np.linspace(xmin, xmax, 200)
    curve_sig = fit_utils.gauss_pdf(x_curve, p['mean'], p['sigma'], xmin, xmax) * p['n_sig'] * bw
    curve_bkg = fit_utils.cheb_bkg_pdf(x_curve, xmin, xmax, p['c0'], p['c1']) * n_bkg * bw
    curve_fit = curve_sig + curve_bkg

    # Build JSON
    result = {
        'meta': {
            'n_input':    int(values.size),
            'model':      'gauss',
            'xmin':  xmin,
            'xmax':  xmax,
            'nbins': nbins,
            'symmetric': False,
        },
        'fit': {
            'converged':      bool(m.valid),
            'valid':          bool(m.valid),
            'fval':           round(float(m.fval), 4),
            'chi2':           round(chi2, 4),
            'ndof':           ndof,
            'chi2_per_ndof':  round(chi2_per_ndof, 6),
            'n_free_params':  m.nfit,
            'parameters': {
                'mean':  {'value': round(float(p['mean']),  6),
                          'error': round(float(e['mean']),  6), 'unit': 'MeV'},
                'sigma': {'value': round(float(p['sigma']), 6),
                          'error': round(float(e['sigma']), 6), 'unit': 'MeV'},
                'n_sig': {'value': round(float(p['n_sig']), 2),
                          'error': round(float(e['n_sig']), 2)},
                'n_bkg': {'value': round(float(n_bkg), 2),
                          'error': round(float(n_bkg_err), 2)},
                'c0':    {'value': round(float(p['c0']),    6),
                          'error': round(float(e['c0']),    6),
                          'note': 'Chebyshev linear coefficient (slope)'},
                'c1':    {'value': round(float(p['c1']),    6),
                          'error': round(float(e['c1']),    6),
                          'note': 'Chebyshev quadratic coefficient (curvature)'},
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
    return result
