"""
fit_dcb.py

Fit an asymmetric Double Crystal Ball (DCB) signal + 2nd-order Chebyshev
background to a distribution of mass values, using iminuit ExtendedBinnedNLL.

Tail parameters for left and right sides are independent: α_L, n_L, α_R, n_R.

This is a library module: it takes an array of numbers and fits it. Reading
ROOT files, applying selection, and reducing to one candidate per event all
happen upstream in driver_dcb.py via cut_utils.
"""

import sys
import os
import numpy as np
from iminuit.cost import NormalConstraint
# Ensure utils/fit_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics

# The DCB tail-shape parameters, as a group.
TAIL_PARAMS = ('alphaL', 'nL', 'alphaR', 'nR')


def fit_dcb(values, xmin=480.0, xmax=620.0, nbins=80,
            mean_init=kinematics.M_ETA, sigma_init=5.0,
            alphaL_init=1.5, nL_init=2.0,
            alphaR_init=1.5, nR_init=2.0,
            c0_init=0.0, c1_init=0.0,
            mean_prior=None, sigma_prior=None, fixed_params=None):
    """
    Run the asymmetric DCB + Chebyshev fit and return (result, m).

    Arguments:
    ----------
    values : 1-D numpy array of values to fit.
    xmin, xmax, nbins : float, float, int : The range and number of bins for the
             histogram.
    mean_init, sigma_init : float : Initial values for the DCB parameters.
    alphaL_init, nL_init : float : Initial values for the left tail parameters.
    alphaR_init, nR_init : float : Initial values for the right tail parameters.
    c0_init, c1_init : float : Initial values for the Chebyshev coefficients.
    mean_prior, sigma_prior : optional (value, error) in MeV; see fit_gauss().
    fixed_params : optional dict {param_name: value}, any of mean, sigma,
             alphaL, nL, alphaR, nR, n_sig, n_bkg, c0, c1. Each is seeded at
             the given value and held fixed through both fit stages instead
             of floating.

    Returns:
    --------
    result : JSON-serialisable dict.
    m      : the final iminuit.Minuit object.
    """
    values = np.asarray(values, dtype=float)

    centers, bin_edges, counts, errors = fit_utils.make_histogram(
        values, xmin=xmin, xmax=xmax, nbins=nbins)
    n_total = counts.sum()
    print(f'[INFO] {int(n_total)} entries in [{xmin}, {xmax}] across {nbins} bins',
          file=sys.stderr)

    # Fit
    cost = fit_utils.make_dcb_cost(counts, bin_edges, xmin, xmax)
    if mean_prior is not None:
        cost = cost + NormalConstraint('mean', mean_prior[0], error=mean_prior[1])
    if sigma_prior is not None:
        cost = cost + NormalConstraint('sigma', sigma_prior[0], error=sigma_prior[1])

    init = dict(
        mean   = mean_init,
        sigma  = sigma_init,
        alphaL = alphaL_init,
        nL     = nL_init,
        alphaR = alphaR_init,
        nR     = nR_init,
        n_sig  = n_total * 0.5,
        n_bkg  = n_total * 0.5,
        c0     = c0_init,
        c1     = c1_init,
    )
    if fixed_params:
        init.update(fixed_params)
    limits = dict(
        sigma=(0.0, None), alphaL=(0.0, None), alphaR=(0.0, None),
        nL=(0.0, None), nR=(0.0, None),
        n_sig=(0.0, None), n_bkg=(0.0, None),
    )

    # Stage 1: fit with c0, c1, and the tail params fixed at their init
    # values (also anything in fixed_params).
    # Stage 2: float everything except fixed_params, starting from stage 1's
    # result.
    # TODO: Try just fixing c0, c1 (also fixed_params) for stage 1.
    # stage1_fixed = set(TAIL_PARAMS) | {'c0', 'c1'} | set(fixed_params or {})
    stage1_fixed = {'c0', 'c1'} | set(fixed_params or {})
    stage1 = fit_utils.run_fit(cost, init, limits, fixed=list(stage1_fixed))
    # init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
    #             n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
                 alphaL=stage1.values['alphaL'], nL=stage1.values['nL'],
                 alphaR=stage1.values['alphaR'], nR=stage1.values['nR'],
                 n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    m = fit_utils.run_fit(cost, init2, limits, fixed=list(fixed_params) if fixed_params else None)
    p, e = m.values, m.errors

    print(f"[INFO] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)
    print(f"[INFO] α_L={p['alphaL']:.3f}±{e['alphaL']:.3f}  n_L={p['nL']:.3f}±{e['nL']:.3f}",
          file=sys.stderr)
    print(f"[INFO] α_R={p['alphaR']:.3f}±{e['alphaR']:.3f}  n_R={p['nR']:.3f}±{e['nR']:.3f}",
          file=sys.stderr)

    # Compute bin predictions. Integrating the PDF over each bin (CDF at the
    # upper edge minus the lower edge) is exact, unlike sampling the PDF at bin
    # centres, which biases bins where the PDF has curvature (i.e. the peak).
    bw = (xmax - xmin) / nbins
    sig_cdf_edges = fit_utils.dcb_cdf(bin_edges, xmin, xmax, p['mean'], p['sigma'],
                                    p['alphaL'], p['nL'], p['alphaR'], p['nR'])
    bkg_cdf_edges = fit_utils.cheb_cdf(bin_edges, xmin, xmax, p['c0'], p['c1'])
    # Both CDFs run 0 -> 1 across the window, so np.diff gives the fraction of
    # each component in each bin; scale by the yields to get expected counts.
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
    x_curve = np.linspace(xmin, xmax, 200)
    curve_sig = fit_utils.dcb_pdf(x_curve, xmin, xmax, p['mean'], p['sigma'],
                                 p['alphaL'], p['nL'], p['alphaR'], p['nR']) * p['n_sig'] * bw
    curve_bkg = fit_utils.cheb_bkg_pdf(x_curve, xmin, xmax, p['c0'], p['c1']) * p['n_bkg'] * bw
    curve_fit = curve_sig + curve_bkg

    # Build JSON. Everything a plotting script needs travels in the result, so
    # plotting never has to re-open the ROOT file or re-run the fit: `meta` for
    # provenance, `fit` for the numbers, `histogram` for the points/residuals,
    # and `curve` for the overlays. Arrays are .tolist()'d since numpy types
    # are not JSON-serialisable.
    result = {
        'meta': {
            'n_input': int(values.size),
            'model': 'dcb_asym',
            'xmin': xmin,
            'xmax': xmax,
            'nbins': nbins,
            'symmetric': False,  # left/right tails floated independently
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
                'mean':  {'value': round(float(p['mean']), 6),
                          'error': round(float(e['mean']), 6), 'unit': 'MeV'},
                'sigma': {'value': round(float(p['sigma']), 6),
                          'error': round(float(e['sigma']), 6), 'unit': 'MeV'},
                'alphaL':{'value': round(float(p['alphaL']), 6),
                          'error': round(float(e['alphaL']), 6)},
                'nL':    {'value': round(float(p['nL']), 6),
                          'error': round(float(e['nL']), 6)},
                'alphaR':{'value': round(float(p['alphaR']), 6),
                          'error': round(float(e['alphaR']), 6)},
                'nR':    {'value': round(float(p['nR']), 6),
                          'error': round(float(e['nR']), 6)},
                'n_sig': {'value': round(float(p['n_sig']), 2),
                          'error': round(float(e['n_sig']), 2),
                          'note': 'DCB integral normalized over the full [xmin, xmax] window'},
                'n_bkg': {'value': round(float(p['n_bkg']), 2),
                          'error': round(float(e['n_bkg']), 2)},
                'c0':    {'value': round(float(p['c0']), 6),
                          'error': round(float(e['c0']), 6),
                          'note': 'Chebyshev linear coefficient (slope)'},
                'c1':    {'value': round(float(p['c1']), 6),
                          'error': round(float(e['c1']), 6),
                          'note': 'Chebyshev quadratic coefficient (curvature)'},
            },
        },
        'histogram': {
            'bin_centers': centers.tolist(),
            'bin_counts': counts.tolist(),
            'bin_errors': errors.tolist(),
            'bin_fit': bin_fit.tolist(),
            'bin_sig': bin_sig.tolist(),
            'bin_bkg': bin_bkg.tolist(),
            'bin_pulls': pulls.tolist(),
        },
        'curve': {
            'x': x_curve.tolist(),
            'fit': curve_fit.tolist(),
            'sig': curve_sig.tolist(),
            'bkg': curve_bkg.tolist(),
        },
    }
    return result, m
