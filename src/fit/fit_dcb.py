"""
fit_dcb.py

Fit an asymmetric Double Crystal Ball (DCB) signal + 2nd-order Chebyshev
background to a distribution of mass values, using iminuit ExtendedBinnedNLL.

Tail parameters for left and right sides are independent: α_L, n_L, α_R, n_R.
Minuit varies inv_nL = 1/n_L and inv_nR = 1/n_R in place of n_L and n_R:
inv_n = 0 is an exponential tail (n -> infinity). Inputs (init values,
fixed_params, tail_prior) and the result dict use n_L and n_R.

This is a library module: it takes an array of numbers and fits it. Reading
ROOT files, applying selection, and reducing to one candidate per event all
happen upstream in driver_mass_fit.py via cut_utils.
"""

import os
import sys

import numpy as np
from iminuit.cost import ExtendedBinnedNLL, NormalConstraint

# Ensure utils/fit_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics

# The DCB tail-shape parameters, as a group.
TAIL_PARAMS = ('alphaL', 'nL', 'alphaR', 'nR')

# The same, as Minuit varies them: n replaced by inv_n = 1/n.
FIT_TAIL_PARAMS = ('alphaL', 'inv_nL', 'alphaR', 'inv_nR')
INV_N = {'nL': 'inv_nL', 'nR': 'inv_nR'}

# inv_n below this is evaluated as n = 1 / INV_N_MIN, an exponential tail to
# numerical precision.
INV_N_MIN = 1e-5
# Upper limit of inv_n in the fit (n >= 0.1).
INV_N_MAX = 10.0


def tail_n(inv_n):
    """n of a DCB tail from inv_n = 1/n, with inv_n floored at INV_N_MIN."""
    return 1.0 / max(inv_n, INV_N_MIN)


def dcb_cdf_inv_n(x, xmin, xmax, mean, sigma, alphaL, inv_nL, alphaR, inv_nR):
    """fit_utils.dcb_cdf with each tail's n given as inv_n = 1/n."""
    return fit_utils.dcb_cdf(x, xmin, xmax, mean, sigma,
                             alphaL, tail_n(inv_nL), alphaR, tail_n(inv_nR))


def fit_dcb(values, xmin=480.0, xmax=620.0, nbins=80,
            mean_init=kinematics.M_ETA, sigma_init=5.0,
            alphaL_init=1.5, nL_init=2.0,
            alphaR_init=1.5, nR_init=2.0,
            tail_prior=None, fixed_params=None):
    """
    Run the asymmetric DCB + background fit and return (result, m).

    Arguments:
    ----------
    values : 1-D numpy array of values to fit.
    xmin, xmax, nbins : float, float, int : The range and number of bins for the
             histogram.
    mean_init, sigma_init : float : Initial values for the DCB parameters.
    alphaL_init, nL_init : float : Initial values for the left tail parameters.
    alphaR_init, nR_init : float : Initial values for the right tail parameters.
    tail_prior : optional (values, cov) for TAIL_PARAMS as a group -- `values`
             a length-4 array, `cov` its 4x4 covariance. Adds a
             NormalConstraint on FIT_TAIL_PARAMS (the same prior with nL, nR
             as inv_nL, inv_nR, covariance propagated) to the cost, so
             the tails float but are penalized for straying from `values`
             (e.g. an MC-derived tail shape) by an amount set by `cov` --
             looser than hard-fixing them via `fixed_params`.
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

    # Fit: extended binned likelihood of n_sig * DCB + n_bkg * Chebyshev, the
    # DCB tails given as alphaL, inv_nL, alphaR, inv_nR
    scaled_cdf, param_names = fit_utils._make_scaled_cdf(
        ('mean', 'sigma') + FIT_TAIL_PARAMS, dcb_cdf_inv_n, fit_utils.cheb_cdf, xmin, xmax)
    cost = ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)
    if tail_prior is not None:
        # tail_prior in (alphaL, inv_nL, alphaR, inv_nR): values inverted, and
        # the covariance propagated with d(1/n)/dn = -1/n^2
        prior_values = np.array(tail_prior[0], dtype=float)
        jacobian = np.array([1.0, -1.0 / prior_values[1]**2, 1.0, -1.0 / prior_values[3]**2])
        prior_values[1] = 1.0 / prior_values[1]
        prior_values[3] = 1.0 / prior_values[3]
        prior_cov = np.asarray(tail_prior[1], dtype=float) * np.outer(jacobian, jacobian)
        cost = cost + NormalConstraint(FIT_TAIL_PARAMS, prior_values, error=prior_cov)

    init = dict(
        mean   = mean_init,
        sigma  = sigma_init,
        alphaL = alphaL_init,
        inv_nL = 1.0 / nL_init,
        alphaR = alphaR_init,
        inv_nR = 1.0 / nR_init,
        n_sig  = n_total * 0.5,
        n_bkg  = n_total * 0.5,
        **fit_utils.BKG_INIT,
    )
    # fixed_params with nL, nR as inv_nL, inv_nR
    fixed = {}
    for name, value in (fixed_params or {}).items():
        if name in INV_N:
            fixed[INV_N[name]] = 1.0 / value
        else:
            fixed[name] = value
    init.update(fixed)
    limits = dict(
        sigma=(0.0, None), alphaL=(0.0, None), alphaR=(0.0, None),
        inv_nL=(0.0, INV_N_MAX), inv_nR=(0.0, INV_N_MAX),
        n_sig=(0.0, None), n_bkg=(0.0, None),
    )

    # Stage 1: fit with the background params, and the tail params fixed at
    # their init values (also anything in fixed_params).
    # Stage 2: float everything except fixed_params, starting from stage 1's
    # result.
    # TODO: Try just fixing the background params (also fixed_params) for stage 1.
    # stage1_fixed = set(TAIL_PARAMS) | set(fit_utils.BKG_PARAMS) | set(fixed_params or {})
    stage1_fixed = set(FIT_TAIL_PARAMS) | set(fixed)
    stage1 = fit_utils.run_fit(cost, init, limits, fixed=list(stage1_fixed))
    # init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
    #             n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
                 alphaL=stage1.values['alphaL'], inv_nL=stage1.values['inv_nL'],
                 alphaR=stage1.values['alphaR'], inv_nR=stage1.values['inv_nR'],
                 n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    m = fit_utils.run_fit(cost, init2, limits, fixed=list(fixed) or None)

    # Fitted values and errors, with inv_nL, inv_nR as nL, nR; error of n from
    # the error of inv_n by |dn/d(inv_n)| = n^2
    p, e = m.values.to_dict(), m.errors.to_dict()
    for name, inv_name in INV_N.items():
        p[name] = tail_n(p[inv_name])
        e[name] = e[inv_name] * p[name]**2

    print(f"[INFO] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)
    print(f"[INFO] α_L={p['alphaL']:.3f}±{e['alphaL']:.3f}  "
          f"1/n_L={p['inv_nL']:.3f}±{e['inv_nL']:.3f}  (n_L={p['nL']:.3f})",
          file=sys.stderr)
    print(f"[INFO] α_R={p['alphaR']:.3f}±{e['alphaR']:.3f}  "
          f"1/n_R={p['inv_nR']:.3f}±{e['inv_nR']:.3f}  (n_R={p['nR']:.3f})",
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

    # Goodness of fit, over the filled mass bins
    chi2 = fit_utils.neyman_chi2(counts, bin_fit)
    ndof = int((counts > 0).sum()) - m.nfit
    chi2_per_ndof = chi2 / ndof if ndof > 0 else float('nan')
    pulls = fit_utils.data_pulls(counts, bin_fit)

    print(f'[INFO] χ²/ndf = {chi2_per_ndof:.4f}  ({chi2:.2f} / {ndof})',
          file=sys.stderr)

    # Smooth curves for plotting
    x_curve = np.linspace(xmin, xmax, 200)
    curve_sig = fit_utils.dcb_pdf(x_curve, xmin, xmax, p['mean'], p['sigma'],
                                 p['alphaL'], p['nL'], p['alphaR'], p['nR']) * p['n_sig'] * bw
    curve_bkg = fit_utils.cheb_bkg_pdf(x_curve, xmin, xmax, p['c0'], p['c1']) * p['n_bkg'] * bw
    curve_fit = curve_sig + curve_bkg

    # Covariance, name-indexed so it survives the round trip through JSON, in
    # nL, nR: rows and columns of inv_nL, inv_nR times dn/d(inv_n) = -n^2.
    # None if HESSE's matrix inversion failed (m.covariance is then None too).
    covariance = None
    if m.covariance is not None:
        names = list(m.parameters)
        jacobian = np.ones(len(names))
        for name, inv_name in INV_N.items():
            jacobian[names.index(inv_name)] = -p[name]**2
            names[names.index(inv_name)] = name
        covariance = {
            'names': names,
            'matrix': (np.asarray(m.covariance) * np.outer(jacobian, jacobian)).tolist(),
        }

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
            'covariance': covariance,
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
                **{name: {'value': round(float(p[name]), 6),
                          'error': round(float(e[name]), 6)}
                   for name in fit_utils.BKG_PARAMS},
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
