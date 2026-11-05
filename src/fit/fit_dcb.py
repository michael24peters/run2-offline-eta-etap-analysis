"""
fit_dcb.py

Fits an asymmetric Double Crystal Ball (DCB) signal + 2nd-order Chebyshev background to a
distribution of mass values (as an array), using iminuit ExtendedBinnedNLL.


"""

import os
import sys

import numpy as np
from iminuit.cost import NormalConstraint

# Ensure utils/fit_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics

# The DCB tail-shape parameters, as a group.
TAIL_PARAMS = ('alphaL', 'nL', 'alphaR', 'nR')
# Default limits: width, tails and yields at least 0
DCB_LIMITS = {'sigma': (0.0, None), 'alphaL': (0.0, None), 'nL': (0.0, None),
              'alphaR': (0.0, None), 'nR': (0.0, None), 'n_sig': (0.0, None), 'n_bkg': (0.0, None)}


def fit_dcb(values, xmin=500.0, xmax=600.0, nbins=100,
            mean_init=kinematics.M_ETA, sigma_init=5.0,
            alphaL_init=1.5, nL_init=2.0,
            alphaR_init=1.5, nR_init=2.0,
            tail_prior=None, sigma_prior=None, fixed_params=None, limits=None):
    """
    Run the asymmetric DCB + background fit and return (result, m).

    Arguments:
    ----------
    values : 1-D numpy array of values to fit.
    xmin, xmax, nbins : float, float, int : The range and number of bins for the histogram.
    mean_init, sigma_init : float : Initial values for the DCB parameters.
    alphaL_init, nL_init : float : Initial values for the left tail parameters.
    alphaR_init, nR_init : float : Initial values for the right tail parameters.
    tail_prior : optional (values, cov) for TAIL_PARAMS as a group -- `values` a length-4
             array/dict, `cov` its 4x4 covariance. Adds
             NormalConstraint(TAIL_PARAMS, values, error=cov) to the cost, so the tails float but
             are penalized for straying from `values` (e.g. an MC-derived tail shape) by an amount
             set by `cov` -- looser than hard-fixing them via `fixed_params`.
    sigma_prior : optional (values, cov) for sigma -- `values` a length-1 array, `cov` its 1x1
             covariance. Adds NormalConstraint(('sigma',), values, error=cov) to the cost, as
             tail_prior does for the tails.
    fixed_params : optional dict {param_name: value}, any of mean, sigma, alphaL, nL, alphaR, nR,
             n_sig, n_bkg, c0, c1. Each is seeded at the given value and held fixed through both fit
             stages instead of floating.
    limits : optional dict {param_name: (lo, hi)} replacing the default limits of those parameters,
             e.g. {'nL': (0, 50), 'nR': (0, 50)}; either bound may be None.

    Returns:
    --------
    result : JSON-serialisable dict.
    m      : the final iminuit.Minuit object.
    """
    values = np.asarray(values, dtype=float)

    counts, bin_edges = np.histogram(values, bins=nbins, range=(xmin, xmax))
    counts = counts.astype(float)
    n_total = counts.sum()
    print(f'[INFO] {int(n_total)} entries in [{xmin}, {xmax}] across {nbins} bins',
          file=sys.stderr)

    # Fit
    cost = fit_utils.make_dcb_cost(counts, bin_edges, xmin, xmax)
    if tail_prior is not None:
        cost = cost + NormalConstraint(TAIL_PARAMS, tail_prior[0], error=tail_prior[1])
    if sigma_prior is not None:
        cost = cost + NormalConstraint(('sigma',), sigma_prior[0], error=sigma_prior[1])

    init = dict(
        mean   = mean_init,
        sigma  = sigma_init,
        alphaL = alphaL_init,
        nL     = nL_init,
        alphaR = alphaR_init,
        nR     = nR_init,
        n_sig  = n_total * 0.5,
        n_bkg  = n_total * 0.5,
        **fit_utils.BKG_INIT,
    )
    if fixed_params:
        init.update(fixed_params)
    # Default limits, with any given in `limits` replacing them
    all_limits = dict(DCB_LIMITS)
    if limits:
        all_limits.update(limits)

    # Stage 1: fit with the tail params fixed at their init values (also anything in fixed_params).
    # Stage 2: float everything except fixed_params, starting from stage 1's result.
    stage1_fixed = set(TAIL_PARAMS) | set(fixed_params or {})
    stage1 = fit_utils.run_fit(cost, init, all_limits, fixed=list(stage1_fixed))
    m = fit_utils.run_fit(cost, stage1.values.to_dict(), all_limits,
                          fixed=list(fixed_params) if fixed_params else None)
    p, e = m.values, m.errors

    print(f"[INFO] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)
    print(f"[INFO] α_L={p['alphaL']:.3f}±{e['alphaL']:.3f}  n_L={p['nL']:.3f}±{e['nL']:.3f}",
          file=sys.stderr)
    print(f"[INFO] α_R={p['alphaR']:.3f}±{e['alphaR']:.3f}  n_R={p['nR']:.3f}±{e['nR']:.3f}",
          file=sys.stderr)

    # Expected counts per mass bin, goodness of fit, and the result
    prediction = fit_utils.bin_prediction(m, counts, bin_edges, xmin, xmax, fit_utils.dcb_cdf,
                                          fit_utils.DCB_PARAMS, fit_utils.cheb_cdf)
    result = fit_utils.fit_result(m, 'dcb_asym', values.size, bin_edges, counts, prediction)
    fit = result['fit']
    print(f"[INFO] χ²/ndf = {fit['chi2_per_ndof']:.4f}  ({fit['chi2']:.2f} / {fit['ndof']})",
          file=sys.stderr)
    return result, m
