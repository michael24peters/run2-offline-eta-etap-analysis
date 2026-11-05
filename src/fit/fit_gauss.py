"""
fit_gauss.py

Fit a Gaussian signal + 2nd-order Chebyshev background to a distribution of mass values, using
iminuit ExtendedBinnedNLL.
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
    xmin, xmax, nbins : float, float, int : The range and number of bins for the histogram.
    mean_init, sigma_init : float : Initial values for the Gaussian parameters.

    Returns:
    --------
    result : JSON-serialisable dict.
    m      : the final iminuit.Minuit object.
    """
    values = np.asarray(values, dtype=float)

    # Histogram of the values in the window
    counts, edges = np.histogram(values, bins=nbins, range=(xmin, xmax))
    counts = counts.astype(float)
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

    # Stage 1: fit with the background params fixed at their initial values (also anything in
    # fixed_params). Stage 2: float everything except fixed_params, starting from stage 1's result.
    stage1_fixed = set(fit_utils.BKG_PARAMS) | set(fixed_params or {})
    stage1 = fit_utils.run_fit(cost, init, limits, fixed=list(stage1_fixed))
    m = fit_utils.run_fit(cost, stage1.values.to_dict(), limits,
                          fixed=list(fixed_params) if fixed_params else None)
    p, e = m.values, m.errors

    print(f"[INFO] valid={m.valid}  "
          f"mean={p['mean']:.3f}±{e['mean']:.3f} MeV  "
          f"σ={p['sigma']:.3f}±{e['sigma']:.3f} MeV",
          file=sys.stderr)

    # Expected counts per mass bin, goodness of fit, and the result
    prediction = fit_utils.bin_prediction(m, counts, edges, xmin, xmax, fit_utils.gauss_cdf,
                                          ('mean', 'sigma'), fit_utils.cheb_cdf)
    result = fit_utils.fit_result(m, 'gauss', values.size, edges, counts, prediction)
    fit = result['fit']
    print(f"[INFO] χ²/ndf = {fit['chi2_per_ndof']:.4f}  ({fit['chi2']:.2f} / {fit['ndof']})",
          file=sys.stderr)
    return result, m
