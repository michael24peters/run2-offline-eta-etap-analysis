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
    fixed_params : optional dict {param_name: value}, any of mean, sigma, n_sig, n_bkg, c0, c1.
             Each is seeded at the given value and held fixed, e.g. {'n_bkg': 0, 'c0': 0, 'c1': 0}
             for no background.

    The fit is one MIGRAD + HESSE stage (fit_utils.run_fit) with every other parameter floating,
    starting at mean_init, sigma_init, n_sig = n_bkg = half the entries in the window and
    c0 = c1 = 0.

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
        sigma=(0.0, None),
        n_sig=(0.0, None),
        n_bkg=(0.0, None),
    )

    # One stage: every parameter floats except fixed_params, held at their given values.
    m = fit_utils.run_fit(cost, init, limits, fixed=list(fixed_params) if fixed_params else None)
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
