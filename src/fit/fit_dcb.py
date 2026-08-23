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
            mean_prior=None, sigma_prior=None, tail_prior=None):
    """
    Run the asymmetric DCB + Chebyshev fit and return (result, m).

    values : 1-D numpy array of masses [MeV], one per event.
    mean_prior, sigma_prior : optional (value, error) in MeV; see fit_gauss().
    tail_prior : optional (values, cov) for TAIL_PARAMS as a group -- `values`
             a length-4 array/dict, `cov` its 4x4 covariance. Adds
             NormalConstraint(TAIL_PARAMS, values, error=cov) to the cost.

    Returns:
    result : JSON-serialisable dict.
    m      : the final iminuit.Minuit object.
    """
    values = np.asarray(values, dtype=float)

    centers, counts, errors = fit_utils.make_histogram(
        values, xmin=xmin, xmax=xmax, nbins=nbins)
    bin_edges = np.linspace(xmin, xmax, nbins + 1)
    N_CAND = counts.sum()
    print(f'[INFO] {int(N_CAND)} entries in [{xmin}, {xmax}] MeV across {nbins} bins',
          file=sys.stderr)

    # Cost function: extended binned NLL comparing `counts` to the expected
    # events per bin from n_sig * DCB + n_bkg * Chebyshev.
    cost = fit_utils.make_dcb_cost(counts, bin_edges, xmin, xmax)  # asymmetric version
    if mean_prior is not None:
        cost = cost + NormalConstraint('mean', mean_prior[0], error=mean_prior[1])
    if sigma_prior is not None:
        cost = cost + NormalConstraint('sigma', sigma_prior[0], error=sigma_prior[1])
    if tail_prior is not None:
        cost = cost + NormalConstraint(TAIL_PARAMS, tail_prior[0], error=tail_prior[1])

    init = dict(
        mean   = mean_init,
        sigma  = sigma_init,
        alphaL = alphaL_init,
        nL     = nL_init,
        alphaR = alphaR_init,
        nR     = nR_init,
        n_sig  = N_CAND * 0.5,
        n_bkg  = N_CAND * 0.5,
        c0     = 0.0,
        c1     = 0.0,
    )
    limits = dict(
        sigma=(0.0, None), alphaL=(0.0, None), alphaR=(0.0, None),
        nL=(0.0, None), nR=(0.0, None),
        n_sig=(0.0, None), n_bkg=(0.0, None),
    )

    # Stage 1: fit with c0, c1, and the tail params fixed at their init
    # values. Stage 2: float them, starting from stage 1's result.
    stage1 = fit_utils.run_fit(cost, init, limits, fixed=['c0', 'c1', *TAIL_PARAMS])
    init2 = dict(init, mean=stage1.values['mean'], sigma=stage1.values['sigma'],
                n_sig=stage1.values['n_sig'], n_bkg=stage1.values['n_bkg'])
    m = fit_utils.run_fit(cost, init2, limits)
    p, e = m.values, m.errors

    n_bkg = p['n_bkg']
    n_bkg_err = e['n_bkg']

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
    sig_cdf_edges = fit_utils.dcb_cdf(bin_edges, p['mean'], p['sigma'],
                                    p['alphaL'], p['nL'], p['alphaR'], p['nR'],
                                    xmin, xmax)
    bkg_cdf_edges = fit_utils.cheb_cdf(bin_edges, xmin, xmax, p['c0'], p['c1'])
    # Both CDFs run 0 -> 1 across the window, so np.diff gives the fraction of
    # each component in each bin; scale by the yields to get expected counts.
    bin_sig = np.diff(sig_cdf_edges) * p['n_sig']
    bin_bkg = np.diff(bkg_cdf_edges) * n_bkg
    bin_fit = bin_sig + bin_bkg

    # Goodness of fit: Neyman (data-based) chi2, with the DATA as the variance
    # estimate. The denominator is the measurement uncertainty rather than a fit
    # uncertainty, which is the same argument the pulls below rest on -- so
    # sum(pulls**2) == chi2 by construction. Empty-DATA bins are skipped
    # (division by zero) and don't count toward ndof, which is
    # (bins used) - (free parameters).
    #
    # Note this is a deliberate choice over the Pearson form (d-f)^2/f. Neyman
    # chi2 is known to bias yields slightly low, because a downward Poisson
    # fluctuation gets a smaller denominator and so more weight. The fit itself
    # is unaffected -- it minimises a Poisson binned NLL, not this chi2, which
    # is only a goodness-of-fit summary reported afterwards.
    mask = counts > 0
    chi2 = float(np.sum((counts[mask] - bin_fit[mask])**2 / counts[mask]))
    ndof = int(mask.sum()) - m.nfit
    chi2_per_ndof = chi2 / ndof if ndof > 0 else float('nan')

    # Data-based pulls: (d - f) / sqrt(d), 0 for empty bins. Same denominator
    # as the chi2 above, so sum(pulls**2) == chi2. fit_gauss.py and fit_dalitz_v3.ipynb use
    # the same convention.
    pulls = np.where(counts > 0, (counts - bin_fit) / np.sqrt(counts), 0.0)

    print(f'[INFO] χ²/ndf = {chi2_per_ndof:.4f}  ({chi2:.2f} / {ndof})',
          file=sys.stderr)

    # Smooth curves for plotting, on a finer grid than the histogram. These are
    # PDF evaluations (not bin integrals), so scaling by yield * bin width `bw`
    # puts them in "counts per bin" units and lets them be drawn on the same
    # axes as the histogram.
    x_curve = np.linspace(xmin, xmax, 200)
    curve_sig = fit_utils.dcb_pdf(x_curve, p['mean'], p['sigma'],
                                 p['alphaL'], p['nL'], p['alphaR'], p['nR'],
                                 xmin, xmax) * p['n_sig'] * bw
    curve_bkg = fit_utils.cheb_bkg_pdf(x_curve, xmin, xmax, p['c0'], p['c1']) * n_bkg * bw
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
                          'error': round(float(e['n_sig']), 2)},
                'n_bkg': {'value': round(float(n_bkg), 2),
                          'error': round(float(n_bkg_err), 2)},
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
