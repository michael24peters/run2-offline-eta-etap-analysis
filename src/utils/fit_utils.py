"""
fit_utils.py

PDFs/CDFs, iminuit cost functions, the fit driver and fit results, on flat numpy arrays.
"""

import math

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import erf, xlogy


# --- Goodness of fit ---

def neyman_chi2(counts, bin_fit):
    """Data-based chi2, sum (data - fit)^2 / data over the mass bins with data > 0."""
    filled = counts > 0
    return float(np.sum((counts[filled] - bin_fit[filled])**2 / counts[filled]))


def data_pulls(counts, bin_fit):
    """Data-based pull (data - fit) / sqrt(data) of each mass bin; 0 for an empty bin."""
    filled = counts > 0
    return np.where(filled, (counts - bin_fit) / np.sqrt(np.where(filled, counts, 1.0)), 0.0)


# --- Chebyshev ---

# Background parameters, in the order the cost functions take them, and their default initial values
# (a flat background).
BKG_PARAMS = ('c0', 'c1')
BKG_INIT = {'c0': 0.0, 'c1': 0.0}

def _xtilde(x, xmin, xmax):
    """
    Define parameterized, normalized [-1, 1] variable for the Chebyshev polynomials.

    x̃ = (2x - (xmin + xmax)) / (xmax - xmin)
    """
    return (2.0 * np.asarray(x, float) - (xmin + xmax)) / (xmax - xmin)


def _cheb_norm(xmin, xmax, c1):
    """Analytic normalization of Chebyshev over [xmin, xmax]."""
    return (xmax - xmin) * (1.0 - c1 / 3.0)


def cheb_bkg_unnorm(x, xmin, xmax, c0, c1):
    """Unnormalized 2nd-order Chebyshev, 1 + c0*T1 + c1*T2."""
    u = _xtilde(x, xmin, xmax)
    return 1.0 + c0 * u + c1 * (2.0 * u**2 - 1.0)


def cheb_bkg_pdf(x, xmin, xmax, c0, c1):
    """Normalized 2nd-order Chebyshev background PDF."""
    u = _xtilde(x, xmin, xmax)
    return (1.0 + c0 * u + c1 * (2.0 * u**2 - 1.0)) / _cheb_norm(xmin, xmax, c1)


def _cheb_antideriv(u, c0, c1):
    """Indefinite integral of second-order Chebyshev w.r.t. x̃."""
    return u + c0 * u**2 / 2.0 + c1 * (2.0 * u**3 / 3.0 - u)


def cheb_cdf(x, xmin, xmax, c0, c1):
    """Analytic Chebyshev CDF normalized to [0, 1] over [xmin, xmax]. Computed via antiderivative of
    the PDF."""
    norm = _cheb_norm(xmin, xmax, c1)
    scale = (xmax - xmin) / 2.0    # Jacobian: dx = (xmax-xmin)/2 · du
    u_lo = -1.0                    # xtilde(xmin) = -1
    u = _xtilde(x, xmin, xmax)
    return scale * (_cheb_antideriv(u, c0, c1)
                    - _cheb_antideriv(u_lo, c0, c1)) / norm


# --- Gaussian ---

def gauss_pdf(x, xmin, xmax, mean, sigma):
    """Normalized Gaussian signal PDF."""
    x = np.asarray(x, float)
    sqrt2 = math.sqrt(2.0)
    norm = 0.5 * (erf((xmax - mean) / (sigma * sqrt2))
                  - erf((xmin - mean) / (sigma * sqrt2)))
    raw = np.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * math.sqrt(2.0 * math.pi))
    return raw / norm


def gauss_cdf(x, xmin, xmax, mean, sigma):
    """Gaussian CDF normalized to [0, 1] over [xmin, xmax]."""
    x = np.asarray(x, float)
    sqrt2 = math.sqrt(2.0)
    erf_lo = erf((xmin - mean) / (sigma * sqrt2))
    erf_hi = erf((xmax - mean) / (sigma * sqrt2))
    erf_x = erf((x - mean) / (sigma * sqrt2))
    norm = 0.5 * (erf_hi - erf_lo)
    return 0.5 * (erf_x - erf_lo) / norm


# --- Double Crystal Ball ---
#
# A Gaussian core with independent power-law tails on each side of the peak:
#
#           {  A_L · (B_L − t)^{−n_L}    if   t < −α_L        (left  tail)
#   f(t) =  {  exp(−t²/2)                if  −α_L ≤ t ≤ α_R   (Gaussian core)
#           {  A_R · (B_R + t)^{−n_R}    if   t >  α_R        (right tail)
#
#   where   t = (x − μ) / σ.

# DCB shape parameters, in the order the DCB functions take them
DCB_PARAMS = ('mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR')

def _dcb_tail(u, alpha, n):
    """
    One DCB tail at distance u = |t| > |alpha| from the mean, in units of sigma: A · (B + u)^(-n)
    with the continuity constants
      A = (n/|α|)^n · exp(-α²/2)
      B = n/|α| - |α|,
    evaluated as exp(-α²/2 + n·log(n/|α|) - n·log(B + u)), which stays finite for large n (A alone
    overflows).
    """
    abs_a = abs(alpha)
    n_over_a = n / abs_a
    return np.exp(-0.5 * alpha * alpha + xlogy(n, n_over_a) - n * np.log(n_over_a - abs_a + u))


def dcb_unnorm(x, mean, sigma, alphaL, nL, alphaR, nR):
    """Unnormalized Double Crystal Ball PDF."""
    x = np.asarray(x, float)
    t = (x - mean) / sigma
    out = np.empty_like(t)
    mc = (-alphaL <= t) & (t <= alphaR)
    ml = t < -alphaL
    mr = t >  alphaR
    out[mc] = np.exp(-0.5 * t[mc]**2)
    out[ml] = _dcb_tail(-t[ml], alphaL, nL)
    out[mr] = _dcb_tail(t[mr], alphaR, nR)
    return out


def _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR, n_grid=1000):
    """(masses, shape): n_grid evenly spaced masses spanning [xmin, xmax] and the unnormalized DCB
    at them."""
    masses = np.linspace(xmin, xmax, n_grid)
    return masses, dcb_unnorm(masses, mean, sigma, alphaL, nL, alphaR, nR)


def dcb_pdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
    """DCB PDF over [xmin, xmax]: dcb_unnorm at x divided by its trapezoid integral on _dcb_grid."""
    # Unnormalized DCB at the grid masses
    masses, shape = _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR)
    # Its integral over [xmin, xmax]
    integral = trapezoid(shape, masses)
    # Unnormalized DCB at x, divided by the integral
    return dcb_unnorm(np.asarray(x, float),
                      mean, sigma, alphaL, nL, alphaR, nR) / integral


def dcb_cdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
    """DCB CDF over [xmin, xmax], from 0 at xmin to 1 at xmax: the cumulative trapezoid integral on
    _dcb_grid, normalized and interpolated to x (_grid_cdf)."""
    # Unnormalized DCB at the grid masses
    masses, shape = _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR)
    # Integrate cumulatively, normalize to [0, 1], interpolate to x
    return _grid_cdf(x, masses, shape)


# --- Weighting by a Dalitz-bin factor f(M) ---
#
# In one Dalitz bin (one m12 x m23 box), the signal and background models of the mass M are
#
#     signal(M)     = f(M) * dcb_unnorm(M)      / its integral over [xmin, xmax]
#     background(M) = f(M) * cheb_bkg_unnorm(M) / its integral over [xmin, xmax]
#
#   M                   gamma mu+ mu- mass [MeV], the fitted variable.
#   f(M)                the Dalitz bin's factor, between 0 and 1 at each M:
#                       kinematics.phase_space_factor or kinematics.dalitz_factor
#                       for the box's m12 and m23 ranges. Analytic.
#   unnormalized shape  a component's curve before division by its integral:
#                       dcb_unnorm (signal), cheb_bkg_unnorm (background).
#   weighted shape      f(M) * unnormalized shape.
#
# Division by the integral makes each model integrate to 1 over [xmin, xmax], making it so n_sig and
# n_bkg are the expected counts in the window.
#
# The weighted shapes have no closed-form integral; they are integrated with the trapezoid rule on a
# fixed grid of sampled masses (2,000 by default):
#
#   factor_grid         (masses, factor) from sample_factor(), one per Dalitz
#                       bin. `masses`: n_grid evenly spaced masses spanning
#                       [xmin, xmax] [MeV]. `factor`: f at those masses.
BKG_FLOOR = 1e-6


def sample_factor(factor_fn, xmin, xmax, n_grid=2000):
    """Evaluate a Dalitz bin's factor f(M) on a grid of masses.

    factor_fn : f(M), taking an array of masses [MeV] and returning f at each, e.g.
                kinematics.phase_space_factor with the bin's m12 and m23 ranges filled in.

    Returns factor_grid = (masses, factor):
        masses : (n_grid,) evenly spaced masses spanning [xmin, xmax] [MeV]
        factor : (n_grid,) f at `masses`
    """
    # n_grid evenly spaced masses spanning [xmin, xmax]
    masses = np.linspace(xmin, xmax, n_grid)
    # f at each of them
    return masses, np.asarray(factor_fn(masses), dtype=float)


def _grid_cdf(x, masses, weighted_shape):
    """CDF at x of a weighted shape given at `masses`, normalized from 0 to 1."""
    # Cumulative trapezoid integral from masses[0]; cumulative[0] = 0
    cumulative = cumulative_trapezoid(weighted_shape, masses, initial=0)
    # Weighted shape 0 everywhere: CDF 0 everywhere
    if cumulative[-1] <= 0:
        return np.zeros_like(np.asarray(x, float))
    # Divide by the total, interpolate to x
    return np.interp(np.asarray(x, float), masses, cumulative / cumulative[-1])


def _grid_pdf(x, masses, weighted_shape):
    """PDF at x of a weighted shape given at `masses`, normalized to integrate to 1."""
    # Integral of the weighted shape over the grid
    integral = trapezoid(weighted_shape, masses)
    # Weighted shape 0 everywhere: PDF 0 everywhere
    if integral <= 0:
        return np.zeros_like(np.asarray(x, float))
    # Weighted shape divided by its integral, interpolated to x
    return np.interp(np.asarray(x, float), masses, weighted_shape / integral)


def weighted_dcb_cdf(factor_grid):
    """CDF of f(M) * DCB(M), as a function with dcb_cdf's arguments.

    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    # Signal CDF function of this Dalitz bin
    def cdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
        # Unnormalized DCB at the grid masses, times f (array broadcast)
        weighted_shape = dcb_unnorm(masses, mean, sigma, alphaL, nL, alphaR, nR) * factor
        # Integrate cumulatively, normalize to [0, 1], interpolate to x
        return _grid_cdf(x, masses, weighted_shape)
    return cdf


def weighted_dcb_pdf(factor_grid):
    """PDF of f(M) * DCB(M), as a function with dcb_pdf's arguments.

    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    def pdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
        # Unnormalized DCB at the grid masses, times f
        weighted_shape = dcb_unnorm(masses, mean, sigma, alphaL, nL, alphaR, nR) * factor
        # Divided by its integral
        return _grid_pdf(x, masses, weighted_shape)
    return pdf


def reachable_range(factor_grid):
    """Finds the lowest and highest masses where factor > 0 (or the whole range if factor is 0
    everywhere).

    This is needed for the weighted Chebyshev, which is defined over [-1, 1] in a normalized
    variable x̃ spanning the reachable range of masses, instead of [xmin, xmax].

    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    # Grid masses where f > 0
    reachable = masses[factor > 0]
    # None, or a single one: the whole grid
    if reachable.size == 0 or reachable.max() <= reachable.min():
        return masses[0], masses[-1]
    return reachable.min(), reachable.max()


def weighted_cheb_cdf(factor_grid):
    """CDF of f(M) * max(Chebyshev(M), BKG_FLOOR), as a function with cheb_cdf's arguments (xmin,
    xmax unused, as in weighted_dcb_cdf).

    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    # Masses the Chebyshev variable x̃ spans
    mass_lo, mass_hi = reachable_range(factor_grid)
    # Background CDF function of this Dalitz bin
    def cdf(x, xmin, xmax, c0, c1):
        # cheb_bkg_unnorm(M, *reachable_range(factor_grid), c0, c1) is the unweighted, unnormalized
        # Chebyshev at the grid masses, which we floor at BKG_FLOOR before multiplying by f.
        cheb = np.clip(cheb_bkg_unnorm(masses, mass_lo, mass_hi, c0, c1),
                       BKG_FLOOR, None)
        # Times f; integrate cumulatively, normalize to [0, 1], interpolate to x
        return _grid_cdf(x, masses, cheb * factor)
    return cdf


def weighted_cheb_pdf(factor_grid):
    """PDF of f(M) * max(Chebyshev(M), BKG_FLOOR), as a function with cheb_bkg_pdf's arguments
    (xmin, xmax unused, as in weighted_dcb_cdf).

    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    # Masses the Chebyshev variable x̃ spans
    mass_lo, mass_hi = reachable_range(factor_grid)
    def pdf(x, xmin, xmax, c0, c1):
        # Unnormalized Chebyshev at the grid masses, floored at BKG_FLOOR
        cheb = np.clip(cheb_bkg_unnorm(masses, mass_lo, mass_hi, c0, c1),
                       BKG_FLOOR, None)
        # Times f, divided by its integral
        return _grid_pdf(x, masses, cheb * factor)
    return pdf


# --- iminuit cost functions ---
#
# Each make_*_cost returns an iminuit.cost.ExtendedBinnedNLL: the extended Poisson likelihood of a
# mass histogram.
#
#   counts      (nbins,) candidates in each mass bin.
#   bin_edges   (nbins + 1,) edges of the mass bins [MeV], spanning
#               [xmin, xmax]. These are mass-histogram edges, not Dalitz-bin
#               boundaries.
#   scaled_cdf  the model as an expected cumulative count at each edge:
#                   n_sig * signal_cdf(bin_edges) + n_bkg * bkg_cdf(bin_edges).
#               ExtendedBinnedNLL takes the expected count in mass bin k as
#               scaled_cdf(bin_edges[k + 1]) - scaled_cdf(bin_edges[k]), the
#               model's integral over that bin.
#
# scaled_cdf takes `*args`, since its parameters depend on the signal model; _make_scaled_cdf
# returns their names for ExtendedBinnedNLL's `name=`.

def _make_scaled_cdf(signal_param_names, signal_cdf, bkg_cdf, xmin, xmax):
    """Returns (scaled_cdf, param_names). scaled_cdf(xe, *params) -> expected cumulative count;
    param_names is signal_param_names + ('n_sig', 'n_bkg')
    + BKG_PARAMS, in the order scaled_cdf expects them."""
    n_sig_params = len(signal_param_names)
    param_names = tuple(signal_param_names) + ('n_sig', 'n_bkg') + BKG_PARAMS

    # Combine into scaled_cdf = n_sig * signal CDF + n_bkg * background CDF
    def scaled_cdf(xe, *args):
        # Split args into signal shape, yields, background shape
        sig_args = args[:n_sig_params]
        n_sig, n_bkg = args[n_sig_params], args[n_sig_params + 1]
        bkg_args = args[n_sig_params + 2:]
        # Expected cumulative count at each mass-bin edge xe
        return (signal_cdf(xe, xmin, xmax, *sig_args) * n_sig
                + bkg_cdf(xe, xmin, xmax, *bkg_args) * n_bkg)

    return scaled_cdf, param_names


def make_gauss_cost(counts, bin_edges, xmin, xmax):
    """ExtendedBinnedNLL cost for Gaussian signal + Chebyshev background. Free parameters: mean,
    sigma, n_sig, n_bkg, c0, c1.
    """
    scaled_cdf, param_names = _make_scaled_cdf(('mean', 'sigma'), gauss_cdf,
                                               cheb_cdf, xmin, xmax)
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


def make_dcb_cost(counts, bin_edges, xmin, xmax, signal_factor_grid=None,
                  bkg_factor_grid=None):
    """
    ExtendedBinnedNLL cost for DCB signal + Chebyshev background. Free parameters: mean, sigma,
    alphaL, nL, alphaR, nR, n_sig, n_bkg, c0, c1.

    signal_factor_grid : optional (masses, factor) from sample_factor(). The signal CDF is then
                         weighted_dcb_cdf(signal_factor_grid) instead of dcb_cdf.
    bkg_factor_grid    : the same for the background: weighted_cheb_cdf
                         instead of cheb_cdf.
    """
    # Signal CDF, weighted by f if signal_factor_grid is given
    signal_cdf = dcb_cdf
    if signal_factor_grid is not None:
        signal_cdf = weighted_dcb_cdf(signal_factor_grid)
    # Background CDF, weighted by f if bkg_factor_grid is given
    bkg_cdf = cheb_cdf
    if bkg_factor_grid is not None:
        bkg_cdf = weighted_cheb_cdf(bkg_factor_grid)
    # Combine into scaled_cdf = n_sig * signal CDF + n_bkg * background CDF
    scaled_cdf, param_names = _make_scaled_cdf(DCB_PARAMS, signal_cdf,
                                               bkg_cdf, xmin, xmax)
    # Extended binned Poisson likelihood of counts in bin_edges
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


# --- Fit ---

def run_fit(cost, initial_values, limits=None, fixed=None):
    """
    Run MIGRAD (minimisation) then HESSE (covariance / symmetric errors).

    Parameters:
    cost           : iminuit cost function (e.g. ExtendedBinnedNLL, or a sum
                     of one with NormalConstraint terms added with `+`)
    initial_values : dict {param_name: start_value}
    limits         : optional dict {param_name: (lo, hi)}, either bound may
                     be None
    fixed          : optional list of param names to hold at their
                     initial_values instead of floating

    Returns:
    m : iminuit.Minuit
        Fitted Minuit object. Key attributes:
          m.values     – best-fit parameter values (MnUserParameters-like dict)
          m.errors     – HESSE symmetric errors
          m.covariance – full covariance matrix, name-indexable, e.g.
                         m.covariance[['c0', 'c1']] for a 2x2 submatrix
          m.valid      – True if MIGRAD converged
          m.fval       – minimum function value (2 * NLL)
          m.nfit       – number of free floating parameters
          m.fmin.edm   – estimated distance to minimum
    """
    m = Minuit(cost, **initial_values)
    for param, lim in (limits or {}).items(): m.limits[param] = lim
    for param in (fixed or []): m.fixed[param] = True
    m.migrad(ncall=100000)
    m.hesse()
    if not m.valid:
        # Not converged: MIGRAD and HESSE once more, from where the first pass ended
        m.migrad(ncall=100000)
        m.hesse()
    return m


# --- Fit results ---

def bin_prediction(m, counts, bin_edges, xmin, xmax, signal_cdf, signal_params, bkg_cdf):
    """
    Expected signal and background counts in each mass bin of fit `m`, with its chi2 and pulls.

    m             : the fitted iminuit.Minuit
    counts        : (nbins,) candidates in each mass bin
    bin_edges     : (nbins + 1,) mass-bin edges [MeV] spanning [xmin, xmax]
    signal_cdf    : signal CDF with the arguments of dcb_cdf: (x, xmin, xmax, *shape)
    signal_params : names of the signal shape parameters in m, in signal_cdf's order
    bkg_cdf       : background CDF with the arguments of cheb_cdf

    Returns {'bin_sig', 'bin_bkg', 'pulls', 'chi2', 'ndof'}: neyman_chi2 and data_pulls over the
    mass bins with data, and ndof the number of those bins minus the free parameters.
    """
    p = m.values
    # Expected counts per mass bin: yield times the CDF difference across the bin
    bin_sig = np.diff(signal_cdf(bin_edges, xmin, xmax,
                                 *(p[name] for name in signal_params))) * p['n_sig']
    bin_bkg = np.diff(bkg_cdf(bin_edges, xmin, xmax,
                              *(p[name] for name in BKG_PARAMS))) * p['n_bkg']
    bin_fit = bin_sig + bin_bkg
    return {'bin_sig': bin_sig, 'bin_bkg': bin_bkg,
            'pulls': data_pulls(counts, bin_fit),
            'chi2': neyman_chi2(counts, bin_fit),
            'ndof': int((counts > 0).sum()) - m.nfit}


def fit_result(m, model, n_input, bin_edges, counts, prediction):
    """
    JSON-serialisable result of a mass fit.

    m          : the fitted iminuit.Minuit
    model      : name of the signal model, e.g. 'gauss'
    n_input    : number of values given to the fit, in the window or not
    bin_edges  : (nbins + 1,) mass-bin edges [MeV]
    counts     : (nbins,) candidates in each mass bin
    prediction : bin_prediction() of m

    Returns {'meta', 'fit', 'histogram'}: the window and model; the fit status, chi2, every
    parameter's value and error, and the covariance (None if HESSE failed); and the data, expected
    signal and background, and pulls of each mass bin.
    """
    chi2, ndof = prediction['chi2'], prediction['ndof']
    covariance = None
    if m.covariance is not None:
        covariance = {'names': list(m.parameters), 'matrix': np.asarray(m.covariance).tolist()}
    parameters = {}
    for name in m.parameters:
        digits = 2 if name in ('n_sig', 'n_bkg') else 6
        parameters[name] = {'value': round(float(m.values[name]), digits),
                            'error': round(float(m.errors[name]), digits)}
    return {
        'meta': {
            'n_input': int(n_input),
            'model': model,
            'xmin': float(bin_edges[0]),
            'xmax': float(bin_edges[-1]),
            'nbins': len(counts),
        },
        'fit': {
            'valid': bool(m.valid),
            'fval': round(float(m.fval), 4),
            'chi2': round(chi2, 4),
            'ndof': ndof,
            'chi2_per_ndof': round(chi2 / ndof, 6) if ndof > 0 else float('nan'),
            'n_free_params': m.nfit,
            'covariance': covariance,
            'parameters': parameters,
        },
        'histogram': {
            'bin_centers': (0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist(),
            'bin_counts': counts.tolist(),
            'bin_sig': prediction['bin_sig'].tolist(),
            'bin_bkg': prediction['bin_bkg'].tolist(),
            'bin_pulls': prediction['pulls'].tolist(),
        },
    }


# --- Priors from a previous fit ---

def build_prior(fit, params, factor=1.0):
    """
    Build a (values, cov) pair for `NormalConstraint(params, values, error=cov)` from a previous
    fit's parameters, with each parameter's uncertainty scaled by `factor`.

    fit    : an iminuit.Minuit.
    params : names to constrain; a parameter not in this list gets no prior.
    factor : stdev multiplier applied to every parameter in `params`, so the covariance scales by
             factor**2. The correlations cov_ij / (sigma_i * sigma_j) of the original fit are
             unchanged.

    Falls back to a diagonal matrix built from `fit.errors` when `fit.covariance` is None (HESSE's
    matrix inversion failed).
    """
    params = list(params)
    values = np.array([fit.values[p] for p in params], dtype=float)
    if fit.covariance is not None:
        cov = np.array(fit.covariance[params], dtype=float)
    else:
        errs = np.array([fit.errors[p] for p in params], dtype=float)
        cov = np.diag(errs**2)

    scaled = factor**2 * cov
    # NormalConstraint requires a bit-exact symmetric matrix.
    return values, (scaled + scaled.T) / 2


def build_prior_from_result(fit_block, params, factor=1.0):
    """build_prior() for a fit saved in a results JSON. fit_block is its 'fit' dict: 'parameters'
    {name: {'value', 'error'}} and 'covariance' {'names', 'matrix'}, or None if HESSE's matrix
    inversion failed."""
    params = list(params)
    values = np.array([fit_block['parameters'][p]['value'] for p in params], 
                      dtype=float)
    covariance = fit_block.get('covariance')
    if covariance is not None:
        index = [covariance['names'].index(p) for p in params]
        cov = np.array(covariance['matrix'], dtype=float)[np.ix_(index, index)]
    else:
        errs = np.array([fit_block['parameters'][p]['error'] for p in params], 
                        dtype=float)
        cov = np.diag(errs**2)

    scaled = factor**2 * cov
    # NormalConstraint requires a bit-exact symmetric matrix.
    return values, (scaled + scaled.T) / 2
