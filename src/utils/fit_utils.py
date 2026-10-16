"""
fit_utils.py

Histogramming, PDFs/CDFs, iminuit cost functions, and the fit driver. Pure
numpy -- this module knows nothing about ROOT files, awkward arrays, events, or
candidates. Selection and candidate reduction happen upstream in cut_utils; by
the time anything here is called, the data is a flat array of values.
"""

import math

import numpy as np
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import erf, xlogy


def make_histogram(values, xmin, xmax, nbins):
    """Bin a flat array of values into the fit window.

    Arguments:
    ----------
    values : 1-D numpy array (e.g. one mass per event, already selected and
             reduced to one candidate per event by cut_utils)
    xmin : float
        The minimum value of the fit window.
    xmax : float
        The maximum value of the fit window.
    nbins : int
        The number of bins to use.

    Returns:
    --------
    centers ndarray (nbins,) : bin centre positions
    edges   ndarray (nbins+1,) : bin edges
    counts  ndarray (nbins,) : observed counts per bin (float)
    errors  ndarray (nbins,) : per-bin uncertainty, max(sqrt(counts), 1)
    """
    counts, edges = np.histogram(values, bins=nbins, range=(xmin, xmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors  = np.maximum(np.sqrt(counts.astype(float)), 1.0)
    return centers, edges, counts.astype(float), errors


# --- Chebyshev ---

# Background parameters, in the order the cost functions take them, and their
# default initial values (a flat background).
BKG_PARAMS = ('c0', 'c1')
BKG_INIT = {'c0': 0.0, 'c1': 0.0}

def _xtilde(x, xmin, xmax):
    """
    Define parameterized, normalized [-1, 1] variable for the Chebyshev
    polynomials.

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
    """Analytic Chebyshev CDF normalized to [0, 1] over [xmin, xmax]. Computed
    via antiderivative of the PDF."""
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
"""
A Gaussian core with independent power-law tails on each side of the peak:

          {  A_L · (B_L − t)^{−n_L}    if   t < −α_L        (left  tail)
  f(t) =  {  exp(−t²/2)                if  −α_L ≤ t ≤ α_R   (Gaussian core)
          {  A_R · (B_R + t)^{−n_R}    if   t >  α_R        (right tail)

  where   t = (x − μ) / σ.
"""

def _dcb_tail(u, alpha, n):
    """
    One DCB tail at distance u = |t| > |alpha| from the mean, in units of
    sigma: A · (B + u)^(-n) with the continuity constants
      A = (n/|α|)^n · exp(-α²/2)
      B = n/|α| - |α|,
    evaluated as exp(-α²/2 + n·log(n/|α|) - n·log(B + u)), which stays finite
    for large n (A alone overflows).
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
    """(masses, shape): n_grid evenly spaced masses spanning [xmin, xmax] and
    the unnormalized DCB at them."""
    masses = np.linspace(xmin, xmax, n_grid)
    return masses, dcb_unnorm(masses, mean, sigma, alphaL, nL, alphaR, nR)


def dcb_pdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
    """DCB PDF over [xmin, xmax]: dcb_unnorm at x divided by its trapezoid
    integral on _dcb_grid."""
    # Unnormalized DCB at the grid masses
    masses, shape = _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR)
    # Its integral over [xmin, xmax]
    integral = trapezoid(shape, masses)
    # Unnormalized DCB at x, divided by the integral
    return dcb_unnorm(np.asarray(x, float),
                      mean, sigma, alphaL, nL, alphaR, nR) / integral


def dcb_cdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
    """DCB CDF over [xmin, xmax], from 0 at xmin to 1 at xmax: the cumulative
    trapezoid integral on _dcb_grid, normalized and interpolated to x
    (_grid_cdf)."""
    # Unnormalized DCB at the grid masses
    masses, shape = _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR)
    # Integrate cumulatively, normalize to [0, 1], interpolate to x
    return _grid_cdf(x, masses, shape)


def dcb_window_yield(p, e, xmin, xmax, window):
    """Signal yield (and its error) from a fitted DCB within an arbitrary
    (lo, hi) sub-window, by scaling n_sig by the DCB CDF's mass fraction
    there. `p`/`e` are the fit's values/errors (e.g. Minuit's m.values,
    m.errors), needing mean/sigma/alphaL/nL/alphaR/nR/n_sig."""
    edge_cdf = dcb_cdf(np.array(window), xmin, xmax, p['mean'], p['sigma'],
                       p['alphaL'], p['nL'], p['alphaR'], p['nR'])
    frac = float(edge_cdf[1] - edge_cdf[0])
    return p['n_sig'] * frac, e['n_sig'] * frac


# --- Weighting by a Dalitz-bin factor f(M) ---
#
# In one Dalitz bin (one m12 x m23 box), the signal and background models of
# the mass M are
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
# Division by the integral makes each model integrate to 1 over [xmin, xmax],
# making it so n_sig and n_bkg are the expected counts in the window.
#
# The weighted shapes have no closed-form integral; they are integrated with
# the trapezoid rule on a fixed grid of sampled masses (2,000 by default):
#
#   factor_grid         (masses, factor) from sample_factor(), one per Dalitz
#                       bin. `masses`: n_grid evenly spaced masses spanning
#                       [xmin, xmax] [MeV]. `factor`: f at those masses.
#
# Fit of one Dalitz bin:
#   1. Histogram the M of the candidates in the box: counts, bin_edges.
#   2. sample_factor(f, xmin, xmax) -> factor_grid, once per Dalitz bin.
#   3. make_dcb_cost(counts, bin_edges, xmin, xmax, signal_factor_grid=...,
#      bkg_factor_grid=...):
#      3.1 build weighted_dcb_cdf(factor_grid) and weighted_cheb_cdf(factor_grid),
#          CDF functions with the same arguments as dcb_cdf and cheb_cdf,
#      3.2 combine them into scaled_cdf = n_sig * signal CDF + n_bkg * background CDF,
#      3.3 wrap scaled_cdf in an ExtendedBinnedNLL of counts.
#   4. For each set of parameters MINUIT tries, ExtendedBinnedNLL calls
#      scaled_cdf(bin_edges, ...), and each weighted CDF:
#      4.1 evaluates the unnormalized shape at `masses`,
#      4.2 multiplies it by `factor`, giving the weighted shape,
#      4.3 integrates the weighted shape cumulatively from xmin (_grid_cdf),
#      4.4 divides by the total, so the CDF runs from 0 at xmin to 1 at xmax,
#      4.5 interpolates to bin_edges.
#      Differences of scaled_cdf between neighbouring edges are the expected
#      counts per mass bin, compared with `counts` in the Poisson likelihood.
#   5. After the fit, the same weighted CDFs give the expected signal and
#      background counts per mass bin; reachable_range gives the masses of the
#      negative-background check.
#
# weighted_dcb_pdf and weighted_cheb_pdf divide the weighted shape by its
# integral in place of 4.3-4.4. These are used for plotting the fit shape.
# dcb_cdf runs 4.1 and 4.3-4.5 with f = 1 on its own grid; cheb_cdf is
# closed-form.
#
# The weighted background's unnormalized shape is max(cheb_bkg_unnorm,
# BKG_FLOOR): a negative density makes the likelihood undefined. The floor is
# applied before the division by the integral, so the background still
# integrates to 1. Bins whose fitted Chebyshev goes below 0 where f > 0 are
# flagged after the fit (bkg_goes_negative).
BKG_FLOOR = 1e-6


def sample_factor(factor_fn, xmin, xmax, n_grid=2000):
    """Evaluate a Dalitz bin's factor f(M) on a grid of masses.

    factor_fn : f(M), taking an array of masses [MeV] and returning f at each,
                e.g. kinematics.phase_space_factor with the bin's m12 and m23
                ranges filled in.

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
    """PDF at x of a weighted shape given at `masses`, normalized to integrate
    to 1."""
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
    """Finds the lowest and highest masses where factor > 0 (or the whole range
    if factor is 0 everywhere).

    This is needed for the weighted Chebyshev, which is defined over [-1, 1] in
    a normalized variable x̃ spanning the reachable range of masses, instead of
    [xmin, xmax].
    
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
    """CDF of f(M) * max(Chebyshev(M), BKG_FLOOR), as a function with
    cheb_cdf's arguments (xmin, xmax unused, as in weighted_dcb_cdf).
    
    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    # Masses the Chebyshev variable x̃ spans
    mass_lo, mass_hi = reachable_range(factor_grid)
    # Background CDF function of this Dalitz bin
    def cdf(x, xmin, xmax, c0, c1):
        # cheb_bkg_unnorm(M, *reachable_range(factor_grid), c0, c1) is the 
        # unweighted, unnormalized Chebyshev at the grid masses, which we floor
        # at BKG_FLOOR before multiplying by f.
        cheb = np.clip(cheb_bkg_unnorm(masses, mass_lo, mass_hi, c0, c1), BKG_FLOOR, None)
        # Times f; integrate cumulatively, normalize to [0, 1], interpolate to x
        return _grid_cdf(x, masses, cheb * factor)
    return cdf


def weighted_cheb_pdf(factor_grid):
    """PDF of f(M) * max(Chebyshev(M), BKG_FLOOR), as a function with
    cheb_bkg_pdf's arguments (xmin, xmax unused, as in weighted_dcb_cdf).
    
    factor_grid : (masses, factor) from sample_factor()
    """
    masses, factor = factor_grid
    # Masses the Chebyshev variable x̃ spans
    mass_lo, mass_hi = reachable_range(factor_grid)
    def pdf(x, xmin, xmax, c0, c1):
        # Unnormalized Chebyshev at the grid masses, floored at BKG_FLOOR
        cheb = np.clip(cheb_bkg_unnorm(masses, mass_lo, mass_hi, c0, c1), BKG_FLOOR, None)
        # Times f, divided by its integral
        return _grid_pdf(x, masses, cheb * factor)
    return pdf


# --- iminuit cost functions ---
#
# Each make_*_cost returns an iminuit.cost.ExtendedBinnedNLL: the extended
# Poisson likelihood of a mass histogram.
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
# scaled_cdf takes `*args`, since its parameters depend on the signal model;
# _make_scaled_cdf returns their names for ExtendedBinnedNLL's `name=`.

def _make_scaled_cdf(signal_param_names, signal_cdf, bkg_cdf, xmin, xmax):
    """Returns (scaled_cdf, param_names). scaled_cdf(xe, *params) -> expected
    cumulative count; param_names is signal_param_names + ('n_sig', 'n_bkg')
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
    """ExtendedBinnedNLL cost for Gaussian signal + Chebyshev background.
    Free parameters: mean, sigma, n_sig, n_bkg, c0, c1.
    """
    scaled_cdf, param_names = _make_scaled_cdf(('mean', 'sigma'), gauss_cdf, cheb_cdf, xmin, xmax)
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


def make_dcb_cost(counts, bin_edges, xmin, xmax, signal_factor_grid=None, bkg_factor_grid=None):
    """
    ExtendedBinnedNLL cost for DCB signal + Chebyshev background.
    Free parameters: mean, sigma, alphaL, nL, alphaR, nR, n_sig, n_bkg, c0, c1.

    signal_factor_grid : optional (masses, factor) from sample_factor(). The
                         signal CDF is then weighted_dcb_cdf(signal_factor_grid)
                         instead of dcb_cdf.
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
    signal_params = ('mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR')
    scaled_cdf, param_names = _make_scaled_cdf(signal_params, signal_cdf, bkg_cdf, xmin, xmax)
    # Extended binned Poisson likelihood of counts in bin_edges
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


def _dcb_sym_cdf(xe, xmin, xmax, mean, sigma, alpha, n):
    return dcb_cdf(xe, xmin, xmax, mean, sigma, alpha, n, alpha, n)


def make_dcb_sym_cost(counts, bin_edges, xmin, xmax):
    """
    ExtendedBinnedNLL cost for symmetric DCB (α_R ≡ α_L, n_R ≡ n_L) +
    Chebyshev background.
    Free parameters: mean, sigma, alpha, n, n_sig, n_bkg, c0, c1 (8 vs 10).
    """
    signal_params = ('mean', 'sigma', 'alpha', 'n')
    scaled_cdf, param_names = _make_scaled_cdf(signal_params, _dcb_sym_cdf, cheb_cdf, xmin, xmax)
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


# --- Fit ---

def run_fit(cost, initial_values, limits=None, fixed=None, seeds=None):
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
    seeds          : optional list of alternative {param: value} dicts,
                     tried in order (each merged over initial_values) until
                     one produces m.valid == True. Returns the lowest-fval
                     attempt if none validate.

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
    best = None
    for start in [initial_values, *({**initial_values, **s} for s in (seeds or []))]:
        m = Minuit(cost, **start)
        for param, lim in (limits or {}).items(): m.limits[param] = lim
        for param in (fixed or []): m.fixed[param] = True
        m.migrad(ncall=100000)
        m.hesse()
        if not m.valid:
            # A single MIGRAD pass can land close enough for HESSE to succeed
            # but still miss MIGRAD's own EDM convergence goal -- standard
            # MINUIT practice is to re-run it once more from there, refining
            # off the now-improved Hessian estimate, before giving up.
            m.migrad(ncall=100000)
            m.hesse()
        if m.valid: return m
        if best is None or m.fval < best.fval: best = m
    return best


# --- Priors from a previous fit ---

def build_prior(fit, params, factor=1.0):
    """
    Build a (values, cov) pair -- ready for
    `NormalConstraint(params, values, error=cov)` -- from a previous fit's
    parameters, with each parameter's uncertainty scaled by `factor`.

    fit    : a live iminuit.Minuit or a LoadedFit (both expose .values,
             .errors, .covariance).
    params : names to constrain. A parameter NOT in this list gets no prior
             at all -- this is how a prior is dropped entirely for a given
             stage of a fit chain (e.g. floating a background parameter
             completely on the per-bin fits after constraining it on the
             data fit).
    factor : a single float (one stdev multiplier applied to every parameter
             in `params`) or a {param_name: float} dict (a per-parameter
             multiplier; a name missing from the dict defaults to 1.0) --
             e.g. factor={'mean': 5.0, 'sigma': 5.0, 'c0': 25.0, 'c1': 25.0}
             loosens background far more than signal in one call, replacing
             the previous SIGNAL_PRIOR_FACTOR / BKG_PRIOR_FACTOR two-call
             pattern. This is a STDEV multiplier, so variances scale by
             factor**2.

    Off-diagonal covariance terms encode correlation: cov_ij says how much
    parameter i and j moved together in the original fit, and
    rho_ij = cov_ij / (sigma_i * sigma_j) is unitless (-1 to 1). Scaling with
    F = diag(factors) as cov' = F @ cov @ F multiplies var_i by factor_i**2
    and cov_ij by factor_i*factor_j for i != j -- every rho_ij is UNCHANGED.
    So each parameter's marginal uncertainty can be loosened independently
    while the correlation structure the original fit measured is preserved
    exactly, rather than discarded.

    Falls back to a diagonal matrix built from `fit.errors` when
    `fit.covariance` is None (HESSE's matrix inversion failed) -- no
    cross-parameter term, but still a usable prior instead of a crash.
    """
    params = list(params)
    if isinstance(factor, dict):
        factors = np.array([factor.get(p, 1.0) for p in params], dtype=float)
    else:
        factors = np.full(len(params), float(factor))

    values = np.array([fit.values[p] for p in params], dtype=float)
    if fit.covariance is not None:
        cov = np.array(fit.covariance[params], dtype=float)
    else:
        errs = np.array([fit.errors[p] for p in params], dtype=float)
        cov = np.diag(errs**2)

    F = np.diag(factors)
    scaled = F @ cov @ F
    # F @ cov @ F is symmetric in exact arithmetic but not always bit-exact
    # symmetric in floating point; NormalConstraint requires the latter.
    return values, (scaled + scaled.T) / 2


# --- Reloading a saved fit ---

class NamedCovariance:
    """Read-only stand-in for iminuit's name-indexable covariance view, e.g.
    cov[['mean', 'sigma']] returns the 2x2 submatrix for those two parameters,
    in that order."""

    def __init__(self, names, matrix):
        self._index = {name: i for i, name in enumerate(names)}
        self._matrix = np.asarray(matrix, dtype=float)

    def __getitem__(self, names):
        idx = [self._index[n] for n in names]
        return self._matrix[np.ix_(idx, idx)]


class LoadedFit:
    """Minuit-like read-only view over a JSON fit result's `fit` block, so
    code written against a live Minuit object (m.values[name],
    m.errors[name], m.valid, m.covariance) runs unchanged against a fit
    reloaded from disk.

    `covariance` is None unless the `fit` block has a `covariance` entry
    ({'names': [...], 'matrix': [[...]]}) -- same as a live fit whose HESSE
    matrix inversion failed, so callers that already fall back to a diagonal
    prior from `errors` in that case need no changes. There is no live HESSE
    run behind a loaded fit, so `fmin` and `limits` aren't reconstructed --
    debugging cells built around those should be skipped for a loaded fit
    rather than adapted.
    """

    def __init__(self, fit_block):
        self.valid = fit_block['valid']
        self.fval = fit_block['fval']
        self.nfit = fit_block['n_free_params']
        params = fit_block['parameters']
        self.parameters = tuple(params)
        self.values = {name: p['value'] for name, p in params.items()}
        self.errors = {name: p['error'] for name, p in params.items()}
        cov = fit_block.get('covariance')
        self.covariance = (NamedCovariance(cov['names'], cov['matrix'])
                            if cov is not None else None)
