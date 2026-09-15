"""
fit_utils.py

Histogramming, PDFs/CDFs, iminuit cost functions, and the fit driver. Pure
numpy -- this module knows nothing about ROOT files, awkward arrays, events, or
candidates. Selection and candidate reduction happen upstream in cut_utils; by
the time anything here is called, the data is a flat array of values.
"""

import math
import numpy as np
from scipy.special import erf
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
# Constraint-based fitting
from iminuit.cost import NormalConstraint

def _trapz(y, x):
    """Trapezoidal integration"""
    y = np.asarray(y)
    x = np.asarray(x)
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.sum(avg * dx)


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


# --- Chebyshev x exponential (positive-by-construction background) ---
#
# pdf(x) ~ exp(k0*u + k1*(2u^2 - 1)), u the same normalized [-1, 1] Chebyshev
# variable as above, i.e., the same Chebyshev polynomial as above, exponentiated.
# Always > 0 everywhere by construction (it's an exp), so it can never produce
# the negative-background dips the plain Chebyshev can.
# k1 < 0 (concave) decays toward zero at BOTH edges; a k0-dominant, k1~0
# shape decays at just one edge -- sign/magnitude of (k0, k1) determine which.

def _grid_norm_cdf(xg, fg):
    """Cumulative-trapezoid CDF over a grid, normalized to [0, 1]. Shared by
    every background model below that has no closed-form CDF."""
    cdf_grid = np.zeros_like(fg)
    cdf_grid[1:] = np.cumsum(0.5 * (fg[:-1] + fg[1:]) * np.diff(xg))
    return cdf_grid / cdf_grid[-1]


def _chebexp_unnorm(u, k0, k1):
    return np.exp(k0 * u + k1 * (2.0 * u**2 - 1.0))


def chebexp_bkg_pdf(x, xmin, xmax, k0, k1, n_grid=1000):
    """Normalized exponentiated-Chebyshev background PDF."""
    xg = np.linspace(xmin, xmax, n_grid)
    fg = _chebexp_unnorm(_xtilde(xg, xmin, xmax), k0, k1)
    norm = _trapz(fg, xg)
    ux = _xtilde(np.asarray(x, float), xmin, xmax)
    return _chebexp_unnorm(ux, k0, k1) / norm


def chebexp_cdf(x, xmin, xmax, k0, k1, n_grid=1000):
    """Exponentiated-Chebyshev CDF normalized to [0, 1] over [xmin, xmax]."""
    xg = np.linspace(xmin, xmax, n_grid)
    fg = _chebexp_unnorm(_xtilde(xg, xmin, xmax), k0, k1)
    cdf_grid = _grid_norm_cdf(xg, fg)
    return np.interp(np.asarray(x, float), xg, cdf_grid)


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

def _dcb_tail_constants(alpha, n):
    """
    Continuity constants A, B for one DCB tail (alpha > 0).
    At the transition point t = -|alpha| (left) or t = |alpha| (right):
      A = (n/|α|)^n · exp(-α²/2)
      B = n/|α| - |α|
    """
    abs_a = abs(alpha)
    A = (n / abs_a) ** n * math.exp(-0.5 * alpha * alpha)
    B = n / abs_a - abs_a
    return A, B


def dcb_unnorm(x, mean, sigma, alphaL, nL, alphaR, nR):
    """Unnormalized Double Crystal Ball PDF."""
    x = np.asarray(x, float)
    t = (x - mean) / sigma
    AL, BL = _dcb_tail_constants(alphaL, nL)
    AR, BR = _dcb_tail_constants(alphaR, nR)
    out = np.empty_like(t)
    mc = (-alphaL <= t) & (t <= alphaR)
    ml = t < -alphaL
    mr = t >  alphaR
    out[mc] = np.exp(-0.5 * t[mc]**2)
    out[ml] = AL * (BL - t[ml]) ** (-nL)
    out[mr] = AR * (BR + t[mr]) ** (-nR)
    return out


def _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR, n_grid=1000):
    """Evaluate unnormalized DCB on a dense grid for CDF/norm computation."""
    xg = np.linspace(xmin, xmax, n_grid)
    fg = dcb_unnorm(xg, mean, sigma, alphaL, nL, alphaR, nR)
    return xg, fg


def dcb_pdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
    """Normalized DCB PDF over [xmin, xmax], computed numerically via trapezoid
    rule over a dense grid."""
    # Evaluate DCB on evenly spaced grid
    xg, fg = _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR)
    # Integrate via trapezoid rule
    norm   = _trapz(fg, xg)
    return dcb_unnorm(np.asarray(x, float),
                      mean, sigma, alphaL, nL, alphaR, nR) / norm


def dcb_cdf(x, xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR):
    """DCB CDF normalized to [0, 1] over [xmin, xmax]. Computed via cumulative
    trapezoid on a dense grid, then interpolated."""
    # Evaluate DCB on evenly spaced grid
    xg, fg   = _dcb_grid(xmin, xmax, mean, sigma, alphaL, nL, alphaR, nR)
    cdf_grid = np.zeros_like(fg)
    # Running total: cdf_grid[i+1] = cdf_grid[i] + avg_i * dx_i
    cdf_grid[1:] = np.cumsum(0.5 * (fg[:-1] + fg[1:]) * np.diff(xg))
    # Normalize to [0, 1]
    cdf_grid = cdf_grid / cdf_grid[-1]
    return np.interp(np.asarray(x, float), xg, cdf_grid)


def dcb_window_yield(p, e, xmin, xmax, window):
    """Signal yield (and its error) from a fitted DCB within an arbitrary
    (lo, hi) sub-window, by scaling n_sig by the DCB CDF's mass fraction
    there. `p`/`e` are the fit's values/errors (e.g. Minuit's m.values,
    m.errors), needing mean/sigma/alphaL/nL/alphaR/nR/n_sig."""
    edge_cdf = dcb_cdf(np.array(window), xmin, xmax, p['mean'], p['sigma'],
                       p['alphaL'], p['nL'], p['alphaR'], p['nR'])
    frac = float(edge_cdf[1] - edge_cdf[0])
    return p['n_sig'] * frac, e['n_sig'] * frac


# --- Pluggable background models ---
#
# A BackgroundModel bundles a background shape's own parameter names with its
# pdf/cdf and default init/limits, so `make_gauss_cost`/`make_dcb_cost` below
# can swap the background out via a `bkg=` name instead of having it
# hardcoded. Registering a future shape (or, per arXiv:1408.6865, eventually
# profiling the choice itself) means adding one entry here -- nothing about
# the signal side, or about existing entries, needs to change.

class BackgroundModel:
    def __init__(self, name, param_names, pdf, cdf, init, limits=None):
        self.name = name
        self.param_names = tuple(param_names)
        self.pdf = pdf
        self.cdf = cdf
        self.init = dict(init)
        self.limits = dict(limits or {})


BACKGROUND_MODELS = {
    'cheb2': BackgroundModel(
        'cheb2', ('c0', 'c1'), cheb_bkg_pdf, cheb_cdf,
        init={'c0': 0.0, 'c1': 0.0}),
    'cheb_exp': BackgroundModel(
        'cheb_exp', ('k0', 'k1'), chebexp_bkg_pdf, chebexp_cdf,
        init={'k0': 0.0, 'k1': 0.0}),
}


# --- iminuit cost functions ---
#
# Each method returns an iminuit.cost.ExtendedBinnedNLL instance. The inner
# function passed to it must return the expected cumulative count at each
# bin edge, cdf[i] = expected events with x < xe[i] (so cdf[0] ~ 0,
# cdf[-1] ~ n_sig + n_bkg) -- exactly `signal_cdf(...)*n_sig +
# bkg_cdf(...)*n_bkg`. Its parameter list varies with the chosen background
# model, so it's a `*args` function whose signature iminuit can't infer by
# introspection; `_make_scaled_cdf` returns the matching parameter names
# alongside it, for ExtendedBinnedNLL's own `name=` argument.

def _make_scaled_cdf(signal_param_names, signal_cdf, bkg_model, xmin, xmax):
    """Returns (scaled_cdf, param_names). scaled_cdf(xe, *params) -> expected
    cumulative count; param_names is signal_param_names + ('n_sig', 'n_bkg')
    + bkg_model.param_names, in the order scaled_cdf expects them."""
    n_sig_params = len(signal_param_names)
    param_names = (tuple(signal_param_names) + ('n_sig', 'n_bkg')
                   + bkg_model.param_names)

    def scaled_cdf(xe, *args):
        sig_args = args[:n_sig_params]
        n_sig, n_bkg = args[n_sig_params], args[n_sig_params + 1]
        bkg_args = args[n_sig_params + 2:]
        return (signal_cdf(xe, xmin, xmax, *sig_args) * n_sig
                + bkg_model.cdf(xe, xmin, xmax, *bkg_args) * n_bkg)

    return scaled_cdf, param_names


def make_gauss_cost(counts, bin_edges, xmin, xmax, bkg='cheb2'):
    """ExtendedBinnedNLL cost for Gaussian signal + a pluggable background
    (`bkg`, a BACKGROUND_MODELS key; default 'cheb2' matches prior behavior).
    Free parameters: mean, sigma, n_sig, n_bkg, then the background model's
    own parameters (c0, c1 for 'cheb2').
    """
    model = BACKGROUND_MODELS[bkg]
    scaled_cdf, param_names = _make_scaled_cdf(('mean', 'sigma'), gauss_cdf, model, xmin, xmax)
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


def make_dcb_cost(counts, bin_edges, xmin, xmax, bkg='cheb2'):
    """
    ExtendedBinnedNLL cost for DCB signal + a pluggable background (`bkg`, a
    BACKGROUND_MODELS key; default 'cheb2' matches prior behavior).
    Free parameters: mean, sigma, alphaL, nL, alphaR, nR, n_sig, n_bkg, then
    the background model's own parameters (c0, c1 for 'cheb2').
    """
    model = BACKGROUND_MODELS[bkg]
    signal_params = ('mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR')
    scaled_cdf, param_names = _make_scaled_cdf(signal_params, dcb_cdf, model, xmin, xmax)
    return ExtendedBinnedNLL(counts, bin_edges, scaled_cdf, name=param_names)


def _dcb_sym_cdf(xe, xmin, xmax, mean, sigma, alpha, n):
    return dcb_cdf(xe, xmin, xmax, mean, sigma, alpha, n, alpha, n)


def make_dcb_sym_cost(counts, bin_edges, xmin, xmax, bkg='cheb2'):
    """
    ExtendedBinnedNLL cost for symmetric DCB (α_R ≡ α_L, n_R ≡ n_L) + a
    pluggable background (`bkg`, a BACKGROUND_MODELS key).
    Free parameters: mean, sigma, alpha, n, n_sig, n_bkg, then the
    background model's own parameters (8 vs 10 for 'cheb2').
    """
    model = BACKGROUND_MODELS[bkg]
    signal_params = ('mean', 'sigma', 'alpha', 'n')
    scaled_cdf, param_names = _make_scaled_cdf(signal_params, _dcb_sym_cdf, model, xmin, xmax)
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
    exactly, rather than discarded (a diagonal-only prior is a strictly
    worse approximation of what the earlier fit actually learned -- it is
    used here only as the fallback below, when there is no real covariance
    to draw from). A single scalar factor is the special case F = factor*I,
    identical to the earlier `cov * factor**2` code.

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
