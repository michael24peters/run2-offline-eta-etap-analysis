"""
Unit tests for cut_utils's fitting counterpart, src/utils/fit_utils.py.

Covers shared infrastructure: run_fit's optional limits, that
fit_gauss()/fit_dcb() hand back the fitted Minuit object, the Chebyshev
background (pdf, cost and result JSON), and build_prior(). All toy data, no ROOT files, a
fixed RNG seed -- consistent with the house style in conftest.py.
"""

import os
import sys

import numpy as np
import pytest
from iminuit import Minuit
from iminuit.cost import NormalConstraint
from scipy.integrate import trapezoid

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'fit'))

import fit_utils
from fit_dcb import fit_dcb, tail_n
from fit_gauss import fit_gauss

XMIN, XMAX = 480.0, 620.0


def toy_histogram(rng, n_sig, n_bkg, mean=547.9, sigma=8.0, nbins=60):
    """Gaussian signal + flat background, histogrammed over [XMIN, XMAX]."""
    vals = np.concatenate([rng.normal(mean, sigma, n_sig),
                           rng.uniform(XMIN, XMAX, n_bkg)])
    counts, edges = np.histogram(vals, bins=nbins, range=(XMIN, XMAX))
    return counts.astype(float), edges


# --- run_fit: limits is now optional -----------------------------------------

def test_run_fit_without_limits():
    """limits=None (or omitted) must not raise -- the `(limits or {})` guard."""
    rng = np.random.default_rng(0)
    counts, edges = toy_histogram(rng, 1500, 6000)
    cost = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    init = dict(mean=547.9, sigma=8.0, n_sig=1500.0, n_bkg=6000.0, c0=0.0, c1=0.0)
    m = fit_utils.run_fit(cost, init)
    assert isinstance(m, Minuit)
    assert m.valid


def test_run_fit_partial_limits():
    """Only some parameters need a bound; the rest are simply never boxed."""
    rng = np.random.default_rng(0)
    counts, edges = toy_histogram(rng, 1500, 6000)
    cost = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    init = dict(mean=547.9, sigma=8.0, n_sig=1500.0, n_bkg=6000.0, c0=0.0, c1=0.0)
    m = fit_utils.run_fit(cost, init, limits=dict(sigma=(0.0, None)))
    assert m.valid


# --- NormalConstraint is actually wired into the minimised cost --------------

def test_constraint_pulls_a_low_statistics_fit():
    """With few signal events, sigma is poorly determined by the data alone.
    A NormalConstraint on sigma must pull the fitted value measurably closer
    to the constrained value than the same fit with no constraint -- proof the
    constraint changes what gets minimised, not just decoration on the cost."""
    rng = np.random.default_rng(1)
    counts, edges = toy_histogram(rng, n_sig=40, n_bkg=400, sigma=8.0)
    limits = dict(sigma=(0.0, None), n_sig=(0.0, None), n_bkg=(0.0, None))
    init = dict(mean=547.9, sigma=8.0, n_sig=40.0, n_bkg=400.0, c0=0.0, c1=0.0)

    cost_free = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    m_free = fit_utils.run_fit(cost_free, init, limits, fixed=['c0', 'c1'])

    cost_constrained = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    cost_constrained = cost_constrained + NormalConstraint('sigma', 5.0, error=0.3)
    m_constrained = fit_utils.run_fit(cost_constrained, init, limits,
                                      fixed=['c0', 'c1'])

    assert (abs(m_constrained.values['sigma'] - 5.0)
            < abs(m_free.values['sigma'] - 5.0))


# --- fit_gauss() / fit_dcb() return (result, m) ------------------------------

def test_fit_gauss_returns_result_and_minuit():
    rng = np.random.default_rng(2)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    result, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    assert isinstance(result, dict)
    assert isinstance(m, Minuit)
    assert result['fit']['parameters']['sigma']['value'] == pytest.approx(
        m.values['sigma'])


def test_fit_gauss_minuit_covariance_is_usable():
    """The whole point of returning m: a later fit can build a real,
    correlation-aware constraint from it."""
    rng = np.random.default_rng(2)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    _, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    cov = np.array(m.covariance[['c0', 'c1']])
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0 and cov[1, 1] > 0


def test_fit_dcb_returns_result_and_minuit():
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    result, m = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    assert isinstance(result, dict)
    assert isinstance(m, Minuit)


def test_fit_dcb_fixed_tails_given_as_n_hold_inv_n():
    """fixed_params gives the tails as nL, nR; the fit holds inv_nL, inv_nR at
    1/n, and the result reports the given n."""
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    tails = {'alphaL': 1.2, 'nL': 4.0, 'alphaR': 1.5, 'nR': 2.5}
    result, m = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60, fixed_params=tails)
    assert m.fixed['inv_nL'] and m.values['inv_nL'] == pytest.approx(0.25)
    assert m.fixed['inv_nR'] and m.values['inv_nR'] == pytest.approx(0.4)
    params = result['fit']['parameters']
    assert params['nL']['value'] == pytest.approx(4.0)
    assert params['nR']['value'] == pytest.approx(2.5)


def test_fit_dcb_result_reports_n_with_errors_from_inv_n():
    """The result names the tails nL, nR, with the error of n that of inv_n
    times n^2 and the variance of n that of inv_n times n^4."""
    rng = np.random.default_rng(12)
    vals = np.concatenate([547.9 + 8.0 * rng.standard_t(3, 4000),
                           rng.uniform(XMIN, XMAX, 4000)])
    result, m = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    covariance = result['fit']['covariance']
    assert 'inv_nL' not in covariance['names']
    for name, inv_name in (('nL', 'inv_nL'), ('nR', 'inv_nR')):
        param = result['fit']['parameters'][name]
        assert param['value'] == pytest.approx(tail_n(m.values[inv_name]), rel=1e-5)
        assert param['error'] == pytest.approx(m.errors[inv_name] * param['value']**2, rel=1e-4)
        i = covariance['names'].index(name)
        assert covariance['matrix'][i][i] == pytest.approx(
            m.covariance[inv_name, inv_name] * param['value']**4, rel=1e-4)


def test_fit_gauss_fixed_params_stay_fixed():
    """fixed_params holds its parameters at the given values through both
    stages, as for a signal-only fit of truth-matched MC."""
    rng = np.random.default_rng(3)
    vals = rng.normal(547.9, 8.0, 1500)
    _, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60,
                     fixed_params={'n_bkg': 0.0, **fit_utils.BKG_INIT})
    assert m.fixed['n_bkg'] and m.values['n_bkg'] == 0.0
    assert m.values['n_sig'] == pytest.approx(1500, rel=0.05)


# --- Chebyshev background ---------------------------------------------------

def test_background_pdf_integrates_to_one_and_is_nonnegative():
    """The Chebyshev pdf is a real, non-negative density over [XMIN, XMAX] at
    its default and at mild coefficients (it can dip negative in principle,
    but not at these values)."""
    xg = np.linspace(XMIN, XMAX, 3000)
    for c0, c1 in [(fit_utils.BKG_INIT['c0'], fit_utils.BKG_INIT['c1']), (0.3, -0.2)]:
        y = fit_utils.cheb_bkg_pdf(xg, XMIN, XMAX, c0, c1)
        assert (y >= 0).all()
        assert trapezoid(y, xg) == pytest.approx(1.0, abs=1e-3)


def test_make_gauss_cost_recovers_toy_params():
    """make_gauss_cost builds a cost Minuit can actually fit -- proof the
    dynamically-named signature (_make_scaled_cdf / ExtendedBinnedNLL's
    name=) works end to end, not just that describe() parses it."""
    rng = np.random.default_rng(10)
    sig = rng.normal(547.9, 8.0, 2000)
    bkg = rng.uniform(XMIN, XMAX, 6000)
    counts, edges = np.histogram(np.concatenate([sig, bkg]), bins=60, range=(XMIN, XMAX))
    counts = counts.astype(float)

    cost = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    init = dict(mean=547.9, sigma=8.0, n_sig=2000.0, n_bkg=6000.0, **fit_utils.BKG_INIT)
    limits = dict(n_sig=(0.0, None), n_bkg=(0.0, None))
    m = fit_utils.run_fit(cost, init, limits)

    assert m.valid
    assert set(m.parameters) == {'mean', 'sigma', 'n_sig', 'n_bkg', *fit_utils.BKG_PARAMS}
    assert m.values['mean'] == pytest.approx(547.9, abs=1.0)


def test_fit_dcb_bkg_params_roundtrip_through_json():
    """fit_dcb's result JSON names the background parameters c0, c1, and
    build_prior_from_result on that JSON gives the live fit's values and
    covariance for them (so a later fit reading this JSON back finds them)."""
    rng = np.random.default_rng(11)
    sig = rng.normal(547.9, 8.0, 2000)
    bkg = rng.uniform(XMIN, XMAX, 6000)
    vals = np.concatenate([sig, bkg])

    result, m = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)

    for name in fit_utils.BKG_PARAMS:
        assert name in result['fit']['parameters']
        # JSON values are round()ed to 6 decimals -- an absolute, not relative,
        # tolerance, so a value near 0 needs abs= rather than approx()'s default
        # relative one.
        assert result['fit']['parameters'][name]['value'] == pytest.approx(
            m.values[name], abs=1e-5)

    values, cov = fit_utils.build_prior_from_result(result['fit'], fit_utils.BKG_PARAMS)
    live_values, live_cov = fit_utils.build_prior(m, fit_utils.BKG_PARAMS)
    assert values == pytest.approx(live_values, abs=1e-5)
    assert cov == pytest.approx(live_cov)


# --- build_prior -----------------------------------------------------------

def test_build_prior_scalar_factor_matches_cov_times_factor_squared():
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    _, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)

    values, cov = fit_utils.build_prior(m, ['n_sig', 'n_bkg'], factor=5.0)
    raw = np.array(m.covariance[['n_sig', 'n_bkg']])

    assert values == pytest.approx([m.values['n_sig'], m.values['n_bkg']])
    assert cov == pytest.approx(raw * 25.0)


def test_build_prior_excluded_param_gets_no_prior():
    """A parameter simply left out of `params` is how a prior is dropped
    entirely -- build_prior only ever returns values/cov for what it's asked
    for, so the caller's NormalConstraint(params, ...) never mentions it."""
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    _, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)

    values, cov = fit_utils.build_prior(m, ['n_sig'], factor=5.0)

    assert values.shape == (1,)
    assert cov.shape == (1, 1)


def test_build_prior_falls_back_to_diagonal_when_covariance_is_none():
    """A results JSON with no 'covariance' block (HESSE's matrix inversion
    failed on the original fit) must fall back to a diagonal matrix built
    from the errors, not raise."""
    fit_block = {
        'valid': True, 'fval': 0.0, 'n_free_params': 2,
        'parameters': {
            'n_sig': {'value': 100.0, 'error': 10.0},
            'n_bkg': {'value': 500.0, 'error': 20.0},
        },
    }
    values, cov = fit_utils.build_prior_from_result(fit_block, ['n_sig', 'n_bkg'], factor=2.0)

    assert values == pytest.approx([100.0, 500.0])
    assert cov == pytest.approx(np.diag([10.0**2, 20.0**2]) * 4.0)
    assert cov[0, 1] == 0.0


# --- DCB tails ---------------------------------------------------------------

def test_dcb_tails_match_the_power_law_and_stay_finite_for_large_n():
    """dcb_unnorm's tails equal A * (B + u)^(-n), A = (n/alpha)^n exp(-alpha^2/2),
    B = n/alpha - alpha, where that is finite, and stay finite where A alone
    overflows."""
    t = np.linspace(-20, 20, 4001)
    for alpha, n in [(1.5, 2.0), (0.8, 5.5), (2.5, 0.2), (1.2, 40.0)]:
        A = (n / alpha)**n * np.exp(-0.5 * alpha**2)
        B = n / alpha - alpha
        expected = np.where(np.abs(t) <= alpha, np.exp(-0.5 * t**2),
                            A * (B + np.abs(t))**(-n))
        assert np.allclose(fit_utils.dcb_unnorm(t, 0.0, 1.0, alpha, n, alpha, n), expected,
                           rtol=1e-12, atol=0)
    y = fit_utils.dcb_unnorm(t, 0.0, 1.0, 0.5, 2000.0, 0.5, 2000.0)
    assert np.all(np.isfinite(y)) and np.all(y >= 0)
