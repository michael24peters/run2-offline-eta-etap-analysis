"""
Unit tests for cut_utils's fitting counterpart, src/utils/fit_utils.py.

Covers shared infrastructure: run_fit's optional limits/seeds, that
fit_gauss()/fit_dcb() hand back the fitted Minuit object, the pluggable
background models (BACKGROUND_MODELS / bkg= on the cost builders / bkg_model=
on fit_gauss()/fit_dcb()), and build_prior(). All toy data, no ROOT files, a
fixed RNG seed -- consistent with the house style in conftest.py.
"""

import os
import sys

import numpy as np
import pytest
from iminuit import Minuit
from iminuit.cost import NormalConstraint

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'fit'))

import fit_utils
from fit_gauss import fit_gauss
from fit_dcb import fit_dcb

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


# --- run_fit: seeds recovers from a bad starting point -----------------------

def test_seeds_recovers_a_degenerate_start():
    """A start with the peak far outside the window AND a collapsed sigma is
    stuck there (MIGRAD's own retries do not save it -- verified below); a
    seeds= alternative gives run_fit a real starting point to try instead."""
    rng = np.random.default_rng(0)
    counts, edges = toy_histogram(rng, 1500, 6000)
    limits = dict(sigma=(0.0, None), n_sig=(0.0, None), n_bkg=(0.0, None))
    bad_init = dict(mean=200.0, sigma=1e-6, n_sig=1500.0, n_bkg=6000.0, c0=0.0, c1=0.0)

    cost = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    without_seeds = fit_utils.run_fit(cost, bad_init, limits)
    assert not without_seeds.valid

    cost2 = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    with_seeds = fit_utils.run_fit(cost2, bad_init, limits,
                                   seeds=[dict(mean=547.9, sigma=8.0)])
    assert with_seeds.valid


def test_seeds_returns_lowest_fval_when_nothing_converges():
    """If every attempt is invalid, run_fit must still return something (the
    best of a bad lot) rather than None or raising."""
    rng = np.random.default_rng(0)
    counts, edges = toy_histogram(rng, 1500, 6000)
    cost = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX)
    # sigma pinned to a single absurd value everywhere it's tried
    bad_init = dict(mean=547.9, sigma=1e-6, n_sig=1500.0, n_bkg=6000.0, c0=0.0, c1=0.0)
    m = fit_utils.run_fit(cost, bad_init, seeds=[dict(mean=200.0)])
    assert m is not None
    assert isinstance(m, Minuit)


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


# --- mean_prior / sigma_prior / tail_fixed / fit_window wiring ---------------

def test_fit_gauss_sigma_prior_pulls_the_fit():
    """sigma_prior=(value, error) must behave like the manual constraint
    above -- it is the same mechanism, exposed as a keyword."""
    rng = np.random.default_rng(1)
    vals = np.concatenate([rng.normal(547.9, 8.0, 40),
                           rng.uniform(XMIN, XMAX, 400)])
    result_free, _ = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    result_constrained, _ = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60,
                                      sigma_prior=(5.0, 0.3))
    free_sigma = result_free['fit']['parameters']['sigma']['value']
    constrained_sigma = result_constrained['fit']['parameters']['sigma']['value']
    assert abs(constrained_sigma - 5.0) < abs(free_sigma - 5.0)


# --- Pluggable background models ----------------------------------------------

def toy_bkg_from_model(rng, n, bkg_model, oversample=5):
    """Rejection-sample n values from a BACKGROUND_MODELS entry's own pdf over
    [XMIN, XMAX], at its default init -- matched toy data for a model whose
    density isn't flat, so a fit seeded at that same default init converges
    quickly instead of fighting a mismatched toy background."""
    model = fit_utils.BACKGROUND_MODELS[bkg_model]
    params = [model.init[p] for p in model.param_names]
    xg = np.linspace(XMIN, XMAX, 2000)
    ymax = model.pdf(xg, XMIN, XMAX, *params).max() * 1.05
    out = []
    while len(out) < n:
        x = rng.uniform(XMIN, XMAX, n * oversample)
        y = rng.uniform(0.0, ymax, n * oversample)
        out.extend(x[y < model.pdf(x, XMIN, XMAX, *params)])
    return np.array(out[:n])


@pytest.mark.parametrize('bkg_model', ['cheb2', 'cheb_exp'])
def test_background_pdf_integrates_to_one_and_is_nonnegative(bkg_model):
    """Every registered background's pdf must be a real, non-negative density
    over [XMIN, XMAX] across a spread of its own parameter values -- 'cheb2'
    can dip negative in principle (that's the whole reason 'cheb_exp' exists)
    but not at these mild parameter values, so it's included too."""
    model = fit_utils.BACKGROUND_MODELS[bkg_model]
    xg = np.linspace(XMIN, XMAX, 3000)
    default = model.init
    variants = [default]
    if bkg_model == 'cheb2':
        variants.append({'c0': 0.3, 'c1': -0.2})
    elif bkg_model == 'cheb_exp':
        variants += [{'k0': 3.0, 'k1': -2.0}, {'k0': -1.0, 'k1': -3.0}]
    for variant in variants:
        params = [variant[p] for p in model.param_names]
        y = model.pdf(xg, XMIN, XMAX, *params)
        assert (y >= 0).all()
        assert fit_utils._trapz(y, xg) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize('bkg_model', ['cheb2', 'cheb_exp'])
def test_make_gauss_cost_recovers_toy_params_for_each_bkg_model(bkg_model):
    """make_gauss_cost(..., bkg=bkg_model) must build a cost Minuit can
    actually fit, for every registered model -- proof the dynamically-named
    signature (_make_scaled_cdf / ExtendedBinnedNLL's name=) works end to
    end, not just that describe() parses it."""
    rng = np.random.default_rng(10)
    model = fit_utils.BACKGROUND_MODELS[bkg_model]
    sig = rng.normal(547.9, 8.0, 2000)
    bkg = toy_bkg_from_model(rng, 6000, bkg_model)
    counts, edges = np.histogram(np.concatenate([sig, bkg]), bins=60, range=(XMIN, XMAX))
    counts = counts.astype(float)

    cost = fit_utils.make_gauss_cost(counts, edges, XMIN, XMAX, bkg=bkg_model)
    init = dict(mean=547.9, sigma=8.0, n_sig=2000.0, n_bkg=6000.0, **model.init)
    limits = dict(n_sig=(0.0, None), n_bkg=(0.0, None), **model.limits)
    m = fit_utils.run_fit(cost, init, limits)

    assert m.valid
    assert set(m.parameters) == {'mean', 'sigma', 'n_sig', 'n_bkg', *model.param_names}
    assert m.values['mean'] == pytest.approx(547.9, abs=1.0)


@pytest.mark.parametrize('bkg_model', ['cheb2', 'cheb_exp'])
def test_fit_dcb_bkg_model_roundtrips_through_json(bkg_model):
    """fit_dcb(bkg_model=...) must record meta.background_model, name the
    background parameters in the result JSON after that model's own
    param_names, and a LoadedFit built from that JSON must expose them under
    the same names (so plot_dcb.py / a later per-bin fit reading this JSON
    back doesn't need to guess which shape produced it)."""
    rng = np.random.default_rng(11)
    model = fit_utils.BACKGROUND_MODELS[bkg_model]
    sig = rng.normal(547.9, 8.0, 2000)
    bkg = toy_bkg_from_model(rng, 6000, bkg_model)
    vals = np.concatenate([sig, bkg])

    result, m = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60, bkg_model=bkg_model)

    assert result['meta']['background_model'] == bkg_model
    for name in model.param_names:
        assert name in result['fit']['parameters']
        # JSON values are round()ed to 6 decimals -- an absolute, not relative,
        # tolerance, so a value near 0 needs abs= rather than approx()'s default
        # relative one.
        assert result['fit']['parameters'][name]['value'] == pytest.approx(
            m.values[name], abs=1e-5)

    loaded = fit_utils.LoadedFit(result['fit'])
    for name in model.param_names:
        assert name in loaded.values


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


def test_build_prior_per_parameter_factor_preserves_correlation():
    """A {param: factor} dict must scale each parameter's own marginal error
    by its own factor while leaving the correlation coefficient between them
    exactly as the original fit measured it -- the whole point of the
    outer-product (F @ cov @ F) scaling over a plain cov*factor**2."""
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    _, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    raw = np.array(m.covariance[['n_sig', 'n_bkg']])

    def corr(cov):
        return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    _, cov = fit_utils.build_prior(m, ['n_sig', 'n_bkg'],
                                   factor={'n_sig': 5.0, 'n_bkg': 1.0})

    assert cov[0, 0] == pytest.approx(raw[0, 0] * 25.0)
    assert cov[1, 1] == pytest.approx(raw[1, 1])
    assert corr(cov) == pytest.approx(corr(raw))


def test_build_prior_missing_dict_entry_defaults_to_one():
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(547.9, 8.0, 1500),
                           rng.uniform(XMIN, XMAX, 6000)])
    _, m = fit_gauss(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    raw = np.array(m.covariance[['n_sig', 'n_bkg']])

    _, cov = fit_utils.build_prior(m, ['n_sig', 'n_bkg'], factor={'n_sig': 3.0})

    assert cov[0, 0] == pytest.approx(raw[0, 0] * 9.0)
    assert cov[1, 1] == pytest.approx(raw[1, 1])  # n_bkg missing -> factor 1.0


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
    """A LoadedFit with no 'covariance' block in its JSON (HESSE's matrix
    inversion failed on the original fit) must fall back to a diagonal
    matrix built from .errors, not raise."""
    fit_block = {
        'valid': True, 'fval': 0.0, 'n_free_params': 2,
        'parameters': {
            'n_sig': {'value': 100.0, 'error': 10.0},
            'n_bkg': {'value': 500.0, 'error': 20.0},
        },
    }
    loaded = fit_utils.LoadedFit(fit_block)
    assert loaded.covariance is None

    values, cov = fit_utils.build_prior(loaded, ['n_sig', 'n_bkg'], factor=2.0)

    assert values == pytest.approx([100.0, 500.0])
    assert cov == pytest.approx(np.diag([10.0**2, 20.0**2]) * 4.0)
    assert cov[0, 1] == 0.0
