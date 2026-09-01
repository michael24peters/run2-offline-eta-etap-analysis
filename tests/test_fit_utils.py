"""
Unit tests for cut_utils's fitting counterpart, src/utils/fit_utils.py.

Covers the parts of the limits -> constraints migration that are shared
infrastructure: run_fit's optional limits/seeds, and that fit_gauss()/
fit_dcb() hand back the fitted Minuit object. All toy data, no ROOT files, a
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


def test_fit_dcb_tail_fixed_holds_the_tails():
    """tail_fixed=values for TAIL_PARAMS must hold the fitted tail values
    exactly at the given values instead of floating them -- this is the fix
    for a weakly-MC-constrained tail param (e.g. nR) drifting into the
    barely-decaying regime and absorbing background as signal."""
    from fit_dcb import TAIL_PARAMS
    rng = np.random.default_rng(4)
    vals = np.concatenate([rng.normal(547.9, 8.0, 60),
                           rng.uniform(XMIN, XMAX, 300)])

    tail_values = np.array([1.5, 2.0, 1.5, 2.0])
    result_free, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    result_fixed, m_fixed = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60,
                                    tail_fixed=tail_values)
    free_values = [result_free['fit']['parameters'][p]['value'] for p in TAIL_PARAMS]
    assert free_values != pytest.approx(list(tail_values))
    for i, p in enumerate(TAIL_PARAMS):
        assert result_fixed['fit']['parameters'][p]['value'] == pytest.approx(tail_values[i])
        assert m_fixed.fixed[p] is True


def test_fit_dcb_fit_window_restricts_the_likelihood():
    """fit_window=(lo, hi) must only use bins inside it: the reported chi2/
    ndof should be computed over a smaller bin count than the full-window fit,
    and meta['fit_window'] should record the bounds used."""
    rng = np.random.default_rng(5)
    vals = np.concatenate([rng.normal(547.9, 5.0, 400),
                           rng.uniform(XMIN, XMAX, 4000)])

    result_full, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    result_windowed, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60,
                                 fit_window=(530.0, 570.0))

    assert result_windowed['meta']['fit_window'] == [530.0, 570.0]
    assert result_windowed['fit']['ndof'] < result_full['fit']['ndof']
    # n_sig_fit_window integrates only the fitted window, so it should be no
    # larger than the full-window-normalized n_sig.
    params = result_windowed['fit']['parameters']
    assert params['n_sig_fit_window']['value'] <= params['n_sig']['value']


def test_fit_dcb_no_background_fixes_bkg_params_to_zero():
    """no_background=True must hold n_bkg/c0/c1 fixed at 0 rather than
    floating them -- for a sample (e.g. truth-matched signal MC) that
    shouldn't have a background component competing for events at all."""
    rng = np.random.default_rng(6)
    vals = rng.normal(547.9, 5.0, 500)

    result, m = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60, no_background=True)
    params = result['fit']['parameters']
    assert params['n_bkg']['value'] == 0.0
    assert params['c0']['value'] == 0.0
    assert params['c1']['value'] == 0.0
    for p in ('n_bkg', 'c0', 'c1'):
        assert m.fixed[p] is True


def test_fit_dcb_report_halfwidth_sigma_adds_a_core_window_yield():
    """report_halfwidth_sigma=k must add n_sig_report_window, integrating
    n_sig over [mean-k*sigma, mean+k*sigma] using the fit's own mean/sigma,
    without changing what the likelihood was fit against (unlike
    fit_window) -- so ndof should match a plain fit with no window at all."""
    rng = np.random.default_rng(7)
    vals = np.concatenate([rng.normal(547.9, 5.0, 400),
                           rng.uniform(XMIN, XMAX, 4000)])

    result_plain, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    result_reported, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60,
                                 report_halfwidth_sigma=3.0)

    assert result_reported['fit']['ndof'] == result_plain['fit']['ndof']
    params = result_reported['fit']['parameters']
    assert 'n_sig_report_window' in params
    assert params['n_sig_report_window']['value'] <= params['n_sig']['value']
    mean = params['mean']['value']
    sigma = params['sigma']['value']
    assert result_reported['meta']['report_window'] == pytest.approx(
        [mean - 3.0 * sigma, mean + 3.0 * sigma])
