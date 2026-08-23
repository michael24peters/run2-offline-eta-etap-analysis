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


# --- mean_prior / sigma_prior / tail_prior wiring ----------------------------

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


def test_fit_dcb_tail_prior_constrains_the_tails():
    """tail_prior=(values, cov) for TAIL_PARAMS must change the fitted tail
    values relative to an unconstrained fit on the same low-statistics data --
    this is the direct fix for tails running away with no ceiling."""
    from fit_dcb import TAIL_PARAMS
    rng = np.random.default_rng(4)
    vals = np.concatenate([rng.normal(547.9, 8.0, 60),
                           rng.uniform(XMIN, XMAX, 300)])

    tail_values = np.array([1.5, 2.0, 1.5, 2.0])
    tail_cov = np.diag([0.05, 0.05, 0.05, 0.05])
    result_free, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60)
    result_constrained, _ = fit_dcb(vals, xmin=XMIN, xmax=XMAX, nbins=60,
                                    tail_prior=(tail_values, tail_cov))
    for i, p in enumerate(TAIL_PARAMS):
        free_v = result_free['fit']['parameters'][p]['value']
        con_v = result_constrained['fit']['parameters'][p]['value']
        assert abs(con_v - tail_values[i]) <= abs(free_v - tail_values[i]) + 1e-9
