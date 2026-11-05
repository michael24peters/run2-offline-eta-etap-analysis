"""
Unit tests for the Dalitz phase-space factor: kinematics.m123_limits / kinematics.m12_limits /
kinematics.phase_space_factor, the Dalitz-decay factor kinematics.dalitz_factor, and the
phase-space-weighted shapes in fit_utils (weighted_dcb_cdf, weighted_cheb_cdf/pdf,
make_dcb_cost(signal_factor_grid=, bkg_factor_grid=)). Toy data only, fixed RNG seeds.
"""

import os
import sys

import numpy as np
import pytest
from scipy.integrate import trapezoid

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'utils'))

import fit_utils
import kinematics

XMIN, XMAX = 420.0, 680.0
MMU = kinematics.M_MU / 1000.0
META = kinematics.M_ETA / 1000.0


def toy_decays(rng, n, M):
    """gamma mu+ mu- four-vectors (GeV) for systems of mass M (GeV, array), built in the mu mu rest
    frame from a random dimuon mass and a random muon direction -- independent of the closed-form
    limits under test."""
    m23 = rng.uniform(2 * MMU + 1e-4, M - 1e-4)
    q = np.sqrt(m23**2 / 4 - MMU**2)                    # muon momentum
    k = (M**2 - m23**2) / (2 * m23)                     # photon energy
    cos_g = rng.uniform(-1, 1, n)
    sin_g = np.sqrt(1 - cos_g**2)
    cos_m = rng.uniform(-1, 1, n)
    sin_m = np.sqrt(1 - cos_m**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    gamma = (k * sin_g, 0 * k, k * cos_g, k)
    mup = (q * sin_m * np.cos(phi), q * sin_m * np.sin(phi), q * cos_m,
           np.sqrt(q**2 + MMU**2))
    mum = tuple(-c for c in mup[:3]) + (mup[3],)
    return gamma, mup, mum


def test_m123_limits_contain_the_true_mass():
    rng = np.random.default_rng(1)
    M = rng.uniform(0.3, 0.8, 20000)
    gamma, mup, mum = toy_decays(rng, M.size, M)
    m12s = kinematics.invariant_mass2(*gamma, *mup)
    m23s = kinematics.invariant_mass2(*mup, *mum)
    lo, hi = kinematics.m123_limits(m12s, m23s)
    assert np.all(lo <= M * 1000 + 1e-6)
    assert np.all(hi >= M * 1000 - 1e-6)
    # And the limits are tight: some toys sit right on each edge.
    assert np.min(M * 1000 - lo) < 0.5
    assert np.min(hi - M * 1000) < 0.5


def test_m12_limits_contain_the_true_m12():
    rng = np.random.default_rng(1)
    M = rng.uniform(0.3, 0.8, 20000)
    gamma, mup, mum = toy_decays(rng, M.size, M)
    m12s = kinematics.invariant_mass2(*gamma, *mup)
    m23s = kinematics.invariant_mass2(*mup, *mum)
    lo, hi = kinematics.m12_limits(M * 1000, m23s)
    assert np.all(lo <= m12s + 1e-9)
    assert np.all(hi >= m12s - 1e-9)
    # And the limits are tight: some toys sit right on each edge.
    assert np.min(m12s - lo) < 1e-3
    assert np.min(hi - m12s) < 1e-3


def test_phase_space_factor_matches_uniform_sampling():
    """The semi-analytic factor equals the Monte Carlo recipe: sample the bin uniformly, count how
    often M is inside m123_limits."""
    rng = np.random.default_rng(2)
    M = np.linspace(XMIN, XMAX, 27)
    for m12_range, m23_range in [((0.015, 0.030), (0.2879, 0.3100)),
                                 ((0.015, 0.030), (0.0886, 0.1107)),
                                 ((0.105, 0.120), (0.0443, 0.0664))]:
        m12 = rng.uniform(*m12_range, 400000)
        m23 = rng.uniform(*m23_range, 400000)
        lo, hi = kinematics.m123_limits(m12, m23)
        mc = np.array([np.mean((lo <= m) & (m <= hi)) for m in M])
        f = kinematics.phase_space_factor(M, m12_range, m23_range)
        assert np.max(np.abs(f - mc)) < 5e-3


def test_phase_space_factor_bounds():
    M = np.linspace(XMIN, XMAX, 100)
    f = kinematics.phase_space_factor(M, (0.015, 0.030), (0.0886, 0.1107))
    assert np.all((f >= 0) & (f <= 1))
    # Bin sits entirely outside the eta's Dalitz region...
    assert kinematics.phase_space_factor([kinematics.M_ETA], (0.015, 0.030),
                                         (0.0886, 0.1107))[0] == 0.0
    # ...and a bin deep inside is fully reachable near the eta mass. (Not across the whole 420-680
    # window: at high M the low-m12 corner of most bins becomes unreachable too.)
    f_in = kinematics.phase_space_factor(np.linspace(530, 565, 20),
                                         (0.090, 0.105), (0.1107, 0.1329))
    assert np.allclose(f_in, 1.0)


def dalitz_weight(m12, m23):
    """eta -> gamma mu+ mu- matrix element at the eta mass for m12 = m^2(gamma mu+), m23 = q^2 =
    m^2(mu+ mu-) (GeV^2), with m13 = m^2(gamma mu-):

        |F(q^2)|^2 / q^4 [q^2 ((m12 - m^2)^2 + (m13 - m^2)^2) + 2 m^2 (m_eta^2 - q^2)^2]

    0 for q^2 <= 4 m^2. Written independently of dalitz_factor's form."""
    m_mu2, m_eta2 = MMU**2, META**2
    m13 = m_eta2 + 2 * m_mu2 - m23 - m12
    form_factor2 = (1 - m23 / kinematics.FF_LAMBDA**2)**-2
    weight = form_factor2 / m23**2 * (m23 * ((m12 - m_mu2)**2 + (m13 - m_mu2)**2)
                                      + 2 * m_mu2 * (m_eta2 - m23)**2)
    return np.where(m23 > 4 * m_mu2, weight, 0.0)


def test_dalitz_factor_matches_weighted_sampling():
    """The semi-analytic signal factor equals the Monte Carlo recipe: sample the bin uniformly,
    weight by the matrix element, and take the weighted fraction with M inside m123_limits."""
    rng = np.random.default_rng(5)
    M = np.linspace(XMIN, XMAX, 27)
    for m12_range, m23_range in [((0.015, 0.030), (0.2879, 0.3100)),
                                 ((0.015, 0.030), (0.0886, 0.1107)),
                                 ((0.105, 0.120), (0.0443, 0.0664)),
                                 ((0.180, 0.195), (0.0443, 0.0664))]:
        m12 = rng.uniform(*m12_range, 400000)
        m23 = rng.uniform(*m23_range, 400000)
        weight = dalitz_weight(m12, m23)
        m_min, m_max = kinematics.m123_limits(m12, m23)
        f = kinematics.dalitz_factor(M, m12_range, m23_range)
        for m, f_m in zip(M, f, strict=True):
            reachable = (m_min <= m) & (m <= m_max)
            mc = np.sum(weight * reachable) / np.sum(weight)
            err = np.sqrt(np.sum((weight * (reachable - mc))**2)) / np.sum(weight)
            assert abs(f_m - mc) < 4 * err + 3e-3


def test_dalitz_factor_bounds():
    M = np.linspace(XMIN, XMAX, 100)
    f = kinematics.dalitz_factor(M, (0.015, 0.030), (0.0886, 0.1107))
    assert np.all((f >= 0) & (f <= 1))
    # Bin entirely outside the eta's Dalitz region...
    assert kinematics.dalitz_factor([kinematics.M_ETA], (0.015, 0.030),
                                    (0.0886, 0.1107))[0] == 0.0
    # ...and a bin entirely inside it near the eta mass.
    f_in = kinematics.dalitz_factor(np.linspace(530, 565, 20),
                                    (0.090, 0.105), (0.1107, 0.1329))
    assert np.allclose(f_in, 1.0)


def test_dalitz_factor_approaches_phase_space_factor_for_small_bins():
    """Bins centred on the boundary of D(m_eta): the largest difference over M between dalitz_factor
    and phase_space_factor shrinks with the bin size."""
    M = np.linspace(XMIN, XMAX, 261)
    for q2_0 in [0.06, 0.12, 0.20, 0.27]:
        # The two m12 edges of D(m_eta) at this q2.
        for m12_0 in kinematics.m12_limits(kinematics.M_ETA, q2_0):
            max_diffs = []
            for scale in [1, 1 / 4, 1 / 16]:
                # Grid bin size times scale, centred on the boundary point.
                width12, width23 = 0.015 * scale, 0.31 / 14 * scale
                m12_range = (m12_0 - width12 / 2, m12_0 + width12 / 2)
                m23_range = (q2_0 - width23 / 2, q2_0 + width23 / 2)
                diff = np.abs(kinematics.dalitz_factor(M, m12_range, m23_range)
                              - kinematics.phase_space_factor(M, m12_range, m23_range))
                max_diffs.append(diff.max())
            assert max_diffs[0] > max_diffs[1] > max_diffs[2], (q2_0, m12_0, max_diffs)
            assert max_diffs[2] < 0.2 * max_diffs[0], (q2_0, m12_0, max_diffs)
            assert max_diffs[2] < 0.015, (q2_0, m12_0, max_diffs)


def test_unit_factor_reproduces_the_unweighted_shapes():
    factor_grid = fit_utils.sample_factor(np.ones_like, XMIN, XMAX)
    x = np.linspace(XMIN, XMAX, 50)
    dcb_args = (547.9, 12.0, 1.5, 2.0, 1.5, 2.0)
    assert np.allclose(fit_utils.weighted_dcb_cdf(factor_grid)(x, XMIN, XMAX, *dcb_args),
                       fit_utils.dcb_cdf(x, XMIN, XMAX, *dcb_args), atol=1e-4)
    assert np.allclose(fit_utils.weighted_cheb_cdf(factor_grid)(x, XMIN, XMAX, 0.3, -0.2),
                       fit_utils.cheb_cdf(x, XMIN, XMAX, 0.3, -0.2), atol=1e-4)


def test_weighted_background_is_zero_where_forbidden():
    m12_range, m23_range = (0.015, 0.030), (0.2879, 0.3100)
    factor_grid = fit_utils.sample_factor(
        lambda M: kinematics.phase_space_factor(M, m12_range, m23_range), XMIN, XMAX)
    weighted_cdf = fit_utils.weighted_cheb_cdf(factor_grid)
    weighted_pdf = fit_utils.weighted_cheb_pdf(factor_grid)
    # f = 0 below ~541 MeV in this bin, so no background probability there, even for a Chebyshev
    # that is negative somewhere (floored, not flipped).
    for c0, c1 in [(0.0, 0.0), (1.5, 0.0)]:
        assert weighted_cdf(np.array([535.0]), XMIN, XMAX, c0, c1)[0] == 0.0
        y = weighted_pdf(np.linspace(XMIN, XMAX, 500), XMIN, XMAX, c0, c1)
        assert np.all(y >= 0)
        assert trapezoid(y, np.linspace(XMIN, XMAX, 500)) == pytest.approx(1, abs=1e-2)


def test_weighted_fit_recovers_background_toy():
    """Background drawn from f * Chebyshev in an edge bin: the weighted cost fits it back with the
    signal yield fixed at 0 (as for a bin with f(m_eta) = 0), including the zero-factor_grid signal
    component."""
    rng = np.random.default_rng(3)
    m12_range, m23_range = (0.015, 0.030), (0.0886, 0.1107)
    factor_grid = fit_utils.sample_factor(
        lambda M: kinematics.phase_space_factor(M, m12_range, m23_range), XMIN, XMAX)
    masses, factor = factor_grid
    shape = fit_utils.cheb_bkg_unnorm(masses, XMIN, XMAX, -0.5, 0.1) * factor
    cdf = np.concatenate([[0], np.cumsum(0.5 * (shape[1:] + shape[:-1]) * np.diff(masses))])
    vals = np.interp(rng.uniform(0, cdf[-1], 3000), cdf, masses)
    counts, edges = np.histogram(vals, bins=80, range=(XMIN, XMAX))

    cost = fit_utils.make_dcb_cost(counts.astype(float), edges, XMIN, XMAX,
                                   signal_factor_grid=factor_grid, bkg_factor_grid=factor_grid)
    init = dict(mean=547.9, sigma=12.0, alphaL=1.5, nL=2.0, alphaR=1.5, nR=2.0,
                n_sig=0.0, n_bkg=3000.0, c0=0.0, c1=0.0)
    m = fit_utils.run_fit(cost, init, dict(n_bkg=(0, None)),
                          fixed=['mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'n_sig'])
    assert m.valid
    assert m.values['n_bkg'] == pytest.approx(3000, rel=0.05)
    # Only ~40% of the window is reachable here, so c0/c1 are individually degenerate; compare the
    # fitted shape to the true one instead.
    fit_cdf = fit_utils.weighted_cheb_cdf(factor_grid)(edges, XMIN, XMAX,
                                                  m.values['c0'], m.values['c1'])
    assert np.max(np.abs(fit_cdf - np.interp(edges, masses, cdf / cdf[-1]))) < 0.03


def edge_bin_factor(M):
    """Phase-space factor of an edge bin that reaches only part of the window."""
    return kinematics.phase_space_factor(M, (0.015, 0.030), (0.0886, 0.1107))


@pytest.mark.parametrize('weighted', ['signal', 'bkg'])
def test_each_factor_changes_only_its_own_component(weighted):
    """With the other component's yield at 0, a cost with a factor on one side equals the unweighted
    cost; with this component's yield at 0 it differs."""
    factor_grid = fit_utils.sample_factor(edge_bin_factor, XMIN, XMAX)
    rng = np.random.default_rng(4)
    counts, edges = np.histogram(rng.normal(547.9, 15.0, 2000), bins=80, range=(XMIN, XMAX))
    counts = counts.astype(float)
    if weighted == 'signal':
        cost_weighted = fit_utils.make_dcb_cost(counts, edges, XMIN, XMAX,
                                                signal_factor_grid=factor_grid)
    else:
        cost_weighted = fit_utils.make_dcb_cost(counts, edges, XMIN, XMAX,
                                                bkg_factor_grid=factor_grid)
    cost_unweighted = fit_utils.make_dcb_cost(counts, edges, XMIN, XMAX)

    # Parameter order: mean, sigma, alphaL, nL, alphaR, nR, n_sig, n_bkg, c0, c1.
    dcb_params = (547.9, 15.0, 1.5, 2.0, 1.5, 2.0)
    signal_only = (*dcb_params, 2000.0, 0.0, 0.2, -0.1)
    bkg_only = (*dcb_params, 0.0, 2000.0, 0.2, -0.1)
    untouched, touched = (bkg_only, signal_only) if weighted == 'signal' else (signal_only, bkg_only)
    assert cost_weighted(*untouched) == pytest.approx(cost_unweighted(*untouched), rel=1e-12)
    assert cost_weighted(*touched) != pytest.approx(cost_unweighted(*touched), rel=1e-3)
