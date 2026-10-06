"""
Unit tests for the Dalitz phase-space factor: kinematics.m123_limits /
kinematics.phase_space_factor, the signal factor kinematics.kroll_wada_factor,
and the phase-space-weighted shapes in fit_utils (weighted_dcb_cdf,
weighted_bkg_model, make_dcb_cost(weight=)). Toy data only, fixed RNG seeds.
"""

import os
import sys

import numpy as np
import pytest

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'utils'))

import fit_utils
import kinematics

XMIN, XMAX = 420.0, 680.0
MMU = kinematics.M_MU / 1000.0
META = kinematics.M_ETA / 1000.0
FF_LAMBDA = 0.71907


def toy_decays(rng, n, M):
    """gamma mu+ mu- four-vectors (GeV) for systems of mass M (GeV, array),
    built in the mu mu rest frame from a random dimuon mass and a random
    muon direction -- independent of the closed-form limits under test."""
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


def test_phase_space_factor_matches_uniform_sampling():
    """The semi-analytic factor equals the Monte Carlo recipe: sample the bin
    uniformly, count how often M is inside m123_limits."""
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
    # ...and a bin deep inside is fully reachable near the eta mass. (Not
    # across the whole 420-680 window: at high M the low-m12 corner of most
    # bins becomes unreachable too.)
    f_in = kinematics.phase_space_factor(np.linspace(530, 565, 20),
                                         (0.090, 0.105), (0.1107, 0.1329))
    assert np.allclose(f_in, 1.0)


def kroll_wada_toys(rng, n):
    """(m12, m23, weight) for points uniform over the eta's Dalitz region,
    weighted by the eta -> gamma mu+ mu- matrix element in invariant form,

        |F(s)|^2 / s^2 [s ((t - m^2)^2 + (u - m^2)^2) + 2 m^2 (m_eta^2 - s)^2]

    with s = m23, t = m12, u = m^2(gamma mu-) -- independent of the
    (s, cos theta*) form that kroll_wada_factor integrates."""
    mmu2, meta2 = MMU**2, META**2
    s = rng.uniform(4 * mmu2, meta2, n)
    t = rng.uniform(mmu2, (META - MMU)**2, n)
    lo, hi = kinematics.m123_limits(t, s)
    inside = (lo <= kinematics.M_ETA) & (kinematics.M_ETA <= hi)
    s, t = s[inside], t[inside]
    u = meta2 + 2 * mmu2 - s - t
    ff2 = (1 - s / FF_LAMBDA**2)**-2
    w = ff2 / s**2 * (s * ((t - mmu2)**2 + (u - mmu2)**2) + 2 * mmu2 * (meta2 - s)**2)
    return t, s, w


def test_kroll_wada_factor_matches_weighted_sampling():
    """The semi-analytic signal factor equals the Monte Carlo recipe: eta
    decays weighted by the matrix element, photon rescaled to mass M,
    counted in the bin."""
    rng = np.random.default_rng(4)
    t, s, w = kroll_wada_toys(rng, 4000000)
    M = np.linspace(440, 660, 12)
    for m12_range, m23_range in [((0.015, 0.030), (0.2879, 0.3100)),
                                 ((0.015, 0.030), (0.0886, 0.1107)),
                                 ((0.105, 0.120), (0.0443, 0.0664)),
                                 ((0.060, 0.075), (0.0886, 0.1107))]:
        f = kinematics.kroll_wada_factor(M, m12_range, m23_range, FF_LAMBDA)
        in_m23 = (m23_range[0] <= s) & (s < m23_range[1])
        for m, f_m in zip(M, f):
            M2 = (m / 1000)**2
            t_m = MMU**2 + (t - MMU**2) * (M2 - s) / (META**2 - s)
            in_bin = in_m23 & (M2 > s) & (m12_range[0] <= t_m) & (t_m < m12_range[1])
            mc = np.sum(w * in_bin) / np.sum(w)
            err = np.sqrt(np.sum((w * (in_bin - mc))**2)) / np.sum(w)
            assert abs(f_m - mc) < 4 * err + 1e-4


def test_kroll_wada_factor_normalisation():
    """Bins tiling the plane hold all the signal for M >= m_eta, and at
    m_eta the signal reaches exactly the bins the flat factor does."""
    M = np.array([kinematics.M_ETA, 600.0])
    m12_edges = np.linspace(0.0, 0.40, 21)
    m23_edges = np.linspace(0.0, 0.32, 17)
    total = sum(kinematics.kroll_wada_factor(M, m12_edges[i:i + 2], m23_edges[j:j + 2], FF_LAMBDA)
                for i in range(20) for j in range(16))
    assert np.allclose(total, 1.0, atol=1e-3)

    m12_edges = np.linspace(0.0, 0.21, 15)
    m23_edges = np.linspace(0.0, 0.31, 15)
    for i in range(14):
        for j in range(14):
            f_sig = kinematics.kroll_wada_factor([kinematics.M_ETA], m12_edges[i:i + 2],
                                                 m23_edges[j:j + 2], FF_LAMBDA)[0]
            f_flat = kinematics.phase_space_factor([kinematics.M_ETA], m12_edges[i:i + 2],
                                                   m23_edges[j:j + 2])[0]
            assert (f_sig > 0) == (f_flat > 0), (i, j, f_sig, f_flat)


def test_unit_weight_reproduces_the_unweighted_shapes():
    weight = fit_utils.make_weight(np.ones_like, XMIN, XMAX)
    x = np.linspace(XMIN, XMAX, 50)
    dcb_args = (547.9, 12.0, 1.5, 2.0, 1.5, 2.0)
    assert np.allclose(fit_utils.weighted_dcb_cdf(weight)(x, XMIN, XMAX, *dcb_args),
                       fit_utils.dcb_cdf(x, XMIN, XMAX, *dcb_args), atol=1e-4)
    model = fit_utils.BACKGROUND_MODELS['cheb2']
    wmodel = fit_utils.weighted_bkg_model(model, weight)
    assert np.allclose(wmodel.cdf(x, XMIN, XMAX, 0.3, -0.2),
                       model.cdf(x, XMIN, XMAX, 0.3, -0.2), atol=1e-4)


def test_weighted_background_is_zero_where_forbidden():
    m12_range, m23_range = (0.015, 0.030), (0.2879, 0.3100)
    weight = fit_utils.make_weight(
        lambda M: kinematics.phase_space_factor(M, m12_range, m23_range), XMIN, XMAX)
    wmodel = fit_utils.weighted_bkg_model(fit_utils.BACKGROUND_MODELS['cheb2'], weight)
    # f = 0 below ~541 MeV in this bin, so no background probability there,
    # even for a Chebyshev that is negative somewhere (floored, not flipped).
    for c0, c1 in [(0.0, 0.0), (1.5, 0.0)]:
        assert wmodel.cdf(np.array([535.0]), XMIN, XMAX, c0, c1)[0] == 0.0
        y = wmodel.pdf(np.linspace(XMIN, XMAX, 500), XMIN, XMAX, c0, c1)
        assert np.all(y >= 0)
        assert fit_utils._trapz(y, np.linspace(XMIN, XMAX, 500)) == pytest.approx(1, abs=1e-2)


def test_weighted_fit_recovers_background_toy():
    """Background drawn from f * Chebyshev in an edge bin: the weighted cost
    fits it back with the signal yield fixed at 0 (as for a bin with
    f(m_eta) = 0), including the zero-weight signal component."""
    rng = np.random.default_rng(3)
    m12_range, m23_range = (0.015, 0.030), (0.0886, 0.1107)
    weight = fit_utils.make_weight(
        lambda M: kinematics.phase_space_factor(M, m12_range, m23_range), XMIN, XMAX)
    xg, wg = weight
    shape = fit_utils.cheb_bkg_unnorm(xg, XMIN, XMAX, -0.5, 0.1) * wg
    cdf = np.concatenate([[0], np.cumsum(0.5 * (shape[1:] + shape[:-1]) * np.diff(xg))])
    vals = np.interp(rng.uniform(0, cdf[-1], 3000), cdf, xg)
    counts, edges = np.histogram(vals, bins=80, range=(XMIN, XMAX))

    cost = fit_utils.make_dcb_cost(counts.astype(float), edges, XMIN, XMAX,
                                   bkg='cheb2', weight=weight)
    init = dict(mean=547.9, sigma=12.0, alphaL=1.5, nL=2.0, alphaR=1.5, nR=2.0,
                n_sig=0.0, n_bkg=3000.0, c0=0.0, c1=0.0)
    m = fit_utils.run_fit(cost, init, dict(n_bkg=(0, None)),
                          fixed=['mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'n_sig'])
    assert m.valid
    assert m.values['n_bkg'] == pytest.approx(3000, rel=0.05)
    # Only ~40% of the window is reachable here, so c0/c1 are individually
    # degenerate; compare the fitted shape to the true one instead.
    wmodel = fit_utils.weighted_bkg_model(fit_utils.BACKGROUND_MODELS['cheb2'], weight)
    fit_cdf = wmodel.cdf(edges, XMIN, XMAX, m.values['c0'], m.values['c1'])
    assert np.max(np.abs(fit_cdf - np.interp(edges, xg, cdf / cdf[-1]))) < 0.03
