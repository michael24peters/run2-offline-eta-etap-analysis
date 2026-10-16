"""
Unit tests for the per-Dalitz-bin fit library, src/fit/fit_dalitz.py, on toy
candidates: a Gaussian eta peak and a flat background in mass. Two toys:

    physical_toy     density per unit Dalitz area the same at every mass,
                     zero outside the mass's Dalitz region: in a Dalitz bin,
                     the mass density times the bin's phase-space factor, the
                     model of the phase-space fit
    independent_toy  Dalitz position independent of mass (signal uniform over
                     the eta's Dalitz region, background uniform over the
                     grid): the model of the plain DCB + Chebyshev fit

A 3 x 3 grid over the same m12/m23 ranges as driver_dalitz_fit.py keeps the
fits fast. Fixed RNG seeds.
"""

import numpy as np
import pytest

import kinematics
from fit_dalitz import bin_factor_grid, fit_dalitz, hole_neighbors

XMIN, XMAX = 420.0, 680.0
M12_EDGES = np.linspace(0.0, 0.21, 4)   # [GeV^2]
M23_EDGES = np.linspace(0.0, 0.31, 4)   # [GeV^2]
N_SIG, N_BKG = 4000, 8000
MIN_CANDIDATES = 10
# Global-fit prior on (alphaL, nL, alphaR, nR): Gaussian-like tails, as the
# driver's MC tail prior provides for data
TAIL_PRIOR = (np.array([3.0, 5.0, 3.0, 5.0]), np.diag([0.1**2] * 4))
BOX = (XMAX / 1000.0)**2   # side of the (m12, m23) square the toy points are drawn in [GeV^2]


def draw_signal_masses(rng, size):
    """Signal masses [MeV]: Gaussian at m_eta, 12 MeV wide."""
    return rng.normal(kinematics.M_ETA, 12.0, size)


def draw_bkg_masses(rng, size):
    """Background masses [MeV]: flat over the window."""
    return rng.uniform(XMIN, XMAX, size)


def draw_eta_mass(rng, size):
    """m_eta [MeV] for every candidate."""
    return np.full(size, kinematics.M_ETA)


def phase_space_candidates(rng, draw_masses, n):
    """n candidates (m12s, m23s, masses): a mass from draw_masses and (m12, m23)
    uniform in [0, BOX] x [0, BOX] [GeV^2], kept where kinematics.m123_limits
    contains the mass. In a Dalitz bin, their mass density is draw_masses's
    density times the bin's phase-space factor."""
    kept = []
    n_kept = 0
    while n_kept < n:
        masses = draw_masses(rng, n)
        m12s = rng.uniform(0, BOX, n)
        m23s = rng.uniform(0, BOX, n)
        lo, hi = kinematics.m123_limits(m12s, m23s)
        inside = (lo <= masses) & (masses <= hi)
        kept.append((m12s[inside], m23s[inside], masses[inside]))
        n_kept += int(inside.sum())
    return tuple(np.concatenate(column)[:n] for column in zip(*kept, strict=True))


@pytest.fixture(scope='module')
def physical_toy():
    """m12s, m23s, masses, is_signal: N_SIG signal then N_BKG background
    candidates from phase_space_candidates."""
    rng = np.random.default_rng(7)
    signal = phase_space_candidates(rng, draw_signal_masses, N_SIG)
    background = phase_space_candidates(rng, draw_bkg_masses, N_BKG)
    m12s, m23s, masses = (np.concatenate(pair) for pair in zip(signal, background, strict=True))
    return m12s, m23s, masses, np.arange(masses.size) < N_SIG


@pytest.fixture(scope='module')
def independent_toy():
    """m12s, m23s, masses, is_signal: signal uniform over the eta's Dalitz
    region and background uniform over the grid, whatever their mass."""
    rng = np.random.default_rng(8)
    signal_m12s, signal_m23s, _ = phase_space_candidates(rng, draw_eta_mass, N_SIG)
    m12s = np.concatenate([signal_m12s, rng.uniform(M12_EDGES[0], M12_EDGES[-1], N_BKG)])
    m23s = np.concatenate([signal_m23s, rng.uniform(M23_EDGES[0], M23_EDGES[-1], N_BKG)])
    masses = np.concatenate([draw_signal_masses(rng, N_SIG), draw_bkg_masses(rng, N_BKG)])
    return m12s, m23s, masses, np.arange(masses.size) < N_SIG


@pytest.fixture(scope='module')
def baseline(independent_toy):
    m12s, m23s, masses, _ = independent_toy
    result, _ = fit_dalitz(m12s, m23s, masses, M12_EDGES, M23_EDGES, xmin=XMIN, xmax=XMAX,
                           tail_prior=TAIL_PRIOR, min_candidates=MIN_CANDIDATES)
    return result


@pytest.fixture(scope='module')
def phase_space(physical_toy):
    m12s, m23s, masses, _ = physical_toy
    result, _ = fit_dalitz(m12s, m23s, masses, M12_EDGES, M23_EDGES, xmin=XMIN, xmax=XMAX,
                           tail_prior=TAIL_PRIOR, min_candidates=MIN_CANDIDATES,
                           signal_factor=kinematics.phase_space_factor,
                           bkg_factor=kinematics.phase_space_factor)
    return result


def signal_in_bin(toy, b):
    """Number of signal toys in bin b and the mass window."""
    m12s, m23s, masses, is_signal = toy
    in_bin = ((m12s >= b['m12_range'][0]) & (m12s < b['m12_range'][1])
              & (m23s >= b['m23_range'][0]) & (m23s < b['m23_range'][1]))
    return int((is_signal & in_bin & (masses >= XMIN) & (masses < XMAX)).sum())


def test_bins_at_or_below_min_candidates_are_not_fit(independent_toy, baseline):
    bins = baseline['dalitz']['bins']
    assert len(bins) == 9
    for b in bins:
        assert b['fitted'] == (b['n_cand'] > MIN_CANDIDATES)
    outcome = baseline['dalitz']['outcome']
    assert outcome['skipped'] == sum(not b['fitted'] for b in bins)
    assert outcome['skipped'] + outcome['invalid'] + outcome['valid'] == len(bins)

    m12s, m23s, masses, _ = independent_toy
    result, _ = fit_dalitz(m12s, m23s, masses, M12_EDGES, M23_EDGES, xmin=XMIN, xmax=XMAX,
                           tail_prior=TAIL_PRIOR, min_candidates=N_SIG + N_BKG)
    assert result['dalitz']['outcome'] == {'skipped': 9, 'invalid': 0, 'valid': 0}


def test_counts_add_up_to_the_window_candidates_in_the_grid(physical_toy, phase_space):
    m12s, m23s, masses, _ = physical_toy
    in_grid = ((m12s >= M12_EDGES[0]) & (m12s < M12_EDGES[-1])
               & (m23s >= M23_EDGES[0]) & (m23s < M23_EDGES[-1]))
    in_window = (masses >= XMIN) & (masses < XMAX)
    bins = phase_space['dalitz']['bins']
    closure = phase_space['dalitz']['yield_closure']
    assert sum(sum(b['counts']) for b in bins) == int((in_grid & in_window).sum())
    assert closure['window_candidates_in_bins'] == int((in_grid & in_window).sum())
    assert closure['candidates_in_grid'] == int(in_grid.sum())
    for b in bins:
        assert b['n_cand'] == sum(b['counts'])


def test_baseline_bins_carry_no_factor(baseline):
    assert baseline['dalitz']['signal_factor'] is None
    assert baseline['dalitz']['bkg_factor'] is None
    for b in baseline['dalitz']['bins']:
        assert 'f_sig' not in b and 'f_bkg' not in b
        assert not b.get('n_sig_fixed_zero', False)


def test_bins_the_eta_cannot_reach_have_n_sig_fixed_at_zero(phase_space):
    assert phase_space['dalitz']['signal_factor'] == 'phase_space_factor'
    fixed_zero = []
    for b in phase_space['dalitz']['bins']:
        if not b['fitted']:
            continue
        factor_grid = bin_factor_grid(kinematics.phase_space_factor, b['m12_range'],
                                      b['m23_range'], XMIN, XMAX)
        unreachable = np.interp(kinematics.M_ETA, *factor_grid) == 0.0
        assert b['n_sig_fixed_zero'] == unreachable
        if unreachable:
            fixed_zero.append(b)
            assert b['valid'] and b['n_sig'] == 0.0
    assert fixed_zero, 'the toy grid has no bin the eta cannot reach'


@pytest.mark.parametrize('fit_name, toy_name', [('baseline', 'independent_toy'),
                                                ('phase_space', 'physical_toy')])
def test_summed_signal_matches_the_injected_signal(fit_name, toy_name, request):
    """Summed n_sig over bins where it floats is within 3 combined standard
    deviations of the signal toys those bins hold."""
    result = request.getfixturevalue(fit_name)
    toy = request.getfixturevalue(toy_name)
    floating = [b for b in result['dalitz']['bins']
                if b['fitted'] and not b['n_sig_fixed_zero']]
    assert all(b['valid'] for b in floating)
    summed = sum(b['n_sig'] for b in floating)
    error = np.sqrt(sum(b['n_sig_err']**2 for b in floating))
    injected = sum(signal_in_bin(toy, b) for b in floating)
    assert abs(summed - injected) < 3 * error, (summed, injected, error)


def test_hole_neighbors_lists_fitted_neighbors_of_invalid_bins():
    def entry(i, j, fitted, valid, n_cand):
        return {'i': i, 'j': j, 'fitted': fitted, 'valid': valid, 'n_cand': n_cand}
    result = {'dalitz': {'bins': [
        entry(0, 0, True, True, 50.0), entry(0, 1, False, False, 3.0),
        entry(1, 0, True, False, 40.0), entry(1, 1, True, True, 60.0),
        entry(2, 0, True, True, 70.0),
    ]}}
    assert hole_neighbors(result) == [(1, 0, 40.0, {(-1, 0): 50.0, (1, 0): 70.0, (0, 1): 60.0})]
