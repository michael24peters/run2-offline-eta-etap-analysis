"""
fit_dalitz.py

Per-Dalitz-bin mass fits of eta -> mu+ mu- gamma. Takes one candidate per event
as (m12s, m23s, masses) arrays and fits them.

    global fit_dcb of every candidate in the mass window
      -> one fit per Dalitz bin, seeded and constrained by the global fit
      -> result dict

In each Dalitz bin (one m12 x m23 box) the gamma mu+ mu- mass M is fit with

    n_sig * signal(M) + n_bkg * background(M)

    signal      DCB, times signal_factor(M) of the bin if given
    background  Chebyshev, times bkg_factor(M) of the bin if given

each divided by its integral over the mass window (fit_utils, "Weighting by a
Dalitz-bin factor f(M)"). signal_factor / bkg_factor are functions
f(M, m12_range, m23_range), e.g. kinematics.phase_space_factor or
kinematics.dalitz_factor; None (the default) fits the plain shapes.

Per-bin fit:
    mean, sigma   float, with a NormalConstraint from the global fit's
                  covariance, stdevs times signal_prior_factor
    DCB tails     fixed at the global fit's values
    n_sig, n_bkg  float, from half the bin's candidates each
    c0, c1        float; from the global fit's values without bkg_factor,
                  from fit_utils.BKG_INIT (flat) with it
    With signal_factor 0 at m_eta (a bin the eta cannot reach), n_sig is
    fixed to 0 and mean, sigma to the global fit's values.

signal_only=True (truth-matched signal MC) fixes n_bkg to 0 and c0, c1 to
fit_utils.BKG_INIT in the global fit and in every bin.
"""

import os
import sys

import numpy as np
from iminuit.cost import NormalConstraint

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics
from fit_dcb import TAIL_PARAMS, fit_dcb

SIGNAL_PARAMS = ('mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR')
PRIOR_PARAMS = ('mean', 'sigma')


def bin_mask(m12s, m23s, m12_range, m23_range):
    """True for candidates in the Dalitz bin [m12_lo, m12_hi) x [m23_lo, m23_hi)."""
    return ((m12s >= m12_range[0]) & (m12s < m12_range[1])
            & (m23s >= m23_range[0]) & (m23s < m23_range[1]))


def bin_factor_grid(factor, m12_range, m23_range, xmin, xmax):
    """factor_grid = (masses, factor values) of one Dalitz bin: `factor` with
    the bin's m12 and m23 ranges, sampled by fit_utils.sample_factor."""
    def factor_of_mass(M):
        return factor(M, m12_range, m23_range)
    return fit_utils.sample_factor(factor_of_mass, xmin, xmax)


def fit_bin(counts, mass_edges, xmin, xmax, seed, prior_cov, bkg_seed,
            signal_factor_grid=None, bkg_factor_grid=None, signal_only=False):
    """One Dalitz bin's mass fit.

    counts             (nbins,) candidates of the bin in each mass bin
    mass_edges         (nbins + 1,) mass-bin edges [MeV] spanning [xmin, xmax]
    seed               {name: value} of SIGNAL_PARAMS from the global fit
    prior_cov          (2, 2) covariance of the mean/sigma NormalConstraint
    bkg_seed           {c0, c1} start values without bkg_factor_grid
    signal_factor_grid, bkg_factor_grid : (masses, factor) or None
    signal_only        n_bkg fixed to 0, c0 and c1 to fit_utils.BKG_INIT

    Returns (fit, no_signal): the Minuit object, and whether n_sig was fixed
    to 0 (signal factor 0 at m_eta).
    """
    n_cand = counts.sum()
    no_signal = (signal_factor_grid is not None
                 and bool(np.interp(kinematics.M_ETA, *signal_factor_grid) == 0.0))

    # Extended binned likelihood of signal + background, plus the mean/sigma prior
    cost = (fit_utils.make_dcb_cost(counts, mass_edges, xmin, xmax,
                                    signal_factor_grid=signal_factor_grid,
                                    bkg_factor_grid=bkg_factor_grid)
            + NormalConstraint(PRIOR_PARAMS, (seed['mean'], seed['sigma']), error=prior_cov))

    # Global init shape; half signal and half background default; Chebyshev from
    # bkg_seed, or flat when its variable spans the bin's reachable masses
    init = dict(seed,
                n_sig=0.0 if no_signal else 0.5 * n_cand,
                n_bkg=n_cand if no_signal else 0.5 * n_cand,
                **(bkg_seed if bkg_factor_grid is None else fit_utils.BKG_INIT))
    # Signal only: every candidate is signal
    if signal_only:
        init.update(n_sig=n_cand, n_bkg=0.0, **fit_utils.BKG_INIT)
    limits = dict(sigma=(0.0, None), alphaL=(0.0, None), alphaR=(0.0, None),
                  nL=(0.0, None), nR=(0.0, None), n_sig=(0.0, None), n_bkg=(0.0, None))
    # Tails at the global fit's values; with no signal, its yield and shape too
    fixed = list(TAIL_PARAMS)
    if no_signal:
        fixed += ['n_sig', 'mean', 'sigma']
    if signal_only:
        fixed += ['n_bkg', *fit_utils.BKG_PARAMS]
    return fit_utils.run_fit(cost, init, limits, fixed=fixed), no_signal


def bin_prediction(fit, counts, mass_edges, xmin, xmax,
                   signal_factor_grid=None, bkg_factor_grid=None):
    """Expected signal and background counts per mass bin, Neyman chi2 and
    data-based pulls of one bin's fit.

    Returns {bin_sig, bin_bkg, pulls, chi2, ndof}. chi2 = sum (d - f)^2 / d
    and pulls = (d - f) / sqrt(d) over mass bins with d > 0.
    """
    values = fit.values
    signal_cdf = fit_utils.dcb_cdf
    if signal_factor_grid is not None:
        signal_cdf = fit_utils.weighted_dcb_cdf(signal_factor_grid)
    bkg_cdf = fit_utils.cheb_cdf
    if bkg_factor_grid is not None:
        bkg_cdf = fit_utils.weighted_cheb_cdf(bkg_factor_grid)

    # Expected counts per mass bin: yield times CDF differences between edges
    bin_sig = np.diff(signal_cdf(mass_edges, xmin, xmax,
                                 *(values[name] for name in SIGNAL_PARAMS))) * values['n_sig']
    bin_bkg = np.diff(bkg_cdf(mass_edges, xmin, xmax,
                              *(values[name] for name in fit_utils.BKG_PARAMS))) * values['n_bkg']
    bin_fit = bin_sig + bin_bkg

    # Neyman chi2 and pulls, empty mass bins skipped
    chi2 = fit_utils.neyman_chi2(counts, bin_fit)
    ndof = int((counts > 0).sum()) - fit.nfit
    pulls = fit_utils.data_pulls(counts, bin_fit)
    return {'bin_sig': bin_sig, 'bin_bkg': bin_bkg, 'pulls': pulls, 'chi2': chi2, 'ndof': ndof}


def bkg_goes_negative(fit, xmin, xmax, bkg_factor_grid=None):
    """True if the fitted Chebyshev, before the BKG_FLOOR floor, is below 0 at
    a mass the background can reach: anywhere in [xmin, xmax] without
    bkg_factor_grid, where the factor is > 0 with it."""
    coefficients = [fit.values[name] for name in fit_utils.BKG_PARAMS]
    if bkg_factor_grid is None:
        masses = np.linspace(xmin, xmax, 2000)
        return bool(np.any(fit_utils.cheb_bkg_unnorm(masses, xmin, xmax, *coefficients) < 0))
    masses, factor = bkg_factor_grid
    cheb_range = fit_utils.reachable_range(bkg_factor_grid)
    cheb = fit_utils.cheb_bkg_unnorm(masses[factor > 0], *cheb_range, *coefficients)
    return bool(np.any(cheb < 0))


def fit_dalitz(m12s, m23s, masses, m12_edges, m23_edges,
               xmin=500.0, xmax=600.0, nbins=100,
               mean_init=kinematics.M_ETA, sigma_init=10.0,
               tail_prior=None, signal_prior_factor=5.0, min_candidates=10,
               signal_factor=None, bkg_factor=None, signal_only=False, limits=None):
    """
    Global DCB + Chebyshev fit of every candidate, then one fit per Dalitz bin.

    Arguments:
    ----------
    m12s, m23s : (N,) m^2(gamma mu+), m^2(mu+ mu-) of each candidate [GeV^2]
    masses     : (N,) gamma mu+ mu- mass of each candidate [MeV]
    m12_edges, m23_edges : Dalitz-bin edges on each axis [GeV^2]
    xmin, xmax, nbins : mass window [MeV] and number of mass bins
    mean_init, sigma_init : global-fit start values [MeV]
    tail_prior : (values, cov) of the global fit's TAIL_PARAMS prior, or None;
             see fit_dcb()
    signal_prior_factor : the per-bin mean/sigma prior is the global fit's
             covariance with stdevs times this
    min_candidates : bins with this many candidates in the window or fewer
             are not fit
    signal_factor, bkg_factor : f(M, m12_range, m23_range) multiplying the
             signal / background shape in each bin, or None
    signal_only : fix n_bkg to 0 and c0, c1 to fit_utils.BKG_INIT in the
             global and per-bin fits (truth-matched signal MC)
    limits : the global fit's limits; see fit_dcb()

    Returns:
    --------
    result : fit_dcb()'s result dict of the global fit, plus result['dalitz']
             with one entry per Dalitz bin ('bins'), the outcome counts, the
             yield closure and the mean, median and max chi2/ndof of the
             valid bins.
    global_fit : the global fit's Minuit object.
    """
    mass_edges = np.linspace(xmin, xmax, nbins + 1)
    factor_masses = np.linspace(xmin, xmax, int(round(xmax - xmin)) + 1)   # 1 MeV apart

    # Global fit; its signal shape seeds and constrains every bin
    global_fixed = {'n_bkg': 0.0, **fit_utils.BKG_INIT} if signal_only else None
    result, global_fit = fit_dcb(masses, xmin=xmin, xmax=xmax, nbins=nbins,
                                 mean_init=mean_init, sigma_init=sigma_init,
                                 tail_prior=tail_prior, fixed_params=global_fixed,
                                 limits=limits)
    global_params = result['fit']['parameters']
    seed = {name: global_params[name]['value'] for name in SIGNAL_PARAMS}
    bkg_seed = {name: global_params[name]['value'] for name in fit_utils.BKG_PARAMS}
    _, prior_cov = fit_utils.build_prior(global_fit, PRIOR_PARAMS, factor=signal_prior_factor)

    bins = []
    for i in range(len(m12_edges) - 1):
        for j in range(len(m23_edges) - 1):
            m12_range = (float(m12_edges[i]), float(m12_edges[i + 1]))
            m23_range = (float(m23_edges[j]), float(m23_edges[j + 1]))

            # Histogram the masses of the candidates in the bin
            in_bin = bin_mask(m12s, m23s, m12_range, m23_range)
            counts = np.histogram(masses[in_bin], bins=mass_edges)[0].astype(float)
            n_cand = counts.sum()
            entry = {
                'i': i, 'j': j,
                'm12_range': list(m12_range),
                'm23_range': list(m23_range),
                'n_cand': float(n_cand),          # candidates in the bin and the mass window
                'n_events': int(in_bin.sum()),    # candidates in the bin, any mass
                'counts': counts.astype(int).tolist(),
                'fitted': False,
                'valid': False,
            }
            bins.append(entry)
            if n_cand <= min_candidates:
                continue

            # Each component's factor of this bin on the mass grid
            signal_factor_grid = bkg_factor_grid = None
            if signal_factor is not None:
                signal_factor_grid = bin_factor_grid(signal_factor, m12_range, m23_range,
                                                     xmin, xmax)
            if bkg_factor is not None:
                bkg_factor_grid = bin_factor_grid(bkg_factor, m12_range, m23_range, xmin, xmax)

            fit, no_signal = fit_bin(counts, mass_edges, xmin, xmax, seed, prior_cov, bkg_seed,
                                     signal_factor_grid, bkg_factor_grid, signal_only)
            entry.update({'fitted': True, 'valid': bool(fit.valid), 'n_sig_fixed_zero': no_signal})
            if not fit.valid:
                continue

            # Expected counts per mass bin, chi2 and pulls of the fit
            prediction = bin_prediction(fit, counts, mass_edges, xmin, xmax,
                                        signal_factor_grid, bkg_factor_grid)
            chi2, ndof = prediction['chi2'], prediction['ndof']
            values, errors = fit.values, fit.errors
            entry.update({
                'n_sig': round(float(values['n_sig']), 2),
                'n_sig_err': round(float(errors['n_sig']), 2),
                'n_bkg': round(float(values['n_bkg']), 2),
                'n_bkg_err': round(float(errors['n_bkg']), 2),
                'mean': round(float(values['mean']), 4),
                'mean_err': round(float(errors['mean']), 4),
                'sigma': round(float(values['sigma']), 4),
                'sigma_err': round(float(errors['sigma']), 4),
                **{name: round(float(values[name]), 6) for name in fit_utils.BKG_PARAMS},
                **{f'{name}_err': round(float(errors[name]), 6) for name in fit_utils.BKG_PARAMS},
                'chi2': round(chi2, 4),
                'ndof': ndof,
                'chi2_per_ndof': round(chi2 / ndof, 6) if ndof > 0 else None,
                'bkg_goes_negative': bkg_goes_negative(fit, xmin, xmax, bkg_factor_grid),
                'bin_sig': np.round(prediction['bin_sig'], 4).tolist(),
                'bin_bkg': np.round(prediction['bin_bkg'], 4).tolist(),
                'pulls': np.round(prediction['pulls'], 4).tolist(),
            })
            # Factors at factor_masses, for plotting
            if signal_factor_grid is not None:
                entry['f_sig'] = np.round(np.interp(factor_masses, *signal_factor_grid), 6).tolist()
            if bkg_factor_grid is not None:
                entry['f_bkg'] = np.round(np.interp(factor_masses, *bkg_factor_grid), 6).tolist()

    # Outcome and yield closure: summed per-bin yields vs the global fit
    fitted = [b for b in bins if b['fitted']]
    valid = [b for b in bins if b['valid']]
    in_grid = bin_mask(m12s, m23s, (m12_edges[0], m12_edges[-1]), (m23_edges[0], m23_edges[-1]))
    in_window = (masses >= xmin) & (masses < xmax)
    chi2s = [b['chi2_per_ndof'] for b in valid if b['chi2_per_ndof'] is not None]

    result['dalitz'] = {
        'signal_factor': None if signal_factor is None else signal_factor.__name__,
        'bkg_factor': None if bkg_factor is None else bkg_factor.__name__,
        'ff_lambda_gev': (kinematics.FF_LAMBDA if signal_factor is kinematics.dalitz_factor
                          else None),
        'signal_only': signal_only,
        'm12_edges': [float(edge) for edge in m12_edges],
        'm23_edges': [float(edge) for edge in m23_edges],
        'mass_window': {'xmin': xmin, 'xmax': xmax, 'nbins': nbins},
        'factor_masses': factor_masses.tolist(),
        'min_candidates': min_candidates,
        'prior': {'params': list(PRIOR_PARAMS), 'factor': signal_prior_factor},
        'fixed_in_per_bin_fit': list(TAIL_PARAMS),
        'n_sig_fixed_zero_rule': 'signal factor at m_eta == 0',
        'chi2_convention': 'neyman/data-based: sum (d - f)^2 / d',
        'pull_convention': 'data-based: (d - f) / sqrt(d)',
        'outcome': {'skipped': len(bins) - len(fitted), 'invalid': len(fitted) - len(valid),
                    'valid': len(valid)},
        'yield_closure': {
            'n_sig_global': global_params['n_sig']['value'],
            'n_sig_summed': float(sum(b['n_sig'] for b in valid)),
            'n_bkg_global': global_params['n_bkg']['value'],
            'n_bkg_summed': float(sum(b['n_bkg'] for b in valid)),
            'window_candidates_in_bins': int((in_grid & in_window).sum()),
            'window_candidates': int(in_window.sum()),
            'candidates_in_grid': int(in_grid.sum()),
            'candidates': int(len(masses)),
        },
        'chi2_per_ndof_over_bins': None if not chi2s else {
            'mean': round(float(np.mean(chi2s)), 6),
            'median': round(float(np.median(chi2s)), 6),
            'max': round(float(np.max(chi2s)), 6),
        },
        'bins': bins,
    }
    return result, global_fit


def hole_neighbors(result):
    """For each fitted bin whose fit is invalid: (i, j, n_cand, neighbors),
    with neighbors {(di, dj): n_cand} of the fitted bins directly above,
    below, left and right of it."""
    bins = {(b['i'], b['j']): b for b in result['dalitz']['bins']}
    holes = []
    for (i, j), b in bins.items():
        if not b['fitted'] or b['valid']:
            continue
        neighbors = {}
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = bins.get((i + di, j + dj))
            if neighbor is not None and neighbor['fitted']:
                neighbors[(di, dj)] = neighbor['n_cand']
        holes.append((i, j, b['n_cand'], neighbors))
    return holes
