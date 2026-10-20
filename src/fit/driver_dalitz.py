"""
driver_dalitz.py

Per-Dalitz-bin mass fits of eta -> mu+ mu- gamma from a ROOT file:

    read ROOT -> drop malformed events -> TOS (+ pnn_mu) -> best candidate
              (+ truth match) -> daughter four-vectors -> m12s, m23s, masses
              -> selection.json, raw Dalitz histograms
              -> N_PER_AXIS x N_PER_AXIS grid over M12_RANGE x M23_RANGE
              -> fit_dalitz (global fit, then one fit per bin)
              -> dalitz_fit_<factor>.json
              -> plot_dalitz_bins.plot_dalitz_fit (global fit, yield maps,
                 per-bin pages)

    m12s    m^2(gamma mu+)        [GeV^2]
    m23s    m^2(mu+ mu-) = q^2    [GeV^2]
    masses  m(gamma mu+ mu-)      [MeV]

--mass picks the mass and the daughter momenta together (MASS_SOURCES):

    rebuilt    mass rebuilt from prt_px/py/pz/e (default)
    tag_m      tag_m; m12, m23 from prt_px/py/pz/e
    tag_dtf_m  tag_dtf_m; m12, m23 from the DTF daughters prt_dtf_px/py/pz/e

--factor picks the per-bin factor on the signal and background shapes
(FACTORS):

    none      no factor on either shape (default)
    flat      kinematics.phase_space_factor on the signal and the background
    weighted  kinematics.dalitz_factor on the signal,
              kinematics.phase_space_factor on the background

--truth-match (signal MC only) keeps the events whose best candidate has every
daughter truth-matched to eta -> mu+ mu- gamma
(bkg_utils.signal_candidate_mask), and fits them with no tail prior and the
background fixed to 0 (fit_dalitz signal_only). Its dalitz_fit_<factor>.json
is the --tail-prior of the data and all-candidate MC fits of the same --mass;
the global fit, whose tails make the prior, is the same for every factor.

Without --truth-match, --tail-prior is required: the global fit carries the
tail prior of that fit, its stdevs times TAIL_PRIOR_FACTOR.

The median and RMS of (mass - mass rebuilt from the same daughters) and the
number of candidates outside their own kinematic limits
(kinematics.m123_limits) are printed. No mass-window or Dalitz-grid cut is
applied to the selection.

select_candidates() is the selection on its own, for fit_dalitz.ipynb and
plot_phase_space.py.

Usage:
    python src/fit/driver_dalitz.py INFILE --outdir OUTDIR
        (--truth-match | --tail-prior DALITZ_FIT_JSON)
        [--mass rebuilt|tag_m|tag_dtf_m] [--factor none|flat|weighted]
        [--tos-levels L0 HLT1 HLT2] [--pnn-mu-min X] [--max-events N]
        [--horizontal]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import bkg_utils
import cut_utils
import fit_utils
import kinematics
from fit_dalitz import fit_dalitz, hole_neighbors
from fit_dcb import TAIL_PARAMS
from plot_dalitz import HISTOGRAMS, plot_dalitz_hist
from plot_dalitz_bins import plot_dalitz_fit

# --- Selection ---------------------------------------------------------------

TAG_REF_BRANCH = 'tag_m'          # tag-level reference branch for cut_utils
METRIC_BRANCH = 'tag_dtf_chi2'    # ranks candidates within an event

# Mass and daughter four-momentum branches of each --mass choice; 'mass' None
# means the mass rebuilt from 'momenta'.
PRT_MOMENTA = ['prt_px', 'prt_py', 'prt_pz', 'prt_e']
DTF_MOMENTA = ['prt_dtf_px', 'prt_dtf_py', 'prt_dtf_pz', 'prt_dtf_e']
MASS_SOURCES = {
    'rebuilt':   {'mass': None,        'momenta': PRT_MOMENTA},
    'tag_m':     {'mass': 'tag_m',     'momenta': PRT_MOMENTA},
    'tag_dtf_m': {'mass': 'tag_dtf_m', 'momenta': DTF_MOMENTA},
}

# Daughter name -> PID. Every candidate is built from exactly one of each.
DAUGHTERS = {'gamma': 22, 'mu+': -13, 'mu-': 13}
MOTHER_PID = 221                                      # eta, for --truth-match
TRUTH_BRANCHES = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']   # read with --truth-match

# TOS lines per trigger level; --tos-levels picks the levels to require.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}

GEV = 1000.0   # MeV per GeV

# --- Fit ---------------------------------------------------------------------

XMIN, XMAX, MASS_NBINS = 420.0, 680.0, 80      # mass window [MeV] and its histogram bins
MEAN_INIT, SIGMA_INIT = kinematics.M_ETA, 10.0  # global-fit seeds [MeV]

M12_RANGE = (0.0, 0.21)   # m^2(gamma mu+) range of the grid [GeV^2]
M23_RANGE = (0.0, 0.31)   # m^2(mu+ mu-) range of the grid [GeV^2]
N_PER_AXIS = 14           # bins per axis

MIN_CANDIDATES = 10         # bins with this many candidates or fewer in the window are not fit
SIGNAL_PRIOR_FACTOR = 5.0   # per-bin mean/sigma prior: global-fit stdevs times this
TAIL_PRIOR_FACTOR = 5.0     # global-fit tail prior: MC tail-fit stdevs times this

# --factor -> (signal factor, background factor)
FACTORS = {
    'none':     (None, None),
    'flat':     (kinematics.phase_space_factor, kinematics.phase_space_factor),
    'weighted': (kinematics.dalitz_factor, kinematics.phase_space_factor),
}


def select_candidates(infile, mass='rebuilt', tos_levels=tuple(TOS_LINES), pnn_mu_min=None,
                      truth_match=False, max_events=None):
    """
    Read infile, select, and keep each event's best candidate.

    Returns (m12s, m23s, masses, selection): the Dalitz variables [GeV^2] and
    gamma mu+ mu- mass [MeV] of each event's best candidate, and the
    selection settings and report (the contents of selection.json).
    """
    mass_branch = MASS_SOURCES[mass]['mass']
    momenta = MASS_SOURCES[mass]['momenta']
    tos_lines = {level: TOS_LINES[level] for level in tos_levels}
    branches = ([TAG_REF_BRANCH, METRIC_BRANCH, 'prt_idx_mom', 'prt_pid', 'prt_pnn_mu']
                + ([mass_branch] if mass_branch not in (None, TAG_REF_BRANCH) else [])
                + momenta
                + [line for lines in tos_lines.values() for line in lines]
                + (TRUTH_BRANCHES if truth_match else []))

    # --- Read ---

    tree = cut_utils.read_branches(infile, branches, entry_stop=max_events)
    tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=TAG_REF_BRANCH)
    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}, "
          f"dropped {drop_report['read'] - drop_report['kept']}")
    for reason in ('dropped_no_candidates', 'dropped_idx_out_of_range',
                   'dropped_daughter_length_mismatch', 'dropped_partial_candidate_branch'):
        if drop_report[reason]:
            print(f'[INFO]   {reason}: {drop_report[reason]}')
    n_multi = int(ak.sum(ak.num(tree[TAG_REF_BRANCH]) > 1))
    print(f'[INFO] {n_multi} / {len(tree)} events have >1 candidate')

    # Every candidate must have exactly one daughter of each PID.
    n_cand_total = int(ak.sum(ak.num(tree[TAG_REF_BRANCH])))
    for name, pid in DAUGHTERS.items():
        n_of_pid = cut_utils.reduce(tree, tree['prt_pid'] == pid, 'prt_idx_mom',
                                    how='sum', target_ref=TAG_REF_BRANCH)
        print(f'[INFO] {name:5s} (pid {pid:>3}): {float(ak.mean(ak.flatten(n_of_pid))):.4f} '
              f'daughters per candidate')
        assert ak.all(n_of_pid == 1), f'some candidates do not have exactly one {name}'
    print(f'[INFO] {n_cand_total} candidates, each exactly (gamma, mu+, mu-)')

    # --- Cuts ---

    keep = ak.ones_like(tree[TAG_REF_BRANCH], dtype=bool)
    cutflow = [('all candidates', n_cand_total)]
    triggers = None

    # TOS on at least one line at every requested level.
    if tos_lines:
        tos, tos_report = cut_utils.tos_mask(tree, tos_lines, tag_ref=TAG_REF_BRANCH)
        keep = keep & tos
        triggers = {level: report['lines'] for level, report in tos_report.items()}
        for level, report in tos_report.items():
            cutflow.append((f'TOS up to {level}', report['tos_up_to_level']))
            for line, n in report['lines'].items():
                if n == 0:
                    print(f'[WARN] {level} TOS line {line} passes no candidate')

    # ProbNNmu on every muon daughter; a non-muon daughter passes.
    if pnn_mu_min is not None:
        muon_ok = (tree['prt_pnn_mu'] > pnn_mu_min) | (abs(tree['prt_pid']) != 13)
        keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                       target_ref=TAG_REF_BRANCH)
        cutflow.append((f'pnn_mu > {pnn_mu_min}', int(ak.sum(keep))))

    # --- Best candidate and its daughters ---

    # One candidate per event: smallest METRIC_BRANCH among those kept.
    best = cut_utils.best_candidate_mask(tree, METRIC_BRANCH, keep)
    cutflow.append(('best candidates', int(ak.sum(best))))

    # Truth match: keep the events whose best candidate has every daughter
    # matched to eta -> gamma mu+ mu-.
    if truth_match:
        best = best & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, list(DAUGHTERS.values()),
                                                      target_ref=TAG_REF_BRANCH)
        cutflow.append(('best candidate truth-matched', int(ak.sum(best))))
    n_best = int(ak.sum(best))
    assert n_best > 0, 'no event survives the selection'

    # Daughter-level mask: True for the daughters of each best candidate.
    best_daughter = cut_utils.follow(tree, best, 'prt_idx_mom')

    # (px, py, pz, e) [GeV] of each daughter of the best candidates, event-aligned.
    p4 = {}
    for name, pid in DAUGHTERS.items():
        match = best_daughter & (tree['prt_pid'] == pid)
        p4[name] = [ak.to_numpy(ak.flatten(tree[field][match])) / GEV for field in momenta]
        assert len(p4[name][0]) == n_best, f'{name}: not one per best candidate'

    # Dalitz variables, and the gamma mu+ mu- mass, from the same four-vectors.
    m12s = kinematics.invariant_mass2(*p4['gamma'], *p4['mu+'])
    m23s = kinematics.invariant_mass2(*p4['mu+'], *p4['mu-'])
    gamma_mup = [a + b for a, b in zip(p4['gamma'], p4['mu+'], strict=True)]
    rebuilt = np.sqrt(np.clip(kinematics.invariant_mass2(*gamma_mup, *p4['mu-']), 0.0, None)) * GEV

    # The fitted mass: rebuilt, or the best candidate's tag-level mass.
    if mass_branch is None:
        masses = rebuilt
    else:
        masses = ak.to_numpy(cut_utils.values(tree, mass_branch, best)).astype(float)
        assert len(masses) == n_best, f'{mass_branch}: not one per best candidate'

    # A Dalitz point needs m^2 > 0 on both axes.
    physical = (m12s > 0) & (m23s > 0)
    m12s, m23s, masses, rebuilt = m12s[physical], m23s[physical], masses[physical], rebuilt[physical]
    cutflow.append(('m12 > 0 and m23 > 0', len(masses)))
    for label, n in cutflow:
        print(f'[INFO]   {label:28s} {n:9d}')

    # Consistency of the mass with the daughters m12, m23 come from.
    mass_minus_rebuilt = masses - rebuilt
    mass_lo, mass_hi = kinematics.m123_limits(m12s, m23s)
    n_outside = int(np.sum(~((mass_lo <= masses) & (masses <= mass_hi))))
    print(f"[INFO] {mass} - rebuilt from {momenta[0].rsplit('_', 1)[0]}_*: median "
          f"{np.median(mass_minus_rebuilt):.3f} MeV, RMS "
          f"{np.sqrt(np.mean(mass_minus_rebuilt**2)):.3f} MeV; {n_outside} of {len(masses)} "
          f"outside their kinematic limits")

    # Ranges, against the eta's Dalitz region [GeV^2] and the mass window.
    m_mu, m_eta = kinematics.M_MU / GEV, kinematics.M_ETA / GEV
    print(f'[INFO] m12 range [{m12s.min():.4f}, {m12s.max():.4f}]  '
          f'(eta: [{m_mu**2:.4f}, {(m_eta - m_mu)**2:.4f}])')
    print(f'[INFO] m23 range [{m23s.min():.4f}, {m23s.max():.4f}]  '
          f'(eta: [{(2 * m_mu)**2:.4f}, {m_eta**2:.4f}])')
    print(f'[INFO] mass range [{masses.min():.2f}, {masses.max():.2f}] MeV')

    selection = {
        'meta': {
            'input_file': infile,
            'mass': mass,
            'momenta': momenta,
            'selection': {
                'tos': tos_lines or None,
                'pnn_mu_min': pnn_mu_min,
                'truth_match': truth_match,
                'best_candidate': f'min {METRIC_BRANCH}',
                'daughters': DAUGHTERS,
                'max_events': max_events,
            },
        },
        'selection_report': {
            'events': drop_report,
            'triggers': triggers,
            'cutflow': [{'cut': label, 'candidates': n} for label, n in cutflow],
        },
    }
    return m12s, m23s, masses, selection


def main():
    parser = argparse.ArgumentParser(
        description='Select eta -> mu+ mu- gamma candidates and fit them per Dalitz bin',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--outdir', required=True,
                        help='Directory for selection.json, dalitz_fit_<factor>.json and plots')
    parser.add_argument('--mass', default='rebuilt', choices=list(MASS_SOURCES),
                        help='Mass and daughter momenta (default: rebuilt)')
    parser.add_argument('--factor', default='none', choices=list(FACTORS),
                        help='Per-bin factor on the signal and background shapes (default: none)')
    parser.add_argument('--truth-match', action='store_true',
                        help='Signal MC: keep only candidates truth-matched to eta -> mu+ mu- '
                             'gamma, and fit with no tail prior and no background')
    parser.add_argument('--tail-prior', default=None,
                        help='dalitz_fit_<factor>.json of the --truth-match fit of the same '
                             '--mass, whose global-fit tails are the prior')
    parser.add_argument('--tos-levels', nargs='*', default=list(TOS_LINES),
                        choices=list(TOS_LINES),
                        help='Trigger levels the candidate must be TOS at (default: L0 HLT1 HLT2). '
                             'Pass none for no TOS cut')
    parser.add_argument('--pnn-mu-min', type=float, default=None,
                        help='Every muon daughter must have prt_pnn_mu above this (default: no cut)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--horizontal', action='store_true',
                        help='Also write the per-bin pages with j (m23) as the outer loop')
    args = parser.parse_args()
    if args.truth_match and args.tail_prior is not None:
        parser.error('--truth-match and --tail-prior are exclusive')
    if not args.truth_match and args.tail_prior is None:
        parser.error('--tail-prior is required without --truth-match')
    os.makedirs(args.outdir, exist_ok=True)

    # --- Select ---

    m12s, m23s, masses, selection = select_candidates(
        args.infile, mass=args.mass, tos_levels=args.tos_levels, pnn_mu_min=args.pnn_mu_min,
        truth_match=args.truth_match, max_events=args.max_events)
    selection_path = os.path.join(args.outdir, 'selection.json')
    with open(selection_path, 'w') as f:
        json.dump(selection, f, indent=2)
    print(f'[DONE] {selection_path}')
    for name, hist_range in HISTOGRAMS:
        plot_dalitz_hist(m12s, m23s, hist_range, os.path.join(args.outdir, name))

    # --- Fit ---

    m12_edges = np.linspace(*M12_RANGE, N_PER_AXIS + 1)   # [GeV^2]
    m23_edges = np.linspace(*M23_RANGE, N_PER_AXIS + 1)   # [GeV^2]
    signal_factor, bkg_factor = FACTORS[args.factor]

    # Tail prior: the MC fit's tail parameters and covariance, stdevs widened
    tail_prior = None
    if args.tail_prior is not None:
        with open(args.tail_prior) as f:
            mc_fit = json.load(f)['fit']
        tail_prior = fit_utils.build_prior_from_result(mc_fit, TAIL_PARAMS,
                                                       factor=TAIL_PRIOR_FACTOR)

    result, global_fit = fit_dalitz(m12s, m23s, masses, m12_edges, m23_edges,
                                    xmin=XMIN, xmax=XMAX, nbins=MASS_NBINS,
                                    mean_init=MEAN_INIT, sigma_init=SIGMA_INIT,
                                    tail_prior=tail_prior,
                                    signal_prior_factor=SIGNAL_PRIOR_FACTOR,
                                    min_candidates=MIN_CANDIDATES,
                                    signal_factor=signal_factor, bkg_factor=bkg_factor,
                                    signal_only=args.truth_match)
    result['meta']['input_file'] = args.infile
    result['meta']['tail_prior'] = (None if args.tail_prior is None else
                                    {'path': args.tail_prior, 'factor': TAIL_PRIOR_FACTOR})
    dalitz = result['dalitz']

    # --- Summary ---

    # Global fit: MIGRAD summary and every parameter
    print(global_fit.fmin)
    for name in global_fit.parameters:
        print(f'  {name:8s} value={global_fit.values[name]:12.4f}  '
              f'error={global_fit.errors[name]:10.4f}  limits={global_fit.limits[name]}')
    params = result['fit']['parameters']
    print(f"[INFO] per-bin seeds: mean={params['mean']['value']:.3f}  "
          f"sigma={params['sigma']['value']:.3f}; fixed tails: "
          + '  '.join(f"{name}={params[name]['value']:.3f}" for name in TAIL_PARAMS))

    # Per-bin outcome
    bins = dalitz['bins']
    fitted = [b for b in bins if b['fitted']]
    valid = [b for b in bins if b['valid']]
    outcome = dalitz['outcome']
    print(f"[INFO] {args.factor}: {len(bins)} bins, {outcome['skipped']} skipped "
          f"(<= {MIN_CANDIDATES} candidates), {outcome['invalid']} invalid, "
          f"{outcome['valid']} valid, "
          f"{sum(b['n_sig_fixed_zero'] for b in fitted)} with n_sig fixed to 0")

    # Invalid fits, with the candidate counts of their fitted neighbors
    for i, j, n_cand, neighbors in hole_neighbors(result):
        neighbor_text = ', '.join(f'{offset}: {count:.0f}' for offset, count in neighbors.items())
        print(f'[INFO]   invalid ({i:2d},{j:2d}): n_cand={n_cand:.0f}  neighbors: {neighbor_text}')

    # Valid fits whose Chebyshev goes below 0 where the background is reachable
    negative = [b for b in valid if b['bkg_goes_negative']]
    print(f'[INFO] {len(negative)} of {len(valid)} valid fits with a negative background')
    for b in negative:
        print(f"[INFO]   ({b['i']:2d},{b['j']:2d}): "
              + '  '.join(f'{name}={b[name]:.3f}' for name in fit_utils.BKG_PARAMS))

    # Summed per-bin yields vs the global fit
    closure = dalitz['yield_closure']
    print(f"[INFO] n_sig global {closure['n_sig_global']:.1f}, summed over bins "
          f"{closure['n_sig_summed']:.1f}; n_bkg global {closure['n_bkg_global']:.1f}, summed "
          f"{closure['n_bkg_summed']:.1f}")
    print(f"[INFO] {closure['window_candidates_in_bins']} of {closure['window_candidates']} "
          f"window candidates in the bins; {closure['candidates_in_grid']} of "
          f"{closure['candidates']} candidates (any mass) in the grid")
    chi2 = dalitz['chi2_per_ndof_over_bins']
    if chi2 is not None:
        print(f"[INFO] chi2/ndof over valid bins: mean {chi2['mean']:.3f}, "
              f"median {chi2['median']:.3f}, max {chi2['max']:.3f}")

    # --- Write ---

    json_path = os.path.join(args.outdir, f'dalitz_fit_{args.factor}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'[DONE] {json_path}')

    plot_dalitz_fit(result, os.path.splitext(json_path)[0], horizontal=args.horizontal)


if __name__ == '__main__':
    main()
