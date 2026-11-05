"""
driver_dalitz.py

Per-Dalitz-bin mass fits of eta -> mu+ mu- gamma from a ROOT file:

    read ROOT -> drop malformed events -> (truth match) -> TOS -> muon cuts
              -> random candidate -> daughter four-vectors -> m12s, m23s, masses
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

    tag_dtf_m  tag_dtf_m; m12, m23 from the DTF daughters prt_dtf_px/py/pz/e (default)
    tag_m      tag_m; m12, m23 from prt_px/py/pz/e

--factor picks the per-bin factor on the signal and background shapes (FACTORS):

    none      no factor on either shape (default)
    flat      kinematics.phase_space_factor on the signal and the background
    weighted  kinematics.dalitz_factor on the signal,
              kinematics.phase_space_factor on the background

One candidate per event is picked at random among those that pass the cuts
(cut_utils.random_candidate_mask, seed selection.SEED).

--truth-match (signal MC only) keeps the candidates with every daughter truth-matched to eta -> mu+
mu- gamma (bkg_utils.signal_candidate_mask). It is the first cut, before TOS, pnn_mu and the
candidate pick. The matched candidates are fit with no tail prior and the background fixed to 0
(fit_dalitz signal_only).

--tail-prior takes the 'fit' block of a results JSON (results_dcb.json of driver_mass_fit.py, or
dalitz_fit_<factor>.json): the global fit carries the tail prior of that fit, its stdevs times
TAIL_PRIOR_FACTOR. Without it the tails float free.

--n-max is the upper limit of the global fit's tail exponents n_L and n_R.

The number of candidates outside their own kinematic limits (kinematics.m123_limits) is printed. No
mass-window or Dalitz-grid cut is applied to the selection.

select_candidates() is the selection on its own, for fit_dalitz.ipynb and plot_phase_space.py.

Usage:
    python src/fit/driver_dalitz.py INFILE --outdir OUTDIR
        [--truth-match | --tail-prior RESULTS_JSON]
        [--mass tag_dtf_m|tag_m] [--factor none|flat|weighted]
        [--n-max N] [--max-events N] [--horizontal]
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
import cut_utils
import fit_utils
import kinematics
import selection
from fit_dalitz import fit_dalitz, hole_neighbors
from fit_dcb import TAIL_PARAMS
from plot_dalitz import HISTOGRAMS, plot_dalitz_hist
from plot_dalitz_bins import plot_dalitz_fit

# --- Selection ---------------------------------------------------------------

# Daughter four-momentum branches of each --mass choice
PRT_MOMENTA = ['prt_px', 'prt_py', 'prt_pz', 'prt_e']
DTF_MOMENTA = ['prt_dtf_px', 'prt_dtf_py', 'prt_dtf_pz', 'prt_dtf_e']
MASS_SOURCES = {
    'tag_dtf_m': DTF_MOMENTA,
    'tag_m':     PRT_MOMENTA,
}

# Daughter name -> PID. Every candidate is built from exactly one of each.
DAUGHTERS = {'gamma': 22, 'mu+': -13, 'mu-': 13}

GEV = 1000.0   # MeV per GeV

# --- Fit ---------------------------------------------------------------------

XMIN, XMAX, MASS_NBINS = 500.0, 600.0, 100     # mass window [MeV] and its histogram bins
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


def select_candidates(infile, mass='tag_dtf_m', truth_match=False, max_events=None):
    """
    Read infile, select, and pick one candidate per event at random.

    Returns (m12s, m23s, masses, selection): the Dalitz variables [GeV^2] and gamma mu+ mu- mass
    [MeV] of each event's picked candidate, and the selection settings and report (the contents of
    selection.json).
    """
    momenta = MASS_SOURCES[mass]
    branches = ([mass, 'prt_idx_mom'] + selection.MUON_CUT_BRANCHES
                + [b for b in momenta if b not in selection.MUON_CUT_BRANCHES]
                + [line for lines in selection.TOS_LINES.values() for line in lines]
                + (selection.TRUTH_BRANCHES if truth_match else []))

    # --- Read ---

    tree = cut_utils.read_branches(infile, branches, entry_stop=max_events)
    tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
    tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=mass)
    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}, "
          f"dropped {drop_report['read'] - drop_report['kept']}")
    for reason in ('dropped_no_candidates', 'dropped_idx_out_of_range',
                   'dropped_daughter_length_mismatch', 'dropped_partial_candidate_branch'):
        if drop_report[reason]:
            print(f'[INFO]   {reason}: {drop_report[reason]}')
    n_multi = int(ak.sum(ak.num(tree[mass]) > 1))
    print(f'[INFO] {n_multi} / {len(tree)} events have >1 candidate')

    # Every candidate must have exactly one daughter of each PID.
    n_cand_total = int(ak.sum(ak.num(tree[mass])))
    for name, pid in DAUGHTERS.items():
        n_of_pid = cut_utils.reduce(tree, tree['prt_pid'] == pid, 'prt_idx_mom',
                                    how='sum', target_ref=mass)
        print(f'[INFO] {name:5s} (pid {pid:>3}): {float(ak.mean(ak.flatten(n_of_pid))):.4f} '
              f'daughters per candidate')
        assert ak.all(n_of_pid == 1), f'some candidates do not have exactly one {name}'
    print(f'[INFO] {n_cand_total} candidates, each exactly (gamma, mu+, mu-)')

    # --- Cuts ---

    # [truth match to eta -> gamma mu+ mu-], TOS per trigger level, then each muon cut
    truth_pids = list(DAUGHTERS.values()) if truth_match else None
    steps, triggers = selection.cut_steps(tree, mass, selection.TOS_LINES, truth_pids=truth_pids)
    keep = steps[-1][1]
    cutflow = [(label, int(ak.sum(mask))) for label, mask in steps]
    for level, lines in triggers.items():
        for line, n in lines.items():
            if n == 0:
                print(f'[WARN] {level} TOS line {line} passes no candidate')

    # --- Random candidate and its daughters ---

    # One candidate per event: picked at random among those kept.
    picked = cut_utils.random_candidate_mask(keep, selection.SEED)
    n_picked = int(ak.sum(picked))
    cutflow.append(('random candidates', n_picked))
    assert n_picked > 0, 'no event survives the selection'

    # Daughter-level mask: True for the daughters of each picked candidate.
    picked_daughter = cut_utils.follow(tree, picked, 'prt_idx_mom')

    # (px, py, pz, e) [GeV] of each daughter of the picked candidates, event-aligned.
    p4 = {}
    for name, pid in DAUGHTERS.items():
        match = picked_daughter & (tree['prt_pid'] == pid)
        p4[name] = [ak.to_numpy(ak.flatten(tree[field][match])) / GEV for field in momenta]
        assert len(p4[name][0]) == n_picked, f'{name}: not one per picked candidate'

    # Dalitz variables from the four-vectors, and the picked candidate's mass.
    m12s = kinematics.invariant_mass2(*p4['gamma'], *p4['mu+'])
    m23s = kinematics.invariant_mass2(*p4['mu+'], *p4['mu-'])
    masses = ak.to_numpy(cut_utils.values(tree, mass, picked)).astype(float)
    assert len(masses) == n_picked, f'{mass}: not one per picked candidate'

    # A Dalitz point needs m^2 > 0 on both axes.
    physical = (m12s > 0) & (m23s > 0)
    m12s, m23s, masses = m12s[physical], m23s[physical], masses[physical]
    cutflow.append(('m12 > 0 and m23 > 0', len(masses)))
    for label, n in cutflow:
        print(f'[INFO]   {label:28s} {n:9d}')

    # Consistency of the mass with the daughters m12, m23 come from.
    mass_lo, mass_hi = kinematics.m123_limits(m12s, m23s)
    n_outside = int(np.sum(~((mass_lo <= masses) & (masses <= mass_hi))))
    print(f'[INFO] {n_outside} of {len(masses)} candidates outside the kinematic limits of '
          f'their m12, m23')

    # Ranges, against the eta's Dalitz region [GeV^2] and the mass window.
    m_mu, m_eta = kinematics.M_MU / GEV, kinematics.M_ETA / GEV
    print(f'[INFO] m12 range [{m12s.min():.4f}, {m12s.max():.4f}]  '
          f'(eta: [{m_mu**2:.4f}, {(m_eta - m_mu)**2:.4f}])')
    print(f'[INFO] m23 range [{m23s.min():.4f}, {m23s.max():.4f}]  '
          f'(eta: [{(2 * m_mu)**2:.4f}, {m_eta**2:.4f}])')
    print(f'[INFO] mass range [{masses.min():.2f}, {masses.max():.2f}] MeV')

    selection_json = {
        'meta': {
            'input_file': infile,
            'mass': mass,
            'momenta': momenta,
            'selection': {
                'tos': selection.TOS_LINES,
                'mu_pt_min': selection.MU_PT_MIN,
                'mu_p_min': selection.MU_P_MIN,
                'mu_eta_range': selection.MU_ETA_RANGE,
                'pnn_mu_min': selection.PNN_MU_MIN,
                'mu_ghost_max': selection.MU_GHOST_MAX,
                'mu_ip_chi2_max': selection.MU_IP_CHI2_MAX,
                'truth_match': truth_match,
                'random_candidate': {'seed': selection.SEED},
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
    return m12s, m23s, masses, selection_json


def main():
    parser = argparse.ArgumentParser(
        description='Select eta -> mu+ mu- gamma candidates and fit them per Dalitz bin',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--outdir', required=True,
                        help='Directory for selection.json, dalitz_fit_<factor>.json and plots')
    parser.add_argument('--mass', default='tag_dtf_m', choices=list(MASS_SOURCES),
                        help='Mass and daughter momenta (default: tag_dtf_m)')
    parser.add_argument('--factor', default='none', choices=list(FACTORS),
                        help='Per-bin factor on the signal and background shapes (default: none)')
    parser.add_argument('--truth-match', action='store_true',
                        help='Signal MC: keep only candidates truth-matched to eta -> mu+ mu- '
                             'gamma, and fit with no tail prior and no background')
    parser.add_argument('--tail-prior', default=None,
                        help='results_dcb.json or dalitz_fit_<factor>.json whose tails are '
                             'the global-fit prior (default: none, tails free)')
    parser.add_argument('--n-max', type=float, default=None,
                        help='Upper limit of the global-fit tail exponents n_L and n_R '
                             '(default: none)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--horizontal', action='store_true',
                        help='Also write the per-bin pages with j (m23) as the outer loop')
    args = parser.parse_args()
    if args.truth_match and args.tail_prior is not None:
        parser.error('--truth-match and --tail-prior are exclusive')
    os.makedirs(args.outdir, exist_ok=True)

    # --- Select ---

    m12s, m23s, masses, selection_json = select_candidates(
        args.infile, mass=args.mass, truth_match=args.truth_match, max_events=args.max_events)
    selection_path = os.path.join(args.outdir, 'selection.json')
    with open(selection_path, 'w') as f:
        json.dump(selection_json, f, indent=2)
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
                                    signal_only=args.truth_match,
                                    limits={'nL': (0.0, args.n_max), 'nR': (0.0, args.n_max)})
    result['meta']['input_file'] = args.infile
    result['meta']['tail_prior'] = (None if args.tail_prior is None else
                                    {'path': args.tail_prior, 'factor': TAIL_PRIOR_FACTOR})
    result['meta']['n_max'] = args.n_max
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
