"""
driver_dalitz.py

Select eta -> mu+ mu- gamma candidates from a ROOT file and write each
event's best candidate as its Dalitz variables and gamma mu+ mu- mass:

    read ROOT -> drop malformed events -> TOS (+ pnn_mu) -> best candidate
              (+ truth match)
              -> daughter four-vectors -> m12s, m23s, masses
              -> dalitz_events.npz, selection.json

    m12s    m^2(gamma mu+)        [GeV^2]
    m23s    m^2(mu+ mu-) = q^2    [GeV^2]
    masses  m(gamma mu+ mu-)      [MeV]

--mass picks the mass and the daughter momenta together (MASS_SOURCES):

    rebuilt    mass rebuilt from prt_px/py/pz/e (default)
    tag_m      tag_m; m12, m23 from prt_px/py/pz/e
    tag_dtf_m  tag_dtf_m; m12, m23 from the DTF daughters prt_dtf_px/py/pz/e

--truth-match (signal MC only) keeps the events whose best candidate has
every daughter truth-matched to eta -> mu+ mu- gamma
(bkg_utils.signal_candidate_mask): the signal as the selection picks it, for
driver_dalitz_fit.py --signal-only, whose global fit is the tail prior of the
data fit of the same --mass.

For each, the median and RMS of (mass - mass rebuilt from the same daughters)
and the number of candidates outside their own kinematic limits
(kinematics.m123_limits) go to stderr and selection.json. No mass-window or
Dalitz-grid cut is applied here.

Usage:
    python src/fit/driver_dalitz.py INFILE --outdir OUTDIR
        [--mass rebuilt|tag_m|tag_dtf_m] [--tos-levels L0 HLT1 HLT2]
        [--pnn-mu-min X] [--truth-match] [--max-events N]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import bkg_utils
import cut_utils
import kinematics

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

parser = argparse.ArgumentParser(
    description='Select eta -> mu+ mu- gamma candidates and write their Dalitz variables',
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument('infile', help='Input ROOT file')
parser.add_argument('--outdir', required=True,
                    help='Directory for dalitz_events.npz and selection.json')
parser.add_argument('--mass', default='rebuilt', choices=list(MASS_SOURCES),
                    help='Mass and daughter momenta (default: rebuilt)')
parser.add_argument('--tos-levels', nargs='*', default=list(TOS_LINES),
                    choices=list(TOS_LINES),
                    help='Trigger levels the candidate must be TOS at (default: L0 HLT1 HLT2). '
                         'Pass none for no TOS cut')
parser.add_argument('--pnn-mu-min', type=float, default=None,
                    help='Every muon daughter must have prt_pnn_mu above this (default: no cut)')
parser.add_argument('--truth-match', action='store_true',
                    help='Signal MC: keep only candidates truth-matched to eta -> mu+ mu- gamma')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events (default: all)')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

mass_branch = MASS_SOURCES[args.mass]['mass']
momenta = MASS_SOURCES[args.mass]['momenta']
tos_lines = {level: TOS_LINES[level] for level in args.tos_levels}
branches = ([TAG_REF_BRANCH, METRIC_BRANCH, 'prt_idx_mom', 'prt_pid', 'prt_pnn_mu']
            + ([mass_branch] if mass_branch not in (None, TAG_REF_BRANCH) else [])
            + momenta
            + [line for lines in tos_lines.values() for line in lines]
            + (TRUTH_BRANCHES if args.truth_match else []))

# --- Read --------------------------------------------------------------------

tree = cut_utils.read_branches(args.infile, branches, entry_stop=args.max_events)
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

# --- Cuts --------------------------------------------------------------------

keep = ak.ones_like(tree[TAG_REF_BRANCH], dtype=bool)
cutflow = [('all candidates', n_cand_total)]
tos_report = None

# TOS on at least one line at every requested level.
if tos_lines:
    tos, tos_report = cut_utils.tos_mask(tree, tos_lines, tag_ref=TAG_REF_BRANCH)
    keep = keep & tos
    for level, report in tos_report.items():
        cutflow.append((f'TOS up to {level}', report['tos_up_to_level']))
        for line, n in report['lines'].items():
            if n == 0:
                print(f'[WARN] {level} TOS line {line} passes no candidate')

# ProbNNmu on every muon daughter; a non-muon daughter passes.
if args.pnn_mu_min is not None:
    muon_ok = (tree['prt_pnn_mu'] > args.pnn_mu_min) | (abs(tree['prt_pid']) != 13)
    keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                   target_ref=TAG_REF_BRANCH)
    cutflow.append((f'pnn_mu > {args.pnn_mu_min}', int(ak.sum(keep))))

for label, n in cutflow:
    print(f'[INFO]   {label:24s} {n:9d}')

# --- Best candidate and its daughters ---------------------------------------

# One candidate per event: smallest METRIC_BRANCH among those kept.
best = cut_utils.best_candidate_mask(tree, METRIC_BRANCH, keep)

# Truth match: keep the events whose best candidate has every daughter
# matched to eta -> gamma mu+ mu-.
if args.truth_match:
    cutflow.append(('best candidates', int(ak.sum(best))))
    best = best & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, list(DAUGHTERS.values()),
                                                  target_ref=TAG_REF_BRANCH)
    cutflow.append(('best candidate truth-matched', int(ak.sum(best))))
    for label, n in cutflow[-2:]:
        print(f'[INFO]   {label:28s} {n:9d}')
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
n_unphysical = int((~physical).sum())
print(f'[INFO] {n_best} best candidates, {n_unphysical} dropped with m12 <= 0 or m23 <= 0, '
      f'{len(masses)} written')

# Consistency of the mass with the daughters m12, m23 come from.
mass_minus_rebuilt = masses - rebuilt
mass_lo, mass_hi = kinematics.m123_limits(m12s, m23s)
n_outside = int(np.sum(~((mass_lo <= masses) & (masses <= mass_hi))))
consistency = {
    'median_mass_minus_rebuilt': float(np.median(mass_minus_rebuilt)),
    'rms_mass_minus_rebuilt': float(np.sqrt(np.mean(mass_minus_rebuilt**2))),
    'outside_kinematic_limits': n_outside,
}
print(f"[INFO] {args.mass} - rebuilt from {momenta[0].rsplit('_', 1)[0]}_*: median "
      f"{consistency['median_mass_minus_rebuilt']:.3f} MeV, RMS "
      f"{consistency['rms_mass_minus_rebuilt']:.3f} MeV; {n_outside} of {len(masses)} "
      f"outside their kinematic limits")

# Ranges, against the eta's Dalitz region [GeV^2] and the mass window.
m_mu, m_eta = kinematics.M_MU / GEV, kinematics.M_ETA / GEV
print(f'[INFO] m12 range [{m12s.min():.4f}, {m12s.max():.4f}]  '
      f'(eta: [{m_mu**2:.4f}, {(m_eta - m_mu)**2:.4f}])')
print(f'[INFO] m23 range [{m23s.min():.4f}, {m23s.max():.4f}]  '
      f'(eta: [{(2 * m_mu)**2:.4f}, {m_eta**2:.4f}])')
print(f'[INFO] mass range [{masses.min():.2f}, {masses.max():.2f}] MeV')

# --- Write -------------------------------------------------------------------

events_path = os.path.join(args.outdir, 'dalitz_events.npz')
np.savez(events_path, m12s=m12s, m23s=m23s, masses=masses)

selection = {
    'meta': {
        'input_file': args.infile,
        'mass': args.mass,
        'momenta': momenta,
        'selection': {
            'tos': tos_lines or None,
            'pnn_mu_min': args.pnn_mu_min,
            'truth_match': args.truth_match,
            'best_candidate': f'min {METRIC_BRANCH}',
            'daughters': DAUGHTERS,
            'max_events': args.max_events,
        },
    },
    'selection_report': {
        'events': drop_report,
        'triggers': tos_report,
        'cutflow': [{'cut': label, 'candidates': n} for label, n in cutflow],
        'best_candidates': n_best,
        'dropped_unphysical_m2': n_unphysical,
        'events_written': len(masses),
        'mass_consistency': consistency,
        'branches_read': branches,
    },
}
selection_path = os.path.join(args.outdir, 'selection.json')
with open(selection_path, 'w') as f:
    json.dump(selection, f, indent=2)
print(f'[DONE] {events_path}')
print(f'[DONE] {selection_path}')
