"""
driver_dalitz.py

Select eta -> mu+ mu- gamma candidates from a ROOT file and write each
event's best candidate as its Dalitz variables and gamma mu+ mu- mass:

    read ROOT -> drop malformed events -> TOS (+ pnn_mu) -> best candidate
              -> daughter four-vectors -> m12s, m23s, masses
              -> dalitz_events.npz, selection.json

    m12s    m^2(gamma mu+)        [GeV^2]
    m23s    m^2(mu+ mu-) = q^2    [GeV^2]
    masses  m(gamma mu+ mu-)      [MeV], built from the same daughter
                                  four-vectors as m12s and m23s

No mass-window or Dalitz-grid cut is applied here.

Usage:
    python src/fit/driver_dalitz.py INFILE --outdir OUTDIR
        [--tos-levels L0 HLT1 HLT2] [--pnn-mu-min X] [--max-events N]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils
import kinematics

# --- Selection ---------------------------------------------------------------

MASS_BRANCH   = 'tag_m'           # tag-level reference branch for cut_utils
METRIC_BRANCH = 'tag_dtf_chi2'    # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu', 'prt_px', 'prt_py', 'prt_pz', 'prt_e']
MOMENTUM_FIELDS = ['prt_px', 'prt_py', 'prt_pz', 'prt_e']

# Daughter name -> PID. Every candidate is built from exactly one of each.
DAUGHTERS = {'gamma': 22, 'mu+': -13, 'mu-': 13}

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
parser.add_argument('--tos-levels', nargs='*', default=list(TOS_LINES),
                    choices=list(TOS_LINES),
                    help='Trigger levels the candidate must be TOS at (default: L0 HLT1 HLT2). '
                         'Pass none for no TOS cut')
parser.add_argument('--pnn-mu-min', type=float, default=None,
                    help='Every muon daughter must have prt_pnn_mu above this (default: no cut)')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events (default: all)')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

tos_lines = {level: TOS_LINES[level] for level in args.tos_levels}
branches = ([MASS_BRANCH, METRIC_BRANCH, 'prt_idx_mom'] + DAUGHTER_FIELDS
            + [line for lines in tos_lines.values() for line in lines])

# --- Read --------------------------------------------------------------------

tree = cut_utils.read_branches(args.infile, branches, entry_stop=args.max_events)
tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}, "
      f"dropped {drop_report['dropped_total']}")

# Every candidate must have exactly one daughter of each PID.
n_cand_total = int(ak.sum(ak.num(tree[MASS_BRANCH])))
for name, pid in DAUGHTERS.items():
    n_of_pid = cut_utils.reduce(tree, tree['prt_pid'] == pid, 'prt_idx_mom',
                                how='sum', target_ref=MASS_BRANCH)
    assert ak.all(n_of_pid == 1), f'some candidates do not have exactly one {name}'
print(f'[INFO] {n_cand_total} candidates, each exactly (gamma, mu+, mu-)')

# --- Cuts --------------------------------------------------------------------

keep = ak.ones_like(tree[MASS_BRANCH], dtype=bool)
cutflow = [('all candidates', n_cand_total)]
tos_report = None

# TOS on at least one line at every requested level.
if tos_lines:
    tos, tos_report = cut_utils.tos_mask(tree, tos_lines, tag_ref=MASS_BRANCH)
    keep = keep & tos
    for level, report in tos_report.items():
        cutflow.append((f'TOS up to {level}', report['candidates_tos_cumulative']))

# ProbNNmu on every muon daughter; a non-muon daughter passes.
if args.pnn_mu_min is not None:
    muon_ok = (tree['prt_pnn_mu'] > args.pnn_mu_min) | (abs(tree['prt_pid']) != 13)
    keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                   target_ref=MASS_BRANCH)
    cutflow.append((f'pnn_mu > {args.pnn_mu_min}', int(ak.sum(keep))))

for label, n in cutflow:
    print(f'[INFO]   {label:24s} {n:9d}')

# --- Best candidate and its daughters ---------------------------------------

# One candidate per event: smallest METRIC_BRANCH among those kept.
best = cut_utils.best_candidate_mask(tree, METRIC_BRANCH, keep)
n_best = int(ak.sum(best))
assert n_best > 0, 'no event survives the selection'

# Daughter-level mask: True for the daughters of each best candidate.
best_daughter = cut_utils.follow(tree, best, 'prt_idx_mom')

# (px, py, pz, e) [GeV] of each daughter of the best candidates, event-aligned.
p4 = {}
for name, pid in DAUGHTERS.items():
    match = best_daughter & (tree['prt_pid'] == pid)
    p4[name] = [ak.to_numpy(ak.flatten(tree[field][match])) / GEV for field in MOMENTUM_FIELDS]
    assert len(p4[name][0]) == n_best, f'{name}: not one per best candidate'

# Dalitz variables and the gamma mu+ mu- mass from the same four-vectors.
m12s = kinematics.invariant_mass2(*p4['gamma'], *p4['mu+'])
m23s = kinematics.invariant_mass2(*p4['mu+'], *p4['mu-'])
gamma_mup = [a + b for a, b in zip(p4['gamma'], p4['mu+'], strict=True)]
masses = np.sqrt(np.clip(kinematics.invariant_mass2(*gamma_mup, *p4['mu-']), 0.0, None)) * GEV

# A Dalitz point needs m^2 > 0 on both axes.
physical = (m12s > 0) & (m23s > 0)
m12s, m23s, masses = m12s[physical], m23s[physical], masses[physical]
n_unphysical = int((~physical).sum())
print(f'[INFO] {n_best} best candidates, {n_unphysical} dropped with m12 <= 0 or m23 <= 0, '
      f'{len(masses)} written')

# --- Write -------------------------------------------------------------------

events_path = os.path.join(args.outdir, 'dalitz_events.npz')
np.savez(events_path, m12s=m12s, m23s=m23s, masses=masses)

selection = {
    'meta': {
        'input_file': args.infile,
        'mass': 'rebuilt from daughters',
        'selection': {
            'tos': tos_lines or None,
            'pnn_mu_min': args.pnn_mu_min,
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
        'branches_read': branches,
    },
}
selection_path = os.path.join(args.outdir, 'selection.json')
with open(selection_path, 'w') as f:
    json.dump(selection, f, indent=2)
print(f'[DONE] {events_path}')
print(f'[DONE] {selection_path}')
