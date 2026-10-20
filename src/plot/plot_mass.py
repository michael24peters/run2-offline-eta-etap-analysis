"""
plot_mass.py

Plots the mass distribution of candidates from a ROOT file.

Usage:
    python plot_mass.py input.root [output.png] [--best] [--full-range]
                        [--mass tag_dtf_m|tag_m]
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

# Project imports
# Ensure utils/cut_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils
import kinematics
import plot_style
import plot_utils

# IO
parser = argparse.ArgumentParser(description='Plot mass distributions from ROOT file')
parser.add_argument('input', 
                    help='Input ROOT file')
parser.add_argument('output', nargs='?', default='mass_plot.png',
                    help='Output PNG file (default: mass_plot.png)')
# Flag to only plot "best" candidate (one candidate per event),
# using cut_utils.best_candidate_mask() to select the best candidate.
parser.add_argument('--best', action='store_true', default=False,
                    help='Only plot best candidate per event (default: False)')
# Flag to plot full mass range (400-1200 MeV) instead of zoomed-in range 
# (450-600 MeV).
parser.add_argument('--full-range', action='store_true', default=False,
                    help='Plot full mass range (400-1200 MeV) instead of zoomed-in range (450-600 MeV) (default: False)')
# Selection, matching driver_dcb.py / driver_gauss.py. Defaults to no additional
# cut, so a bare run is a diagnostic overview of everything in the file.
parser.add_argument('--tos', action='store_true', default=False,
                    help='Require candidates to be TOS on at least one line at every '
                         'trigger level (L0, HLT1, HLT2)')
parser.add_argument('--pnn-mu-min', type=float, default=0.4,
                    help='Muon PID threshold (default: 0.4, the floor already applied '
                         'upstream in the ntuple, i.e. no additional cut)')
parser.add_argument('--mass', choices=['tag_dtf_m', 'tag_m'], default='tag_dtf_m',
                    help='Mass branch to plot (default: tag_dtf_m)')
args = parser.parse_args()
print(f'Reading from {args.input} and writing to {args.output}.')

# Raw, per-event jagged array of candidate masses (always needed for the
# candidate-multiplicity diagnostics below, regardless of --best).
# The prt_/trigger branches are read and validated unconditionally so that every
# selection variant runs over the same event sample and differs only by the cut.
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']
MUON_PID = 13
# Trigger-on-signal lines per trigger level; TOS requires at least one line at
# every level (cut_utils.tos_mask). A property of this ntuple, not of cut_utils.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}
TOS_BRANCHES = [branch for lines in TOS_LINES.values() for branch in lines]
tree = cut_utils.read_branches(
    args.input,
    [args.mass, 'tag_dtf_chi2', 'prt_idx_mom'] + DAUGHTER_FIELDS + TOS_BRANCHES)
tree, _ = cut_utils.drop_malformed(tree, tag_ref=args.mass)

# Get lengths of arrays in the mass branch to see how many candidates per event
lengths = ak.to_numpy(ak.num(tree[args.mass]))
print(f"{'─' * 48}")
print(f'Events with 0 candidates: {np.sum(lengths == 0)}')
print(f'Events with 1 candidate:  {np.sum(lengths == 1)}')
print(f'Events with 2+ candidates: {np.sum(lengths >= 2)}')
print(f'Events with 3+ candidates: {np.sum(lengths >= 3)}')
print(f'Max candidates in an event: {lengths.max()}')
# Candidates plotted with --best=False is every candidate (lengths.sum());
# with --best=True it's one per event that has at least one candidate.
# Report both so the effect of --best is visible regardless of which
# mode this run actually uses.
n_all_candidates = int(lengths.sum())
n_best_candidates = int(np.sum(lengths >= 1))
print(f'Candidates plotted (--best=False): {n_all_candidates}')
print(f'Candidates plotted (--best=True):  {n_best_candidates}')
print(f'Difference (candidates dropped by --best selection): {n_all_candidates - n_best_candidates}')
print(f"{'─' * 48}")

# Apply the selection. pnn_mu_min defaults to the upstream floor, so a bare run
# passes every candidate and only --tos / a raised threshold narrow it.
keep = ak.ones_like(tree[args.mass], dtype=bool)
if args.tos:
    tos, _ = cut_utils.tos_mask(tree, TOS_LINES, tag_ref=args.mass)
    keep = keep & tos
if args.pnn_mu_min is not None:
    # "passes OR isn't a muon", reduced onto the candidate, so a candidate is cut
    # only when one of its own muon daughters fails.
    muon_ok = ((tree['prt_pnn_mu'] > args.pnn_mu_min)
               | (abs(tree['prt_pid']) != MUON_PID))
    keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                   target_ref=args.mass)
applied = ', '.join((['TOS'] if args.tos else [])
                    + ([f'pnn_mu>{args.pnn_mu_min}'] if args.pnn_mu_min is not None else []))
mass = tree[args.mass][keep]
print(f'Selection [{applied or "none"}]: '
      f'{int(ak.sum(ak.num(mass)))} / {n_all_candidates} candidates pass')

if args.best:
    # Select the best candidate for each event, by lowest tag_dtf_chi2. `keep` is
    # the eligibility mask, so each event keeps the best candidate that survived
    # the selection; events with none left simply contribute nothing.
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', keep)
    tag_m = cut_utils.values(tree, args.mass, best)
# Flatten to every surviving candidate
else: tag_m = ak.to_numpy(ak.flatten(mass))

print(f'Read {len(tag_m)} entries from {args.input}')
print(f'shape: {tag_m.shape}')
print(f'dtype: {tag_m.dtype}')
print(f'min: {tag_m.min():.2f}')
print(f'max: {tag_m.max():.2f}')
print(f"{'─' * 48}")

# Plot
print('Plotting histogram...')
C = plot_style.COLORS
plot_utils.apply_font()
plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(figsize=(12, 9))
if args.full_range: hist_range = (300, 1200)
else: hist_range = (450, 650)
ax.hist(tag_m, bins=100, range=hist_range, histtype='step', color=C['data'], label='Candidates')
ax.axvline(x=kinematics.M_ETA, color=C['reference'], linestyle='--', label=r'$\eta$')
if args.full_range:
    for label, resonance_mass, linestyle in plot_style.RESONANCES:
        ax.axvline(x=resonance_mass, color=C['reference'], linestyle=linestyle, label=label)
ax.set_xlabel('Mass [MeV]')
ax.set_ylabel('Candidates')
plot_style.style_axes(ax)
ax.legend(frameon=False)
plot_style.save_plot(fig, args.output)
