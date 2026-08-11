"""
plot_mass.py

Plots the mass distribution of candidates from a ROOT file.

Usage:
    python plot_mass.py input.root [output.png] [--best] [--full-range]
"""

import argparse
import os
import sys
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
# Project imports
# Ensure utils/cut_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils

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
                    help='Require candidates to be TOS for at least one trigger')
parser.add_argument('--pnn-mu-min', type=float, default=0.4,
                    help='Muon PID threshold (default: 0.4, the floor already applied '
                         'upstream in the ntuple, i.e. no additional cut)')
args = parser.parse_args()
print(f'Reading from {args.input} and writing to {args.output}.')

# Raw, per-event jagged array of candidate masses (always needed for the
# candidate-multiplicity diagnostics below, regardless of --best).
# The prt_/trigger branches are read and validated unconditionally so that every
# selection variant runs over the same event sample and differs only by the cut.
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']
tree = cut_utils.read_branches(
    args.input,
    ['tag_dtf_m', 'tag_dtf_chi2', 'prt_idx_mom'] + DAUGHTER_FIELDS + cut_utils.TOS_TRIGGERS)
tree = cut_utils.drop_malformed(tree, DAUGHTER_FIELDS, cut_utils.TOS_TRIGGERS)

# Get lengths of arrays in tag_dtf_m to see how many candidates per event
lengths = ak.to_numpy(ak.num(tree['tag_dtf_m']))
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
keep = cut_utils.selection_mask(tree, use_tos=args.tos, pnn_mu_min=args.pnn_mu_min)
applied = ', '.join((['TOS'] if args.tos else [])
                    + ([f'pnn_mu>{args.pnn_mu_min}'] if args.pnn_mu_min is not None else []))
mass = tree['tag_dtf_m'][keep]
print(f'Selection [{applied or "none"}]: '
      f'{int(ak.sum(ak.num(mass)))} / {n_all_candidates} candidates pass')

if args.best:
    # Select the best candidate for each event, by lowest tag_dtf_chi2. `keep` is
    # the eligibility mask, so each event keeps the best candidate that survived
    # the selection; events with none left simply contribute nothing.
    best = cut_utils.best_candidate_mask(tree['tag_dtf_chi2'], keep, method='min')
    tag_m = ak.to_numpy(ak.flatten(tree['tag_dtf_m'][best]))
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
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(12, 9))
if args.full_range: hist_range = (400, 1200)
else: hist_range = (450, 600)
plt.hist(tag_m, bins=100, range=hist_range, histtype='step', color='black', label='Candidates')
plt.axvline(x=547.86, color='tab:red', linestyle='--', label='Eta')
if args.full_range:
    plt.axvline(x=770., color='tab:orange', linestyle='--', label='Rho')
    plt.axvline(x=782.65, color='tab:pink', linestyle='--', label='Omega')
    plt.axvline(x=957.78, color='tab:purple', linestyle='--', label='Eta Prime')
    plt.axvline(x=1019.46, color='tab:brown', linestyle='--', label='Phi')
# plt.xlim(300, 700)
plt.xlabel('Mass [MeV]')
plt.ylabel('Candidates')
plt.title('Mass Distribution of Candidates')
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.savefig(f'{args.output}', dpi=300)
print(f'Saved plot to {args.output}')
