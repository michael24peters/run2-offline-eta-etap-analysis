"""
Overlay the mass distribution under each selection variant on one plot, one
colour per variant, to show what each cut removes.

The variants are nested subsets of one another (no TOS + loose PID contains
TOS + loose PID contains TOS + tight PID), so they are overlaid rather than
stacked: stacking nested samples would count the same candidate several times
and the total would mean nothing. Each variant is drawn as a low-opacity fill
plus a full-opacity outline, matching plot_mass_compare.py.

Usage:
    python plot_mass_variants.py input.root [output.png] [--full-range] [--log]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

# Add the "utils" directory to the Python path so we can import the project modules.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils

# (label, use_tos, pnn_mu_min, colour). Ordered widest selection first so the
# tighter ones draw on top and stay visible.
VARIANTS = [
    ('All, pnn_mu > 0.4',   False, 0.4,  'tab:orange'),
    ('TOS, pnn_mu > 0.4',   True,  0.4,  'tab:green'),
    ('All, pnn_mu > 0.95',  False, 0.95, 'tab:blue'),
    ('TOS, pnn_mu > 0.95',  True,  0.95, 'tab:red'),
]

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

# IO
parser = argparse.ArgumentParser(
    description='Overlay mass distributions for each selection variant')
parser.add_argument('input', help='Input ROOT file')
parser.add_argument('output', nargs='?', default='mass_plot_variants.png',
                    help='Output PNG file (default: mass_plot_variants.png)')
parser.add_argument('--full-range', action='store_true', default=False,
                    help='Plot full mass range (400-1200 MeV) instead of zoomed-in range (450-600 MeV) (default: False)')
parser.add_argument('--log', action='store_true', default=False,
                    help='Log y axis, so the tighter selections stay readable next to the loosest')
args = parser.parse_args()
print(f'Reading from {args.input} and writing to {args.output}.')

# Read once, then apply each variant's mask to the same tree.
tree = cut_utils.read_branches(
    args.input,
    ['tag_dtf_m', 'tag_dtf_chi2', 'prt_idx_mom'] + DAUGHTER_FIELDS + TOS_BRANCHES)
tree, _ = cut_utils.drop_malformed(tree)

# The TOS mask doesn't depend on the variant, so build it once.
tos, _ = cut_utils.tos_mask(tree, TOS_LINES)

samples = []
for label, use_tos, pnn_mu_min, colour in VARIANTS:
    # "passes OR isn't a muon", reduced onto the candidate, so a candidate is cut
    # only when one of its own muon daughters fails.
    muon_ok = ((tree['prt_pnn_mu'] > pnn_mu_min)
               | (abs(tree['prt_pid']) != MUON_PID))
    keep = cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all')
    if use_tos: keep = keep & tos
    # `keep` is the eligibility mask, so each event keeps the best candidate that
    # survived THIS variant's selection; events with none left contribute nothing.
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', keep)
    samples.append((label, colour, cut_utils.values(tree, 'tag_dtf_m', best)))

print(f"{'─' * 56}")
for label, _, sample in samples:
    print(f'{label:24s} {len(sample):>12,d} best candidates')
print(f"{'─' * 56}")

# Plot
print('Plotting histogram...')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(12, 9))

bins = 200
if args.full_range: hist_range = (400, 1200)
else: hist_range = (450, 600)

# Each variant: low-opacity fill for area, full-opacity outline for its edge.
for label, colour, values in samples:
    plt.hist(values, bins=bins, range=hist_range, histtype='stepfilled',
             color=colour, alpha=0.25, label=label)
    plt.hist(values, bins=bins, range=hist_range, histtype='step',
             color=colour, alpha=1.0, linewidth=1.5)

plt.axvline(x=547.86, color='black', linestyle=':', label='Eta')

if args.full_range:
    plt.axvline(x=770., color='tab:orange', linestyle=':', label='Rho')
    plt.axvline(x=782.65, color='tab:pink', linestyle=':', label='Omega')
    plt.axvline(x=957.78, color='tab:purple', linestyle=':', label='Eta Prime')
    plt.axvline(x=1019.46, color='tab:brown', linestyle=':', label='Phi')

if args.log: plt.yscale('log')

plt.xlabel('Mass [MeV]')
plt.ylabel('Candidates')
plt.title('Mass Distribution by Selection')
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.savefig(f'{args.output}', dpi=300)
print(f'Saved plot to {args.output}')
