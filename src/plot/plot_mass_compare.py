"""
Script to compare "all candidates" vs "best candidate per event" mass plots on
one overlaid plot. In this script the "best candidate" is defined as the 
candidate with the lowest tag_dtf_chi2 value per event, but this could be
changed since the best candidate method is more generic than this.

Usage:
    python plot_mass_compare.py input.root [output.png] [--full-range]
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt

# Add the "plot" and "fit" directories to the Python path so we can import the
# project modules.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import cut_utils
import kinematics
import plot_style
import plot_utils

# IO
parser = argparse.ArgumentParser(description='Compare all-candidate vs best-candidate mass distributions')
parser.add_argument('input',
                    help='Input ROOT file')
parser.add_argument('output', nargs='?', default='mass_plot_compare.png',
                    help='Output PNG file (default: mass_plot_compare.png)')
parser.add_argument('--full-range', action='store_true', default=False,
                    help='Plot full mass range (400-1200 MeV) instead of zoomed-in range (450-600 MeV) (default: False)')
args = parser.parse_args()
print(f'Reading from {args.input} and writing to {args.output}.')

# Get mass info as awkward arrays
tree = cut_utils.read_branches(args.input, ['tag_dtf_m', 'tag_dtf_chi2'])

# All candidates, flattened
tag_m_all = ak.to_numpy(ak.flatten(tree['tag_dtf_m']))

# Best candidate per event (lowest tag_dtf_chi2). This script applies no cuts --
# it exists to compare "all candidates" against "best candidate" -- so every
# candidate is eligible to win. Events with no candidates contribute nothing.
eligible = ak.ones_like(tree['tag_dtf_m'], dtype=bool)
best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', eligible)
tag_m_best = cut_utils.values(tree, 'tag_dtf_m', best)

print(f"{'─' * 48}")
print(f'Candidates plotted (all):  {len(tag_m_all)}')
print(f'Candidates plotted (best): {len(tag_m_best)}')
print(f'Difference (candidates dropped by best selection): {len(tag_m_all) - len(tag_m_best)}')
print(f"{'─' * 48}")

# Plot
print('Plotting histogram...')
C = plot_style.COLORS
plot_utils.apply_font()
plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(figsize=(12, 9))

bins = 200
if args.full_range: hist_range = (400, 1200)
else: hist_range = (450, 600)

# Each sample: low-opacity fill plus full-opacity outline, so the overlap of
# the two fills stays readable.
for values, color, label in [(tag_m_all, C['pull'], 'All candidates'),
                             (tag_m_best, C['fit'], 'Best candidate')]:
    ax.hist(values, bins=bins, range=hist_range, histtype='stepfilled',
            color=color, alpha=0.25, label=label)
    ax.hist(values, bins=bins, range=hist_range, histtype='step',
            color=color, alpha=1.0, linewidth=1.5)

ax.axvline(x=kinematics.M_ETA, color=C['reference'], linestyle='--', label=r'$\eta$')
if args.full_range:
    for label, resonance_mass, linestyle in plot_style.RESONANCES:
        ax.axvline(x=resonance_mass, color=C['reference'], linestyle=linestyle, label=label)

ax.set_xlabel('Mass [MeV]')
ax.set_ylabel('Candidates')
plot_style.style_axes(ax)
ax.legend(frameon=False)
plot_style.save_plot(fig, args.output)
