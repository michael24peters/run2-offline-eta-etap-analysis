"""
plot_pulls.py

Plot the pull distribution from a fit results JSON file, as well as a normalized Gaussian
distribution for comparison, scaled to have the same area as the pull distribution. The pull is
defined as:
    pull = (y_data - y_fit) / sigma_data

Usage:
    python src/plot/plot_pulls.py RESULTS_JSON [OUTPUT_PNG]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
import plot_style
import plot_utils

# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Pull plot from json fit results')
parser.add_argument('input', help='Input JSON file')
parser.add_argument('output', nargs='?', default='out/pull_plot.png',
                    help='Output PNG file (default: out/pull_plot.png)')
args = parser.parse_args()
print(f'[INFO] Reading from {args.input} and writing to {args.output}.')

# --- Load JSON ---
with open(args.input) as f:
    data = json.load(f)

# --- Extract pull values ---
hist = data['histogram']
x = np.array(hist['bin_centers'])
pulls = np.array(hist['bin_pulls'])
n_pulls = len(pulls)
print(f'[INFO] Extracted {n_pulls} pull values from: {args.input}')

# --- Plot configuration ---
plot_utils.apply_font()
plt.rcParams['font.size'] = 12

# --- Create pull histogram ---
# Get derived values
mean = np.mean(pulls)
std = np.std(pulls, ddof=1)

# x range for the plot
xmin = min(-4.0, mean - 3.5 * std)
xmax = max( 4.0, mean + 3.5 * std)
nbins = 20  # pull histogram bins

# Histogram the pulls to get bin counts for the Gaussian amplitude seed
# Returns the counts in each bin and the edges of the bins
counts, edges = np.histogram(pulls, bins=nbins, range=(xmin, xmax))
bin_width = edges[1] - edges[0]
# Find the area under the histogram to scale the Gaussian PDF Each bar's area is count * bin_width,
# so total histogram area is sum(counts) * bin_width. A normal gaussian PDF has area 1, so
# multiplying by the histogram's total area exactly scales it to match.
area = counts.sum() * bin_width

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(
    pulls,
    bins=nbins,
    range=(xmin, xmax),
    color=plot_style.COLORS['pull'],
    alpha=0.75,
    edgecolor='white',
    linewidth=0.7,
    label=f'Pull values\n($N_{{\\mathrm{{bins}}}}={n_pulls}$)',
    zorder=2,
)

# Array of 400 evenly spaced points between xmin and xmax
x_gauss = np.linspace(xmin, xmax, 400)
# Calculates pdf for norm and scales it to match the histogram area
y_gauss = area * stats.norm.pdf(x_gauss)
ax.plot(x_gauss, y_gauss,
        label='Normal gaussian\n(scaled to hist area)',
        color=plot_style.COLORS['fit'],
        linewidth=2,
        linestyle='--',
        zorder=3)

# Plot configuration Axis labels, legend
ax.set_xlabel(r'Pull  $(y_{\mathrm{data}} - y_{\mathrm{fit}})\,/\,\sigma_{\mathrm{data}}$',
              fontsize=12)
ax.set_ylabel(plot_style.count_label('Mass bins', edges))
ax.set_xlim(xmin, xmax)
plot_style.style_axes(ax)
# Legend
legend = ax.legend(frameon=False, loc='upper right')

# --- Summary text, below the sample label of the fit's input ---
text = f'$\\mu = {mean:+.3f}$\n$\\sigma = {std:.3f}$'
sample = plot_style.sample_label(data.get('meta', {}).get('input_file', ''))
stats_box = plot_style.add_stats_box(ax, text, sample=sample)
plot_style.reserve_top_margin(ax, max(counts.max(), y_gauss.max()), [stats_box, legend])

# --- Save plot ---
os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
plot_style.save_plot(fig, args.output)

# --- Print summary ---
plot_style.print_summary(f'  N bins            : {n_pulls}\n'
                         f'  Bin width         : {bin_width:.4f}\n'
                         f'  Sample mean       : {mean:+.4f}\n'
                         f'  Sample std        : {std:.4f}',
                         title='Pull distribution summary')
