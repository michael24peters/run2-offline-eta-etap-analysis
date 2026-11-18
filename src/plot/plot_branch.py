"""
plot_branch.py

Rough plot of a branch from a ROOT file, using uproot and matplotlib. Useful for quick checks of
distributions.

Usage:
    python src/plot/plot_branch.py INFILE BRANCH [OUTPUT_PNG] [--n-evts N]
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
import cut_utils
import plot_style
import plot_utils


def plot_branch(path, branch_name, n_evts=None, output='branch_plot.png'):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'[INFO] Output directory created: {output_dir}')

    # Read the branch from the first n_evts events (all if None), flattened to one value per entry
    print(f'[INFO] Loading branch {branch_name}...')
    tree = cut_utils.read_branches(path, [branch_name], entry_stop=n_evts)
    branch_data = ak.to_numpy(ak.flatten(tree[branch_name], axis=None))
    print(f'[INFO] Loaded {len(branch_data)} entries of {branch_name} from {len(tree)} events.')
    print('[INFO] Plotting histogram...')
    # Provide warning if the branch_data is probably too large to plot
    if len(branch_data) > 1e6:
        print(f'[WARNING] Branch data has {len(branch_data)} entries, which '
              f'may take a long time to plot.')
    # Apply font and style
    plot_utils.apply_font()
    fig, ax = plt.subplots()
    # Look only at 0.1-99.9 percentiles so outliers don't stretch the axis
    x_range = np.percentile(branch_data, [0.1, 99.9])
    counts, _, _ = ax.hist(branch_data, bins=150, range=x_range, histtype='step',
                           color=plot_style.COLORS['data'])
    ax.set_xlabel(branch_name)
    # The branch's unit is unknown, so the y label has no bin width
    entries = ('Candidates' if branch_name.startswith('tag_')
               else 'Daughters' if branch_name.startswith('prt_') else 'Entries')
    ax.set_ylabel(entries)
    plot_style.style_axes(ax)
    sample_box = plot_style.add_stats_box(ax, '', sample=plot_style.sample_label(path))
    plot_style.reserve_top_margin(ax, max(counts.max(), 1), [sample_box])
    print('[INFO] Saving plot...')
    plot_style.save_plot(fig, output)
    plt.close(fig)

# Main method to run this directly as a script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a branch from a ROOT file')
    parser.add_argument('input', help='Input ROOT file, or a quoted glob of them')
    parser.add_argument('branch_name', help='Branch name to plot')
    parser.add_argument('--n-evts', '-n', type=int, default=None,
                            help='Number of events to plot (default: all)')
    parser.add_argument('output', nargs='?', default='branch_plot.png',
                    help='Output PNG file (default: branch_plot.png)')
    args = parser.parse_args()
    print(f'[INFO] Reading from {args.input}, plotting branch {args.branch_name}, writing to {args.output}.')
    plot_branch(args.input, args.branch_name, args.n_evts, args.output)
