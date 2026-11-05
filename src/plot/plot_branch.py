'''
plot_branch.py

Rough plot of a branch from a ROOT file, using uproot and matplotlib. Useful for quick checks of
distributions.

Usage:
    python plot_branch.py input.root branch_name [output.png] [--n-evts N]
'''

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
import plot_style
import plot_utils


def plot_branch(input, branch_name, n_evts=-1, output='branch_plot.png'):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'[INFO] Output directory created: {output_dir}')

    # Check if input is a ROOT file or just an array of numbers
    if input.endswith('.root'):
        # Open the ROOT file and get the branch data from the tree
        print('[INFO] Opening ROOT file...')
        tree = uproot.open(input)['tree']
        # Load the branch as a numpy array
        print(f'[INFO] Loading branch {branch_name}...')
        branch_data = tree[branch_name].array(library='np')
        print(f'[INFO] Loaded {len(branch_data)} entries for branch {branch_name}.')
    else:
        # Assume input is just an array of numbers, which needs to be converted to numpy if not
        # already
        branch_data = input if isinstance(input, np.ndarray) else np.array(input)
        print(f'[INFO] Loaded {len(branch_data)} entries from {input}.')
    if n_evts > 0:
        branch_data = branch_data[:n_evts]
        print(f'[INFO] Using first {n_evts} entries for plotting.')
    # Flatten array (only if the branch is a jagged array)
    if branch_data.dtype == object:
        branch_data = np.concatenate(branch_data)
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
    ax.hist(branch_data, bins=150, range=x_range, histtype='step', color=plot_style.COLORS['data'])
    ax.set_xlabel(branch_name)
    ax.set_ylabel('Counts')
    plot_style.style_axes(ax)
    print('[INFO] Saving plot...')
    plot_style.save_plot(fig, output)
    plt.close(fig)

# Main method to run this directly as a script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a branch from a ROOT file')
    parser.add_argument('input', help='Input ROOT file or data file')
    parser.add_argument('branch_name', help='Branch name to plot')
    parser.add_argument('--n-evts', '-n', type=int, default=-1,
                            help='Number of events to plot (default: all)')
    parser.add_argument('output', nargs='?', default='branch_plot.png',
                    help='Output PNG file (default: branch_plot.png)')
    args = parser.parse_args()
    print(f'[INFO] Reading from {args.input}, plotting branch {args.branch_name}, writing to {args.output}.')
    plot_branch(args.input, args.branch_name, args.n_evts, args.output)
