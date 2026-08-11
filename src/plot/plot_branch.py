'''
plot_branch.py

Rough plot of a branch from a ROOT file, using uproot and matplotlib. Useful for
quick checks of distributions.

Usage:
    python plot_branch.py input.root branch_name [output.png]
'''

import argparse
import sys
import os
import uproot
import matplotlib.pyplot as plt
import numpy as np

def plot_branch(input, branch_name, n_evts=-1, output='branch_plot.png'):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'[INFO] Output directory created: {output_dir}')

    # Check if input is a ROOT file or just an array of numbers
    if input.endswith('.root'):
        # Open the ROOT file and get the branch data from the tree
        print(f'[INFO] Opening ROOT file...')
        tree = uproot.open(input)['tree']
        # Load the branch as a numpy array
        print(f'[INFO] Loading branch {branch_name}...')
        branch_data = tree[branch_name].array(library='np')
        print(f'[INFO] Loaded {len(branch_data)} entries for branch {branch_name}.')
    else:
        # Assume input is just an array of numbers, which needs to be converted
        # to numpy if not already
        branch_data = input if isinstance(input, np.ndarray) else np.array(input)
        print(f'[INFO] Loaded {len(branch_data)} entries from {input}.')
    if n_evts > 0:
        branch_data = branch_data[:n_evts]
        print(f'[INFO] Using first {n_evts} entries for plotting.')
    # Convert branch_data from array of arrays to a flat array
    branch_data = np.concatenate(branch_data)
    print(f'[INFO] Plotting histogram...')
    # Provide warning if the branch_data is probably too large to plot
    if len(branch_data) > 1e6:
        print(f'[WARNING] Branch data has {len(branch_data)} entries, which '
              f'may be too large to plot.')
    plt.hist(branch_data, bins=100, color='tab:blue')
    plt.xlabel(branch_name)
    plt.ylabel('Counts')
    plt.title(f'{branch_name} distribution')
    plt.grid(True, alpha=0.3)
    print(f'[INFO] Saving plot...')
    plt.savefig(output, dpi=300)
    print(f'[INFO] Saved plot to {output}.')

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
