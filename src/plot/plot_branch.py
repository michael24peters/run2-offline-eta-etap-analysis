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
# Add the "utils" directory to the Python path so we can import the project modules.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))

parser = argparse.ArgumentParser(description='Plot a branch from a ROOT file')
parser.add_argument('input', help='Input ROOT file')
parser.add_argument('branch', help='Branch name to plot')
parser.add_argument('output', nargs='?', default='branch_plot.png',
                    help='Output PNG file (default: branch_plot.png)')
args = parser.parse_args()
print(f'[INFO] Reading from {args.input}, plotting branch {args.branch}, writing to {args.output}.')

# Create output directory if it doesn't exist
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f'[INFO] Output directory created: {output_dir}')

print(f'[INFO] Opening ROOT file...')
tree = uproot.open(args.input)['tree']
# Load the branch as a numpy array
print(f'[INFO] Loading branch {args.branch}...')
branch_data = tree[args.branch].array(library='np')
print(f'[INFO] Loaded {len(branch_data)} entries for branch {args.branch}.')
print(f'[INFO] Plotting histogram...')
plt.hist(branch_data, bins=100, alpha=0.7)
plt.xlabel(args.branch)
plt.ylabel('Counts')
plt.title(f'{args.branch} distribution')
print(f'[INFO] Saving plot...')
plt.savefig(args.output, dpi=300)
print(f'[INFO] Saved plot to {args.output}.')
