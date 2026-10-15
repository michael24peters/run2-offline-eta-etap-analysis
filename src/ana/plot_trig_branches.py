"""
plot_trig_branches.py

Quick diagnostic: plot all trigger branches as histograms to PNG.

Usage:
    python plot_trig_branches.py input.root
"""

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot trigger branches from a ROOT file.')
parser.add_argument('infile', help='Input ROOT file')

args = parser.parse_args()

with uproot.open(args.infile) as f:
    tree = f['tree']
    # Extract all branches that begin with `tag_l0_`, `tag_hlt1_`, or `tag_hlt2_`
    # from the tree and plot them as histograms.
    branches = []
    for name in tree.keys():
        if name.startswith('tag_l0_') or name.startswith('tag_hlt1_') or name.startswith('tag_hlt2_'):
            branches.append(name)
    # Plot each branch and save to PNG.
    for name in branches:
        flat = ak.to_numpy(ak.flatten(tree[name].array(library='ak')))
        fig, ax = plt.subplots()
        ax.hist(flat, bins=50)
        ax.set_title(name)
        ax.set_xlabel('value')
        ax.set_ylabel('candidates')
        fig.savefig(f'{name}.png', dpi=100)
        plt.close(fig)
        print(f'{name}: min={flat.min():.3f} max={flat.max():.3f} mean={flat.mean():.4f} nonzero={np.sum(flat != 0)}')
