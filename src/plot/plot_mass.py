"""
plot_mass.py

Mass of every candidate in a ROOT file, with no selection.

Usage:
    python src/plot/plot_mass.py INFILE [--output mass.png] [--mass tag_dtf_m]
        [--xmin 398] [--xmax 1108] [--nbins 142]
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils
import kinematics
import plot_style
import plot_utils


def main():
    parser = argparse.ArgumentParser(
        description='Mass of every candidate in a ROOT file',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--output', default='mass.png', help='Output PNG (default: mass.png)')
    parser.add_argument('--mass', default='tag_dtf_m', help='Mass branch (default: tag_dtf_m)')
    parser.add_argument('--xmin', type=float, default=398.0, help='Lower mass bound [MeV]')
    parser.add_argument('--xmax', type=float, default=1108.0, help='Upper mass bound [MeV]')
    parser.add_argument('--nbins', type=int, default=142, help='Number of bins')
    args = parser.parse_args()

    # Read: one mass per candidate, every candidate of every event
    tree = cut_utils.read_branches(args.infile, [args.mass])
    masses = ak.to_numpy(ak.flatten(tree[args.mass]))
    print(f'[INFO] {len(masses)} candidates in {len(tree)} events', file=sys.stderr)

    # Plot
    plot_utils.apply_font()
    fig, ax = plt.subplots()
    ax.hist(masses, bins=np.linspace(args.xmin, args.xmax, args.nbins + 1), histtype='step',
            linewidth=1.5, color=plot_style.COLORS['data'])
    ax.axvline(kinematics.M_ETA, color=plot_style.COLORS['reference'], linestyle='--')
    ax.set_xlabel(plot_style.decay_label({'input_file': args.infile}))
    ax.set_ylabel('Candidates')
    plot_style.style_axes(ax)
    plot_style.save_plot(fig, args.output)


if __name__ == '__main__':
    main()
