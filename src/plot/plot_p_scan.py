"""
plot_p_scan.py

The muon momentum cut step of plot_cut_steps.py for a range of thresholds,
with no fit:

    mass_p_<NN>_gev.png   the mass after the cuts before the p cut in
                          plot_cut_steps.cut_steps ([truth match], TOS per
                          trigger level, muon pT), and after p(muon) > NN GeV
                          on top of them, for NN in P_MIN_GEV

The cuts after p (eta, ProbNNmu, ghost, IP chi2, IsMuonTight, random candidate,
fit window) are not applied. Every muon daughter must pass the threshold; a
non-muon daughter passes.

Usage:
    python src/plot/plot_p_scan.py INFILE --outdir OUTDIR
        [--mass tag_dtf_m|tag_m] [--truth-match PID [PID ...]] [--max-events N]
        [--xmin 398] [--xmax 1108] [--nbins 142] [--logy]
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
from plot_cut_steps import MUON_PID, cut_steps, plot_pair, read_tree

P_MIN_GEV = range(0, 21)   # p(muon) thresholds [GeV]


def mask_before_p_cut(tree, mass_branch, truth_match=None):
    """(label, per-candidate mask) of the cut_steps step just before the p(muon) cut."""
    steps = cut_steps(tree, mass_branch, (-np.inf, np.inf), truth_match)
    labels = [label for label, _ in steps]
    p_step = next(i for i, label in enumerate(labels) if label.startswith('p(muon)'))
    return steps[p_step - 1]


def main():
    parser = argparse.ArgumentParser(
        description='Mass before and after the muon p cut, for each threshold in P_MIN_GEV',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--mass', choices=['tag_dtf_m', 'tag_m'], default='tag_dtf_m',
                        help='Mass branch (default: tag_dtf_m, as driver_mass_fit.py)')
    parser.add_argument('--truth-match', nargs='+', type=int, default=None, metavar='PID',
                        help='Signal MC: first cut is the truth match to eta -> these PIDs')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--xmin', type=float, default=398.0, help='Lower mass bound [MeV]')
    parser.add_argument('--xmax', type=float, default=1108.0, help='Upper mass bound [MeV]')
    parser.add_argument('--nbins', type=int, default=142, help='Mass histogram bins')
    parser.add_argument('--logy', action='store_true', help='Log y axis')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tree, drop_report = read_tree(args.infile, args.mass, args.truth_match, args.max_events)
    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}",
          file=sys.stderr)

    before_label, before = mask_before_p_cut(tree, args.mass, args.truth_match)
    print(f'[INFO]   {"up to " + before_label:32s} {int(ak.sum(before)):9d}', file=sys.stderr)
    p_muon = kinematics.p(tree['prt_px'], tree['prt_py'], tree['prt_pz'])
    not_muon = abs(tree['prt_pid']) != MUON_PID
    before_masses = cut_utils.values(tree, args.mass, before)

    plot_utils.apply_font()
    edges = np.linspace(args.xmin, args.xmax, args.nbins + 1)
    for p_min in P_MIN_GEV:
        muon_ok = (p_muon > p_min * 1000.0) | not_muon
        after = before & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                          target_ref=args.mass)
        label = f'p(muon) > {p_min:g} GeV'
        print(f'[INFO]   {"+ " + label:32s} {int(ak.sum(after)):9d}', file=sys.stderr)

        fig, ax = plt.subplots()
        plot_pair(ax, edges, before_masses, cut_utils.values(tree, args.mass, after),
                  ('previous cuts', f'+ {label}'), logy=args.logy)
        ax.axvline(kinematics.M_ETA, color=plot_style.COLORS['reference'], linestyle='--')
        ax.set_xlabel(f'{args.mass} [MeV]')
        ax.set_ylabel('Candidates')
        plot_style.save_plot(fig, os.path.join(args.outdir, f'mass_p_{p_min:02d}_gev.png'))
        plt.close(fig)


if __name__ == '__main__':
    main()
