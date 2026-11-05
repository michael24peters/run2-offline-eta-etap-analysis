"""
plot_ip_chi2.py

Muon prt_ip_chi2 of every ntuple candidate vs the fit sample (the last step of
plot_cut_steps.cut_steps: [truth match], TOS, muon cuts, random candidate, fit
window [--fit-xmin, --fit-xmax)), with no fit:

    ip_chi2[_logy].png        linear x up to --ip-chi2-max (linear, log y)
    ip_chi2_logx[_logy].png   the same on a log x axis over the full range
                              (whole decades around every value > 0)

prt_ip_chi2 is the IP chi2 of the daughter to its related PV
(IDistanceCalculator::distance(particle, PV, ip, ipChi2) in ../ana/src/ntuple.py),
not the track chi2. -1 is stored when there is no PV; those entries are
counted and printed, and fall outside the plotted range.

Usage:
    python src/plot/plot_ip_chi2.py INFILE --outdir OUTDIR
        [--mass tag_dtf_m|tag_m] [--truth-match PID [PID ...]] [--max-events N]
        [--fit-xmin 500] [--fit-xmax 600] [--ip-chi2-max 30]
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
import plot_style
import plot_utils
from plot_cut_steps import MU_IP_CHI2_MAX, MUON_PID, cut_steps, plot_pair, read_tree


def muon_ip_chi2(tree, mask):
    """prt_ip_chi2 of every muon daughter of the candidates in `mask`."""
    daughter = cut_utils.follow(tree, mask, 'prt_idx_mom') & (abs(tree['prt_pid']) == MUON_PID)
    return ak.to_numpy(ak.flatten(tree['prt_ip_chi2'][daughter]))


def main():
    parser = argparse.ArgumentParser(
        description='Muon IP chi2 of the ntuple vs the fit sample',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--mass', choices=['tag_dtf_m', 'tag_m'], default='tag_dtf_m',
                        help='Mass branch of the fit window (default: tag_dtf_m)')
    parser.add_argument('--truth-match', nargs='+', type=int, default=None, metavar='PID',
                        help='Signal MC: first cut is the truth match to eta -> these PIDs')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--fit-xmin', type=float, default=500.0,
                        help='Lower edge of the fit window [MeV] (driver_mass_fit.py --xmin)')
    parser.add_argument('--fit-xmax', type=float, default=600.0,
                        help='Upper edge of the fit window [MeV] (driver_mass_fit.py --xmax)')
    parser.add_argument('--ip-chi2-max', type=float, default=30.0,
                        help='Upper bound of the linear IP chi2 axis (default: 30)')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tree, drop_report = read_tree(args.infile, args.mass, args.truth_match, args.max_events)
    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}",
          file=sys.stderr)
    steps = cut_steps(tree, args.mass, (args.fit_xmin, args.fit_xmax), args.truth_match)

    ip_before, ip_after = muon_ip_chi2(tree, steps[0][1]), muon_ip_chi2(tree, steps[-1][1])
    for name, values in (('ntuple', ip_before), ('fit sample', ip_after)):
        print(f'[INFO] muon ip_chi2 ({name}): {len(values)} muons, min {values.min():.3f}, '
              f'{int(np.sum(values < 0))} with no PV (-1), '
              f'{int(np.sum(values > args.ip_chi2_max))} above {args.ip_chi2_max:g}',
              file=sys.stderr)

    # Log x: whole decades around every positive value, so nothing is cut off.
    positive = np.concatenate([ip_before, ip_after])
    positive = positive[positive > 0]
    log_edges = np.logspace(np.floor(np.log10(positive.min())),
                            np.ceil(np.log10(positive.max())), 141)
    lin_edges = np.linspace(0.0, args.ip_chi2_max, 121)

    plot_utils.apply_font()
    for logx, logy, name in ((False, False, 'ip_chi2.png'), (False, True, 'ip_chi2_logy.png'),
                             (True, False, 'ip_chi2_logx.png'),
                             (True, True, 'ip_chi2_logx_logy.png')):
        fig, ax = plt.subplots()
        plot_pair(ax, log_edges if logx else lin_edges, ip_before, ip_after,
                  ('ntuple', 'fit sample'), logy=logy)
        if logx:
            ax.set_xscale('log')
            ax.set_xlim(log_edges[0], log_edges[-1])
        ax.axvline(MU_IP_CHI2_MAX, color=plot_style.COLORS['reference'], linestyle='--')
        ax.set_xlabel('prt_ip_chi2 (muons)')
        ax.set_ylabel('Muons')
        plot_style.save_plot(fig, os.path.join(args.outdir, name))
        plt.close(fig)


if __name__ == '__main__':
    main()
