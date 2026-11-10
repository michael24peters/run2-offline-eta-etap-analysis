"""
plot_pnn_mu.py

ProbNNmu of the muon daughters with no cut: data (every candidate) and truth-matched signal MC,
each normalised to unit area, with selection.PNN_MU_MIN marked.

    pnn_mu.png       one entry per muon daughter
    pnn_mu_min.png   the lowest ProbNNmu of each candidate's muons, which selection.PNN_MU_MIN cuts
    *_log.png        the same with a log y axis

The fraction of data and MC above a few thresholds is printed for each.

Usage:
    python src/plot/plot_pnn_mu.py DATA MC --truth-match PID [PID ...] --outdir OUTDIR
    e.g. python src/plot/plot_pnn_mu.py 'ntuple/lhcb/eta2mumumumu/20261109/mag*/*.root' \
             'ntuple/mc/eta2mumumumu/signal_20261109/mag*/*.root' --truth-match -13 -13 13 13 \
             --outdir out/eta2mumumumu/turbo/20261110/pnn_mu
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import bkg_utils
import cut_utils
import plot_style
import plot_utils
import selection

MASS_BRANCH = 'tag_dtf_m'               # the branch defining the candidates
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)  # printed fractions above these
NBINS = 50                              # bins over [0, 1]


def pnn_values(tree, keep):
    """
    (ProbNNmu per muon daughter, lowest ProbNNmu of the muons per candidate) of the candidates
    where `keep`, as flat numpy arrays.
    """
    is_muon = abs(tree['prt_pid']) == selection.MUON_PID
    daughter_keep = cut_utils.follow(tree, keep, 'prt_idx_mom') & is_muon
    per_muon = ak.to_numpy(ak.flatten(tree['prt_pnn_mu'][daughter_keep]))
    # A non-muon daughter counts as 1, so the minimum is over the muons
    muon_pnn = ak.where(is_muon, tree['prt_pnn_mu'], 1.0)
    lowest = cut_utils.reduce(tree, muon_pnn, 'prt_idx_mom', how='min', target_ref=MASS_BRANCH)
    per_candidate = ak.to_numpy(ak.flatten(lowest[keep]))
    return per_muon, per_candidate


def read_tree(path, truth_match=None):
    """The branches pnn_values reads from `path` (and the truth-match branches), malformed events
    dropped."""
    branches = ([MASS_BRANCH, 'prt_idx_mom', 'prt_pid', 'prt_pnn_mu']
                + (selection.TRUTH_BRANCHES if truth_match else []))
    tree, _ = cut_utils.drop_malformed(cut_utils.read_branches(path, branches),
                                       tag_ref=MASS_BRANCH)
    return tree


def plot(data, mc, xlabel, path):
    """`data` and `mc` normalised, PNN_MU_MIN marked; `path` and `path` with _log before .png."""
    edges = np.linspace(0.0, 1.0, NBINS + 1)
    for logy, suffix in ((False, ''), (True, '_log')):
        fig, ax = plt.subplots()
        ax.hist(data, bins=edges, density=True, histtype='step', linewidth=1.5,
                color=plot_style.COLORS['data'], label=f'data ({len(data)})')
        ax.hist(mc, bins=edges, density=True, histtype='step', linewidth=1.5,
                color=plot_style.COLORS['fit'], label=f'truth-matched MC ({len(mc)})')
        ax.axvline(selection.PNN_MU_MIN, color=plot_style.COLORS['reference'], linestyle='--')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Normalised entries')
        plot_style.style_axes(ax)
        ax.legend(frameon=False, loc='upper center')
        plot_style.save_plot(fig, path.replace('.png', f'{suffix}.png'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='ProbNNmu of the muon daughters in data and truth-matched MC, no cut',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('data', help='Data ROOT file, or a quoted glob of them')
    parser.add_argument('mc', help='Signal MC ROOT file, or a quoted glob of them')
    parser.add_argument('--truth-match', nargs='+', type=int, required=True, metavar='PID',
                        help='MC: keep candidates truth-matched to eta -> these daughter PIDs')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data_tree = read_tree(args.data)
    data_muon, data_candidate = pnn_values(data_tree,
                                           ak.ones_like(data_tree[MASS_BRANCH], dtype=bool))
    mc_tree = read_tree(args.mc, args.truth_match)
    truth = bkg_utils.signal_candidate_mask(mc_tree, selection.MOTHER_PID, args.truth_match,
                                            target_ref=MASS_BRANCH)
    mc_muon, mc_candidate = pnn_values(mc_tree, truth)

    plot_utils.apply_font()
    plot(data_muon, mc_muon, 'ProbNNmu (per muon)', os.path.join(args.outdir, 'pnn_mu.png'))
    plot(data_candidate, mc_candidate, 'lowest ProbNNmu of the muons',
         os.path.join(args.outdir, 'pnn_mu_min.png'))
    for label, data, mc in (('per muon', data_muon, mc_muon),
                            ('per candidate (lowest)', data_candidate, mc_candidate)):
        for threshold in THRESHOLDS:
            print(f'[INFO] {label:24s} > {threshold}: data {np.mean(data > threshold):.3f}, '
                  f'MC {np.mean(mc > threshold):.3f}', file=sys.stderr)


if __name__ == '__main__':
    main()
