"""
plot_muon_branch.py

A muon-daughter branch with no cut: data (every candidate) and MC-matched signal MC, each
normalised to unit area, with a cut value marked.

    <branch>.png          one entry per muon daughter
    <branch>_<how>.png    per candidate, the min or max of its muons' values: the min for a
                          lower cut every muon must pass (prt_pnn_mu > X), the max for an upper
                          cut (prt_ip_chi2 < X)
    *_log.png             the same with a log y axis

The inputs are read one chunk of selection.STEP_SIZE at a time (cut_utils.iterate_branches). The
fraction of data and MC passing a few thresholds (> for min, < for max) is printed.

Usage:
    python src/plot/plot_muon_branch.py DATA MC --mc-match PID [PID ...] --outdir OUTDIR \
        --branch prt_pnn_mu --how min [--cut 0.9] [--range 0 1] [--nbins 50]
    e.g. --branch prt_ip_chi2 --how max --cut 6 --range 0 30
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

MASS_BRANCH = 'tag_dtf_m'   # the branch defining the candidates
N_THRESHOLDS = 5            # printed passing fractions at this many values across the range


def muon_values(tree, keep, branch, how):
    """
    (`branch` per muon daughter, its `how` ('min' or 'max') over each candidate's muons) of the
    candidates where `keep`, as flat numpy arrays.
    """
    is_muon = abs(tree['prt_pid']) == selection.MUON_PID
    daughter_keep = cut_utils.follow(tree, keep, 'prt_idx_mom') & is_muon
    per_muon = ak.to_numpy(ak.flatten(tree[branch][daughter_keep]))
    # A non-muon daughter takes the value that never sets the min or max
    neutral = np.inf if how == 'min' else -np.inf
    muon_branch = ak.where(is_muon, tree[branch], neutral)
    per_candidate = cut_utils.reduce(tree, muon_branch, 'prt_idx_mom', how=how,
                                     target_ref=MASS_BRANCH)
    return per_muon, ak.to_numpy(ak.flatten(per_candidate[keep]))


def read_values(path, branch, how, mc_match=None):
    """
    The values (per muon, per candidate) of muon_values for the candidates of `path` (the
    MC-matched ones with `mc_match`), read in chunks, as float32 numpy arrays.
    """
    branches = ([MASS_BRANCH, 'prt_idx_mom', 'prt_pid', branch]
                + (selection.MC_BRANCHES if mc_match else []))
    per_muon, per_candidate = [], []
    for tree in cut_utils.iterate_branches(path, branches, selection.STEP_SIZE):
        tree, _ = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
        if mc_match:
            keep = bkg_utils.signal_candidate_mask(tree, selection.MOTHER_PID, mc_match,
                                                   target_ref=MASS_BRANCH)
        else:
            keep = ak.ones_like(tree[MASS_BRANCH], dtype=bool)
        muon_chunk, candidate_chunk = muon_values(tree, keep, branch, how)
        per_muon.append(muon_chunk.astype(np.float32))
        per_candidate.append(candidate_chunk.astype(np.float32))
    return np.concatenate(per_muon), np.concatenate(per_candidate)


def plot(data, mc, edges, cut, xlabel, path):
    """`data` and `mc` normalised over `edges`, `cut` marked; `path` and `path` with _log."""
    for logy, suffix in ((False, ''), (True, '_log')):
        fig, ax = plt.subplots()
        ax.hist(data, bins=edges, density=True, histtype='step', linewidth=1.5,
                color=plot_style.COLORS['data'], label=f'data ({len(data)})')
        ax.hist(mc, bins=edges, density=True, histtype='step', linewidth=1.5,
                color=plot_style.COLORS['fit'], label=f'MC-matched MC ({len(mc)})')
        if cut is not None:
            ax.axvline(cut, color=plot_style.COLORS['reference'], linestyle='--')
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
        description='A muon-daughter branch in data and MC-matched MC, no cut',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('data', help='Data ROOT file, or a quoted glob of them')
    parser.add_argument('mc', help='Signal MC ROOT file, or a quoted glob of them')
    parser.add_argument('--mc-match', nargs='+', type=int, required=True, metavar='PID',
                        help='MC: keep candidates MC-matched to eta -> these daughter PIDs')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--branch', required=True, help='Muon-daughter branch, e.g. prt_pnn_mu')
    parser.add_argument('--how', choices=['min', 'max'], required=True,
                        help='Per candidate: min (lower cut) or max (upper cut) of its muons')
    parser.add_argument('--cut', type=float, default=None, help='Cut value to mark')
    parser.add_argument('--range', nargs=2, type=float, default=(0.0, 1.0), metavar=('LO', 'HI'),
                        help='Histogram range (default: 0 1)')
    parser.add_argument('--nbins', type=int, default=50, help='Histogram bins (default: 50)')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    edges = np.linspace(args.range[0], args.range[1], args.nbins + 1)

    data_muon, data_candidate = read_values(args.data, args.branch, args.how)
    mc_muon, mc_candidate = read_values(args.mc, args.branch, args.how, args.mc_match)

    plot_utils.apply_font()
    plot(data_muon, mc_muon, edges, args.cut, f'{args.branch} (per muon)',
         os.path.join(args.outdir, f'{args.branch}.png'))
    plot(data_candidate, mc_candidate, edges, args.cut, f'{args.how} {args.branch} of the muons',
         os.path.join(args.outdir, f'{args.branch}_{args.how}.png'))
    thresholds = np.linspace(args.range[0], args.range[1], N_THRESHOLDS + 2)[1:-1]
    for label, data, mc in (('per muon', data_muon, mc_muon),
                            (f'per candidate ({args.how})', data_candidate, mc_candidate)):
        for threshold in thresholds:
            if args.how == 'min':
                data_pass, mc_pass, sign = np.mean(data > threshold), np.mean(mc > threshold), '>'
            else:
                data_pass, mc_pass, sign = np.mean(data < threshold), np.mean(mc < threshold), '<'
            print(f'[INFO] {label:24s} {sign} {threshold:g}: data {data_pass:.3f}, '
                  f'MC {mc_pass:.3f}', file=sys.stderr)


if __name__ == '__main__':
    main()
