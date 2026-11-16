"""
plot_cut_steps.py

Mass before and after each offline selection with no fit, to check the effect of each cut:

    mass_<NN>_<cut>.png       one figure per cut: the mass after every earlier
                              cut, and after this cut too. Figure 01 is the
                              ntuple vs the first cut, figure 02 the first cut
                              vs the first two, and so on.
    mass_<NN>_<cut>_log.png   the same with a log y axis.

The cuts are cumulative, in the order of driver_mass_fit.py: [MC match], TOS per trigger level,
each muon cut, random candidate, then the fit window [--fit-xmin, --fit-xmax) (driver_mass_fit.py's
--xmin/--xmax defaults), so the last step is exactly what the mass fit sees. The cuts are
selection.cut_steps with selection.TOS_LINES and MUON_THRESHOLDS, or selection.TOS_LINES_4MU and
MUON_THRESHOLDS_4MU for an input path with an eta2mumumumu segment, as in
driver_mass_fit.py. The input is read, selected and histogrammed in chunks of selection.STEP_SIZE
(cut_utils.iterate_branches), the random candidate of chunk i picked with seed selection.SEED + i,
as in the drivers.

Usage:
    python src/plot/plot_cut_steps.py INFILE --outdir OUTDIR
        [--mass tag_dtf_m|tag_m] [--mc-match PID [PID ...]] [--max-events N] [--pnn-mu-min X]
        [--fit-xmin 500] [--fit-xmax 600]
        [--xmin 398] [--xmax 1108] [--nbins 142]
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils
import kinematics
import plot_style
import plot_utils
import selection

def tree_branches(mass_branch, tos_lines, mc_match=None):
    """The branches cut_steps reads with `tos_lines`."""
    return ([mass_branch, 'prt_idx_mom'] + selection.MUON_CUT_BRANCHES
            + [line for lines in tos_lines.values() for line in lines]
            + (selection.MC_BRANCHES if mc_match else []))


def cut_steps(tree, mass_branch, tos_lines, muon_thresholds, fit_window, seed,
              mc_match=None):
    """
    selection.cut_steps with `tos_lines` and `muon_thresholds`, then the random candidate
    (`seed`) and the fit window (lo, hi) [MeV] on mass_branch.

    Returns [(label, mask), ...]: the per-candidate mask after each step, starting with ('all
    candidates', every candidate).
    """
    steps, _ = selection.cut_steps(tree, mass_branch, tos_lines, mc_pids=mc_match,
                                   muon_thresholds=muon_thresholds)

    keep = cut_utils.random_candidate_mask(steps[-1][1], seed)
    steps.append(('random candidate', keep))

    # The fit's histogram range: what the mass fit actually sees.
    lo, hi = fit_window
    keep = keep & (tree[mass_branch] >= lo) & (tree[mass_branch] < hi)
    steps.append((f'{lo:g} <= {mass_branch} < {hi:g}', keep))
    return steps


def slug(label):
    """File-name form of a step label."""
    return re.sub(r'[^a-z0-9]+', '_', re.sub(r'[$\\{}]', '', label.lower())).strip('_')


def plot_pair(ax, edges, before, after, labels, logy=False):
    """Step histograms of `before` (black) and `after` (blue) on `ax`, each (counts per bin of
    `edges`, total candidates), with room above the tallest bin for the legend."""
    C = plot_style.COLORS
    top = 0
    for (counts, total), label, color in ((before, labels[0], C['data']),
                                          (after, labels[1], C['fit'])):
        ax.stairs(counts, edges, linewidth=1.5, color=color, label=f'{label} ({total})')
        top = max(top, counts.max())
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(0.5, top * 30)
    else:
        ax.set_ylim(0, top * 1.25)
    plot_style.style_axes(ax)
    ax.legend(frameon=False, loc='upper right')


def main():
    parser = argparse.ArgumentParser(
        description='Mass before and after each offline cut',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--mass', choices=['tag_dtf_m', 'tag_m'], default='tag_dtf_m',
                        help='Mass branch (default: tag_dtf_m, as driver_mass_fit.py)')
    parser.add_argument('--mc-match', nargs='+', type=int, default=None, metavar='PID',
                        help='Signal MC: first cut is the MC match to eta -> these PIDs')
    parser.add_argument('--pnn-mu-min', type=float, default=None,
                        help="ProbNNmu threshold in place of the channel's "
                             "(selection.MUON_THRESHOLDS or MUON_THRESHOLDS_4MU)")
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--fit-xmin', type=float, default=500.0,
                        help='Lower edge of the fit window [MeV] (driver_mass_fit.py --xmin)')
    parser.add_argument('--fit-xmax', type=float, default=600.0,
                        help='Upper edge of the fit window [MeV] (driver_mass_fit.py --xmax)')
    parser.add_argument('--xmin', type=float, default=398.0, help='Lower mass bound [MeV]')
    parser.add_argument('--xmax', type=float, default=1108.0, help='Upper mass bound [MeV]')
    parser.add_argument('--nbins', type=int, default=142, help='Mass histogram bins')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    is_4mu = 'eta2mumumumu' in args.infile.split('/')
    tos_lines = selection.TOS_LINES_4MU if is_4mu else selection.TOS_LINES
    muon_thresholds = selection.MUON_THRESHOLDS_4MU if is_4mu else selection.MUON_THRESHOLDS
    if args.pnn_mu_min is not None:
        muon_thresholds = {**muon_thresholds, 'pnn_mu_min': args.pnn_mu_min}

    # --- Read, select and histogram one chunk of selection.STEP_SIZE at a time ---
    # Per step: the mass histogram and the candidate count, summed over the chunks. Chunk i picks
    # the random candidate with seed selection.SEED + i, as the drivers.
    edges = np.linspace(args.xmin, args.xmax, args.nbins + 1)
    drop_report, labels, histograms, totals = {}, [], [], []
    chunks = cut_utils.iterate_branches(args.infile,
                                        tree_branches(args.mass, tos_lines, args.mc_match),
                                        selection.STEP_SIZE, entry_stop=args.max_events)
    for i, tree in enumerate(chunks):
        tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
        tree, chunk_drop_report = cut_utils.drop_malformed(tree, tag_ref=args.mass)
        drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
        steps = cut_steps(tree, args.mass, tos_lines, muon_thresholds,
                          (args.fit_xmin, args.fit_xmax),
                          selection.SEED + i, args.mc_match)
        if not labels:
            labels = [label for label, _ in steps]
            histograms = [np.zeros(args.nbins) for _ in steps]
            totals = [0 for _ in steps]
        for step, (_, mask) in enumerate(steps):
            masses = cut_utils.values(tree, args.mass, mask)
            histograms[step] += np.histogram(masses, bins=edges)[0]
            totals[step] += len(masses)
        print(f"[INFO] chunk {i}: {drop_report['read']} events read", file=sys.stderr)

    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}",
          file=sys.stderr)
    for label, total in zip(labels, totals):
        print(f'[INFO]   {label:32s} {total:9d}', file=sys.stderr)

    plot_utils.apply_font()
    xlabel = f'{args.mass} [MeV]'

    # --- Mass: step i-1 vs step i ---
    for i in range(1, len(labels)):
        label_after = labels[i]
        for logy, suffix in ((False, ''), (True, '_log')):
            fig, ax = plt.subplots()
            plot_pair(ax, edges, (histograms[i - 1], totals[i - 1]), (histograms[i], totals[i]),
                      (labels[0] if i == 1 else 'previous cuts', f'+ {label_after}'),
                      logy=logy)
            ax.axvline(kinematics.M_ETA, color=plot_style.COLORS['reference'], linestyle='--')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Candidates')
            name = f'mass_{i:02d}_{slug(label_after)}{suffix}.png'
            plot_style.save_plot(fig, os.path.join(args.outdir, name))
            plt.close(fig)


if __name__ == '__main__':
    main()
