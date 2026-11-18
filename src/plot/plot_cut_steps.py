"""
plot_cut_steps.py

Mass before and after each offline selection with no fit, to check the effect of each cut on the
candidates in the fit window [--fit-xmin, --fit-xmax) (driver_mass_fit.py's --xmin/--xmax
defaults):

    mass_01_fit_window.png    every candidate over the full mass range [--xmin, --xmax), and those
                              in the fit window
    mass_<NN>_<cut>.png       from 02 on, the fit window only (--fit-nbins bins): the candidates
                              in the window after every earlier cut, and after this cut too.
                              Figure 02 is the window vs the window and the first cut, and so on.
    mass_<NN>_<cut>_log.png   the same with a log y axis.

The cuts are cumulative, in the order of driver_mass_fit.py: [MC match], TOS per trigger level,
each muon cut, then the random candidate, picked among the candidates that pass the cuts over the
whole mass range. The figures apply the fit window first, as a step of its own; the last figure
is the random candidates in the window. The cuts are selection.cut_steps with selection.TOS_LINES and
MUON_THRESHOLDS, or selection.TOS_LINES_4MU and MUON_THRESHOLDS_4MU for an input path with an
eta2mumumumu segment, as in driver_mass_fit.py. The input is read, selected and histogrammed in
chunks of selection.STEP_SIZE (cut_utils.iterate_branches), the random candidate of chunk i picked
with seed selection.SEED + i, as in the drivers. Every figure carries plot_style.sample_label of
INFILE.

Usage:
    python src/plot/plot_cut_steps.py INFILE --outdir OUTDIR
        [--mass tag_dtf_m|tag_m] [--mc-match PID [PID ...]] [--max-events N] [--pnn-mu-min X]
        [--fit-xmin 500] [--fit-xmax 600] [--fit-nbins 100]
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

def cut_steps(tree, mass_branch, tos_lines, muon_thresholds, seed, mc_match=None):
    """
    selection.cut_steps with `tos_lines` and `muon_thresholds`, then the random candidate
    (`seed`).

    Returns [(label, mask), ...]: the per-candidate mask after each step, starting with ('all
    candidates', every candidate).
    """
    steps, _ = selection.cut_steps(tree, mass_branch, tos_lines, mc_pids=mc_match,
                                   muon_thresholds=muon_thresholds)

    keep = cut_utils.random_candidate_mask(steps[-1][1], seed)
    steps.append(('random candidate', keep))
    return steps


def slug(label):
    """File-name form of a step label."""
    return re.sub(r'[^a-z0-9]+', '_', re.sub(r'[$\\{}]', '', label.lower())).strip('_')


def plot_pair(ax, edges, before, after, labels, sample, logy=False):
    """Step histograms of `before` (black) and `after` (blue) on `ax`, each (counts per bin of
    `edges`, total candidates), the text `sample` in the upper left, with room above the tallest
    bin for it and the legend."""
    C = plot_style.COLORS
    top = 1
    for (counts, total), label, color in ((before, labels[0], C['data']),
                                          (after, labels[1], C['fit'])):
        ax.stairs(counts, edges, linewidth=1.5, color=color, label=f'{label} ({total})')
        top = max(top, counts.max())
    plot_style.style_axes(ax)
    legend = ax.legend(frameon=False, loc='upper right')
    sample_box = plot_style.add_stats_box(ax, '', sample=sample)
    if logy:
        ax.set_yscale('log')
        plot_style.reserve_top_margin(ax, top, [sample_box, legend], ymin=0.5, growth=2.0,
                                      max_iter=12)
    else:
        plot_style.reserve_top_margin(ax, top, [sample_box, legend])


def save_pair(outdir, name, edges, before, after, labels, sample, xlabel):
    """plot_pair with the eta mass marked, saved as outdir/name.png and outdir/name_log.png."""
    for logy, suffix in ((False, ''), (True, '_log')):
        fig, ax = plt.subplots()
        plot_pair(ax, edges, before, after, labels, sample, logy=logy)
        ax.axvline(kinematics.M_ETA, color=plot_style.COLORS['reference'], linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_style.count_label('Candidates', edges, 'MeV'))
        plot_style.save_plot(fig, os.path.join(outdir, f'{name}{suffix}.png'))
        plt.close(fig)


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
    parser.add_argument('--fit-nbins', type=int, default=100,
                        help='Mass histogram bins in the fit window (driver_mass_fit.py --nbins)')
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
    # Per step: the histogram over the fit window of its candidates in the window, and their
    # count; over the full range, the histograms of every candidate and of those in the window.
    # All summed over the chunks. Chunk i picks the random candidate with seed selection.SEED + i,
    # as the drivers.
    edges = np.linspace(args.xmin, args.xmax, args.nbins + 1)
    window_edges = np.linspace(args.fit_xmin, args.fit_xmax, args.fit_nbins + 1)
    full_all, full_window, n_all = np.zeros(args.nbins), np.zeros(args.nbins), 0
    drop_report, labels, histograms, totals = {}, [], [], []
    chunks = cut_utils.iterate_branches(args.infile,
                                        selection.branches(args.mass, tos_lines, args.mc_match),
                                        selection.STEP_SIZE, entry_stop=args.max_events)
    for i, tree in enumerate(chunks):
        tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
        tree, chunk_drop_report = cut_utils.drop_malformed(tree, tag_ref=args.mass)
        drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
        steps = cut_steps(tree, args.mass, tos_lines, muon_thresholds,
                          selection.SEED + i, args.mc_match)
        in_window = (tree[args.mass] >= args.fit_xmin) & (tree[args.mass] < args.fit_xmax)
        if not labels:
            labels = [label for label, _ in steps]
            histograms = [np.zeros(args.fit_nbins) for _ in steps]
            totals = [0 for _ in steps]
        for step, (_, mask) in enumerate(steps):
            masses = cut_utils.values(tree, args.mass, mask & in_window)
            histograms[step] += np.histogram(masses, bins=window_edges)[0]
            totals[step] += len(masses)
        masses = cut_utils.values(tree, args.mass, steps[0][1])
        full_all += np.histogram(masses, bins=edges)[0]
        n_all += len(masses)
        masses = cut_utils.values(tree, args.mass, steps[0][1] & in_window)
        full_window += np.histogram(masses, bins=edges)[0]
        print(f"[INFO] chunk {i}: {drop_report['read']} events read", file=sys.stderr)

    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}",
          file=sys.stderr)
    print(f"[INFO]   {'all candidates, full range':32s} {n_all:9d}", file=sys.stderr)
    print('[INFO] in the fit window:', file=sys.stderr)
    for label, total in zip(labels, totals):
        print(f'[INFO]   {label:32s} {total:9d}', file=sys.stderr)

    plot_utils.apply_font()
    xlabel = 'DTF candidate mass [MeV]' if args.mass == 'tag_dtf_m' else f'{args.mass} [MeV]'
    sample = plot_style.sample_label(args.infile)
    window = f'fit window {args.fit_xmin:g}–{args.fit_xmax:g} MeV'

    # --- Mass: every candidate vs those in the fit window, full range ---
    save_pair(args.outdir, 'mass_01_fit_window', edges, (full_all, n_all),
              (full_window, totals[0]), (labels[0], f'+ {window}'), sample, xlabel)

    # --- Mass in the fit window: step i-1 vs step i ---
    for i in range(1, len(labels)):
        save_pair(args.outdir, f'mass_{i + 1:02d}_{slug(labels[i])}', window_edges,
                  (histograms[i - 1], totals[i - 1]), (histograms[i], totals[i]),
                  (window if i == 1 else 'previous cuts', f'+ {labels[i]}'), sample, xlabel)


if __name__ == '__main__':
    main()
