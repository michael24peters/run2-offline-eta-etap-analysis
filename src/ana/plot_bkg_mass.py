"""
plot_bkg_mass.py

Mass of the MC candidates the mass fit sees, split by background source
(bkg_utils.candidate_categories against eta -> --daughter-pids):

    bkg_mass.png        one step histogram per category with candidates, the eta mass marked
    bkg_mass_log.png    the same with a log y axis

The candidates are those of driver_mass_fit.py with no MC match: selection.cut_steps with
selection.TOS_LINES and MUON_THRESHOLDS, or TOS_LINES_4MU and MUON_THRESHOLDS_4MU for an input path
with an eta2mumumumu segment, then one random candidate per event. The input is read in chunks of
selection.STEP_SIZE, chunk i picking with seed selection.SEED + i. The count and fraction of each
category print to stderr.

Usage:
    python src/ana/plot_bkg_mass.py INFILE --daughter-pids PID [PID ...] --outdir OUTDIR
        [--max-events N] [--xmin 398] [--xmax 1108] [--nbins 142]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
import bkg_utils
import cut_utils
import kinematics
import plot_style
import plot_utils
import selection

MASS_BRANCH = 'tag_dtf_m'   # as driver_mass_fit.py

# Legend label and color of each category of bkg_utils.candidate_categories (Okabe-Ito)
CATEGORY_STYLE = {
    'mc_matched':             ('MC-matched', '#000000'),
    'unmatched':              ('Daughter with no generator match', '#0072B2'),
    'muon_misid':             ('Misidentified muon', '#009E73'),
    'photon_misid':           ('Misidentified photon', '#D55E00'),
    'muon_not_from_signal':   (r'Muon not from the $\eta$', '#CC79A7'),
    'photon_not_from_signal': (r'Photon not from the $\eta$', '#56B4E9'),
    'several':                ('Two or more of these', '#999999'),
}


def plot_categories(ax, edges, histograms, counts, sample, logy=False):
    """Step histogram of each category with candidates on `ax` (counts per bin of `edges`, total
    in the legend), the text `sample` in the upper left, with room above the tallest bin for it and
    the legend."""
    top = 1
    for name, (label, color) in CATEGORY_STYLE.items():
        if counts[name] == 0:
            continue
        ax.stairs(histograms[name], edges, linewidth=1.5, color=color,
                  label=f'{label} ({counts[name]})')
        top = max(top, histograms[name].max())
    plot_style.style_axes(ax)
    legend = ax.legend(frameon=False, loc='upper right', fontsize='small')
    sample_box = plot_style.add_stats_box(ax, '', sample=sample)
    if logy:
        ax.set_yscale('log')
        plot_style.reserve_top_margin(ax, top, [sample_box, legend], ymin=0.5, growth=2.0,
                                      max_iter=12)
    else:
        plot_style.reserve_top_margin(ax, top, [sample_box, legend])
        plot_style.format_y_axis(ax)


def main():
    parser = argparse.ArgumentParser(
        description='MC candidate mass by background source',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input MC ROOT file, or a quoted glob of them')
    parser.add_argument('--daughter-pids', nargs='+', type=int, required=True, metavar='PID',
                        help='Signal daughter PIDs: -13 13, -13 13 22, or -13 -13 13 13')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--xmin', type=float, default=398.0, help='Lower mass bound [MeV]')
    parser.add_argument('--xmax', type=float, default=1108.0, help='Upper mass bound [MeV]')
    parser.add_argument('--nbins', type=int, default=142, help='Mass histogram bins')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    is_4mu = 'eta2mumumumu' in args.infile.split('/')
    tos_lines = selection.TOS_LINES_4MU if is_4mu else selection.TOS_LINES
    muon_thresholds = selection.MUON_THRESHOLDS_4MU if is_4mu else selection.MUON_THRESHOLDS

    # --- Read, select, categorise and histogram one chunk of selection.STEP_SIZE at a time ---
    edges = np.linspace(args.xmin, args.xmax, args.nbins + 1)
    histograms = {name: np.zeros(args.nbins) for name in CATEGORY_STYLE}
    counts = {name: 0 for name in CATEGORY_STYLE}
    drop_report = {}
    chunks = cut_utils.iterate_branches(args.infile,
                                        selection.branches(MASS_BRANCH, tos_lines, mc_match=True),
                                        selection.STEP_SIZE, entry_stop=args.max_events)
    for i, tree in enumerate(chunks):
        tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
        tree, chunk_drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
        drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
        steps, _ = selection.cut_steps(tree, MASS_BRANCH, tos_lines,
                                       muon_thresholds=muon_thresholds)
        picked = cut_utils.random_candidate_mask(steps[-1][1], selection.SEED + i)
        categories = bkg_utils.candidate_categories(tree, selection.MOTHER_PID, args.daughter_pids,
                                                    target_ref=MASS_BRANCH)
        for name, in_category in categories.items():
            masses = cut_utils.values(tree, MASS_BRANCH, picked & in_category)
            histograms[name] += np.histogram(masses, bins=edges)[0]
            counts[name] += len(masses)
        print(f"[INFO] chunk {i}: {drop_report['read']} events read", file=sys.stderr)

    n_picked = sum(counts.values())
    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}; "
          f"{n_picked} random candidates", file=sys.stderr)
    for name, n in counts.items():
        print(f'[INFO]   {name:24s} {n:9d}  {n / max(n_picked, 1):7.2%}', file=sys.stderr)

    plot_utils.apply_font()
    sample = plot_style.sample_label(args.infile)
    for logy, suffix in ((False, ''), (True, '_log')):
        fig, ax = plt.subplots()
        plot_categories(ax, edges, histograms, counts, sample, logy=logy)
        ax.axvline(kinematics.M_ETA, color=plot_style.COLORS['reference'], linestyle='--')
        ax.set_xlabel('DTF candidate mass [MeV]')
        ax.set_ylabel(plot_style.count_label('Candidates', edges, 'MeV'))
        plot_style.save_plot(fig, os.path.join(args.outdir, f'bkg_mass{suffix}.png'))
        plt.close(fig)


if __name__ == '__main__':
    main()
