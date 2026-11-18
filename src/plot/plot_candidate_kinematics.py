"""
plot_candidate_kinematics.py

pT and pseudorapidity of the eta candidates of two trigger selections, TOS only (no muon cut):
prompt (selection.TOS_LINES) and displaced (selection.TOS_LINES_DISPLACED). Per selection, one
candidate per event is picked at random among its TOS candidates (seed selection.SEED + chunk
index), and kept if its DTF mass is in --window.

    pt_fraction.png                each selection normalised to unit area, linear y
    pt_counts.png                  candidates, log y
    pseudorapidity_fraction.png
    pseudorapidity_counts.png

The input is read one chunk of selection.STEP_SIZE at a time (cut_utils.iterate_branches).

Usage:
    python src/plot/plot_candidate_kinematics.py 'ntuple/lhcb/eta2mumu/20261109/mag*/*.root' \
        --outdir OUTDIR [--window 535 560] [--max-events N]
"""

import argparse
import os
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

MASS_BRANCH = 'tag_dtf_m'   # the branch defining the candidates, and the window's mass
WINDOW      = (535.0, 560.0)   # default DTF mass window [MeV]
# (label, TOS lines, colour) of each selection
SELECTIONS = [
    ('Prompt',    selection.TOS_LINES,           plot_style.COLORS['fit']),
    ('Displaced', selection.TOS_LINES_DISPLACED, plot_style.COLORS['pull']),
]
PT_EDGES  = np.linspace(0.0, 25000.0, 101)   # pT [MeV], 250 MeV bins
ETA_EDGES = np.linspace(1.5, 5.0, 71)        # pseudorapidity, 0.05 bins


def select(tree, tos_lines, window, seed):
    """Per-candidate mask: one candidate per event, picked at random among those TOS on
    `tos_lines`, kept if its MASS_BRANCH is in `window`."""
    tos, _ = cut_utils.tos_mask(tree, tos_lines, tag_ref=MASS_BRANCH)
    picked = cut_utils.random_candidate_mask(tos, seed)
    mass = tree[MASS_BRANCH]
    return picked & (mass >= window[0]) & (mass <= window[1])


def plot(values, edges, xlabel, unit, box_text, sample, outdir, name):
    """
    `values` ({selection label: array}) as one step histogram per selection over `edges`:
    outdir/name_fraction.png, each selection normalised to unit area (linear y), and
    outdir/name_counts.png, candidates (log y).
    """
    for scale in ('fraction', 'counts'):
        fig, ax = plt.subplots()
        top = 0
        for label, _, color in SELECTIONS:
            data = values[label]
            weights = np.ones(len(data)) / max(len(data), 1) if scale == 'fraction' else None
            counts, _ = np.histogram(data, bins=edges, weights=weights)
            ax.stairs(counts, edges, linewidth=1.5, color=color, label=f'{label} ({len(data):,})')
            top = max(top, counts.max())
        ax.set_xlabel(xlabel)
        entries = 'Fraction of candidates' if scale == 'fraction' else 'Candidates'
        ax.set_ylabel(plot_style.count_label(entries, edges, unit))
        plot_style.style_axes(ax)
        legend = ax.legend(frameon=False, loc='upper right')
        sample_box = plot_style.add_stats_box(ax, box_text, sample=sample)
        if scale == 'counts':
            ax.set_yscale('log')
            plot_style.reserve_top_margin(ax, max(top, 1), [sample_box, legend], ymin=0.5,
                                          growth=2.0, max_iter=12)
        else:
            plot_style.reserve_top_margin(ax, top, [sample_box, legend])
        plot_style.save_plot(fig, os.path.join(outdir, f'{name}_{scale}.png'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='pT and pseudorapidity of the eta candidates, prompt and displaced TOS',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('infile', help='Input ROOT file, or a quoted glob of them')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--window', nargs=2, type=float, default=WINDOW, metavar=('LO', 'HI'),
                        help='DTF mass window [MeV] (default: 535 560)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tos_branches = {line for _, tos_lines, _ in SELECTIONS
                    for lines in tos_lines.values() for line in lines}
    branches = [MASS_BRANCH, 'prt_idx_mom', 'tag_px', 'tag_py', 'tag_pz'] + sorted(tos_branches)

    pts = {label: [] for label, _, _ in SELECTIONS}
    etas = {label: [] for label, _, _ in SELECTIONS}
    drop_report = {}
    chunks = cut_utils.iterate_branches(args.infile, branches, selection.STEP_SIZE,
                                        entry_stop=args.max_events)
    for i, tree in enumerate(chunks):
        tree, chunk_drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
        drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
        for label, tos_lines, _ in SELECTIONS:
            keep = select(tree, tos_lines, args.window, selection.SEED + i)
            px = cut_utils.values(tree, 'tag_px', keep)   # <- awkward -> numpy
            py = cut_utils.values(tree, 'tag_py', keep)
            pz = cut_utils.values(tree, 'tag_pz', keep)
            pts[label].append(kinematics.pt(px, py))
            etas[label].append(kinematics.eta(px, py, pz))
        print(f"[INFO] chunk {i}: {drop_report['read']} events read", file=sys.stderr)
    pts = {label: np.concatenate(arrays) for label, arrays in pts.items()}
    etas = {label: np.concatenate(arrays) for label, arrays in etas.items()}

    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}",
          file=sys.stderr)
    for label, _, _ in SELECTIONS:
        print(f'[INFO] {label}: {len(pts[label])} candidates; '
              f'{np.sum(pts[label] > PT_EDGES[-1])} above pT {PT_EDGES[-1]:g} MeV, '
              f'{np.sum((etas[label] < ETA_EDGES[0]) | (etas[label] > ETA_EDGES[-1]))} '
              f'outside pseudorapidity {ETA_EDGES[0]:g}-{ETA_EDGES[-1]:g}', file=sys.stderr)

    plot_utils.apply_font()
    sample = plot_style.sample_label(args.infile)
    box_text = f'DTF mass {args.window[0]:g}–{args.window[1]:g} MeV'
    plot(pts, PT_EDGES, r'$p_T$ of the $\eta$ candidate [MeV]', 'MeV', box_text, sample,
         args.outdir, 'pt')
    plot(etas, ETA_EDGES, r'Pseudorapidity of the $\eta$ candidate', '', box_text, sample,
         args.outdir, 'pseudorapidity')


if __name__ == '__main__':
    main()
