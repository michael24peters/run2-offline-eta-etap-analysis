"""
plot_decay_length.py

Calculate and plot decay length and pseudo-lifetime for eta candidates from a ROOT file, for two
trigger selections, TOS only (no muon cut): prompt (selection.TOS_LINES) and displaced
(selection.TOS_LINES_DISPLACED). Per selection, one candidate per event is picked at random among
its TOS candidates (seed selection.SEED + chunk index), and kept if its DTF mass is in --window and
it has a PV.
Pseudo-lifetime: tau = L * m / (p * c), derived from tau = L / (beta*gamma*c).

    decay_length_fraction.png      each selection normalised to unit area, linear y
    decay_length_counts.png        candidates, log y
    pseudo_lifetime_fraction.png
    pseudo_lifetime_counts.png

The input is read one chunk of selection.STEP_SIZE at a time (cut_utils.iterate_branches).

Usage:
    python plot_decay_length.py 'ntuple/lhcb/eta2mumu/20261109/mag*/*.root' --outdir OUTDIR \
        [--window 535 560] [--max-events N]
"""

import argparse
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
import cut_utils
import kinematics
import plot_style
import plot_utils
import selection

C_MM_PER_PS = 0.299_792_458   # speed of light in mm/ps
MASS_BRANCH = 'tag_dtf_m'      # the branch defining the candidates, and the window's mass
WINDOW      = (535.0, 560.0)   # eta -> mu mu fit window [MeV]
# (label, TOS lines, colour) of each selection
SELECTIONS = [
    ('Prompt',    selection.TOS_LINES,           plot_style.COLORS['fit']),
    ('Displaced', selection.TOS_LINES_DISPLACED, plot_style.COLORS['pull']),
]
DECAY_LENGTH_EDGES = np.linspace(0.0, 1.0, 101)   # decay length [mm], 0.01 mm bins
LIFETIME_EDGES     = np.linspace(0.0, 0.1, 101)   # pseudo-lifetime [ps], 0.001 ps bins

def parse_args():
    parser = argparse.ArgumentParser(description='Decay length and pseudo-lifetime from ROOT tree.')
    parser.add_argument('input', help='Input ROOT file, or a quoted glob of them')
    parser.add_argument('--outdir', required=True, help='Directory for the PNGs')
    parser.add_argument('--window', nargs=2, type=float, default=WINDOW, metavar=('LO', 'HI'),
                        help='DTF mass window [MeV] (default: 535 560)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    return parser.parse_args()

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

args = parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f'[INFO] Reading from {args.input} and writing to {args.outdir}.')

tos_branches = {line for _, tos_lines, _ in SELECTIONS
                for lines in tos_lines.values() for line in lines}
branches = ([MASS_BRANCH, 'prt_idx_mom', 'tag_x', 'tag_y', 'tag_z', 'tag_idx_pvr',
             'tag_px', 'tag_py', 'tag_pz', 'tag_m', 'pvr_x', 'pvr_y', 'pvr_z']
            + sorted(tos_branches))

decay_lengths = {label: [] for label, _, _ in SELECTIONS}
pseudo_lifetimes = {label: [] for label, _, _ in SELECTIONS}
no_pv = {label: 0 for label, _, _ in SELECTIONS}
drop_report = {}
chunks = cut_utils.iterate_branches(args.input, branches, selection.STEP_SIZE,
                                    entry_stop=args.max_events)
for i, tree in enumerate(chunks):
    tree, chunk_drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
    drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
    # tag_idx_pvr is -1 for a candidate with no PV
    has_pv = cut_utils.valid(tree, 'tag_idx_pvr')
    for label, tos_lines, _ in SELECTIONS:
        keep = select(tree, tos_lines, args.window, selection.SEED + i)
        no_pv[label] += int(ak.sum(keep & ~has_pv))
        keep = keep & has_pv

        # tag_idx_pvr tells us which PV in the pvr_* arrays belongs to this candidate.
        ipv = ak.values_astype(tree['tag_idx_pvr'][keep], np.int64)

        # Decay length (L): distance between decay vertex (tag_x/y/z) and PV (pvr_x/y/z), in mm.
        dx = ak.to_numpy(ak.flatten(tree['tag_x'][keep] - tree['pvr_x'][ipv]))
        dy = ak.to_numpy(ak.flatten(tree['tag_y'][keep] - tree['pvr_y'][ipv]))
        dz = ak.to_numpy(ak.flatten(tree['tag_z'][keep] - tree['pvr_z'][ipv]))
        L = np.sqrt(dx**2 + dy**2 + dz**2)

        # Pseudo-lifetime (tau) = L / (beta*gamma*c), with beta*gamma = p / m (natural units), so
        # tau = L * m / (p * c)
        p = kinematics.p(cut_utils.values(tree, 'tag_px', keep),
                         cut_utils.values(tree, 'tag_py', keep),
                         cut_utils.values(tree, 'tag_pz', keep))
        m = cut_utils.values(tree, 'tag_m', keep)

        # Units: L [mm], m [MeV], p [MeV], c [mm/ps] -> tau [ps]
        tau = L * m / (p * C_MM_PER_PS)

        decay_lengths[label].append(L)
        pseudo_lifetimes[label].append(tau)
    print(f"[INFO] chunk {i}: {drop_report['read']} events read")

decay_lengths = {label: np.concatenate(arrays) for label, arrays in decay_lengths.items()}
pseudo_lifetimes = {label: np.concatenate(arrays) for label, arrays in pseudo_lifetimes.items()}

print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}")
for label, _, _ in SELECTIONS:
    print(f'[INFO] {label}: {len(decay_lengths[label])} candidates, '
          f'{no_pv[label]} more with no PV left out')
    print(f'[INFO]   Decay length:    [{decay_lengths[label].min():.4f}, '
          f'{decay_lengths[label].max():.4f}] mm, '
          f'{np.sum(decay_lengths[label] > DECAY_LENGTH_EDGES[-1])} above '
          f'{DECAY_LENGTH_EDGES[-1]:g} mm')
    print(f'[INFO]   Pseudo-lifetime: [{pseudo_lifetimes[label].min():.4e}, '
          f'{pseudo_lifetimes[label].max():.4e}] ps, '
          f'{np.sum(pseudo_lifetimes[label] > LIFETIME_EDGES[-1])} above '
          f'{LIFETIME_EDGES[-1]:g} ps')

plot_utils.apply_font()
sample = plot_style.sample_label(args.input)
box_text = f'DTF mass {args.window[0]:g}–{args.window[1]:g} MeV'
plot(decay_lengths, DECAY_LENGTH_EDGES, 'Decay length [mm]', 'mm', box_text, sample,
     args.outdir, 'decay_length')
plot(pseudo_lifetimes, LIFETIME_EDGES, 'Pseudo-lifetime [ps]', 'ps', box_text, sample,
     args.outdir, 'pseudo_lifetime')
