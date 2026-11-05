"""
plot_cut_steps.py

Mass before and after each offline selection with no fit, to check the effect of each cut:

    mass_<NN>_<cut>.png       one figure per cut: the mass after every earlier
                              cut, and after this cut too. Figure 01 is the
                              ntuple vs the first cut, figure 02 the first cut
                              vs the first two, and so on.

The cuts are cumulative, in the order of driver_mass_fit.py: [truth match], TOS per trigger level,
each muon cut, random candidate, then the fit window [--fit-xmin, --fit-xmax) (driver_mass_fit.py's
--xmin/--xmax defaults), so the last step is exactly what the mass fit sees. The selection constants
are copied from driver_mass_fit.py; keep them in step.


Usage:
    python src/plot/plot_cut_steps.py INFILE --outdir OUTDIR
        [--mass tag_dtf_m|tag_m] [--truth-match PID [PID ...]] [--max-events N]
        [--fit-xmin 500] [--fit-xmax 600]
        [--xmin 398] [--xmax 1108] [--nbins 142] [--logy]
"""

import argparse
import os
import re
import sys

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import bkg_utils
import cut_utils
import kinematics
import plot_style
import plot_utils

# --- Selection (as driver_mass_fit.py) ---------------------------------------

SEED            = 0
DAUGHTER_FIELDS = ['prt_pid', 'prt_px', 'prt_py', 'prt_pz', 'prt_pnn_mu',
                   'prt_prb_ghost', 'prt_ip_chi2', 'prt_tight_mu']
MUON_PID        = 13
MOTHER_PID      = 221
TRUTH_BRANCHES  = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']

MU_PT_MIN       = 1000.0           # pT [MeV]
MU_P_MIN        = 10000.0          # p [MeV]
MU_ETA_RANGE    = (2.0, 4.5)       # pseudorapidity
PNN_MU_MIN      = 0.9              # ProbNNmu
MU_GHOST_MAX    = 0.2              # ghost probability
MU_IP_CHI2_MAX  = 6.0              # IP chi2

TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0'],                    # Hlt1DiMuonNoIP
    'HLT2': ['tag_hlt2_tos0'],                    # Hlt2ExoticaPrmptDiMuonTurbo
}


def read_tree(infile, mass_branch, truth_match=None, max_events=None):
    """
    The branches cut_steps reads from `infile`, malformed events dropped.

    Returns (tree, drop_report) as cut_utils.drop_malformed.
    """
    branches = ([mass_branch, 'prt_idx_mom'] + DAUGHTER_FIELDS
                + [line for lines in TOS_LINES.values() for line in lines]
                + (TRUTH_BRANCHES if truth_match else []))
    tree = cut_utils.read_branches(infile, branches, entry_stop=max_events)
    tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
    return cut_utils.drop_malformed(tree, tag_ref=mass_branch)


def cut_steps(tree, mass_branch, fit_window, truth_match=None):
    """
    Cumulative selection of driver_mass_fit.py, one step per cut, ending with the fit window (lo,
    hi) [MeV] on mass_branch.

    Returns [(label, mask), ...]: the per-candidate mask after each step, starting with ('ntuple',
    all candidates).
    """
    keep = ak.ones_like(tree[mass_branch], dtype=bool)
    steps = [('ntuple', keep)]

    if truth_match:
        keep = keep & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, truth_match,
                                                      target_ref=mass_branch)
        steps.append(('truth-matched', keep))

    # TOS one level at a time, so each level is its own step.
    for level, lines in TOS_LINES.items():
        tos, _ = cut_utils.tos_mask(tree, {level: lines}, tag_ref=mass_branch, base=keep)
        keep = keep & tos
        steps.append((' | '.join(lines), keep))

    # Labels are the ntuple branches each cut reads; pT, p and eta have no branch of their own
    # (computed from prt_px/py/pz).
    px, py, pz = tree['prt_px'], tree['prt_py'], tree['prt_pz']
    eta = ak.unflatten(kinematics.eta(ak.to_numpy(ak.flatten(px)), ak.to_numpy(ak.flatten(py)),
                                      ak.to_numpy(ak.flatten(pz))), ak.num(px))
    muon_cuts = {
        f'pT(muon) > {MU_PT_MIN / 1000:g} GeV':   kinematics.pt(px, py) > MU_PT_MIN,
        f'p(muon) > {MU_P_MIN / 1000:g} GeV':     kinematics.p(px, py, pz) > MU_P_MIN,
        rf'${MU_ETA_RANGE[0]:g} < \eta < {MU_ETA_RANGE[1]:g}$':
            (eta > MU_ETA_RANGE[0]) & (eta < MU_ETA_RANGE[1]),
        f'prt_pnn_mu > {PNN_MU_MIN}':             tree['prt_pnn_mu'] > PNN_MU_MIN,
        f'prt_prb_ghost < {MU_GHOST_MAX}':        tree['prt_prb_ghost'] < MU_GHOST_MAX,
        f'0 <= prt_ip_chi2 < {MU_IP_CHI2_MAX:g}':   # -1 is stored when there is no PV
            (tree['prt_ip_chi2'] >= 0) & (tree['prt_ip_chi2'] < MU_IP_CHI2_MAX),
        'prt_tight_mu > 0.5':                     tree['prt_tight_mu'] > 0.5,
    }
    for label, passes in muon_cuts.items():
        muon_ok = passes | (abs(tree['prt_pid']) != MUON_PID)
        keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                       target_ref=mass_branch)
        steps.append((label, keep))

    keep = cut_utils.random_candidate_mask(keep, SEED)
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
    """Step histograms of `before` (black) and `after` (blue) on `ax`, with room above the tallest
    bin for the legend."""
    C = plot_style.COLORS
    top = 0
    for values, label, color in ((before, labels[0], C['data']), (after, labels[1], C['fit'])):
        counts, _, _ = ax.hist(values, bins=edges, histtype='step', linewidth=1.5, color=color,
                               label=f'{label} ({len(values)})')
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
    parser.add_argument('--truth-match', nargs='+', type=int, default=None, metavar='PID',
                        help='Signal MC: first cut is the truth match to eta -> these PIDs')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events (default: all)')
    parser.add_argument('--fit-xmin', type=float, default=500.0,
                        help='Lower edge of the fit window [MeV] (driver_mass_fit.py --xmin)')
    parser.add_argument('--fit-xmax', type=float, default=600.0,
                        help='Upper edge of the fit window [MeV] (driver_mass_fit.py --xmax)')
    parser.add_argument('--xmin', type=float, default=398.0, help='Lower mass bound [MeV]')
    parser.add_argument('--xmax', type=float, default=1108.0, help='Upper mass bound [MeV]')
    parser.add_argument('--nbins', type=int, default=142, help='Mass histogram bins')
    parser.add_argument('--logy', action='store_true', help='Log y axis on the mass plots')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tree, drop_report = read_tree(args.infile, args.mass, args.truth_match, args.max_events)
    print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}",
          file=sys.stderr)

    steps = cut_steps(tree, args.mass, (args.fit_xmin, args.fit_xmax), args.truth_match)
    for label, mask in steps:
        print(f'[INFO]   {label:32s} {int(ak.sum(mask)):9d}', file=sys.stderr)

    plot_utils.apply_font()
    xlabel = f'{args.mass} [MeV]'

    # --- Mass: step i-1 vs step i ---
    edges = np.linspace(args.xmin, args.xmax, args.nbins + 1)
    for i in range(1, len(steps)):
        (label_before, before), (label_after, after) = steps[i - 1], steps[i]
        fig, ax = plt.subplots()
        plot_pair(ax, edges, cut_utils.values(tree, args.mass, before),
                  cut_utils.values(tree, args.mass, after),
                  (label_before if i == 1 else 'previous cuts', f'+ {label_after}'),
                  logy=args.logy)
        ax.axvline(kinematics.M_ETA, color=plot_style.COLORS['reference'], linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Candidates')
        plot_style.save_plot(fig, os.path.join(args.outdir, f'mass_{i:02d}_{slug(label_after)}.png'))
        plt.close(fig)


if __name__ == '__main__':
    main()
