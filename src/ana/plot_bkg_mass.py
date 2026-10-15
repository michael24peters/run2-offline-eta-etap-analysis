"""
plot_bkg_mass.py

Script to plot eta mass distributions broken down by background category. Each
candidate is assigned to exactly one category based on the combination of
daughter error types from bkg.py classification. Uses uproot for ROOT I/O (no
PyROOT dependency).

Usage:
    python plot_bkg_mass.py input.root [--outfile out/output.png]
                                       [--verbose] [--dtrs 2]
                                       [--log] [--debug]
                                       [--pnnmu_cut 0.5]
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

# Python auto-adds this script's own directory (src/ana/) to sys.path, so bkg
# is importable directly when this script is run as `python plot_bkg_mass.py`.
from bkg import Candidate, ErrorType, classify_candidate

# --- Category definitions ---
# Ordered for legend display: signal first, then mismatch group, error group,
# then catch-all.
CATEGORIES = [
    'Signal',
    'Muon PID mismatch only',
    'Dimuon PID mismatch only',
    'Photon PID mismatch only',
    'Combination PID mismatch',
    'Muon error only',
    'Dimuon error only',
    'Photon error only',
    'Combination error',
    'Other error',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot eta mass distributions by background error category.')
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('--outfile', '-o', default='out/output.png',
                        help='Output file prefix (default: out/output.png)')
    parser.add_argument('--dtrs', type=int, default=2,
                        help='Number of daughters per candidate (default: 2)')
    parser.add_argument('--pnnmu_cut', type=float, default=None,
                        help='PROBNNmu cut applied to muon daughters')
    parser.add_argument('--log', default=False, action='store_true',
                        help='Enable logging to file (default: False)')
    parser.add_argument('--debug', '-d', action='store_true', default=False,
                        help='Enable debug logging (default: False)')
    return parser.parse_args()


# --- Per-candidate label ---
def get_candidate_label(can: Candidate) -> str:
    """
    Map a Candidate to exactly one category label based on the full set of
    daughter error types. Candidates with a mix of PID-mismatch and error
    daughters (or any OTHER_ERROR) fall into 'Other error'.
    """
    # Signal candidates
    if can.is_sig: return 'Signal'

    has_mup_mm = any(d.err_type == ErrorType.MUP_PID_MISMATCH for d in can.dtrs)
    has_mum_mm = any(d.err_type == ErrorType.MUM_PID_MISMATCH for d in can.dtrs)
    has_pho_mm = any(d.err_type == ErrorType.PHOTON_PID_MISMATCH for d in can.dtrs)
    has_mup_err = any(d.err_type == ErrorType.MUP_COMBINATORICS for d in can.dtrs)
    has_mum_err = any(d.err_type == ErrorType.MUM_COMBINATORICS for d in can.dtrs)
    has_pho_err = any(d.err_type == ErrorType.PHOTON_COMBINATORICS for d in can.dtrs)
    has_other = any(d.err_type == ErrorType.OTHER_ERROR for d in can.dtrs)

    has_any_mm = has_mup_mm or has_mum_mm or has_pho_mm
    has_any_err = has_mup_err or has_mum_err or has_pho_err

    # Catch-all for other errors and mixed categories not captured
    if has_other or (has_any_mm and has_any_err):
        return 'Other error'

    # PID mismatch errors
    if has_any_mm:
        if has_pho_mm and (has_mup_mm or has_mum_mm): return 'Combination PID mismatch'
        if has_mup_mm and has_mum_mm: return 'Dimuon PID mismatch only'
        if has_pho_mm: return 'Photon PID mismatch only'
        return 'Muon PID mismatch only'

    # Non-PID errors
    if has_any_err:
        if has_pho_err and (has_mup_err or has_mum_err): return 'Combination error'
        if has_mup_err and has_mum_err: return 'Dimuon error only'
        if has_pho_err: return 'Photon error only'
        return 'Muon error only'

    return 'Other error'


# --- Event loop ---
def run_event_loop(args) -> dict[str, list[float]]:
    """
    Loop over events, classify each candidate, and collect tag_m values
    grouped by category label.

    All branches are read once via a single bulk uproot call, then converted
    to plain Python lists -- classify_candidate's per-candidate branching
    logic stays a Python loop either way, so this only replaces PyROOT's
    per-entry tree.GetEntry() calls with pre-loaded in-memory data.
    """
    masses: dict[str, list[float]] = defaultdict(list)

    with uproot.open(args.infile) as f:
        branches = f['tree'].arrays(
            ['tag_pid', 'tag_m', 'prt_pid', 'prt_idx_gen', 'prt_idx_mom',
             'mc_pid', 'mc_idx_mom', 'prt_pnn_mu'],
            library='ak',
        )

    tag_pid_evts = ak.to_list(branches['tag_pid'])
    tag_m_evts = ak.to_list(branches['tag_m'])
    prt_pid_evts = ak.to_list(branches['prt_pid'])
    prt_idx_gen_evts = ak.to_list(branches['prt_idx_gen'])
    prt_idx_mom_evts = ak.to_list(branches['prt_idx_mom'])
    mc_pid_evts = ak.to_list(branches['mc_pid'])
    mc_idx_mom_evts = ak.to_list(branches['mc_idx_mom'])
    probnn_mu_evts = ak.to_list(branches['prt_pnn_mu'])

    for entryIdx in range(len(tag_pid_evts)):
        if entryIdx % 500_000 == 0 and entryIdx > 0:
            logging.info(f'Processed {entryIdx:,d} events...')

        tag_pid = [int(pid) for pid in tag_pid_evts[entryIdx]]
        tag_m = [float(m) for m in tag_m_evts[entryIdx]]
        prt_pid = [int(pid) for pid in prt_pid_evts[entryIdx]]
        prt_idx_gen = [int(idx) for idx in prt_idx_gen_evts[entryIdx]]
        prt_idx_mom = [int(idx) for idx in prt_idx_mom_evts[entryIdx]]
        mc_pid = [int(pid) for pid in mc_pid_evts[entryIdx]]
        mc_idx_mom = [int(idx) for idx in mc_idx_mom_evts[entryIdx]]
        probnn_mu = [float(p) for p in probnn_mu_evts[entryIdx]]

        # Skip events with no candidates
        if not tag_pid: continue

        for i in range(len(tag_pid)):
            if tag_pid[i] != 221:
                continue
            candidate = classify_candidate(entryIdx, i, prt_pid, prt_idx_gen,
                prt_idx_mom, mc_pid, mc_idx_mom, probnn_mu, args)
            if candidate is None: continue
            label = get_candidate_label(candidate)
            masses[label].append(tag_m[i])

    return masses


# --- Plot ---
def plot_masses(masses: dict[str, list[float]], outfile: str) -> None:
    """
    Overlay eta mass histograms for each category on a single matplotlib
    figure and save to {outfile}.png.
    """
    bins = np.arange(400, 710, 10)  # 400–700 MeV, 10 MeV bins
    colors = plt.get_cmap('tab10').colors  # 10 distinct colours

    # Font settings
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 14
    # Creature plot
    plt.figure(figsize=(12, 9))

    for idx, label in enumerate(CATEGORIES):
        data = masses.get(label, [])
        if not data:
            continue
        arr = np.array(data)
        plt.hist(arr, bins=bins, histtype='step',
                color=colors[idx % len(colors)],
                label=f'{label} (n={len(arr):,})')

    plt.axvline(x=547.86, color='black', linestyle='--', linewidth=1,
               label='PDG $m_\\eta$ = 547.86 MeV')
    plt.xlabel('Mass [MeV]')
    plt.ylabel('Candidates')
    plt.title('Eta Mass Signal vs Background Categories')
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=11)

    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    plt.savefig(outfile + '.png', dpi=300, bbox_inches='tight')
    logging.info(f'Saved plot to {outfile}.png')


# --- Main ---
def main():
    args = parse_args()
    # Set up output plot file
    if not args.outfile.endswith('.png'): args.outfile += '.png'
    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # Log to file if requested, otherwise just log to console
    if args.log:
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=os.path.splitext(args.outfile)[0] + '.log',
            filemode='w',
        )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Reading from {args.infile}, writing to {args.outfile}.')
    if args.pnnmu_cut is not None:
        logging.info(f'Applying PROBNNmu cut: {args.pnnmu_cut}')

    # Run the event loop to classify candidates and collect masses by category
    masses = run_event_loop(args)

    # Log the number of candidates in each category and their percentage of the
    # total
    total = sum(len(v) for v in masses.values())
    for label in CATEGORIES:
        n = len(masses.get(label, []))
        if n:
            logging.info(f'  {label + ":":<35} {n:6,d}  ({100*n/total:.1f}%)')

    # Plot the mass distributions and save to file
    plot_masses(masses, args.outfile)
    logging.info(f'Plot saved to {args.outfile}.')


if __name__ == '__main__':
    main()
