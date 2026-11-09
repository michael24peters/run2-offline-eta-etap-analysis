"""
skim.py

Skim of a data ntuple production for the drivers: every event with at least one candidate that
passes driver_mass_fit's selection, with only the branches the drivers read.

    read the sorted glob in STEP_SIZE chunks (uproot.iterate)
      -> event_mask: align_muons, drop_malformed, selection.cut_steps with selection.TOS_LINES,
         then any candidate passing the last cut
      -> OUTFILE, TTree 'tree': the kept events, every branch as stored (prt_tight_mu unaligned)
      -> skim.json next to OUTFILE: input files, branches, TOS_LINES, cut labels, events read,
         events kept and candidates kept (passing the last cut)

The selection reads only driver_mass_fit's branches (SELECTION_BRANCHES), so drop_malformed checks
the same branches as in driver_mass_fit. No mass-window cut is applied.

The branches written are WRITE_BRANCHES (both mass branches of driver_dalitz.MASS_SOURCES and
their daughter four-momenta, prt_idx_mom, selection.MUON_CUT_BRANCHES, run_n, evt_n) and every
tag_*_tos* branch of the input. uproot adds a counter branch n<branch> per jagged branch.

Usage:
    python src/root/skim.py 'ntuple/lhcb/<channel>/<date>/mag*/*.root' \
        ntuple/lhcb/<channel>/<date>/skim.root
"""

import argparse
import glob
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'fit'))
import cut_utils
import selection
from driver_dalitz import MASS_SOURCES

MASS_BRANCH = 'tag_dtf_m'   # MASS_BRANCH of driver_mass_fit
STEP_SIZE = '2 GB'          # uproot.iterate chunk size

# The branches of driver_mass_fit, the only ones the selection reads
SELECTION_BRANCHES = ([MASS_BRANCH, 'prt_idx_mom'] + selection.MUON_CUT_BRANCHES
                      + [line for lines in selection.TOS_LINES.values() for line in lines])
# Written with every tag_*_tos* branch of the input; in order, without duplicates
WRITE_BRANCHES = list(dict.fromkeys(
    list(MASS_SOURCES) + [branch for momenta in MASS_SOURCES.values() for branch in momenta]
    + ['prt_idx_mom'] + selection.MUON_CUT_BRANCHES + ['run_n', 'evt_n']))


def event_mask(tree):
    """
    Per-event numpy mask of `tree`: True for every event with at least one candidate passing
    driver_mass_fit's selection (align_muons, drop_malformed, then selection.cut_steps with
    selection.TOS_LINES). Reads the SELECTION_BRANCHES of `tree` and leaves `tree` unchanged.

    Returns (mask, steps): the mask, and the [(label, keep), ...] of cut_steps on the events
    drop_malformed keeps.
    """
    events = tree[SELECTION_BRANCHES]
    # Row of each event in `tree`; drop_malformed checks only tag_* and prt_* fields
    events['skim_index'] = np.arange(len(events))
    events['prt_tight_mu'] = cut_utils.align_muons(events, 'prt_tight_mu')   # stored for muons only
    events, _ = cut_utils.drop_malformed(events, tag_ref=MASS_BRANCH)
    steps, _ = selection.cut_steps(events, MASS_BRANCH, selection.TOS_LINES)
    passes = ak.any(steps[-1][1], axis=1)
    mask = np.zeros(len(tree), dtype=bool)
    mask[ak.to_numpy(events['skim_index'][passes])] = True
    return mask, steps


def main():
    parser = argparse.ArgumentParser(
        description='Skim a data ntuple production to the events and branches the drivers read',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('inglob', help='Quoted glob of the input ROOT files')
    parser.add_argument('outfile', help='Output ROOT file; skim.json is written next to it')
    args = parser.parse_args()

    paths = sorted(glob.glob(args.inglob))
    if not paths:
        parser.error(f'no file matches {args.inglob}')
    with uproot.open(paths[0]) as f:
        branches = WRITE_BRANCHES + f['tree'].keys(filter_name='tag_*_tos*')

    n_read = n_kept = n_candidates = 0
    with uproot.recreate(args.outfile) as out:
        chunks = uproot.iterate({path: 'tree' for path in paths}, branches,
                                step_size=STEP_SIZE, library='ak')
        for i, chunk in enumerate(chunks):
            mask, steps = event_mask(chunk)
            kept = {branch: chunk[branch][mask] for branch in branches}
            if i == 0:
                out.mktree('tree', kept)   # a TTree; assigning out['tree'] writes an RNTuple
            else:
                out['tree'].extend(kept)
            n_read += len(chunk)
            n_kept += int(np.sum(mask))
            n_candidates += int(ak.sum(steps[-1][1]))
            print(f'[INFO] events: read {n_read}, kept {n_kept}', file=sys.stderr)

    report = {
        'input_files': paths,
        'branches': branches,
        'tos_lines': selection.TOS_LINES,
        'cuts': [label for label, _ in steps],
        'events_read': n_read,
        'events_kept': n_kept,
        'candidates_kept': n_candidates,
    }
    report_path = os.path.join(os.path.dirname(args.outfile), 'skim.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[DONE] {args.outfile}, {report_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
