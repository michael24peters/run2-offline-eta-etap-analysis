"""
skim.py

Skim of an ntuple production for the drivers: per event, one candidate picked at random among
those that pass driver_mass_fit's selection, kept if MASS_BRANCH is in the --window range, with its
daughters, and only the branches the drivers read.

    read the sorted glob in STEP_SIZE chunks (uproot.iterate)
      -> candidate_mask: align_muons, drop_malformed, selection.cut_steps, the random candidate
         (cut_utils.random_candidate_mask, seed selection.SEED + chunk index), then the window
      -> keep_candidates: the kept candidate and its daughters, prt_idx_mom renumbered to it,
         events with no kept candidate dropped
      -> OUTFILE, TTree 'tree'
      -> OUTFILE with .json for .root: input files, branches, TOS lines, window, cut labels,
         the drop_malformed report summed over the chunks (events read, kept, dropped per reason,
         including events with no candidate), events kept and candidates kept

The selection is driver_mass_fit's for eta -> mu+ mu- and eta -> mu+ mu- gamma
(selection.TOS_LINES), without an MC match. drop_malformed checks every branch written.

An MC skim is never the input of an MC-matched run (--mc-match): the MC match is the first cut of
every MC-matched run, before TOS, the muon cuts and the candidate pick, and a skim has already
picked its candidate with no MC match. MC-matched runs read the raw files (mag*/*.root).

The branches written are WRITE_BRANCHES (both mass branches of driver_dalitz.MASS_SOURCES and
their daughter four-momenta, prt_idx_mom, selection.MUON_CUT_BRANCHES, run_n, evt_n), every
tag_*_tos* branch of the input and, in MC, selection.MC_BRANCHES. prt_tight_mu keeps one entry
per muon daughter, as stored; the mc_* branches are kept whole. uproot adds a counter branch
n<branch> per jagged branch.

Usage:
    python src/root/skim.py 'ntuple/<lhcb|mc>/<channel>/<production>/mag*/*.root' \
        ntuple/<lhcb|mc>/<channel>/<production>/skim_eta.root --window 450 650

The skims of a production, next to its mag* directories:

    skim_eta.root       --window 450 650 (eta2mumugamma) or 500 600 (eta2mumu)
    skim_etaprime.root  --window 800 1100, data only
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

MASS_BRANCH = 'tag_dtf_m'               # MASS_BRANCH of driver_mass_fit
STEP_SIZE = '2 GB'                      # uproot.iterate chunk size
MUON_ONLY_BRANCHES = ['prt_tight_mu']   # written branches stored one entry per muon

# Written with every tag_*_tos* branch of the input; in order, without duplicates
WRITE_BRANCHES = list(dict.fromkeys(
    list(MASS_SOURCES) + [branch for momenta in MASS_SOURCES.values() for branch in momenta]
    + ['prt_idx_mom'] + selection.MUON_CUT_BRANCHES + ['run_n', 'evt_n']))


def candidate_mask(tree, tos_lines, window, seed):
    """
    Per event of `tree`, one candidate picked at random (`seed`) among those that pass
    driver_mass_fit's selection (align_muons, drop_malformed, then selection.cut_steps with
    `tos_lines`), kept if MASS_BRANCH is in `window`, (low, high) [MeV] with both ends included.
    drop_malformed checks every field of `tree`; `tree` is left unchanged.

    Returns (rows, keep, steps, drop_report):
        rows        : numpy indices of the events of `tree` that drop_malformed keeps
        keep        : per-candidate mask of those events, shaped like MASS_BRANCH, at most one
                      candidate per event
        steps       : [(label, keep), ...] of cut_steps, then the random candidate and the window
        drop_report : the report of drop_malformed
    """
    events = tree[tree.fields]   # a new array: the fields set below leave `tree` unchanged
    # Row of each event in `tree`; drop_malformed checks only tag_* and prt_* fields
    events['skim_index'] = np.arange(len(events))
    events['prt_tight_mu'] = cut_utils.align_muons(events, 'prt_tight_mu')   # stored for muons only
    events, drop_report = cut_utils.drop_malformed(events, tag_ref=MASS_BRANCH)
    steps, _ = selection.cut_steps(events, MASS_BRANCH, tos_lines)

    picked = cut_utils.random_candidate_mask(steps[-1][1], seed)
    steps.append(('random candidate', picked))
    low, high = window
    mass = events[MASS_BRANCH]
    keep = picked & (mass >= low) & (mass <= high)
    steps.append((f'{MASS_BRANCH} in {low:g}-{high:g} MeV', keep))
    return ak.to_numpy(events['skim_index']), keep, steps, drop_report


def keep_candidates(tree, keep):
    """
    Every field of `tree` reduced to the candidates where `keep` and their daughters, in the events
    with at least one kept candidate. `tree` holds well-formed events (drop_malformed) with
    prt_tight_mu as stored; `keep` is per candidate, shaped like MASS_BRANCH.

        tag_*               the kept candidates; a branch empty in an event stays empty
        prt_*               the daughters of the kept candidates; prt_idx_mom renumbered to them
        MUON_ONLY_BRANCHES  the muon daughters of the kept candidates
        any other field     whole (run_n, evt_n, mc_*)

    Returns {field: awkward array}.
    """
    daughter_keep = cut_utils.follow(tree, keep, 'prt_idx_mom')
    muon_keep = daughter_keep[abs(tree['prt_pid']) == selection.MUON_PID]

    # Index of each candidate among the kept ones of its event, e.g. [[F, T, T]] -> [[0, 0, 1]]
    n_candidates = ak.to_numpy(ak.num(keep))
    flat_keep = ak.to_numpy(ak.flatten(keep)).astype(np.int64)
    kept_before = np.cumsum(flat_keep) - flat_keep   # kept candidates before each, over all events
    first = np.repeat(np.cumsum(n_candidates) - n_candidates, n_candidates)   # its event's first
    new_index = ak.unflatten((kept_before - kept_before[first]).astype(np.float64), n_candidates)

    has_kept = ak.any(keep, axis=1)
    out = {}
    for field in tree.fields:
        if field == 'prt_idx_mom':
            values = cut_utils.follow(tree, new_index, 'prt_idx_mom')[daughter_keep]
        elif field in MUON_ONLY_BRANCHES:
            values = tree[field][muon_keep]
        elif field.startswith('prt_'):
            values = tree[field][daughter_keep]
        elif field.startswith('tag_'):
            # Per candidate: its event stores the branch
            present = ak.broadcast_arrays(ak.num(tree[field]) > 0, keep)[0]
            values = tree[field][keep[present]]
        else:
            values = tree[field]
        out[field] = values[has_kept]
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Skim an ntuple production to the candidates and branches the drivers read',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('inglob', help='Quoted glob of the input ROOT files')
    parser.add_argument('outfile', help='Output ROOT file; its report is OUTFILE with .json')
    parser.add_argument('--window', nargs=2, type=float, required=True, metavar=('LOW', 'HIGH'),
                        help='MASS_BRANCH range of the kept candidates [MeV], both ends included')
    args = parser.parse_args()

    paths = sorted(glob.glob(args.inglob))
    if not paths:
        parser.error(f'no file matches {args.inglob}')
    with uproot.open(paths[0]) as f:
        input_branches = f['tree'].keys()
        branches = (WRITE_BRANCHES + f['tree'].keys(filter_name='tag_*_tos*')
                    + [branch for branch in selection.MC_BRANCHES if branch in input_branches])

    drop_report = {}
    n_kept = n_candidates = 0
    if os.path.exists(args.outfile):
        os.remove(args.outfile)   # uproot.recreate leaves the old file's bytes beyond the new end
    with uproot.recreate(args.outfile) as out:
        chunks = uproot.iterate({path: 'tree' for path in paths}, branches,
                                step_size=STEP_SIZE, library='ak')
        for i, chunk in enumerate(chunks):
            rows, keep, steps, chunk_drop_report = candidate_mask(chunk, selection.TOS_LINES,
                                                                  args.window, selection.SEED + i)
            kept = keep_candidates(chunk[rows], keep)
            if i == 0:
                out.mktree('tree', kept)   # a TTree; assigning out['tree'] writes an RNTuple
            else:
                out['tree'].extend(kept)
            drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
            n_kept += len(kept['run_n'])
            n_candidates += int(ak.sum(keep))
            print(f"[INFO] events: read {drop_report['read']}, kept {n_kept}", file=sys.stderr)

    report = {
        'input_files': paths,
        'branches': branches,
        'tos_lines': selection.TOS_LINES,
        'random_candidate': {'seed': selection.SEED, 'step_size': STEP_SIZE},
        'window': args.window,
        'cuts': [label for label, _ in steps],
        'drop_malformed': drop_report,
        'events_kept': n_kept,
        'candidates_kept': n_candidates,
    }
    report_path = os.path.splitext(args.outfile)[0] + '.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[DONE] {args.outfile}, {report_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
