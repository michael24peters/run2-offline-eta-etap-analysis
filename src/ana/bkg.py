"""
bkg.py

Script to analyze background for eta -> mu+ mu- (gamma). Requires MC generator
level information to classify candidates as signal or background. Uses uproot
for ROOT I/O (no PyROOT dependency).

Usage:
    python bkg.py input.root [--outfile out/output.txt] [--verbose]
                  [--dtrs 2] [--log] [--debug] [--pnnmu_cut 0.5]
"""

# TODO: Remove log file option and just log to console, which is what the rest
# of this project does. May be a future feature to log to file, but not needed
# for now.

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import awkward as ak
import uproot


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('-o', '--outfile', default='out/output.txt',
                        help='Output text file')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('--pnnmu_cut', type=float, default=None,
                        help='PROBNNmu cut value applied to input file')
    # Store number of daughters per candidate, 2 for eta -> mu+ mu-, 3 for 
    # eta -> mu+ mu- gamma, etc.
    parser.add_argument('--dtrs', action='store', default=2, type=int,
                        help='Number of daughters per candidate (default: 2)')
    parser.add_argument('--log', default=False, action='store_true',
                        help='Enable logging to file (default: False)')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='Enable debug logging (default: False)')
    return parser.parse_args()

# Possible error categories for a decay candidate
class ErrorType(str, Enum):
    MUP_PID_MISMATCH = 'MUP_PID_MISMATCH'
    MUP_ONLY_PID_MISMATCH = 'MUP_ONLY_PID_MISMATCH'
    MUM_PID_MISMATCH = 'MUM_PID_MISMATCH'
    MUM_ONLY_PID_MISMATCH = 'MUM_ONLY_PID_MISMATCH'
    DIMUON_PID_MISMATCH = 'DIMUON_PID_MISMATCH'
    PHOTON_PID_MISMATCH = 'PHOTON_PID_MISMATCH'
    MUP_ERROR = 'MUP_COMBINATORICS'
    MUP_ONLY_ERROR = 'MUP_ONLY_COMBINATORICS'
    MUM_ERROR = 'MUM_COMBINATORICS'
    MUM_ONLY_ERROR = 'MUM_ONLY_COMBINATORICS'
    DIMUON_ERROR = 'DIMUON_COMBINATORICS'
    PHOTON_ERROR = 'PHOTON_COMBINATORICS'
    OTHER_ERROR = 'OTHER_ERROR'

# Data class to hold daughter information for a candidate
@dataclass
class Daughter:
    prt_pid: int
    prt_idx_gen: int
    mc_pid: int | None
    mc_idx_mom: int | None
    err_type: ErrorType

# Data class to hold per candidate information for analytics
@dataclass
class Candidate:
    is_sig: bool
    evt: int
    can_idx: int
    dtrs: list[Daughter]
    has_dimu_mismatch: bool
    has_dimu_err: bool

# Data class to hold overall analytics results for output
@dataclass
class Analytics:
    ncan: int
    nsig: int
    nbkg: int
    err_counters: dict[ErrorType, int]
    mup_mismatches: list[int]
    mum_mismatches: list[int]
    pho_mismatches: list[int]
    other_mismatches: list[int]
    # Derived quantities
    prob_dimu_given_mu: float
    prob_mup_err: float
    prob_mum_err: float
    prob_pho_err: float
    # Sorted (pid, count) pairs for each daughter's PID mismatch
    pid_freq: dict[str, list[tuple[int, int]]]


# ──────────────────────────────────────────────────────────────────────────────

def classify_candidate(entryIdx, i, prt_pid, prt_idx_gen, prt_idx_mom, mc_pid, mc_idx_mom,
                       probnn_mu, args):
    """
    For a single candidate, classify as signal or background and determine error
    type if background. Returns a Candidate object with all relevant info for
    analytics.
    """
    # NOTE: can only handle one gen-level eta candidate per event, improve
    # later if needed, or just select the first one and ignore the others.
    # If so, then this check is needed to prevent out-of-bounds errors.
    #
    # Define j0 and j1 depending on number of daughters per candidate (ndtrs)
    ndtrs = args.dtrs
    j0, j1 = i * ndtrs, i * ndtrs + ndtrs
    if j1 > len(prt_pid) or j1 > len(prt_idx_gen) or j1 > len(prt_idx_mom):
        logging.warning(f'Event {entryIdx}, Candidate {i} skipped, has incomplete daughter info.')
        return None

    # Apply PROBNNmu cut if specified
    if args.pnnmu_cut is not None:
        for j in range(j0, j1):
            if abs(prt_pid[j]) == 13 and probnn_mu[j] < args.pnnmu_cut:
                return None

    is_signal = True
    dtrs: list[Daughter] = []
    dimu_mismatch = [False, False]
    dimu_err = [False, False]

    # Requires at least one MC eta candidate
    # NOTE: cannot handle multiple eta candidates (very rare)
    if mc_pid[0] != 221: is_signal = False

    for j in range(j0, j1):
        if prt_idx_mom[j] != i: break  # Skip failed reco, shouldn't happen

        is_pid_mismatch, is_from_eta = False, True

        # Particle has no MC match
        if prt_idx_gen[j] == -1:
            is_signal = False
            is_from_eta = False
            err_type = ErrorType.OTHER_ERROR
        # Particle matches to a MC particle with a different pid
        elif mc_pid[prt_idx_gen[j]] != prt_pid[j]:
            is_signal = False
            is_from_eta = False
            is_pid_mismatch = True
        # Particle is correct pid but didn't come from eta candidate
        elif mc_idx_mom[prt_idx_gen[j]] != i:
            is_signal = False
            is_from_eta = False
        
        # Background
        if not is_from_eta:
            # Classify PID mismatch type
            if is_pid_mismatch:
                # MUON_PID_MISMATCH condition
                if prt_pid[j] == -13:
                    err_type = ErrorType.MUP_PID_MISMATCH
                    dimu_mismatch[0] = True
                # MUON_PID_MISMATCH condition
                elif prt_pid[j] == 13:
                    err_type = ErrorType.MUM_PID_MISMATCH
                    dimu_mismatch[1] = True
                # PHOTON_PID_MISMATCH condition
                elif prt_pid[j] == 22:
                    err_type = ErrorType.PHOTON_PID_MISMATCH
                # Some other error which should be studied
                else:
                    err_type = ErrorType.OTHER_ERROR
            # Classify daughter error type
            else:
                if prt_pid[j] == -13:
                    err_type = ErrorType.MUP_COMBINATORICS
                    dimu_err[0] = True
                elif prt_pid[j] == 13:
                    err_type = ErrorType.MUM_COMBINATORICS
                    dimu_err[1] = True
                elif prt_pid[j] == 22:
                    err_type = ErrorType.PHOTON_COMBINATORICS
                else: 
                    err_type = ErrorType.OTHER_ERROR
        # Correctly matched dtr of signal candidate
        else: err_type = None

        # Add daughter to list
        try: 
            dtrs.append(Daughter(prt_pid=prt_pid[j],
                prt_idx_gen=prt_idx_gen[j],
                mc_pid=mc_pid[prt_idx_gen[j]],
                mc_idx_mom=mc_idx_mom[prt_idx_gen[j]],
                err_type=err_type))
        # Handles any indexing issues which can arise for various reasons
        except IndexError: 
            dtrs.append(Daughter(prt_pid=prt_pid[j],
                prt_idx_gen=prt_idx_gen[j],
                mc_pid=None,
                mc_idx_mom=None,
                err_type=err_type))
            logging.warning(f'Event {entryIdx}, Candidate {i} has incomplete daughter info.')

    # Return Candidate object with all reco-to-gen matching and background
    # classification
    return Candidate(is_sig=is_signal,
                     evt=entryIdx,
                     can_idx=i,
                     dtrs=dtrs,
                     has_dimu_mismatch=all(dimu_mismatch),
                     has_dimu_err=all(dimu_err))


# ──────────────────────────────────────────────────────────────────────────────

def run_event_loop(args):
    """
    Loop over events in the input ROOT file, classify candidates as signal or
    background, determine error types for background candidates, and return as
    a list of Candidate objects for analytics.

    All branches are read once via a single bulk uproot call, then converted
    to plain Python lists -- classify_candidate's per-candidate branching
    logic stays a Python loop either way, so this only replaces PyROOT's
    per-entry tree.GetEntry() calls with pre-loaded in-memory data.
    """
    with uproot.open(args.infile) as f:
        branches = f['tree'].arrays(
            ['tag_pid', 'prt_pid', 'prt_idx_gen', 'prt_idx_mom',
             'mc_pid', 'mc_idx_mom', 'prt_pnn_mu'],
            library='ak',
        )

    # Explicitly define types for easier handling
    tag_pid_evts = ak.to_list(branches['tag_pid'])
    prt_pid_evts = ak.to_list(branches['prt_pid'])
    prt_idx_gen_evts = ak.to_list(branches['prt_idx_gen'])
    prt_idx_mom_evts = ak.to_list(branches['prt_idx_mom'])
    mc_pid_evts = ak.to_list(branches['mc_pid'])
    mc_idx_mom_evts = ak.to_list(branches['mc_idx_mom'])
    probnn_mu_evts = ak.to_list(branches['prt_pnn_mu'])

    # List of Candidate objects
    candidates = []

    # Event loop
    for entryIdx in range(len(tag_pid_evts)):
        # Print status every 500,000 events
        check_interval = 500000
        if entryIdx % check_interval == 0 and entryIdx > 0:
            logging.info(f'Processed {entryIdx:,d} events...')

        # Reconstructed particle information
        tag_pid = [int(pid) for pid in tag_pid_evts[entryIdx]]
        prt_pid = [int(pid) for pid in prt_pid_evts[entryIdx]]
        # MC-matching index information
        prt_idx_gen = [int(idx) for idx in prt_idx_gen_evts[entryIdx]]
        prt_idx_mom = [int(idx) for idx in prt_idx_mom_evts[entryIdx]]
        # Generator particle information
        mc_pid = [int(pid) for pid in mc_pid_evts[entryIdx]]
        mc_idx_mom = [int(idx) for idx in mc_idx_mom_evts[entryIdx]]
        # PROBNNmu branch
        probnn_mu = [float(prob) for prob in probnn_mu_evts[entryIdx]]

        # Skip empty events
        ntags = len(tag_pid)
        if ntags == 0: continue

        # Get per-candidate-level information and background analysis
        for i in range(ntags):
            if tag_pid[i] != 221: continue  # skip failed reco/non-eta candidates
            candidate = classify_candidate(entryIdx, i, prt_pid, prt_idx_gen,
                                           prt_idx_mom, mc_pid, mc_idx_mom,
                                           probnn_mu, args)
            if candidate is not None: candidates.append(candidate)
    return candidates


# ──────────────────────────────────────────────────────────────────────────────

def get_analytics(candidates: list[Candidate]) -> Analytics:
    """
    Get analytics from list of Candidate objects, including counts, error rates,
    and list of PID mismatches ranked by frequency for each particle type.
    Returns an Analytics dataclass object containing all relevant info for text
    and JSON output.
    """
    # List of all error types
    ERROR_TYPES = list(ErrorType)
    # Counters
    err_counters = {err: 0 for err in ERROR_TYPES}
    mup_mismatches, mum_mismatches, pho_mismatches, other_mismatches = [], [], [], []
    mup_mismatch_only_count, mum_mismatch_only_count = 0, 0
    mup_err_only_count, mum_err_only_count = 0, 0

    # Signal (and background) candidates
    nsig = sum(1 for c in candidates if c.is_sig)

    # Count errors
    for can in candidates:
        # Dimuon error conditions are on a per-candidate basis
        if can.has_dimu_mismatch:
            err_counters[ErrorType.DIMUON_PID_MISMATCH] += 1
        if can.has_dimu_err:
            err_counters[ErrorType.DIMUON_COMBINATORICS] += 1
        # Count per-daughter-level errors
        for dtr in can.dtrs:
            # Skip signal, add background
            if dtr.err_type is None: continue
            # TODO: Doesn't this handle everything except dimuon errors? Might
            # be able to simplify this.
            err_counters[dtr.err_type] += 1
            # Append to mismatch lists for pid frequency tables
            # PID mismatch errors
            if dtr.err_type == ErrorType.MUP_PID_MISMATCH:
                mup_mismatches.append(dtr.mc_pid)
                if not can.has_dimu_mismatch:
                    mup_mismatch_only_count += 1
            elif dtr.err_type == ErrorType.MUM_PID_MISMATCH:
                mum_mismatches.append(dtr.mc_pid)
                if not can.has_dimu_mismatch:
                    mum_mismatch_only_count += 1
            # Other errors
            elif dtr.err_type == ErrorType.MUP_COMBINATORICS:
                if not can.has_dimu_err:
                    mup_err_only_count += 1
            elif dtr.err_type == ErrorType.MUM_COMBINATORICS:
                if not can.has_dimu_err:
                    mum_err_only_count += 1
            elif dtr.err_type == ErrorType.PHOTON_PID_MISMATCH:
                pho_mismatches.append(dtr.mc_pid)
            elif dtr.err_type == ErrorType.OTHER_ERROR:
                other_mismatches.append(dtr.mc_pid)

    err_counters[ErrorType.MUP_ONLY_PID_MISMATCH] = mup_mismatch_only_count
    err_counters[ErrorType.MUM_ONLY_PID_MISMATCH] = mum_mismatch_only_count
    err_counters[ErrorType.MUP_ONLY_COMBINATORICS] = mup_err_only_count
    err_counters[ErrorType.MUM_ONLY_COMBINATORICS] = mum_err_only_count

    # Useful values for probability calculations
    ncan = len(candidates)
    denom_mu = max(err_counters[ErrorType.MUP_COMBINATORICS],
                   err_counters[ErrorType.MUM_COMBINATORICS], 
                   1)

    # Return Analytics object with all relevant info for output
    return Analytics(
        ncan=ncan,
        nsig=nsig,
        nbkg=ncan - nsig,
        err_counters=err_counters,
        mup_mismatches=mup_mismatches,
        mum_mismatches=mum_mismatches,
        pho_mismatches=pho_mismatches,
        other_mismatches=other_mismatches,
        prob_dimu_given_mu=err_counters[ErrorType.DIMUON_COMBINATORICS] / denom_mu,
        prob_mup_err=err_counters[ErrorType.MUP_COMBINATORICS] / ncan if ncan else 0,
        prob_mum_err=err_counters[ErrorType.MUM_COMBINATORICS] / ncan if ncan else 0,
        prob_pho_err=err_counters[ErrorType.PHOTON_COMBINATORICS] / ncan if ncan else 0,
        pid_freq={
            'MU+':    Counter(mup_mismatches).most_common(),
            'MU-':    Counter(mum_mismatches).most_common(),
            'PHOTON': Counter(pho_mismatches).most_common(),
            'OTHER':  Counter(other_mismatches).most_common(),
        }
    )


# ──────────────────────────────────────────────────────────────────────────────

def format_pid_freq_table(label, rows: list[tuple[int, int]]) -> str:
    """
    Return a plain text table for (pid, count) rows.
    """
    out = ''
    if not rows: return out

    # Determine column widths based on data
    pid_w = max(len(' PID '), max(len(str(pid)) for pid, _ in rows))
    cnt_w = max(len(' # '), max(len(str(cnt)) for _, cnt in rows))
    
    # Label
    if label == 'PHOTON': out += f'\n{"═" * 4} {label} ' + f'{"═" * 4}\n'
    else: out += f'\n{"═" * 5} {label} ' + f'{"═" * 5}\n'
    # Header
    out += f"┌─{' PID ':>{pid_w}}─┬─{' # ':<{cnt_w}}─┐\n"
    out += f"├─{'─' * pid_w}─┼─{'─' * cnt_w}─┤\n"
    # Data
    for pid, count in rows:
        out += f'│ {pid:>{pid_w}d} │ {count:<{cnt_w}d} │\n'
    out += '└' + '─' * (pid_w + 2) + '┴' + '─' * (cnt_w + 2) + '┘\n'
    return out


# ──────────────────────────────────────────────────────────────────────────────

def format_text(result: Analytics, verbose: bool, candidates: list[Candidate]) -> str:
    """
    Return a formatted string of background analysis results, including counts,
    error rates, and list of PID mismatches ranked by frequency for each
    particle type. If verbose=True, also include candidate-level information and
    error log. Designed for printing to console or writing to text file.
    """
    # Header
    W = 48
    out = '═' * W + '\n'
    out += f"{'Background analysis results':^{W}}\n"
    out += '═' * W + '\n'
    out += '  *_MISMATCH: Daughter has MC match but reco\n    pid does not match gen pid.\n'
    out += '  *_ERROR: Daughter has MC match but reco did\n    not match to candidate gen dtr.\n'
    # High-level counts
    out += '─' * W + '\n'
    out += f'  {"Total candidates processed:":<31} {result.ncan:5d}\n'
    out += f'  {"Total signal candidates:":<31} {result.nsig:5d}\n'
    out += f'  {"Total background candidates:":<31} {result.nbkg:5d}\n'
    # Error counts
    out += '─' * W + '\n'
    out += f"{'Background error counts':^{W}}\n"
    out += '─' * W + '\n'
    for err, count in result.err_counters.items():
        out += f'  {err.name + ":":<31} {count:5d}\n'
    # Error rates
    out += '─' * W + '\n'
    out += f"{'Background error rates':^{W}}\n"
    out += '─' * W + '\n'
    out += f'  P(dimuon error | muon error) = {result.prob_dimu_given_mu:.4f}\n'
    out += f'  P(mu+ error)                 = {result.prob_mup_err:.4f}\n'
    out += f'  P(mu- error)                 = {result.prob_mum_err:.4f}\n'
    out += f'  P(photon error)              = {result.prob_pho_err:.4f}\n'
    out += '─' * W + '\n'
    # PID mismatch frequency tables per daughter
    out += f"{'List of PID mismatches (ranked by frequency)':^{W}}\n"
    out += '─' * W + '\n'
    for label, rows in result.pid_freq.items():
        out += format_pid_freq_table(label, rows)
    out += '─' * W + '\n'

    # Verbose candidate-level information
    if verbose:
        W = 120
        out += '═' * W + '\n'
        out += f"{'Verbose candidate information':^{W}}\n"
        out += '═' * W + '\n'
        # event, cand index, dtr pid, gen idx, mc pid, mc mom idx, error type
        for can in candidates:
            out += f'\nEvent {can.evt}, Candidate {can.can_idx}:\n'
            for dtr in can.dtrs:
                out += f'  Daughter PID {dtr.prt_pid:3d}, Gen idx {dtr.prt_idx_gen:2d}, '
                out += f'MC PID {str(dtr.mc_pid):>5}, MC mom idx {str(dtr.mc_idx_mom):>2}, '
                out += f'Error type: {dtr.err_type}\n'
        out += '─' * W + '\n'

    return out


# ──────────────────────────────────────────────────────────────────────────────

def format_json(result: Analytics, verbose: bool, candidates: list[Candidate]) -> dict:
    """
    Return a formatted JSON dictionary of background analysis results, including
    counts, error rates, and list of PID mismatches ranked by frequency for each
    particle type. If verbose=True, also include candidate-level information and
    error log.
    """
    out = {
        'total_candidates': result.ncan,
        'total_signal':     result.nsig,
        'total_background': result.nbkg,
        'err_counters':     {err.name: count for err, count in result.err_counters.items()},
        'error_rates': {
            'P(dimuon error | muon error)': result.prob_dimu_given_mu,
            'P(mu+ error)':                result.prob_mup_err,
            'P(mu- error)':                result.prob_mum_err,
            'P(photon error)':             result.prob_pho_err,
        },
        'pid_mismatches': {
            label: dict(rows) if rows else None
            for label, rows in result.pid_freq.items()
        }
    }
    if verbose:
        out['candidates'] = [
            {
                'event': can.evt,
                'candidate_index': can.can_idx,
                'has_dimu_mismatch': can.has_dimu_mismatch,
                'has_dimu_err': can.has_dimu_err,
                'daughters': [
                    {'prt_pid': d.prt_pid, 'prt_idx_gen': d.prt_idx_gen,
                     'mc_pid': d.mc_pid, 'mc_idx_mom': d.mc_idx_mom,
                     'err_type': d.err_type.name if d.err_type else None}
                    for d in can.dtrs
                ]
            }
            for can in candidates
        ]
    return out


# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Parse command line arguments
    args = parse_args()
    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO
    # Set basic logging configuration to log to console with specified level and
    # format
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if args.log:
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            # Strip any file type extensions before adding .log to avoid double
            # extensions. If no extension, just add .log to the end.
            filename=os.path.splitext(args.outfile)[0] + '.log',
            filemode='w'
        )
    # Also print to stderr
    logging.getLogger().addHandler(logging.StreamHandler())

    # Set up IO
    # Leading directories for output file prefix created, if any
    # Strip any file extensions if present
    if args.outfile.endswith('.txt') or args.outfile.endswith('.json'):
        args.outfile = os.path.splitext(args.outfile)[0]
    # Create leading directories, not including the file name itself (which 
    # currently does not have an extension, but will be added later for text and
    # JSON output)
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    if args.pnnmu_cut is not None:
        logging.info(f'Applying PROBNNmu cut: {args.pnnmu_cut}')
    logging.info(f'Reading from {args.infile}, writing to {args.outfile}.')

    # Run analysis
    candidates = run_event_loop(args)
    result = get_analytics(candidates)
    # Get results as text string and JSON dictionary
    text_out = format_text(result, args.verbose, candidates)
    json_out = format_json(result, args.verbose, candidates)
    
    # Write results to text and JSON files
    print(text_out)
    with open(args.outfile + '.txt', 'w') as f:
        f.write(text_out)
    with open(args.outfile + '.json', 'w') as f:
        json.dump(json_out, f, indent=2)
    logging.info(f'DONE: Results saved to: {args.outfile}.txt and {args.outfile}.json')

if __name__ == '__main__':
    main()
