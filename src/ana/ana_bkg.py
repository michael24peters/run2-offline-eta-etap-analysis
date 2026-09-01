"""
ana_bkg.py

Vectorized rewrite of bkg.py: classify eta candidates as signal or background
by matching reco daughters to MC truth, and tabulate error categories. Uses
cut_utils (awkward masks) instead of a per-event Python loop, and groups
daughters by prt_idx_mom instead of assuming a fixed daughter count per
candidate.

Usage:
    python ana_bkg.py input.root [outfile out/output.txt] [--verbose]
                      [--pnnmu_cut 0.5] [--mother_pid 221]
                      [--daughter_pids -13 13]
"""

import argparse
import json
import os
import sys
from collections import Counter
from enum import Enum

import awkward as ak
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils

BRANCHES = ['tag_pid', 'prt_pid', 'prt_idx_gen', 'prt_idx_mom', 'prt_pnn_mu',
            'mc_pid', 'mc_idx_mom']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('outfile', default='out/output.txt',
                        help='Output file prefix, writes .txt and .json')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Include a per-candidate, per-daughter dump of '
                             'background candidates')
    parser.add_argument('--pnnmu_cut', type=float, default=None,
                        help='PROBNNmu cut applied to muon daughters; a '
                             'candidate is dropped if any muon daughter fails it')
    parser.add_argument('--mother_pid', type=int, default=221,
                        help='Mother PID for the signal decay (default: 221, eta)')
    parser.add_argument('--daughter_pids', nargs='+', type=int, default=[-13, 13],
                        help='Daughter PIDs for the signal decay '
                             '(default: -13 13, eta -> mu+ mu-)')
    return parser.parse_args()


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


# ──────────────────────────────────────────────────────────────────────────────

def build_masks(tree, args):
    """
    Vectorized reco-to-gen matching. Every entry is a mask on some collection,
    following cut_utils' style throughout.

    A daughter is correctly matched iff it has a generator match, that match's
    PID agrees with the daughter's own reco PID, AND that match's mother is
    the real signal mother (found via decay_mask) -- not merely "some
    generator particle at position i", which is what bkg.py's classify_candidate
    got wrong for the second and later candidates in multi-candidate events.
    """
    signal_mother = cut_utils.decay_mask(tree, args.mother_pid, args.daughter_pids)

    has_gen_match = cut_utils.valid(tree, 'prt_idx_gen')
    pid_at_gen = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    mc_mom_at_gen = cut_utils.follow(tree, tree['mc_idx_mom'], 'prt_idx_gen')
    pid_match = has_gen_match & (pid_at_gen == tree['prt_pid'])

    # Two-hop: daughter -> its gen particle (prt_idx_gen) -> that particle's
    # mother (mc_idx_mom) -> is that mother the real signal eta?
    gen_is_child_of_signal = (cut_utils.valid(tree, 'mc_idx_mom')
                              & cut_utils.follow(tree, signal_mother, 'mc_idx_mom'))
    is_child_of_signal_mother = has_gen_match & cut_utils.follow(
        tree, gen_is_child_of_signal, 'prt_idx_gen')

    is_signal_dtr = pid_match & is_child_of_signal_mother

    mup, mum, pho = tree['prt_pid'] == -13, tree['prt_pid'] == 13, tree['prt_pid'] == 22
    other = ~(mup | mum | pho)

    no_match = ~has_gen_match
    mismatch = has_gen_match & ~pid_match
    combinatoric = has_gen_match & pid_match & ~is_child_of_signal_mother

    mup_pid_mismatch, mum_pid_mismatch = mismatch & mup, mismatch & mum
    pho_pid_mismatch, other_pid_mismatch = mismatch & pho, mismatch & other
    mup_combinatoric, mum_combinatoric = combinatoric & mup, combinatoric & mum
    pho_combinatoric, other_combinatoric = combinatoric & pho, combinatoric & other
    # OTHER_ERROR is a catch-all across all three root causes, matching bkg.py
    other_error = no_match | other_pid_mismatch | other_combinatoric

    if args.pnnmu_cut is not None:
        pnn_ok = (tree['prt_pnn_mu'] >= args.pnnmu_cut) | (abs(tree['prt_pid']) != 13)
    else:
        pnn_ok = ak.ones_like(tree['prt_pid'], dtype=bool)
    eligible = ((tree['tag_pid'] == args.mother_pid)
               & cut_utils.reduce(tree, pnn_ok, 'prt_idx_mom', how='all',
                                  target_ref='tag_pid'))
    eligible_dtr = cut_utils.follow(tree, eligible, 'prt_idx_mom')

    is_signal = eligible & cut_utils.reduce(tree, is_signal_dtr, 'prt_idx_mom',
                                            how='all', target_ref='tag_pid')

    def per_candidate(mask):
        return cut_utils.reduce(tree, mask, 'prt_idx_mom', how='any',
                                target_ref='tag_pid')

    mup_mism_c, mum_mism_c = per_candidate(mup_pid_mismatch), per_candidate(mum_pid_mismatch)
    has_dimu_mismatch = eligible & mup_mism_c & mum_mism_c
    mup_mism_only = eligible & mup_mism_c & ~has_dimu_mismatch
    mum_mism_only = eligible & mum_mism_c & ~has_dimu_mismatch

    mup_comb_c, mum_comb_c = per_candidate(mup_combinatoric), per_candidate(mum_combinatoric)
    has_dimu_err = eligible & mup_comb_c & mum_comb_c
    mup_err_only = eligible & mup_comb_c & ~has_dimu_err
    mum_err_only = eligible & mum_comb_c & ~has_dimu_err

    return dict(
        eligible=eligible, eligible_dtr=eligible_dtr, is_signal=is_signal,
        has_gen_match=has_gen_match, pid_match=pid_match,
        is_child_of_signal_mother=is_child_of_signal_mother,
        pid_at_gen=pid_at_gen, mc_mom_at_gen=mc_mom_at_gen,
        mup_pid_mismatch=mup_pid_mismatch, mum_pid_mismatch=mum_pid_mismatch,
        pho_pid_mismatch=pho_pid_mismatch, other_pid_mismatch=other_pid_mismatch,
        mup_combinatoric=mup_combinatoric, mum_combinatoric=mum_combinatoric,
        pho_combinatoric=pho_combinatoric, other_combinatoric=other_combinatoric,
        other_error=other_error,
        has_dimu_mismatch=has_dimu_mismatch, has_dimu_err=has_dimu_err,
        mup_mism_only=mup_mism_only, mum_mism_only=mum_mism_only,
        mup_err_only=mup_err_only, mum_err_only=mum_err_only,
    )


# ──────────────────────────────────────────────────────────────────────────────

def get_analytics(tree, m):
    """
    Counts, error rates, and PID-mismatch frequency tables, computed directly
    from the boolean masks in `m` -- no per-candidate Python object needed.
    """
    def count(mask):
        return int(ak.sum(ak.flatten(mask & m['eligible_dtr'])))

    def count_can(mask):
        return int(ak.sum(ak.flatten(mask)))

    def mismatched_pids(mask):
        # The generator PID a mismatched daughter actually turned out to be.
        # Restricted to has_gen_match (folded into `mask` already), so this
        # never falls into the -1 -> wraps-to-last-entry trap cut_utils warns
        # about -- unlike bkg.py's plain-Python mc_pid[prt_idx_gen[j]] lookup.
        vals = ak.to_numpy(ak.flatten(m['pid_at_gen'][mask & m['eligible_dtr']]))
        return vals.astype(int).tolist()

    ncan = count_can(m['eligible'])
    nsig = count_can(m['is_signal'])

    err_counters = {err: 0 for err in ErrorType}
    err_counters[ErrorType.MUP_PID_MISMATCH] = count(m['mup_pid_mismatch'])
    err_counters[ErrorType.MUM_PID_MISMATCH] = count(m['mum_pid_mismatch'])
    err_counters[ErrorType.PHOTON_PID_MISMATCH] = count(m['pho_pid_mismatch'])
    err_counters[ErrorType.MUP_ERROR] = count(m['mup_combinatoric'])
    err_counters[ErrorType.MUM_ERROR] = count(m['mum_combinatoric'])
    err_counters[ErrorType.PHOTON_ERROR] = count(m['pho_combinatoric'])
    err_counters[ErrorType.OTHER_ERROR] = count(m['other_error'])
    err_counters[ErrorType.DIMUON_PID_MISMATCH] = count_can(m['has_dimu_mismatch'])
    err_counters[ErrorType.DIMUON_ERROR] = count_can(m['has_dimu_err'])
    err_counters[ErrorType.MUP_ONLY_PID_MISMATCH] = count_can(m['mup_mism_only'])
    err_counters[ErrorType.MUM_ONLY_PID_MISMATCH] = count_can(m['mum_mism_only'])
    err_counters[ErrorType.MUP_ONLY_ERROR] = count_can(m['mup_err_only'])
    err_counters[ErrorType.MUM_ONLY_ERROR] = count_can(m['mum_err_only'])

    denom_mu = max(err_counters[ErrorType.MUP_ERROR],
                   err_counters[ErrorType.MUM_ERROR], 1)

    return {
        'ncan': ncan, 'nsig': nsig, 'nbkg': ncan - nsig,
        'err_counters': err_counters,
        'prob_dimu_given_mu': err_counters[ErrorType.DIMUON_ERROR] / denom_mu,
        'prob_mup_err': err_counters[ErrorType.MUP_ERROR] / ncan if ncan else 0,
        'prob_mum_err': err_counters[ErrorType.MUM_ERROR] / ncan if ncan else 0,
        'prob_pho_err': err_counters[ErrorType.PHOTON_ERROR] / ncan if ncan else 0,
        'pid_freq': {
            'MU+': Counter(mismatched_pids(m['mup_pid_mismatch'])).most_common(),
            'MU-': Counter(mismatched_pids(m['mum_pid_mismatch'])).most_common(),
            'PHOTON': Counter(mismatched_pids(m['pho_pid_mismatch'])).most_common(),
            'OTHER': Counter(mismatched_pids(m['other_pid_mismatch'])).most_common(),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────

def _err_label(has_match, pid_ok, from_signal, pid):
    """Per-daughter error label, for the verbose dump only."""
    if not has_match:
        return ErrorType.OTHER_ERROR
    if not pid_ok:
        if pid == -13: return ErrorType.MUP_PID_MISMATCH
        if pid == 13: return ErrorType.MUM_PID_MISMATCH
        if pid == 22: return ErrorType.PHOTON_PID_MISMATCH
        return ErrorType.OTHER_ERROR
    if not from_signal:
        if pid == -13: return ErrorType.MUP_ERROR
        if pid == 13: return ErrorType.MUM_ERROR
        if pid == 22: return ErrorType.PHOTON_ERROR
        return ErrorType.OTHER_ERROR
    return None


def collect_verbose_candidates(tree, m):
    """
    Per-candidate, per-daughter dump for --verbose, restricted to background
    candidates. Subselects just the background EVENTS with awkward fancy
    indexing first, so the plain-Python loop below is bounded by background
    candidates, not the full dataset.
    """
    bg = m['eligible'] & ~m['is_signal']
    n_events = len(tree)
    evt_num = ak.broadcast_arrays(ak.Array(np.arange(n_events)), tree['tag_pid'])[0]

    bg_evt = ak.to_numpy(ak.flatten(evt_num[bg])).astype(int)
    if len(bg_evt) == 0:
        return []
    events = sorted(set(bg_evt.tolist()))

    sub_bg = ak.to_list(bg[events])
    sub_mom = ak.to_list(tree['prt_idx_mom'][events])
    sub_pid = ak.to_list(tree['prt_pid'][events])
    sub_gen = ak.to_list(tree['prt_idx_gen'][events])
    sub_mcpid = ak.to_list(m['pid_at_gen'][events])
    sub_mcmom = ak.to_list(m['mc_mom_at_gen'][events])
    sub_match = ak.to_list(m['has_gen_match'][events])
    sub_pidok = ak.to_list(m['pid_match'][events])
    sub_fromsig = ak.to_list(m['is_child_of_signal_mother'][events])
    sub_dimu_mm = ak.to_list(m['has_dimu_mismatch'][events])
    sub_dimu_err = ak.to_list(m['has_dimu_err'][events])

    out = []
    for k, evt in enumerate(events):
        for can, is_bg in enumerate(sub_bg[k]):
            if not is_bg:
                continue
            daughters = []
            for j, mom in enumerate(sub_mom[k]):
                if mom != can:
                    continue
                match = sub_match[k][j]
                pid = int(sub_pid[k][j])
                err = _err_label(match, sub_pidok[k][j], sub_fromsig[k][j], pid)
                daughters.append({
                    'prt_pid': pid,
                    'prt_idx_gen': int(sub_gen[k][j]),
                    'mc_pid': int(sub_mcpid[k][j]) if match else None,
                    'mc_idx_mom': int(sub_mcmom[k][j]) if match else None,
                    'err_type': err,
                })
            out.append({
                'evt': evt, 'can_idx': can, 'daughters': daughters,
                'has_dimu_mismatch': bool(sub_dimu_mm[k][can]),
                'has_dimu_err': bool(sub_dimu_err[k][can]),
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────

def format_pid_freq_table(label, rows):
    """Return a plain text table for (pid, count) rows."""
    out = ''
    if not rows: return out

    pid_w = max(len(' PID '), max(len(str(pid)) for pid, _ in rows))
    cnt_w = max(len(' # '), max(len(str(cnt)) for _, cnt in rows))

    if label == 'PHOTON': out += f'\n{"═" * 4} {label} ' + f'{"═" * 4}\n'
    else: out += f'\n{"═" * 5} {label} ' + f'{"═" * 5}\n'
    out += f"┌─{' PID ':>{pid_w}}─┬─{' # ':<{cnt_w}}─┐\n"
    out += f"├─{'─' * pid_w}─┼─{'─' * cnt_w}─┤\n"
    for pid, count in rows:
        out += f'│ {pid:>{pid_w}d} │ {count:<{cnt_w}d} │\n'
    out += '└' + '─' * (pid_w + 2) + '┴' + '─' * (cnt_w + 2) + '┘\n'
    return out


def format_text(result, verbose, verbose_candidates):
    W = 48
    out = '═' * W + '\n'
    out += f"{'Background analysis results':^{W}}\n"
    out += '═' * W + '\n'
    out += '  *_MISMATCH: Daughter has MC match but reco\n    pid does not match gen pid.\n'
    out += '  *_ERROR: Daughter has MC match but reco did\n    not match to candidate gen dtr.\n'
    out += '─' * W + '\n'
    out += f'  {"Total candidates processed:":<31} {result["ncan"]:5d}\n'
    out += f'  {"Total signal candidates:":<31} {result["nsig"]:5d}\n'
    out += f'  {"Total background candidates:":<31} {result["nbkg"]:5d}\n'
    out += '─' * W + '\n'
    out += f"{'Background error counts':^{W}}\n"
    out += '─' * W + '\n'
    for err, count in result['err_counters'].items():
        out += f'  {err.name + ":":<31} {count:5d}\n'
    out += '─' * W + '\n'
    out += f"{'Background error rates':^{W}}\n"
    out += '─' * W + '\n'
    out += f'  P(dimuon error | muon error) = {result["prob_dimu_given_mu"]:.4f}\n'
    out += f'  P(mu+ error)                 = {result["prob_mup_err"]:.4f}\n'
    out += f'  P(mu- error)                 = {result["prob_mum_err"]:.4f}\n'
    out += f'  P(photon error)              = {result["prob_pho_err"]:.4f}\n'
    out += '─' * W + '\n'
    out += f"{'List of PID mismatches (ranked by frequency)':^{W}}\n"
    out += '─' * W + '\n'
    for label, rows in result['pid_freq'].items():
        out += format_pid_freq_table(label, rows)
    out += '─' * W + '\n'

    if verbose:
        W2 = 120
        out += '═' * W2 + '\n'
        out += f"{'Verbose candidate information':^{W2}}\n"
        out += '═' * W2 + '\n'
        for can in verbose_candidates:
            out += f"\nEvent {can['evt']}, Candidate {can['can_idx']}:\n"
            for dtr in can['daughters']:
                out += f"  Daughter PID {dtr['prt_pid']:3d}, Gen idx {dtr['prt_idx_gen']:2d}, "
                out += f"MC PID {str(dtr['mc_pid']):>5}, MC mom idx {str(dtr['mc_idx_mom']):>2}, "
                out += f"Error type: {dtr['err_type']}\n"
        out += '─' * W2 + '\n'

    return out


def format_json(result, verbose, verbose_candidates):
    out = {
        'total_candidates': result['ncan'],
        'total_signal': result['nsig'],
        'total_background': result['nbkg'],
        'err_counters': {err.name: count for err, count in result['err_counters'].items()},
        'error_rates': {
            'P(dimuon error | muon error)': result['prob_dimu_given_mu'],
            'P(mu+ error)': result['prob_mup_err'],
            'P(mu- error)': result['prob_mum_err'],
            'P(photon error)': result['prob_pho_err'],
        },
        'pid_mismatches': {
            label: dict(rows) if rows else None
            for label, rows in result['pid_freq'].items()
        },
    }
    if verbose:
        out['candidates'] = [
            {
                'event': can['evt'],
                'candidate_index': can['can_idx'],
                'has_dimu_mismatch': can['has_dimu_mismatch'],
                'has_dimu_err': can['has_dimu_err'],
                'daughters': [
                    {'prt_pid': d['prt_pid'], 'prt_idx_gen': d['prt_idx_gen'],
                     'mc_pid': d['mc_pid'], 'mc_idx_mom': d['mc_idx_mom'],
                     'err_type': d['err_type'].name if d['err_type'] else None}
                    for d in can['daughters']
                ],
            }
            for can in verbose_candidates
        ]
    return out


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.outfile.endswith('.txt') or args.outfile.endswith('.json'):
        args.outfile = os.path.splitext(args.outfile)[0]
    os.makedirs(os.path.dirname(args.outfile) or '.', exist_ok=True)

    print(f'[INFO] Reading from {args.infile}.', file=sys.stderr)
    tree, drop_report = cut_utils.drop_malformed(
        cut_utils.read_branches(args.infile, BRANCHES), tag_ref='tag_pid')
    print(f"[INFO] events: read {drop_report['read']:,d}, kept {drop_report['kept']:,d} "
          f"(dropped {drop_report['dropped_total']:,d})", file=sys.stderr)
    if args.pnnmu_cut is not None:
        print(f'[INFO] Applying PROBNNmu cut: {args.pnnmu_cut}', file=sys.stderr)

    masks = build_masks(tree, args)
    result = get_analytics(tree, masks)
    verbose_candidates = collect_verbose_candidates(tree, masks) if args.verbose else []

    text_out = format_text(result, args.verbose, verbose_candidates)
    json_out = format_json(result, args.verbose, verbose_candidates)

    print(text_out)
    with open(args.outfile + '.txt', 'w') as f:
        f.write(text_out)
    with open(args.outfile + '.json', 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f'[INFO] DONE: results saved to {args.outfile}.txt and {args.outfile}.json',
          file=sys.stderr)


if __name__ == '__main__':
    main()
