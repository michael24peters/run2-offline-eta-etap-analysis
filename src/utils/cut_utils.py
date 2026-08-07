"""
cut_utils.py

Utilities for cutting and selecting events in the analysis pipeline.

An event contains zero or more candidates, and each candidate has 2+ daughters
(depending on the decay). The file does NOT store it that way. Taking a real
event with three candidates:

    tag_dtf_m    = [1080.5, 551.5, 1022.0]     3 candidates
    prt_pnn_mu   = [0.42, 0.947, -100,         9 daughters, one flat list
                    0.42, 0.947, -100,         (all candidates run together)
                    0.42, 0.947, -100]
    prt_idx_mom  = [0,0,0, 1,1,1, 2,2,2]       parent candidate of each daughter

These functions regroup the daughters by candidate, and then the candidates by
event, so the data can be treated as a jagged array of events, each with a
jagged array of candidates, each with a jagged array of daughters. The jagged
structure is preserved by awkward arrays. An example for two events might look
like:

    tag_dtf_m    = [[1080.5, 551.5, 1022.0], [547.9]]   2 events
    prt_pnn_mu   = [[[0.42, 0.947, -100], [0.42, 0.947, -100], [0.42, 0.947, -100]],
                     [[0.42, 0.947, -100]]]              2 events
    prt_idx_mom  = [[[0,0,0], [1,1,1], [2,2,2]], [[0,0,0]]] 2 events

The jagged array would then be indexed like `tag_dtf_m[0][1]` to get the second
candidate of the first event (551.5), or `prt_pnn_mu[1][0][2]` to get the third
daughter of the first candidate of the second event (-100). This is done by
the `_group_daughters()` and `_align_candidates()` functions below.
All cut functions return **per-candidate boolean masks**, shaped like the
reference tag branch (usually `tag_dtf_m`), so a caller can compose them with
plain `&`.

TODO: create generic pid/pnn cuts method instead of just pnn_mu, and make it
configurable in driver_*.py.
"""

import copy
import sys

import awkward as ak
import numpy as np

# Diagnostics accumulated as a run progresses
_REPORT = {}


def report():
    """Diagnostics collected, for writing to JSON (or printing to stderr)."""
    return copy.deepcopy(_REPORT)


# Trigger-on-signal branches OR'd together by `tos_mask`
# TODO: 20260807 tag_hlt1_tos2 for next anaprod
TOS_TRIGGERS = ['tag_l0_tos0', 'tag_l0_tos1', 'tag_hlt1_tos0',
                'tag_hlt1_tos1', 'tag_hlt2_tos0', 'tag_hlt2_tos1']


def read_branches(root_path, branches, tree='tree', entry_stop=None):
    """Read `branches` as awkward arrays.

    entry_stop : optional cap on events read, for quick iteration on large files.
    """
    import uproot
    with uproot.open(root_path) as f:
        return f[tree].arrays(list(branches), library='ak', entry_stop=entry_stop)


def drop_malformed(tree, daughter_fields=(), candidate_fields=(),
                   tag_ref='tag_dtf_m', idx_branch='prt_idx_mom'):
    """
    Drop events that cannot be interpreted, which include:

      - events with no candidates
            tag_dtf_m = []  0 candidates
            -> drop this event
      - events whose `daughter_fields` are inconsistent with idx_branch
            tag_dtf_m = [1080.5, 551.5]          2 candidates
            prt_idx_mom = [0,0,0, 1,1,1, 2,2,2]  3 candidates, mismatch
            -> drop this event
      - events where daughter branch and idx_branch lengths do not match
            prt_idx_mom = [0,0,0, 1,1,1]  2 candidates, 6 daughters
            prt_pnn_mu = [0.42, 0.947]    2 daughters, mismatch
            -> drop this event
      - events where a `candidate_fields` branch holds a partial number of
        entries
            tag_dtf_m = [1080.5, 551.5, 1022.0]  3 candidates
            tag_hlt2_tos0 = [1.0, 0.0, 1.0]      ok, one per candidate
            tag_hlt2_tos0 = []                   ok, line not evaluated
            -> will fill with specified value ( default:-1.0) for each candidate
               value
            tag_hlt2_tos0 = [1.0, 0.0]           bad, only 2 of 3 candidates
            -> drop this event; can't be sure which candidate value is missing
            
    These should never happen, but this is a just-in-case filter to ensure
    everything that happens later can assume a well-formed tree.
    """
    # Initial number of events
    n_before = len(tree)
    # Initial number of candidates
    n_tag = ak.num(tree[tag_ref])

    # Filter out events with no candidates
    has_candidates = n_tag > 0
    # Filter out events where every daughter's parent index does not point to a
    # candidate in the event
    idx_in_range = ak.all(tree[idx_branch] < n_tag, axis=1)
    # Filter out events where a daughter branch and idx_branch disagree on how
    # many daughters the event has
    matched_daughter_lengths = ak.ones_like(n_tag, dtype=bool)
    for field in daughter_fields:
        matched_daughter_lengths = matched_daughter_lengths & (
            ak.num(tree[field]) == ak.num(tree[idx_branch]))
    # Filter out events whose candidate fields are malformed: either a differing
    # length between candidate and tag, or a partial list
    whole_candidate_fields = ak.ones_like(n_tag, dtype=bool)
    partial_by_field = {}
    for field in candidate_fields:
        n_field = ak.num(tree[field])
        # Event's per-candidate branch must either have the same number of
        # entries as the tag branch, or be empty
        whole = (n_field == n_tag) | (n_field == 0)
        partial_by_field[field] = int(ak.sum(has_candidates & ~whole))
        whole_candidate_fields = whole_candidate_fields & whole
    # Combine all four filters into one per-event mask
    keep = (has_candidates & idx_in_range & matched_daughter_lengths
            & whole_candidate_fields)
    n_kept = int(ak.sum(keep))
    # Write report for JSON output
    _REPORT['events'] = {
        'read': n_before,
        'kept': n_kept,
        'dropped_total': n_before - n_kept,
        'dropped_no_candidates': int(ak.sum(~has_candidates)),
        'dropped_idx_out_of_range': int(ak.sum(has_candidates & ~idx_in_range)),
        'dropped_daughter_length_mismatch': int(ak.sum(has_candidates & ~matched_daughter_lengths)),
        'dropped_partial_candidate_branch': int(ak.sum(has_candidates & ~whole_candidate_fields)),
        'partial_by_branch': {f: n for f, n in partial_by_field.items() if n},
    }
    # Print summary
    for entry in _REPORT['events']:
        print(f'[INFO] {entry}: {_REPORT["events"][entry]}')
    return tree[keep]


def _align_candidates(tree, field, tag_ref='tag_dtf_m', fill_value=-1.0):
    """
    Expand a per-candidate branch to exactly one value per candidate, filling
    with `fill_value` for events where the branch is absent (stored as an empty
    list).

    Assumes `drop_malformed()` has already run, so each event's branch holds
    either one entry per candidate or none at all.
    """
    # This is the shape everything must match, e.g., n_tag = [3, 1, 2]
    n_tag = ak.num(tree[tag_ref])
    # Per-candidate flag: does this candidate's event carry the branch at all?
    # e.g., [3==3, 0==1, 2==2] -> [True, False, True]
    present = ak.num(tree[field]) == n_tag
    # broadcast_arrays expands the per-event `present` to per-candidate, e.g.,
    # [[True, True, True], [False], [True, True]]. [0] selects the first of the
    # two broadcasted arrays, which is the one we want. ak.flatten() collapses
    # the jagged structure to a single list of candidates, e.g.,
    # [True, True, True, False, True, True]. ak.to_numpy() turns it into a plain
    # numpy array
    present_per_candidate = ak.to_numpy(
        ak.flatten(ak.broadcast_arrays(present, tree[tag_ref])[0]))

    # Fill with `fill_value`, one per candidate, e.g.,
    # [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    values = np.full(int(ak.sum(n_tag)), fill_value)
    # Overwrite the candidates whose event had a value. If they did not have a
    # value, they remain `fill_value`. In our running example, since the second
    # event had a False, the second event's candidate is left as -1.0, e.g.,
    # [6.42, 0.947, 1.822, -1.0, 0.42, 2.947].
    values[present_per_candidate] = ak.to_numpy(ak.flatten(tree[field]))
    # Unflatten back to the original event structure, e.g.,
    # [[6.42, 0.947, 1.822], [-1.0], [0.42, 2.947]]
    return ak.unflatten(values, n_tag)


def _group_daughters(tree, field, tag_ref='tag_dtf_m', idx_branch='prt_idx_mom'):
    """
    Changes current [event][daughter] structure from ROOT file into jagged
    [event][candidate][daughter].
    """
    # Array of number of candidates per event, e.g., n_tag = [3, 1, 2] for three
    # events with 3, 1, and 2 candidates.
    n_tag = ak.num(tree[tag_ref])
    if len(tree) == 0: return tree[field]

    # Candidates in all preceding events, e.g., offset = [0, 3, 4] for
    # n_tag = [3, 1, 2].
    offset = np.concatenate([[0], np.cumsum(ak.to_numpy(n_tag))[:-1]])
    # e.g., idx_branch = [0,0,0, 1,1,1, 2,2,2], [0,0,0], [1,1,1] for
    # n_tag = [3, 1, 2]; adding the offset makes each candidate unique per
    # event, e.g., tree[idx_branch] + offset = [0,0,0, 1,1,1, 2,2,2], [3,3,3],
    # [4,4,4, 5,5,5]. run_lengths then counts consecutive entries, which in this
    # case tells us the number of daughters per candidate, e.g.,
    # counts = [3,3,3, 3, 3,3].
    counts = ak.run_lengths(ak.flatten(tree[idx_branch] + offset))

    # ak.flatten(tree[field]) is a single list of all the daughters in the
    # event, then ak.unflatten() groups them by candidate, e.g., 
    # [d0,d1,d2], [d3,d4,d5], [d6,d7,d8], [d9,d10,d11], [d12,d13,d14], 
    # [d15,d16,d17].
    per_candidate = ak.unflatten(ak.flatten(tree[field]), counts)
    # Then we unflatten the candidates back to the original event structure,
    # e.g., [[d0,d1,d2], [d3,d4,d5], [d6,d7,d8]], [[d9,d10,d11]], 
    # [[d12,d13,d14], [d15,d16,d17]].
    # Now when we want to access a daughter from a candidate in an event, we can
    # index like tree[field][event][candidate][daughter].
    return ak.unflatten(per_candidate, n_tag)


def tos_mask(tree, triggers=TOS_TRIGGERS, tag_ref='tag_dtf_m'):
    """
    Per-candidate mask: True where the candidate is TOS for at least one of the
    specified triggers.

    Trigger branches hold 0.0/1.0 as float64, so `> 0.5` recovers the boolean.
    They are also aligned first: an HLT2 line that was not evaluated for an
    event is stored as an empty list rather than a row of zeros, and counts as
    not fired.
    """
    n_tag = ak.num(tree[tag_ref])
    n_events = len(tree)
    not_evaluated, ignored = {}, []
    mask = None
    for trigger in triggers:
        # An absent branch contributes nothing to the OR, so it should be
        # reported if a line is absent everywhere and not contributing to the
        # selection.
        n_absent = int(ak.sum(ak.num(tree[trigger]) != n_tag))
        if n_absent:
            not_evaluated[trigger] = n_absent
            if n_absent == n_events: ignored.append(trigger)
        # Makes sure the trigger branch has one entry per candidate, filling
        # absent 
        fired = _align_candidates(tree, trigger, tag_ref) > 0.5
        # Updates mask to True when a trigger has fired. Updates each trigger
        # for mask in place so the final mask is True where any trigger has
        # fired.
        mask = fired if mask is None else (mask | fired)

    # Write report for JSON output
    _REPORT['triggers'] = {
        'requested': list(triggers),
        # Lines fully available everywhere; these are the ones actually driving
        # the OR in every event.
        'always_evaluated': [t for t in triggers if t not in not_evaluated],
        # branch -> number of events where it was absent and read as not fired
        'not_evaluated_in_events': not_evaluated,
        # Absent in every event, so contributing nothing to the selection
        'ignored': ignored,
    }
    # Write report to stderr
    print(f"[INFO] TOS triggers requested: "
          f"{_REPORT['triggers']['requested']}", file=sys.stderr)
    print(f"[INFO] TOS triggers always evaluated: "
          f"{_REPORT['triggers']['always_evaluated']}", file=sys.stderr)
    if not_evaluated:
        print(f"[INFO] TOS triggers not evaluated in some events: "
              f"{_REPORT['triggers']['not_evaluated_in_events']}",
              file=sys.stderr)
    if ignored:
        print(f"[INFO] TOS triggers ignored (absent in all events): {ignored}",
              file=sys.stderr)
    return mask


def pnn_mask(tree, threshold=0.95, abspid=13, pnn_field='prt_pnn_mu',
             pid_field='prt_pid', cut='min'):
    """
    Per-candidate mask: True where every*daughter of the specified pid has 
    pnn > threshold.

    Only daughters selected as the specified pid are considered. For example, a
    pnn_mu cut is only applied to muons, and a candidate with no muons passes
    the cut. The `cut` argument can be 'min' or 'max' to require every daughter
    to exceed or be below the threshold, respectively.

    Arguments:
    ----------
    threshold : float, the pnn cut value
    abspid    : int, the absolute pid of the daughter to cut on (default 13 for muons)
    pnn_field : str, the name of the pnn branch (default 'prt_pnn_mu')
    pid_field : str, the name of the pid branch (default 'prt_pid')
    cut       : str, the type of cut to apply ('min' or 'max', default 'min')

    Returns:
    --------
    mask : np.ndarray, boolean array of shape (n_candidates,) where True
           indicates that the candidate passes the pnn cut.
    """
    pid = _group_daughters(tree, pid_field)
    pnn = _group_daughters(tree, pnn_field)
    # Mask for pid of interest only, e.g., pnn_mu cut is only applied to muons.
    is_muon = abs(pid) == abspid
    # axis=-1 collapses the innermost (daughter) level, so the result is one
    # bool per candidate, shaped like tag_dtf_m. ak.all() returns True for 
    # candidates with no muons, which is what we want.
    if cut == 'min': return ak.all(pnn[is_muon] > threshold, axis=-1)
    elif cut == 'max': return ak.all(pnn[is_muon] < threshold, axis=-1)
    else: raise ValueError(f"Invalid cut {cut!r}. Must be 'min' or 'max'.")


def selection_mask(tree, use_tos=True, pnn_mu_min=0.95, tag_ref='tag_dtf_m'):
    """
    Per-candidate mask combining the standard cuts, so the fit drivers and the
    diagnostic plotters select identically.

    use_tos     : require the candidate to be TOS for at least one trigger
    pnn_mu_min  : require every muon daughter above this; None drops the cut

    With both disabled every candidate passes, which is the "no selection"
    baseline in a selection comparison.
    """
    mask = ak.ones_like(tree[tag_ref], dtype=bool)
    stages = [('all candidates', int(ak.sum(ak.num(mask))))]
    if use_tos:
        mask = mask & tos_mask(tree, tag_ref=tag_ref)
        stages.append(('TOS', int(ak.sum(mask))))
    if pnn_mu_min is not None:
        mask = mask & pnn_mask(tree, pnn_mu_min)
        stages.append((f'pnn_mu > {pnn_mu_min}', int(ak.sum(mask))))

    _REPORT['cuts'] = {
        'tos': use_tos,
        'pnn_mu_min': pnn_mu_min,
        # Candidates surviving after each cut is applied, in order
        'cutflow_candidates': [{'stage': s, 'candidates': n} for s, n in stages],
    }
    return mask


def find_best_candidates(metrics, method='min', target=None):
    """
    Reduce each event to its single best candidate, by `metrics`.

    metrics : per-candidate awkward array to rank on (e.g. tag_dtf_chi2),
              already filtered to the candidates that passed selection
    method  : 'min' / 'max' -- smallest / largest metric wins
              'closest'     -- metric nearest `target` wins
    target  : required for method='closest'

    Returns a per-event index into `metrics` with `keepdims=True`, i.e. a
    one-element list per event rather than a bare number. That keeps the array
    jagged when you index with it, so the result is still one-candidate-per-event
    and every other per-candidate array can be indexed the same way.

    Every event in `metrics` must have at least one candidate -- `ak.argmin` on
    an empty list yields None, which becomes a silently masked numpy entry later.
    Filter with `ak.num(metrics) > 0` first.
    """
    if method == 'min':
        return ak.argmin(metrics, axis=1, keepdims=True)
    if method == 'max':
        return ak.argmax(metrics, axis=1, keepdims=True)
    if method == 'closest':
        if target is None:
            raise ValueError("method='closest' requires a target value")
        return ak.argmin(abs(metrics - target), axis=1, keepdims=True)
    raise ValueError(f"Invalid method {method!r}. Must be 'min', 'max', or 'closest'.")


def transverse_momentum(px, py):
    """pT = sqrt(px^2 + py^2), elementwise over any-shaped px/py fields."""
    return np.sqrt(px**2 + py**2)

def invariant_mass2(px1, py1, pz1, e1, px2, py2, pz2, e2):
    """Compute M^2 of the sum of two 4-vectors."""
    e  = e1  + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    return e**2 - px**2 - py**2 - pz**2
