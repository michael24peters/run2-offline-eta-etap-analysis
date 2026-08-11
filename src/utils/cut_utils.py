"""
cut_utils.py

Utilities for cutting and selecting events in the analysis pipeline.

The structure of the ntuple is assumed to look similar to this, though the names
can be different:

    tag_dtf_m    = [1080.5, 551.5, 1022.0]     3 candidates
    prt_pnn_mu   = [0.42, 0.947, -100,         9 daughters, one flat list
                    0.42, 0.947, -100,         (all candidates run together)
                    0.42, 0.947, -100]
    prt_idx_mom  = [0,0,0, 1,1,1, 2,2,2]       parent candidate of each daughter

These cuts are always at the candidate level, so they return a boolean mask
shaped like the tag reference branch (usually `tag_dtf_m`). The mask be composed
with a plain `&` to combine multiple cuts. The method `to_daughter_mask()` can
be used to apply the candidate-level mask to the daughter level.

TODO: Need similar to_daughter_mask() for generator particles.

The goal is to have a single, consistent way to apply cuts and selections which
can then be applied to any branch at any level easily. This traversal is done
with the index branches, which link the collections together. There are three
index branches:

1. Candidate -> daughter: prt_idx_mom - links each daughter to its mother
   particle.
2. Candidate -> generator particle: prt_idx_gen - links each daughter to its
   generator particle (candidate).
3. Generator particle -> generator mother: mc_idx_mom - links each generator
   daughter to its mother generator particle.

    follow(tree, values, idx_branch)   coarse -> fine  (one target -> many)
    reduce(tree, values, idx_branch)   fine -> coarse  (many -> one, so it
                                                        takes an aggregation)
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

    Arguments:
    ----------
    tree             : awkward array, the event tree
    daughter_fields  : list of str, the per-daughter branches to check
    candidate_fields : list of str, the per-candidate branches to check
    tag_ref          : str, the branch to use as the reference for candidate 
                       structure
    idx_branch       : str, the branch to use as the reference for daughter
                       indices

    Returns:
    --------
    tree : awkward array, the filtered event tree
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

    NOTE: This is not currently used, but it is a useful reference for if you
    ever want to set up this jagged structure in the future and work with that.

    Note this assumes each candidate's daughters are stored contiguously and in
    order (it counts runs of equal prt_idx_mom); reduce() does not.
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

    Arguments:
    ----------
    tree     : awkward array, the event tree
    triggers : list of str, the trigger branches to OR together
    tag_ref  : str, the branch defining the candidate level structure to reduce

    Returns:
    --------
    mask : np.ndarray, boolean array of shape (n_candidates,) where True
           indicates that the candidate is TOS for at least one trigger.
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
             pid_field='prt_pid', cut='min', tag_ref='tag_dtf_m'):
    """
    Per-candidate mask: True where every daughter of the specified pid has 
    pnn > threshold. If a single daughter of the specified pid fails, the
    candidate mask is False.

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
    tag_ref   : str, the branch defining the candidate level to reduce onto

    Returns:
    --------
    mask : np.ndarray, boolean array of shape (n_candidates,) where True
           indicates that the candidate passes the pnn cut.
    """
    pid, pnn = tree[pid_field], tree[pnn_field]   # stay at daughter level
    if cut == 'min': passes = pnn > threshold
    elif cut == 'max': passes = pnn < threshold
    else: raise ValueError(f"Invalid cut {cut!r}. Must be 'min' or 'max'.")

    # A daughter of some other pid isn't subject to this cut, so it can never
    # fail it. Folding that in as "passes OR isn't of this pid" turns the cut
    # into a plain all-reduction, and is what makes a candidate with no daughters
    # of `abspid` pass (an empty segment reduces to the identity, True).
    return reduce(tree, passes | (abs(pid) != abspid), 'prt_idx_mom',
                  how='all', target_ref=tag_ref)


def best_candidate_mask(metrics, eligible, method='min', target=None):
    """
    Per-candidate mask, True only for each event's single best candidate.

    metrics  : per-candidate array to rank on (e.g. tag_dtf_chi2), shaped like
               the tag branch -- the FULL candidate axis, not a pre-filtered one
    eligible : per-candidate mask of the candidates allowed to compete, i.e. the
               output of selection_mask() and friends
    method   : 'min' / 'max' -- smallest / largest metric wins
               'closest'     -- metric nearest `target` wins
    target   : required for method='closest'

    `eligible` is required, because we want to find the best SURVIVING
    candidate. If the best candidate is independently selected and then ANDed
    with the cuts, the best could be a cut candidate and the entire event would
    be dropped when it should have been kept.

    Returns a mask on the full candidate axis, so it composes with `&` like every
    other mask here and indexes any other per-candidate branch directly. Events
    with no eligible candidate simply select nothing -- no need to filter empty
    events first, and no option type, so ak.to_numpy gives a plain ndarray rather
    than a MaskedArray.

    To reach the daughter branches of the winning candidates, pass this mask to
    follow(tree, mask, 'prt_idx_mom').
    """
    # Reduce every method to "smallest score wins", so there is one argmin below.
    if method == 'min':
        score = metrics
    elif method == 'max':
        score = -metrics                    # argmin of the negated metric
    elif method == 'closest':
        if target is None:
            raise ValueError("method='closest' requires a target value")
        score = abs(metrics - target)
    else:
        raise ValueError(f"Invalid method {method!r}. Must be 'min', 'max', or 'closest'.")

    # Push ineligible candidates out of the running before the argmin, so the
    # winner is the best candidate that actually survived the cuts.
    score = ak.where(eligible, score, np.inf)
    idx = ak.argmin(score, axis=1, keepdims=True)
    # An event with no candidates gives None here; fill_none(-1) makes it match
    # nothing rather than silently selecting candidate 0.
    return (ak.local_index(metrics, axis=1) == ak.fill_none(idx, -1)) & eligible


# --- Crossing between collections --------------------------------------------
#
# The ROOT file stores several collections per event -- candidates (tag_*), reco
# daughters (prt_*), and in MC also generator particles (mc_*) and vertices
# (pvr_*/mcpvr_*). They have different lengths and are connected by index
# branches:
#
#   prt_idx_mom : daughter -> candidate (prt_* -> tag_*)
#   prt_idx_gen : daughter -> gen particle (prt_* -> mc_*)
#   mc_idx_mom  : gen particle -> gen mother (mc_* -> mc_*)
#
# Note that there is no direct way to go from candidate -> gen candidate. You
# have to go candidate -> daughter -> gen daughters -> gen candidate. This is
# because MC generator particles are stored in a single collection per event,
# not split into tags (candidates) and particles (daughters) like the reco side.
# 
# To go from one level to another, we follow the index branches. The operations
# are:
# 
#   1. Candidate level mask to the daughter level mask
#   2. Daughter level mask to generator daughter level mask
#   3. Candidate level mask to generator candidate-level mask (composite)
#   ---
#   4. Daughter level mask to the candidate level mask
#   5. Generator daughter level mask to daughter level mask
#   6. Generator level candidate mask to candidate level mask (composite)
#   7. Generator daughter level mask to generator candidate level mask
#
# These processes should be deterministic, so each conversion should be
# exactly invertible. follow() goes along the index branch, and reduce() goes
# against it. Composite conversion are done by chaining the two functions
# together.

def follow(tree, values, idx_branch):
    """
    Coarse -> fine. Re-index `values` down to the level of `idx_branch`.

    `values` must be indexed by whatever `idx_branch` points INTO, and the result
    comes back at the level of `idx_branch` itself.

    Example: follow(tree, tag_dtf_m, 'prt_idx_mom') takes the per-candidate
    tag_dtf_m and returns a per-daughter array. I/O might look something like
    this:

    tag_dtf_m = [1080.5, 551.5, 1022.0]     3 candidates
    prt_idx_mom = [0,0,0, 1,1,1, 2,2,2]       9 daughters
    follow(tree, tag_dtf_m, 'prt_idx_mom') = [1080.5, 1080.5, 1080.5, 551.5, 
                                              551.5, 551.5, 1022.0, 1022.0, 
                                              1022.0]
    
    follow() can also be used on mask to propagate the selection down to the
    finer level. For example, I/O might look like this:

    mask = [True, False, True]  3 candidates
    prt_idx_mom = [0,0,0, 1,1,1, 2,2,2]  9 daughters
    follow(tree, mask, 'prt_idx_mom') = [True, True, True, False, False, False,
                                         True, True, True]
    """
    return values[ak.values_astype(tree[idx_branch], np.int64)]


# Value each aggregation returns for a target entry that nothing points at, e.g.
# a candidate with no daughters. These are the identity elements, so an empty
# segment behaves like an empty ak.all()/ak.any()/sum.
_REDUCE_IDENTITY = {'all': True, 'any': False, 'sum': 0.0,
                    'min': np.inf, 'max': -np.inf}


def reduce(tree, values, idx_branch, how='all', target_ref='tag_dtf_m'):
    """
    Fine -> coarse. Aggregate `values` from the level of `idx_branch` onto the
    level `idx_branch` points into.

    The inverse of follow(). Many source entries map to one target entry, so an
    aggregation is required rather than optional:

        how='all' / 'any'  : bool over the segment   (values must be boolean)
        how='sum'          : sum over the segment    (bools sum to a count)
        how='min' / 'max'  : extremum over the segment

    values     : array at the source level, same jagged shape as `idx_branch`
    target_ref : any branch whose per-event length defines the TARGET collection.
                 'tag_dtf_m' for daughters -> candidates; for daughters -> gen
                 particles (idx_branch='prt_idx_gen') pass an mc_* branch.

    Aggregation is done flat, over a global target id built from the index branch
    plus a per-event offset.

        # every muon daughter of the candidate passes the PID cut
        reduce(tree, (pnn > 0.95) | (abs(pid) != 13), 'prt_idx_mom', how='all')
        # candidate has exactly one photon daughter
        reduce(tree, pid == 22, 'prt_idx_mom', how='sum') == 1
    """
    if how not in _REDUCE_IDENTITY:
        raise ValueError(f"Invalid how {how!r}. Must be one of "
                         f"{sorted(_REDUCE_IDENTITY)}.")

    n_target = ak.num(tree[target_ref])
    n_source = ak.num(tree[idx_branch])
    # Running total of target entries, so every candidate in the file gets a
    # unique id and the whole thing reduces in one flat pass.
    starts = np.concatenate([[0], np.cumsum(ak.to_numpy(n_target))])
    n_total = int(starts[-1])
    if n_total == 0: return ak.unflatten(np.empty(0, dtype=bool), n_target)

    # Local index -> global id, by adding the event's own starting offset.
    ids = (ak.to_numpy(ak.flatten(ak.values_astype(tree[idx_branch], np.int64)))
           + np.repeat(starts[:-1], ak.to_numpy(n_source)))
    vals = ak.to_numpy(ak.flatten(values))

    if how == 'all':
        # No failing source entry anywhere in the segment.
        out = np.bincount(ids[~vals], minlength=n_total) == 0
    elif how == 'any':
        out = np.bincount(ids[vals], minlength=n_total) > 0
    elif how == 'sum':
        out = np.bincount(ids, weights=vals.astype(float), minlength=n_total)
    else:
        ufunc = np.minimum if how == 'min' else np.maximum
        out = np.full(n_total, _REDUCE_IDENTITY[how], dtype=float)
        ufunc.at(out, ids, vals)

    return ak.unflatten(out, n_target)


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
