"""
cut_utils.py

Read a ROOT ntuple, build per-candidate selection masks, and move those masks
between the collections the file stores.

The ntuple keeps several collections per event. They have different lengths and
are stored flat, one list per event, with no nesting:

    tag_dtf_m    = [1080.5, 551.5, 1022.0]     3 candidates      (tag_*)
    prt_pnn_mu   = [0.42, 0.94, 0.42,          6 daughters, all candidates
                    0.95, 0.31, 0.88]          run together      (prt_*)
    prt_idx_mom  = [0, 0, 1, 1, 2, 2]          parent candidate of each daughter
    mc_pid       = [221.0, -13.0, 13.0]        3 generator particles (mc_*)

Everything here is a mask, and every mask is shaped like the collection it
belongs to. Masks on the same collection compose with plain `&` and `|`. There
is deliberately no cut-specific helper beyond `tos_mask` -- a PID cut, a
kinematic cut and a multiplicity cut are all just comparisons on a branch,
moved to the level you want with `follow` / `reduce`:

    # every muon daughter of the candidate has pnn_mu > 0.95
    muon_ok = (tree['prt_pnn_mu'] > 0.95) | (abs(tree['prt_pid']) != 13)
    pnn = reduce(tree, muon_ok, 'prt_idx_mom', how='all')

    # every daughter has pT > 500 MeV
    fast = reduce(tree, kinematics.pt(tree['prt_px'], tree['prt_py']) > 500,
                  'prt_idx_mom', how='all')

    # candidate has exactly one photon daughter
    one_gamma = reduce(tree, tree['prt_pid'] == 22, 'prt_idx_mom', how='sum') == 1

    keep = tos & pnn & fast & one_gamma

Crossing between collections is done with the index branches that link them:

    prt_idx_mom : daughter       -> candidate      (prt_* -> tag_*)
    prt_idx_gen : daughter       -> gen particle   (prt_* -> mc_*)
    mc_idx_mom  : gen particle   -> gen mother     (mc_* -> mc_*)

    follow(tree, array, idx_branch)   coarse -> fine  (one source -> many)
    reduce(tree, array, idx_branch)   fine -> coarse  (many -> one, so it
                                                       takes an aggregation)
    valid(tree, idx_branch)           mask of entries whose index points
                                      somewhere; AND it in before traversing an
                                      index branch that can point nowhere

There is no direct candidate -> gen candidate link: go candidate -> daughter ->
gen daughter -> gen candidate. MC generator particles are one flat collection
per event, not split into tags and daughters like the reco side.

NOTE: `mc_idx_mom` is -1 for a particle with no mother, but passing in -1 
silently indexes backwards into the previous event, so make sure to AND with
`valid(tree, 'mc_idx_mom')` to ignore invalid indices before traversal.

The typical pipeline, with TOS_LINES = {'L0': [...], 'HLT1': [...], 'HLT2': [...]}
(TOS on at least one line at every level, see tos_mask):

    tree, report = drop_malformed(read_branches(infile, BRANCHES))
    tos, _       = tos_mask(tree, TOS_LINES)
    keep         = tos & reduce(tree, muon_ok, 'prt_idx_mom', how='all')
    best         = best_candidate_mask(tree, 'tag_dtf_chi2', keep)
    masses       = values(tree, 'tag_dtf_m', best)   # flat numpy, ready to fit
"""

import awkward as ak
import numpy as np
import collections


def read_branches(root_path, branches, tree='tree', entry_stop=None):
    """Read `branches` as awkward arrays.

    entry_stop : optional, specifies number of events to read (Default: all).
    """
    import uproot
    with uproot.open(root_path) as f:
        return f[tree].arrays(list(branches), library='ak', entry_stop=entry_stop)


def _prefix(branch):
    """Collection a branch belongs to: 'prt_idx_mom' -> 'prt_'.

    The ntuple names every branch after its collection (tag_, prt_, mc_, pvr_,
    mcpvr_, evt_), which is what lets drop_malformed work out for itself which
    branches live at which level.
    """
    return branch.split('_')[0] + '_'


def drop_malformed(tree, tag_ref='tag_dtf_m', idx_branch='prt_idx_mom',
                   daughter_fields=None, candidate_fields=None):
    """
    Drop events that cannot be interpreted, which include:

      - events with no candidates
            tag_dtf_m = []  0 candidates
            -> drop this event
      - events whose daughter fields are inconsistent with idx_branch
            tag_dtf_m = [1080.5, 551.5]          2 candidates
            prt_idx_mom = [0,0,0, 1,1,1, 2,2,2]  3 candidates, mismatch
            -> drop this event
      - events where a daughter branch and idx_branch lengths do not match
            prt_idx_mom = [0,0,0, 1,1,1]  2 candidates, 6 daughters
            prt_pnn_mu = [0.42, 0.947]    2 daughters, mismatch
            -> drop this event
      - events where a candidate field holds a partial number of entries
            tag_dtf_m = [1080.5, 551.5, 1022.0]  3 candidates
            tag_hlt2_tos0 = [1.0, 0.0, 1.0]      ok, one per candidate
            tag_hlt2_tos0 = []                   ok, line not evaluated
            -> align() will fill this in later
            tag_hlt2_tos0 = [1.0, 0.0]           bad, only 2 of 3 candidates
            -> drop this event; can't be sure which candidate value is missing

    These should never happen, but this is a just-in-case filter to ensure
    everything that happens later can assume a well-formed tree.

    By default EVERY branch in the tree sharing idx_branch's prefix is checked as
    a daughter field, and every branch sharing tag_ref's prefix as a candidate
    field. That is the point: forgetting to list a branch here is how a ragged
    event survives to blow up inside reduce() much later. Pass the lists
    explicitly only to check less than everything.

    Arguments:
    ----------
    tree             : awkward array, the event tree
    tag_ref          : str, the branch defining the candidate level
    idx_branch       : str, the branch defining the daughter level
    daughter_fields  : list of str, or None to use every `idx_branch`-prefixed
                       branch in the tree
    candidate_fields : list of str, or None to use every `tag_ref`-prefixed
                       branch in the tree

    Returns:
    --------
    (tree, report) : the filtered event tree, and a dict of what was dropped

    Dict Keys:
    ----------
    read : int, number of events read
    kept : int, number of events kept
    dropped_total : int, number of events dropped
    dropped_no_candidates : int, number of events dropped for having no
                            candidates
    dropped_idx_out_of_range : int, number of events dropped for having an index
                               that is out of range
    dropped_daughter_length_mismatch : int, number of events dropped for having
                                       a mismatch in the length of daughter
                                       branches
    dropped_partial_candidate_branch : int, number of events dropped for having
                                       a candidate branch with a partial list
    partial_by_branch : dict, candidate branch -> number of events dropped for
                        that branch
    daughter_fields_checked : list of str, the daughter fields checked
    candidate_fields_checked : list of str, the candidate fields checked
    """
    if daughter_fields is None:
        daughter_fields = [f for f in tree.fields
                           if f.startswith(_prefix(idx_branch))]
    if candidate_fields is None:
        candidate_fields = [f for f in tree.fields
                            if f.startswith(_prefix(tag_ref))]

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

    report = {
        'read': n_before,
        'kept': n_kept,
        'dropped_total': n_before - n_kept,
        'dropped_no_candidates': int(ak.sum(~has_candidates)),
        'dropped_idx_out_of_range': int(ak.sum(has_candidates & ~idx_in_range)),
        'dropped_daughter_length_mismatch': int(ak.sum(has_candidates & ~matched_daughter_lengths)),
        'dropped_partial_candidate_branch': int(ak.sum(has_candidates & ~whole_candidate_fields)),
        'partial_by_branch': {f: n for f, n in partial_by_field.items() if n},
        'daughter_fields_checked': list(daughter_fields),
        'candidate_fields_checked': list(candidate_fields),
    }
    return tree[keep], report


def align(tree, field, tag_ref='tag_dtf_m', fill_value=-1.0):
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
    out = np.full(int(ak.sum(n_tag)), fill_value)
    # Overwrite the candidates whose event had a value. If they did not have a
    # value, they remain `fill_value`. In our running example, since the second
    # event had a False, the second event's candidate is left as -1.0, e.g.,
    # [6.42, 0.947, 1.822, -1.0, 0.42, 2.947].
    out[present_per_candidate] = ak.to_numpy(ak.flatten(tree[field]))
    # Unflatten back to the original event structure, e.g.,
    # [[6.42, 0.947, 1.822], [-1.0], [0.42, 2.947]]
    return ak.unflatten(out, n_tag)


def tos_mask(tree, levels, tag_ref='tag_dtf_m'):
    """
    Per-candidate mask: True where the candidate is TOS on at least one line at
    every trigger level -- AND across levels, OR within a level.

        levels = {'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],
                  'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],
                  'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1']}
        TOS = (l0_tos0 | l0_tos1) & (hlt1_tos0 | hlt1_tos1) & (hlt2_tos0 | hlt2_tos1)

    The only cut-specific helper here, because it is valid to have an absent
    trigger line: a line that was never evaluated is stored as an empty list,
    and must read as "did not fire" without changing the mask out of candidate
    shape.

    Arguments:
    ----------
    tree    : awkward array, the event tree
    levels  : dict, trigger level -> list of that level's TOS branches
    tag_ref : str, the branch defining the candidate level

    Returns:
    --------
    (mask, report) : per-candidate boolean mask, and a dict keyed by level
                     recording which lines were actually available to drive
                     that level's OR and how many candidates pass

    Report keys, per level:
    -----------------------
    requested                  : list of str, the level's TOS branches
    always_evaluated           : branches present in every event
    not_evaluated_in_events    : branch -> number of events where it was absent
                                 and read as not fired
    ignored                    : branches absent in every event, so contributing
                                 nothing to the level's OR
    candidates_tos             : int, candidates TOS at this level
    candidates_tos_cumulative  : int, candidates TOS at this level and every
                                 level before it
    """
    if not isinstance(levels, dict):
        raise TypeError("levels must be a dict of trigger level -> list of TOS "
                        "branches, e.g. {'L0': [...], 'HLT1': [...], 'HLT2': [...]}")

    n_tag = ak.num(tree[tag_ref])
    n_events = len(tree)
    mask = ak.ones_like(tree[tag_ref], dtype=bool)
    report = {}
    for level, triggers in levels.items():
        if not triggers:
            raise ValueError(f'trigger level {level!r} has no TOS branches')
        not_evaluated, ignored = {}, []
        level_tos = ak.zeros_like(tree[tag_ref], dtype=bool)
        for trigger in triggers:
            # An absent branch contributes nothing to the OR, so it should be
            # reported if a line is absent everywhere and not contributing to
            # the selection.
            n_absent = int(ak.sum(ak.num(tree[trigger]) != n_tag))
            if n_absent:
                not_evaluated[trigger] = n_absent
                if n_absent == n_events: ignored.append(trigger)
            # Makes sure the trigger branch has one entry per candidate,
            # filling absent lines with -1.0, which reads as not fired.
            level_tos = level_tos | (align(tree, trigger, tag_ref) > 0.5)
        mask = mask & level_tos

        report[level] = {
            'requested': list(triggers),
            'always_evaluated': [t for t in triggers if t not in not_evaluated],
            'not_evaluated_in_events': not_evaluated,
            'ignored': ignored,
            'candidates_tos': int(ak.sum(level_tos)),
            'candidates_tos_cumulative': int(ak.sum(mask)),
        }
    return mask, report


def best_candidate_mask(tree, metric_branch, eligible):
    """
    Per-candidate mask, True only for each event's single best candidate.

    tree          : awkward array, the event tree
    metric_branch : str, per-candidate branch to rank on (e.g. 'tag_dtf_chi2'),
                    shaped like the tag branch -- the FULL candidate axis, not a
                    pre-filtered one
    eligible      : per-candidate mask of the candidates allowed to compete, i.e.
                    the combined cuts

    The smallest metric wins. Every call site ranks on tag_dtf_chi2, where
    smaller is better; to rank on something where larger is better, negate it
    at the call site.

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
    # Get metrics from the tree. Smallest wins, so the metric IS the score.
    metrics = tree[metric_branch]

    # -------------------------------------------------------------------------
    # Example:
    #     metrics  = [[0.42, 0.95, 1.82], [7.0, 2.0], [4.0, 9.0], []]
    #     eligible = [[T,    F,    T   ], [F,   F  ], [T,   T  ], []]
    #                  event 0             event 1     event 2    event 3
    #                  normal (mixed)      NONE pass   all pass   no candidates
    # -------------------------------------------------------------------------

    # Push ineligible candidates out of the running before the argmin, so the
    # winner is the best candidate that actually survived the cuts.
    #     score -> [[0.42, inf, 1.82], [inf, inf], [4.0, 9.0], []]
    score = ak.where(eligible, metrics, np.inf)

    # argmin(axis=1) asks each event separately "which position in YOUR list
    # holds the smallest score?". keepdims=True keeps the result nested 
    # one-per-event as [[0]] rather than flattening to [0].
    #     idx -> [[0], [0], [0], [None]]
    #             ^     ^          ^
    #             |     |          event 3 has nothing to pick, so None.
    #             |     event 1 is ALL inf -- argmin returns 0, but we still
    #             |     want to exclide ineligible events, so we make sure to
    #             |     include `& eligible` in the return below.
    #             event 0: 0.42 < 1.82, so candidate 0 wins.
    idx = ak.argmin(score, axis=1, keepdims=True)

    # local_index(axis=1) numbers each candidate by its position within its own
    # event -- essentially an index of its indices.
    #     ak.local_index(metrics, axis=1) -> [[0, 1, 2], [0, 1], [0, 1], []]
    #
    # fill_none(idx, -1) turns event 3's None into -1, which no position can
    # equal.
    #     fill_none(idx, -1) -> [[0], [0], [0], [-1]]
    #
    # Comapring the two looks like:
    #     [[0], [0], [0], [-1]] == [[0, 1, 2], [0, 1], [0, 1], []]
    #
    # The two sides do not have the same shape, but awkward broadcasts them to
    # the same shape, so the comparison is done by event. The broadcasted
    # comparison looks like:
    #     [[0, 0, 0], [0, 0], [0, 0], [-1, -1, -1]] == [[0, 1, 2], [0, 1], [0, 1], []]
    # 
    # The result is True at exactly one position per event, where the
    # candidate's local index matches the event's winning index. The result
    # looks like:
    #    [[T, F, F], [T, F], [T, F], []]
    #                 ^
    #                 event 1's phantom winner, an ineligible candidate
    #
    # We get rid of this ineligible candidate with the final `& eligible`:
    #    return [[T, F, F], [F, F], [T, F], []]
    return (ak.local_index(metrics, axis=1) == ak.fill_none(idx, -1)) & eligible


def values(tree, branch, mask):
    """
    `branch` under `mask`, as a flat numpy array -- the awkward/numpy boundary.

    This is the last step of a selection: everything upstream is jagged and
    per-event, everything downstream (histograms, fits, plots) is a flat array
    of numbers.

        masses = values(tree, 'tag_dtf_m', best)   # one mass per surviving event

    `mask` must be shaped like `branch`, which is what a mask from this module
    always is.

    Example:
        tag_dtf_m = [[1080.5, 551.5, 1022.0], [1050.0]]
        best      = [[False, True, False], [True]]
        values(...) = [551.5, 1050.0]
    
    Example:
        branch = [[True, False, False], [True]]
        mask = [[False, True, False], [True]]
        values(...) = [False, True]   
    """
    return ak.to_numpy(ak.flatten(tree[branch][mask]))


# --- Crossing between collections --------------------------------------------

def follow(tree, array, idx_branch):
    """
    Coarse -> fine. Re-index `array` down to the level of `idx_branch`.

    `array` must be indexed by whatever `idx_branch` points INTO, and the result
    comes back at the level of `idx_branch` itself.

    NOTE: a negative index (mc_idx_mom uses -1 for "no mother") is NOT caught
    here -- numpy-style it silently wraps to the last entry. AND in
    valid(tree, idx_branch) to handle those.
    """
    # The index branch is stored as float64, but indexes are just integers, so
    # convert them to in64.
    idx = ak.values_astype(tree[idx_branch], np.int64)

    # Broadcast array to the same jagged structure as idx_branch
    return array[idx]


def valid(tree, idx_branch):
    """
    Mask of source entries that point to something. Index branches with no mother or
    no matching MC particle get a value of `-1`. A `-1` is a valid index to numpy
    (it means the last entry), so it is never caught as an error -- follow() reads
    the wrong particle, and reduce() writes into the previous event. Mask it off
    with this instead.


    An example demonstrating the need for this function:

        mc_idx_mom       [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        follow(mc_pid)   [[13.0, 221.0, 221.0], [13.0, 221.0, 221.0]]
                           ^^^^ the eta's "mother" came back as a mu-

    Example:

        mc_idx_mom = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -1.0]]
        valid(...) = [[F,    T,   T  ], [F,    T,   T,   F   ]]
                       ^                 ^               ^
                       the eta           the eta         a muon produced
                       (no mother)                       outside the decay
        
        From here, follow() can be used safely, and looks like:
        follow(mc_pid, mc_idx_mom)[valid(mc_idx_mom)] = 
            = [[221.0, 221.0], [221.0, 221.0]]
    """

    return tree[idx_branch] >= 0


def reduce(tree, array, idx_branch, how='all', target_ref='tag_dtf_m'):
    """
    Fine -> coarse. Aggregate `array` from the level of `idx_branch` onto the
    level `idx_branch` points into.

    This is essentially the inverse of follow(), except in going from many to
    one, an aggregation method must be specified.

        how='all' / 'any'  : bool over the segment
        how='sum'          : sum over the segment
        how='min' / 'max'  : min/max over the segment

    A target entry that nothing points at (a candidate with no daughters) gets
    the identity of its aggregation: True for 'all', False for 'any', 0 for
    'sum', +inf for 'min', -inf for 'max'.

    array      : at the source level, same jagged shape as `idx_branch`
    target_ref : any branch whose per-event length defines the TARGET collection.

    Example 1: Every muon daughter of the candidate that passes the PID cut

        tree['prt_pid'] = [[-13.0, 13.0, -13.0, 13.0], [-13.0, 13.0, 211.0, 22.0]]
        tree['prt_pnn_mu'] = [[0.99, 0.96, 0.42, 0.05], [0.88, 0.91, 0.95, 0.01]]
        array = (tree['prt_pnn_mu'] > 0.95) & (abs(tree['prt_pid']) == 13)
            = [[T, T, F, F], [F, F, T, F]] & [[T, T, T, T], [T, T, F, F]] =
                                   = [[T, T, F, F], [F, F,  F,  F]]
        idx_branch = 'prt_idx_mom' = [[0, 0, 1, 1], [0, 0, -1, -1]]
        reduce(tree, array, 'prt_idx_mom', how='all') 
            = [[T, F], [F]]
    
    Example 2: Candidate has exactly one photon daughter
        tree['prt_pid'] = [[-13.0, 13.0, -13.0, 22.0], [-13.0, 13.0, 211.0, 22.0, 22.0]]
        array = (tree['prt_pid'] == 22)  = [[F, F,  F, T], [F, F,  F,  T,  T]]
        idx_branch = 'prt_idx_mom'       = [[0, 0, -1, 0], [0, 0, -1, -1, 0]]
        reduce(tree, array, 'prt_idx_mom', how='sum', target_ref='prt_pid') = [[1, 0], [1]] == 1
            = [[T, F], [T]]

    Example 3: Number of daughters per generator candidate
        tree['mc_pid'] = [[221.0, -13.0, 13.0], [221.0, -13.0, 13.0, 211.0, 22.0]]
        array = ak.ones_like(tree['mc_pid'], dtype=bool) = [[T, T, T], [T, T, T, T, T]]
        idx_branch = 'mc_idx_mom' = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -1.0, -1.0]]
        reduce(tree, array, 'mc_idx_mom', how='sum', target_ref='mc_pid') = [[2], [2]]

    Note that for all of these example, it is assumed the negative indices are
    masked off with valid(tree, idx_branch) before reducing. Otherwise, the
    negative would get counted as a daughter of the last candidate, which
    confuses the result. In example 3, the sum of the daughters of the eta is 2,
    but the negative indices if counted would make the second event's eta have
    4 daughters instead of 2.

    NOTE: as in follow(), a negative index is not caught -- it wraps into the
    previous event's segment. AND in valid(tree, 'mc_idx_mom') before reducing
    along mc_idx_mom. This must be handled by the caller.
    """
    if how not in ('all', 'any', 'sum', 'min', 'max'):
        raise ValueError(f"Invalid how {how!r}. Must be 'all', 'any', 'sum', 'min' or 'max'.")

    # -------------------------------------------------------------------------
    # Example: 2 events, reducing daughters onto candidates with how='all':
    #
    #     tag_dtf_m   = [[10.0, 20.0], [30.0]]  3 candidates -- target structure
    #     prt_idx_mom = [[0, 0, 1, 1], [0, 0]]  6 daughters  -- source indices
    #     array       = [[T, T, F, T], [T, F]]  source values to reduce
    #
    # Wanted: one bool per CANDIDATE -- [[True, False], [False]]. Candidate 0 of
    # event 0 has daughters [T, T] so it passes; candidate 1 has [F, T] so it
    # fails; event 1's only candidate has [T, F] so it fails, i.e., True only if
    # all daughters are True.
    # -------------------------------------------------------------------------

    n_target = ak.num(tree[target_ref])   # candidates per event -> [2, 1]
    n_source = ak.num(tree[idx_branch])   # daughters  per event -> [4, 2]

    # --- Step 1 ---
    # Lay the events' candidates end to end and record the starting offsets:
    #     cumsum([2, 1])            = [2, 3]
    #     concatenate([0], ...)     = [0, 2, 3] <- starts
    # Read it as: event 0's candidates occupy global ids 0-1, event 1's occupy
    # global id 2, and the file holds starts[-1] = 3 candidates in total.
    starts = np.concatenate([[0], np.cumsum(ak.to_numpy(n_target))])
    n_total = int(starts[-1])  # -> 3 (in the example above)
    # Return empty array of correct shape if there are no candidates
    if n_total == 0: return ak.unflatten(np.empty(0, dtype=bool), n_target)

    # --- Step 2 ---
    # Rewrite each daughter's LOCAL index as a GLOBAL id. A daughter's stored
    # index is local to its own event ("candidate 0 of MY event"). Adding that
    # event's starting offset makes it globally unique.
    #     flatten(prt_idx_mom)          = [0, 0, 1, 1, 0, 0]   local
    #     repeat(starts[:-1], n_source) = [0, 0, 0, 0, 2, 2]   each event's offset,
    #                                                          repeated once per
    #                                                          daughter it owns
    #     ids                           = [0, 0, 1, 1, 2, 2]   global
    local = ak.to_numpy(ak.flatten(ak.values_astype(tree[idx_branch], np.int64)))

    # Some index branches can point nowhere: mc_idx_mom = -1 for a particle with
    # no mother. Those entries must be dropped BEFORE the offset is added,
    # because afterwards a -1 is indistinguishable from a real id. Take a second
    # example, 2 events of 3 generator particles each:
    #
    #     mc_idx_mom = [[-1, 0, 0], [-1, 0, 0]]
    #     flattened  = [-1, 0, 0, -1, 0, 0]
    #     offsets    = [ 0, 0, 0,  3, 3, 3]
    #     naive ids  = [-1, 0, 0,  2, 3, 3]
    #                    ^          ^
    #                    |          event 1's -1 became 2 -- a REAL id, pointing
    #                    |          at event 0's third particle. Event 1's result
    #                    |          gets written into event 0, silently.
    #                    rejected outright by np.bincount below
    #
    # Filtering on the LOCAL index catches both, since only -1 is negative there:
    #     points_somewhere = [F, T, T, F, T, T]
    #     ids              = [0, 0, 3, 3]
    #
    # This is why no caller needs to AND in valid() before reducing: a motherless
    # particle simply never enters the aggregation. (follow() is different -- its
    # output must keep the shape of the index branch, so it has no entry to drop
    # and the caller does still need valid() there.)
    #
    # Mask for non-negative local indices.
    points_somewhere = local >= 0
    ids = (local + np.repeat(starts[:-1], ak.to_numpy(n_source)))[points_somewhere]
    vals = ak.to_numpy(ak.flatten(array))[points_somewhere]   # -> [T, T, F, T, T, F]

    # --- Step 3 ---
    # Now `ids` and `vals` are two plain numpy arrays of the same length, lined
    # up daughter by daughter:
    #
    #     daughter #   0  1  2  3  4  5
    #     ids        [ 0, 0, 1, 1, 2, 2 ]   which candidate
    #     vals       [ T, T, F, T, T, F ]   did it pass?
    #
    # np.bincount(x, minlength=n) counts how many times each value 0..n-1 appears
    # in x. So bincount over a subset of `ids` counts, per candidate, how many of
    # its daughters were in that subset. Picking the subset picks the question.
    if how == 'all':
        # "every daughter passed" is the same as "no daughter failed", so count
        # the FAILING daughters per candidate and require zero.
        #
        # ~vals inverts the pass flags -- these are the failures:
        #     vals   = [T, T, F, T, T, F]
        #     ~vals  = [F, F, T, F, F, T]
        #
        # ids[~vals] is BOOLEAN MASK indexing; it keeps the entries of `ids` 
        # whose mask entry is True, and discards the rest.
        #
        #     position:  0     1     2     3     4     5
        #     ids:       0     0     1     1     2     2
        #     ~vals:     F     F     T     F     F     T
        #                skip  skip  KEEP  skip  skip  KEEP
        #                            |                 |
        #                            v                 v
        #     ids[~vals] =         [ 1,                2 ]
        #
        # Read this as "a daughter of candidate 1 failed, and a daughter of
        # candidate 2 failed". It is a list of complaints, one per failing
        # daughter, labelled by which candidate the complaint is against. The
        # same candidate appears once per failing daughter, so bincount tallies
        # the complaints:
        #     np.bincount([1, 2], minlength=3) = [0, 1, 1]
        #                                         ^  ^  ^
        #                                         |  |  candidate 2: 1 complaint
        #                                         |  candidate 1: 1 complaint
        #                                         candidate 0: no complaints
        #     == 0                             = [True, False, False]
        #
        # A candidate that nothing points at collects no complaints, so it comes
        # back True -- the identity of an empty all(). That is deliberate (it is
        # what lets a candidate with no muons pass a muon cut) but it is a trap
        # when propagating a selection upward: use how='any' there, whose
        # identity is False.
        out = np.bincount(ids[~vals], minlength=n_total) == 0
    elif how == 'any':
        # Mirror image: count the PASSING daughters per candidate, require >= 1.
        #     ids[vals] = [0, 0, 1, 2]   (positions 0, 1, 3, 4 survive the mask)
        #     bincount  = [2, 1, 1]
        #     > 0       = [True, True, True]
        out = np.bincount(ids[vals], minlength=n_total) > 0
    elif how == 'sum':
        # `weights` adds vals[k] to bucket ids[k] instead of adding 1, so booleans
        # come out as a count and floats as a total.
        #     -> [2.0, 1.0, 1.0]
        out = np.bincount(ids, weights=vals.astype(float), minlength=n_total)
    elif how == 'min':
        # Start every candidate at +inf (identity for min), then lower it to
        # each of its daughters' values in turn.
        out = np.full(n_total, np.inf)
        np.minimum.at(out, ids, vals.astype(float))
    else:  # 'max'
        # Mirror image of 'min': start at -inf, raise to each daughter's value.
        out = np.full(n_total, -np.inf)
        np.maximum.at(out, ids, vals.astype(float))

    # --- Step 4: back to the per-event structure -----------------------------
    # `out` is flat, one entry per candidate in the file, in the same global-id
    # order the ids were built from. Cutting it back up with the same n_target
    # restores the jagged shape the caller expects, so the result composes with
    # `&` against any other per-candidate mask.
    #     unflatten([True, False, False], [2, 1]) = [[True, False], [False]]
    return ak.unflatten(out, n_target)

def decay_mask(tree, mother_pid, daughter_pids):
    """
    Per generator mother mask: True for each particle with mother_pid whose
    daughters are exactly the specified daughter_pids (could be in any order but
    has the exact same multiplicity).

    Works for any decay by changing one list:

        decay_mask(tree, 221, [-13, 13])            eta -> mu mu
        decay_mask(tree, 221, [-13, 13, 22])        eta -> mu mu gamma
        decay_mask(tree, 221, [-13, -13, 13, 13])   eta -> 4mu
        decay_mask(tree, 221, [-13, 13, -11, 11])   eta -> mu mu e e

    Generator level only. Mothers and daughters are read from the same branch
    (mc_pid), which works only because mc_idx_mom points back into its own
    collection. The reco side has no equivalent -- prt_idx_mom crosses from
    prt_* to tag_* -- so there is nothing to parameterise here.
    
    We ask counting questions to deduce the decay, rather than trying to match
    the daughters and resulting in combinatorial problems.

    Start with a simple example:

        mc_pid     = [[221, -13, 13, 22]]
        mc_idx_mom = [[ -1,   0,  0,  0]]

        reduce(pid == -13, how='sum') -> [[1, 0, 0, 0]]   one mu+ daughter
        reduce(pid ==  13, how='sum') -> [[1, 0, 0, 0]]   one mu- daughter
        reduce(pid ==  22, how='sum') -> [[1, 0, 0, 0]]   one gamma daughter
        reduce(ones,       how='sum') -> [[3, 0, 0, 0]]   three daughters total

    This tells us everything we need to know: a mother has exactly one mu+, one
    mu-, and one gamma, and no other daughters.
    """
    # Get the mother particles
    mothers = tree['mc_pid'] == mother_pid

    # Get the daughters of those mothers, and check that each mother has the
    # right number of daughters and the right PIDs.
    # Get daughters
    # Example: n_all = [[3, 0, 0, 0]] for the eta -> mu mu gamma example above.
    n_all = reduce(tree, ak.ones_like(tree['mc_pid'], dtype=bool), 'mc_idx_mom',
                   how='sum', target_ref='mc_pid')
    # Check that each mother has the correct number of daughters
    ok = mothers & (n_all == len(daughter_pids))

    # Now check PIDs using a Counter, which turns [-13, 13, -13, 13] into
    # {13: 2, -13: 2}.
    # code == PID
    # want == number of each PID
    # mc_pid                       [[221,-13,13], [221,-13,-13,13], [221,-13,13,22]]
    # decay_mask(221,[-13,13])     [[T,  F,  F ], [F,  F,  F,  F ], [F,  F,  F, F ]]
    # decay_mask(221,[-13,13,22])  [[F,  F,  F ], [F,  F,  F,  F ], [T,  F,  F, F ]]
    # decay_mask(221,[-13,-13,13]) [[F,  F,  F ], [T,  F,  F,  F ], [F,  F,  F, F ]]
    for code, want in collections.Counter(daughter_pids).items():
        ok = ok & (reduce(tree, tree['mc_pid'] == code, 'mc_idx_mom', how='sum',
                          target_ref='mc_pid') == want)
    return ok
