"""
Unit tests for cut_utils.follow().

follow() is coarse -> fine: it re-indexes an array down to the level of an index
branch. One source entry can be handed to many targets, so nothing is
aggregated and the result always has the shape of the index branch.
"""

import awkward as ak
import numpy as np
from conftest import make_tree

import cut_utils


# --- basic behaviour ---------------------------------------------------------

def test_values_are_broadcast_to_daughters(reco):
    """Each daughter receives the value of the candidate it points at."""
    out = cut_utils.follow(reco, reco['tag_dtf_m'], 'prt_idx_mom')
    assert out.tolist() == [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0]]


def test_propagates_a_mask(reco):
    """The usual use: push a candidate selection down to the daughter level."""
    best = ak.Array([[True, False], [True]])
    out = cut_utils.follow(reco, best, 'prt_idx_mom')
    assert out.tolist() == [[True, True, False, False], [True, True]]


def test_result_has_the_shape_of_the_index_branch(reco):
    """Not the shape of the input array -- follow() moves to the finer level."""
    out = cut_utils.follow(reco, reco['tag_dtf_m'], 'prt_idx_mom')
    assert ak.num(out).tolist() == ak.num(reco['prt_idx_mom']).tolist()
    assert ak.num(out).tolist() != ak.num(reco['tag_dtf_m']).tolist()


def test_float_index_branch_is_cast_to_int():
    """Index branches are float64 in the ntuple; 2.0 must become 2, not 1 or 3."""
    tree = make_tree(tag=[[0.0, 1.0, 2.0]], idx=[[2.0, 0.0, 1.0]])
    out = cut_utils.follow(tree, ak.Array([[100.0, 200.0, 300.0]]), 'idx')
    assert out.tolist() == [[300.0, 100.0, 200.0]]


# --- the properties that make it safe ----------------------------------------

def test_indices_never_cross_an_event_boundary(reco):
    """Event 1's index 0 must pick event 1's candidate, not event 0's.

    This is the whole reason follow() can be a plain `array[idx]`: awkward
    applies a jagged index to a jagged array event by event.
    """
    out = cut_utils.follow(reco, reco['tag_dtf_m'], 'prt_idx_mom')
    assert out[1].tolist() == [30.0, 30.0]      # not [10.0, 10.0]


def test_repeated_indices_duplicate_the_source():
    """One candidate should feed an arbitrary number of daughters."""
    tree = make_tree(tag=[[7.0]], idx=[[0, 0, 0, 0, 0]])
    out = cut_utils.follow(tree, tree['tag'], 'idx')
    assert out.tolist() == [[7.0, 7.0, 7.0, 7.0, 7.0]]


def test_unreferenced_source_entries_are_simply_dropped():
    """A candidate no daughter points at contributes nothing. No error."""
    tree = make_tree(tag=[[1.0, 2.0, 3.0]], idx=[[0, 2]])
    out = cut_utils.follow(tree, tree['tag'], 'idx')
    assert out.tolist() == [[1.0, 3.0]]          # 2.0 never appears


def test_out_of_order_indices():
    """Daughters need not be stored grouped or sorted by their mother."""
    tree = make_tree(tag=[[10.0, 20.0]], idx=[[1, 0, 1, 0]])
    out = cut_utils.follow(tree, tree['tag'], 'idx')
    assert out.tolist() == [[20.0, 10.0, 20.0, 10.0]]


def test_event_with_no_daughters():
    """An event could have a candidate with no daughters, or only one daughter."""
    tree = make_tree(tag=[[1.0], [2.0]], idx=[[0], []])
    out = cut_utils.follow(tree, tree['tag'], 'idx')
    assert out.tolist() == [[1.0], []]


def test_empty_tree():
    """An empty tree should return an empty array."""
    tree = make_tree(tag=[[]], idx=[[]])
    assert cut_utils.follow(tree, tree['tag'], 'idx').tolist() == [[]]


# --- crossing collections, not just candidate -> daughter --------------------

def test_follow_works_on_any_index_branch():
    """follow() knows nothing about "mothers" or "daughters" -- it moves a value
    along whatever index branch it is given. Here prt_idx_gen carries a
    generator-level mask down to the reco level.
    """
    tree = make_tree(
        mc_pid=[[221, -13, 13, 22]],
        prt_idx_gen=[[1, 2, 3]],          # 3 reco daughters -> gen particles
    )
    is_photon = tree['mc_pid'] == 22
    out = cut_utils.follow(tree, is_photon, 'prt_idx_gen')
    assert out.tolist() == [[False, False, True]]
    assert ak.num(out).tolist() == [3]     # reco-shaped, not gen-shaped


# --- valid tests --------------------------------------------------------------

def test_negative_index_fails_silently(gen):
    """-1 means "no mother", but follow() does numpy-style indexing, so it
    reads the LAST entry instead of raising.

    This is asserted deliberately: the docstring promises this behaviour and
    tells callers to use valid(). If follow() ever starts raising or masking
    instead, this test should fail so the docs get updated with it.

    So in a way, we're actually checking for the presence of an issue: follow()
    is not a safe operation on its own, and the caller must use valid() to
    ignore the wrapped values.
    """
    out = cut_utils.follow(gen, gen['mc_pid'], 'mc_idx_mom')
    # Entry 0 is the eta, whose mc_idx_mom is -1 -> wraps to mc_pid[-1] == 13.
    assert out[0].tolist() == [13.0, 221.0, 221.0]



def test_valid_flags_the_invalid_entries(gen):
    """Should flag -1 entries as invalid, and everything else as valid."""
    ok = cut_utils.valid(gen, 'mc_idx_mom')
    assert ok.tolist() == [[False, True, True], [False, True, True]]



def test_valid_preserves_shape(gen):
    """valid() itself never changes any shape -- it returns a mask shaped like
    the index branch, exactly as follow() does. Only USING it as a boolean index
    collapses anything, which is what the next three tests separate.
    """
    ok = cut_utils.valid(gen, 'mc_idx_mom')
    assert ak.num(ok).tolist() == ak.num(gen['mc_idx_mom']).tolist()
    raw = cut_utils.follow(gen, gen['mc_pid'], 'mc_idx_mom')
    assert ak.num(raw).tolist() == ak.num(gen['mc_idx_mom']).tolist()


def test_valid_as_a_boolean_index_shrinks_the_array(gen):
    """DIAGNOSTIC USE ONLY. Indexing with valid() drops the bad entries, but the
    result is no longer aligned with the generator particles -- 3 entries become
    2 -- so it can no longer be ANDed against any other generator-level mask.
    Fine for "what mother PIDs appear?", useless for building a selection.
    """
    raw = cut_utils.follow(gen, gen['mc_pid'], 'mc_idx_mom')
    ok = cut_utils.valid(gen, 'mc_idx_mom')
    assert raw[ok].tolist() == [[221.0, 221.0], [221.0, 221.0]]
    assert ak.num(raw[ok]).tolist() == [2, 2]        # was [3, 3]


def test_valid_anded_into_a_boolean_result_keeps_shape(gen):
    """THE PATTERN TO USE for masks. The wrapped entry is forced False and the
    array stays generator-shaped, so it still composes with `&`.
    """
    is_eta = gen['mc_pid'] == 221
    mother_is_eta = cut_utils.follow(gen, is_eta, 'mc_idx_mom')
    ok = cut_utils.valid(gen, 'mc_idx_mom')
    out = mother_is_eta & ok
    assert out.tolist() == [[False, True, True], [False, True, True]]
    assert ak.num(out).tolist() == ak.num(gen['mc_pid']).tolist()


def test_valid_with_ak_where_keeps_shape_for_values(gen):
    """THE PATTERN TO USE for value arrays, where `&` is not available. ak.where
    substitutes a sentinel instead of dropping the entry, so position k still
    means generator particle k.
    """
    raw = cut_utils.follow(gen, gen['mc_pid'], 'mc_idx_mom')
    ok = cut_utils.valid(gen, 'mc_idx_mom')
    out = ak.where(ok, raw, np.nan)
    assert ak.num(out).tolist() == [3, 3]
    assert np.isnan(out[0][0])
    assert out[0][1] == 221.0


def test_wrapped_value_is_wrong_not_merely_missing(gen):
    """Why the guard is not optional: the unguarded entry is not a null, it is a
    plausible-looking PID belonging to a different particle in the same event."""
    raw = cut_utils.follow(gen, gen['mc_pid'], 'mc_idx_mom')
    assert raw[0][0] == 13.0                        # reads mc_pid[-1], the mu-
    assert raw[0][0] in gen['mc_pid'][0].tolist()   # indistinguishable by value


def test_valid_is_all_true_when_nothing_points_nowhere(reco):
    ok = cut_utils.valid(reco, 'prt_idx_mom')
    assert ak.all(ok)


# --- round trip with reduce() ------------------------------------------------

def test_follow_then_reduce_any_recovers_the_mask(reco):
    """follow() down and reduce(how='any') back up is the identity, provided
    every candidate has at least one daughter."""
    original = ak.Array([[True, False], [True]])
    down = cut_utils.follow(reco, original, 'prt_idx_mom')
    back = cut_utils.reduce(reco, down, 'prt_idx_mom', how='any',
                            target_ref='tag_dtf_m')
    assert back.tolist() == original.tolist()


def test_follow_then_reduce_all_also_recovers_the_mask(reco):
    """'all' works too here, because follow() gives every daughter of a
    candidate the same value -- so all and any agree."""
    original = ak.Array([[True, False], [True]])
    down = cut_utils.follow(reco, original, 'prt_idx_mom')
    back = cut_utils.reduce(reco, down, 'prt_idx_mom', how='all',
                            target_ref='tag_dtf_m')
    assert back.tolist() == original.tolist()


# --- every use of follow() this framework has ---------------------------------
#
# follow() knows nothing about "mothers" or "daughters". It moves a value down
# an index branch: one source entry can be handed to many targets, and the
# result always has the shape of the index branch. These are the four index
# branches it gets pointed at in practice.

'''
The five follow() use cases:

--------------------------------------------------------------------------------

1. Candidate mass → each daughter. Input is candidate-shaped, output daughter-
   shaped:

in   [[10.0, 20.0], [30.0]]
out  [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0]]

Candidate 0's value goes to daughters 0 and 1; candidate 1's to daughters 2 and
3.

--------------------------------------------------------------------------------

2. Best candidate → its daughters:

best         [[False, True], [True]]
dmask        [[False, False, True, True], [True, True]]
prt_pid[dmask]  [[-13.0, 13.0], [-13.0, 13.0]]     SHRUNK 4 -> 2

follow keeps the shape (4 daughters); the subsequent [dmask] is what shrinks it.

--------------------------------------------------------------------------------

3. Generator level PID of each reco daughter.

mc_pid       [221, -13, 13, 211]
prt_idx_gen  [1, 3]
  daughter 0 -> gen 1 -> -13
  daughter 1 -> gen 3 -> 211   a PION reconstructed as a muon
follow       [[-13.0, 211.0]]

--------------------------------------------------------------------------------

4. Generator mother PID, on two decays:

mc_pid      [221, -13, 13, 22, 221, -13, 13, 22]
mc_idx_mom  [ -1,   0,  0,  0,  -1,   4,  4,  4]
follow      [ 22, 221,221,221,  22, 221,221,221]   idx 0 and 4 are -1 wraps
valid       [  F,   T,  T,  T,   F,   T,  T,  T]
mother[ok]  [221, 221, 221, 221, 221, 221]         6 real mother PIDs

--------------------------------------------------------------------------------

5. Round trip. reduce up, follow back down:

per-daughter  [[True, True, False, True], [True, True]]
reduce all    [[True, False], [True]]          candidate 1 had a False
follow back   [[True, True, False, False], [True, True]]

Daughter 3 was True on the way up and comes back False — it's been overwritten
by its candidate's verdict. That's the lossy step, but this is okay because the
candidate-level verdict is what matters in this context.
'''

def test_use_candidate_kinematics_at_the_daughter_level(reco):
    """1. tag_* -> prt_* : give every daughter its own candidate's mass, so a
    per-daughter cut can reference a candidate-level quantity."""
    out = cut_utils.follow(reco, reco['tag_dtf_m'], 'prt_idx_mom')
    assert out.tolist() == [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0]]


def test_use_best_candidate_selects_its_daughters(reco):
    """2. tag_* -> prt_* : the commonest call in the drivers -- reach the
    daughter branches of the winning candidate."""
    best = ak.Array([[False, True], [True]])
    dmask = cut_utils.follow(reco, best, 'prt_idx_mom')
    assert dmask.tolist() == [[False, False, True, True], [True, True]]
    kept = reco['prt_pid'][dmask]
    assert kept.tolist() == [[-13.0, 13.0], [-13.0, 13.0]]


def test_use_truth_pid_of_each_reco_daughter():
    """3. mc_* -> prt_* : pull a generator-level quantity down onto the reco
    daughters that matched it. The basis of every MC-truth cross-check."""
    tree = make_tree(mc_pid=[[221, -13, 13, 211]], prt_idx_gen=[[1, 3]])
    truth = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    assert truth.tolist() == [[-13.0, 211.0]]


def test_use_stray_flag_pushed_onto_reco_daughters(gen_with_strays):
    """3b. mc_* -> prt_* with a MASK: "is my generator match a stray?", the
    entry point for asking why background candidates get reconstructed."""
    tree = ak.with_field(gen_with_strays, ak.Array([[1, 2, 4, 5]]), 'prt_idx_gen')
    signal = cut_utils.decay_mask(tree, 221, [-13, 13, 22])
    stray = ~cut_utils.valid(tree, 'mc_idx_mom') & ~signal
    out = cut_utils.follow(tree, stray, 'prt_idx_gen')
    # Reco daughters 0,1 matched real decay products; 2,3 matched strays.
    assert out.tolist() == [[False, False, True, True]]


def test_use_generator_mother_pid(two_gen_candidates):
    """4. mc_* -> mc_* : "what is my mother?" -- the only self-referential
    index branch, and the one that needs valid() because of the -1."""
    mother = cut_utils.follow(two_gen_candidates, two_gen_candidates['mc_pid'],
                              'mc_idx_mom')
    ok = cut_utils.valid(two_gen_candidates, 'mc_idx_mom')
    assert mother[ok].tolist() == [[221.0] * 6]


def test_use_broadcast_a_candidate_verdict_back_down(reco):
    """5. Round trip: reduce() up to a per-candidate verdict, follow() it back
    down so each daughter carries its candidate's outcome."""
    vals = ak.Array([[True, True, False, True], [True, True]])
    verdict = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='all',
                               target_ref='tag_dtf_m')
    out = cut_utils.follow(reco, verdict, 'prt_idx_mom')
    assert out.tolist() == [[True, True, False, False], [True, True]]
