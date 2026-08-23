"""
Null / zero-cardinality tests across the whole API.

The "zero, one, many" heuristic: for anything countable, test it with none, with
one, and with several. The one-and-many cases live in the per-function files;
this file sweeps the ZERO end across every entry point, because that is the
column most easily forgotten and the one that tends to raise rather than return
a wrong answer.

There are several distinct kinds of "nothing" in this ntuple, and they are not
interchangeable:

  1. zero events            -- an empty file, or drop_malformed() rejecting all
  2. zero entries per event -- events exist, a collection is empty in them
  3. zero of one collection  -- generator particles but no reco, or vice versa
  4. zero matches            -- the data is fine, nothing satisfies the request
"""

import awkward as ak
import numpy as np
import pytest
from conftest import make_tree, IN

import cut_utils
import fid_reqs


@pytest.fixture
def zero_events():
    """A 0-event tree that still carries its float64 types.

    Built by slicing a populated tree to nothing, which is what uproot returns
    for an empty file and what `tree[keep]` returns when drop_malformed rejects
    every event. Types matter here -- see test_untyped_empty_tree_is_rejected.
    """
    full = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]],
                     mc_px=[[1, 2, 3]], mc_py=[[1, 2, 3]], mc_pz=[[1, 2, 3]],
                     tag_dtf_m=[[550.0]], prt_idx_mom=[[0, 0]],
                     prt_idx_gen=[[1, 2]], prt_pid=[[-13, 13]])
    return full[:0]


# --- 1. zero events ----------------------------------------------------------

def test_follow_on_zero_events(zero_events):
    """Every function must survive an empty file rather than raising, because
    drop_malformed() legitimately produces one."""
    out = cut_utils.follow(zero_events, zero_events['mc_pid'], 'mc_idx_mom')
    assert out.tolist() == []


def test_valid_on_zero_events(zero_events):
    assert cut_utils.valid(zero_events, 'mc_idx_mom').tolist() == []


@pytest.mark.parametrize('how', ['all', 'any', 'sum'])
def test_reduce_on_zero_events(zero_events, how):
    """Exercises the n_total == 0 short circuit for every aggregation."""
    vals = zero_events['mc_pid'] > 0
    out = cut_utils.reduce(zero_events, vals, 'mc_idx_mom', how=how,
                           target_ref='mc_pid')
    assert out.tolist() == []


def test_decay_mask_on_zero_events(zero_events):
    assert cut_utils.decay_mask(zero_events, 221, [-13, 13]).tolist() == []


def test_best_candidate_mask_on_zero_events(zero_events):
    eligible = zero_events['tag_dtf_m'] > 0
    out = cut_utils.best_candidate_mask(zero_events, 'tag_dtf_m', eligible)
    assert out.tolist() == []


def test_values_on_zero_events(zero_events):
    """The awkward -> numpy boundary must give an empty ndarray, not raise, so
    a fit downstream sees len 0 rather than a traceback."""
    out = cut_utils.values(zero_events, 'tag_dtf_m', zero_events['tag_dtf_m'] > 0)
    assert len(out) == 0
    assert isinstance(out, np.ndarray)


def test_fiducial_mask_on_zero_events(zero_events):
    assert fid_reqs.fiducial_mask(zero_events).tolist() == []


def test_fiducial_masks_on_zero_events(zero_events):
    """All six keys must come back empty and same-shaped, so the caller's
    ak.any(..., axis=1) still works."""
    m = fid_reqs.fiducial_masks(zero_events, 221, [-13, 13])
    for name, mask in m.items():
        assert mask.tolist() == [], name
    keep = ak.to_numpy(ak.any(m['in_acceptance'], axis=1))
    assert keep.tolist() == []


def test_drop_malformed_can_reject_every_event():
    """The realistic route to a 0-event tree: every event malformed. The report
    must still be well formed so a driver can log it."""
    tree = make_tree(tag_dtf_m=[[550.0], [549.0]],
                     prt_idx_mom=[[0, 0], [0, 0]],
                     prt_pid=[[-13, 13], [-13, 13]],
                     prt_extra=[[9.0], [9.0]])       # ragged vs prt_idx_mom
    kept, report = cut_utils.drop_malformed(tree)
    assert len(kept) == 0
    assert report['kept'] == 0
    assert report['dropped_daughter_length_mismatch'] == 2


def test_pipeline_survives_an_emptied_tree():
    """End to end on nothing: drop everything, then run the usual chain. This is
    the sequence a driver would execute on a pathological input file."""
    tree = make_tree(tag_dtf_m=[[550.0]], prt_idx_mom=[[0, 0]],
                     prt_pid=[[-13, 13]], prt_extra=[[9.0]])
    kept, _ = cut_utils.drop_malformed(tree)
    muon_ok = kept['prt_pid'] > 0
    sel = cut_utils.reduce(kept, muon_ok, 'prt_idx_mom', how='all',
                           target_ref='tag_dtf_m')
    best = cut_utils.best_candidate_mask(kept, 'tag_dtf_m', sel)
    masses = cut_utils.values(kept, 'tag_dtf_m', best)
    assert len(masses) == 0


def test_hand_built_untyped_branch_breaks_reduce():
    """A hazard for toy trees only, pinned so it is understood rather than
    rediscovered.

    ak.Array([[]]) infers `var * unknown`, not `var * float64`, because there is
    no value to infer a type from. values_astype cannot fix an unknown, so the
    index array reaches np.bincount as float64 and it refuses to cast.

    uproot never produces this -- a ROOT branch carries a declared type whether
    or not any entry is filled, verified against a real file where all three
    read events had empty tag_dtf_m and the type was still float64. Hence
    make_tree() builds via ak.unflatten to match. If a notebook builds a tree by
    hand and hits this, the fix is the tree, not reduce().
    """
    untyped = ak.Array({'mc_pid': ak.Array([[221.0, -13.0, 13.0]]),
                        'prt_idx_gen': ak.Array([[]])})
    assert 'unknown' in str(untyped['prt_idx_gen'].type)
    with pytest.raises(TypeError, match='[Cc]ast'):
        cut_utils.reduce(untyped, untyped['prt_idx_gen'] >= 0, 'prt_idx_gen',
                         how='any', target_ref='mc_pid')


def test_properly_typed_empty_branch_is_fine():
    """The same shape built the way uproot builds it works, which is what makes
    the case above a toy-tree artifact rather than a framework bug."""
    tree = make_tree(mc_pid=[[221, -13, 13]], prt_idx_gen=[[]])
    assert 'float64' in str(tree['prt_idx_gen'].type)
    out = cut_utils.reduce(tree, tree['prt_idx_gen'] >= 0, 'prt_idx_gen',
                           how='any', target_ref='mc_pid')
    assert out.tolist() == [[False, False, False]]


def test_untyped_empty_tree_is_rejected():
    """A hand-built `ak.Array([])` is depth-1, not a jagged per-event array, and
    reduce() raises on it. Documented so the failure is understood as "your tree
    is the wrong shape" rather than "reduce is broken on empty input"."""
    flat = ak.Array({'mc_pid': ak.Array([]), 'mc_idx_mom': ak.Array([])})
    with pytest.raises(Exception, match='axis'):
        cut_utils.reduce(flat, flat['mc_pid'] > 0, 'mc_idx_mom', how='all',
                         target_ref='mc_pid')


# --- 2. zero entries per event ----------------------------------------------

def test_events_with_no_generator_particles():
    """Events exist but the generator collection is empty in them."""
    tree = make_tree(mc_pid=[[], []], mc_idx_mom=[[], []],
                     mc_px=[[], []], mc_py=[[], []], mc_pz=[[], []])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[], []]
    m = fid_reqs.fiducial_masks(tree, 221, [-13, 13])
    assert m['signal'].tolist() == [[], []]
    keep = ak.to_numpy(ak.any(m['in_acceptance'], axis=1))
    assert keep.tolist() == [False, False]


def test_some_events_empty_some_not():
    """The realistic shape of an MC file: most events have nothing. The
    populated events must be unaffected by their empty neighbours."""
    tree = make_tree(mc_pid=[[], [221, -13, 13], []],
                     mc_idx_mom=[[], [-1, 0, 0], []])
    out = cut_utils.decay_mask(tree, 221, [-13, 13])
    assert out.tolist() == [[], [True, False, False], []]


def test_candidate_with_no_daughters():
    """A candidate nothing points at: all-reduces to the identity True, which is
    correct but is exactly why 'any' is used for propagating selections."""
    tree = make_tree(tag_dtf_m=[[550.0, 549.0]], prt_idx_mom=[[0, 0]],
                     prt_pid=[[-13, 13]])
    ones = ak.ones_like(tree['prt_idx_mom'], dtype=bool)
    n = cut_utils.reduce(tree, ones, 'prt_idx_mom', how='sum',
                         target_ref='tag_dtf_m')
    assert n.tolist() == [[2.0, 0.0]]


# --- 3. one collection empty, another populated ------------------------------

def test_generator_particles_but_no_reco():
    """MC truth exists, nothing was reconstructed -- the normal case for an
    event that failed the trigger. Crossing gen -> reco must give empty, not
    raise."""
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]],
                     prt_idx_gen=[[]], tag_dtf_m=[[]], prt_idx_mom=[[]])
    down = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    assert down.tolist() == [[]]
    up = cut_utils.reduce(tree, tree['prt_idx_gen'] >= 0, 'prt_idx_gen',
                          how='any', target_ref='mc_pid')
    assert up.tolist() == [[False, False, False]]


def test_reco_candidates_but_no_generator_particles():
    """Real data has no mc_* collection at all. Reducing reco onto an empty
    generator collection must return an empty gen-shaped array."""
    tree = make_tree(mc_pid=[[]], prt_idx_gen=[[]], prt_idx_mom=[[0, 0]],
                     tag_dtf_m=[[550.0]], prt_pid=[[-13, 13]])
    out = cut_utils.reduce(tree, tree['prt_idx_gen'] >= 0, 'prt_idx_gen',
                           how='any', target_ref='mc_pid')
    assert out.tolist() == [[]]


def test_mixed_events_one_reconstructed_one_not():
    """Two events, only one reconstructed. Per-event isolation must hold across
    the null boundary."""
    tree = make_tree(mc_pid=[[221, -13, 13], [221, -13, 13]],
                     mc_idx_mom=[[-1, 0, 0], [-1, 0, 0]],
                     prt_idx_gen=[[1, 2], []],
                     prt_idx_mom=[[0, 0], []],
                     tag_dtf_m=[[550.0], []])
    out = cut_utils.reduce(tree, tree['prt_idx_gen'] >= 0, 'prt_idx_gen',
                           how='any', target_ref='mc_pid')
    assert out.tolist() == [[False, True, True], [False, False, False]]


# --- 4. zero matches (data is fine, nothing qualifies) -----------------------

def test_no_candidate_passes_the_cut():
    """Every candidate cut, so best_candidate_mask must select nothing rather
    than falling back to candidate 0."""
    tree = make_tree(tag_dtf_m=[[550.0, 549.0]], tag_dtf_chi2=[[17.0, 14.0]])
    nothing = ak.Array([[False, False]])
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', nothing)
    assert best.tolist() == [[False, False]]
    assert len(cut_utils.values(tree, 'tag_dtf_m', best)) == 0


def test_no_decay_matches_the_request():
    """The record is well formed, it just is not the decay asked for."""
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]])
    assert not ak.any(cut_utils.decay_mask(tree, 511, [-13, 13]))


def test_empty_daughter_pids_matches_only_childless_mothers():
    """A degenerate argument: asking for a decay with no daughters. It should
    match mothers that have none, not everything and not nothing."""
    tree = make_tree(mc_pid=[[221, -13, 13, 221]], mc_idx_mom=[[-1, 0, 0, -1]])
    out = cut_utils.decay_mask(tree, 221, [])
    assert out.tolist() == [[False, False, False, True]]


def test_no_particle_passes_fiducial():
    """Everything out of acceptance: signal found, nothing kept, and the
    failure reasons must still be reported."""
    FWD = (600.0, 800.0, 74200.0)
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]],
                     mc_px=[[FWD[0]] * 3], mc_py=[[FWD[1]] * 3],
                     mc_pz=[[FWD[2]] * 3])
    m = fid_reqs.fiducial_masks(tree, 221, [-13, 13])
    assert int(ak.sum(m['signal'])) == 1
    assert int(ak.sum(m['in_acceptance'])) == 0


def test_no_strays_in_a_clean_event():
    """The stray key must be empty when the record holds only the decay."""
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]],
                     mc_px=[[IN[0]] * 3], mc_py=[[IN[1]] * 3],
                     mc_pz=[[IN[2]] * 3])
    m = fid_reqs.fiducial_masks(tree, 221, [-13, 13])
    assert int(ak.sum(m['stray'])) == 0
