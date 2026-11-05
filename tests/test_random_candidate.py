"""
Unit tests for cut_utils.random_candidate_mask().

Reduces each event to one randomly chosen candidate, restricted to the candidates that survived the
cuts.

The subtlety worth testing hardest: `eligible` is applied TWICE -- once via ak.where to push losers
out of the draw, and again as a final `& eligible`. The second one is not redundant. argmax over an
event where every score is -inf still has to return something, and it returns 0, so without the
final AND an event where nothing survived the cuts would silently select candidate 0.
"""

import awkward as ak
import numpy as np
import pytest

import cut_utils
from conftest import make_tree

SEEDS = range(200)


@pytest.fixture
def cands():
    """Four events covering every branch of the selection logic.

        event 0: 3 candidates, some eligible   -- the normal case
        event 1: 2 candidates, NONE eligible   -- the all -inf case
        event 2: 2 candidates, all eligible    -- no cut interaction
        event 3: 0 candidates                  -- argmax returns None
    """
    return make_tree(
        tag_dtf_m=[[550.0, 551.0, 552.0], [560.0, 561.0], [570.0, 571.0], []],
    )


@pytest.fixture
def eligible():
    return ak.Array([[True, False, True], [False, False], [True, True], []])


# --- the all-ineligible case -------------------------------------------------

def test_all_ineligible_event_selects_nothing(eligible):
    """Event 1 has no surviving candidate. It must select NOTHING.

    Without the final `& eligible`, argmax over [-inf, -inf] returns 0 (ties go to the first entry)
    and candidate 0 would be picked despite being cut.
    """
    for seed in SEEDS:
        picked = cut_utils.random_candidate_mask(eligible, seed)
        assert picked[1].tolist() == [False, False]


def test_argmax_alone_would_have_picked_candidate_zero(eligible):
    """Pins the reason the final AND is needed, so nobody 'simplifies' it away.

    Reproduces the internals up to the point before `& eligible` and shows the phantom winner it
    would otherwise leave behind.
    """
    score = ak.where(eligible, ak.ones_like(eligible, dtype=float), -np.inf)
    idx = ak.argmax(score, axis=1, keepdims=True)
    without_and = ak.local_index(eligible, axis=1) == ak.fill_none(idx, -1)
    assert without_and[1].tolist() == [True, False]      # phantom winner
    with_and = without_and & eligible
    assert with_and[1].tolist() == [False, False]        # corrected


def test_every_candidate_ineligible_everywhere():
    """A whole file where nothing passed the cuts: no picks at all."""
    nothing = ak.Array([[False, False], [False]])
    picked = cut_utils.random_candidate_mask(nothing, 0)
    assert int(ak.sum(picked)) == 0


def test_all_ineligible_does_not_leak_into_values(cands, eligible):
    """The consequence that matters downstream: a cut event must contribute no entry to the fitted
    array, not a spurious one."""
    picked = cut_utils.random_candidate_mask(eligible, 0)
    masses = cut_utils.values(cands, 'tag_dtf_m', picked)
    assert len(masses) == 2                       # events 0 and 2 only
    assert masses[0] in (550.0, 552.0)            # event 0's eligible candidates
    assert masses[1] in (570.0, 571.0)            # event 2's candidates


# --- normal selection --------------------------------------------------------

def test_exactly_one_pick_per_event_with_an_eligible_candidate(eligible):
    for seed in SEEDS:
        picked = cut_utils.random_candidate_mask(eligible, seed)
        assert ak.sum(picked, axis=1).tolist() == [1, 0, 1, 0]


def test_pick_is_always_eligible(eligible):
    for seed in SEEDS:
        picked = cut_utils.random_candidate_mask(eligible, seed)
        assert ak.all(picked == (picked & eligible))


def test_ineligible_candidate_never_picked(eligible):
    """Candidate 1 of event 0 was cut, so it must not compete. This is the whole reason `eligible`
    goes IN rather than being ANDed afterwards."""
    for seed in SEEDS:
        picked = cut_utils.random_candidate_mask(eligible, seed)
        assert not picked[0][1]


def test_each_eligible_candidate_is_picked_about_equally_often():
    two = ak.Array([[True, True]])
    first = sum(bool(cut_utils.random_candidate_mask(two, seed)[0][0])
                for seed in range(400))
    assert 150 < first < 250


def test_different_seeds_pick_different_candidates():
    five = ak.Array([[True] * 5])
    chosen = {cut_utils.random_candidate_mask(five, seed)[0].tolist().index(True)
              for seed in SEEDS}
    assert len(chosen) == 5


def test_same_seed_gives_the_same_mask(eligible):
    first = cut_utils.random_candidate_mask(eligible, 7)
    again = cut_utils.random_candidate_mask(eligible, 7)
    assert first.tolist() == again.tolist()


# --- truth matching first ----------------------------------------------------

def test_cutting_all_but_one_candidate_keeps_the_event():
    """If only the second candidate is matched, the event keeps it for every seed rather than being
    dropped."""
    matched = ak.Array([[False, True]])
    for seed in SEEDS:
        picked = cut_utils.random_candidate_mask(matched, seed)
        assert picked.tolist() == [[False, True]]


def test_picking_before_matching_would_lose_events():
    """Pins why matching goes first: pick-then-match drops the event whenever the pick is the
    unmatched candidate."""
    everything = ak.Array([[True, True]])
    matched = ak.Array([[False, True]])
    kept = sum(int(ak.sum(cut_utils.random_candidate_mask(everything, seed) & matched))
               for seed in range(400))
    assert 150 < kept < 250                       # about half the events survive


# --- shape, empties, errors --------------------------------------------------

def test_event_with_no_candidates_selects_nothing(eligible):
    """argmax returns None there; fill_none(-1) must make it match no position rather than
    defaulting to 0."""
    picked = cut_utils.random_candidate_mask(eligible, 0)
    assert picked[3].tolist() == []


def test_result_is_candidate_shaped(eligible):
    picked = cut_utils.random_candidate_mask(eligible, 0)
    assert ak.num(picked).tolist() == ak.num(eligible).tolist()


def test_result_is_a_plain_ndarray_not_masked(cands, eligible):
    """No option type in the output, so ak.to_numpy gives a plain ndarray -- which is what lets
    values() hand a clean array to a fit."""
    picked = cut_utils.random_candidate_mask(eligible, 0)
    out = cut_utils.values(cands, 'tag_dtf_m', picked)
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, np.ma.MaskedArray)


def test_single_candidate_event():
    assert cut_utils.random_candidate_mask(ak.Array([[True]]), 0).tolist() == [[True]]


def test_events_are_independent(eligible):
    """An all-ineligible event must not disturb its neighbours."""
    for seed in SEEDS:
        picked = cut_utils.random_candidate_mask(eligible, seed)
        assert ak.sum(picked[0]) == 1
        assert ak.sum(picked[2]) == 1
