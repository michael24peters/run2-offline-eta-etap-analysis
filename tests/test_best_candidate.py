"""
Unit tests for cut_utils.best_candidate_mask().

Reduces each event to its single best candidate, ranked by a per-candidate
metric, restricted to the candidates that survived the cuts.

The subtlety worth testing hardest: `eligible` is applied TWICE -- once via
ak.where to push losers out of the ranking, and again as a final `& eligible`.
The second one is not redundant. argmin over an event where every score is inf
still has to return something, and it returns 0, so without the final AND an
event where nothing survived the cuts would silently select candidate 0.
"""

import awkward as ak
import numpy as np
import pytest

import cut_utils
from conftest import make_tree


@pytest.fixture
def cands():
    """Four events covering every branch of the selection logic.

        event 0: 3 candidates, some eligible   -- the normal case
        event 1: 2 candidates, NONE eligible   -- the all-inf case
        event 2: 2 candidates, all eligible    -- no cut interaction
        event 3: 0 candidates                  -- argmin returns None
    """
    return make_tree(
        tag_dtf_chi2=[[0.42, 0.95, 1.82], [7.0, 2.0], [4.0, 9.0], []],
        tag_dtf_m=[[550.0, 551.0, 552.0], [560.0, 561.0], [570.0, 571.0], []],
    )


@pytest.fixture
def eligible():
    return ak.Array([[True, False, True], [False, False], [True, True], []])


# --- the all-ineligible case -------------------------------------------------

def test_all_ineligible_event_selects_nothing(cands, eligible):
    """Event 1 has no surviving candidate. It must select NOTHING.

    Without the final `& eligible`, argmin over [inf, inf] returns 0 (ties go to
    the first entry) and candidate 0 would be picked despite being cut.
    """
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert best[1].tolist() == [False, False]


def test_argmin_alone_would_have_picked_candidate_zero(cands, eligible):
    """Pins the reason the final AND is needed, so nobody 'simplifies' it away.

    Reproduces the internals up to the point before `& eligible` and shows the
    phantom winner it would otherwise leave behind.
    """
    score = ak.where(eligible, cands['tag_dtf_chi2'], np.inf)
    idx = ak.argmin(score, axis=1, keepdims=True)
    without_and = ak.local_index(cands['tag_dtf_chi2'], axis=1) == ak.fill_none(idx, -1)
    assert without_and[1].tolist() == [True, False]      # phantom winner
    with_and = without_and & eligible
    assert with_and[1].tolist() == [False, False]        # corrected


def test_every_candidate_ineligible_everywhere():
    """A whole file where nothing passed the cuts: no winners at all."""
    tree = make_tree(tag_dtf_chi2=[[1.0, 2.0], [3.0]])
    nothing = ak.Array([[False, False], [False]])
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', nothing)
    assert int(ak.sum(best)) == 0


def test_all_ineligible_does_not_leak_into_values(cands, eligible):
    """The consequence that matters downstream: a cut event must contribute no
    entry to the fitted array, not a spurious one."""
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    masses = cut_utils.values(cands, 'tag_dtf_m', best)
    assert masses.tolist() == [550.0, 570.0]      # events 0 and 2 only


def test_tie_among_ineligible_candidates():
    """Identical metrics AND all ineligible -- the worst case for a tie break."""
    tree = make_tree(tag_dtf_chi2=[[5.0, 5.0, 5.0]])
    nothing = ak.Array([[False, False, False]])
    assert int(ak.sum(cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', nothing))) == 0


def test_infinite_metric_on_an_eligible_candidate():
    """A candidate whose metric is genuinely inf is eligible and is the only
    one, so it must still win -- inf as a real value must not be confused with
    inf as the "pushed out of the ranking" sentinel."""
    tree = make_tree(tag_dtf_chi2=[[1.0, 2.0]])
    tree = ak.with_field(tree, ak.Array([[np.inf, 5.0]]), 'metric')
    eligible = ak.Array([[True, False]])
    best = cut_utils.best_candidate_mask(tree, 'metric', eligible)
    assert best.tolist() == [[True, False]]


# --- normal selection --------------------------------------------------------

def test_min_picks_the_smallest(cands, eligible):
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert best[0].tolist() == [True, False, False]      # 0.42 < 1.82
    assert best[2].tolist() == [True, False]             # 4.0 < 9.0


def test_ineligible_candidate_never_wins_even_if_best(cands, eligible):
    """Candidate 1 of event 0 has chi2 0.95, better than the winning 1.82's
    rival -- but it was cut, so it must not compete. This is the whole reason
    `eligible` goes IN rather than being ANDed afterwards."""
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert not best[0][1]


def test_cutting_the_globally_best_candidate_keeps_the_event():
    """If the best candidate is cut, the event keeps its best SURVIVOR rather
    than being dropped."""
    tree = make_tree(tag_dtf_chi2=[[1.0, 5.0, 9.0]])
    eligible = ak.Array([[False, True, True]])
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', eligible)
    assert best.tolist() == [[False, True, False]]       # 5.0, not nothing


def test_exactly_one_winner_per_event(cands, eligible):
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert ak.all(ak.sum(best, axis=1) <= 1)


def test_winner_is_always_eligible(cands, eligible):
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert ak.all(best == (best & eligible))


def test_ties_pick_the_first(cands):
    """Documented consequence of argmin: with a degenerate metric the winner is
    candidate 0, i.e. arbitrary. Relevant because tag_dtf_chi2 is degenerate in
    a large fraction of real multi-candidate events."""
    tree = make_tree(tag_dtf_chi2=[[5.0, 5.0, 5.0]])
    eligible = ak.Array([[True, True, True]])
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', eligible)
    assert best.tolist() == [[True, False, False]]


# --- shape, empties, errors --------------------------------------------------

def test_event_with_no_candidates_selects_nothing(cands, eligible):
    """argmin returns None there; fill_none(-1) must make it match no position
    rather than defaulting to 0."""
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert best[3].tolist() == []


def test_result_is_candidate_shaped(cands, eligible):
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert ak.num(best).tolist() == ak.num(cands['tag_dtf_chi2']).tolist()


def test_result_is_a_plain_ndarray_not_masked(cands, eligible):
    """No option type in the output, so ak.to_numpy gives a plain ndarray --
    which is what lets values() hand a clean array to a fit."""
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    out = cut_utils.values(cands, 'tag_dtf_m', best)
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, np.ma.MaskedArray)


def test_single_candidate_event():
    tree = make_tree(tag_dtf_chi2=[[3.0]])
    eligible = ak.Array([[True]])
    assert cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', eligible).tolist() \
        == [[True]]


def test_negative_metrics():
    """Nothing assumes the metric is positive; min over negatives must work."""
    tree = make_tree(tag_dtf_chi2=[[-5.0, -1.0, -9.0]])
    eligible = ak.Array([[True, True, True]])
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', eligible)
    assert best.tolist() == [[False, False, True]]


def test_events_are_independent(cands, eligible):
    """An all-ineligible event must not disturb its neighbours."""
    best = cut_utils.best_candidate_mask(cands, 'tag_dtf_chi2', eligible)
    assert best[0].tolist() == [True, False, False]
    assert best[2].tolist() == [True, False]
