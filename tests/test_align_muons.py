"""
Unit tests for cut_utils.align_muons().

The ntuple stores prt_mu, prt_loose_mu and prt_tight_mu for muon daughters
only, one entry per muon in daughter order. In eta -> mu mu gamma that is 2
entries per 3-daughter candidate, so the branch does not line up with prt_pid
by position; align_muons expands it to one entry per daughter.
"""

import pytest

import cut_utils
from conftest import make_tree


# --- eta -> mu mu: every daughter is a muon ----------------------------------

def test_mumu_branch_is_unchanged():
    tree = make_tree(prt_pid=[[-13, 13, -13, 13]],
                     prt_tight_mu=[[1, 0, 1, 1]])
    assert cut_utils.align_muons(tree, 'prt_tight_mu').tolist() == [[1.0, 0.0, 1.0, 1.0]]


# --- eta -> mu mu gamma: the photon has no entry -----------------------------

def test_mumugamma_values_land_on_their_muons():
    """The second candidate's first muon is daughter 3, not daughter 2."""
    tree = make_tree(prt_pid=[[-13, 13, 22, -13, 13, 22]],
                     prt_tight_mu=[[1, 1, 0, 1]])
    assert cut_utils.align_muons(tree, 'prt_tight_mu').tolist() == [[1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]


def test_photon_gets_the_fill_value():
    tree = make_tree(prt_pid=[[-13, 13, 22]], prt_tight_mu=[[1, 1]])
    assert cut_utils.align_muons(tree, 'prt_tight_mu', fill_value=-1.0).tolist() == [[1.0, 1.0, -1.0]]


def test_photon_first_in_daughter_order():
    """Muon values follow the muons' order, wherever the photon sits."""
    tree = make_tree(prt_pid=[[22, -13, 13]], prt_tight_mu=[[0, 1]])
    assert cut_utils.align_muons(tree, 'prt_tight_mu').tolist() == [[0.0, 0.0, 1.0]]


def test_several_events_and_an_empty_event():
    tree = make_tree(prt_pid=[[-13, 13, 22], [], [13, -13, 22]],
                     prt_tight_mu=[[1, 0], [], [0, 1]])
    assert cut_utils.align_muons(tree, 'prt_tight_mu').tolist() == [
        [1.0, 0.0, 0.0], [], [0.0, 1.0, 0.0]]


def test_result_is_shaped_like_prt_pid():
    tree = make_tree(prt_pid=[[-13, 13, 22], [], [13, -13]],
                     prt_tight_mu=[[1, 1], [], [1, 0]])
    aligned = cut_utils.align_muons(tree, 'prt_tight_mu')
    assert aligned.type.length == 3
    assert [len(event) for event in aligned.tolist()] == [3, 0, 2]


# --- malformed events --------------------------------------------------------

def test_missing_muon_entry_raises():
    tree = make_tree(prt_pid=[[-13, 13, 22]], prt_tight_mu=[[1]])
    with pytest.raises(ValueError):
        cut_utils.align_muons(tree, 'prt_tight_mu')


def test_extra_entry_in_one_event_and_missing_in_another_raises():
    """The totals agree (4 muons, 4 entries); the events do not."""
    tree = make_tree(prt_pid=[[-13, 13], [-13, 13]],
                     prt_tight_mu=[[1, 1, 1], [1]])
    with pytest.raises(ValueError):
        cut_utils.align_muons(tree, 'prt_tight_mu')
