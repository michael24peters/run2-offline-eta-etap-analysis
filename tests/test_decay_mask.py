"""
Unit tests for cut_utils.decay_mask().

decay_mask identifies mothers whose daughters are exactly a given multiset of
PIDs, by asking counting questions that reduce onto the mother rather than
materialising each mother's daughter list.
"""

import awkward as ak

import cut_utils
from conftest import make_tree

# --- the decays this analysis actually cares about ---------------------------

def test_finds_eta_to_mumu():
    """Baseline: the mother is flagged, its daughters are not."""
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]])
    out = cut_utils.decay_mask(tree, 221, [-13, 13])
    assert out.tolist() == [[True, False, False]]


def test_finds_eta_to_mumugamma():
    tree = make_tree(mc_pid=[[221, -13, 13, 22]], mc_idx_mom=[[-1, 0, 0, 0]])
    assert cut_utils.decay_mask(tree, 221, [-13, 13, 22]).tolist() \
        == [[True, False, False, False]]


def test_finds_eta_to_four_muons():
    """A repeated PID must be matched with the right multiplicity, which is what
    the Counter in decay_mask is for."""
    tree = make_tree(mc_pid=[[221, -13, -13, 13, 13]],
                     mc_idx_mom=[[-1, 0, 0, 0, 0]])
    assert cut_utils.decay_mask(tree, 221, [-13, -13, 13, 13]).tolist() \
        == [[True, False, False, False, False]]


def test_finds_eta_to_mumuee():
    tree = make_tree(mc_pid=[[221, -13, 13, -11, 11]],
                     mc_idx_mom=[[-1, 0, 0, 0, 0]])
    assert cut_utils.decay_mask(tree, 221, [-13, 13, -11, 11]).tolist() \
        == [[True, False, False, False, False]]


# --- rejection ---------------------------------------------------------------

def test_rejects_wrong_multiplicity():
    """eta -> mu+ mu- mu- must NOT match [-13, 13]; counting, not membership."""
    tree = make_tree(mc_pid=[[221, -13, 13, 13]], mc_idx_mom=[[-1, 0, 0, 0]])
    assert not ak.any(cut_utils.decay_mask(tree, 221, [-13, 13]))


def test_rejects_an_extra_daughter():
    """The total-count check is what rejects a radiative extra: eta -> mu mu
    gamma gamma must not pass a [-13, 13, 22] test."""
    tree = make_tree(mc_pid=[[221, -13, 13, 22, 22]],
                     mc_idx_mom=[[-1, 0, 0, 0, 0]])
    assert not ak.any(cut_utils.decay_mask(tree, 221, [-13, 13, 22]))


def test_rejects_a_missing_daughter():
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]])
    assert not ak.any(cut_utils.decay_mask(tree, 221, [-13, 13, 22]))


def test_rejects_wrong_mother_pid():
    """Same daughters under a different mother is a different decay."""
    tree = make_tree(mc_pid=[[443, -13, 13]], mc_idx_mom=[[-1, 0, 0]])
    assert not ak.any(cut_utils.decay_mask(tree, 221, [-13, 13]))
    assert ak.any(cut_utils.decay_mask(tree, 443, [-13, 13]))


def test_daughter_order_does_not_matter():
    """The record may store daughters in any order; a multiset is a multiset."""
    a = make_tree(mc_pid=[[221, -13, 13, 22]], mc_idx_mom=[[-1, 0, 0, 0]])
    b = make_tree(mc_pid=[[221, 22, 13, -13]], mc_idx_mom=[[-1, 0, 0, 0]])
    assert cut_utils.decay_mask(a, 221, [-13, 13, 22]).tolist() \
        == cut_utils.decay_mask(b, 221, [-13, 13, 22]).tolist()


def test_charge_conjugate_is_not_matched():
    """[-13, 13] and [13, 13] are different multisets -- sign matters."""
    tree = make_tree(mc_pid=[[221, 13, 13]], mc_idx_mom=[[-1, 0, 0]])
    assert not ak.any(cut_utils.decay_mask(tree, 221, [-13, 13]))


# --- the messy generator record ----------------------------------------------

def test_strays_do_not_break_the_match(gen_with_strays):
    """MC-matched junk appended to the record is motherless, so it is nobody's
    daughter and cannot inflate a daughter count."""
    out = cut_utils.decay_mask(gen_with_strays, 221, [-13, 13, 22])
    assert out.tolist() == [[True, False, False, False, False, False, False]]


def test_two_complete_decays_both_match(two_gen_candidates):
    """Both etas are flagged independently, at their own indices."""
    out = cut_utils.decay_mask(two_gen_candidates, 221, [-13, 13, 22])
    assert out.tolist() == [[True, False, False, False,
                             True, False, False, False]]
    assert int(ak.sum(out)) == 2


def test_only_the_well_formed_decay_matches():
    """Two etas, one good and one missing its photon: flag exactly one."""
    tree = make_tree(mc_pid=[[221, -13, 13, 22, 221, -13, 13]],
                     mc_idx_mom=[[-1, 0, 0, 0, -1, 4, 4]])
    out = cut_utils.decay_mask(tree, 221, [-13, 13, 22])
    assert out.tolist() == [[True, False, False, False, False, False, False]]


def test_childless_mother_pid_does_not_match():
    """A stray particle that happens to have the mother PID has no daughters,
    so its counts are all zero and it must not be mistaken for a decay."""
    tree = make_tree(mc_pid=[[221, 221, -13, 13]], mc_idx_mom=[[-1, -1, 0, 0]])
    out = cut_utils.decay_mask(tree, 221, [-13, 13])
    assert out.tolist() == [[True, False, False, False]]


# --- edge cases --------------------------------------------------------------

def test_empty_event():
    tree = make_tree(mc_pid=[[]], mc_idx_mom=[[]])
    assert cut_utils.decay_mask(tree, 221, [-13, 13]).tolist() == [[]]


def test_events_are_independent():
    """One good event, one bad -- the bad one must not poison the good one."""
    tree = make_tree(mc_pid=[[221, -13, 13], [221, -13, 13, 22]],
                     mc_idx_mom=[[-1, 0, 0], [-1, 0, 0, 0]])
    out = cut_utils.decay_mask(tree, 221, [-13, 13])
    assert out.tolist() == [[True, False, False], [False, False, False, False]]
