"""
bkg_utils.candidate_categories: each category on a toy eta -> mu+ mu- gamma event, and every
candidate in exactly one category.
"""

import awkward as ak

import bkg_utils
from conftest import make_tree

DAUGHTER_PIDS = [-13, 13, 22]

# Generator record: the signal eta (0) -> mu+ (1) mu- (2) gamma (3), and motherless strays: a pi+
# (4), a mu- (5), a photon (6) and a pi0 (7).
MC_PID     = [221, -13, 13, 22, 211, 13, 22, 111]
MC_IDX_MOM = [-1, 0, 0, 0, -1, -1, -1, -1]

# Candidates of reco mu+ mu- gamma: the generator match of each daughter (-1: none), and the one
# category each lands in
CANDIDATES = [
    ([1, 2, 3],   'mc_matched'),
    ([1, 2, -1],  'unmatched'),                # photon with no generator match
    ([1, 4, 3],   'muon_misid'),               # mu- matched to the pi+
    ([1, 2, 7],   'photon_misid'),             # photon matched to the pi0
    ([1, 5, 3],   'muon_not_from_signal'),     # mu- matched to the stray mu-
    ([1, 2, 6],   'photon_not_from_signal'),   # photon matched to the stray photon
    ([1, 4, 6],   'several'),                  # muon_misid and photon_not_from_signal
    ([-1, -1, 3], 'unmatched'),                # two daughters, one reason
]


def event_tree():
    """One event holding every candidate of CANDIDATES."""
    return make_tree(
        tag_dtf_m=[[550.0] * len(CANDIDATES)],
        prt_pid=[DAUGHTER_PIDS * len(CANDIDATES)],
        prt_idx_mom=[[k for k in range(len(CANDIDATES)) for _ in DAUGHTER_PIDS]],
        prt_idx_gen=[[gen for gens, _ in CANDIDATES for gen in gens]],
        mc_pid=[MC_PID],
        mc_idx_mom=[MC_IDX_MOM],
    )


def categories_of(categories, event, k):
    """Names of the categories candidate k of `event` is in."""
    return [name for name, mask in categories.items() if mask[event][k]]


def test_each_candidate_lands_in_its_category():
    categories = bkg_utils.candidate_categories(event_tree(), 221, DAUGHTER_PIDS, 'tag_dtf_m')
    for k, (_, expected) in enumerate(CANDIDATES):
        assert categories_of(categories, 0, k) == [expected]


def test_categories_in_order():
    categories = bkg_utils.candidate_categories(event_tree(), 221, DAUGHTER_PIDS, 'tag_dtf_m')
    assert list(categories) == ['mc_matched', 'unmatched', 'muon_misid', 'photon_misid',
                                'muon_not_from_signal', 'photon_not_from_signal', 'several']


def test_mc_matched_is_the_signal_candidate_mask():
    tree = event_tree()
    categories = bkg_utils.candidate_categories(tree, 221, DAUGHTER_PIDS, 'tag_dtf_m')
    assert (categories['mc_matched'].tolist()
            == bkg_utils.signal_candidate_mask(tree, 221, DAUGHTER_PIDS, 'tag_dtf_m').tolist())


def test_every_candidate_in_exactly_one_category():
    """eta -> mu+ mu-: two events, one with no candidates."""
    tree = make_tree(
        tag_dtf_m=[[550.0, 548.0, 547.0], []],
        prt_pid=[[-13, 13, -13, 13, -13, 13], []],
        prt_idx_mom=[[0, 0, 1, 1, 2, 2], []],
        prt_idx_gen=[[1, 2, 1, 3, -1, 4], []],
        mc_pid=[[221, -13, 13, 211, 13], [221, -13, 13]],
        mc_idx_mom=[[-1, 0, 0, -1, -1], [-1, 0, 0]],
    )
    categories = bkg_utils.candidate_categories(tree, 221, [-13, 13], 'tag_dtf_m')
    n_categories = sum(ak.values_astype(mask, int) for mask in categories.values())
    assert n_categories.tolist() == [[1, 1, 1], []]
    assert categories_of(categories, 0, 0) == ['mc_matched']
    assert categories_of(categories, 0, 1) == ['muon_misid']
    assert categories_of(categories, 0, 2) == ['several']   # unmatched and muon_not_from_signal
