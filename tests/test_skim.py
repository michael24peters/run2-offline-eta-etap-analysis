"""
skim.event_mask on toy eta -> mu+ mu- and eta -> mu+ mu- gamma trees.

Every candidate is TOS at every level. Every muon has pT 1500 MeV, p 15075 MeV and eta 3.0, and
passes every muon cut unless noted.
"""

import skim
from conftest import make_tree


def mumu_tree():
    """
    Four eta -> mu+ mu- events:

      event 0: 2 candidates, a muon of candidate 0 has ProbNNmu 0.5     candidate 1 passes
      event 1: 1 candidate with a muon of ProbNNmu 0.5                   no candidate passes
      event 2: 1 candidate, a daughter's prt_idx_mom is 1                malformed
      event 3: 1 candidate                                               passes
    """
    n_daughters = [4, 2, 2, 2]
    return make_tree(
        tag_dtf_m=[[548, 550], [547], [549], [546]],
        tag_l0_tos0=[[1, 1], [1], [1], [1]],
        tag_l0_tos1=[[0, 0], [0], [0], [0]],
        tag_hlt1_tos0=[[1, 1], [1], [1], [1]],
        tag_hlt2_tos0=[[1, 1], [1], [1], [1]],
        prt_idx_mom=[[0, 0, 1, 1], [0, 0], [0, 1], [0, 0]],
        prt_pid=[[-13, 13, -13, 13], [-13, 13], [-13, 13], [-13, 13]],
        prt_pnn_mu=[[0.99, 0.5, 0.99, 0.99], [0.99, 0.5], [0.99, 0.99], [0.99, 0.99]],
        prt_prb_ghost=[[0.01] * n for n in n_daughters],
        prt_ip_chi2=[[1.0] * n for n in n_daughters],
        prt_tight_mu=[[1] * n for n in n_daughters],
        prt_px=[[900] * n for n in n_daughters],
        prt_py=[[1200] * n for n in n_daughters],
        prt_pz=[[15000] * n for n in n_daughters],
    )


def test_event_with_one_passing_candidate_is_kept():
    mask, _ = skim.event_mask(mumu_tree())
    assert mask[0]


def test_event_with_no_passing_candidate_is_dropped():
    mask, _ = skim.event_mask(mumu_tree())
    assert not mask[1]


def test_malformed_event_is_dropped():
    """The passing event after the malformed one keeps its own row."""
    mask, _ = skim.event_mask(mumu_tree())
    assert mask.tolist() == [True, False, False, True]


def test_mumugamma_tight_mu_one_entry_per_muon():
    """
    Two eta -> mu+ mu- gamma events of 2 candidates each, prt_tight_mu stored one entry per muon:
    in event 0 both muons of candidate 0 are tight, in event 1 one muon of each candidate is not.
    """
    tree = make_tree(
        tag_dtf_m=[[548, 550], [548, 550]],
        tag_l0_tos0=[[1, 1], [1, 1]],
        tag_l0_tos1=[[0, 0], [0, 0]],
        tag_hlt1_tos0=[[1, 1], [1, 1]],
        tag_hlt2_tos0=[[1, 1], [1, 1]],
        prt_idx_mom=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        prt_pid=[[-13, 13, 22, -13, 13, 22], [-13, 13, 22, -13, 13, 22]],
        prt_pnn_mu=[[0.99, 0.99, 0.0] * 2, [0.99, 0.99, 0.0] * 2],
        prt_prb_ghost=[[0.01, 0.01, 1.0] * 2, [0.01, 0.01, 1.0] * 2],
        prt_ip_chi2=[[1.0, 1.0, -1.0] * 2, [1.0, 1.0, -1.0] * 2],
        prt_tight_mu=[[1, 1, 0, 0], [1, 0, 0, 1]],
        prt_px=[[900] * 6, [900] * 6],
        prt_py=[[1200] * 6, [1200] * 6],
        prt_pz=[[15000] * 6, [15000] * 6],
    )
    mask, _ = skim.event_mask(tree)
    assert mask.tolist() == [True, False]
    assert tree['prt_tight_mu'].tolist() == [[1, 1, 0, 0], [1, 0, 0, 1]]   # left as stored
