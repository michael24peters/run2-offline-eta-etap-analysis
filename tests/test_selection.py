"""
selection.cut_steps and selection.muon_cuts on a toy eta -> mu+ mu- (gamma) tree.

  event 0: candidate 0 passes everything; candidate 1 is not L0 TOS but is HLT1 and HLT2 TOS
  event 1: candidate 0 has a muon with ProbNNmu 0.5; candidate 1 has a photon daughter that fails
           every muon requirement
"""

import awkward as ak

import selection
from conftest import make_tree


def toy_tree():
    """Every daughter has pT 1500 MeV, p 15075 MeV and eta 3.0, so passes the kinematic cuts."""
    return make_tree(
        tag_dtf_m=[[548, 550], [547, 546]],
        tag_l0_tos0=[[1, 0], [1, 1]],
        tag_l0_tos1=[[0, 0], [0, 0]],
        tag_hlt1_tos0=[[1, 1], [1, 1]],
        tag_hlt2_tos0=[[1, 1], [1, 1]],
        prt_idx_mom=[[0, 0, 1, 1], [0, 0, 1, 1, 1]],
        prt_pid=[[-13, 13, -13, 13], [-13, 13, -13, 13, 22]],
        prt_pnn_mu=[[0.99] * 4, [0.99, 0.5, 0.99, 0.99, 0.0]],
        prt_prb_ghost=[[0.01] * 4, [0.01] * 4 + [1.0]],
        prt_ip_chi2=[[1.0] * 4, [1.0] * 4 + [-1.0]],
        prt_tight_mu=[[1] * 4, [1] * 4 + [0]],
        prt_px=[[900] * 4, [900] * 5],
        prt_py=[[1200] * 4, [1200] * 5],
        prt_pz=[[15000] * 4, [15000] * 5],
    )


def test_steps_are_cumulative_and_in_order():
    steps, _ = selection.cut_steps(toy_tree(), 'tag_dtf_m', selection.TOS_LINES)
    labels = [label for label, _ in steps]
    assert labels[:4] == ['all candidates', 'TOS up to L0', 'TOS up to HLT1', 'TOS up to HLT2']
    assert labels[4:] == list(selection.muon_cuts(toy_tree()))
    assert ak.to_list(dict(steps)['TOS up to HLT2']) == [[True, False], [True, True]]
    assert ak.to_list(dict(steps)['pnn_mu > 0.9']) == [[True, False], [False, True]]
    assert ak.to_list(steps[-1][1]) == [[True, False], [False, True]]


def test_muon_cuts_pass_every_non_muon_daughter():
    tree = toy_tree()
    is_photon = tree['prt_pid'] == 22
    for label, muon_ok in selection.muon_cuts(tree).items():
        assert ak.all(muon_ok[is_photon]), label


def test_line_counts_are_among_the_candidates_before_any_tos_cut():
    _, triggers = selection.cut_steps(toy_tree(), 'tag_dtf_m', selection.TOS_LINES)
    assert triggers['L0'] == {'tag_l0_tos0': 3, 'tag_l0_tos1': 0}
    assert triggers['HLT1'] == {'tag_hlt1_tos0': 4}   # counts the candidate that is not L0 TOS
