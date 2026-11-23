"""
selection.parse_cut and selection.cut_mask, and selection.cut_steps and selection.daughter_masks on
a toy eta -> mu+ mu- (gamma) tree.

  event 0: candidate 0 passes everything; candidate 1 is not L0 TOS but is HLT1 and HLT2 TOS
  event 1: candidate 0 has a muon with ProbNNmu 0.5; candidate 1 has a photon daughter that fails
           every muon requirement
"""

import awkward as ak
import numpy as np
import pytest

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
    assert labels[4:] == selection.MUON_CUTS
    assert ak.to_list(dict(steps)['TOS up to HLT2']) == [[True, False], [True, True]]
    assert ak.to_list(dict(steps)['pnn_mu > 0.9']) == [[True, False], [False, True]]
    assert ak.to_list(steps[-1][1]) == [[True, False], [False, True]]


def test_muon_cuts_pass_every_non_muon_daughter():
    tree = toy_tree()
    is_photon = tree['prt_pid'] == 22
    masks = selection.daughter_masks(tree, selection.MUON_CUTS, selection.MUON_PID)
    for label, muon_ok in masks.items():
        assert ak.all(muon_ok[is_photon]), label


def test_muon_with_no_pv_fails_the_ip_chi2_cut():
    tree = toy_tree()
    tree['prt_ip_chi2'] = ak.Array([[1.0, -1.0, 1.0, 1.0], [1.0] * 4 + [-1.0]])
    masks = selection.daughter_masks(tree, ['0 <= ip_chi2 < 6'], selection.MUON_PID)
    assert ak.to_list(masks['0 <= ip_chi2 < 6']) == [[True, False, True, True], [True] * 5]


def test_photon_cuts_apply_to_photons_only():
    steps, _ = selection.cut_steps(toy_tree(), 'tag_dtf_m', selection.TOS_LINES,
                                   muon_cuts=[], photon_cuts=['pt > 2000'])
    assert [label for label, _ in steps][-1] == 'photon pt > 2000'
    assert ak.to_list(steps[-1][1]) == [[True, False], [True, False]]   # every photon has pT 1500


def test_parse_cut_condition_and_range():
    assert selection.parse_cut('pt > 1000') == [('pt', '>', 1000.0)]
    assert selection.parse_cut('2 < eta < 4.5') == [('eta', '>', 2.0), ('eta', '<', 4.5)]
    assert selection.parse_cut('0 <= ip_chi2 < 6') == [('ip_chi2', '>=', 0.0),
                                                        ('ip_chi2', '<', 6.0)]
    with pytest.raises(ValueError):
        selection.parse_cut('pt>1000')


def test_cut_mask_range_on_numpy():
    masses = np.array([534.9, 535.0, 547.9, 560.0])
    assert selection.cut_mask(masses, '535 <= dtf_m < 560').tolist() == [False, True, True, False]


def test_line_counts_are_among_the_candidates_before_any_tos_cut():
    _, triggers = selection.cut_steps(toy_tree(), 'tag_dtf_m', selection.TOS_LINES)
    assert triggers['L0'] == {'tag_l0_tos0': 3, 'tag_l0_tos1': 0}
    assert triggers['HLT1'] == {'tag_hlt1_tos0': 4}   # counts the candidate that is not L0 TOS
