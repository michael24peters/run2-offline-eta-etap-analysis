"""
skim.candidate_mask and skim.keep_candidates on toy eta -> mu+ mu- and eta -> mu+ mu- gamma trees.

Every candidate is TOS at every level. Every muon has pT ~1500 MeV, p ~15075 MeV and eta ~3.0, and
passes every muon cut unless noted.
"""

import awkward as ak
import numpy as np

import selection
import skim
from conftest import make_tree

ETA_WINDOW = (450.0, 650.0)
ETAPRIME_WINDOW = (800.0, 1100.0)


def mumu_tree():
    """
    Five eta -> mu+ mu- events, prt_px numbering each daughter:

      event 0: 3 candidates: 548 MeV with a muon of ProbNNmu 0.5, 550 MeV, 560 MeV
      event 1: 1 candidate at 1100 MeV, the upper edge of ETAPRIME_WINDOW
      event 2: 1 candidate at 750 MeV, between the windows
      event 3: 1 candidate, a daughter's prt_idx_mom is 1 (malformed)
      event 4: 1 candidate at 547 MeV with a muon of ProbNNmu 0.5
    """
    tree = make_tree(
        tag_dtf_m=[[548, 550, 560], [1100], [750], [549], [547]],
        tag_l0_tos0=[[1, 1, 1], [1], [1], [1], [1]],
        tag_l0_tos1=[[0, 0, 0], [0], [0], [0], [0]],
        tag_hlt1_tos0=[[1, 1, 1], [1], [1], [1], [1]],
        tag_hlt2_tos0=[[1, 1, 1], [1], [1], [1], [1]],
        tag_hlt2_tos1=[[5, 6, 7], [], [8], [9], [4]],   # empty in event 1
        prt_idx_mom=[[0, 0, 1, 1, 2, 2], [0, 0], [0, 0], [0, 1], [0, 0]],
        prt_pid=[[-13, 13] * 3, [-13, 13], [-13, 13], [-13, 13], [-13, 13]],
        prt_pnn_mu=[[0.99, 0.5, 0.99, 0.99, 0.99, 0.99], [0.99] * 2, [0.99] * 2, [0.99] * 2,
                    [0.99, 0.5]],
        prt_prb_ghost=[[0.01] * 6] + [[0.01] * 2] * 4,
        prt_ip_chi2=[[1.0] * 6] + [[1.0] * 2] * 4,
        prt_tight_mu=[[1] * 6] + [[1] * 2] * 4,
        prt_px=[[900, 901, 910, 911, 920, 921], [930, 931], [940, 941], [950, 951], [960, 961]],
        prt_py=[[1200] * 6] + [[1200] * 2] * 4,
        prt_pz=[[15000] * 6] + [[15000] * 2] * 4,
    )
    return ak.with_field(tree, np.array([10.0, 11.0, 12.0, 13.0, 14.0]), 'run_n')


def skim_tree(tree, window):
    """keep_candidates of candidate_mask with selection.TOS_LINES and seed selection.SEED."""
    rows, keep, _, _ = skim.candidate_mask(tree, selection.TOS_LINES, window, selection.SEED)
    return skim.keep_candidates(tree[rows], keep)


def test_one_picked_candidate_per_event_inside_the_window():
    rows, keep, steps, _ = skim.candidate_mask(mumu_tree(), selection.TOS_LINES, ETA_WINDOW,
                                               selection.SEED)
    assert rows.tolist() == [0, 1, 2, 4]   # event 3 is malformed
    assert keep.tolist()[1:] == [[False], [False], [False]]
    assert keep.tolist()[0] in ([False, True, False], [False, False, True])
    assert [label for label, _ in steps[-2:]] == ['random candidate', 'tag_dtf_m in 450-650 MeV']


def test_window_includes_its_edges():
    _, keep, _, _ = skim.candidate_mask(mumu_tree(), selection.TOS_LINES, ETAPRIME_WINDOW,
                                        selection.SEED)
    assert keep.tolist() == [[False, False, False], [True], [False], [False]]


def test_window_applies_after_the_random_candidate():
    """200 events with a passing candidate at 550 and one at 750 MeV: the event is kept only when
    the pick is the 550 MeV candidate."""
    n_events = 200
    tree = make_tree(
        tag_dtf_m=[[550, 750]] * n_events,
        tag_l0_tos0=[[1, 1]] * n_events,
        tag_l0_tos1=[[0, 0]] * n_events,
        tag_hlt1_tos0=[[1, 1]] * n_events,
        tag_hlt2_tos0=[[1, 1]] * n_events,
        prt_idx_mom=[[0, 0, 1, 1]] * n_events,
        prt_pid=[[-13, 13, -13, 13]] * n_events,
        prt_pnn_mu=[[0.99] * 4] * n_events,
        prt_prb_ghost=[[0.01] * 4] * n_events,
        prt_ip_chi2=[[1.0] * 4] * n_events,
        prt_tight_mu=[[1] * 4] * n_events,
        prt_px=[[900] * 4] * n_events,
        prt_py=[[1200] * 4] * n_events,
        prt_pz=[[15000] * 4] * n_events,
    )
    out = skim_tree(tree, ETA_WINDOW)
    assert 0 < len(out['tag_dtf_m']) < n_events
    assert set(ak.flatten(out['tag_dtf_m']).tolist()) == {550}


def test_kept_candidate_keeps_its_daughters_renumbered():
    out = skim_tree(mumu_tree(), ETA_WINDOW)
    assert out['tag_dtf_m'].tolist() in ([[550]], [[560]])
    assert out['prt_idx_mom'].tolist() == [[0, 0]]
    expected_px = [[910, 911]] if out['tag_dtf_m'].tolist() == [[550]] else [[920, 921]]
    assert out['prt_px'].tolist() == expected_px
    assert out['run_n'].tolist() == [10]


def test_tag_branch_empty_in_an_event_stays_empty():
    assert skim_tree(mumu_tree(), ETA_WINDOW)['tag_hlt2_tos1'].tolist() in ([[6]], [[7]])
    assert skim_tree(mumu_tree(), ETAPRIME_WINDOW)['tag_hlt2_tos1'].tolist() == [[]]


def mumugamma_tree():
    """
    Two eta -> mu+ mu- gamma events of 2 candidates each, prt_tight_mu stored one entry per muon:
    in event 0 a muon of candidate 0 is not tight, in event 1 candidate 1 is at 300 MeV and has a
    muon that is not tight.
    """
    return make_tree(
        tag_dtf_m=[[548, 550], [549, 300]],
        tag_l0_tos0=[[1, 1], [1, 1]],
        tag_l0_tos1=[[0, 0], [0, 0]],
        tag_hlt1_tos0=[[1, 1], [1, 1]],
        tag_hlt2_tos0=[[1, 1], [1, 1]],
        prt_idx_mom=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        prt_pid=[[-13, 13, 22, -13, 13, 22], [-13, 13, 22, -13, 13, 22]],
        prt_pnn_mu=[[0.99, 0.99, 0.0] * 2, [0.99, 0.99, 0.0] * 2],
        prt_prb_ghost=[[0.01, 0.01, 1.0] * 2, [0.01, 0.01, 1.0] * 2],
        prt_ip_chi2=[[1.0, 1.0, -1.0] * 2, [1.0, 1.0, -1.0] * 2],
        prt_tight_mu=[[1, 0, 1, 1], [1, 1, 0, 1]],
        prt_px=[[900, 901, 902, 910, 911, 912], [920, 921, 922, 930, 931, 932]],
        prt_py=[[1200] * 6, [1200] * 6],
        prt_pz=[[15000] * 6, [15000] * 6],
    )


def test_mumugamma_tight_mu_keeps_one_entry_per_kept_muon():
    tree = mumugamma_tree()
    out = skim_tree(tree, ETA_WINDOW)
    assert out['prt_tight_mu'].tolist() == [[1, 1], [1, 1]]
    assert out['prt_px'].tolist() == [[910, 911, 912], [920, 921, 922]]
    assert out['prt_idx_mom'].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert tree['prt_tight_mu'].tolist() == [[1, 0, 1, 1], [1, 1, 0, 1]]   # left as stored


def test_skim_of_a_skim_keeps_everything():
    for tree in [mumu_tree(), mumugamma_tree()]:
        skimmed = ak.Array(skim_tree(tree, ETA_WINDOW))
        rows, keep, _, _ = skim.candidate_mask(skimmed, selection.TOS_LINES, ETA_WINDOW,
                                               selection.SEED)
        assert len(rows) == len(skimmed)
        assert ak.all(keep)
