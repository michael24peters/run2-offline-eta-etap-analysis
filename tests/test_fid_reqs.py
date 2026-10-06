"""
Unit tests for src/root/fid_reqs.py.

fid_reqs decides which MC events could plausibly have been reconstructed, so
downstream efficiency denominators mean something. The tests build generator
records with momenta chosen to sit just inside or just outside the acceptance
(see IN / FWD / LOW_PT in conftest) rather than relying on real files.
"""

import awkward as ak
import numpy as np
import pytest
from conftest import kinematic_tree, make_tree, IN, FWD, LOW_PT

import fid_reqs


MUMUGAMMA = [-13, 13, 22]
MUMU = [-13, 13]


# --- fiducial_mask: the per-particle predicate -------------------------------

def test_particle_inside_acceptance_passes():
    """Baseline: eta ~ 3.0, pT = 1000, p ~ 10050 clears every cut."""
    tree = kinematic_tree([[(13, -1, *IN)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[True]]


def test_particle_outside_eta_range_fails():
    """FWD sits at eta = 5.0, past the 4.5 edge, with pT and p both fine --
    so this isolates the eta cut."""
    tree = kinematic_tree([[(13, -1, *FWD)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[False]]


def test_soft_particle_fails():
    tree = kinematic_tree([[(13, -1, *LOW_PT)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[False]]


def test_photon_and_muon_use_different_rules():
    """A photon has no p > 3000 requirement, so a momentum that fails for a
    muon can still pass for a photon. Pins the per-PID table.

    pT = 700 and eta = 2.10 clear those two cuts; p = 2901 sits just under the
    muon-only momentum threshold.
    """
    mid = (420.0, 560.0, 2815.0)
    tree = kinematic_tree([[(13, -1, *mid), (22, -1, *mid)]])
    out = fid_reqs.fiducial_mask(tree).tolist()[0]
    assert out[0] is False            # muon fails p > 3000
    assert out[1] is True             # photon has no p requirement


def test_unknown_pid_fails_by_default():
    """A PID absent from FIDUCIAL_REQS (here a neutron) must fail closed, not
    pass by omission."""
    tree = kinematic_tree([[(2112, -1, *IN)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[False]]


def test_mother_pid_is_not_in_the_table():
    """The eta itself is never detected, so it has no fiducial rule and comes
    back False. Downstream logic must therefore not require the mother to pass
    -- see test_mother_failing_does_not_block_acceptance."""
    tree = kinematic_tree([[(221, -1, *IN)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[False]]


def test_zero_momentum_particle_fails_without_dividing_by_zero():
    """p == 0 would make eta() divide by zero; the guard must catch it."""
    tree = kinematic_tree([[(13, -1, 0.0, 0.0, 0.0)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[False]]


def test_result_is_shaped_like_the_generator_collection():
    tree = kinematic_tree([[(221, -1, *IN), (13, 0, *IN)], [(13, -1, *IN)]])
    out = fid_reqs.fiducial_mask(tree)
    assert ak.num(out).tolist() == [2, 1]


def test_empty_event():
    tree = kinematic_tree([[], [(13, -1, *IN)]])
    assert fid_reqs.fiducial_mask(tree).tolist() == [[], [True]]


# --- fiducial_masks: the usual case ------------------------------------------

def test_clean_decay_all_daughters_inside():
    """The bread-and-butter case: one eta, three good daughters, keep it."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN),
                            (13, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['signal'])) == 1
    assert int(ak.sum(m['in_acceptance'])) == 1


def test_clean_decay_with_one_daughter_outside():
    """One forward muon must cost the whole decay its acceptance."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *FWD),
                            (13, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['signal'])) == 1
    assert int(ak.sum(m['in_acceptance'])) == 0


def test_mother_failing_does_not_block_acceptance():
    """The eta has no fiducial rule so fiducial_mask calls it False, but only
    the DAUGHTERS are reduced over -- the mother's own flag must not matter."""
    tree = kinematic_tree([[(221, -1, *FWD), (-13, 0, *IN),
                            (13, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['in_acceptance'])) == 1


def test_mumu_decay_without_a_photon():
    """The mother_pid/daughter_pids arguments must actually switch decay."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMU)
    assert int(ak.sum(m['in_acceptance'])) == 1
    # ...and the same tree must NOT match the three-body decay.
    assert int(ak.sum(fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)['signal'])) == 0


def test_all_masks_are_generator_shaped():
    """Every key must be the same shape so they compose with & and |."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN),
                            (13, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    for name, mask in m.items():
        assert ak.num(mask).tolist() == [4], name


# --- fiducial_masks: multiple decays -----------------------------------------

def test_two_good_decays_in_one_event():
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN), (22, 0, *IN),
                            (221, -1, *IN), (-13, 4, *IN), (13, 4, *IN), (22, 4, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['signal'])) == 2
    assert int(ak.sum(m['in_acceptance'])) == 2


def test_one_good_one_bad_decay_in_one_event():
    """ak.any keeps the event; ak.all would drop it. The two policies must
    demonstrably differ here, which is the open decision in the script."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN), (22, 0, *IN),
                            (221, -1, *IN), (-13, 4, *FWD), (13, 4, *IN), (22, 4, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['signal'])) == 2
    assert int(ak.sum(m['in_acceptance'])) == 1
    keep_any = bool(ak.any(m['in_acceptance'], axis=1)[0])
    keep_all = int(ak.sum(m['in_acceptance'])) == int(ak.sum(m['signal']))
    assert keep_any is True
    assert keep_all is False


def test_events_are_independent():
    """A bad event must not change the verdict on a good one."""
    good = [(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN), (22, 0, *IN)]
    bad = [(221, -1, *IN), (-13, 0, *FWD), (13, 0, *IN), (22, 0, *IN)]
    tree = kinematic_tree([good, bad])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    keep = ak.to_numpy(ak.any(m['in_acceptance'], axis=1))
    assert keep.tolist() == [True, False]


# --- fiducial_masks: the messy record ----------------------------------------

def test_strays_do_not_affect_the_keep_decision():
    """MC-matched junk appended to the record is motherless, so it is nobody's
    daughter and must not drag the decay out of acceptance -- even when the
    strays themselves fail fiducial."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN), (22, 0, *IN),
                            (22, -1, *FWD), (211, -1, *FWD)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['in_acceptance'])) == 1
    assert int(ak.sum(m['stray'])) == 2


def test_stray_excludes_the_signal_mother():
    """The eta is motherless too, so 'stray' must subtract the signal or it
    would report every decay as background."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN), (22, 0, *IN),
                            (211, -1, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert m['stray'].tolist() == [[False, False, False, False, True]]


def test_event_with_no_signal_decay():
    """Pure background. Nothing to keep, and nothing should raise."""
    tree = kinematic_tree([[(211, -1, *IN), (22, -1, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['signal'])) == 0
    assert int(ak.sum(m['in_acceptance'])) == 0
    assert bool(ak.any(m['in_acceptance'], axis=1)[0]) is False


def test_extra_daughter_disqualifies_the_decay():
    """eta -> mu mu gamma gamma is not eta -> mu mu gamma, so it is not signal
    even though every particle is inside acceptance."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN),
                            (22, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['signal'])) == 0


# --- the failure-reason breakdown --------------------------------------------

def test_failed_muon_names_the_muon():
    """When a muon is the reason the decay was lost, failed_muon must flag its
    mother -- otherwise the diagnostic breakdown is silently empty."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *FWD),
                            (13, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['failed_muon'])) == 1
    assert int(ak.sum(m['failed_gamma'])) == 0


def test_failed_gamma_names_the_photon():
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN),
                            (13, 0, *IN), (22, 0, *FWD)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['failed_gamma'])) == 1
    assert int(ak.sum(m['failed_muon'])) == 0


def test_failed_electron_names_the_electron():
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN),
                            (-11, 0, *FWD), (11, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, [-13, 13, -11, 11])
    assert int(ak.sum(m['failed_electron'])) == 1


def test_a_decay_can_fail_for_two_reasons_at_once():
    """failed_* keys are diagnostics, not a partition -- they may overlap."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *FWD),
                            (13, 0, *IN), (22, 0, *FWD)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    assert int(ak.sum(m['failed_muon'])) == 1
    assert int(ak.sum(m['failed_gamma'])) == 1


def test_in_acceptance_and_failed_are_mutually_exclusive():
    """A decay cannot both be kept and have lost a daughter."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *FWD),
                            (13, 0, *IN), (22, 0, *IN)],
                           [(221, -1, *IN), (-13, 0, *IN),
                            (13, 0, *IN), (22, 0, *IN)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    overlap = m['in_acceptance'] & (m['failed_muon'] | m['failed_gamma'])
    assert int(ak.sum(overlap)) == 0


def test_every_lost_decay_has_at_least_one_reason():
    """signal minus in_acceptance must be fully explained by the failed_* keys;
    a decay lost for no stated reason means the breakdown is broken."""
    tree = kinematic_tree([[(221, -1, *IN), (-13, 0, *FWD),
                            (13, 0, *IN), (22, 0, *IN)],
                           [(221, -1, *IN), (-13, 0, *IN),
                            (13, 0, *IN), (22, 0, *FWD)]])
    m = fid_reqs.fiducial_masks(tree, 221, MUMUGAMMA)
    lost = m['signal'] & ~m['in_acceptance']
    explained = m['failed_muon'] | m['failed_gamma'] | m['failed_electron']
    assert int(ak.sum(lost)) == 2
    assert int(ak.sum(lost & ~explained)) == 0


# --- fiducial_candidate_mask: reco candidates from in-acceptance decays ------

def reco_on(gen, prt_idx_gen, prt_idx_mom, n_candidates):
    """A one-event-per-row tree: the generator record `gen` (kinematic_tree
    rows) plus reco daughters wired to generator particles by `prt_idx_gen`
    and to candidates by `prt_idx_mom`."""
    kin = kinematic_tree(gen)
    rec = make_tree(prt_idx_gen=prt_idx_gen, prt_idx_mom=prt_idx_mom,
                    tag_m=[[0.0] * n for n in n_candidates])
    return ak.Array({**{f: kin[f] for f in kin.fields},
                     **{f: rec[f] for f in rec.fields}})


ETA_MUMUGAMMA_IN = [(221, -1, *IN), (-13, 0, *IN), (13, 0, *IN), (22, 0, *IN)]


def test_candidate_from_in_acceptance_decay_passes():
    tree = reco_on([ETA_MUMUGAMMA_IN], [[1, 2, 3]], [[0, 0, 0]], [1])
    mask = fid_reqs.fiducial_candidate_mask(tree, 221, MUMUGAMMA, target_ref='tag_m')
    assert mask.tolist() == [[True]]


def test_candidate_from_decay_with_a_daughter_outside_fails():
    """The decay is out of acceptance as a whole, so every candidate built
    from it fails, including through its in-acceptance daughters."""
    gen = [(221, -1, *IN), (-13, 0, *FWD), (13, 0, *IN), (22, 0, *IN)]
    tree = reco_on([gen], [[1, 2, 3]], [[0, 0, 0]], [1])
    mask = fid_reqs.fiducial_candidate_mask(tree, 221, MUMUGAMMA, target_ref='tag_m')
    assert mask.tolist() == [[False]]


def test_daughter_without_generator_match_fails():
    tree = reco_on([ETA_MUMUGAMMA_IN], [[1, 2, -1]], [[0, 0, 0]], [1])
    mask = fid_reqs.fiducial_candidate_mask(tree, 221, MUMUGAMMA, target_ref='tag_m')
    assert mask.tolist() == [[False]]


def test_candidate_with_a_stray_daughter_fails():
    """Candidate 0 is the signal decay; candidate 1 swaps the photon for a
    motherless stray."""
    gen = ETA_MUMUGAMMA_IN + [(22, -1, *IN)]
    tree = reco_on([gen], [[1, 2, 3, 1, 2, 4]], [[0, 0, 0, 1, 1, 1]], [2])
    mask = fid_reqs.fiducial_candidate_mask(tree, 221, MUMUGAMMA, target_ref='tag_m')
    assert mask.tolist() == [[True, False]]
