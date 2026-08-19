"""
Unit tests for cut_utils.reduce().

reduce() is fine -> coarse: it aggregates values from the level of an index
branch onto the level that index branch points into. Many sources map to one
target, so an aggregation is required.

The tests are grouped as: the five `how` modes, the identity elements, the
-1 "points nowhere" sentinel, event isolation, shape/edge cases, and round
trips against follow().
"""

import awkward as ak
import numpy as np
import pytest
from conftest import make_tree

import cut_utils


# --- the five aggregations ---------------------------------------------------

def test_all(reco):
    """True only where every daughter of the candidate is True.

    cand 0 has daughters [T, T] -> True
    cand 1 has daughters [F, T] -> False
    event 1's cand has  [T, F]  -> False
    """
    vals = ak.Array([[True, True, False, True], [True, False]])
    out = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='all',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[True, False], [False]]


def test_any(reco):
    """True where any daughter of the candidate is True.
    
    cand 0 has daughters [T, T] -> True
    cand 1 has daughters [F, T] -> True
    event 1's cand has  [F, F]  -> False

    """
    vals = ak.Array([[True, True, False, True], [False, False]])
    out = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='any',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[True, True], [False]]


def test_sum_counts_booleans(reco):
    """Booleans sum to a count -- the "how many daughters of this PID" idiom."""
    is_muplus = reco['prt_pid'] == -13
    out = cut_utils.reduce(reco, is_muplus, 'prt_idx_mom', how='sum',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[1.0, 1.0], [1.0]]


def test_sum_totals_floats(reco):
    vals = ak.Array([[1.5, 2.5, 10.0, 20.0], [0.25, 0.75]])
    out = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='sum',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[4.0, 30.0], [1.0]]


def test_min(reco):
    vals = ak.Array([[5.0, 2.0, 9.0, 7.0], [1.0, 4.0]])
    out = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='min',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[2.0, 7.0], [1.0]]


def test_max(reco):
    vals = ak.Array([[5.0, 2.0, 9.0, 7.0], [1.0, 4.0]])
    out = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='max',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[5.0, 9.0], [4.0]]


def test_invalid_how_raises(reco):
    """The five `how` modes are the only valid ones."""
    vals = ak.Array([[True, True, True, True], [True, True]])
    with pytest.raises(ValueError, match='Invalid how'):
        cut_utils.reduce(reco, vals, 'prt_idx_mom', how='median',
                         target_ref='tag_dtf_m')


# --- identity elements -------------------------------------------------------
#
# A target that nothing points at reduces to the identity of its aggregation.
# This is load-bearing: it is what lets a candidate with no muon daughters pass
# a muon cut, and it is the trap that makes how='all' the wrong choice when
# propagating a selection upward.

@pytest.mark.parametrize('how, identity', [
    ('all', True),
    ('any', False),
    ('sum', 0.0),
    ('min', np.inf),
    ('max', -np.inf),
])
def test_unreferenced_target_gets_the_identity(how, identity):
    """A generator particle with no children lands on the identity.

    eta -> mu+ mu-: the eta (index 0) collects both muons, but the muons
    themselves are final state and collect nothing. That is the norm on the
    generator side -- most mc_* entries have no children at all -- which is why
    the identity is worth pinning for every aggregation.
    """
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]])
    vals = (ak.Array([[True, True, True]]) if how in ('all', 'any')
            else ak.Array([[1.0, 2.0, 3.0]]))
    out = cut_utils.reduce(tree, vals, 'mc_idx_mom', how=how, target_ref='mc_pid')
    assert out[0][1] == identity      # the mu+, no children
    assert out[0][2] == identity      # the mu-, no children


def test_all_identity_is_why_any_is_right_for_propagating_up():
    """Why how='any' is the right choice for propagating selections up.

    When pushing a reco selection up onto generator particles: one reco daughter
    matched generator particle 1. With how='all' the three generator particles
    nothing points at come back True as well, because an empty all() is True --
    so the selection appears to cover the whole record. how='any' reports only
    the particle that was really matched.
    """
    tree = make_tree(mc_pid=[[221, -13, 13, 22]], prt_idx_gen=[[1]])
    selected = ak.Array([[True]])
    as_all = cut_utils.reduce(tree, selected, 'prt_idx_gen', how='all',
                              target_ref='mc_pid')
    as_any = cut_utils.reduce(tree, selected, 'prt_idx_gen', how='any',
                              target_ref='mc_pid')
    assert as_all.tolist() == [[True, True, True, True]]     # 3 are spurious
    assert as_any.tolist() == [[False, True, False, False]]  # only the match


# --- the -1 "points nowhere" sentinel ---------------------------------------

@pytest.mark.parametrize('how', ['all', 'any', 'sum', 'min', 'max'])
def test_sentinel_does_not_raise(gen, how):
    """mc_idx_mom is -1 for the eta. Every aggregation must cope with it.

    Before the sentinel was handled, 'all' and 'sum' raised ValueError from
    np.bincount and 'min'/'max' silently wrapped via ufunc.at.
    """
    vals = (ak.Array([[True, True, True], [True, True, True]])
            if how in ('all', 'any') else
            ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    out = cut_utils.reduce(gen, vals, 'mc_idx_mom', how=how, target_ref='mc_pid')
    assert ak.num(out).tolist() == [3, 3]


@pytest.mark.parametrize('how, vals, guarded', [
    ('all', [[True, True, True], [False, True, True]], 'or'),
    ('any', [[True, False, False], [True, False, False]], 'and'),
    ('sum', [[True, False, False], [True, False, False]], 'and'),
])
def test_guarding_with_valid_is_now_a_no_op(gen, how, vals, guarded):
    """Callers used to have to AND/OR valid() in before reducing, with the
    polarity depending on `how`. reduce() drops non-pointing entries itself now,
    so the guard changes nothing -- both spellings must agree."""
    vals = ak.Array(vals)
    ok = cut_utils.valid(gen, 'mc_idx_mom')
    guard = (vals | ~ok) if guarded == 'or' else (vals & ok)
    plain = cut_utils.reduce(gen, vals, 'mc_idx_mom', how=how, target_ref='mc_pid')
    with_guard = cut_utils.reduce(gen, guard, 'mc_idx_mom', how=how,
                                  target_ref='mc_pid')
    assert plain.tolist() == with_guard.tolist()


def test_sentinel_does_not_leak_into_the_previous_event(gen):
    """Regression test for the subtle one.

    The offset that makes ids globally unique is added per event. Event 1's -1
    plus its offset of 3 becomes 2 -- a perfectly valid id pointing at event 0's
    third particle. So the sentinel MUST be filtered on the local index, before
    the offset is added. Filtering afterwards on `ids >= 0` catches event 0's -1
    but not event 1's, and event 1's failure gets written into event 0.

    Here only event 1's eta fails. Event 0 must be untouched.
    """
    vals = ak.Array([[True, True, True], [False, True, True]])
    out = cut_utils.reduce(gen, vals, 'mc_idx_mom', how='all',
                           target_ref='mc_pid')
    assert out[0].tolist() == [True, True, True]   # event 0 unaffected
    assert out[1].tolist() == [True, True, True]   # the eta is not its own child


def test_sentinel_entries_do_not_contribute_to_sums(gen):
    """The eta is motherless, so it must not be counted as anyone's daughter."""
    everything = ak.ones_like(gen['mc_pid'], dtype=bool)
    out = cut_utils.reduce(gen, everything, 'mc_idx_mom', how='sum',
                           target_ref='mc_pid')
    # The eta (index 0) has exactly 2 daughters; the muons have none.
    assert out.tolist() == [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_all_sentinels_reduces_to_all_identity():
    """An index branch that points nowhere at all leaves every target at the
    identity rather than raising."""
    tree = make_tree(mc_pid=[[221, 22]], mc_idx_mom=[[-1, -1]])
    vals = ak.Array([[False, False]])
    out = cut_utils.reduce(tree, vals, 'mc_idx_mom', how='all',
                           target_ref='mc_pid')
    assert out.tolist() == [[True, True]]


# --- event isolation ---------------------------------------------------------

def test_events_are_independent(reco):
    """Flipping a value in event 1 must not change event 0's result."""
    base = ak.Array([[True, True, True, True], [True, True]])
    flipped = ak.Array([[True, True, True, True], [False, True]])
    a = cut_utils.reduce(reco, base, 'prt_idx_mom', how='all', target_ref='tag_dtf_m')
    b = cut_utils.reduce(reco, flipped, 'prt_idx_mom', how='all', target_ref='tag_dtf_m')
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() != b[1].tolist()


def test_same_local_index_in_different_events_stays_separate():
    """Both events use local index 0; the offset must keep them apart."""
    tree = make_tree(tag=[[1.0], [2.0]], idx=[[0], [0]])
    vals = ak.Array([[True], [False]])
    out = cut_utils.reduce(tree, vals, 'idx', how='all', target_ref='tag')
    assert out.tolist() == [[True], [False]]


# --- shape and edge cases ----------------------------------------------------

def test_result_is_shaped_like_target_ref(reco):
    vals = ak.Array([[True, True, True, True], [True, True]])
    out = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='all',
                           target_ref='tag_dtf_m')
    assert ak.num(out).tolist() == ak.num(reco['tag_dtf_m']).tolist()


def test_target_ref_can_be_a_different_collection():
    """prt_idx_gen maps reco daughters onto generator particles, so the result
    is gen-shaped even though the input was reco-shaped."""
    tree = make_tree(
        mc_pid=[[221, -13, 13, 22]],       # 4 generator particles
        prt_idx_gen=[[1, 2, 3]],           # 3 reco daughters
    )
    vals = ak.Array([[True, False, True]])
    out = cut_utils.reduce(tree, vals, 'prt_idx_gen', how='any',
                           target_ref='mc_pid')
    assert ak.num(out).tolist() == [4]
    assert out.tolist() == [[False, True, False, True]]


def test_out_of_order_indices():
    """Sources need not be stored grouped by target."""
    tree = make_tree(tag=[[10.0, 20.0]], idx=[[1, 0, 1, 0]])
    vals = ak.Array([[True, True, False, True]])
    out = cut_utils.reduce(tree, vals, 'idx', how='all', target_ref='tag')
    assert out.tolist() == [[True, False]]


def test_event_with_no_sources():
    tree = make_tree(tag=[[1.0], [2.0]], idx=[[0], []])
    vals = ak.Array([[True], []])
    out = cut_utils.reduce(tree, vals, 'idx', how='any', target_ref='tag')
    assert out.tolist() == [[True], [False]]


def test_empty_tree():
    """No targets anywhere -- the n_total == 0 short circuit."""
    tree = make_tree(tag=[[], []], idx=[[], []])
    vals = ak.Array([[], []])
    out = cut_utils.reduce(tree, vals, 'idx', how='all', target_ref='tag')
    assert out.tolist() == [[], []]


def test_float_index_branch_is_cast_not_rounded():
    tree = make_tree(tag=[[0.0, 0.0, 0.0]], idx=[[2.0, 2.0]])
    vals = ak.Array([[True, False]])
    out = cut_utils.reduce(tree, vals, 'idx', how='all', target_ref='tag')
    assert out.tolist() == [[True, True, False]]


# --- round trips -------------------------------------------------------------

def test_reduce_then_follow_broadcasts_the_verdict_back(reco):
    """Not an identity -- reduce() loses per-daughter detail by construction --
    but every daughter must come back with its own candidate's verdict."""
    vals = ak.Array([[True, True, False, True], [True, False]])
    up = cut_utils.reduce(reco, vals, 'prt_idx_mom', how='all',
                          target_ref='tag_dtf_m')
    down = cut_utils.follow(reco, up, 'prt_idx_mom')
    assert down.tolist() == [[True, True, False, False], [False, False]]


def test_sum_of_ones_equals_daughter_multiplicity(reco):
    ones = ak.ones_like(reco['prt_idx_mom'], dtype=bool)
    out = cut_utils.reduce(reco, ones, 'prt_idx_mom', how='sum',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[2.0, 2.0], [2.0]]


# --- every use of reduce() this framework has ---------------------------------
#
# reduce() moves a value UP an index branch, collapsing many sources onto one
# target. The aggregation is the whole vocabulary: 'all' for requirements,
# 'any' for propagating a selection, 'sum' for counting, 'min'/'max' for
# extrema. These are the calls the analysis actually makes.

def test_use_pid_cut_every_muon_must_pass(reco):
    """1. 'all' : the pnn_mu cut. "passes OR isn't a muon" folded in, so a
    candidate with no muon daughters passes rather than fails."""
    pnn = ak.Array([[0.99, 0.97, 0.99, 0.20], [0.96, 0.98]])
    is_muon = abs(reco['prt_pid']) == 13
    muon_ok = (pnn > 0.95) | ~is_muon
    out = cut_utils.reduce(reco, muon_ok, 'prt_idx_mom', how='all',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[True, False], [True]]


def test_use_kinematic_cut_every_daughter_fast_enough(reco):
    """2. 'all' : a pT requirement applied at the daughter level and reduced
    onto the candidate, exactly like the PID cut."""
    pt = ak.Array([[600.0, 700.0, 600.0, 100.0], [800.0, 900.0]])
    out = cut_utils.reduce(reco, pt > 500, 'prt_idx_mom', how='all',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[True, False], [True]]


def test_use_multiplicity_exactly_one_photon():
    """3. 'sum' == 1 : the Dalitz well-formedness requirement."""
    tree = make_tree(tag_dtf_m=[[550.0, 549.0]],
                     prt_pid=[[-13, 13, 22, -13, 13, 22, 22]],
                     prt_idx_mom=[[0, 0, 0, 1, 1, 1, 1]])
    n_gamma = cut_utils.reduce(tree, tree['prt_pid'] == 22, 'prt_idx_mom',
                               how='sum', target_ref='tag_dtf_m')
    assert (n_gamma == 1).tolist() == [[True, False]]


def test_use_daughter_multiplicity_per_candidate(reco):
    """4. 'sum' of ones : how many daughters does each candidate have?"""
    ones = ak.ones_like(reco['prt_idx_mom'], dtype=bool)
    out = cut_utils.reduce(reco, ones, 'prt_idx_mom', how='sum',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[2.0, 2.0], [2.0]]


def test_use_worst_daughter_pnn_per_candidate(reco):
    """5. 'min' : the weakest daughter's PID score, for a cut expressed as a
    candidate-level quantity rather than an all-reduction."""
    pnn = ak.Array([[0.99, 0.97, 0.99, 0.20], [0.96, 0.98]])
    out = cut_utils.reduce(reco, pnn, 'prt_idx_mom', how='min',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[0.97, 0.20], [0.96]]


def test_use_highest_daughter_pt_per_candidate(reco):
    """6. 'max' : the leading daughter's pT, a common trigger-adjacent cut."""
    pt = ak.Array([[600.0, 1500.0, 600.0, 100.0], [800.0, 900.0]])
    out = cut_utils.reduce(reco, pt, 'prt_idx_mom', how='max',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[1500.0, 600.0], [900.0]]


def test_use_flag_candidates_containing_a_stray_match():
    """7. 'any' : push a per-daughter truth flag up to condemn its candidate --
    the background-analysis chain, gen -> reco -> candidate."""
    tree = make_tree(
        tag_dtf_m=[[550.0, 549.0]],
        mc_pid=[[221, -13, 13, 211]],
        mc_idx_mom=[[-1, 0, 0, -1]],
        prt_idx_gen=[[1, 2, 1, 3]],
        prt_idx_mom=[[0, 0, 1, 1]],
    )
    signal = cut_utils.decay_mask(tree, 221, [-13, 13])
    stray = ~cut_utils.valid(tree, 'mc_idx_mom') & ~signal
    matched_stray = cut_utils.follow(tree, stray, 'prt_idx_gen')
    out = cut_utils.reduce(tree, matched_stray, 'prt_idx_mom', how='any',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[False, True]]


def test_use_reco_daughters_onto_generator_particles():
    """8. prt_* -> mc_* : reduce across collections, not just up a level.
    "was this generator particle reconstructed at all?"."""
    tree = make_tree(mc_pid=[[221, -13, 13, 22]], prt_idx_gen=[[1, 2]])
    everything = ak.ones_like(tree['prt_idx_gen'], dtype=bool)
    out = cut_utils.reduce(tree, everything, 'prt_idx_gen', how='any',
                           target_ref='mc_pid')
    assert out.tolist() == [[False, True, True, False]]


def test_use_reconstruction_multiplicity_per_generator_particle():
    """9. prt_* -> mc_* with 'sum' : how many reco tracks claimed the same
    generator particle? >1 means candidates are sharing a track."""
    tree = make_tree(mc_pid=[[221, -13, 13]], prt_idx_gen=[[1, 2, 1, 2]])
    everything = ak.ones_like(tree['prt_idx_gen'], dtype=bool)
    out = cut_utils.reduce(tree, everything, 'prt_idx_gen', how='sum',
                           target_ref='mc_pid')
    assert out.tolist() == [[0.0, 2.0, 2.0]]


def test_use_fiducial_all_daughters_of_a_generator_mother():
    """10. mc_* -> mc_* with 'all' : the fid_reqs requirement, on the only
    self-referential index branch."""
    tree = make_tree(mc_pid=[[221, -13, 13, 22]], mc_idx_mom=[[-1, 0, 0, 0]])
    ok = ak.Array([[False, True, True, False]])   # mother has no rule; gamma out
    out = cut_utils.reduce(tree, ok, 'mc_idx_mom', how='all',
                           target_ref='mc_pid')
    assert out.tolist() == [[False, True, True, True]]


def test_use_name_the_failing_species():
    """11. mc_* -> mc_* with 'any' : which mothers lost a MUON specifically --
    the failure-reason breakdown in fid_reqs.fiducial_masks."""
    tree = make_tree(mc_pid=[[221, -13, 13, 22]], mc_idx_mom=[[-1, 0, 0, 0]])
    ok = ak.Array([[False, False, True, True]])   # the mu+ failed
    lost_muon = cut_utils.reduce(tree, ~ok & (abs(tree['mc_pid']) == 13),
                                 'mc_idx_mom', how='any', target_ref='mc_pid')
    lost_gamma = cut_utils.reduce(tree, ~ok & (tree['mc_pid'] == 22),
                                  'mc_idx_mom', how='any', target_ref='mc_pid')
    assert lost_muon.tolist() == [[True, False, False, False]]
    assert lost_gamma.tolist() == [[False, False, False, False]]


def test_use_count_daughters_of_a_generator_mother():
    """12. mc_* -> mc_* with 'sum' : the counting that decay_mask is built on,
    and the call that was impossible before the -1 sentinel was handled."""
    tree = make_tree(mc_pid=[[221, -13, 13, 22]], mc_idx_mom=[[-1, 0, 0, 0]])
    n = cut_utils.reduce(tree, ak.ones_like(tree['mc_pid'], dtype=bool),
                         'mc_idx_mom', how='sum', target_ref='mc_pid')
    assert n.tolist() == [[3.0, 0.0, 0.0, 0.0]]


# --- identifying a decay by counting its daughters ----------------------------
#
# The pattern decay_mask() is built from, spelled out one reduce at a time. A
# mother is the decay you want when every required PID appears the right number
# of times AND the total daughter count matches -- the second check is what
# rejects a mother carrying an extra daughter on top of the ones you asked for.

@pytest.fixture
def three_decays():
    """One event, three generator mothers with different daughter sets.

        index 0  eta -> mu+ mu- gamma      the decay we want
        index 4  eta -> mu+ mu-            missing the photon
        index 7  eta -> mu+ mu- gamma gamma  one photon too many
    """
    return make_tree(
        mc_pid=[[221, -13, 13, 22,
                 221, -13, 13,
                 221, -13, 13, 22, 22]],
        mc_idx_mom=[[-1, 0, 0, 0,
                     -1, 4, 4,
                     -1, 7, 7, 7, 7]],
    )


def n_daughters_of_pid(tree, pid_code):
    """How many daughters of this PID does each generator particle have?"""
    return cut_utils.reduce(tree, tree['mc_pid'] == pid_code, 'mc_idx_mom',
                            how='sum', target_ref='mc_pid')


def test_count_muplus_daughters(three_decays):
    """One mu+ under each of the three mothers; everything else has none."""
    n = n_daughters_of_pid(three_decays, -13)
    assert n.tolist() == [[1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0]]


def test_count_muminus_daughters(three_decays):
    n = n_daughters_of_pid(three_decays, 13)
    assert n.tolist() == [[1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0]]


def test_count_photon_daughters(three_decays):
    """This is the count that separates the three mothers: 1, 0 and 2."""
    n = n_daughters_of_pid(three_decays, 22)
    assert n[0][0] == 1.0      # eta -> mu mu gamma
    assert n[0][4] == 0.0      # eta -> mu mu, no photon
    assert n[0][7] == 2.0      # eta -> mu mu gamma gamma


def test_count_total_daughters(three_decays):
    """Reducing a constant True counts children regardless of PID. The
    motherless mothers contribute nothing to anyone, so they count 0."""
    ones = ak.ones_like(three_decays['mc_pid'], dtype=bool)
    n = cut_utils.reduce(three_decays, ones, 'mc_idx_mom', how='sum',
                         target_ref='mc_pid')
    assert n[0][0] == 3.0
    assert n[0][4] == 2.0
    assert n[0][7] == 4.0
    assert n[0][1] == 0.0      # a final-state muon has no children


def test_species_counts_alone_do_not_identify_the_decay(three_decays):
    """Why the total is needed: mother 7 has exactly one mu+ and one mu- and at
    least one gamma, so PID counts alone would wrongly accept it."""
    right_species = ((n_daughters_of_pid(three_decays, -13) == 1)
                     & (n_daughters_of_pid(three_decays, 13) == 1)
                     & (n_daughters_of_pid(three_decays, 22) >= 1))
    assert bool(right_species[0][0])          # the real decay, correctly accepted
    assert bool(right_species[0][7])          # the impostor slips through too


def test_species_counts_plus_total_identify_the_decay(three_decays):
    """Adding the total count rejects the impostor, and the whole thing is
    exactly what decay_mask() does internally."""
    ones = ak.ones_like(three_decays['mc_pid'], dtype=bool)
    total = cut_utils.reduce(three_decays, ones, 'mc_idx_mom', how='sum',
                             target_ref='mc_pid')
    is_signal = ((three_decays['mc_pid'] == 221)
                 & (n_daughters_of_pid(three_decays, -13) == 1)
                 & (n_daughters_of_pid(three_decays, 13) == 1)
                 & (n_daughters_of_pid(three_decays, 22) == 1)
                 & (total == 3))
    assert is_signal.tolist() == [[True, False, False, False,
                                   False, False, False,
                                   False, False, False, False, False]]


def test_matches_decay_mask(three_decays):
    """The hand-rolled version and the library function must agree."""
    ones = ak.ones_like(three_decays['mc_pid'], dtype=bool)
    total = cut_utils.reduce(three_decays, ones, 'mc_idx_mom', how='sum',
                             target_ref='mc_pid')
    manual = ((three_decays['mc_pid'] == 221)
              & (n_daughters_of_pid(three_decays, -13) == 1)
              & (n_daughters_of_pid(three_decays, 13) == 1)
              & (n_daughters_of_pid(three_decays, 22) == 1)
              & (total == 3))
    library = cut_utils.decay_mask(three_decays, 221, [-13, 13, 22])
    assert manual.tolist() == library.tolist()
