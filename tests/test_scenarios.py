"""
Scenario tests: the situations this ntuple actually produces.

Each test builds the smallest tree that reproduces one real-world structure and
pins down what the framework should say about it. These are the cases that broke
earlier implementations -- stride grouping, the -1 sentinel, shared tracks --
so they are written as narratives rather than API coverage.
"""

import awkward as ak

import cut_utils
from conftest import make_tree

# --- mis-identification ------------------------------------------------------

def test_reco_muon_matched_to_a_generator_pion():
    """A reco track called a muon whose MC match is a pion (211).

    Real: 'matched gen pids' on eta2mumu MC comes back as [-13, 13, 211]. The
    framework must report the TRUTH pid, not the reco hypothesis, when asked
    through prt_idx_gen.
    """
    tree = make_tree(
        mc_pid=[[221, -13, 13, 211]],
        mc_idx_mom=[[-1, 0, 0, -1]],
        prt_pid=[[-13, 13]],        # both reconstructed as muons
        prt_idx_gen=[[1, 3]],       # ...but track 1 really matched the pion
    )
    truth = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    assert truth.tolist() == [[-13.0, 211.0]]

    misid = truth != tree['prt_pid']
    assert misid.tolist() == [[False, True]]


def test_misidentified_daughter_flags_its_candidate():
    """One mis-ID'd daughter must condemn the whole candidate, via reduce."""
    tree = make_tree(
        tag_dtf_m=[[550.0, 549.0]],
        mc_pid=[[221, -13, 13, 211]],
        mc_idx_mom=[[-1, 0, 0, -1]],
        prt_pid=[[-13, 13, -13, 13]],
        prt_idx_gen=[[1, 2, 1, 3]],
        prt_idx_mom=[[0, 0, 1, 1]],
    )
    truth = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    misid = truth != tree['prt_pid']
    bad = cut_utils.reduce(tree, misid, 'prt_idx_mom', how='any',
                           target_ref='tag_dtf_m')
    # Candidate 0's daughters both match truth; candidate 1 holds the pion.
    assert bad.tolist() == [[False, True]]


def test_a_pion_is_not_a_signal_daughter():
    """A stray pion in the record must not count toward the decay."""
    tree = make_tree(mc_pid=[[221, -13, 13, 211]], mc_idx_mom=[[-1, 0, 0, -1]])
    assert cut_utils.decay_mask(tree, 221, [-13, 13]).tolist() \
        == [[True, False, False, False]]


# --- multiple reco candidates per generator candidate ------------------------

def test_two_candidates_share_one_generator_particle(shared_track):
    """Daughters 1 and 2 both matched gen particle 2 -- a shared track.

    This is why daughter -> gen -> daughter is one-way lossy: 'any' widens the
    selection to both candidates on the way back down.
    """
    selected = ak.Array([[True, False]])                    # candidate 0 only
    dmask = cut_utils.follow(shared_track, selected, 'prt_idx_mom')
    assert dmask.tolist() == [[True, True, False, False]]

    gen = cut_utils.reduce(shared_track, dmask, 'prt_idx_gen', how='any',
                           target_ref='mc_pid')
    back = cut_utils.follow(shared_track, gen, 'prt_idx_gen')
    # Daughter 2 comes back True even though its candidate was not selected.
    assert back.tolist() == [[True, True, True, False]]
    assert back.tolist() != dmask.tolist()


def test_two_candidates_from_one_generator_decay():
    """Two reco candidates, both pointing at the same generator eta.

    Counting candidates per generator mother is a reduce chain: reco daughter
    -> gen particle, then gen daughter -> gen mother.
    """
    tree = make_tree(
        tag_dtf_m=[[550.0, 549.0]],
        prt_idx_mom=[[0, 0, 1, 1]],
        prt_idx_gen=[[1, 2, 1, 2]],
        mc_pid=[[221, -13, 13]],
        mc_idx_mom=[[-1, 0, 0]],
    )
    everything = ak.ones_like(tree['prt_idx_gen'], dtype=bool)
    matched_gen = cut_utils.reduce(tree, everything, 'prt_idx_gen', how='any',
                                   target_ref='mc_pid')
    assert matched_gen.tolist() == [[False, True, True]]

    n_reco_per_gen = cut_utils.reduce(tree, everything, 'prt_idx_gen',
                                      how='sum', target_ref='mc_pid')
    # Each generator muon was reconstructed twice, once per candidate.
    assert n_reco_per_gen.tolist() == [[0.0, 2.0, 2.0]]


def test_best_candidate_survives_a_shared_track(shared_track):
    """Sharing a track must not stop best_candidate_mask picking exactly one."""
    tree = ak.with_field(shared_track,
                         ak.Array([[17.0, 14.0]]), 'tag_dtf_chi2')
    eligible = ak.Array([[True, True]])
    best = cut_utils.best_candidate_mask(tree, 'tag_dtf_chi2', eligible)
    assert best.tolist() == [[False, True]]      # lower chi2 wins
    assert int(ak.sum(best)) == 1


# --- generator particles with no daughters -----------------------------------

def test_motherless_daughter_particle():
    """A muon with mc_idx_mom == -1: reconstructed and matched, but not from
    the signal decay. It must not be counted as anyone's daughter."""
    tree = make_tree(mc_pid=[[221, -13, 13, -13]],
                     mc_idx_mom=[[-1, 0, 0, -1]])
    everything = ak.ones_like(tree['mc_pid'], dtype=bool)
    n_daughters = cut_utils.reduce(tree, everything, 'mc_idx_mom', how='sum',
                                   target_ref='mc_pid')
    assert n_daughters.tolist() == [[2.0, 0.0, 0.0, 0.0]]


def test_childless_particles_reduce_to_the_identity():
    """The muons have no daughters, so how='all' gives them True (empty all)
    and how='any' gives them False. Both are correct and both are traps."""
    tree = make_tree(mc_pid=[[221, -13, 13]], mc_idx_mom=[[-1, 0, 0]])
    vals = ak.Array([[True, False, False]])
    as_all = cut_utils.reduce(tree, vals, 'mc_idx_mom', how='all',
                              target_ref='mc_pid')
    as_any = cut_utils.reduce(tree, vals, 'mc_idx_mom', how='any',
                              target_ref='mc_pid')
    assert as_all.tolist() == [[False, True, True]]   # eta fails; muons vacuous
    assert as_any.tolist() == [[False, False, False]]


def test_every_particle_motherless():
    """A record of pure strays -- no decay at all. Nothing should match and
    nothing should raise."""
    tree = make_tree(mc_pid=[[22, 22, 211]], mc_idx_mom=[[-1, -1, -1]])
    assert not ak.any(cut_utils.decay_mask(tree, 221, [-13, 13]))
    everything = ak.ones_like(tree['mc_pid'], dtype=bool)
    counts = cut_utils.reduce(tree, everything, 'mc_idx_mom', how='sum',
                              target_ref='mc_pid')
    assert counts.tolist() == [[0.0, 0.0, 0.0]]


def test_strays_are_identifiable(gen_with_strays):
    """The background-analysis entry point: motherless AND not the signal."""
    signal = cut_utils.decay_mask(gen_with_strays, 221, [-13, 13, 22])
    stray = ~cut_utils.valid(gen_with_strays, 'mc_idx_mom') & ~signal
    assert stray.tolist() == [[False, False, False, False, True, True, True]]
    pids = ak.flatten(gen_with_strays['mc_pid'][stray]).tolist()
    assert pids == [22.0, -211.0, 2112.0]


# --- multiple generator candidates -------------------------------------------

def test_two_generator_decays_are_independent(two_gen_candidates):
    """Each eta's daughter counts must come from its own subtree."""
    everything = ak.ones_like(two_gen_candidates['mc_pid'], dtype=bool)
    n = cut_utils.reduce(two_gen_candidates, everything, 'mc_idx_mom',
                         how='sum', target_ref='mc_pid')
    assert n.tolist() == [[3.0, 0, 0, 0, 3.0, 0, 0, 0]]


def test_any_versus_all_over_generator_mothers(two_gen_candidates):
    """With two decays, 'does the event have a good one' and 'are they all
    good' genuinely differ. This is the open decision in fid_reqs.py."""
    signal = cut_utils.decay_mask(two_gen_candidates, 221, [-13, 13, 22])
    # Pretend the second eta's photon (index 7) is out of acceptance.
    ok = ak.Array([[True] * 7 + [False]])
    good = cut_utils.reduce(two_gen_candidates, ok, 'mc_idx_mom', how='all',
                            target_ref='mc_pid') & signal
    assert good.tolist() == [[True, False, False, False,
                              False, False, False, False]]
    assert bool(ak.any(good, axis=1)[0]) is True            # ak.any -> keep
    assert int(ak.sum(good)) != int(ak.sum(signal))         # ak.all -> drop


def test_second_decay_daughters_point_at_the_second_mother(two_gen_candidates):
    """Guards against an off-by-one in the offset arithmetic: daughter 5 must
    resolve to the eta at index 4, not the one at index 0."""
    mother_pid = cut_utils.follow(two_gen_candidates, two_gen_candidates['mc_pid'],
                                  'mc_idx_mom')
    ok = cut_utils.valid(two_gen_candidates, 'mc_idx_mom')
    assert mother_pid[ok].tolist() == [[221.0, 221.0, 221.0,
                                        221.0, 221.0, 221.0]]


def test_generator_decays_in_separate_events_stay_separate():
    """The same structure split across two events must give the same answer as
    one event holding both -- the per-event offset is the only thing keeping
    them apart."""
    one = make_tree(mc_pid=[[221, -13, 13, 221, -13, 13]],
                    mc_idx_mom=[[-1, 0, 0, -1, 3, 3]])
    two = make_tree(mc_pid=[[221, -13, 13], [221, -13, 13]],
                    mc_idx_mom=[[-1, 0, 0], [-1, 0, 0]])
    a = cut_utils.decay_mask(one, 221, [-13, 13])
    b = cut_utils.decay_mask(two, 221, [-13, 13])
    assert ak.to_list(ak.flatten(a)) == ak.to_list(ak.flatten(b))
