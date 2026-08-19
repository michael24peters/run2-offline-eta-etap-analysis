"""
Recipes: one test per question you would actually ask this framework.

Each test is named for the question, and its body is the shortest correct way to
answer it with valid() / follow() / reduce(). Read this file as a cookbook --
if you know what you want to ask but not which primitive to reach for, the
answer is probably here.

One event, eta -> mu+ mu- gamma, small enough to read by eye.

  GENERATOR
    index        0     1    2    3  |  4      5     6
    mc_pid     [221, -13,  13,  22, | 22,  -211, 2112]
    mc_idx_mom [ -1,   0,   0,   0, | -1,    -1,   -1]
                 |    |____|____|   |  |______|______|
                 |      daughters   |    strays: MC-matched junk that the
                 the eta,           |    reconstruction picked up. Motherless,
                 top of the chain   |    so nobody's children.

  RECO -- two candidates. COMPOSITION IS FIXED by the decay descriptor: every
  candidate is exactly one mu+, one mu- and one gamma, verified on real MC
  (5914 of 5914 candidates). A candidate with any other composition is
  malformed, not background.

    index         0    1    2  |  3    4    5
    prt_idx_mom [  0,   0,   0 |  1,   1,   1]
    prt_pid     [-13,  13,  22 |-13,  13,  22]
    prt_idx_gen [  1,   2,   3 |  1,   5,   4]
                                      ^    ^
                                      |    matched the STRAY photon
                                      matched the STRAY pion: a mis-ID

  What DOES vary is truth matching -- which generator particle each track
  really was. On real MC 25% of reco daughters match a motherless generator
  particle and 5.6% disagree with their own reco PID. Candidate 0 here is
  cleanly matched; candidate 1 shares a track and picks up two bad matches.
  That is the distinction the recipes below turn on: composition is guaranteed,
  truth matching is not.
"""

import awkward as ak
import numpy as np
import pytest
from conftest import make_tree

import cut_utils


@pytest.fixture
def event():
    """One event: a clean decay, MC-matched junk, and two reco candidates."""
    return make_tree(
        mc_pid=[[221, -13, 13, 22, 22, -211, 2112]],
        mc_idx_mom=[[-1, 0, 0, 0, -1, -1, -1]],
        tag_dtf_m=[[550.0, 549.0]],
        tag_dtf_chi2=[[14.0, 17.0]],
        prt_idx_mom=[[0, 0, 0, 1, 1, 1]],
        prt_idx_gen=[[1, 2, 3, 1, 5, 4]],
        prt_pid=[[-13, 13, 22, -13, 13, 22]],
        prt_pnn_mu=[[0.99, 0.97, 0.0, 0.99, 0.20, 0.0]],
    )


# --- picking particles out of the generator record ---------------------------

def test_which_generator_particles_are_daughters(event):
    """"Do I have a mother?" -- valid() is the whole answer, no follow needed."""
    assert cut_utils.valid(event, 'mc_idx_mom').tolist() \
        == [[False, True, True, True, False, False, False]]


def test_which_generator_particles_are_top_of_a_chain(event):
    """The complement. Includes both the signal eta AND the strays, which is
    why it is never the background definition on its own."""
    assert (~cut_utils.valid(event, 'mc_idx_mom')).tolist() \
        == [[True, False, False, False, True, True, True]]


def test_which_generator_particles_are_mothers(event):
    """"Does anything point at me?". Different from "is top of a chain": the eta
    is both, a stray is neither."""
    ones = ak.ones_like(event['mc_pid'], dtype=bool)
    has_children = cut_utils.reduce(event, ones, 'mc_idx_mom', how='any',
                                    target_ref='mc_pid')
    assert has_children.tolist() \
        == [[True, False, False, False, False, False, False]]


def test_how_many_children_does_each_particle_have(event):
    """Same question, counted instead of flagged."""
    ones = ak.ones_like(event['mc_pid'], dtype=bool)
    n = cut_utils.reduce(event, ones, 'mc_idx_mom', how='sum',
                         target_ref='mc_pid')
    assert n.tolist() == [[3.0, 0, 0, 0, 0, 0, 0]]


def test_what_is_my_mothers_pid(event):
    """follow() along mc_idx_mom, guarded. Without the ak.where, index 0 reads
    mc_pid[-1] = 2112 and claims the eta's mother is a neutron."""
    raw = cut_utils.follow(event, event['mc_pid'], 'mc_idx_mom')
    guarded = ak.where(cut_utils.valid(event, 'mc_idx_mom'), raw, np.nan)
    assert raw[0][0] == 2112.0                    # the wrap
    assert guarded[0][1:4].tolist() == [221.0, 221.0, 221.0]
    assert np.isnan(guarded[0][0])


def test_which_particles_are_daughters_of_an_eta(event):
    """A boolean version of the above: follow the mask down, AND off the wrap."""
    is_eta = event['mc_pid'] == 221
    out = cut_utils.follow(event, is_eta, 'mc_idx_mom') & cut_utils.valid(
        event, 'mc_idx_mom')
    assert out.tolist() == [[False, True, True, True, False, False, False]]


# --- signal versus background ------------------------------------------------

def test_which_particles_are_the_signal_decay(event):
    """decay_mask flags the MOTHER of a matching decay, not its daughters."""
    assert cut_utils.decay_mask(event, 221, [-13, 13, 22]).tolist() \
        == [[True, False, False, False, False, False, False]]


def test_which_particles_are_stray_background(event):
    """Motherless AND not the signal. The `& ~signal` is what stops every signal
    eta being counted as background -- it is motherless too."""
    signal = cut_utils.decay_mask(event, 221, [-13, 13, 22])
    stray = ~cut_utils.valid(event, 'mc_idx_mom') & ~signal
    assert stray.tolist() == [[False, False, False, False, True, True, True]]
    assert ak.flatten(event['mc_pid'][stray]).tolist() == [22.0, -211.0, 2112.0]


def test_signal_daughters_and_strays_do_not_overlap(event):
    """The two categories partition the interesting particles: a particle is a
    decay product or it is junk, never both."""
    signal = cut_utils.decay_mask(event, 221, [-13, 13, 22])
    ok = cut_utils.valid(event, 'mc_idx_mom')
    daughters = cut_utils.follow(event, signal, 'mc_idx_mom') & ok
    stray = ~ok & ~signal
    assert int(ak.sum(daughters & stray)) == 0


# --- crossing to the reco side -----------------------------------------------

def test_what_did_each_reco_daughter_really_match(event):
    """follow() along prt_idx_gen pulls truth PIDs onto the reco daughters."""
    truth = cut_utils.follow(event, event['mc_pid'], 'prt_idx_gen')
    assert truth.tolist() == [[-13.0, 13.0, 22.0, -13.0, -211.0, 22.0]]


def test_which_reco_daughters_are_misidentified(event):
    """Compare the reco hypothesis against the truth match, elementwise.
    Daughter 4 was reconstructed as a mu- but really matched a pion."""
    truth = cut_utils.follow(event, event['mc_pid'], 'prt_idx_gen')
    assert (truth != event['prt_pid']).tolist() \
        == [[False, False, False, False, True, False]]


def test_which_reco_daughters_matched_a_stray(event):
    """Push a generator-level background flag DOWN onto the reco daughters --
    the entry point for asking why background candidates get reconstructed.
    Daughter 4 matched the stray pion, daughter 5 the stray photon."""
    signal = cut_utils.decay_mask(event, 221, [-13, 13, 22])
    stray = ~cut_utils.valid(event, 'mc_idx_mom') & ~signal
    assert cut_utils.follow(event, stray, 'prt_idx_gen').tolist() \
        == [[False, False, False, False, True, True]]


def test_which_generator_particles_were_reconstructed(event):
    """The other direction: reduce the reco daughters UP onto the generator
    collection. 'any' because one reco track is enough. The neutron (index 6)
    was never reconstructed."""
    ones = ak.ones_like(event['prt_idx_gen'], dtype=bool)
    out = cut_utils.reduce(event, ones, 'prt_idx_gen', how='any',
                           target_ref='mc_pid')
    assert out.tolist() == [[False, True, True, True, True, True, False]]


def test_how_many_reco_tracks_claimed_each_generator_particle(event):
    """>1 means candidates are sharing a track. Generator particle 1 (the mu+)
    was reconstructed twice, once in each candidate."""
    ones = ak.ones_like(event['prt_idx_gen'], dtype=bool)
    n = cut_utils.reduce(event, ones, 'prt_idx_gen', how='sum',
                         target_ref='mc_pid')
    assert n.tolist() == [[0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0]]


def test_which_candidates_are_built_from_stray_matches(event):
    """The full background question: gen -> reco -> candidate. Candidate 1 is
    contaminated; candidate 0 is clean."""
    signal = cut_utils.decay_mask(event, 221, [-13, 13, 22])
    stray = ~cut_utils.valid(event, 'mc_idx_mom') & ~signal
    matched_stray = cut_utils.follow(event, stray, 'prt_idx_gen')
    out = cut_utils.reduce(event, matched_stray, 'prt_idx_mom', how='any',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[False, True]]


# --- candidate-level selection -----------------------------------------------

def test_which_candidates_pass_a_pid_cut(event):
    """"Every muon daughter passes" -- an all-reduction, with non-muons folded
    in as automatic passes so the photons do not fail the candidate."""
    muon_ok = ((event['prt_pnn_mu'] > 0.95)
               | (abs(event['prt_pid']) != 13))
    out = cut_utils.reduce(event, muon_ok, 'prt_idx_mom', how='all',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[True, False]]


def test_which_candidates_have_exactly_one_photon(event):
    """A multiplicity requirement: sum the PID flag, compare to 1.

    On this ntuple it is a no-op -- the decay descriptor guarantees exactly one
    photon per candidate, and it does hold on real MC. The recipe exists as a
    SAFETY CHECK on that guarantee (fit_dalitz_new.ipynb runs it before
    computing Dalitz variables), and it is not a no-op for a channel whose
    reconstruction does not fix the composition.
    """
    n_gamma = cut_utils.reduce(event, event['prt_pid'] == 22, 'prt_idx_mom',
                               how='sum', target_ref='tag_dtf_m')
    assert n_gamma.tolist() == [[1.0, 1.0]]
    assert ak.all(n_gamma == 1)


def test_multiplicity_check_catches_a_malformed_candidate():
    """What the previous recipe would report if the guarantee were violated.
    Candidate 1 carries two photons, which is malformed for this channel --
    not background, just wrong."""
    bad = make_tree(
        tag_dtf_m=[[550.0, 549.0]],
        prt_idx_mom=[[0, 0, 0, 1, 1, 1, 1]],
        prt_pid=[[-13, 13, 22, -13, 13, 22, 22]],
    )
    n_gamma = cut_utils.reduce(bad, bad['prt_pid'] == 22, 'prt_idx_mom',
                               how='sum', target_ref='tag_dtf_m')
    assert (n_gamma == 1).tolist() == [[True, False]]


def test_worst_muon_pnn_per_candidate(event):
    """When you want the value rather than a pass/fail, use min. Push non-muons
    to +inf first or they win the minimum -- photons carry pnn_mu = 0."""
    is_muon = abs(event['prt_pid']) == 13
    out = cut_utils.reduce(event, ak.where(is_muon, event['prt_pnn_mu'], np.inf),
                           'prt_idx_mom', how='min', target_ref='tag_dtf_m')
    assert out.tolist() == [[0.97, 0.20]]


def test_propagate_a_candidate_selection_to_its_daughters(event):
    """follow() down. The result is daughter-shaped, so it indexes prt_* directly."""
    best = cut_utils.best_candidate_mask(event, 'tag_dtf_chi2',
                                         ak.Array([[True, True]]))
    dmask = cut_utils.follow(event, best, 'prt_idx_mom')
    assert best.tolist() == [[True, False]]          # lower chi2 wins
    assert dmask.tolist() == [[True, True, True, False, False, False]]


def test_propagate_a_daughter_flag_up_to_its_candidate(event):
    """reduce() up with 'any' -- one bad muon condemns the candidate."""
    bad_muon = (event['prt_pnn_mu'] < 0.5) & (abs(event['prt_pid']) == 13)
    out = cut_utils.reduce(event, bad_muon, 'prt_idx_mom', how='any',
                           target_ref='tag_dtf_m')
    assert out.tolist() == [[False, True]]


def test_full_chain_candidate_to_generator_mother(event):
    """The composite crossing: candidate -> daughter -> gen particle -> gen
    mother. There is no direct link, so it is three hops."""
    best = cut_utils.best_candidate_mask(event, 'tag_dtf_chi2',
                                         ak.Array([[True, True]]))
    dmask = cut_utils.follow(event, best, 'prt_idx_mom')
    gen = cut_utils.reduce(event, dmask, 'prt_idx_gen', how='any',
                           target_ref='mc_pid')
    mother = cut_utils.reduce(event, gen & cut_utils.valid(event, 'mc_idx_mom'),
                              'mc_idx_mom', how='any', target_ref='mc_pid')
    assert gen.tolist() == [[False, True, True, True, False, False, False]]
    assert mother.tolist() == [[True, False, False, False, False, False, False]]
    assert ak.flatten(event['mc_pid'][mother]).tolist() == [221.0]
