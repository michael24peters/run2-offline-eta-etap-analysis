"""
bkg_utils.py

MC matching against a signal decay (signal_daughter_mask, signal_candidate_mask) and the background
source of each candidate (candidate_categories). For MC, MC matching is always the first cut.
"""

import awkward as ak
import numpy as np

import cut_utils


def signal_daughter_mask(tree, mother_pid, daughter_pids):
    """
    Per-daughter MC matching against a signal decay mother_pid -> daughter_pids.

    A daughter is correctly matched iff it has a generator match, that match's PID agrees with the
    daughter's own reco PID, AND that match's mother is the real signal mother (found via
    cut_utils.decay_mask) -- not merely "some generator particle at position i".

    Returns a dict:
        has_gen_match             : daughter has a valid prt_idx_gen
        pid_at_gen                : true PID of the matched gen particle
        mc_mom_at_gen             : mc_idx_mom of the matched gen particle
        pid_match                 : has_gen_match & (pid_at_gen == reco PID)
        is_child_of_signal_mother : has_gen_match & the matched gen
                                    particle's mother is the signal decay
        is_signal_dtr             : pid_match & is_child_of_signal_mother
    """
    # First, find the signal mothers in the event
    signal_mother = cut_utils.decay_mask(tree, mother_pid, daughter_pids)

    # For each reco daughter, check if it has a gen match
    has_gen_match = cut_utils.valid(tree, 'prt_idx_gen')
    # For each reco daughter, follow its gen match to find the gen PID
    pid_at_gen = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    mc_mom_at_gen = cut_utils.follow(tree, tree['mc_idx_mom'], 'prt_idx_gen')
    # For each reco daughter, check if its gen match's PID agrees with the reco PID
    pid_match = has_gen_match & (pid_at_gen == tree['prt_pid'])

    # Two-hop: daughter -> its gen particle (prt_idx_gen) -> that particle's mother (mc_idx_mom) ->
    # is that mother the real signal mother?
    gen_is_child_of_signal = (cut_utils.valid(tree, 'mc_idx_mom')
                              & cut_utils.follow(tree, signal_mother, 'mc_idx_mom'))
    is_child_of_signal_mother = has_gen_match & cut_utils.follow(
        tree, gen_is_child_of_signal, 'prt_idx_gen')

    is_signal_dtr = pid_match & is_child_of_signal_mother

    return dict(
        has_gen_match=has_gen_match, pid_at_gen=pid_at_gen,
        mc_mom_at_gen=mc_mom_at_gen, pid_match=pid_match,
        is_child_of_signal_mother=is_child_of_signal_mother,
        is_signal_dtr=is_signal_dtr,
    )


def signal_candidate_mask(tree, mother_pid, daughter_pids, target_ref):
    """
    Per-candidate mask: True where every daughter of the candidate matches the signal decay
    mother_pid -> daughter_pids (see signal_daughter_mask).

    target_ref : any branch whose per-event length defines the candidate collection, passed through
                 to cut_utils.reduce.
    """
    m = signal_daughter_mask(tree, mother_pid, daughter_pids)
    return cut_utils.reduce(tree, m['is_signal_dtr'], 'prt_idx_mom',
                            how='all', target_ref=target_ref)


def candidate_categories(tree, mother_pid, daughter_pids, target_ref):
    """
    Per-candidate masks of the background source of each candidate against the signal decay
    mother_pid -> daughter_pids (see signal_daughter_mask). A daughter that is not a signal
    daughter has one reason:

        unmatched        no generator match
        misid            the generator match has a different PID
        not_from_signal  the right PID, but not a daughter of the signal mother

    with misid and not_from_signal split by the daughter's reco species, muon (|pid| 13) or photon
    (pid 22).

    target_ref : any branch whose per-event length defines the candidate collection, passed through
                 to cut_utils.reduce.

    Returns {category: per-candidate mask}, in this order:
        mc_matched              every daughter is a signal daughter (signal_candidate_mask)
        unmatched, muon_misid, photon_misid, muon_not_from_signal, photon_not_from_signal
                                the one reason found among the candidate's daughters
        several                 two or more different reasons
    For candidates whose daughters are muons and photons, every candidate is in exactly one.
    """
    m = signal_daughter_mask(tree, mother_pid, daughter_pids)
    is_muon = abs(tree['prt_pid']) == 13
    is_photon = tree['prt_pid'] == 22
    misid = m['has_gen_match'] & ~m['pid_match']
    not_from_signal = m['pid_match'] & ~m['is_child_of_signal_mother']
    daughter_reasons = {
        'unmatched':              ~m['has_gen_match'],
        'muon_misid':             misid & is_muon,
        'photon_misid':           misid & is_photon,
        'muon_not_from_signal':   not_from_signal & is_muon,
        'photon_not_from_signal': not_from_signal & is_photon,
    }

    # Each reason onto the candidates (True if any daughter has it), and the number of different
    # reasons per candidate
    reasons = {}
    n_reasons = ak.zeros_like(tree[target_ref], dtype=np.int64)
    for name, daughter_has_reason in daughter_reasons.items():
        reasons[name] = cut_utils.reduce(tree, daughter_has_reason, 'prt_idx_mom', how='any',
                                         target_ref=target_ref)
        n_reasons = n_reasons + reasons[name]

    categories = {'mc_matched': cut_utils.reduce(tree, m['is_signal_dtr'], 'prt_idx_mom',
                                                 how='all', target_ref=target_ref)}
    for name, has_reason in reasons.items():
        categories[name] = has_reason & (n_reasons == 1)
    categories['several'] = n_reasons > 1
    return categories
