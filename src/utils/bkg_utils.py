"""
bkg_utils.py

MC truth-matching against a signal decay (signal_daughter_mask,
signal_candidate_mask) -- imported by ana_bkg.py and driver_dcb_mc_prior.py.
"""

import cut_utils


def signal_daughter_mask(tree, mother_pid, daughter_pids):
    """
    Per-daughter MC truth-matching against a signal decay
    mother_pid -> daughter_pids.

    A daughter is correctly matched iff it has a generator match, that
    match's PID agrees with the daughter's own reco PID, AND that match's
    mother is the real signal mother (found via cut_utils.decay_mask) -- not
    merely "some generator particle at position i".

    Returns a dict:
        has_gen_match             : daughter has a valid prt_idx_gen
        pid_at_gen                : true PID of the matched gen particle
        mc_mom_at_gen             : mc_idx_mom of the matched gen particle
        pid_match                 : has_gen_match & (pid_at_gen == reco PID)
        is_child_of_signal_mother : has_gen_match & the matched gen
                                    particle's mother is the signal decay
        is_signal_dtr             : pid_match & is_child_of_signal_mother
    """
    signal_mother = cut_utils.decay_mask(tree, mother_pid, daughter_pids)

    has_gen_match = cut_utils.valid(tree, 'prt_idx_gen')
    pid_at_gen = cut_utils.follow(tree, tree['mc_pid'], 'prt_idx_gen')
    mc_mom_at_gen = cut_utils.follow(tree, tree['mc_idx_mom'], 'prt_idx_gen')
    pid_match = has_gen_match & (pid_at_gen == tree['prt_pid'])

    # Two-hop: daughter -> its gen particle (prt_idx_gen) -> that particle's
    # mother (mc_idx_mom) -> is that mother the real signal mother?
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
    Per-candidate mask: True where every daughter of the candidate matches
    the signal decay mother_pid -> daughter_pids (see signal_daughter_mask).

    target_ref : any branch whose per-event length defines the candidate
                 collection, passed through to cut_utils.reduce.
    """
    m = signal_daughter_mask(tree, mother_pid, daughter_pids)
    return cut_utils.reduce(tree, m['is_signal_dtr'], 'prt_idx_mom',
                            how='all', target_ref=target_ref)
