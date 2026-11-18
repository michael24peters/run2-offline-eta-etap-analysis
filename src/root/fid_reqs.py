"""
fid_reqs.py

LHCb fiducial requirements on generator-level particles, as masks:

    fiducial_mask(tree)                         per generator particle
    fiducial_masks(tree, mother, daughters)     per generator particle, by decay and failure
    fiducial_candidate_mask(tree, mother, daughters, target_ref)   per reco candidate
"""

import os
import sys

import awkward as ak
import numpy as np

# cut_utils and kinematics, from src/utils
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils
import kinematics

# LHCb fiducial requirements: 2 < eta < 4.5 for every particle, and per |pid| the pT and p minima
# [MeV] (p_min None: no p requirement). A PID not listed fails.
FIDUCIAL_ETA_RANGE = (2.0, 4.5)
FIDUCIAL_REQS = {
    13: (500.0, 3000.0),   # muons
    11: (500.0, 3000.0),   # electrons
    22: (500.0, None),     # photons
}


def fiducial_mask(tree):
    """Per generator particle mask: True for the particles that pass the fiducial requirements."""
    pid = ak.to_numpy(ak.flatten(tree['mc_pid'])).astype(int)
    px = ak.to_numpy(ak.flatten(tree['mc_px']))
    py = ak.to_numpy(ak.flatten(tree['mc_py']))
    pz = ak.to_numpy(ak.flatten(tree['mc_pz']))
    p, pt = kinematics.p(px, py, pz), kinematics.pt(px, py)
    eta = kinematics.eta(px, py, pz)

    # [False, False, ...]
    ok = np.zeros(len(pid), dtype=bool)
    in_eta = (FIDUCIAL_ETA_RANGE[0] < eta) & (eta < FIDUCIAL_ETA_RANGE[1])
    # Apply fiducial requirements to each particle type (PID).
    for abs_pid, (pt_min, p_min) in FIDUCIAL_REQS.items():
        # True for all particles of this PID [13, -13, 22, ...] -> [True, True, False, ...] for
        # abs_pid=13
        mask = np.abs(pid) == abs_pid
        # For each particle of this PID, check if it passes the fiducial reqs.
        passes = in_eta & (pt > pt_min)
        if p_min is not None:
            passes = passes & (p > p_min)
        ok[mask] = passes[mask]
    # Back to per-event structure; p > 0 also fails particles with no momentum
    return ak.unflatten(ok & (p > 0), ak.num(tree['mc_pid']))


def fiducial_masks(tree, mother_pid, daughter_pids):
    """
    Per generator particle masks of the signal decay mother_pid -> daughter_pids, each shaped like
    mc_pid; every mask but 'stray' is True at signal mothers only. They compose with & and |.

    Keys:
    'signal':          the mother of a signal decay (cut_utils.decay_mask)
    'in_acceptance':   ...whose daughters all pass the fiducial requirements
    'failed_muon':     ...with at least one muon daughter failing them
    'failed_gamma':    ...with at least one photon daughter failing them
    'failed_electron': ...with at least one electron daughter failing them
    'stray':           motherless generator particles other than the signal mother
    """
    # Get the per-particle fiducial mask for the whole tree, i.e., all generator level particles in
    # the ntuple.
    ok = fiducial_mask(tree)
    # The signal mothers: True at each mother_pid particle whose daughters are exactly daughter_pids.
    signal = cut_utils.decay_mask(tree, mother_pid, daughter_pids)

    # Per generator particle: every one of its daughters passes the fiducial requirements.
    daughters_ok = cut_utils.reduce(tree, ok, 'mc_idx_mom', how='all',
                                    target_ref='mc_pid')

    def fail_mask(pid):
        """Per generator particle: a daughter selected by the mask `pid` fails the fiducial
        requirements."""
        return cut_utils.reduce(tree, ~ok & pid, 'mc_idx_mom', how='any',
                                target_ref='mc_pid')

    pid = tree['mc_pid']
    return {
        'signal':          signal,
        'in_acceptance':   signal & daughters_ok,
        'failed_muon':     signal & fail_mask(abs(pid) == 13),
        'failed_gamma':    signal & fail_mask(pid == 22),
        'failed_electron': signal & fail_mask(abs(pid) == 11),
        'stray':           ~cut_utils.valid(tree, 'mc_idx_mom') & ~signal,
    }


def fiducial_candidate_mask(tree, mother_pid, daughter_pids, target_ref):
    """Per reco candidate mask: True where every daughter's generator match is a daughter of a
    signal decay mother_pid -> daughter_pids that is in acceptance (fiducial_masks 'in_acceptance').
    A daughter with no generator match fails.

    target_ref : any branch whose per-event length defines the candidate collection, passed through
                 to cut_utils.reduce.
    """
    in_acceptance = fiducial_masks(tree, mother_pid, daughter_pids)['in_acceptance']
    # Generator particle -> its mother is an in-acceptance signal decay.
    child_of_acc = (cut_utils.valid(tree, 'mc_idx_mom')
                    & cut_utils.follow(tree, in_acceptance, 'mc_idx_mom'))
    # Reco daughter -> its generator match is such a particle.
    dtr_of_acc = (cut_utils.valid(tree, 'prt_idx_gen')
                  & cut_utils.follow(tree, child_of_acc, 'prt_idx_gen'))
    return cut_utils.reduce(tree, dtr_of_acc, 'prt_idx_mom', how='all',
                            target_ref=target_ref)
