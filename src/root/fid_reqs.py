"""
fid_reqs.py

Script to apply fiducial requirements to generator-level particles in a ROOT
file. Uses uproot + awkward for I/O (no PyROOT/ROOT dependency).

Usage:
    python fid_reqs.py input.root mother_pid daughter_pid1 daughter_pid2 ...

Examples:
    python fid_reqs.py input.root 221 13 -13  |  (eta -> mu+ mu-)
    python fid_reqs.py input.root 221 13 -13 22  |  (eta -> mu+ mu- gamma)
"""

import argparse
import os
import sys

import awkward as ak
import numpy as np
import uproot

# Ensure utils/fit_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import kinematics
import cut_utils

# LHCb fiducial requirements.
FIDUCIAL_REQS = {
    13: lambda pt, p, eta: (2.0 < eta) & (eta < 4.5) & (pt > 500) & (p > 3000),
    11: lambda pt, p, eta: (2.0 < eta) & (eta < 4.5) & (pt > 500) & (p > 3000),
    22: lambda pt, p, eta: (2.0 < eta) & (eta < 4.5) & (pt > 500),
}


def fiducial_mask(tree):
    '''Per-generator particle mask: True for all particles that pass the LHCb
    fiducial requirements.
    '''
    pid = ak.to_numpy(ak.flatten(tree['mc_pid'])).astype(int)
    px = ak.to_numpy(ak.flatten(tree['mc_px']))
    py = ak.to_numpy(ak.flatten(tree['mc_py']))
    pz = ak.to_numpy(ak.flatten(tree['mc_pz']))
    p, pt = kinematics.p(px, py, pz), kinematics.pt(px, py)
    eta = kinematics.eta(px, py, pz)

    # [False, False, ...]
    ok = np.zeros(len(pid), dtype=bool)
    # Apply fiducial requirements to each particle type (PID).
    for ptype, reqs in FIDUCIAL_REQS.items():
        # True for all particles of this PID 
        # [13, -13, 22, ...] -> [True, True, False, ...] for ptype=13
        mask = np.abs(pid) == ptype
        # For each particle of this PID, check if it passes the fiducial reqs.
        if mask.any(): ok[mask] = reqs(pt[mask], p[mask], eta[mask])
    # Unflatten back to per-event structure, also require momentum > 0 to avoid
    # divison by zero and unphysical momenta, just in case.
    return ak.unflatten(ok & (p > 0), ak.num(tree['mc_pid']))


def fiducial_masks(tree, mother_pid, daughter_pids):
    '''Returns a dict of per-event boolean masks for the mother and daughter
    particles of a given decay. True for all particles that pass the LHCb
    fiducial requirements. Every mask is the same shape and can be composed
    with logical operators to select events with all daughters passing.

    Keys:
    'signal':          Confirmed mother and daughters of the specified decay
    'in_acceptance':   ...and every daughter passes fiducial requirements
    'failed_muon':     ...except for at least one muon daughter that fails
    'failed_gamma':    ...except for at least one photon daughter that fails
    'failed_electron': ...except for at least one electron daughter that fails
    'stray':           Generator particles with no mother that are not the signal
                       mother/candidate; i.e., MC-matched background particles
    '''
    # Get the per-particle fiducial mask for the whole tree, i.e., all generator
    # level particles in the ntuple.
    ok = fiducial_mask(tree)
    # Select only the mother and daughter particles of the given decay, i.e.,
    # ignore background, strays, etc. and isolate to only pure signal. Formatted
    # as a per-particle mask. True for all mother and daughter particles of
    # the signal, False for all others.
    signal = cut_utils.decay_mask(tree, mother_pid, daughter_pids)

    # Now check whether ALL daughters are in acceptance (not just signal).
    daughters_ok = cut_utils.reduce(tree, ok, 'mc_idx_mom', how='all',
                                    target_ref='mc_pid')

    def fail_mask(pid):
        '''
        Returns every mother that has a daughter of the specified PID failing
        the fiducial requirements.
        '''
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


# -------------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('mother_pid', type=int, help='Mother PID for the decay')
    parser.add_argument('daughter_pids', nargs='+', type=int,
                        help='Daughter PIDs for the decay, space separated')
    args = parser.parse_args()

    # Define output path based on input path, file name always the same
    outfile = os.path.join(os.path.dirname(args.infile), 'fiducial_requirements.root')

    with uproot.open(args.infile) as fin:
        events = fin['tree'].arrays(library='ak')

    masks = fiducial_masks(events, args.mother_pid, args.daughter_pids)

    # One boolean per event. `ak.any` = "the event has at least one fully fiducial
    # mother"; swap for ak.all over `signal` if every mother must pass instead.
    mask = ak.to_numpy(ak.any(masks['in_acceptance'], axis=1))

    for name, m in masks.items():
        print(f'{name:14s}: {int(ak.sum(m)):9d} particles '
              f'{int(ak.sum(ak.any(m, axis=1))):9d} events')
    print(f'Kept {int(mask.sum())} / {len(mask)} events '
          f'({100.0 * mask.sum() / len(mask):.1f}%)')

    kept = events[mask]
    with uproot.recreate(outfile) as fout:
        fout['tree'] = kept

    print(f'Wrote {len(kept)} events to {outfile}')
