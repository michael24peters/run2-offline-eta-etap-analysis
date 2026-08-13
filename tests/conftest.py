"""
Shared pytest fixtures and path setup.

`src/utils` and `src/root` are not packages, so they go on sys.path the same way
every script in the repo does it.

Every fixture here is a hand-built toy tree -- no ROOT files, no network. The
trees are small enough to verify by eye, which is the point: a test that needs
a 600 MB ntuple to explain itself is not a unit test.

Ntuple conventions the fixtures reproduce:
  - every branch is float64, including the index branches
  - `mc_idx_mom` is -1 for a particle with no mother in the record
  - `prt_*` daughters are flat per event, wired to candidates by `prt_idx_mom`
  - `prt_idx_gen` links a reco daughter to the generator particle it matched
"""

import os
import sys

import awkward as ak
import numpy as np
import pytest

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'src', 'root'))


def make_tree(**fields):
    """Build a toy tree from plain nested lists.

    Everything is float64, to match the real ntuple -- casting index branches
    back to int is part of what follow() and reduce() do, and the tests should
    exercise that rather than side-step it.

    Built via ak.unflatten rather than ak.Array so that a branch which is empty
    in EVERY event still comes out typed `var * float64`. ak.Array would infer
    `var * unknown` there, which uproot never produces (a ROOT branch carries a
    declared type whether or not any entry is filled) and which would make these
    tests fail for a reason the real pipeline cannot hit.
    """
    built = {}
    for name, rows in fields.items():
        flat = np.array([float(v) for event in rows for v in event],
                        dtype=np.float64)
        counts = np.array([len(event) for event in rows], dtype=np.int64)
        built[name] = ak.unflatten(flat, counts)
    return ak.Array(built)


def kinematic_tree(particles):
    """Build an mc_* tree from (pid, mother_idx, px, py, pz) tuples per event.

    Momenta are chosen by the caller so a particle either clears or misses the
    fiducial cuts (eta in [2, 4.5], pT > 500, p > 3000 for muons).
    """
    return ak.Array({
        'mc_pid':     ak.Array([[float(p[0]) for p in ev] for ev in particles]),
        'mc_idx_mom': ak.Array([[float(p[1]) for p in ev] for ev in particles]),
        'mc_px':      ak.Array([[float(p[2]) for p in ev] for ev in particles]),
        'mc_py':      ak.Array([[float(p[3]) for p in ev] for ev in particles]),
        'mc_pz':      ak.Array([[float(p[4]) for p in ev] for ev in particles]),
    })


# (px, py, pz) triples calibrated against the fiducial cuts in fid_reqs.py
# (eta in [2, 4.5]; pT > 500; and p > 3000 for muons and electrons), so a test
# can say IN or FWD instead of three magic numbers. Verified values:
#
#            pT       p       eta    muon   photon
#   IN     1000.0  10049.9   3.00    pass    pass
#   FWD    1000.0  74206.7   5.00    fail    fail    <- fails on eta ALONE
#   LOW_PT  100.0   1005.0   3.00    fail    fail    <- fails on pT (and p)
#
# FWD is the one to reach for when a test needs a particle to fail for exactly
# one reason; LOW_PT trips more than one cut at once for muons.
IN = (600.0, 800.0, 10000.0)
FWD = (600.0, 800.0, 74200.0)
LOW_PT = (60.0, 80.0, 1000.0)


# --- reco-side fixtures ------------------------------------------------------

@pytest.fixture
def reco():
    """Two events of reco candidates and their daughters.

        event 0: 2 candidates, 4 daughters -- daughters 0,1 -> cand 0
                                                        2,3 -> cand 1
        event 1: 1 candidate,  2 daughters -- both -> cand 0
    """
    return make_tree(
        tag_dtf_m=[[10.0, 20.0], [30.0]],
        prt_idx_mom=[[0, 0, 1, 1], [0, 0]],
        prt_pid=[[-13, 13, -13, 13], [-13, 13]],
    )


@pytest.fixture
def shared_track():
    """One event, 2 candidates that SHARE a reco track.

    Daughters 1 and 2 both matched generator particle 2, which is what makes the
    daughter -> gen -> daughter round trip lossy. A real and common situation in
    multi-candidate events.

        prt_idx_mom = [0, 0, 1, 1]   two candidates
        prt_idx_gen = [1, 2, 2, 3]   daughters 1 and 2 share gen particle 2
    """
    return make_tree(
        tag_dtf_m=[[550.0, 549.0]],
        prt_idx_mom=[[0, 0, 1, 1]],
        prt_idx_gen=[[1, 2, 2, 3]],
        prt_pid=[[-13, 13, -13, 13]],
        mc_pid=[[221, -13, 13, 13]],
        mc_idx_mom=[[-1, 0, 0, -1]],
    )


# --- generator-side fixtures -------------------------------------------------

@pytest.fixture
def gen():
    """Two events of generator particles, eta -> mu+ mu-.

    Index 0 is the eta and has NO mother, so mc_idx_mom is -1 there. That
    sentinel is what most of the reduce() tests are about.
    """
    return make_tree(
        mc_pid=[[221, -13, 13], [221, -13, 13]],
        mc_idx_mom=[[-1, 0, 0], [-1, 0, 0]],
    )


@pytest.fixture
def gen_with_strays():
    """One event: a clean eta -> mu mu gamma decay plus MC-matched junk.

    Taken from real eta2mumugamma MC, where the generator record holds the
    signal decay AND every particle the reconstruction matched onto. The strays
    are motherless, so they are nobody's daughters.

        mc_pid     = [221, -13, 13, 22,  22, -211, 2112]
        mc_idx_mom = [ -1,   0,  0,  0,  -1,   -1,   -1]
                      |---- the decay ---|  |--- strays ---|
    """
    return make_tree(
        mc_pid=[[221, -13, 13, 22, 22, -211, 2112]],
        mc_idx_mom=[[-1, 0, 0, 0, -1, -1, -1]],
    )


@pytest.fixture
def two_gen_candidates():
    """One event with TWO complete eta -> mu mu gamma decays.

    Second eta is at index 4 and its daughters point at 4, not 0. Exercises the
    case where 'any' and 'all' over mothers disagree.
    """
    return make_tree(
        mc_pid=[[221, -13, 13, 22, 221, -13, 13, 22]],
        mc_idx_mom=[[-1, 0, 0, 0, -1, 4, 4, 4]],
    )
