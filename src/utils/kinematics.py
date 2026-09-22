"""
kinematics.py

Four-vector arithmetic. Elementwise, so it works on plain numpy arrays and on
awkward arrays.
"""

import numpy as np

# PDG masses [MeV]. Every fit in this repo works in MeV; divide by 1000 for
# the GeV^2 Dalitz variables.
M_MU  = 105.6583755
M_ETA = 547.862


def pt(px, py):
    """Transverse momentum."""
    return np.sqrt(px**2 + py**2)


def p(px, py, pz):
    """Momentum magnitude."""
    return np.sqrt(px**2 + py**2 + pz**2)


def invariant_mass2(px1, py1, pz1, e1, px2, py2, pz2, e2):
    """Mass squared of the sum of two 4-vectors.

    Squared rather than the mass itself because reconstructed pairs can come out
    with M^2 < 0, and Dalitz variables are conventionally m^2 anyway. Take the
    square root yourself if you want a mass, after checking positivity.
    """
    e  = e1  + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    return e**2 - px**2 - py**2 - pz**2


def eta(px, py, pz):
    """
    Vectorized pseudorapidity, matching ROOT.TVector3::PseudoRapidity().
    
    Uses np.where instead of try/except to handle division by zero.
    Source: https://root.cern.ch/doc/master/classTVector3.html#aedc6fc6f5f6f3f3d4e4f4e4f4f4f4f4
    Code: https://root.cern.ch/doc/master/TVector3_8cxx_source.html#l00345
    """
    px, py, pz = np.asarray(px, float), np.asarray(py, float), np.asarray(pz, float)
    mom = p(px, py, pz)  # momentum
    cos_theta = np.where(mom != 0, pz / np.where(mom != 0, mom, 1.0), 1.0)  # avoid div by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        log_eta = -0.5 * np.log((1.0 - cos_theta) / (1.0 + cos_theta))
    edge_eta = np.where(pz == 0, 0.0, np.where(pz > 0, 1e10, -1e10))
    return np.where(cos_theta**2 < 1, log_eta, edge_eta)


def m123_limits(m12s, m23s):
    """
    Kinematic range [M_min, M_max] (MeV) of the gamma mu+ mu- invariant mass
    for a given Dalitz point, m12s = m^2(gamma mu+), m23s = m^2(mu+ mu-) in
    GeV^2.

    In the mu+ mu- rest frame, with theta* the mu+ angle and
    beta = sqrt(1 - 4 m_mu^2 / m23s):

        m12s = m_mu^2 + (M^2 - m23s) (1 - beta cos theta*) / 2

    so cos theta* in [-1, 1] bounds M^2 between
    m23s + 2 (m12s - m_mu^2) / (1 +/- beta). This is an identity on the
    four-vectors, so it is exact only for a mass built from the same
    daughter momenta as m12s/m23s (not a DTF-refitted mass).
    """
    mmu2 = (M_MU / 1000.0)**2
    m12s, m23s = np.asarray(m12s, float), np.asarray(m23s, float)
    beta = np.sqrt(np.clip(1.0 - 4.0 * mmu2 / m23s, 0.0, None))
    x = m12s - mmu2
    with np.errstate(divide='ignore', invalid='ignore'):
        lo = np.sqrt(m23s + 2.0 * x / (1.0 + beta))
        hi = np.sqrt(m23s + 2.0 * x / (1.0 - beta))
    return lo * 1000.0, hi * 1000.0


def phase_space_factor(M, m12_range, m23_range, n_m23=400):
    """
    Fraction of the Dalitz bin m12_range x m23_range (GeV^2, m12 =
    m^2(gamma mu+), m23 = m^2(mu+ mu-)) that is kinematically reachable by a
    gamma mu+ mu- system of invariant mass M (MeV, array).

    1 where the whole bin is inside the Dalitz region of mass M, 0 where none
    of it is. Equivalent to sampling (m12, m23) uniformly in the bin and
    counting how often M falls inside m123_limits(), but noise-free: at fixed
    m23 the allowed m12 is an interval (see m123_limits), so its overlap with
    the bin's m12 range is computed exactly and only m23 is gridded.
    """
    mmu2 = (M_MU / 1000.0)**2
    M2 = (np.asarray(M, float) / 1000.0)[:, None]**2
    s = np.linspace(*m23_range, n_m23)[None, :]
    beta = np.sqrt(np.clip(1.0 - 4.0 * mmu2 / s, 0.0, None))
    half = np.clip(M2 - s, 0.0, None) / 2.0
    lo, hi = mmu2 + half * (1.0 - beta), mmu2 + half * (1.0 + beta)
    a, b = m12_range
    overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    return overlap.mean(axis=1) / (b - a)
