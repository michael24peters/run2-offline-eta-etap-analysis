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
    counting how often M falls inside m123_limits(): at fixed
    m23 the allowed m12 is an interval (m123_limits), so its overlap with
    the bin's m12 range is computed exactly and only m23 is gridded.
    """
    mmu2 = (M_MU / 1000.0)**2
    M2 = (np.asarray(M, float) / 1000.0)[:, None]**2
    s = np.linspace(*m23_range, n_m23)[None, :]
    with np.errstate(divide='ignore'):   # m23 = 0 at the grid's bottom edge
        beta = np.sqrt(np.clip(1.0 - 4.0 * mmu2 / s, 0.0, None))
    half = np.clip(M2 - s, 0.0, None) / 2.0
    lo, hi = mmu2 + half * (1.0 - beta), mmu2 + half * (1.0 + beta)
    a, b = m12_range
    overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    return overlap.mean(axis=1) / (b - a)


def kroll_wada_s_density(s, ff_lambda):
    """
    Unnormalized m^2(mu+ mu-) spectrum dGamma/ds of eta -> gamma gamma* ->
    gamma mu+ mu- (Kroll-Wada), s in GeV^2, with the pole form factor
    F(s) = 1 / (1 - s / ff_lambda^2), ff_lambda in GeV:

        dGamma/ds ~ |F(s)|^2 / s * (1 - s / m_eta^2)^3 * beta * (1 + 2 m_mu^2 / s)

    0 outside [4 m_mu^2, m_eta^2].
    """
    mmu2 = (M_MU / 1000.0)**2
    meta2 = (M_ETA / 1000.0)**2
    s = np.asarray(s, float)
    inside = (s > 4.0 * mmu2) & (s < meta2)
    s_in = np.where(inside, s, meta2 / 2.0)
    beta = np.sqrt(1.0 - 4.0 * mmu2 / s_in)
    ff2 = (1.0 - s_in / ff_lambda**2)**-2
    w = ff2 / s_in * (1.0 - s_in / meta2)**3 * beta * (1.0 + 2.0 * mmu2 / s_in)
    return np.where(inside, w, 0.0)


def kroll_wada_factor(M, m12_range, m23_range, ff_lambda, n_m23=400):
    """
    Fraction of eta -> gamma mu+ mu- signal at gamma mu+ mu- mass M (MeV,
    array) that falls in the Dalitz bin m12_range x m23_range (GeV^2, m12 =
    m^2(gamma mu+), m23 = m^2(mu+ mu-)). The signal counterpart of
    phase_space_factor.

    The signal density is the Kroll-Wada one (kroll_wada_s_density for s =
    m23, ff_lambda in GeV), with the mu+ angle theta* in the mu+ mu- rest
    frame distributed as

        1 + cos^2 theta* + (4 m_mu^2 / s) sin^2 theta*.

    A signal candidate at M != m_eta is a true eta decay with its photon
    energy scaled by (M^2 - s) / (m_eta^2 - s). That leaves s and theta*
    unchanged, and moves m12 along

        m12 = m_mu^2 + (M^2 - s) (1 - beta cos theta*) / 2

    (see m123_limits), so s stays below m_eta^2 at every M.

    Summed over bins that tile the Dalitz plane, the factor is 1 for
    M >= m_eta. The m12 overlap is integrated exactly, over cos theta*; m23
    is sampled at the midpoints of n_m23 strips.
    """
    mmu2 = (M_MU / 1000.0)**2
    meta2 = (M_ETA / 1000.0)**2

    s_norm = np.linspace(4.0 * mmu2, meta2, 200001)
    norm = np.trapezoid(kroll_wada_s_density(s_norm, ff_lambda), s_norm)

    M2 = (np.asarray(M, float) / 1000.0)[:, None]**2
    lo, hi = m23_range
    s = (lo + (np.arange(n_m23) + 0.5) * (hi - lo) / n_m23)[None, :]
    w = kroll_wada_s_density(s, ff_lambda) / norm

    # cos theta* at the bin's m12 edges, clipped to [-1, 1].
    valid = (s > 4.0 * mmu2) & (M2 > s)
    s_in = np.where(s > 4.0 * mmu2, s, 1.0)
    beta = np.sqrt(1.0 - 4.0 * mmu2 / s_in)
    span = np.where(valid, M2 - s, 1.0)
    a, b = m12_range
    c_hi = np.clip((1.0 - 2.0 * (a - mmu2) / span) / beta, -1.0, 1.0)
    c_lo = np.clip((1.0 - 2.0 * (b - mmu2) / span) / beta, -1.0, 1.0)

    # Angular weight between c_lo and c_hi, as a fraction of [-1, 1].
    A = 1.0 + 4.0 * mmu2 / s_in
    ang = A * (c_hi - c_lo) + beta**2 * (c_hi**3 - c_lo**3) / 3.0
    ang_frac = np.where(valid, ang / (8.0 / 3.0 * (1.0 + 2.0 * mmu2 / s_in)), 0.0)
    return (hi - lo) * (w * ang_frac).mean(axis=1)
