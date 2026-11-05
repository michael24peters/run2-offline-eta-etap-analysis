"""
kinematics.py

Four-vector arithmetic. Elementwise, so it works on plain numpy arrays and on awkward arrays.
"""

import numpy as np

# PDG masses [MeV]. Every fit in this repo works in MeV; divide by 1000 for the GeV^2 Dalitz
# variables.
M_MU  = 105.6583755
M_ETA = 547.862

# Pole parameter [GeV] of the eta transition form factor F(s) = 1 / (1 - s / FF_LAMBDA^2), from NA60
# 2016 (arXiv:1608.07898): Lambda^-2 = 1.934 GeV^-2.
FF_LAMBDA = 1.934**-0.5


def pt(px, py):
    """Transverse momentum."""
    return np.sqrt(px**2 + py**2)


def p(px, py, pz):
    """Momentum magnitude."""
    return np.sqrt(px**2 + py**2 + pz**2)


def invariant_mass2(px1, py1, pz1, e1, px2, py2, pz2, e2):
    """Mass squared of the sum of two 4-vectors.

    Squared rather than the mass itself because reconstructed pairs can come out with M^2 < 0, and
    Dalitz variables are conventionally m^2 anyway. Take the square root yourself if you want a
    mass, after checking positivity.
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
    Kinematic range [M_min, M_max] (MeV) of the gamma mu+ mu- invariant mass for a given Dalitz
    point, m12s = m^2(gamma mu+), m23s = m^2(mu+ mu-) in GeV^2.

    In the mu+ mu- rest frame, with theta* the mu+ angle and beta = sqrt(1 - 4 m_mu^2 / m23s):

        m12s = m_mu^2 + (M^2 - m23s) (1 - beta cos theta*) / 2

    so cos theta* in [-1, 1] bounds M^2 between m23s + 2 (m12s - m_mu^2) / (1 +/- beta).
    """
    m_mu2 = (M_MU / 1000.0)**2                  # m_mu^2 [GeV^2]
    m12s, m23s = np.asarray(m12s, float), np.asarray(m23s, float)
    # Muon speed in the mu+ mu- rest frame.
    beta = np.sqrt(np.clip(1.0 - 4.0 * m_mu2 / m23s, 0.0, None))
    m12_excess = m12s - m_mu2                   # m12 above its minimum, m_mu^2
    with np.errstate(divide='ignore', invalid='ignore'):
        m_min = np.sqrt(m23s + 2.0 * m12_excess / (1.0 + beta))
        m_max = np.sqrt(m23s + 2.0 * m12_excess / (1.0 - beta))
    return m_min * 1000.0, m_max * 1000.0


def m12_limits(M, m23s):
    """
    Kinematic range [m12_min, m12_max] (GeV^2) of m12 = m^2(gamma mu+) for a gamma mu+ mu- system of
    invariant mass M (MeV) and m23s = m^2(mu+ mu-) (GeV^2). The inverse of m123_limits.

    In the mu+ mu- rest frame (the frame where the pair's total momentum is zero) the muons fly
    apart back to back, each with energy E_mu = sqrt(m23s) / 2 and speed beta =
    sqrt(1 - 4 m_mu^2 / m23s), and the photon has energy E_gamma = (M^2 - m23s) / (2 sqrt(m23s)).
    With theta* the angle between the mu+ and the photon,

        m12 = (p_gamma + p_mu+)^2 = m_mu^2 + 2 E_gamma E_mu (1 - beta cos theta*)
            = m_mu^2 + (M^2 - m23s) / 2 * (1 - beta cos theta*)

    (the sqrt(m23s) in E_gamma cancels the one in E_mu). cos theta* runs over [-1, 1], so m12 runs
    over [m12_min, m12_max]. The range has zero width where m23s >= M^2 (photon energy 0) and where
    m23s <= 4 m_mu^2 (beta = 0).
    """
    m_mu2 = (M_MU / 1000.0)**2
    M2 = (np.asarray(M, float) / 1000.0)**2
    m23s = np.asarray(m23s, float)
    with np.errstate(divide='ignore'):   # m23s = 0 gives beta = 0
        beta = np.sqrt(np.clip(1.0 - 4.0 * m_mu2 / m23s, 0.0, None))
    # cos theta* = +1: mu+ along the photon
    m12_min = m_mu2 + np.clip(M2 - m23s, 0.0, None) / 2.0 * (1.0 - beta)
    # cos theta* = -1: mu+ against the photon
    m12_max = m_mu2 + np.clip(M2 - m23s, 0.0, None) / 2.0 * (1.0 + beta)
    return m12_min, m12_max


def phase_space_factor(M, m12_range, m23_range, n_m23=400):
    """
    Fraction of the Dalitz bin m12_range x m23_range (GeV^2, m12 = m^2(gamma mu+), m23 =
    m^2(mu+ mu-)) that is kinematically reachable by a gamma mu+ mu- system of invariant mass M.

    Every point of the bin counts equally (flat phase space); dalitz_factor is the same fraction
    with points weighted by the eta Dalitz-decay density. 1 where the whole bin is inside the Dalitz
    region of mass M, 0 where none of it is.

    Arguments:
    ----------
    M: array of trial masses (MeV) to evaluate the factor at. m12_range: (m12_min, m12_max) of the
    bin (GeV^2, m12 = m^2(gamma mu+)). m23_range: (m23_min, m23_max) of the bin (GeV^2, m23 =
    m^2(mu+ mu-)). n_m23: number of equal q^2 = m23 strips across the bin; the factor is
        sampled at their midpoints.

    Returns:
    --------
    Array of the same length as M, with the fraction of the bin reachable at each mass. 1 where the
    whole bin is inside the Dalitz region of mass M, 0 where none of it is.
    """
    M = np.asarray(M, float)[:, None]
    # Midpoints of n_m23 equal strips across the bin's q2 range, one column per strip.
    q2 = (m23_range[0] + (np.arange(n_m23) + 0.5) * (m23_range[1] - m23_range[0]) / n_m23)[None, :]

    # Reachable m12 at each (M, q2).
    lo12, hi12 = m12_limits(M, q2)

    # TODO: the ntuple keeps only photons with pT > 500 MeV (~/ana/src/ana.py), which empties the
    # part of D(M) where the photon is soft in the mu+ mu- frame, E*_gamma = (M^2 - q2) / (2
    # sqrt(q2)) -> 0, i.e. near a bin's lowest reachable masses; this factor counts that part as
    # populated. The loss depends on the photon's lab boost, not only on (M, m12, m23): a turn-on
    # from data.
    #
    # At each (M, q2): the length of [lo12, hi12] inside the bin's m12 range, 0 where they do not
    # overlap. Averaged over the q2 values and divided by the bin's m12 width, this is (reachable
    # area) / (bin area).
    overlap12 = np.clip(np.minimum(hi12, m12_range[1]) - np.maximum(lo12, m12_range[0]), 0.0, None)
    return overlap12.mean(axis=1) / (m12_range[1] - m12_range[0])


def dalitz_factor(M, m12_range, m23_range, n_m23=400):
    """
    Dalitz-decay counterpart of phase_space_factor: the fraction of the Dalitz bin m12_range x
    m23_range that is kinematically reachable by a gamma mu+ mu- system of invariant mass M (array
    of trial masses, as in phase_space_factor), with each point of the bin weighted by the eta
    Dalitz-decay density instead of counting equally:

        f(M) = sum over q^2 strips of int_{reachable m12 in the bin} density dm12
               / sum over q^2 strips of int_{bin's m12 range} density dm12

    i.e. int_{bin and D(M)} density dA / int_{bin} density dA. phase_space_factor is the same ratio
    with density = 1. 1 where the whole bin is inside the Dalitz region of mass M, 0 where none of
    it is. At fixed q^2 the density is quadratic in m12, so the m12 integral is exact; q^2 is
    sampled at the midpoints of n_m23 strips.

    The density of eta -> gamma gamma* -> gamma mu+ mu- decays over the Dalitz plot at the eta mass,
    |M|^2 for m12 = m^2(gamma mu+) and q^2 = m23 = m^2(mu+ mu-) in GeV^2 (the Kroll-Wada matrix
    element with the pole form factor F(q^2) = 1 / (1 - q^2 / FF_LAMBDA^2)):

        |F(q^2)|^2 / q^2 [(m_eta^2 - q^2)^2 (1 + 4 m_mu^2 / q^2) + (m13 - m12)^2]

    with m13 = m_eta^2 + 2 m_mu^2 - q^2 - m12. 0 for q^2 <= 4 m_mu^2. Valid for q^2 < FF_LAMBDA^2.

    True, gen-level eta decays fill D(m_eta) with this density. The formula is defined at any (m12,
    q^2) above threshold, including outside D(m_eta), where it is the same expression continued.
    """
    m_mu2 = (M_MU / 1000.0)**2
    m_eta2 = (M_ETA / 1000.0)**2
    M = np.asarray(M, float)[:, None]
    bin_lo23, bin_hi23 = m23_range  # the bin's m23 edges
    bin_lo12, bin_hi12 = m12_range  # the bin's m12 edges
    # Midpoints of n_m23 equal strips across the bin's q2 range, one column per strip.
    q2 = (bin_lo23 + (np.arange(n_m23) + 0.5) * (bin_hi23 - bin_lo23) / n_m23)[None, :]

    # Reachable m12 at each (M, q2), clipped to the bin's m12 range:
    # [reach_lo12, reach_hi12] is the part of the bin inside D(M) (zero length where none is).
    lo12, hi12 = m12_limits(M, q2)
    reach_lo12 = np.clip(lo12, bin_lo12, bin_hi12)
    reach_hi12 = np.clip(hi12, bin_lo12, bin_hi12)

    # Make sure values are phsyical before evaluating the density. The density is 0 for q^2 <= 4
    # m_mu^2
    physical = q2 > 4.0 * m_mu2
    q2_safe = np.where(physical, q2, 1.0) # stand-in 1 below threshold, where q2_weight is 0
    q2_weight = np.where(physical, (1.0 - q2_safe / FF_LAMBDA**2)**-2 / q2_safe, 0.0)   # |F|^2 / q2

    base_term = (m_eta2 - q2_safe)**2 * (1.0 + 4.0 * m_mu2 / q2_safe)   # part not depending on m12
    mid12 = (m_eta2 + 2.0 * m_mu2 - q2_safe) / 2.0   # m12 where m13 = m12 (cos theta* = 0)

    # m12 integral of the density over the reachable part and over the whole bin, from its
    # antiderivative q2_weight * (base_term m12 + (4/3) (m12 - mid12)^3).
    reach_integral = q2_weight * (base_term * (reach_hi12 - reach_lo12)
                                  + 4.0 / 3.0 * ((reach_hi12 - mid12)**3 - (reach_lo12 - mid12)**3))
    bin_integral = q2_weight * (base_term * (bin_hi12 - bin_lo12)
                                + 4.0 / 3.0 * ((bin_hi12 - mid12)**3 - (bin_lo12 - mid12)**3))

    # TODO: the photon pT > 500 MeV loss noted in phase_space_factor applies here too. Sum over the
    # q2 strips (the strip width cancels in the ratio). A bin entirely below threshold has bin_total
    # 0 and factor 0.
    bin_total = bin_integral.sum(axis=1)
    return np.where(bin_total > 0,
                    reach_integral.sum(axis=1) / np.where(bin_total > 0, bin_total, 1.0), 0.0)
