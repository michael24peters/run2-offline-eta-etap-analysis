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

# Pole parameter [GeV] of the eta transition form factor F(s) = 1 / (1 - s /
# FF_LAMBDA^2), from NA60 2016 (arXiv:1608.07898): Lambda^-2 = 1.934 GeV^-2.
FF_LAMBDA = 1.934**-0.5


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
    m23s + 2 (m12s - m_mu^2) / (1 +/- beta).
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


def phase_space_factor(M, m12_range, m23_range, n_m23=400):
    """
    Fraction of the Dalitz bin m12_range x m23_range (GeV^2, m12 =
    m^2(gamma mu+), m23 = m^2(mu+ mu-)) that is kinematically reachable by a
    gamma mu+ mu- system of invariant mass M.

    Every point of the bin counts equally (flat phase space); dalitz_factor
    is the same fraction with points weighted by the eta Dalitz-decay
    density. 1 where the whole bin is inside the Dalitz region of mass M, 0
    where none of it is.

    Arguments:
    ----------
    M: array of trial masses (MeV) to evaluate the factor at.
    m12_range: (m12_min, m12_max) of the bin (GeV^2, m12 = m^2(gamma mu+)).
    m23_range: (m23_min, m23_max) of the bin (GeV^2, m23 = m^2(mu+ mu-)).
    n_m23: number of q^2 = m23 points to sample across the bin.

    Returns:
    --------
    Array of the same length as M, with the fraction of the bin reachable at
    each mass.
    1 where the whole bin is inside the Dalitz region of mass M, 0
    where none of it is.
    """
    m_mu2 = (M_MU / 1000.0)**2
    M2 = (np.asarray(M, float) / 1000.0)[:, None]**2
    q2 = np.linspace(*m23_range, n_m23)[None, :]  # n_m23 values of m23 = q^2 spanning the bin, one column each

    # Reachable m12 at each (M, q2). In the mu+ mu- rest frame (the frame where
    # the pair's total momentum is zero: the muons fly apart back to back, each
    # with speed beta), energy-momentum conservation gives the photon energy
    # (M^2 - q2) / (2 sqrt(q2)) and
    #     m12 = m_mu^2 + (M^2 - q2) (1 - beta cos theta*) / 2,
    # theta* = angle between the mu+ and the photon. cos theta* runs over
    # [-1, 1], so m12 runs over [lo12, hi12].
    with np.errstate(divide='ignore'):   # q2 = 0 at the grid's bottom edge
        beta = np.sqrt(np.clip(1.0 - 4.0 * m_mu2 / q2, 0.0, None))  # muon speed; 0 at or below threshold
    half_span = np.clip(M2 - q2, 0.0, None) / 2.0  # (M^2 - q2) / 2; 0 where q2 >= M^2 (photon energy 0)
    lo12 = m_mu2 + half_span * (1.0 - beta)  # cos theta* = +1: mu+ along the photon
    hi12 = m_mu2 + half_span * (1.0 + beta)  # cos theta* = -1: mu+ against the photon

    # At each (M, q2): the length of [lo12, hi12] inside the bin's m12 range,
    # 0 where they do not overlap. Averaged over the q2 values and divided by
    # the bin's m12 width, this is (reachable area) / (bin area).
    overlap12 = np.clip(np.minimum(hi12, m12_range[1]) - np.maximum(lo12, m12_range[0]), 0.0, None)
    return overlap12.mean(axis=1) / (m12_range[1] - m12_range[0])


def dalitz_density(m12s, m23s):
    """
    Unnormalized density of eta Dalitz decays, eta -> gamma gamma* ->
    gamma mu+ mu-, over the Dalitz plot at the eta mass: |M|^2 for m12s =
    m^2(gamma mu+) and m23s = q^2 = m^2(mu+ mu-) in GeV^2 (the Kroll-Wada
    matrix element with the pole form factor F(q^2) = 1 / (1 - q^2 /
    FF_LAMBDA^2)):

        |F(q^2)|^2 / q^2 [(m_eta^2 - q^2)^2 (1 + 4 m_mu^2 / q^2) + (m13s - m12s)^2]

    with m13s = m_eta^2 + 2 m_mu^2 - q^2 - m12s.
    0 for q^2 <= 4 m_mu^2. Valid for q^2 < FF_LAMBDA^2.

    True, gen-level eta decays fill D(m_eta) with this density. The formula is
    defined at any (m12s, m23s) above threshold, including outside D(m_eta),
    where it is the same expression continued.
    """
    m_mu2 = (M_MU / 1000.0)**2
    m_eta2 = (M_ETA / 1000.0)**2
    m12s, q2 = np.asarray(m12s, float), np.asarray(m23s, float)

    # Below the dimuon threshold (q2 <= 4 m_mu^2) no muon pair exists and the
    # density is 0; q2_safe stands in a 1 there so the divisions stay finite.
    physical = q2 > 4.0 * m_mu2
    q2_safe = np.where(physical, q2, 1.0)

    form_factor2 = (1.0 - q2_safe / FF_LAMBDA**2)**-2  # |F(q^2)|^2
    m13s = m_eta2 + 2.0 * m_mu2 - q2_safe - m12s  # m13 = m_eta^2 + 2 m_mu^2 - q2 - m12, so m12 + m13 + q2 = m_eta^2 + 2 m_mu^2

    # |M|^2 = |F|^2 / q2 * (m_eta^2 - q2)^2 [1 + cos^2 theta* + (4 m_mu^2 / q2) sin^2 theta*]:
    #   1 / q2                the virtual photon: 1/q2^2 from its propagator, times q2 from its decay to mu+ mu-
    #   (m_eta^2 - q2)^2      the eta -> gamma gamma* coupling, from the photons' momenta
    #   [...]                 the mu+ direction in the mu+ mu- frame; with
    #                         m13 - m12 = (m_eta^2 - q2) beta cos theta*, (m_eta^2 - q2)^2 [...] is
    #                         (m_eta^2 - q2)^2 (1 + 4 m_mu^2 / q2) + (m13 - m12)^2
    density = form_factor2 / q2_safe * ((m_eta2 - q2_safe)**2 * (1.0 + 4.0 * m_mu2 / q2_safe)
                                        + (m13s - m12s)**2)
    # 0 below physical threshold.
    return np.where(physical, density, 0.0)


def dalitz_factor(M, m12_range, m23_range, n_m23=400):
    """
    Dalitz-decay counterpart of phase_space_factor: the fraction of the
    Dalitz bin m12_range x m23_range that is kinematically reachable by a
    gamma mu+ mu- system of invariant mass M (array of trial masses, as in
    phase_space_factor), with each point of the bin weighted by
    dalitz_density instead of counting equally:

        f(M) = sum over q^2 strips of int_{reachable m12 in the bin} density dm12
               / sum over q^2 strips of int_{bin's m12 range} density dm12

    i.e. int_{bin and D(M)} density dA / int_{bin} density dA.
    phase_space_factor is the same ratio with density = 1. 1 where the whole
    bin is inside the Dalitz region of mass M, 0 where none of it is. At
    fixed q^2 the density is quadratic in m12, so the m12 integral is exact;
    q^2 is sampled at the midpoints of n_m23 strips.
    """
    m_mu2 = (M_MU / 1000.0)**2
    m_eta2 = (M_ETA / 1000.0)**2
    M2 = (np.asarray(M, float) / 1000.0)[:, None]**2
    bin_lo23, bin_hi23 = m23_range  # the bin's m23 edges
    bin_lo12, bin_hi12 = m12_range  # the bin's m12 edges
    # Midpoints of n_m23 equal strips across the bin's q2 range, one column per strip.
    q2 = (bin_lo23 + (np.arange(n_m23) + 0.5) * (bin_hi23 - bin_lo23) / n_m23)[None, :]

    # Reachable m12 at each (M, q2), as in phase_space_factor, clipped to the
    # bin's m12 range: [reach_lo12, reach_hi12] is the part of the bin inside
    # D(M) (zero length where none is).
    beta = np.sqrt(np.clip(1.0 - 4.0 * m_mu2 / q2, 0.0, None))  # muon speed in the mu+ mu- frame
    half_span = np.clip(M2 - q2, 0.0, None) / 2.0  # (M^2 - q2) / 2
    reach_lo12 = np.clip(m_mu2 + half_span * (1.0 - beta), bin_lo12, bin_hi12)
    reach_hi12 = np.clip(m_mu2 + half_span * (1.0 + beta), bin_lo12, bin_hi12)

    # dalitz_density at fixed q2, written as a polynomial in m12:
    #     density = q2_weight * (base_term + 4 (m12 - mid12)^2),
    # since m13 - m12 = 2 (mid12 - m12).
    physical = q2 > 4.0 * m_mu2
    q2_safe = np.where(physical, q2, 1.0)       # stand-in 1 below threshold, where q2_weight is 0
    q2_weight = np.where(physical, (1.0 - q2_safe / FF_LAMBDA**2)**-2 / q2_safe, 0.0)   # |F|^2 / q2
    base_term = (m_eta2 - q2_safe)**2 * (1.0 + 4.0 * m_mu2 / q2_safe)   # part not depending on m12
    mid12 = (m_eta2 + 2.0 * m_mu2 - q2_safe) / 2.0   # m12 where m13 = m12 (cos theta* = 0)

    # m12 integral of the density over the reachable part and over the whole
    # bin, from its antiderivative q2_weight * (base_term m12 + (4/3) (m12 - mid12)^3).
    reach_integral = q2_weight * (base_term * (reach_hi12 - reach_lo12)
                                  + 4.0 / 3.0 * ((reach_hi12 - mid12)**3 - (reach_lo12 - mid12)**3))
    bin_integral = q2_weight * (base_term * (bin_hi12 - bin_lo12)
                                + 4.0 / 3.0 * ((bin_hi12 - mid12)**3 - (bin_lo12 - mid12)**3))

    # Sum over the q2 strips (the strip width cancels in the ratio). A bin
    # entirely below threshold has bin_total 0 and factor 0.
    bin_total = bin_integral.sum(axis=1)
    return np.where(bin_total > 0,
                    reach_integral.sum(axis=1) / np.where(bin_total > 0, bin_total, 1.0), 0.0)
