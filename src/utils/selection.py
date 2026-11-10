"""
selection.py

Selection constants and per-candidate masks shared by driver_mass_fit.py, driver_dalitz.py and
plot_cut_steps.py. The scripts choose what to call and in which order.

    muon_cuts(tree)    daughter-level mask of each muon requirement
    cut_steps(...)     truth match, TOS per level, then each muon cut, one mask per step
"""

import awkward as ak

import bkg_utils
import cut_utils
import kinematics

SEED              = 0                # random candidate pick; the drivers' chunk i takes SEED + i
STEP_SIZE         = '1 GB'           # chunk size the drivers read and select at a time
MUON_PID          = 13               # |pid| the muon cuts apply to
MOTHER_PID        = 221              # eta, for the truth match
TRUTH_BRANCHES    = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']   # read for the truth match
MUON_CUT_BRANCHES = ['prt_pid', 'prt_px', 'prt_py', 'prt_pz', 'prt_pnn_mu',
                     'prt_prb_ghost', 'prt_ip_chi2', 'prt_tight_mu']   # read by muon_cuts

# Muon requirements of the dark-photon ananote (LHCb-ANA-2017-027, Tables 1 and 3), with p at the
# 2018 Hlt1DiMuonNoIP threshold. Every muon daughter must pass each; a non-muon daughter passes.
MU_PT_MIN       = 1000.0           # pT [MeV]
MU_P_MIN        = 10000.0          # p [MeV]
MU_ETA_RANGE    = (2.0, 4.5)       # pseudorapidity
PNN_MU_MIN      = 0.9              # ProbNNmu
MU_GHOST_MAX    = 0.2              # ghost probability
MU_IP_CHI2_MAX  = 6.0              # IP chi2

# The muon thresholds that differ by channel, as muon_cuts takes them ('pt_min' None: no pT cut)
MUON_THRESHOLDS = {'pt_min': MU_PT_MIN, 'p_min': MU_P_MIN, 'pnn_mu_min': PNN_MU_MIN}
# eta -> 4mu: no pT cut, p > 5 GeV, ProbNNmu > 0.8
MUON_THRESHOLDS_4MU = {'pt_min': None, 'p_min': 5000.0, 'pnn_mu_min': 0.8}

# TOS lines per trigger level.
# TODO: use the commented-out lines again.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0'],                    # Hlt1DiMuonNoIP
    # 'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0'],                    # Hlt2ExoticaPrmptDiMuonTurbo
    # 'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}
# eta -> 4mu: L0 only L0DiMuon, HLT1 also Hlt1MultiDiMuonNoIP (two dimuon pairs), HLT2 also
# Hlt2ExoticaQuadMuonNoIP on its candidates rebuilt offline (a Full-stream line: tag_hlt2_tos4 is
# -1 in Turbo data)
TOS_LINES_4MU = {'L0':   ['tag_l0_tos0'],                            # L0DiMuon
                 'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos2'],          # + Hlt1MultiDiMuonNoIP
                 'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_rebuilt_tos4']}  # + Hlt2ExoticaQuadMuonNoIP


def muon_cuts(tree, thresholds=MUON_THRESHOLDS):
    """
    {label: daughter-level mask} of each muon requirement above, True for every non-muon daughter,
    with the pT, p and ProbNNmu thresholds of `thresholds` (MUON_THRESHOLDS or
    MUON_THRESHOLDS_4MU). Reads MUON_CUT_BRANCHES.
    """
    pt_min, p_min, pnn_mu_min = (thresholds['pt_min'], thresholds['p_min'],
                                 thresholds['pnn_mu_min'])
    px, py, pz = tree['prt_px'], tree['prt_py'], tree['prt_pz']
    eta = ak.unflatten(kinematics.eta(ak.to_numpy(ak.flatten(px)), ak.to_numpy(ak.flatten(py)),
                                      ak.to_numpy(ak.flatten(pz))), ak.num(px))
    cuts = {}
    if pt_min is not None:
        cuts[f'pT > {pt_min:g}'] = kinematics.pt(px, py) > pt_min
    cuts.update({
        f'p > {p_min:g}':        kinematics.p(px, py, pz) > p_min,
        f'{MU_ETA_RANGE[0]:g} < eta < {MU_ETA_RANGE[1]:g}':
            (eta > MU_ETA_RANGE[0]) & (eta < MU_ETA_RANGE[1]),
        f'pnn_mu > {pnn_mu_min}': tree['prt_pnn_mu'] > pnn_mu_min,
        f'ghost_prob < {MU_GHOST_MAX}': tree['prt_prb_ghost'] < MU_GHOST_MAX,
        f'ip_chi2 < {MU_IP_CHI2_MAX:g}':   # -1 is stored when there is no PV
            (tree['prt_ip_chi2'] >= 0) & (tree['prt_ip_chi2'] < MU_IP_CHI2_MAX),
        'IsMuonTight':           tree['prt_tight_mu'] > 0.5,
    })
    not_muon = abs(tree['prt_pid']) != MUON_PID
    return {label: passes | not_muon for label, passes in cuts.items()}


def cut_steps(tree, mass_branch, tos_lines, truth_pids=None, muon_thresholds=MUON_THRESHOLDS):
    """
    The selection one cut at a time, each on top of the ones before: the truth match to eta ->
    truth_pids (if given), TOS on at least one line of each level of tos_lines, then each of
    muon_cuts on every muon daughter.

    tree        : awkward event tree
    mass_branch : the tag_* branch defining the candidates
    tos_lines   : {level: [TOS branches]}, e.g. TOS_LINES
    truth_pids  : signal daughter PIDs, e.g. [-13, 13, 22], or None for no truth match
    muon_thresholds : the thresholds of muon_cuts, MUON_THRESHOLDS or MUON_THRESHOLDS_4MU

    Returns (steps, triggers):
        steps    : [(label, keep), ...], the per-candidate mask after each cut, starting with
                   ('all candidates', every candidate)
        triggers : {level: {line: candidates TOS on that line}}, among the truth-matched
                   candidates (all candidates without a truth match)
    """
    keep = ak.ones_like(tree[mass_branch], dtype=bool)
    steps = [('all candidates', keep)]

    # Truth match first: every daughter matches eta -> truth_pids
    if truth_pids:
        keep = keep & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, truth_pids,
                                                      target_ref=mass_branch)
        steps.append(('truth-matched', keep))

    # TOS one level at a time; the line counts are among the candidates before any TOS cut
    before_tos = keep
    triggers = {}
    for level, lines in tos_lines.items():
        tos, report = cut_utils.tos_mask(tree, {level: lines}, tag_ref=mass_branch,
                                         base=before_tos)
        keep = keep & tos
        triggers[level] = report[level]['lines']
        steps.append((f'TOS up to {level}', keep))

    # Muon cuts, each on every muon daughter
    for label, muon_ok in muon_cuts(tree, muon_thresholds).items():
        keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                       target_ref=mass_branch)
        steps.append((label, keep))
    return steps, triggers
