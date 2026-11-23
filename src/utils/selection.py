"""
selection.py

Selection constants and per-candidate masks. The scripts choose what to call and in which order.

A cut is a string of a quantity, comparison operators and thresholds, separated by spaces: one
condition ('pnn_mu > 0.9') or a range ('2 < eta < 4.5'). MUON_CUTS apply to every muon daughter,
PHOTON_CUTS to every photon daughter.

    branches(...)                    the branches cut_steps reads
    parse_cut(cut)                   [(quantity, operator, threshold), ...] of a cut
    cut_mask(values, cut)            mask of the values passing a cut
    quantity(tree, name)             daughter-level values of a cut quantity
    daughter_masks(tree, cuts, pid)  daughter-level mask of each cut
    cut_steps(...)                   MC match, TOS per level, then each muon and photon cut, one
                                     mask per step
"""

import operator

import awkward as ak

import bkg_utils
import cut_utils
import kinematics

SEED              = 0                # random candidate pick; the drivers' chunk i takes SEED + i
STEP_SIZE         = '1 GB'           # chunk size the drivers read and select at a time
MUON_PID          = 13               # |pid| the muon cuts apply to
PHOTON_PID        = 22               # pid the photon cuts apply to
MOTHER_PID        = 221              # eta, for the MC match
MC_BRANCHES       = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']   # read for the MC match
MUON_CUT_BRANCHES = ['prt_pid', 'prt_px', 'prt_py', 'prt_pz', 'prt_pnn_mu',
                     'prt_prb_ghost', 'prt_ip_chi2', 'prt_tight_mu']   # read by the cuts

# Comparison operators a cut can use, and each one's mirror for the low side of a range
OPERATORS = {'>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le}
MIRRORED  = {'>': '<', '>=': '<=', '<': '>', '<=': '>='}

# Muon requirements of the dark-photon ananote (LHCb-ANA-2017-027, Tables 1 and 3), with p at the
# 2018 Hlt1DiMuonNoIP threshold. Every muon daughter must pass each; a non-muon daughter passes.
# pt and p in MeV; ip_chi2 is -1 where no PV is stored.
MUON_CUTS = ['pt > 1000', 'p > 10000', '2 < eta < 4.5', 'pnn_mu > 0.9', 'prb_ghost < 0.2',
             '0 <= ip_chi2 < 6', 'tight_mu > 0.5']
# eta -> 4mu: no pT cut, p > 5 GeV, ProbNNmu > 0.6
MUON_CUTS_4MU = ['p > 5000', '2 < eta < 4.5', 'pnn_mu > 0.6', 'prb_ghost < 0.2',
                 '0 <= ip_chi2 < 6', 'tight_mu > 0.5']
# Photon requirements: every photon daughter must pass each; a non-photon daughter passes.
PHOTON_CUTS = []

# TOS lines per trigger level.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0'],                    # Hlt1DiMuonNoIP
    'HLT2': ['tag_hlt2_tos0'],                    # Hlt2ExoticaPrmptDiMuonTurbo
}
# eta -> 4mu: HLT1 also Hlt1MultiDiMuonNoIP (two dimuon pairs), HLT2 also Hlt2ExoticaQuadMuonNoIP
# on its candidates rebuilt offline (a Full-stream line: tag_hlt2_tos4 is -1 in Turbo data)
TOS_LINES_4MU = {**TOS_LINES,
                 'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos2'],          # + Hlt1MultiDiMuonNoIP
                 'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_rebuilt_tos4']}  # + Hlt2ExoticaQuadMuonNoIP
# Displaced: HLT2 Hlt2ExoticaDisplDiMuon on its candidates rebuilt offline (a Full-stream line:
# tag_hlt2_tos3 is -1 in Turbo data) in place of the prompt line
TOS_LINES_DISPLACED = {**TOS_LINES,
                       'HLT2': ['tag_hlt2_rebuilt_tos3']}   # Hlt2ExoticaDisplDiMuon


def branches(mass_branch, tos_lines, mc_match=False):
    """
    The branches cut_steps reads: mass_branch, prt_idx_mom, MUON_CUT_BRANCHES, every TOS branch of
    tos_lines and, with mc_match, MC_BRANCHES.
    """
    return ([mass_branch, 'prt_idx_mom'] + MUON_CUT_BRANCHES
            + [line for lines in tos_lines.values() for line in lines]
            + (MC_BRANCHES if mc_match else []))


def parse_cut(cut):
    """
    [(quantity, operator, threshold), ...] of a cut string, its words separated by spaces:
    'pt > 1000' gives [('pt', '>', 1000.0)]; the range '2 < eta < 4.5' gives
    [('eta', '>', 2.0), ('eta', '<', 4.5)].
    """
    words = cut.split()
    if len(words) == 3:
        name, op, threshold = words
        return [(name, op, float(threshold))]
    if len(words) == 5:
        low, low_op, name, high_op, high = words
        return [(name, MIRRORED[low_op], float(low)), (name, high_op, float(high))]
    raise ValueError(f"cut {cut!r} is neither 'quantity op threshold' nor "
                     "'low op quantity op high'")


def cut_mask(values, cut):
    """Mask of the values (any array) passing every condition of `cut`."""
    conditions = parse_cut(cut)
    _, op, threshold = conditions[0]
    mask = OPERATORS[op](values, threshold)
    for _, op, threshold in conditions[1:]:
        mask = mask & OPERATORS[op](values, threshold)
    return mask


def quantity(tree, name):
    """
    Daughter-level values of the cut quantity `name`: 'pt', 'p' [MeV] and 'eta' from prt_px,
    prt_py, prt_pz, any other name the branch prt_<name>.
    """
    px, py, pz = tree['prt_px'], tree['prt_py'], tree['prt_pz']
    if name == 'pt':
        return kinematics.pt(px, py)
    if name == 'p':
        return kinematics.p(px, py, pz)
    if name == 'eta':
        return ak.unflatten(kinematics.eta(ak.to_numpy(ak.flatten(px)),
                                           ak.to_numpy(ak.flatten(py)),
                                           ak.to_numpy(ak.flatten(pz))), ak.num(px))
    return tree[f'prt_{name}']


def daughter_masks(tree, cuts, pid):
    """
    {cut: daughter-level mask} of each cut of `cuts`: cut_mask of the cut's quantity for every
    daughter with |prt_pid| == pid, True for every other daughter.
    """
    other = abs(tree['prt_pid']) != pid
    masks = {}
    for cut in cuts:
        name = parse_cut(cut)[0][0]
        masks[cut] = cut_mask(quantity(tree, name), cut) | other
    return masks


def cut_steps(tree, mass_branch, tos_lines, mc_pids=None, muon_cuts=MUON_CUTS,
              photon_cuts=PHOTON_CUTS):
    """
    The selection one cut at a time, each on top of the ones before: the MC match to eta ->
    mc_pids (if given), TOS on at least one line of each level of tos_lines, each cut of
    muon_cuts on every muon daughter, then each cut of photon_cuts on every photon daughter.

    tree        : awkward event tree
    mass_branch : the tag_* branch defining the candidates
    tos_lines   : {level: [TOS branches]}, e.g. TOS_LINES
    mc_pids     : signal daughter PIDs, e.g. [-13, 13, 22], or None for no MC match
    muon_cuts   : cut strings, MUON_CUTS or MUON_CUTS_4MU
    photon_cuts : cut strings, PHOTON_CUTS

    Returns (steps, triggers):
        steps    : [(label, keep), ...], the per-candidate mask after each cut, starting with
                   ('all candidates', every candidate); a muon cut's label is the cut, a photon
                   cut's 'photon <cut>'
        triggers : {level: {line: candidates TOS on that line}}, among the MC-matched
                   candidates (all candidates without an MC match)
    """
    keep = ak.ones_like(tree[mass_branch], dtype=bool)
    steps = [('all candidates', keep)]

    # MC match first: every daughter matches eta -> mc_pids
    if mc_pids:
        keep = keep & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, mc_pids,
                                                      target_ref=mass_branch)
        steps.append(('MC-matched', keep))

    # TOS one level at a time; the line counts are among the candidates before any TOS cut
    before_tos = keep
    triggers = {}
    for level, lines in tos_lines.items():
        tos, report = cut_utils.tos_mask(tree, {level: lines}, tag_ref=mass_branch,
                                         base=before_tos)
        keep = keep & tos
        triggers[level] = report[level]['lines']
        steps.append((f'TOS up to {level}', keep))

    # Muon cuts, each on every muon daughter, labelled by the cut; then photon cuts, each on every
    # photon daughter, labelled 'photon <cut>'
    for cuts, pid, prefix in ((muon_cuts, MUON_PID, ''), (photon_cuts, PHOTON_PID, 'photon ')):
        for cut, daughter_ok in daughter_masks(tree, cuts, pid).items():
            keep = keep & cut_utils.reduce(tree, daughter_ok, 'prt_idx_mom', how='all',
                                           target_ref=mass_branch)
            steps.append((prefix + cut, keep))
    return steps, triggers
