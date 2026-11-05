"""
driver_mass_fit.py

Gaussian and DCB + Chebyshev mass fits of one ROOT file:

    read ROOT → [truth-match] → select → random candidate → fit_gauss → fit_dcb (seeded with the Gauss σ)
              → selection.json, results_gauss.json, gauss_mass_fit.png,
                results_dcb.json, dcb_mass_fit.png

This script owns everything about the input: which file, which branches, which selection.
fit_gauss() and fit_dcb() downstream see only a numpy array of masses.

One candidate per event is picked at random among those that pass the cuts
(cut_utils.random_candidate_mask, seed SEED).

--truth-match PID... (signal MC only) keeps the candidates whose every daughter matches eta -> PIDs
(bkg_utils.signal_candidate_mask). It is the first cut, before TOS, pnn_mu and the candidate pick.
The DCB fit runs with n_bkg fixed to 0 and the background shape at its init values; the Gaussian fit
keeps its background floating. Its results_dcb.json is the --tail-prior of the data fit. The PIDs
are -13 13 (eta -> mu+ mu-), -13 13 22 (eta -> mu+ mu- gamma) or -13 -13 13 13 (eta -> 4mu).

An input path with an eta2mumumumu segment (eta -> 4mu) takes TOS_LINES_4MU in place of TOS_LINES.

--sigma-prior RESULTS_JSON (with --truth-match) takes the results_gauss.json of an earlier
--truth-match run and adds NormalConstraint(sigma, Gaussian sigma, Gaussian variance *
SIGMA_PRIOR_FACTOR**2) to the DCB fit.

--tail-prior RESULTS_JSON fixes the DCB tails at the MC values with TAIL_PRIOR_FACTOR = 0; otherwise
it floats them under NormalConstraint(TAIL_PARAMS, MC values, MC covariance * TAIL_PRIOR_FACTOR**2),
starting from the MC values.

A single power-law tail cannot describe both the near and deep tails of MC:
nL/nR are a compromise that over-predicts the deepest tail (~17% at 17-23 sigma). See README.md.

Usage:
    python src/fit/driver_mass_fit.py INFILE --outdir OUTDIR
        [--truth-match PID [PID ...] [--sigma-prior RESULTS_JSON] | --tail-prior RESULTS_JSON]
        [--max-events N]
        [--xmin 500] [--xmax 600] [--nbins 100] [--mean-init 548] [--sigma-init 5]
        [--alphaL-init 1.5] [--nL-init 2] [--alphaR-init 1.5] [--nR-init 2] [--n-max N]
"""

import argparse
import json
import os
import sys

import awkward as ak

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import bkg_utils
import cut_utils
import fit_utils
import kinematics
from fit_dcb import TAIL_PARAMS, fit_dcb
from fit_gauss import fit_gauss
from plot_mass_fit import dcb_text, gauss_text, plot_fit

# --- Selection ---------------------------------------------------------------

MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit
SEED            = 0                # random candidate pick
DAUGHTER_FIELDS = ['prt_pid', 'prt_px', 'prt_py', 'prt_pz', 'prt_pnn_mu',
                   'prt_prb_ghost', 'prt_ip_chi2', 'prt_tight_mu']   # branches the cuts below read
MUON_PID        = 13               # |pid| the muon cuts apply to
MOTHER_PID      = 221              # eta, for --truth-match
TRUTH_BRANCHES  = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']   # read with --truth-match

# Muon requirements of the dark-photon ananote (LHCb-ANA-2017-027, Tables 1 and 3), with p at the
# 2018 Hlt1DiMuonNoIP threshold. Every muon daughter must pass each; a non-muon daughter passes.
MU_PT_MIN       = 1000.0           # pT [MeV]
MU_P_MIN        = 10000.0          # p [MeV]
MU_ETA_RANGE    = (2.0, 4.5)       # pseudorapidity
PNN_MU_MIN      = 0.9              # ProbNNmu
MU_GHOST_MAX    = 0.2              # ghost probability
MU_IP_CHI2_MAX  = 6.0              # IP chi2

# TOS lines per trigger level.
# TODO: use the commented-out lines again.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0'],                    # Hlt1DiMuonNoIP
    # 'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0'],                    # Hlt2ExoticaPrmptDiMuonTurbo
    # 'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}
# eta -> 4mu: HLT1 also takes Hlt1MultiDiMuonNoIP (two dimuon pairs)
TOS_LINES_4MU = {**TOS_LINES, 'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos2']}   # + Hlt1MultiDiMuonNoIP

TAIL_PRIOR_FACTOR = 5.0   # --tail-prior: MC tail-fit stdevs times this; 0 fixes the tails
SIGMA_PRIOR_FACTOR = 5.0  # --sigma-prior: Gaussian sigma stdev times this

parser = argparse.ArgumentParser(
    description='Gaussian and DCB + Chebyshev mass fits',
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument('infile', help='Input ROOT file')
parser.add_argument('--outdir', required=True,
                    help='Directory for selection.json, results_*.json and *_mass_fit.png')
parser.add_argument('--truth-match', nargs='+', type=int, default=None, metavar='PID',
                    help='Signal MC: keep candidates truth-matched to eta -> these daughter '
                         'PIDs (-13 13, -13 13 22, or -13 -13 13 13) and fit with no background')
parser.add_argument('--tail-prior', default=None,
                    help='results_dcb.json of a --truth-match fit whose tails are the DCB prior')
parser.add_argument('--sigma-prior', default=None,
                    help='results_gauss.json of a --truth-match fit whose sigma is the DCB sigma '
                         'prior (with --truth-match)')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events (default: all)')
parser.add_argument('--xmin', type=float, default=500.0, help='Lower mass bound [MeV]')
parser.add_argument('--xmax', type=float, default=600.0, help='Upper mass bound [MeV]')
parser.add_argument('--nbins', type=int, default=100, help='Number of histogram bins')
parser.add_argument('--mean-init', type=float, default=kinematics.M_ETA, help='Initial mean [MeV]')
parser.add_argument('--sigma-init', type=float, default=5.0,
                    help='Initial sigma [MeV] of the Gaussian')
parser.add_argument('--alphaL-init', type=float, default=1.5, help='Initial left tail α_L')
parser.add_argument('--nL-init', type=float, default=2.0, help='Initial left tail n_L')
parser.add_argument('--alphaR-init', type=float, default=1.5, help='Initial right tail α_R')
parser.add_argument('--nR-init', type=float, default=2.0, help='Initial right tail n_R')
parser.add_argument('--n-max', type=float, default=None,
                    help='Upper limit of the DCB tail exponents n_L and n_R (default: none)')
args = parser.parse_args()
if args.truth_match and args.tail_prior:
    parser.error('--truth-match and --tail-prior are exclusive')
if args.sigma_prior and not args.truth_match:
    parser.error('--sigma-prior needs --truth-match')
os.makedirs(args.outdir, exist_ok=True)
tos_lines = TOS_LINES_4MU if 'eta2mumumumu' in args.infile.split('/') else TOS_LINES

branches = ([MASS_BRANCH, 'prt_idx_mom'] + DAUGHTER_FIELDS
            + [line for lines in tos_lines.values() for line in lines]
            + (TRUTH_BRANCHES if args.truth_match else []))

# --- Read --------------------------------------------------------------------

tree = cut_utils.read_branches(args.infile, branches, entry_stop=args.max_events)
tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}, "
      f"dropped {drop_report['read'] - drop_report['kept']}", file=sys.stderr)
for reason in ('dropped_no_candidates', 'dropped_idx_out_of_range',
               'dropped_daughter_length_mismatch', 'dropped_partial_candidate_branch'):
    if drop_report[reason]:
        print(f'[INFO]   {reason}: {drop_report[reason]}', file=sys.stderr)

# --- Select ------------------------------------------------------------------
# Every cut is a per-candidate boolean mask shaped like MASS_BRANCH, so they combine with a plain
# `&` and adding one is a single line.

keep = ak.ones_like(tree[MASS_BRANCH], dtype=bool)
cutflow = [('all candidates', int(ak.sum(ak.num(keep))))]

# Truth match first: every daughter matches eta -> args.truth_match.
if args.truth_match:
    keep = keep & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, args.truth_match,
                                                  target_ref=MASS_BRANCH)
    cutflow.append(('truth-matched', int(ak.sum(keep))))

# TOS on at least one line at every trigger level.
tos, tos_report = cut_utils.tos_mask(tree, tos_lines, tag_ref=MASS_BRANCH, base=keep)
keep = keep & tos
triggers = {level: report['lines'] for level, report in tos_report.items()}
for level, report in tos_report.items():
    cutflow.append((f'TOS up to {level}', report['tos_up_to_level']))
    for line, n in report['lines'].items():
        if n == 0:
            print(f'[WARN] {level} TOS line {line} passes no candidate', file=sys.stderr)

# Muon cuts, each on every muon daughter; a non-muon daughter passes.
px, py, pz = tree['prt_px'], tree['prt_py'], tree['prt_pz']
eta = ak.unflatten(kinematics.eta(ak.to_numpy(ak.flatten(px)), ak.to_numpy(ak.flatten(py)),
                                  ak.to_numpy(ak.flatten(pz))), ak.num(px))
muon_cuts = {
    f'pT > {MU_PT_MIN:g}':   kinematics.pt(px, py) > MU_PT_MIN,
    f'p > {MU_P_MIN:g}':     kinematics.p(px, py, pz) > MU_P_MIN,
    f'{MU_ETA_RANGE[0]:g} < eta < {MU_ETA_RANGE[1]:g}':
        (eta > MU_ETA_RANGE[0]) & (eta < MU_ETA_RANGE[1]),
    f'pnn_mu > {PNN_MU_MIN}': tree['prt_pnn_mu'] > PNN_MU_MIN,
    f'ghost_prob < {MU_GHOST_MAX}': tree['prt_prb_ghost'] < MU_GHOST_MAX,
    f'ip_chi2 < {MU_IP_CHI2_MAX:g}':   # -1 is stored when there is no PV
        (tree['prt_ip_chi2'] >= 0) & (tree['prt_ip_chi2'] < MU_IP_CHI2_MAX),
    'IsMuonTight':           tree['prt_tight_mu'] > 0.5,
}
for label, passes in muon_cuts.items():
    muon_ok = passes | (abs(tree['prt_pid']) != MUON_PID)
    keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                   target_ref=MASS_BRANCH)
    cutflow.append((label, int(ak.sum(keep))))

# One candidate per event: picked at random among those kept.
picked = cut_utils.random_candidate_mask(keep, SEED)
masses = cut_utils.values(tree, MASS_BRANCH, picked)  # <- awkward → numpy
cutflow.append(('random candidates', len(masses)))
for label, n in cutflow:
    print(f'[INFO]   {label:24s} {n:9d}', file=sys.stderr)

selection = {
    'meta': {
        'input_file': args.infile,
        'mass_branch': MASS_BRANCH,
        'selection': {
            'tos': tos_lines,
            'mu_pt_min': MU_PT_MIN,
            'mu_p_min': MU_P_MIN,
            'mu_eta_range': MU_ETA_RANGE,
            'pnn_mu_min': PNN_MU_MIN,
            'mu_ghost_max': MU_GHOST_MAX,
            'mu_ip_chi2_max': MU_IP_CHI2_MAX,
            'mu_tight': True,
            'truth_match': args.truth_match,
            'random_candidate': {'seed': SEED},
            'max_events': args.max_events,
        },
    },
    'selection_report': {
        'events': drop_report,
        'triggers': triggers,
        'cutflow': [{'cut': label, 'candidates': n} for label, n in cutflow],
    },
}
selection_path = os.path.join(args.outdir, 'selection.json')
with open(selection_path, 'w') as f:
    json.dump(selection, f, indent=2)
print(f'[DONE] JSON saved to: {selection_path}', file=sys.stderr)

# --- Fit ---------------------------------------------------------------------
# The turbo DCB fit is the last of three runs of this script (run_mass_fits.sh):
#
#   1. MC, --truth-match              Gaussian, then DCB with no background and no prior
#   2. MC, --truth-match              Gaussian, then DCB with no background, sigma under
#      --sigma-prior <1>/results_gauss.json   a prior from the Gaussian sigma of 1
#   3. turbo, --tail-prior            Gaussian, then DCB + Chebyshev, tails under a prior
#      <2>/results_dcb.json                   from the DCB tails of 2
#
# Every run fits the Gaussian first and starts the DCB sigma from its sigma (or from the prior's).
# fit_gauss() and fit_dcb() each run two MIGRAD stages: stage 1 with the background (Gaussian) or
# the tails (DCB) fixed at their init values, stage 2 with every parameter floating except
# fixed_params.

fits = []   # (name, result, stats-box builder) of each fit, written below

# The Gaussian fit's sigma seeds the DCB. Truth-matched MC: the DCB has no background, n_bkg fixed
# to 0 and the background shape parameters at their init values (fit_utils.BKG_INIT).
gauss_result, m_gauss = fit_gauss(
    masses, xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=args.sigma_init)
gauss_result['meta']['input_file'] = args.infile
fits.append(('gauss', gauss_result, gauss_text))
sigma_init = abs(m_gauss.values['sigma'])   # the Gaussian fit's sigma can have either sign
fixed_params = None
if args.truth_match:
    fixed_params = {'n_bkg': 0.0, **fit_utils.BKG_INIT}

# DCB sigma: held near the Gaussian sigma of a --truth-match fit, with that fit's stdev widened by
# SIGMA_PRIOR_FACTOR.
sigma_prior = None
if args.sigma_prior:
    with open(args.sigma_prior) as f:
        gauss_fit = json.load(f)['fit']
    sigma_prior = fit_utils.build_prior_from_result(gauss_fit, ['sigma'],
                                                    factor=SIGMA_PRIOR_FACTOR)
    sigma_init = sigma_prior[0][0]
    print(f"[INFO] Gaussian sigma (prior, x{SIGMA_PRIOR_FACTOR:.0f} widened stdev): "
          f"{gauss_fit['parameters']['sigma']['value']:.4f}"
          f"±{gauss_fit['parameters']['sigma']['error']:.4f}", file=sys.stderr)

# DCB tails: fixed at the MC values (TAIL_PRIOR_FACTOR 0), from the MC tail prior widened by
# TAIL_PRIOR_FACTOR, or free.
tail_init = (args.alphaL_init, args.nL_init, args.alphaR_init, args.nR_init)
tail_prior = None
if args.tail_prior:
    with open(args.tail_prior) as f:
        mc_fit = json.load(f)['fit']
    tail_prior = fit_utils.build_prior_from_result(mc_fit, TAIL_PARAMS, factor=TAIL_PRIOR_FACTOR)
    tail_init = tail_prior[0]
    print(f'[INFO] MC tail shape (prior, x{TAIL_PRIOR_FACTOR:.0f} widened stdev; 0 = fixed): '
          + ', '.join(f"{name}={mc_fit['parameters'][name]['value']:.4f}"
                      f"±{mc_fit['parameters'][name]['error']:.4f}" for name in TAIL_PARAMS),
          file=sys.stderr)
    if TAIL_PRIOR_FACTOR == 0:
        fixed_params = dict(zip(TAIL_PARAMS, tail_init))
        tail_prior = None

dcb_result, m_dcb = fit_dcb(
    masses, xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=sigma_init,
    alphaL_init=tail_init[0], nL_init=tail_init[1],
    alphaR_init=tail_init[2], nR_init=tail_init[3],
    tail_prior=tail_prior, sigma_prior=sigma_prior, fixed_params=fixed_params,
    limits={'nL': (0.0, args.n_max), 'nR': (0.0, args.n_max)})
dcb_result['meta']['input_file'] = args.infile
dcb_result['meta']['n_max'] = args.n_max
if args.sigma_prior:
    dcb_result['meta']['sigma_prior_from'] = args.sigma_prior
    dcb_result['meta']['sigma_prior_factor'] = SIGMA_PRIOR_FACTOR
if args.tail_prior:
    dcb_result['meta']['tail_prior_from'] = args.tail_prior
    dcb_result['meta']['tail_prior_factor'] = TAIL_PRIOR_FACTOR
fits.append(('dcb', dcb_result, dcb_text))

# --- Write -------------------------------------------------------------------

for name, result, build_text in fits:
    json_path = os.path.join(args.outdir, f'results_{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[DONE] JSON saved to: {json_path}', file=sys.stderr)
    plot_fit(result, os.path.join(args.outdir, f'{name}_mass_fit.png'), build_text)
