"""
driver_dcb_mc_prior.py

Two-stage DCB fit. Fits truth-matched signal MC for the tail-shape parameters
(alphaL, nL, alphaR, nR), then fits data with the tails floating under a
Gaussian prior centred on the MC values.

    read MC root   → select → truth-match → best candidate
                   → fit_dcb(fixed_params={n_bkg: 0, background shape at init})
                   → tail_values, tail_cov
    read data root → select → best candidate
                   → fit_dcb(tail_prior=(tail_values, tail_cov * factor**2))
                   → results/plot

MC leg: n_bkg is fixed to 0 and the background shape parameters are fixed at
their model init values, so the fit has no background component.

Data leg: the tails are penalised by NormalConstraint(TAIL_PARAMS,
tail_values, error=tail_cov). The MC covariance is scaled by factor**2
(--tail-prior-factor), giving the data fit `factor` MC standard deviations of
slack per tail parameter.

Both fits use the full [xmin, xmax] window. The selection (TOS, pnn_mu cut,
best candidate by METRIC_BRANCH) is identical for both files; truth-matching
applies to the MC file only.

A single power-law tail cannot describe both the near and deep tails of MC:
nL/nR are a compromise that over-predicts the deepest tail (~17% at 17-23
sigma). See README.md.

Usage:
    python driver_dcb_mc_prior.py mc.root data.root [--outdir out] [--max-events N]
                       [--xmin 480] [--xmax 620] [--nbins 80]
                       [--mean-init 548] [--sigma-init 5]
                       [--alphaL-init 1.5] [--nL-init 2]
                       [--alphaR-init 1.5] [--nR-init 2]
                       [--tail-prior-factor 5]
"""

import argparse
import json
import os
import sys

import awkward as ak

# Add the "utils", "plot" and "fit" directories to the Python path so we can
# import the project modules.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'fit'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import bkg_utils
import cut_utils
import fit_utils
import kinematics
from fit_dcb import TAIL_PARAMS, fit_dcb
from plot_dcb import plot_dcb

# --- Selection ---------------------------------------------------------------
# Selection constants, applied identically to the MC and data files.

MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit
METRIC_BRANCH   = 'tag_dtf_chi2'   # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']   # branches the cuts below read
PNN_MU_MIN      = 0.4              # every muon daughter must exceed this
MUON_PID        = 13               # |pid| the pnn_mu cut applies to
MOTHER_PID      = 221              # signal mother PID, for truth-matching (eta)
DAUGHTER_PIDS   = [-13, 13]        # signal daughter PIDs, for truth-matching

# Trigger-on-signal lines per trigger level. The candidate must be TOS on at
# least one line at every level (cut_utils.tos_mask).
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}
TOS_BRANCHES = [branch for lines in TOS_LINES.values() for branch in lines]

BRANCHES = ([MASS_BRANCH, METRIC_BRANCH, 'prt_idx_mom']
            + DAUGHTER_FIELDS + TOS_BRANCHES)

# MC-only branches for truth-matching; read for the MC file only (the data
# file has no mc_* / prt_idx_gen branches).
MC_TRUTH_BRANCHES = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Fit signal MC to get a tail-shape prior, then fit data with it',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument('mc_infile', help='Input ROOT file, signal MC')
parser.add_argument('data_infile', help='Input ROOT file, data')
parser.add_argument('--outdir', default='out',
                    help='Output directory for results_dcb_*.json and dcb_mass_fit_*.png')
parser.add_argument('--label', default='',
                    help='Suffix for the output filenames, e.g. `--label nopnn` writes '
                         'results_dcb_data_nopnn.json / dcb_mass_fit_data_nopnn.png. Lets '
                         'several selections live side by side in one --outdir')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events, from each file (default: all)')
parser.add_argument('--pnn-mu-min', type=float, default=PNN_MU_MIN,
                    help=f'Muon PID threshold (default: {PNN_MU_MIN}, the floor already '
                         'applied upstream in the ntuple, i.e. no additional cut). Raise it '
                         'to tighten, e.g. --pnn-mu-min 0.95')
parser.add_argument('--no-tos', action='store_true',
                    help='Skip the TOS trigger requirement, for selection studies')
parser.add_argument('--no-truth-match', action='store_true',
                    help='Skip MC truth-matching on the MC leg, for selection studies')
parser.add_argument('--xmin', type=float, default=480.0, help='Lower mass bound [MeV]')
parser.add_argument('--xmax', type=float, default=620.0, help='Upper mass bound [MeV]')
parser.add_argument('--nbins', type=int, default=80, help='Number of histogram bins')
parser.add_argument('--mean-init', type=float, default=kinematics.M_ETA, help='Initial mean [MeV]')
parser.add_argument('--sigma-init', type=float, default=5.0, help='Initial sigma [MeV]')
parser.add_argument('--alphaL-init', type=float, default=1.5, help='Initial left tail α_L')
parser.add_argument('--nL-init', type=float, default=2.0, help='Initial left tail n_L')
parser.add_argument('--alphaR-init', type=float, default=1.5, help='Initial right tail α_R')
parser.add_argument('--nR-init', type=float, default=2.0, help='Initial right tail n_R')
parser.add_argument('--tail-prior-factor', type=float, default=5.0,
                    help='Stdev multiplier applied to the MC tail-parameter covariance '
                         'before it constrains the data fit (default: 5.0). Larger = looser '
                         'prior; smaller = tails closer to the MC values')
args = parser.parse_args()


def select_masses(infile, tag, truth_match=False):
    """
    Read `infile`, apply the selection constants above, keep the best
    candidate per event, and return (masses, selection_report), where masses
    is a flat numpy array of MASS_BRANCH values.

    `tag` ('mc' or 'data') labels the [INFO] log lines.

    `truth_match` (MC file only): require every daughter of a candidate to
    match the signal decay MOTHER_PID -> DAUGHTER_PIDS, i.e. a matched
    generator particle whose true PID equals the reco PID and whose mother
    is the signal mother (bkg_utils.signal_candidate_mask).
    """
    branches = list(BRANCHES) + (MC_TRUTH_BRANCHES if truth_match else [])
    tree = cut_utils.read_branches(infile, branches, entry_stop=args.max_events)
    tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)

    use_tos = not args.no_tos
    pnn_mu_min = args.pnn_mu_min

    keep = ak.ones_like(tree[MASS_BRANCH], dtype=bool)
    tos_report = None
    if use_tos:
        tos, tos_report = cut_utils.tos_mask(tree, TOS_LINES, tag_ref=MASS_BRANCH)
        keep = keep & tos
    if pnn_mu_min is not None:
        # Non-muon daughters pass automatically, so the cut is an 'all'
        # reduction over daughters onto the candidate.
        muon_ok = ((tree['prt_pnn_mu'] > pnn_mu_min)
                   | (abs(tree['prt_pid']) != MUON_PID))
        keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                       target_ref=MASS_BRANCH)
    n_after_reco_cuts = int(ak.sum(keep))

    truth_report = None
    if truth_match:
        # Per candidate: every daughter matches the signal decay
        # MOTHER_PID -> DAUGHTER_PIDS (see bkg_utils.signal_daughter_mask).
        truth_matched = bkg_utils.signal_candidate_mask(
            tree, MOTHER_PID, DAUGHTER_PIDS, target_ref=MASS_BRANCH)

        keep = keep & truth_matched
        truth_report = {
            'truth_matched': int(ak.sum(truth_matched)),
        }

    best = cut_utils.best_candidate_mask(tree, METRIC_BRANCH, keep)
    masses = cut_utils.values(tree, MASS_BRANCH, best)  # <- awkward → numpy

    print(f"[INFO] [{tag}] events: read {drop_report['read']}, kept {drop_report['kept']}, "
          f"dropped {drop_report['dropped_total']}", file=sys.stderr)
    for reason in ('dropped_no_candidates', 'dropped_idx_out_of_range',
                   'dropped_daughter_length_mismatch',
                   'dropped_partial_candidate_branch'):
        if drop_report[reason]:
            print(f'[INFO] [{tag}]   {reason}: {drop_report[reason]}', file=sys.stderr)
    for level, r in (tos_report or {}).items():
        print(f"[INFO] [{tag}] TOS at every level up to {level}: "
              f"{r['candidates_tos_cumulative']} candidates", file=sys.stderr)
        if r['ignored']:
            print(f"[WARN] [{tag}] {level} TOS lines absent in every event "
                  f"(contributing nothing): {r['ignored']}", file=sys.stderr)

    applied = ', '.join((['TOS'] if use_tos else [])
                        + ([f'pnn_mu>{pnn_mu_min}'] if pnn_mu_min is not None else [])) or 'none'
    cutflow = (f'[INFO] [{tag}] start: {int(ak.sum(ak.num(keep)))} candidates -> '
               f'cuts ({applied}): {n_after_reco_cuts} candidates')
    if truth_match:
        cutflow += f' -> truth-match: {int(ak.sum(keep))} candidates'
    cutflow += f' -> best candidates: {len(masses)} candidates'
    print(cutflow, file=sys.stderr)

    selection_report = {
        'events': drop_report,
        'triggers': tos_report,
        'truth_match': truth_report,
        'best_candidate': {
            'metric': METRIC_BRANCH, 'method': 'min',
            'events_with_best_candidate': len(masses),
        },
        'branches_read': branches,
    }
    return masses, selection_report


os.makedirs(args.outdir, exist_ok=True)
suffix = f'_{args.label}' if args.label else ''

# --- 1. Fit MC to get the tail-shape prior ------------------------------------
# Truth-matched MC has no background: n_bkg is fixed to 0 and the background
# shape parameters at their init values (fit_utils.BKG_INIT).
mc_masses, mc_selection_report = select_masses(args.mc_infile, 'mc',
                                                truth_match=not args.no_truth_match)
mc_result, m_mc = fit_dcb(
    mc_masses,
    xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=args.sigma_init,
    alphaL_init=args.alphaL_init, nL_init=args.nL_init,
    alphaR_init=args.alphaR_init, nR_init=args.nR_init,
    fixed_params={'n_bkg': 0.0, **fit_utils.BKG_INIT},
)

tail_values, tail_cov = fit_utils.build_prior(m_mc, TAIL_PARAMS,
                                              factor=args.tail_prior_factor)

print(f'[INFO] MC tail shape (prior for data fit, x{args.tail_prior_factor:.0f} '
      f'widened stdev): ' + ', '.join(
    f'{p}={v:.4f}±{m_mc.errors[p]:.4f}' for p, v in zip(TAIL_PARAMS, tail_values, strict=True)
), file=sys.stderr)

mc_result['meta']['input_file'] = args.mc_infile
mc_result['meta']['selection'] = {
    'tos': None if args.no_tos else TOS_LINES, 'pnn_mu_min': args.pnn_mu_min,
    'truth_match': not args.no_truth_match,
    'best_candidate': f'min {METRIC_BRANCH}',
}
mc_result['selection_report'] = mc_selection_report

mc_json_path = os.path.join(args.outdir, f'results_dcb_mc{suffix}.json')
with open(mc_json_path, 'w') as f:
    json.dump(mc_result, f, indent=2)
print(f'[DONE] MC JSON saved to: {mc_json_path}', file=sys.stderr)
plot_dcb(mc_result, os.path.join(args.outdir, f'dcb_mass_fit_mc{suffix}.png'))

# --- 2. Fit data with the MC tail shape as a (widened) prior ------------------
data_masses, data_selection_report = select_masses(args.data_infile, 'data')
data_result, m_data = fit_dcb(
    data_masses,
    xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=args.sigma_init,
    alphaL_init=tail_values[0], nL_init=tail_values[1],
    alphaR_init=tail_values[2], nR_init=tail_values[3],
    tail_prior=(tail_values, tail_cov),
)

data_result['meta']['input_file'] = args.data_infile
data_result['meta']['tail_prior_from'] = args.mc_infile
data_result['meta']['tail_prior_factor'] = args.tail_prior_factor
data_result['meta']['selection'] = {
    'tos': None if args.no_tos else TOS_LINES, 'pnn_mu_min': args.pnn_mu_min,
    'best_candidate': f'min {METRIC_BRANCH}',
}
data_result['selection_report'] = data_selection_report

data_json_path = os.path.join(args.outdir, f'results_dcb_data{suffix}.json')
with open(data_json_path, 'w') as f:
    json.dump(data_result, f, indent=2)
print(f'[DONE] data JSON saved to: {data_json_path}', file=sys.stderr)
plot_dcb(data_result, os.path.join(args.outdir, f'dcb_mass_fit_data{suffix}.png'))
