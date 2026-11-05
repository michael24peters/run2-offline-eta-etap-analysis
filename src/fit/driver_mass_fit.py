"""
driver_mass_fit.py

Gaussian and DCB + Chebyshev mass fits of one ROOT file:

    read ROOT → [truth-match] → select → random candidate → fit_gauss → fit_dcb (seeded with the Gauss σ)
              → selection.json, results_gauss.json, gauss_mass_fit.png,
                results_dcb.json, dcb_mass_fit.png

This script owns everything about the input: which file, which branches, which selection (the
constants and cut_steps of selection.py). fit_gauss() and fit_dcb() downstream see only a numpy
array of masses.

One candidate per event is picked at random among those that pass the cuts
(cut_utils.random_candidate_mask, seed selection.SEED).

--truth-match PID... (signal MC only) keeps the candidates whose every daughter matches eta -> PIDs
(bkg_utils.signal_candidate_mask). It is the first cut, before TOS, pnn_mu and the candidate pick.
The DCB fit runs with n_bkg fixed to 0 and the background shape at its init values; the Gaussian fit
keeps its background floating. Its results_dcb.json is the --tail-prior of the data fit. The PIDs
are -13 13 (eta -> mu+ mu-), -13 13 22 (eta -> mu+ mu- gamma) or -13 -13 13 13 (eta -> 4mu).

An input path with an eta2mumumumu segment (eta -> 4mu) takes selection.TOS_LINES_4MU in place of
selection.TOS_LINES.

--sigma-prior RESULTS_JSON (with --truth-match) takes the results_gauss.json of an earlier
--truth-match run and adds NormalConstraint(sigma, Gaussian sigma, Gaussian variance *
SIGMA_PRIOR_FACTOR**2) to the DCB fit.

--tail-prior RESULTS_JSON floats the DCB tails under NormalConstraint(TAIL_PARAMS, MC values, MC
covariance * TAIL_PRIOR_FACTOR**2), starting from the MC values.

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
import cut_utils
import fit_utils
import kinematics
import selection
from fit_dcb import TAIL_PARAMS, fit_dcb
from fit_gauss import fit_gauss
from plot_mass_fit import dcb_text, gauss_text, plot_fit

# --- Selection ---------------------------------------------------------------

MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit

TAIL_PRIOR_FACTOR = 5.0   # --tail-prior: MC tail-fit stdevs times this
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
tos_lines = (selection.TOS_LINES_4MU if 'eta2mumumumu' in args.infile.split('/')
             else selection.TOS_LINES)

branches = ([MASS_BRANCH, 'prt_idx_mom'] + selection.MUON_CUT_BRANCHES
            + [line for lines in tos_lines.values() for line in lines]
            + (selection.TRUTH_BRANCHES if args.truth_match else []))

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
# [truth match], TOS per trigger level, then each muon cut (selection.cut_steps). Every cut is a
# per-candidate boolean mask shaped like MASS_BRANCH.

steps, triggers = selection.cut_steps(tree, MASS_BRANCH, tos_lines, truth_pids=args.truth_match)
keep = steps[-1][1]
cutflow = [(label, int(ak.sum(mask))) for label, mask in steps]
for level, lines in triggers.items():
    for line, n in lines.items():
        if n == 0:
            print(f'[WARN] {level} TOS line {line} passes no candidate', file=sys.stderr)

# One candidate per event: picked at random among those kept.
picked = cut_utils.random_candidate_mask(keep, selection.SEED)
masses = cut_utils.values(tree, MASS_BRANCH, picked)  # <- awkward → numpy
cutflow.append(('random candidates', len(masses)))
for label, n in cutflow:
    print(f'[INFO]   {label:24s} {n:9d}', file=sys.stderr)

selection_json = {
    'meta': {
        'input_file': args.infile,
        'mass_branch': MASS_BRANCH,
        'selection': {
            'tos': tos_lines,
            'mu_pt_min': selection.MU_PT_MIN,
            'mu_p_min': selection.MU_P_MIN,
            'mu_eta_range': selection.MU_ETA_RANGE,
            'pnn_mu_min': selection.PNN_MU_MIN,
            'mu_ghost_max': selection.MU_GHOST_MAX,
            'mu_ip_chi2_max': selection.MU_IP_CHI2_MAX,
            'truth_match': args.truth_match,
            'random_candidate': {'seed': selection.SEED},
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
    json.dump(selection_json, f, indent=2)
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

# DCB tails: under the MC tail prior widened by TAIL_PRIOR_FACTOR, or free.
tail_init = (args.alphaL_init, args.nL_init, args.alphaR_init, args.nR_init)
tail_prior = None
if args.tail_prior:
    with open(args.tail_prior) as f:
        mc_fit = json.load(f)['fit']
    tail_prior = fit_utils.build_prior_from_result(mc_fit, TAIL_PARAMS, factor=TAIL_PRIOR_FACTOR)
    tail_init = tail_prior[0]
    print(f'[INFO] MC tail shape (prior, x{TAIL_PRIOR_FACTOR:.0f} widened stdev): '
          + ', '.join(f"{name}={mc_fit['parameters'][name]['value']:.4f}"
                      f"±{mc_fit['parameters'][name]['error']:.4f}" for name in TAIL_PARAMS),
          file=sys.stderr)

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
