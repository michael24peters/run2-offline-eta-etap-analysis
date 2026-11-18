"""
driver_mass_fit.py

Gaussian and DCB + Chebyshev mass fits of one input (a ROOT file or a quoted glob of them):

    read ROOT → [MC match] → select → random candidate → [fit_gauss] → [fit_dcb]
              → selection.json, [results_gauss.json, gauss_mass_fit.png],
                [results_dcb.json, dcb_mass_fit.png]

Which fits run, and where their start values and priors come from:

    --mc-match PID...       Gaussian, then DCB, both with no background (signal-only MC); the DCB
                            starts sigma at the Gaussian sigma
    no prior                Gaussian + Chebyshev, then DCB + Chebyshev starting sigma at the
                            Gaussian sigma, tails free (data, or MC without --mc-match)
    --sigma-prior JSON      DCB + Chebyshev only (data): sigma, and with --tail-prior JSON the
      [--tail-prior JSON]   tails, start at the values of an --mc-match results_dcb.json and are
                            held near them
    --gauss-only            the Gaussian of the first two rows only

fit_gauss is one MIGRAD stage. fit_dcb is two: tails fixed at their start values, then every
parameter not fixed floats. A prior is a term of the cost in both stages.

The input is read in chunks of selection.STEP_SIZE (cut_utils.iterate_branches) and selected with
selection.cut_steps; one candidate per event is picked at random among those that pass
(cut_utils.random_candidate_mask, seed selection.SEED + chunk index). The selection report sums the
chunks. The fits see only the numpy array of picked masses.

--mc-match PID... (signal MC) keeps the candidates whose every daughter matches eta -> PIDs
(bkg_utils.signal_candidate_mask): -13 13, -13 13 22 or -13 -13 13 13. It is the first cut, before
TOS, the muon cuts and the candidate pick, so its input is the raw MC files (mag*/*.root), never a
skim_eta.root (src/root/skim.py), whose candidate was picked with no MC match. n_bkg, c0 and c1 are
fixed at 0 (meta 'fixed_params' of results_gauss.json and results_dcb.json, left out of the plots).
Its results_dcb.json is the --sigma-prior and --tail-prior of the data fit.

An input path with an eta2mumumumu segment (eta -> 4mu) takes selection.TOS_LINES_4MU and
selection.MUON_THRESHOLDS_4MU in place of selection.TOS_LINES and selection.MUON_THRESHOLDS.

--sigma-prior RESULTS_JSON starts the DCB sigma at the MC value and holds it under
NormalConstraint(sigma, MC value, MC variance * SIGMA_PRIOR_FACTOR**2), from the 'fit' block of an
--mc-match results_dcb.json. No Gaussian is fit.

--tail-prior RESULTS_JSON floats the DCB tails under NormalConstraint(TAIL_PARAMS, MC values, MC
covariance * TAIL_PRIOR_FACTOR**2), starting from the MC values, in data or in a --mc-match fit
(the tails of one MC sample as the prior of another). --tail-prior-factor and --sigma-prior-factor
set TAIL_PRIOR_FACTOR and SIGMA_PRIOR_FACTOR for one run; the defaults are x5, and x2
(TAIL_PRIOR_FACTOR_4MU, SIGMA_PRIOR_FACTOR_4MU) for an input path with an eta2mumumumu segment. No
hard limit on n_L, n_R unless --n-max is given.

Usage:
    python src/fit/driver_mass_fit.py INFILE --outdir OUTDIR
        [--mc-match PID [PID ...] | --sigma-prior RESULTS_JSON] [--tail-prior RESULTS_JSON]
        [--tail-prior-factor F] [--sigma-prior-factor F] [--gauss-only]
        [--max-events N] [--pnn-mu-min X]
        [--xmin 500] [--xmax 600] [--nbins 100] [--mean-init 548] [--sigma-init 5]
        [--alphaL-init 1.5] [--nL-init 2] [--alphaR-init 1.5] [--nR-init 2] [--n-max N]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

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

# --- Settings ----------------------------------------------------------------

MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit

TAIL_PRIOR_FACTOR = 5.0   # --tail-prior: MC tail-fit stdevs times this
SIGMA_PRIOR_FACTOR = 5.0  # --sigma-prior: MC sigma stdev times this
TAIL_PRIOR_FACTOR_4MU = 2.0    # eta -> 4mu, in place of TAIL_PRIOR_FACTOR
SIGMA_PRIOR_FACTOR_4MU = 2.0   # eta -> 4mu, in place of SIGMA_PRIOR_FACTOR

parser = argparse.ArgumentParser(
    description='Gaussian and DCB + Chebyshev mass fits',
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument('infile', help='Input ROOT file, or a quoted glob of them')
parser.add_argument('--outdir', required=True,
                    help='Directory for selection.json, results_*.json and *_mass_fit.png')
parser.add_argument('--mc-match', nargs='+', type=int, default=None, metavar='PID',
                    help='Signal MC: keep candidates MC-matched to eta -> these daughter '
                         'PIDs (-13 13, -13 13 22, or -13 -13 13 13) and fit with no background')
parser.add_argument('--tail-prior-factor', type=float, default=None,
                    help=f'--tail-prior stdevs times this (default: {TAIL_PRIOR_FACTOR:g}, '
                         f'{TAIL_PRIOR_FACTOR_4MU:g} for eta2mumumumu)')
parser.add_argument('--sigma-prior-factor', type=float, default=None,
                    help=f'sigma prior stdev times this (default: {SIGMA_PRIOR_FACTOR:g}, '
                         f'{SIGMA_PRIOR_FACTOR_4MU:g} for eta2mumumumu)')
parser.add_argument('--tail-prior', default=None,
                    help='results_dcb.json of a --mc-match fit whose tails are the DCB prior')
parser.add_argument('--sigma-prior', default=None,
                    help='results_dcb.json of a --mc-match fit whose sigma is the start value and '
                         'prior of the DCB sigma; no Gaussian is fit')
parser.add_argument('--gauss-only', action='store_true',
                    help='Fit the Gaussian only, no DCB')
parser.add_argument('--pnn-mu-min', type=float, default=None,
                    help="ProbNNmu threshold in place of the channel's "
                         "(selection.MUON_THRESHOLDS or MUON_THRESHOLDS_4MU)")
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
if args.mc_match and args.sigma_prior:
    parser.error('--mc-match and --sigma-prior are exclusive')
if args.gauss_only and (args.sigma_prior or args.tail_prior):
    parser.error('--gauss-only takes no prior: the Gaussian fit has none')
os.makedirs(args.outdir, exist_ok=True)
is_4mu = 'eta2mumumumu' in args.infile.split('/')
tos_lines = selection.TOS_LINES_4MU if is_4mu else selection.TOS_LINES
muon_thresholds = selection.MUON_THRESHOLDS_4MU if is_4mu else selection.MUON_THRESHOLDS
if args.pnn_mu_min is not None:
    muon_thresholds = {**muon_thresholds, 'pnn_mu_min': args.pnn_mu_min}
if args.tail_prior_factor is None:
    args.tail_prior_factor = TAIL_PRIOR_FACTOR_4MU if is_4mu else TAIL_PRIOR_FACTOR
if args.sigma_prior_factor is None:
    args.sigma_prior_factor = SIGMA_PRIOR_FACTOR_4MU if is_4mu else SIGMA_PRIOR_FACTOR

branches = selection.branches(MASS_BRANCH, tos_lines, mc_match=args.mc_match)

# --- Read and select, one chunk of selection.STEP_SIZE at a time ------------
# Per chunk: drop malformed events, [MC match], TOS per trigger level, then each muon cut
# (selection.cut_steps), every cut a per-candidate boolean mask shaped like MASS_BRANCH; then one
# candidate per event, picked at random among those kept. The reports sum over the chunks.

drop_report, triggers, cutflow, mass_chunks = {}, {}, {}, []
chunks = cut_utils.iterate_branches(args.infile, branches, selection.STEP_SIZE,
                                    entry_stop=args.max_events)
for i, tree in enumerate(chunks):
    tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
    tree, chunk_drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
    steps, chunk_triggers = selection.cut_steps(tree, MASS_BRANCH, tos_lines,
                                                mc_pids=args.mc_match,
                                                muon_thresholds=muon_thresholds)
    # TODO: investigate an eta2mumugamma cut m(mu+ mu-) < --xmin, which keeps every candidate's
    # kinematic edge m(mu+ mu- gamma) >= m(mu+ mu-) below the fit window.
    picked = cut_utils.random_candidate_mask(steps[-1][1], selection.SEED + i)
    mass_chunks.append(cut_utils.values(tree, MASS_BRANCH, picked))  # <- awkward → numpy
    drop_report = cut_utils.add_counts(drop_report, chunk_drop_report)
    triggers = cut_utils.add_counts(triggers, chunk_triggers)
    cutflow = cut_utils.add_counts(cutflow, {label: int(ak.sum(mask)) for label, mask in steps})
    print(f"[INFO] chunk {i}: {drop_report['read']} events read", file=sys.stderr)
masses = np.concatenate(mass_chunks)

print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}, "
      f"dropped {drop_report['read'] - drop_report['kept']}", file=sys.stderr)
for reason in ('dropped_no_candidates', 'dropped_idx_out_of_range',
               'dropped_daughter_length_mismatch', 'dropped_partial_candidate_branch'):
    if drop_report[reason]:
        print(f'[INFO]   {reason}: {drop_report[reason]}', file=sys.stderr)
for level, lines in triggers.items():
    for line, n in lines.items():
        if n == 0:
            print(f'[WARN] {level} TOS line {line} passes no candidate', file=sys.stderr)
cutflow = list(cutflow.items())
cutflow.append(('random candidates', len(masses)))
for label, n in cutflow:
    print(f'[INFO]   {label:24s} {n:9d}', file=sys.stderr)

selection_json = {
    'meta': {
        'input_file': args.infile,
        'mass_branch': MASS_BRANCH,
        'selection': {
            'tos': tos_lines,
            'mu_pt_min': muon_thresholds['pt_min'],
            'mu_p_min': muon_thresholds['p_min'],
            'mu_eta_range': selection.MU_ETA_RANGE,
            'pnn_mu_min': muon_thresholds['pnn_mu_min'],
            'mu_ghost_max': selection.MU_GHOST_MAX,
            'mu_ip_chi2_max': selection.MU_IP_CHI2_MAX,
            'mc_match': args.mc_match,
            'random_candidate': {'seed': selection.SEED, 'step_size': selection.STEP_SIZE},
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
# The fits of the docstring table; run_mass_fits.sh chains them into the data DCB fit.

fits = []   # (name, result, stats-box builder) of each fit, written below

# MC-matched MC is signal only: the Gaussian and the DCB have no background, n_bkg, c0 and c1 fixed
# at 0 (fit_utils.BKG_INIT).
fixed_params = {'n_bkg': 0.0, **fit_utils.BKG_INIT} if args.mc_match else None

# DCB sigma: started at and held near the MC DCB sigma (--sigma-prior, no Gaussian), or started at
# the sigma of the Gaussian fitted here.
sigma_prior = None
if args.sigma_prior:
    with open(args.sigma_prior) as f:
        mc_sigma_fit = json.load(f)['fit']
    sigma_prior = fit_utils.build_prior_from_result(mc_sigma_fit, ['sigma'],
                                                    factor=args.sigma_prior_factor)
    sigma_init = sigma_prior[0][0]
    print(f"[INFO] MC sigma (start and prior, x{args.sigma_prior_factor:g} widened stdev): "
          f"{mc_sigma_fit['parameters']['sigma']['value']:.4f}"
          f"±{mc_sigma_fit['parameters']['sigma']['error']:.4f}", file=sys.stderr)
else:
    gauss_result, m_gauss = fit_gauss(
        masses, xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
        mean_init=args.mean_init, sigma_init=args.sigma_init, fixed_params=fixed_params)
    gauss_result['meta']['input_file'] = args.infile
    if args.mc_match:
        gauss_result['meta']['fixed_params'] = fixed_params
    fits.append(('gauss', gauss_result, gauss_text))
    sigma_init = m_gauss.values['sigma']

# DCB tails: under the MC tail prior widened by TAIL_PRIOR_FACTOR, or free.
tail_init = (args.alphaL_init, args.nL_init, args.alphaR_init, args.nR_init)
tail_prior = None
if args.tail_prior:
    with open(args.tail_prior) as f:
        mc_fit = json.load(f)['fit']
    tail_prior = fit_utils.build_prior_from_result(mc_fit, TAIL_PARAMS,
                                                   factor=args.tail_prior_factor)
    tail_init = tail_prior[0]
    print(f'[INFO] MC tail shape (start and prior, x{args.tail_prior_factor:g} widened stdev): '
          + ', '.join(f"{name}={mc_fit['parameters'][name]['value']:.4f}"
                      f"±{mc_fit['parameters'][name]['error']:.4f}" for name in TAIL_PARAMS),
          file=sys.stderr)

if not args.gauss_only:
    dcb_result, m_dcb = fit_dcb(
        masses, xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
        mean_init=args.mean_init, sigma_init=sigma_init,
        alphaL_init=tail_init[0], nL_init=tail_init[1],
        alphaR_init=tail_init[2], nR_init=tail_init[3],
        tail_prior=tail_prior, sigma_prior=sigma_prior, fixed_params=fixed_params,
        limits={'nL': (0.0, args.n_max), 'nR': (0.0, args.n_max)})
    dcb_result['meta']['input_file'] = args.infile
    dcb_result['meta']['n_max'] = args.n_max
    if args.mc_match:
        dcb_result['meta']['fixed_params'] = fixed_params
    if args.sigma_prior:
        dcb_result['meta']['sigma_prior_from'] = args.sigma_prior
        dcb_result['meta']['sigma_prior_factor'] = args.sigma_prior_factor
    if args.tail_prior:
        dcb_result['meta']['tail_prior_from'] = args.tail_prior
        dcb_result['meta']['tail_prior_factor'] = args.tail_prior_factor
    fits.append(('dcb', dcb_result, dcb_text))

# --- Write -------------------------------------------------------------------

for name, result, build_text in fits:
    json_path = os.path.join(args.outdir, f'results_{name}.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[DONE] JSON saved to: {json_path}', file=sys.stderr)
    plot_fit(result, os.path.join(args.outdir, f'{name}_mass_fit.png'), build_text)
