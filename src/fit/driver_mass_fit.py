"""
driver_mass_fit.py

Gaussian and DCB + Chebyshev mass fits of one ROOT file:

    read ROOT → select → best candidate → fit_gauss → fit_dcb (seeded with the Gauss σ)
              → selection.json, results_gauss.json, gauss_mass_fit.png,
                results_dcb.json, dcb_mass_fit.png

This script owns everything about the input: which file, which branches, which
selection. fit_gauss() and fit_dcb() downstream see only a numpy array of
masses.

--truth-match PID... (signal MC only) keeps the candidates whose every daughter
matches eta -> PIDs (bkg_utils.signal_candidate_mask), before the best
candidate is chosen, and runs only the DCB fit, from --sigma-init, with n_bkg
fixed to 0 and the background shape at its init values: no Gaussian fit and
no results_gauss.json. Its results_dcb.json is the --tail-prior of the data
fit.

--tail-prior RESULTS_JSON floats the DCB tails under
NormalConstraint(TAIL_PARAMS, MC values, MC covariance * TAIL_PRIOR_FACTOR**2),
starting from the MC values.

A single power-law tail cannot describe both the near and deep tails of MC:
nL/nR are a compromise that over-predicts the deepest tail (~17% at 17-23
sigma). See README.md.

Usage:
    python src/fit/driver_mass_fit.py INFILE --outdir OUTDIR
        [--truth-match PID [PID ...] | --tail-prior RESULTS_JSON]
        [--tos-levels L0 HLT1 HLT2] [--pnn-mu-min 0.4] [--max-events N]
        [--xmin 480] [--xmax 620] [--nbins 80] [--mean-init 548] [--sigma-init 5]
        [--alphaL-init 1.5] [--nL-init 2] [--alphaR-init 1.5] [--nR-init 2]
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
METRIC_BRANCH   = 'tag_dtf_chi2'   # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']   # branches the cuts below read
PNN_MU_MIN      = 0.4              # every muon daughter must exceed this
MUON_PID        = 13               # |pid| the pnn_mu cut applies to
MOTHER_PID      = 221              # eta, for --truth-match
TRUTH_BRANCHES  = ['prt_idx_gen', 'mc_idx_mom', 'mc_pid']   # read with --truth-match

# TOS lines per trigger level; --tos-levels picks the levels to require.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}

TAIL_PRIOR_FACTOR = 5.0   # --tail-prior: MC tail-fit stdevs times this

parser = argparse.ArgumentParser(
    description='Gaussian and DCB + Chebyshev mass fits',
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument('infile', help='Input ROOT file')
parser.add_argument('--outdir', required=True,
                    help='Directory for selection.json, results_*.json and *_mass_fit.png')
parser.add_argument('--truth-match', nargs='+', type=int, default=None, metavar='PID',
                    help='Signal MC: keep candidates truth-matched to eta -> these daughter '
                         'PIDs (-13 13, or -13 13 22) and fit with no background')
parser.add_argument('--tail-prior', default=None,
                    help='results_dcb.json of a --truth-match fit whose tails are the DCB prior')
parser.add_argument('--tos-levels', nargs='*', default=list(TOS_LINES), choices=list(TOS_LINES),
                    help='Trigger levels the candidate must be TOS at (default: L0 HLT1 HLT2). '
                         'Pass none for no TOS cut')
parser.add_argument('--pnn-mu-min', type=float, default=PNN_MU_MIN,
                    help=f'Muon PID threshold (default: {PNN_MU_MIN}, the floor already '
                         'applied upstream in the ntuple, i.e. no additional cut)')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events (default: all)')
parser.add_argument('--xmin', type=float, default=480.0, help='Lower mass bound [MeV]')
parser.add_argument('--xmax', type=float, default=620.0, help='Upper mass bound [MeV]')
parser.add_argument('--nbins', type=int, default=80, help='Number of histogram bins')
parser.add_argument('--mean-init', type=float, default=kinematics.M_ETA, help='Initial mean [MeV]')
parser.add_argument('--sigma-init', type=float, default=5.0,
                    help='Initial sigma [MeV] of the Gaussian, or of the DCB with --truth-match')
parser.add_argument('--alphaL-init', type=float, default=1.5, help='Initial left tail α_L')
parser.add_argument('--nL-init', type=float, default=2.0, help='Initial left tail n_L')
parser.add_argument('--alphaR-init', type=float, default=1.5, help='Initial right tail α_R')
parser.add_argument('--nR-init', type=float, default=2.0, help='Initial right tail n_R')
args = parser.parse_args()
if args.truth_match and args.tail_prior:
    parser.error('--truth-match and --tail-prior are exclusive')
os.makedirs(args.outdir, exist_ok=True)

tos_lines = {level: TOS_LINES[level] for level in args.tos_levels}
branches = ([MASS_BRANCH, METRIC_BRANCH, 'prt_idx_mom'] + DAUGHTER_FIELDS
            + [line for lines in tos_lines.values() for line in lines]
            + (TRUTH_BRANCHES if args.truth_match else []))

# --- Read --------------------------------------------------------------------

tree = cut_utils.read_branches(args.infile, branches, entry_stop=args.max_events)
tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}, "
      f"dropped {drop_report['read'] - drop_report['kept']}", file=sys.stderr)
for reason in ('dropped_no_candidates', 'dropped_idx_out_of_range',
               'dropped_daughter_length_mismatch', 'dropped_partial_candidate_branch'):
    if drop_report[reason]:
        print(f'[INFO]   {reason}: {drop_report[reason]}', file=sys.stderr)

# --- Select ------------------------------------------------------------------
# Every cut is a per-candidate boolean mask shaped like MASS_BRANCH, so they
# combine with a plain `&` and adding one is a single line.

keep = ak.ones_like(tree[MASS_BRANCH], dtype=bool)
cutflow = [('all candidates', int(ak.sum(ak.num(keep))))]
triggers = None

# TOS on at least one line at every requested level.
if tos_lines:
    tos, tos_report = cut_utils.tos_mask(tree, tos_lines, tag_ref=MASS_BRANCH)
    keep = keep & tos
    triggers = {level: report['lines'] for level, report in tos_report.items()}
    for level, report in tos_report.items():
        cutflow.append((f'TOS up to {level}', report['tos_up_to_level']))
        for line, n in report['lines'].items():
            if n == 0:
                print(f'[WARN] {level} TOS line {line} passes no candidate', file=sys.stderr)

# ProbNNmu on every muon daughter; a non-muon daughter passes.
if args.pnn_mu_min is not None:
    muon_ok = ((tree['prt_pnn_mu'] > args.pnn_mu_min)
               | (abs(tree['prt_pid']) != MUON_PID))
    keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                   target_ref=MASS_BRANCH)
    cutflow.append((f'pnn_mu > {args.pnn_mu_min}', int(ak.sum(keep))))

# Truth match: every daughter matches eta -> args.truth_match.
if args.truth_match:
    keep = keep & bkg_utils.signal_candidate_mask(tree, MOTHER_PID, args.truth_match,
                                                  target_ref=MASS_BRANCH)
    cutflow.append(('truth-matched', int(ak.sum(keep))))

# One candidate per event: smallest METRIC_BRANCH among those kept.
best = cut_utils.best_candidate_mask(tree, METRIC_BRANCH, keep)
masses = cut_utils.values(tree, MASS_BRANCH, best)  # <- awkward → numpy
cutflow.append(('best candidates', len(masses)))
for label, n in cutflow:
    print(f'[INFO]   {label:24s} {n:9d}', file=sys.stderr)

selection = {
    'meta': {
        'input_file': args.infile,
        'mass_branch': MASS_BRANCH,
        'selection': {
            'tos': tos_lines or None,
            'pnn_mu_min': args.pnn_mu_min,
            'truth_match': args.truth_match,
            'best_candidate': f'min {METRIC_BRANCH}',
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

fits = []   # (name, result, stats-box builder) of each fit, written below

# Truth-matched MC: no background, n_bkg fixed to 0 and the background shape
# parameters at their init values (fit_utils.BKG_INIT); the DCB starts from
# --sigma-init. Otherwise the Gaussian fit, whose sigma seeds the DCB.
fixed_params = None
sigma_init = args.sigma_init
if args.truth_match:
    fixed_params = {'n_bkg': 0.0, **fit_utils.BKG_INIT}
else:
    gauss_result, m_gauss = fit_gauss(
        masses, xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
        mean_init=args.mean_init, sigma_init=args.sigma_init)
    gauss_result['meta']['input_file'] = args.infile
    fits.append(('gauss', gauss_result, gauss_text))
    sigma_init = m_gauss.values['sigma']

# DCB tails: from the MC tail prior, widened by TAIL_PRIOR_FACTOR, or free.
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
    tail_prior=tail_prior, fixed_params=fixed_params)
dcb_result['meta']['input_file'] = args.infile
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
