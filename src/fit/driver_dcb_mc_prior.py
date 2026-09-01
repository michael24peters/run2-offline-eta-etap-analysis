"""
driver_dcb_mc_prior.py

Two-stage DCB fit: fit signal MC first to pin down the tail-shape parameters
(alphaL, nL, alphaR, nR) with no background component, then fit data with the
tails held fixed and quote a signal yield restricted to a core window.

    read MC root   → select → best candidate
                    → fit_dcb(fixed_params={n_bkg,c0,c1: 0}) → tail_values
    read data root → select → best candidate
                    → fit_dcb(fixed_params=dict(zip(TAIL_PARAMS, tail_values)))
                    → n_sig_report_window (this driver, via fit_utils.dcb_window_yield)
                    → results/plot

A soft NormalConstraint prior on the tails isn't enough: if MC only weakly
constrains one of the four (nR is nearly degenerate with alphaR at large n),
the data fit can drift it into the barely-decaying regime and reclassify
background as signal. Hard-fixing removes that degree of freedom outright.

The full mass window is kept for both fits -- restricting the data fit's
likelihood to a narrow core window was tried and made things worse: floating
the Chebyshev background's 2 shape parameters on a ~40 MeV slice leaves them
unconstrained (~150% relative error), and extrapolating that far outside the
fitted slice diverges badly. The full window gives the background a proper
baseline and keeps it well-constrained.

A single power-law DCB tail also can't simultaneously fit the near-tail and
deep-tail regions of the (background-free) MC -- the near tail wants a
heavier tail than the deep tail actually supports, so nL/nR end up as a
compromise that measurably over-predicts the deepest tail (~17% over MC's own
counts 17-23 sigma out). This is a real modeling limitation, not something a
parameter fix can remove. Two mitigations: fixing n_bkg/c0/c1 to 0 on the MC
fit (background shouldn't float at all in a truth-matched sample, and doing
so was additionally distorting the tail-shape estimate -- nR came back far
more precisely determined without it); and, on the data fit's result,
n_sig_report_window -- the signal yield integrated over a sane core window
(--report-halfwidth-sigma wide), decoupled from the over-extrapolating far
tail. See README.md.

Same selection flags apply to both files, since the tail-shape fix is only
valid if MC and data were selected the same way.

Usage:
    python driver_dcb_mc_prior.py mc.root data.root [--outdir out] [--max-events N]
                       [--xmin 480] [--xmax 620] [--nbins 80]
                       [--mean-init 548] [--sigma-init 5]
                       [--alphaL-init 1.5] [--nL-init 2]
                       [--alphaR-init 1.5] [--nR-init 2]
                       [--report-halfwidth-sigma 10]
"""

import sys
import os
import json
import argparse

import numpy as np
import awkward as ak

# Add the "utils", "plot" and "fit" directories to the Python path so we can
# import the project modules.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'fit'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import cut_utils
import fit_utils
import kinematics
from fit_dcb import fit_dcb, TAIL_PARAMS
from plot_dcb import plot_dcb

# --- Selection ---------------------------------------------------------------
# Edit these to change what gets fit. Applied identically to the MC file and
# the data file -- the tail prior is only meaningful if both were selected the
# same way.

MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit
METRIC_BRANCH   = 'tag_dtf_chi2'   # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']   # branches the cuts below read
PNN_MU_MIN      = 0.4              # every muon daughter must exceed this
MUON_PID        = 13               # |pid| the pnn_mu cut applies to

# Trigger-on-signal lines OR'd together by the TOS cut. These are a property of
# this ntuple, not of the framework, so they live here rather than in cut_utils.
TOS_TRIGGERS = ['tag_l0_tos0', 'tag_l0_tos1', 'tag_hlt1_tos0',
                'tag_hlt1_tos1', 'tag_hlt2_tos0', 'tag_hlt2_tos1']

BRANCHES = ([MASS_BRANCH, METRIC_BRANCH, 'prt_idx_mom']
            + DAUGHTER_FIELDS + TOS_TRIGGERS)

# MC-only branches for truth-matching. Not in BRANCHES since the data file has
# no mc_* / prt_idx_gen branches -- read_branches() would fail on the data leg
# if these were read unconditionally.
MC_TRUTH_BRANCHES = ['prt_idx_gen', 'mc_idx_mom']

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
parser.add_argument('--report-halfwidth-sigma', type=float, default=10.0,
                    help='Half-width, in units of the fitted sigma, of the core window used '
                         'to compute n_sig_report_window -- a single power-law DCB tail '
                         'over-extrapolates far from the peak (see README.md), so this is '
                         'the number to quote as the signal yield rather than n_sig itself')
args = parser.parse_args()


def select_masses(infile, tag, truth_match=False):
    """
    Read `infile`, apply the selection block above, reduce to one candidate
    per event, and return a flat numpy array of MASS_BRANCH values.

    `tag` ('mc' or 'data') only affects the [INFO] log lines, so a run mixing
    both files in one terminal is still readable.

    `truth_match` : only meaningful for the MC file (it reads mc_*/prt_idx_gen
    branches the data file doesn't have). Requires every daughter of a
    candidate to have a matched generator particle, and all of those matched
    particles to share one common generator mother.
    """
    branches = list(BRANCHES) + (MC_TRUTH_BRANCHES if truth_match else [])
    tree = cut_utils.read_branches(infile, branches, entry_stop=args.max_events)
    tree, drop_report = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)

    use_tos = not args.no_tos
    pnn_mu_min = args.pnn_mu_min

    keep = ak.ones_like(tree[MASS_BRANCH], dtype=bool)
    tos_report = None
    if use_tos:
        tos, tos_report = cut_utils.tos_mask(tree, TOS_TRIGGERS, tag_ref=MASS_BRANCH)
        keep = keep & tos
    if pnn_mu_min is not None:
        # A daughter of some other PID isn't subject to a muon cut, so it can
        # never fail it. Folding that in as "passes OR isn't a muon" turns the
        # cut into a plain all-reduction onto the candidate.
        muon_ok = ((tree['prt_pnn_mu'] > pnn_mu_min)
                   | (abs(tree['prt_pid']) != MUON_PID))
        keep = keep & cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all',
                                       target_ref=MASS_BRANCH)
    n_after_reco_cuts = int(ak.sum(keep))

    truth_report = None
    if truth_match:
        # True per daughter: its matched gen particle exists (prt_idx_gen >= 0)
        # and is a valid index into this event's mc_ collection.
        n_mc = ak.num(tree['mc_idx_mom'])
        has_gen_match = cut_utils.valid(tree, 'prt_idx_gen') & (tree['prt_idx_gen'] < n_mc)
        # True per candidate: every one of its daughters has a gen match.
        all_gen_matched = cut_utils.reduce(tree, has_gen_match, 'prt_idx_mom',
                                            how='all', target_ref=MASS_BRANCH)

        # Per daughter, the mc_idx_mom of the gen particle it matched to --
        # i.e. the generator-level mother index of that daughter's true
        # particle. follow(tree, tree['mc_idx_mom'], 'prt_idx_gen') re-indexes
        # mc_idx_mom (currently at gen-particle level) down to daughter level:
        # one value per daughter, looking up mc_idx_mom[prt_idx_gen[i]] for
        # each daughter i.
        #   e.g. prt_idx_gen = [2, 0, 1], mc_idx_mom = [-1, 5, 5]
        #     -> mom_idx_at_gen = [5, -1, 5]  (daughters 0 and 2 share mother 5)
        mom_idx_at_gen = cut_utils.follow(tree, tree['mc_idx_mom'], 'prt_idx_gen')
        # Per candidate, the smallest/largest mom_idx_at_gen among its
        # daughters. If every daughter's gen particle points to the same
        # mother, min and max both equal that shared index.
        #   e.g. daughters' mom_idx_at_gen = [5, 5, 5] -> min=max=5 (agree)
        #        daughters' mom_idx_at_gen = [5, 5, 7] -> min=5, max=7 (disagree)
        mom_min = cut_utils.reduce(tree, mom_idx_at_gen, 'prt_idx_mom',
                                    how='min', target_ref=MASS_BRANCH)
        mom_max = cut_utils.reduce(tree, mom_idx_at_gen, 'prt_idx_mom',
                                    how='max', target_ref=MASS_BRANCH)
        # True per candidate: all daughters' matched gen particles share one
        # mother (min == max), and that mother index is real (>= 0) -- rules
        # out two daughters both matching a gen particle with no mother
        # (-1 == -1), which would otherwise look like agreement.
        truth_matched = all_gen_matched & (mom_min == mom_max) & (mom_min >= 0)

        keep = keep & truth_matched
        truth_report = {
            'all_daughters_gen_matched': int(ak.sum(all_gen_matched)),
            'shared_valid_mother': int(ak.sum((mom_min == mom_max) & (mom_min >= 0))),
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
    if tos_report and tos_report['ignored']:
        print(f"[INFO] [{tag}] TOS lines absent in every event (contributing nothing): "
              f"{tos_report['ignored']}", file=sys.stderr)

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
# fixed_params holds n_bkg/c0/c1 at 0: this sample is truth-matched signal MC,
# so it shouldn't have any real background -- letting them float anyway (as a
# plain fit_dcb() call would) lets a spurious background component compete
# for events and distort the tail-shape parameters (nR in particular came out
# far less precisely determined with it floating: see README.md).
mc_masses, mc_selection_report = select_masses(args.mc_infile, 'mc',
                                                truth_match=not args.no_truth_match)
mc_result, m_mc = fit_dcb(
    mc_masses,
    xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=args.sigma_init,
    alphaL_init=args.alphaL_init, nL_init=args.nL_init,
    alphaR_init=args.alphaR_init, nR_init=args.nR_init,
    fixed_params={'n_bkg': 0.0, 'c0': 0.0, 'c1': 0.0},
)

tail_values = np.array([m_mc.values[p] for p in TAIL_PARAMS])

print(f'[INFO] MC tail shape (fixed for data fit): ' + ', '.join(
    f'{p}={v:.4f}±{m_mc.errors[p]:.4f}' for p, v in zip(TAIL_PARAMS, tail_values)
), file=sys.stderr)

mc_result['meta']['input_file'] = args.mc_infile
mc_result['meta']['selection'] = {
    'tos': not args.no_tos, 'pnn_mu_min': args.pnn_mu_min,
    'truth_match': not args.no_truth_match,
    'best_candidate': f'min {METRIC_BRANCH}',
}
mc_result['selection_report'] = mc_selection_report

mc_json_path = os.path.join(args.outdir, f'results_dcb_mc{suffix}.json')
with open(mc_json_path, 'w') as f:
    json.dump(mc_result, f, indent=2)
print(f'[DONE] MC JSON saved to: {mc_json_path}', file=sys.stderr)
plot_dcb(mc_result, {'output': os.path.join(args.outdir, f'dcb_mass_fit_mc{suffix}.png')})

# --- 2. Fit data with the MC tail shape fixed ---------------------------------
data_masses, data_selection_report = select_masses(args.data_infile, 'data')
data_result, m_data = fit_dcb(
    data_masses,
    xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=args.sigma_init,
    fixed_params=dict(zip(TAIL_PARAMS, tail_values)),
)

# n_sig_report_window: a single power-law DCB tail can't fit both the
# near-tail and deep-tail of the real shape at once (see README.md, ~17%
# over-prediction 17-23 sigma out), so quote the signal yield in a sane core
# window -- [mean - k*sigma, mean + k*sigma], from the fit's own mean/sigma --
# instead of trusting the over-extrapolated tail integral.
p_data = m_data.values
report_window = (p_data['mean'] - args.report_halfwidth_sigma * p_data['sigma'],
                  p_data['mean'] + args.report_halfwidth_sigma * p_data['sigma'])
n_sig_rw, n_sig_rw_err = fit_utils.dcb_window_yield(
    m_data.values, m_data.errors, args.xmin, args.xmax, report_window)
data_result['meta']['report_window'] = list(report_window)
data_result['fit']['parameters']['n_sig_report_window'] = {
    'value': round(float(n_sig_rw), 2),
    'error': round(float(n_sig_rw_err), 2),
    'note': 'signal yield within report_window, independent of the likelihood -- use this '
            'to quote a core-region count when the fitted tail shape over-extrapolates '
            'further out',
}

data_result['meta']['input_file'] = args.data_infile
data_result['meta']['tail_fixed_from'] = args.mc_infile
data_result['meta']['selection'] = {
    'tos': not args.no_tos, 'pnn_mu_min': args.pnn_mu_min,
    'best_candidate': f'min {METRIC_BRANCH}',
}
data_result['selection_report'] = data_selection_report

data_json_path = os.path.join(args.outdir, f'results_dcb_data{suffix}.json')
with open(data_json_path, 'w') as f:
    json.dump(data_result, f, indent=2)
print(f'[DONE] data JSON saved to: {data_json_path}', file=sys.stderr)
plot_dcb(data_result, {'output': os.path.join(args.outdir, f'dcb_mass_fit_data{suffix}.png')})
