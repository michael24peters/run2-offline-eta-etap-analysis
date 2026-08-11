"""
driver_gauss.py

Run the Gaussian + Chebyshev mass fit on a ROOT file and produce the fit
plot in one step:

    read ROOT → select → best candidate → fit_gauss → results_gauss.json → gauss_mass_fit.png

This script owns everything about the input: which file, which branches, which
selection. fit_gauss() downstream sees only a numpy array of masses.

Usage:
    python driver_gauss.py input.root [--outdir out] [--max-events N]
                          [--xmin 480] [--xmax 620] [--nbins 80]
                          [--mean-init 548] [--sigma-init 10]
"""

import sys
import os
import json
import argparse

import awkward as ak

# Add the "utils", "plot" and "fit" directories to the Python path so we can
# import the project modules.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'fit'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import cut_utils
from fit_gauss import fit_gauss
from plot_gauss import plot_gauss

# --- Configuration ------------------------------------------------------------
# Edit these to change what gets fit. Adding a cut is one line in the block
# further down; adding a branch to cut on means listing it here too.

MASS_BRANCH     = 'tag_dtf_m'  # the quantity being fit
METRIC_BRANCH   = 'tag_dtf_chi2'  # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']  # per-daughter, need regrouping
PNN_MU_MIN      = 0.95  # every muon daughter must exceed this

BRANCHES = ([MASS_BRANCH, METRIC_BRANCH, 'prt_idx_mom']
            + DAUGHTER_FIELDS + cut_utils.TOS_TRIGGERS)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Fit and plot a Gaussian + Chebyshev mass fit in one step',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument('infile', help='Input ROOT file')
parser.add_argument('--outdir', default='out',
                    help='Output directory for results_gauss.json and gauss_mass_fit.png')
parser.add_argument('--label', default='',
                    help='Suffix for the output filenames, e.g. --label nopnn writes '
                         'results_gauss_nopnn.json / gauss_mass_fit_nopnn.png. Lets several '
                         'selections live side by side in one --outdir')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events (default: all)')
parser.add_argument('--pnn-mu-min', type=float, default=PNN_MU_MIN,
                    help=f'Muon PID threshold (default: {PNN_MU_MIN}, the floor already '
                         'applied upstream in the ntuple, i.e. no additional cut). Raise it '
                         'to tighten, e.g. --pnn-mu-min 0.95')
parser.add_argument('--no-tos', action='store_true',
                    help='Skip the TOS trigger requirement, for selection studies')
parser.add_argument('--xmin', type=float, default=480.0, help='Lower mass bound [MeV]')
parser.add_argument('--xmax', type=float, default=620.0, help='Upper mass bound [MeV]')
parser.add_argument('--nbins', type=int, default=80, help='Number of histogram bins')
parser.add_argument('--mean-init', type=float, default=547.9, help='Initial mean [MeV]')
parser.add_argument('--sigma-init', type=float, default=10.0, help='Initial sigma [MeV]')
args = parser.parse_args()

# Read the ROOT file.
tree = cut_utils.read_branches(args.infile, BRANCHES, entry_stop=args.max_events)
tree = cut_utils.drop_malformed(tree, DAUGHTER_FIELDS, cut_utils.TOS_TRIGGERS,
                                tag_ref=MASS_BRANCH)

# --- Selection ----------------------------------------------------------------
# selection_mask returns a per-candidate boolean mask shaped like MASS_BRANCH.
# Note we never index the whole tree with it -- that would silently truncate
# the prt_* fields. Either cut can be switched off for selection studies.
use_tos = not args.no_tos
pnn_mu_min = args.pnn_mu_min
keep = cut_utils.selection_mask(tree, use_tos=use_tos, pnn_mu_min=pnn_mu_min,
                                tag_ref=MASS_BRANCH)

# Reduce to one candidate per event. `keep` goes in as the eligibility mask, so
# each event keeps the best candidate that actually survived the cuts; the result
# is a mask on the same candidate axis, ready to index any per-candidate branch.
best = cut_utils.best_candidate_mask(tree[METRIC_BRANCH], keep, method='min')
masses = ak.to_numpy(ak.flatten(tree[MASS_BRANCH][best]))  # <- awkward → numpy

applied = ', '.join((['TOS'] if use_tos else [])
                    + ([f'pnn_mu>{pnn_mu_min}'] if pnn_mu_min is not None else [])) or 'none'
print(f'[INFO] cutflow: {int(ak.sum(ak.num(keep)))} candidates -> '
      f'cuts [{applied}] -> {int(ak.sum(keep))} candidates -> '
      f'{len(masses)} events with a best candidate', file=sys.stderr)

# --- Fit ---------------------------------------------------------------------
result = fit_gauss(
    masses,
    xmin=args.xmin, xmax=args.xmax, nbins=args.nbins,
    mean_init=args.mean_init, sigma_init=args.sigma_init,
)
# Where the numbers came from -- the fit itself only ever saw an array.
result['meta']['input_file'] = args.infile
result['meta']['selection'] = {
    'tos': use_tos,
    'pnn_mu_min': pnn_mu_min,
    'best_candidate': f'min {METRIC_BRANCH}',
}
# Everything the selection did, so the numbers behind this fit can be audited
# from the results file alone rather than from terminal scrollback: events read
# and why any were dropped, which trigger lines were actually available, and
# candidates surviving each cut.
result['selection_report'] = cut_utils.report()
result['selection_report']['best_candidate'] = {
    'metric': METRIC_BRANCH, 'method': 'min',
    'events_with_best_candidate': len(masses),
}
result['selection_report']['branches_read'] = BRANCHES

# Save results to JSON. --label suffixes both filenames so runs with different
# selections can share an --outdir instead of overwriting each other.
os.makedirs(args.outdir, exist_ok=True)
suffix = f'_{args.label}' if args.label else ''
json_path = os.path.join(args.outdir, f'results_gauss{suffix}.json')
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'[done] JSON saved to: {json_path}', file=sys.stderr)

# Plot the fit results from plot pipeline
plot_gauss(result, {'output': os.path.join(args.outdir, f'gauss_mass_fit{suffix}.png')})
