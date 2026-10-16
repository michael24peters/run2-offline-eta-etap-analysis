"""
driver_dalitz_fit.py

Per-Dalitz-bin mass fits of eta -> mu+ mu- gamma, from the candidates
driver_dalitz.py wrote:

    dalitz_events.npz -> N_PER_AXIS x N_PER_AXIS grid over M12_RANGE x M23_RANGE
                      -> fit_dalitz (global fit, then one fit per bin)
                      -> dalitz_fit_<factor>.json

    <factor>  none    no factor on either shape (default)
              flat    --phase-space: kinematics.phase_space_factor on the
                      signal and the background
              dalitz  --phase-space --dalitz-factor: kinematics.dalitz_factor
                      on the signal, kinematics.phase_space_factor on the
                      background

The global fit carries the tail prior of an MC DCB fit (--tail-prior), its
stdevs times TAIL_PRIOR_FACTOR.

Usage:
    python src/fit/driver_dalitz_fit.py OUTDIR/dalitz_events.npz
        [--phase-space [--dalitz-factor]] [--tail-prior PATH] [--outdir DIR]
"""

import argparse
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics
from fit_dalitz import fit_dalitz, hole_neighbors
from fit_dcb import TAIL_PARAMS

# --- Configuration -----------------------------------------------------------

XMIN, XMAX, MASS_NBINS = 420.0, 680.0, 80      # mass window [MeV] and its histogram bins
MEAN_INIT, SIGMA_INIT = kinematics.M_ETA, 10.0  # global-fit seeds [MeV]

M12_RANGE = (0.0, 0.21)   # m^2(gamma mu+) range of the grid [GeV^2]
M23_RANGE = (0.0, 0.31)   # m^2(mu+ mu-) range of the grid [GeV^2]
N_PER_AXIS = 14           # bins per axis

MIN_CANDIDATES = 10         # bins with this many candidates or fewer in the window are not fit
SIGNAL_PRIOR_FACTOR = 5.0   # per-bin mean/sigma prior: global-fit stdevs times this
TAIL_PRIOR_FACTOR = 5.0     # global-fit tail prior: MC tail-fit stdevs times this
TAIL_PRIOR_PATH = os.path.join(_here, '..', '..', 'out', 'eta2mumugamma', 'signal',
                               '20261006', 'results_dcb_truth_rebuilt.json')

parser = argparse.ArgumentParser(
    description='Per-Dalitz-bin mass fits',
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument('events', help='dalitz_events.npz from driver_dalitz.py')
parser.add_argument('--phase-space', action='store_true',
                    help='Multiply signal and background by the phase-space factor of each bin')
parser.add_argument('--dalitz-factor', action='store_true',
                    help='With --phase-space: the Dalitz factor instead on the signal')
parser.add_argument('--tail-prior', default=TAIL_PRIOR_PATH,
                    help='Results JSON of the MC DCB fit whose tails seed the global-fit prior')
parser.add_argument('--outdir', default=None,
                    help="Output directory (default: the events file's directory)")
args = parser.parse_args()
if args.dalitz_factor and not args.phase_space:
    parser.error('--dalitz-factor requires --phase-space')
outdir = args.outdir or os.path.dirname(os.path.abspath(args.events))
os.makedirs(outdir, exist_ok=True)

signal_factor = bkg_factor = None
factor_label = 'none'
if args.phase_space:
    signal_factor = bkg_factor = kinematics.phase_space_factor
    factor_label = 'flat'
if args.dalitz_factor:
    signal_factor = kinematics.dalitz_factor
    factor_label = 'dalitz'

# --- Inputs ------------------------------------------------------------------

events = np.load(args.events)
m12s, m23s, masses = events['m12s'], events['m23s'], events['masses']
with open(os.path.join(os.path.dirname(os.path.abspath(args.events)), 'selection.json')) as f:
    selection = json.load(f)

m12_edges = np.linspace(*M12_RANGE, N_PER_AXIS + 1)   # [GeV^2]
m23_edges = np.linspace(*M23_RANGE, N_PER_AXIS + 1)   # [GeV^2]

# Tail prior: the MC fit's tail parameters and covariance, stdevs widened
with open(args.tail_prior) as f:
    mc_fit = fit_utils.LoadedFit(json.load(f)['fit'])
tail_prior = fit_utils.build_prior(mc_fit, TAIL_PARAMS, factor=TAIL_PRIOR_FACTOR)

# --- Fit ---------------------------------------------------------------------

result, global_fit = fit_dalitz(m12s, m23s, masses, m12_edges, m23_edges,
                                xmin=XMIN, xmax=XMAX, nbins=MASS_NBINS,
                                mean_init=MEAN_INIT, sigma_init=SIGMA_INIT,
                                tail_prior=tail_prior, signal_prior_factor=SIGNAL_PRIOR_FACTOR,
                                min_candidates=MIN_CANDIDATES,
                                signal_factor=signal_factor, bkg_factor=bkg_factor)
result['meta'].update(selection['meta'])
result['meta']['tail_prior'] = {'path': args.tail_prior, 'factor': TAIL_PRIOR_FACTOR}
result['selection_report'] = selection['selection_report']
dalitz = result['dalitz']

# --- Summary -----------------------------------------------------------------

# Global fit: MIGRAD summary and every parameter
print(global_fit.fmin)
for name in global_fit.parameters:
    print(f'  {name:8s} value={global_fit.values[name]:12.4f}  '
          f'error={global_fit.errors[name]:10.4f}  limits={global_fit.limits[name]}')
params = result['fit']['parameters']
print(f"[INFO] per-bin seeds: mean={params['mean']['value']:.3f}  "
      f"sigma={params['sigma']['value']:.3f}; fixed tails: "
      + '  '.join(f"{name}={params[name]['value']:.3f}" for name in TAIL_PARAMS))

# Per-bin outcome
bins = dalitz['bins']
fitted = [b for b in bins if b['fitted']]
valid = [b for b in bins if b['valid']]
outcome = dalitz['outcome']
print(f"[INFO] {factor_label}: {len(bins)} bins, {outcome['skipped']} skipped "
      f"(<= {MIN_CANDIDATES} candidates), {outcome['invalid']} invalid, {outcome['valid']} valid, "
      f"{sum(b['n_sig_fixed_zero'] for b in fitted)} with n_sig fixed to 0")

# Invalid fits, with the candidate counts of their fitted neighbors
for i, j, n_cand, neighbors in hole_neighbors(result):
    neighbor_text = ', '.join(f'{offset}: {count:.0f}' for offset, count in neighbors.items())
    print(f'[INFO]   invalid ({i:2d},{j:2d}): n_cand={n_cand:.0f}  neighbors: {neighbor_text}')

# Valid fits whose Chebyshev goes below 0 where the background is reachable
negative = [b for b in valid if b['bkg_goes_negative']]
print(f'[INFO] {len(negative)} of {len(valid)} valid fits with a negative background')
for b in negative:
    print(f"[INFO]   ({b['i']:2d},{b['j']:2d}): "
          + '  '.join(f'{name}={b[name]:.3f}' for name in fit_utils.BKG_PARAMS))

# Summed per-bin yields vs the global fit
closure = dalitz['yield_closure']
print(f"[INFO] n_sig global {closure['n_sig_global']:.1f}, summed over bins "
      f"{closure['n_sig_summed']:.1f}; n_bkg global {closure['n_bkg_global']:.1f}, summed "
      f"{closure['n_bkg_summed']:.1f}")
print(f"[INFO] {closure['window_candidates_in_bins']} of {closure['window_candidates']} window "
      f"candidates in the bins; {closure['candidates_in_grid']} of {closure['candidates']} "
      f"candidates (any mass) in the grid")
chi2s = np.array([b['chi2_per_ndof'] for b in valid if b['chi2_per_ndof'] is not None])
if chi2s.size:
    print(f'[INFO] chi2/ndof over valid bins: median {np.median(chi2s):.3f}, '
          f'max {chi2s.max():.3f}')

# --- Write -------------------------------------------------------------------

json_path = os.path.join(outdir, f'dalitz_fit_{factor_label}.json')
with open(json_path, 'w') as f:
    json.dump(result, f, indent=1)
print(f'[DONE] {json_path}')
