"""
fit_dalitz_bins.py

Per-Dalitz-bin mass fits of eta -> mu+ mu- gamma. In each bin, the
gamma mu+ mu- mass spectrum is fit with

    n_sig * f_sig(M) * DCB(M)  +  n_bkg * f_bkg(M) * Chebyshev(M)

each shape renormalized over the mass window, where f(M) is the fraction of
the bin reachable by a gamma mu+ mu- system of mass M:

    f_bkg   kinematics.phase_space_factor   every point of the bin counts equally
    f_sig   kinematics.phase_space_factor   (--signal-factor flat), or
            kinematics.dalitz_factor        (--signal-factor dalitz): points
                                            weighted by the eta Dalitz-decay density

Steps: global DCB + Chebyshev fit of every event in the mass window (its
mean/sigma become a prior for every bin, its tails are held fixed in every
bin) -> one fit per bin -> dalitz_fit_<binning>_<signal-factor>.json.

Binnings:
    grid   14 x 14 bins over m12 in M12_RANGE and m23 in M23_RANGE
    q2     14 strips over m23 in M23_RANGE (the grid's rows), each covering
           every m12 a candidate in the mass window can have

Bins the eta itself cannot reach (signal factor 0 at m_eta) have n_sig fixed
to 0.

Usage:
    python src/fit/fit_dalitz_bins.py OUTDIR/dalitz_events.npz
        --binning grid|q2 --signal-factor flat|dalitz [--tail-prior PATH]
"""

import argparse
import json
import os
import sys

import numpy as np
from iminuit.cost import ExtendedBinnedNLL, NormalConstraint

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics
from fit_dcb import TAIL_PARAMS, fit_dcb

# --- Configuration -----------------------------------------------------------

XMIN, XMAX, MASS_NBINS = 420.0, 680.0, 80      # mass window [MeV] and its histogram bins
MEAN_INIT, SIGMA_INIT = kinematics.M_ETA, 10.0  # global-fit seeds [MeV]
BKG_MODEL = 'cheb2'

M12_RANGE = (0.0, 0.21)   # m^2(gamma mu+) range of the grid [GeV^2]
M23_RANGE = (0.0, 0.31)   # m^2(mu+ mu-) range of the grid and strips [GeV^2]
N_PER_AXIS = 14           # bins per axis

MIN_CANDIDATES = 10         # bins with this many candidates or fewer in the window are not fit
SIGNAL_PRIOR_FACTOR = 5.0   # per-bin mean/sigma prior: global-fit stdevs times this
TAIL_PRIOR_FACTOR = 5.0     # global-fit tail prior: MC tail-fit stdevs times this
TAIL_PRIOR_PATH = os.path.join(_here, '..', '..', 'out', 'eta2mumugamma', 'signal',
                               '20261006', 'results_dcb_truth_rebuilt.json')

FACTOR_GRID = np.linspace(XMIN, XMAX, 2000)   # masses the factors are evaluated at in the fit
PLOT_GRID = np.linspace(XMIN, XMAX, 261)      # masses the factors are saved at, 1 MeV apart

SIGNAL_FACTORS = {'flat': kinematics.phase_space_factor, 'dalitz': kinematics.dalitz_factor}
SIGNAL_PARAMS = ('mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR')
PRIOR_PARAMS = ('mean', 'sigma')

parser = argparse.ArgumentParser(
    description='Per-Dalitz-bin mass fits',
    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
parser.add_argument('events', help='dalitz_events.npz from driver_dalitz.py')
parser.add_argument('--binning', required=True, choices=['grid', 'q2'])
parser.add_argument('--signal-factor', required=True, choices=list(SIGNAL_FACTORS))
parser.add_argument('--tail-prior', default=TAIL_PRIOR_PATH,
                    help='Results JSON of the MC DCB fit whose tails seed the global-fit prior')
parser.add_argument('--outdir', default=None,
                    help="Output directory (default: the events file's directory)")
args = parser.parse_args()
outdir = args.outdir or os.path.dirname(os.path.abspath(args.events))
os.makedirs(outdir, exist_ok=True)

# --- Inputs ------------------------------------------------------------------

events = np.load(args.events)
m12s, m23s, masses = events['m12s'], events['m23s'], events['masses']
with open(os.path.join(os.path.dirname(os.path.abspath(args.events)), 'selection.json')) as f:
    selection = json.load(f)

# Bin edges [GeV^2]. A q2 strip spans every m12 up to (XMAX)^2, above any
# m12 a candidate with M < XMAX can have.
m23_edges = np.linspace(*M23_RANGE, N_PER_AXIS + 1)
if args.binning == 'grid':
    m12_edges = np.linspace(*M12_RANGE, N_PER_AXIS + 1)
else:
    m12_edges = np.array([0.0, (XMAX / 1000.0)**2])
mass_edges = np.linspace(XMIN, XMAX, MASS_NBINS + 1)

# --- Global fit --------------------------------------------------------------

# Tail prior: the MC fit's tail parameters and covariance, stdevs widened.
with open(args.tail_prior) as f:
    mc_fit = fit_utils.LoadedFit(json.load(f)['fit'])
tail_prior = fit_utils.build_prior(mc_fit, TAIL_PARAMS, factor=TAIL_PRIOR_FACTOR)

result, global_fit = fit_dcb(masses, xmin=XMIN, xmax=XMAX, nbins=MASS_NBINS,
                             mean_init=MEAN_INIT, sigma_init=SIGMA_INIT,
                             bkg_model=BKG_MODEL, tail_prior=tail_prior)
result['meta'].update(selection['meta'])
result['meta']['tail_prior'] = {'path': args.tail_prior, 'factor': TAIL_PRIOR_FACTOR}
result['selection_report'] = selection['selection_report']

# Per-bin seeds from the global fit, and its mean/sigma covariance (stdevs
# widened) as the per-bin prior.
global_params = result['fit']['parameters']
seed = {name: global_params[name]['value'] for name in SIGNAL_PARAMS}
_, prior_cov = fit_utils.build_prior(global_fit, PRIOR_PARAMS, factor=SIGNAL_PRIOR_FACTOR)
print(f"[INFO] global fit valid={global_fit.valid}  mean={seed['mean']:.3f}  "
      f"sigma={seed['sigma']:.3f}  n_sig={global_params['n_sig']['value']:.1f}")

# --- Per-bin fits ------------------------------------------------------------

bkg_model_base = fit_utils.BACKGROUND_MODELS[BKG_MODEL]
signal_factor = SIGNAL_FACTORS[args.signal_factor]
bins = []
for i in range(len(m12_edges) - 1):
    for j in range(len(m23_edges) - 1):
        m12_range = (m12_edges[i], m12_edges[i + 1])
        m23_range = (m23_edges[j], m23_edges[j + 1])
        in_bin = ((m12s >= m12_range[0]) & (m12s < m12_range[1])
                  & (m23s >= m23_range[0]) & (m23s < m23_range[1]))
        counts = np.histogram(masses[in_bin], bins=mass_edges)[0].astype(float)
        n_cand = counts.sum()
        entry = {
            'i': i, 'j': j,
            'm12_range': [float(v) for v in m12_range],
            'm23_range': [float(v) for v in m23_range],
            'n_cand': float(n_cand),          # candidates in the bin and the mass window
            'n_events': int(in_bin.sum()),    # candidates in the bin, any mass
            'counts': counts.astype(int).tolist(),
            'fitted': False,
            'valid': False,
        }
        bins.append(entry)
        if n_cand <= MIN_CANDIDATES:
            continue

        # Factors on FACTOR_GRID, as fit_utils's (masses, factor) weights.
        bkg_weight = (FACTOR_GRID, kinematics.phase_space_factor(FACTOR_GRID, m12_range, m23_range))
        sig_weight = (FACTOR_GRID, signal_factor(FACTOR_GRID, m12_range, m23_range))
        no_signal = bool(np.interp(kinematics.M_ETA, *sig_weight) == 0.0)

        # Extended binned likelihood of f_sig * DCB + f_bkg * Chebyshev, plus the prior.
        sig_cdf = fit_utils.weighted_dcb_cdf(sig_weight)
        bkg_model = fit_utils.weighted_bkg_model(bkg_model_base, bkg_weight)
        scaled_cdf, param_names = fit_utils._make_scaled_cdf(SIGNAL_PARAMS, sig_cdf, bkg_model,
                                                             XMIN, XMAX)
        cost = (ExtendedBinnedNLL(counts, mass_edges, scaled_cdf, name=param_names)
                + NormalConstraint(PRIOR_PARAMS, (seed['mean'], seed['sigma']), error=prior_cov))

        # Start from the global signal shape, half signal and half background,
        # and a flat Chebyshev (its [-1, 1] variable spans the bin's reachable masses).
        init = dict(seed,
                    n_sig=0.0 if no_signal else 0.5 * n_cand,
                    n_bkg=n_cand if no_signal else 0.5 * n_cand,
                    **bkg_model_base.init)
        limits = dict(sigma=(0.0, None), alphaL=(0.0, None), alphaR=(0.0, None),
                      nL=(0.0, None), nR=(0.0, None), n_sig=(0.0, None), n_bkg=(0.0, None),
                      **bkg_model_base.limits)
        # Tails held at the global fit's values; with no signal, its yield and shape too.
        fixed = list(TAIL_PARAMS)
        if no_signal:
            fixed += ['n_sig', 'mean', 'sigma']
        fit = fit_utils.run_fit(cost, init, limits, fixed=fixed)
        entry.update({'fitted': True, 'valid': bool(fit.valid), 'n_sig_fixed_zero': no_signal})
        if not fit.valid:
            continue

        # Expected counts per mass bin for each component.
        values, errors = fit.values, fit.errors
        bkg_values = [values[name] for name in bkg_model.param_names]
        bin_sig = np.diff(sig_cdf(mass_edges, XMIN, XMAX,
                                  *(values[name] for name in SIGNAL_PARAMS))) * values['n_sig']
        bin_bkg = np.diff(bkg_model.cdf(mass_edges, XMIN, XMAX, *bkg_values)) * values['n_bkg']
        bin_fit = bin_sig + bin_bkg

        # Neyman chi2 and pulls: the data as the variance, empty mass bins skipped.
        filled = counts > 0
        chi2 = float(np.sum((counts[filled] - bin_fit[filled])**2 / counts[filled]))
        ndof = int(filled.sum()) - fit.nfit
        pulls = np.where(filled, (counts - bin_fit) / np.sqrt(np.where(filled, counts, 1.0)), 0.0)

        # True where the raw Chebyshev goes negative at some reachable mass.
        reachable = bkg_weight[1] > 0
        bkg_negative = bool(np.any(bkg_model.unnorm(FACTOR_GRID[reachable], XMIN, XMAX,
                                                    *bkg_values) < 0))

        entry.update({
            'n_sig': round(float(values['n_sig']), 2),
            'n_sig_err': round(float(errors['n_sig']), 2),
            'n_bkg': round(float(values['n_bkg']), 2),
            'n_bkg_err': round(float(errors['n_bkg']), 2),
            'mean': round(float(values['mean']), 4),
            'mean_err': round(float(errors['mean']), 4),
            'sigma': round(float(values['sigma']), 4),
            'sigma_err': round(float(errors['sigma']), 4),
            **{name: round(float(values[name]), 6) for name in bkg_model.param_names},
            **{f'{name}_err': round(float(errors[name]), 6) for name in bkg_model.param_names},
            'chi2': round(chi2, 4),
            'ndof': ndof,
            'chi2_per_ndof': round(chi2 / ndof, 6) if ndof > 0 else None,
            'bkg_goes_negative': bkg_negative,
            'bin_sig': np.round(bin_sig, 4).tolist(),
            'bin_bkg': np.round(bin_bkg, 4).tolist(),
            'pulls': np.round(pulls, 4).tolist(),
            'f_sig': np.round(np.interp(PLOT_GRID, *sig_weight), 6).tolist(),
            'f_bkg': np.round(np.interp(PLOT_GRID, *bkg_weight), 6).tolist(),
        })

# --- Summary -----------------------------------------------------------------

fitted = [b for b in bins if b['fitted']]
valid = [b for b in bins if b['valid']]
chi2s = np.array([b['chi2_per_ndof'] for b in valid if b['chi2_per_ndof'] is not None])
n_sig_global = global_params['n_sig']['value']
n_bkg_global = global_params['n_bkg']['value']
n_sig_summed = float(sum(b['n_sig'] for b in valid))
n_bkg_summed = float(sum(b['n_bkg'] for b in valid))
in_grid = ((m12s >= m12_edges[0]) & (m12s < m12_edges[-1])
           & (m23s >= m23_edges[0]) & (m23s < m23_edges[-1]))
in_window = (masses >= XMIN) & (masses < XMAX)

print(f'[INFO] {args.binning} / {args.signal_factor}: {len(bins)} bins, '
      f'{len(bins) - len(fitted)} skipped, {len(fitted) - len(valid)} invalid, {len(valid)} valid, '
      f"{sum(b['n_sig_fixed_zero'] for b in fitted)} with n_sig fixed to 0, "
      f"{sum(b.get('bkg_goes_negative', False) for b in valid)} with negative background")
print(f'[INFO] n_sig global {n_sig_global:.1f}, summed over bins {n_sig_summed:.1f}; '
      f'n_bkg global {n_bkg_global:.1f}, summed {n_bkg_summed:.1f}; '
      f'{int((in_grid & in_window).sum())} of {int(in_window.sum())} window candidates in the bins')
print(f'[INFO] chi2/ndof over valid bins: median {np.median(chi2s):.3f}, max {chi2s.max():.3f}')

# --- Write -------------------------------------------------------------------

result['dalitz'] = {
    'binning': args.binning,
    'signal_factor': args.signal_factor,
    'background_factor': 'flat',
    'ff_lambda_gev': kinematics.FF_LAMBDA if args.signal_factor == 'dalitz' else None,
    'm12_edges': m12_edges.tolist(),
    'm23_edges': m23_edges.tolist(),
    'mass_window': {'xmin': XMIN, 'xmax': XMAX, 'nbins': MASS_NBINS},
    'factor_masses': PLOT_GRID.tolist(),
    'min_candidates': MIN_CANDIDATES,
    'background_model': BKG_MODEL,
    'prior': {'params': list(PRIOR_PARAMS), 'factor': SIGNAL_PRIOR_FACTOR},
    'fixed_in_per_bin_fit': list(TAIL_PARAMS),
    'n_sig_fixed_zero_rule': 'signal factor at m_eta == 0',
    'chi2_convention': 'neyman/data-based: sum (d - f)^2 / d',
    'pull_convention': 'data-based: (d - f) / sqrt(d)',
    'outcome': {'skipped': len(bins) - len(fitted), 'invalid': len(fitted) - len(valid),
                'valid': len(valid)},
    'yield_closure': {
        'n_sig_global': n_sig_global, 'n_sig_summed': n_sig_summed,
        'n_bkg_global': n_bkg_global, 'n_bkg_summed': n_bkg_summed,
        'window_candidates_in_bins': int((in_grid & in_window).sum()),
        'window_candidates': int(in_window.sum()),
    },
    'bins': bins,
}
json_path = os.path.join(outdir, f'dalitz_fit_{args.binning}_{args.signal_factor}.json')
with open(json_path, 'w') as f:
    json.dump(result, f, indent=1)
print(f'[DONE] {json_path}')
