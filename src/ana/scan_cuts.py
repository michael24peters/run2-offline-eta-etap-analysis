"""
scan_cuts.py

Stepwise scan of offline cuts by the figure of merit F = S / sqrt(S + B) of the data mass fit:

    for each step (one quantity):
        one pass over the data and one over the MC: per point (the step's cuts, and 'off'), the
        mass of the random candidate per event among those passing [MC match], TOS, the cuts
        chosen so far and the point's cut
        per point: MC Gaussian -> MC DCB -> data DCB + Chebyshev with the MC sigma and tails as
        priors (the chain of run_mass_fits.sh), and F
        the point with the highest F whose MC DCB and data DCB fits are valid joins every later
        step; 'off' adds no cut. The MC Gaussian only starts the MC DCB sigma, so its validity
        does not count (a signal-only Gaussian does not describe the tails).

--channel eta2mumu scans MUON_STEPS (cuts on every muon) on top of FIXED_MUON_CUTS. --channel
eta2mumugamma scans PHOTON_STEPS (cuts on every photon) on top of the muon cuts of --muon-cuts.
Each cut is a string such as 'pnn_mu > 0.9' or '0 <= ip_chi2 < 6' (selection.parse_cut). The
inputs, MC (MC-matched, the first cut), windows, bins and Gaussian sigma start of each channel are
in CHANNELS.

Each pass reads the branches of driver_mass_fit.py (selection.branches) in chunks of
selection.STEP_SIZE, the random candidate of chunk i picked with seed selection.SEED + i, so a
point with the selection of a driver_mass_fit.py run gives that run's masses and fits. Only the
masses in the fit window are kept; the fits histogram only the window.

S and B are the fitted signal and background in mean +- SR_SIGMAS sigma of the data DCB fit
(fit_utils.dcb_cdf, cheb_cdf).

Writes into --outdir:
    <NN>_<quantity>/<point>/results_gauss_mc.json, results_dcb_mc.json, results_dcb.json
    summary.txt         per step, per point: F, S, B, n_sig, n_bkg, chi2/ndf (data), MC sigma,
                        validity of the MC DCB and data DCB fits; the chosen point marked *; the
                        final cut lists
    muon_cuts.txt       the final muon cuts, one per line
    photon_cuts.txt     the final photon cuts, one per line

Usage:
    python src/ana/scan_cuts.py --channel eta2mumu --outdir OUTDIR [--max-events N]
        [--max-points N]
    python src/ana/scan_cuts.py --channel eta2mumugamma --muon-cuts CUT [CUT ...] --outdir OUTDIR
        [--max-events N] [--max-points N]
"""

import argparse
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'fit'))
sys.path.insert(0, os.path.join(_here, '..', 'plot'))
import cut_utils
import fit_utils
import selection
from fit_dcb import TAIL_PARAMS, fit_dcb
from fit_gauss import fit_gauss
from plot_cut_steps import slug

REPO = os.path.normpath(os.path.join(_here, '..', '..'))

# --- Settings ----------------------------------------------------------------

MASS_BRANCH = 'tag_dtf_m'
SIGMA_PRIOR_FACTOR = 5.0   # MC sigma stdev times this, as driver_mass_fit.py
TAIL_PRIOR_FACTOR = 5.0    # MC tail stdevs times this, as driver_mass_fit.py
SR_SIGMAS = 2.0            # signal region of F: mean +- SR_SIGMAS sigma of the data DCB fit

# Inputs (relative to the repository), MC-matched daughter PIDs, fit windows (xmin, xmax, nbins)
# [MeV] and Gaussian sigma start [MeV] of each channel, as run_mass_fits.sh; eta2mumugamma MC: phsp
# (generator photon pT > 0.5 GeV)
CHANNELS = {
    'eta2mumu': {
        'data': 'ntuple/lhcb/eta2mumu/20261109/mag*/*.root',
        'mc': 'ntuple/mc/eta2mumu/signal_20261109/mag*/*.root',
        'mc_pids': [-13, 13],
        'mc_window': (500.0, 600.0, 100),
        'data_window': (535.0, 560.0, 100),
        'sigma_init': 5.0,
    },
    'eta2mumugamma': {
        'data': 'ntuple/lhcb/eta2mumugamma/20261109/mag*/*.root',
        'mc': 'ntuple/mc/eta2mumugamma/signal_20261109/phsp/mag*/*.root',
        'mc_pids': [-13, 13, 22],
        'mc_window': (450.0, 650.0, 200),
        'data_window': (450.0, 650.0, 200),
        'sigma_init': 13.0,
    },
}

# eta -> mu+ mu-: muon cuts at every point (acceptance, IsMuonTight), then one step per quantity,
# in order. The lowest thresholds are those of the trigger or the ntuple (ProbNNmu 0.8 in HLT2,
# p 10 GeV and pT 0.5 GeV in HLT1).
FIXED_MUON_CUTS = ['2 < eta < 4.5', 'tight_mu > 0.5']
MUON_STEPS = [
    ['pnn_mu > 0.8', 'pnn_mu > 0.85', 'pnn_mu > 0.9', 'pnn_mu > 0.925', 'pnn_mu > 0.95',
     'pnn_mu > 0.97', 'pnn_mu > 0.98', 'pnn_mu > 0.99'],
    ['0 <= ip_chi2 < 2', '0 <= ip_chi2 < 3', '0 <= ip_chi2 < 4', '0 <= ip_chi2 < 5',
     '0 <= ip_chi2 < 6', '0 <= ip_chi2 < 7', '0 <= ip_chi2 < 8', '0 <= ip_chi2 < 9',
     '0 <= ip_chi2 < 10'],
    ['pt > 600', 'pt > 750', 'pt > 1000', 'pt > 1250', 'pt > 1500', 'pt > 1750', 'pt > 2000'],
    ['p > 10000', 'p > 11000', 'p > 12500', 'p > 14000', 'p > 15000', 'p > 16000', 'p > 17500',
     'p > 20000'],
    ['prb_ghost < 0.2', 'prb_ghost < 0.1', 'prb_ghost < 0.05', 'prb_ghost < 0.03',
     'prb_ghost < 0.02', 'prb_ghost < 0.01'],
]
# eta -> mu+ mu- gamma: one step per photon quantity, in order (ntuple: photon pT > 500 MeV)
PHOTON_STEPS = [
    ['pt > 500', 'pt > 600', 'pt > 650', 'pt > 750', 'pt > 850', 'pt > 900', 'pt > 1000',
     'pt > 1250', 'pt > 1500', 'pt > 1750', 'pt > 2000'],
    ['p > 3000', 'p > 5000', 'p > 7500', 'p > 10000', 'p > 15000', 'p > 20000'],
    ['2 < eta < 4.5'],
]


def picked_masses(infile, mc_pids, muon_cuts, photon_cuts, points, pid, window, max_events):
    """
    One pass over infile. Per point (a cut on every daughter with |pid| == pid, or None for no
    cut), the DTF mass of the candidate picked at random per event among those passing [the MC
    match to mc_pids], TOS, muon_cuts, photon_cuts and the point's cut, for the picks in `window`
    (xmin, xmax, nbins).

    Returns {point: numpy array of masses}.
    """
    xmin, xmax, _ = window
    in_window = f'{xmin:g} <= {MASS_BRANCH} <= {xmax:g}'   # the range np.histogram counts
    masses = {point: [] for point in points}
    chunks = cut_utils.iterate_branches(
        infile, selection.branches(MASS_BRANCH, selection.TOS_LINES, mc_match=bool(mc_pids)),
        selection.STEP_SIZE, entry_stop=max_events)
    for i, tree in enumerate(chunks):
        tree['prt_tight_mu'] = cut_utils.align_muons(tree, 'prt_tight_mu')   # stored for muons only
        tree, _ = cut_utils.drop_malformed(tree, tag_ref=MASS_BRANCH)
        steps, _ = selection.cut_steps(tree, MASS_BRANCH, selection.TOS_LINES, mc_pids=mc_pids,
                                       muon_cuts=muon_cuts, photon_cuts=photon_cuts)
        for point in points:
            keep = steps[-1][1]
            if point is not None:
                daughter_ok = selection.daughter_masks(tree, [point], pid)[point]
                keep = keep & cut_utils.reduce(tree, daughter_ok, 'prt_idx_mom', how='all',
                                               target_ref=MASS_BRANCH)
            picked = cut_utils.random_candidate_mask(keep, selection.SEED + i)
            values = cut_utils.values(tree, MASS_BRANCH, picked)
            masses[point].append(values[selection.cut_mask(values, in_window)])
        print(f'[INFO] {infile}: chunk {i} done', file=sys.stderr)
    return {point: np.concatenate(chunk_masses) for point, chunk_masses in masses.items()}


def fit_chain(mc_masses, data_masses, channel):
    """
    The fits of run_mass_fits.sh on one point: MC Gaussian (no background, sigma started at the
    channel's sigma_init) -> MC DCB (no background, sigma started at the Gaussian sigma) -> data
    DCB + Chebyshev (sigma and tails started at the MC DCB values and held under NormalConstraint
    priors with the MC stdevs times SIGMA_PRIOR_FACTOR and TAIL_PRIOR_FACTOR).

    Returns {'gauss_mc', 'dcb_mc', 'dcb'}: the result dicts of the three fits.
    """
    settings = CHANNELS[channel]
    mc_xmin, mc_xmax, mc_nbins = settings['mc_window']
    xmin, xmax, nbins = settings['data_window']
    no_background = {'n_bkg': 0.0, **fit_utils.BKG_INIT}

    gauss_mc, m_gauss = fit_gauss(mc_masses, xmin=mc_xmin, xmax=mc_xmax, nbins=mc_nbins,
                                  sigma_init=settings['sigma_init'], fixed_params=no_background)
    dcb_mc, _ = fit_dcb(mc_masses, xmin=mc_xmin, xmax=mc_xmax, nbins=mc_nbins,
                        sigma_init=m_gauss.values['sigma'], fixed_params=no_background)

    sigma_prior = fit_utils.build_prior_from_result(dcb_mc['fit'], ['sigma'],
                                                    factor=SIGMA_PRIOR_FACTOR)
    tail_prior = fit_utils.build_prior_from_result(dcb_mc['fit'], TAIL_PARAMS,
                                                   factor=TAIL_PRIOR_FACTOR)
    alphaL, nL, alphaR, nR = tail_prior[0]
    dcb, _ = fit_dcb(data_masses, xmin=xmin, xmax=xmax, nbins=nbins,
                     sigma_init=sigma_prior[0][0], alphaL_init=alphaL, nL_init=nL,
                     alphaR_init=alphaR, nR_init=nR, tail_prior=tail_prior,
                     sigma_prior=sigma_prior)
    for result in (gauss_mc, dcb_mc):
        result['meta']['fixed_params'] = no_background
    return {'gauss_mc': gauss_mc, 'dcb_mc': dcb_mc, 'dcb': dcb}


def figure_of_merit(dcb):
    """
    (F, n_sig_sr, n_bkg_sr) of a data DCB + Chebyshev result: the fitted signal and background in
    mean +- SR_SIGMAS sigma (inside the fit window), F = n_sig_sr / sqrt(n_sig_sr + n_bkg_sr).
    """
    values = {name: parameter['value'] for name, parameter in dcb['fit']['parameters'].items()}
    xmin, xmax = dcb['meta']['xmin'], dcb['meta']['xmax']
    edges = np.array([max(xmin, values['mean'] - SR_SIGMAS * values['sigma']),
                      min(xmax, values['mean'] + SR_SIGMAS * values['sigma'])])
    signal_cdf = fit_utils.dcb_cdf(edges, xmin, xmax, values['mean'], values['sigma'],
                                   values['alphaL'], values['nL'], values['alphaR'], values['nR'])
    background_cdf = fit_utils.cheb_cdf(edges, xmin, xmax, values['c0'], values['c1'])
    n_sig_sr = values['n_sig'] * (signal_cdf[1] - signal_cdf[0])
    n_bkg_sr = values['n_bkg'] * (background_cdf[1] - background_cdf[0])
    return n_sig_sr / np.sqrt(n_sig_sr + n_bkg_sr), n_sig_sr, n_bkg_sr


def summary_row(row, chosen):
    """One summary.txt line of a point's row {'point', 'fits', 'fom', 'n_sig_sr', 'n_bkg_sr',
    'valid'}, marked * if chosen."""
    data_fit = row['fits']['dcb']['fit']
    n_sig = data_fit['parameters']['n_sig']
    return (f"{'*' if chosen else ' '} {row['point'] or 'off':20s} {row['fom']:10.2f} "
            f"{row['n_sig_sr']:12.0f} {row['n_bkg_sr']:12.0f} {n_sig['value']:12.0f} "
            f"{n_sig['error']:9.0f} {data_fit['parameters']['n_bkg']['value']:12.0f} "
            f"{data_fit['chi2_per_ndof']:8.3f} "
            f"{row['fits']['dcb_mc']['fit']['parameters']['sigma']['value']:8.3f} "
            f"{row['valid']}")


def write_lines(path, lines):
    """Write `lines` to `path`, one per line."""
    with open(path, 'w') as f:
        f.write(''.join(f'{line}\n' for line in lines))


def main():
    parser = argparse.ArgumentParser(
        description='Stepwise scan of offline cuts by S / sqrt(S + B) of the data mass fit',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--channel', required=True, choices=list(CHANNELS))
    parser.add_argument('--outdir', required=True, help='Directory for the results')
    parser.add_argument('--muon-cuts', nargs='+', default=None, metavar='CUT',
                        help='eta2mumugamma: the muon cuts of every point (the eta2mumu scan '
                             'result)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Only read the first N events of each input (default: all)')
    parser.add_argument('--max-points', type=int, default=None,
                        help="Only the first N cuts of each step, besides 'off' (default: all)")
    args = parser.parse_args()
    if args.channel == 'eta2mumugamma' and args.muon_cuts is None:
        parser.error('--channel eta2mumugamma needs --muon-cuts')
    os.makedirs(args.outdir, exist_ok=True)
    settings = CHANNELS[args.channel]

    # eta2mumu: muon steps on top of FIXED_MUON_CUTS; eta2mumugamma: photon steps on top of
    # --muon-cuts
    if args.channel == 'eta2mumu':
        muon_cuts, photon_cuts = list(FIXED_MUON_CUTS), []
        scan_steps, pid = MUON_STEPS, selection.MUON_PID
    else:
        muon_cuts, photon_cuts = list(args.muon_cuts), []
        scan_steps, pid = PHOTON_STEPS, selection.PHOTON_PID
    data_path = os.path.join(REPO, settings['data'])
    mc_path = os.path.join(REPO, settings['mc'])

    header = (f"  {'cut':20s} {'F':>10s} {'S':>12s} {'B':>12s} {'n_sig':>12s} {'error':>9s} "
              f"{'n_bkg':>12s} {'chi2/ndf':>8s} {'MC sigma':>8s} valid")
    lines = [f'{args.channel}: S, B in mean +- {SR_SIGMAS:g} sigma of the data DCB fit; '
             f'F = S / sqrt(S + B); * = chosen', '']
    for step, cuts in enumerate(scan_steps, start=1):
        points = [None] + cuts[:args.max_points]
        name = selection.parse_cut(cuts[0])[0][0]
        step_dir = os.path.join(args.outdir, f'{step:02d}_{name}')
        print(f'[INFO] step {step}: {name}, on top of {muon_cuts + photon_cuts}', file=sys.stderr)

        # --- Masses of every point: one pass over the MC, one over the data ---
        mc_masses = picked_masses(mc_path, settings['mc_pids'], muon_cuts, photon_cuts, points,
                                  pid, settings['mc_window'], args.max_events)
        data_masses = picked_masses(data_path, None, muon_cuts, photon_cuts, points, pid,
                                    settings['data_window'], args.max_events)

        # --- Fit chain and F of every point; write the fit JSONs ---
        rows = []
        for point in points:
            fits = fit_chain(mc_masses[point], data_masses[point], args.channel)
            fom, n_sig_sr, n_bkg_sr = figure_of_merit(fits['dcb'])
            valid = fits['dcb_mc']['fit']['valid'] and fits['dcb']['fit']['valid']
            point_cuts = [point] if point else []
            if pid == selection.MUON_PID:
                point_muon_cuts, point_photon_cuts = muon_cuts + point_cuts, photon_cuts
            else:
                point_muon_cuts, point_photon_cuts = muon_cuts, photon_cuts + point_cuts
            point_dir = os.path.join(step_dir, slug(point) if point else 'off')
            os.makedirs(point_dir, exist_ok=True)
            for fit, result in fits.items():
                result['meta']['input_file'] = mc_path if fit.endswith('_mc') else data_path
                result['meta']['muon_cuts'] = point_muon_cuts
                result['meta']['photon_cuts'] = point_photon_cuts
                with open(os.path.join(point_dir, f'results_{fit}.json'), 'w') as f:
                    json.dump(result, f, indent=2)
            rows.append({'point': point, 'fits': fits, 'fom': fom, 'n_sig_sr': n_sig_sr,
                         'n_bkg_sr': n_bkg_sr, 'valid': valid})
            print(f"[INFO] {point or 'off'}: F = {fom:.2f}, valid {valid}", file=sys.stderr)

        # --- Choose the valid point of highest F; 'off' (listed first) wins ties ---
        best = None
        for row in rows:
            if row['valid'] and (best is None or row['fom'] > best['fom']):
                best = row
        if best is None:
            print(f'[WARN] step {step}: no point with valid MC and data DCB fits, no cut',
                  file=sys.stderr)
        elif best['point'] is not None:
            if pid == selection.MUON_PID:
                muon_cuts.append(best['point'])
            else:
                photon_cuts.append(best['point'])

        lines += [f'step {step}: {name}', header]
        lines += [summary_row(row, row is best) for row in rows]
        lines.append('')
        write_lines(os.path.join(args.outdir, 'summary.txt'), lines)

    lines += ['muon cuts:'] + [f'  {cut}' for cut in muon_cuts]
    lines += ['photon cuts:'] + [f'  {cut}' for cut in photon_cuts]
    write_lines(os.path.join(args.outdir, 'summary.txt'), lines)
    write_lines(os.path.join(args.outdir, 'muon_cuts.txt'), muon_cuts)
    write_lines(os.path.join(args.outdir, 'photon_cuts.txt'), photon_cuts)
    print(f'[DONE] {args.outdir}: summary.txt, muon_cuts.txt, photon_cuts.txt', file=sys.stderr)


if __name__ == '__main__':
    main()
