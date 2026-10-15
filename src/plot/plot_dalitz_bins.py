"""
plot_dalitz_bins.py

Figures from one fit_dalitz_bins.py results JSON, written next to it, with the
JSON's name (without .json) as prefix:

    global_fit.png               the global DCB + Chebyshev fit (the same for
                                 every results JSON of one events file)
    <prefix>_signal_yield.png    n_sig per bin                 (grid)
    <prefix>_bkg_yield.png       n_bkg per bin                 (grid)
    <prefix>_q2_yield.png        n_sig per q^2 strip           (q2)
    <prefix>_per_bin_fits.pdf    one page per bin, in order of i then j

Usage:
    python src/plot/plot_dalitz_bins.py RESULTS.json
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics
import plot_style
import plot_utils
from plot_dcb import plot_dcb
from plot_mass_fit import build_fit_figure
from plot_phase_space import dalitz_boundary

SIGNAL_KEYS = {'mean', 'sigma', 'n_sig', 'n_bkg'}
C = plot_style.COLORS

parser = argparse.ArgumentParser(description='Figures from a fit_dalitz_bins.py results JSON')
parser.add_argument('results', help='dalitz_fit_<binning>_<signal-factor>.json')
args = parser.parse_args()

with open(args.results) as f:
    result = json.load(f)
dalitz = result['dalitz']
prefix = os.path.splitext(args.results)[0]
binning = dalitz['binning']

m12_edges = np.array(dalitz['m12_edges'])
m23_edges = np.array(dalitz['m23_edges'])
xmin, xmax, nbins = (dalitz['mass_window'][key] for key in ('xmin', 'xmax', 'nbins'))
mass_edges = np.linspace(xmin, xmax, nbins + 1)
mass_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])
mass_bin_width = (xmax - xmin) / nbins
bkg_names = fit_utils.BACKGROUND_MODELS[dalitz['background_model']].param_names
bins = {(b['i'], b['j']): b for b in dalitz['bins']}

# --- Global fit --------------------------------------------------------------

plot_dcb(result, os.path.join(os.path.dirname(os.path.abspath(args.results)), 'global_fit.png'))

# --- Yields ------------------------------------------------------------------

plot_utils.apply_font()
plt.rcParams['font.size'] = 11
boundary12, boundary23 = dalitz_boundary(kinematics.M_ETA)

if binning == 'grid':
    # One map per yield, on the Dalitz grid, with the eta's Dalitz region.
    for key, name, label in [('n_sig', 'signal', 'Fitted signal yield'),
                             ('n_bkg', 'bkg', 'Fitted background yield')]:
        yields = np.zeros((len(m12_edges) - 1, len(m23_edges) - 1))
        for (i, j), b in bins.items():
            if b['valid']:
                yields[i, j] = b[key]
        fig, ax = plt.subplots(figsize=(7, 6))
        mesh = ax.pcolormesh(m12_edges, m23_edges, yields.T, cmap='viridis', shading='auto')
        fig.colorbar(mesh, ax=ax, label=label, pad=0.13)
        ax.plot(boundary12, boundary23, color='red', lw=1, label=r'Dalitz region at $m_\eta$')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel(r'$m^2_{\gamma\mu^+}$ [GeV$^2$]')
        ax.set_ylabel(r'$m^2_{\mu^+\mu^-}$ [GeV$^2$]')
        plot_style.add_bin_index_axes(ax, m12_edges, m23_edges)
        plot_style.save_plot(fig, f'{prefix}_{name}_yield.png')
        plt.close(fig)
else:
    # Signal yield per q^2 strip; strips without a valid fit are left out.
    strips = [bins[(0, j)] for j in range(len(m23_edges) - 1) if bins[(0, j)]['valid']]
    q2_lo = np.array([b['m23_range'][0] for b in strips])
    q2_hi = np.array([b['m23_range'][1] for b in strips])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(0.5 * (q2_lo + q2_hi), [b['n_sig'] for b in strips],
                xerr=0.5 * (q2_hi - q2_lo), yerr=[b['n_sig_err'] for b in strips],
                fmt='o', color=C['data'], markersize=5, capsize=0)
    # Dimuon threshold and the eta's largest q^2.
    for q2_limit in (4 * (kinematics.M_MU / 1000.0)**2, (kinematics.M_ETA / 1000.0)**2):
        ax.axvline(q2_limit, color=C['reference'], linestyle='--', lw=1)
    ax.set_xlabel(r'$q^2 = m^2_{\mu^+\mu^-}$ [GeV$^2$]')
    ax.set_ylabel('Fitted signal yield per strip (not efficiency-corrected)')
    ax.set_ylim(bottom=0)
    plot_style.style_axes(ax)
    plot_style.save_plot(fig, f'{prefix}_q2_yield.png')
    plt.close(fig)

# --- Per-bin pages -----------------------------------------------------------

def make_build_text(n_cand):
    """Stats-box builder for a per-bin page: the plot_dcb layout without the
    tails (fixed per bin), with n_cand next to chi2/ndof."""
    def build_text(params, chi2_per_ndof):
        mean, mean_err = plot_utils.val_err(params['mean'])
        sigma, sigma_err = plot_utils.val_err(params['sigma'])
        n_sig, n_sig_err = plot_utils.val_err(params['n_sig'])
        n_bkg, n_bkg_err = plot_utils.val_err(params['n_bkg'])
        bkg_row = [(name, name, *plot_utils.val_err(params[name]), '.3f', '')
                   for name in params if name not in SIGNAL_KEYS]
        stats = [
            [(r'\mu', 'μ', mean, mean_err, '.2f', 'MeV')],
            [(r'\sigma', 'σ', sigma, sigma_err, '.2f', 'MeV')],
            [(r'n_{sig}', 'n_sig', n_sig, n_sig_err, '.0f', '')],
            [(r'n_{bkg}', 'n_bkg', n_bkg, n_bkg_err, '.0f', '')],
            bkg_row,
        ]
        extra = (
            f'$n_{{cand}}$ = {n_cand:.0f}\n$\\chi^2/ndof$ = {chi2_per_ndof:.2f}',
            f'n_cand = {n_cand:.0f}\nχ²/ndof = {chi2_per_ndof:.2f}',
        )
        return plot_style.build_stats_text(stats, extra=extra)
    return build_text


pdf_path = f'{prefix}_per_bin_fits.pdf'
with PdfPages(pdf_path) as pdf:
    for i in range(len(m12_edges) - 1):
        for j in range(len(m23_edges) - 1):
            b = bins[(i, j)]
            counts = np.array(b['counts'], dtype=float)
            title = (f'Bin ({i}, {j}): m12=[{b["m12_range"][0]:.4f}, {b["m12_range"][1]:.4f}], '
                     f'm23=[{b["m23_range"][0]:.4f}, {b["m23_range"][1]:.4f}]')

            if b['valid']:
                # Fit page: data, fit components, the factors, pulls.
                bin_sig, bin_bkg = np.array(b['bin_sig']), np.array(b['bin_bkg'])
                phase_space = {'x': dalitz['factor_masses'], 'f': b['f_bkg']}
                if dalitz['signal_factor'] != 'flat':
                    phase_space['f_sig'] = b['f_sig']
                params = {
                    'mean': {'value': b['mean'], 'error': b['mean_err']},
                    'sigma': {'value': b['sigma'], 'error': b['sigma_err']},
                    'n_sig': {'value': b['n_sig'], 'error': b['n_sig_err']},
                    'n_bkg': {'value': b['n_bkg'], 'error': b['n_bkg_err']},
                    **{name: {'value': b[name], 'error': b[f'{name}_err']} for name in bkg_names},
                }
                chi2_per_ndof = b['chi2_per_ndof'] if b['chi2_per_ndof'] is not None else np.nan
                data = {
                    'meta': result['meta'],
                    'phase_space': phase_space,
                    'fit': {'parameters': params, 'chi2_per_ndof': chi2_per_ndof},
                    'histogram': {
                        'bin_centers': mass_centers, 'bin_counts': counts,
                        'bin_errors': np.sqrt(counts), 'bin_fit': bin_sig + bin_bkg,
                        'bin_sig': bin_sig, 'bin_bkg': bin_bkg, 'bin_pulls': b['pulls'],
                    },
                }
                fig, ax, ax_pull, _, _ = build_fit_figure(
                    data, make_build_text(b['n_cand']), x_lim=(xmin, xmax), title=title)
            else:
                # Data-only page in the fit pages' style, with the reason there is no fit.
                plot_utils.apply_font()
                plt.rcParams['font.size'] = 19
                fig, (ax, ax_pull) = plt.subplots(2, 1, figsize=(12, 12), sharex=True,
                                                  gridspec_kw={'height_ratios': [3, 1]})
                ax.set_title(title)
                ax.set_xlim(xmin, xmax)
                ax.errorbar(mass_centers, counts, yerr=np.sqrt(counts), fmt='+',
                            color=C['data'], markersize=7, markeredgewidth=1.3,
                            elinewidth=1.3, capsize=0, label='Data')
                ax.axvline(x=kinematics.M_ETA, color=C['reference'], linestyle='--',
                           label='PDG Mass')
                plot_style.style_axes(ax)
                legend = ax.legend(loc='upper right', frameon=True, fancybox=False,
                                   edgecolor='none', framealpha=1)
                ax.set_ylabel(f'Candidates / ({mass_bin_width:.1f} MeV/c²)')
                if b['fitted']:
                    status = f"FIT DID NOT CONVERGE\nn_cand = {b['n_cand']:.0f}"
                else:
                    status = (f"NO FIT ATTEMPTED\n{b['n_cand']:.0f} candidates "
                              f"(threshold is {dalitz['min_candidates']})")
                status_box = ax.text(0.02, 0.97, status, transform=ax.transAxes, va='top',
                                     ha='left', family='monospace', color='red')
                plot_style.reserve_top_margin(ax, max(float(counts.max()), 1.0),
                                              artists=[status_box, legend])
                plot_style.format_y_axis(ax)
                ax_pull.text(0.5, 0.5, 'no fit', transform=ax_pull.transAxes,
                             ha='center', va='center', color='gray', fontsize=9)
                ax_pull.axhline(0, color=C['data'], lw=0.8)
                ax_pull.set_ylim(-5, 5)
                plot_style.style_axes(ax_pull, minor_y=False)
                ax_pull.set_ylabel('Pulls')
                ax_pull.set_xlabel(plot_style.decay_label(result['meta']))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
print(f'[DONE] {pdf_path}')
