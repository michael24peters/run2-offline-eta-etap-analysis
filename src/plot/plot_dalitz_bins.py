"""
plot_dalitz_bins.py

Figures from one driver_dalitz_fit.py result, written with `prefix` (the
results JSON's path without .json):

    global_fit.png                      the global DCB + Chebyshev fit, next to
                                        the results JSON (the same for every
                                        results JSON of one events file)
    <prefix>_signal_yield.png           n_sig per bin
    <prefix>_bkg_yield.png              n_bkg per bin
    <prefix>_per_bin_fits.pdf           one page per bin, i (m12) outer, j (m23) inner
    <prefix>_per_bin_fits_horizontal.pdf  the same pages, j outer, i inner
                                        (horizontal=True / --horizontal)

plot_dalitz_fit(result, prefix, horizontal=False) writes all of them;
driver_dalitz_fit.py calls it after the fit.

Usage:
    python src/plot/plot_dalitz_bins.py RESULTS.json [--horizontal]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics
import plot_style
import plot_utils
from plot_mass_fit import build_fit_figure, dcb_text, plot_fit
from plot_phase_space import dalitz_boundary

SIGNAL_KEYS = {'mean', 'sigma', 'n_sig', 'n_bkg'}
C = plot_style.COLORS


def mass_binning(dalitz):
    """(edges, centers, width) [MeV] of the per-bin mass histograms."""
    xmin, xmax, nbins = (dalitz['mass_window'][key] for key in ('xmin', 'xmax', 'nbins'))
    edges = np.linspace(xmin, xmax, nbins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:]), (xmax - xmin) / nbins


def plot_yield_maps(result, prefix):
    """<prefix>_signal_yield.png and <prefix>_bkg_yield.png: the fitted n_sig
    and n_bkg of each valid bin on the Dalitz grid, with the eta's Dalitz
    region. Bins without a valid fit are 0."""
    dalitz = result['dalitz']
    m12_edges = np.array(dalitz['m12_edges'])
    m23_edges = np.array(dalitz['m23_edges'])
    boundary12, boundary23 = dalitz_boundary(kinematics.M_ETA)
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 11
    for key, name, label in [('n_sig', 'signal', 'Fitted signal yield'),
                             ('n_bkg', 'bkg', 'Fitted background yield')]:
        yields = np.zeros((len(m12_edges) - 1, len(m23_edges) - 1))
        for b in dalitz['bins']:
            if b['valid']:
                yields[b['i'], b['j']] = b[key]
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


def make_build_text(n_cand):
    """Stats-box builder for a per-bin page: the dcb_text layout without the
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


def bin_page(result, b):
    """The figure of one bin's page: data, fit components, the bin's factors
    and pulls for a valid fit; otherwise the data and the reason there is no
    fit."""
    dalitz = result['dalitz']
    mass_edges, mass_centers, mass_bin_width = mass_binning(dalitz)
    xmin, xmax = mass_edges[0], mass_edges[-1]
    counts = np.array(b['counts'], dtype=float)
    title = (f"Bin ({b['i']}, {b['j']}): m12=[{b['m12_range'][0]:.4f}, {b['m12_range'][1]:.4f}], "
             f"m23=[{b['m23_range'][0]:.4f}, {b['m23_range'][1]:.4f}]")

    if b['valid']:
        bin_sig, bin_bkg = np.array(b['bin_sig']), np.array(b['bin_bkg'])
        params = {
            'mean': {'value': b['mean'], 'error': b['mean_err']},
            'sigma': {'value': b['sigma'], 'error': b['sigma_err']},
            'n_sig': {'value': b['n_sig'], 'error': b['n_sig_err']},
            'n_bkg': {'value': b['n_bkg'], 'error': b['n_bkg_err']},
            **{name: {'value': b[name], 'error': b[f'{name}_err']}
               for name in fit_utils.BKG_PARAMS},
        }
        chi2_per_ndof = b['chi2_per_ndof'] if b['chi2_per_ndof'] is not None else np.nan
        data = {
            'meta': result['meta'],
            'fit': {'parameters': params, 'chi2_per_ndof': chi2_per_ndof},
            'histogram': {
                'bin_centers': mass_centers, 'bin_counts': counts,
                'bin_errors': np.sqrt(counts), 'bin_fit': bin_sig + bin_bkg,
                'bin_sig': bin_sig, 'bin_bkg': bin_bkg, 'bin_pulls': b['pulls'],
            },
        }
        # The bin's factors, when the fit used them
        if 'f_bkg' in b:
            data['phase_space'] = {'x': dalitz['factor_masses'], 'f': b['f_bkg']}
            if dalitz['signal_factor'] != dalitz['bkg_factor']:
                data['phase_space']['f_sig'] = b['f_sig']
        fig, _ = build_fit_figure(data, make_build_text(b['n_cand']),
                                  x_lim=(xmin, xmax), title=title)
        return fig

    # Data-only page in the fit pages' style, with the reason there is no fit
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 19
    fig, (ax, ax_pull) = plt.subplots(2, 1, figsize=(12, 12), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1]})
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.errorbar(mass_centers, counts, yerr=np.sqrt(counts), fmt='+',
                color=C['data'], markersize=7, markeredgewidth=1.3,
                elinewidth=1.3, capsize=0, label='Data')
    ax.axvline(x=kinematics.M_ETA, color=C['reference'], linestyle='--', label='PDG Mass')
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
    return fig


def write_per_bin_pdf(result, pdf_path, horizontal=False):
    """One bin_page per bin into pdf_path: i (m12) outer and j (m23) inner, or
    j outer and i inner with horizontal=True."""
    bins = {(b['i'], b['j']): b for b in result['dalitz']['bins']}
    n12 = len(result['dalitz']['m12_edges']) - 1
    n23 = len(result['dalitz']['m23_edges']) - 1
    if horizontal:
        order = [(i, j) for j in range(n23) for i in range(n12)]
    else:
        order = [(i, j) for i in range(n12) for j in range(n23)]
    with PdfPages(pdf_path) as pdf:
        for i, j in order:
            fig = bin_page(result, bins[(i, j)])
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f'[DONE] {pdf_path}')


def plot_dalitz_fit(result, prefix, horizontal=False):
    """global_fit.png next to prefix, the yield maps, and the per-bin pages
    (also in horizontal order with horizontal=True)."""
    plot_fit(result, os.path.join(os.path.dirname(os.path.abspath(prefix)), 'global_fit.png'),
             dcb_text)
    plot_yield_maps(result, prefix)
    write_per_bin_pdf(result, f'{prefix}_per_bin_fits.pdf')
    if horizontal:
        write_per_bin_pdf(result, f'{prefix}_per_bin_fits_horizontal.pdf', horizontal=True)


if __name__ == '__main__':
    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description='Figures from a driver_dalitz_fit.py results JSON')
    parser.add_argument('results', help='dalitz_fit_<factor>.json')
    parser.add_argument('--horizontal', action='store_true',
                        help='Also write the per-bin pages with j (m23) as the outer loop')
    args = parser.parse_args()
    with open(args.results) as f:
        result = json.load(f)
    plot_dalitz_fit(result, os.path.splitext(args.results)[0], horizontal=args.horizontal)
