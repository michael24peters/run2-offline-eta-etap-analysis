"""
plot_mass_fit.py

Fit + pull plots of mass-fit result dicts (fit_gauss / fit_dcb / a per-bin
Dalitz fit), with the stats box of the fit model:

    plot_fit(result, output, gauss_text)   Gaussian + Chebyshev
    plot_fit(result, output, dcb_text)     DCB + Chebyshev
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_style
import plot_utils


def gauss_text(params, chi2_per_ndof):
    """Stats box of a Gaussian + Chebyshev fit: (plot_text, term_text)."""
    mean, mean_err = plot_utils.val_err(params['mean'])
    sigma, sigma_err = plot_utils.val_err(params['sigma'])

    stats = [
        [(r'\mu', 'μ', mean, mean_err, '.2f', 'MeV')],
        [(r'\sigma', 'σ', sigma, sigma_err, '.2f', 'MeV')],
    ]
    extra = (
        fr'$\chi^2/ndof$ = {chi2_per_ndof:.2f}',
        f'χ²/ndof = {chi2_per_ndof:.2f}',
    )
    return plot_style.build_stats_text(stats, extra=extra)


def dcb_text(params, chi2_per_ndof):
    """Stats box of a DCB + Chebyshev fit: (plot_text, term_text)."""
    mean, mean_err   = plot_utils.val_err(params['mean'])
    sigma, sigma_err = plot_utils.val_err(params['sigma'])
    n_sig, n_sig_err = plot_utils.val_err(params['n_sig'])
    n_bkg, n_bkg_err = plot_utils.val_err(params['n_bkg'])

    # DCB tail parameters (asymmetric)
    alphaL, alphaL_err = plot_utils.val_err(params.get('alphaL', {}))
    nL, nL_err         = plot_utils.val_err(params.get('nL', {}))
    alphaR, alphaR_err = plot_utils.val_err(params.get('alphaR', {}))
    nR, nR_err         = plot_utils.val_err(params.get('nR', {}))

    # Whatever key in `params` isn't one of the known signal/yield names
    # above is a background parameter (c0, c1).
    SIGNAL_KEYS = {'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'n_sig', 'n_bkg'}
    bkg_row = [(name, name, *plot_utils.val_err(params[name]), '.3f', '')
               for name in params if name not in SIGNAL_KEYS]

    # Rows of (latex_symbol, plain_symbol, value, error, format string, unit)
    stats = [
        [(r'\mu', 'μ', mean, mean_err, '.2f', 'MeV')],
        [(r'\sigma', 'σ', sigma, sigma_err, '.2f', 'MeV')],
        [(r'\alpha_L', 'α_L', alphaL, alphaL_err, '.2f', ''),
         (r'n_L', 'n_L', nL, nL_err, '.1f', '')],
        [(r'\alpha_R', 'α_R', alphaR, alphaR_err, '.2f', ''),
         (r'n_R', 'n_R', nR, nR_err, '.2f', '')],
        [(r'n_{sig}', 'n_sig', n_sig, n_sig_err, '.0f', '')],
        [(r'n_{bkg}', 'n_bkg', n_bkg, n_bkg_err, '.0f', '')],
        bkg_row,
    ]
    # Values which are not a value-error pair (e.g., chi2/ndof). Must provide
    # exact strings for desired LaTeX and terminal output.
    extra = (
        fr'$\chi^2/ndof$ = {chi2_per_ndof:.2f}',
        f'χ²/ndof = {chi2_per_ndof:.2f}',
    )
    return plot_style.build_stats_text(stats, extra=extra)


def build_fit_figure(data, build_text, x_lim=None, title=None):
    """
    Draw a fit + pull figure from a result dict. Does not save or print
    anything; plot_fit() does both.

    build_text(params, chi2_per_ndof) returns the stats box as
    (plot_text, term_text): gauss_text, dcb_text, or a per-bin Dalitz one.

    x_lim: (xmin, xmax) override; defaults to the data range.
    title: optional plot title (e.g. to identify one page of a multi-page
    PDF); omitted entirely when not given.

    If `data` has a 'phase_space' entry ({'x': [...], 'f': [...]}, the
    per-bin phase-space factor f(M) the fit multiplied its shapes by), it is
    drawn in its own panel between the fit and the pulls, on the same mass
    axis. An optional 'f_sig' in it is a separate signal factor (with 'f'
    then the background's), drawn dashed in the same panel.

    Returns (fig, term_text).
    """
    # Extract histogram data
    hist = data['histogram']
    x = np.array(hist['bin_centers'])
    y = np.array(hist['bin_counts'])
    yerr = np.array(hist['bin_errors'])
    fit = np.array(hist['bin_fit'])
    bkg = np.array(hist['bin_bkg'])
    pulls = np.array(hist['bin_pulls'])
    # Font
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 19
    # Create figure with two panels, or three with a phase-space factor
    phase_space = data.get('phase_space')
    if phase_space is None:
        fig, (ax, ax_pull) = plt.subplots(
            2, 1, figsize=(12, 12),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
    else:
        fig, (ax, ax_f, ax_pull) = plt.subplots(
            3, 1, figsize=(12, 15),
            gridspec_kw={'height_ratios': [3, 1, 1]},
            sharex=True
        )
        ax_f.plot(phase_space['x'], phase_space['f'],
                  color=plot_style.COLORS['phase_space'], linewidth=1.8,
                  label='background (flat)')
        if 'f_sig' in phase_space:
            ax_f.plot(phase_space['x'], phase_space['f_sig'],
                      color=plot_style.COLORS['phase_space'], linewidth=1.8,
                      linestyle='--', label='signal (Dalitz decay)')
            ax_f.legend(loc='center right', fontsize=12, frameon=False)
        ax_f.axvline(x=547.86, color=plot_style.COLORS['reference'], linestyle='--')
        ax_f.set_ylim(-0.05, 1.1)
        ax_f.set_yticks([0, 0.5, 1])
        ax_f.set_ylabel('$f(M)$')
        plot_style.style_axes(ax_f, minor_y=False)
    if title:
        ax.set_title(title)

    # Top panel: data + fit
    xmin, xmax = x_lim if x_lim is not None else (x.min() - 0.5, x.max() + 0.5)
    ax.set_xlim(xmin, xmax)
    ax.errorbar(x, y, yerr=yerr,
                fmt='+',
                label='Data',
                color=plot_style.COLORS['data'],
                markersize=7,
                markeredgewidth=1.3,
                elinewidth=1.3,
                capsize=0
    )
    # Background as a filled region from zero, total fit drawn on top of it --
    # the gap between the fill's top edge and the fit curve reads as signal.
    ax.fill_between(x, 0, bkg, label='Background',
                     color=plot_style.COLORS['background'], alpha=0.35,
                     linewidth=0.8, edgecolor=plot_style.COLORS['background'])
    ax.plot(x, fit, label='Total fit', color=plot_style.COLORS['fit'], linewidth=1.8)
    ax.axvline(x=547.86, color=plot_style.COLORS['reference'],
               linestyle='--', label='PDG Mass')

    # Ticks and grid
    plot_style.style_axes(ax)

    # Bin width, reused for both the y-label and the pull-bar width below.
    bin_width = x[1] - x[0]

    legend = ax.legend(loc='upper right', frameon=True, fancybox=False,
                        edgecolor='none', framealpha=1)
    ax.set_ylabel(f'Candidates / ({bin_width:.1f} MeV/c²)')

    # Add fit parameters to plot
    params = data['fit']['parameters']
    chi2_per_ndof = data['fit']['chi2_per_ndof']
    plot_text, term_text = build_text(params, chi2_per_ndof)
    stats_box = plot_style.add_stats_box(ax, plot_text)

    # Reserve headroom above the data/curves for the stats box + legend,
    # sized by actually measuring what was rendered -- not a fixed fraction.
    content_max = max(fit.max(), (y + yerr).max())
    plot_style.reserve_top_margin(ax, content_max, artists=[stats_box, legend])
    plot_style.format_y_axis(ax)

    # Bottom panel: pulls
    ax_pull.bar(x, pulls, width=bin_width, color=plot_style.COLORS['pull'], edgecolor='none')
    ax_pull.axhline(0, color=plot_style.COLORS['data'], lw=0.8)
    ymin, ymax = int(np.floor(pulls.min())) - 1, int(np.ceil(pulls.max())) + 1
    ax_pull.set_ylim(ymin, ymax)
    ax_pull.set_yticks(np.arange(ymin, ymax + 1, 1))
    plot_style.style_axes(ax_pull, minor_y=False)
    # Labels
    ax_pull.set_ylabel('Pulls')
    ax_pull.set_xlabel(plot_style.decay_label(data.get('meta', {})))

    # The leftmost x-tick label sits right where the pull axis's bottom
    # y-tick label ends, so it collides with it -- drop it.
    xticklabels = ax_pull.get_xticklabels()
    if xticklabels:
        min(xticklabels, key=lambda t: t.get_position()[0]).set_visible(False)

    return fig, term_text


def plot_fit(data, output, build_text):
    """Fit + pull plot of a result dict, saved to output, with its stats
    printed to the terminal."""
    fig, term_text = build_fit_figure(data, build_text)
    plot_style.save_plot(fig, output)
    plot_style.print_summary(term_text)
