"""
plot_mass_fit.py

Library for plotting fit results from JSON files.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_utils
import plot_style


def build_fit_figure(data, build_text, x_lim=None, title=None):
    """
    Draw a fit + pull figure from an already-loaded result dict. Does not
    save or print anything -- that's plot_fit_data(), which wraps this for
    the single-plot-per-JSON-file callers (plot_dcb.py, plot_gauss.py). A
    caller that needs the figure itself (e.g. to add it to a multi-page
    PdfPages) should call this directly instead.

    build_text(params, chi2_per_ndof) builds the stats-box text -- this is
    what lets each caller (DCB vs Gauss, or a per-bin Dalitz fit) show its
    own stats without this function knowing about any of them.

    x_lim: (xmin, xmax) override; defaults to the data range.
    title: optional plot title (e.g. to identify one page of a multi-page
    PDF); omitted entirely when not given.

    If `data` has a 'phase_space' entry ({'x': [...], 'f': [...]}, the
    per-bin phase-space factor f(M) the fit multiplied its shapes by), it is
    drawn in its own panel between the fit and the pulls, on the same mass
    axis.

    Returns (fig, ax, ax_pull, plot_text, term_text).
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
                  color=plot_style.COLORS['phase_space'], linewidth=1.8)
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

    # Mark the region actually used in the likelihood, if the fit was
    # restricted (fit_dcb's fit_window) -- curves elsewhere on the plot are
    # extrapolated from the fixed/fitted shape, not fit against those bins.
    fit_window = data.get('meta', {}).get('fit_window')
    if fit_window:
        for ax_i in (ax, ax_pull):
            for edge in fit_window:
                ax_i.axvline(x=edge, color=plot_style.COLORS['reference'],
                             linestyle=':', linewidth=1.2)

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
    # Get text from build_text, with error handling for compatibility
    try:
        plot_text, term_text = build_text(params, chi2_per_ndof)
    except:
        plot_text = build_text(params, chi2_per_ndof)
        term_text = None

    stats_box = plot_style.add_stats_box(ax, plot_text)

    # Reserve headroom above the data/curves for the stats box + legend,
    # sized by actually measuring what was rendered -- not a fixed fraction.
    content_max = max(fit.max(), (y + yerr).max())
    plot_style.reserve_top_margin(ax, content_max, artists=[stats_box, legend])
    plot_style.format_y_axis(ax)

    if fit_window:
        ax.text(fit_window[0], ax.get_ylim()[1], ' fit region',
                color=plot_style.COLORS['reference'], fontsize=12, va='top', ha='left')

    # Bottom panel: pulls
    ax_pull.bar(x, pulls, width=bin_width, color=plot_style.COLORS['pull'], edgecolor='none')
    ax_pull.axhline(0, color=plot_style.COLORS['data'], lw=0.8)
    # y axis config
    # Bins outside fit_window (if set) were never fit against, so their pulls
    # can be huge and shouldn't blow out the axis scale for the bins that
    # matter.
    scale_pulls = pulls
    if fit_window:
        in_window = (x >= fit_window[0]) & (x <= fit_window[1])
        if in_window.any():
            scale_pulls = pulls[in_window]
    ymin, ymax = int(np.floor(scale_pulls.min())) - 1, int(np.ceil(scale_pulls.max())) + 1
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

    return fig, ax, ax_pull, plot_text, term_text

def plot_fit_data(data, output, build_text, x_lim=None, title=None):
    """Render a fit + pull plot from an already-loaded result dict, save it,
    and print its summary."""
    fig, ax, ax_pull, plot_text, term_text = build_fit_figure(
        data, build_text, x_lim=x_lim, title=title)
    plot_style.save_plot(fig, output)
    plot_style.print_summary(term_text if term_text else plot_text)
