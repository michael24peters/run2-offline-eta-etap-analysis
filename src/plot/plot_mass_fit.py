"""
plot_mass_fit.py

Library for plotting fit results from JSON files.

Usage:
    python plot_mass_fit.py input.json [output.png]
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_utils
import plot_style


def build_fit_figure(data, config):
    """
    Draw a fit + pull figure from an already-loaded result dict. Does not
    save or print anything -- that's plot_fit_data(), which wraps this for
    the single-plot-per-JSON-file callers (plot_dcb.py, plot_gauss.py). A
    caller that needs the figure itself (e.g. to add it to a multi-page
    PdfPages) should call this directly instead.

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
    plot_utils.apply_font(config.get('font', 'serif'))
    plt.rcParams['font.size'] = config.get('font_size', 19)
    # Create figure with two panels
    fig, (ax, ax_pull) = plt.subplots(
        2, 1, figsize=config.get('figsize', (12, 12)),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # Top panel: data + fit
    # Set x-axis limits based on config or data range
    if config.get('x_lim') is not None: xmin, xmax = config['x_lim']
    else: xmin, xmax = x.min() - 0.5, x.max() + 0.5
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
    if config.get('pdg_line', True):
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
    ax.set_ylabel(config.get('y_label', f'Candidates / ({bin_width:.1f} MeV/c²)'))

    # Add fit parameters to plot
    params = data['fit']['parameters']
    chi2_per_ndof = data['fit']['chi2_per_ndof']
    # Get text from config function, with error handling for compatibility
    try:
        plot_text, term_text = config['build_text'](params, chi2_per_ndof)
    except:
        plot_text = config['build_text'](params, chi2_per_ndof)
        term_text = None

    stats_box = plot_style.add_stats_box(ax, plot_text,
                                          show=config.get('show_stats', True))

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
    # Set y axis limits based on config or pull range. Bins outside
    # fit_window (if set) were never fit against, so their pulls can be huge
    # and shouldn't blow out the axis scale for the bins that matter.
    if config.get('pull_ylim') is not None:
        ymin, ymax = config['pull_ylim']
    else:
        scale_pulls = pulls
        if fit_window:
            in_window = (x >= fit_window[0]) & (x <= fit_window[1])
            if in_window.any():
                scale_pulls = pulls[in_window]
        ymin, ymax = int(np.floor(scale_pulls.min())) - 1, int(np.ceil(scale_pulls.max())) + 1
    ax_pull.set_ylim(ymin, ymax)
    ax_pull.set_yticks(np.arange(ymin, ymax + 1, 1))  # TODO: test ticks
    plot_style.style_axes(ax_pull, minor_y=False)
    # Labels
    ax_pull.set_ylabel('Pulls')
    ax_pull.set_xlabel(config.get('x_label', plot_style.decay_label(data.get('meta', {}))))
    # Show x labels on both panels, comment out for only bottom panel
    # ax.tick_params(axis='x', which='both', labelbottom=True)

    # The leftmost x-tick label sits right where the pull axis's bottom
    # y-tick label ends, so it collides with it -- drop it.
    xticklabels = ax_pull.get_xticklabels()
    if xticklabels:
        min(xticklabels, key=lambda t: t.get_position()[0]).set_visible(False)

    return fig, ax, ax_pull, plot_text, term_text

def plot_fit_data(data, config):
    """Render a fit + pull plot from an already-loaded result dict, save it,
    and print its summary."""
    output = config['output'] if 'output' in config else 'out/fit_result.png'
    fig, ax, ax_pull, plot_text, term_text = build_fit_figure(data, config)
    plot_style.save_plot(fig, output)
    plot_style.print_summary(term_text if term_text else plot_text)

def plot_fit(config):
    # Parse arguments
    args = plot_utils.parse_args()
    output = config['output'] if 'output' in config else 'out/fit_result.png'
    print(f'[INFO] Reading from {args.input} and writing to {output}.')

    # Load JSON data and plot
    data = plot_utils.load_data(args.input)
    plot_fit_data(data, config)
