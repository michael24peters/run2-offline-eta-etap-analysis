"""
plot_style.py

Repeatable plot style for figures: text-box stat summaries, text placement (auto
or manual), and summary statistics both to terminal and matplotlib.
"""

import math

import matplotlib.ticker as mticker

# Shared fit-plot palette. fit/background/pull validated pairwise
# colorblind-safe (dataviz skill's validate_palette.js, all-pairs, light mode).
COLORS = {
    'data':       '#000000',
    'fit':        '#2a78d6',
    'background': '#1baf7a',
    'pull':       '#eb6834',
    'reference':  '#898781',
}

# LaTeX decay string for each channel, keyed by the channel segment of
# meta.input_file ('ntuple/lhcb/<channel>/<date>/combined_files.root').
CHANNEL_LABELS = {
    'eta2mumu':      r'\eta \to \mu^+ \mu^-',
    'eta2mumugamma': r'\eta \to \mu^+ \mu^- \gamma',
}

def decay_label(meta):
    """
    LaTeX x-axis mass label for a fit-result's meta dict, derived from the
    channel segment of meta['input_file']. Falls back to a bare 'Mass' label
    if no known channel is found.
    """
    segments = meta.get('input_file', '').split('/')
    for segment in segments:
        if segment in CHANNEL_LABELS:
            return fr'$m({CHANNEL_LABELS[segment]})$ [MeV/c$^2$]'
    return 'Mass [MeV/c²]'

def build_stats_text(stats, extra=None):
    """
    Build the matplotlib (LaTeX) and terminal (plain Unicode) renderings of a
    fit's stats box from one shared spec, so each symbol is written once.

    stats: list of "rows"; each row is a list of one or more
        (latex_symbol, plain_symbol, value, error, fmt, unit) tuples. Tuples
        within a row are joined with ", " onto one line (so e.g. alphaL and
        nL can share a row, matching a compact stats box layout).
    extra: optional (plot_line, term_line) tuple appended as a final row,
        for values that aren't a value/error pair (e.g. chi2/ndof).

    Returns (plot_text, term_text).
    """
    def render_row(row, latex):
        """
        Render a single row of the stats box for latex or terminal output.
        """
        parts = []
        for sym_latex, sym_plain, value, error, fmt, unit in row:
            sym = f'${sym_latex}$' if latex else sym_plain
            unit_str = f' {unit}' if unit else ''
            parts.append(f'{sym} = {value:{fmt}} ± {error:{fmt}}{unit_str}')
        return ', '.join(parts)

    plot_lines = [render_row(row, latex=True) for row in stats]
    term_lines = [render_row(row, latex=False) for row in stats]
    if extra is not None:
        # Append the extra row the output exactly as given (no val/err format).
        plot_lines.append(extra[0])
        term_lines.append(extra[1])
    return '\n'.join(plot_lines), '\n'.join(term_lines)

def add_stats_box(ax, plot_text, show=True):
    """
    Draw a boxed text annotation in `ax`'s upper-left corner. Does nothing and
    returns None if show=False or plot_text is empty -- this is the
    "statistics / no statistics" toggle. Returns the Text artist otherwise, so
    the caller can measure it (see reserve_top_margin).
    """
    if not show or not plot_text:
        return None
    return ax.text(0.02, 0.97, plot_text,
                    transform=ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='none',
                              edgecolor='none'))

def reserve_top_margin(ax, content_max, artists, ymin=0, growth=1.15, max_iter=8):
    """
    Grow ax's y-limit (from content_max upward) until none of `artists`
    (already placed at fixed axes-fraction corners, e.g. the stats box and
    legend) extends down past content_max in data coordinates -- so the
    reserved headroom is sized to whatever was actually rendered, rather than
    a hand-tuned fraction.
    """
    artists = [a for a in artists if a is not None]
    if not artists:
        ax.set_ylim(ymin, content_max * growth)
        return
    ymax = content_max * growth
    for _ in range(max_iter):
        ax.set_ylim(ymin, ymax)
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lowest = min(
            artist.get_window_extent(renderer=renderer)
                  .transformed(ax.transData.inverted()).y0
            for artist in artists
        )
        if lowest >= content_max:
            return
        ymax *= growth
    ax.set_ylim(ymin, ymax)

def style_axes(ax, minor_y=True):
    """Inward, mirrored major+minor ticks on all four sides, and heavier
    spines -- shared chrome for every panel of a fit plot. Set minor_y=False
    to omit minor ticks on the y-axis only (e.g. the pull panel's integer
    scale has no need for them)."""
    ax.minorticks_on()
    if not minor_y:
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(which='major', direction='in', top=True, right=True, width=1.6, length=7.0)
    ax.tick_params(which='minor', direction='in', top=True, right=True, width=1.2, length=4.0)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

def format_y_axis(ax):
    """
    Plain integer y-tick labels, offset by whatever power of ten keeps the
    mantissa a short integer (e.g. 0, 100, 200, 300 with a x10^2 marker)
    instead of matplotlib's default single-digit-plus-decimal mantissa (0.0,
    0.5, ..., 3.0 x10^4). Values under 1000 need no offset at all. Reads ax's
    current y-limit, so call this after the axis range is final (i.e. after
    reserve_top_margin).
    """
    ymax = ax.get_ylim()[1]
    if ymax < 1000:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f'{v:.0f}'))
        return
    divisor_oom = math.floor(math.log10(ymax)) - 2
    divisor = 10 ** divisor_oom
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f'{v / divisor:.0f}'))
    ax.text(0, 1.0, fr'$\times 10^{{{divisor_oom}}}$', transform=ax.transAxes,
            ha='left', va='bottom')

def print_summary(text, width=48, title='Fit summary'):
    """Print a summary to the terminal."""
    print()
    print('─' * width)
    print(f'{title:^{width-4}}')
    print('─' * width)
    print(text)
    print('─' * width)

def save_plot(fig, output, dpi=300):
    """Save the plot to disk."""
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    print(f'[DONE] Plot saved to {output}.')
