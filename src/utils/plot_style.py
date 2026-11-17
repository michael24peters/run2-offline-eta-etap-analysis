"""
plot_style.py

Repeatable plot style for figures: text-box stat summaries, text placement (auto or manual), and
summary statistics both to terminal and matplotlib.
"""

import math

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

import kinematics

# Shared fit-plot palette. fit/background/pull/phase_space colorblind-safe.
COLORS = {
    'data':        '#000000',
    'fit':         '#2a78d6',
    'background':  '#1baf7a',
    'pull':        '#eb6834',
    'reference':   '#898781',
    'phase_space': '#4a3aa7',
}

# LaTeX decay string for each channel, keyed by the channel segment of meta.input_file
# ('ntuple/lhcb/<channel>/<date>/combined_files.root').
CHANNEL_LABELS = {
    'eta2mumu':      r'\eta \to \mu^+ \mu^-',
    'eta2mumugamma': r'\eta \to \mu^+ \mu^- \gamma',
    'eta2mumumumu':  r'\eta \to \mu^+ \mu^- \mu^+ \mu^-',
}

DATA_YEAR = 2018   # every ntuple: 2018 Turbo data (Collision18) or 2018-conditions MC
BOX_FONT_SCALE = 0.75   # add_stats_box text size, relative to the current font size

def sample_label(path, sample=None):
    """
    '<decay> <sample> <year>' text of an input path, e.g. '$\\eta \\to \\mu^+ \\mu^-$ MC 2018': the
    decay from the channel segment of `path`, the sample 'MC' for a path with an 'mc' segment and
    'Data' for one with an 'lhcb' segment, unless `sample` is given. '' if no known channel is
    found.
    """
    segments = path.split('/')
    channels = [segment for segment in segments if segment in CHANNEL_LABELS]
    if not channels:
        return ''
    if sample is None:
        sample = 'MC' if 'mc' in segments else 'Data' if 'lhcb' in segments else ''
    parts = [f'${CHANNEL_LABELS[channels[0]]}$', sample, str(DATA_YEAR)]
    return ' '.join(part for part in parts if part)

def decay_label(meta):
    """
    LaTeX x-axis mass label for a fit-result's meta dict, derived from the channel segment of
    meta['input_file'], with η′ in place of η when the fit window meta['xmin']–meta['xmax'] holds
    the η′ mass (e.g. an η′ → μ⁺μ⁻μ⁺μ⁻ fit on the eta2mumumumu ntuple). Falls back to a bare
    'Mass' label if no known channel is found.
    """
    segments = meta.get('input_file', '').split('/')
    for segment in segments:
        if segment in CHANNEL_LABELS:
            decay = CHANNEL_LABELS[segment]
            if meta.get('xmin', math.inf) <= kinematics.M_ETA_PRIME <= meta.get('xmax', -math.inf):
                decay = decay.replace(r'\eta', r"\eta'", 1)
            return fr'$m({decay})$ [MeV]'
    return 'Mass [MeV]'

def count_label(entries, edges, unit=''):
    """
    y-axis label of a histogram of `entries` (e.g. 'Candidates', 'Muons') over the bin `edges`:
    '<entries> / (<bin width> <unit>)' for bins of one width, e.g. 'Candidates / (1 MeV)' or
    'Muons / (0.5)' for a quantity with no unit, and '<entries>' alone for bins of varying width
    (e.g. log-spaced). Units are natural units: MeV, GeV, never MeV/c or MeV/c².
    """
    widths = np.diff(edges)
    if not np.allclose(widths, widths[0], rtol=1e-6, atol=0):
        return entries
    width = f'{widths[0]:.4g} {unit}'.strip()
    return f'{entries} / ({width})'

def build_stats_text(stats, extra=None):
    """
    Build the matplotlib (LaTeX) and terminal (plain Unicode) renderings of a fit's stats box from
    one shared spec, so each symbol is written once.

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

def add_stats_box(ax, plot_text, sample='', textprops=None, fontsize=None):
    """
    Draw a white box in `ax`'s upper-left corner: `sample` (a sample_label) above `plot_text`
    (with `textprops`, e.g. {'color': 'red'}), both in `fontsize` points (default: BOX_FONT_SCALE
    times the current font size). Does nothing and returns None if both are empty. Returns the box
    artist otherwise, so the caller can measure it (see reserve_top_margin).
    """
    size = fontsize if fontsize is not None else BOX_FONT_SCALE * mpl.rcParams['font.size']
    rows = []
    if sample:
        rows.append(TextArea(sample, textprops={'fontsize': size}))
    if plot_text:
        rows.append(TextArea(plot_text, textprops={'fontsize': size, **(textprops or {})}))
    if not rows:
        return None
    box = AnchoredOffsetbox(loc='upper left', child=VPacker(children=rows, align='left', pad=0,
                                                            sep=0.4 * size),
                            bbox_to_anchor=(0.02, 0.97), bbox_transform=ax.transAxes,
                            borderpad=0, pad=0.4, frameon=True)
    box.patch.set_boxstyle('round,pad=0,rounding_size=0.4')
    box.patch.set_facecolor('white')
    box.patch.set_edgecolor('none')
    ax.add_artist(box)
    return box

def reserve_top_margin(ax, content_max, artists, ymin=0, growth=1.15, max_iter=8):
    """
    Grow ax's y-limit (from content_max upward) until none of `artists` (already placed at fixed
    axes-fraction corners, e.g. the stats box and legend) extends down past content_max in data
    coordinates -- so the reserved headroom is sized to whatever was actually rendered, rather than
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
    """Inward, mirrored major+minor ticks on all four sides, and heavier spines. Set minor_y=False
    to disable minor ticks on the y-axis."""
    ax.minorticks_on()
    if not minor_y:
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(which='major', direction='in', top=True, right=True, width=1.6, length=7.0)
    ax.tick_params(which='minor', direction='in', top=True, right=True, width=1.2, length=4.0)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

def add_bin_index_axes(ax, x_edges, y_edges, x_label='bin i', y_label='bin j'):
    """Secondary top/right axes giving each Dalitz bin's (i, j) grid index, alongside the primary
    physical-value axes on the bottom/left -- so a bin spotted on a Dalitz grid figure can be read
    off directly, in the same (i, j) numbering used for e.g. per_bin_fits.pdf's page titles. Assumes
    uniform bin edges, matching every Dalitz grid in this codebase (np.linspace)."""
    ax.tick_params(which='both', top=False, right=False)  # would collide with these
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    x0, dx = x_edges[0], (x_edges[-1] - x_edges[0]) / nx
    y0, dy = y_edges[0], (y_edges[-1] - y_edges[0]) / ny

    top = ax.secondary_xaxis('top', functions=(lambda x: (x - x0) / dx,
                                                lambda i: x0 + i * dx))
    top.set_xticks([i + 0.5 for i in range(nx)])
    top.set_xticklabels([str(i) for i in range(nx)], fontsize=8)
    top.set_xlabel(x_label, fontsize=9)
    top.tick_params(length=3)

    right = ax.secondary_yaxis('right', functions=(lambda y: (y - y0) / dy,
                                                     lambda j: y0 + j * dy))
    right.set_yticks([j + 0.5 for j in range(ny)])
    right.set_yticklabels([str(j) for j in range(ny)], fontsize=8)
    right.set_ylabel(y_label, fontsize=9)
    right.tick_params(length=3)
    return top, right

def format_y_axis(ax):
    """
    Plain integer y-tick labels, offset by whatever power of ten keeps the mantissa a short integer
    (e.g. 0, 100, 200, 300 with a x10^2 marker) instead of matplotlib's default
    single-digit-plus-decimal mantissa (0.0, 0.5, ..., 3.0 x10^4). Values under 1000 need no offset
    at all. Reads ax's current y-limit, so call this after the axis range is final (i.e. after
    reserve_top_margin).
    """
    # Candidate counts are always non-negative integers, so tick positions should be too --
    # otherwise adjacent fractional ticks can round to the same displayed integer (e.g. a near-empty
    # bin autoscaled to [0, 1]). nbins/steps match mpl's default AutoLocator, so normal-range plots
    # keep the same tick choices as before; integer=True is the only real change.
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10], integer=True))
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
