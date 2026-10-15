"""
plot_q2.py

Plot the m^2_23 = m^2(mu+ mu-) = q^2 distribution of the candidates in a
dalitz_events.npz (written by src/fit/driver_dalitz.py), in the style of the
mass-fit plots.

Only candidates with mass inside --mass-range are plotted (default: the
420-680 MeV window of fit_dalitz_bins.py). --truth-match uses the file's
'matched' array (signal MC files written with truth matching):

    all       every candidate (default)
    matched   truth-matched candidates only
    both      both histograms on one plot

Usage:
    python src/plot/plot_q2.py EVENTS [OUTPUT]
        [--mass-range LO HI] [--truth-match all|matched|both] [--label TEXT]

OUTPUT defaults to q2.png next to EVENTS.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import kinematics
import plot_style
import plot_utils

M23_EDGES = np.arange(0.040, 0.450 + 1e-9, 0.005)   # [GeV^2]
M23_THRESHOLD = 4.0 * (kinematics.M_MU / 1000.0)**2  # 4 m_mu^2 [GeV^2]
ETA_MASS2 = (kinematics.M_ETA / 1000.0)**2           # m_eta^2 [GeV^2]

parser = argparse.ArgumentParser(description='Plot m^2(mu+ mu-) from a dalitz_events.npz')
parser.add_argument('events', help='dalitz_events.npz from driver_dalitz.py')
parser.add_argument('output', nargs='?', default=None,
                    help='Output PNG (default: q2.png next to EVENTS)')
parser.add_argument('--mass-range', nargs=2, type=float, default=[420.0, 680.0],
                    metavar=('LO', 'HI'),
                    help='Keep candidates with LO <= m(gamma mu+ mu-) < HI [MeV] (default: 420 680)')
parser.add_argument('--truth-match', choices=['all', 'matched', 'both'], default='all',
                    help="Plot every candidate, the truth-matched ones (needs a 'matched' "
                         "array), or both (default: all)")
parser.add_argument('--label', default='Data',
                    help="Legend label; truth-matched gets ' (truth-matched)' appended (default: 'Data')")
args = parser.parse_args()
output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.events)), 'q2.png')

# --- Read and select ---------------------------------------------------------

events = np.load(args.events)
m23s, masses = events['m23s'], events['masses']
mass_lo, mass_hi = args.mass_range
in_window = (masses >= mass_lo) & (masses < mass_hi)

# (legend label, colour, stats-box name, selection) of each histogram.
series = []
if args.truth_match in ('all', 'both'):
    series.append((args.label, plot_style.COLORS['data'], 'candidates', in_window))
if args.truth_match in ('matched', 'both'):
    matched = in_window & events['matched'].astype(bool)
    series.append((f'{args.label} (truth-matched)', plot_style.COLORS['fit'],
                   'truth-matched', matched))

histograms = []   # (label, colour, counts)
stats_lines = []
for label, color, stats_name, selection in series:
    selected_m23s = m23s[selection]
    counts = np.histogram(selected_m23s, bins=M23_EDGES)[0]
    histograms.append((label, color, counts))
    stats_lines.append(f'{len(selected_m23s)} {stats_name}')
    print(f'[INFO] {label}: {selection.sum()} / {len(selection)} candidates kept from {args.events}, '
          f'm23 in [{selected_m23s.min():.4f}, {selected_m23s.max():.4f}] GeV^2, '
          f'{len(selected_m23s) - int(counts.sum())} outside the plotted range')
bin_width = M23_EDGES[1] - M23_EDGES[0]

# --- Plot --------------------------------------------------------------------

plot_utils.apply_font()
plt.rcParams['font.size'] = 19
fig, ax = plt.subplots(figsize=(12, 9))
for label, color, counts in histograms:
    ax.stairs(counts, M23_EDGES, label=label, color=color, linewidth=1.8)
ax.axvline(M23_THRESHOLD, color=plot_style.COLORS['reference'], linestyle='--',
           label=r'$4m_\mu^2$ and $m_\eta^2$')
ax.axvline(ETA_MASS2, color=plot_style.COLORS['reference'], linestyle='--')
ax.set_xlim(M23_EDGES[0], M23_EDGES[-1])
ax.set_xlabel(r'$m^2_{23} = m^2(\mu^+\mu^-)$ [GeV$^2$]')
ax.set_ylabel(f'Candidates / ({bin_width:.3f} GeV$^2$)')
plot_style.style_axes(ax)

legend = ax.legend(loc='upper right', frameon=True, fancybox=False,
                   edgecolor='none', framealpha=1)
stats_lines.append(fr'${mass_lo:.0f} \leq m(\gamma\mu^+\mu^-) < {mass_hi:.0f}$ MeV')
stats_box = plot_style.add_stats_box(ax, '\n'.join(stats_lines))
content_max = max(float(counts.max()) for _, _, counts in histograms)
plot_style.reserve_top_margin(ax, content_max, artists=[stats_box, legend])
plot_style.format_y_axis(ax)
plot_style.save_plot(fig, output)
