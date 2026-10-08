"""
plot_yield_ratio.py

Compare the per-bin signal yields of two fit_dalitz_bins.py results with the
same binning, one with the flat signal factor and one with the Dalitz-decay
signal factor:

    grid   map of n_sig(dalitz) / n_sig(flat) on the Dalitz grid
    q2     n_sig of both against q^2, and their ratio

Bins without a valid fit in both, or with n_sig fixed to 0, are left out.
Both fits use the same candidates, so their yields and errors are strongly
correlated; the ratio carries no error bar.

Prints a per-bin table of both yields, their ratio and both chi2/ndof, and
writes yield_ratio_<binning>.png next to the Dalitz-factor results.

Usage:
    python src/plot/plot_yield_ratio.py FLAT.json DALITZ.json
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import kinematics
import plot_style
import plot_utils
from plot_phase_space import dalitz_boundary

C = plot_style.COLORS

parser = argparse.ArgumentParser(description='Signal yields with the flat vs Dalitz signal factor')
parser.add_argument('flat', help='dalitz_fit_<binning>_flat.json')
parser.add_argument('dalitz', help='dalitz_fit_<binning>_dalitz.json')
args = parser.parse_args()

with open(args.flat) as f:
    flat = json.load(f)['dalitz']
with open(args.dalitz) as f:
    dalitz = json.load(f)['dalitz']
assert flat['binning'] == dalitz['binning'], 'the two results have different binnings'
assert (flat['signal_factor'], dalitz['signal_factor']) == ('flat', 'dalitz')
binning = flat['binning']
m12_edges = np.array(flat['m12_edges'])
m23_edges = np.array(flat['m23_edges'])

# Bins with a valid fit and a free n_sig in both results.
flat_bins = {(b['i'], b['j']): b for b in flat['bins']}
dalitz_bins = {(b['i'], b['j']): b for b in dalitz['bins']}
compared = [key for key in flat_bins
            if flat_bins[key]['valid'] and dalitz_bins[key]['valid']
            and not flat_bins[key]['n_sig_fixed_zero'] and not dalitz_bins[key]['n_sig_fixed_zero']
            and flat_bins[key]['n_sig'] > 0]

# --- Table -------------------------------------------------------------------

print(f"{'bin':>9} {'n_cand':>8} {'n_sig flat':>18} {'n_sig dalitz':>18} {'ratio':>7} "
      f"{'chi2/ndof flat':>15} {'dalitz':>7}")
for key in sorted(compared):
    fb, db = flat_bins[key], dalitz_bins[key]
    print(f"{str(key):>9} {fb['n_cand']:8.0f} {fb['n_sig']:10.1f} ± {fb['n_sig_err']:5.1f} "
          f"{db['n_sig']:10.1f} ± {db['n_sig_err']:5.1f} {db['n_sig'] / fb['n_sig']:7.3f} "
          f"{fb['chi2_per_ndof'] or np.nan:15.3f} {db['chi2_per_ndof'] or np.nan:7.3f}")
print(f"summed n_sig over valid bins: flat {flat['yield_closure']['n_sig_summed']:.1f}, "
      f"dalitz {dalitz['yield_closure']['n_sig_summed']:.1f}, "
      f"global {flat['yield_closure']['n_sig_global']:.1f}")

# --- Figure ------------------------------------------------------------------

plot_utils.apply_font()
plt.rcParams['font.size'] = 11
out_path = os.path.join(os.path.dirname(os.path.abspath(args.dalitz)), f'yield_ratio_{binning}.png')

if binning == 'grid':
    # Ratio map: blue below 1, gray at 1, orange above 1; uncompared bins blank.
    ratio = np.full((len(m12_edges) - 1, len(m23_edges) - 1), np.nan)
    for (i, j) in compared:
        ratio[i, j] = dalitz_bins[(i, j)]['n_sig'] / flat_bins[(i, j)]['n_sig']
    cmap = LinearSegmentedColormap.from_list('ratio', [C['fit'], '#e4e3df', C['pull']])
    norm = TwoSlopeNorm(vcenter=1.0, vmin=min(np.nanmin(ratio), 0.99),
                        vmax=max(np.nanmax(ratio), 1.01))
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(m12_edges, m23_edges, ratio.T, cmap=cmap, norm=norm, shading='auto')
    fig.colorbar(mesh, ax=ax, label=r'$n_{sig}$(Dalitz) / $n_{sig}$(flat)', pad=0.13)
    boundary12, boundary23 = dalitz_boundary(kinematics.M_ETA)
    ax.plot(boundary12, boundary23, color=C['data'], lw=1, label=r'Dalitz region at $m_\eta$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel(r'$m^2_{\gamma\mu^+}$ [GeV$^2$]')
    ax.set_ylabel(r'$m^2_{\mu^+\mu^-}$ [GeV$^2$]')
    plot_style.add_bin_index_axes(ax, m12_edges, m23_edges)
else:
    # Yields of both against q^2 (top), their ratio (bottom).
    keys = sorted(compared)   # (0, j): in order of q^2
    q2_mid = np.array([np.mean(flat_bins[key]['m23_range']) for key in keys])
    q2_half = np.array([np.diff(flat_bins[key]['m23_range'])[0] / 2 for key in keys])
    offset = 0.15 * q2_half   # side-by-side points within a strip
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
    for bins, color, marker, label, shift in [(flat_bins, C['phase_space'], 'o', 'flat', -offset),
                                              (dalitz_bins, C['fit'], 's', 'Dalitz decay', offset)]:
        ax.errorbar(q2_mid + shift, [bins[key]['n_sig'] for key in keys],
                    yerr=[bins[key]['n_sig_err'] for key in keys], fmt=marker, color=color,
                    markersize=5, capsize=0, label=f'signal factor: {label}')
    ax.set_ylabel('Fitted signal yield per strip')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=9)
    ax_ratio.plot(q2_mid, [dalitz_bins[key]['n_sig'] / flat_bins[key]['n_sig'] for key in keys],
                  'o', color=C['data'], markersize=5)
    ax_ratio.axhline(1.0, color=C['reference'], linestyle='--', lw=1)
    ax_ratio.set_ylabel('Dalitz / flat')
    ax_ratio.set_xlabel(r'$q^2 = m^2_{\mu^+\mu^-}$ [GeV$^2$]')
    for axis in (ax, ax_ratio):
        plot_style.style_axes(axis)

plot_style.save_plot(fig, out_path)
