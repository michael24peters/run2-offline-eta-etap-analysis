"""
plot_signal_factor.py

Figures for src/fit/dalitz-signal-factor.md comparing the flat phase-space
factor (kinematics.phase_space_factor) with the Dalitz-decay factor
(kinematics.dalitz_factor). Kinematics only, no data:

    density.png       kinematics.dalitz_density over the eta's Dalitz region,
                      and the q^2 spectrum it gives next to flat phase space
    factors_grid.png  both factors against M for six grid bins
    convergence.png   the largest difference between the two factors against
                      bin size, for bins centred on the edge of the eta's
                      Dalitz region
    factors_q2.png    both factors against M for four q^2 strips, each scaled
                      to 1 at m_eta

Usage:
    python src/plot/plot_signal_factor.py [--outdir out/signal_factor]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import kinematics
import plot_style
import plot_utils
from plot_phase_space import dalitz_boundary

XMIN, XMAX = 420.0, 680.0                 # mass window [MeV]
M12_EDGES = np.linspace(0.0, 0.21, 15)    # fit_dalitz_bins.py's grid [GeV^2]
M23_EDGES = np.linspace(0.0, 0.31, 15)
STRIP_M12 = (0.0, (XMAX / 1000.0)**2)     # m12 range of a q^2 strip [GeV^2]
GRID_BINS = [(4, 4), (6, 8), (7, 2), (1, 13), (1, 8), (12, 7)]
STRIPS = [2, 4, 8, 12]                    # m23 indices of the q^2 strips shown
EDGE_Q2 = [0.06, 0.12, 0.20, 0.27]        # q^2 of the edge points in convergence.png [GeV^2]
SCALES = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]   # bin size as a fraction of a grid bin
M_GRID = np.linspace(XMIN, XMAX, 521)     # masses the factors are drawn at [MeV]

M_MU2 = (kinematics.M_MU / 1000.0)**2     # m_mu^2 [GeV^2]
M_ETA2 = (kinematics.M_ETA / 1000.0)**2   # m_eta^2 [GeV^2]
C = plot_style.COLORS
FLAT = dict(color=C['phase_space'], linestyle='-', lw=2, label='flat phase space')
DALITZ = dict(color=C['fit'], linestyle='--', lw=2, label='Dalitz decay')
M_LABEL = r'$M = m(\gamma\mu^+\mu^-)$ [MeV]'
M12_LABEL = r'$m^2_{\gamma\mu^+}$ [GeV$^2$]'
Q2_LABEL = r'$q^2 = m^2_{\mu^+\mu^-}$ [GeV$^2$]'

parser = argparse.ArgumentParser(description='Flat vs Dalitz-decay factor figures')
parser.add_argument('--outdir', default=os.path.join(_here, '..', '..', 'out', 'signal_factor'))
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
plot_utils.apply_font()
plt.rcParams['font.size'] = 12

# --- density.png -------------------------------------------------------------

fig, (ax_map, ax_q2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Density over the plane, shown only inside D(m_eta), on a log colour scale.
m12_grid, q2_grid = np.meshgrid(np.linspace(0.0, 0.21, 500), np.linspace(4 * M_MU2, 0.31, 500))
m_min, m_max = kinematics.m123_limits(m12_grid, q2_grid)
inside = (m_min <= kinematics.M_ETA) & (kinematics.M_ETA <= m_max)
density = np.where(inside, kinematics.dalitz_density(m12_grid, q2_grid), np.nan)
mesh = ax_map.pcolormesh(m12_grid, q2_grid, density / np.nanmax(density), cmap='Purples',
                         norm=LogNorm(vmin=1e-3, vmax=1), shading='auto')
fig.colorbar(mesh, ax=ax_map, label='Dalitz-decay density (max = 1)')
boundary12, boundary23 = dalitz_boundary(kinematics.M_ETA)
ax_map.plot(boundary12, boundary23, color=C['data'], lw=1)
ax_map.set_xlim(0.0, 0.21)
ax_map.set_ylim(0.0, 0.31)
ax_map.set_xlabel(M12_LABEL)
ax_map.set_ylabel(Q2_LABEL)

# q^2 spectrum: the density summed across D(m_eta) at each q^2, against the
# length of D(m_eta) at that q^2 (flat phase space); both of unit area.
q2 = np.linspace(4 * M_MU2, M_ETA2, 600)[1:-1]
beta = np.sqrt(1 - 4 * M_MU2 / q2)                       # muon speed in the mu+ mu- frame
lo12 = M_MU2 + (M_ETA2 - q2) * (1 - beta) / 2            # m12 edges of D(m_eta) at each q^2
hi12 = M_MU2 + (M_ETA2 - q2) * (1 + beta) / 2
flat_spectrum = hi12 - lo12
dalitz_spectrum = np.array([np.trapezoid(kinematics.dalitz_density(np.linspace(lo, hi, 400),
                                                                   np.full(400, value)),
                                         np.linspace(lo, hi, 400))
                            for lo, hi, value in zip(lo12, hi12, q2)])
ax_q2.plot(q2, flat_spectrum / np.trapezoid(flat_spectrum, q2), **FLAT)
ax_q2.plot(q2, dalitz_spectrum / np.trapezoid(dalitz_spectrum, q2), **DALITZ)
ax_q2.set_xlabel(Q2_LABEL)
ax_q2.set_ylabel(r'$d\Gamma/dq^2$ (unit area) [GeV$^{-2}$]')
ax_q2.set_ylim(bottom=0)
ax_q2.legend(loc='upper right')
for ax in (ax_map, ax_q2):
    plot_style.style_axes(ax)
plot_style.save_plot(fig, os.path.join(args.outdir, 'density.png'), dpi=150)
plt.close(fig)

# --- factors_grid.png --------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
for ax, (i, j) in zip(axes.flat, GRID_BINS):
    m12_range, m23_range = (M12_EDGES[i], M12_EDGES[i + 1]), (M23_EDGES[j], M23_EDGES[j + 1])
    ax.plot(M_GRID, kinematics.phase_space_factor(M_GRID, m12_range, m23_range), **FLAT)
    ax.plot(M_GRID, kinematics.dalitz_factor(M_GRID, m12_range, m23_range), **DALITZ)
    ax.axvline(kinematics.M_ETA, color=C['reference'], linestyle=':', lw=1)
    ax.set_title(f'bin ({i}, {j})', fontsize=12)
    ax.set_ylim(-0.03, 1.08)
    plot_style.style_axes(ax)
axes[0, 0].legend(loc='lower right', fontsize=10)
for ax in axes[1]:
    ax.set_xlabel(M_LABEL)
for ax in axes[:, 0]:
    ax.set_ylabel('$f(M)$')
plot_style.save_plot(fig, os.path.join(args.outdir, 'factors_grid.png'), dpi=150)
plt.close(fig)

# --- convergence.png ---------------------------------------------------------

# Colour by q^2, solid for the left (low-m12) edge of D(m_eta), dashed for the right.
fig, ax = plt.subplots(figsize=(8, 6))
for q2_0, color in zip(EDGE_Q2, [C['fit'], C['background'], C['pull'], C['phase_space']]):
    beta0 = np.sqrt(1 - 4 * M_MU2 / q2_0)
    edges12 = [M_MU2 + (M_ETA2 - q2_0) * (1 - beta0) / 2, M_MU2 + (M_ETA2 - q2_0) * (1 + beta0) / 2]
    for m12_0, linestyle, side in zip(edges12, ['-', '--'], ['left', 'right']):
        max_diffs = []
        for scale in SCALES:
            width12, width23 = 0.015 * scale, 0.31 / 14 * scale
            m12_range = (m12_0 - width12 / 2, m12_0 + width12 / 2)
            m23_range = (q2_0 - width23 / 2, q2_0 + width23 / 2)
            max_diffs.append(np.max(np.abs(kinematics.dalitz_factor(M_GRID, m12_range, m23_range)
                                           - kinematics.phase_space_factor(M_GRID, m12_range,
                                                                           m23_range))))
        ax.plot(SCALES, max_diffs, marker='o', markersize=5, color=color, linestyle=linestyle,
                lw=1.6, label=f'$q^2$ = {q2_0:.2f} GeV$^2$, {side} edge')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(SCALES)
ax.set_xticklabels(['1', '1/2', '1/4', '1/8', '1/16'])
plot_style.style_axes(ax)
ax.minorticks_off()
ax.set_xlabel('bin size (fraction of a grid bin, both axes)')
ax.set_ylabel(r'max over $M$ of |Dalitz $-$ flat|')
ax.legend(loc='lower right', fontsize=8, ncol=2)
plot_style.save_plot(fig, os.path.join(args.outdir, 'convergence.png'), dpi=150)
plt.close(fig)

# --- factors_q2.png ----------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, j in zip(axes.flat, STRIPS):
    m23_range = (M23_EDGES[j], M23_EDGES[j + 1])
    for factor, style in [(kinematics.phase_space_factor, FLAT), (kinematics.dalitz_factor, DALITZ)]:
        values = factor(M_GRID, STRIP_M12, m23_range)
        ax.plot(M_GRID, values / np.interp(kinematics.M_ETA, M_GRID, values), **style)
    ax.axvline(kinematics.M_ETA, color=C['reference'], linestyle=':', lw=1)
    ax.set_title(f'strip {j}: $q^2$ in [{m23_range[0]:.3f}, {m23_range[1]:.3f}] GeV$^2$',
                 fontsize=11)
    ax.set_ylim(bottom=0)
    plot_style.style_axes(ax)
axes[0, 0].legend(loc='upper left', fontsize=10)
for ax in axes[1]:
    ax.set_xlabel(M_LABEL)
for ax in axes[:, 0]:
    ax.set_ylabel(r'$f(M) / f(m_\eta)$')
plot_style.save_plot(fig, os.path.join(args.outdir, 'factors_q2.png'), dpi=150)
plt.close(fig)
