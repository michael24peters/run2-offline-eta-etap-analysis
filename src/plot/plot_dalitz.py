"""
plot_dalitz.py

Raw Dalitz histograms (no fit) of the candidates driver_dalitz.py selects,
with the eta's Dalitz region. driver_dalitz.py writes HISTOGRAMS:

    dalitz_raw.png       m12 in [0, 1.0], m23 in [0, 1.2] GeV^2 (every candidate)
    dalitz_raw_zoom.png  m12 in [0, 0.22], m23 in [0, 0.35] GeV^2 (the eta's region)
"""

import os
import sys

import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import kinematics
import plot_style
import plot_utils
from plot_phase_space import dalitz_boundary

# (file name, [[m12_lo, m12_hi], [m23_lo, m23_hi]]) [GeV^2] of each histogram
HISTOGRAMS = [
    ('dalitz_raw.png', [[0, 1.0], [0, 1.2]]),
    ('dalitz_raw_zoom.png', [[0, 0.22], [0, 0.35]]),
]
N_BINS = 100   # per axis


def plot_dalitz_hist(m12s, m23s, hist_range, output, bins=N_BINS):
    """2D histogram of (m12s, m23s) [GeV^2] over hist_range, with the eta's
    Dalitz region, saved to output."""
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 11
    fig, ax = plt.subplots(figsize=(8, 6))
    *_, mesh = ax.hist2d(m12s, m23s, bins=bins, cmap='viridis', range=hist_range)
    fig.colorbar(mesh, ax=ax, label='Counts')
    boundary12, boundary23 = dalitz_boundary(kinematics.M_ETA)
    ax.plot(boundary12, boundary23, color='red', lw=1, label=r'Dalitz region at $m_\eta$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel(r'$m^2_{\gamma\mu^+}$ [GeV$^2$]')
    ax.set_ylabel(r'$m^2_{\mu^+\mu^-}$ [GeV$^2$]')
    plot_style.save_plot(fig, output)
    plt.close(fig)

