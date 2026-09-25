"""
plot_pt.py

Shared pT-histogram plotting library: single-species and multi-species overlay
histograms in house style. Used by plot_mc_pt.py and plot_reco_pt.py.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_utils
import plot_style

def _new_axes():
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 16
    return plt.subplots(figsize=(8, 6))

def plot_pt_hist(pt, xlabel, title, output, xmax, bins=100):
    """Single-species pT histogram: black step, log-scale y."""
    fig, ax = _new_axes()
    ax.hist(pt, bins=bins, range=(0, xmax), histtype='step', color=plot_style.COLORS['data'])
    ax.set_yscale('log')
    plot_style.style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entries')
    ax.set_title(title)
    plot_style.save_plot(fig, output)
    plt.close(fig)

def plot_pt_overlay(series, xlabel, title, output, xmax, bins=100):
    """
    Multiple species overlaid on one pT histogram.

    series : list of (label, pt_array, color) tuples.
    """
    fig, ax = _new_axes()
    for label, pt, color in series:
        ax.hist(pt, bins=bins, range=(0, xmax), histtype='step', color=color, label=label)
    ax.set_yscale('log')
    plot_style.style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entries')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='none', framealpha=1)
    plot_style.save_plot(fig, output)
    plt.close(fig)
