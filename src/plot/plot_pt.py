"""
plot_pt.py

Shared pT-histogram plotting library: single-species and multi-species overlay histograms in the
custom style. Used by plot_mc_pt.py and plot_reco_pt.py.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import plot_style
import plot_utils


def _new_axes():
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 16
    return plt.subplots(figsize=(8, 6))

def plot_pt_hist(pt, xlabel, output, xmax, bins=100, entries='Particles', sample=''):
    """Single-species pT [MeV] histogram: black step, log-scale y, the y axis `entries` per bin,
    the text `sample` (plot_style.sample_label) in the upper left."""
    fig, ax = _new_axes()
    edges = np.linspace(0, xmax, bins + 1)
    counts, _, _ = ax.hist(pt, bins=edges, histtype='step', color=plot_style.COLORS['data'])
    ax.set_yscale('log')
    plot_style.style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(plot_style.count_label(entries, edges, 'MeV'))
    sample_box = plot_style.add_stats_box(ax, '', sample=sample)
    plot_style.reserve_top_margin(ax, max(counts.max(), 1), [sample_box], ymin=0.5, growth=2.0,
                                  max_iter=12)
    plot_style.save_plot(fig, output)
    plt.close(fig)

def plot_pt_overlay(series, xlabel, output, xmax, bins=100, entries='Particles', sample=''):
    """
    Multiple species overlaid on one pT [MeV] histogram, the y axis `entries` per bin, the text
    `sample` (plot_style.sample_label) in the upper left.

    series : list of (label, pt_array, color) tuples.
    """
    fig, ax = _new_axes()
    edges = np.linspace(0, xmax, bins + 1)
    top = 1
    for label, pt, color in series:
        counts, _, _ = ax.hist(pt, bins=edges, histtype='step', color=color, label=label)
        top = max(top, counts.max())
    ax.set_yscale('log')
    plot_style.style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(plot_style.count_label(entries, edges, 'MeV'))
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='none',
                       framealpha=1)
    sample_box = plot_style.add_stats_box(ax, '', sample=sample)
    plot_style.reserve_top_margin(ax, top, [sample_box, legend], ymin=0.5, growth=2.0,
                                  max_iter=12)
    plot_style.save_plot(fig, output)
    plt.close(fig)
