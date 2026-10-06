"""
plot_phase_space.py

Figures for src/fit/phase-space-factor.md, explaining the Dalitz phase-space
factor f(M).

Usage:
    python src/plot/plot_phase_space.py \
        [--results out/eta2mumugamma/turbo/20260922/dalitz_fit_results.json] \
        [--events  out/eta2mumugamma/turbo/20260922/dalitz_events.npz] \
        [--outdir  out/phase_space_factor]

Pure kinematics (always made): f(m_eta) map, edge-bin zooms, the step-by-step
construction of f for one bin, and f(M) for a set of example bins.
With --results: per-bin decompositions (what f and renormalization do to the
signal and background shapes). With --events: the within-bin uniformity
figure, which needs the candidates' Dalitz positions.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import fit_utils
import kinematics
import plot_style
import plot_utils

MMU2 = (kinematics.M_MU / 1000.0)**2
XMIN, XMAX = 420.0, 680.0
M12_EDGES = np.linspace(0.0, 0.21, 15)   # the notebook's Dalitz grid
M23_EDGES = np.linspace(0.0, 0.31, 15)
C = plot_style.COLORS
M12_LABEL = r'$m^2_{12} = m^2(\gamma\mu^+)$ [GeV$^2$]'
M23_LABEL = r'$m^2_{23} = m^2(\mu^+\mu^-)$ [GeV$^2$]'
M_LABEL = r'$M = m(\gamma\mu^+\mu^-)$ [MeV]'

# Example bins, (i, j) = (m12 index, m23 index).
EDGE_BINS = [(1, 13), (2, 12), (2, 4), (7, 8), (7, 2), (12, 5)]
FACTOR_BINS = [(4, 4), (6, 8), (7, 2), (1, 13), (1, 4), (12, 7)]
DECOMP_BINS = [(4, 4), (6, 8), (7, 2), (1, 13), (12, 7)]
CONSTRUCTION_BIN, CONSTRUCTION_MASSES = (6, 8), [505.0, 520.0, 530.0, 540.0, 550.0]
UNIFORMITY_BIN = (1, 8)
MASS_SLICES = [(475.0, 485.0), (525.0, 535.0), (545.0, 551.0), (640.0, 650.0)]
MASS_POINTS, MASS_HALF_WIDTH = [475.0, 530.0, 548.0, 645.0], 1.0   # [MeV]
WORKED_POINT = (0.020, 0.300)   # (m12, m23) [GeV^2] of the worked example
STRIPS_MASS, STRIPS_N = 530.0, 20


def style():
    plot_utils.apply_font()
    plt.rcParams['font.size'] = 13


def bin_ranges(i, j):
    return (M12_EDGES[i], M12_EDGES[i + 1]), (M23_EDGES[j], M23_EDGES[j + 1])


def f_of(M, i, j):
    return kinematics.phase_space_factor(np.atleast_1d(M), *bin_ranges(i, j))


def inside_region(m12, m23, M):
    """True where the Dalitz point (m12, m23) [GeV^2] is kinematically
    reachable by a system of mass M [MeV] -- eq. (1) of the write-up."""
    M2 = (M / 1000.0)**2
    ok = (m23 > 4 * MMU2) & (m23 < M2)
    beta = np.sqrt(np.clip(1 - 4 * MMU2 / np.where(ok, m23, 1.0), 0, None))
    half = np.clip(M2 - m23, 0, None) / 2
    return ok & (m12 >= MMU2 + half * (1 - beta)) & (m12 <= MMU2 + half * (1 + beta))


def dalitz_boundary(M, n=3000):
    """Closed (m12, m23) boundary [GeV^2] of the Dalitz region at mass M."""
    M2 = (M / 1000.0)**2
    s = np.linspace(4 * MMU2, M2, n)
    beta = np.sqrt(np.clip(1 - 4 * MMU2 / s, 0, None))
    lo = MMU2 + (M2 - s) * (1 - beta) / 2
    hi = MMU2 + (M2 - s) * (1 + beta) / 2
    return np.concatenate([lo, hi[::-1]]), np.concatenate([s, s[::-1]])


def draw_bin_at_mass(ax, i, j, M, margin=0.6, n=400):
    """Zoom on bin (i, j): the Dalitz region at mass M in light grey, the
    part of the bin inside it in violet, the bin outline in black."""
    (a, b), (lo, hi) = bin_ranges(i, j)
    w, h = b - a, hi - lo
    x0, x1 = max(a - margin * w, 0), b + margin * w
    y0, y1 = max(lo - margin * h, 0), hi + margin * h
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    inside = inside_region(X, Y, M)
    in_bin = (X >= a) & (X <= b) & (Y >= lo) & (Y <= hi)
    img = np.ones(X.shape + (4,))                        # white
    img[inside] = matplotlib.colors.to_rgba('#e4e3df')   # region: light grey
    img[inside & in_bin] = matplotlib.colors.to_rgba(C['phase_space'], 0.75)
    ax.imshow(img, origin='lower', extent=(x0, x1, y0, y1), aspect='auto',
              interpolation='nearest')
    bx, by = dalitz_boundary(M)
    ax.plot(bx, by, color=C['data'], lw=1.2)
    ax.add_patch(Rectangle((a, lo), w, h, fill=False, ec=C['data'], lw=1.6))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    plot_style.style_axes(ax)


def fig_f_at_meta_map(outdir):
    f_eta = np.array([[f_of(kinematics.M_ETA, i, j)[0] for j in range(14)]
                      for i in range(14)])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mesh = ax.pcolormesh(M12_EDGES, M23_EDGES, f_eta.T, cmap='Purples',
                         vmin=0, vmax=1, edgecolors='white', linewidth=0.3)
    fig.colorbar(mesh, ax=ax, label=r'$f(m_\eta)$', pad=0.13)
    bx, by = dalitz_boundary(kinematics.M_ETA)
    ax.plot(bx, by, color=C['data'], lw=1.4, label=r'$\eta$ Dalitz boundary')
    for (i, j) in EDGE_BINS:
        (a, b), (lo, hi) = bin_ranges(i, j)
        ax.add_patch(Rectangle((a, lo), b - a, hi - lo, fill=False,
                               ec=C['pull'], lw=1.8))
        right = b > 0.17   # label on the left near the plot's right edge
        ax.text(a - 0.002 if right else b + 0.002, hi - 0.002, f'({i},{j})',
                color=C['data'], fontsize=10, va='top', ha='right' if right else 'left')
    ax.set_xlabel(M12_LABEL); ax.set_ylabel(M23_LABEL)
    ax.legend(loc='upper right', fontsize=10)
    plot_style.style_axes(ax)
    plot_style.add_bin_index_axes(ax, M12_EDGES, M23_EDGES)
    plot_style.save_plot(fig, os.path.join(outdir, 'f_at_meta_map.png'), dpi=150)


def fig_edge_bins(outdir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (i, j) in zip(axes.flat, EDGE_BINS):
        draw_bin_at_mass(ax, i, j, kinematics.M_ETA)
        ax.set_title(fr'Bin ({i},{j}):  $f(m_\eta)$ = {f_of(kinematics.M_ETA, i, j)[0]:.2f}',
                     fontsize=12)
        ax.set_xlabel(r'$m^2_{12}$ [GeV$^2$]'); ax.set_ylabel(r'$m^2_{23}$ [GeV$^2$]')
        ax.tick_params(labelsize=9)
    handles = [Rectangle((0, 0), 1, 1, fc=C['phase_space'], alpha=0.75),
               Rectangle((0, 0), 1, 1, fc='#e4e3df'),
               Rectangle((0, 0), 1, 1, fill=False, ec=C['data'], lw=1.6)]
    fig.legend(handles, [r'part of the bin reachable at $M = m_\eta$',
                         r'rest of the $\eta$ Dalitz region',
                         'Dalitz bin'],
               loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(outdir, 'edge_bins.png'), dpi=150)
    print(f"[DONE] Plot saved to {os.path.join(outdir, 'edge_bins.png')}.")


def fig_construction(outdir):
    i, j = CONSTRUCTION_BIN
    masses = CONSTRUCTION_MASSES
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, len(masses), height_ratios=[1.1, 1], figure=fig)
    for k, M in enumerate(masses):
        ax = fig.add_subplot(gs[0, k])
        draw_bin_at_mass(ax, i, j, M, margin=0.9)
        ax.set_title(f'M = {M:.0f} MeV\nf = {f_of(M, i, j)[0]:.2f}', fontsize=12)
        ax.set_xlabel(r'$m^2_{12}$ [GeV$^2$]', fontsize=10)
        if k == 0: ax.set_ylabel(r'$m^2_{23}$ [GeV$^2$]', fontsize=10)
        ax.tick_params(labelsize=8)
    ax = fig.add_subplot(gs[1, :])
    Mg = np.linspace(XMIN, XMAX, 1500)
    ax.plot(Mg, f_of(Mg, i, j), color=C['phase_space'], lw=2, label=f'f(M) for bin ({i},{j})')
    ax.plot(masses, f_of(masses, i, j), 'o', color=C['phase_space'], ms=9,
            mfc='white', mew=2, label='the five masses above')
    ax.axvline(kinematics.M_ETA, color=C['reference'], ls='--', label=r'$m_\eta$')
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel(M_LABEL); ax.set_ylabel('f(M)')
    ax.legend(loc='lower right', fontsize=11)
    plot_style.style_axes(ax)
    fig.tight_layout()
    out = os.path.join(outdir, f'f_construction_{i}_{j}.png')
    fig.savefig(out, dpi=150); print(f'[DONE] Plot saved to {out}.')


def fig_factors(outdir):
    Mg = np.linspace(XMIN, XMAX, 1500)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    for ax, (i, j) in zip(axes.flat, FACTOR_BINS):
        f = f_of(Mg, i, j)
        ax.plot(Mg, f, color=C['phase_space'], lw=2)
        ax.axvline(kinematics.M_ETA, color=C['reference'], ls='--')
        ax.set_title(fr'Bin ({i},{j}):  $f(m_\eta)$ = {f_of(kinematics.M_ETA, i, j)[0]:.2f}',
                     fontsize=12)
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(-0.05, 1.08)
        plot_style.style_axes(ax)
    for ax in axes[1]: ax.set_xlabel('M [MeV]')
    for ax in axes[:, 0]: ax.set_ylabel('f(M)')
    plot_style.save_plot(fig, os.path.join(outdir, 'phase_space_factors.png'), dpi=150)


def fig_decomposition(outdir, results, i, j):
    """Bin (i, j)'s f, and what multiplying by f and renormalizing does to
    its background (fitted c0, c1) and signal (global-fit DCB) shapes."""
    g = results['fit']['parameters']
    entry = next(b for b in results['dalitz']['bins'] if (b['i'], b['j']) == (i, j))
    dcb = [g[p]['value'] for p in ('mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR')]
    weight = fit_utils.make_weight(lambda M: f_of(M, i, j), XMIN, XMAX)
    xg, wg = weight
    wmodel = fit_utils.weighted_bkg_model(fit_utils.BACKGROUND_MODELS['cheb2'], weight)
    c = (entry['c0'], entry['c1'])

    cheb = np.clip(wmodel.unnorm(xg, XMIN, XMAX, *c), 0, None)
    cheb /= fit_utils._trapz(cheb, xg)
    dcb_alone = fit_utils.dcb_pdf(xg, XMIN, XMAX, *dcb)
    shapes = [
        ('Background', C['background'], cheb, 'Chebyshev',
         wmodel.pdf(xg, XMIN, XMAX, *c)),
        ('Signal', C['fit'], dcb_alone, 'DCB',
         fit_utils.weighted_dcb_pdf(weight)(xg, XMIN, XMAX, *dcb)),
    ]
    # A bin whose n_sig is fixed to 0 has no signal shape to show.
    if entry.get('n_sig_fixed_zero'):
        shapes = shapes[:1]
    fig, axes = plt.subplots(len(shapes) + 1, 1, figsize=(10, 4 + 3.5 * len(shapes)),
                             sharex=True, gridspec_kw={'height_ratios': [1] + [2] * len(shapes)})
    axes[0].plot(xg, wg, color=C['phase_space'], lw=2, label='$f(M)$')
    axes[0].set_ylim(-0.05, 1.1); axes[0].set_ylabel('$f(M)$')
    axes[0].legend(loc='lower right', fontsize=11)
    for ax, (name, color, alone, label, renorm) in zip(axes[1:], shapes):
        kept = fit_utils._trapz(wg * alone, xg)
        ax.plot(xg, alone, color=color, ls='--', lw=1.6, label=f'{label} alone (area 1)')
        ax.plot(xg, wg * alone, color=color, ls=':', lw=2,
                label=fr'$f \times$ {label}, before renormalizing (area {kept:.2f})')
        ax.plot(xg, renorm, color=color, lw=2.2,
                label=fr'$f \times$ {label}, renormalized (area 1)')
        ax.set_ylabel(f'{name} pdf [1/MeV]')
        ax.set_ylim(0, 1.45 * max(alone.max(), renorm.max()))   # headroom for the legend
        ax.legend(loc='upper left' if xg[np.argmax(renorm)] > 560 else 'upper right',
                  fontsize=10)
    for ax in axes:
        ax.axvline(kinematics.M_ETA, color=C['reference'], ls='--', lw=1)
        plot_style.style_axes(ax)
    axes[-1].set_xlim(XMIN, XMAX); axes[-1].set_xlabel(M_LABEL)
    plot_style.save_plot(fig, os.path.join(outdir, f'decomposition_{i}_{j}.png'), dpi=150)


def fig_worked_example(outdir, events):
    """The worked-example Dalitz point against D(M) at six masses, and the
    masses of the candidates reconstructed near it."""
    x, y = WORKED_POINT
    m_min, m_max = (float(v) for v in kinematics.m123_limits(x, y))
    masses = [kinematics.M_ETA, m_min, 620.0, 680.0, m_max, 760.0]
    names = [r' ($m_\eta$)', r' ($M_\mathrm{min}$)', '', '', r' ($M_\mathrm{max}$)', '']
    x0, x1, y0, y1 = 0.005, 0.040, 0.270, 0.320
    fig = plt.figure(figsize=(18, 8.5))
    gs = GridSpec(2, len(masses), height_ratios=[1.15, 1], figure=fig)
    X, Y = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 400))
    for k, (M, name) in enumerate(zip(masses, names)):
        ax = fig.add_subplot(gs[0, k])
        img = np.ones(X.shape + (4,))
        img[inside_region(X, Y, M)] = matplotlib.colors.to_rgba('#e4e3df')
        ax.imshow(img, origin='lower', extent=(x0, x1, y0, y1), aspect='auto',
                  interpolation='nearest')
        ax.plot(*dalitz_boundary(M), color=C['data'], lw=1.2)
        ax.plot(x, y, 'o', color=C['pull'], ms=8, mec=C['data'], mew=0.8)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([0.01, 0.02, 0.03])
        ax.set_title(f'M = {M:.1f} MeV{name}', fontsize=12)
        ax.set_xlabel(r'$m^2_{12}$ [GeV$^2$]', fontsize=10)
        if k == 0: ax.set_ylabel(r'$m^2_{23}$ [GeV$^2$]', fontsize=10)
        else: ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=9)
        plot_style.style_axes(ax)

    ev = np.load(events)
    m12, m23, Mc = (ev[k].astype(float) for k in ('m12s', 'm23s', 'masses'))
    near = (abs(m12 - x) < 0.001) & (abs(m23 - y) < 0.002)
    ax = fig.add_subplot(gs[1, :])
    lo_axis, hi_axis = 520.0, 780.0
    ax.axvspan(m_min, m_max, color=C['phase_space'], alpha=0.15, lw=0,
               label=fr'$[M_\mathrm{{min}}, M_\mathrm{{max}}]$ = [{m_min:.1f}, {m_max:.1f}] MeV')
    ax.axvspan(XMAX, hi_axis, fc='none', ec=C['reference'], hatch='//', lw=0,
               label='outside the mass window')
    ax.hist(Mc[near], bins=np.arange(lo_axis, hi_axis + 1, 5), histtype='step',
            color=C['data'], lw=1.8, label=f'{near.sum()} candidates near the point')
    ax.axvline(kinematics.M_ETA, color=C['reference'], ls='--', label=r'$m_\eta$')
    ax.set_xlim(lo_axis, hi_axis)
    ax.set_xlabel(M_LABEL); ax.set_ylabel('Candidates / 5 MeV')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc='upper right', fontsize=11)
    plot_style.style_axes(ax)
    fig.tight_layout()
    out = os.path.join(outdir, 'worked_example.png')
    fig.savefig(out, dpi=150); print(f'[DONE] Plot saved to {out}.')


def fig_strips(outdir):
    """How f is measured at one mass: the bin cut into horizontal strips,
    each strip's covered length (violet), and the covered fraction per strip."""
    i, j = CONSTRUCTION_BIN
    (a, b), (lo, hi) = bin_ranges(i, j)
    M = STRIPS_MASS
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6), sharey=True,
                                   gridspec_kw={'width_ratios': [1.4, 1]})
    draw_bin_at_mass(ax0, i, j, M, margin=0.6)
    # Redraw the bin as strips: covered part violet, rest white.
    ax0.add_patch(Rectangle((a, lo), b - a, hi - lo, fc='white', ec='none', zorder=2))
    edges = np.linspace(lo, hi, STRIPS_N + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    M2, h = (M / 1000.0)**2, edges[1] - edges[0]
    beta = np.sqrt(1 - 4 * MMU2 / centres)
    s_lo = MMU2 + (M2 - centres) * (1 - beta) / 2
    s_hi = MMU2 + (M2 - centres) * (1 + beta) / 2
    left, right = np.maximum(s_lo, a), np.minimum(s_hi, b)
    frac = np.clip(right - left, 0, None) / (b - a)
    for y, l, r in zip(edges[:-1], left, right):
        if r > l:
            ax0.add_patch(Rectangle((l, y), r - l, h, fc=C['phase_space'], alpha=0.75,
                                    ec='none', zorder=3))
        ax0.add_patch(Rectangle((a, y), b - a, h, fill=False, ec=C['data'], lw=0.5, zorder=4))
    ax0.add_patch(Rectangle((a, lo), b - a, hi - lo, fill=False, ec=C['data'], lw=1.6, zorder=5))
    ax0.set_xlabel(r'$m^2_{12}$ [GeV$^2$]'); ax0.set_ylabel(r'$m^2_{23}$ [GeV$^2$]')

    ax1.barh(centres, frac, height=0.85 * h, color=C['phase_space'], alpha=0.75)
    ax1.axvline(frac.mean(), color=C['data'], ls='--', lw=1.4,
                label=f'mean over strips = {frac.mean():.2f}')
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel('covered fraction of the strip')
    ax1.legend(loc='upper right', fontsize=11)
    plot_style.style_axes(ax1)
    fig.tight_layout()
    out = os.path.join(outdir, f'f_strips_{i}_{j}.png')
    fig.savefig(out, dpi=150); print(f'[DONE] Plot saved to {out}.')


def fig_mass_slices(outdir, events):
    """The candidates' Dalitz plot in narrow slices of M, each with the
    Dalitz boundaries at the slice's edges, and the construction bin outlined."""
    ev = np.load(events)
    m12, m23, M = (ev[k].astype(float) for k in ('m12s', 'm23s', 'masses'))
    i, j = CONSTRUCTION_BIN
    (a, b), (lo, hi) = bin_ranges(i, j)
    fig, axes = plt.subplots(1, len(MASS_SLICES), figsize=(5 * len(MASS_SLICES), 5.6),
                             sharex=True, sharey=True)
    for ax, (m_lo, m_hi) in zip(axes, MASS_SLICES):
        sel = (M >= m_lo) & (M < m_hi)
        ax.hist2d(m12[sel], m23[sel], bins=(M12_EDGES.size - 1) * 6,
                  range=[[0, M12_EDGES[-1]], [0, M23_EDGES[-1]]],
                  cmap='Greys', cmin=1)
        ax.plot(*dalitz_boundary(kinematics.M_ETA), color=C['reference'], ls='--', lw=1.2)
        for M_edge, ls in ((m_lo, ':'), (m_hi, '-')):
            ax.plot(*dalitz_boundary(M_edge), color=C['phase_space'], ls=ls, lw=1.4)
        ax.add_patch(Rectangle((a, lo), b - a, hi - lo, fill=False, ec=C['pull'], lw=2))
        in_bin = sel & (m12 >= a) & (m12 < b) & (m23 >= lo) & (m23 < hi)
        ax.set_title(f'{m_lo:.0f} < M < {m_hi:.0f} MeV\n'
                     f'{sel.sum()} candidates, {in_bin.sum()} in bin ({i},{j})', fontsize=12)
        ax.set_xlim(0, M12_EDGES[-1]); ax.set_ylim(0, M23_EDGES[-1])
        ax.set_xlabel(r'$m^2_{12}$ [GeV$^2$]')
        plot_style.style_axes(ax)
    axes[0].set_ylabel(r'$m^2_{23}$ [GeV$^2$]')
    handles = [plt.Line2D([], [], color=C['phase_space'], ls=':', lw=1.4),
               plt.Line2D([], [], color=C['phase_space'], ls='-', lw=1.4),
               plt.Line2D([], [], color=C['reference'], ls='--', lw=1.2),
               Rectangle((0, 0), 1, 1, fill=False, ec=C['pull'], lw=2)]
    fig.legend(handles, ['Dalitz boundary at the lower edge of the slice',
                         'Dalitz boundary at the upper edge of the slice',
                         r'Dalitz boundary at $M = m_\eta$',
                         f'bin ({i},{j})'],
               loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(outdir, 'mass_slices.png')
    fig.savefig(out, dpi=150); print(f'[DONE] Plot saved to {out}.')


def fig_mass_points(outdir, events):
    """The candidates' Dalitz plot at single values of M, each with the
    Dalitz boundary at that M, and the construction bin outlined. Candidates
    are those within MASS_HALF_WIDTH of M."""
    ev = np.load(events)
    m12, m23, M = (ev[k].astype(float) for k in ('m12s', 'm23s', 'masses'))
    i, j = CONSTRUCTION_BIN
    (a, b), (lo, hi) = bin_ranges(i, j)
    fig, axes = plt.subplots(1, len(MASS_POINTS), figsize=(5 * len(MASS_POINTS), 5.6),
                             sharex=True, sharey=True)
    for ax, M0 in zip(axes, MASS_POINTS):
        sel = abs(M - M0) < MASS_HALF_WIDTH
        ax.hist2d(m12[sel], m23[sel], bins=(M12_EDGES.size - 1) * 6,
                  range=[[0, M12_EDGES[-1]], [0, M23_EDGES[-1]]],
                  cmap='Greys', cmin=1)
        ax.plot(*dalitz_boundary(kinematics.M_ETA), color=C['reference'], ls='--', lw=1.2)
        ax.plot(*dalitz_boundary(M0), color=C['phase_space'], lw=1.4)
        ax.add_patch(Rectangle((a, lo), b - a, hi - lo, fill=False, ec=C['pull'], lw=2))
        in_bin = sel & (m12 >= a) & (m12 < b) & (m23 >= lo) & (m23 < hi)
        ax.set_title(f'M = {M0:.0f} MeV\n'
                     f'{sel.sum()} candidates with |M - {M0:.0f}| < {MASS_HALF_WIDTH:.0f} MeV,\n'
                     f'{in_bin.sum()} of them in bin ({i},{j})', fontsize=12)
        ax.set_xlim(0, M12_EDGES[-1]); ax.set_ylim(0, M23_EDGES[-1])
        ax.set_xlabel(r'$m^2_{12}$ [GeV$^2$]')
        plot_style.style_axes(ax)
    axes[0].set_ylabel(r'$m^2_{23}$ [GeV$^2$]')
    handles = [plt.Line2D([], [], color=C['phase_space'], lw=1.4),
               plt.Line2D([], [], color=C['reference'], ls='--', lw=1.2),
               Rectangle((0, 0), 1, 1, fill=False, ec=C['pull'], lw=2)]
    fig.legend(handles, ['Dalitz boundary at the panel\'s M',
                         r'Dalitz boundary at $M = m_\eta$',
                         f'bin ({i},{j})'],
               loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(outdir, 'mass_points.png')
    fig.savefig(out, dpi=150); print(f'[DONE] Plot saved to {out}.')


def fig_uniformity(outdir, events, i, j):
    """How the bin's candidates are actually spread across it, and f
    computed from that spread instead of from area."""
    ev = np.load(events)
    (a, b), (lo, hi) = bin_ranges(i, j)
    sel = ((ev['m12s'] >= a) & (ev['m12s'] < b) & (ev['m23s'] >= lo) & (ev['m23s'] < hi))
    m12, m23 = ev['m12s'][sel].astype(float), ev['m23s'][sel].astype(float)
    Mg = np.linspace(XMIN, XMAX, 600)
    lim_lo, lim_hi = kinematics.m123_limits(m12, m23)
    f_data = np.array([np.mean((lim_lo <= M) & (M <= lim_hi)) for M in Mg])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={'width_ratios': [1, 1.3]})
    h = ax0.hist2d(m12, m23, bins=20, range=[[a, b], [lo, hi]], cmap='Greys')
    fig.colorbar(h[3], ax=ax0, label='Candidates')
    ax0.set_xlabel(r'$m^2_{12}$ [GeV$^2$]'); ax0.set_ylabel(r'$m^2_{23}$ [GeV$^2$]')
    ax0.set_title(f'Bin ({i},{j}): where its {sel.sum()} candidates sit')
    plot_style.style_axes(ax0)
    ax1.plot(Mg, f_of(Mg, i, j), color=C['phase_space'], lw=2,
             label='f(M): fraction of the bin\'s area (used in the fit)')
    ax1.plot(Mg, f_data, color=C['data'], ls='--', lw=2,
             label='fraction of the bin\'s candidates (actual spread)')
    ax1.axvline(kinematics.M_ETA, color=C['reference'], ls='--', lw=1)
    ax1.set_xlim(XMIN, XMAX); ax1.set_ylim(-0.05, 1.08)
    ax1.set_xlabel(M_LABEL); ax1.set_ylabel('reachable fraction')
    ax1.set_title('Reachable fraction at each M')
    ax1.legend(loc='lower right', fontsize=10)
    plot_style.style_axes(ax1)
    plot_style.save_plot(fig, os.path.join(outdir, f'uniformity_{i}_{j}.png'), dpi=150)


def main():
    parser = argparse.ArgumentParser(description='Phase-space factor figures.')
    parser.add_argument('--results', default=None,
                        help='dalitz_fit_results.json, for the decomposition figures')
    parser.add_argument('--events', default=None,
                        help='dalitz_events.npz, for the uniformity figure')
    parser.add_argument('--outdir', default=os.path.join(_here, '..', '..', 'out',
                                                         'phase_space_factor'))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    style()
    fig_f_at_meta_map(args.outdir)
    fig_edge_bins(args.outdir)
    fig_construction(args.outdir)
    fig_strips(args.outdir)
    fig_factors(args.outdir)
    if args.results:
        with open(args.results) as fh:
            results = json.load(fh)
        for (i, j) in DECOMP_BINS:
            fig_decomposition(args.outdir, results, i, j)
    if args.events:
        fig_mass_slices(args.outdir, args.events)
        fig_mass_points(args.outdir, args.events)
        fig_worked_example(args.outdir, args.events)
        fig_uniformity(args.outdir, args.events, *UNIFORMITY_BIN)
    plt.close('all')
    print(f'[DONE] figures in {os.path.abspath(args.outdir)}')


if __name__ == '__main__':
    main()
