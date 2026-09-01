"""
bkg_utils.py

Derive (c0, c1) limits for the 2nd-order Chebyshev background in fit_utils.py
that guarantee the background PDF never goes negative inside the fit window.

TODO: This script doesn't actually do anything. It just derives the safe limits
for the 2nd-order Chebyshev background in fit_utils.py, and prints them to
stdout. Will probably delete this script or refactor to generally return (set)
the safe limits for any PDF in the future.

The minimum of a quadratic on a closed interval occurs at an endpoint or at
its vertex (if the vertex falls inside the interval); taking the min over
{s(-1), s(1), s(vertex) if in range} always gives the true minimum on [-1, 1],
regardless of whether the parabola opens up or down. Working through that:

  - s(u) is symmetric under (c0, u) -> (-c0, -u), so the feasible c0 range for
    a given c1 is always [-c0_max(c1), +c0_max(c1)].
  - s(0) = 1 - c1, so c1 > 1 is infeasible for ANY c0 -- the curve dips below
    zero at bin center regardless of the linear term.
  - For c1 outside [-1, 1], no c0 keeps s(u) >= 0 everywhere.
  - For c1 in [-1, 1/3]: the binding constraint is the endpoints,
        c0_max(c1) = 1 + c1
  - For c1 in [1/3, 1]: the vertex falls inside [-1, 1] and is the binding
    constraint,
        c0_max(c1) = sqrt(8 * c1 * (1 - c1))
    (The two pieces agree at c1 = 1/3: both give 4/3.)

Largest symmetric box:
-----------------------
Minuit's `limits` are independent per-parameter (lo, hi) boxes, but the exact
region above is not a rectangle. The largest box of the form
[-a, a] x [-a, a] entirely inside the exact region is found by requiring the
box's worst corner to just touch the boundary: c0_max(-a) = a, i.e.
1 + (-a) = a  =>  a = 0.5 (this lands in the c1 <= 1/3 linear regime, so the
piecewise formula above applies directly with no further case-splitting).
So |c0| <= 0.5 and |c1| <= 0.5 is the largest symmetric, window-independent,
per-parameter Minuit limits box that is *guaranteed* safe.
"""

import argparse
import numpy as np

import fit_utils

parser = argparse.ArgumentParser(
    description='Derive safe (c0, c1) limits for the Chebyshev background '
                 'that prevent it from going negative inside the fit window.')
parser.add_argument('--xmin', type=float, default=480.0, help='Lower mass bound [MeV]')
parser.add_argument('--xmax', type=float, default=620.0, help='Upper mass bound [MeV]')
parser.add_argument('--nbins', type=int, default=80, help='Number of histogram bins')
args = parser.parse_args()


def c0_max(c1):
    """Largest c0 (feasible range is [-c0_max, c0_max]) keeping the 2nd-order
    Chebyshev shape non-negative over u in [-1, 1]. None if c1 itself makes
    every c0 infeasible."""
    if c1 < -1.0 or c1 > 1.0:
        return None
    if c1 <= 1.0 / 3.0:
        return 1.0 + c1
    return np.sqrt(8.0 * c1 * (1.0 - c1))


print('=' * 70)
print('Exact feasibility boundary: c0 in [-c0_max(c1), +c0_max(c1)]')
print('(window-independent -- depends only on c0, c1)')
print('=' * 70)
print(f'{"c1":>8} | {"c0_max":>10}')
for c1 in [-1.0, -0.75, -0.5, -1/3, 0.0, 1/3, 0.5, 2/3, 0.75, 1.0]:
    print(f'{c1:8.4f} | {c0_max(c1):10.4f}')
print(f'{"|c1|>1":>8} | infeasible for any c0 (background hits 0 at bin '
      f'center regardless of c0)')

a = 0.5
print()
print('=' * 70)
print('Largest symmetric box safe for Minuit limits=dict(c0=(...), c1=(...))')
print('=' * 70)
print(f'  c0: ({-a}, {a})')
print(f'  c1: ({-a}, {a})')
print('  (touches zero, not negative, only at the corner c1=-0.5, c0=+-0.5, '
       'at the respective opposite window edge)')

# --- Numeric verification against the real cheb_bkg_pdf, on a grid much finer
# than nbins, over the *continuous* window -- bin centers alone would miss the
# worst case, which sits exactly at the window edges for c1 < 0.
x_fine = np.linspace(args.xmin, args.xmax, 20001)


def min_pdf(c0, c1):
    return float(np.min(fit_utils.cheb_bkg_pdf(x_fine, args.xmin, args.xmax, c0, c1)))


print()
print('=' * 70)
print(f'Numeric check on cheb_bkg_pdf over [{args.xmin}, {args.xmax}] '
      f'({x_fine.size} pts)')
print('=' * 70)
safe_corner = (a, -a)     # worst corner of the derived safe box
# Two ways to land outside the *exact* feasible region (not just outside the
# safe box -- c0=1.0, c1=0.8 is outside the box but still inside the true
# region, since c0_max(0.8) = sqrt(8*0.8*0.2) = 1.131 > 1.0):
#   1. c0 too large for its c1 (c0_max(0.5) = sqrt(2) = 1.414 < 2.0)
#   2. c1 alone outside [-1, 1] -- infeasible for ANY c0, since s(0) = 1 - c1
outside_c0 = (2.0, 0.5)
outside_c1 = (0.0, 1.2)
print(f'  safe box corner   c0={safe_corner[0]:+.2f}, c1={safe_corner[1]:+.2f} '
      f'-> min(pdf) = {min_pdf(*safe_corner):.6f}  (>= 0, as predicted)')
print(f'  c0 too large      c0={outside_c0[0]:+.2f}, c1={outside_c0[1]:+.2f} '
      f'(c0_max={c0_max(outside_c0[1]):.4f}) -> min(pdf) = {min_pdf(*outside_c0):.6f}  '
      f'(negative -> unphysical)')
print(f'  c1 out of [-1,1]  c0={outside_c1[0]:+.2f}, c1={outside_c1[1]:+.2f} '
      f'(infeasible for any c0) -> min(pdf) = {min_pdf(*outside_c1):.6f}  '
      f'(negative -> unphysical)')

# --- Sanity table at the actual fit's bin edges/centers, per the requested
# --xmin/--xmax/--nbins interface. This does NOT change the limits above (the
# math is window-independent) -- it's just a concrete look at what the real
# fit_gauss.py binning would see for the safe-box corner vs. the outside example.
_, edges, _, _ = fit_utils.make_histogram(np.array([]), args.xmin, args.xmax, args.nbins)
centers = 0.5 * (edges[:-1] + edges[1:])
print()
print('=' * 70)
print(f'Bin-center sanity table ({args.nbins} bins, first/last 3 shown)')
print('=' * 70)
for label, (c0, c1) in [('safe box corner', safe_corner), ('c0 too large', outside_c0)]:
    vals = fit_utils.cheb_bkg_pdf(centers, args.xmin, args.xmax, c0, c1)
    shown = list(zip(centers[:3], vals[:3])) + [('...', '...')] + list(zip(centers[-3:], vals[-3:]))
    print(f'  {label} (c0={c0:+.2f}, c1={c1:+.2f}):')
    for c, v in shown:
        if c == '...':
            print('    ...')
        else:
            print(f'    x={c:8.3f}  pdf={v: .6f}')
