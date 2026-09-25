"""
plot_mc_pt.py

MC truth transverse momentum (pT) for the eta and its mu+, mu-, gamma
daughters, from a signal eta -> mu mu gamma ROOT ntuple.

Usage:
    python plot_mc_pt.py input.root [output_dir]
"""

import argparse
import os
import sys

import awkward as ak

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils
import kinematics
import plot_style

sys.path.insert(0, _here)
import plot_pt

# (file-name suffix, PID, LaTeX symbol, x-axis max [MeV/c])
PARTICLES = [
    ('eta',      221, r'\eta',    30000),
    ('mu_plus',  -13, r'\mu^+',   15000),
    ('mu_minus', 13,  r'\mu^-',   15000),
    ('gamma',    22,  r'\gamma',  15000),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot MC truth pT for eta -> mu mu gamma and its daughters')
    parser.add_argument('input', help='Input ROOT file')
    parser.add_argument('output_dir', nargs='?',
                        default=os.path.join(_here, '..', '..', 'out', 'quick_plots'),
                        help='Output directory (default: out/quick_plots)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Reading from {args.input}.')

    tree = cut_utils.read_branches(args.input, ['mc_pid', 'mc_px', 'mc_py', 'mc_idx_mom'])

    # Signal etas: generator particles with exactly one mu+, one mu- and one
    # gamma daughter. daughter_mask brings that mother-level selection down to
    # the daughter level via mc_idx_mom (see cut_utils.follow), so each
    # daughter species can then be picked out by PID.
    eta_mask = cut_utils.decay_mask(tree, 221, [-13, 13, 22])
    daughter_mask = (cut_utils.follow(tree, eta_mask, 'mc_idx_mom')
                     & cut_utils.valid(tree, 'mc_idx_mom'))
    pt_all = kinematics.pt(tree['mc_px'], tree['mc_py'])

    masks = {
        'eta':      eta_mask,
        'mu_plus':  daughter_mask & (tree['mc_pid'] == -13),
        'mu_minus': daughter_mask & (tree['mc_pid'] == 13),
        'gamma':    daughter_mask & (tree['mc_pid'] == 22),
    }

    channel = fr'${plot_style.CHANNEL_LABELS["eta2mumugamma"]}$'
    pts = {}
    for name, pid, symbol, xmax in PARTICLES:
        pt = ak.to_numpy(ak.flatten(pt_all[masks[name]]))
        pts[name] = pt
        output = os.path.join(args.output_dir, f'mc_pt_{name}.png')
        plot_pt.plot_pt_hist(pt, fr'$p_T({symbol})$ [MeV/$c$]',
                              fr'MC truth $p_T({symbol})$, {channel}', output, xmax)
        print(f'[INFO] {name} (PID {pid}): n={len(pt)}, '
              f'p_T range [{pt.min():.2f}, {pt.max():.2f}] MeV/c')

    dimuon_output = os.path.join(args.output_dir, 'mc_pt_dimuon.png')
    plot_pt.plot_pt_overlay(
        [(r'$\mu^+$', pts['mu_plus'], plot_style.COLORS['fit']),
         (r'$\mu^-$', pts['mu_minus'], plot_style.COLORS['pull'])],
        r'$p_T(\mu)$ [MeV/$c$]', fr'MC truth $p_T(\mu)$, {channel}', dimuon_output, xmax=15000)
    print(f'[INFO] dimuon overlay saved to {dimuon_output}')
