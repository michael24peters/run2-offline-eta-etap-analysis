"""
plot_reco_pt.py

Reconstructed transverse momentum (pT) for the mu+, mu-, and gamma daughters,
from a signal eta -> mu mu gamma ROOT ntuple.

Usage:
    python plot_reco_pt.py input.root [output_dir]
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
    ('mu_plus',  -13, r'\mu^+',   15000),
    ('mu_minus', 13,  r'\mu^-',   15000),
    ('gamma',    22,  r'\gamma',  15000),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot reconstructed pT for eta -> mu mu gamma daughters')
    parser.add_argument('input', help='Input ROOT file')
    parser.add_argument('output_dir', nargs='?',
                        default=os.path.join(_here, '..', '..', 'out', 'quick_plots'),
                        help='Output directory (default: out/quick_plots)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Reading from {args.input}.')

    tree = cut_utils.read_branches(args.input, ['prt_pid', 'prt_px', 'prt_py'])
    pt_all = kinematics.pt(tree['prt_px'], tree['prt_py'])

    masks = {
        'mu_plus':  tree['prt_pid'] == -13,
        'mu_minus': tree['prt_pid'] == 13,
        'gamma':    tree['prt_pid'] == 22,
    }

    channel = fr'${plot_style.CHANNEL_LABELS["eta2mumugamma"]}$'
    pts = {}
    for name, pid, symbol, xmax in PARTICLES:
        pt = ak.to_numpy(ak.flatten(pt_all[masks[name]]))
        pts[name] = pt
        output = os.path.join(args.output_dir, f'reco_pt_{name}.png')
        plot_pt.plot_pt_hist(pt, fr'$p_T({symbol})$ [MeV/$c$]',
                              fr'Reconstructed $p_T({symbol})$, {channel}', output, xmax)
        print(f'[INFO] {name} (PID {pid}): n={len(pt)}, '
              f'p_T range [{pt.min():.2f}, {pt.max():.2f}] MeV/c')

    dimuon_output = os.path.join(args.output_dir, 'reco_pt_dimuon.png')
    plot_pt.plot_pt_overlay(
        [(r'$\mu^+$', pts['mu_plus'], plot_style.COLORS['fit']),
         (r'$\mu^-$', pts['mu_minus'], plot_style.COLORS['pull'])],
        r'$p_T(\mu)$ [MeV/$c$]', fr'Reconstructed $p_T(\mu)$, {channel}', dimuon_output, xmax=15000)
    print(f'[INFO] dimuon overlay saved to {dimuon_output}')
