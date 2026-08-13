"""
plot_dalitz.py

Plot dalitz plot of eta -> mu+ mu- gamma candidates from a ROOT file.

Usage:
    python plot_dalitz.py input.root [--type dalitz|phsp]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import uproot
# Ensure utils/kinematics.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'utils'))
from kinematics import invariant_mass2

def parse_args():

    parser = argparse.ArgumentParser(description='Dalitz plot from ROOT tree.')
    parser.add_argument('input', help='Input ROOT file')
    parser.add_argument(
        '--type', dest='plot_type', choices=['dalitz', 'phsp'], default='dalitz',
        help="Plot type: 'dalitz' -> dalitz_dalitz.png, 'phsp' -> dalitz-phsp.png"
    )
    return parser.parse_args()

args = parse_args()

# Output file path
_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out')
if args.plot_type == 'dalitz': outfile = os.path.join(_out_dir, 'dalitz_dalitz.png')  
else: outfile = os.path.join(_out_dir, 'dalitz-phsp.png')
print(f'[INFO] Reading from {args.input} and writing to {outfile}.')

# Read tree with uproot.
with uproot.open(args.input) as f:
    tree = f['tree']
    branches = tree.arrays(
        ['prt_pid', 'prt_px', 'prt_py', 'prt_pz', 'prt_e'],
        library='np'
    )

# Define variables for m^2(gamma, mu+) and m^2(mu+, mu-).
# Loop over events and find the gamma, mu+, and mu- for each event to compute
# the invariant masses.
pids = branches['prt_pid']
pxs = branches['prt_px']
pys = branches['prt_py']
pzs = branches['prt_pz']
es = branches['prt_e']

m12s = []
m23s = []

for i in range(len(pids)):
    p1 = p2 = p3 = None  # gamma, mu+, mu-

    for j, pid in enumerate(pids[i]):
        # Convert from MeV to GeV
        px = pxs[i][j] / 1000.
        py = pys[i][j] / 1000.
        pz = pzs[i][j] / 1000.
        e = es[i][j] / 1000.

        if pid == 22: p1 = (px, py, pz, e)
        elif pid == -13: p2 = (px, py, pz, e)
        elif pid == 13: p3 = (px, py, pz, e)

    if None not in (p1, p2, p3):
        m12 = invariant_mass2(*p1, *p2)   # m^2(gamma, mu+)
        m23 = invariant_mass2(*p2, *p3)   # m^2(mu+, mu-)
        m12s.append(m12)
        m23s.append(m23)

m12s = np.array(m12s)
m23s = np.array(m23s)

print(f'm12 range: [{m12s.min():.4f}, {m12s.max():.4f}]')
print(f'm23 range: [{m23s.min():.4f}, {m23s.max():.4f}]')

# Plot.
fig, ax = plt.subplots(figsize=(7, 6))
h, xedges, yedges = np.histogram2d(m12s, m23s, bins=100,
                                    range=[[0, m12s.max()+.01], [0, m23s.max()+.01]])
# Mask empty bins so they appear white.
# Also mask bins with very low entries to reduce noise (which is likely caused
# by some errors anyway).
# Keep in mind that this will include some background (error) particles that
# are correctly pid identified but are not from eta candidate.
h_masked = np.ma.masked_where(h <= 1, h)

mesh = ax.pcolormesh(xedges, yedges, h_masked.T, cmap='viridis')
fig.colorbar(mesh, ax=ax, label='Entries')

ax.set_xlabel(r'$m^2_{\gamma,\mu^+}$ [GeV$^2$]')
ax.set_ylabel(r'$m^2_{\mu^+,\mu^-}$ [GeV$^2$]')
ax.set_title(r'$\eta \to \mu\mu\gamma$ Dalitz plot')

# ax.text(0.025, 0.95, "Requires > 1 entry/bin to show",
#         transform=ax.transAxes,
#         verticalalignment='top',
# )

fig.tight_layout()
fig.savefig(outfile, dpi=150)
print(f'Saved: {outfile}')
