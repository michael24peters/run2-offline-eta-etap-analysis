"""
plot_q2.py

Plot q^2 (mu+, mu-) from a ROOT tree.

Usage:
    python plot_q2.py input.root
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

# Ensure utils/kinematics.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'utils'))
from kinematics import invariant_mass2


def parse_args():

    parser = argparse.ArgumentParser(description='Plot q2 from ROOT tree.')
    parser.add_argument('input', help='Input ROOT file')
    return parser.parse_args()

args = parse_args()

outfile = 'out/q2.png'
print(f'[INFO] Reading from {args.input} and writing to {outfile}.')

# Branches of interest
IS_MC = False # True = MC | False = real data
# MC
if IS_MC: branches = ['mc_pid', 'mc_px', 'mc_py', 'mc_pz', 'mc_e']
else: branches = ['prt_pid', 'prt_px', 'prt_py', 'prt_pz', 'prt_e']

# Read tree with uproot.
with uproot.open(args.input) as f:
    tree = f['tree']
    branches = tree.arrays(
        branches,
        library='np'
    )

# Define variables for m^2(gamma, mu+) and m^2(mu+, mu-).
# mc contains all MC particles, including the eta and its daughters. We need to
# loop over events and find the gamma, mu+, and mu- for each event to compute
# the invariant masses.
# Reco contains only reconstructed particles, i.e., muons and photons, no eta.
if IS_MC:
    pids = branches['mc_pid']
    pxs = branches['mc_px']
    pys = branches['mc_py']
    pzs = branches['mc_pz']
    es = branches['mc_e']
else:
    pids = branches['prt_pid']
    pxs = branches['prt_px']
    pys = branches['prt_py']
    pzs = branches['prt_pz']
    es = branches['prt_e']

q2s = []

for i in range(len(pids)):
    p1 = p2 = None  # mu+, mu-

    for j, pid in enumerate(pids[i]):
        # Convert from MeV to GeV
        px = pxs[i][j] / 1000.
        py = pys[i][j] / 1000.
        pz = pzs[i][j] / 1000.
        e = es[i][j] / 1000.

        # None check used since only one candidate max is expected per event.
        # Don't overwrite if multiple candidates are found, since if that
        # happens, it's because there was a reconstruction error (i.e.,
        # background).
        if pid == -13 and p1 is None: p1 = (px, py, pz, e)
        elif pid == 13 and p2 is None: p2 = (px, py, pz, e)

    if None not in (p1, p2):
        q2 = invariant_mass2(*p1, *p2)  # m^2(mu+, mu-)
        q2s.append(q2)

q2s = np.array(q2s)

print(f'[INFO] q2 range: [{q2s.min():.4f}, {q2s.max():.4f}]')

# Plot 1d histogram of q^2
fig, ax = plt.subplots(figsize=(7, 6))
ax.hist(q2s, bins=200, range=(0.04, 0.92))
# Get bin edges and counts
bin_edges = np.linspace(0.04, 0.92, 201)
counts = np.histogram(q2s, bins=bin_edges)[0]
# Save to JSON file
import json

json_outfile = 'out/q2_histogram.json'  
with open(json_outfile, 'w') as f:
    json.dump({
        'bin_edges': bin_edges.tolist(),
        'counts': counts.tolist()
    }, f, indent=4)
print(f'[DONE] q2 histogram data saved to {json_outfile}')

ax.set_xlabel(r'$q^2 (m^2_{\mu^+,\mu^-})$ [GeV$^2$]')
ax.set_ylabel('Counts')
ax.set_title(r'$\eta \to \mu^+\mu^-\gamma$ plot ($q^2$)')
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(outfile, dpi=150)
print(f'[DONE] q2 histogram saved to {outfile}')
