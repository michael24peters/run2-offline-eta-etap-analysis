"""
plot_decay_length.py

Calculate and plot decay length and pseudo-lifetime for eta candidates from a
ROOT file. Pseudo-lifetime: tau = L * m / (p * c), derived from 
tau = L / (beta*gamma*c).

Usage:
    python plot_decay_length.py input.root
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import uproot

C_MM_PER_S = 299_792_458_000.0  # speed of light in mm/s

def parse_args():
    parser = argparse.ArgumentParser(description='Decay length and pseudo-lifetime from ROOT tree.')
    parser.add_argument('input', help='Input ROOT file')
    return parser.parse_args()

args = parse_args()

_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'out')
outfile = os.path.join(_out_dir, 'decay_length.png')
print(f'[INFO] Reading from {args.input} and writing to {outfile}.')

with uproot.open(args.input) as f:
    tree = f['tree']
    branches = tree.arrays(
        ['tag_x', 'tag_y', 'tag_z', 'tag_idx_pvr',
         'tag_px', 'tag_py', 'tag_pz', 'tag_m',
         'pvr_x', 'pvr_y', 'pvr_z'],
        library='np'
    )

decay_lengths = []
pseudo_lifetimes = []

for i in range(len(branches['tag_x'])):
    for j in range(len(branches['tag_x'][i])):
        # tag_idx_pvr tells us which PV in the pvr_* arrays belongs to this candidate.
        ipv = int(branches['tag_idx_pvr'][i][j])

        # Decay length (L): distance between decay vertex (tag_x/y/z) and 
        # PV (pvr_x/y/z), in mm.
        dx = branches['tag_x'][i][j] - branches['pvr_x'][i][ipv]
        dy = branches['tag_y'][i][j] - branches['pvr_y'][i][ipv]
        dz = branches['tag_z'][i][j] - branches['pvr_z'][i][ipv]
        L = np.sqrt(dx**2 + dy**2 + dz**2)

        # Pseudo-lifetime (tau) = L / (beta*gamma*c).
        # In HEP units (p in MeV/c, m in MeV/c²), so tau = L * m / (p * c)
        p = np.sqrt(branches['tag_px'][i][j]**2 +
                    branches['tag_py'][i][j]**2 +
                    branches['tag_pz'][i][j]**2)
        m = branches['tag_m'][i][j]

        # Units: L [mm], m [MeV/c^2], p [MeV/c], c [mm/s] -> tau [s]
        tau = L * m / (p * C_MM_PER_S)

        decay_lengths.append(L)
        pseudo_lifetimes.append(tau)

decay_lengths = np.array(decay_lengths)
pseudo_lifetimes = np.array(pseudo_lifetimes)

dl_mask = decay_lengths > 100
print(f'[INFO] Removing {dl_mask.sum()} outlier candidate(s):')
# Print outlier decay lengths
print(f'[INFO]   Outlier decay lengths: {decay_lengths[dl_mask]}')
pl_mask = pseudo_lifetimes > 5e-12
print(f'[INFO] Removing {pl_mask.sum()} outlier candidate(s).')
# Print outlier pseudo-lifetimes
print(f'[INFO]   Outlier pseudo-lifetimes: {pseudo_lifetimes[pl_mask]}')

print(f'[INFO] Decay length:    [{decay_lengths[~dl_mask].min():.4f}, {decay_lengths[~dl_mask].max():.4f}] mm')
print(f'[INFO] Pseudo-lifetime: [{pseudo_lifetimes[~pl_mask].min():.4e}, {pseudo_lifetimes[~pl_mask].max():.4e}] s')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))


axes[0].hist(decay_lengths[~dl_mask], bins=100, histtype='step', color='tab:blue')
axes[0].set_xlabel('Decay length [mm]')
axes[0].set_ylabel('Entries')
axes[0].set_title(r'$\eta \to \mu\mu\gamma$ decay length')
axes[0].set_yscale('log')


axes[1].hist(pseudo_lifetimes[~pl_mask], bins=100, histtype='step', color='tab:blue')
axes[1].set_xlabel('Pseudo-lifetime [s]')
axes[1].set_ylabel('Entries')
axes[1].set_title(r'$\eta \to \mu\mu\gamma$ pseudo-lifetime')
axes[1].set_yscale('log')

fig.tight_layout()
fig.savefig(outfile, dpi=300)
print(f'[INFO] Saved: {outfile}')
