"""
trig_eff.py

Trigger efficiency of every line and of each level's TOS selection, per
candidate, two ways:

    true     eff = N(TOS) / N                  (with MC truth, the actual efficiency)
    TISTOS   eff = N(TIS & TOS) / N(TIS)       (the data-driven estimate)

TOS selection: TOS on at least one of the level's TOS_LINES. TIS: TIS on at
least one line of the level (every line in LINES, including the dedicated TIS
line). Each level is evaluated on the candidates that pass the TOS selection
of every earlier level (L0 -> HLT1 -> HLT2, as cut_utils.tos_mask applies
them), so the per-level efficiencies multiply to the total.

With --daughter-pids, the candidates are the MC-truth-matched ones (every
daughter matches eta -> daughter-pids, bkg_utils.signal_candidate_mask) whose
generated decay is in the fiducial region (fid_reqs.fiducial_candidate_mask);
decays outside it are not counted. Without, every candidate.

Errors are binomial: sqrt(eff (1 - eff) / denominator).

Usage:
    python trig_eff.py input.root [output.json] [--daughter-pids -13 13 [22]]
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
sys.path.insert(0, os.path.join(_here, '..', 'root'))
import cut_utils
import bkg_utils
from fid_reqs import fiducial_candidate_mask

TAG_REF    = 'tag_m'
MOTHER_PID = 221

# Every line of each level, in ntuple order: line i is tag_<level>_tos<i> /
# tag_<level>_tis<i> (~/ana/src/ntuple.py l0Trgs / hlt1Trgs / hlt2Trgs).
LINES = {
    'L0':   ['L0DiMuon', 'L0Muon', 'L0Hadron'],
    'HLT1': ['Hlt1DiMuonNoIP', 'Hlt1DiMuonLowMass', 'Hlt1MultiDiMuonNoIP', 'Hlt1TrackMVA'],
    'HLT2': ['Hlt2ExoticaPrmptDiMuonTurbo', 'Hlt2ExoticaDiMuonNoIPTurbo', 'Hlt2Topo'],
}
# TOS selection per level, as in src/fit/driver_dcb.py.
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}


def branch(level, kind, i):
    """tag_<level>_<kind><i>, e.g. branch('HLT1', 'tos', 2) = 'tag_hlt1_tos2'."""
    return f'tag_{level.lower()}_{kind}{i}'


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('input', help='Input ROOT file')
parser.add_argument('output', nargs='?', default=None, help='Output JSON file (optional)')
parser.add_argument('--daughter-pids', nargs='+', type=int, default=None,
                    help='Truth-match candidates to eta -> these daughters in the '
                         'fiducial region (signal MC), e.g. -13 13 or -13 13 22. '
                         'Default: every candidate')
parser.add_argument('--max-events', type=int, default=None,
                    help='Only read the first N events (default: all)')
args = parser.parse_args()

trigger_branches = [branch(level, kind, i) for level, names in LINES.items()
                    for kind in ('tos', 'tis') for i in range(len(names))]
truth_branches = (['prt_pid', 'prt_idx_gen', 'mc_pid', 'mc_idx_mom',
                   'mc_px', 'mc_py', 'mc_pz']
                  if args.daughter_pids else [])
branches = [TAG_REF, 'prt_idx_mom'] + truth_branches + trigger_branches

print(f'[INFO] Reading from {args.input}.')
tree, drop_report = cut_utils.drop_malformed(
    cut_utils.read_branches(args.input, branches, entry_stop=args.max_events),
    tag_ref=TAG_REF)
print(f"[INFO] events: read {drop_report['read']}, kept {drop_report['kept']}")

if args.daughter_pids:
    matched = bkg_utils.signal_candidate_mask(tree, MOTHER_PID, args.daughter_pids,
                                              target_ref=TAG_REF)
    fiducial = fiducial_candidate_mask(tree, MOTHER_PID, args.daughter_pids,
                                       target_ref=TAG_REF)
    population = matched & fiducial
    population_label = f'truth-matched eta -> {args.daughter_pids}, gen decay fiducial'
    truth_report = {'truth_matched': int(ak.sum(matched)),
                    'truth_matched_not_fiducial': int(ak.sum(matched & ~fiducial))}
    print(f"[INFO] {truth_report['truth_matched']} truth-matched candidates, "
          f"{truth_report['truth_matched_not_fiducial']} of them from decays outside "
          f"the fiducial region (not counted)")
else:
    population = ak.ones_like(tree[TAG_REF], dtype=bool)
    population_label = 'all candidates'
    truth_report = None
population = ak.to_numpy(ak.flatten(population))
print(f'[INFO] {int(population.sum())} candidates ({population_label}) '
      f'of {len(population)}')


def fired(name):
    """Per-candidate flat numpy bool of a trigger branch; absent reads as not fired."""
    return ak.to_numpy(ak.flatten(cut_utils.align(tree, name, TAG_REF))) > 0.5


def efficiency(num, den):
    """{'num', 'den', 'eff', 'err'} for boolean numerator/denominator masks."""
    n, d = int(num.sum()), int(den.sum())
    eff = n / d if d else float('nan')
    err = float(np.sqrt(eff * (1.0 - eff) / d)) if d else float('nan')
    return {'num': n, 'den': d, 'eff': eff, 'err': err}


def row(label, tos, base, tis):
    """True and TISTOS efficiency of `tos` on the candidates in `base`."""
    true = efficiency(base & tos, base)
    tistos = efficiency(base & tis & tos, base & tis)
    print(f"  {label:34s} {true['num']:8d} / {true['den']:<8d} {true['eff']:7.4f} ± {true['err']:.4f}"
          f"   {tistos['num']:7d} / {tistos['den']:<7d} {tistos['eff']:7.4f} ± {tistos['err']:.4f}")
    return {'true': true, 'tistos': tistos}


results = {
    'meta': {
        'input_file': args.input,
        'population': population_label,
        'lines': LINES,
        'tos_selection': TOS_LINES,
        'tis': 'TIS on at least one line of the level',
        'conditioning': 'each level on candidates TOS at every earlier level',
        'true': 'N(TOS) / N',
        'tistos': 'N(TIS & TOS) / N(TIS)',
        'err': 'binomial: sqrt(eff (1 - eff) / den)',
    },
    'selection_report': {'events': drop_report, 'truth': truth_report},
    'levels': {},
}

base = population.copy()
total_true = 1.0
total_tistos = 1.0
print(f"\n  {'':34s} {'true: N(TOS) / N':>34s}   {'TISTOS: N(TIS&TOS) / N(TIS)':>32s}")
for level, names in LINES.items():
    tis = np.zeros_like(base)
    for i in range(len(names)):
        tis |= fired(branch(level, 'tis', i))
    tos_sel = np.zeros_like(base)
    for name in TOS_LINES[level]:
        tos_sel |= fired(name)

    print(f'{"─" * 110}\n{level} (on {int(base.sum())} candidates, '
          f'{int((base & tis).sum())} TIS)')
    lines = {name: row(name, fired(branch(level, 'tos', i)), base, tis)
             for i, name in enumerate(names)}
    selection = row('TOS selection (TOS_LINES)', tos_sel, base, tis)
    results['levels'][level] = {
        'candidates': int(base.sum()),
        'candidates_tis': int((base & tis).sum()),
        'lines': lines,
        'tos_selection': selection,
    }
    total_true *= selection['true']['eff']
    total_tistos *= selection['tistos']['eff']
    base = base & tos_sel

results['total'] = {
    'candidates_tos_every_level': int(base.sum()),
    'true': efficiency(base, population),
    'true_product_of_levels': total_true,
    'tistos_product_of_levels': total_tistos,
}
print(f'{"─" * 110}\nTotal TOS selection: true {total_true:.4f} '
      f'({int(base.sum())} / {int(population.sum())}), TISTOS product {total_tistos:.4f}')

if args.output:
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[DONE] JSON saved to {args.output}')
