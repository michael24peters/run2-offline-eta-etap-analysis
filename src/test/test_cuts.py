'''
test_cuts.py

Script to test cut_utils.py. This is a standalone script to run from the
command line. It prints out partial results to the console to be manually
reviewed. It is not a unit test, but rather a quick way to check that the cuts
are working as expected.
'''

import sys
import os
import numpy as np
import awkward as ak
# Ensure utils/cut_utils.py is importable
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'utils'))
import cut_utils

# Constants
INPUT_FILE = '/data/home/michael24peters/anaroot/ntuple/mc/eta2mumu/signal_20260716/combined_files.root'
MAX_EVENTS      = 200_000          # candidates are sparse; cap the read
MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit
METRIC_BRANCH   = 'tag_dtf_chi2'   # ranks candidates within an event
DAUGHTER_BRANCHES = ['prt_pid', 'prt_pnn_mu']   # per-daughter, need regrouping
MC_BRANCHES     = ['mc_pid']       # per-generator-particle
INDEX_BRANCHES  = ['prt_idx_mom', 'prt_idx_gen', 'mc_idx_mom']  # link levels
PNN_MU_MIN      = 0.95             # every muon daughter must exceed this

# dict.fromkeys() de-duplicates while keeping order; uproot chokes on repeats.
BRANCHES = list(dict.fromkeys(
    [MASS_BRANCH, METRIC_BRANCH] + DAUGHTER_BRANCHES + MC_BRANCHES
    + INDEX_BRANCHES + cut_utils.TOS_TRIGGERS))


def pick_events(tree, n=3, tag_ref=MASS_BRANCH):
    """
    Indices of `n` events that actually hold candidates, multi-candidate events
    first.

    Only ~1% of events reconstruct a candidate, so tree[:3] is all empty lists
    and shows nothing about how the levels line up. Every preview below prints
    the SAME event indices across every branch, so the candidate, daughter and
    generator levels can be read against each other.
    """
    n_tag = ak.to_numpy(ak.num(tree[tag_ref]))
    # Multi-candidate events are the interesting ones -- they are where the
    # index branches stop being a trivial one-to-one map.
    order = np.flatnonzero(n_tag > 1).tolist() + np.flatnonzero(n_tag == 1).tolist()
    return order[:n]

# Load the ROOT files and read the branches of interest into an awkward array
tree = cut_utils.read_branches(INPUT_FILE, BRANCHES, entry_stop=MAX_EVENTS)

# drop_malformed()
print(f'--- drop_malformed() ---')
# `prt_idx_gen` is a per-daughter branch too, and it does NOT always agree with
# prt_idx_mom on how many daughters an event has. Leaving it out of
# daughter_fields lets a ragged event through and reduce() blows up later.
new_tree = cut_utils.drop_malformed(tree,
                                    DAUGHTER_BRANCHES + ['prt_idx_gen'],
                                    cut_utils.TOS_TRIGGERS,
                                    tag_ref=MASS_BRANCH)
# Check difference between tree and new_tree
print(f'[TEST] Original events: {len(tree)}, kept events: {len(new_tree)}')
# Compare matching key arrays and see whether they are the same. Note the
# per-EVENT count is what changes here; the per-candidate totals are the
# interesting number, so print both.
for name in [MASS_BRANCH, METRIC_BRANCH] + cut_utils.TOS_TRIGGERS:
    if name in tree.fields and name in new_tree.fields:
        n_old = int(ak.sum(ak.num(tree[name])))
        n_new = int(ak.sum(ak.num(new_tree[name])))
        print(f'[TEST] Branch: {name}, events {len(tree[name])} -> '
              f'{len(new_tree[name])}, entries {n_old} -> {n_new}')
    else:
        print(f'[TEST] Branch: {name} not found in one of the trees.')

# _align_candidates()
print(f'--- _align_candidates() ---')
tree = new_tree  # Use the cleaned tree for further tests
preview = pick_events(tree)  # events were dropped, so the indices moved
n_tag = ak.num(tree[MASS_BRANCH])
n_tag_np = ak.to_numpy(n_tag)   # candidates per event, for picking preview events
# Aligning the mass branch against itself is a no-op, so align a TOS trigger
# instead -- that is the branch that is actually allowed to come back empty.
probe = cut_utils.TOS_TRIGGERS[-1]
aligned = cut_utils._align_candidates(tree, probe, tag_ref=MASS_BRANCH)
absent = ak.to_numpy(ak.num(tree[probe]) != n_tag)
print(f'[TEST] _align_candidates({probe}): events={len(aligned)}, '
      f'candidates={int(ak.sum(ak.num(aligned)))} '
      f'(tag has {int(ak.sum(n_tag))}), events missing the branch={int(absent.sum())}')
# An event that is MISSING the branch is the whole point of this function, so
# preview one of those if the sample has any.
shown = np.flatnonzero(absent)[:2].tolist() or preview
if not absent.any():
    print(f'[TEST]   - no event is missing {probe} here; showing normal events')
for i in shown:
    print(f'[TEST]   - event {i}: {len(tree[MASS_BRANCH][i])} candidates, '
          f'raw={tree[probe][i].tolist()} -> aligned={aligned[i].tolist()}')
assert ak.all(ak.num(aligned) == n_tag), 'aligned branch is not candidate-shaped'

# Candidate to daughter level
print(f'--- candidate to daughter ---')
# follow() re-indexes VALUES down a level; it does not return indices. Feeding
# it the mass branch gives each daughter the mass of its own candidate.
mass_per_daughter = cut_utils.follow(tree, tree[MASS_BRANCH], INDEX_BRANCHES[0])
print(f'[TEST] follow({MASS_BRANCH}, prt_idx_mom): events={len(mass_per_daughter)}')
for i in preview:
    print(f'[TEST]   - event {i}: {tree[MASS_BRANCH][i].tolist()} -> '
          f'{mass_per_daughter[i].tolist()}')
# Print one candidate entry, then its daughter entries, to check that eta -> mu mu
# looks right. Each candidate should have two muon daughters.
print(f'[TEST] Candidate to daughter level:')
for i in preview:
    idx_mom = tree[INDEX_BRANCHES[0]][i]
    for j, mass in enumerate(tree[MASS_BRANCH][i].tolist()):
        # Daughters of candidate j are the ones whose prt_idx_mom points at j.
        of_this_candidate = idx_mom == j
        info = {b: tree[b][i][of_this_candidate].tolist() for b in DAUGHTER_BRANCHES}
        print(f'  Event {i} candidate {j}: {MASS_BRANCH}={mass:.2f}, '
              f'n_daughters={int(ak.sum(of_this_candidate))}, {info}')

# Candidate cuts
print(f'--- cuts ---')
tos = cut_utils.tos_mask(tree, tag_ref=MASS_BRANCH)
pnn = cut_utils.pnn_mask(tree, threshold=PNN_MU_MIN, tag_ref=MASS_BRANCH)
# Combine cuts with plain `&` and `|` operators
selected = tos & pnn
best = cut_utils.best_candidate_mask(tree[METRIC_BRANCH], selected, method='min')
n_cand = int(ak.sum(n_tag))
masks = [('tos_mask', tos), ('pnn_mask', pnn),
         ('tos & pnn', selected), ('best_candidate', best)]
for label, mask in masks:
    print(f'[TEST] {label:15s}: {int(ak.sum(mask)):6d} / {n_cand} candidates '
          f'({int(ak.sum(ak.any(mask, axis=1)))} events)')
    assert ak.all(ak.num(mask) == n_tag), f'{label} is not candidate-shaped'
# Show the masks side by side on events where best_candidate actually had a
# choice to make: several candidates surviving the cuts, so the min-chi2 tie
# break is visible. Fall back to weaker cases if the sample has none.
n_selected = ak.to_numpy(ak.sum(selected, axis=1))
contested = (np.flatnonzero(n_selected > 1).tolist()
             + np.flatnonzero((n_selected == 1) & (n_tag_np > 1)).tolist()
             + np.flatnonzero(n_selected == 1).tolist())
for i in contested[:2]:
    print(f'[TEST]   - event {i}: {METRIC_BRANCH}='
          f'{[round(x, 2) for x in tree[METRIC_BRANCH][i].tolist()]}')
    for label, mask in masks:
        print(f'[TEST]       {label:15s} {mask[i].tolist()}')
# The best candidate must be a subset of what survived the cuts, and there can
# be at most one per event.
assert ak.all(best == (best & selected)), 'best candidate escaped the cuts'
assert ak.all(ak.sum(best, axis=1) <= 1), 'more than one best candidate'

# Apply follow() cut mask to daughters
print(f'--- candidate -> daughter cut mask ---')
daughter_mask = cut_utils.follow(tree, best, INDEX_BRANCHES[0])
print(f'[TEST] follow(best, prt_idx_mom): '
      f'{int(ak.sum(daughter_mask))} / {int(ak.sum(ak.num(daughter_mask)))} daughters')
kept_pnn = ak.flatten(tree['prt_pnn_mu'][daughter_mask])
kept_pid = ak.flatten(tree['prt_pid'][daughter_mask])
print(f'[TEST]   - surviving daughter pids: '
      f'{np.unique(ak.to_numpy(kept_pid)).tolist()}')
print(f'[TEST]   - surviving prt_pnn_mu min: {float(ak.min(kept_pnn)):.4f} '
      f'(cut was {PNN_MU_MIN})')

# Daughter to generator level daughters
print(f'--- daughter -> gen level daughter ---')
# 'any' is the right aggregation for pushing a selection up a level: 'all' would
# also light up every generator particle that nothing points at, since the
# identity of an empty ak.all() is True.
gen_daughters = cut_utils.reduce(tree, daughter_mask, INDEX_BRANCHES[1],
                                 how='any', target_ref=MC_BRANCHES[0])
print(f'[TEST] reduce(daughters, prt_idx_gen): '
      f'{int(ak.sum(gen_daughters))} / {int(ak.sum(ak.num(gen_daughters)))} gen particles')
print(f'[TEST]   - matched gen pids: '
      f'{np.unique(ak.to_numpy(ak.flatten(tree[MC_BRANCHES[0]][gen_daughters]))).tolist()}')

# Generator level daughter to generator level candidate
print(f'--- gen level daughter -> gen level candidate ---')
# mc_idx_mom is -1 for a particle with no mother in the record. Left alone that
# -1 indexes backwards into the previous event, so mask it off first.
has_mother = tree[INDEX_BRANCHES[2]] >= 0
gen_candidates = cut_utils.reduce(tree, gen_daughters & has_mother,
                                  INDEX_BRANCHES[2], how='any',
                                  target_ref=MC_BRANCHES[0])
print(f'[TEST] reduce(gen daughters, mc_idx_mom): '
      f'{int(ak.sum(gen_candidates))} gen candidates')
print(f'[TEST]   - gen candidate pids: '
      f'{np.unique(ak.to_numpy(ak.flatten(tree[MC_BRANCHES[0]][gen_candidates]))).tolist()}'
      f'  (221 = eta)')

# Walk the whole chain on a few surviving events, so every level can be read
# against the one above it. Multi-candidate events go first.
survived = ak.to_numpy(ak.any(best, axis=1))
chain = (np.flatnonzero(survived & (n_tag_np > 1)).tolist()
         + np.flatnonzero(survived & (n_tag_np == 1)).tolist())[:3]
print(f'[TEST] Full chain, candidate -> daughter -> gen daughter -> gen candidate:')
for i in chain:
    print(f'  Event {i}')
    print(f'    candidate  {MASS_BRANCH}={[round(x, 1) for x in tree[MASS_BRANCH][i].tolist()]}'
          f'  best={best[i].tolist()}')
    print(f'    daughter   prt_pid={tree["prt_pid"][i].tolist()}'
          f'  prt_idx_mom={tree[INDEX_BRANCHES[0]][i].tolist()}'
          f'  mask={daughter_mask[i].tolist()}')
    print(f'               prt_pnn_mu='
          f'{[round(x, 3) for x in tree["prt_pnn_mu"][i].tolist()]}'
          f'  prt_idx_gen={tree[INDEX_BRANCHES[1]][i].tolist()}')
    print(f'    gen        {MC_BRANCHES[0]}={tree[MC_BRANCHES[0]][i].tolist()}'
          f'  mc_idx_mom={tree[INDEX_BRANCHES[2]][i].tolist()}')
    print(f'               gen_daughter={gen_daughters[i].tolist()}'
          f'  gen_candidate={gen_candidates[i].tolist()}')

# ------------------------------------------------------------------------------

# Reverse above tests: now go from generator level candidates to candidates
# should return the same results as above at each step

# Generator level candidate to generator level daughter
back_gen_daughters = (cut_utils.follow(tree, gen_candidates, INDEX_BRANCHES[2])
                      & has_mother)
differs = back_gen_daughters != gen_daughters
n_diff = int(ak.sum(differs))
print(f'[TEST] gen candidate -> gen daughter: {n_diff} / '
      f'{int(ak.sum(gen_daughters))} mismatched gen particles')
if n_diff:
    # Also one-way lossy, in both directions. Going up, a gen daughter with no
    # mother (mc_idx_mom == -1, i.e. a muon or mis-ID'd pion that is not a decay
    # product of the eta) has nothing to reduce onto and is dropped. Coming back
    # down, every true child of the selected eta lights up, including the ones
    # no reco daughter was matched to.
    lost = gen_daughters & ~back_gen_daughters
    gained = back_gen_daughters & ~gen_daughters
    print(f'[TEST]   - {int(ak.sum(lost))} lost (motherless: '
          f'{int(ak.sum(lost & ~has_mother))}), {int(ak.sum(gained))} gained '
          f'(unmatched true children)')
    print(f'[TEST]   - pids involved: '
          f'{np.unique(ak.to_numpy(ak.flatten(tree[MC_BRANCHES[0]][differs]))).tolist()}')

# Generator level daughter to daughter
back_daughters = cut_utils.follow(tree, gen_daughters, INDEX_BRANCHES[1])
n_diff = int(ak.sum(back_daughters != daughter_mask))
print(f'[TEST] gen daughter -> daughter: {n_diff} mismatched daughters')
if n_diff:
    # Not a bug: in a multi-candidate event two reco daughters can point at the
    # SAME generator particle, so the round trip is one-way lossy by
    # construction. Show that every mismatch lives in such an event.
    shared = ak.num(tree[INDEX_BRANCHES[1]]) != ak.num(
        ak.Array([np.unique(ak.to_numpy(row)) for row in tree[INDEX_BRANCHES[1]]]))
    bad_events = ak.any(back_daughters != daughter_mask, axis=1)
    print(f'[TEST]   - {int(ak.sum(bad_events))} events differ; '
          f'{int(ak.sum(bad_events & shared))} of them share a gen particle '
          f'between candidates')

# Daughter to candidate
back_candidates = cut_utils.reduce(tree, back_daughters, INDEX_BRANCHES[0],
                                   how='any', target_ref=MASS_BRANCH)
n_diff = int(ak.sum(back_candidates != best))
print(f'[TEST] daughter -> candidate: {n_diff} mismatched candidates '
      f'({int(ak.sum(back_candidates))} vs {int(ak.sum(best))} selected)')

print('[TEST] done.')
