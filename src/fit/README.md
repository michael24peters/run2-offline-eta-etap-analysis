# README

## Pipeline

```
driver_*.py   read ROOT → apply selection → reduce to 1 candidate/event → numpy masses
   ↓ numpy array
fit_*.py      fit the numbers. Blind to files, cuts, candidates.
   ↓ result dict
plot_*.py     render
```

Awkward arrays own everything jagged (ROOT I/O, event/candidate structure) and
live only in `src/utils/cut_utils.py`. Numpy owns everything flat (histogram,
PDFs, curves) and lives in `src/utils/fit_utils.py` and the `fit_*` modules.
The boundary is one `ak.to_numpy(ak.flatten(...))` call in the driver.

The drivers are the only entry point -- `fit_dcb.py` / `fit_gauss.py` are
library modules with no CLI, since they take an array rather than a file.

## Usage

One-shot fit + plot. Every flag shown explicitly, with its default value:

For double crystal ball:

```
python src/fit/driver_dcb.py data.root \
    --outdir out \
    --max-events 500000 \
    --xmin 480 --xmax 620 --nbins 80 \
    --mean-init 547.862 --sigma-init 10.0 \
    --alphaL-init 1.5 --nL-init 2.0 \
    --alphaR-init 1.5 --nR-init 2.0
```

For gaussian:

```
python src/fit/driver_gauss.py data.root \
    --outdir out \
    --max-events 500000 \
    --xmin 480 --xmax 620 --nbins 80 \
    --mean-init 547.862 --sigma-init 10.0
```

Each driver runs the fit, saves `results_dcb.json` / `results_gauss.json`,
and writes `dcb_mass_fit.png` / `gauss_mass_fit.png` into `--outdir`.
`--max-events` caps how many events are read, for quick iteration on large
files; omit it to read everything.

The drivers only expose fit hyperparameters as flags -- plot styling (font,
colors, ticks, stats box, etc.) is fixed in `plot_mass_fit.py`/`plot_style.py`,
not driver-configurable.

## Running a full fit, start to finish (eta2mumu / eta2mumugamma)

Two stages, driven by hand; no MC needed for either of them. `fit_dcb()` / `fit_gauss()` already run their own internal two-stage MIGRAD (background -- and for DCB, tails -- fixed first, then floated from that result) on every call, so this is the outer loop on top of that: a quick Gaussian check, then the DCB fit seeded from what it found.

### 1. Gaussian

Confirms the selection gives a clean peak and gives a rough sigma to seed the DCB fit with.

eta2mumu:
```
python src/fit/driver_gauss.py ntuple/lhcb/eta2mumu/20260716/combined_files.root \
    --outdir out/eta2mumu/turbo/$(date +%Y%m%d) \
    --pnn-mu-min 0.95 \
    --mean-init 547.862 \
    --sigma-init 5.0
```

eta2mumugamma -- use `20260608`, not `20260716` (the `20260716` eta2mumugamma production writes each event ~5x, TBD why):
```
python src/fit/driver_gauss.py ntuple/lhcb/eta2mumugamma/20260608/combined_files.root \
    --outdir out/eta2mumugamma/turbo/$(date +%Y%m%d) \
    --xmin 450 --xmax 650 \
    --pnn-mu-min 0.95 \
    --mean-init 547.862 \
    --sigma-init 10.0
```

Read the `[INFO] ... mean=...  σ=...` line on stderr and look at `gauss_mass_fit.png`. `--mean-init` doesn't need setting -- it already defaults to `kinematics.M_ETA` (547.862 MeV). TOS is on by default; add `--no-tos` to turn it off.

### 2. DCB fit, seeded from stage 1

Same file, same selection flags, `--sigma-init` set to whatever stage 1 reported:
```
python src/fit/driver_dcb.py ntuple/lhcb/eta2mumu/20260716/combined_files.root \
    --outdir out/eta2mumu/turbo/$(date +%Y%m%d) \
    --pnn-mu-min 0.95 \
    --sigma-init <σ from step 1> \
    --alphaL-init 1.5 --nL-init 2.0 --alphaR-init 1.5 --nR-init 2.0
```
```
python src/fit/driver_dcb.py ntuple/lhcb/eta2mumugamma/20260608/combined_files.root \
    --outdir out/eta2mumugamma/turbo/$(date +%Y%m%d) \
    --xmin 450 --xmax 650 \
    --pnn-mu-min 0.95 \
    --sigma-init <σ from step 1> \
    --alphaL-init 1.5 --nL-init 2.0 --alphaR-init 1.5 --nR-init 2.0
```

## Fixing tails from an MC fit

Fit signal MC first to pin down the tail-shape parameters (alphaL, nL, alphaR,
nR), then constrain them in the data fit with a `NormalConstraint` prior
centered on the MC values, its covariance widened by `--tail-prior-factor`
standard deviations (default 5) rather than hard-fixed.

Hard-fixing was tried first, on the reasoning that a soft prior isn't enough
if MC only weakly constrains one of the four (CB tails are nearly degenerate
at large n, so a weak constraint lets the data fit drift into the
barely-decaying regime and reclassify background as signal). It does remove
that degree of freedom outright -- but it also removes the data fit's only
way to compensate for real MC/data differences in tail shape, and measured on
`eta2mumugamma`: both legs converged (`valid=True`), but the data fit's
chi2/ndof came back at ~21 (vs. ~1.2 for the unconstrained-tail single-stage
fit), and locking ~150 per-bin Dalitz fits to that MC-only shape produced
*more* non-converging bins than before, with a median chi2/ndof of ~12 among
the bins that did converge. A numerically valid but systematically wrong
shape isn't worth trading real bins for, so the tails float with a wide-slack
prior instead: real freedom to move away from the MC value where data
disagrees, while still anchored against drifting into the degenerate regime.

Keep the full mass window for both fits. Restricting the data fit's
likelihood to a narrow core window around the peak was tried, to stop the
tail from predicting non-negligible density tens of sigma out -- but it made
things worse: floating the Chebyshev background's 2 shape parameters against
only a ~40 MeV slice leaves them essentially unconstrained, and extrapolating
that across the rest of the window diverges badly. The full window gives the
background a real baseline and keeps it well-constrained.

The heavy-tail-at-large-sigma effect is real, though, and worth understanding
rather than fitting around: a single power-law CB tail can't simultaneously
match the near-tail and deep-tail regions of the (background-free) MC. The
near tail (better populated) wants a heavier tail than the deep tail actually
supports, so nL/nR end up as a compromise -- concretely, the fitted MC model
over-predicts its own deep-tail bin counts by ~17% (480-498 MeV, 17-23 sigma
from the peak), while under-predicting slightly in the 500-525 MeV near-tail
region. No parameter fix removes this; it's the functional form running out
of flexibility. Mitigation: `fixed_params={'n_bkg': 0.0, 'c0': 0.0, 'c1': 0.0}`
on the MC fit. The sample is truth-matched signal MC, so it shouldn't have a
floating background component at all -- and letting one float anyway was
additionally distorting the tail-shape estimate (`nR` came back at 4.98+/-1.25
with it floating vs. 2.10+/-0.16 fixed to zero -- the former was largely
unconstrained noise, not real precision).

```python
import sys
sys.path.insert(0, "src/utils")
sys.path.insert(0, "src/fit")
import numpy as np
from fit_dcb import fit_dcb, TAIL_PARAMS

# 1. Fit signal MC -- same selection driver_dcb.py would apply
#    (read_branches -> drop_malformed -> cuts -> best_candidate_mask ->
#    values), against ntuple/mc/<channel>/signal_*/combined_files.root.
#    fixed_params holds n_bkg/c0/c1 at 0: a truth-matched sample shouldn't
#    have any background.
mc_masses = ...
_, m_mc = fit_dcb(mc_masses, sigma_init=..., alphaL_init=1.5, nL_init=2.0,
                  alphaR_init=1.5, nR_init=2.0,
                  fixed_params={'n_bkg': 0.0, 'c0': 0.0, 'c1': 0.0})

tail_values = np.array([m_mc.values[p] for p in TAIL_PARAMS])
tail_cov = np.array(m_mc.covariance[list(TAIL_PARAMS)])

# 2. Fit data with the tail shape as a widened prior, not fixed -- gives the
#    data fit TAIL_PRIOR_FACTOR MC-fit standard deviations of slack per tail
#    parameter to move away from the MC value where the data disagrees.
TAIL_PRIOR_FACTOR = 5.0
data_masses = ...
result, m_data = fit_dcb(data_masses, sigma_init=...,
                         tail_prior=(tail_values, tail_cov * TAIL_PRIOR_FACTOR**2))
```

## Changing the selection

The easiest way to change the existing selection-criteria is to change the top-level variables:

```python
MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit
METRIC_BRANCH   = 'tag_dtf_chi2'   # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']
PNN_MU_MIN      = 0.95  # ProbNNMu minimum for selection
MUON_PID        = 13
TOS_TRIGGERS    = ['tag_l0_tos0', ...]   # ntuple-specific, so not in cut_utils
```

Every cut is structured as a per-candidate boolean mask shaped like `tag_dtf_m`, so they combine with a plain `&` and adding one is a line or two. Most selection criteria (i.e., kinematic cuts) can be applied with `reduce()`:

```python
tos, tos_report = cut_utils.tos_mask(tree, TOS_TRIGGERS)

# every muon daughter passes the PID cut ("passes OR isn't a muon", so a
# candidate with no muon daughters passes rather than fails)
muon_ok = (tree['prt_pnn_mu'] > PNN_MU_MIN) | (abs(tree['prt_pid']) != MUON_PID)
pnn = cut_utils.reduce(tree, muon_ok, 'prt_idx_mom', how='all')

# every daughter has pT > 500 MeV
fast = cut_utils.reduce(tree, kinematics.pt(tree['prt_px'],
                        tree['prt_py']) > 500, 'prt_idx_mom', how='all')

# candidate has exactly one photon daughter
one_gamma = cut_utils.reduce(tree, tree['prt_pid'] == 22, 'prt_idx_mom',
                             how='sum') == 1

keep = tos & pnn & fast & one_gamma
```

The full pipeline is six lines:

```python
tree, drop_report = cut_utils.drop_malformed(cut_utils.read_branches(infile, BRANCHES))
keep   = ...                                                   # as above
best   = cut_utils.best_candidate_mask(tree, METRIC_BRANCH, keep)
masses = cut_utils.values(tree, MASS_BRANCH, best)             # flat numpy
result = fit_gauss(masses, ...)
```

`values()` is where awkward is converted to numpy: everything upstream is jagged and per-event, everything downstream is a flat array of numbers.

Two things to know before editing that block:

- **Never index the whole tree with a per-candidate mask.** `tree[mask]`
  silently truncates the `prt_*` fields -- a 3-candidate / 9-daughter event
  comes back with 3 candidates and 3 daughters, no error. Mask individual
  `tag_*` branches instead, which is what the drivers do.
- **Drop empty events before the best-candidate step.** `ak.argmin` on an
  event with no surviving candidates returns `None`, which becomes a silently
  masked numpy entry. The drivers filter on `ak.num(metric) > 0` first.

Cuts are applied *before* the best-candidate reduction, so the survivor is the
best among candidates that passed rather than an event being discarded because
its arbitrarily-chosen candidate failed. Reordering is a matter of moving two
lines in the driver.

`drop_malformed()` removes events with no candidates and events whose `prt_*`
branches are inconsistent with `prt_idx_mom` (~0.1% of current ntuples). It works
out which branches live at which level from their name prefixes, so **every**
`prt_*` branch read is checked as a daughter field and every `tag_*` branch as a
candidate field — forgetting to list one is how a ragged event survives to blow
up inside `reduce()` much later.

Nothing in `cut_utils` prints. `drop_malformed()` and `tos_mask()` return their
diagnostics as a second return value, and the driver decides what to log and what
to write into the results JSON under `selection_report`.

When traversing `mc_idx_mom`, AND in `cut_utils.valid(tree, 'mc_idx_mom')` first.
A generator particle with no mother stores -1, which `follow()`/`reduce()` treat
as a real index and throws no error, which then produces the wrong entry.

## Known ntuple quirks

Open items, surfaced by the drivers on stderr every run. Each is worked around
rather than solved, and each is worth chasing upstream in the ntuple production.

**Ragged HLT2 trigger branches.** `tag_hlt2_tos0` / `tag_hlt2_tos1` are not
always written with one entry per candidate. Two distinct cases:

- *Absent* (empty list): the line was not evaluated for that event. Treated as
  "did not fire" and zero-filled by `_align_candidates()`. Reported as
  `[INFO] <branch>: not evaluated in N/M events`. If a branch is ever absent in
  *all* events it contributes nothing to the TOS OR -- i.e. that trigger is
  silently not part of the selection -- so it is escalated to `[WARN]`.
- *Partial* (fewer entries than candidates): there is no way to tell which
  candidate each entry belongs to, so `drop_malformed()` drops those events
  rather than assuming positional correspondence. Reported as
  `[WARN] <branch>: partial in N events`.

Measured on `ntuple/lhcb/.../20260716/combined_files.root`:

| channel | partial (events dropped) | `tag_hlt2_tos0` absent | `tag_hlt2_tos1` absent |
|---|---|---|---|
| `eta2mumu` | none | none | none |
| `eta2mumugamma` | 16.8% | 88.3% | 52.1% |

So `eta2mumu` is clean, while `eta2mumugamma` loses ~17% of events to the
ambiguity and runs with HLT2 contributing to the TOS OR only where it was
actually evaluated. **Unresolved:** why the HLT2 branches are written per event
rather than per candidate, and what a partial list is meant to signify.

**Inconsistent `prt_*` lengths.** In a small fraction of events (~0.1% in MC)
the `prt_*` branches disagree in length with `prt_idx_mom`, or an index points
past the last candidate. Those events are dropped and counted.

> **Caveat on the best-candidate metric.** In current ntuples `tag_dtf_chi2` is
> identical across all candidates in ~61% of multi-candidate events, with a
> median mass spread of ~300 MeV within those events -- so `method='min'` is
> effectively picking candidate 0 arbitrarily most of the time.
> `find_best_candidates()` takes `metric=` / `method=` so a better discriminant
> can be substituted.


## Fitting an arbitrary array

The fit modules were written in a generic way so as to fit any numpy array of values:

```python
import sys
sys.path.insert(0, "src/fit")
from fit_dcb import fit_dcb

result = fit_dcb(array, xmin=480, xmax=620, nbins=80)   # array: numpy array of values
```
