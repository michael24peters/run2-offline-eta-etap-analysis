# README

## Pipeline

```
driver_*.py   read ROOT → apply selection → reduce to 1 candidate/event → numpy masses
   ↓ numpy array
fit_*.py      fit the numbers. Blind to files, cuts, candidates.
   ↓ result dict
plot_*.py     render
```

Awkward arrays handle everything jagged (ROOT I/O, event/candidate structure) and
live only in `src/utils/cut_utils.py`. Numpy handles everything flat (histogram,
PDFs, curves) and lives in `src/utils/fit_utils.py` and the `fit_*` modules.
The conversion occurs in the `ak.to_numpy(ak.flatten(...))` call in the driver.

Most everything is a library, so the drivers are the only entry point for fitting
(`fit_dcb.py` / `fit_gauss.py`, etc.).

## Usage

`driver_mass_fit.py` reads a ROOT file, selects, keeps the best candidate per
event, and runs the Gaussian fit, then the DCB fit seeded with the Gaussian
sigma. The flag values below are the defaults.

```
python src/fit/driver_mass_fit.py data.root \
    --outdir OUT \
    --tos-levels L0 HLT1 HLT2 --pnn-mu-min 0.4 \
    --xmin 480 --xmax 620 --nbins 80 \
    --mean-init 547.862 --sigma-init 5.0 \
    --alphaL-init 1.5 --nL-init 2.0 --alphaR-init 1.5 --nR-init 2.0
```

It writes into `--outdir`:

- `selection.json`: the selection settings, the events dropped as malformed,
  the candidates TOS on each trigger line, and the cutflow
- `results_gauss.json`, `gauss_mass_fit.png`
- `results_dcb.json`, `dcb_mass_fit.png`

`--max-events N` reads only the first N events, for quick iteration on large
files. The drivers only expose fit hyperparameters as flags. Plot styling (font,
colors, ticks, stats box, etc.) is fixed in `plot_mass_fit.py`/`plot_style.py`,
and must be changed there manually.

## DCB tails from signal MC

On data the DCB tails are poorly constrained. They come from truth-matched
signal MC instead: fixed at the MC values with `TAIL_PRIOR_FACTOR` 0 in
`driver_mass_fit.py`, or otherwise a `NormalConstraint` prior on
(alphaL, nL, alphaR, nR) centred on the MC values, with the MC covariance
widened by `TAIL_PRIOR_FACTOR` standard deviations:

```
# 1. truth-matched signal MC: DCB only, from --sigma-init, no background (n_bkg = 0, c0 = c1 = 0)
python src/fit/driver_mass_fit.py ntuple/mc/eta2mumugamma/signal_20261006/combined_files.root \
    --outdir out/eta2mumugamma/signal/$(date +%Y%m%d)/truth --truth-match -13 13 22
# 2. data, tails under the MC prior
python src/fit/driver_mass_fit.py ntuple/lhcb/eta2mumugamma/20261006/combined_files.root \
    --outdir out/eta2mumugamma/turbo/$(date +%Y%m%d)/mc_prior \
    --tail-prior out/eta2mumugamma/signal/$(date +%Y%m%d)/truth/results_dcb.json
```

`--truth-match` takes the signal daughter PIDs: `-13 13` for eta -> mu+ mu-,
`-13 13 22` for eta -> mu+ mu- gamma. It skips the Gaussian fit: with no
background, a Gaussian has no stable minimum on the radiative tail of
eta -> mu+ mu- gamma MC.

`src/fit/run_mass_fits.sh [--channel "eta2mumu eta2mumugamma"]` runs the full
sequence for each channel on the 20261006 ntuples, into
`out/<channel>/<turbo|signal>/<today>/`: data and signal MC with no prior, the
truth-matched MC (`signal/<today>/truth/`), and data with its tail prior
(`turbo/<today>/mc_prior/`).

## Changing the selection

The easiest way to change the existing selection-criteria is to change the top-level variables:

```python
MASS_BRANCH     = 'tag_dtf_m'      # the quantity being fit
METRIC_BRANCH   = 'tag_dtf_chi2'   # ranks candidates within an event
DAUGHTER_FIELDS = ['prt_pid', 'prt_pnn_mu']
PNN_MU_MIN      = 0.95  # ProbNNMu minimum for selection
MUON_PID        = 13
TOS_LINES       = {'L0': [...], 'HLT1': [...], 'HLT2': [...]}   # ntuple-specific, so not in cut_utils
```

Every cut is structured as a per-candidate boolean mask shaped like `tag_dtf_m`, so they combine with a plain `&` and adding one is a line or two.

TOS means the candidate is TOS on at least one line at every trigger level (AND across levels, OR within a level). `TOS_LINES` maps each level to its TOS branches, and `tos_mask()` takes that dict; a flat list of branches raises a `TypeError`. Every driver, notebook and plot script uses the same lines:

```python
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0', 'tag_hlt1_tos1'],   # Hlt1DiMuonNoIP, Hlt1DiMuonLowMass
    'HLT2': ['tag_hlt2_tos0', 'tag_hlt2_tos1'],   # Hlt2ExoticaPrmptDiMuonTurbo, Hlt2ExoticaDiMuonNoIPTurbo
}
``` Most selection criteria (i.e., kinematic cuts) can be applied with `reduce()`:

```python
tos, tos_report = cut_utils.tos_mask(tree, TOS_LINES)

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
branches are inconsistent with `prt_idx_mom`. It works
out which branches live at which level from their name prefixes, so **every**
`prt_*` branch read is checked as a daughter field and every `tag_*` branch as a
candidate field — forgetting to list one is how a ragged event survives to blow
up inside `reduce()` much later.

Nothing in `cut_utils` prints. `drop_malformed()` and `tos_mask()` return their
diagnostics as a second return value, and the driver decides what to log and what
to write into `selection.json` under `selection_report`.

When traversing `mc_idx_mom`, AND in `cut_utils.valid(tree, 'mc_idx_mom')` first.
A generator particle with no mother stores -1, which `follow()`/`reduce()` treat
as a real index and throws no error, which then produces the wrong entry.

## Known ntuple quirks

Properties of the ntuple productions that the selection depends on.

**Trigger branch numbering depends on the production.** `tag_<level>_tos<i>` is
the i-th line of `l0Trgs` / `hlt1Trgs` / `hlt2Trgs` in the `~/ana/src/ntuple.py`
that wrote the file, and that order has changed. `TOS_LINES` assumes the July
2026 order or later:

| ntuple | `tag_hlt2_tos0` | `tag_hlt2_tos1` | `tag_hlt2_tos2` |
|---|---|---|---|
| data 20260507, 20260608 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDisplDiMuon | Hlt2ExoticaDiMuonNoIPTurbo |
| data and MC 20260716 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | Hlt2ExoticaDisplDiMuon |
| `ntuple.py` since 2026-08-05 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | -- |
| `ntuple.py` since 2026-10-05 (MC 20261006) | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | Hlt2Topo (TIS line) |

L0 (`tos0` L0DiMuon, `tos1` L0Muon) and HLT1 (`tos0` Hlt1DiMuonNoIP, `tos1`
Hlt1DiMuonLowMass) keep the same order in all of them. Since 2026-10-05 each
level has further lines after those: L0 `tos2` L0Hadron (TIS line), HLT1
`tos2` Hlt1MultiDiMuonNoIP and `tos3` Hlt1TrackMVA (TIS line).

> **Caveat on the best-candidate metric.** In the 20261006 `eta2mumugamma`
> ntuples `tag_dtf_chi2` is the same for every candidate, to a relative
> difference below 1e-9, in 95% of multi-candidate data events and 91% of
> multi-candidate MC events, with a median mass spread of 180 MeV (data) and
> 300 MeV (MC) within those events. In `eta2mumu` this never occurs. The values
> differ only by floating-point rounding, so taking the minimum selects the
> best candidate by rounding noise in most multi-candidate `eta2mumugamma`
> events. `best_candidate_mask()` takes the ranking branch as `metric_branch`
> (`METRIC_BRANCH` in the drivers), so a better discriminant can be substituted.


## Per-Dalitz-bin fits: the phase-space factor

Full write-up, with derivation, figures and a code walkthrough:
[`phase-space-factor.md`](phase-space-factor.md).

In a Dalitz bin, only part of the m(gamma mu+ mu-) window is kinematically
reachable. In the mu+ mu- rest frame, with theta* the mu+ angle and
beta = sqrt(1 - 4 m_mu^2 / m23),

    m12 = m_mu^2 + (M^2 - m23) (1 - beta cos theta*) / 2

(m12 = m^2(gamma mu+), m23 = m^2(mu+ mu-)), so cos theta* in [-1, 1] bounds
the total mass M at each Dalitz point (`kinematics.m123_limits`). For a bin,
`kinematics.phase_space_factor` gives f(M), the fraction of the bin reachable
at mass M -- the same thing as sampling (m12, m23) uniformly in the bin and
counting how often M is within those limits, but computed without MC noise.
f is 1 where the whole bin is reachable and 0 where none of it is, which is
what takes edge bins to zero at the ends of the mass window.

`fit_dalitz.fit_dalitz` multiplies the DCB signal and the (plain Chebyshev)
background by a factor f in every bin when given one (`signal_factor`,
`bkg_factor`; `fit_utils.make_dcb_cost(..., signal_factor_grid=,
bkg_factor_grid=)`), and fixes n_sig = 0 in bins whose signal factor is 0 at
m_eta, which lie entirely outside the eta's Dalitz region. Without a factor
(the default) both shapes are the plain ones. The background has no per-bin
prior; mean/sigma keep theirs.

The limits are an identity on the four-vectors, so the fitted mass must come
from the same daughter momenta as m12/m23. `driver_dalitz.py --mass` picks the
mass and the daughters together:

| `--mass` | mass | daughters | mass - rebuilt (median, RMS) | outside own limits |
|---|---|---|---|---|
| `rebuilt` (default) | rebuilt from the daughters | `prt_px/py/pz/e` | 0, 0 | 0 |
| `tag_m` | `tag_m` | `prt_px/py/pz/e` | 0.07 MeV, 4.1 MeV | 0.7% |
| `tag_dtf_m` | `tag_dtf_m` | `prt_dtf_px/py/pz/e` | 0.00 MeV, 0.04 MeV | 8 of 3.0M |

(20261006 Turbo data; the driver prints these numbers.) `tag_dtf_m` with the DTF daughters satisfies the identity
like the rebuilt mass, at the DTF resolution (sigma ~14 MeV vs ~16.5 MeV
rebuilt).

Per bin, the Chebyshev's [-1, 1] variable spans the bin's reachable mass
range (where f > 0), not [xmin, xmax]: fit on only a slice of its range, its
coefficients are near-degenerate and MIGRAD runs them off to infinity. So
per-bin c0/c1 are not comparable to the global fit's. The plain Chebyshev can
still go negative inside the reachable region; the fit floors it at
`fit_utils.BKG_FLOOR`, and `bkg_goes_negative` flags any bin where that
happened.

The global fit carries a widened MC tail prior (`--tail-prior`): on the
rebuilt mass the unconstrained tails run off to n ~ 50-150 and the fit is
invalid. The prior of each mass is the global fit of truth-matched signal MC
of the same mass (`driver_dalitz.py --truth-match`).

Known limitation: f assumes background is uniform across the bin. Near the
low-m12 boundary (the i = 1 column) it isn't, and the data turn on more
slowly than f predicts, which is where the worst per-bin chi2 values are.

### Pipeline

```
python src/fit/driver_dalitz.py ntuple/lhcb/eta2mumugamma/20261006/combined_files.root \
    --outdir OUT (--truth-match | --tail-prior TRUTH_OUT/dalitz_fit_<factor>.json) \
    [--mass rebuilt|tag_m|tag_dtf_m] [--factor none|flat|weighted] \
    [--tos-levels L0 HLT1 HLT2] [--pnn-mu-min X] [--max-events N] [--horizontal]
python src/plot/plot_dalitz_bins.py OUT/dalitz_fit_<factor>.json [--horizontal]   # re-plot only
```

`driver_dalitz.py` reads ROOT -> TOS (default L0, HLT1 and HLT2) -> best
candidate (min `tag_dtf_chi2`) -> m12, m23 and the `--mass` mass of every
event, then fits them on the 14 x 14 grid (`fit_dalitz.fit_dalitz`). It writes
into `--outdir`:

- `selection.json`: the selection settings, the events dropped as malformed,
  the candidates TOS on each trigger line, and the cutflow
- `dalitz_raw.png`, `dalitz_raw_zoom.png`: raw Dalitz histograms
- `dalitz_fit_<factor>.json`: the global fit, one entry per Dalitz bin, the
  outcome counts, the yield closure, and the mean, median and max chi2/ndof of
  the valid bins
- `global_fit.png`, `dalitz_fit_<factor>_{signal,bkg}_yield.png`,
  `dalitz_fit_<factor>_per_bin_fits.pdf` (with `--horizontal` also
  `*_per_bin_fits_horizontal.pdf`)

`--factor`: `none` (default) fits the plain shapes. `flat` puts
`kinematics.phase_space_factor` on both shapes. `weighted` puts
`kinematics.dalitz_factor` on the signal instead: the fraction of the bin
reachable at M with each point weighted by the eta -> gamma gamma* -> gamma
mu+ mu- (Kroll-Wada) density with the NA60 2016 pole form factor
(`FF_LAMBDA`).

`--truth-match` (signal MC) keeps the events whose best candidate is
truth-matched to eta -> mu+ mu- gamma, and fits with the background fixed to 0
and no prior. Every other run needs `--tail-prior`: the `dalitz_fit_<factor>.json`
of the truth-matched run of the same `--mass`.

`fit_dalitz.ipynb` runs the same selection (`driver_dalitz.select_candidates`)
and fit interactively on one ROOT file.

`src/fit/run_dalitz.sh [--mass "rebuilt tag_m tag_dtf_m"] [--factor "none flat weighted"]`
runs the full sequence for each mass and factor on the 20261006 ntuples: the
truth-matched MC (`out/eta2mumugamma/signal/<today>/dalitz_truth/<mass>`), then
all selected signal MC (`signal/<today>/dalitz/<mass>`) and data
(`turbo/<today>/dalitz/<mass>`) with its tail prior.

The library and the scripts reproduce the per-bin fits of the previous
`fit_dalitz.ipynb` (`PHASE_SPACE` True and False) bin for bin on the 20261006
data.


## Fitting an arbitrary array

The fit modules were written in a generic way so as to fit any numpy array of values:

```python
import sys
sys.path.insert(0, "src/fit")
from fit_dcb import fit_dcb

result = fit_dcb(array, xmin=480, xmax=620, nbins=80)   # array: numpy array of values
```
