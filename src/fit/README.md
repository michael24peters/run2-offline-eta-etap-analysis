# README

## Pipeline

```
driver_*.py   read ROOT → [truth-match] → apply selection → pick 1 random candidate/event → numpy masses
   ↓ numpy array
fit_*.py      fit the numbers. Blind to files, cuts, candidates.
   ↓ result dict
plot_*.py     render
```

Awkward arrays handle everything jagged (ROOT I/O, event/candidate structure) and live only in `src/utils/cut_utils.py`. Numpy handles everything flat (histogram, PDFs, curves) and lives in `src/utils/fit_utils.py` and the `fit_*` modules. The conversion occurs in the `ak.to_numpy(ak.flatten(...))` call in the driver.

For MC, truth-matching is always the first cut: it is applied before TOS, pnn_mu and the candidate pick.

Most everything is a library, so the drivers are the only entry point for fitting (`fit_dcb.py` / `fit_gauss.py`, etc.).

## Usage

`driver_mass_fit.py` reads a ROOT file, selects, picks one candidate per event at random, and runs the Gaussian fit, then the DCB fit seeded with the Gaussian sigma. The flag values below are the defaults.

```
python src/fit/driver_mass_fit.py data.root \
    --outdir OUT \
    --xmin 500 --xmax 600 --nbins 100 \
    --mean-init 547.862 --sigma-init 5.0 \
    --alphaL-init 1.5 --nL-init 2.0 --alphaR-init 1.5 --nR-init 2.0
```

It writes into `--outdir`:

- `selection.json`: the selection settings, the events dropped as malformed, the candidates TOS on each trigger line, and the cutflow
- `results_gauss.json`, `gauss_mass_fit.png`
- `results_dcb.json`, `dcb_mass_fit.png`

`--n-max N` puts an upper limit N on the DCB tail exponents n_L and n_R (no limit by default).

`--max-events N` reads only the first N events, for quick iteration on large files. The drivers only expose fit hyperparameters as flags. Plot styling (font, colors, ticks, stats box, etc.) is fixed in `plot_mass_fit.py`/`plot_style.py`, and must be changed there manually.

## Fit chain: MC Gaussian, MC DCB, data DCB

```
MC gauss  -(sigma prior)->  MC dcb  -(tail prior)->  data dcb
```

On data the DCB tails are poorly constrained. They come from truth-matched signal MC instead: a `NormalConstraint` prior on (alphaL, nL, alphaR, nR) centred on the MC values, with the MC covariance widened by `TAIL_PRIOR_FACTOR` standard deviations (`--tail-prior`). The truth-matched MC DCB takes its sigma from the truth-matched MC Gaussian: a `NormalConstraint` prior on sigma centred on the Gaussian sigma, with its variance widened by `SIGMA_PRIOR_FACTOR` standard deviations (`--sigma-prior`, with `--truth-match`):

```
# 1. truth-matched signal MC: Gaussian, then DCB with no background (n_bkg = 0, c0 = c1 = 0), no prior
python src/fit/driver_mass_fit.py ntuple/mc/eta2mumugamma/signal_20261006/combined_files.root \
    --outdir out/eta2mumugamma/signal/$(date +%Y%m%d)/truth_no_prior --truth-match -13 13 22
# 2. the same selection, DCB sigma under the Gaussian prior of 1
python src/fit/driver_mass_fit.py ntuple/mc/eta2mumugamma/signal_20261006/combined_files.root \
    --outdir out/eta2mumugamma/signal/$(date +%Y%m%d)/truth --truth-match -13 13 22 \
    --sigma-prior out/eta2mumugamma/signal/$(date +%Y%m%d)/truth_no_prior/results_gauss.json
# 3. data, tails under the MC prior of 2
python src/fit/driver_mass_fit.py ntuple/lhcb/eta2mumugamma/20261006/combined_files.root \
    --outdir out/eta2mumugamma/turbo/$(date +%Y%m%d)/mc_prior \
    --tail-prior out/eta2mumugamma/signal/$(date +%Y%m%d)/truth/results_dcb.json
```

`--truth-match` takes the signal daughter PIDs: `-13 13` for eta -> mu+ mu-, `-13 13 22` for eta -> mu+ mu- gamma, `-13 -13 13 13` for eta -> 4mu. For an `eta2mumumumu` input path, HLT1 TOS also accepts Hlt1MultiDiMuonNoIP (`TOS_LINES_4MU`); every muon cut applies to all four muons. The Gaussian fit keeps its Chebyshev background floating: with no background, a Gaussian has no stable minimum on the radiative tail of the MC. The DCB fit runs with no background.

`src/fit/run_mass_fits.sh [--channel "eta2mumu eta2mumugamma"]` runs the chain for each channel on the 20261006 ntuples, into `out/<channel>/<turbo|signal>/<today>/`: the truth-matched MC without a prior (`signal/<today>/truth_no_prior/`), with the sigma prior (`signal/<today>/truth/`), and data with its tail prior (`turbo/<today>/mc_prior/`). Data with no prior (`turbo/<today>/no_prior/`) and the truth-matched MC without a prior are the comparison runs. The data jobs pass `--n-max 50` (`N_MAX` in the script). The header of `run_mass_fits.sh` lists every run and fit in order.

### Fit stages

`fit_gauss()` and `fit_dcb()` each fit in two MIGRAD stages, each followed by HESSE (and by MIGRAD and HESSE once more if the fit has not converged, `fit_utils.run_fit`). Stage 2 starts from the stage-1 values.

| fit | stage 1 | stage 2 |
|---|---|---|
| Gaussian | c0, c1 fixed at 0 (flat background); mean, sigma, n_sig, n_bkg float | every parameter floats |
| DCB | alphaL, nL, alphaR, nR fixed at their start values (the MC values with `--tail-prior`); mean, sigma, yields, c0, c1 float | every parameter floats |

With `--truth-match`, n_bkg, c0 and c1 of the DCB stay fixed at 0 through both stages. The priors (`--sigma-prior`, `--tail-prior`) are terms of the cost in both stages. The DCB starts sigma at the Gaussian sigma of the same run (at the prior's with `--sigma-prior`).

## Changing the selection

The selection constants (`SEED`, the muon cuts `MU_*` and `PNN_MU_MIN`, `TOS_LINES`, `TOS_LINES_4MU`) and the functions that apply them (`muon_cuts`, `cut_steps`) are in `src/utils/selection.py`, shared by `driver_mass_fit.py`, `driver_dalitz.py` and `plot_cut_steps.py`; the mass branch (`MASS_BRANCH`) is set in each driver. Each driver calls `selection.cut_steps` (truth match, TOS per trigger level, then each muon cut, each its own `selection.json` cutflow line), then picks the random candidate.

The muon cuts are the muon requirements of the dark-photon ananote (LHCb-ANA-2017-027, Tables 1 and 3) that the existing ntuples can apply, plus `IsMuonTight`, with p > 10 GeV (the 2018 Hlt1DiMuonNoIP threshold) in place of the ananote's 20 GeV. They are constants, not flags.

Every cut is structured as a per-candidate boolean mask shaped like `tag_dtf_m`, so they combine with a plain `&` and adding one is a line or two.

TOS means the candidate is TOS on at least one line at every trigger level (AND across levels, OR within a level). `TOS_LINES` maps each level to its TOS branches, and `tos_mask()` takes that dict; a flat list of branches raises a `TypeError`. Both drivers, and the notebook through `select_candidates`, use `selection.TOS_LINES`:

```python
TOS_LINES = {
    'L0':   ['tag_l0_tos0', 'tag_l0_tos1'],       # L0DiMuon, L0Muon
    'HLT1': ['tag_hlt1_tos0'],                    # Hlt1DiMuonNoIP
    'HLT2': ['tag_hlt2_tos0'],                    # Hlt2ExoticaPrmptDiMuonTurbo
}
```

Most selection criteria (i.e., kinematic cuts) can be applied with `reduce()`:

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

The full pipeline:

```python
tree, drop_report = cut_utils.drop_malformed(cut_utils.read_branches(infile, BRANCHES))
keep   = ...                                                   # as above
picked = cut_utils.random_candidate_mask(keep, SEED)
masses = cut_utils.values(tree, MASS_BRANCH, picked)           # flat numpy
result = fit_gauss(masses, ...)
```

`values()` is where awkward is converted to numpy: everything upstream is jagged and per-event, everything downstream is a flat array of numbers.

Two things to know before editing that block:

- **Never index the whole tree with a per-candidate mask.** `tree[mask]` silently truncates the `prt_*` fields -- a 3-candidate / 9-daughter event comes back with 3 candidates and 3 daughters, no error. Mask individual `tag_*` branches instead, which is what the drivers do.
- **No need to drop empty events before the candidate pick.** `random_candidate_mask()` selects nothing in an event with no eligible candidate (its final `& eligible` removes the phantom `argmax` winner of an all `-inf` event) and returns a plain boolean mask, so `values()` gives a plain ndarray rather than a masked one.

Cuts are applied *before* the random pick, so the pick is made among the candidates that passed rather than an event being discarded because its arbitrarily-picked candidate failed. For MC, truth-matching comes before every other cut. Reordering is a matter of moving lines in `selection.cut_steps`.

`drop_malformed()` removes events with no candidates and events whose `prt_*` branches are inconsistent with `prt_idx_mom`. It works out which branches live at which level from their name prefixes, so **every** `prt_*` branch read is checked as a daughter field and every `tag_*` branch as a candidate field — forgetting to list one is how a ragged event survives to blow up inside `reduce()` much later.

Nothing in `cut_utils` prints. `drop_malformed()` and `tos_mask()` return their diagnostics as a second return value, and the driver decides what to log and what to write into `selection.json` under `selection_report`.

When following `mc_idx_mom`, AND in `cut_utils.valid(tree, 'mc_idx_mom')` first. A generator particle with no mother stores -1, which `follow()` treats as a real index, with no error, and so returns the wrong entry. `reduce()` leaves negative indices out of the aggregation.

## Known ntuple quirks

Properties of the ntuple productions that the selection depends on.

**Trigger branch numbering depends on the production.** `tag_<level>_tos<i>` is the i-th line of `l0Trgs` / `hlt1Trgs` / `hlt2Trgs` in the `../ana/src/ntuple.py` that wrote the file, and that order has changed. `TOS_LINES` assumes the July 2026 order or later:

| ntuple | `tag_hlt2_tos0` | `tag_hlt2_tos1` | `tag_hlt2_tos2` |
|---|---|---|---|
| data 20260507, 20260608 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDisplDiMuon | Hlt2ExoticaDiMuonNoIPTurbo |
| data and MC 20260716 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | Hlt2ExoticaDisplDiMuon |
| `ntuple.py` since 2026-08-05 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | -- |
| `ntuple.py` since 2026-10-05 (MC 20261006) | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | Hlt2Topo (TIS line) |

L0 (`tos0` L0DiMuon, `tos1` L0Muon) and HLT1 (`tos0` Hlt1DiMuonNoIP, `tos1` Hlt1DiMuonLowMass) keep the same order in all of them. Since 2026-10-05 each level has further lines after those: L0 `tos2` L0Hadron (TIS line), HLT1 `tos2` Hlt1MultiDiMuonNoIP and `tos3` Hlt1TrackMVA (TIS line).

**`prt_mu`, `prt_loose_mu` and `prt_tight_mu` hold one entry per muon.** In `eta2mumugamma` an event with two candidates has `prt_pid` = [-13, 13, 22, -13, 13, 22] and `prt_tight_mu` = [1, 1, 1, 1]: the photon has no entry, so the branch does not line up with `prt_pid` by position, and `drop_malformed()` drops every event for a daughter-length mismatch if it is read as is. `cut_utils.align_muons()` expands it to one entry per daughter (0 for the photon); it runs between `read_branches()` and `drop_malformed()`. In `eta2mumu` every daughter is a muon and the branch is unchanged.

**Candidates cannot be ranked by `tag_dtf_chi2`.** In the 20261006 `eta2mumugamma` ntuples it is the same for every candidate of most multi-candidate events, up to floating-point rounding. The drivers pick one candidate per event at random (`random_candidate_mask()`, seed `selection.SEED`).


## Per-Dalitz-bin fits: the phase-space factor

Full write-up, with derivation, figures and a code walkthrough: `docs/phase-space-factor.md`.

In a Dalitz bin, only part of the m(gamma mu+ mu-) window is kinematically reachable. In the mu+ mu- rest frame, with theta* the mu+ angle and beta = sqrt(1 - 4 m_mu^2 / m23),

    m12 = m_mu^2 + (M^2 - m23) (1 - beta cos theta*) / 2

(m12 = m^2(gamma mu+), m23 = m^2(mu+ mu-)), so cos theta* in [-1, 1] bounds the total mass M at each Dalitz point (`kinematics.m123_limits`). For a bin, `kinematics.phase_space_factor` gives f(M), the fraction of the bin reachable at mass M -- the same thing as sampling (m12, m23) uniformly in the bin and counting how often M is within those limits, but computed without MC noise. f is 1 where the whole bin is reachable and 0 where none of it is, which is what takes edge bins to zero at the ends of the mass window.

`fit_dalitz.fit_dalitz` multiplies the DCB signal and the (plain Chebyshev) background by a factor f in every bin when given one (`signal_factor`, `bkg_factor`; `fit_utils.make_dcb_cost(..., signal_factor_grid=, bkg_factor_grid=)`), and fixes n_sig = 0 in bins whose signal factor is 0 at m_eta, which lie entirely outside the eta's Dalitz region. Without a factor (the default) both shapes are the plain ones. The background has no per-bin prior; mean/sigma keep theirs.

The limits are an identity on the four-vectors, so the fitted mass must come from the same daughter momenta as m12/m23. `MASS_BRANCH` in `driver_dalitz.py` picks the mass, and `MASS_SOURCES` the daughters with it: `tag_dtf_m` (the default) with `prt_dtf_px/py/pz/e`, or `tag_m` with `prt_px/py/pz/e`. The driver prints how many candidates fall outside the limits of their own m12, m23.

Per bin, the Chebyshev's [-1, 1] variable spans the bin's reachable mass range (where f > 0), not [xmin, xmax]: fit on only a slice of its range, its coefficients are near-degenerate and MIGRAD runs them off to infinity. So per-bin c0/c1 are not comparable to the global fit's. The plain Chebyshev can still go negative inside the reachable region; the fit floors it at `fit_utils.BKG_FLOOR`, and `bkg_goes_negative` flags any bin where that happened.

The global fit carries a widened tail prior (`--tail-prior`). The prior is the `fit` block of any results JSON: the data DCB fit of `driver_mass_fit.py` (`results_dcb.json`), or the global fit of a `driver_dalitz.py --truth-match` run (`dalitz_fit_<factor>.json`). `--n-max` caps n_L and n_R of the global fit.

Known limitation: f assumes background is uniform across the bin, which does not hold near the low-m12 boundary (the i = 1 column).

### Pipeline

```
python src/fit/driver_dalitz.py ntuple/lhcb/eta2mumugamma/20261006/combined_files.root \
    --outdir OUT [--truth-match | --tail-prior RESULTS_JSON] \
    [--factor none|flat|weighted] [--n-max N] [--max-events N]
python src/plot/plot_dalitz_bins.py OUT/dalitz_fit_<factor>.json   # re-plot only
```

`driver_dalitz.py` reads ROOT -> truth match (`--truth-match`) -> TOS (L0, HLT1 and HLT2) -> muon cuts -> random candidate -> m12, m23 and the `MASS_BRANCH` mass of every event, then fits them on the 14 x 14 grid (`fit_dalitz.fit_dalitz`). It writes into `--outdir`:

- `selection.json`: the selection settings, the events dropped as malformed, the candidates TOS on each trigger line, and the cutflow
- `dalitz_raw.png`, `dalitz_raw_zoom.png`: raw Dalitz histograms
- `dalitz_fit_<factor>.json`: the global fit, one entry per Dalitz bin, the outcome counts, the yield closure, and the mean, median and max chi2/ndof of the valid bins
- `global_fit.png`, `dalitz_fit_<factor>_{signal,bkg}_yield.png`, `dalitz_fit_<factor>_per_bin_fits.pdf`

`--factor`: `none` (default) fits the plain shapes. `flat` puts `kinematics.phase_space_factor` on both shapes. `weighted` puts `kinematics.dalitz_factor` on the signal instead: the fraction of the bin reachable at M with each point weighted by the eta -> gamma gamma* -> gamma mu+ mu- (Kroll-Wada) density with the NA60 2016 pole form factor (`FF_LAMBDA`).

`--truth-match` (signal MC) keeps the candidates truth-matched to eta -> mu+ mu- gamma, before TOS, the muon cuts and the candidate pick, and fits with the background fixed to 0 and no prior. Without `--truth-match` or `--tail-prior` the tails float free.

`fit_dalitz.ipynb` runs the same selection (`driver_dalitz.select_candidates`) and fit interactively on one ROOT file.

`src/fit/run_dalitz.sh [--factor "none flat weighted"]` runs the data fits for each factor on the 20261006 ntuples, both with `--n-max 50`: with the tails of the 20261027 data DCB mass fit (`turbo/20261027/mc_prior/results_dcb.json`, `TAIL_PRIOR`) as prior (`out/eta2mumugamma/turbo/<today>/dalitz`), and with no prior (`turbo/<today>/dalitz_no_prior`).


## Fitting an arbitrary array

The fit modules were written in a generic way so as to fit any numpy array of values:

```python
import sys
sys.path.insert(0, "src/fit")
from fit_dcb import fit_dcb

result = fit_dcb(array, xmin=500, xmax=600, nbins=100)   # array: numpy array of values
```
