# Fits

## Pipeline

```
driver_*.py   read ROOT → [MC match] → apply selection → pick 1 random candidate/event → numpy masses
   ↓ numpy array
fit_*.py      fit the numbers. Blind to files, cuts, candidates.
   ↓ result dict
plot_*.py     render
```

Awkward arrays hold everything jagged (ROOT I/O, event/candidate structure) up to `cut_utils.values(tree, branch, mask)`, which the drivers call. Numpy holds everything flat after it (histograms, PDFs, curves: `src/utils/fit_utils.py` and the `fit_*` modules).

For MC, MC matching is always the first cut: it is applied before TOS, pnn_mu and the candidate pick. Every MC-matched run (`--mc-match`) therefore reads the raw MC files (`mag*/*.root`), never a `skim_eta.root`: a skim has already picked its candidate with no MC match.

`fit_*.py` are libraries; `driver_*.py` are the entry points.

## Usage

`driver_mass_fit.py` reads a ROOT file, or a quoted glob of them, in chunks of `selection.STEP_SIZE`, selects and picks one candidate per event at random in each chunk, and runs the Gaussian fit, then the DCB fit seeded with the Gaussian sigma. The flag values below are the defaults.

```
python src/fit/driver_mass_fit.py data.root \
    --outdir OUT \
    --xmin 500 --xmax 600 --nbins 100 \
    --mean-init 547.862 --sigma-init 5.0 \
    --alphaL-init 1.5 --nL-init 2.0 --alphaR-init 1.5 --nR-init 2.0
```

It writes into `--outdir`:

- `selection.json`: the selection settings, the events dropped as malformed, the candidates TOS on each trigger line, and the cutflow
- `results_gauss.json`, `gauss_mass_fit.png` (not with `--sigma-prior`)
- `results_dcb.json`, `dcb_mass_fit.png` (not with `--gauss-only`)

`--n-max N` puts an upper limit N on the DCB tail exponents n_L and n_R (no limit by default). The fit chain holds the tails with priors instead, and `run_mass_fits.sh` passes no `--n-max`.

`--max-events N` reads only the first N events.

Plot styling (font, colors, ticks, stats box) is set in `plot_mass_fit.py` and `plot_style.py`, not by flags. Every fit plot starts its stats box with `plot_style.sample_label` of the input path (decay, Data or MC, year), labels its points Data or MC, and counts `Candidates / (<bin width> MeV)` on its y axis (`plot_style.count_label`). The `--mc-match` fits record their fixed n_bkg, c0 and c1 in the meta `fixed_params` of `results_gauss.json` and `results_dcb.json`; their plots leave them out of the stats box and draw no background.

## Fit chain: MC Gaussian, MC DCB, data DCB

```
MC gauss  -(sigma start)->  MC dcb  -(sigma and tail start + prior)->  data dcb
```

The MC fits use MC-matched signal MC (`--mc-match`). It is signal only, so the Gaussian and the DCB both run with no background (n_bkg, c0, c1 fixed at 0). The Gaussian sigma is only the start value of the MC DCB sigma, with no prior. On data the DCB sigma and tails come from the MC DCB: they start at the MC values and are under a `NormalConstraint` prior centred on them, with the MC covariance widened by `SIGMA_PRIOR_FACTOR` (`--sigma-prior`) and `TAIL_PRIOR_FACTOR` (`--tail-prior`) standard deviations, x5 for eta2mumu and eta2mumugamma, x2 for eta2mumumumu. The data run with `--sigma-prior` fits no Gaussian.

eta2mumumumu has one more link: its MC-matched DCB takes the tails of the eta2mumu MC-matched DCB as `--tail-prior` (x2; `mc_tail_prior` in `run_mass_fits.sh`).

```
# 1. MC-matched signal MC: Gaussian, then DCB, both with no background (n_bkg = 0, c0 = c1 = 0);
#    the DCB starts sigma at the Gaussian sigma
python src/fit/driver_mass_fit.py 'ntuple/mc/eta2mumugamma/signal_20261109/pi0_dalitz/mag*/*.root' \
    --outdir out/eta2mumugamma/signal/$(date +%Y%m%d)/mc_matched --mc-match -13 13 22 \
    --xmin 450 --xmax 650 --nbins 200 --sigma-init 13
# 2. data DCB, sigma and tails started at and held near those of 1
python src/fit/driver_mass_fit.py 'ntuple/lhcb/eta2mumugamma/20261109/mag*/*.root' \
    --outdir out/eta2mumugamma/turbo/$(date +%Y%m%d)/mc_prior --xmin 450 --xmax 650 --nbins 200 \
    --sigma-prior out/eta2mumugamma/signal/$(date +%Y%m%d)/mc_matched/results_dcb.json \
    --tail-prior out/eta2mumugamma/signal/$(date +%Y%m%d)/mc_matched/results_dcb.json
```

Other runs of the same driver:

- data Gaussian: `driver_mass_fit.py DATA --gauss-only`, no prior
- MC Gaussian, MC DCB: `--mc-match PID... [--gauss-only]`; without `--mc-match` the MC has candidates not matched to the signal decay, and the fits keep their Chebyshev background floating, as on data

`--mc-match` takes the signal daughter PIDs: `-13 13` for eta -> mu+ mu-, `-13 13 22` for eta -> mu+ mu- gamma, `-13 -13 13 13` for eta -> 4mu. For an `eta2mumumumu` input path, the drivers use `selection.TOS_LINES_4MU` (HLT1 also Hlt1MultiDiMuonNoIP, HLT2 also the rebuilt Hlt2ExoticaQuadMuonNoIP) and `selection.MUON_THRESHOLDS_4MU`; every muon cut applies to all four muons. The MC-matched eta2mumugamma Gaussian starts at `--sigma-init 13`.

`src/fit/run_mass_fits.sh [--channel "eta2mumu eta2mumugamma eta2mumumumu"]` runs the chain for each channel on the 20261109 ntuples into `out/<channel>/<turbo|signal>/<today>/`: the MC-matched MC (`signal/<today>/mc_matched/`) and the data DCB with its sigma and tail priors (`turbo/<today>/mc_prior/`). eta2mumumumu needs eta2mumu in the same run. The script header lists every run, input and fit; `mass_window`, `data_window` and `sigma_init` in the script set the windows and sigma starts per channel.

### Fit stages

Each stage is MIGRAD then HESSE (and MIGRAD and HESSE once more if the fit has not converged, `fit_utils.run_fit`).

| fit | stage 1 | stage 2 |
|---|---|---|
| Gaussian | every parameter not fixed floats | none |
| DCB | alphaL, nL, alphaR, nR fixed at their start values (the MC values with `--tail-prior`); mean, sigma and the parameters not fixed float | every parameter not fixed floats, from the stage-1 values |

With `--mc-match`, n_bkg, c0 and c1 are fixed at 0 in every stage of both fits. The priors (`--sigma-prior`, `--tail-prior`) are terms of the DCB cost in both stages; in stage 1 the tails sit at the centre of their prior.

## Changing the selection

The selection constants (`SEED`, `STEP_SIZE`, `MUON_PID`, `MOTHER_PID`, the branch lists `MC_BRANCHES` and `MUON_CUT_BRANCHES`, the muon cuts `MU_*` and `PNN_MU_MIN`, `MUON_THRESHOLDS`, `MUON_THRESHOLDS_4MU`, `TOS_LINES`, `TOS_LINES_4MU`, `TOS_LINES_DISPLACED`) and the functions that apply them (`branches`, `muon_cuts`, `cut_steps`) are in `src/utils/selection.py`; the mass branch (`MASS_BRANCH`) is set in each driver. Each driver calls `selection.cut_steps` (MC match, TOS per trigger level, then each muon cut, each its own `selection.json` cutflow line), then picks the random candidate, one chunk of `STEP_SIZE` at a time (`cut_utils.iterate_branches`), and sums the reports of the chunks (`cut_utils.add_counts`).

The muon cuts are the muon requirements of the dark-photon ananote (LHCb-ANA-2017-027, Tables 1 and 3) that the existing ntuples can apply, plus `IsMuonTight`, with p > 10 GeV (the 2018 Hlt1DiMuonNoIP threshold) in place of the ananote's 20 GeV. They are constants; `driver_mass_fit.py --pnn-mu-min` overrides the ProbNNmu threshold.

Every cut is structured as a per-candidate boolean mask shaped like `tag_dtf_m`, so they combine with a plain `&` and adding one is a line or two.

TOS means the candidate is TOS on at least one line at every trigger level (AND across levels, OR within a level). `TOS_LINES` maps each level to its TOS branches, and `tos_mask()` takes that dict; a flat list of branches raises a `TypeError`. The drivers use `selection.TOS_LINES`, or `selection.TOS_LINES_4MU` for an `eta2mumumumu` input path (`driver_mass_fit.py`); `src/utils/selection.py` lists their lines.

Cuts on daughters are reduced onto candidates with `cut_utils.reduce()`; its docstring has examples.

Two things to know when writing a cut:

- **Never index the whole tree with a per-candidate mask.** `tree[mask]` silently truncates the `prt_*` fields -- a 3-candidate / 9-daughter event comes back with 3 candidates and 3 daughters, no error. Mask individual `tag_*` branches instead, which is what the drivers do.
- **No need to drop empty events before the candidate pick.** `random_candidate_mask()` selects nothing in an event with no eligible candidate (its final `& eligible` removes the phantom `argmax` winner of an all `-inf` event) and returns a plain boolean mask, so `values()` gives a plain ndarray rather than a masked one.

Cuts are applied *before* the random pick, so the pick is made among the candidates that passed rather than an event being discarded because its arbitrarily-picked candidate failed. For MC, MC matching comes before every other cut. Reordering is a matter of moving lines in `selection.cut_steps`.

`drop_malformed()` removes events with no candidates and events whose `prt_*` branches are inconsistent with `prt_idx_mom`. It works out which branches live at which level from their name prefixes, so **every** `prt_*` branch read is checked as a daughter field and every `tag_*` branch as a candidate field.

Nothing in `cut_utils` prints. `drop_malformed()` and `tos_mask()` return their diagnostics as a second return value, and the driver decides what to log and what to write into `selection.json` under `selection_report`.

When following `mc_idx_mom`, AND in `cut_utils.valid(tree, 'mc_idx_mom')` first. A generator particle with no mother stores -1, which `follow()` treats as a real index, with no error, and so returns the wrong entry. `reduce()` leaves negative indices out of the aggregation.

## Known ntuple quirks

Properties of the ntuple productions that the selection depends on.

**Trigger branch numbering depends on the production.** `tag_<level>_tos<i>` is the i-th line of `l0Trgs` / `hlt1Trgs` / `hlt2Trgs` in the `../ana/src/ntuple.py` that wrote the file. `TOS_LINES` assumes the order of the 20261109 data and MC:

| level | `tos0` | `tos1` | `tos2` | `tos3` | `tos4` |
|---|---|---|---|---|---|
| L0 | L0DiMuon | L0Muon | L0Hadron (TIS line) | | |
| HLT1 | Hlt1DiMuonNoIP | Hlt1DiMuonLowMass | Hlt1MultiDiMuonNoIP | Hlt1TrackMVA (TIS line) | |
| HLT2 | Hlt2ExoticaPrmptDiMuonTurbo | Hlt2ExoticaDiMuonNoIPTurbo | Hlt2Topo (TIS line) | Hlt2ExoticaDisplDiMuon | Hlt2ExoticaQuadMuonNoIP |

HLT2 `tos3` and `tos4` are Full-stream lines that are -1 in Turbo data; `tag_hlt2_rebuilt_tos3` and `tag_hlt2_rebuilt_tos4` hold their TOS on the candidates rebuilt offline.

**`prt_mu`, `prt_loose_mu` and `prt_tight_mu` hold one entry per muon.** In `eta2mumugamma` an event with two candidates has `prt_pid` = [-13, 13, 22, -13, 13, 22] and `prt_tight_mu` = [1, 1, 1, 1]: the photon has no entry, so the branch does not line up with `prt_pid` by position, and `drop_malformed()` drops every event for a daughter-length mismatch if it is read as is. `cut_utils.align_muons()` expands it to one entry per daughter (0 for the photon); it runs between `read_branches()` and `drop_malformed()`. In `eta2mumu` every daughter is a muon and the branch is unchanged.

**Candidates cannot be ranked by `tag_dtf_chi2`.** In the 20261109 `eta2mumugamma` ntuples it is the same for every candidate of most multi-candidate events, up to floating-point rounding. The drivers pick one candidate per event at random (`random_candidate_mask()`, seed `selection.SEED` plus the chunk index).


## Per-Dalitz-bin fits: the phase-space factor

In a Dalitz bin, only part of the m(gamma mu+ mu-) window is kinematically reachable. In the mu+ mu- rest frame, with theta* the mu+ angle and beta = sqrt(1 - 4 m_mu^2 / m23),

    m12 = m_mu^2 + (M^2 - m23) (1 - beta cos theta*) / 2

(m12 = m^2(gamma mu+), m23 = m^2(mu+ mu-)), so cos theta* in [-1, 1] bounds the total mass M at each Dalitz point (`kinematics.m123_limits`). For a bin, `kinematics.phase_space_factor` gives f(M), the fraction of the bin reachable at mass M -- the same thing as sampling (m12, m23) uniformly in the bin and counting how often M is within those limits, but computed without MC noise. f is 1 where the whole bin is reachable and 0 where none of it is, which is what takes edge bins to zero at the ends of the mass window.

`fit_dalitz.fit_dalitz` multiplies the DCB signal and the (plain Chebyshev) background by a factor f in every bin when given one (`signal_factor`, `bkg_factor`; `fit_utils.make_dcb_cost(..., signal_factor_grid=, bkg_factor_grid=)`), and fixes n_sig = 0 in bins whose signal factor is 0 at m_eta, which lie entirely outside the eta's Dalitz region. Without a factor (the default) both shapes are the plain ones. The background has no per-bin prior; mean/sigma keep theirs.

The limits are an identity on the four-vectors, so the fitted mass must come from the same daughter momenta as m12/m23. `MASS_BRANCH` in `driver_dalitz.py` picks the mass, and `MASS_SOURCES` the daughters with it: `tag_dtf_m` (the default) with `prt_dtf_px/py/pz/e`, or `tag_m` with `prt_px/py/pz/e`. The driver prints how many candidates fall outside the limits of their own m12, m23.

Per bin, the Chebyshev's [-1, 1] variable spans the bin's reachable mass range (where f > 0), not [xmin, xmax], so per-bin c0/c1 are not comparable to the global fit's. The plain Chebyshev can still go negative inside the reachable region; the fit floors it at `fit_utils.BKG_FLOOR`, and `bkg_goes_negative` flags any bin where that happened.

The global fit carries a widened tail prior (`--tail-prior`). The prior is the `fit` block of any results JSON: the data DCB fit of `driver_mass_fit.py` (`results_dcb.json`), or the global fit of a `driver_dalitz.py --mc-match` run (`dalitz_fit_<factor>.json`). `--n-max` caps n_L and n_R of the global fit.

Known limitation: f assumes background is uniform across the bin, which does not hold near the low-m12 boundary (the i = 1 column).

### Pipeline

```
python src/fit/driver_dalitz.py 'ntuple/lhcb/eta2mumugamma/20261109/mag*/*.root' \
    --outdir OUT [--mc-match | --tail-prior RESULTS_JSON] \
    [--factor none|flat|weighted] [--n-max N] [--max-events N]
python src/plot/plot_dalitz_bins.py OUT/dalitz_fit_<factor>.json   # re-plot only
```

`driver_dalitz.py` reads ROOT -> MC match (`--mc-match`) -> TOS (L0, HLT1 and HLT2) -> muon cuts -> random candidate -> m12, m23 and the `MASS_BRANCH` mass of every event, then fits them in 450-650 MeV (200 bins), the window of the eta2mumugamma mass fit, on the 14 x 14 grid (`fit_dalitz.fit_dalitz`). It writes into `--outdir`:

- `selection.json`: the selection settings, the events dropped as malformed, the candidates TOS on each trigger line, and the cutflow
- `dalitz_raw.png`, `dalitz_raw_zoom.png`: raw Dalitz histograms
- `dalitz_fit_<factor>.json`: the global fit, one entry per Dalitz bin, the outcome counts, the yield closure, and the mean, median and max chi2/ndof of the valid bins
- `global_fit.png`, `dalitz_fit_<factor>_{signal,bkg}_yield.png`, `dalitz_fit_<factor>_per_bin_fits.pdf`

`--factor`: `none` (default) fits the plain shapes. `flat` puts `kinematics.phase_space_factor` on both shapes. `weighted` puts `kinematics.dalitz_factor` on the signal instead: the fraction of the bin reachable at M with each point weighted by the eta -> gamma gamma* -> gamma mu+ mu- (Kroll-Wada) density with the NA60 2016 pole form factor (`FF_LAMBDA`).

`--mc-match` (signal MC) keeps the candidates MC-matched to eta -> mu+ mu- gamma, before TOS, the muon cuts and the candidate pick, and fits with the background fixed to 0 and no prior. Without `--mc-match` or `--tail-prior` the tails float free.

`fit_dalitz.ipynb` runs the same selection (`driver_dalitz.select_candidates`) and fit interactively on one input.

`src/fit/run_dalitz.sh [--factor "none flat weighted"]` runs the data fit for each factor on the 20261109 eta2mumugamma raw files (`mag*/*.root`), with the tails of the `run_mass_fits.sh` data DCB fit of the same ntuples (`TAIL_PRIOR` in the script) as prior and no `--n-max`, into `out/eta2mumugamma/turbo/<today>/dalitz`.


## Fitting an arbitrary array

`fit_*.py` fit any numpy array and return `(result, m)`:

```python
import sys
sys.path.insert(0, "src/fit")
from fit_dcb import fit_dcb

result, m = fit_dcb(array, xmin=500, xmax=600, nbins=100)   # array: numpy array of values
```
