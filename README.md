# anaroot

LHCb analysis of η → μμ(γ): background classification, mass fits, and Dalitz
analysis.

## Layout

```
src/root/   file prep: fiducial cuts, concatenation (PyROOT for TFileMerger; uproot elsewhere)
src/ana/    MC-truth background classification and diagnostics
src/fit/    mass fits (Gaussian, DCB, Dalitz bin-by-bin) -- see src/fit/README.md
src/plot/   plotting scripts
src/utils/  shared library code (fit math, plot styling, cuts)
ntuple/     input ROOT files
out/        generated plots, JSON fit results
```

ROOT I/O is uproot+awkward everywhere except `src/root/concat_roots.py`, which
uses PyROOT's `TFileMerger` for whole-file merging (no uproot equivalent).

## Environment

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

PyROOT itself isn't pip-installable -- it comes from the LCG/`lb-conda` stack,
needed only for `concat_roots.py`.
