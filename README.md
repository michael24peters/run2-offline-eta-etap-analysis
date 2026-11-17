# anaroot

LHCb analysis of η → μμ(γ)(μμ): background classification, mass fits, and Dalitz analysis.

## Layout

```
src/root/   fiducial requirements (fid_reqs.py), skims (skim.py)
src/ana/    MC background classification and diagnostics
src/fit/    mass fits (Gaussian, DCB, Dalitz bin-by-bin) -- see src/fit/README.md
src/plot/   plotting scripts
src/utils/  shared library code (fit math, plot styling, cuts)
ntuple/     input ROOT files
out/        generated plots, JSON fit results
```

ROOT I/O is uproot + awkward.

## Environment

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
