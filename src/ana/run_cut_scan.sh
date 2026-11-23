#!/usr/bin/env bash
# Stepwise scan of the offline cuts (src/ana/scan_cuts.py) on the 20261109 ntuples, and the
# cut-step figures (src/plot/plot_cut_steps.py, data) of the chosen cuts, into --outdir
# (default out/explain/<today>/cut_scan/):
#
#   1. eta2mumu: muon cuts, one quantity per step                    eta2mumu/
#   2. eta2mumu data cut steps with the muon cuts of 1                eta2mumu/cut_steps/
#      (fit window 535-560 MeV, 100 bins)
#   3. eta2mumugamma: photon cuts, one quantity per step, on top of   eta2mumugamma/
#      the muon cuts of 1
#   4. eta2mumugamma data cut steps with the muon and photon cuts     eta2mumugamma/cut_steps/
#      of 3 (fit window 450-650 MeV, 200 bins)
#
# Each step takes the cut lists of the one before from its muon_cuts.txt and photon_cuts.txt.
# Every job prints to the terminal with its lines prefixed [job]; the script stops at the first
# failed job.
#
# Usage:
#   src/ana/run_cut_scan.sh [--outdir DIR] [--max-events N] [--max-points N]

set -eo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export MPLBACKEND=Agg   # figures are only saved
NTUPLE=20261109
OUT=out/explain/$(date +%Y%m%d)/cut_scan
SCAN_ARGS=()
PLOT_ARGS=()
while [ $# -gt 0 ]; do
    case $1 in
        --outdir) OUT=$2; shift 2 ;;
        --max-events) SCAN_ARGS+=(--max-events "$2"); PLOT_ARGS+=(--max-events "$2"); shift 2 ;;
        --max-points) SCAN_ARGS+=(--max-points "$2"); shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

# run LABEL CMD...: run CMD with its output prefixed [LABEL]
run() {
    local label=$1; shift
    "$@" 2>&1 | sed -u "s|^|[$label] |"
}

# cut_args FLAG FILE: FLAG and the cuts of FILE (one per line), nothing for an empty FILE
cut_args() {
    local cuts
    mapfile -t cuts < "$2"
    if [ ${#cuts[@]} -gt 0 ]; then
        CUT_ARGS=("$1" "${cuts[@]}")
    else
        CUT_ARGS=()
    fi
}

# 1. eta2mumu muon cuts
run "eta2mumu scan" $PY src/ana/scan_cuts.py --channel eta2mumu --outdir $OUT/eta2mumu \
    "${SCAN_ARGS[@]}"

# 2. eta2mumu cut steps
cut_args --muon-cuts $OUT/eta2mumu/muon_cuts.txt
MUON_ARGS=("${CUT_ARGS[@]}")
run "eta2mumu cut steps" $PY src/plot/plot_cut_steps.py "ntuple/lhcb/eta2mumu/$NTUPLE/mag*/*.root" \
    --outdir $OUT/eta2mumu/cut_steps --fit-xmin 535 --fit-xmax 560 --fit-nbins 100 \
    "${MUON_ARGS[@]}" "${PLOT_ARGS[@]}"

# 3. eta2mumugamma photon cuts on top of the eta2mumu muon cuts
run "eta2mumugamma scan" $PY src/ana/scan_cuts.py --channel eta2mumugamma \
    --outdir $OUT/eta2mumugamma "${MUON_ARGS[@]}" "${SCAN_ARGS[@]}"

# 4. eta2mumugamma cut steps
cut_args --photon-cuts $OUT/eta2mumugamma/photon_cuts.txt
PHOTON_ARGS=("${CUT_ARGS[@]}")
run "eta2mumugamma cut steps" $PY src/plot/plot_cut_steps.py \
    "ntuple/lhcb/eta2mumugamma/$NTUPLE/mag*/*.root" --outdir $OUT/eta2mumugamma/cut_steps \
    --fit-xmin 450 --fit-xmax 650 --fit-nbins 200 "${MUON_ARGS[@]}" "${PHOTON_ARGS[@]}" \
    "${PLOT_ARGS[@]}"

echo "cut scan done: $OUT"
