#!/usr/bin/env bash
# Mass fits of the NTUPLE ntuples into out/<channel>/<turbo|signal>/<today>/,
# for each channel (both by default):
#
#   1. truth-matched signal MC, Gaussian then DCB, no prior   signal/<date>/truth_no_prior/
#   2. truth-matched signal MC, Gaussian then DCB with the
#      Gaussian sigma of 1 as prior                           signal/<date>/truth/
#   3. data, Gaussian then DCB with the DCB tails of 2 as
#      prior                                                  turbo/<date>/mc_prior/
#   4. data, Gaussian then DCB, no prior                      turbo/<date>/no_prior/
#
# 1 -> 2 -> 3 is the chain; 1 and 4 are the no-prior comparisons. The data jobs
# (3, 4) pass --n-max N_MAX, the upper limit of the DCB tail exponents n_L, n_R.
#
# Every job prints to the terminal with its lines prefixed [job]; failed jobs
# are listed at the end.
#
# Usage:
#   src/fit/run_mass_fits.sh [--channel "eta2mumu eta2mumugamma"]

set -o pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
NTUPLE=20261006
N_MAX=50
DATE=$(date +%Y%m%d)
CHANNELS="eta2mumu eta2mumugamma"
while [ $# -gt 0 ]; do
    case $1 in
        --channel) CHANNELS=$2; shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done
FAILED=$(mktemp)
trap 'rm -f "$FAILED"' EXIT

# run LABEL CMD...: run CMD with its output prefixed [LABEL]; list LABEL in FAILED if it fails
run() {
    local label=$1; shift
    if ! "$@" 2>&1 | sed -u "s|^|[$label] |"; then
        echo "$label: $*" >> "$FAILED"
    fi
}

# Signal daughter PIDs of each channel, for --truth-match
daughter_pids() {
    case $1 in
        eta2mumu)      echo "-13 13" ;;
        eta2mumugamma) echo "-13 13 22" ;;
        eta2mumumumu)  echo "-13 -13 13 13" ;;
    esac
}

# mass_fits CHANNEL: steps 1-4 for one channel
mass_fits() {
    local channel=$1
    local data=ntuple/lhcb/$channel/$NTUPLE/combined_files.root
    local mc=ntuple/mc/$channel/signal_$NTUPLE/combined_files.root
    local turbo=out/$channel/turbo/$DATE
    local signal=out/$channel/signal/$DATE
    run "$channel no_prior" $PY src/fit/driver_mass_fit.py $data --outdir $turbo/no_prior --n-max $N_MAX &
    run "$channel truth_no_prior" $PY src/fit/driver_mass_fit.py $mc --outdir $signal/truth_no_prior \
        --truth-match $(daughter_pids $channel)
    run "$channel truth" $PY src/fit/driver_mass_fit.py $mc --outdir $signal/truth \
        --truth-match $(daughter_pids $channel) --sigma-prior $signal/truth_no_prior/results_gauss.json
    run "$channel mc_prior" $PY src/fit/driver_mass_fit.py $data --outdir $turbo/mc_prior \
        --tail-prior $signal/truth/results_dcb.json --n-max $N_MAX
    wait
}

for channel in $CHANNELS; do
    mass_fits $channel &
done
wait

if [ -s "$FAILED" ]; then
    echo "FAILED jobs:"
    cat "$FAILED"
    exit 1
fi
echo "all mass fits done: out/<channel>/<turbo|signal>/$DATE"
