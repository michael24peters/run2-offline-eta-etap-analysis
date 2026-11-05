#!/usr/bin/env bash
# Mass fits of the NTUPLE ntuples into out/<channel>/<turbo|signal>/<today>/, for each channel (both
# by default). The data DCB fit is the end of a chain of three runs of driver_mass_fit.py, each
# passing a prior to the next through its results JSON:
#
#   1. truth-matched signal MC                                  signal/<date>/truth_no_prior/
#      a. Gaussian + Chebyshev
#      b. DCB with no background (n_bkg, c0, c1 fixed at 0), sigma started at the sigma of 1a
#   2. truth-matched signal MC, --sigma-prior 1a                signal/<date>/truth/
#      a. Gaussian + Chebyshev (the same fit as 1a)
#      b. DCB with no background, sigma under NormalConstraint(sigma of 1a, its stdev times
#         SIGMA_PRIOR_FACTOR) and started there
#   3. data, --tail-prior 2b                                    turbo/<date>/mc_prior/
#      a. Gaussian + Chebyshev
#      b. DCB + Chebyshev, sigma started at the sigma of 3a, the tails alphaL, nL, alphaR, nR
#         under NormalConstraint(tails of 2b, their covariance with stdevs times TAIL_PRIOR_FACTOR)
#         and started there
#
# Every fit runs MIGRAD in two stages, each followed by HESSE; stage 2 starts from the stage-1
# values:
#
#   Gaussian   stage 1: c0, c1 fixed at 0 (flat background)     stage 2: every parameter floats
#   DCB        stage 1: tails fixed at their start values        stage 2: every parameter floats
#
# with n_bkg, c0, c1 of the MC DCB fixed at 0 in both, and the priors in the cost of both.
#
# Comparisons, with no prior: the DCB of 1 (1b), and 4. data, Gaussian then DCB with the tails free
# (turbo/<date>/no_prior/). The data jobs (3, 4) pass --n-max N_MAX, the upper limit of the DCB tail
# exponents n_L, n_R.
#
# Every job prints to the terminal with its lines prefixed [job]; failed jobs are listed at the end.
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
