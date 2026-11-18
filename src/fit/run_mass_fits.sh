#!/usr/bin/env bash
# Mass fits of the NTUPLE ntuples into out/<channel>/<turbo|signal>/<today>/, for each channel (all
# three by default). The data DCB fit is the end of a chain of two runs of driver_mass_fit.py, the
# first passing a prior to the second through its results JSON:
#
#   1. MC-matched signal MC                                     signal/<date>/mc_matched/
#      a. Gaussian with no background (n_bkg, c0, c1 fixed at 0), sigma started at sigma_init
#      b. DCB with no background (n_bkg, c0, c1 fixed at 0), sigma started at the sigma of 1a.
#         eta2mumumumu: the tails under NormalConstraint(tails of the eta2mumu 1b, their covariance
#         with stdevs times TAIL_PRIOR_FACTOR_4MU) and started there; its 1b runs after eta2mumu's
#   2. data, --sigma-prior 1b --tail-prior 1b                   turbo/<date>/mc_prior/
#      DCB + Chebyshev only: sigma and the tails alphaL, nL, alphaR, nR started at the values of 1b
#      and under NormalConstraint(1b values, 1b covariance with stdevs times SIGMA_PRIOR_FACTOR /
#      TAIL_PRIOR_FACTOR, x2 for eta2mumumumu)
#
# Each MIGRAD stage is followed by HESSE; a DCB's stage 2 starts from its stage-1 values:
#
#   Gaussian   one stage: every parameter not fixed floats
#   DCB        stage 1: tails fixed at their start values        stage 2: every parameter not fixed
#                                                                 floats
#
# with the priors in the cost of both DCB stages. No fit has a hard limit on n_L, n_R. The
# MC-matched jobs fit the window of mass_window, the data jobs that of data_window.
#
# Inputs: the raw files (mag*/*.root) of every channel, data and MC (eta2mumugamma MC:
# pi0_dalitz), no skim. The MC match is the first cut of every MC-matched run, before TOS, the muon
# cuts and the candidate pick.
#
# Every job prints to the terminal with its lines prefixed [job]; failed jobs are listed at the end.
#
# Usage:
#   src/fit/run_mass_fits.sh [--channel "eta2mumu eta2mumugamma eta2mumumumu"]
#   (eta2mumumumu needs eta2mumu in the same run)

set -o pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export MPLBACKEND=Agg   # figures are only saved
NTUPLE=20261109
DATE=$(date +%Y%m%d)
CHANNELS="eta2mumu eta2mumugamma eta2mumumumu"
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

# Data input of each channel: the raw files (a quoted glob)
data_input() {
    echo "ntuple/lhcb/$1/$NTUPLE/mag*/*.root"
}

# Signal MC input of each channel: the raw files (a quoted glob), MC match first
mc_input() {
    case $1 in
        eta2mumugamma) echo "ntuple/mc/$1/signal_$NTUPLE/pi0_dalitz/mag*/*.root" ;;
        *)             echo "ntuple/mc/$1/signal_$NTUPLE/mag*/*.root" ;;
    esac
}

# Signal daughter PIDs of each channel, for --mc-match
daughter_pids() {
    case $1 in
        eta2mumu)      echo "-13 13" ;;
        eta2mumugamma) echo "-13 13 22" ;;
        eta2mumumumu)  echo "-13 -13 13 13" ;;
    esac
}

# Mass window of each channel, at least 100 bins, none wider than 1 MeV
mass_window() {
    case $1 in
        eta2mumugamma) echo "--xmin 450 --xmax 650 --nbins 200" ;;
        *)             echo "--xmin 500 --xmax 600 --nbins 100" ;;
    esac
}

# Data fit window of each channel, at least 100 bins, none wider than 1 MeV: eta2mumu 535-560 MeV,
# the others mass_window
data_window() {
    case $1 in
        eta2mumu) echo "--xmin 535 --xmax 560 --nbins 100" ;;
        *)        mass_window $1 ;;
    esac
}

# Gaussian sigma start of each channel [MeV]
sigma_init() {
    case $1 in
        eta2mumugamma) echo "--sigma-init 13" ;;
        *)             echo "--sigma-init 5" ;;
    esac
}

# Tail prior of the MC-matched DCB: eta2mumumumu takes the tails of the eta2mumu MC-matched DCB
mc_tail_prior() {
    case $1 in
        eta2mumumumu) echo "--tail-prior out/eta2mumu/signal/$DATE/mc_matched/results_dcb.json" ;;
    esac
}

# mc_matched CHANNEL: step 1 for one channel
mc_matched() {
    local channel=$1
    run "$channel mc_matched" $PY src/fit/driver_mass_fit.py "$(mc_input $channel)" \
        --outdir out/$channel/signal/$DATE/mc_matched $(mass_window $channel) \
        $(sigma_init $channel) --mc-match $(daughter_pids $channel) $(mc_tail_prior $channel)
}

# data_dcb CHANNEL: step 2 for one channel
data_dcb() {
    local channel=$1
    local prior=out/$channel/signal/$DATE/mc_matched/results_dcb.json
    run "$channel mc_prior" $PY src/fit/driver_mass_fit.py "$(data_input $channel)" \
        --outdir out/$channel/turbo/$DATE/mc_prior $(data_window $channel) \
        --sigma-prior $prior --tail-prior $prior
}

# Step 1 for every channel, eta2mumumumu after the others (it takes the eta2mumu tails), then
# step 2 for every channel
for channel in $CHANNELS; do
    if [ "$channel" != eta2mumumumu ]; then
        mc_matched $channel &
    fi
done
wait
for channel in $CHANNELS; do
    if [ "$channel" = eta2mumumumu ]; then
        mc_matched $channel
    fi
done
for channel in $CHANNELS; do
    data_dcb $channel &
done
wait

if [ -s "$FAILED" ]; then
    echo "FAILED jobs:"
    cat "$FAILED"
    exit 1
fi
echo "all mass fits done: out/<channel>/<turbo|signal>/$DATE"
