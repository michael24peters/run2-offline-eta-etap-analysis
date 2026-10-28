#!/usr/bin/env bash
# Dalitz fits of eta -> mu+ mu- gamma data from the NTUPLE ntuples into
# out/eta2mumugamma/turbo/<today>/, for each --mass and --factor (all by
# default), one mass after the other:
#
#   1. with the DCB tails of TAIL_PRIOR as prior    turbo/<date>/dalitz/<mass>
#   2. no prior                                     turbo/<date>/dalitz_no_prior/<mass>
#
# TAIL_PRIOR is the tag_dtf_m data DCB fit of run_mass_fits.sh; it is the prior
# for every --mass. Both jobs pass --n-max N_MAX, the upper limit of the DCB
# tail exponents n_L, n_R.
#
# Every job prints to the terminal with its lines prefixed [job]; failed jobs
# are listed at the end. Jobs start READ_PAUSE seconds apart, so that only one
# holds a full ROOT read (~4 GB for data) at a time.
#
# Usage:
#   src/fit/run_dalitz.sh [--mass "rebuilt tag_m tag_dtf_m"] [--factor "none flat weighted"]

set -o pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
NTUPLE=20261006
TAIL_PRIOR=out/eta2mumugamma/turbo/20261027/mc_prior/results_dcb.json
N_MAX=50
DATE=$(date +%Y%m%d)
READ_PAUSE=20
MASSES="rebuilt tag_m tag_dtf_m"
FACTORS="none flat weighted"
while [ $# -gt 0 ]; do
    case $1 in
        --mass) MASSES=$2; shift 2 ;;
        --factor) FACTORS=$2; shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done
FAILED=$(mktemp)
trap 'rm -f "$FAILED"' EXIT

DATA=ntuple/lhcb/eta2mumugamma/$NTUPLE/combined_files.root

# run LABEL CMD...: run CMD with its output prefixed [LABEL]; list LABEL in FAILED if it fails
run() {
    local label=$1; shift
    if ! "$@" 2>&1 | sed -u "s|^|[$label] |"; then
        echo "$label: $*" >> "$FAILED"
    fi
}

for mass in $MASSES; do
    turbo=out/eta2mumugamma/turbo/$DATE/dalitz/$mass
    no_prior=out/eta2mumugamma/turbo/$DATE/dalitz_no_prior/$mass

    for factor in $FACTORS; do
        run "$mass $factor turbo" $PY src/fit/driver_dalitz.py $DATA --outdir $turbo \
            --mass $mass --factor $factor --tail-prior $TAIL_PRIOR --n-max $N_MAX --horizontal &
        sleep $READ_PAUSE
        run "$mass $factor no_prior" $PY src/fit/driver_dalitz.py $DATA --outdir $no_prior \
            --mass $mass --factor $factor --n-max $N_MAX --horizontal &
        sleep $READ_PAUSE
    done
    wait
done

if [ -s "$FAILED" ]; then
    echo "FAILED jobs:"
    cat "$FAILED"
    exit 1
fi
echo "all Dalitz fits done: out/eta2mumugamma/turbo/$DATE/dalitz*"
