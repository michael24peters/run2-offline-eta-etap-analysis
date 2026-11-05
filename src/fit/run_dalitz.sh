#!/usr/bin/env bash
# Dalitz fits of eta -> mu+ mu- gamma data from the NTUPLE ntuples into
# out/eta2mumugamma/turbo/<today>/, for each --factor (all by default):
#
#   1. with the DCB tails of TAIL_PRIOR as prior    turbo/<date>/dalitz
#   2. no prior                                     turbo/<date>/dalitz_no_prior
#
# TAIL_PRIOR is the data DCB fit of run_mass_fits.sh. Both jobs pass --n-max N_MAX, the upper limit
# of the DCB tail exponents n_L, n_R.
#
# Every job prints to the terminal with its lines prefixed [job]; failed jobs are listed at the end.
# Jobs start READ_PAUSE seconds apart, so that only one holds a full ROOT read at a time.
#
# Usage:
#   src/fit/run_dalitz.sh [--factor "none flat weighted"]

set -o pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
NTUPLE=20261006
TAIL_PRIOR=out/eta2mumugamma/turbo/20261027/mc_prior/results_dcb.json
N_MAX=50
DATE=$(date +%Y%m%d)
READ_PAUSE=20
FACTORS="none flat weighted"
while [ $# -gt 0 ]; do
    case $1 in
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

TURBO=out/eta2mumugamma/turbo/$DATE/dalitz
NO_PRIOR=out/eta2mumugamma/turbo/$DATE/dalitz_no_prior

for factor in $FACTORS; do
    run "$factor turbo" $PY src/fit/driver_dalitz.py $DATA --outdir $TURBO \
        --factor $factor --tail-prior $TAIL_PRIOR --n-max $N_MAX &
    sleep $READ_PAUSE
    run "$factor no_prior" $PY src/fit/driver_dalitz.py $DATA --outdir $NO_PRIOR \
        --factor $factor --n-max $N_MAX &
    sleep $READ_PAUSE
done
wait

if [ -s "$FAILED" ]; then
    echo "FAILED jobs:"
    cat "$FAILED"
    exit 1
fi
echo "all Dalitz fits done: out/eta2mumugamma/turbo/$DATE/dalitz*"
