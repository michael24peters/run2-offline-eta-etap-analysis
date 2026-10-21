#!/usr/bin/env bash
# Dalitz fits of eta -> mu+ mu- gamma from the NTUPLE ntuples into
# out/eta2mumugamma/<turbo|signal>/<today>/, for each --mass and --factor
# (all by default), one mass after the other:
#
#   1. truth-matched signal MC, signal only        signal/<date>/dalitz_truth/<mass>
#   2. all selected signal MC candidates and data,  signal/<date>/dalitz/<mass>
#      with the global-fit tails of 1 as prior      turbo/<date>/dalitz/<mass>
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
MC=ntuple/mc/eta2mumugamma/signal_$NTUPLE/combined_files.root

# run LABEL CMD...: run CMD with its output prefixed [LABEL]; list LABEL in FAILED if it fails
run() {
    local label=$1; shift
    if ! "$@" 2>&1 | sed -u "s|^|[$label] |"; then
        echo "$label: $*" >> "$FAILED"
    fi
}

for mass in $MASSES; do
    truth=out/eta2mumugamma/signal/$DATE/dalitz_truth/$mass
    mc_all=out/eta2mumugamma/signal/$DATE/dalitz/$mass
    turbo=out/eta2mumugamma/turbo/$DATE/dalitz/$mass

    for factor in $FACTORS; do
        run "$mass $factor truth" $PY src/fit/driver_dalitz.py $MC --outdir $truth \
            --mass $mass --factor $factor --truth-match --horizontal &
        sleep $READ_PAUSE
    done
    wait

    for factor in $FACTORS; do
        prior=$truth/dalitz_fit_$factor.json
        run "$mass $factor signal" $PY src/fit/driver_dalitz.py $MC --outdir $mc_all \
            --mass $mass --factor $factor --tail-prior $prior --horizontal &
        sleep $READ_PAUSE
        run "$mass $factor turbo" $PY src/fit/driver_dalitz.py $DATA --outdir $turbo \
            --mass $mass --factor $factor --tail-prior $prior --horizontal &
        sleep $READ_PAUSE
    done
    wait
done

if [ -s "$FAILED" ]; then
    echo "FAILED jobs:"
    cat "$FAILED"
    exit 1
fi
echo "all Dalitz fits done: out/eta2mumugamma/<turbo|signal>/$DATE/dalitz*"
