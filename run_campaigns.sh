#!/usr/bin/env bash
# Full 68-PRB test matrix: every channel x modulation x receiver x IoT, plus the
# channel-estimator study with IRC. Jobs run in parallel on the GPU.
#   OUT=/root/reports/x/campaigns WORKERS=3 ./run_campaigns.sh
# PRB=12 shrinks the allocation: a 2-UE job with the LMMSE estimators needs
# 21 GB of GPU memory at 68 PRB and 1 GB at 12 PRB.
set -euo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-results/campaigns}
WORKERS=${WORKERS:-3}
MC=${MC:-8}
BATCH=${BATCH:-8}
PY=${PY:-.venv/bin/python}
PRB=${PRB:-68}
ALL_EST=perfect,ls_nn,ls_lin,ls_lin_time_avg,lmmse_ce,lmmse_exp,ls_hard_window,ls_soft_window
mkdir -p "$OUT/receivers" "$OUT/ce" "$OUT/logs"

snr_for() {
  case "$1" in
    qpsk)  echo "-12:20:2" ;;
    qam16) echo "-6:26:2" ;;
    qam64) echo "0:32:2" ;;
  esac
}

JOBS="$OUT/jobs.txt"
: > "$JOBS"
common="--num-prb $PRB --fft-size 1024 --batch-size $BATCH --max-mc-iter $MC --num-target-bit-errors 500 --num-target-block-errors 30"

# 1) receivers: 4 channels x 3 modulations x {1 UE r1, 2 UE r1, 2 UE r2}
for ch in cdl-b cdl-c umi uma; do
  for mod in qpsk qam16 qam64; do
    for cfg in "1 1" "2 1" "2 2"; do
      set -- $cfg
      name="rx_${ch}_${mod}_ue$1_r$2"
      echo "$name|$PY -m nr_ul_sim --channel $ch --modulation $mod --num-ue $1 --rank $2 --iot-db 0,10,20 --snr-db=$(snr_for $mod) --receivers mr,lmmse,ideal_mmse,zf,irc --estimators ls_lin $common --outdir $OUT/receivers/$name" >> "$JOBS"
    done
  done
done

# 2) all channel estimators with IRC: 4 channels x 3 modulations x {1 UE, 2 UE r1}, IoT 0/10
for ch in cdl-b cdl-c umi uma; do
  for mod in qpsk qam16 qam64; do
    for nue in 1 2; do
      name="ce_${ch}_${mod}_ue${nue}"
      echo "$name|$PY -m nr_ul_sim --channel $ch --modulation $mod --num-ue $nue --rank 1 --iot-db 0,10 --snr-db=$(snr_for $mod) --receivers irc --estimators $ALL_EST $common --outdir $OUT/ce/$name" >> "$JOBS"
    done
  done
done

# 3) IRC covariance estimators and speed (CDL-C, 16QAM, 2 UE)
for cov in perfect residual estimated; do
  name="cov_${cov}"
  echo "$name|$PY -m nr_ul_sim --channel cdl-c --modulation qam16 --num-ue 2 --rank 1 --iot-db 10,20 --snr-db=-6:26:2 --receivers lmmse,irc --estimators ls_lin --iot-cov $cov $common --outdir $OUT/receivers/$name" >> "$JOBS"
done
for sp in 3 10 30; do
  name="speed_${sp}"
  echo "$name|$PY -m nr_ul_sim --channel cdl-c --modulation qam16 --num-ue 1 --rank 1 --speed-kmh $sp --iot-db 0 --snr-db=-6:26:2 --receivers irc --estimators $ALL_EST $common --outdir $OUT/ce/$name" >> "$JOBS"
done

echo "$(wc -l < "$JOBS") jobs -> $OUT ($WORKERS workers)"
run_job() {
  local line="$1" name cmd
  name=${line%%|*}; cmd=${line#*|}
  if ls "$OUT"/*/"$name"/*.json >/dev/null 2>&1; then echo "skip $name"; return 0; fi
  local t0=$SECONDS
  if $cmd > "$OUT/logs/$name.log" 2>&1; then
    echo "done $name ($((SECONDS - t0)) s)"
  else
    echo "FAIL $name ($((SECONDS - t0)) s) see $OUT/logs/$name.log"
  fi
}
export -f run_job
export OUT
xargs -P "$WORKERS" -I{} -d '\n' bash -c 'run_job "$@"' _ {} < "$JOBS" 2>&1 | tee -a "$OUT/progress.log"
echo "all jobs finished"
