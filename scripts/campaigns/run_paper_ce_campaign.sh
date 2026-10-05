#!/usr/bin/env bash
# Link-level tests of the paper channel estimators (EqDeepRx, A-MMSE), 68 PRB, IRC.
#   OUT=/root/reports/paper_ce/campaigns WORKERS=2 scripts/campaigns/run_paper_ce_campaign.sh
# Needs models/paper_ce from scripts/training/train_paper_ce.py.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root

OUT=${OUT:-results/paper_ce}
WORKERS=${WORKERS:-2}
MC=${MC:-16}
BATCH=${BATCH:-4}
PY=${PY:-.venv/bin/python}
EST=perfect,ls_lin,ls_soft_window,lmmse_exp,ls_fir,denoise_nn,lmmse_data,lmmse_data_1d,a_mmse,ra_a_mmse
mkdir -p "$OUT/ce" "$OUT/incm" "$OUT/logs"

snr_for() {
  case "$1" in
    qpsk)  echo "-12:20:2" ;;
    qam16) echo "-6:26:2" ;;
    qam64) echo "0:32:2" ;;
  esac
}
speed_for() {  # UMa/UMi at 30 km/h (inside the 0-126 km/h training range), CDL at 3 km/h
  case "$1" in
    uma|umi) echo 30 ;;
    *) echo 3 ;;
  esac
}

JOBS="$OUT/jobs.txt"
: > "$JOBS"
common="--num-prb 68 --fft-size 1024 --batch-size $BATCH --max-mc-iter $MC --num-target-bit-errors 500 --num-target-block-errors 30"

# 1) every estimator with IRC: 4 channels x 3 modulations x {1 UE r1, 2 UE r1, 2 UE r2}, IoT 0 / 10 dB
for ch in uma umi cdl-b cdl-c; do
  for mod in qpsk qam16 qam64; do
    for cfg in "1 1" "2 1" "2 2"; do
      set -- $cfg
      name="ce_${ch}_${mod}_ue$1_r$2"
      echo "$name|$PY -m nr_ul_sim --channel $ch --speed-kmh $(speed_for $ch) --modulation $mod --num-ue $1 --rank $2 --iot-db 0,10 --snr-db=$(snr_for $mod) --receivers irc --estimators $EST $common --outdir $OUT/ce/$name" >> "$JOBS"
    done
  done
done

# 2) interference covariance for IRC: perfect / wideband residual / EqDeepRx INCM (2 PRB, OAS)
for ch in uma cdl-c; do
  for nue in 1 2; do
    for cov in perfect residual incm_oas; do
      name="incm_${ch}_ue${nue}_${cov}"
      echo "$name|$PY -m nr_ul_sim --channel $ch --speed-kmh $(speed_for $ch) --modulation qam16 --num-ue $nue --rank 1 --num-interferers 1 --iot-db 5,10,20 --snr-db=-6:30:2 --receivers lmmse,irc --estimators ls_lin,denoise_nn,lmmse_data --iot-cov $cov $common --outdir $OUT/incm/$name" >> "$JOBS"
    done
  done
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
