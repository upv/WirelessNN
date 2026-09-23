#!/usr/bin/env bash
# IRC x 5 channel estimators x QPSK/16QAM/64QAM, 20 single-thread shards.
#   ./run_irc_ce.sh            start (or continue: shards are resumable)
#   ./run_irc_ce.sh merge      merge finished shards into $OUT
#   TOTAL=20000 OUT=dataset/irc_ce20k ./run_irc_ce.sh
set -euo pipefail
cd "$(dirname "$0")"

TOTAL=${TOTAL:-10000}
OUT=${OUT:-dataset/irc_ce10k}
SHARDS=${SHARDS:-20}
PER=$((TOTAL / SHARDS))

if [[ "${1:-}" == "merge" ]]; then
  exec .venv/bin/python merge_shards.py "$OUT" "$OUT"/shards/s*
fi

mkdir -p "$OUT/shards"
for k in $(seq 0 $((SHARDS - 1))); do
  id=$(printf %02d "$k")
  dir="$OUT/shards/s$id"
  resume=()
  [[ -f "$dir/meta.jsonl" ]] && resume=(--resume)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid nohup .venv/bin/python -m nr_ul_sim.dataset \
    --num-samples "$PER" --start-index $((k * PER)) --outdir "$dir" --seed 0 "${resume[@]}" \
    --channels cdl-b,cdl-c,umi,uma --modulations qpsk,qam16,qam64 --receivers irc \
    --estimators perfect,ls_nn,ls_lin,ls_lin_time_avg,lmmse_ce \
    --num-prb 8 --fft-size 1024 --snr-min -20 --snr-max 40 \
    --coarse-step 4 --refine-db 0.5 --batch-size 2 --max-mc-iter 10 \
    --num-target-bit-errors 200 --num-freq-bins 64 --num-symbol-bins 4 \
    >> "$OUT/shard$id.log" 2>&1 < /dev/null &
done
echo "launched $SHARDS shards -> $OUT/shards"
