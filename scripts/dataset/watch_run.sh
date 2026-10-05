#!/usr/bin/env bash
# Merge the sharded run when all shards finish, or stop them at the deadline and
# merge what is done.   DEADLINE_H=23.5 OUT=dataset/irc_ce10k scripts/dataset/watch_run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."   # repository root

OUT=${OUT:-dataset/irc_ce10k}
DEADLINE_H=${DEADLINE_H:-23.5}
start=$(stat -c %Y "$OUT/shards/s00/dataset_config.json")
deadline=$(awk -v s="$start" -v h="$DEADLINE_H" 'BEGIN{printf "%d", s + h * 3600}')
echo "watching $OUT, deadline $(date -d @"$deadline")"

running() { pgrep -f "[p]ython -m nr_ul_sim.dataset .*--outdir $OUT/" | wc -l; }

while (( $(running) > 0 )) && (( $(date +%s) < deadline )); do
  sleep 60
done
if (( $(running) > 0 )); then
  echo "$(date) deadline reached, stopping $(running) shards"
  pkill -f "[p]ython -m nr_ul_sim.dataset .*--outdir $OUT/"
  sleep 10
fi
echo "$(date) merging"
OUT="$OUT" scripts/dataset/run_irc_ce.sh merge && .venv/bin/python scripts/dataset/analyze_ce.py "$OUT"
echo "$(date) finished"
