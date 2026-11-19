#!/usr/bin/env bash
# Train every ce_zoo model in sequence on the Channelformer 4x2 dataset, then print a summary.
# Run from the repository root:
#   bash experiments/ce_zoo/run_all.sh [data dir, default data/cf_4x2] [runs dir, default runs/zoo]
# Environment: PYTHON (default python), EPOCHS (60), MAX_HOURS (2 per model), DMRS (2,11).
set -u
DATA=${1:-data/cf_4x2}
RUNS=${2:-runs/zoo}
PY=${PYTHON:-python}
export PYTHONPATH=.:experiments/ce_zoo
mkdir -p "$RUNS"

$PY experiments/ce_zoo/models.py || exit 1
for m in swift denoiserx chea helena chea_xl; do
  if [ -f "$RUNS/$m/summary.json" ]; then echo "skip $m"; continue; fi
  echo "=== $m $(date -u +%FT%TZ)"
  $PY experiments/ce_zoo/train.py --model $m --data "$DATA" --out "$RUNS/$m" --dmrs "${DMRS:-2,11}" \
      --epochs "${EPOCHS:-60}" --max-hours "${MAX_HOURS:-2}" > "$RUNS/$m.log" 2>&1
  echo "exit $? $(tail -1 "$RUNS/$m.log")"
done
$PY experiments/ce_zoo/summarize.py "$RUNS" | tee "$RUNS/SUMMARY.md"
