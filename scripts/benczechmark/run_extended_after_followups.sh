#!/bin/bash
# Wait for run_followups.sh to complete, then run benczechmark_extended (16 tasks) on reference models.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_mvp_eval.sh"

echo "[$(date +%H:%M:%S)] waiting for run_followups.sh to finish..."
while pgrep -f run_followups.sh >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] followups done; starting extended runs"

REF_MODELS=(
  "Qwen/Qwen3.6-27B"
  "google/gemma-4-31B"   # base
)

for M in "${REF_MODELS[@]}"; do
  echo "=== [$(date +%H:%M:%S)] $M : EXTENDED limit=200 ==="
  TASKS=benczechmark_extended SUFFIX="_extended" LIMIT=200 \
    "$RUNNER" "$M" || echo "  extended run failed for $M (rc=$?)"
done

echo "=== [$(date +%H:%M:%S)] all extended runs done ==="
