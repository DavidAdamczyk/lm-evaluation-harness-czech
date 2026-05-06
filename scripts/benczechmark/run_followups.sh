#!/bin/bash
# Sequential follow-up runs: csfever NLI (limit=200) + hellaswag (limit=1000) for the 4 reference models.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_mvp_eval.sh"

MODELS=(
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "google/gemma-4-31B-it"
  "google/gemma-4-26B-A4B-it"
)

# gemma-4-31B-it has cache snapshot quirk; pin revision
declare -A REV
REV["google/gemma-4-31B-it"]="439edf5652646a0d1bd8b46bfdc1d3645761a445"

for M in "${MODELS[@]}"; do
  R="${REV[$M]:-}"
  echo "=== [$(date +%H:%M:%S)] $M : NLI limit=200 ==="
  REVISION="$R" TASKS=benczechmark_csfever_nli SUFFIX="_nli" LIMIT=200 \
    "$RUNNER" "$M" || echo "  NLI run failed for $M (rc=$?)"

  echo "=== [$(date +%H:%M:%S)] $M : HellaSwag limit=1000 ==="
  REVISION="$R" TASKS=benczechmark_hellaswag SUFFIX="_hsw1000" LIMIT=1000 \
    "$RUNNER" "$M" || echo "  HSW run failed for $M (rc=$?)"
done

echo "=== [$(date +%H:%M:%S)] all follow-ups done ==="
