#!/bin/bash
# Generative eval: triviaQA + sqad32 + summarization (prompt-0 each), 4 models, base-mode.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_mvp_eval.sh"

MODELS=(
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "google/gemma-4-31B-it"
  "google/gemma-4-26B-A4B-it"
)

declare -A REV
REV["google/gemma-4-31B-it"]="439edf5652646a0d1bd8b46bfdc1d3645761a445"

for M in "${MODELS[@]}"; do
  R="${REV[$M]:-}"
  echo "=== [$(date +%H:%M:%S)] $M : generative_lite ==="
  REVISION="$R" TASKS=benczechmark_generative_lite SUFFIX="_gen" LIMIT=200 \
    "$RUNNER" "$M" || echo "  generative run failed for $M (rc=$?)"
done

echo "=== [$(date +%H:%M:%S)] all generative runs done ==="
