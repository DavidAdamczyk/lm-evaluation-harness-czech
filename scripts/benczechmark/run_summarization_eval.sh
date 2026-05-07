#!/bin/bash
# Summarization eval: 5 prompt variants × 200 docs on 4 IT models with chat template + per-model thinking suppression.
# ROUGE-2 F-mid is the only generative metric that survives reasoning-tuned models in lm-eval-harness pipeline.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_mvp_eval.sh"

declare -A THINK
THINK["Qwen/Qwen3.6-27B"]="false"
THINK["Qwen/Qwen3.6-35B-A3B"]="false"
THINK["google/gemma-4-31B-it"]="true"
THINK["google/gemma-4-26B-A4B-it"]="true"

declare -A REV
REV["google/gemma-4-31B-it"]="439edf5652646a0d1bd8b46bfdc1d3645761a445"

MODELS=(
  "Qwen/Qwen3.6-27B"
  "Qwen/Qwen3.6-35B-A3B"
  "google/gemma-4-31B-it"
  "google/gemma-4-26B-A4B-it"
)

for M in "${MODELS[@]}"; do
  R="${REV[$M]:-}"
  T="${THINK[$M]}"
  echo "=== [$(date +%H:%M:%S)] $M : summarization chat + enable_thinking=$T ==="
  REVISION="$R" ENABLE_THINKING="$T" APPLY_CHAT_TEMPLATE=1 \
    TASKS=benczechmark_summarization SUFFIX="_sumr" LIMIT=200 \
    "$RUNNER" "$M" || echo "  summarization run failed for $M (rc=$?)"
done

echo "=== [$(date +%H:%M:%S)] all summarization runs done ==="
