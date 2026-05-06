#!/bin/bash
# Re-evaluate IT models with --apply_chat_template after the lm_eval JsonChatStr.rstrip patch.
# Runs MVP (4 tasks) + NLI (csfever, limit=200) + HSW (limit=1000) with chat template applied.
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
  echo "=== [$(date +%H:%M:%S)] $M : MVP + chat template ==="
  REVISION="$R" APPLY_CHAT_TEMPLATE=1 TASKS=benczechmark_mvp LIMIT=200 \
    "$RUNNER" "$M" || echo "  MVP+chat run failed for $M (rc=$?)"

  echo "=== [$(date +%H:%M:%S)] $M : NLI + chat template ==="
  REVISION="$R" APPLY_CHAT_TEMPLATE=1 TASKS=benczechmark_csfever_nli SUFFIX="_nli" LIMIT=200 \
    "$RUNNER" "$M" || echo "  NLI+chat run failed for $M (rc=$?)"

  echo "=== [$(date +%H:%M:%S)] $M : HellaSwag + chat template ==="
  REVISION="$R" APPLY_CHAT_TEMPLATE=1 TASKS=benczechmark_hellaswag SUFFIX="_hsw1000" LIMIT=1000 \
    "$RUNNER" "$M" || echo "  HSW+chat run failed for $M (rc=$?)"
done

echo "=== [$(date +%H:%M:%S)] all chat-template re-eval runs done ==="
