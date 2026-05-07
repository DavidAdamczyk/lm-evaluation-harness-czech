#!/bin/bash
# Orchestration: deploy a model in vLLM, run a BenCzechMark task group, save results.
# Usage: ./run_mvp_eval.sh <hf_repo>
# Example: ./run_mvp_eval.sh Qwen/Qwen3.6-27B
#
# Env vars:
#   PORT, GPU_MEM, MAX_LEN, LIMIT, IMAGE   — vLLM/eval knobs
#   TASKS                                  — lm_eval task or group (default: benczechmark_mvp)
#   SUFFIX                                 — appended to output dir (default: empty)
#   REVISION                               — pin a specific HF revision/commit
#   APPLY_CHAT_TEMPLATE=1                  — pass --apply_chat_template to lm_eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${1:?model HF repo required}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_LEN="${MAX_LEN:-16384}"
LIMIT="${LIMIT:-200}"
IMAGE="${IMAGE:-vllm/vllm-openai:gemma4-0505-cu130}"
CONTAINER_NAME="vllm-mvp-$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '-')"

TASKS="${TASKS:-benczechmark_mvp}"
SUFFIX="${SUFFIX:-}"
EXTRA_LM_EVAL_ARGS=""
if [ "${APPLY_CHAT_TEMPLATE:-0}" = "1" ]; then
  SUFFIX="${SUFFIX}_chat"
  EXTRA_LM_EVAL_ARGS="--apply_chat_template"
fi
OUTPUT_BASE="$REPO_ROOT/results/mvp_$(echo "$MODEL" | sed 's|/|__|g' | tr '.' '_')${SUFFIX}"
mkdir -p "$OUTPUT_BASE"
LOG_FILE="$OUTPUT_BASE/run.log"

echo "=== MVP eval: $MODEL ===" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Image: $IMAGE" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_BASE" | tee -a "$LOG_FILE"

# Stop any prior vllm container
docker ps -aq --filter "name=^vllm-mvp-" | xargs -r docker rm -f >/dev/null 2>&1 || true

echo "[$(date +%H:%M:%S)] starting vLLM container..." | tee -a "$LOG_FILE"
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || echo)" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface/" \
  "$IMAGE" \
  "$MODEL" \
    ${REVISION:+--revision "$REVISION"} \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --trust-remote-code \
    --max-model-len "$MAX_LEN" \
    --max-num-batched-tokens "$MAX_LEN" \
    --max-num-seqs 8 \
    --max-logprobs 500 \
    >> "$LOG_FILE" 2>&1

echo "[$(date +%H:%M:%S)] waiting for vLLM ready (max 25 min)..." | tee -a "$LOG_FILE"
READY=0
for i in $(seq 1 1500); do
  if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "[$(date +%H:%M:%S)] vLLM ready after ${i}s" | tee -a "$LOG_FILE"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[$(date +%H:%M:%S)] CONTAINER DIED. dumping last 100 lines:" | tee -a "$LOG_FILE"
    docker logs --tail 100 "$CONTAINER_NAME" 2>&1 | tee -a "$LOG_FILE"
    exit 2
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "[$(date +%H:%M:%S)] vLLM ready timeout" | tee -a "$LOG_FILE"
  docker logs --tail 200 "$CONTAINER_NAME" 2>&1 | tee -a "$LOG_FILE"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  exit 3
fi

echo "[$(date +%H:%M:%S)] running lm_eval..." | tee -a "$LOG_FILE"
cd "$REPO_ROOT"
source .venv/bin/activate

MODEL_ARGS="model=${MODEL},base_url=http://localhost:${PORT}/v1/completions,num_concurrent=8,tokenized_requests=False,max_length=${MAX_LEN}"
if [ -n "${ENABLE_THINKING:-}" ]; then
  # Forwarded to tokenizer.apply_chat_template via TemplateAPI patch.
  MODEL_ARGS="${MODEL_ARGS},enable_thinking=${ENABLE_THINKING}"
fi
if [ -n "${CHAT_TEMPLATE_FILE:-}" ]; then
  # Path to a Jinja file overriding tokenizer.chat_template (TemplateAPI patch).
  MODEL_ARGS="${MODEL_ARGS},chat_template_file=${CHAT_TEMPLATE_FILE}"
fi

set +e
lm_eval \
  --model local-completions \
  --tasks "$TASKS" \
  --model_args "$MODEL_ARGS" \
  --batch_size 1 \
  --limit "$LIMIT" \
  --output_path "$OUTPUT_BASE" \
  --log_samples \
  $EXTRA_LM_EVAL_ARGS \
  >> "$LOG_FILE" 2>&1
EVAL_RC=$?
set -e

echo "[$(date +%H:%M:%S)] lm_eval rc=$EVAL_RC" | tee -a "$LOG_FILE"
echo "[$(date +%H:%M:%S)] stopping vLLM..." | tee -a "$LOG_FILE"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exit $EVAL_RC
