#!/usr/bin/env bash
# Run a data_process command against local vLLM servers (one per model, one GPU group each).
#
#   MODELS="qwen=/shared/models/Qwen3-8B,llama=/shared/models/Llama-3.1-8B-Instruct" \
#   TP=1 CMD='python -m data_process.vote_gt --input raw.jsonl --output voted.jsonl --n 2 --min-agree 0.6 {ENDPOINTS}' \
#   sbatch --partition=gpu --gres=gpu:8 --nodes=1 slurm/data/run_with_vllm.sbatch
#
# {ENDPOINTS} in CMD expands to "--endpoint name=http://127.0.0.1:PORT/v1 ..." for every model.
# Run it from inside the repo (or export RLX_ROOT). Single node on purpose: data jobs are embarrassingly parallel, so run several of them for scale.
#SBATCH --job-name=rlx-data
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
set -euo pipefail

RLX_ROOT="${RLX_ROOT:-$(cd "$(dirname "${SLURM_SUBMIT_DIR:-.}/x")" && git rev-parse --show-toplevel)}"
source "$RLX_ROOT/slurm/common/env.sh"

: "${MODELS:?set MODELS=name=path[,name=path...]}"
: "${CMD:?set CMD to the data_process command}"
TP="${TP:-1}"
BASE_PORT="${BASE_PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NGPU="${GPUS_PER_NODE:-8}"
LOG_DIR="${LOG_DIR:-${SLURM_SUBMIT_DIR:-.}/vllm_logs_${SLURM_JOB_ID:-local}}"
mkdir -p "$LOG_DIR"

IFS=',' read -ra SPECS <<< "$MODELS"
(( ${#SPECS[@]} * TP <= NGPU )) || { echo "need $(( ${#SPECS[@]} * TP )) GPUs, have $NGPU" >&2; exit 2; }

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

ENDPOINTS=""
for i in "${!SPECS[@]}"; do
  name="${SPECS[$i]%%=*}"; path="${SPECS[$i]#*=}"
  port=$(( BASE_PORT + i ))
  first=$(( i * TP )); devs=$(seq -s, "$first" $(( first + TP - 1 )))
  CUDA_VISIBLE_DEVICES="$devs" python -m vllm.entrypoints.openai.api_server \
    --model "$path" --served-model-name "$name" --port "$port" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" ${VLLM_EXTRA_ARGS:-} \
    > "$LOG_DIR/vllm_$name.log" 2>&1 &
  PIDS+=($!)
  ENDPOINTS+=" --endpoint ${name}=http://127.0.0.1:${port}/v1@${name}"
done

for i in "${!SPECS[@]}"; do
  port=$(( BASE_PORT + i ))
  for _ in $(seq 1 "${READY_TIMEOUT_S:-1200}"); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null && break
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "vLLM for ${SPECS[$i]%%=*} died; see $LOG_DIR" >&2; exit 1; }
    sleep 1
  done
done

cd "$RLX_ROOT"
eval "${CMD//\{ENDPOINTS\}/$ENDPOINTS}"
