#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU index}"
PORT="${2:?port}"
MODEL_DIR="${3:?local pinned model directory}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Two independent 48 GiB replicas avoid unmeasured cross-GPU communication.
# Defaults are a starting profile, not an empirical throughput optimum.
command=(vllm serve "$MODEL_DIR" \
  --host 127.0.0.1 --port "$PORT" --served-model-name mai-gym \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --limit-mm-per-prompt '{"image":10}' \
  --enable-chunked-prefill --enable-prefix-caching \
  --generation-config vllm)
LOG_DIR="${SERVICE_LOG_DIR:-runs/services/gpu_${GPU_ID}_port_${PORT}}"
mkdir -p "$LOG_DIR"
printf '%q ' "${command[@]}" > "$LOG_DIR/launch.command"
printf '\n' >> "$LOG_DIR/launch.command"
python3 -m pip freeze > "$LOG_DIR/packages.txt"
exec > >(tee -a "$LOG_DIR/server.log") 2>&1
exec "${command[@]}"
