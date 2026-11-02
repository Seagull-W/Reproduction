#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?请传入GPU编号}"
PORT="${2:?请传入服务端口}"
MODEL_DIR="${3:?请传入模型目录}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec vllm serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name mai-gym \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm