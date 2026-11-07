#!/usr/bin/env bash
# Optional short diagnostic run. Invoking this starts a model service locally.
# Does not create or start any CompShare instance. Do not use concurrently with DCGM profiling.
set -euo pipefail
GPU_ID="${1:?GPU index}"
PORT="${2:?port}"
MODEL_DIR="${3:?model directory}"
PROFILE_DIR="${4:?new profile output directory}"
PROFILE_SECONDS="${PROFILE_SECONDS:-30}"
PROFILE_DELAY="${PROFILE_DELAY:-120}"
mkdir "$PROFILE_DIR"
nsys profile --help > "$PROFILE_DIR/nsys-help.txt"
nsys --version > "$PROFILE_DIR/nsys-version.txt"
if rg -q -- '--gpu-metrics-devices' "$PROFILE_DIR/nsys-help.txt"; then
  METRICS_FLAG="--gpu-metrics-devices=$GPU_ID"
else
  METRICS_FLAG="--gpu-metrics-device=$GPU_ID"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec nsys profile --trace=cuda,nvtx,osrt --sample=none \
  "$METRICS_FLAG" --gpu-metrics-frequency=1000 \
  --delay="$PROFILE_DELAY" --duration="$PROFILE_SECONDS" --kill=none \
  --output="$PROFILE_DIR/inference" \
  bash "$SCRIPT_DIR/serve.sh" "$GPU_ID" "$PORT" "$MODEL_DIR"
