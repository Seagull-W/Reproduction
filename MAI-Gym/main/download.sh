#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:?请传入模型保存目录}"

hf download Tongyi-MAI/MAI-Gym-9B \
  --revision 176656ae6a034a0ebf57e8fa67e73015b6e2d98a \
  --local-dir "$MODEL_DIR"