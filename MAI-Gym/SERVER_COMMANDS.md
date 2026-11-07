# 服务器命令清单

适用：Ubuntu/Debian、root shell、两个标称 48GB GPU；Docker provider。这里只提供命令，未在实例执行。CompShare 容器必须确实拥有 KVM 和 Docker daemon 权限；仅安装 Python 包或 Docker CLI 不能赋予宿主机权限。下载和依赖准备尽量在无卡环境完成，GPU 检查与模型启动在用户自行启动有卡模式后执行。

## 1. 同步当前代码（本地执行）

当前代码尚不能假定已推送 GitHub，所以直接打包本地目录。实例可连接后再执行 cp。

```bash
tar --exclude='MAI-Gym/runs' --exclude='MAI-Gym/.venv' \
    --exclude='__pycache__' --exclude='*.pyc' \
    -C /home/seagull/Reproduction -czf /tmp/MAI-Gym.tar.gz MAI-Gym
compshare --json instance cp cpod-1x2b7fhaphm7 /tmp/MAI-Gym.tar.gz :/root/MAI-Gym.tar.gz
```

服务器解压（已有目录时先保留旧版本；不要覆盖唯一实验数据）：

```bash
mkdir -p /root/Reproduction
tar -xzf /root/MAI-Gym.tar.gz -C /root/Reproduction
```

## 2. 设置目录、查看真实硬件（服务器执行）

`MAI_DATA` 必须改为有足够空间的实际挂载目录。已查系统盘只有 50 GiB，不能假定它足够放两个 Python 环境、权重、qcow2、Docker 写层和原始轨迹。此处不购买或扩容存储。

```bash
export PROJECT=/root/Reproduction/MAI-Gym
export MAI_DATA=/data/mai-eval
mkdir -p "$MAI_DATA"
df -hT "$MAI_DATA"
findmnt -T "$MAI_DATA"

cat > "$PROJECT/server.env" <<'ENV'
export PROJECT=/root/Reproduction/MAI-Gym
export MAI_DATA=/data/mai-eval
export OSWORLD="$MAI_DATA/OSWorld"
export INFER="$MAI_DATA/venvs/infer"
export EVAL="$MAI_DATA/venvs/eval"
export MODEL="$MAI_DATA/models/MAI-Gym-9B"
export MODEL_REV=176656ae6a034a0ebf57e8fa67e73015b6e2d98a
export HF_HOME="$MAI_DATA/hf-cache"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$OSWORLD${PYTHONPATH:+:$PYTHONPATH}"
export SERVICE_LOG_DIR="$MAI_DATA/service-logs"
export MAX_MODEL_LEN=32768
export MAX_NUM_SEQS=8
export MAX_BATCHED_TOKENS=8192
export GPU_MEMORY_UTILIZATION=0.90
ENV
source "$PROJECT/server.env"
mkdir -p "$MAI_DATA/setup" "$MAI_DATA/runs" "$MAI_DATA/models" "$MAI_DATA/venvs"

# 有卡模式下运行：
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
nvidia-smi topo -m
nproc
free -h
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max
ls -l /dev/kvm
```

每次新终端先 `source /root/Reproduction/MAI-Gym/server.env`。修改目录时同时修改上面的初始 MAI_DATA 和 server.env。

## 3. 系统依赖与 Docker/KVM

```bash
apt-get update
apt-get install -y git curl ca-certificates unzip build-essential pkg-config \
    tmux qemu-utils ffmpeg libgl1 libglib2.0-0 libsndfile1 libchromaprint1 \
    tesseract-ocr libasound2-dev portaudio19-dev

# 仅在没有 Docker CLI 时安装。启动 daemon 需要宿主机提供对应能力。
if ! command -v docker >/dev/null 2>&1; then
    apt-get install -y docker.io
fi

docker version
docker info
python3 - <<'PY'
import os
assert os.path.exists('/dev/kvm'), '没有 /dev/kvm：不要继续同机 Docker 评测'
assert os.access('/dev/kvm', os.R_OK | os.W_OK), '/dev/kvm 不可读写：需平台开放权限'
PY
```

如果 docker info 无法连接 daemon，或 KVM 检查失败，停止同机 Docker 路线。不要在普通容器里盲目运行 systemctl、modprobe 或 privileged 命令来绕过平台限制。先让平台提供可用宿主机能力，或在另一台支持 KVM 的机器运行 OSWorld。

## 4. 两个独立 Python 环境

OSWorld 当前依赖旧版 torch/transformers，不能与 vLLM 混装。下面固定 OSWorld 源码到本次查阅 commit；vLLM 首次安装后记录实际版本，并检查是否具有该模型实现。这里未声称整套依赖已在目标机器解析/安装成功。

```bash
python3 -m pip install --user -U uv
source "$PROJECT/server.env"
uv venv --python 3.12 --seed "$INFER"
uv venv --python 3.12 --seed "$EVAL"

uv pip install --python "$INFER/bin/python" --torch-backend=auto \
    vllm huggingface_hub
"$INFER/bin/python" -c 'import torch, vllm; from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration; print(torch.__version__, vllm.__version__, torch.cuda.is_available())'
"$INFER/bin/python" -m pip freeze > "$MAI_DATA/setup/infer.freeze.txt"

# 以下假定 OSWORLD 目录尚不存在；已有 checkout 时不要覆盖本地修改。
git clone https://github.com/xlang-ai/OSWorld.git "$OSWORLD"
git -C "$OSWORLD" checkout --detach 8903ebeb0efdf74331bd7dc37ed91ccbbff22317
git -C "$OSWORLD" rev-parse HEAD > "$MAI_DATA/setup/osworld.commit.txt"

uv pip install --python "$EVAL/bin/python" 'numpy==1.26.4' Cython setuptools wheel
uv pip install --python "$EVAL/bin/python" --torch-backend=cpu \
    --no-build-isolation-package fastdtw \
    -r "$OSWORLD/requirements.txt" -r "$PROJECT/requirements-eval.txt"
"$EVAL/bin/python" -m playwright install chromium
uv pip check --python "$EVAL/bin/python"
"$EVAL/bin/python" -m pip freeze > "$MAI_DATA/setup/eval.freeze.txt"
cd "$OSWORLD"
"$EVAL/bin/python" -c 'from desktop_env.desktop_env import DesktopEnv; print("DesktopEnv import OK")'
"$EVAL/bin/python" "$PROJECT/main/preflight.py" \
    --provider docker --output "$MAI_DATA/setup/preflight.json"
```

如果 Qwen3.5 实现导入失败，当前 vLLM 不适用；不要直接跳过继续启动。先定位版本并按官方安装说明选择支持版本。若发生依赖解析失败，保留完整日志，不使用 --no-deps 隐藏冲突。

## 5. 下载权重和桌面镜像（不启动环境）

```bash
PATH="$INFER/bin:$PATH" bash "$PROJECT/main/download.sh" "$MODEL"

docker pull happysixd/osworld-docker
docker image inspect happysixd/osworld-docker \
    > "$MAI_DATA/setup/docker-image.json"

cd "$OSWORLD"
"$EVAL/bin/python" - <<'PY'
import os
from pathlib import Path
from desktop_env.providers.docker.manager import DockerVMManager
path = Path(DockerVMManager().get_vm_path(os_type='Ubuntu', region=None)).resolve()
assert path.is_file(), path
(Path(os.environ['MAI_DATA']) / 'setup/base-image.path').write_text(str(path))
print(path)
PY
sha256sum "$(cat "$MAI_DATA/setup/base-image.path")" \
    > "$MAI_DATA/setup/base-image.sha256"
```

当前 Docker provider 会为每个 DesktopEnv 创建独立容器，qcow2 挂载为只读，所以后面的 worker 可以共享同一个基础镜像路径。每个容器仍会产生自己的运行时写层、占用 CPU/RAM/磁盘。

上游标注公开 Ubuntu 镜像可能缺少截图 X11 连接泄漏修复。正式长测前查看 `$OSWORLD/docs/guest-image-status.md`，完成 guest 内修复/新镜像验收；只更新 host 代码不能证明 guest 已修复。短冒烟通过也不能证明长测通过。

## 6. 生成实际配置与任务清单

```bash
source "$PROJECT/server.env"
"$EVAL/bin/python" - <<'PY'
import json, os
from pathlib import Path
project = Path(os.environ['PROJECT'])
root = Path(os.environ['OSWORLD'])
data = Path(os.environ['MAI_DATA'])
image = (data/'setup/base-image.path').read_text().strip()
workers = [dict(endpoint=f'http://127.0.0.1:{8000+i//4}/v1',
                gpu_index=i//4, path_to_vm=image) for i in range(8)]
config = project/'configs'
(config/'workers.server.json').write_text(json.dumps(workers, indent=2))
(config/'workers.smoke.json').write_text(json.dumps(workers[:1], indent=2))
index = json.loads((root/'evaluation_examples/test_all.json').read_text())
paths = [root/'evaluation_examples/examples'/domain/f'{task_id}.json'
         for domain, ids in index.items() for task_id in ids]
assert paths and all(p.is_file() for p in paths)
(config/'tasks.server.json').write_text(json.dumps([str(p) for p in paths], indent=2))
smoke = next(p for p in paths if p.parent.name == 'os')
(config/'tasks.smoke.json').write_text(json.dumps([str(smoke)], indent=2))
print('tasks:', len(paths), 'smoke:', smoke)
print('smoke instruction:', json.loads(smoke.read_text())['instruction'])
PY
```

先读打印的冒烟任务，确认其操作和配置适合调试。全量清单由固定版本的官方 test_all.json 生成，不把测试目录中的其他 JSON 混进去；依赖 Google/OAuth 等任务仍需按 OSWorld 的 SETUP_GUIDELINE.md 准备。不能把凭据缺失产生的失败当模型能力结论。

## 7. 启动双副本服务（明确消耗 GPU）

创建仅用于本次项目的启动脚本，tmux 会话只运行对应副本：

```bash
cat > "$PROJECT/start_replica.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source /root/Reproduction/MAI-Gym/server.env
source "$INFER/bin/activate"
cd "$PROJECT"
export SERVICE_LOG_DIR="$MAI_DATA/service-logs/gpu_$1"
exec bash main/serve.sh "$1" "$2" "$MODEL"
SH

tmux new-session -d -s mai-gpu0 "bash '$PROJECT/start_replica.sh' 0 8000"
tmux new-session -d -s mai-gpu1 "bash '$PROJECT/start_replica.sh' 1 8001"

tmux ls
tail -n 80 "$MAI_DATA/service-logs/gpu_0/server.log"
tail -n 80 "$MAI_DATA/service-logs/gpu_1/server.log"
curl --fail --silent http://127.0.0.1:8000/v1/models
curl --fail --silent http://127.0.0.1:8001/v1/models
```

模型加载可能较慢；只有两次 /models 返回 mai-gym 后再执行评测。若显存不足，先检查是否有其他 GPU 进程，再调低 MAX_NUM_SEQS/MAX_BATCHED_TOKENS；不要直接减少截图窗口或输出上限来改变实验定义。

## 8. 计划检查与单任务冒烟

所有评测从 OSWORLD 目录执行，避免官方任务的相对资源路径错位。输出目录必须全新。

```bash
source "$PROJECT/server.env"
cd "$OSWORLD"
"$EVAL/bin/python" "$PROJECT/main/run_batch.py" \
    --manifest-path "$PROJECT/configs/tasks.smoke.json" \
    --workers-path "$PROJECT/configs/workers.smoke.json" \
    --prompt-path "$PROJECT/main/system_prompt.txt" --provider docker \
    --model-revision "$MODEL_REV" --model-dir "$MODEL" \
    --output-dir "$MAI_DATA/runs/plan-001" \
    --max-steps 15 --max-tokens 2048 --trials 1 --time-limit 1800 --dry-run

"$EVAL/bin/python" "$PROJECT/main/run_batch.py" \
    --manifest-path "$PROJECT/configs/tasks.smoke.json" \
    --workers-path "$PROJECT/configs/workers.smoke.json" \
    --prompt-path "$PROJECT/main/system_prompt.txt" --provider docker \
    --model-revision "$MODEL_REV" --model-dir "$MODEL" \
    --output-dir "$MAI_DATA/runs/smoke-001" \
    --max-steps 15 --max-tokens 2048 --trials 1 --time-limit 1800 \
    --episode-timeout 1200 --min-free-gib 10 --telemetry-interval 2

cat "$MAI_DATA/runs/smoke-001/summary.json"
cat "$MAI_DATA/runs/smoke-001/telemetry/status.json"
```

检查真实截图、动作坐标、模型输出、verifier、HTTP 错误和 telemetry 的 missing 字段。15/2048/1 是调试参数，不是官方统一配置。

## 9. 批量与参数调整

先在 server.env 调整每个服务的引擎设置，重启对应服务才能生效。worker 数由 workers.server.json 的列表长度决定，不等于引擎 max_num_seqs。CPU/RAM 不足时先用 2 个 worker/卡；否则起步 4 个/卡，并结合 calibrate.py 和相同任务集的端到端吞吐再测试 8 个/卡。

批量示例参数为 50 步、2048 输出 token、1 trial、2 小时总上限；正式评测前按实验定义更改，不能把它们当官方设置。

```bash
source "$PROJECT/server.env"
cd "$OSWORLD"
"$EVAL/bin/python" "$PROJECT/main/run_batch.py" \
    --manifest-path "$PROJECT/configs/tasks.server.json" \
    --workers-path "$PROJECT/configs/workers.server.json" \
    --prompt-path "$PROJECT/main/system_prompt.txt" --provider docker \
    --model-revision "$MODEL_REV" --model-dir "$MODEL" \
    --output-dir "$MAI_DATA/runs/batch-001" \
    --max-steps 50 --max-tokens 2048 --trials 1 --time-limit 7200 \
    --episode-timeout 1800 --min-free-gib 10 --telemetry-interval 2
```

若已有经核验的 DCGM exporter，追加 `--dcgm-url http://127.0.0.1:9400/metrics`。没有 exporter 时不要填假地址。NVML、主机指标、vLLM /metrics 默认采集；更深层计数器的 Nsight 命令见 RUNNING.md。

批量结束后两个模型服务仍在运行。停止本项目对应 tmux 会话中的服务：

```bash
tmux send-keys -t mai-gpu0 C-c
tmux send-keys -t mai-gpu1 C-c
nvidia-smi
```

这只结束模型进程，不关闭 CompShare 实例、不停止云计费。云资源生命周期仍需另外核对。

## 10. CPU 环境作图、校验与回传

不要为画图额外占用 GPU；切换无卡或在已有 CPU 主机上执行。路径以实际保留/复制的位置为准。

```bash
source "$PROJECT/server.env"
"$EVAL/bin/python" "$PROJECT/main/analyze.py" "$MAI_DATA/runs/batch-001" --plots --inventory
"$EVAL/bin/python" "$PROJECT/main/verify_artifacts.py" "$MAI_DATA/runs/batch-001"
```

服务器仍可连接时，本地执行回传，确认目标磁盘空间足够：

```bash
mkdir -p /home/seagull/mai-results
compshare --json instance cp cpod-1x2b7fhaphm7 \
    :/data/mai-eval/runs/batch-001 /home/seagull/mai-results/
python3 /home/seagull/Reproduction/MAI-Gym/main/verify_artifacts.py \
    /home/seagull/mai-results/batch-001
```

校验通过后才认为形成完整实例外副本；不自动删除源文件。系统盘回收时间以平台最新状态为准。

参考（2026-11-06 核对）：[OSWorld 安装](https://github.com/xlang-ai/OSWorld)、[Docker provider](https://github.com/xlang-ai/OSWorld/blob/8903ebeb0efdf74331bd7dc37ed91ccbbff22317/desktop_env/providers/docker/provider.py)、[依赖](https://github.com/xlang-ai/OSWorld/blob/8903ebeb0efdf74331bd7dc37ed91ccbbff22317/requirements.txt)、[guest 镜像状态](https://github.com/xlang-ai/OSWorld/blob/8903ebeb0efdf74331bd7dc37ed91ccbbff22317/docs/guest-image-status.md)、[vLLM 安装](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。
