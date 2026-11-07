# MAI-Gym 模型评测

使用 [Tongyi-MAI/MAI-Gym-9B](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B) 后训练权重进行基准测试，保存中途轨迹与最终结果。本项目不进行训练；当前 PyAutoGUI 协议与提示词为本地适配，不宣称复现官方分数。

## 从哪里开始

| 需要做什么 | 文档 |
| --- | --- |
| 在服务器安装、配置、启动、评测和回传数据 | [服务器命令清单](#服务器命令清单) |
| 理解调度、上下文窗口、原始数据、遥测和统计口径；进行校准与诊断 | [实现与数据说明](#实现与数据说明) |
| 核对权重、缓存、截图和并发的显存预算 | [显存评估](#显存评估) |

首次部署从下方服务器命令清单开始。本 README 集中维护项目说明、完整操作命令、实现约定和显存计算。

## 已验证到哪一步

最近保存的[本地验证记录](evidence/local-validation-2026-11-06.json)为 2026-11-06：18 项 CPU/mock 测试通过，详见[测试日志](evidence/local-tests-2026-11-06.txt)。现有记录不包含真实 GPU 吞吐或 OSWorld 评测结果。

双卡配置为未实测起点。上机先核实 GPU、CPU/RAM、Docker/KVM 和存储条件，再完成单任务冒烟与并发校准。

## 产出与保存

遵循[全仓库产出与存储准则](../docs/EXPERIMENT_ARTIFACTS.md)。本项目不产生训练检查点；权重固定 revision 后作为共享缓存。代码、配置、摘要和证据索引进入 Git，大体积截图、轨迹和权重另存。采集字段与缺失值处理见实现与数据说明，回传与校验命令见服务器命令清单。

## 服务器命令清单

适用：Ubuntu/Debian、root shell、两个标称 48GB GPU；Docker provider。这里只提供命令，未在实例执行。CompShare 容器必须确实拥有 KVM 和 Docker daemon 权限；仅安装 Python 包或 Docker CLI 不能赋予宿主机权限。下载和依赖准备尽量在无卡环境完成，GPU 检查与模型启动在用户自行启动有卡模式后执行。

### 1. 同步当前代码（本地执行）

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

### 2. 设置目录、查看真实硬件（服务器执行）

`MAI_DATA` 必须改为有足够空间的实际挂载目录。2026-11-05 的[实例快照](evidence/instance-spec-2026-11-05.json)记录系统盘为 50 GiB，不能假定它足够放两个 Python 环境、权重、qcow2、Docker 写层和原始轨迹。此处不购买或扩容存储。

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

### 3. 系统依赖与 Docker/KVM

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

### 4. 两个独立 Python 环境

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

### 5. 下载权重和桌面镜像（不启动环境）

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

### 6. 生成实际配置与任务清单

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

### 7. 启动双副本服务（明确消耗 GPU）

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

### 8. 计划检查与单任务冒烟

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

### 9. 批量与参数调整

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

若已有经核验的 DCGM exporter，追加 `--dcgm-url http://127.0.0.1:9400/metrics`。没有 exporter 时不要填假地址。NVML、主机指标、vLLM /metrics 默认采集；更深层计数器的 Nsight 命令见[实现与数据说明](#实现与数据说明)。

批量结束后两个模型服务仍在运行。停止本项目对应 tmux 会话中的服务：

```bash
tmux send-keys -t mai-gpu0 C-c
tmux send-keys -t mai-gpu1 C-c
nvidia-smi
```

这只结束模型进程，不关闭 CompShare 实例、不停止云计费。云资源生命周期仍需另外核对。

### 10. CPU 环境作图、校验与回传

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

## 实现与数据说明

以下说明评测行为、产物与诊断工具。补充工具命令从 MAI-Gym 项目目录执行，Python 使用对应的评测或推理环境。

### 调度与配置

采用两卡双副本、每个副本 TP=1、任务内固定 endpoint、worker 之间共享动态任务队列。每个 worker 有独立桌面，完成一个任务就领取下一个，避免静态分配导致的尾部空等。一个任务内部仍顺序观察/推理/执行，不能为吞吐破坏状态依赖。

[初始配置](configs/dual-48gb.json) 是未实测起点：4 个 worker/副本，8 个最大引擎并发，32K 总上下文，8192 batched tokens，90% 显存预算。模型权重约 17.527 GiB；余量用于多图预填充、KV/线性状态和运行时。并发最大值不等于最快设置，也不能仅凭 GPU busy% 证明有效吞吐。上下文、生成上限和图像策略必须在各组间一致。

`serve.sh` 保存实际启动命令、packages 和日志；服务器命令清单通过 `SERVICE_LOG_DIR` 指定位置，默认位置为 `runs/services/`。引擎配置修改后需要重启服务。前缀缓存和 chunked prefill 须在实际引擎版本上验证，调整时保留实际命令。

图片上限为 10；全量文字仍会增长，超出上下文时保留 HTTP 错误而不静默裁剪。显存预算推导见[显存评估](#显存评估)，不能仅用 GPU busy% 判断有效吞吐。

### 任务与停止行为

任务清单为 JSON 路径数组，相对路径按清单目录解析。重复 trial 通过 `--trials` 明确展开，不允许重复任务路径暗中增加权重。复制 `configs/workers.example.json` 并填写每个独立桌面标识；Docker 模式的 path_to_vm 是只读 qcow2 镜像，可由不同容器共享；VMware 等已有实例路径必须分别配置。所有 `REPLACE` 均会被拒绝。旧的 `main/wokers.json` 是此前手工文件，没有删除，不作为默认配置。

`--dry-run` 只校验并保存冻结计划，不请求模型、不启动 worker/VM/遥测。实际运行使用全新的输出目录。调试步数、token 和 trial 数不代表官方设置；先验证单任务截图、坐标、动作格式、verifier 和数据回传，再扩展并发。

`run_one.py` 是同一个 runner 的单任务入口，需要 `--time-limit` 和 `--model-revision`，因而也有遥测、监督和批量格式的产物。运行超时/磁盘不足停止本次进程，不调用云平台关机，也不自动停止另外运行的模型服务；任务结束不是停止计费。

每个任务成功/失败后即时落盘。失败 worker 不复用可能损坏的桌面，也不自动重试该任务；其他 worker 继续领取待执行任务。父进程设置总时长、初始化时长、每 episode 时长和磁盘阈值。SIGTERM 先尝试走 finally 关闭环境；硬 kill 或构造中断时不能保证 provider 资源已经释放，须检查 `state.json`、`close_error.json` 和实例实际状态。

### 采集内容与可重建性

- 冻结任务、trial、worker 配置、模型 revision、实际源码和包版本；动态领取记录以及未完成任务清单。
- 每轮原图、图像 SHA256/尺寸、完整观察字段、原始 HTTP body（包括错误响应）、完整模型 JSON（含 usage/reasoning 等实际返回字段）、解析动作、执行结果。
- 每步请求快照保留全部文本，用带哈希的文件引用代替重复 base64。`replay_request.py` 验证图片后重建请求，不联网、不重新执行动作。
- N=10/O=2：第 11 步只保留第 9～11 步的真实图片，第 19 步只保留 17～19；所有文字/回答保留，退出窗口的图片替换为文本占位符。
- 请求时间含队列和完整生成延迟；保存 reset、执行、verifier 和 episode 时长。当前非流式接口不声称有逐 token 时间或客户端 TTFT；TTFT/ITL 分布来自可用的 vLLM exporter 原始指标。
- controller 的命令参数、返回 stdout/stderr 等实际返回字段和二进制文件独立落盘，避免 OSWorld step 丢弃底层返回值。
- verifier 的 result/expected getter 和 metric 返回值单独记录，原始返回值不变；无法 JSON 序列化的自定义对象标注类型和 repr，不假装已完整序列化。worker.log 保留 OSWorld 日志。
- NVML 每 2 秒采集：实际显存、GPU busy/显存控制器 busy、功耗/功率上限、累计能耗、温度、SM/显存/图形频率、P-state、限频原因、PCIe RX/TX/速率/宽度/replay、ECC、风扇和计算进程显存；设备不支持的项记录 null 和具体原因。
- 同步保留 CPU、RAM、swap、磁盘、I/O、网络和 vLLM 原始 Prometheus 文本（gzip）。包括引擎队列、缓存、抢占、prefill/decode、TTFT/ITL 等服务实际公开的指标。不把进程总显存说成 PyTorch allocated/reserved。

```bash
python main/replay_request.py runs/RUN/worker_00/task_00000_trial_000/step_0011.request.json /tmp/request.json
```

### 深层 GPU 硬件计数器

GPU busy% 不是 SM occupancy 或 Tensor Core 使用率。若服务器有可用 DCGM exporter，以 `--dcgm-url http://127.0.0.1:9400/metrics` 保存全部原始指标，并为公开的 SM/Tensor/DRAM profiling 指标作图。项目不自动启动 exporter 或申请权限。4090 类设备与容器可能不支持 DCGM profiling，不做支持承诺。

另提供 `profile_service.sh`，在单独短诊断时段采集 CUDA kernel/API、NVTX 和 Nsight GPU Metrics。这个脚本显式启动本机模型服务，需使用空闲端口，不能对正式测评服务重复启动；不要与 DCGM profiling 同时占用计数器。

```bash
PROFILE_DELAY=120 PROFILE_SECONDS=30 \
  bash main/profile_service.sh 0 8000 /data/models/MAI-Gym-9B runs/profile-gpu0
```

延迟和时长应覆盖服务就绪后的代表性推理；通过另一终端发送真实请求。保留 `.nsys-rep`，可在 CPU 主机用 `nsys export --type sqlite --output profile.sqlite inference.nsys-rep` 导出 SQLite。`--kill=none` 不在采样结束时杀模型，需自行结束诊断服务。深度 profiling 有额外开销，单独记录，不能与无 profiler 的吞吐结果直接混合。权限不足保留错误，不更改驱动权限。

参考：[NVML](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)、[DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)、[vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)。

### 并发校准与结果口径

`calibrate.py` 是主动推理工具，只有手动运行才会发请求。使用不同任务早/中/晚阶段的请求快照，比较每卡 2/4/8 并发。保存原始回答、错误、延迟、token/s 和遥测；只对无错误且 token 计量完整的组给出实测最佳值，不改动正式评测参数。

```bash
python main/calibrate.py --requests /data/early.request.json /data/late.request.json \
  --endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
  --concurrency 2 4 8 --repeats 3 --output-dir runs/calibration-001
```

重复输入会提高缓存命中；这只是推理端上限参考。最终依据相同任务集合的端到端 episodes/hour、错误率、CPU/RAM 和 GPU 指标选 worker 数；不能为提高 token/s 降低评测完整性。两个服务各自限流，并保留任务内端点固定。

离线作图、文件校验与回传命令见[服务器命令清单](#服务器命令清单)，在 CPU 环境执行。

生成原始结果汇总、未完成清单、CSV、PNG/SVG/PDF、能耗来源说明、绘图输入哈希和逐文件 SHA256 清单。完成样本均分与全量均分分开：有缺失时全量均分为 null，异常不自动算零分；score=1 的比例单独命名，不把任意分数阈值当官方成功率。能耗是设备总量，不能归因给单个任务；没有累计计数器时梯形积分明确标为估计。

将完整 run 复制到已有持久磁盘后，在目标目录再次运行 verify_artifacts.py。通过才登记已校验副本；当前不会自动上传、购买存储或删除源数据。图表可重建，原始截图和响应应持续保留。首次真实运行需用户确定停卡安排和回传位置；本实现尚未验证云平台计费、无卡切换或真实环境关闭。

### 本地验证

```bash
python -m unittest discover -s tests -v
```

测试使用本地假 HTTP 服务和 fake OSWorld，包括 27 步窗口与请求重建、错误原文、verifier、spawn 动态调度、trial 唯一性、超时/初始化失败、缺失分母、无 GPU 遥测和图表/校验清单。CPU/mock 通过不代表 GPU 已测，也不产生真实评测结果。

## 显存评估

评估日期：2026 年 11 月 1 日。目标硬件：用户提供的 40 系 GPU，两张、每张标称 48GB；具体型号、PCIe 拓扑与驱动尚未实测。

**单卡 48GiB 在受控截图规模和并发下具有运行 BF16 模型的容量基础；双卡优先各部署一个副本，按任务分流。** 权重精确占用为 **17.527 GiB**。按本文假设的 7GiB 运行开销、每请求 32K 总 token 计算，单卡 4 并发约 **28.719GiB**，8 并发约 **32.911GiB**；这些是容量预算，不是实测峰值或速度承诺。

本节保留 2026-11-01 的容量评估依据，核验值、公式结果与工程预算分别标注。当前验证状态见[已验证到哪一步](#已验证到哪一步)，实际部署命令见[服务器命令清单](#服务器命令清单)。本节的 85% 显存预算用于容量敏感性分析，不是当前启动脚本的默认值。

### 1 依据与复现边界

主要依据如下，访问日期均为 2026 年 11 月 1 日。

| 依据 | 用途 |
| --- | --- |
| [本地复现要求](#mai-gym-模型评测) | 确定只做评测 |
| [模型页面](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B) | 模型身份、BF16 类型 |
| [模型配置](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/blob/176656ae6a034a0ebf57e8fa67e73015b6e2d98a/config.json) | 层数、头数、维度、视觉配置 |
| [权重索引](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/blob/176656ae6a034a0ebf57e8fa67e73015b6e2d98a/model.safetensors.index.json) | 精确权重字节数 |
| [处理器配置](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/blob/176656ae6a034a0ebf57e8fa67e73015b6e2d98a/processor_config.json) | 截图 patch 与 merge 参数 |
| [论文附录 A.4](https://arxiv.org/html/2610.20939v1#A1.SS4) | 原生截图、历史窗口、评测采样条件 |
| [项目仓库](https://github.com/Tongyi-MAI/MAI-Gym) | 评测 harness 发布状态 |
| [vLLM Qwen3.5 实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)及[状态形状实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/mamba_utils/) | 混合注意力状态形状和 TP 分片 |
| [vLLM 显存控制](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)及[缓存配置](https://docs.vllm.ai/en/latest/api/vllm/config/cache/) | CUDA Graph、缓存池及 dtype 控制 |

核验时模型 revision 为 `176656ae6a034a0ebf57e8fa67e73015b6e2d98a`。配置与索引保存在 [evidence](evidence/)；[weight_summary.json](evidence/weight_summary.json) 来自该 revision 四个 safetensors 分片的头部，包含 tensor 形状、dtype、分片内偏移和分组汇总。没有下载约 18.82GB 的完整权重。配置快照由当时 main 获取，正式运行仍应固定上述 revision，避免后续 main 变化。

评估时上游 README 写有框架代码待发布；当时核查上游提交 `dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9`，其中包含 agents、prompt、parser 和 HTTP backend，但不是完整评测基础设施。因此正式复现前需核对实际可用 harness、任务版本及 verifier。本文不编造 `mai-gym evaluate` 命令。

论文描述历史窗口 K=10、重叠 O=2；当时上游配置示例中的截图保留数量仍为待填项。本地现已按用户要求实现 N=10/O=2，窗口外图片替换为占位符、文字保留，具体行为见[实现与数据说明](#实现与数据说明)。评测温度为 0，每任务 3 次 trial，使用对应 benchmark 的步数限制。本文 32K/64K 上下文和并发值是部署预算场景，不是官方统一评测配置。[论文](https://arxiv.org/html/2610.20939v1)

公开 agent 实现还揭示了显存评估的重要边界：移动端窗口策略把窗口外图片替换成占位符，仍保留文字消息；电脑端 `max_history_turns` 默认是 0，即不主动截断文字历史，`window_size` 默认也未启用。**图片窗口有界不代表文字上下文有界**，长任务的思考、动作与命令输出仍会累积。正式运行必须显式记录实例化参数，不能仅使用构造器默认值便认为对齐论文。[移动端窗口实现](https://github.com/Tongyi-MAI/MAI-Gym/blob/dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9/src/mai_gym/agents/implementations/mai_gym_mobile_use_windows_wise.py)、[电脑端实现](https://github.com/Tongyi-MAI/MAI-Gym/blob/dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9/src/mai_gym/agents/implementations/mai_gym_computer_use.py)

### 2 硬件容量与计量单位

所有显存表采用 GiB：`1 GiB = 2^30 bytes`，`1 MiB = 2^20 bytes`。厂商 GB 与 GiB 不应混用：若设备实际可用 48GiB，则总计 96GiB；若每卡只有十进制 48GB，则每卡约 44.703GiB。应使用实际驱动报告替换输入。

两张卡并不自动形成一个可任意分配的 96GB 地址空间。双副本需要每卡完整装下模型；张量并行需要引擎主动切分权重和缓存，部分模块仍可能复制。

这里不根据“40 系 48GB”猜测具体 SKU，也不假设 NVLink、GPU P2P 或 PCIe 带宽已验证。首次运行按[服务器命令清单](#服务器命令清单)记录实际显存和拓扑。

双副本几乎没有模型层之间的跨卡通信，适合互联能力未知的两张卡。TP=2 则需额外确认 NCCL/P2P 可用性和实际吞吐。

### 3 总显存如何计算

推理逻辑需求采用以下分解，单位统一为字节后再换算：

```text
M_peak ≈ M_weights + M_full_attention_KV + M_linear_states
       + M_live_activations + M_workspace + M_graphs
       + M_runtime_and_allocator + M_other_GPU_processes
```

这里的峰值应取预填充、解码、窗口切换、加载和引擎初始化等阶段的最大值。各阶段不一定同时达到自身最大值，逐项预算相加是规划方式，不是精确的实际峰值模拟。

纯推理使用 inference mode，不保留反向传播图，因此训练用的梯度、Adam 一阶/二阶状态和 FP32 主权重 **均不计入本次预算**。也不额外加载训练参考模型、奖励模型或 MTP 草稿模型。若 benchmark 需要本地 VLM judge，其显存必须另行加算；优先在推理空闲时运行，或用独立服务。

还要区分两种读数：

- **逻辑有效需求**：当前权重、有效缓存和工作张量的大小，对应本文公式。
- **引擎保留显存**：vLLM/SGLang 常预分配缓存池。即使只有一条请求，`nvidia-smi` 也可能显示接近显存预算上限；不能再在该读数上重复加一次 KV cache。

PyTorch `memory_allocated` 与 `memory_reserved` 也不能相加，前者通常已包含在后者内；进程外 CUDA 分配还可能不在 PyTorch 统计中。[vLLM 缓存配置](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)

### 4 权重占用的精确计算

权重索引给出的 `metadata.total_size` 为 **18,819,627,488 bytes**。四个分片头部合计 760 个 tensor，全部 BF16，合计 **9,409,813,744 个参数**。

```text
M_weights = 参数数量 × 每参数字节数
          = 9,409,813,744 × 2
          = 18,819,627,488 bytes
          = 18.819627488 GB
          = 17.527143925 GiB
```

| 分组 | 参数数量 | BF16 字节数 | GiB |
| --- | ---: | ---: | ---: |
| 输入 embedding | 1,017,118,720 | 2,034,237,440 | 1.894531 |
| 语言解码器及 norm | 6,919,565,824 | 13,839,131,648 | 12.888696 |
| 视觉编码器及连接模块 | 456,010,480 | 912,020,960 | 0.849386 |
| 输出 lm_head | 1,017,118,720 | 2,034,237,440 | 1.894531 |
| 合计 | 9,409,813,744 | 18,819,627,488 | 17.527144 |

输入 embedding 和输出 head 不共享权重。可交叉核验：`248320 × 4096 = 1,017,118,720`，两者分别计数。视觉权重已经包含在总数内，不再额外加一次。“9B × 2 bytes”的粗算仅约 16.76GiB，会少算约 0.76GiB。[配置与权重](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/tree/176656ae6a034a0ebf57e8fa67e73015b6e2d98a)

上述数值是原始 tensor 存储量。加载转置、对齐、打包或临时副本仍需运行预算。模型应直接以 BF16 分片加载；不要先把整模 FP32 放进 GPU 再转换，FP32 权重本身就约 35.05GiB，转换时还有额外峰值。

### 5 混合注意力缓存的计算

#### 5.1 不能将 32 层全部按普通 KV cache 计算

模型有 32 层语言解码器，配置每 4 层才有 1 层 full attention：**8 层全注意力，24 层线性注意力**。全注意力为 16 个 query head、4 个 KV head，每头维度 256。这里 head_dim 必须读配置，不能用 `4096 / 16` 之外的习惯值替代；KV head 也不能误用 query head 数量。[config.json](evidence/config.json)

设 C 为一条请求的总缓存 token 数，包括系统提示、历史文字、图片展开后的 token 和已生成输出；B 为同时保留缓存的请求数量，BF16 缓存每元素 2 字节。

```text
M_KV = B × C × L_full × 2(K 与 V) × H_KV × D_head × bytes
     = B × C × 8 × 2 × 4 × 256 × 2
     = B × C × 32768 bytes
     = B × C × 32 KiB
```

所以每条请求 32768 token 对应 **1GiB KV cache**；65536 对应 2GiB；262144 对应 8GiB。如果错误地用 32 层全注意力计算，会放大 4 倍。

对不同长度的请求，应使用 `32768 × sum(C_i)`。分页缓存用块大小 q 向上取整：`C_i → ceil(C_i/q) × q`；实际引擎还可能有混合层 page 对齐开销。基础估算不依赖前缀共享，从而避免把未命中的共享收益当成固定容量。

#### 5.2 线性注意力状态

24 层 Gated DeltaNet 不保存每一个历史 token 的完整 KV。配置为 16 个 key head、32 个 value head，key/value 维度均为 128，卷积核大小 4。

按 **FP32 recurrent state** 和 **BF16 convolution state** 预算：

```text
单层 recurrent state = 32 × 128 × 128 × 4 = 2,097,152 bytes = 2 MiB
卷积通道 = 2 × 16 × 128 + 32 × 128 = 8192
单层 conv state = 8192 × (4 - 1) × 2 = 49,152 bytes
24 层总状态 = 24 × (2,097,152 + 49,152)
             = 51,511,296 bytes
             = 49.125 MiB = 0.047973633 GiB / 请求
```

这是 vLLM 所用 `kernel_size - 1` 状态形状、没有 speculative token 的计算。其他实现若存 4 个 conv 位置，总量为 49.5MiB。采用 FP32 recurrent state 是与 checkpoint 配置 `mamba_ssm_dtype` 对应的保守预算；引擎的实际 dtype 由运行配置决定，不能只看 checkpoint 后便假定已生效。[状态形状实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/mamba_utils/)

这里仅计算每请求**一个当前状态**。若引擎为了前缀复用保存多个历史状态检查点、预分配状态槽，或者做 speculative decoding，状态池会更大：应将公式中的 B 替换为实际状态槽数，或引入每请求状态副本数 R。脚本提供 `--state-copies R`。例如 B=8、R=10 时状态共约 3.838GiB，比 R=1 多约 3.454GiB。前缀缓存是否省显存取决于复用收益与状态保存策略，需实测。

#### 5.3 每请求缓存合计

以下设 R=1，不含权重和临时激活。

| 总 token C | 全注意力 KV GiB | 线性状态 GiB | 合计 GiB |
| ---: | ---: | ---: | ---: |
| 8192 | 0.250 | 0.048 | 0.298 |
| 16384 | 0.500 | 0.048 | 0.548 |
| 32768 | 1.000 | 0.048 | 1.048 |
| 65536 | 2.000 | 0.048 | 2.048 |
| 131072 | 4.000 | 0.048 | 4.048 |
| 262144 | 8.000 | 0.048 | 8.048 |

`max_position_embeddings=262144` 是配置上限，不代表应默认开到 256K，也不表示 256K 多图请求一定能通过预填充。

### 6 截图与激活值如何估算

#### 6.1 图片 token 数

视觉 patch_size=16、spatial_merge_size=2。普通单张静态截图经过实际 processor resize 后，设尺寸为 H′×W′、均按处理器规则对齐：

```text
视觉编码器 patch 数 N_patch = (H′/16) × (W′/16)
进入语言模型的图片 token N_img = N_patch / 4 = H′W′ / 1024
```

静态图片按 temporal_patch_size=2 的方式处理，不应因此再把语言图片 token 乘 2。最可靠的计数是 processor 返回的 `image_grid_thw`：`N_img = product(grid_thw) / merge_size^2`。再由最终 tokenized input 长度确认，包含图片边界等少量特殊 token。[处理器配置](evidence/processor_config.json)

| 假设处理后尺寸 H′×W′ | N_patch | 每图语言 token | 每图 KV 增量 |
| --- | ---: | ---: | ---: |
| 768×1024 | 3072 | 768 | 24MiB |
| 1088×1920 | 8160 | 2040 | 63.75MiB |
| 1440×2560 | 14400 | 3600 | 112.5MiB |
| 2160×3840 | 32400 | 8100 | 253.125MiB |

1088×1920 只是一个已对齐的预算例子，并不宣称任意 1080p 图像都会被 processor 转成该尺寸。

例如，假设保留 10 张 1088×1920 图片、6000 个文字/模板 token、2048 个输出 token，则 C≈28448，小于 32768。相同设置换成 10 张 4K 图片，则 C≈89048，32K 和 64K 都不够。图片保留数、文字历史长度与实际处理分辨率必须一起记录，不能仅根据“历史 10 步”设上下文。

发布的 processor 配置中 `size.longest_edge=16777216`、`shortest_edge=65536` 是该处理器用于面积控制的字段，不应读成图像边长。允许面积上限对应约 16384 个合并后图片 token，默认上限并不低。正式复现不应为了省显存擅自降低原生分辨率或删除历史；这种改动应单独标为资源受限实验。

#### 6.2 为什么图片预填充可能成为峰值

视觉 hidden_size=1152，MLP intermediate_size=4304。以单图 N_patch=8160、BF16 为例：

```text
一个视觉 hidden 张量 = 8160 × 1152 × 2 ≈ 17.93 MiB
一个视觉 MLP intermediate = 8160 × 4304 × 2 ≈ 66.99 MiB
一组 QKV 张量 ≈ 3 × 17.93 = 53.79 MiB
```

如果将完整的视觉注意力分数显式物化，16 heads 的单层矩阵为 `16 × 8160² × 2 ≈ 1.984GiB`。这只是低效全矩阵实现的风险示例；实际分窗、融合 attention 或 memory-efficient kernel 会改变峰值，不能把该矩阵直接加到所有后端的预算里。多张图片同时编码时，临时张量还会放大。

#### 6.3 语言预填充与 logits

设一次调度实际预填充 token 数为 T，而非所有请求的完整 C：

```text
语言 hidden = T × 4096 × 2
一个 MLP 中间张量 = T × 12288 × 2
```

T=4096 时分别为 32MiB、96MiB；门控 MLP 可能同时保留多份中间结果。推理时层间可复用缓冲区，不能简单乘 32 层当作激活常驻量。chunked prefill 可控制语言侧 T，但不保证视觉编码也被同样切块。

词表有 248320 项。如果错误地保留全部 32768 个位置的 BF16 logits，单请求就需要 `32768 × 248320 × 2 / 2^30 = 15.156GiB`，FP32 则为 30.312GiB。正常生成应只保留需要采样的位置，避免开启全量 hidden_states、attentions 或所有位置 logits 日志。生成过程文本、截图和动作记录应存 CPU/磁盘。

#### 6.4 工程开销预算

无法只靠 config 精确求出 CUDA workspace、融合算子生命周期和分配器碎片。以下是**待实测校准的规划假设**，适用于受控图片批次、分块预填充、普通解码，不是官方或实测值：

| 项目 | 基准预算 GiB | 说明 |
| --- | ---: | --- |
| CUDA context、基础引擎和运行时缓冲 | 1.5 | 与后端、进程数有关 |
| 语言和视觉激活及算子 workspace | 3.5 | 高分辨率多图可能超过 |
| Graph、分配器余量、碎片和未展开小张量 | 2.0 | eager 模式也保留该规划余量 |
| 预算合计 O | 7.0 | 容量表使用；另做 O=10 的敏感性分析 |

使用单卡 48GiB、显存利用预算 85%：引擎目标预算 U=40.8GiB，卡外余量 7.2GiB 用于系统占用与峰值浮动。O 是目标预算内部的运行开销，与这 7.2GiB 卡外余量不同。若其他 GPU 进程已占用较多，应重新降低 U。

### 7 单卡与双副本容量表

采用 BF16 权重及 KV、FP32 线性状态、R=1、O=7GiB，无前缀共享、无量化、无 CPU offload、无额外 judge：

```text
M_single(B,C) = 17.527143925 + 7
              + B × (C / 32768 + 0.047973633) GiB
```

| 单请求总 token | B=1 GiB | B=4 GiB | B=8 GiB | B=16 GiB | 40.8GiB 下代数并发上限 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 24.825 | 25.719 | 26.911 | 29.295 | 54 |
| 16384 | 25.075 | 26.719 | 28.911 | 33.295 | 29 |
| 32768 | 25.575 | 28.719 | 32.911 | 41.295 | 15 |
| 65536 | 26.575 | 32.719 | 40.911 | 57.295 | 7 |
| 131072 | 28.575 | 40.719 | 56.911 | 89.295 | 4 |
| 262144 | 32.575 | 56.719 | 88.911 | 153.295 | 2 |

代数上限为 `floor((U - W - O)/(C/32768 + state))`。它忽略实际 page padding、额外状态槽、编码器缓存和批次变化，**不能直接当作生产并发设置**。尤其不能把 8K 场景的 54 直接设置为高分辨率多图并发。

双副本时两卡各承担 B 请求，系统共 2B 请求，显存要逐卡判断：32K、每卡 B=4 为每卡 28.719GiB、系统 8 并发；每卡 B=8 为每卡 32.911GiB、系统 16 并发。汇总用量约 57.438GiB 和 65.822GiB，不表示某卡可向另一卡借空闲容量。

敏感性示例：

- O 从 7 增至 10GiB，32K/B=8 从 32.911 增至 35.911GiB；64K/B=8 从 40.911 增至 43.911GiB。
- 同为 32K/B=8，若每请求状态副本 R 从 1 增至 10，增加约 3.454GiB。
- 若单卡只有 44.703GiB，85% 预算是约 37.998GiB；32K 的代数并发上限从 15 降至 12。

因此推荐先每卡 B=1 验证，然后 B=2、4；32K/B=8 是后续压测候选。64K 从每卡 1–2 开始，再验证 4，并不默认启用每卡 8。

### 8 并行方案选择

| 方案 | 权重分布 | 主要收益 | 对本次任务的判断 |
| --- | --- | --- | --- |
| 两卡双副本，每个 TP=1 | 每卡 17.527GiB | 不同任务独立推理，基本无跨卡层通信 | 首选 |
| 单副本 TP=2 | 每卡约 8.764–9.188GiB，外加复制/对齐开销 | 单请求缓存和语言计算可分摊 | 大上下文或单卡峰值过高时考虑 |
| PP=2 或逐层 device_map | 按层分摊但可能不均衡 | 模型跨卡放置 | 9B 单卡已能放下，通常没有必要 |
| 单卡模型加另一卡 judge/环境渲染 | 模型卡占完整权重 | 隔离额外负载 | 存在重型本地 judge 或 GPU 渲染时考虑 |

这里“双副本”是推理请求级数据并行，不需要训练用的 DistributedDataParallel、梯度同步或 ZeRO。

TP=2 理想权重为 `17.527/2=8.764GiB`。若视觉模块完整复制而语言权重切半，则每卡为 `(17.527-0.849386)/2+0.849386=9.188265GiB`。KV 的 4 个头可分为每卡 2 个，线性状态 head 也可分；在实现支持且采用相应策略时，每请求每卡缓存约为单卡的一半。[vLLM Qwen3.5 实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

用视觉复制假设、O=7GiB，仅作 TP 容量示例：

```text
M_TP2_per_GPU ≈ 9.188265 + 7 + B × (C/65536 + 0.023986816)
```

| TP=2 整个服务并发 B | C | 每卡估算 GiB |
| ---: | ---: | ---: |
| 8 | 32768 | 20.380 |
| 16 | 32768 | 24.572 |
| 8 | 65536 | 24.380 |
| 16 | 65536 | 32.572 |

TP=2 的 O 仍是假设，**没有按 2 平分运行开销**，也没有据此声称 TP 吞吐更高。NCCL 缓冲、all-reduce、视觉 TP 模式及不均衡必须在实机测量。`device_map=auto` 仅代表放置策略，不能视为高吞吐 TP 实现。

任务分发建议按 `(benchmark, task_id, trial_id)` 分区，轨迹在整个执行期间固定同一推理副本，以便缓存复用。一个任务的后续动作依赖环境前一步反馈，不能将同一轨迹的不同步骤并行执行；并行单位是独立任务或拥有独立环境的 trial。三个 trial 不必同时在 GPU 上驻留。

### 9 实测如何校准预算

安装与启动命令统一见[服务器命令清单](#服务器命令清单)，不再维护另一套部署参数。实测应覆盖真实截图与任务历史：

1. 分别记录权重加载、视觉 profile、缓存分配后的显存，以及实际缓存 dtype、token blocks 和状态槽数。
2. 覆盖单图、窗口上限图片数、最大的原生截图与最长文字历史，记录 processor 后 token 数。输入加输出预算必须满足上下文上限，不能静默截断。
3. 按需要的上下文长度逐步增加并发 B=1、2、4、8；加入第二副本后测试缓存未命中、混合长短请求和长轨迹。比较 eager 与 CUDA Graph 时只改变这一项。
4. 用实测峰值校准运行开销 O 和状态副本 R。环境渲染或本地 judge 若共享 GPU，应计入其他进程占用；增加桌面环境还会消耗 CPU、RAM 和磁盘。

轮询可能漏掉瞬时峰值，应结合引擎 profile。若直接使用 Transformers，在模型进程中重置并读取 PyTorch 峰值统计；客户端调用 CUDA 内存 API 无法测到服务端显存。具体 NVML、DCGM、Nsight 与并发校准工具见[实现与数据说明](#实现与数据说明)。

正式对照需固定 prompt、chat template、thinking 设置、动作解析、图像处理、历史策略、温度、步数和 verifier。改变分辨率、量化或历史窗口应单独标注，不能作为默认容量补救。

### 10 可重复计算与验收标准

附带[离线计算脚本](estimate_vram.py)，仅需 Python 标准库，不会下载模型或占用 GPU：

```bash
python3 MAI-Gym/estimate_vram.py
python3 MAI-Gym/estimate_vram.py --overhead-gib 10
python3 MAI-Gym/estimate_vram.py --gpu-gib 44.70348358
python3 MAI-Gym/estimate_vram.py --tp 2
python3 MAI-Gym/estimate_vram.py --state-copies 10
```

脚本从保存的配置提取层数和维度，校验权重汇总与索引一致，重算单请求缓存、B=1/4/8/16 用量与代数容量。TP 场景采用视觉复制、语言分片的明确假设；运行开销由参数输入，不将其伪装成测量值。

本阶段可以确定：权重原始字节数、BF16 KV 每 token 字节数、单份线性状态形状，以及给定上下文/并发/开销假设下的容量。**本阶段不能确定：真实 GPU 峰值、极限并发、tokens/s、总评测时长或最终 benchmark 分数。** 这些必须由目标机器和实际 harness 验证。

执行建议是 BF16 双副本开始，每卡先验证 1–4 并发；用真实历史统计选择 32K 或更长上下文，完成峰值压测再提高并发。若单卡被长上下文或图片预填充峰值限制，再测试 TP=2。无需为了装下权重提前引入量化、训练分片或 CPU offload。
