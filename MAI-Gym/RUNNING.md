# MAI-Gym 本地实现与服务器运行说明

本目录已从编程指导阶段转为可测试的评测客户端。当前只完成本地 CPU/mock 验证，未启动 CompShare 实例、模型服务或真实 OSWorld。模型动作协议和 system_prompt.txt 是自定义 PyAutoGUI 适配，不宣称复现官方分数。

## 已查规格与性能起点

[只读 API 快照](evidence/instance-spec-2026-11-05.json)：实例处于 Stopped，平台型号标签为 `4090_48G`，当前 API 配置为 0 GPU / 2 CPU / 4096 MiB RAM / 50 GiB 系统盘；用户确认有卡模式为两张、每张标称 48GB。GPU 模式的 CPU、RAM、驱动实际显存、PCIe 拓扑、KVM 和 Docker 权限尚未核实。容器实例不等于具备嵌套虚拟化；若不能在该实例内运行 OSWorld，可以将桌面环境放在另一个支持的主机上，由评测 worker 请求模型端点。

采用两卡双副本、每个副本 TP=1、任务内固定 endpoint、worker 之间共享动态任务队列。每个 worker 有独立桌面，完成一个任务就领取下一个，避免静态分配导致的尾部空等。一个任务内部仍顺序观察/推理/执行，不能为吞吐破坏状态依赖。

[初始配置](configs/dual-48gb.json) 是未实测起点：4 个 worker/副本，8 个最大引擎并发，32K 总上下文，8192 batched tokens，90% 显存预算。模型权重约 17.527 GiB；余量用于多图预填充、KV/线性状态和运行时。并发最大值不等于最快设置，也不能仅凭 GPU busy% 证明有效吞吐。上下文、生成上限和图像策略必须在各组间一致。

## 服务器准备（本文命令未执行）

在与固定版本 OSWorld 兼容的 Python 环境中安装 `requirements-eval.txt`；vLLM 与 OSWorld 分别按其版本要求部署，不让此依赖文件覆盖 torch。运行时会保存实际 packages、源码副本、OSWorld commit（若可读）和显式模型 revision。客户端和模型服务可以使用不同 venv。

```bash
python -m pip install -r requirements-eval.txt
python main/preflight.py --output preflight.json --provider docker
```

检查 GPU 模式的 CPU/RAM、cgroup 限额和 /dev/kvm 权限。确认存储足够同时容纳模型、独立桌面磁盘及原始截图；50 GiB 不是已验证充足容量。固定模型 revision 为实际部署的 revision，不能只记录 `mai-gym` 别名。当前参考 revision 为 `176656ae6a034a0ebf57e8fa67e73015b6e2d98a`。

分别在两个终端启动模型服务：

```bash
bash main/serve.sh 0 8000 /data/models/MAI-Gym-9B
bash main/serve.sh 1 8001 /data/models/MAI-Gym-9B
```

`serve.sh` 保存实际启动命令、packages 和日志到 `runs/services/`。使用 `MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`MAX_BATCHED_TOKENS`、`GPU_MEMORY_UTILIZATION` 环境变量调整；不动态更改正式 run 的实验设置。`--limit-mm-per-prompt '{"image":10}'` 对应截图上限。全量文字仍会增长，超出上下文时保留 HTTP 错误而不静默裁剪。前缀缓存和 chunked prefill 在实际 vLLM/混合注意力版本上须启动验证；不支持时调整后记录实际命令。

本机 HTTP 端点应绕过代理：例如在服务器 shell 中设置 `NO_PROXY=localhost,127.0.0.1`。不要将服务端口直接暴露到公网。

## 计划、冒烟和批量

任务清单为 JSON 路径数组，相对路径按清单目录解析。重复 trial 通过 `--trials` 明确展开，不允许重复任务路径暗中增加权重。复制 `configs/workers.example.json` 并填写每个独立桌面标识；Docker 模式的 path_to_vm 是只读 qcow2 镜像，可由不同容器共享；VMware 等已有实例路径必须分别配置。所有 `REPLACE` 均会被拒绝。旧的 `main/wokers.json` 是此前手工文件，没有删除，不作为默认配置。

```bash
python main/run_batch.py \
  --manifest-path tasks.json --workers-path workers.json \
  --prompt-path main/system_prompt.txt --provider docker \
  --model-revision 176656ae6a034a0ebf57e8fa67e73015b6e2d98a \
  --model-dir /data/models/MAI-Gym-9B \
  --output-dir runs/plan-001 --max-steps 15 --max-tokens 2048 \
  --trials 1 --time-limit 1800 --dry-run
```

`--dry-run` 只校验并保存冻结计划，不请求模型、不启动 worker/VM/遥测。实际运行使用一个全新的输出目录并去掉 `--dry-run`。上述 15 步/2048 token/1 trial 只是小规模调试例子，不是官方统一参数。先验证单任务原生分辨率、坐标、动作格式、verifier、数据回传，再扩展到多个 worker。provider 的 `path_to_vm` 含义按实际部署确认，独立字符串不自动证明底层环境隔离；Docker 由 provider 每次创建独立容器，镜像路径相同不表示容器相同。

`run_one.py` 是同一个 runner 的单任务入口，需要 `--time-limit` 和 `--model-revision`，因而也有遥测、监督和批量格式的产物。运行超时/磁盘不足停止本次进程，不调用云平台关机，也不自动停止另外运行的模型服务；任务结束不是停止计费。

每个任务成功/失败后即时落盘。失败 worker 不复用可能损坏的桌面，也不自动重试该任务；其他 worker 继续领取待执行任务。父进程设置总时长、初始化时长、每 episode 时长和磁盘阈值。SIGTERM 先尝试走 finally 关闭环境；硬 kill 或构造中断时不能保证 provider 资源已经释放，须检查 `state.json`、`close_error.json` 和实例实际状态。

## 采集内容与可重建性

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

## 深层 GPU 硬件计数器

GPU busy% 不是 SM occupancy 或 Tensor Core 使用率。若服务器有可用 DCGM exporter，以 `--dcgm-url http://127.0.0.1:9400/metrics` 保存全部原始指标，并为公开的 SM/Tensor/DRAM profiling 指标作图。项目不自动启动 exporter 或申请权限。4090 类设备与容器可能不支持 DCGM profiling，不做支持承诺。

另提供 `profile_service.sh`，在单独短诊断时段采集 CUDA kernel/API、NVTX 和 Nsight GPU Metrics。这个脚本显式启动本机模型服务，需使用空闲端口，不能对正式测评服务重复启动；不要与 DCGM profiling 同时占用计数器。

```bash
PROFILE_DELAY=120 PROFILE_SECONDS=30 \
  bash main/profile_service.sh 0 8000 /data/models/MAI-Gym-9B runs/profile-gpu0
```

延迟和时长应覆盖服务就绪后的代表性推理；通过另一终端发送真实请求。保留 `.nsys-rep`，可在 CPU 主机用 `nsys export --type sqlite --output profile.sqlite inference.nsys-rep` 导出 SQLite。`--kill=none` 不在采样结束时杀模型，需自行结束诊断服务。深度 profiling 有额外开销，单独记录，不能与无 profiler 的吞吐结果直接混合。权限不足保留错误，不更改驱动权限。

参考：[NVML](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)、[DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)、[vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)。

## 并发校准与离线图表

`calibrate.py` 是主动推理工具，只有手动运行才会发请求。使用不同任务早/中/晚阶段的请求快照，比较每卡 2/4/8 并发。保存原始回答、错误、延迟、token/s 和遥测；只对无错误且 token 计量完整的组给出实测最佳值，不改动正式评测参数。

```bash
python main/calibrate.py --requests /data/early.request.json /data/late.request.json \
  --endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
  --concurrency 2 4 8 --repeats 3 --output-dir runs/calibration-001
```

重复输入会提高缓存命中；这只是推理端上限参考。最终依据相同任务集合的端到端 episodes/hour、错误率、CPU/RAM 和 GPU 指标选 worker 数；不能为提高 token/s 降低评测完整性。两个服务各自限流，并保留任务内端点固定。

停止评测、切换到 CPU 环境之后作图与校验：

```bash
python main/analyze.py runs/RUN --plots --inventory
python main/verify_artifacts.py runs/RUN
```

生成原始结果汇总、未完成清单、CSV、PNG/SVG/PDF、能耗来源说明、绘图输入哈希和逐文件 SHA256 清单。完成样本均分与全量均分分开：有缺失时全量均分为 null，异常不自动算零分；score=1 的比例单独命名，不把任意分数阈值当官方成功率。能耗是设备总量，不能归因给单个任务；没有累计计数器时梯形积分明确标为估计。

将完整 run 复制到已有持久磁盘后，在目标目录再次运行 verify_artifacts.py。通过才登记已校验副本；当前不会自动上传、购买存储或删除源数据。图表可重建，原始截图和响应应持续保留。首次真实运行需用户确定停卡安排和回传位置；本实现尚未验证云平台计费、无卡切换或真实环境关闭。

## 本地验证

```bash
python -m unittest discover -s tests -v
```

测试使用本地假 HTTP 服务和 fake OSWorld，包括 27 步窗口与请求重建、错误原文、verifier、spawn 动态调度、trial 唯一性、超时/初始化失败、缺失分母、无 GPU 遥测和图表/校验清单。CPU/mock 通过不代表 GPU 已测，也不产生真实评测结果。
