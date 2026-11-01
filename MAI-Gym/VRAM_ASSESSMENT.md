# MAI-Gym-9B 双卡 48GB 基准测试显存评估

评估日期：2026 年 11 月 1 日。目标硬件：用户提供的 40 系 GPU，两张、每张标称 48GB；具体型号、PCIe 拓扑与驱动尚未实测。

根据[本目录 README](README.md)，本次复现是使用 `Tongyi-MAI/MAI-Gym-9B` 后训练权重进行基准测试，保留过程与结果，不进行 SFT 或在线强化学习。**单卡 48GiB 在受控截图规模和并发下具有运行 BF16 模型的容量基础；双卡优先各部署一个副本，按任务分流。** 权重精确占用为 **17.527 GiB**。按本文假设的 7GiB 运行开销、每请求 32K 总 token 计算，单卡 4 并发约 **28.719GiB**，8 并发约 **32.911GiB**；这些是容量预算，不是实测峰值或速度承诺。

当前工作环境未提供 `nvidia-smi`，尚未加载模型、执行 GPU 推理或运行基准。下文将核验值、公式结果和工程预算分别标注。先完成本阶段评估，再在目标机器做小规模实测。

## 1 依据与复现边界

主要依据如下，访问日期均为 2026 年 11 月 1 日。

| 依据 | 用途 |
| --- | --- |
| [本地复现要求](README.md) | 确定只做评测 |
| [模型页面](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B) | 模型身份、BF16 类型 |
| [模型配置](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/blob/176656ae6a034a0ebf57e8fa67e73015b6e2d98a/config.json) | 层数、头数、维度、视觉配置 |
| [权重索引](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/blob/176656ae6a034a0ebf57e8fa67e73015b6e2d98a/model.safetensors.index.json) | 精确权重字节数 |
| [处理器配置](https://huggingface.co/Tongyi-MAI/MAI-Gym-9B/blob/176656ae6a034a0ebf57e8fa67e73015b6e2d98a/processor_config.json) | 截图 patch 与 merge 参数 |
| [论文附录 A.4](https://arxiv.org/html/2610.20939v1#A1.SS4) | 原生截图、历史窗口、评测采样条件 |
| [项目仓库](https://github.com/Tongyi-MAI/MAI-Gym) | 评测 harness 发布状态 |
| [vLLM Qwen3.5 实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)及[状态形状实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/mamba_utils/) | 混合注意力状态形状和 TP 分片 |
| [vLLM 显存控制](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)及[缓存配置](https://docs.vllm.ai/en/latest/api/vllm/config/cache/) | CUDA Graph、缓存池及 dtype 控制 |

核验时模型 revision 为 `176656ae6a034a0ebf57e8fa67e73015b6e2d98a`。配置与索引保存在 [evidence](evidence/)；[weight_summary.json](evidence/weight_summary.json) 来自该 revision 四个 safetensors 分片的头部，包含 tensor 形状、dtype、分片内偏移和分组汇总。没有下载约 18.82GB 的完整权重。配置快照由当时 main 获取，正式运行仍应固定上述 revision，避免后续 main 变化。

项目 README 仍写有框架代码待发布；已核查上游提交 `dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9`，其中包含 agents、prompt、parser 和 HTTP backend，但不是完整评测基础设施。因此正式复现前需核对实际可用 harness、任务版本及 verifier。本文不编造 `mai-gym evaluate` 命令。

论文描述历史窗口 K=10、重叠 O=2；截图保留数量在配置示例中仍为待填项，**10 步不等于确认保留 10 张图片**。评测温度为 0，每任务 3 次 trial，使用对应 benchmark 的步数限制。本文 32K/64K 上下文和并发值是部署预算场景，不是官方统一评测配置。[论文](https://arxiv.org/html/2610.20939v1)

公开 agent 实现还揭示了显存评估的重要边界：移动端窗口策略把窗口外图片替换成占位符，仍保留文字消息；电脑端 `max_history_turns` 默认是 0，即不主动截断文字历史，`window_size` 默认也未启用。**图片窗口有界不代表文字上下文有界**，长任务的思考、动作与命令输出仍会累积。正式运行必须显式记录实例化参数，不能仅使用构造器默认值便认为对齐论文。[移动端窗口实现](https://github.com/Tongyi-MAI/MAI-Gym/blob/dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9/src/mai_gym/agents/implementations/mai_gym_mobile_use_windows_wise.py)、[电脑端实现](https://github.com/Tongyi-MAI/MAI-Gym/blob/dc279a3735f0d59f9b9b9806f1fc8cd99c28e6f9/src/mai_gym/agents/implementations/mai_gym_computer_use.py)

## 2 硬件容量与计量单位

所有显存表采用 GiB：`1 GiB = 2^30 bytes`，`1 MiB = 2^20 bytes`。厂商 GB 与 GiB 不应混用：若设备实际可用 48GiB，则总计 96GiB；若每卡只有十进制 48GB，则每卡约 44.703GiB。应使用实际驱动报告替换输入。

两张卡并不自动形成一个可任意分配的 96GB 地址空间。双副本需要每卡完整装下模型；张量并行需要引擎主动切分权重和缓存，部分模块仍可能复制。

这里不根据“40 系 48GB”猜测具体 SKU，也不假设 NVLink、GPU P2P 或 PCIe 带宽已验证。首次运行记录：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version,pci.bus_id --format=csv
nvidia-smi topo -m
```

双副本几乎没有模型层之间的跨卡通信，适合互联能力未知的两张卡。TP=2 则需额外确认 NCCL/P2P 可用性和实际吞吐。

## 3 总显存如何计算

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

## 4 权重占用的精确计算

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

## 5 混合注意力缓存的计算

### 5.1 不能将 32 层全部按普通 KV cache 计算

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

### 5.2 线性注意力状态

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

### 5.3 每请求缓存合计

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

## 6 截图与激活值如何估算

### 6.1 图片 token 数

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

### 6.2 为什么图片预填充可能成为峰值

视觉 hidden_size=1152，MLP intermediate_size=4304。以单图 N_patch=8160、BF16 为例：

```text
一个视觉 hidden 张量 = 8160 × 1152 × 2 ≈ 17.93 MiB
一个视觉 MLP intermediate = 8160 × 4304 × 2 ≈ 66.99 MiB
一组 QKV 张量 ≈ 3 × 17.93 = 53.79 MiB
```

如果将完整的视觉注意力分数显式物化，16 heads 的单层矩阵为 `16 × 8160² × 2 ≈ 1.984GiB`。这只是低效全矩阵实现的风险示例；实际分窗、融合 attention 或 memory-efficient kernel 会改变峰值，不能把该矩阵直接加到所有后端的预算里。多张图片同时编码时，临时张量还会放大。

### 6.3 语言预填充与 logits

设一次调度实际预填充 token 数为 T，而非所有请求的完整 C：

```text
语言 hidden = T × 4096 × 2
一个 MLP 中间张量 = T × 12288 × 2
```

T=4096 时分别为 32MiB、96MiB；门控 MLP 可能同时保留多份中间结果。推理时层间可复用缓冲区，不能简单乘 32 层当作激活常驻量。chunked prefill 可控制语言侧 T，但不保证视觉编码也被同样切块。

词表有 248320 项。如果错误地保留全部 32768 个位置的 BF16 logits，单请求就需要 `32768 × 248320 × 2 / 2^30 = 15.156GiB`，FP32 则为 30.312GiB。正常生成应只保留需要采样的位置，避免开启全量 hidden_states、attentions 或所有位置 logits 日志。生成过程文本、截图和动作记录应存 CPU/磁盘。

### 6.4 工程开销预算

无法只靠 config 精确求出 CUDA workspace、融合算子生命周期和分配器碎片。以下是**待实测校准的规划假设**，适用于受控图片批次、分块预填充、普通解码，不是官方或实测值：

| 项目 | 基准预算 GiB | 说明 |
| --- | ---: | --- |
| CUDA context、基础引擎和运行时缓冲 | 1.5 | 与后端、进程数有关 |
| 语言和视觉激活及算子 workspace | 3.5 | 高分辨率多图可能超过 |
| Graph、分配器余量、碎片和未展开小张量 | 2.0 | eager 模式也保留该规划余量 |
| 预算合计 O | 7.0 | 容量表使用；另做 O=10 的敏感性分析 |

使用单卡 48GiB、显存利用预算 85%：引擎目标预算 U=40.8GiB，卡外余量 7.2GiB 用于系统占用与峰值浮动。O 是目标预算内部的运行开销，与这 7.2GiB 卡外余量不同。若其他 GPU 进程已占用较多，应重新降低 U。

## 7 单卡与双副本容量表

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

## 8 并行方案选择

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

## 9 部署起点与评测一致性

优先沿用实际官方 harness 所兼容的后端；论文 generation 使用 SGLang。以下 vLLM 命令仅为 **Qwen3.5 架构支持下的本地服务起点**，不是 MAI-Gym 官方评测命令，也尚未在目标硬件验证。Qwen 官方模型页列出了 vLLM/SGLang 服务方式。[Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)

分别在两个终端启动，前提是已安装支持该模型和硬件的 vLLM：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Tongyi-MAI/MAI-Gym-9B \
  --revision 176656ae6a034a0ebf57e8fa67e73015b6e2d98a \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype float32 \
  --enforce-eager --port 8000
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Tongyi-MAI/MAI-Gym-9B \
  --revision 176656ae6a034a0ebf57e8fa67e73015b6e2d98a \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype float32 \
  --enforce-eager --port 8001
```

`--enforce-eager` 是初期排除 Graph 内存变量的选择；通过压测后可单独比较开启 Graph 的速度与峰值。KV 使用 BF16 模型下的默认非量化类型，并检查启动日志确认。引擎版本需固定并检查 `vllm serve --help` 中对应参数；启动后重点记录混合缓存管理、prefix cache、状态 dtype、可用 token blocks 和视觉 profile 情况。

多模态图片数量上限必须匹配 harness 实际保留图片数，显式检查 `limit_mm_per_prompt`，不要默认只有一张；但也不要在未测量前把它设成任意很大的数。单步生成预算在客户端设置，确保 `input_tokens + max_tokens <= max_model_len`。32K 不足时应增加服务上下文并重算容量，不应默默截断 benchmark 输入。

正式复现要保持 prompt、chat template、thinking 设置、tool parser、图像预处理、历史策略、温度、任务步数与 verifier 一致。基础模型支持某 reasoning parser 并不证明 MAI-Gym harness 应使用该 parser；先验证原始输出到动作的映射。量化或更改历史窗口可能改变分数，不作为默认容量补救。

## 10 上机验证与中途结果记录

建议按以下顺序进行，验证阶段使用真实截图与任务历史，而不仅是随机文本：

1. 记录两卡总/空闲显存、拓扑、CPU/内存与磁盘条件，以及驱动、CUDA、PyTorch、Transformers、推理后端版本和模型 revision。
2. 单卡启动服务，记录加载权重、视觉 profile、缓存分配完成后的显存。验证一张图、一条动作和解析过程。
3. 用真实图片分别测试 1 张、实际窗口上限张数，以及任务中最大的原生截图；测试窗口切换、长命令输出和最长生成预算。记录 processor 后 token 数。
4. 在上下文 8K、16K、32K、64K 等实际需要的档位，逐步增加 B=1、2、4、8。不能以短上下文成功替代最长输入压测。
5. 启动第二副本，重复双卡场景；测试冷启动、缓存未命中、混合长短请求及长轨迹。比较 eager 与 Graph 时只改变这一项。
6. 选择无 OOM、无持续抢占/交换、token 预算满足任务、延迟可接受并保留显存余量的配置，再开展完整 benchmark。若任务真实输入超过预算，记录为配置失败，不能计作模型能力失败。

GPU 侧采样示例，在运行评测时另开终端：

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
  --format=csv -lms 200 > gpu_memory.csv
```

200ms 采样可能漏掉短峰值，应结合引擎 profile 与 worker 内的 CUDA/PyTorch 峰值统计。若使用 Transformers 直接推理，在模型所在进程 `torch.cuda.reset_peak_memory_stats()` 后运行，再 synchronize 并读取 `max_memory_allocated()`、`max_memory_reserved()`；在服务客户端调用这些 API 无法测到服务端 GPU 内存。

建议每步保存以下 CPU/磁盘字段，避免把 GPU tensor 引用长期留在日志对象中：

| 类别 | 字段 |
| --- | --- |
| 身份 | run_id、benchmark 版本、task_id、trial_id、step_id、模型 revision、引擎配置 |
| 观测 | 原始截图路径及尺寸、处理后 grid/token 数、历史窗口和图片数 |
| 推理 | 输入/输出 token 数、完整原始输出与思考文本、解析动作、首 token 延迟、总延迟 |
| 资源 | 每卡显存、引擎 cache 使用、活跃请求数、抢占/OOM/超时事件 |
| 结果 | verifier 输出、任务成功标记、停止原因、环境错误分类 |

中途成功率应标明已完成任务/试验数，不把未完成任务默认为失败；最终同时报告每 trial 结果和与官方定义一致的汇总指标。保留失败轨迹，以便区分模型动作错误、解析错误、环境失败和资源配置失败。

环境数量与 GPU 活跃请求数并不相等。任务在等待 UI/网络时不持续执行 GPU kernel；可用队列保持模型繁忙。但增加环境也会消耗 CPU、系统内存、磁盘及可能的 GPU 渲染显存。若渲染或本地 judge 共享这两张卡，需加入 `M_other_GPU_processes` 后重算，不能直接沿用表格。

## 11 可重复计算与验收标准

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
