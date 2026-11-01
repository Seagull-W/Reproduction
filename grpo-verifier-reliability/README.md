# GRPO 与验证器可靠性：从看懂代码到提出研究问题

> **状态：教学项目与复现计划。没有真实大模型训练结果，不能据此宣称复现了论文。**
> 本目录新增内容独立于仓库已有项目。CPU 示例用于理解机制；GPU 入口需要在实际租卡环境验证。

## 我们要回答什么？

训练 reward 变高，是否代表模型真的变好？当判分器会犯错时，模型可能学会“拿分”而非完成任务。
本项目先建立可信实验流程，再研究 **验证器错误的结构如何影响训练**。不仅追求最终分数，也要知道数据、采样、奖励、优势与梯度如何连起来。

## 阅读路线（按顺序，不必一次读完）

1. [论文导读](docs/01_paper.md)：作者做了什么？我们与原文有哪些不同？
2. [任务框架](docs/02_plan.md)：第一轮如何验收，哪些结论现在还不能下。
3. [GRPO 公式到代码](docs/03_code_walkthrough.md)：先看无依赖数学，再看 token 级可微实现。
4. 运行下方 CPU 实验，观察 **训练奖励** 和 **干净评测** 是两条不同曲线。
5. [GPU 操作手册](docs/04_gpu.md)：准备本地模型后，才进入真实小模型训练。
6. [网页沉淀模板](docs/05_knowledge_page.md)：把证据、失败与边界一起写下来。

## 不租 GPU 也能开始

需要 Python 3.10+，CPU 部分只用标准库。在本目录执行：

```bash
python -m unittest discover -s tests -v
python scripts/audit_rewards.py
python scripts/toy_train.py --mode clean --seed 0 --out runs/toy-clean-s0
python scripts/toy_train.py --mode random_fp --seed 0 --out runs/toy-random-s0
python scripts/toy_train.py --mode structured_fp --seed 0 --out runs/toy-structured-s0
```

每个输出目录包含 `metrics.csv` 和 `manifest.json`；已存在的目录不会被覆盖。
`clean_test_expected_accuracy` 是有限动作策略在保留题目上的精确期望分数，**不是 LLM benchmark**。
这三个命令的种子相同以减少无关差异，但训练策略改变后轨迹会分叉；需要多个种子，不应把一条曲线当统计证据。

## 文件如何对应训练流程？

```text
任务与干净答案            src/verifier_lab/tasks.py
          ↓ 采样回答
有噪声的训练判分器        src/verifier_lab/rewards.py
          ↓ 组内相对优势
数学与 clipping          src/verifier_lab/grpo.py
          ↓ 策略更新
有限动作教学训练          scripts/toy_train.py
可微 token 级损失参考     src/verifier_lab/torch_loss.py
真实 HF 模型训练入口      scripts/train_hf.py（由 TRL 执行更新）
          ↓ 独立保留集
干净验证器评测            scripts/evaluate_hf.py
```

`torch_loss.py` 是独立学习参考，**不会被 TRL 入口调用**，避免误以为手写 loss 已经过真实模型集成验证。

## 范围与资源

- 本项目不是通用后训练框架：先单进程、单卡、短回答、小任务，保持可读性。
- 可用租卡资源：40 系 48GB×2（最多4张）或50系32GB×2，双卡整机约6–7元/小时，月预算不超过1000元。
- 100元仅是首轮试验的**建议上限**，不是支出授权；脚本不租卡、不关机、不自动限制账单。
- 两张卡的显存不能自动合成一块。第一版可只暴露一张卡；另一张卡不被利用也可能继续计费。
- 先测真实吞吐与峰值，决定继续或停止；不预先承诺某模型能放下、多少小时完成。

## 产出与存储

遵循 [全仓库准则](../docs/EXPERIMENT_ARTIFACTS.md)，新实验可使用其中的运行清单、产物清单与记录模板。逐样本输出/双奖励、采样组与更新关系、独立评测、环境和资源记录优先保存，并形成实例外副本；图表从这些证据重建。

检查点默认在实例上滚动保留最近两个完整版本，研究节点另行保护，final 保留到评测和阶段复盘完成。当前无需把长期网盘归档作为开跑前提。大文件上传、下载与离线分析安排在无卡/CPU 环境；实例停止后的磁盘期限仍需登记。

现有脚本尚未实现完整监控、统一事件/采样组标识、自动回传或里程碑保护，具体缺口见准则第 11 节；准则与模板不自动补齐这些能力。

## 已有验证与未完成部分

见 [VALIDATION.md](VALIDATION.md)。CPU 代码通过不等于 GPU 训练通过；GPU 冒烟、恢复一致性、多种子实验均须实际运行后填结果。

论文和第三方实现的链接在导读中；本目录未复制作者仓库源码。
