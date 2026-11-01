# 04｜可选 GPU 路径（尚未实机验证）

## 先知道限制

这是普通 Transformers 生成 + TRL 的单进程 BF16 全参数训练，不是 LoRA、参数分片或 vLLM 配置。
没有把两个GPU的显存相加使用。为了先验证算法，第一版拒绝 WORLD_SIZE>1。
40/50系具体型号、驱动和 CUDA 支持需要在租用机器检查；本目录未完成依赖安装或GPU集成测试。

模型必须是已准备好的**本地目录**。代码设置 local_files_only，不自动下载，不执行 trust_remote_code。
建议从约0.5–2B模型的短回答试验起步，但不保证任务难度合适或某配置能放入显存。

## 环境与基线

先在独立虚拟环境安装官方、匹配 GPU 的 PyTorch，再安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
# 先按 PyTorch 官方说明选择 CUDA wheel，再执行下面命令。
python -m pip install -r requirements-hf.txt
python -m unittest discover -s tests -v
python -m pip freeze > environment-lock.txt
```

依赖文件是有界的候选环境，不是经过实机验证的 lockfile。入口固定 TRL 0.25.1 API；更换版本需重新测试。
参考：[TRL 0.25.1 文档](https://huggingface.co/docs/trl/v0.25.1/en/grpo_trainer)。

以下 `/path/to/local-model` 要替换为真实目录：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hf.py   --model /path/to/local-model --out runs/base-eval.json
CUDA_VISIBLE_DEVICES=0 python scripts/train_hf.py   --model /path/to/local-model --mode clean --steps 20 --seed 0 --out runs/hf-clean-s0
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_hf.py   --model runs/hf-clean-s0/final --out runs/hf-clean-s0/test-eval.json
```

基线和训练后评测必须使用同一 test seed/size、prompt、max_new_tokens 和 greedy 解码。
训练 temperature=1，评测 greedy：两者不同是刻意设计，因此不可直接把训练采样正确率与 greedy 测试正确率混为同一指标。

## 恢复检查

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_hf.py   --model /path/to/local-model --mode clean --steps 30 --seed 0   --out runs/hf-clean-s0 --resume runs/hf-clean-s0/checkpoint-20
```

`--steps 30` 是目标总更新步数，不是额外30步。上面的命令是**延长训练**，会改变学习率调度的总时长，不能当作严格的中断恢复对照。
若要验证中断恢复一致性，连续组和中断组从一开始都设置相同的总目标步数（例如30），中断组在第10步保存后停止，再使用同一总步数30恢复。
恢复时保持 mode、seed、模型与数据不变；更改优化器或调度设置也会破坏严格对照。
首次使用时，应比较“连续运行”与“中途恢复”的步数、样本及结果差异；GPU非确定性和奖励调用顺序可能影响严格一致性，尚未验证。
原始 reward trace 会追加，恢复重放区间可能有重复记录，应按 step/样本分析而非直接累计当独立样本。

## 噪声实验与预算

clean 路径正确后，使用**新的输出目录**，将 mode 改成 random_fp 或 structured_fp。
不先启动九个长任务。先离线审计，实测一小段，再按 `configs/pilot.json` 讨论正式运行预算。
随机扰动使用 seed+题目+回答+验证事件哈希，可复现但不是论文的原噪声实现。

保存内容：配置清单、pip freeze、逐样本原始输出/双奖励/FP/FN、checkpoint、final 模型与独立评测。
运行清单不会自动证明模型文件版本；还应记录模型来源 revision 或本地权重哈希、GPU信息与实际账单。
