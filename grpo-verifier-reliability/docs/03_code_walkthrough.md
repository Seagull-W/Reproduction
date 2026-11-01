# 03｜GRPO：从公式读到代码

## 一轮的数据流

一道题 → 采样 G 个回答 → 各回答得到训练奖励 → 同题组内标准化 → 固定采样时概率 → 当前模型重算概率 → clipped loss → 更新。
不同题目必须分别算组内均值，不能把一个 batch 的所有回答误当同一组。

## 1. 相对优势

对同题的奖励 r_i：

`A_i = (r_i - mean(r)) / (population_std(r) + 1e-8)`

这里明确选总体标准差。全相同奖励的优势为0，这一组没有学习信号。代码在 `grpo.py:advantages`。
实际库可能采用不同标准差约定；因此参考实现与 TRL 不保证数值逐项相同。

## 2. 概率比与裁剪

`ratio = exp(logp_current - logp_old)`

`J = min(ratio * A, clip(ratio, 1-epsilon, 1+epsilon) * A)`

最大化 J 等价于最小化 -J。正优势限制过度增加概率，负优势限制过度减少概率。
`old_logp` 来自采样时策略，更新过程中固定，不能用当前模型的概率替换。

CPU 示例把每个完整回答视作一个离散动作；对 softmax logits 使用链式法则：
`d log pi(action) / d logits[j] = 1[j == action] - pi(j)`。
这解释了 `toy_train.py` 里的两层循环，不依赖自动求导隐藏细节。

## 3. 从一个动作变成 token 序列

`torch_loss.py` 输入形状：
- current/old log probability、mask：`[B,G,T]`
- rewards：`[B,G]`
- 优势沿 token 维广播，同一回答的 token 共享序列级奖励产生的优势。

先算每个 token 的 J，再用 mask 排除 prompt/padding。默认按回答实际长度平均，再对题目与回答平均。
提供 constant 长度归一化选项便于对照，但这**不是 Dr.GRPO 全配方**。
参考实现未加入 KL（beta=0），也没有 DAPO 的动态采样、非对称裁剪或长答案处理。

## 4. 为什么还有 TRL？

手写 loss 便于学习；真实训练还涉及 tokenizer、生成、梯度累积、检查点等。
`train_hf.py` 将这些交给固定版本 TRL，显式指定 `loss_type="grpo"`，避免默认算法变化。
两条路径独立：通过 CPU/手写 loss 测试不意味着 TRL GPU 集成已经通过。

## 建议自己修改的练习

1. 去掉标准差缩放，观察梯度幅度，解释奖励尺度的影响。
2. 在 toy 固定一组采样，多次更新，观察 ratio 进入裁剪区。
3. 令所有奖励相同，验证参数不变。
4. 对 token loss 改 padding 长度，确认有效 token 梯度不变。
5. 给错误模式增加新的子群，先离线审计，再决定是否训练。
