# 01｜论文导读：判分器为什么是研究对象

## 原始资料

- **Delay, Plateau, or Collapse: Evaluating the Impact of Systematic Verification Error on RLVR**
- Kazuki Egashira, Mark Vero, Jasper Dekoninck, Florian E. Dorner, Robin Staab, Martin Vechev；COLM 2026。
- [作者页面](https://www.sri.inf.ethz.ch/publications/egashira2026delay)
- [论文 v1 全文](https://arxiv.org/html/2605.02909v1) · [摘要与版本](https://arxiv.org/abs/2605.02909)
- [作者代码](https://github.com/eth-sri/llm-verifier-noise) · [MIT 许可](https://github.com/eth-sri/llm-verifier-noise/blob/main/LICENSE)

先读摘要、实验设置和附录 A。作者研究随机与系统性的验证错误，尤其说明：假阳性的影响不只由总体错误率决定，还依赖错误模式。不要把“所有噪声只会减慢训练”当成普遍结论。

## 最重要的区别

论文主实验采用 **DAPO**，在 Reasoning Gym 的 `decimal_chain_sum` 上实验；使用 Qwen3-1.7B-Base 和 OLMo3-7B。
本项目从 **GRPO + 整数加法** 入门，是小预算教学适配，不是原设置重跑。

| 维度 | 论文主设置的核对起点 | 本目录第一阶段 |
|---|---|---|
| 算法 | DAPO | 显式 GRPO；beta=0 |
| 数据 | decimal_chain_sum，多项小数运算 | 两个整数相加，固定不相交题目池 |
| 模型 | 上述两种模型 | CPU 有限动作；GPU 模型由本地目录指定 |
| 规模 | 500 steps；64题×4回答/步；最大回答2048 | 先20步冒烟；4回答组；最大回答32 |
| 优化器 | Adafactor | TRL 默认优化器，需保存解析后配置 |
| 错误 | 随机FP/FN、格式/语言FN、相对误差/词触发FP | 仅FP：随机与偶数题上的 answer+1 |

这张表只是阅读核对表，不代替逐项核对论文、作者版本和配置文件。作者代码依赖与本项目不同，不要混装到同一环境。

## 读论文时带着三个问题

1. 什么被称为“正确”：模型给的字符串，还是题目真正的数学答案？
2. 同一错误模式，在不同模型、任务与训练阶段是否会被触发？
3. 训练奖励上涨时，干净评测是否同样上涨？

本项目的随机假阳性率默认1/6，来自有限动作示例的粗略对照设计，**不是论文参数**。真实测得的错误率可能不同，更不能假定训练全程保持一致。

## 进一步研究的候选问题（尚未查重完成）

在固定人工或干净验证预算下，按错误模式分层检查，是否比均匀抽查更早发现能力下降？
先证明现象可复现，再谈检测或缓解。原文已研究交替使用 oracle/noisy verifier，因此“交替验证器”本身不能直接宣称创新。

若后续引用或修改作者源码，应固定 commit 并保留 MIT 版权与许可；这里目前只提供链接和独立教学实现。
