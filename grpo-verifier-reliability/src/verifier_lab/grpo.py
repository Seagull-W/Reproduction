"""无第三方依赖的数学参考。最大化 J；训练损失 L = -J。"""

import math


def advantages(rewards, eps=1e-8):
    if len(rewards) < 2:
        raise ValueError("GRPO requires at least two samples per prompt")
    mean = sum(rewards) / len(rewards)
    # 显式使用总体标准差，不与 torch.std 默认无偏估计混淆。
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + eps) for r in rewards]


def clipped_objective(ratio, advantage, epsilon=0.2):
    if ratio <= 0 or not 0 < epsilon < 1:
        raise ValueError("invalid ratio or epsilon")
    clipped = max(1 - epsilon, min(1 + epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)


def objective_logprob_derivative(ratio, advantage, epsilon=0.2):
    # 对 log pi 求导；进入被截断的平坦区后导数为 0。
    # 边界点选择未截断一侧子梯度。
    if (advantage > 0 and ratio > 1 + epsilon) or (
        advantage < 0 and ratio < 1 - epsilon
    ):
        return 0.0
    return ratio * advantage


def softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    return [x / sum(e) for x in e]
