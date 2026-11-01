"""可微 token 级 GRPO 参考实现。不是训练器；HF 入口由 TRL 执行更新。"""

import torch


def grpo_loss(
    logp,
    old_logp,
    rewards,
    mask,
    epsilon=0.2,
    normalization="sequence",
    max_length=None,
):
    """输入 [题目数, 组大小, token数]；rewards 为 [题目数, 组大小]。

    logp 必须来自当前模型；old_logp 必须来自采样策略，不得用当前概率冒充。
    mask 只包含回答 token（含有效 EOS），不包含 prompt/padding。
    为便于学习，beta=0；没有实现 reference KL、动态采样或 DAPO。
    """
    if logp.ndim != 3 or logp.shape != old_logp.shape or mask.shape != logp.shape:
        raise ValueError("logp, old_logp, mask must have identical [B,G,T] shape")
    if rewards.shape != logp.shape[:2] or logp.shape[1] < 2:
        raise ValueError("rewards must be [B,G], G>=2")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1)")
    if normalization not in {"sequence", "constant"}:
        raise ValueError("unknown normalization")
    mask = mask.to(logp.dtype)
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must be binary")
    lengths = mask.sum(-1)
    if torch.any(lengths == 0):
        raise ValueError("empty completions must be excluded before loss")
    with torch.no_grad():
        centered = rewards - rewards.mean(dim=1, keepdim=True)
        advantage = centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
    ratio = torch.exp(logp - old_logp.detach())
    objective = torch.minimum(
        ratio * advantage.unsqueeze(-1),
        ratio.clamp(1 - epsilon, 1 + epsilon) * advantage.unsqueeze(-1),
    )
    sums = (objective * mask).sum(-1)
    if normalization == "sequence":
        return -(sums / lengths).mean()
    if max_length is None or max_length < logp.shape[-1]:
        raise ValueError("provide fixed max_length >= T")
    # 这里只改变长度归一化，不声称实现 Dr.GRPO 的全部配方。
    return -(sums / max_length).mean()
