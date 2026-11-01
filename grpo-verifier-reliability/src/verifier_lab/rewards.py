"""有缺陷的训练奖励与可信评测分离。纯函数便于审计和复现。"""

import hashlib
from .tasks import clean_reward, parse_integer


def uniform_hash(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


def observed_reward(
    task, text, mode="clean", seed=0, nonce="", false_positive_rate=1 / 6
):
    if mode not in {"clean", "random_fp", "structured_fp"}:
        raise ValueError("unknown reward mode")
    if not 0 <= false_positive_rate <= 1:
        raise ValueError("false_positive_rate must be in [0,1]")
    truth = clean_reward(task, text)
    if truth or mode == "clean":
        return truth
    if mode == "random_fp":
        # nonce 应区分每次验证事件；同一次事件重复运行则结果相同。
        u = uniform_hash(f"{seed}|{task.id}|{text}|{nonce}")
        return float(u < false_positive_rate)
    # 人为设计、可学习的错误模式：a 为偶数时，将 answer+1 当成正确。
    # 这不是论文中的原始扰动；错误率随任务/策略分布变化。
    return float(task.a % 2 == 0 and parse_integer(text) == task.answer + 1)
