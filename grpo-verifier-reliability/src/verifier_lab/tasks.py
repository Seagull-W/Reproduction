"""任务和干净判分器：这部分永远不读取训练奖励的模式。"""

from dataclasses import dataclass
import random
import hashlib
import re


@dataclass(frozen=True)
class Task:
    id: str
    a: int
    b: int

    @property
    def answer(self):
        return self.a + self.b

    @property
    def prompt(self):
        return f"Compute {self.a} + {self.b}. Reply with only the integer answer."


def make_tasks(split, size, seed=0):
    # 不同 split 使用不相交的算式池；测试集与训练集不是换 seed 后再抽同一池。
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    # 对交换律等价题使用同一个哈希，避免 a+b 与 b+a 跨集合泄漏。
    # 不按答案取模划分，否则测试集会只包含某一类答案。
    def is_test(a, b):
        key = f"{min(a,b)}:{max(a,b)}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 5 == 0

    pool = [
        (a, b)
        for a in range(1, 51)
        for b in range(1, 51)
        if is_test(a, b) == (split == "test")
    ]
    if not 0 < size <= len(pool):
        raise ValueError(f"size must be in 1..{len(pool)}")
    random.Random(seed).shuffle(pool)
    return [Task(f"{split}:{a}:{b}", a, b) for a, b in pool[:size]]


def parse_integer(text):
    # 严格格式：不从长回答里捡一个碰巧正确的数字。
    text = text.strip()
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else None


def clean_reward(task, text):
    return float(parse_integer(text) == task.answer)


def candidates(task):
    # 仅供有限动作 toy；LM 自由生成不能访问这些候选或标准答案。
    return [str(task.answer + d) for d in (0, 1, -1, 2)]
