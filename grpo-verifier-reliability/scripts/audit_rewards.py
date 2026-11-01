#!/usr/bin/env python3
"""冻结、均匀候选分布上的离线审计，不是训练结果。"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verifier_lab.tasks import make_tasks, candidates, clean_reward
from verifier_lab.rewards import observed_reward

for mode in ("clean", "random_fp", "structured_fp"):
    fp = neg = fn = pos = 0
    for t in make_tasks("test", 400, 0):
        for i, s in enumerate(candidates(t)):
            true = clean_reward(t, s)
            obs = observed_reward(t, s, mode, 0, str(i))
            neg += not true
            pos += bool(true)
            fp += not true and bool(obs)
            fn += bool(true) and not obs
    print(
        json.dumps(
            dict(
                mode=mode,
                false_positive_rate=fp / neg,
                false_negative_rate=fn / pos,
                negative_count=neg,
                positive_count=pos,
            )
        )
    )
