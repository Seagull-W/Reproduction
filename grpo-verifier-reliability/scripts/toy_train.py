#!/usr/bin/env python3
"""CPU 教学训练：两类问题 × 四个动作，不是语言模型实验。"""

import argparse, csv, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verifier_lab.tasks import make_tasks, candidates, clean_reward
from verifier_lab.rewards import observed_reward
from verifier_lab.grpo import advantages, softmax, objective_logprob_derivative


def evaluate(logits, tasks):
    # 对四个动作精确积分，避免把采样噪声误当能力变化。
    return sum(
        sum(
            p * clean_reward(t, s)
            for p, s in zip(softmax(logits[t.a % 2]), candidates(t))
        )
        for t in tasks
    ) / len(tasks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", choices=["clean", "random_fp", "structured_fp"], default="clean"
    )
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.steps < 1 or args.group_size < 2:
        ap.error("steps>=1 and group-size>=2 required")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    train = make_tasks("train", 128, args.seed)
    test = make_tasks("test", 128, args.seed)
    logits = [[0.0] * 4 for _ in range(2)]
    rows = []
    for step in range(args.steps + 1):
        row = {"step": step, "clean_test_expected_accuracy": evaluate(logits, test)}
        if step == args.steps:
            rows.append(row)
            break
        task = rng.choice(train)
        cls = task.a % 2
        old = softmax(logits[cls])
        actions = rng.choices(range(4), weights=old, k=args.group_size)
        rewards = [
            observed_reward(
                task, candidates(task)[a], args.mode, args.seed, f"{step}:{i}"
            )
            for i, a in enumerate(actions)
        ]
        adv = advantages(rewards)
        row.update(
            train_observed_reward=sum(rewards) / len(rewards),
            zero_advantage_group=int(all(x == 0 for x in adv)),
        )
        rows.append(row)
        # 固定 old policy 和同一组优势，复用该组做两次更新，才能看到 clipping 的作用。
        for _ in range(2):
            probs = softmax(logits[cls])
            grad = [0.0] * 4
            for a, A in zip(actions, adv):
                coeff = objective_logprob_derivative(probs[a] / old[a], A) / len(
                    actions
                )
                for j in range(4):
                    grad[j] += coeff * ((j == a) - probs[j])
            logits[cls] = [x + 0.1 * g for x, g in zip(logits[cls], grad)]
    with (args.out / "metrics.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "clean_test_expected_accuracy",
                "train_observed_reward",
                "zero_advantage_group",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                **vars(args),
                "out": str(args.out),
                "kind": "finite_action_cpu_teaching_demo",
                "not_paper_reproduction": True,
                "logits": logits,
            },
            indent=2,
        )
    )
    print(json.dumps(rows[-1]))


if __name__ == "__main__":
    main()
