#!/usr/bin/env python3
"""只用干净 verifier 评测原模型/训练后模型，输出逐题结果；不读取训练奖励配置。"""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verifier_lab.tasks import make_tasks, clean_reward, parse_integer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refuse to overwrite existing evaluation")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    # 只在不相交的测试题上做固定 greedy 评测，不读取 noisy reward。
    model.eval()
    rows = []
    for task in make_tasks("test", args.size, args.seed):
        inputs = tok(task.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tok.pad_token_id
            )
        ids = output[0, inputs.input_ids.shape[1] :]
        text = tok.decode(ids, skip_special_tokens=True)
        rows.append(
            dict(
                task_id=task.id,
                prompt=task.prompt,
                completion=text,
                correct=clean_reward(task, text),
                format_valid=parse_integer(text) is not None,
                tokens=len(ids),
                reached_length_limit=len(ids) == 32
                and (tok.eos_token_id is None or int(ids[-1]) != tok.eos_token_id),
            )
        )
    result = dict(
        model=str(args.model.resolve()),
        split="test",
        seed=args.seed,
        decoding="greedy",
        max_new_tokens=32,
        accuracy=sum(r["correct"] for r in rows) / len(rows),
        samples=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))


if __name__ == "__main__":
    main()
