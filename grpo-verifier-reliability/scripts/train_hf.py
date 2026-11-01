#!/usr/bin/env python3
"""可选的真实小模型 GRPO 入口：本环境未运行 GPU，先阅读 docs/04_gpu.md。"""

import argparse, json, os, sys, subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verifier_lab.tasks import Task, make_tasks, clean_reward
from verifier_lab.rewards import observed_reward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=Path,
        required=True,
        help="已自行准备好的本地模型目录；不自动下载",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--mode", choices=["clean", "random_fp", "structured_fp"], default="clean"
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    if not args.model.is_dir():
        ap.error("model must be an existing local directory")
    if args.steps < 1:
        ap.error("steps>=1 required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        ap.error(
            "first version supports single process only; do not silently change batch semantics"
        )
    if args.resume and not args.resume.is_dir():
        ap.error("resume checkpoint not found")
    if args.out.exists() and not args.resume:
        ap.error("output exists; choose a fresh directory or explicitly resume")
    if args.resume:
        manifest_path = args.out / "manifest.json"
        if not manifest_path.exists():
            ap.error("resume requires the original manifest.json")
        original = json.loads(manifest_path.read_text())
        for key, value in {
            "model": str(args.model.resolve()),
            "mode": args.mode,
            "seed": args.seed,
        }.items():
            if original.get(key) != value:
                ap.error(f"resume mismatch in {key}; use original configuration")
        if args.resume.resolve().parent != args.out.resolve():
            ap.error("checkpoint must belong to this output directory")
        if not (args.resume / "trainer_state.json").exists():
            ap.error("checkpoint missing trainer_state.json")
        previous_step = json.loads((args.resume / "trainer_state.json").read_text())[
            "global_step"
        ]
        if args.steps <= previous_step:
            ap.error("target steps must exceed checkpoint global_step")
    # 1. 验证运行参数之后，才加载可选依赖和本地模型。
    import torch
    import trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if trl.__version__ != "0.25.1":
        raise RuntimeError(
            "This entry point targets trl==0.25.1; verify API before upgrading"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required for this optional entry point; use toy_train.py on CPU"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 support required; revise config deliberately if unavailable"
        )
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=bool(args.resume))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    # prompt 只含题目，答案列只给奖励函数；采用普通文本，避免 chat 模板隐式改变任务。
    train = Dataset.from_list(
        [
            dict(prompt=t.prompt, task_id=t.id, a=t.a, b=t.b)
            for t in make_tasks("train", 256, args.seed)
        ]
    )

    # 2. 同一回答同时保存真实判分和训练判分，但仅后者参与优化。
    def reward(completions, task_id, a, b, **kwargs):
        step = trainer.state.global_step
        rows = []
        for i, (text, tid, x, y) in enumerate(zip(completions, task_id, a, b)):
            if not isinstance(text, str):
                raise TypeError("expected plain-text completions")
            task = Task(tid, x, y)
            truth = clean_reward(task, text)
            observed = observed_reward(task, text, args.mode, args.seed, f"{step}:{i}")
            rows.append(
                dict(
                    step=step,
                    task_id=tid,
                    completion=text,
                    clean_reward=truth,
                    observed_reward=observed,
                    false_positive=int(not truth and observed),
                    false_negative=int(truth and not observed),
                )
            )
        # 原始样本用于排查：奖励上涨是否只是解析、格式或 verifier 漏洞。
        with (args.out / "reward_trace.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [r["observed_reward"] for r in rows]

    # 3. 把关键算法选项写明，避免库默认值悄悄改变实验。
    config = GRPOConfig(
        output_dir=str(args.out),
        max_steps=args.steps,
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=128,
        max_completion_length=32,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        temperature=1.0,
        top_p=1.0,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward,
        train_dataset=train,
        processing_class=tokenizer,
    )
    # 4. 先保存环境与配置，再启动训练；结果必须来自真实执行。
    manifest = {
        **vars(args),
        "model": str(args.model.resolve()),
        "out": str(args.out),
        "resume": str(args.resume) if args.resume else None,
        "kind": "integer_addition_GRPO_teaching_adaptation",
        "paper_reproduction": False,
        "training_config": config.to_dict(),
        "python": sys.version,
        "torch": torch.__version__,
        "trl": trl.__version__,
    }
    # resume 时保留最初的清单，并额外记录恢复调用。
    name = (
        "manifest-resume-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
        if args.resume
        else "manifest.json"
    )
    (args.out / name).write_text(json.dumps(manifest, indent=2, default=str))
    with (args.out / "pip-freeze.txt").open("w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model(str(args.out / "final"))
    tokenizer.save_pretrained(args.out / "final")


if __name__ == "__main__":
    main()
