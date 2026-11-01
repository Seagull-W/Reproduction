"""短命令也作为接口测试：没有 torch 时仍应能学和跑。"""

import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_toy_reproducibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            outputs = []
            for n in ("first", "second"):
                out = Path(d) / n
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts/toy_train.py"),
                    "--steps",
                    "5",
                    "--mode",
                    "random_fp",
                    "--seed",
                    "7",
                    "--out",
                    str(out),
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                outputs.append((out / "metrics.csv").read_text())
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertTrue(manifest["not_paper_reproduction"])
            self.assertEqual(outputs[0], outputs[1])
            again = subprocess.run(cmd, capture_output=True)
            self.assertNotEqual(again.returncode, 0)

    def test_help_without_gpu_dependencies(self):
        for name in ("toy_train.py", "train_hf.py", "evaluate_hf.py"):
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), "--help"],
                check=True,
                capture_output=True,
            )

    def test_audit_counts_and_no_false_negatives(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/audit_rewards.py")],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [json.loads(line) for line in r.stdout.splitlines()]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["false_negative_rate"], 0)
            self.assertEqual(row["negative_count"], 1200)
        self.assertEqual(rows[0]["false_positive_rate"], 0)
