import math, random, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verifier_lab.tasks import *
from verifier_lab.rewards import *
from verifier_lab.grpo import *


class CoreTests(unittest.TestCase):
    def test_split_disjoint(self):
        train = make_tasks("train", 1500)
        test = make_tasks("test", 400)
        train_pairs = {tuple(sorted((t.a, t.b))) for t in train}
        test_pairs = {tuple(sorted((t.a, t.b))) for t in test}
        self.assertFalse(train_pairs & test_pairs)
        for tasks in (train, test):
            self.assertEqual({t.answer % 5 for t in tasks}, set(range(5)))

    def test_strict_parser(self):
        self.assertEqual(parse_integer("  +42 "), 42)
        for text in ("The answer is 42", "42 43", "42.0", ""):
            self.assertIsNone(parse_integer(text))

    def test_clean(self):
        t = Task("x", 2, 3)
        self.assertEqual(clean_reward(t, "5"), 1)
        self.assertEqual(clean_reward(t, "6"), 0)

    def test_fp_only(self):
        for t in make_tasks("test", 100):
            for mode in ("clean", "random_fp", "structured_fp"):
                self.assertEqual(observed_reward(t, str(t.answer), mode), 1)

    def test_structured(self):
        self.assertEqual(observed_reward(Task("x", 2, 3), "6", "structured_fp"), 1)
        self.assertEqual(observed_reward(Task("y", 3, 3), "7", "structured_fp"), 0)

    def test_random_reproducible(self):
        t = Task("x", 2, 3)
        self.assertEqual(
            observed_reward(t, "6", "random_fp", 3, "event"),
            observed_reward(t, "6", "random_fp", 3, "event"),
        )
        self.assertEqual(observed_reward(t, "6", "random_fp", false_positive_rate=0), 0)
        self.assertEqual(observed_reward(t, "6", "random_fp", false_positive_rate=1), 1)

    def test_advantages(self):
        a = advantages([0, 1, 0, 1])
        self.assertAlmostEqual(sum(a), 0)
        self.assertEqual(advantages([1, 1, 1, 1]), [0] * 4)
        with self.assertRaises(ValueError):
            advantages([1])

    def test_clipping_direction(self):
        self.assertAlmostEqual(clipped_objective(1.5, 1), 1.2)
        self.assertAlmostEqual(clipped_objective(0.5, -1), -0.8)
        self.assertEqual(objective_logprob_derivative(1.5, 1), 0)
        self.assertEqual(objective_logprob_derivative(0.5, -1), 0)
        self.assertAlmostEqual(objective_logprob_derivative(0.5, 1), 0.5)

    def test_derivative_finite_difference(self):
        for ratio, A in [(0.7, 1), (1.1, -1), (1.4, 1), (0.6, -1)]:
            h = 1e-6
            numerical = (
                clipped_objective(ratio * math.exp(h), A)
                - clipped_objective(ratio * math.exp(-h), A)
            ) / (2 * h)
            self.assertAlmostEqual(
                numerical, objective_logprob_derivative(ratio, A), places=6
            )

    def test_softmax(self):
        self.assertAlmostEqual(sum(softmax([1000, 1001])), 1)


if __name__ == "__main__":
    unittest.main()
