import tempfile
import unittest
from pathlib import Path

from ai_kernel_benchmarks import BenchmarkCase, BenchmarkPlan, default_plan, load_plan


REPO_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkSuiteTests(unittest.TestCase):
    def test_default_plan_covers_flash_attention_3(self) -> None:
        plan = default_plan()

        self.assertEqual(plan.name, "latest-ai-kernels")
        self.assertIn("flash-attention-3", {case.name for case in plan.cases})

    def test_benchmark_case_requires_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "metrics must not be empty"):
            BenchmarkCase(
                name="broken-case",
                category="attention",
                frameworks=("pytorch",),
                devices=("nvidia-hopper",),
                metrics=(),
            )

    def test_sample_plan_round_trips(self) -> None:
        plan = load_plan(REPO_ROOT / "benchmarks" / "plans" / "latest_ai_kernels.json")

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "plan.json"
            written = plan.write(target)

            self.assertEqual(load_plan(written), plan)

    def test_plan_requires_unique_case_names(self) -> None:
        case = BenchmarkCase(
            name="flash-attention-3",
            category="attention",
            frameworks=("pytorch",),
            devices=("nvidia-hopper",),
            metrics=("latency_ms",),
        )

        with self.assertRaisesRegex(ValueError, "case names must be unique"):
            BenchmarkPlan(
                name="duplicate",
                focus="Ensure bad plans are rejected.",
                cases=(case, case),
            )


if __name__ == "__main__":
    unittest.main()
