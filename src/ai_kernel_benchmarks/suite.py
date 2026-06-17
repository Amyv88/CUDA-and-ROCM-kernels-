from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


def _ensure_values(field_name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    category: str
    frameworks: tuple[str, ...]
    devices: tuple[str, ...]
    metrics: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        _ensure_values("frameworks", self.frameworks)
        _ensure_values("devices", self.devices)
        _ensure_values("metrics", self.metrics)


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    name: str
    focus: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        _ensure_values("cases", self.cases)
        case_names = {case.name for case in self.cases}
        if len(case_names) != len(self.cases):
            raise ValueError("case names must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "focus": self.focus,
            "cases": [asdict(case) for case in self.cases],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def default_plan() -> BenchmarkPlan:
    return BenchmarkPlan(
        name="latest-ai-kernels",
        focus="Track representative CUDA and ROCm benchmark cases for modern attention kernels.",
        cases=(
            BenchmarkCase(
                name="flash-attention-3",
                category="attention",
                frameworks=("pytorch", "triton"),
                devices=("nvidia-hopper",),
                metrics=("latency_ms", "throughput_tflops", "memory_gb_s"),
                notes="Primary coverage for the latest FlashAttention release family.",
            ),
            BenchmarkCase(
                name="paged-attention",
                category="inference",
                frameworks=("pytorch",),
                devices=("nvidia-hopper", "amd-mi300x"),
                metrics=("tokens_per_second", "latency_ms"),
                notes="Covers LLM serving style kernels.",
            ),
            BenchmarkCase(
                name="grouped-gemm",
                category="training",
                frameworks=("hipblaslt", "cutlass"),
                devices=("nvidia-hopper", "amd-mi300x"),
                metrics=("throughput_tflops", "occupancy_pct"),
                notes="Useful for fused MLP and MoE style workloads.",
            ),
        ),
    )


def load_plan(path: str | Path) -> BenchmarkPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkPlan(
        name=payload["name"],
        focus=payload["focus"],
        cases=tuple(
            BenchmarkCase(
                name=case["name"],
                category=case["category"],
                frameworks=tuple(case["frameworks"]),
                devices=tuple(case["devices"]),
                metrics=tuple(case["metrics"]),
                notes=case.get("notes", ""),
            )
            for case in payload["cases"]
        ),
    )
