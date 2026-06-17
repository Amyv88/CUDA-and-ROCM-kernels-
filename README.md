# CUDA-and-ROCM-kernels-

This repository now follows a minimal software-project layout for testing and benchmarking modern AI kernels across CUDA and ROCm hardware.

## Project layout

- `src/ai_kernel_benchmarks/` - reusable Python package code
- `benchmarks/plans/` - benchmark plan definitions for kernels such as FlashAttention-3
- `tests/` - focused unit tests for benchmark metadata and validation

## Current benchmark focus

The initial benchmark plan tracks representative workloads for:

- FlashAttention-3
- Paged attention
- Grouped GEMM / fused MLP style kernels

## Local validation

Run the unit tests from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
