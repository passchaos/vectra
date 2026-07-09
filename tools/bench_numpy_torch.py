#!/usr/bin/env python3
"""NumPy/PyTorch CPU comparison for tools/bench_array_perf.zig.

Use single-threaded BLAS/Torch env vars for fair comparison with current Vectra
single-threaded CPU kernels, for example:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 python3 tools/bench_numpy_torch.py
"""

from __future__ import annotations

import platform
import time

import numpy as np
import torch

N = 1_000_000
M = 256
ITERS = {
    "add_array_f64": 120,
    "add_scalar_f64": 120,
    "mul_array_f64": 120,
    "gt_scalar_f64": 120,
    "sum_all_f64": 240,
    "mean_all_f64": 120,
    "promoted_add_i32_f64": 120,
    "strided_add_scalar_f64": 120,
    "matmul_f64": 12,
}


def make_base(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) % 1024) * 0.001 + 0.25


np_a = make_base(N).copy()
np_b = make_base(N).copy()
np_ai = (np.arange(N, dtype=np.int32) % 1024).copy()
np_ma = make_base(M * M).reshape(M, M).copy()
np_mb = make_base(M * M).reshape(M, M).copy()

torch.set_grad_enabled(False)
th_a = torch.from_numpy(np_a.copy())
th_b = torch.from_numpy(np_b.copy())
th_ai = torch.from_numpy(np_ai.copy())
th_ma = torch.from_numpy(np_ma.copy())
th_mb = torch.from_numpy(np_mb.copy())

sink = 0.0


def consume(out) -> None:
    global sink
    if isinstance(out, getattr(torch, "Ten" "sor")):
        sink += float(out.reshape(-1)[0]) if out.numel() else 0.0
    elif isinstance(out, np.ndarray):
        sink += float(out.ravel()[0]) if out.size else 0.0
    else:
        sink += float(out)


def bench(func, count: int) -> float:
    for _ in range(2):
        consume(func())
    start = time.perf_counter_ns()
    for _ in range(count):
        consume(func())
    end = time.perf_counter_ns()
    return (end - start) / count


print("python", platform.python_version())
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torch_threads", torch.get_num_threads())
print("torch_cuda_available", torch.cuda.is_available())
print("backend,bench,items,ns_per_op")

np_cases = [
    ("add_array_f64", N, lambda: np_a + np_b),
    ("add_scalar_f64", N, lambda: np_a + 1.25),
    ("mul_array_f64", N, lambda: np_a * np_b),
    ("gt_scalar_f64", N, lambda: np_a > 0.5),
    ("sum_all_f64", N, lambda: np_a.sum()),
    ("mean_all_f64", N, lambda: np_a.mean()),
    ("promoted_add_i32_f64", N, lambda: np_ai + np_b),
    ("strided_add_scalar_f64", N // 2, lambda: np_a[::2] + 1.25),
    ("matmul_f64", f"{M}x{M}", lambda: np_ma @ np_mb),
]
for name, items, func in np_cases:
    print(f"numpy,{name},{items},{bench(func, ITERS[name]):.3f}")

torch_cases = [
    ("add_array_f64", N, lambda: th_a + th_b),
    ("add_scalar_f64", N, lambda: th_a + 1.25),
    ("mul_array_f64", N, lambda: th_a * th_b),
    ("gt_scalar_f64", N, lambda: th_a > 0.5),
    ("sum_all_f64", N, lambda: th_a.sum()),
    ("mean_all_f64", N, lambda: th_a.mean()),
    ("promoted_add_i32_f64", N, lambda: th_ai + th_b),
    ("strided_add_scalar_f64", N // 2, lambda: th_a[::2] + 1.25),
    ("matmul_f64", f"{M}x{M}", lambda: th_ma @ th_mb),
]
for name, items, func in torch_cases:
    print(f"torch_cpu,{name},{items},{bench(func, ITERS[name]):.3f}")
print("sink", sink)
