#!/usr/bin/env python3
"""Compare Vectra/Axiom CUDA matmul+add with PyTorch addmm/torch.compile.

Examples:
    python3 tools/bench_matmul_add_compare.py --smoke
    python3 tools/bench_matmul_add_compare.py --execute --m 16384 --n 4096 --k 4096 --warmup 3 --iters 5

The script emits JSON lines. Vectra rows are forwarded from
`example-large-matmul-add`; PyTorch rows use the same shape/dtype/device and
synchronize CUDA so elapsed time is comparable with Vectra's synchronized Axiom
routes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Use a small quick shape unless m/n/k are provided.")
    mode.add_argument("--execute", action="store_true", help="Use the production shape unless m/n/k are provided.")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead", help="Mode passed to torch.compile.")
    parser.add_argument("--skip-torch-compile", action="store_true")
    parser.add_argument("--skip-vectra", action="store_true")
    parser.add_argument("--baseline", choices=("torch_addmm", "torch_eager_matmul_add", "torch_compile"), default="torch_addmm")
    parser.add_argument("--max-ratio", type=float, default=None, help="Fail if Vectra matmul_add avg_us / baseline avg_us exceeds this value.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def shape_from_args(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.execute:
        shape = [16384, 4096, 4096]
    else:
        shape = [512, 512, 512]
    if args.m is not None:
        shape[0] = args.m
    if args.n is not None:
        shape[1] = args.n
    if args.k is not None:
        shape[2] = args.k
    if min(shape) <= 0:
        raise SystemExit("m/n/k must be positive")
    return tuple(shape)  # type: ignore[return-value]


def emit(row: dict[str, Any]) -> dict[str, Any]:
    print(json.dumps(row, separators=(",", ":")))
    return row


def run_vectra(args: argparse.Namespace, m: int, n: int, k: int, warmup: int, iters: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.skip_vectra:
        return rows
    cmd = [
        "zig",
        "build",
        "-Doptimize=ReleaseFast",
        "example-large-matmul-add",
        "--",
        "--execute",
        "--backend=cuda",
        "--dtype=f32",
        f"--m={m}",
        f"--n={n}",
        f"--k={k}",
        f"--warmup={warmup}",
        f"--iters={iters}",
        "--require-cuda",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=args.repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    rows.append(emit({
        "kind": "compare_invocation",
        "runner": "vectra",
        "cmd": cmd,
        "elapsed_ms": elapsed_ms,
        "returncode": proc.returncode,
    }))
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            rows.append(emit({"kind": "vectra_output", "line": line}))
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append(emit({"kind": "vectra_output", "line": line}))
            continue
        row.setdefault("kind", "vectra_large_matmul_add")
        row["source"] = "vectra"
        rows.append(emit(row))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return rows


def bench_torch_case(name: str, func: Callable[[], Any], sync: Callable[[], None], warmup: int, iters: int) -> tuple[float, Any]:
    result = None
    for _ in range(warmup):
        result = func()
    sync()
    start = time.perf_counter_ns()
    for _ in range(iters):
        result = func()
    sync()
    elapsed_us = (time.perf_counter_ns() - start) / 1000.0
    return elapsed_us, result


def run_torch(args: argparse.Namespace, m: int, n: int, k: int, warmup: int, iters: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    import torch

    row_base: dict[str, Any] = {
        "kind": "torch_matmul_add_compare",
        "source": "torch",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dtype": args.dtype,
        "shape": [m, n, k],
        "warmup": warmup,
        "iters": iters,
    }
    if not torch.cuda.is_available():
        rows.append(emit({**row_base, "skipped": True, "reason": "torch_cuda_unavailable"}))
        return rows

    torch.set_grad_enabled(False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as exc:  # pragma: no cover - version dependent
        rows.append(emit({**row_base, "kind": "torch_warning", "warning": f"set_float32_matmul_precision failed: {exc}"}))

    device = torch.device("cuda")
    dtype = torch.float32
    a = torch.ones((m, k), device=device, dtype=dtype)
    b = torch.ones((k, n), device=device, dtype=dtype)
    c = torch.ones((m, n), device=device, dtype=dtype)
    torch.cuda.synchronize()

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("torch_addmm", lambda: torch.addmm(c, a, b)),
        ("torch_eager_matmul_add", lambda: a @ b + c),
    ]

    if not args.skip_torch_compile and hasattr(torch, "compile"):
        def matmul_add(x: Any, y: Any, z: Any) -> Any:
            return x @ y + z

        try:
            compiled = torch.compile(matmul_add, mode=args.torch_compile_mode)
            cases.append((f"torch_compile_{args.torch_compile_mode}", lambda: compiled(a, b, c)))
        except Exception as exc:
            rows.append(emit({**row_base, "op": "torch_compile", "skipped": True, "reason": f"compile_create_failed:{exc}"}))

    for name, func in cases:
        try:
            elapsed_us, result = bench_torch_case(name, func, torch.cuda.synchronize, warmup, iters)
            first = float(result.reshape(-1)[0].item()) if result is not None and result.numel() else 0.0
            rows.append(emit({
                **row_base,
                "op": name,
                "elapsed_us": int(elapsed_us),
                "avg_us": elapsed_us / iters,
                "first": first,
                "ok": True,
            }))
        except Exception as exc:
            rows.append(emit({**row_base, "op": name, "ok": False, "error": repr(exc)}))
    return rows


def summarize(args: argparse.Namespace, vectra_rows: list[dict[str, Any]], torch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectra_by_op = {row.get("op"): row for row in vectra_rows if row.get("source") == "vectra" and row.get("ok") is True}
    torch_by_op = {row.get("op"): row for row in torch_rows if row.get("source") == "torch" and row.get("ok") is True}
    baseline_key = args.baseline
    if baseline_key == "torch_compile":
        prefix = f"torch_compile_{args.torch_compile_mode}"
        baseline_row = next((row for op, row in torch_by_op.items() if isinstance(op, str) and op.startswith(prefix)), None)
        baseline_key = prefix
    else:
        baseline_row = torch_by_op.get(baseline_key)
    vectra_matmul_add = vectra_by_op.get("matmul_add")
    vectra_then_add = vectra_by_op.get("matmul_then_add")
    result: dict[str, Any] = {
        "kind": "matmul_add_compare_summary",
        "baseline": baseline_key,
        "max_ratio": args.max_ratio,
        "ok": False,
    }
    if baseline_row is None:
        result["reason"] = "missing_baseline"
        return result
    baseline_avg = float(baseline_row["avg_us"])
    result["baseline_avg_us"] = baseline_avg
    for label, row in (("vectra_matmul_add", vectra_matmul_add), ("vectra_matmul_then_add", vectra_then_add)):
        if row is None:
            result[f"{label}_missing"] = True
            continue
        avg = float(row["avg_us"])
        result[f"{label}_avg_us"] = avg
        result[f"{label}_ratio"] = avg / baseline_avg if baseline_avg else None
    ratios = [value for key, value in result.items() if key.startswith("vectra_") and key.endswith("_ratio") and isinstance(value, float)]
    if not ratios:
        result["reason"] = "missing_vectra_rows"
        return result
    result["best_ratio"] = min(ratios)
    result["worst_ratio"] = max(ratios)
    result["ok"] = args.max_ratio is None or result["worst_ratio"] <= args.max_ratio
    if not result["ok"]:
        result["reason"] = "ratio_exceeds_threshold"
    return result


def main() -> None:
    args = parse_args()
    m, n, k = shape_from_args(args)
    warmup = args.warmup if args.warmup is not None else (3 if args.execute else 1)
    iters = args.iters if args.iters is not None else (5 if args.execute else 2)
    emit({
        "kind": "matmul_add_compare_plan",
        "m": m,
        "n": n,
        "k": k,
        "warmup": warmup,
        "iters": iters,
        "dtype": args.dtype,
        "repo": str(args.repo),
        "baseline": args.baseline,
        "max_ratio": args.max_ratio,
    })
    vectra_rows = run_vectra(args, m, n, k, warmup, iters)
    torch_rows = run_torch(args, m, n, k, warmup, iters)
    summary = summarize(args, vectra_rows, torch_rows)
    emit(summary)
    if args.max_ratio is not None and not summary.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
