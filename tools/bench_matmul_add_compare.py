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


def run_vectra(args: argparse.Namespace, m: int, n: int, k: int, warmup: int, iters: int) -> None:
    if args.skip_vectra:
        return
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
    print(json.dumps({
        "kind": "compare_invocation",
        "runner": "vectra",
        "cmd": cmd,
        "elapsed_ms": elapsed_ms,
        "returncode": proc.returncode,
    }, separators=(",", ":")))
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            print(json.dumps({"kind": "vectra_output", "line": line}, separators=(",", ":")))
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"kind": "vectra_output", "line": line}, separators=(",", ":")))
            continue
        row.setdefault("kind", "vectra_large_matmul_add")
        row["source"] = "vectra"
        print(json.dumps(row, separators=(",", ":")))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


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


def run_torch(args: argparse.Namespace, m: int, n: int, k: int, warmup: int, iters: int) -> None:
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
        print(json.dumps({**row_base, "skipped": True, "reason": "torch_cuda_unavailable"}, separators=(",", ":")))
        return

    torch.set_grad_enabled(False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as exc:  # pragma: no cover - version dependent
        print(json.dumps({**row_base, "kind": "torch_warning", "warning": f"set_float32_matmul_precision failed: {exc}"}, separators=(",", ":")))

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
            print(json.dumps({**row_base, "op": "torch_compile", "skipped": True, "reason": f"compile_create_failed:{exc}"}, separators=(",", ":")))

    for name, func in cases:
        try:
            elapsed_us, result = bench_torch_case(name, func, torch.cuda.synchronize, warmup, iters)
            first = float(result.reshape(-1)[0].item()) if result is not None and result.numel() else 0.0
            print(json.dumps({
                **row_base,
                "op": name,
                "elapsed_us": int(elapsed_us),
                "avg_us": elapsed_us / iters,
                "first": first,
                "ok": True,
            }, separators=(",", ":")))
        except Exception as exc:
            print(json.dumps({**row_base, "op": name, "ok": False, "error": repr(exc)}, separators=(",", ":")))


def main() -> None:
    args = parse_args()
    m, n, k = shape_from_args(args)
    warmup = args.warmup if args.warmup is not None else (3 if args.execute else 1)
    iters = args.iters if args.iters is not None else (5 if args.execute else 2)
    print(json.dumps({
        "kind": "matmul_add_compare_plan",
        "m": m,
        "n": n,
        "k": k,
        "warmup": warmup,
        "iters": iters,
        "dtype": args.dtype,
        "repo": str(args.repo),
    }, separators=(",", ":")))
    run_vectra(args, m, n, k, warmup, iters)
    run_torch(args, m, n, k, warmup, iters)


if __name__ == "__main__":
    sys.exit(main())
