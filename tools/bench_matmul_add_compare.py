#!/usr/bin/env python3
"""Compare Vectra/Axiom CUDA matmul+add with PyTorch addmm/torch.compile.

Examples:
    python3 tools/bench_matmul_add_compare.py --smoke
    python3 tools/bench_matmul_add_compare.py --execute --m 16384 --n 4096 --k 4096 --warmup 3 --iters 5
    python3 tools/bench_matmul_add_compare.py --smoke --dtype f16
    python3 tools/bench_matmul_add_compare.py --smoke --dtype bf16

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
    parser.add_argument("--dtype", choices=("f32", "f64", "f16", "bf16"), default="f32")
    parser.add_argument("--repeat", type=int, default=1, help="Run the full Vectra+PyTorch comparison N times and aggregate ratios with median/worst summaries.")
    parser.add_argument(
        "--op",
        choices=("all", "matmul", "matmul_add", "matmul_then_add", "matmul_then_sub", "matmul_then_add_sqrt", "matmul_then_add_exp"),
        default="all",
        help="Optionally isolate one Vectra large_matmul_add op; default runs all Vectra ops.",
    )
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead", help="Mode passed to torch.compile.")
    parser.add_argument("--skip-torch-compile", action="store_true")
    parser.add_argument("--skip-vectra", action="store_true")
    parser.add_argument(
        "--baseline",
        choices=(
            "auto",
            "torch_matmul",
            "torch_addmm",
            "torch_eager_matmul_add",
            "torch_addmm_sub",
            "torch_eager_matmul_sub",
            "torch_addmm_sqrt",
            "torch_eager_matmul_add_sqrt",
            "torch_addmm_exp",
            "torch_eager_matmul_add_exp",
            "torch_compile",
            "torch_best",
        ),
        default="auto",
    )
    parser.add_argument("--max-ratio", type=float, default=None, help="Fail if selected Vectra op avg_us / baseline avg_us exceeds this value.")
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
        f"--dtype={args.dtype}",
        f"--m={m}",
        f"--n={n}",
        f"--k={k}",
        f"--warmup={warmup}",
        f"--iters={iters}",
        "--require-cuda",
    ]
    if args.op != "all":
        cmd.append(f"--op={args.op}")
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


def default_baseline_for_op(op_name: str) -> str:
    if op_name == "matmul":
        return "torch_matmul"
    if op_name in ("matmul_add", "matmul_then_add", "all"):
        return "torch_addmm"
    if op_name == "matmul_then_sub":
        return "torch_addmm_sub"
    if op_name == "matmul_then_add_sqrt":
        return "torch_addmm_sqrt"
    if op_name == "matmul_then_add_exp":
        return "torch_addmm_exp"
    raise ValueError(f"unsupported op for baseline selection: {op_name}")


def torch_expression(op_name: str, a: Any, b: Any, c: Any, k: int) -> Any:
    if op_name == "matmul":
        return a @ b
    if op_name in ("matmul_add", "matmul_then_add", "all"):
        return torch_addmm_expr(a, b, c)
    if op_name == "matmul_then_sub":
        return torch_addmm_sub_expr(a, b, c)
    if op_name == "matmul_then_add_sqrt":
        return torch_addmm_sqrt_expr(a, b, c)
    if op_name == "matmul_then_add_exp":
        return torch_addmm_exp_expr(a, b, c, k)
    raise ValueError(f"unsupported torch expression op: {op_name}")


def torch_addmm_expr(a: Any, b: Any, c: Any) -> Any:
    import torch

    return torch.addmm(c, a, b)


def torch_addmm_sub_expr(a: Any, b: Any, c: Any) -> Any:
    import torch

    return torch.addmm(c, a, b, beta=-1.0, alpha=1.0)


def torch_addmm_sqrt_expr(a: Any, b: Any, c: Any) -> Any:
    import torch

    return torch.sqrt(torch.addmm(c, a, b))


def torch_addmm_exp_expr(a: Any, b: Any, c: Any, k: int) -> Any:
    import torch

    return torch.exp(torch.addmm(c, a, b) * (1.0 / float(k + 1)))


def torch_dtype_from_name(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "f32":
        return torch.float32
    if dtype_name == "f64":
        return torch.float64
    if dtype_name == "f16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


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
    dtype = torch_dtype_from_name(torch, args.dtype)
    a = torch.ones((m, k), device=device, dtype=dtype)
    b = torch.ones((k, n), device=device, dtype=dtype)
    c = torch.ones((m, n), device=device, dtype=dtype)
    torch.cuda.synchronize()

    if args.op == "all":
        cases: list[tuple[str, Callable[[], Any]]] = [
            ("torch_addmm", lambda: torch.addmm(c, a, b)),
            ("torch_eager_matmul_add", lambda: a @ b + c),
        ]
    elif args.op == "matmul":
        cases = [("torch_matmul", lambda: a @ b)]
    elif args.op in ("matmul_add", "matmul_then_add"):
        cases = [
            ("torch_addmm", lambda: torch.addmm(c, a, b)),
            ("torch_eager_matmul_add", lambda: a @ b + c),
        ]
    elif args.op == "matmul_then_sub":
        cases = [
            ("torch_addmm_sub", lambda: torch.addmm(c, a, b, beta=-1.0, alpha=1.0)),
            ("torch_eager_matmul_sub", lambda: a @ b - c),
        ]
    elif args.op == "matmul_then_add_sqrt":
        cases = [
            ("torch_addmm_sqrt", lambda: torch.sqrt(torch.addmm(c, a, b))),
            ("torch_eager_matmul_add_sqrt", lambda: torch.sqrt(a @ b + c)),
        ]
    elif args.op == "matmul_then_add_exp":
        scale = 1.0 / float(k + 1)
        cases = [
            ("torch_addmm_exp", lambda: torch.exp(torch.addmm(c, a, b) * scale)),
            ("torch_eager_matmul_add_exp", lambda: torch.exp((a @ b + c) * scale)),
        ]
    else:
        raise ValueError(f"unsupported op: {args.op}")

    if not args.skip_torch_compile and hasattr(torch, "compile"):
        def selected_expression(x: Any, y: Any, z: Any) -> Any:
            return torch_expression(args.op, x, y, z, k)

        try:
            compiled = torch.compile(selected_expression, mode=args.torch_compile_mode)
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
    vectra_by_op = {row.get("op"): row for row in vectra_rows if row.get("source") == "vectra" and row.get("ok") is True and "avg_us" in row}
    vectra_skipped_by_op = {row.get("op"): row for row in vectra_rows if row.get("source") == "vectra" and row.get("skipped") is True}
    torch_by_op = {row.get("op"): row for row in torch_rows if row.get("source") == "torch" and row.get("ok") is True}
    baseline_key = default_baseline_for_op(args.op) if args.baseline == "auto" else args.baseline
    if baseline_key == "torch_compile":
        prefix = f"torch_compile_{args.torch_compile_mode}"
        baseline_row = next((row for op, row in torch_by_op.items() if isinstance(op, str) and op.startswith(prefix)), None)
        baseline_key = prefix
    elif baseline_key == "torch_best":
        baseline_row = min(torch_by_op.values(), key=lambda row: float(row["avg_us"])) if torch_by_op else None
        if baseline_row is not None:
            baseline_key = f"torch_best:{baseline_row.get('op')}"
    else:
        baseline_row = torch_by_op.get(baseline_key)

    result: dict[str, Any] = {
        "kind": "matmul_add_compare_summary",
        "requested_op": args.op,
        "baseline": baseline_key,
        "max_ratio": args.max_ratio,
        "ok": False,
    }
    if baseline_row is None:
        result["reason"] = "missing_baseline"
        return result
    baseline_avg = float(baseline_row["avg_us"])
    result["baseline_avg_us"] = baseline_avg

    if args.op == "all":
        vectra_targets = [
            ("vectra_matmul_add", vectra_by_op.get("matmul_add")),
            ("vectra_matmul_then_add", vectra_by_op.get("matmul_then_add")),
        ]
    else:
        vectra_targets = [(f"vectra_{args.op}", vectra_by_op.get(args.op))]

    for label, row in vectra_targets:
        if row is None:
            op_name = label[len("vectra_"):] if label.startswith("vectra_") else label
            skipped = vectra_skipped_by_op.get(op_name)
            if skipped is not None:
                result[f"{label}_skipped"] = True
                result[f"{label}_skip_reason"] = skipped.get("reason", "unknown")
            else:
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


def aggregate_summaries(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    ratios = [float(row["worst_ratio"]) for row in summaries if isinstance(row.get("worst_ratio"), (float, int))]
    best_ratios = [float(row["best_ratio"]) for row in summaries if isinstance(row.get("best_ratio"), (float, int))]
    result: dict[str, Any] = {
        "kind": "matmul_add_compare_repeat_summary",
        "repeat": args.repeat,
        "requested_op": args.op,
        "baseline": summaries[-1].get("baseline") if summaries else (default_baseline_for_op(args.op) if args.baseline == "auto" else args.baseline),
        "baseline_request": args.baseline,
        "max_ratio": args.max_ratio,
        "ok": False,
        "runs": len(summaries),
        "successful_ratio_runs": len(ratios),
    }
    if not ratios:
        result["reason"] = "missing_repeat_ratios"
        return result
    result["median_worst_ratio"] = median(ratios)
    result["worst_ratio"] = max(ratios)
    result["best_ratio"] = min(best_ratios) if best_ratios else min(ratios)
    result["all_worst_ratios"] = ratios
    result["ok"] = args.max_ratio is None or result["worst_ratio"] <= args.max_ratio
    if not result["ok"]:
        result["reason"] = "repeat_ratio_exceeds_threshold"
    return result


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise SystemExit("repeat must be positive")
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
        "op": args.op,
        "repeat": args.repeat,
        "repo": str(args.repo),
        "baseline": args.baseline,
        "max_ratio": args.max_ratio,
    })
    summaries: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        if args.repeat > 1:
            emit({"kind": "matmul_add_compare_repeat", "index": repeat_index + 1, "repeat": args.repeat})
        vectra_rows = run_vectra(args, m, n, k, warmup, iters)
        torch_rows = run_torch(args, m, n, k, warmup, iters)
        summary = summarize(args, vectra_rows, torch_rows)
        summary["repeat_index"] = repeat_index + 1
        summaries.append(summary)
        emit(summary)
    final_summary = aggregate_summaries(args, summaries) if args.repeat > 1 else summaries[-1]
    if args.repeat > 1:
        emit(final_summary)
    if args.max_ratio is not None and not final_summary.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
