#!/usr/bin/env python3
"""Audit Vectra's NumPy/PyTorch-style Array API coverage.

This is not a claim that the active goal is complete.  It is a regression gate
for the long-term roadmap: keep the public Array surface broad enough for
NumPy/PyTorch-style numerical computing, make missing areas explicit, and keep
autograd out of Vectra's scope.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ARRAY = REPO / "src" / "array.zig"
BUILD = REPO / "build.zig"
COVERAGE_DOC = REPO / "docs" / "ARRAY_API_COVERAGE.md"


@dataclass(frozen=True)
class Capability:
    category: str
    name: str
    required_symbols: tuple[str, ...]
    references: tuple[str, ...]


CAPABILITIES: tuple[Capability, ...] = (
    Capability("creation", "constructors_and_initializers", ("empty", "zeros", "ones", "full", "arange", "linspace", "eye", "fromSlice"), ("numpy", "torch")),
    Capability("creation", "random_sampling", ("rand", "randn", "randint", "normal", "uniform", "poisson", "gamma", "beta"), ("numpy.random", "torch.distributions")),
    Capability("shape", "reshape_and_views", ("reshape", "view", "flatten", "ravel", "squeeze", "unsqueeze", "expandDims"), ("numpy", "torch")),
    Capability("shape", "broadcasting", ("broadcastTo", "broadcastWith", "broadcastShape", "expandAs"), ("numpy", "torch")),
    Capability("shape", "layout_and_strides", ("asStrided", "stridesFor", "isContiguous", "contiguous", "storageOffset"), ("numpy", "torch")),
    Capability("indexing", "slicing_and_take", ("slice", "narrow", "select", "take", "takeAlongAxis", "gather"), ("numpy", "torch")),
    Capability("indexing", "scatter_and_put", ("scatter", "scatterAdd", "scatterReduce", "putAlongAxis", "indexPut"), ("numpy", "torch")),
    Capability("indexing", "masking_and_where", ("where", "maskedSelect", "maskedFill", "maskedScatter", "copyWhere"), ("numpy", "torch")),
    Capability("elementwise", "arithmetic", ("add", "sub", "mul", "div", "pow", "remainder", "floorDiv"), ("numpy.ufunc", "torch")),
    Capability("elementwise", "comparisons", ("equal", "notEqual", "less", "lessEqual", "greater", "greaterEqual", "isClose"), ("numpy", "torch")),
    Capability("elementwise", "math_ufuncs", ("sqrt", "rsqrt", "exp", "log", "sin", "cos", "tan", "erlang"), ("numpy.ufunc", "torch")),
    Capability("elementwise", "activation_numerics_without_autograd", ("relu", "gelu", "sigmoid", "silu", "softplus", "softmax", "logSoftmax"), ("torch.nn.functional",)),
    Capability("reductions", "axis_reductions", ("sum", "prod", "mean", "variance", "stddev", "min", "max", "all", "any"), ("numpy", "torch")),
    Capability("reductions", "nan_reductions", ("nansum", "nanmean", "nanvar", "nanstd", "nanmin", "nanmax"), ("numpy", "torch")),
    Capability("statistics", "order_statistics", ("median", "quantile", "percentile", "nanmedian", "nanquantile", "nanpercentile"), ("numpy", "torch")),
    Capability("statistics", "histogram_and_counts", ("histogram", "bincount", "countNonzero", "unique", "uniqueWithCounts"), ("numpy", "torch")),
    Capability("linear_algebra", "matrix_products", ("matmul", "matmulAdd", "bmm", "dot", "vdot", "inner", "outer", "tensordot", "tensorDot", "einsum"), ("numpy.linalg", "torch.linalg")),
    Capability("linear_algebra", "factorizations_and_solves", ("solve", "lstsq", "cholesky", "qr", "svd", "eigh", "lu"), ("numpy.linalg", "torch.linalg")),
    Capability("linear_algebra", "matrix_properties", ("trace", "det", "inv", "pinv", "matrixRank", "matrixNorm", "matrixPower"), ("numpy.linalg", "torch.linalg")),
    Capability("sorting", "sorting_and_topk", ("sort", "argsort", "partition", "argpartition", "topk", "kthValue"), ("numpy", "torch")),
    Capability("sets", "set_and_search", ("isin", "intersect1d", "union1d", "setdiff1d", "setxor1d", "searchsorted", "bucketize"), ("numpy", "torch")),
    Capability("fft_signal", "fft", ("fft", "ifft", "rfft", "irfft", "fft2", "ifft2"), ("numpy.fft", "torch.fft")),
    Capability("fft_signal", "signal_helpers", ("convolve1d", "convolve2d", "correlate1d", "correlate2d", "gradient"), ("numpy", "scipy", "torch")),
    Capability("device", "device_transfers_and_creation", ("cpu", "cuda", "mps", "emptyOn", "zerosOn", "onesOn", "fullOn", "fromSliceOn"), ("torch",)),
    Capability("device", "backend_policy_evidence", ("deviceBackend", "deviceBackendName", "fusionStatus", "isCuda", "isMps"), ("torch", "Axiom")),
    Capability("io", "serialization", ("saveArchive", "loadArchive", "toArchive", "fromArchive", "toBytes", "fromBytes"), ("numpy", "torch")),
)

KNOWN_GAPS: tuple[dict[str, Any], ...] = (
    {
        "name": "mps_kernel_runtime_parity",
        "reason": "Axiom now owns a real Metal/MPS device, command-queue, shared-buffer storage ABI, executable f32 2D Metal kernels for same-shape elementwise/scalar, unary math (abs/square/sqrt/exp/log/exp2/expm1/log1p/log2/log10/sin/cos/tan), matmul/matmulAdd, rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, dot/inner/outer/matvec/vecmat, transpose, row/column broadcast add/sub/mul/div plus rank<=6 general broadcast add/sub/mul/div and rank>2 last-dim broadcast via MPS composition, reductions, softmax, logSoftmax, softmin, and logSoftmin, plus f16 same-shape elementwise/scalar/device-scalar-broadcast/abs/square/sqrt/exp, 2D matmul/matmulAdd, rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, dot/inner/outer/matvec/vecmat, transpose, row/column broadcast add/sub/mul/div plus rank<=6 general broadcast add/sub/mul/div and rank>2 last-dim broadcast via MPS composition, sum/prod/min/max reductions, softmax, logSoftmax, softmin, and logSoftmin, plus BF16 same-shape elementwise/scalar/device-scalar-broadcast/abs/square/sqrt/exp, 2D matmul/matmulAdd, rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, dot/inner/outer/matvec/vecmat, transpose, row/column broadcast add/sub/mul/div plus rank<=6 general broadcast add/sub/mul/div and rank>2 last-dim broadcast via MPS composition, sum/prod/min/max reductions, softmax, logSoftmax, softmin, and logSoftmin covered by axiom-mps-storage-smoke; composed MPS coverage now also keeps f32/f16/BF16 rsqrt/leakyRelu/silu/hardsigmoid/hardswish/softshrink/tanh/tanhshrink/gelu/elu/celu/selu/SELU/relu6/hardtanh/clipArray, powScalar(-1/-0.5/0/0.5/1/2/3), softmin/logSoftmin, norm/normalize/cosineSimilarity/pairwiseDistance, mean/variance/stddev plus a focused rank-3 sum/prod/min/max reduction smoke, and mse/l1/smoothL1/huber(.none) on real MPS storage. CUDA/MPS host fallback is explicit and off by default, with axiom-device-fallback-policy-smoke covering the default error path. Remaining MPS dtypes/shapes still need runtime parity.",
        "target_layer": "axiom",
    },
    {
        "name": "strided_transposed_device_gemm_lowering",
        "reason": "Axiom supports contiguous/padded row-major f32/f64/f16/BF16 GEMM and matmul-add memrefs plus f32/f64/f16/BF16 transposed/non-row-major GEMM through copy-pack/GEMM/copy-unpack runtime seeds; fused pack/unpack kernels still need lowering; negative-stride GEMM enters the copy-pack/GEMM/copy-unpack runtime route, and Vectra CUDA dot/vdot, common inner/outer, norm(p=2), bmm, batched matvec/vecmat, plus N-D matmul with equal, whole-operand-broadcastable, or mixed per-axis broadcastable leading batch dimensions now call Axiom's batched/GEMM memref runtime. The Axiom runtime prefers native cuBLAS strided-batched GEMM for contiguous positive-stride batches and falls back to explicit loop-over-per-batch GEMM for higher-rank/mixed broadcast layouts; native mixed-broadcast throughput lowering remains future work.",
        "target_layer": "axiom",
    },
    {
        "name": "complete_numpy_torch_dtype_promotion_matrix",
        "reason": "dtype-promotion-smoke checks the full current Vectra DType pair matrix plus representative promoted array and scalar values; exact NumPy/PyTorch scalar edge-case compatibility still needs expansion.",
        "target_layer": "vectra",
    },
    {
        "name": "einsum_and_general_contraction_syntax",
        "reason": "A bounded einsum-smoke exists for common binary contractions, output reordering, implicit output inference, and common explicit/ellipsis batched matmul, matvec, vecmat, and batched-dot spellings; full NumPy einsum syntax is not covered yet.",
        "target_layer": "vectra_axiom",
    },
    {
        "name": "sparse_tensor_array_api",
        "reason": "Sparse layouts are outside the current dense Array surface.",
        "target_layer": "future",
    },
)

SCOPED_OUT: tuple[dict[str, str], ...] = (
    {
        "name": "autograd",
        "reason": "User clarified autograd is not required; Forge, not Vectra, owns training/autograd semantics.",
    },
)

REQUIRED_BUILD_SNIPPETS = (
    "api-boundary-audit",
    "dtype-promotion-smoke",
    "einsum-smoke",
    "contraction-smoke",
    "indexing-smoke",
    "shape-view-smoke",
    "axiom-descriptor-smoke",
    "axiom-gemm-layout-smoke",
    "axiom-dialect-lowering-smoke",
    "axiom-mps-storage-smoke",
    "axiom-cuda-device-smoke",
    "axiom-backend-policy-smoke",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def public_methods(source: str) -> set[str]:
    return set(re.findall(r"\bpub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source))


def capability_row(capability: Capability, methods: set[str]) -> dict[str, Any]:
    missing = [symbol for symbol in capability.required_symbols if symbol not in methods]
    return {
        "category": capability.category,
        "name": capability.name,
        "status": "covered" if not missing else "missing",
        "required_symbols": list(capability.required_symbols),
        "missing_symbols": missing,
        "references": list(capability.references),
    }


def main() -> int:
    array_text = read(ARRAY)
    methods = public_methods(array_text) | public_methods(read(REPO / "src" / "root.zig"))
    rows = [capability_row(capability, methods) for capability in CAPABILITIES]
    missing_required = [row for row in rows if row["status"] != "covered"]

    build_text = read(BUILD)
    missing_build_steps = [snippet for snippet in REQUIRED_BUILD_SNIPPETS if snippet not in build_text]

    doc_missing = []
    if not COVERAGE_DOC.exists():
        doc_missing.append(str(COVERAGE_DOC.relative_to(REPO)))
    else:
        doc_text = read(COVERAGE_DOC)
        for snippet in ("Autograd is out of scope", "Known gaps", "array-api-coverage-audit"):
            if snippet not in doc_text:
                doc_missing.append(f"{COVERAGE_DOC.relative_to(REPO)}:{snippet}")

    issues: list[dict[str, Any]] = []
    for row in missing_required:
        issues.append({
            "kind": "missing_required_array_api_symbol",
            "category": row["category"],
            "capability": row["name"],
            "missing_symbols": row["missing_symbols"],
        })
    for snippet in missing_build_steps:
        issues.append({"kind": "missing_required_backend_gate", "snippet": snippet})
    for missing in doc_missing:
        issues.append({"kind": "missing_array_api_coverage_doc", "item": missing})

    result = {
        "kind": "vectra_array_api_coverage_audit",
        "ok": not issues,
        "scope": "NumPy/PyTorch-style dense Array computation API; autograd excluded",
        "capability_count": len(rows),
        "covered_count": sum(1 for row in rows if row["status"] == "covered"),
        "missing_count": len(missing_required),
        "known_gap_count": len(KNOWN_GAPS),
        "scoped_out": SCOPED_OUT,
        "known_gaps": KNOWN_GAPS,
        "capabilities": rows,
        "issues": issues,
        "issue_count": len(issues),
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
