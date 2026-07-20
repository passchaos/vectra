#!/usr/bin/env python3
"""Audit Vectra's public API boundary for the Forge/Axiom stack.

Vectra should stay an Array/NDArray numerical library. Forge should wrap Vectra
Array values as Tensor and own autograd/training semantics. Axiom should own
compiler/backend/kernel abstractions. This audit intentionally checks only
Vectra's public source surface and boundary docs so implementation comments in
backend bridges do not create false positives.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "src" / "root.zig"
ARRAY = REPO / "src" / "array.zig"
AXIOM_BACKEND = REPO / "src" / "backends" / "axiom_backend.zig"
README = REPO / "README.md"
BOUNDARY_DOC = REPO / "docs" / "API_BOUNDARY.md"
BUILD = REPO / "build.zig"
ZON = REPO / "build.zig.zon"

PUBLIC_SOURCE_FILES = tuple(sorted((REPO / "src").rglob("*.zig")))
AXIAL_GUARD_FILES = (BUILD, ZON, ROOT, ARRAY, AXIOM_BACKEND)
# Array implementation code must not bypass the Axiom target facade.  The root
# module should likewise expose backend diagnostics through axiom_backend rather
# than publishing target-specific bridge modules as API surface.
TARGET_FACADE_CLIENT_FILES = (ARRAY,)
PUBLIC_TARGET_BRIDGE_GUARD_FILES = (
    README,
    *tuple(sorted((REPO / "docs").glob("*.md"))),
    *tuple(sorted((REPO / "examples").glob("*.zig"))),
    *tuple(sorted((REPO / "tools").glob("*.zig"))),
)

BANNED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("public_tensor_alias", re.compile(r"\bpub\s+const\s+Tensor\b")),
    ("public_parameter_type", re.compile(r"\bpub\s+const\s+Parameter\b")),
    ("public_module_type", re.compile(r"\bpub\s+const\s+Module\b")),
    ("public_optimizer_type", re.compile(r"\bpub\s+const\s+Optimizer\b")),
    ("autograd_symbol", re.compile(r"\b(?:Autograd|autograd|requires_grad|requiresGrad|grad_fn|backward|zeroGrad|zero_grad)\b")),
)

REQUIRED_ROOT_SNIPPETS = (
    "pub const Array = array_mod.Array;",
    "pub const NDArray = array_mod.NDArray;",
    "pub const ArrayView = array_mod.ArrayView;",
    "pub const NDArrayView = array_mod.NDArrayView;",
    'pub const axiom_backend = @import("backends/axiom_backend.zig");',
)

REQUIRED_README_SNIPPETS = (
    "Vectra intentionally uses `Array`/`NDArray`",
    "automatic differentiation, training, and inference belong in the sibling `../forge` deep-learning framework",
    "linalg/memref/gpu dialect counts",
)

REQUIRED_BOUNDARY_SNIPPETS = (
    "Vectra owns Array",
    "Forge owns Tensor",
    "Axiom owns backend and kernel lowering",
    "Vectra lowers array operations through Axiom dialects",
    "Do not add a `Tensor` alias to Vectra",
    "Do not add autograd to Vectra",
)

REQUIRED_AXIOM_BACKEND_SNIPPETS = (
    "pub fn defaultExecutionTarget() DialectBackend",
    "pub fn executeElementwise(",
    "pub fn executeElementwiseScalar(",
    "pub fn executeMatmul(",
    "pub fn executeMatmulAdd(",
    "pub fn executeReduction(",
    "pub fn executeBroadcastAdd(",
    "pub fn executeTrace(",
    "pub fn executeDet(",
    "pub fn executeEigh(",
    "pub fn executeEigvalsh(",
    "pub const cpu = struct",
    "pub const cuda = struct",
    "pub fn transferStorage(",
    "pub fn hostElementCapacity(",
    "pub fn fillAllocated(",
    "pub fn reductionRuntimeCapability(",
    "pub fn broadcastAddRuntimeCapability(",
    "pub fn transposeRuntimeCapability(",
    "pub fn unaryRuntimeCapability(",
    "plannedMpsRuntimeCapability(",
    "pub fn planPendingMatmul(",
    "pub fn hostFallbackAllowed(",
    "pub fn shouldRestoreDeviceAfterHostCast(",
    "pub fn pendingMatmulDeviceSupported(",
)

FORBIDDEN_ROOT_TARGET_SPLIT_SNIPPETS = (
    "tryCpuMatmulAdd",
    "tryCudaMatmulAdd",
)

FORBIDDEN_ROOT_BACKEND_EXPORT_SNIPPETS = (
    'pub const axiom_cpu = @import("backends/axiom_cpu.zig");',
    'pub const axiom_cuda = @import("backends/axiom_cuda.zig");',
)

FORBIDDEN_PUBLIC_TARGET_BRIDGE_SNIPPETS = (
    "vx.axiom_cpu",
    "vx.axiom_cuda",
    '@import("backends/axiom_cpu.zig")',
    '@import("backends/axiom_cuda.zig")',
)

FORBIDDEN_ARRAY_TARGET_SPLIT_SNIPPETS = (
    "pendingCudaMatmul",
)

FORBIDDEN_ARRAY_STORAGE_SPLIT_SNIPPETS = (
    "axiom_backend.uploadStorage(",
    "axiom_backend.downloadStorage(",
    "axiom_backend.copyStorage(",
)

FORBIDDEN_AXIAL_SNIPPETS = (
    '@import("axial")',
    'b.dependency("axial"',
    '.axial',
    'axial_cuda',
    'axial-accelerator-smoke',
)

FORBIDDEN_DIRECT_ACCELERATOR_SNIPPETS = (
    "axiom_cpu_backend",
    "axiom_cuda_backend",
    '@import("backends/axiom_cuda.zig")',
    '@import("backends/axiom_cpu.zig")',
    "tryDeviceBinary",
    "tryDeviceUnary",
    "trySqrt",
    "tryExp",
    "runPendingMatmul",
    "runPendingMatmulAdd",
    "axiom_cuda_backend.BinaryOp",
    "axiom_cuda_backend.UnaryOp",
)

PUBLIC_FN_PATTERN = re.compile(r"(?m)^\s*pub\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

CUDA_ONLY_BUILD_SNIPPETS = (
    "example-axiom-cuda-bridge",
    "example-large-matmul-add",
    "bench-matmul-add-compare",
    "axiom-cuda-smoke",
    "axiom-cuda-dispatch-smoke",
    "axiom-cuda-device-smoke",
    "fusion-smoke",
    "fusion-production-gate",
)

MPS_ONLY_BUILD_SNIPPETS = (
    "axiom-mps-storage-smoke",
    "axiom-mps-gelu-smoke",
    "axiom-mps-rank3-smoke",
    "axiom-mps-rank3-broadcast-smoke",
    "axiom-mps-bmm-smoke",
    "axiom-mps-batched-vector-matmul-smoke",
    "axiom-mps-higher-rank-bmm-smoke",
    "axiom-mps-broadcast-bmm-smoke",
    "axiom-mps-mixed-bmm-smoke",
    "axiom-mps-inner-outer-smoke",
)

REQUIRED_PLATFORM_BUILD_SNIPPETS = (
    'const enable_axiom_cuda = !is_macos_target;',
    'const enable_axiom_cuda_dispatch = enable_axiom_cuda;',
    'b.option(bool, "axiom-cuda", "Compatibility flag: Axiom CUDA wrapping is enabled on non-macOS targets") orelse !is_macos_target',
    'b.option(bool, "axiom-cuda-dispatch", "Compatibility flag: supported CUDA dispatch uses Axiom on non-macOS targets") orelse !is_macos_target',
)

REQUIRED_TEST_GATE_DEPENDENCIES = (
    "run_mod_tests",
    "run_exe_tests",
    "api_boundary_audit_quiet_cmd",
    "array_api_coverage_audit_quiet_cmd",
    "dtype_promotion_smoke_cmd",
    "einsum_smoke_cmd",
    "contraction_smoke_cmd",
    "indexing_smoke_cmd",
    "shape_view_smoke_cmd",
    "axiom_cpu_dispatch_smoke_cmd",
    "axiom_backend_policy_smoke_cmd",
    "axiom_device_fallback_policy_smoke_cmd",
    "axiom_dialect_lowering_smoke_cmd",
    "axiom_descriptor_smoke_cmd",
    "axiom_gemm_layout_smoke_cmd",
    "basic_array_example_cmd",
    "axiom_backend_policy_example_cmd",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def public_fn_naming_issues(path: Path, text: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for match in PUBLIC_FN_PATTERN.finditer(text):
        name = match.group(1)
        if "_" not in name:
            continue
        line = text.count("\n", 0, match.start()) + 1
        issues.append({
            "kind": "non_zig_style_public_function",
            "path": str(path.relative_to(REPO)),
            "line": line,
            "match": name,
            "reason": "Public function APIs should use Zig-style camelCase names only; avoid snake_case or trailing-underscore aliases.",
        })
    return issues


def find_block_end(text: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def gated_build_step_issues(
    build_text: str,
    *,
    gate: str,
    snippets: tuple[str, ...],
    missing_gate_kind: str,
    unterminated_gate_kind: str,
    missing_step_kind: str,
    ungated_step_kind: str,
    reason: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    gate_ranges: list[tuple[int, int]] = []
    search_from = 0
    while True:
        gate_start = build_text.find(gate, search_from)
        if gate_start < 0:
            break
        gate_open = build_text.find("{", gate_start)
        gate_end = find_block_end(build_text, gate_open)
        if gate_end < 0:
            return [{
                "kind": unterminated_gate_kind,
                "path": "build.zig",
                "snippet": gate,
            }]
        gate_ranges.append((gate_start, gate_end))
        search_from = gate_end + 1

    if not gate_ranges:
        return [{
            "kind": missing_gate_kind,
            "path": "build.zig",
            "snippet": gate,
        }]

    for snippet in snippets:
        positions = [match.start() for match in re.finditer(re.escape(snippet), build_text)]
        if not positions:
            issues.append({"kind": missing_step_kind, "path": "build.zig", "snippet": snippet})
            continue
        for position in positions:
            if any(start <= position <= end for start, end in gate_ranges):
                continue
            line = build_text.count("\n", 0, position) + 1
            issues.append({
                "kind": ungated_step_kind,
                "path": "build.zig",
                "line": line,
                "snippet": snippet,
                "reason": reason,
            })
    return issues


def cuda_build_step_gating_issues(build_text: str) -> list[dict[str, Any]]:
    return gated_build_step_issues(
        build_text,
        gate="if (!is_macos_target) {",
        snippets=CUDA_ONLY_BUILD_SNIPPETS,
        missing_gate_kind="missing_cuda_non_macos_build_gate",
        unterminated_gate_kind="unterminated_cuda_non_macos_build_gate",
        missing_step_kind="missing_cuda_only_build_step",
        ungated_step_kind="cuda_only_build_step_not_gated",
        reason="CUDA/NVVM build steps must live inside the non-macOS build graph gate.",
    )


def mps_build_step_gating_issues(build_text: str) -> list[dict[str, Any]]:
    return gated_build_step_issues(
        build_text,
        gate="if (is_macos_target) {",
        snippets=MPS_ONLY_BUILD_SNIPPETS,
        missing_gate_kind="missing_mps_macos_build_gate",
        unterminated_gate_kind="unterminated_mps_macos_build_gate",
        missing_step_kind="missing_mps_only_build_step",
        ungated_step_kind="mps_only_build_step_not_gated",
        reason="MPS/Metal build steps must live inside the macOS build graph gate.",
    )


def test_gate_dependency_issues(build_text: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    step_start = build_text.find('const test_step = b.step("test", "Run tests");')
    if step_start < 0:
        return [{
            "kind": "missing_test_step",
            "path": "build.zig",
            "snippet": 'const test_step = b.step("test", "Run tests");',
        }]
    comment_after_step = build_text.find("// Just like flags", step_start)
    step_body = build_text[step_start:comment_after_step if comment_after_step >= 0 else len(build_text)]
    for dependency in REQUIRED_TEST_GATE_DEPENDENCIES:
        snippet = f"test_step.dependOn(&{dependency}.step);"
        if snippet not in step_body:
            issues.append({
                "kind": "missing_test_gate_dependency",
                "path": "build.zig",
                "snippet": snippet,
            })
    return issues


def main() -> int:
    quiet = "--quiet" in sys.argv[1:]
    issues: list[dict[str, Any]] = []

    for path in PUBLIC_SOURCE_FILES:
        text = read(path)
        issues.extend(public_fn_naming_issues(path, text))
        for issue_kind, pattern in BANNED_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                issues.append({
                    "kind": issue_kind,
                    "path": str(path.relative_to(REPO)),
                    "line": line,
                    "match": match.group(0),
                })

    root_text = read(ROOT)
    for snippet in REQUIRED_ROOT_SNIPPETS:
        if snippet not in root_text:
            issues.append({"kind": "missing_root_array_export", "path": "src/root.zig", "snippet": snippet})
    if "pub fn tryMatmulAddTarget(target: DialectBackend," not in root_text:
        issues.append({"kind": "missing_target_based_public_helper", "path": "src/root.zig", "snippet": "tryMatmulAddTarget"})
    for snippet in FORBIDDEN_ROOT_TARGET_SPLIT_SNIPPETS:
        if snippet in root_text:
            issues.append({"kind": "target_split_public_helper", "path": "src/root.zig", "snippet": snippet})
    for snippet in FORBIDDEN_ROOT_BACKEND_EXPORT_SNIPPETS:
        if snippet in root_text:
            issues.append({"kind": "target_specific_backend_public_export", "path": "src/root.zig", "snippet": snippet})

    readme_text = read(README)
    for snippet in REQUIRED_README_SNIPPETS:
        if snippet not in readme_text:
            issues.append({"kind": "missing_readme_boundary", "path": "README.md", "snippet": snippet})

    if not BOUNDARY_DOC.exists():
        issues.append({"kind": "missing_boundary_doc", "path": "docs/API_BOUNDARY.md"})
        boundary_text = ""
    else:
        boundary_text = read(BOUNDARY_DOC)
    for snippet in REQUIRED_BOUNDARY_SNIPPETS:
        if snippet not in boundary_text:
            issues.append({"kind": "missing_boundary_doc_snippet", "path": "docs/API_BOUNDARY.md", "snippet": snippet})

    build_text = read(BUILD)
    issues.extend(cuda_build_step_gating_issues(build_text))
    issues.extend(mps_build_step_gating_issues(build_text))
    issues.extend(test_gate_dependency_issues(build_text))
    for snippet in REQUIRED_PLATFORM_BUILD_SNIPPETS:
        if snippet not in build_text:
            issues.append({"kind": "missing_platform_build_gate", "path": "build.zig", "snippet": snippet})
    for snippet in ('b.dependency("axiom"', 'axiom-dialect-lowering-smoke'):
        if snippet not in build_text:
            issues.append({"kind": "missing_axiom_build_snippet", "path": "build.zig", "snippet": snippet})

    zon_text = read(ZON)
    for snippet in (".axiom", '.path = "../axiom"'):
        if snippet not in zon_text:
            issues.append({"kind": "missing_axiom_zon_snippet", "path": "build.zig.zon", "snippet": snippet})

    axiom_backend_text = read(AXIOM_BACKEND)
    for snippet in REQUIRED_AXIOM_BACKEND_SNIPPETS:
        if snippet not in axiom_backend_text:
            issues.append({"kind": "missing_axiom_target_facade", "path": "src/backends/axiom_backend.zig", "snippet": snippet})

    for path in AXIAL_GUARD_FILES:
        text = read(path)
        for snippet in FORBIDDEN_AXIAL_SNIPPETS:
            if snippet in text:
                issues.append({"kind": "forbidden_axial_dependency", "path": str(path.relative_to(REPO)), "snippet": snippet})

    for path in TARGET_FACADE_CLIENT_FILES:
        text = read(path)
        for snippet in FORBIDDEN_ARRAY_TARGET_SPLIT_SNIPPETS:
            if snippet in text:
                issues.append({"kind": "target_split_array_helper", "path": str(path.relative_to(REPO)), "snippet": snippet})
        for snippet in FORBIDDEN_DIRECT_ACCELERATOR_SNIPPETS:
            if snippet in text:
                issues.append({"kind": "direct_accelerator_dispatch_outside_axiom_backend", "path": str(path.relative_to(REPO)), "snippet": snippet})
        for snippet in FORBIDDEN_ARRAY_STORAGE_SPLIT_SNIPPETS:
            if snippet in text:
                issues.append({"kind": "direct_storage_transfer_outside_axiom_backend", "path": str(path.relative_to(REPO)), "snippet": snippet})

    # Docs, examples, and smoke tools are part of the practical public surface:
    # users copy these snippets first.  Keep them target-facade-oriented so the
    # lower-level CPU/CUDA bridge modules can continue shrinking behind
    # `vx.axiom_backend` instead of becoming stable APIs.
    for path in PUBLIC_TARGET_BRIDGE_GUARD_FILES:
        text = read(path)
        for snippet in FORBIDDEN_PUBLIC_TARGET_BRIDGE_SNIPPETS:
            if snippet in text:
                issues.append({"kind": "target_specific_bridge_public_usage", "path": str(path.relative_to(REPO)), "snippet": snippet})

    row = {
        "kind": "vectra_api_boundary_audit",
        "ok": not issues,
        "checked_public_sources": [str(path.relative_to(REPO)) for path in PUBLIC_SOURCE_FILES],
        "checked_target_facade_surfaces": [str(path.relative_to(REPO)) for path in PUBLIC_TARGET_BRIDGE_GUARD_FILES],
        "boundary_doc": str(BOUNDARY_DOC.relative_to(REPO)),
        "issues": issues,
        "issue_count": len(issues),
        "policy": {
            "vectra": "Array/NDArray numerical library",
            "forge": "Tensor/autograd/module/optimizer training framework over Vectra Array",
            "axiom": "linalg/memref/gpu dialects plus backend, compiler, kernel, CUDA/MPS/native lowering",
        },
    }
    if not quiet or not row["ok"]:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return 0 if row["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
