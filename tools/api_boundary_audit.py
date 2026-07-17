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
README = REPO / "README.md"
BOUNDARY_DOC = REPO / "docs" / "API_BOUNDARY.md"
BUILD = REPO / "build.zig"
ZON = REPO / "build.zig.zon"

PUBLIC_SOURCE_FILES = (ROOT, ARRAY)

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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    issues: list[dict[str, Any]] = []

    for path in PUBLIC_SOURCE_FILES:
        text = read(path)
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
    for snippet in ('b.dependency("axiom"', 'axiom-dialect-lowering-smoke'):
        if snippet not in build_text:
            issues.append({"kind": "missing_axiom_build_snippet", "path": "build.zig", "snippet": snippet})

    zon_text = read(ZON)
    for snippet in (".axiom", '.path = "../axiom"'):
        if snippet not in zon_text:
            issues.append({"kind": "missing_axiom_zon_snippet", "path": "build.zig.zon", "snippet": snippet})

    row = {
        "kind": "vectra_api_boundary_audit",
        "ok": not issues,
        "checked_public_sources": [str(path.relative_to(REPO)) for path in PUBLIC_SOURCE_FILES],
        "boundary_doc": str(BOUNDARY_DOC.relative_to(REPO)),
        "issues": issues,
        "issue_count": len(issues),
        "policy": {
            "vectra": "Array/NDArray numerical library",
            "forge": "Tensor/autograd/module/optimizer training framework over Vectra Array",
            "axiom": "linalg/memref/gpu dialects plus backend, compiler, kernel, CUDA/MPS/native lowering",
        },
    }
    print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return 0 if row["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
