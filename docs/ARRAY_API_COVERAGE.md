# Vectra Array API Coverage Audit

This document tracks Vectra's long-term goal of matching NumPy/PyTorch-style
dense array computation while routing acceleration through Axiom.  It is a
roadmap gate, not a declaration that the goal is complete.

Autograd is out of scope.  The user clarified that automatic differentiation is
not required for Vectra; training/autograd semantics belong in Forge or another
upper layer.  This audit therefore checks array computation, shape semantics,
dtype/device behavior, and Axiom backend evidence only.

Run:

```sh
zig build array-api-coverage-audit
zig build dtype-promotion-smoke
zig build einsum-smoke
```

The build step runs `tools/array_api_coverage_audit.py`, which statically checks
that representative NumPy/PyTorch-style capability groups have public Array API
symbols and that the Axiom backend smoke gates remain wired into the build.

## Covered capability groups

The audit currently checks representative public symbols for:

- constructors and initializers;
- random sampling;
- reshape/view/broadcast/stride layout helpers;
- slicing, gather/take, scatter/put, masking, and `where`;
- arithmetic, comparisons, math ufuncs, and activation-style numerical ops;
- reductions, nan reductions, order statistics, histograms, and counts;
- matrix products, linear solves/factorizations, and matrix properties;
- sorting/top-k and set/search helpers;
- FFT/signal helpers;
- device transfers/creation and backend policy evidence;
- archive/byte serialization.

The audit deliberately groups aliases and snake/camel-case variants under one
capability.  Its job is to prevent large regressions and make gaps explicit,
not to prove every NumPy/PyTorch overload or edge-case behavior.

## Known gaps

- True MPS runtime ABI: MPS remains planned/unavailable until Axiom owns real
  Metal/MPS storage and runtime execution.
- Strided/transposed device GEMM lowering: Axiom now supports contiguous and
  padded row-major f32/f64/f16/BFloat16 GEMM memrefs through cuBLAS leading
  dimensions.  It still needs bufferization/layout-transform lowering for
  transposed, negative-stride, and batched device GEMM layouts.
- Complete NumPy/PyTorch dtype promotion matrix: `dtype-promotion-smoke` now
  checks every pair in Vectra's current `DType` set plus representative promoted
  value operations.  More NumPy/PyTorch scalar edge cases still need expansion.
- `einsum`/general contraction syntax: Vectra now has a bounded binary
  explicit-subscript `einsum-smoke` for common contractions and output
  reordering, including the common batched matmul form `bij,bjk->bik`, but full
  NumPy-compatible syntax (ellipsis, repeated labels, more than two operands,
  and optimized path planning) is not covered yet.
- Sparse tensor/array layouts are outside the current dense Array surface.

## Backend evidence

Backend correctness and policy are covered by separate gates:

- `zig build api-boundary-audit`
- `zig build axiom-descriptor-smoke`
- `zig build axiom-dialect-lowering-smoke`
- `zig build axiom-backend-policy-smoke`
- `zig build axiom-cpu-dispatch-smoke`
- `zig build axiom-cuda-dispatch-smoke`
- `zig build axiom-cuda-device-smoke`

The CUDA device smoke reports memref fingerprints for eager/pending CUDA paths
so CI can detect regressions back to raw pointer/shape ABI calls.
