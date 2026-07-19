# Axiom accelerator backend

Vectra imports the sibling Axiom compiler/runtime package by default for the
supported CPU and CUDA tensor paths. CPU-backed `Array(f32/f64)`
elementwise/scalar/unary (`sqrt/exp/exp2/expm1/log/log1p/log2/log10/sin/cos/tan/asin/acos/atan/square`)/matmul, matrix-vector, vector-matrix, dot/vdot, trace,
determinant, inverse, solve, Cholesky, QR, LU, and triangular-solve operations
plus Frobenius/one/inf/two/nuclear matrix norms, SVD, singular values, matrix rank, condition number, pseudo-inverse, and least-squares route through Axiom CPU lowering to
Veyra. CUDA-owning `Array(f32)` values allocate real device storage when the CUDA
driver can retain the requested primary context, and supported kernels consume
existing device pointers rather than staging through host arrays.

## Validation

```sh
zig build test
zig build axiom-cpu-dispatch-smoke
zig build axiom-backend-policy-smoke
zig build axiom-dialect-lowering-smoke
zig build axiom-cuda-dispatch-smoke
zig build axiom-cuda-device-smoke
zig build -Daxiom-cuda-expect=ran axiom-cuda-smoke
zig build -Doptimize=ReleaseFast example-large-matmul-add -- --execute --backend=cuda --require-cuda
```

CUDA validation requires a CUDA/libnvvm/PTXAS-capable host.

## Axiom dialect route

The long-term descriptor/lowering/runtime plan is in [`AXIOM_ARCHITECTURE_PLAN.md`](AXIOM_ARCHITECTURE_PLAN.md). This bridge document records the current executable CUDA surface; new work should first check whether it advances that architecture.

Vectra uses Axiom directly as the compiler/runtime boundary.  Frontends should model array work as linalg/memref/gpu-style records and let Axiom lower them through its structured-linalg, schedule, CUDA Tile/NVVM, and runtime ABI layers.  `vx.axiom_backend.lowerMatmulDialect(...)` is the first public Vectra-facing helper for that route; it returns Axiom's `DialectMatmulLoweringReport` with concrete evidence for:

- linalg/memref/gpu dialect registration and operation counts;
- operation-store, memref-type, structured-linalg, and schedule fingerprints;
- CUDA Tile/NVVM handoff fingerprints when the requested backend is CUDA;
- explicit CPU, CUDA, and planned MPS backend statuses.

Validation:

```sh
zig build axiom-dialect-lowering-smoke
```

MPS is intentionally represented as `planned_mps` until Axiom owns a real Metal/MPS runtime ABI.  That keeps dynamic backend policy honest without routing through another facade or pretending CUDA execution occurred.

## CUDA owning-array behavior

- `Device.cuda(index).isAvailable()` is true when Axiom can load the CUDA driver
  and retain that device's primary context. `Device.mps(index)` is part of the
  public backend selector surface, but `isAvailable()` is false until Axiom owns
  a real Metal/MPS storage/runtime ABI.
- `Array.*On(..., vx.cuda(i))`, deterministic `Context.*With(vx.onDevice(...))`
  creation helpers, and `.cuda(i)` allocate/copy real device storage.
- `.cpu()` explicitly downloads CUDA storage.
- `ArrayView.cuda()` remains unsupported until view/device storage semantics are
  implemented; `Array.mps()` / `ArrayView.mps()` return `InvalidDevice` today while
  dialect lowering reports the planned MPS route.
- CUDA `Array(f32/f64/f16/BFloat16).add/sub/mul/div` first builds Axiom
  memref-backed elementwise specs from Vectra device-storage descriptors, then
  lowers those specs to Axiom's cached compact CUDA elementwise ABI.  This keeps
  the Array API behind the Axiom descriptor/lowering boundary while retaining
  cached-kernel performance for contiguous device arrays.
- CUDA `Array(f32).add/sub/mul/div/exp2/expm1/log1p/log2/log10/sin/cos/tan/sum/prod/min/max(axis=0/1)/broadcast-add/sub/mul/div(row/column)/transpose/softmax(axis=0/1)/logSoftmax(axis=0/1)/maximum/minimum/addcmul/addcdiv/lerp/neg/abs/reciprocal/square/sqrt/rsqrt/exp/relu/threshold/leakyRelu/relu6/clip/clipArray/elu/celu/sigmoid/silu/hardsigmoid/hardswish/softsign/softshrink/powScalar(-1/-0.5/0/0.5/1/2/3)/mseLoss(.none)/l1Loss(.none)/smoothL1Loss(.none)/huberLoss(.none)` and `Array(f64).logSoftmax(axis=0/1)/sum/prod/min/max(axis=0/1)/broadcast-add/sub/mul/div(row/column)/transpose/maximum/addcmul/addcdiv/lerp/neg/abs/reciprocal/square/sqrt/rsqrt/exp/relu/threshold/leakyRelu/relu6/clip/clipArray/elu/celu/sigmoid/silu/hardsigmoid/hardswish/softsign/softshrink/powScalar(-1/-0.5/0/0.5/1/2/3)/mseLoss(.none)/l1Loss(.none)/smoothL1Loss(.none)/huberLoss(.none)` use Axiom device unary/elementwise
  kernels through descriptor-backed specs where the current runtime ABI has a
  matching lowering. f16 and BFloat16 2D `sum/prod/min/max(axis=0/1)` reductions, row/column broadcast binary, transpose, softmax(axis=0/1), logSoftmax(axis=0/1), plus widened activation/powScalar combinations such as `relu/sigmoid/softsign/clip/powScalar(-1/-0.5/0/0.5/1/2/3)` are covered by the CUDA device smoke.
- CUDA `Array(f32/f64/f16/BFloat16).matmul` builds an Axiom memref-backed
  `TensorGemmSpec` from device descriptors before lowering to Axiom's cached
  cuBLAS-backed GEMM runtime for PyTorch-class throughput.  Axiom also accepts
  padded row-major f32/f64/f16/BFloat16 GEMM memrefs via cuBLAS leading
  dimensions, routes transposed/non-row-major/negative-stride GEMM through its
  explicit copy-pack/GEMM/copy-unpack seed, and exposes rank-3 batched GEMM
  descriptors through a native cuBLAS strided-batched runtime for contiguous
  positive-stride batches, with an explicit loop-over-per-batch fallback for
  other legal layouts or older cuBLAS symbols.  Vectra CUDA
  `Array(f32/f64/f16/BFloat16).dot/vdot`, common `inner/outer`, `norm(p=2)`, `bmm`, N-D `matmul` cases whose leading batch
  dimensions are equal, whole-operand broadcastable, or mixed per-axis
  broadcastable, plus batched matrix-vector/vector-matrix `matmul` now call that
  Axiom batched GEMM runtime, flattening affine cases and using Axiom's
  higher-rank loop fallback for mixed broadcasts rather than falling back to a
  host loop; fused pack/unpack kernels and native mixed-broadcast throughput
  lowering remain future Axiom work.
- CUDA `vx.matmulAdd(Array(f32/f64/f16/BFloat16), ...)` builds an Axiom
  memref-backed matmul-add spec with separate addend/output descriptors before
  lowering to cached cuBLASLt-backed GEMM epilogues where available.  Padded
  row-major matmul-add memrefs use cuBLASLt leading dimensions.  Fused f32
  `matmul + add + sqrt/exp` uses the same memref-backed matmul-add spec before
  selecting Axiom's fused unary epilogue seed.

## CPU-backed policy

`vx.axiom_backend` reports and executes the shared route policy:

- `selectElementwise(T, op, policy, lhs, rhs)` / `elementwise(...)`
- `selectScalarElementwise(T, op, policy, input, scalar, side)` /
  `elementwiseScalar(...)`
- `tryElementwiseScalarBroadcast(T, op, policy, lhs, rhs)`
- `selectMatmul(T, policy, lhs, rhs)` / `matmul(...)`

Supported CPU routes currently cover f32/f64 same-shape add/sub/mul/div, sqrt/exp/exp2/expm1/log/log1p/log2/log10/sin/cos/tan/asin/acos/atan/square, scalar
and one-element scalar-broadcast add/sub/mul/div, 2D matmul, matrix-vector,
vector-matrix, dot/vdot, trace, determinant, inverse, solve, Cholesky, QR, LU,
triangular solve, Frobenius/one/inf/two/nuclear matrix norms, SVD, singular values, matrix rank, condition number, pseudo-inverse, and least-squares through Axiom
CPU→Veyra. Unsupported shapes or dtypes return explicit errors or fall back only
where Vectra still has a non-Axiom generic implementation.

## `vx.axiom_cuda` API surface

- `enabled()`
- `planArrayF32(array, name)`
- `tryAddF32(lhs, rhs)` / `trySubF32` / `tryMulF32` / `tryDivF32`
- `tryAddF16(lhs, rhs)` / `trySubF16` / `tryMulF16` / `tryDivF16`
- `tryAddBF16(lhs, rhs)` / `trySubBF16` / `tryMulBF16` / `tryDivBF16`
- `tryAddViewF32(lhs_view, rhs_view)` / `trySubViewF32` / `tryMulViewF32` /
  `tryDivViewF32` / `tryAbsViewF32` / `trySqrtViewF32` / `tryExpViewF32`; f64 also exposes `tryAbsViewF64` / `trySqrtViewF64` / `tryExpViewF64`
- `trySaxpyF32(alpha, x, y)`
- `tryAddScalarF32(input, scalar)` / `tryMulScalarF32` / `tryDivScalarF32`
- `trySaxpyScalarF32(alpha, scalar_x, y)`
- `tryMatmulF32(lhs, rhs)` / `tryMatmulF16(lhs, rhs)` /
  `tryMatmulBF16(lhs, rhs)`
- `tryDeviceBmmF32(lhs, rhs)` / `tryDeviceBmmF64(lhs, rhs)` /
  `tryDeviceBmmF16(lhs, rhs)` / `tryDeviceBmmBF16(lhs, rhs)` plus
  `lastCudaDeviceBatchedGemmReport()` for Axiom rank-3 batched GEMM runtime
  fingerprints
- `vx.axiom_cpu.tryMatvecF32/F64`, `tryVecmatF32/F64`, `tryDotF32/F64`,
  `tryTraceF32/F64`, `tryDetF32/F64`, `tryInverseF32/F64`, and
  `trySolveF32/F64`, `tryCholeskyF32/F64`, `tryQrF32/F64`, `tryLuF32/F64`,
  `trySolveTriangularF32/F64`, `tryMatrixNormF32/F64`, `trySvdF32/F64`,
  `trySingularValuesF32/F64`, `tryMatrixRankF32/F64`, `tryCondF32/F64`, `tryPinvF32/F64`, `tryLstsqF32/F64` for CPU
  matrix-vector, vector-matrix, dot/vdot, trace, determinant, inverse, solve,
  Cholesky, QR, LU, triangular-solve, Frobenius/one/inf/two/nuclear matrix-norm, SVD, singular-value, matrix-rank, condition-number, pseudo-inverse, and least-squares
  lowering through Axiom CPU→Veyra
- `tryDeviceBinaryF32(op, lhs, rhs)`
- `tryDeviceMatmulF32(lhs, rhs)`
- `tryDeviceMatmulAddF32(lhs, rhs, addend)`
- `toDeviceF32(allocator, host)` and `DeviceArrayF32`
- `runSmoke(allocator)`

## Current limits

- Automatic Axiom dispatch covers contiguous same-shape elementwise,
  CPU f32/f64 sqrt/exp/exp2/expm1/log/log1p/log2/log10/sin/cos/tan/asin/acos/atan/square, scalar/one-element scalar-broadcast, CPU f32/f64
  row/column-bias broadcast add, CPU f32/f64 2D axis reductions
  (`sum/prod/min/max`), CPU f32/f64 2D transpose, and contiguous 2D matmul for the dtypes
  listed in [`CUDA_DTYPE_SUPPORT.md`](CUDA_DTYPE_SUPPORT.md).
- General non-f32/f64/f16/BFloat16 reduction/broadcast/transpose CUDA eager execution, non-f32/f64/general softmax, random CUDA creation, and
  full CUDA view storage are not exposed yet; f32/f64/f16/BFloat16 host-backed 1D positive-stride view add/sub/mul/div now enter Axiom through descriptor-backed memref elementwise runtimes (`runTensorElementwiseBinaryMemRefsF32` for f32 and generic `runTensorElementwiseBinaryMemRefsNative` for f64/f16/BFloat16) before reusing the existing CUDA host-slice launch path; the f32/f64/f16/BFloat16 unary view bridges carry unary memref legality evidence, and the f32/f64/f16/BFloat16 scalar view bridges also carry zero-stride scalar memref descriptors. Host-backed 1D positive-stride `ArrayView(f32/f64/f16/BFloat16).add/sub/mul/div` and `ArrayView(f32/f64/f16/BFloat16).addScalar/subScalar/mulScalar/divScalar` plus f32 `log`, f32/f64/f16/BFloat16 `abs/sqrt/exp/neg/square/reciprocal` can execute through the central Axiom target facade with CUDA strided/unary/zero-stride scalar launch provenance when the default target is CUDA. CUDA f32 owning `Array.log()`, CUDA f32/f64/f16/BFloat16 2D `sum/prod/min/max(axis=0/1)`, and CUDA f32/f64/f16/BFloat16 2D `softmax(axis=0/1)` plus CUDA f32/f64/f16/BFloat16 2D `logSoftmax(axis=0/1)` have
  eager Axiom runtimes; other CUDA reduction/broadcast/transpose/softmax/logSoftmax dtypes
  dialect lowerings are visible through `vx.axiom_backend.lowerReductionDialect(...)`,
  `lowerBroadcastAddDialect(...)`, and `lowerTransposeDialect(...)`, until Axiom exposes matching eager runtime ABIs.
- Dialect-lowering helpers accept both CPU arrays and CUDA-tagged/device arrays
  when the operation shape and dtype are structurally valid. This keeps Vectra
  MLIR-like: array operations are described once and lowered for the requested
  Axiom target (`.cpu/.cuda/.mps`), while eager execution remains gated by the
  runtime capability reports above.
- f16 and BFloat16 reductions, broadcast binary, and transpose use Axiom typed CUDA kernels for contiguous 2D `sum/prod/min/max(axis=0/1)`, row/column bias binary, and 2D transpose. Their matmul paths call Axiom typed SIMT GEMM seed entry points, which
  report typed launch/readiness metadata while using widened f32 compute today.
- f64 CUDA same-shape/scalar elementwise, square/sqrt/exp, matmul, and matmulAdd/fusion
  are exposed for owning CUDA arrays; broader CUDA dtype reductions/broadcast/view
  storage remain future work.
