# Axiom accelerator backend

Vectra imports the sibling Axiom compiler/runtime package by default for the
supported CPU and CUDA tensor paths. CPU-backed `Array(f32/f64)`
elementwise/scalar/matmul, matrix-vector, vector-matrix, dot/vdot, trace,
determinant, inverse, solve, Cholesky, QR, and LU operations route through Axiom CPU
lowering to Veyra. CUDA-owning `Array(f32)` values allocate real device storage
when the CUDA driver can retain the requested primary context, and supported
kernels consume existing device pointers rather than staging through host
arrays.

## Validation

```sh
zig build test
zig build axiom-cpu-dispatch-smoke
zig build axiom-backend-policy-smoke
zig build axiom-cuda-dispatch-smoke
zig build axiom-cuda-device-smoke
zig build -Daxiom-cuda-expect=ran axiom-cuda-smoke
zig build -Doptimize=ReleaseFast example-large-matmul-add -- --execute --backend=cuda --require-cuda
```

CUDA validation requires a CUDA/libnvvm/PTXAS-capable host.

## CUDA owning-array behavior

- `Device.cuda(index).isAvailable()` is true when Axiom can load the CUDA driver
  and retain that device's primary context.
- `Array.*On(..., vx.cuda(i))`, deterministic `Context.*With(vx.onDevice(...))`
  creation helpers, and `.cuda(i)` allocate/copy real device storage.
- `.cpu()` explicitly downloads CUDA storage.
- `ArrayView.cuda()` remains unsupported until view/device storage semantics are
  implemented.
- CUDA `Array(f32).add/sub/mul/div` launch Axiom cached device-pointer
  elementwise kernels with existing device pointers, avoiding repeated
  compile/module-load overhead after the first operation per op.
- CUDA `Array(f32).matmul` uses Axiom's cached cuBLAS-backed SGEMM wrapper first
  for PyTorch-class throughput and falls back to the Axiom PTX/CUDA Tile IR seed
  if cuBLAS is unavailable.
- CUDA `vx.matmulAdd(Array(f32), Array(f32), Array(f32))` uses Axiom's cached
  cuBLAS SGEMM wrapper with `beta=1` so the addend is consumed in the GEMM
  epilogue instead of launching a separate add kernel.

## CPU-backed policy

`vx.axiom_backend` reports and executes the shared route policy:

- `selectElementwise(T, op, policy, lhs, rhs)` / `elementwise(...)`
- `selectScalarElementwise(T, op, policy, input, scalar, side)` /
  `elementwiseScalar(...)`
- `tryElementwiseScalarBroadcast(T, op, policy, lhs, rhs)`
- `selectMatmul(T, policy, lhs, rhs)` / `matmul(...)`

Supported CPU routes currently cover f32/f64 same-shape add/sub/mul/div, scalar
and one-element scalar-broadcast add/sub/mul/div, 2D matmul, matrix-vector,
vector-matrix, dot/vdot, trace, determinant, inverse, solve, Cholesky, QR, and LU
through Axiom CPU→Veyra. Unsupported shapes or dtypes return explicit errors or
fall back only where Vectra still has a non-Axiom generic implementation.

## `vx.axiom_cuda` API surface

- `enabled()`
- `planArrayF32(array, name)`
- `tryAddF32(lhs, rhs)` / `trySubF32` / `tryMulF32` / `tryDivF32`
- `tryAddF16(lhs, rhs)` / `trySubF16` / `tryMulF16` / `tryDivF16`
- `tryAddBF16(lhs, rhs)` / `trySubBF16` / `tryMulBF16` / `tryDivBF16`
- `tryAddViewF32(lhs_view, rhs_view)` / `trySubViewF32` / `tryMulViewF32` /
  `tryDivViewF32`
- `trySaxpyF32(alpha, x, y)`
- `tryAddScalarF32(input, scalar)` / `tryMulScalarF32` / `tryDivScalarF32`
- `trySaxpyScalarF32(alpha, scalar_x, y)`
- `tryMatmulF32(lhs, rhs)` / `tryMatmulF16(lhs, rhs)` /
  `tryMatmulBF16(lhs, rhs)`
- `vx.axiom_cpu.tryMatvecF32/F64`, `tryVecmatF32/F64`, `tryDotF32/F64`,
  `tryTraceF32/F64`, `tryDetF32/F64`, `tryInverseF32/F64`, and
  `trySolveF32/F64`, `tryCholeskyF32/F64`, `tryQrF32/F64`, `tryLuF32/F64` for CPU
  matrix-vector, vector-matrix, dot/vdot, trace, determinant, inverse, solve,
  Cholesky, QR, and LU lowering through Axiom CPU→Veyra
- `tryDeviceBinaryF32(op, lhs, rhs)`
- `tryDeviceMatmulF32(lhs, rhs)`
- `tryDeviceMatmulAddF32(lhs, rhs, addend)`
- `toDeviceF32(allocator, host)` and `DeviceArrayF32`
- `runSmoke(allocator)`

## Current limits

- Automatic Axiom dispatch covers contiguous same-shape elementwise,
  scalar/one-element scalar-broadcast, and contiguous 2D matmul for the dtypes
  listed in [`CUDA_DTYPE_SUPPORT.md`](CUDA_DTYPE_SUPPORT.md).
- General broadcast lowering, reductions, softmax, random CUDA creation, and
  CUDA view storage are not exposed yet.
- f16 and BFloat16 matmul call Axiom typed SIMT GEMM seed entry points, which
  report typed launch/readiness metadata while using widened f32 compute today.
- f64 CUDA tensor runtime support is not exposed yet; f64 supported accelerator
  routes are CPU→Veyra.
