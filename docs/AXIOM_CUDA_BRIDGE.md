# Optional Axiom accelerator bridge

Vectra can optionally call the sibling Axiom next-generation compiler
infrastructure for small CPU and CUDA tensor accelerator seeds.  The bridge is
deliberately opt-in and host-slice based today: it validates that Vectra
`Array(f32/f64)` metadata can be mapped into Axiom CPU lowering and that
`Array(f32)` metadata can be mapped into Axiom tensor buffer/copy plans and real
CUDA runtime kernels, without claiming that `Array.cuda()` is already a
persistent device-resident storage backend.

## Enable the bridge

Default builds do not import Axiom:

```sh
zig build test
zig build axiom-cuda-smoke -Daxiom-cuda-expect=disabled
```

CUDA-capable hosts can opt in:

```sh
zig build -Daxiom-cuda=true -Daxiom-cuda-expect=ran axiom-cuda-smoke
zig build -Daxiom-cuda-dispatch=true axiom-cuda-dispatch-smoke
zig build -Daxiom-cuda=true axiom-cuda-device-smoke
zig build -Daxiom-cpu-dispatch=true axiom-cpu-dispatch-smoke
zig build -Daxiom-cpu-dispatch=true axiom-backend-policy-smoke
```

The CUDA smoke gate runs f32 add/sub/mul/div, f32 SAXPY, scalar-broadcast f32
add/SAXPY, experimental 1D positive-stride view add/sub/mul/div, 2D f32
matmul, native f16/BFloat16 same-shape elementwise seeds, and Axiom-owned
widened f16/BFloat16 GEMM/provenance seeds.  f32 elementwise/SAXPY paths use Axiom's
builder-style CUDA tensor runtime; f16/BFloat16 same-shape elementwise dispatch
now tries Axiom's native typed runtime seed before falling back to widened f32
compute; f16/BFloat16 elementwise provenance calls Axiom's widened elementwise runtime APIs directly;
matmul now builds Axiom CUDA Tile IR and hands it to Axiom's
Tile-IR-to-CUTILE GEMM runtime bridge.  It also reports
Vectra-to-Axiom buffer planning evidence:

- logical element count
- required backing span
- logical and required byte counts
- linear-copy compatibility
- device-copy-plan status and fingerprints
- `matmul_tile_ir_ok` evidence proving the matmul smoke went through the Axiom
  CUDA Tile IR bridge
- `f16_add_ok`, `f16_matmul_ok`, `bf16_add_ok`, and `bf16_matmul_ok` evidence for
  native/widened dtype bridges
- `f16_native_execution_fingerprint` evidence when the native f16 elementwise
  runtime seed is available
- `bf16_native_execution_fingerprint` evidence when the native BFloat16
  elementwise runtime seed is available


## Automatic dispatch and policy

`-Daxiom-cuda=true` only exposes explicit `vx.axiom_cuda.*` bridge calls.
`-Daxiom-cuda-dispatch=true` also lets supported ordinary `Array(f32)` methods
try Axiom CUDA first and fall back through the unified policy when unsupported or
unavailable.  `-Daxiom-cpu-dispatch=true` lets supported ordinary `Array(f32/f64)`
methods try Axiom CPU lowering to Veyra before the existing direct CPU path.
The current automatic dispatch covers contiguous same-shape `add/sub/mul/div`,
scalar `addScalar/subScalar/mulScalar/divScalar`, scalar-array broadcast
`add/sub/mul/div`, and contiguous 2D `matmul`.  CUDA native seed routes cover
f32 and same-shape f16/BFloat16 elementwise operations; f16/BFloat16 matmul
currently uses Axiom's widened GEMM runtime bridge before narrowing back.
For elementwise provenance, Vectra consumes Axiom's
`runTensorElementwiseBinaryF16Widened` and
`runTensorElementwiseBinaryBF16Widened` runtime reports instead of rebuilding
those reports locally; those Axiom reports can carry compute-run fingerprints
when the widened bridge delegates its f32 compute step to Axiom's f32 CUDA
elementwise runtime.
CPU scalar routes cover f32/f64 through Axiom CPU→Veyra.

`vx.axiom_backend` is the shared policy seam for both CPU and CUDA paths:

- `selectElementwise(T, op, policy, lhs, rhs)` reports the selected direct CPU,
  Axiom CPU→Veyra, or Axiom CUDA route for contiguous same-shape f32/f64
  add/sub/mul/div.
- `elementwise(T, op, policy, lhs, rhs)` executes that route and falls back to
  direct CPU if an optional Axiom route is disabled or unavailable.
- `selectScalarElementwise(T, op, policy, input, scalar, side)` and
  `elementwiseScalar(T, op, policy, input, scalar, side)` cover scalar
  add/sub/mul/div with explicit scalar-left/scalar-right semantics.
- `tryElementwiseScalarBroadcast(T, op, policy, lhs, rhs)` recognizes one-element
  scalar-array broadcast and routes it through the same policy.
- `selectMatmul(T, policy, lhs, rhs)` and `matmul(T, policy, lhs, rhs)` do the
  same for contiguous 2D matmul.

## Current API surface

`vx.axiom_cuda` exposes:

- `enabled()`
- `planArrayF32(array, name)`
- `tryAddF32(lhs, rhs)`
- `trySubF32(lhs, rhs)`
- `tryMulF32(lhs, rhs)`
- `tryDivF32(lhs, rhs)`
- `tryAddF16(lhs, rhs)`
- `trySubF16(lhs, rhs)`
- `tryMulF16(lhs, rhs)`
- `tryDivF16(lhs, rhs)`
- `tryAddBF16(lhs, rhs)`
- `trySubBF16(lhs, rhs)`
- `tryMulBF16(lhs, rhs)`
- `tryDivBF16(lhs, rhs)`
- `tryAddViewF32(lhs_view, rhs_view)`
- `trySubViewF32(lhs_view, rhs_view)`
- `tryMulViewF32(lhs_view, rhs_view)`
- `tryDivViewF32(lhs_view, rhs_view)`
- `trySaxpyF32(alpha, x, y)`
- `tryAddScalarF32(input, scalar)`
- `tryMulScalarF32(input, scalar)`
- `tryDivScalarF32(input, scalar)`
- `trySaxpyScalarF32(alpha, scalar_x, y)`
- `tryMatmulF32(lhs, rhs)`
- `tryMatmulF16(lhs, rhs)`
- `tryMatmulBF16(lhs, rhs)`
- `toDeviceF32(allocator, host)`
- `DeviceArrayF32`
- `runSmoke(allocator)`

The `try*` functions return `null` when the optional backend is disabled,
unsupported by the current array metadata, or unavailable at runtime.  Callers
should fall back to Vectra's CPU/Veyra paths in that case.

## Current limits

- Only contiguous same-shape `Array(f32/f16/BFloat16/f64)` add/sub/mul/div, scalar and one-element scalar-broadcast `Array(f32/f16/BFloat16/f64)` add/sub/mul/div, experimental 1D positive-stride `ArrayView(f32)` add/sub/mul/div bridge calls, and contiguous 2D f32/f16/BFloat16/f64 matmul inputs are covered by automatic policy dispatch.
- The bridge does not change `Device.cuda(index).isAvailable()` yet.
- An explicit `DeviceArrayF32` handle can acquire/release Axiom pool-backed device buffers; ordinary `.cuda()` persistent storage is still intentionally unavailable.
- Only scalar-array broadcast dispatch is covered; no general broadcast lowering, reductions, or softmax bridge is exposed through Vectra yet.
- The CUDA elementwise bridge is native for contiguous same-shape `Array(f32)`,
  `Array(f16)`, and `Array(BFloat16)` inputs when Axiom's typed CUBIN paths are
  available.
- The CUDA matmul bridge is native for contiguous 2D `Array(f32)` inputs and
  routes through Axiom CUDA Tile IR before the CUTILE/GEMM runtime path. f16 and
  BFloat16 elementwise provenance uses Axiom widened runtime reports, including
  f32 CUDA compute evidence when that runtime route is available; f16 and
  BFloat16 matmul use Axiom-owned widened GEMM runtime reports today, with typed SIMT GEMM
  launch-plan readiness metadata reported separately. f64 matmul routes through Axiom
  CPU→Veyra when `-Daxiom-cpu-dispatch=true`.
- The explicit ArrayView bridge is currently fallback-safe: it may return `null` on hosts where the strided CUDA runtime path reports `CudaError`, and is not part of the strict `ran` smoke gate yet.
- f64 CUDA tensor runtime support is not exposed yet.
- See [`CUDA_DTYPE_SUPPORT.md`](CUDA_DTYPE_SUPPORT.md) for the local CUDA dtype
  support matrix and current Vectra/Axiom bridge status.

This is the first integration seam for a future CuPy/PyTorch-like Vectra GPU
backend, not the final GPU backend itself.


Axiom CPU dispatch seed: `-Daxiom-cpu-dispatch=true` routes supported contiguous same-shape and scalar/broadcast `Array(f32/f64).add/sub/mul/div` plus contiguous 2D `Array(f32/f64).matmul` calls through Axiom CPU lowering to Veyra before falling back to Vectra CPU paths.


Unified Axiom backend policy seed: `vx.axiom_backend` reports and routes supported elementwise and matmul calls across direct CPU, Axiom CPU→Veyra, and Axiom CUDA policies; `Array.add/sub/mul/div` and `Array.matmul` now use this policy when Axiom CPU/CUDA dispatch flags are enabled.
