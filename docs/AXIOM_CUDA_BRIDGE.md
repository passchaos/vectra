# Optional Axiom CUDA bridge

Vectra can optionally call the sibling Axiom next-generation compiler
infrastructure for a small CUDA tensor accelerator seed.  The bridge is
deliberately opt-in and host-slice based today: it validates that Vectra
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
```

The smoke gate runs f32 add, f32 mul, f32 SAXPY, scalar-broadcast f32 add/SAXPY, and 2D f32 matmul through Axiom's
builder-style CUDA tensor runtime.  It also reports Vectra-to-Axiom buffer
planning evidence:

- logical element count
- required backing span
- logical and required byte counts
- linear-copy compatibility
- device-copy-plan status and fingerprints

## Current API surface

`vx.axiom_cuda` exposes:

- `enabled()`
- `planArrayF32(array, name)`
- `tryAddF32(lhs, rhs)`
- `tryMulF32(lhs, rhs)`
- `trySaxpyF32(alpha, x, y)`
- `tryAddScalarF32(input, scalar)`
- `trySaxpyScalarF32(alpha, scalar_x, y)`
- `tryMatmulF32(lhs, rhs)`
- `runSmoke(allocator)`

The `try*` functions return `null` when the optional backend is disabled,
unsupported by the current array metadata, or unavailable at runtime.  Callers
should fall back to Vectra's CPU/Veyra paths in that case.

## Current limits

- Only `Array(f32)` contiguous same-shape host arrays, scalar-broadcast f32 vector inputs, and contiguous 2D f32 matmul inputs are covered.
- The bridge does not change `Device.cuda(index).isAvailable()` yet.
- No persistent CUDA allocation/cache API is owned by Vectra yet.
- No broadcast lowering, reductions, or softmax bridge is exposed through Vectra yet.
- The matmul bridge is limited to contiguous 2D `Array(f32)` inputs.
- f64 linalg remains Veyra/CPU first until Axiom exposes matching tensor runtime
  support.

This is the first integration seam for a future CuPy/PyTorch-like Vectra GPU
backend, not the final GPU backend itself.
