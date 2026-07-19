# Vectra + Axiom architecture plan

This plan is the architecture-first roadmap for aligning Vectra's array-compute
surface with NumPy/PyTorch while keeping CPU, CUDA, and MPS acceleration owned by
Axiom.  It is intentionally not an operation-by-operation checklist.  New eager
bridges are acceptable only when they move the implementation toward the target
lowering/runtime architecture described here.

## Target state

Vectra is the array frontend.  Axiom is the compiler/runtime/backend.  The
intended execution ladder is:

```text
Vectra Array / ArrayView operation
  -> Vectra operation descriptor (dtype, shape, strides, device, aliasing)
  -> Axiom linalg + memref + gpu dialect records
  -> Axiom canonicalization / broadcast expansion / fusion planning
  -> Axiom bufferization and view descriptor normalization
  -> Axiom target lowering (CPU / CUDA / MPS)
  -> Axiom runtime ABI launch with capability-gated fallback/error semantics
```

The public Vectra API may remain NumPy/PyTorch-like (`add`, `matmul`, `sum`,
`softmax`, `view`, `cuda`, etc.), but the implementation should not become a
parallel backend framework.  Vectra chooses a target and describes the work;
Axiom owns how that work is lowered and executed.

## Non-negotiable design constraints

- **No Axial dependency:** do not reintroduce `@import("axial")`,
  `b.dependency("axial")`, `axial_cuda`, `.axial`, or Axial smoke targets.
- **No high-level CPU/CUDA/MPS branching in Array APIs:** Array and ArrayView
  methods may call `src/backends/axiom_backend.zig`, but CUDA/MPS-specific ABI
  knowledge must not leak into the user-facing array layer.
- **MPS honesty:** MPS is a dialect target until Axiom owns real Metal/MPS
  storage, command-queue, kernel, synchronization, and runtime ABI support.  Do
  not pretend CPU execution is MPS execution for device arrays.
- **Capability gates over silent staging:** unsupported target/dtype/layout
  combinations must report planned/lowering-only/unavailable or return an
  explicit error.  Avoid hidden CPU round-trips for CUDA/MPS device arrays.
- **Local project-z first:** when Axiom needs math or linear-algebra kernels,
  check sibling libraries under `~/project-z` before adding external
  dependencies.

## Descriptor-first runtime architecture

The next major refactor should replace per-operation host-slice ABI growth with a
single descriptor model that Axiom can lower for every target.

### 1. Operation descriptors in Vectra

Vectra should build small, data-only descriptors from arrays/views:

- dtype and promoted dtype;
- rank, shape, strides, offset, storage span, and alias/overlap metadata;
- device backend/index and ownership/lifetime contract;
- operation kind (`unary`, `binary`, `reduction`, `matmul`, `broadcast`,
  `transpose`, `softmax`, etc.);
- semantic options such as axis, keepdims, scalar side, broadcasting policy,
  reduction identity, NaN policy, and math accuracy policy.

These descriptors are not backend kernels.  They are the frontend contract that
Axiom consumes.

### 2. Axiom memref/view descriptors

Axiom needs a reusable memref-like descriptor ABI for host and device buffers:

```text
base pointer / device pointer
storage offset
rank
sizes[]
strides[]
element type
address space / device target
layout flags (contiguous, dense-strided, broadcast, overlapping, reversed)
```

The descriptor must support scalar broadcast (`stride == 0`), positive/negative
strides, empty tensors, non-zero offsets, and explicit materialization decisions.
This is the foundation for eventually implementing `ArrayView.cuda()` without
copying through host storage.

### 3. Axiom lowering stages

Axiom should own these transformations:

1. **Shape and dtype legality:** validate ranks, broadcasting, accumulation
   dtype, exact/inexact math, and target support.
2. **Canonical linalg form:** map Vectra operations into linalg.generic,
   linalg.matmul, linalg.reduce, linalg.transpose, or equivalent structured ops.
3. **Broadcast/view normalization:** decide whether a view can be lowered as
   strided memref, scalar-broadcast memref, or must be materialized.
4. **Fusion planning:** fuse unary/binary chains, matmul epilogues, reductions
   with post-ops, and activation-style operations where legal.
5. **Bufferization:** allocate/reuse output buffers, preserve aliasing rules, and
   avoid unsafe in-place writes for overlapping views.
6. **Target lowering:** CPU loops/vectorization/Veyra, CUDA kernels/cuBLAS/cuBLASLt,
   and future MPS/Metal kernels.
7. **Runtime launch:** submit via an Axiom-owned runtime ABI with deterministic
   reports/fingerprints.

### 4. Target policy

Vectra exposes target selection (`cpu`, `cuda`, `mps`) and a default backend
selector.  Axiom returns one of:

- `executable`: supported and ran on the requested target;
- `lowering_only`: valid IR/lowering exists but no eager runtime ABI yet;
- `planned`: accepted architectural target, not yet implemented;
- `unavailable`: target/device/runtime unavailable;
- `unsupported`: invalid dtype/layout/operation combination.

Vectra should use these statuses to decide whether to run, fall back to a real
CPU path for CPU arrays, or return an explicit error.

## Implementation milestones

### Milestone A: descriptor ABI seed

- Add Axiom `TensorMemRefDescriptor` / `TensorViewDescriptor` records with
  deterministic fingerprints and text/json evidence.
- Add Vectra conversion helpers from `Array` and `ArrayView` to those records.  The first smoke gate is `zig build axiom-descriptor-smoke`.
- Add verifier coverage for contiguous, strided, scalar-broadcast, empty,
  non-zero-offset, and rejected-overlap examples.

### Milestone B: descriptor-backed unary/binary execution

- Replace specialized host-slice unary/binary bridges with descriptor-backed
  Axiom entry points.
- Keep current f32/f64/f16/BFloat16 paths as compatibility wrappers around the
  descriptor ABI, not as independently growing API families.
- Validate f32/f64/f16/BFloat16 unary/binary/scalar broadcast through the same
  runtime descriptor contract.  The current first executable slices are f32
  strided `ArrayView.add/sub/mul/div` via Axiom
  `runTensorElementwiseBinaryMemRefsF32` and f64/f16/BFloat16 strided
  `ArrayView.add/sub/mul/div` plus scalar broadcast through Axiom's generic
  `runTensorElementwiseBinaryMemRefsNative` ABI, and f32/f64/f16/BFloat16 strided
  `ArrayView.abs/sqrt/exp` plus f32 `ArrayView.log/exp2/expm1/log1p/log2/log10/sin/cos/tan` through unary memref runtime ABIs;
  `axiom-cuda-smoke` must report non-zero f32/f64/f16/BFloat16 strided,
  unary, and scalar-broadcast memref legality fingerprints so
  the bridge cannot silently regress to stride-only host-slice calls.

### Milestone C: reductions, broadcasts, transpose, and softmax

- Lower all supported 1D/2D reductions and broadcast adds via descriptor-backed
  linalg records.
- Generalize to N-D rank where the descriptor/lowering can prove legality.
- Keep target capability reports honest for unsupported dtype/axis/rank.

### Milestone D: device view storage

- Implement CUDA device ArrayView descriptors with offset/stride semantics.
- Add `ArrayView.cuda()` only when it can return a real device view/storage
  contract rather than a host-backed tag.
- Add stream/context lifetime rules and synchronization evidence.

### Milestone E: MPS runtime ABI

- Add Axiom Metal/MPS storage allocation, upload/download, command queue,
  kernel launch, synchronization, and runtime reports.
- Make MPS eager execution executable slice by slice as the runtime ABI grows;
  macOS must use MPS/Metal rather than CUDA paths.

### Milestone F: fusion and graph capture

- Add Axiom fusion plans for common NumPy/PyTorch chains: matmul+add, unary
  post-ops, reductions followed by normalization, softmax/logSoftmax, and loss
  kernels.
- Add Vectra-side graph/plan capture only as data that Axiom lowers; do not add a
  Vectra-owned compiler.

## Acceptance gates

A change that claims progress toward this architecture should provide evidence
from the relevant subset of:

- `zig build api-boundary-audit` for no Axial and boundary compliance;
- `zig build axiom-dialect-lowering-smoke` for linalg/memref/gpu route evidence;
- descriptor verifier tests in Axiom for shape/stride/alias/device contracts;
- `zig build axiom-descriptor-smoke` for Vectra Array/ArrayView to Axiom descriptor conversion evidence;
- `zig build test` in Axiom and Vectra;
- `zig build axiom-backend-policy-smoke` for dynamic target policy;
- `zig build axiom-cpu-dispatch-smoke` for CPU runtime paths;
- `zig build axiom-cuda-dispatch-smoke`, `axiom-cuda-device-smoke`, and
  `axiom-cuda-smoke -- --json` for CUDA runtime paths;
- real Metal/MPS storage ABI evidence plus executable-kernel evidence for any
  claimed MPS slice.

## Current gaps vs. the target state

- Many current eager CUDA paths still call typed runtime wrappers instead of a
  unified memref/view descriptor ABI.
- Host-backed ArrayView CUDA coverage exists for selected 1D positive-stride
  cases, but full device view storage and `ArrayView.cuda()` remain unimplemented.
- Fusion is mostly ad hoc (for example matmul epilogues) rather than a general
  Axiom pass over linalg IR.
- MPS has real Metal storage on macOS plus f32 2D kernels for same-shape
  elementwise/scalar, unary math
  (`abs/square/sqrt/exp/log/exp2/expm1/log1p/log2/log10/sin/cos/tan`), matmul/matmulAdd,
  transpose, broadcast-add, reductions, softmax, and logSoftmax, plus f16
  same-shape elementwise/scalar/abs/square/sqrt/exp, 2D matmul, transpose,
  broadcast-add, sum/prod/min/max reductions, softmax, and logSoftmax; remaining
  MPS dtypes/shapes still need parity.
- NumPy/PyTorch API surface breadth is large; parity must be measured by
  descriptor/lowering coverage, runtime capability, and correctness/performance
  gates rather than by counting isolated method names.
