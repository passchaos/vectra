# Vectra / Forge / Axiom API Boundary

This repository intentionally keeps Vectra focused on numerical arrays while the
sibling projects own higher and lower layers of the stack.

## Ownership

- **Vectra owns Array**: `Array(T)`, `NDArray(T)`, `ArrayView(T)`, dtype/device
  metadata, broadcasting, reductions, linalg/stat helpers, and explicit native
  CPU/CUDA dispatch through Axiom.
- **Forge owns Tensor**: a future Forge `Tensor` should wrap Vectra `Array`
  storage and own autograd, parameters, modules, losses, optimizers, training
  loops, inference ergonomics, graph capture, and in-place policy decisions.
- **Axiom owns backend and kernel lowering**: linalg/memref/gpu dialects,
  CPU/CUDA/MPS/native runtime dispatch contracts, concrete compiler IR, runtime
  launch, autotune/cache, and compiler evidence gates.
- **Vectra lowers array operations through Axiom dialects**: Vectra may expose
  Array conveniences such as `matmul`, but the compiler/runtime route starts in
  Axiom's MLIR-like linalg/memref/gpu records instead of a Vectra-owned kernel
  DSL.

## Guardrails

- Do not add a `Tensor` alias to Vectra.
- Do not add autograd to Vectra.
- Do not add `Parameter`, `Module`, `Optimizer`, `requires_grad`, `backward`, or
  training-loop semantics to Vectra.
- Do not place CUTILE/CuteDSL/SIMT kernel DSL APIs in Forge or Vectra; route
  compiler/runtime work through Axiom dialect records and backend passes.
- It is OK for Vectra to expose Array methods that are convenient for numerical
  computing, such as `a.matmul(b)` or `a.matmulAdd(b, c)`, as long as they remain
  Array operations without autograd ownership.

## Forge interop metadata

Vectra exposes a small data-only boundary manifest from `src/forge_interop.zig`
through `vx.forge_interop`, `vx.ForgeInteropBoundary`,
`vx.forgeArrayInteropBoundary`, and `vx.forgeInteropBoundary()`. The manifest is
made only of string and boolean literals, states that Vectra has no Forge
dependency, and identifies Vectra's role as the Array/NDArray data interop layer.

Forge may inspect or mirror this metadata when wrapping Vectra `Array` values,
but core Forge operation lowering should remain in Forge's graph pipeline and
flow through `Forge IR -> Axiom dialect/runtime`. Do not route Forge op semantics through
Vectra merely because an Array value is used for storage or data exchange.

## Intended dependency direction

```text
Axiom backend/compiler/runtime  <-  Vectra Array  <-  Forge Tensor/autograd
```

The Axiom boundary starts with linalg/memref/gpu dialect records and lowers from
there into CPU/CUDA/MPS/native backend contracts.  Vectra should not depend on an
intermediate compute-facade package for array execution.
