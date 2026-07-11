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
- **Axiom owns backend and kernel lowering**: CPU/CUDA/native runtime dispatch,
  fusion, CUTILE/CuteDSL/CUDA SIMT abstractions, kernel IR, lowering, runtime
  launch, autotune/cache, and compiler evidence gates.

## Guardrails

- Do not add a `Tensor` alias to Vectra.
- Do not add autograd to Vectra.
- Do not add `Parameter`, `Module`, `Optimizer`, `requires_grad`, `backward`, or
  training-loop semantics to Vectra.
- Do not place CUTILE/CuteDSL/SIMT kernel DSL APIs in Forge or Vectra; keep them
  in Axiom until they are stable enough to become a standalone lower-level
  dependency consumed by Axiom.
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
flow through `Forge IR -> Axial/Axiom`. Do not route Forge op semantics through
Vectra merely because an Array value is used for storage or data exchange.

## Intended dependency direction

```text
Axiom backend/compiler/runtime  <-  Vectra Array  <-  Forge Tensor/autograd
```

If a future standalone CUTILE-style package is introduced, it should sit below
Axiom rather than above Vectra or Forge:

```text
cutile-like kernel DSL  <-  Axiom  <-  Vectra  <-  Forge
```
