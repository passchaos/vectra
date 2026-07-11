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

## Intended dependency direction

```text
Axiom backend/compiler/runtime  <-  Vectra Array  <-  Forge Tensor/autograd
```

If a future standalone CUTILE-style package is introduced, it should sit below
Axiom rather than above Vectra or Forge:

```text
cutile-like kernel DSL  <-  Axiom  <-  Vectra  <-  Forge
```
