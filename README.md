# Vectra

Vectra is a Zig 0.16 experimental data processing and numerical computing library.
It aims for a familiar Python-like surface inspired by NumPy/CuPy/SciPy/Pandas/Polars,
while leaning toward PyTorch-style fluent array methods for common operations. Vectra intentionally uses `Array`/`NDArray` as the primary user-facing name; automatic differentiation, training, and inference belong in the sibling `../forge` deep-learning framework.

> Status: early scaffold with a real, tested CPU core. The full NumPy/SciPy/Pandas
> ecosystem is enormous; this repository starts with a coherent architecture and
> useful primitives that can be expanded backend-by-backend and module-by-module.

## What is included now

- `Array(T)` / `NDArray(T)` with shape/strides, typed storage, `reshape/view`, `flatten/ravel`, `squeeze/unsqueeze`, `permute/swapaxes/movedim`, `transpose`.
- Creation helpers: `array`, `ndarray`, `zeros`, `ones`, `full`, `empty`, `eye`, `arange`, `linspace`, `rand`, `randn`, `uniform`, `normal`, `randint`, `bernoulli`; random generation uses the local `../alea` backend.
- NumPy/PyTorch-like indexing helpers: `get/at`, `set/put`, `select`, `narrow`, `take/indexSelect`, `gather`, `scatter/scatterScalar`, `maskedSelect`, `slice1d`.
- Broadcasting elementwise arithmetic/comparisons: `add/sub/mul/div/pow`, scalar variants, `maximum/minimum`, `whereMask`, `allclose`.
- Array transforms: `broadcastTo`, `repeat`, `tile`, `cat/concatenate`, `stack`, `sort`, `argsort`.
- Reductions/statistics: `sum`, `prod`, `min`, `max`, `allAxis`, `anyAxis`, `mean`, `variance`, `stddev`, `norm`, `logsumexp`, `cumsum`, `cumprod`, `argmin`, `argmax`, `argminAxis/argmaxAxis`, `topk`, `histogram`.
- Neural/math functions: `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `relu`, `sigmoid`, `softmax`, `logSoftmax/log_softmax`, `clip`.
- Linear algebra: `matmul/mm`, `bmm`, `matvec`, `dot`, `outer`, `diagonal`, `trace`, `triu/tril`, `linalg.eye`, `det`, `inverse`, `solve`, `lu`, `solveTriangular`, `cholesky`, `qr`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`; f64 `linalg.matmul`/`matvec`/`trace`/`det`/`solve`/`inverse`/`lu`/`solveTriangular`/`cholesky`/`qr`/`svd`/`lstsq`/`singularValues`/`matrixRank`/`cond`/`pinv`/`matrixNorm`/`eigh`/`eigvalsh` use the local `../veyra` backend when available.
- SciPy-like stats helpers: `stats.zscore`, `normalize`, `pearsonr`.
- `Series(T)` and heterogeneous `DataFrame` with select/filter/sort/head/tail/describe/group-by-sum.
- CSV read/write with simple type inference.
- Device API placeholder (`Device.cpu`, `Device.cuda(index)`) for future CuPy/PyTorch-like GPU backends.

## Example

```zig
const std = @import("std");
const vx = @import("vectra");

pub fn demo(allocator: std.mem.Allocator) !void {
    var a = try vx.array(f64, allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var bias = try vx.ones(f64, allocator, &.{3});
    defer bias.deinit();

    var y = try a.add(bias);      // NumPy/PyTorch-like broadcasting
    defer y.deinit();

    var probs = try y.softmax(1); // PyTorch-like method API
    defer probs.deinit();

    var picked_idx = try vx.array(usize, allocator, &.{ 2, 0 }, &.{2});
    defer picked_idx.deinit();
    var picked = try y.indexSelect(1, picked_idx); // torch.index_select / np.take style
    defer picked.deinit();

    var df = try vx.DataFrame.init(allocator, &.{
        .{ .name = "city", .data = .{ .string = &.{ "hz", "bj", "hz" } } },
        .{ .name = "sales", .data = .{ .f64 = &.{ 2.0, 3.0, 5.0 } } },
    });
    defer df.deinit();

    var grouped = try df.groupBySum("city", "sales");
    defer grouped.deinit();
}
```

## Alea backend

Vectra uses the sibling [`../alea`](../alea) Zig package as a local path dependency for random generation and distributions. Current array random helpers delegate seeded scalar random streams to Alea for uniform, normal, integer range, and Bernoulli generation. Future random distributions should prefer Alea rather than reimplementing RNG kernels inside Vectra.

## Veyra backend

Vectra uses the sibling [`../veyra`](../veyra) Zig package as a local path dependency for foundational math and linear algebra. Current f64 `linalg` paths delegate matrix multiplication, matrix-vector products, trace, determinant, solve, inverse, LU, triangular solve, Cholesky, QR, SVD, least-squares, rank/condition helpers, pseudo-inverse, matrix norms, and symmetric eigen decomposition to Veyra-compatible dense matrix APIs while Tensor methods keep dependency-free generic fallbacks. Future SciPy-like and high-performance BLAS/LAPACK-style work should prefer Veyra where it already provides tested kernels or decompositions.

## Development priorities

Future work should follow the documented order in [`docs/DEVELOPMENT_GUIDE.md`](docs/DEVELOPMENT_GUIDE.md):

1. NumPy/CuPy/PyTorch-like array and tensor functionality first.
2. SciPy-like numerical algorithms second.
3. Pandas/Polars-like DataFrame and lazy tabular processing third.

## Development

```sh
zig build test
zig build run
```

The package targets Zig `0.16.0` and uses the new `std.Io` writer APIs.

## Roadmap

- Richer indexing/views and non-contiguous tensors.
- More dtypes, nullable values, categorical/string kernels.
- Polars-like lazy query plans and expression DSL.
- BLAS/LAPACK/FFT/sparse integrations.
- GPU backend implementation behind the existing `Device` surface.
- Arrow/Parquet IPC support.
