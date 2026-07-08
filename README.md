# Vectra

Vectra is a Zig 0.16 experimental data processing and numerical computing library.
It aims for a familiar Python-like surface inspired by NumPy/CuPy/SciPy/Pandas/Polars,
while leaning toward PyTorch-style fluent tensor methods for common operations.

> Status: early scaffold with a real, tested CPU core. The full NumPy/SciPy/Pandas
> ecosystem is enormous; this repository starts with a coherent architecture and
> useful primitives that can be expanded backend-by-backend and module-by-module.

## What is included now

- `Tensor(T)` with shape/strides, typed storage, `reshape/view`, `flatten/ravel`, `squeeze/unsqueeze`, `permute/swapaxes/movedim`, `transpose`.
- Creation helpers: `tensor`, `zeros`, `ones`, `full`, `empty`, `eye`, `arange`, `linspace`, `rand`, `randn`, `randint`.
- NumPy/PyTorch-like indexing helpers: `get/at`, `set/put`, `select`, `narrow`, `take/indexSelect`, `maskedSelect`, `slice1d`.
- Broadcasting elementwise arithmetic/comparisons: `add/sub/mul/div/pow`, scalar variants, `maximum/minimum`, `whereMask`, `allclose`.
- Array transforms: `broadcastTo`, `repeat`, `tile`, `cat/concatenate`, `stack`, `sort`, `argsort`.
- Reductions/statistics: `sum`, `prod`, `min`, `max`, `mean`, `variance`, `stddev`, `cumsum`, `cumprod`, `argmin`, `argmax`, `histogram`.
- Neural/math functions: `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `relu`, `sigmoid`, `softmax`, `clip`.
- Linear algebra: `matmul/mm`, `bmm`, `dot`, `linalg.eye`, `trace`, `det`, `inverse`, `solve`.
- SciPy-like stats helpers: `stats.zscore`, `normalize`, `pearsonr`.
- `Series(T)` and heterogeneous `DataFrame` with select/filter/sort/head/tail/describe/group-by-sum.
- CSV read/write with simple type inference.
- Device API placeholder (`Device.cpu`, `Device.cuda(index)`) for future CuPy/PyTorch-like GPU backends.

## Example

```zig
const std = @import("std");
const vx = @import("vectra");

pub fn demo(allocator: std.mem.Allocator) !void {
    var a = try vx.tensor(f64, allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var bias = try vx.ones(f64, allocator, &.{3});
    defer bias.deinit();

    var y = try a.add(bias);      // NumPy/PyTorch-like broadcasting
    defer y.deinit();

    var probs = try y.softmax(1); // PyTorch-like method API
    defer probs.deinit();

    var picked_idx = try vx.tensor(usize, allocator, &.{ 2, 0 }, &.{2});
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
