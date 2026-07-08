# Vectra

Vectra is a Zig 0.16 experimental data processing and numerical computing library.
It aims for a familiar Python-like surface inspired by NumPy/CuPy/SciPy/Pandas/Polars,
while leaning toward PyTorch-style fluent array methods for common operations. Vectra intentionally uses `Array`/`NDArray` as the primary user-facing name; automatic differentiation, training, and inference belong in the sibling `../forge` deep-learning framework.

> Status: early scaffold with a real, tested CPU core. The full NumPy/SciPy/Pandas
> ecosystem is enormous; this repository starts with a coherent architecture and
> useful primitives that can be expanded backend-by-backend and module-by-module.

## What is included now

- `Array(T)` / `NDArray(T)` with shape/strides, typed storage (`bool`, `i8/i16/i32/i64/isize`, `u8/u16/u32/u64/usize`, `BFloat16`, `f16/f32/f64`, `Complex64`/`Complex128`), dtype metadata (`canCastDType`, `promoteDType`, `resultDType`, `promoteType`), `reshape/view`, `flatten/ravel`, `squeeze/unsqueeze`, `permute/swapaxes/movedim`, `transpose`; `ArrayView(T)` / `NDArrayView(T)` provides non-owning strided views with shared storage, offsets, non-contiguous slicing, permutation, broadcasting, mutation, and `toArray/contiguous` materialization.
- Object-style construction on `Array(T)` / `NDArray(T)`: `fromSlice`, `fromScalar`, `zeros`, `ones`, `full`, `empty`, `emptyLike`, `zerosLike`, `onesLike`, `fullLike`, `eye`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid` with `MeshGridIndexing.xy/ij`, `rand`, `randn`, `uniform`, `normal`, `randint`, `bernoulli`, `permutation`, `shuffle/shuffleInPlace`, `choice`, `choiceWeighted`, `exponential`, `gamma`, `beta`, `poisson`, `multinomial`, `dirichlet`, `lognormal`, `studentT`, `cauchy`, `laplace`, `weibull`; random generation uses the local `../alea` backend.
- NumPy/PyTorch-like indexing helpers: `get/at`, `set/put`, scalar signed negative-index variants (`getSigned/atSigned`, `setSigned/putSigned`, `selectSigned`), `select`, `narrow`, `take/indexSelect`, batch signed negative-index variants (`takeSigned`, `indexSelectSigned`, `takeAlongAxisSigned`, `gatherSigned`, `putFlatSigned`, `putFlatScalarSigned`), `takeMode` with `IndexMode.raise/wrap/clip`, `takeAlongAxis/putAlongAxis`, coordinate indexing with prefix-shaped coordinate arrays (`ravelCoords`, `unravelFlat`, `takeCoords`, `putCoords/putCoordsScalar`) and broadcasted coordinate arrays (`ravelMultiIndex`, `takeMultiIndex`, `putMultiIndex/putMultiIndexScalar`), `gather`, `scatter/scatterScalar`, `scatterAdd`, `scatterReduce`, `scatterReduceScalar`, `scatterAddScalar`, `maskedSelect`, `maskedFill`, `maskedScatter`, `maskedPut/maskedPutScalar`, `putMask/putMaskScalar`, `copyWhere`, object-style masked in-place helpers (`maskedFillAssign`, `maskedCopyFrom`, `maskedCopyFromView`, `copyWhereAssign`, `copyWhereAssignView` plus `ArrayView.maskedFill/maskedCopyFrom*/copyWhereFrom*`), `whereIndices`, `putFlat/putFlatMode/putFlatScalar/putFlatScalarMode`, `indexPut/indexPutScalar`, `compress`, `flatNonzero`, `nonzero/argwhere/countNonzero`, `isin`, `slice1d`.
- Broadcasting elementwise arithmetic/comparisons: `add/sub/mul/div/pow`, promoted mixed-dtype variants (`addPromote`, `subPromote`, `mulPromote`, `divPromote`, `maximumPromote`, `minimumPromote`), complex helpers (`real`, `imag`, `conj/conjugate`, `magnitude/absComplex`, complex `exp/log/sqrt/sin/cos/tan/...` and complex finiteness predicates), `floorDiv`, `mod/remainder`, scalar variants, `maximum/minimum`, `hypot`, `atan2`, `copysign`, `heaviside`, comparisons (`eq/equal`, `ne/notEqual`, `gt/greater`, `ge/greaterEqual`, `lt/less`, `le/lessEqual`, scalar variants), boolean logic (`logicalNot`, `logicalAnd`, `logicalOr`, `logicalXor`, scalar variants), `whereMask`, `isclose`, `allclose`; object-style in-place helpers include `fill`, `copyFrom/copyFromView`, `copyFromArray`, `add/sub/mul/divAssign`, `add/sub/mul/divAssignView`, and scalar assignment variants on `Array`/`ArrayView`.
- Array transforms are object/type methods: `reshape/view`, `reshapeInfer/viewInfer` with one inferred dimension, `flatten/ravel`, `flattenAxes`, `unflatten`, `squeeze/unsqueeze`, `transpose`, `permute`, `swapaxes`, `movedim`, `broadcastTo`, `repeat`, `repeatInterleave/repeatInterleaveScalar`, `tile`, `slice/sliceAxis/slice1d`, `split`, `chunk`, `flip/flipAxes`, `roll/rollAxes`, `padConstant`, `padEdge`, `padReflect`, `padWrap`, `padSymmetric`, and type-level `cat/concatenate`, `stack`, `hstack`, `vstack`, `dstack`, `columnStack`; non-copying view helpers include `asView`, safe `asStrided`, PyTorch-like `unfold`, `sliceAxisView`, `sliceView`, `selectView`, `narrowView`, `permuteView`, `transposeView`, and `broadcastView`.
- Sorting/selection helpers: `sort`, `sortBy`, `sortDescending`, `argsort`, `argsortAxis`, `argsortDescending`, `sortWithIndices`, `partition`, `argpartition`, and `topk(sorted=true/false)`.
- Discrete/search/set helpers: `unique`, `uniqueWithCounts`, `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `bincount`, `bincountWeighted`, `searchsorted` with `SearchSide.left/right`, PyTorch-like `bucketize`, NumPy-like `digitize`, and broadcasted `clipArray`.
- Reductions/statistics as methods: `sum`, `prod`, `min`, `max`, `allAxis`, `anyAxis`, `mean`, `variance`, `stddev`, `median`, `quantile`, `percentile`, weighted stats (`weightedMean`, `average`, `weightedVariance/weightedVar`, `weightedStddev/weightedStd`, `weightedQuantile`, `weightedMedian`, `weightedCov`, `weightedCorrcoef`), `cov`, `corrcoef`, nan-aware stats (`nanToNum/nan_to_num`, `nansum`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax`, `nanmedian`, `nanquantile`, `nanpercentile`, `nanCov`, `nanCorrcoef`), `norm`, `logsumexp`, `cumsum`, `cumprod`, `cumsumAxis`, `cumprodAxis`, `diff`, `argmin`, `argmax`, `argminAxis/argmaxAxis`, `histogram`.
- `ArrayView(T)` can now directly call object-style materializing wrappers for `softmax/logSoftmax`, `norm`, `sort/sortBy/sortDescending`, `argsort/argsortAxis`, `topk`, `matmul/matmulArray`, `dot`, `outer`, `trace`, and complex helpers (`real`, `imag`, `conj`, `magnitude`) while preserving non-copying indexing/mutation for view operations.
- `ArrayView(T)` also exposes common materializing elementwise wrappers: `square`, `reciprocal`, `sign/signbit`, `exp/expm1`, `log/log1p/log2/log10`, `sqrt`, `floor/ceil/round/trunc`, `deg2rad/rad2deg`, trigonometric and hyperbolic functions, `relu`, `sigmoid`, `clip`, `isNan/isInf/isFinite`, `isclose`, and `allclose`.
- `ArrayView(T)` statistics wrappers include `variance/stddev`, `median/quantile/percentile`, `average`, nan-aware reductions (`nansum/nanmean/nanvar/nanstd/nanmin/nanmax/nanmedian/nanquantile/nanpercentile`), cumulative operations (`cumsum/cumprod/cumsumAxis/cumprodAxis/diff`), and arg reductions (`argmin/argmax/argminAxis/argmaxAxis`).
- `ArrayView(T)` also exposes materializing indexing/search/shape wrappers such as `repeat/repeatInterleave/repeatInterleaveScalar`, `tile`, `flip/flipAxes`, `roll/rollAxes`, `padConstant/padEdge/padReflect/padWrap/padSymmetric`, `slice1d`, `split/chunk`, `take/takeSigned`, `indexSelect/indexSelectSigned`, `gather/gatherSigned`, `maskedSelect`, `compress`, `nonzero/argwhere/whereIndices`, `unique`, `searchsorted`, `bucketize`, `digitize`, and `isin`.
- Neural/math functions: `neg`, `abs`, `square`, `reciprocal`, `sign/signbit`, `nextAfter/nextafter`, `ldexp`, `frexp`, `exp/expm1`, `log/log1p/log2/log10`, `sqrt`, `floor`, `ceil`, `round`, `trunc`, `deg2rad/rad2deg`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `hypot`, `copysign`, `heaviside`, `sinh`, `cosh`, `tanh`, `relu`, `sigmoid`, `softmax`, `logsumexp`, `logSoftmax/log_softmax`, `clip/clamp`, `isNan/isnan`, `isInf/isinf`, `isFinite/isfinite`.
- Linear algebra, signal basics, and contractions: `diag/diagflat`, `diagonal`, `trace`, `triu/tril`, generalized `matmul/mm` with vector/matrix/batched broadcasting semantics, `bmm`, `matvec`, `dot`, `inner`, `vecdot`, `vdot`, `outer`, `cross`, `contractAxes`, 1D/2D `convolve*`/`correlate*` with `ConvMode.full/same/valid`, real `rfft/irfft`, complex `fft/ifft`, `fftAxis/ifftAxis`, `fftAxes/ifftAxes`, `fft2/ifft2`, `linalg.eye`, `det`, `inverse`, `solve`, `lu`, `solveTriangular`, `cholesky`, `qr`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`; f64 `linalg.matmul`/`matvec`/`trace`/`det`/`solve`/`inverse`/`lu`/`solveTriangular`/`cholesky`/`qr`/`svd`/`lstsq`/`singularValues`/`matrixRank`/`cond`/`pinv`/`matrixNorm`/`eigh`/`eigvalsh` use the local `../veyra` backend when available, while Array object methods keep generic in-core fallbacks.
- SciPy-like stats helpers: `stats.zscore`, `normalize`, `pearsonr`.
- Sparse CSR/CSC bridge: `CsrMatrix`, `CscMatrix`, `csrFromDense`, `csrFromCompressed`, `cscFromDense`, `cscFromCompressed`, CSR-to-dense, transpose, transpose products, row/column stats, diagonal/trace diagnostics, bandwidth, symmetry checks, triangular solve, and Veyra-backed f64 CSR kernels.
- `Series(T)` and heterogeneous `DataFrame` with select/filter/sort/head/tail/describe/group-by-sum.
- CSV read/write with simple type inference.
- Array IO helpers: `toBytes/fromBytes` for raw data, `toArchive/fromArchive` for a simple dtype+shape binary archive, and object-style file helpers `saveArchive/saveArchiveToDir` plus `loadArchive/loadArchiveFromDir`.
- Device API placeholder (`Device.cpu`, `Device.cuda(index)`) for future CuPy/PyTorch-like GPU backends.

## Example

```zig
const std = @import("std");
const vx = @import("vectra");

pub fn demo(allocator: std.mem.Allocator) !void {
    var a = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var bias = try vx.Array(f64).ones(allocator, &.{3});
    defer bias.deinit();

    var y = try a.add(bias);      // NumPy/PyTorch-like broadcasting
    defer y.deinit();

    var probs = try y.softmax(1); // PyTorch-like method API
    defer probs.deinit();

    var picked_idx = try vx.Array(usize).fromSlice(allocator, &.{ 2, 0 }, &.{2});
    defer picked_idx.deinit();
    var picked = try y.indexSelect(1, picked_idx); // torch.index_select / np.take style
    defer picked.deinit();

    var labels = try vx.Array(i32).fromSlice(allocator, &.{ 2, 1, 2, 3 }, &.{4});
    defer labels.deinit();
    var counts = try labels.bincount(5);
    defer counts.deinit();

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

Vectra uses the sibling [`../alea`](../alea) Zig package as a local path dependency for random generation and distributions. Current array random helpers delegate seeded scalar random streams to Alea for uniform, normal, integer range, Bernoulli, exponential, gamma, beta, Poisson, multinomial, Dirichlet, log-normal, Student-t, Cauchy, Laplace, and Weibull generation. Future random distributions should prefer Alea rather than reimplementing RNG kernels inside Vectra.

## Veyra backend

Vectra uses the sibling [`../veyra`](../veyra) Zig package as a local path dependency for foundational math and linear algebra. Current f64 `linalg` paths delegate matrix multiplication, matrix-vector products, trace, determinant, solve, inverse, LU, triangular solve, Cholesky, QR, SVD, least-squares, rank/condition helpers, pseudo-inverse, matrix norms, and symmetric eigen decomposition to Veyra-compatible dense matrix APIs while Array methods keep dependency-free generic fallbacks. Future SciPy-like and high-performance BLAS/LAPACK-style work should prefer Veyra where it already provides tested kernels or decompositions.

## Development priorities

Future work should follow the documented order in [`docs/DEVELOPMENT_GUIDE.md`](docs/DEVELOPMENT_GUIDE.md):

1. NumPy/CuPy/PyTorch-like Array/NDArray functionality first.
2. SciPy-like numerical algorithms second.
3. Pandas/Polars-like DataFrame and lazy tabular processing third.

## Development

```sh
zig build test
zig build run
```

The package targets Zig `0.16.0` and uses the new `std.Io` writer APIs.

## Roadmap

- Broader view-aware kernels on top of the current `ArrayView`/`NDArrayView` non-contiguous storage model.
- Nullable values, categorical/string kernels and richer promotion policy.
- Polars-like lazy query plans and expression DSL.
- BLAS/LAPACK/high-performance FFT/sparse integrations.
- GPU backend implementation behind the existing `Device` surface.
- Arrow/Parquet IPC support.
