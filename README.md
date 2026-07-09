# Vectra

Vectra is a Zig 0.16 experimental data processing and numerical computing library.
It aims for a familiar Python-like surface inspired by NumPy/CuPy/SciPy/Pandas/Polars,
while leaning toward PyTorch-style fluent array methods for common operations. Vectra intentionally uses `Array`/`NDArray` as the primary user-facing name; automatic differentiation, training, and inference belong in the sibling `../forge` deep-learning framework.

> Status: early scaffold with a real, tested CPU core. The full NumPy/SciPy/Pandas
> ecosystem is enormous; this repository starts with a coherent architecture and
> useful primitives that can be expanded backend-by-backend and module-by-module.

## What is included now

- `Array(T)` / `NDArray(T)` with shape/strides and metadata helpers (`ndim/dim/rank/numDims`, `numel/nelement`, `size/shapeAt`, `stride/strideAt`, `nbytes`, `elementSize`, `isEmpty`, `isContiguous/is_contiguous`), deep-copy helpers (`clone/copy/detach`), typed storage (`bool`, `i8/i16/i32/i64/isize`, `u8/u16/u32/u64/usize`, `BFloat16`, `f16/f32/f64`, `Complex64`/`Complex128`), dtype metadata (`canCastDType`, `promoteDType`, `resultDType`, `promoteType`), `reshape/view`, `flatten/ravel`, `squeeze/unsqueeze`, `permute/swapaxes/movedim`, `transpose`; `ArrayView(T)` / `NDArrayView(T)` provides non-owning strided views with shared storage, offsets, non-contiguous slicing, permutation, broadcasting, mutation, and `toArray/copy/detach/contiguous` materialization.
- Object-style construction on `Array(T)` / `NDArray(T)`: `fromSlice`, `fromScalar`, `zeros`, `ones`, `full`, `empty`, `emptyLike`, `zerosLike`, `onesLike`, `fullLike`, PyTorch-like `newEmpty/newZeros/newOnes/newFull` aliases, `eye/identity/eyeRect`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid` with `MeshGridIndexing.xy/ij`, `rand`, `randn`, `uniform`, `normal`, `randint`, `bernoulli`, `permutation`, `shuffle/shuffleInPlace`, `choice`, `choiceWeighted`, `exponential`, `gamma`, `beta`, `poisson`, `multinomial`, `dirichlet`, `lognormal`, `studentT`, `cauchy`, `laplace`, `weibull`, plus Alea-backed continuous distributions such as `halfNormal`, `chiSquared/chi`, `erlang`, `fisherF`, `triangular`, `arcsine`, `logistic`, `logLogistic`, `kumaraswamy`, `powerFunction`, `rayleigh`, `maxwell`, `pareto`, `gumbel`, `frechet`, `skewNormal`, `pert`, `inverseGaussian`, and `normalInverseGaussian`; random generation uses the local `../alea` backend.
- NumPy/PyTorch-like indexing helpers: `get/at`, `set/put`, scalar signed negative-index variants (`getSigned/get_signed/atSigned/at_signed`, `setSigned/set_signed/putSigned/put_signed`, `selectSigned/select_signed`), `select`, `narrow/narrowSigned`, `take/indexSelect`, batch signed negative-index variants (`takeSigned/take_signed/takeSignedMode/take_signed_mode`, `indexSelectSigned/index_select_signed`, `takeAlongAxisSigned/take_along_axis_signed`, `gatherSigned/gather_signed`, `putFlatSigned`, `putFlatScalarSigned`), `takeMode/takeSignedMode` with `IndexMode.raise/wrap/clip`, `takeAlongAxis/putAlongAxis`, coordinate indexing with prefix-shaped coordinate arrays (`ravelCoords`, `unravelFlat`, `takeCoords`, `putCoords/putCoordsScalar`) and broadcasted coordinate arrays (`ravelMultiIndex`, `takeMultiIndex`, `putMultiIndex/putMultiIndexScalar`), `gather`, `scatter/scatterScalar`, `scatterAdd`, `scatterReduce`, `scatterReduceScalar`, `scatterAddScalar`, `maskedSelect/masked_select`, `maskedFill/masked_fill`, `maskedScatter/masked_scatter`, `maskedPut/masked_put/maskedPutScalar/masked_put_scalar`, `putMask/put_mask/putMaskScalar/put_mask_scalar`, `copyWhere/copy_where/where/whereScalar/where_scalar`, object-style masked in-place helpers (`maskedFillAssign`, `maskedCopyFrom`, `maskedCopyFromView`, `copyWhereAssign`, `copyWhereAssignView` plus `ArrayView.maskedFill/maskedCopyFrom*/copyWhereFrom*`), `whereIndices`, `putFlat/put_flat/putFlatMode/putFlatScalar/put_flat_scalar/putFlatScalarMode`, `indexPut/index_put/indexPutScalar/index_put_scalar`, `compress`, `flatNonzero`, `nonzero/argwhere/countNonzero/countNonzeroAxis/count_nonzero`, `isin`, `slice1d`.
- Broadcasting elementwise arithmetic/comparisons: `add/sub/mul/div/pow`, promoted mixed-dtype variants (`addPromote`, `subPromote`, `mulPromote`, `divPromote`, `maximumPromote`, `minimumPromote`), complex helpers (`real`, `imag`, `conj/conjugate`, `magnitude/absComplex`, `angle/phase`, `isreal/iscomplex`, complex `exp/log/sqrt/sin/cos/tan/...` and complex finiteness predicates), `floorDiv`, `mod/remainder`, scalar variants, `maximum/minimum`, `fmax/fmin`, `hypot`, `atan2`, `logAddExp/logaddexp`, `logAddExp2/logaddexp2`, `xlogy`, `copysign`, `heaviside`, comparisons (`eq/equal`, `ne/notEqual`, `gt/greater`, `ge/greaterEqual`, `lt/less`, `le/lessEqual`, scalar variants including `equalScalar/greaterScalar/lessScalar/...`), boolean logic (`logicalNot`, `logicalAnd`, `logicalOr`, `logicalXor`, scalar variants including `equalScalar/greaterScalar/lessScalar/...`), `where/whereScalar/whereMask`, `isclose/isClose/iscloseScalar/isCloseScalar` with equal-NaN variants, `allclose/allClose/allcloseScalar/allCloseScalar` with equal-NaN variants; object-style in-place helpers include `fill`, `copyFrom/copyFromView`, `copyFromArray`, `add/sub/mul/divAssign`, `add/sub/mul/divAssignView`, and scalar assignment variants on `Array`/`ArrayView`.
- Array transforms are object/type methods: `reshape/view`, `reshapeInfer/viewInfer` with one inferred dimension, `reshapeAs/viewAs`, `flatten/ravel`, `flattenAxes/flattenRange/flattenFrom`, `unflatten`, `atLeast1d/atLeast2d/atLeast3d`, `squeeze/unsqueeze` plus `squeezeDim/squeezeAxes/unsqueezeDim/unsqueezeAxes/expandDims`, `transpose`, `permute`, `swapaxes/swapDims`, `movedim/moveaxis/moveaxes`, `broadcastTo/expand/expandAs/expand_as`, `repeat`, `repeatInterleave/repeatInterleaveScalar`, NumPy-like `tile` with leading-rank alignment, `slice/sliceAxis/slice1d`, `split`, `splitWithSizes/splitAtIndices`, `chunk`, `unbind`, `flip/flipAxes/fliplr/flipud`, `roll/rollFlat/rollAxes`, `rot90`, `padConstant`, `padEdge`, `padReflect`, `padWrap`, `padSymmetric`, and type-level `cat/concatenate`, `stack`, `hstack`, `vstack`, `dstack`, `columnStack`; non-copying view helpers include `asView`, safe `asStrided`, PyTorch-like `unfold`, `sliceAxisView`, `sliceView`, `selectView`, `narrowView`, `narrowSigned`, `permuteView`, `swapaxesView`, `movedimView`, `transposeView/T_`, `view/viewInfer/viewAs`, and `broadcastView/expandView`, `atLeast1d/atLeast2d/atLeast3d`.
- Sorting/selection helpers: `sort`, `sortBy`, `sortDescending`, `argsort`, `argsortAxis`, `argsortDescending`, `sortWithIndices`, `partition`, `argpartition`, `kthValue/kth_value`, and `topk(sorted=true/false)`.
- Discrete/search/set helpers: `unique`, `uniqueWithCounts`, `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `bincount`, `bincountWeighted`, `searchsorted` with `SearchSide.left/right`, PyTorch-like `bucketize`, NumPy-like `digitize`, and broadcasted `clipArray`.
- Reductions/statistics as methods: `sum/sumAxes/sumToSize`, `prod/prodAxes`, `min/amin/minAxes/aminAxes`, `max/amax/maxAxes/amaxAxes`, `ptp/ptpAxes`, `allAxis/allAxes`, `anyAxis/anyAxes`, `mean/meanAxes`, `variance/varianceAxes`, `stddev/stddevAxes`, `median/medianAxes`, `quantile/quantileAxes`, `percentile/percentileAxes`, weighted stats (`weightedMean`, `average`, `weightedVariance/weightedVar`, `weightedStddev/weightedStd`, `weightedQuantile`, `weightedMedian`, `weightedCov`, `weightedCorrcoef`), `cov`, `corrcoef`, nan-aware stats (`nanToNum/nan_to_num/nanToNumDefault/nan_to_num_default`, `nansum/nansumAxes`, `nanmean/nanmeanAxes`, `nanvar/nanvarAxes`, `nanstd/nanstdAxes`, `nanmin/nanminAxes`, `nanmax/nanmaxAxes`, `nanmedian/nanmedianAxes`, `nanquantile/nanquantileAxes`, `nanpercentile/nanpercentileAxes`, `nanCov`, `nanCorrcoef`), `norm/normAxes`, `logsumexp/logsumexpAxes`, `logcumsumexp`, `cumsum`, `cumprod`, `cummax`, `cummin`, `cumsumAxis`, `cumprodAxis`, `cummaxAxis`, `cumminAxis`, `diff/diffWith/ediff1d`, `gradient`, `trapezoid/trapz`, `argmin`, `argmax`, `argminAxis/argmaxAxis`, `nanargmin/nanargmax`, `nanargminAxis/nanargmaxAxis`, `histogram`.
- `ArrayView(T)` can now directly call object-style materializing wrappers for `softmax/logSoftmax`, `norm`, `sort/sortBy/sortDescending`, `argsort/argsortAxis`, `topk`, `matmul/matmulArray`, `bmm`, `matvec`, `dot`, `vdot/vecdot`, `inner`, `outer`, `cross`, `contractAxes`, `convolve1d/correlate1d/convolve2d/correlate2d`, `trace/traceOffset`, `diagonal/diag/diagflat`, `triu/tril`, object-style linalg wrappers (`det`, `inverse/inv`, `solve`, `cholesky`, `qr`, `lu`, `solveTriangular`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`), and complex helpers (`real`, `imag`, `conj`, `magnitude`, `angle/phase`, `isreal/iscomplex`) while preserving non-copying indexing/mutation for view operations.
- `ArrayView(T)` also exposes common materializing dtype/elementwise/broadcast wrappers: `astype`, promoted mixed-dtype variants (`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`), `neg/negative`, `positive`, `abs/absolute/fabs`, `square`, `reciprocal`, `sign/signbit`, `exp/exp2/expm1`, `log/log1p/log2/log10/lgamma/gammaln`, `sqrt/rsqrt/cbrt`, `floor/ceil/round/trunc`, `deg2rad/radians` and `rad2deg/degrees`, `sinc`, trigonometric and inverse/hyperbolic aliases including `arcsin/arccos/arctan/arctan2` and `asinh/acosh/atanh` plus `arcsinh/arccosh/arctanh`, `relu/leakyRelu`, `sigmoid/expit`, `logit`, `softplus`, `softsign`, `gelu`, `clip`, `clipMin/clipMax/clampMin/clampMax`, `pow/floorDiv/mod/remainder`, `maximum/minimum`, `fmax/fmin`, `hypot/atan2`, `logAddExp/logAddExp2/xlogy`, `lerp`, `addcmul/addcdiv`, `nextAfter/copysign/heaviside`, advanced scalar variants (`powScalar/floorDivScalar/modScalar/maximumScalar/...`), `ldexpScalar`, `clipArray`, `isNan/isInf/isFinite/isNormal`, `isclose/isClose/iscloseScalar/isCloseScalar` with equal-NaN variants, `allclose/allClose/allcloseScalar/allCloseScalar` with equal-NaN variants, comparison aliases (`equal/greater/less`), and bool `logicalNot/logicalAnd/logicalOr/logicalXor` wrappers.
- `ArrayView(T)` statistics wrappers include `min/amin`, `max/amax`, `ptp`, `sumAxes/prodAxes/minAxes/aminAxes/maxAxes/amaxAxes/ptpAxes/meanAxes/varianceAxes/stddevAxes`, `variance/stddev`, `median/medianAxes/quantile/quantileAxes/percentile/percentileAxes`, `average`, weighted statistics (`weightedMean/weightedVariance/weightedVar/weightedStddev/weightedStd/weightedQuantile/weightedMedian`), nan-aware reductions (`nanToNum/nan_to_num/nanToNumDefault/nan_to_num_default`, `nansum/nansumAxes/nanmean/nanmeanAxes/nanvar/nanvarAxes/nanstd/nanstdAxes/nanmin/nanminAxes/nanmax/nanmaxAxes/nanmedian/nanmedianAxes/nanquantile/nanquantileAxes/nanpercentile/nanpercentileAxes`), `logsumexp`, covariance/correlation wrappers (`cov/corrcoef/weightedCov/weightedCorrcoef/nanCov/nanCorrcoef`), cumulative/integration operations (`cumsum/cumprod/cummax/cummin/logcumsumexp/cumsumAxis/cumprodAxis/cummaxAxis/cumminAxis/logcumsumexpAxis/diff/gradient/trapezoid/trapz`), and arg reductions (`argmin/argmax/argminAxis/argmaxAxis/nanargmin/nanargmax/nanargminAxis/nanargmaxAxis`) plus bool `all/any/allAxis/anyAxis`.
- `ArrayView(T)` also exposes metadata and like/new allocation helpers plus materializing indexing/search/shape wrappers such as `repeat/repeatInterleave/repeatInterleaveScalar`, NumPy-like `tile`, `flip/flipAxes/fliplr/flipud`, `roll/rollFlat/rollAxes`, `rot90`, `padConstant/padEdge/padReflect/padWrap/padSymmetric`, `slice1d`, `split/splitWithSizes/splitAtIndices/chunk/unbind`, `take/takeSigned`, `takeAlongAxis/takeAlongAxisSigned/putAlongAxis`, `indexSelect/indexSelectSigned`, `gather/gatherSigned`, `putFlat/putFlatMode/putFlatScalar*`, `indexPut/indexPutScalar`, coordinate and multi-index helpers, `scatter/scatterScalar/scatterAdd/scatterReduce*`, `maskedSelect`, `where/whereScalar`, `compress`, `nonzero/argwhere/whereIndices/countNonzeroAxis/countNonzeroAxes/count_nonzero`, `unique/uniqueWithCounts`, `union1d/intersect1d/setdiff1d/setxor1d`, `bincount/bincountWeighted`, `histogram`, `searchsorted`, `bucketize`, `digitize`, and `isin`.
- Neural/math functions: `neg/negative`, `positive`, `abs/absolute/fabs`, `astype`, promoted mixed-dtype variants (`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`), `square`, `reciprocal`, `sign/signbit`, `nextAfter/nextafter`, `ldexp`, `frexp`, `exp/exp2/expm1`, `log/log1p/log2/log10/lgamma/gammaln`, `sqrt/rsqrt/cbrt`, `floor`, `ceil`, `round`, `trunc`, `deg2rad/radians`, `rad2deg/degrees`, `sinc`, `sin`, `cos`, `tan`, `asin/arcsin`, `acos/arccos`, `atan/arctan`, `atan2/arctan2`, `hypot`, `copysign`, `heaviside`, `sinh`, `cosh`, `tanh`, `asinh/arcsinh`, `acosh/arccosh`, `atanh/arctanh`, `relu/leakyRelu`, `sigmoid/expit`, `logit`, `softplus`, `softsign`, `gelu`, `softmax`, `logsumexp`, `logcumsumexp`, `logSoftmax/log_softmax`, `clip/clamp`, `clipMin/clipMax/clampMin/clampMax`, `lerp`, `addcmul/addcdiv`, `fmax/fmin`, `isNan/isnan`, `isInf/isinf`、`isPosInf/isposinf`、`isNegInf/isneginf`, `isFinite/isfinite`, `isNormal/isnormal`.
- Linear algebra, signal basics, and contractions: `diag/diagflat`, `diagonal`, `trace/traceOffset`, `triu/tril`, object-style `det`, `inverse/inv`, `solve`, `cholesky`, `qr`, `lu`, `solveTriangular/solve_triangular`, `svd`, `lstsq`, `singularValues/singular_values`, `matrixRank/matrix_rank`, `cond`, `pinv`, `matrixNorm/matrix_norm`, `eigh`, `eigvalsh`, generalized `matmul/mm` with vector/matrix/batched broadcasting semantics, `bmm`, `matvec`, `dot`, `inner`, `vecdot`, `vdot`, `outer`, `cross`, `contractAxes`, 1D/2D `convolve*`/`correlate*` with `ConvMode.full/same/valid`, real `rfft/irfft`, complex `fft/ifft`, `fftAxis/ifftAxis`, `fftAxes/ifftAxes`, `fft2/ifft2`, `linalg.eye`, `det`, `inverse`, `solve`, `lu`, `solveTriangular`, `cholesky`, `qr`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`; f64 object-style `Array.det/inverse/solve/cholesky/qr/lu/solveTriangular/svd/lstsq/singularValues/matrixRank/cond/pinv/matrixNorm/eigh/eigvalsh` and `linalg.matmul`/`matvec`/`trace`/`det`/`solve`/`inverse`/`lu`/`solveTriangular`/`cholesky`/`qr`/`svd`/`lstsq`/`singularValues`/`matrixRank`/`cond`/`pinv`/`matrixNorm`/`eigh`/`eigvalsh` use the local `../veyra` backend when available, while non-f64 Array methods keep generic in-core fallbacks where implemented and return explicit errors where no fallback backend exists yet.
- SciPy-like stats helpers: `stats.zscore`, `normalize`, `pearsonr`.
- Sparse CSR/CSC bridge: `CsrMatrix`, `CscMatrix`, `csrFromDense`, `csrFromCompressed`, `cscFromDense`, `cscFromCompressed`, CSR-to-dense, transpose, transpose products, row/column stats, diagonal/trace diagnostics, bandwidth, symmetry checks, triangular solve, and Veyra-backed f64 CSR kernels.
- `Series(T)` and heterogeneous `DataFrame` with select/filter/sort/head/tail/describe/group-by-sum.
- CSV read/write with simple type inference.
- Array IO helpers: `toBytes/fromBytes` for raw data, `toArchive/fromArchive` for a simple dtype+shape binary archive, and object-style file helpers `saveArchive/saveArchiveToDir` plus `loadArchive/loadArchiveFromDir`.
- Device API placeholder (`Device.cpu`, `Device.cuda(index)`, object-style `to/cpu/cuda` on `Array`/`ArrayView`) for future CuPy/PyTorch-like GPU backends.

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

Vectra uses the sibling [`../alea`](../alea) Zig package as a local path dependency for random generation and distributions. Current array random helpers delegate seeded scalar random streams to Alea for uniform, normal, integer range, Bernoulli, exponential, gamma, beta, Poisson, multinomial, Dirichlet, log-normal, Student-t, Cauchy, Laplace, Weibull, half-normal, chi/chi-squared, Erlang, Fisher-F, triangular, arcsine, logistic/log-logistic, Kumaraswamy, power-function, Rayleigh, Maxwell, Pareto, Gumbel, Frechet, skew-normal, PERT, inverse-Gaussian, and normal-inverse-Gaussian generation. Future random distributions should prefer Alea rather than reimplementing RNG kernels inside Vectra.

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
