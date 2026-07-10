# Vectra

Vectra is a Zig 0.16 experimental data processing and numerical computing library.
It aims for a familiar Python-like surface inspired by NumPy/CuPy/SciPy/Pandas/Polars,
while leaning toward PyTorch-style fluent array methods for common operations. Vectra intentionally uses `Array`/`NDArray` as the primary user-facing name; automatic differentiation, training, and inference belong in the sibling `../forge` deep-learning framework.

> Status: early scaffold with a real, tested CPU core. The full NumPy/SciPy/Pandas
> ecosystem is enormous; this repository starts with a coherent architecture and
> useful primitives that can be expanded backend-by-backend and module-by-module.

## What is included now

- `Array(T)` / `NDArray(T)` with shape/strides and metadata helpers (`ndim/dim/rank/numDims/num_dims`, `numel/nelement`, `size/shapeAt/shape_at`, `stride/strideAt/stride_at`, `nbytes/num_bytes`, `elementSize/element_size/itemsize`, `storageOffset/storage_offset`, `dataPtr/data_ptr`, `isEmpty/is_empty`, `isContiguous/is_contiguous`, `isMatrix/isSquare/isBatchedMatrix`, scalar/vector predicates (`isScalar/isVector/isRowVector/isColumnVector/isVectorLike` plus snake_case aliases), scalar/flat export (`item/itemValue/item_value`, `scalarValue/scalar_value`, `asSlice/as_slice`, `asConstSlice/as_const_slice`, `copyToSlice/copy_to_slice`, `toOwnedSlice/to_owned_slice`), storage span/sharing metadata (`storageSize/storage_size`, `storageNbytes/storage_nbytes`, `storageSpan/storage_span`, `storageRange/storage_range`, `storageEndOffset/storage_end_offset`, `sharesStorage/sameStorage`, `mayOverlap` plus Array/View variants), shape comparison (`sameShape/same_shape`, `shapeEquals/shape_equals`, `hasShape/has_shape`, Array/View cross-shape helpers), `broadcastShape/broadcastShapes/broadcastWith`), device metadata (`Device.isCpu/isCuda/backendName/sameDevice/isAvailable` plus snake_case aliases, `deviceBackend/deviceIndex/deviceBackendName/isCpu/isCuda/isDeviceAvailable/sameDevice` on arrays/views), deep-copy helpers (`clone/copy/detach`), typed storage (`bool`, `i8/i16/i32/i64/isize`, `u8/u16/u32/u64/usize`, `BFloat16`, `f16/f32/f64`, `Complex64`/`Complex128`), dtype metadata (`dtypeName/dtype_name`, `dtypeTag/dtype_tag`, `dtypeByteSize/dtype_byte_size`, `dtypeBitSize/dtype_bit_size`, `isFloatDtype/isIntegerDtype/isSignedDtype/isUnsignedDtype/isComplexDtype/isBoolDtype/isRealDtype/isNumericDtype`, `canCastToDtype/can_cast_to_dtype`, plus type-level `canCastDType`, `promoteDType`, `resultDType`, `promoteType`), `reshape/view`, `flatten/ravel`, `squeeze/unsqueeze`, `permute/swapaxes/movedim`, `transpose`, PyTorch-like `matrixTranspose/matrix_transpose/mT`, `adjoint/mH/H_`, and `matrixPower/matrix_power`; `ArrayView(T)` / `NDArrayView(T)` provides non-owning strided views with shared storage, offsets, non-contiguous slicing, permutation, broadcasting, mutation, scalar/flat export helpers, contiguous and 1D strided fast paths for core linear kernels and scalar close comparisons, and `toArray/copy/detach/contiguous` materialization.
- Object-style construction on `Array(T)` / `NDArray(T)`: `fromSlice`, `fromScalar`, `zeros`, `ones`, `full`, `empty`, `emptyLike`, `zerosLike`, `onesLike`, `fullLike`, PyTorch-like `newEmpty/newZeros/newOnes/newFull` aliases, `eye/identity/eyeRect`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid` with `MeshGridIndexing.xy/ij`, `rand`, `randn`, `uniform`, `normal`, `randint`, `bernoulli`, `permutation`, `shuffle/shuffleInPlace`, `choice`, `choiceWeighted`, `exponential`, `gamma`, `beta`, `poisson`, `multinomial`, `dirichlet`, `lognormal`, `studentT`, `cauchy`, `laplace`, `weibull`, plus Alea-backed continuous distributions such as `halfNormal`, `chiSquared/chi`, `erlang`, `fisherF`, `triangular`, `arcsine`, `logistic`, `logLogistic`, `kumaraswamy`, `powerFunction`, `rayleigh`, `maxwell`, `pareto`, `gumbel`, `frechet`, `skewNormal`, `pert`, `inverseGaussian`, and `normalInverseGaussian`; random generation uses the local `../alea` backend.
- NumPy/PyTorch-like indexing helpers: `get/at`, `set/put`, scalar signed negative-index variants (`getSigned/get_signed/atSigned/at_signed`, `setSigned/set_signed/putSigned/put_signed`, `selectSigned/select_signed`), `select`, `narrow/narrowSigned`, `take/indexSelect`, batch signed negative-index variants (`takeSigned/take_signed/takeSignedMode/take_signed_mode`, `indexSelectSigned/index_select_signed`, `takeAlongAxisSigned/take_along_axis_signed`, `gatherSigned/gather_signed`, `putFlatSigned`, `putFlatScalarSigned`), `takeMode/takeSignedMode` with `IndexMode.raise/wrap/clip`, `takeAlongAxis/putAlongAxis`, coordinate indexing with prefix-shaped coordinate arrays (`ravelCoords`, `unravelFlat`, `takeCoords`, `putCoords/putCoordsScalar`) and broadcasted coordinate arrays (`ravelMultiIndex`, `takeMultiIndex`, `putMultiIndex/putMultiIndexScalar`), `gather`, `scatter/scatterScalar`, `scatterAdd`, `scatterReduce`, `scatterReduceScalar`, `scatterAddScalar`, `maskedSelect/masked_select`, `maskedFill/masked_fill`, `maskedScatter/masked_scatter`, `maskedPut/masked_put/maskedPutScalar/masked_put_scalar`, `putMask/put_mask/putMaskScalar/put_mask_scalar`, `copyWhere/copy_where/where/whereScalar/where_scalar`, object-style masked in-place helpers (`maskedFillAssign`, `maskedCopyFrom`, `maskedCopyFromView`, `copyWhereAssign`, `copyWhereAssignView` plus `ArrayView.maskedFill/maskedCopyFrom*/copyWhereFrom*`), `whereIndices`, `putFlat/put_flat/putFlatMode/putFlatScalar/put_flat_scalar/putFlatScalarMode`, `indexPut/index_put/indexPutScalar/index_put_scalar`, `compress`, `flatNonzero`, `nonzero/argwhere/countNonzero/countNonzeroAxis/count_nonzero`, `isin`, `slice1d`.
- Broadcasting elementwise arithmetic/comparisons: same-shape fast paths plus f32/f64 owning SIMD fast paths, caller-owned `Array.*Out` reuse-buffer APIs, and ArrayView caller-owned `*Out`/`*ScalarOut` materialization helpers for same-shape and scalar `add/sub/mul/div`, and broadcasted `add/sub/mul/div/pow`, same-shape fast promoted mixed-dtype variants (`addPromote`, `subPromote`, `mulPromote`, `divPromote`, `maximumPromote`, `minimumPromote`), complex helpers (`real`, `imag`, `conj/conjugate`, `magnitude/absComplex`, `angle/phase`, `isreal/iscomplex`, complex `exp/log/sqrt/sin/cos/tan/...` and complex finiteness predicates), `floorDiv`, `mod/remainder`, scalar variants, `maximum/minimum`, `fmax/fmin`, `hypot`, `atan2`, `logAddExp/logaddexp`, `logAddExp2/logaddexp2`, `xlogy`, `copysign`, `heaviside`, same-shape comparison fast paths (`eq/equal`, `ne/notEqual`, `gt/greater`, `ge/greaterEqual`, `lt/less`, `le/lessEqual`, scalar variants including `equalScalar/greaterScalar/lessScalar/...`) with f32/f64 SIMD acceleration for same-shape and scalar comparisons, boolean logic (`logicalNot`, view-aware `logicalAnd`, `logicalOr`, `logicalXor`, scalar variants including `equalScalar/greaterScalar/lessScalar/...`), `where/whereScalar/whereMask`, view-aware `isclose/isClose` and `allclose/allClose` with equal-NaN variants plus scalar close variants (`iscloseScalar/isCloseScalar`, `allcloseScalar/allCloseScalar`); object-style in-place helpers include `fill/fill_/zero_`, `copyFrom/copyFromView/copy_`, `copyFromArray/copy_from_array`, `add/sub/mul/divAssign`, `add/sub/mul/divAssignView`, PyTorch-style mutating aliases (`add_/sub_/mul_/div_`, scalar `addScalar_/add_scalar_`, etc.), and masked/copy-where mutating aliases (`masked_fill_`, `masked_copy_from`, `copy_where_`) on `Array`/`ArrayView`.
- Array transforms are object/type methods: `reshape/view`, `reshapeInfer/viewInfer` with one inferred dimension, `reshapeAs/viewAs`, `flatten/ravel`, `flattenAxes/flattenRange/flattenFrom`, `unflatten`, `atLeast1d/atLeast2d/atLeast3d`, `squeeze/unsqueeze` plus `squeezeDim/squeezeAxes/unsqueezeDim/unsqueezeAxes/expandDims`, `transpose`, `matrixTranspose/matrix_transpose/mT`, `adjoint/mH/H_`, `matrixPower/matrix_power`, `permute`, `swapaxes/swapDims`, `movedim/moveaxis/moveaxes`, `broadcastTo/broadcast_to/expand/expandAs/expand_as`, `repeat`, `repeatInterleave/repeat_interleave/repeatInterleaveScalar/repeat_interleave_scalar`, NumPy-like `tile` with leading-rank alignment, `slice/sliceAxis/slice1d`, `split`, `splitWithSizes/splitAtIndices`, `chunk`, `unbind`, `flip/flipAxes/fliplr/flipud`, `roll/rollFlat/rollAxes`, `rot90`, `padConstant`, `padEdge`, `padReflect`, `padWrap`, `padSymmetric`, and type-level `cat/concatenate`, `stack`, `hstack`, `vstack`, `dstack`, `columnStack`; non-copying view helpers include `asView`, safe `asStrided`, PyTorch-like `unfold`, `sliceAxisView/slice_axis_view`, `sliceView/slice_view`, `selectView/select_view`, `selectSignedView/select_signed_view`, `narrowView/narrow_view`, `narrowSigned/narrowSignedView/narrow_signed_view`, `unfoldView/unfold_view`, `permuteView/permute_view`, `swapaxesView/swapaxes_view`, `swapDimsView/swap_dims_view`, `movedimView/movedim_view`, `moveaxisView/moveaxis_view`, `moveaxesView/moveaxes_view/move_axes_view`, `transposeView/transpose_view/TView/T_view`, `matrixTransposeView/matrix_transpose_view/mTView/mT_view`, `matrixPower`, `diagonalView/diagonalAxesView`, zero-copy reshape/flatten aliases (`reshapeView/reshapeInferView`, `flattenView/flattenAxesView/flattenRangeView/flattenFromView`, `ravelView`, `unflattenView` plus snake_case variants), zero-copy squeeze/unsqueeze aliases (`squeezeView/squeezeDimView/squeezeAxesView`, `unsqueezeView/unsqueezeDimView/unsqueezeAxesView`, `expandDimsView`, `atLeast1dView/atLeast2dView/atLeast3dView` plus snake_case variants), `view/viewInfer/viewAs`, and zero-copy broadcast/expand aliases (`broadcastView/broadcast_view`, `broadcastToView/broadcast_to_view`, `expandView/expand_view`, `expandToView/expand_to_view`, `expandAs*`/`broadcastAs*` Array/View helpers), `atLeast1d/atLeast2d/atLeast3d`.
- Sorting/selection helpers: `sort`, `sortBy`, `sortDescending`, `argsort`, `argsortAxis`, `argsortDescending`, `sortWithIndices`, `partition`, `argpartition`, `kthValue/kth_value`, and `topk(sorted=true/false)`.
- Discrete/search/set helpers: `unique`, `uniqueWithCounts`, `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `bincount`, `bincountWeighted`, `searchsorted` with `SearchSide.left/right`, PyTorch-like `bucketize`, NumPy-like `digitize`, and broadcasted `clipArray`.
- Reductions/statistics as methods: f32/f64 flat `sum`/`mean` SIMD fast paths, `sum/sumAxes/sumToSize`, `prod/prodAxes`, `min/amin/minAxes/aminAxes`, `max/amax/maxAxes/amaxAxes`, `ptp/ptpAxes`, `allAxis/allAxes`, `anyAxis/anyAxes`, `mean/meanAxes`, `variance/varianceAxes`, `stddev/stddevAxes`, `median/medianAxes`, `quantile/quantileAxes`, `percentile/percentileAxes`, plus PyTorch-style object aliases using `dim/keepdim` naming (`sumDim/sumDims`, `meanDim/meanDims`, `varDim/varDims`, `stdDim/stdDims`, `argmaxDim/argminDim`, `topkDim/kthValueDim`, `softmaxDim/logSoftmaxDim`, `normDim/normDims`, `logsumexpDim/logsumexpDims`, cumulative `*Dim`, nan-aware `nan*Dim/nan*Dims`, `countNonzeroDim/countNonzeroDims`, `allDim/allDims`, `anyDim/anyDims` with snake_case variants), weighted stats (`weightedMean`, `average`, `weightedVariance/weightedVar`, `weightedStddev/weightedStd`, `weightedQuantile`, `weightedMedian`, `weightedCov`, `weightedCorrcoef`), `cov`, `corrcoef`, nan-aware stats (`nanToNum/nan_to_num/nanToNumDefault/nan_to_num_default`, `nansum/nansumAxes`, `nanmean/nanmeanAxes`, `nanvar/nanvarAxes`, `nanstd/nanstdAxes`, `nanmin/nanminAxes`, `nanmax/nanmaxAxes`, `nanmedian/nanmedianAxes`, `nanquantile/nanquantileAxes`, `nanpercentile/nanpercentileAxes`, `nanCov`, `nanCorrcoef`), `norm/normAxes`, `logsumexp/logsumexpAxes`, `logcumsumexp`, `cumsum`, `cumprod`, `cummax`, `cummin`, `cumsumAxis`, `cumprodAxis`, `cummaxAxis`, `cumminAxis`, `diff/diffWith/ediff1d`, `gradient`, `trapezoid/trapz`, `argmin`, `argmax`, `argminAxis/argmaxAxis`, `nanargmin/nanargmax`, `nanargminAxis/nanargmaxAxis`, `histogram`.
- `ArrayView(T)` can now directly call object-style materializing wrappers for `softmax/logSoftmax`, `norm`, `sort/sortBy/sortDescending`, `argsort/argsortAxis`, `topk`, `matmul/matmulArray`, `bmm`, `matvec`, `dot`, `vdot/vecdot`, `inner`, `outer`, `cross`, `contractAxes`, `convolve1d/correlate1d/convolve2d/correlate2d`, `trace/traceOffset/traceAxes`, `diagonal/diagonalAxes/diag/diagflat`, `triu/tril`, object-style linalg wrappers (`det`, `inverse/inv`, `solve`, `cholesky`, `qr`, `lu`, `solveTriangular`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`), and complex helpers (`real`, `imag`, `conj`, `magnitude`, `angle/phase`, `isreal/iscomplex`) while preserving non-copying indexing/mutation for view operations.
- `ArrayView(T)` also exposes common materializing dtype/elementwise/broadcast wrappers: direct strided `astype` and promoted mixed-dtype variants (`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`), view-aware `neg/negative`, `positive`, `abs/absolute/fabs`, `square`, `reciprocal`, `sign/signbit`, direct strided traversal for transcendental unary math (`exp/exp2/expm1`, `log/log1p/log2/log10/lgamma/gammaln`, `sqrt/rsqrt/cbrt`, `floor/ceil/round/trunc`, `deg2rad/radians`, `rad2deg/degrees`, `sinc`, trigonometric and inverse/hyperbolic aliases including `arcsin/arccos/arctan` and `asinh/acosh/atanh` plus `arcsinh/arccosh/arctanh`), activation-style unary math (`relu/leakyRelu`, `sigmoid/expit`, `logit`, `softplus`, `softsign`, `gelu`), `clip`, `clipMin/clipMax/clampMin/clampMax`, direct view-aware binary/ternary/scalar elementwise variants (`pow/floorDiv/mod`, `maximum/minimum`, `fmax/fmin`, `hypot/atan2`, `logAddExp/logAddExp2/xlogy`, `nextAfter/copysign/heaviside`, `lerp`, `addcmul/addcdiv`, `clipArray`, plus `*Scalar` forms such as `powScalar`, `logAddExpScalar`, and `ldexpScalar`), `isNan/isInf/isFinite/isNormal`, direct view-aware close comparisons (`isclose/isClose`, `iscloseScalar/isCloseScalar`, `allclose/allClose`, `allcloseScalar/allCloseScalar`) with equal-NaN variants, comparison aliases (`equal/greater/less`), direct view-aware unary predicate/logical wrappers (`signbit`, `isReal/iscomplex`, `isNan/isInf/isFinite/isNormal`, `logicalNot`, scalar and Array logical wrappers), PyTorch-style `*Dim` / `*Dims` aliases for reductions, sort/select, softmax/logsumexp/norm, cumulative, bool and nan-aware operations, and bool `logicalAnd/logicalOr/logicalXor` wrappers.
- `ArrayView(T)` statistics wrappers include view-aware `min/amin`, `max/amax`, `ptp`, `sum`, `prod`, `mean`, `variance/stddev` flat, single-axis, and multi-axis reductions, `sumAxes/prodAxes/minAxes/aminAxes/maxAxes/amaxAxes/ptpAxes/meanAxes/varianceAxes/stddevAxes` and matching `*Dim/*Dims` PyTorch-style aliases, `variance/stddev`, view-aware `median/quantile/percentile` flat, single-axis, and direct multi-axis reductions plus `medianAxes/medianDim`, `quantileAxes/quantileDim`, `percentileAxes/percentileDim`, `average`, weighted statistics (`weightedMean/weightedVariance/weightedVar/weightedStddev/weightedStd/weightedQuantile/weightedMedian`), view-aware nan-aware reductions and cleanup, including direct multi-axis sum/mean/min/max and quantile/median/percentile (`nanToNum/nan_to_num/nanToNumDefault/nan_to_num_default`, `nansum/nansumAxes/nansumDim`, `nanmean/nanmeanAxes/nanmeanDim`, `nanvar/nanvarAxes/nanvarDim`, `nanstd/nanstdAxes/nanstdDim`, `nanmin/nanminAxes/nanminDim`, `nanmax/nanmaxAxes/nanmaxDim`, `nanmedian/nanmedianAxes/nanmedianDim`, `nanquantile/nanquantileAxes/nanquantileDim`, `nanpercentile/nanpercentileAxes/nanpercentileDim`), `logsumexp/logsumexpDim`, covariance/correlation wrappers (`cov/corrcoef/weightedCov/weightedCorrcoef/nanCov/nanCorrcoef`), cumulative/integration operations (`cumsum/cumprod/cummax/cummin/logcumsumexp/cumsumAxis/cumsumDim/cumprodAxis/cumprodDim/cummaxAxis/cummaxDim/cumminAxis/cumminDim/logcumsumexpAxis/logcumsumexpDim/diff/gradient/trapezoid/trapz`), and arg reductions (`argmin/argmax/argminAxis/argmaxAxis/argminDim/argmaxDim/nanargmin/nanargmax/nanargminAxis/nanargmaxAxis/nanargminDim/nanargmaxDim`) plus bool `all/any/allAxis/anyAxis/allDim/anyDim`.
- `ArrayView(T)` also exposes metadata and like/new allocation helpers plus materializing indexing/search/shape wrappers such as `repeat/repeatInterleave/repeatInterleaveScalar`, NumPy-like `tile`, `flip/flipAxes/fliplr/flipud`, `roll/rollFlat/rollAxes`, `rot90`, `padConstant/padEdge/padReflect/padWrap/padSymmetric`, `slice1d`, `split/splitWithSizes/splitAtIndices/chunk/unbind`, `take/takeSigned`, `takeAlongAxis/takeAlongAxisSigned/putAlongAxis`, `indexSelect/indexSelectSigned`, `gather/gatherSigned`, `putFlat/putFlatMode/putFlatScalar*`, `indexPut/indexPutScalar`, coordinate and multi-index helpers, `scatter/scatterScalar/scatterAdd/scatterReduce*`, `maskedSelect`, `where/whereScalar`, `compress`, `nonzero/argwhere/whereIndices/countNonzeroAxis/countNonzeroAxes/count_nonzero`, `unique/uniqueWithCounts`, `union1d/intersect1d/setdiff1d/setxor1d`, `bincount/bincountWeighted`, `histogram`, `searchsorted`, `bucketize`, `digitize`, and `isin`.
- Neural/math functions: `neg/negative`, `positive`, `abs/absolute/fabs`, `astype`, promoted mixed-dtype variants (`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`), `square`, `reciprocal`, `sign/signbit`, `nextAfter/nextafter`, `ldexp`, `frexp`, `exp/exp2/expm1`, `log/log1p/log2/log10/lgamma/gammaln`, `sqrt/rsqrt/cbrt`, `floor`, `ceil`, `round`, `trunc`, `deg2rad/radians`, `rad2deg/degrees`, `sinc`, `sin`, `cos`, `tan`, `asin/arcsin`, `acos/arccos`, `atan/arctan`, `atan2/arctan2`, `hypot`, `copysign`, `heaviside`, `sinh`, `cosh`, `tanh`, `asinh/arcsinh`, `acosh/arccosh`, `atanh/arctanh`, `relu/leakyRelu`, `sigmoid/expit`, `logit`, `softplus`, `softsign`, `gelu`, `softmax`, `logsumexp`, `logcumsumexp`, `logSoftmax/log_softmax`, `clip/clamp`, `clipMin/clipMax/clampMin/clampMax`, `lerp`, `addcmul/addcdiv`, `fmax/fmin`, `isNan/isnan`, `isInf/isinf`、`isPosInf/isposinf`、`isNegInf/isneginf`, `isFinite/isfinite`, `isNormal/isnormal`.
- Linear algebra, signal basics, and contractions: `diag/diagflat/diagEmbed`, `diagonal/diagonalAxes/diagonalView/diagonalAxesView/fillDiagonal/diagonalScatter`, `trace/traceOffset/traceAxes`, `diagEmbed/diag_embed`, `triu/tril`, object-style matrix predicates (`isDiagonalMatrix`, `isUpperTriangular`, `isLowerTriangular`, `isSymmetric`, `isHermitian`), `matrixPower/matrix_power`, `det`, `inverse/inv`, `solve`, `cholesky`, `qr`, `lu`, `solveTriangular/solve_triangular`, `svd`, `lstsq`, `singularValues/singular_values`, `matrixRank/matrix_rank`, `cond`, `pinv`, `matrixNorm/matrix_norm`, `eigh`, `eigvalsh`, generalized `matmul/mm` with vector/matrix/batched broadcasting semantics plus Axiom CPU→Veyra-backed f32/f64 2D matrix-matrix, matrix-vector, vector-matrix, dot/vdot, trace, determinant, inverse, solve, Cholesky, QR, LU, triangular-solve, Frobenius/one/inf matrix-norm, SVD, singular-value, matrix-rank, and condition-number paths, `bmm`, `matvec`, `dot`, `inner`, `vecdot`, `vdot`, `outer`, `cross`, `contractAxes`, 1D/2D `convolve*`/`correlate*` with `ConvMode.full/same/valid`, real `rfft/irfft`, complex `fft/ifft`, `fftAxis/ifftAxis`, `fftAxes/ifftAxes`, `fft2/ifft2`, `linalg.eye`, `det`, `inverse`, `solve`, `lu`, `solveTriangular`, `cholesky`, `qr`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`; remaining SVD-derived high-level helpers compose over Axiom-backed SVD/singular-values/rank/condition-number until Axiom exposes matching wrappers, while non-covered Array methods keep generic in-core fallbacks where implemented and return explicit errors where no fallback backend exists yet.
- SciPy-like stats helpers: `stats.zscore`, `normalize`, `pearsonr`.
- Sparse CSR/CSC bridge: `CsrMatrix`, `CscMatrix`, `csrFromDense`, `csrFromCompressed`, `cscFromDense`, `cscFromCompressed`, CSR-to-dense, transpose, transpose products, row/column stats, diagonal/trace diagnostics, bandwidth, symmetry checks, triangular solve, and Veyra-backed f64 CSR kernels.
- `Series(T)` and heterogeneous `DataFrame` with select/filter/sort/head/tail/describe/group-by-sum.
- CSV read/write with simple type inference.
- Array IO helpers: `toBytes/fromBytes` for raw data, `toArchive/fromArchive` for a simple dtype+shape binary archive, and object-style file helpers `saveArchive/saveArchiveToDir` plus `loadArchive/loadArchiveFromDir`.
- Device API (`Device.cpu`, `Device.cuda(index)`, object-style `to/cpu/cuda` on `Array`) backed by Axiom by default for supported accelerator paths: CPU-backed `Array(f32/f64)` add/sub/mul/div/scalar-broadcast/matmul/matvec/dot/trace/det/inverse/solve/cholesky/qr/lu/solveTriangular/matrixNorm/svd/singularValues/matrixRank/cond route through Axiom CPU→Veyra, and CUDA-resident owning `Array(f32)` storage is available when a CUDA device can be retained. `fromSliceOn`/`emptyOn`/`zerosOn`/`onesOn`/`fullOn`, deterministic `Context` creation helpers such as `arrayWith`/`zerosWith`/`onesWith`, and `.cuda()` allocate directly in device memory, while `.cpu()` explicitly downloads. CUDA `Array(f32)` same-device `add/sub/mul/div`, `matmul`, and `matmulAdd` launch using existing device pointers; large f32 GEMM/GEMM+add uses Axiom's cached cuBLAS-backed SGEMM wrapper for PyTorch-class throughput, with the Axiom PTX seed retained as fallback/provenance. `ArrayView.cuda()` remains unsupported until view/device storage semantics are implemented.

## Example

```zig
const std = @import("std");
const vx = @import("vectra");

pub fn demo(allocator: std.mem.Allocator) !void {
    // Zig still makes allocation explicit, but Vectra's `Context` keeps the
    // everyday array surface close to NumPy/PyTorch.
    const np = vx.withAllocator(allocator);

    var a = try np.array(f64, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var bias = try np.ones(f64, &.{3});
    defer bias.deinit();

    var y = try vx.add(a, bias);  // NumPy/PyTorch-like broadcasting
    defer y.deinit();

    var probs = try y.softmax(1); // PyTorch-like method API
    defer probs.deinit();

    var picked_idx = try np.array(usize, &.{ 2, 0 }, &.{2});
    defer picked_idx.deinit();
    var picked = try y.indexSelect(1, picked_idx); // torch.index_select / np.take style
    defer picked.deinit();

    var labels = try np.array(i32, &.{ 2, 1, 2, 3 }, &.{4});
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

The fully explicit `vx.Array(T).fromSlice(allocator, ...)` and method surface is
still available when you need fine-grained control; the `vx.withAllocator(...)`
context and top-level `vx.add/vx.matmul/vx.sum/...` helpers are the intended
short-form front door for examples and application code. Ordinary array creation
and random creation do not require a seed: `try np.rand(f32, &.{ m, k })` uses the
context RNG stream. Creation options keep `dtype` as a Zig type parameter and
`device` as runtime metadata, e.g. `try np.zerosWith(vx.onDevice(f64, vx.cuda(0)), &.{ rows, cols })` or `try vx.Array(f32).fromSliceOn(allocator, values, dims, vx.cuda(0))`. Random CUDA creation is not exposed until a device RNG kernel exists; use CPU `rand` or explicit seeded CPU creation when reproducible host data is needed.
Use `vx.withSeed(...)` or `vx.seeded(...)` only when reproducible random values
are required; device is optional and defaults to `vx.cpu`.

Vectra also exposes layered Array abstractions for code that needs more or less
static information:

- `vx.StaticArray(T, vx.StaticLayout(...))`: dtype, shape, strides, and layout
  order are all comptime-known, CUTE-style metadata; use this for tile/kernel
  planning and fixed-shape kernels.
- `vx.SymbolicArray(T, vx.SymbolicLayout(...))`: dtype and layout expression are
  comptime-known, while selected extents are runtime-bound symbolic dimensions.
  Symbolic dimension expressions support `vx.symbol("M")`, `vx.dim(16)`, and
  `vx.dimAdd` / `vx.dimSub` / `vx.dimMul` so layouts can encode values such as
  `2 * M`, `K + 1`, or `N - tile`.
- `vx.Array(T)`: dtype is static, shape/strides layout and device are runtime
  metadata; this remains the main NumPy/PyTorch-like user array.
- `vx.AnyArray`: dtype and layout are runtime metadata for dynamic dispatch,
  serialization, and heterogeneous containers.

`device` is runtime metadata in every layer and follows PyTorch-like dispatch:
CPU arrays use Axiom CPU lowering for supported operations, CUDA arrays use Axiom
CUDA when a CUDA device is available, and mixed-device operations fail with
`InvalidDevice`. CUDA-resident Array storage is available for owning `Array`
values; supported f32 CUDA kernels consume those device pointers directly, while
unsupported operations return explicit errors or require an explicit `.cpu()`
transfer. Axiom host-slice bridge paths remain exposed through `vx.axiom_cuda` /
`vx.axiom_backend` for smoke coverage and provenance.

More runnable examples live under [`examples/`](examples):

```sh
zig build examples
zig build example-basic-array
zig build example-axiom-backend-policy
zig build example-axiom-cuda-bridge
zig build example-large-matmul-add
zig build example-large-matmul-add-smoke
```

`example-basic-array` is a Vectra CPU example that uses Axiom-backed supported
kernels where available. The Axiom examples run in the default build; CUDA routes
run when a CUDA device is available and otherwise report a skipped CUDA backend.
`example-large-matmul-add` keeps the user-facing body close to PyTorch:
device-aware creation followed by the same `vx.matmulAdd` call for CPU and CUDA
tensors. It documents the `Y = A[M,K] * B[K,N] + C[M,N]` workload. The checked-in
execute size is a CUDA stress run (`M = 4096 * 4`, `N = 4096`, `K = 4096`) and
dry-runs by default; pass `-- --smoke` for a tiny executable check or
`-- --execute --backend=cuda --require-cuda` for the production CUDA benchmark.
The execute mode warms up and averages repeated `matmulAdd` iterations so it can
be compared directly with a PyTorch CUDA tensor reuse benchmark.


## Axiom accelerator backend

Vectra imports the sibling [`../axiom`](../axiom) package by default. Supported
CPU-backed `Array(f32/f64)` same-shape and scalar/broadcast add/sub/mul/div,
2D matmul, matrix-vector, vector-matrix, dot/vdot, trace, determinant, inverse,
solve, Cholesky, QR, LU, triangular solve, Frobenius/one/inf matrix norms, SVD, singular values, matrix rank, and condition number flow through Axiom CPU lowering to Veyra. Supported CUDA owning-array
f32 add/sub/mul/div, matmul, and fused matmul+add use existing device pointers
through Axiom CUDA. Large f32 GEMM and GEMM+add use Axiom's cached
cuBLAS-backed SGEMM wrapper; the Axiom PTX/CUDA Tile IR seeds remain as
fallback/provenance paths.

Validation commands:

```sh
zig build test
zig build axiom-cpu-dispatch-smoke
zig build axiom-backend-policy-smoke
zig build axiom-cuda-dispatch-smoke
zig build axiom-cuda-device-smoke
zig build -Daxiom-cuda-expect=ran axiom-cuda-smoke
zig build -Doptimize=ReleaseFast example-large-matmul-add -- --execute --backend=cuda --require-cuda
```

CUDA commands require a CUDA/libnvvm/PTXAS-capable host. `Array(f16)` and
`Array(BFloat16)` have native Axiom CUDA same-shape elementwise seeds plus Axiom
typed SIMT GEMM runtime seed entry points for contiguous 2D matmul; the typed
GEMM seed reports launch-plan readiness metadata and the explicit
`widened_f32_cuda_compute` route while using widened f32 compute underneath
today. See [`docs/AXIOM_CUDA_BRIDGE.md`](docs/AXIOM_CUDA_BRIDGE.md) and
[`docs/CUDA_DTYPE_SUPPORT.md`](docs/CUDA_DTYPE_SUPPORT.md) for the supported
surface, local CUDA dtype matrix, and current limits.

## Alea backend

Vectra uses the sibling [`../alea`](../alea) Zig package as a local path dependency for random generation and distributions. Current array random helpers delegate seeded scalar random streams to Alea for uniform, normal, integer range, Bernoulli, exponential, gamma, beta, Poisson, multinomial, Dirichlet, log-normal, Student-t, Cauchy, Laplace, Weibull, half-normal, chi/chi-squared, Erlang, Fisher-F, triangular, arcsine, logistic/log-logistic, Kumaraswamy, power-function, Rayleigh, Maxwell, Pareto, Gumbel, Frechet, skew-normal, PERT, inverse-Gaussian, and normal-inverse-Gaussian generation. Future random distributions should prefer Alea rather than reimplementing RNG kernels inside Vectra.

## Veyra backend

Vectra uses the sibling [`../veyra`](../veyra) Zig package as a local path dependency for foundational math and linear algebra, but supported CPU tensor paths should enter through Axiom first. Current f32/f64 `Array`/`linalg` paths for matrix multiplication, matrix-vector products, dot/vdot, trace, determinant, inverse, solve, Cholesky, QR, LU, triangular solve, Frobenius/one/inf matrix norms, SVD, singular values, matrix rank, and condition number route through Axiom CPU→Veyra; least-squares, pseudo-inverse, two/nuclear matrix norms, and symmetric eigen decomposition still compose over those paths or call Veyra-compatible dense matrix APIs directly until Axiom exposes matching front-door wrappers. Non-covered dtypes and non-contiguous/batched Array methods keep generic in-core fallbacks where implemented.

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

## Performance comparison

Array performance should be compared against local NumPy/PyTorch before and after performance changes. The repository includes a Zig benchmark step plus a Python comparison script:

```sh
zig build bench --release=fast
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 tools/bench_numpy_torch.py
```

The current high-value benchmark set covers large f64 elementwise/scalar ops, flat reductions, promoted i32+f64 arithmetic, strided scalar/array ops, f64 dot/matvec/vecmat, and 256x256 f64 matmul.

## Roadmap

- Broader view-aware kernels and more simple-stride fast paths on top of the current `ArrayView`/`NDArrayView` storage model.
- Nullable values, categorical/string kernels and richer promotion policy.
- Polars-like lazy query plans and expression DSL.
- BLAS/LAPACK/high-performance FFT/sparse integrations.
- Broader GPU coverage behind the existing `Device` surface; see [`docs/AXIOM_CUDA_BRIDGE.md`](docs/AXIOM_CUDA_BRIDGE.md) for the current Axiom CUDA backend surface.
- Arrow/Parquet IPC support.
