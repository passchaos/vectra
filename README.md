# Vectra

Vectra is a Zig 0.16 experimental data processing and numerical computing library.
It aims for a familiar Python-like surface inspired by NumPy/CuPy/SciPy/Pandas/Polars,
while leaning toward PyTorch-style fluent array methods for common operations. Vectra intentionally uses `Array`/`NDArray` as the primary user-facing name; automatic differentiation, training, and inference belong in the sibling `../forge` deep-learning framework.

> Status: early scaffold with a real, tested CPU core. The full NumPy/SciPy/Pandas
> ecosystem is enormous; this repository starts with a coherent architecture and
> useful primitives that can be expanded backend-by-backend and module-by-module.

## What is included now

- `Array(T)` / `NDArray(T)` with shape/strides and metadata helpers (`ndim/dim/rank/numDims`, `numel/nelement`, `size/shapeAt`, `stride/strideAt`, `nbytes`, `elementSize/itemsize`, `storageOffset`, `dataPtr`, `isEmpty`, `isContiguous`, `isMatrix/isSquare/isBatchedMatrix`, scalar/vector predicates (`isScalar/isVector/isRowVector/isColumnVector/isVectorLike`), scalar/flat export (`item/itemValue`, `scalarValue`, `asSlice`, `asConstSlice`, `copyToSlice`, `toOwnedSlice`), storage span/sharing metadata (`storageSize`, `storageNbytes`, `storageSpan`, `storageRange`, `storageEndOffset`, `sharesStorage/sameStorage`, `mayOverlap` plus Array/View variants), shape comparison (`sameShape`, `shapeEquals`, `hasShape`, Array/View cross-shape helpers), `broadcastShape/broadcastShapes/broadcastWith`), device metadata (`Device.isCpu/isCuda/backendName/sameDevice/isAvailable`, `deviceBackend/deviceIndex/deviceBackendName/isCpu/isCuda/isDeviceAvailable/sameDevice` on arrays/views), deep-copy helpers (`clone/copy/detach`), typed storage (`bool`, `i8/i16/i32/i64/isize`, `u8/u16/u32/u64/usize`, `BFloat16`, `f16/f32/f64`, `Complex64`/`Complex128`), dtype metadata (`dtypeName`, `dtypeTag`, `dtypeByteSize`, `dtypeBitSize`, `isFloatDtype/isIntegerDtype/isSignedDtype/isUnsignedDtype/isComplexDtype/isBoolDtype/isRealDtype/isNumericDtype`, `canCastToDtype`, plus type-level `canCastDType`, `promoteDType`, `resultDType`, `promoteType`), `reshape/view`, `flatten/ravel`, `squeeze/unsqueeze`, `permute/swapaxes/movedim`, `transpose`, PyTorch-like `matrixTranspose/mT`, `adjoint/mH/adjoint`, and `matrixPower`; `ArrayView(T)` / `NDArrayView(T)` provides non-owning strided views with shared storage, offsets, non-contiguous slicing, permutation, broadcasting, mutation, scalar/flat export helpers, contiguous and 1D strided fast paths for core linear kernels and scalar close comparisons, and `toArray/copy/detach/contiguous` materialization.
- Object-style construction on `Array(T)` / `NDArray(T)`: `fromSlice`, `fromScalar`, `zeros`, `ones`, `full`, `empty`, `emptyLike`, `zerosLike`, `onesLike`, `fullLike`, PyTorch-like `newEmpty/newZeros/newOnes/newFull` aliases, `eye/identity/eyeRect`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid` with `MeshGridIndexing.xy/ij`, `rand`, `randn`, `uniform`, `normal`, `randint`, `bernoulli`, `permutation`, `shuffle/shuffleInPlace`, `choice`, `choiceWeighted`, `exponential`, `gamma`, `beta`, `poisson`, `multinomial`, `dirichlet`, `lognormal`, `studentT`, `cauchy`, `laplace`, `weibull`, plus Alea-backed continuous distributions such as `halfNormal`, `chiSquared/chi`, `erlang`, `fisherF`, `triangular`, `arcsine`, `logistic`, `logLogistic`, `kumaraswamy`, `powerFunction`, `rayleigh`, `maxwell`, `pareto`, `gumbel`, `frechet`, `skewNormal`, `pert`, `inverseGaussian`, and `normalInverseGaussian`; `rand`, CPU `uniform`, and CPU `randint` use Philox, while the remaining distribution helpers continue to use the local `../alea` backend where noted.
- NumPy/PyTorch-like indexing helpers: `get/at`, `set/put`, scalar signed negative-index variants (`getSigned/atSigned`, `setSigned/putSigned`, `selectSigned`), `select`, `narrow/narrowSigned`, `take/indexSelect`, batch signed negative-index variants (`takeSigned/takeSignedMode`, `indexSelectSigned`, `takeAlongAxisSigned`, `gatherSigned`, `putFlatSigned`, `putFlatScalarSigned`), `takeMode/takeSignedMode` with `IndexMode.raise/wrap/clip`, `takeAlongAxis/putAlongAxis`, coordinate indexing with prefix-shaped coordinate arrays (`ravelCoords`, `unravelFlat`, `takeCoords`, `putCoords/putCoordsScalar`) and broadcasted coordinate arrays (`ravelMultiIndex`, `takeMultiIndex`, `putMultiIndex/putMultiIndexScalar`), `gather`, `scatter/scatterScalar`, `scatterAdd`, `scatterReduce`, `scatterReduceScalar`, `scatterAddScalar`, `maskedSelect`, `maskedFill`, `maskedScatter`, `maskedPut/maskedPutScalar`, `putMask/putMaskScalar`, `copyWhere/where/whereScalar`, object-style masked in-place helpers (`maskedFillAssign`, `maskedCopyFrom`, `maskedCopyFromView`, `copyWhereAssign`, `copyWhereAssignView` plus `ArrayView.maskedFill/maskedCopyFrom*/copyWhereFrom*`), `whereIndices`, `putFlat/putFlatMode/putFlatScalar/putFlatScalarMode`, `indexPut/indexPutScalar`, `compress`, `flatNonzero`, `nonzero/argwhere/countNonzero/countNonzeroAxis/countNonzero`, `isin`, `slice1d`.
- Broadcasting elementwise arithmetic/comparisons: same-shape fast paths plus f32/f64 owning SIMD fast paths, caller-owned `Array.*Out` reuse-buffer APIs, and ArrayView caller-owned `*Out`/`*ScalarOut` materialization helpers for same-shape and scalar `add/sub/mul/div`, and broadcasted `add/sub/mul/div/pow`, same-shape fast promoted mixed-dtype variants (`addPromote`, `subPromote`, `mulPromote`, `divPromote`, `maximumPromote`, `minimumPromote`), complex helpers (`real`, `imag`, `conj/conjugate`, `magnitude/absComplex`, `angle/phase`, `isreal/iscomplex`, complex `exp/log/sqrt/sin/cos/tan/...` and complex finiteness predicates), `floorDiv`, `mod/remainder`, scalar variants, `maximum/minimum`, `fmax/fmin`, `hypot`, `atan2`, `logAddExp/logaddexp`, `logAddExp2/logaddexp2`, `xlogy`, `copysign`, `heaviside`, same-shape comparison fast paths (`eq/equal`, `ne/notEqual`, `gt/greater`, `ge/greaterEqual`, `lt/less`, `le/lessEqual`, scalar variants including `equalScalar/greaterScalar/lessScalar/...`) with f32/f64 SIMD acceleration for same-shape and scalar comparisons, boolean logic (`logicalNot`, view-aware `logicalAnd`, `logicalOr`, `logicalXor`, scalar variants including `equalScalar/greaterScalar/lessScalar/...`), `where/whereScalar/whereMask`, view-aware `isclose/isClose` and `allclose/allClose` with equal-NaN variants plus scalar close variants (`iscloseScalar/isCloseScalar`, `allcloseScalar/allCloseScalar`); object-style in-place helpers include `fill`, `copyFrom/copyFromView/copyFrom`, `copyFromArray`, `add/sub/mul/divAssign`, `add/sub/mul/divAssignView`, PyTorch-style mutating aliases (`addAssign/subAssign/mulAssign/divAssign`, scalar `addScalarAssign`, etc.), and masked/copy-where mutating aliases (`maskedFillAssign`, `maskedCopyFrom`, `copyWhereAssign`) on `Array`/`ArrayView`.
- Array transforms are object/type methods: `reshape/view`, `reshapeInfer/viewInfer` with one inferred dimension, `reshapeAs/viewAs`, `flatten/ravel`, `flattenAxes/flattenRange/flattenFrom`, `unflatten`, `atLeast1d/atLeast2d/atLeast3d`, `squeeze/unsqueeze` plus `squeezeDim/squeezeAxes/unsqueezeDim/unsqueezeAxes/expandDims`, `transpose`, `matrixTranspose/mT`, `adjoint/mH/adjoint`, `matrixPower`, `permute`, `swapaxes/swapDims`, `movedim/moveaxis/moveaxes`, `broadcastTo/expand/expandAs`, `repeat`, `repeatInterleave/repeatInterleaveScalar`, NumPy-like `tile` with leading-rank alignment, `slice/sliceAxis/slice1d`, `split`, `splitWithSizes/splitAtIndices`, `chunk`, `unbind`, `flip/flipAxes/fliplr/flipud`, `roll/rollFlat/rollAxes`, `rot90`, `padConstant`, `padEdge`, `padReflect`, `padWrap`, `padSymmetric`, and type-level `cat/concatenate`, `stack`, `hstack`, `vstack`, `dstack`, `columnStack`; non-copying view helpers include `asView`, safe `asStrided`, PyTorch-like `unfold`, `sliceAxisView`, `sliceView`, `selectView`, `selectSignedView`, `narrowView`, `narrowSigned/narrowSignedView`, `unfoldView`, `permuteView`, `swapaxesView`, `swapDimsView`, `movedimView`, `moveaxisView`, `moveaxesView`, `transposeView/TView`, `matrixTransposeView/mTView`, `matrixPower`, `diagonalView/diagonalAxesView`, zero-copy reshape/flatten aliases (`reshapeView/reshapeInferView`, `flattenView/flattenAxesView/flattenRangeView/flattenFromView`, `ravelView`, `unflattenView`), zero-copy squeeze/unsqueeze aliases (`squeezeView/squeezeDimView/squeezeAxesView`, `unsqueezeView/unsqueezeDimView/unsqueezeAxesView`, `expandDimsView`, `atLeast1dView/atLeast2dView/atLeast3dView`), `view/viewInfer/viewAs`, and zero-copy broadcast/expand aliases (`broadcastView`, `broadcastToView`, `expandView`, `expandToView`, `expandAs*`/`broadcastAs*` Array/View helpers), `atLeast1d/atLeast2d/atLeast3d`.
- Sorting/selection helpers: `sort`, `sortBy`, `sortDescending`, `argsort`, `argsortAxis`, `argsortDescending`, `sortWithIndices`, `partition`, `argpartition`, `kthValue`, and `topk(sorted=true/false)`.
- Discrete/search/set helpers: `unique`, `uniqueWithCounts`, `union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `bincount`, `bincountWeighted`, `searchsorted` with `SearchSide.left/right`, PyTorch-like `bucketize`, NumPy-like `digitize`, and broadcasted `clipArray`.
- Reductions/statistics as methods: f32/f64 flat `sum`/`mean` SIMD fast paths, `sum/sumAxes/sumToSize`, `prod/prodAxes`, `min/amin/minAxes/aminAxes`, `max/amax/maxAxes/amaxAxes`, `ptp/ptpAxes`, `allAxis/allAxes`, `anyAxis/anyAxes`, `mean/meanAxes`, `variance/varianceAxes`, `stddev/stddevAxes`, `median/medianAxes`, `quantile/quantileAxes`, `percentile/percentileAxes`, plus PyTorch-style object aliases using `dim/keepdim` naming (`sumDim/sumDims`, `meanDim/meanDims`, `varDim/varDims`, `stdDim/stdDims`, `argmaxDim/argminDim`, `topkDim/kthValueDim`, `softmaxDim/softminDim/logSoftmaxDim/logSoftminDim`, `normDim/normDims`, `cosineSimilarityDim/pairwiseDistanceDim`, `logsumexpDim/logsumexpDims`, cumulative `*Dim`, nan-aware `nan*Dim/nan*Dims`, `countNonzeroDim/countNonzeroDims`, `allDim/allDims`, `anyDim/anyDims`), weighted stats (`weightedMean`, `average`, `weightedVariance/weightedVar`, `weightedStddev/weightedStd`, `weightedQuantile`, `weightedMedian`, `weightedCov`, `weightedCorrcoef`), `cov`, `corrcoef`, nan-aware stats (`nanToNum/nanToNumDefault`, `nansum/nansumAxes`, `nanmean/nanmeanAxes`, `nanvar/nanvarAxes`, `nanstd/nanstdAxes`, `nanmin/nanminAxes`, `nanmax/nanmaxAxes`, `nanmedian/nanmedianAxes`, `nanquantile/nanquantileAxes`, `nanpercentile/nanpercentileAxes`, `nanCov`, `nanCorrcoef`), `norm/normAxes`, `normalize`, `cosineSimilarity`, `pairwiseDistance`, `logsumexp/logsumexpAxes`, `logcumsumexp`, `cumsum`, `cumprod`, `cummax`, `cummin`, `cumsumAxis`, `cumprodAxis`, `cummaxAxis`, `cumminAxis`, `diff/diffWith/ediff1d`, `gradient`, `trapezoid/trapz`, `argmin`, `argmax`, `argminAxis/argmaxAxis`, `nanargmin/nanargmax`, `nanargminAxis/nanargmaxAxis`, `histogram`.
- `ArrayView(T)` can now directly call object-style materializing wrappers for `softmax/softmin/logSoftmax/logSoftmin`, `norm`, `sort/sortBy/sortDescending`, `argsort/argsortAxis`, `topk`, `matmul/matmulArray`, `bmm`, `matvec`, `dot`, `vdot/vecdot`, `inner`, `outer`, `cross`, `contractAxes`, `convolve1d/correlate1d/convolve2d/correlate2d`, `trace/traceOffset/traceAxes`, `diagonal/diagonalAxes/diag/diagflat`, `triu/tril`, object-style linalg wrappers (`det`, `inverse/inv`, `solve`, `cholesky`, `qr`, `lu`, `solveTriangular`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`), and complex helpers (`real`, `imag`, `conj`, `magnitude`, `angle/phase`, `isreal/iscomplex`) while preserving non-copying indexing/mutation for view operations.
- `ArrayView(T)` also exposes common materializing dtype/elementwise/broadcast wrappers: direct strided `astype` and promoted mixed-dtype variants (`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`), view-aware `neg/negative`, `positive`, `abs/absolute/fabs`, `square`, `reciprocal`, `sign/signbit`, direct strided traversal for transcendental unary math (`exp/exp2/expm1`, `log/log1p/log2/log10/lgamma/gammaln`, `sqrt/rsqrt/cbrt`, `floor/ceil/round/trunc`, `deg2rad/radians`, `rad2deg/degrees`, `sinc`, trigonometric and inverse/hyperbolic aliases including `arcsin/arccos/arctan` and `asinh/acosh/atanh` plus `arcsinh/arccosh/arctanh`), activation-style unary math (`relu/leakyRelu/relu6`, `threshold`, `hardtanh/hardTanh`, `hardshrink/hardShrink`, `softshrink/softShrink`, `tanhshrink/tanhShrink`, `elu/celu`, `selu/SELU`, `glu/gluDim`, `sigmoid/expit`, `silu/SiLU/swish`, `mish`, `hardsigmoid/hardSigmoid`, `hardswish/hardSwish`, `logsigmoid/logSigmoid`, `logit`, `softplus`, `softsign`, `gelu`), `clip`, `clipMin/clipMax/clampMin/clampMax`, direct view-aware binary/ternary/scalar elementwise variants (`pow/floorDiv/mod`, `maximum/minimum`, `fmax/fmin`, `hypot/atan2`, `logAddExp/logAddExp2/xlogy`, `nextAfter/copysign/heaviside`, `lerp`, `addcmul/addcdiv`, `clipArray`, plus `*Scalar` forms such as `powScalar`, `logAddExpScalar`, and `ldexpScalar`), `isNan/isInf/isFinite/isNormal`, direct view-aware close comparisons (`isclose/isClose`, `iscloseScalar/isCloseScalar`, `allclose/allClose`, `allcloseScalar/allCloseScalar`) with equal-NaN variants, comparison aliases (`equal/greater/less`), direct view-aware unary predicate/logical wrappers (`signbit`, `isReal/iscomplex`, `isNan/isInf/isFinite/isNormal`, `logicalNot`, scalar and Array logical wrappers), PyTorch-style `*Dim` / `*Dims` aliases for reductions, sort/select, softmax/softmin/logsumexp/norm, nllLoss/crossEntropy, and binary cross-entropy wrappers, cumulative, bool and nan-aware operations, and bool `logicalAnd/logicalOr/logicalXor` wrappers.
- `ArrayView(T)` statistics wrappers include view-aware `min/amin`, `max/amax`, `ptp`, `sum`, `prod`, `mean`, `variance/stddev` flat, single-axis, and multi-axis reductions, `sumAxes/prodAxes/minAxes/aminAxes/maxAxes/amaxAxes/ptpAxes/meanAxes/varianceAxes/stddevAxes` and matching `*Dim/*Dims` PyTorch-style aliases, `variance/stddev`, view-aware `median/quantile/percentile` flat, single-axis, and direct multi-axis reductions plus `medianAxes/medianDim`, `quantileAxes/quantileDim`, `percentileAxes/percentileDim`, `average`, weighted statistics (`weightedMean/weightedVariance/weightedVar/weightedStddev/weightedStd/weightedQuantile/weightedMedian`), view-aware nan-aware reductions and cleanup, including direct multi-axis sum/mean/min/max and quantile/median/percentile (`nanToNum/nanToNumDefault`, `nansum/nansumAxes/nansumDim`, `nanmean/nanmeanAxes/nanmeanDim`, `nanvar/nanvarAxes/nanvarDim`, `nanstd/nanstdAxes/nanstdDim`, `nanmin/nanminAxes/nanminDim`, `nanmax/nanmaxAxes/nanmaxDim`, `nanmedian/nanmedianAxes/nanmedianDim`, `nanquantile/nanquantileAxes/nanquantileDim`, `nanpercentile/nanpercentileAxes/nanpercentileDim`), `logsumexp/logsumexpDim`, covariance/correlation wrappers (`cov/corrcoef/weightedCov/weightedCorrcoef/nanCov/nanCorrcoef`), cumulative/integration operations (`cumsum/cumprod/cummax/cummin/logcumsumexp/cumsumAxis/cumsumDim/cumprodAxis/cumprodDim/cummaxAxis/cummaxDim/cumminAxis/cumminDim/logcumsumexpAxis/logcumsumexpDim/diff/gradient/trapezoid/trapz`), and arg reductions (`argmin/argmax/argminAxis/argmaxAxis/argminDim/argmaxDim/nanargmin/nanargmax/nanargminAxis/nanargmaxAxis/nanargminDim/nanargmaxDim`) plus bool `all/any/allAxis/anyAxis/allDim/anyDim`.
- `ArrayView(T)` also exposes metadata and like/new allocation helpers plus materializing indexing/search/shape wrappers such as `repeat/repeatInterleave/repeatInterleaveScalar`, NumPy-like `tile`, `flip/flipAxes/fliplr/flipud`, `roll/rollFlat/rollAxes`, `rot90`, `padConstant/padEdge/padReflect/padWrap/padSymmetric`, `slice1d`, `split/splitWithSizes/splitAtIndices/chunk/unbind`, `take/takeSigned`, `takeAlongAxis/takeAlongAxisSigned/putAlongAxis`, `indexSelect/indexSelectSigned`, `gather/gatherSigned`, `putFlat/putFlatMode/putFlatScalar*`, `indexPut/indexPutScalar`, coordinate and multi-index helpers, `scatter/scatterScalar/scatterAdd/scatterReduce*`, `maskedSelect`, `where/whereScalar`, `compress`, `nonzero/argwhere/whereIndices/countNonzeroAxis/countNonzeroAxes/countNonzero`, `unique/uniqueWithCounts`, `union1d/intersect1d/setdiff1d/setxor1d`, `bincount/bincountWeighted`, `histogram`, `searchsorted`, `bucketize`, `digitize`, and `isin`.
- Neural/math functions: `neg/negative`, `positive`, `abs/absolute/fabs`, `astype`, promoted mixed-dtype variants (`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`), `square`, `reciprocal`, `sign/signbit`, `nextAfter/nextafter`, `ldexp`, `frexp`, `exp/exp2/expm1`, `log/log1p/log2/log10/lgamma/gammaln`, `sqrt/rsqrt/cbrt`, `floor`, `ceil`, `round`, `trunc`, `deg2rad/radians`, `rad2deg/degrees`, `sinc`, `sin`, `cos`, `tan`, `asin/arcsin`, `acos/arccos`, `atan/arctan`, `atan2/arctan2`, `hypot`, `copysign`, `heaviside`, `sinh`, `cosh`, `tanh`, `asinh/arcsinh`, `acosh/arccosh`, `atanh/arctanh`, `relu/leakyRelu/relu6`, `threshold`, `hardtanh/hardTanh`, `hardshrink/hardShrink`, `softshrink/softShrink`, `tanhshrink/tanhShrink`, `elu/celu`, `selu/SELU`, `glu/gluDim`, `sigmoid/expit`, `silu/SiLU/swish`, `mish`, `hardsigmoid/hardSigmoid`, `hardswish/hardSwish`, `logsigmoid/logSigmoid`, `logit`, `softplus`, `softsign`, `gelu`, `softmax`, `logsumexp`, `logcumsumexp`, `logSoftmax`、`logSoftmin`, `nllLoss`, `crossEntropy`, `binaryCrossEntropy`, `binaryCrossEntropyWithLogits`, `multiLabelSoftMarginLoss`, `poissonNllLoss`, `mseLoss`, `l1Loss`, `smoothL1Loss`, `huberLoss`, `klDiv`, `marginRankingLoss`, `cosineEmbeddingLoss`, `softMarginLoss`, `hingeLoss`, `hingeEmbeddingLoss`, `multiMarginLoss`, and `tripletMarginLoss` with `LossReduction.none/sum/mean`, `clip/clamp`, `clipMin/clipMax/clampMin/clampMax`, `lerp`, `addcmul/addcdiv`, `fmax/fmin`, `isNan/isnan`, `isInf/isinf`、`isPosInf/isposinf`、`isNegInf/isneginf`, `isFinite/isfinite`, `isNormal/isnormal`.
- Linear algebra, signal basics, and contractions: `diag/diagflat/diagEmbed`, `diagonal/diagonalAxes/diagonalView/diagonalAxesView/fillDiagonal/diagonalScatter`, `trace/traceOffset/traceAxes`, `diagEmbed`, `triu/tril`, object-style matrix predicates (`isDiagonalMatrix`, `isUpperTriangular`, `isLowerTriangular`, `isSymmetric`, `isHermitian`), `matrixPower`, `det`, `inverse/inv`, `solve`, `cholesky`, `qr`, `lu`, `solveTriangular`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`, generalized `matmul/mm` with vector/matrix/batched broadcasting semantics plus Axiom CPU→Veyra-backed f32/f64 2D matrix-matrix, matrix-vector, vector-matrix, dot/vdot, trace, determinant, inverse, solve, Cholesky, QR, LU, triangular-solve, Frobenius/one/inf/two/nuclear matrix-norm, SVD, singular-value, matrix-rank, condition-number, pseudo-inverse, least-squares, symmetric-eigen, and Hermitian-eigenvalue paths, `bmm`, `matvec`, `dot`, `inner`, `vecdot`, `vdot`, `outer`, `cross`, `contractAxes`, 1D/2D `convolve*`/`correlate*` with `ConvMode.full/same/valid`, real `rfft/irfft`, complex `fft/ifft`, `fftAxis/ifftAxis`, `fftAxes/ifftAxes`, `fft2/ifft2`, `linalg.eye`, `det`, `inverse`, `solve`, `lu`, `solveTriangular`, `cholesky`, `qr`, `svd`, `lstsq`, `singularValues`, `matrixRank`, `cond`, `pinv`, `matrixNorm`, `eigh`, `eigvalsh`; non-covered Array methods keep generic in-core fallbacks where implemented and return explicit errors where no fallback backend exists yet.
- SciPy-like stats helpers: `stats.zscore`, `normalize`, `pearsonr`.
- Sparse CSR/CSC bridge: `CsrMatrix`, `CscMatrix`, `csrFromDense`, `csrFromCompressed`, `cscFromDense`, `cscFromCompressed`, CSR-to-dense, transpose, transpose products, row/column stats, diagonal/trace diagnostics, bandwidth, symmetry checks, triangular solve, and Veyra-backed f64 CSR kernels.
- `Series(T)` and heterogeneous CPU `DataFrame` with select/filter/sort/head/tail/describe/group-by-sum, plus a cuDF/Polars-inspired `DeviceDataFrame`/`DeviceColumn` owning fixed-width columns on a shared Vectra device (CPU/CUDA/MPS) with non-owning table/column views, row concat/append/vstack, row distinct/drop-duplicates, nullable validity metadata, eager same-dtype column arithmetic/comparisons, generated boolean-mask filtering, lightweight `DeviceLazyFrame` dataframe or Parquet-scan sources with derived-column expressions, lazy groupby aggregations, lazy equi/asof joins, lazy row concat/append/vstack, lazy distinct/drop-duplicates, select/positional-select-drop/first-last-column-select/column-reverse-name-sort/name-pattern-select/name-pattern-drop/dtype-select/dtype-drop/nullability-select-drop/nan-presence-select-drop/inf-presence-select-drop/signed-inf-presence-select-drop/zero-presence-select-drop/signed-zero-presence-select-drop/non-zero-presence-select-drop/positive-presence-select-drop/signbit-presence-select-drop/negative-presence-select-drop/finite-presence-select-drop/normal-presence-select-drop/subnormal-presence-select-drop/non-finite-presence-select-drop/cast/fill-null/fill-nan/fill-inf/fill-signed-inf/fill-zero/fill-signed-zero/fill-non-zero/fill-positive/fill-signbit/fill-negative/fill-finite/fill-normal/fill-subnormal/fill-non-finite/abs-column/neg-column/square-column/reciprocal-column/sign-column/sqrt-column/rsqrt-column/cbrt-column/floor-column/ceil-column/round-column/trunc-column/deg2rad-column/rad2deg-column/expit-column/logit-column/softplus-column/logsigmoid-column/relu-column/leaky-relu-column/relu6-column/pow-scalar-column/floor-div-scalar-column/mod-scalar-column/remainder-scalar-column/log-add-exp-scalar-column/log-add-exp2-scalar-column/xlogy-scalar-column/fmax-scalar-column/fmin-scalar-column/hypot-scalar-column/atan2-scalar-column/next-after-scalar-column/copysign-scalar-column/heaviside-scalar-column/ldexp-scalar-column/lerp-scalar-column/addcmul-scalar-column/addcdiv-scalar-column/clip-array-column/where-scalar-column/where-column/isin-column/masked-put-scalar-column/put-flat-column/put-flat-scalar-column/put-flat-signed-scalar-column/put-flat-mode-scalar-column/isclose-scalar-column/allclose-scalar-gate/count-nonzero-column/zero-count-column/zero-ratio-column/nonzero-ratio-column/first-zero-index-column/last-zero-index-column/first-nonzero-index-column/last-nonzero-index-column/positive-zero-count-column/negative-zero-count-column/positive-zero-ratio-column/negative-zero-ratio-column/positive-count-column/negative-count-column/signbit-count-column/positive-ratio-column/negative-ratio-column/signbit-ratio-column/first-positive-index-column/last-positive-index-column/first-negative-index-column/last-negative-index-column/first-signbit-index-column/last-signbit-index-column/first-nan-index-column/last-nan-index-column/first-inf-index-column/last-inf-index-column/first-finite-index-column/last-finite-index-column/first-non-finite-index-column/last-non-finite-index-column/nan-count-column/inf-count-column/positive-inf-count-column/negative-inf-count-column/finite-count-column/normal-count-column/subnormal-count-column/non-finite-count-column/nan-ratio-column/inf-ratio-column/positive-inf-ratio-column/negative-inf-ratio-column/finite-ratio-column/normal-ratio-column/subnormal-ratio-column/non-finite-ratio-column/count-distinct-column/n-unique-column/null-count-column/valid-count-column/null-ratio-column/valid-ratio-column/first-valid-index-column/last-valid-index-column/first-null-index-column/last-null-index-column/mode-column/sum-column/prod-column/mean-column/median-column/quantile-column/variance-column/stddev-column/sem-column/cv-column/skewness-column/kurtosis-column/mean-abs-column/rms-column/l1-norm-column/l2-norm-column/geometric-mean-column/harmonic-mean-column/mad-column/iqr-column/min-column/max-column/ptp-column/argmin-column/argmax-column/bool-scalar-gate/logical-scalar-column/logical-column/threshold-column/hardtanh-column/maximum-scalar-column/minimum-scalar-column/clip-min-column/clip-max-column/hardshrink-column/softshrink-column/tanhshrink-column/elu-column/celu-column/softsign-column/hardsigmoid-column/hardswish-column/silu-column/swish-column/mish-column/gelu-column/selu-column/exp-column/exp2-column/expm1-column/sin-column/cos-column/tan-column/asin-column/acos-column/atan-column/sinh-column/cosh-column/tanh-column/asinh-column/acosh-column/atanh-column/log-column/log1p-column/lgamma-column/sinc-column/log2-column/log10-column/coalesce-column/drop-null/filter-null/null-predicate-column/nan-inf-signed-inf-zero-signed-zero-non-zero-positive-signbit-negative-finite-normal-subnormal-non-finite-predicate-column/drop-nan/filter-nan/drop-inf/filter-inf/drop-signed-inf/filter-signed-inf/drop-zero/filter-zero/drop-signed-zero/filter-signed-zero/drop-non-zero/filter-non-zero/drop-positive/filter-positive/drop-signbit/filter-signbit/drop-negative/filter-negative/drop-finite/filter-finite/drop-normal/filter-normal/drop-subnormal/filter-subnormal/drop-non-finite/filter-non-finite/row-null-valid-nan-inf-signed-inf-zero-signed-zero-non-zero-positive-signbit-negative-finite-normal-subnormal-count/literal-column/positioned-literal-column/copy-column/row-index/rename/bulk-rename/move/drop/boolean-column-filter/boolean-column-drop/where-indices-column/scalar-filter/filter-mask/sort/reverse/roll-row/shift-row/slice/signed-slice/signed-stepped-slice/drop-row/drop-row-mode/signed-drop-row/signed-drop-row-mode/drop-row-by-column/drop-row-by-column-mode/repeat-row/tile-row/repeat-row-by-column/stepped-slice/stride/sample/replacement-sample/take/optional-take/take-by-column/take-by-column-mode/take-mode/signed-take/signed-take-mode/head/tail plans, explain/scan-pushdown/adjacent-op/top-k optimization, stable argsort/sort-by-key/top-k with null placement, `rankProfileBy` rank/window columns (ordinal/competition/dense/percent/cume-dist), `rollingRankProfile` trailing rank columns (count/rank/percent/cume-dist), `bucketProfile` empirical distribution columns (ECDF/equal-frequency bucket/lower-tail/upper-tail), `validityProfile` data-quality columns (is-null/is-valid/valid-streak/null-streak), `rollingValidityProfile` trailing data-quality columns (window-count/valid-count/null-count/valid-rate/null-rate), `expandingValidityProfile` cumulative data-quality columns (count/valid-count/null-count/valid-rate/null-rate), `classificationProfile` bool evaluation columns (TP/FP/TN/FN/correct), `rollingClassificationProfile` trailing classification columns (count/TP/FP/TN/FN/accuracy/precision/recall), `expandingClassificationProfile` cumulative classification columns (count/TP/FP/TN/FN/accuracy/precision/recall), `boolTransitionProfile` bool sequence columns (rising/falling/toggled/true-streak/false-streak), `rollingBoolTransitionProfile` trailing bool transition columns (transition-count/rising-count/falling-count/toggle-count/rising-rate/falling-rate/toggle-rate), `expandingBoolTransitionProfile` cumulative bool transition columns (transition-count/rising-count/falling-count/toggle-count/rising-rate/falling-rate/toggle-rate), `rollingBoolProfile` trailing bool-window columns (true-count/false-count/true-rate/any/all), `expandingBoolProfile` cumulative bool columns (true-count/false-count/true-rate/any/all), `clipProfile` bounded-cleaning columns (clipped/below/above/in-range), `rollingClipProfile` trailing clip summary columns (count/mean-clipped/clipped-rate/below-rate/above-rate/in-range-rate), `expandingClipProfile` cumulative clip summary columns (count/mean-clipped/clipped-rate/below-rate/above-rate/in-range-rate), `thresholdProfile` threshold-distance columns (distance/absolute/above/below/at), `rollingThresholdProfile` trailing threshold summary columns (count/mean-distance/mean-absolute-distance/above-rate/below-rate/at-rate), `expandingThresholdProfile` cumulative threshold summary columns (count/mean-distance/mean-absolute-distance/above-rate/below-rate/at-rate), `emaProfile` exponential-smoothing columns (EMA/residual/ratio), `linearFitProfile` two-column model diagnostics (fitted/residual/residual-z/slope), `rollingLinearFitProfile` trailing two-column regression diagnostics (pair-count/slope/intercept/fitted/residual/residual-z), `expandingLinearFitProfile` cumulative regression diagnostics (pair-count/slope/intercept/fitted/residual/residual-z), `errorProfile` forecast/error diagnostics (error/absolute/squared/APE/SMAPE), `rollingErrorProfile` trailing error summary columns (count/MAE/RMSE/MAPE/SMAPE), `expandingErrorProfile` cumulative error summary columns (count/MAE/RMSE/MAPE/SMAPE), `rollingCorrelationProfile` two-column rolling statistics (pair-count/covariance/correlation/beta), `expandingCorrelationProfile` cumulative two-column statistics (pair-count/covariance/correlation/beta), `rollingProfile` trailing-window columns (count/sum/mean/variance/stddev), `rollingMomentProfile` trailing higher-moment columns (count/M3/M4/skewness/kurtosis), `rollingRangeProfile` trailing range columns (low/high/range/position), `rollingNormalizeProfile` trailing normalization columns (centered/z-score/min-max), `expandingNormalizeProfile` cumulative normalization columns (centered/z-score/min-max), `rollingQuantileProfile` trailing distribution columns (q1/median/q3/IQR), `expandingQuantileProfile` cumulative distribution columns (q1/median/q3/IQR), `rollingDrawdownProfile` trailing risk columns (peak/drawdown/drawdown-pct/peak-age), `rollingRobustProfile` trailing robust-stat columns (median-centered/MAD-z/IQR-outlier/winsorized), `lagProfile`/`leadProfile` sequence feature columns (lag/lead, backward/forward diff, pct-change), `signProfile` sign-state columns (sign/sign-flip/positive/negative/zero streak), `rollingSignProfile` trailing sign composition columns (count/positive-rate/negative-rate/zero-rate/flip-rate), `expandingSignProfile` cumulative sign composition columns (count/positive-rate/negative-rate/zero-rate/flip-rate), `expandingProfile` cumulative columns (count/sum/mean/min/max), `expandingRankProfile` cumulative rank columns (count/rank/percent/cume-dist), `expandingRobustProfile` cumulative robust-stat columns (median-centered/MAD-z/IQR-outlier/winsorized), `expandingMomentProfile` cumulative higher-moment columns (count/M3/M4/skewness/kurtosis), `extremaProfile` running extrema columns (low/high/new-low/new-high), `standardizeProfile` scale columns (centered/z-score/min-max), `robustProfile` outlier-resistant columns (median-centered/MAD-z/IQR-outlier/winsorized), `drawdownProfile` risk/time-series columns (running-peak/drawdown/drawdown-pct), `trendProfile` sequence columns (trend/up-streak/down-streak/flat-streak/reversal), `rollingTrendProfile` trailing trend composition columns (count/up-rate/down-rate/flat-rate/reversal-rate), `expandingTrendProfile` cumulative trend composition columns (count/up-rate/down-rate/flat-rate/reversal-rate), `changePointProfile` sequence jump columns (delta/absolute-delta/percent-change/change-point), `rollingChangePointProfile` trailing jump summary columns (count/change-count/change-rate/mean-abs-delta/max-abs-delta), `expandingChangePointProfile` cumulative jump summary columns (count/change-count/change-rate/mean-abs-delta/max-abs-delta), `crossoverProfile` two-column signal columns (spread/ratio/cross-above/cross-below), `rollingCrossoverProfile` trailing crossover signal summaries (count/cross-above-count/cross-below-count/cross-above-rate/cross-below-rate/mean-abs-spread), `expandingCrossoverProfile` cumulative crossover signal summaries (count/cross-above-count/cross-below-count/cross-above-rate/cross-below-rate/mean-abs-spread), fixed-width `groupByCount`/`groupBySum`/`groupByMin`/`groupByMax`/`groupByMean` plus one-pass single/multi-key `groupByStats`/`groupByStatsOn` and `groupByProfile`/`groupByProfileOn` moment profiles (count/sum/mean/variance/stddev/skewness/kurtosis), fixed-width `innerJoinOn`/`leftJoinOn`/`fullJoinOn`/`semiJoinOn`/`antiJoinOn` for multi-key equi joins plus single-key `innerJoin`/`leftJoin`/`fullJoin`/`semiJoin`/`antiJoin` and ordered `asofJoin`, select/drop by position/range/first-last plus reverse/name-sort, name prefix/suffix/contains, dtype/class, and nullability/null-presence, column copy/move, head/tail/slice/signed-slice/signed-stepped-slice/drop-row/stepped-slice/stride/sample/replacement-sample/take/filter, device transfer, Boltha/Arrow schema/record-batch/table export, Boltha/Parquet byte round-trip, range-pruned reads, lazy `DeviceParquetScan`, and lazy Parquet scalar-filter/projection pushdown, and round-trip conversion for legacy fixed-width dataframe columns.
- CSV read/write with simple type inference.
- Array IO helpers: `toBytes/fromBytes` for raw data, `toArchive/fromArchive` for a simple dtype+shape binary archive, and object-style file helpers `saveArchive/saveArchiveToDir` plus `loadArchive/loadArchiveFromDir`.
- Device API (`Device.cpu`, `Device.cuda(index)`, `Device.mps(index)`, object-style `to/cpu/cuda/mps` on `Array`) backed by Axiom by default for supported accelerator paths: CPU-backed `Array(f32/f64)` add/sub/mul/div/sqrt/exp/exp2/expm1/log/log1p/log2/log10/sin/cos/tan/asin/acos/atan/square/scalar-broadcast/2D row-bias and column-bias broadcast add/2D axis reductions (`sum/prod/min/max`)/2D transpose/matmul/matvec/dot/trace/det/inverse/solve/cholesky/qr/lu/solveTriangular/matrixNorm(.fro/.one/.inf/.two/.nuclear)/svd/singularValues/matrixRank/cond/pinv/lstsq/eigh/eigvalsh route through Axiom CPU->Veyra, and CUDA-resident owning `Array(f32)` storage is available when a CUDA device can be retained. `fromSliceOn`/`emptyOn`/`zerosOn`/`onesOn`/`fullOn`, deterministic `Context` creation helpers such as `arrayWith`/`zerosWith`/`onesWith`, and `.cuda()` allocate directly in device memory, while `.cpu()` explicitly downloads. CUDA `Array(f32)` same-device `add/sub/mul/div`, `log/exp2/expm1/log1p/log2/log10`, `sin/cos/tan`, `softmax(axis=0/1)`, `matmul`, and `matmulAdd`, CUDA f64 same-shape elementwise, softmax(axis=0/1), 2D reductions/broadcast-add/transpose, and matmul, plus typed f16/BFloat16 2D reductions, broadcast-add, transpose, softmax, and matmul seeds launch using existing device pointers; large f32 GEMM+add uses Axiom's cached cuBLAS-backed SGEMM wrapper for PyTorch-class throughput, with the Axiom PTX seed retained as fallback/provenance. Direct `ArrayView.cuda()` remains unsupported until full view/device storage semantics are implemented; host-backed 1D positive-stride `ArrayView(f32/f64).add/sub/mul/div`, scalar add/sub/mul/div, and f32/f64 `sqrt/exp` can execute through Axiom CPU->Veyra memref runtimes under the CPU target, while host-backed 1D positive-stride `ArrayView(f32/f64/f16/BFloat16).add/sub/mul/div` and `ArrayView(f32/f64/f16/BFloat16).addScalar/subScalar/mulScalar/divScalar` plus f32 `log/exp2/expm1/log1p/log2/log10/sin/cos/tan`, f32/f64 `abs/sqrt/exp`, and f32/f64/f16/BFloat16 `neg/square/reciprocal` can execute through the central Axiom target facade using CUDA strided/unary/zero-stride scalar launch provenance when the default target is CUDA; `Device.mps(index)` and `.mps(index)` now use Axiom's Metal/MPS runtime ABI on macOS for real shared `MTLBuffer` owning-array storage, CPU<->MPS copies, same-device copies, fills, owning Array reshape/reshapeInfer/flatten storage evidence, f32/f16 contiguous 2D matmul/matmulAdd through `MPSMatrixMultiplication`, BF16 2D matmul/matmulAdd through Metal kernels because Apple `MPSMatrixMultiplication` does not accept BF16 input matrices, f32 same-shape elementwise/scalar/device-scalar-broadcast including div/rdiv/unary Metal kernels plus matmul-then-add/sub/sqrt/exp device-storage chains, rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, and dot/inner/outer/matvec/vecmat, f32 transpose, row/column broadcast add/sub/mul/div plus rank<=6 general broadcast add/sub/mul/div and rank>2 last-dim broadcast via MPS composition, 2D/flat/all-axes reductions and `ptp`, softmax/logSoftmax/logsumexp, f32/f16/BFloat16 composed `rsqrt/leakyRelu/silu/hardsigmoid/hardswish/softshrink/tanh/tanhshrink/gelu/elu/celu/selu/SELU/relu6/hardtanh/clipArray`, `powScalar(-1/-0.5/0/0.5/1/2/3)`, `softmin/logSoftmin`, metric helpers (`norm/normalize/cosineSimilarity/pairwiseDistance`), and statistics helpers (`mean/variance/stddev`) plus focused rank-3 `sum/prod/min/max` and partial-axes `variance/stddev` reduction smoke, f32/f16/BFloat16 `mseLoss/l1Loss/smoothL1Loss/huberLoss(.none)`, plus f16 same-shape elementwise/scalar/device-scalar-broadcast/unary math (`abs/square/sqrt/exp/log/exp2/expm1/log1p/log2/log10/sin/cos/tan`), rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, and dot/inner/outer/matvec/vecmat, transpose, row/column broadcast add/sub/mul/div plus rank<=6 general broadcast add/sub/mul/div and rank>2 last-dim broadcast via MPS composition, sum/prod/min/max reductions, softmax, logSoftmax, softmin, and logSoftmin, plus BF16 same-shape elementwise/scalar/device-scalar-broadcast/unary math (`abs/square/sqrt/exp/log/exp2/expm1/log1p/log2/log10/sin/cos/tan`), rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, and dot/inner/outer/matvec/vecmat, transpose, row/column broadcast add/sub/mul/div plus rank<=6 general broadcast add/sub/mul/div and rank>2 last-dim broadcast via MPS composition, sum/prod/min/max reductions, softmax, logSoftmax, softmin, and logSoftmin; remaining MPS dtype/shape coverage stays capability-gated.

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
    var picked = try y.indexSelect(1, picked_idx); // torch.indexSelect / np.take style
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

    var sales_col = try vx.DeviceColumn.fromSlice(f64, allocator, &.{ 2.0, 3.0, 5.0 }, vx.cpu);
    defer sales_col.deinit();
    var units_col = try vx.DeviceColumn.fromSliceWithValidity(i64, allocator, &.{ 1, 2, 3 }, &.{ true, false, true }, vx.cpu);
    defer units_col.deinit();
    var device_df = try vx.DeviceDataFrame.init(allocator, &.{
        .{ .name = "sales", .data = sales_col },
        .{ .name = "units", .data = units_col },
    });
    defer device_df.deinit();
    var table_view = try device_df.view(); // cuDF-like non-owning table metadata
    defer table_view.deinit();
    var expensive = try device_df.compareColumnScalar("sales", f64, 2.5, .gt);
    defer expensive.deinit();
    var filtered_device_df = try device_df.filterColumnMask(expensive);
    defer filtered_device_df.deinit();
    var lazy_df = try vx.DeviceLazyFrame.init(allocator, device_df);
    defer lazy_df.deinit();
    try lazy_df.withColumnScalar("sales_x2", "sales", f64, 2.0, .mul);
    try lazy_df.withColumnCompareScalar("expensive_x2", "sales_x2", f64, 6.0, .gt);
    try lazy_df.filterColumn("expensive_x2");
    try lazy_df.sortBy("sales", .{ .descending = true });
    try lazy_df.select(&.{ "sales", "units", "sales_x2", "expensive_x2" });
    var lazy_result = try lazy_df.collect();
    defer lazy_result.deinit();
    var lazy_grouped_df = try vx.DeviceLazyFrame.init(allocator, device_df);
    defer lazy_grouped_df.deinit();
    try lazy_grouped_df.groupBySum("units", "sales", "lazy_sales_sum");
    var lazy_grouped_result = try lazy_grouped_df.collect();
    defer lazy_grouped_result.deinit();
    var sorted_device_df = try device_df.sortBy("sales", .{ .descending = true });
    defer sorted_device_df.deinit();
    var top2_device_df = try device_df.topKBy("sales", 2, .{ .descending = true });
    defer top2_device_df.deinit();
    var stacked_device_df = try device_df.concatRows(filtered_device_df);
    defer stacked_device_df.deinit();
    var deduped_device_df = try stacked_device_df.distinctOn(&.{"sales"});
    defer deduped_device_df.deinit();
    var lazy_stack_df = try vx.DeviceLazyFrame.init(allocator, device_df);
    defer lazy_stack_df.deinit();
    try lazy_stack_df.concatRows(filtered_device_df);
    try lazy_stack_df.distinctOn(&.{"sales"});
    var lazy_stacked_result = try lazy_stack_df.collect();
    defer lazy_stacked_result.deinit();
    var grouped_device_df = try device_df.groupBySum("units", "sales", "sales_sum");
    defer grouped_device_df.deinit();
    var mean_device_df = try device_df.groupByMean("units", "sales", "sales_mean");
    defer mean_device_df.deinit();
    var stats_device_df = try device_df.groupByStats("units", "sales", "sales");
    defer stats_device_df.deinit();
    var stats_on_df = try device_df.groupByStatsOn(&.{"units"}, "sales", "sales");
    defer stats_on_df.deinit();
    var joined_device_df = try device_df.innerJoin(lookup_df, "units", "units", .{});
    defer joined_device_df.deinit();
    var joined_on_df = try device_df.innerJoinOn(lookup_df, &.{"units"}, &.{"units"}, .{});
    defer joined_on_df.deinit();
    var lazy_join_df = try vx.DeviceLazyFrame.init(allocator, device_df);
    defer lazy_join_df.deinit();
    try lazy_join_df.innerJoinOn(lookup_df, &.{"units"}, &.{"units"}, .{});
    var lazy_join_result = try lazy_join_df.collect();
    defer lazy_join_result.deinit();
    var left_joined_device_df = try device_df.leftJoin(lookup_df, "units", "units", .{});
    defer left_joined_device_df.deinit();
    var left_joined_on_df = try device_df.leftJoinOn(lookup_df, &.{"units"}, &.{"units"}, .{});
    defer left_joined_on_df.deinit();
    var full_joined_device_df = try device_df.fullJoin(lookup_df, "units", "units", .{});
    defer full_joined_device_df.deinit();
    var full_joined_on_df = try device_df.fullJoinOn(lookup_df, &.{"units"}, &.{"units"}, .{});
    defer full_joined_on_df.deinit();
    var semi_joined_device_df = try device_df.semiJoin(lookup_df, "units", "units");
    defer semi_joined_device_df.deinit();
    var semi_joined_on_df = try device_df.semiJoinOn(lookup_df, &.{"units"}, &.{"units"});
    defer semi_joined_on_df.deinit();
    var anti_joined_device_df = try device_df.antiJoin(lookup_df, "units", "units");
    defer anti_joined_device_df.deinit();
    var anti_joined_on_df = try device_df.antiJoinOn(lookup_df, &.{"units"}, &.{"units"});
    defer anti_joined_on_df.deinit();
    var asof_joined_df = try device_df.asofJoin(lookup_df, "units", "units", .{ .strategy = .nearest });
    defer asof_joined_df.deinit();
    var lazy_asof_df = try vx.DeviceLazyFrame.init(allocator, device_df);
    defer lazy_asof_df.deinit();
    try lazy_asof_df.asofJoin(lookup_df, "units", "units", .{ .strategy = .nearest });
    var lazy_asof_result = try lazy_asof_df.collect();
    defer lazy_asof_result.deinit();
    var arrow_batch = try device_df.toArrowRecordBatch(allocator); // Boltha/Arrow boundary
    defer arrow_batch.deinit(allocator);
    const parquet_bytes = try device_df.toParquetBytes(allocator); // Boltha/Parquet boundary
    defer allocator.free(parquet_bytes);
    var parquet_df = try vx.DeviceDataFrame.fromParquetBytes(allocator, parquet_bytes, vx.cpu);
    defer parquet_df.deinit();
    var pruned_df = try vx.DeviceDataFrame.fromParquetBytesPruned(
        allocator,
        parquet_bytes,
        "sales",
        .{ .f64 = .{ .min = 4.0 } },
        vx.cpu,
    );
    defer pruned_df.deinit();
    var scan = try vx.DeviceParquetScan.init(allocator, parquet_bytes, vx.cpu);
    defer scan.deinit();
    try scan.whereRange("sales", .{ .f64 = .{ .min = 4.0 } });
    try scan.select(&.{ "sales", "units" });
    var scanned_df = try scan.collect();
    defer scanned_df.deinit();
    var lazy_scan_df = try vx.DeviceLazyFrame.scanParquetBytes(allocator, parquet_bytes, vx.cpu);
    defer lazy_scan_df.deinit();
    try lazy_scan_df.filterColumnScalar("sales", f64, 4.0, .ge);
    try lazy_scan_df.select(&.{ "sales", "units" });
    var pushed_down_df = try lazy_scan_df.collect();
    defer pushed_down_df.deinit();
}
```

The fully explicit `vx.Array(T).fromSlice(allocator, ...)` and method surface is
still available when you need fine-grained control; the `vx.withAllocator(...)`
context and top-level `vx.add/vx.matmul/vx.sum/...` helpers are the intended
short-form front door for examples and application code. Ordinary array creation
and random creation do not require a seed: `try np.rand(f32, &.{ m, k })` uses the
context RNG stream. Creation helpers keep `dtype` as the explicit Zig type
parameter and use a fixed `CreationOptions` value in the final argument for
runtime device/seed metadata, e.g. `try np.zerosWith(f32, &.{ rows, cols }, vx.onDevice(vx.cuda(0)))`
or `try vx.Array(f32).fromSliceOn(allocator, values, dims, vx.cuda(0))`. `rand`
uses a Philox stream; `randWith(T, dims, vx.seededOn(device, seed))` generates
directly into supported CUDA/MPS device storage when that accelerator is selected,
never by CPU generation followed by upload. Arrays implement Zig's standard `{f}` formatter
with a PyTorch-like `tensor(...)` representation, e.g. `try writer.print("{f}", .{array})`;
`DeviceDataFrame` follows the same device model for tabular fixed-width data:
construct on `vx.cpu`, `vx.cuda(index)`, or `vx.mps(index)`, inspect zero-copy
`DeviceDataFrameView` metadata for backend bridges, and transfer the whole table
with `df.to(device)` / `df.cuda(index)` / `df.mps(index)` when the target is available.
`DeviceDataFrame.concatRows`/`appendRows`/`vstack` concatenate compatible
fixed-width tables by rows while preserving nullable validity.
`distinctRows`/`distinctOn` and `dropDuplicates*` keep the first row for each
full-row or subset-key combination; the row concat and distinct operations are
available in lazy plans as well.
`DeviceLazyFrame` stages derived-column expressions (`withColumnBinary`,
`withColumnScalar`, `withColumnCompare`, and `withColumnCompareScalar`), lazy
groupby aggregations (`groupByCount`, `groupBySum`/`Min`/`Max`/`Mean`,
`groupByStats`, multi-key `groupByStatsOn`, and grouped moment profiles through
`groupByProfile`/`groupByProfileOn`), lazy `rankProfileBy`,
lazy `rollingProfile`, lazy `rollingMomentProfile`, lazy `rollingRankProfile`, lazy `lagProfile`, lazy `expandingProfile`, lazy `expandingRankProfile`, lazy `expandingMomentProfile`,
lazy `standardizeProfile`, lazy `robustProfile`, lazy `expandingRobustProfile`, lazy `rollingRobustProfile`, lazy `drawdownProfile`,
lazy `rollingDrawdownProfile`,
lazy `trendProfile`, lazy `rollingTrendProfile`, lazy `expandingTrendProfile`, lazy `changePointProfile`, lazy `rollingChangePointProfile`, lazy `expandingChangePointProfile`, lazy `extremaProfile`, lazy `rollingRangeProfile`, lazy `crossoverProfile`, lazy `rollingCrossoverProfile`, lazy `expandingCrossoverProfile`,
lazy `linearFitProfile`, lazy `rollingLinearFitProfile`, lazy `expandingLinearFitProfile`, lazy `errorProfile`, lazy `rollingErrorProfile`, lazy `expandingErrorProfile`, lazy `rollingCorrelationProfile`, lazy `expandingCorrelationProfile`, lazy `validityProfile`, lazy `rollingValidityProfile`, lazy `expandingValidityProfile`,
lazy `classificationProfile`, lazy `rollingClassificationProfile`, lazy `expandingClassificationProfile`,
lazy `boolTransitionProfile`, lazy `rollingBoolTransitionProfile`, lazy `expandingBoolTransitionProfile`,
lazy `rollingBoolProfile`, lazy `expandingBoolProfile`,
lazy `signProfile`, lazy `rollingSignProfile`, lazy `expandingSignProfile`,
lazy `clipProfile`, lazy `rollingClipProfile`, lazy `expandingClipProfile`, lazy `thresholdProfile`, lazy `rollingThresholdProfile`,
lazy `expandingThresholdProfile`, lazy `leadProfile`,
lazy `rollingNormalizeProfile`, lazy `expandingNormalizeProfile`,
lazy `rollingQuantileProfile`, lazy `expandingQuantileProfile`,
lazy `emaProfile`,
lazy `bucketProfile`,
lazy `innerJoinOn`/`leftJoinOn`/`fullJoinOn`/`semiJoinOn`/`antiJoinOn` plus ordered `asofJoin`, lazy `concatRows`/`appendRows`/`vstack`, lazy `distinctRows`/`distinctOn`,
select/positional-select-drop/first-last-column-select/column-reverse-name-sort/name-pattern-select/name-pattern-drop/dtype-select/dtype-drop/nullability-select-drop/nan-presence-select-drop/inf-presence-select-drop/signed-inf-presence-select-drop/zero-presence-select-drop/signed-zero-presence-select-drop/non-zero-presence-select-drop/positive-presence-select-drop/signbit-presence-select-drop/negative-presence-select-drop/finite-presence-select-drop/normal-presence-select-drop/subnormal-presence-select-drop/non-finite-presence-select-drop/cast/fill-null/fill-nan/fill-inf/fill-signed-inf/fill-zero/fill-signed-zero/fill-non-zero/fill-positive/fill-signbit/fill-negative/fill-finite/fill-normal/fill-subnormal/fill-non-finite/abs-column/neg-column/square-column/reciprocal-column/sign-column/sqrt-column/rsqrt-column/cbrt-column/floor-column/ceil-column/round-column/trunc-column/deg2rad-column/rad2deg-column/expit-column/logit-column/softplus-column/logsigmoid-column/relu-column/leaky-relu-column/relu6-column/pow-scalar-column/floor-div-scalar-column/mod-scalar-column/remainder-scalar-column/log-add-exp-scalar-column/log-add-exp2-scalar-column/xlogy-scalar-column/fmax-scalar-column/fmin-scalar-column/hypot-scalar-column/atan2-scalar-column/next-after-scalar-column/copysign-scalar-column/heaviside-scalar-column/ldexp-scalar-column/lerp-scalar-column/addcmul-scalar-column/addcdiv-scalar-column/clip-array-column/where-scalar-column/where-column/isin-column/masked-put-scalar-column/put-flat-column/put-flat-scalar-column/put-flat-signed-scalar-column/put-flat-mode-scalar-column/isclose-scalar-column/allclose-scalar-gate/count-nonzero-column/zero-count-column/zero-ratio-column/nonzero-ratio-column/first-zero-index-column/last-zero-index-column/first-nonzero-index-column/last-nonzero-index-column/positive-zero-count-column/negative-zero-count-column/positive-zero-ratio-column/negative-zero-ratio-column/positive-count-column/negative-count-column/signbit-count-column/positive-ratio-column/negative-ratio-column/signbit-ratio-column/first-positive-index-column/last-positive-index-column/first-negative-index-column/last-negative-index-column/first-signbit-index-column/last-signbit-index-column/first-nan-index-column/last-nan-index-column/first-inf-index-column/last-inf-index-column/first-finite-index-column/last-finite-index-column/first-non-finite-index-column/last-non-finite-index-column/nan-count-column/inf-count-column/positive-inf-count-column/negative-inf-count-column/finite-count-column/normal-count-column/subnormal-count-column/non-finite-count-column/nan-ratio-column/inf-ratio-column/positive-inf-ratio-column/negative-inf-ratio-column/finite-ratio-column/normal-ratio-column/subnormal-ratio-column/non-finite-ratio-column/count-distinct-column/n-unique-column/null-count-column/valid-count-column/null-ratio-column/valid-ratio-column/first-valid-index-column/last-valid-index-column/first-null-index-column/last-null-index-column/mode-column/sum-column/prod-column/mean-column/median-column/quantile-column/variance-column/stddev-column/sem-column/cv-column/skewness-column/kurtosis-column/mean-abs-column/rms-column/l1-norm-column/l2-norm-column/geometric-mean-column/harmonic-mean-column/mad-column/iqr-column/min-column/max-column/ptp-column/argmin-column/argmax-column/bool-scalar-gate/logical-scalar-column/logical-column/threshold-column/hardtanh-column/maximum-scalar-column/minimum-scalar-column/clip-min-column/clip-max-column/hardshrink-column/softshrink-column/tanhshrink-column/elu-column/celu-column/softsign-column/hardsigmoid-column/hardswish-column/silu-column/swish-column/mish-column/gelu-column/selu-column/exp-column/exp2-column/expm1-column/sin-column/cos-column/tan-column/asin-column/acos-column/atan-column/sinh-column/cosh-column/tanh-column/asinh-column/acosh-column/atanh-column/log-column/log1p-column/lgamma-column/sinc-column/log2-column/log10-column/coalesce-column/drop-null/filter-null/null-predicate-column/nan-inf-signed-inf-zero-signed-zero-non-zero-positive-signbit-negative-finite-normal-subnormal-non-finite-predicate-column/drop-nan/filter-nan/drop-inf/filter-inf/drop-signed-inf/filter-signed-inf/drop-zero/filter-zero/drop-signed-zero/filter-signed-zero/drop-non-zero/filter-non-zero/drop-positive/filter-positive/drop-signbit/filter-signbit/drop-negative/filter-negative/drop-finite/filter-finite/drop-normal/filter-normal/drop-subnormal/filter-subnormal/drop-non-finite/filter-non-finite/row-null-valid-nan-inf-signed-inf-zero-signed-zero-non-zero-positive-signbit-negative-finite-normal-subnormal-count/literal-column/positioned-literal-column/copy-column/row-index/rename/bulk-rename/move/drop/boolean-column-filter/boolean-column-drop/where-indices-column/scalar-filter/filter-mask/sort/reverse/roll-row/shift-row/slice/signed-slice/signed-stepped-slice/drop-row/drop-row-mode/signed-drop-row/signed-drop-row-mode/drop-row-by-column/drop-row-by-column-mode/repeat-row/tile-row/repeat-row-by-column/stepped-slice/stride/sample/replacement-sample/take/optional-take/take-by-column/take-by-column-mode/take-mode/signed-take/signed-take-mode/head/tail operations over either an eager
dataframe source or a `DeviceParquetScan` source, exposes `explain()`, folds
adjacent select/head/tail operations and exposes lazy row reversal/slicing/dropping/stepped-slicing/striding/sampling/replacement-sampling/gathering, rewrites adjacent sort+head into `topKBy`,
and pushes conservative Parquet range/projection scan metadata into Boltha before
materializing source columns. Derived columns reuse Vectra `Array` arithmetic and
comparison kernels, so they keep the same CPU/CUDA/MPS dispatch seam as eager
expressions. It executes through `collect()` today, preserving a future Axiom
lowering boundary for query optimization/fusion.
Eager schema/expression helpers such as `selectByNamePrefix`, `selectByNameSuffix`,
`selectByNameContains`, `dropByNamePrefix`, `dropByNameSuffix`,
`dropByNameContains`, `selectByDTypes`, `selectNumeric`, `dropByDTypes`,
`dropNumeric`, `dropFloat`, `dropInteger`, `dropBool`, `selectNullableColumns`, `selectNonNullableColumns`, `selectColumnsWithNulls`, `selectColumnsWithoutNulls`, `dropNullableColumns`, `dropNonNullableColumns`, `dropColumnsWithNulls`, `dropColumnsWithoutNulls`, `selectColumnsWithNaNs`, `selectColumnsWithoutNaNs`, `dropColumnsWithNaNs`, `dropColumnsWithoutNaNs`, `selectColumnsWithInfs`, `selectColumnsWithoutInfs`, `dropColumnsWithInfs`, `dropColumnsWithoutInfs`, `selectColumnsWithPositiveInfs`, `selectColumnsWithoutPositiveInfs`, `dropColumnsWithPositiveInfs`, `dropColumnsWithoutPositiveInfs`, `selectColumnsWithNegativeInfs`, `selectColumnsWithoutNegativeInfs`, `dropColumnsWithNegativeInfs`, `dropColumnsWithoutNegativeInfs`, `selectColumnsWithZeros`, `selectColumnsWithoutZeros`, `dropColumnsWithZeros`, `dropColumnsWithoutZeros`, `selectColumnsWithPositiveZeros`, `selectColumnsWithoutPositiveZeros`, `dropColumnsWithPositiveZeros`, `dropColumnsWithoutPositiveZeros`, `selectColumnsWithNegativeZeros`, `selectColumnsWithoutNegativeZeros`, `dropColumnsWithNegativeZeros`, `dropColumnsWithoutNegativeZeros`, `selectColumnsWithNonZeros`, `selectColumnsWithoutNonZeros`, `dropColumnsWithNonZeros`, `dropColumnsWithoutNonZeros`, `selectColumnsWithPositives`, `selectColumnsWithoutPositives`, `dropColumnsWithPositives`, `dropColumnsWithoutPositives`, `selectColumnsWithSignBits`, `selectColumnsWithoutSignBits`, `dropColumnsWithSignBits`, `dropColumnsWithoutSignBits`, `selectColumnsWithNegatives`, `selectColumnsWithoutNegatives`, `dropColumnsWithNegatives`, `dropColumnsWithoutNegatives`, `selectColumnsWithFinites`, `selectColumnsWithoutFinites`, `dropColumnsWithFinites`, `dropColumnsWithoutFinites`, `selectColumnsWithNormals`, `selectColumnsWithoutNormals`, `dropColumnsWithNormals`, `dropColumnsWithoutNormals`, `selectColumnsWithSubnormals`, `selectColumnsWithoutSubnormals`, `dropColumnsWithSubnormals`, `dropColumnsWithoutSubnormals`, `selectColumnsWithNonFinites`, `selectColumnsWithoutNonFinites`, `dropColumnsWithNonFinites`, `dropColumnsWithoutNonFinites`, `castColumn`,
`withColumnAbs`, `withColumnNeg`, `withColumnNegative`, `withColumnSquare`, `withColumnReciprocal`, `withColumnSign`, `withColumnSqrt`, `withColumnRsqrt`, `withColumnCbrt`, `withColumnFloor`, `withColumnCeil`, `withColumnRound`, `withColumnTrunc`, `withColumnDeg2rad`, `withColumnRad2deg`, `withColumnExpit`, `withColumnLogit`, `withColumnSoftplus`, `withColumnLogsigmoid`, `withColumnRelu`, `withColumnLeakyRelu`, `withColumnRelu6`, `withColumnPowScalar`, `withColumnFloorDivScalar`, `withColumnModScalar`, `withColumnRemainderScalar`, `withColumnLogAddExpScalar`, `withColumnLogAddExp2Scalar`, `withColumnXlogyScalar`, `withColumnFmaxScalar`, `withColumnFminScalar`, `withColumnHypotScalar`, `withColumnAtan2Scalar`, `withColumnNextAfterScalar`, `withColumnCopysignScalar`, `withColumnHeavisideScalar`, `withColumnLdexpScalar`, `withColumnLerpScalar`, `withColumnAddcmulScalar`, `withColumnAddcdivScalar`, `withColumnClipArray`, `withColumnWhereScalar`, `withColumnWhere`, `withColumnIsIn`, `withColumnIsInInverted`, `withColumnIsin`, `withColumnIsinInverted`, `withColumnMaskedPutScalar`, `withColumnPutMaskScalar`, `withColumnPutFlat`, `withColumnPutFlatScalar`, `withColumnIndexPut`, `withColumnIndexPutScalar`, `withColumnPutFlatScalarSigned`, `withColumnIndexPutScalarSigned`, `withColumnPutFlatScalarMode`, `withColumnIscloseScalar`, `withColumnIscloseScalarEqualNan`, `allcloseColumnScalar`, `allcloseColumnScalarEqualNan`, `countNonzeroColumn`, `zeroCountColumn`, `countZeroColumn`, `zeroRatioColumn`, `nonzeroRatioColumn`, `nonZeroRatioColumn`, `firstZeroIndexColumn`, `lastZeroIndexColumn`, `firstNonzeroIndexColumn`, `lastNonzeroIndexColumn`, `positiveZeroCountColumn`, `negativeZeroCountColumn`, `positiveZeroRatioColumn`, `negativeZeroRatioColumn`, `positiveCountColumn`, `negativeCountColumn`, `signBitCountColumn`, `positiveRatioColumn`, `negativeRatioColumn`, `signBitRatioColumn`, `firstPositiveIndexColumn`, `lastPositiveIndexColumn`, `firstNegativeIndexColumn`, `lastNegativeIndexColumn`, `firstSignBitIndexColumn`, `lastSignBitIndexColumn`, `firstNanIndexColumn`, `firstNaNIndexColumn`, `lastNanIndexColumn`, `lastNaNIndexColumn`, `firstInfIndexColumn`, `lastInfIndexColumn`, `firstFiniteIndexColumn`, `lastFiniteIndexColumn`, `firstNonFiniteIndexColumn`, `lastNonFiniteIndexColumn`, `nanCountColumn`, `infCountColumn`, `positiveInfCountColumn`, `negativeInfCountColumn`, `finiteCountColumn`, `normalCountColumn`, `subnormalCountColumn`, `nonFiniteCountColumn`, `nanRatioColumn`, `infRatioColumn`, `positiveInfRatioColumn`, `negativeInfRatioColumn`, `finiteRatioColumn`, `normalRatioColumn`, `subnormalRatioColumn`, `nonFiniteRatioColumn`, `countDistinctColumn`, `nUniqueColumn`, `nullCountColumn`, `validCountColumn`, `nullRatioColumn`, `validRatioColumn`, `firstValidIndexColumn`, `lastValidIndexColumn`, `firstNullIndexColumn`, `lastNullIndexColumn`, `modeColumn`, `sumColumn`, `prodColumn`, `meanColumn`, `medianColumn`, `quantileColumn`, `varianceColumn`, `varColumn`, `stddevColumn`, `stdColumn`, `semColumn`, `cvColumn`, `skewnessColumn`, `skewColumn`, `kurtosisColumn`, `kurtColumn`, `meanAbsColumn`, `rmsColumn`, `l1NormColumn`, `l2NormColumn`, `geometricMeanColumn`, `geoMeanColumn`, `harmonicMeanColumn`, `harmMeanColumn`, `madColumn`, `medianAbsDevColumn`, `iqrColumn`, `minColumn`, `maxColumn`, `ptpColumn`, `argminColumn`, `argmaxColumn`, `anyColumn`, `allColumn`, `countTrueColumn`, `countFalseColumn`, `withColumnLogicalAndScalar`, `withColumnLogicalOrScalar`, `withColumnLogicalXorScalar`, `withColumnLogicalAnd`, `withColumnLogicalOr`, `withColumnLogicalXor`, `withColumnThreshold`, `withColumnHardtanh`, `withColumnMaximumScalar`, `withColumnMinimumScalar`, `withColumnClipMin`, `withColumnClipMax`, `withColumnHardshrink`, `withColumnSoftshrink`, `withColumnTanhshrink`, `withColumnElu`, `withColumnCelu`, `withColumnSoftsign`, `withColumnHardsigmoid`, `withColumnHardswish`, `withColumnSilu`, `withColumnSwish`, `withColumnMish`, `withColumnGelu`, `withColumnSelu`, `withColumnExp`, `withColumnExp2`, `withColumnExpm1`, `withColumnSin`, `withColumnCos`, `withColumnTan`, `withColumnAsin`, `withColumnAcos`, `withColumnAtan`, `withColumnSinh`, `withColumnCosh`, `withColumnTanh`, `withColumnAsinh`, `withColumnAcosh`, `withColumnAtanh`, `withColumnLog`, `withColumnLog1p`, `withColumnLgamma`, `withColumnSinc`, `withColumnLog2`, `withColumnLog10`, `fillNullColumn`, `fillNaNColumn`, `fillInfColumn`, `fillPositiveInfColumn`, `fillNegativeInfColumn`, `fillZeroColumn`, `fillPositiveZeroColumn`, `fillNegativeZeroColumn`, `fillNonZeroColumn`, `fillPositiveColumn`, `fillSignBitColumn`, `fillNegativeColumn`, `fillFiniteColumn`, `fillNormalColumn`, `fillSubnormalColumn`, `fillNonFiniteColumn`, `coalesceColumns`, `dropNulls`, `filterNullsColumn`, `dropNaNs`, `dropNaNsColumn`, `filterNaNsColumn`, `dropInfs`, `dropInfsColumn`, `filterInfsColumn`, `dropPositiveInfs`, `dropPositiveInfsColumn`, `filterPositiveInfsColumn`, `dropNegativeInfs`, `dropNegativeInfsColumn`, `filterNegativeInfsColumn`, `dropZeros`, `dropZerosColumn`, `filterZerosColumn`, `dropPositiveZeros`, `dropPositiveZerosColumn`, `filterPositiveZerosColumn`, `dropNegativeZeros`, `dropNegativeZerosColumn`, `filterNegativeZerosColumn`, `dropNonZeros`, `dropNonZerosColumn`, `filterNonZerosColumn`, `dropPositives`, `dropPositivesColumn`, `filterPositivesColumn`, `dropSignBits`, `dropSignBitsColumn`, `filterSignBitsColumn`, `dropNegatives`, `dropNegativesColumn`, `filterNegativesColumn`, `dropFinites`, `dropFinitesColumn`, `filterFinitesColumn`, `dropNormals`, `dropNormalsColumn`, `filterNormalsColumn`, `dropSubnormals`, `dropSubnormalsColumn`, `filterSubnormalsColumn`, `dropNonFinites`, `dropNonFinitesColumn`, `filterNonFinitesColumn`, `isNullColumn`, `isValidColumn`, `isNanColumn`, `isZeroColumn`, `isPositiveZeroColumn`, `isNegativeZeroColumn`, `isNonZeroColumn`, `isPositiveColumn`, `isSignBitColumn`, `isNegativeColumn`, `isInfColumn`, `isPositiveInfColumn`, `isNegativeInfColumn`, `isFiniteColumn`, `isNormalColumn`, `isSubnormalColumn`, `isNonFiniteColumn`, `withRowNullCount`, `withRowValidCount`, `withRowNaNCount`, `withRowInfCount`, `withRowPositiveInfCount`, `withRowNegativeInfCount`, `withRowZeroCount`, `withRowPositiveZeroCount`, `withRowNegativeZeroCount`, `withRowNonZeroCount`, `withRowPositiveCount`, `withRowSignBitCount`, `withRowNegativeCount`, `withRowFiniteCount`, `withRowNormalCount`, `withRowSubnormalCount`, `withRowNonFiniteCount`, `sliceRowsSigned`, `sliceSigned`, `sliceRowsSignedStep`, `sliceSignedStep`, `dropRows`, `dropRowsMode`, `dropRowsSigned`, `dropRowsSignedMode`, `dropRowRange`, `dropFirstRows`, `dropLastRows`, `dropRowsByColumn`, `dropRowsByColumnMode`, `dropRowsByColumnMask`, `whereIndicesColumn`, `argwhereColumn`, `take`, `takeOptional`, `takeOptionalRows`, `takeByColumn`, `takeRowsByColumn`, `takeByColumnMode`, `takeRowsByColumnMode`, `takeMode`, `takeSigned`, `takeSignedMode`, `rollRows`, `shiftRows`, `repeatRows`, `tileRows`, `repeatRowsByColumn`, `withColumnAt`, `withColumnBefore`, `withColumnAfter`,
`withColumnLiteral`, `withColumnLiteralAt`, `withColumnLiteralBefore`, `withColumnLiteralAfter`,
`copyColumn`, `copyColumnAt`, `copyColumnBefore`, `copyColumnAfter`,
`withRowIndex`, `renameColumn`, `renameColumns`, `addColumnNamePrefix`, `addColumnNameSuffix`,
`moveColumn`, `moveColumnBefore`, `moveColumnAfter`, `reverseColumns`, `sortColumnsByName`, `dropColumn`, `compareColumnScalar`, `addColumns`, `filterColumn`, and `filterColumnMask` reuse Vectra `Array` operations, so supported dtypes route
through the same Axiom CPU/CUDA/MPS dispatch instead of a CUDA-only dataframe
implementation. Nullable boolean predicate masks now follow query-engine semantics:
null predicate rows are treated as not selected rather than requiring a prefilled
host mask.
Sorting follows the cuDF/Polars pattern of producing a stable row order and then
gathering the whole table (`argsortBy` + `take` under `sortBy`/`topKBy`), which gives a
single future hook for replacing the current host order materialization with
Axiom sort/top-k kernels. `rankProfileBy` reuses that stable order to append
ordinal rank, competition rank, dense rank, percent rank, and cumulative
distribution columns in original-row order, honoring descending and null
placement options without forcing callers to choose one rank convention up
front.
`rollingProfile` appends trailing row-count-window count/sum/mean/population
variance/stddev columns for numeric fixed-width data. It honors nullable input
values and `min_periods`, marking insufficient windows null while leaving the
row count column valid, which gives time-series style feature engineering a
single future Axiom rolling-kernel seam.
`rollingMomentProfile` appends trailing count, third/fourth central moments,
skewness, and excess kurtosis for nullable numeric windows, sharing
`min_periods` semantics with other rolling profiles.
`rollingRankProfile` appends trailing valid-observation count, current-row rank,
percent rank, and cumulative distribution within each row-count window for
ordered fixed-width columns.
`rollingRangeProfile` appends trailing low/high/range and current-position
within range columns, honoring nullable inputs and `min_periods` for oscillator
and volatility-style features.
`rollingNormalizeProfile` appends trailing-window centered, z-score, and min-max
scaled values, propagating nulls and emitting NaN for zero-variance or
zero-range windows.
`expandingNormalizeProfile` appends cumulative centered, z-score, and min-max
scaled values with nullable `min_periods` semantics.
`rollingQuantileProfile` appends trailing q1/median/q3/IQR columns using valid
window observations and nullable outputs for insufficient windows.
`expandingQuantileProfile` appends cumulative q1/median/q3/IQR columns using
valid observations and nullable outputs until `min_periods` is met.
`rollingDrawdownProfile` appends trailing-window peak, drawdown, drawdown
percentage, and peak-age columns, honoring nullable rows and `min_periods` for
local risk/time-underwater diagnostics.
`rollingRobustProfile` appends trailing-window median-centered values,
MAD-based robust z-scores, Tukey-IQR outlier flags, and winsorized values for
nullable numeric windows.
`lagProfile` appends lag, absolute difference, and percent-change columns for a
configurable row offset. Nullable current/lagged inputs propagate to diff and
percent-change validity, and zero lagged values produce NaN percent changes
rather than panicking, giving sequence feature engineering a compact API seam.
`leadProfile` mirrors `lagProfile` with forward-looking lead, forward
difference, and forward percent-change columns, including null propagation and
NaN percent changes for zero current values.
`signProfile` appends sign, sign-flip, and positive/negative/zero streak
diagnostics for numeric sequences, resetting streaks across null observations.
`expandingProfile` appends cumulative count/sum/mean/min/max columns for numeric
fixed-width data, honoring nullable values and `min_periods` so cumulative
feature engineering can share the same future Axiom scan-kernel boundary.
`expandingRankProfile` appends cumulative valid-observation count, current-row
rank, percent rank, and cumulative distribution for ordered fixed-width columns.
`expandingRobustProfile` appends cumulative median-centered values, MAD-based
robust z-scores, Tukey-IQR outlier flags, and winsorized values for nullable
numeric prefixes.
`expandingMomentProfile` appends cumulative count, third/fourth central moments,
skewness, and excess kurtosis for nullable numeric sequences, preserving the
same `min_periods` semantics for distribution-shape diagnostics.
`standardizeProfile` appends centered, population z-score, and min-max scaled
columns from whole-column statistics, propagating nulls and emitting NaN for
zero-variance or zero-range scale factors instead of failing.
`robustProfile` appends median-centered values, MAD-based robust z-scores,
Tukey-IQR outlier flags, and winsorized values from nullable numeric columns,
providing outlier-resistant feature engineering without multiple passes through
the public dataframe API.
`drawdownProfile` appends running peak, absolute drawdown, and percentage
drawdown columns, honoring nullable sequence rows and `min_periods` for
risk/time-series style analyses.
`extremaProfile` appends running low/high and new-low/new-high flags, honoring
nullable sequence rows and `min_periods` for record-break diagnostics.
`trendProfile` appends signed trend direction plus up/down/flat streak lengths
and reversal flags for configurable row offsets, resetting streak state across
null or insufficient-history rows.
`crossoverProfile` appends two-column spread, ratio, cross-above, and cross-below
signal columns for same-dtype numeric columns, propagating nulls and producing
NaN ratios for zero denominators.
`rollingCrossoverProfile` appends trailing crossover counts, directional cross
rates, and mean absolute spread for same-dtype numeric column pairs, using
`DeviceCrossoverOptions.periods` for event lookback and `DeviceRollingOptions` for
window/min-period semantics.
`expandingCrossoverProfile` appends the cumulative counterpart of those crossover
summaries with `DeviceExpandingOptions.min_periods`, making long-horizon spread
and cross-rate monitoring available in eager and lazy plans.
`linearFitProfile` fits one global ordinary least-squares line between two
same-dtype numeric columns and appends fitted value, residual, residual z-score,
and slope diagnostics.
`rollingLinearFitProfile` fits a trailing ordinary least-squares line for each
row and appends valid-pair count, slope, intercept, current-row fitted value,
residual, and residual z-score columns, honoring nullable pairs and
`min_periods` for local model diagnostics.
`expandingLinearFitProfile` fits a cumulative ordinary least-squares line for
each row and appends valid-pair count, slope, intercept, fitted value, residual,
and residual z-score diagnostics for expanding model monitoring.
`errorProfile` appends forecast/evaluation diagnostics (`actual - predicted`,
absolute error, squared error, APE, and SMAPE) for same-dtype numeric columns,
propagating null pairs and using NaN for zero denominators.
`rollingErrorProfile` appends trailing valid-pair count, MAE, RMSE, MAPE, and
SMAPE summaries for same-dtype numeric forecast/evaluation columns.
`expandingErrorProfile` appends cumulative valid-pair count, MAE, RMSE, MAPE,
and SMAPE summaries for same-dtype numeric forecast/evaluation columns.
`rollingCorrelationProfile` appends trailing valid-pair count, population
covariance, Pearson correlation, and beta columns for same-dtype numeric column
pairs, propagating null pairs and emitting NaN for zero-variance windows.
`expandingCorrelationProfile` appends cumulative valid-pair count, population
covariance, Pearson correlation, and beta columns for same-dtype numeric column
pairs, skipping null pairs and sharing `min_periods` semantics.
`validityProfile` appends null/valid flags and consecutive valid/null streak
lengths for any fixed-width column, making data-quality diagnostics available in
eager and lazy plans.
`rollingValidityProfile` appends trailing window count, valid/null counts, and
valid/null rates for fixed-width columns, making null-density monitoring a
first-class eager/lazy operation.
`expandingValidityProfile` appends cumulative valid/null counts and rates with
`min_periods` validity, making long-horizon null-density monitoring available in
eager and lazy plans.
`classificationProfile` appends row-level TP/FP/TN/FN/correct diagnostics for
nullable boolean actual/predicted columns.
`rollingClassificationProfile` appends trailing confusion-matrix counts plus
accuracy, precision, and recall for nullable boolean actual/predicted columns.
`expandingClassificationProfile` appends cumulative confusion-matrix counts plus
accuracy, precision, and recall for long-horizon classifier monitoring.
`boolTransitionProfile` appends rising/falling/toggled flags and true/false
streak lengths for nullable boolean sequence columns.
`rollingBoolTransitionProfile` appends trailing transition counts and rates for
nullable boolean sequence columns, using `DeviceTrendOptions.periods` for the
lookback and `DeviceRollingOptions` for window/min-period semantics.
`expandingBoolTransitionProfile` appends the cumulative counterpart of those
transition counts and rates with `DeviceExpandingOptions.min_periods`.
`rollingBoolProfile` appends trailing true/false counts, true-rate, any, and
all columns for nullable boolean sequences. Counts remain valid audit columns
that ignore null observations, while rate/any/all require the current row to be
valid and the trailing window to meet `min_periods`.
`expandingBoolProfile` appends cumulative true/false counts, true-rate, any, and
all diagnostics for nullable boolean sequences using the same `min_periods`
validity convention.
`clipProfile` appends clipped values plus below/above/in-range flags for fixed
numeric bounds, propagating nulls and giving bounded data-cleaning a first-class
dataframe operation.
`rollingClipProfile` appends trailing valid-count, mean clipped value, clipped
rate, below/above rates, and in-range rate for bounded data-quality monitoring.
`expandingClipProfile` appends the cumulative counterpart, skipping nulls and
carrying forward accumulated clipping summaries once `min_periods` is reached.
`thresholdProfile` appends signed/absolute threshold distance and above/below/at
flags for scalar-threshold feature engineering.
`rollingThresholdProfile` appends trailing valid-count, mean signed/absolute
threshold distance, and above/below/at rates for scalar-threshold drift
monitoring. Count is always materialized, while rate/distance summaries honor
`window` and `min_periods` across nullable observations.
`expandingThresholdProfile` appends the cumulative counterpart for threshold
drift monitoring, carrying forward the accumulated state across null rows once
`min_periods` valid observations have been seen.
`bucketProfile` appends right-continuous empirical CDF, equal-frequency bucket,
and configurable lower/upper tail flags from a column's valid-value
distribution, enabling percentile-aware feature engineering without leaving the
dataframe API.
`emaProfile` appends exponential moving average, residual, and ratio columns for
configurable smoothing factors, skipping null observations so gaps do not update
the smoother state.
`groupByCount` plus numeric `groupBySum`/`groupByMin`/`groupByMax`/`groupByMean`
and one-pass single/multi-key `groupByStats`/`groupByStatsOn` provide the same
sort/hash-groupby API seam used by cuDF and Polars. `groupByProfile` and
`groupByProfileOn` extend that seam with count/sum/mean plus population
variance, standard deviation, skewness, and excess kurtosis from an online
moment accumulator, so heavy profiling scans do not need to schedule several
separate groupbys. Today these routines materialize keys/values through
`Array.toOwnedSlice` for all devices and keep null-key/null-value rows out of
the aggregate, while future Axiom lowering can replace the implementation
behind the same methods.
`innerJoinOn`/`leftJoinOn`/`fullJoinOn`/`semiJoinOn`/`antiJoinOn` plus
`innerJoin`/`leftJoin`/`fullJoin`/`semiJoin`/`antiJoin` similarly follow cuDF's
hash-join shape: build/probe row index pairs, gather the left/right tables as
needed, and then concatenate payload columns for joins that materialize right
payloads.  The current implementations skip null keys for matching, materialize
row pairs on the host for every device, and make unmatched left/full-join
payloads nullable, leaving a stable API seam for Axiom hash-join lowering.  The
`*JoinOn` variants accept multiple fixed-width key columns with matching dtype
pairs; the non-`On` join names remain single-key convenience wrappers.
`asofJoin` covers ordered nearest-key joins with previous/next/nearest
strategies, preserving all left rows and nullable right payloads when no
candidate exists.
Arrow/Parquet/IPC interoperability is intentionally delegated to the sibling
[`../boltha`](../boltha) package: `DeviceDataFrame.toArrowSchema`,
`toArrowRecordBatch`, and `toArrowTable` materialize Boltha-owned Arrow objects
from CPU/CUDA/MPS columns rather than reimplementing Arrow inside Vectra.
`toParquetBytes` and `fromParquetBytes` reuse Boltha's simple Parquet
reader/writer and allow readback directly onto the requested Vectra device.
`fromParquetBytesPruned` sends fixed-width numeric range predicates to
Boltha's statistics/Bloom-filter pruning reader before materializing the
surviving Arrow table as a `DeviceDataFrame`. `DeviceParquetScan.lazy()` and
`DeviceLazyFrame.scanParquetBytes(...)` reuse the same path for scalar-filter
pushdown and apply projection while crossing the Arrow -> Vectra device-column
boundary, so unused columns are not uploaded into CPU/CUDA/MPS arrays.
`toOwnedTensorString` returns the same format as an owned string.

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
CUDA when a CUDA device is available, macOS MPS arrays use Axiom's Metal/MPS
runtime ABI, and mixed-device operations fail with `InvalidDevice`.
CUDA-resident and MPS-resident owning `Array` storage are real device allocations
for supported backends; supported kernels consume those device pointers directly,
while unsupported operations return explicit errors or require an explicit
`.cpu()` transfer. `-Ddevice-host-fallback=true` explicitly opts CUDA/MPS
owning arrays into host generic fallback for diagnostics; the default is off.
Public backend capability reports and smoke/provenance
diagnostics are exposed through `vx.axiom_backend`; target-specific bridge
modules remain an internal implementation detail behind that facade.

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
device-aware creation followed by `vx.matmul` and `vx.matmulAdd` calls. It
documents `Y = A[M,K] * B[K,N]`, explicit `vx.matmulAdd`, the user-written
`tmp = A.matmul(B); Y = tmp.add(C)` / `tmp.sub(C)` forms, and follow-on
`sqrt(tmp + C)` / `exp(chain)` / `exp2(chain)` / `expm1(chain)` unary chains, emitting
one JSON result per backend/dtype/op. The checked-in execute size is a CUDA
stress run (`M = 4096 * 4`, `N = 4096`, `K = 4096`) and dry-runs by default;
pass `-- --smoke` for a tiny executable check, `-- --dtype=all --backend=both`
to compare f32/f64/f16/BFloat16 CPU paths plus supported CUDA paths, or
`-- --execute --backend=cuda --dtype=f32 --require-cuda` for the production CUDA
benchmark. CUDA owning arrays benchmark f32/f64/f16/BFloat16 matmul,
matmulAdd, automatic matmul-then-add/sub fusion, and supported sqrt/exp unary
chains through Axiom cuBLAS/cuBLASLt-backed device GEMM plus cached CUDA unary
post-op routes. `--retain-outputs`
intentionally keeps each iteration output alive to expose allocation/reuse effects
versus the default PyTorch-like single-output reuse loop.


## Axiom dialect lowering layer

The architecture-first roadmap is documented in [`docs/AXIOM_ARCHITECTURE_PLAN.md`](docs/AXIOM_ARCHITECTURE_PLAN.md). New acceleration work should move Vectra toward that descriptor/linalg/memref/gpu plan rather than adding isolated short-term backend branches.

Vectra now routes array-compute intent directly into the sibling [`../axiom`](../axiom) package instead of depending on a separate facade package.  The front door follows an MLIR-like progression: Vectra describes Array operations as Axiom `linalg` operations over `memref` storage contracts, Axiom attaches `gpu` launch metadata for accelerator backends, and Axiom-owned passes lower those records toward CPU, CUDA, or MPS runtime paths.  The smoke gate for this route is:

```sh
zig build axiom-dialect-lowering-smoke
zig build axiom-descriptor-smoke
```

The current public evidence API is `vx.axiom_backend.lowerMatmulDialect(...)`, which reports registered linalg/memref/gpu dialect counts, operation-store fingerprints, schedule fingerprints, CUDA Tile/NVVM handoff fingerprints for CUDA, and an explicit `planned_mps` status for MPS operation-kernel lowering while Axiom's MPS runtime ABI reports real Metal storage availability on macOS. These lowering helpers now validate structural array contracts separately from eager runtime storage contracts, so CPU arrays and CUDA-tagged/device arrays can both describe the same linalg/memref/gpu program while runtime capability reports remain honest. `vx.axiom_backend.lowerElementwiseDialect(...)` reports the same linalg.generic/schedule/vector/gpu route for same-shape add/sub/mul/div; `vx.axiom_backend.lowerReductionDialect(...)` reports linalg.generic/vector/gpu reductions for row/column sum/max/min/prod while `vx.axiom_backend.reductionRuntimeCapability(.cuda)` reports executable f32/f64/f16/BFloat16 2D `sum/prod/min/max` and capability-gates other CUDA reduction dtypes/ops; `vx.axiom_backend.lowerBroadcastAddDialect(...)` reports row/column broadcast lowering, `vx.axiom_backend.broadcastAddRuntimeCapability(.cpu)` reports executable CPU row/column bias-add through Axiom CPU->Veyra, and `broadcastAddRuntimeCapability(.cuda/.mps)` reports executable row/column broadcast add/sub/mul/div for supported dtypes while other broadcast dtypes/shapes remain gated; `vx.axiom_backend.lowerUnaryDialect(...)` reports linalg.generic copy/square/cube plus abs/sqrt/exp/log unary routes while `vx.axiom_backend.unaryRuntimeCapability(.cpu/.cuda, .log)` reports executable f32/f64 CPU and f32 CUDA log coverage. Axiom MPS runtime capability now reports executable f32/f16/BFloat16 unary math (`abs/square/sqrt/exp/log/exp2/expm1/log1p/log2/log10/sin/cos/tan`), f32/f16 contiguous 2D matmul/matmulAdd through `MPSMatrixMultiplication`, BF16 2D matmul/matmulAdd through Metal kernels because Apple `MPSMatrixMultiplication` rejects BF16 input matrices, f32/f16/BFloat16 matmul-then-add/sub/sqrt/exp storage chains, rank-3 BMM, higher-rank equal-batch BMM, rank-3 whole-batch broadcast BMM, rank-4/rank-5/rank-6 mixed-batch BMM, rank-3/rank-4 batched matvec/vecmat, 2D plus flat/all-axes reductions and `ptp`, row/column broadcast add/sub/mul/div, transpose, softmax, logSoftmax, logsumexp, composed activation chains (`rsqrt/leakyRelu/silu/hardsigmoid/hardswish/softshrink/tanh/tanhshrink/gelu/elu/celu/selu/SELU/relu6/hardtanh/clipArray`), composed `powScalar(-1/-0.5/0/0.5/1/2/3)`, composed metric helpers (`norm/normalize/cosineSimilarity/pairwiseDistance`), and composed statistics helpers (`mean/variance/stddev`) plus focused rank-3 `sum/prod/min/max` and partial-axes `variance/stddev` reduction, rank-3/rank-4/rank-5 broadcast, rank-4/rank-5/rank-6 mixed-batch BMM, and rank-3/rank-4 batched matvec/vecmat smokes over Metal shared-buffer storage; remaining MPS dtypes/shapes stay capability-gated while lowering reports still use `planned_mps` for the structural MPS GPU route. `vx.axiom_backend.lowerTransposeDialect(...)` reports 2D transpose lowering while `vx.axiom_backend.transposeRuntimeCapability(.cuda/.mps)` reports executable 2D transpose for supported dtypes/targets and gates other transpose dtypes/shapes; `vx.axiom_backend.softmaxRuntimeCapability(.cuda/.mps)` reports executable 2D axis softmax for supported dtypes/targets while other softmax dtypes/shapes remain gated; `vx.axiom_backend.logSoftmaxRuntimeCapability(.cuda/.mps)` reports executable 2D axis logSoftmax for supported dtypes/targets while other logSoftmax dtypes/shapes remain gated; and `vx.setDefaultDialectBackend(.cpu/.cuda/.mps)` switches the default dialect-lowering target dynamically for helpers that use the default route, while CPU eager Array dispatch maps `.mps` defaults back to CPU unless the Array already owns real MPS storage.


## Axiom accelerator backend

Vectra imports the sibling [`../axiom`](../axiom) package by default. Supported
CPU-backed `Array(f32/f64)` same-shape and scalar/broadcast add/sub/mul/div plus sqrt/exp/exp2/expm1/log/log1p/log2/log10/sin/cos/tan/asin/acos/atan/square,
Axiom-composed `powScalar(-1/-0.5/0/0.5/1/2/3)`, 2D row/column broadcast add/sub/mul/div, 2D axis reductions (`sum/prod/min/max`), 2D transpose, 2D matmul, matrix-vector, vector-matrix, dot/vdot, trace, determinant, inverse,
solve, Cholesky, QR, LU, triangular solve, Frobenius/one/inf/two/nuclear matrix norms, SVD, singular values, matrix rank, condition number, pseudo-inverse, least-squares, symmetric eigen decomposition (`eigh`), and Hermitian eigenvalues (`eigvalsh`) flow through Axiom CPU lowering to Veyra. Supported CUDA owning-array
f32 add/sub/mul/div/exp2/expm1/log1p/log2/log10/sin/cos/tan/sum/prod/min/max(axis=0/1)/broadcast-add(row/column)/transpose/softmax(axis=0/1)/logSoftmax(axis=0/1)/maximum/minimum/addcmul/addcdiv/lerp/neg/abs/reciprocal/square/sqrt/rsqrt/exp/relu/threshold/leakyRelu/relu6/clip/clipArray/elu/celu/sigmoid/silu/hardsigmoid/hardswish/softsign/softshrink/powScalar(-1/-0.5/0/0.5/1/2/3)/mseLoss(.none)/l1Loss(.none)/smoothL1Loss(.none)/huberLoss(.none), f32 matmul/fused matmul+add, f64 same-shape/scalar elementwise plus softmax(axis=0/1)/logSoftmax(axis=0/1)/sum/prod/min/max(axis=0/1)/broadcast-add(row/column)/transpose/maximum/addcmul/addcdiv/lerp/neg/abs/reciprocal/square/sqrt/rsqrt/exp/relu/threshold/leakyRelu/relu6/clip/clipArray/elu/celu/sigmoid/silu/hardsigmoid/hardswish/softsign/softshrink/powScalar(-1/-0.5/0/0.5/1/2/3)/mseLoss(.none)/l1Loss(.none)/smoothL1Loss(.none)/huberLoss(.none), DGEMM matmul/matmulAdd, and f16 and BFloat16 sum/prod/min/max(axis=0/1), broadcast-add(row/column), transpose, softmax(axis=0/1), logSoftmax(axis=0/1), matmul/fused
matmul+add plus widened elementwise/activation/powScalar(-1/-0.5/0/0.5/1/2/3) combinations such as relu/sigmoid/softsign/clip use existing device pointers through Axiom CUDA. Large f32 GEMM and
GEMM+add use Axiom's cached cuBLAS-backed SGEMM wrapper; BFloat16 GEMM uses Axiom's cuBLAS `cublasGemmEx` BF16 device wrapper and BFloat16 GEMM+add uses cuBLASLt with separate C/D pointers to avoid pre-copying the addend. The Axiom PTX/CUDA Tile
IR seeds remain as fallback/provenance paths.

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

## Random Backends

Vectra uses Philox for the `rand` stream so CPU, CUDA, and MPS can share one deterministic algorithm. CUDA `randWith(f32/f64/f16/BFloat16, ..., vx.seededOn(device, seed))` and MPS `randWith(f32/f16/BFloat16, ..., vx.seededOn(device, seed))` write directly into device storage; MPS also supports direct f32/f16/BFloat16 `normalWith`/`randnWith`. The sibling [`../alea`](../alea) Zig package remains the local path dependency for higher-level distribution helpers such as Bernoulli, exponential, gamma, beta, Poisson, multinomial, Dirichlet, log-normal, Student-t, Cauchy, Laplace, Weibull, half-normal, chi-squared, Erlang, Fisher-F, triangular, arcsine, logistic/log-logistic, Kumaraswamy, power-function, Rayleigh, Maxwell, Pareto, Gumbel, Frechet, skew-normal, PERT, inverse-Gaussian, and normal-inverse-Gaussian generation until those distributions grow device-side Philox kernels.

## Veyra backend

Vectra uses the sibling [`../veyra`](../veyra) Zig package as a local path dependency for foundational math and linear algebra, but supported CPU tensor paths should enter through Axiom first. Current f32/f64 `Array`/`linalg` paths for matrix multiplication, matrix-vector products, dot/vdot, trace, determinant, inverse, solve, Cholesky, QR, LU, triangular solve, Frobenius/one/inf/two/nuclear matrix norms, SVD, singular values, matrix rank, condition number, pseudo-inverse, least-squares, symmetric eigen decomposition, and Hermitian eigenvalues route through Axiom CPU→Veyra. Non-covered dtypes and non-contiguous/batched Array methods keep generic in-core fallbacks where implemented.

## Forge interop boundary

Vectra remains an Array/NDArray data and numerical interop layer rather than a
training framework. It exposes `vx.forge_interop` /
`vx.forgeArrayInteropBoundary` as a data-only manifest describing how Forge may
wrap Vectra Array metadata without Vectra importing or depending on Forge. Core
Forge operation lowering should flow through `Forge IR -> Axiom dialect/runtime`; Vectra
should not become the lowering path for Forge Tensor, autograd, module, optimizer,
or training/inference semantics.

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
python3 tools/bench_matmul_add_compare.py --smoke --op matmulAdd --repeat 2
python3 tools/bench_matmul_add_compare.py --smoke --op matmulAdd --dtype f16
python3 tools/bench_matmul_add_compare.py --execute --m 16384 --n 4096 --k 4096 --warmup 3 --iters 5 --max-ratio 1.05
zig build bench-matmul-add-compare-smoke
zig build bench-matmul-add-compare-production
zig build bench-matmul-add-compare-production-compile
zig build bench-matmul-add-compare-bf16-large
zig build bench-matmul-add-compare-bf16-stability
zig build bench-matmul-add-compare-f64-exp-large
zig build fusion-smoke
zig build fusion-production-gate
zig build bench-matmul-add-compare -- --execute --m 16384 --n 4096 --k 4096 --warmup 3 --iters 5 --max-ratio 1.05
```

The current high-value benchmark set covers large f64 elementwise/scalar ops, flat reductions, promoted i32+f64 arithmetic, strided scalar/array ops, f64 dot/inner/outer/matvec/vecmat, and 256x256 f64 matmul.
`tools/bench_matmul_add_compare.py` emits JSONL rows for the same CUDA dtype
(`--dtype=f32/f64/f16/bf16`) and
shape through Vectra/Axiom `large_matmul_add`, PyTorch `torch.addmm`, eager
`a @ b + c`, and `torch.compile`, so matmul+add performance work has a
repeatable local PyTorch/torch.compile baseline. Pass `--op` to isolate one
Vectra expression when tuning kernel launch/algorithm effects; the default
`--baseline=auto` selects the matching PyTorch expression for matmul, add, sub,
sqrt, or exp variants; `--baseline=torch_best` gates against the fastest emitted
PyTorch eager/addmm/compile row for that op. Pass `--repeat` to run the full comparison multiple
times and gate on the worst observed ratio, then pass `--baseline` and
`--max-ratio` to turn the comparison into a failing performance gate based on
the selected Vectra op's average time divided by the selected PyTorch baseline.
Use `--max-first-error` and `--max-checksum-error` to make correctness drift fail
alongside performance regressions.

CUDA owning `Array(f32/f64/f16/BFloat16).matmul` may return a pending
matmul/fusion handle so a following add/sub/unary can fuse into one backend
operation.  Timing only `a.matmul(b)` can therefore measure handle creation
rather than GEMM completion.  Use `out.hasPendingWork()` to detect this state,
then call `out.materializeAndSynchronize()` (or download with `out.cpu()`) when
you need a clear completion point.  The `large_matmul_add` benchmark already
materializes pending CUDA matmul results and synchronizes the device before it
reports elapsed time.

## Roadmap

- Descriptor-first Axiom lowering/runtime convergence; see [`docs/AXIOM_ARCHITECTURE_PLAN.md`](docs/AXIOM_ARCHITECTURE_PLAN.md).
- Broader view-aware kernels and more simple-stride fast paths on top of the current `ArrayView`/`NDArrayView` storage model.
- Nullable values, categorical/string kernels and richer promotion policy.
- Polars-like lazy query plans and expression DSL.
- BLAS/LAPACK/high-performance FFT/sparse integrations.
- Broader GPU coverage behind the existing `Device` surface; see [`docs/AXIOM_CUDA_BRIDGE.md`](docs/AXIOM_CUDA_BRIDGE.md) for the current Axiom CUDA backend surface.
- Arrow/Parquet IPC support.
