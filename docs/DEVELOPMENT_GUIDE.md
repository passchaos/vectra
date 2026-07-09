# Vectra 后续开发指导

本文件记录本项目初始化阶段和用户明确确认过的功能开发方向，后续实现、重构、测试和 API 设计应优先遵循这里的顺序与原则。

## 1. 总体目标

Vectra 目标是在 Zig 中实现一套完整的数据处理与数值计算库，能力长期对标 Python 生态中的：

- NumPy / CuPy
- PyTorch
- SciPy
- Pandas / Polars

接口应尽量让熟悉 Python 数值计算生态的用户容易迁移；但在同一件事情 NumPy 与 PyTorch 都有常见接口时，优先考虑更符合 PyTorch 使用习惯的类型/对象式 API，不再保留重复的 NumPy 风格过程式数组包装。

## 2. 功能开发优先级

后续开发必须按以下顺序推进，不要过早把主要精力放到 DataFrame 或上层分析功能上。

### P0：数组核心，类比 NumPy、CuPy、PyTorch

这是当前最高优先级。

优先完善：

- 数组数据结构：shape、stride、dtype、device、内存布局、metadata helpers（`dim/rank/numDims/num_dims`、`nelement`、`shapeAt/shape_at/strideAt/stride_at`、`nbytes/num_bytes/elementSize/element_size/itemsize`、`storageOffset/storage_offset`、`dataPtr/data_ptr`、`isEmpty/is_empty/is_contiguous`、`isMatrix/isSquare/isBatchedMatrix`、scalar/vector predicates（`isScalar/isVector/isRowVector/isColumnVector/isVectorLike` 及 snake_case aliases）、scalar/flat export（`item/itemValue/item_value`、`scalarValue/scalar_value`、`asSlice/as_slice`、`asConstSlice/as_const_slice`、`copyToSlice/copy_to_slice`、`toOwnedSlice/to_owned_slice`）、storage span/sharing metadata（`storageSize/storage_size`、`storageNbytes/storage_nbytes`、`storageSpan/storage_span`、`storageRange/storage_range`、`storageEndOffset/storage_end_offset`、`sharesStorage/sameStorage`、`mayOverlap` 及 Array/View variants）、shape comparison（`sameShape/same_shape`、`shapeEquals/shape_equals`、`hasShape/has_shape`、Array/View cross-shape helpers）、`broadcastShape/broadcastShapes/broadcastWith`、对象式 device 查询如 `Device.isCpu/isCuda/backendName/sameDevice/isAvailable` 与数组/视图上的 `deviceBackend/deviceIndex/deviceBackendName/isCpu/isCuda/isDeviceAvailable/sameDevice`、对象式 dtype 查询如 `dtypeName/dtype_name`、`dtypeTag/dtype_tag`、`dtypeByteSize/dtype_byte_size`、`dtypeBitSize/dtype_bit_size`、`is*Dtype`、`canCastToDtype/can_cast_to_dtype`）和深拷贝 helpers（`clone/copy/detach`）、contiguous/non-contiguous view；当前已有 `ArrayView` / `NDArrayView` 非 owning 视图基础，后续继续把更多 kernel 做成 view-aware。
- 创建函数：通过 `Array(T)` / `NDArray(T)` 的类型方法创建，例如 `fromSlice`、`zeros`、`ones`、`empty`、`full`、like helpers、`newEmpty/newZeros/newOnes/newFull`、`eye/identity/eyeRect`、`arange`、`linspace`、`logspace`、`geomspace`、`meshgrid`、随机初始化等。
- 索引与切片：整数索引、range/slice、bool mask、坐标索引、take/gather/scatter、IndexMode raise/wrap/clip、masked/index put、compress、advanced indexing、membership/search helpers。
- 形状变换：reshape/view/reshapeAs/viewAs、flatten/ravel/flattenRange/flattenFrom、squeeze/unsqueeze/squeezeDim/squeezeAxes/unsqueezeDim/unsqueezeAxes/expandDims、transpose、matrixTranspose/matrix_transpose/mT、adjoint/mH/H_、matrixPower/matrix_power、permute/swapaxes/swapDims/movedim/moveaxis/moveaxes、broadcast、repeat/tile（含 NumPy-like leading-rank 对齐）、slice/sliceAxis/slice1d、split/splitWithSizes/splitAtIndices/chunk/unbind、hstack/vstack/dstack/columnStack、flip、roll、padConstant。
- 广播逐元素运算：加减乘除、幂、floorDiv/mod/remainder、hypot/atan2、copysign/heaviside、比较及 scalar 比较（`equalScalar/greaterScalar/lessScalar/...`）、逻辑运算及 scalar 逻辑、where、clip/clamp/clipArray、maximum/minimum。
- 归约与统计：sum/sumAxes、prod/prodAxes、min/amin/minAxes/aminAxes、max/amax/maxAxes/amaxAxes、ptp/ptpAxes、mean/meanAxes、var/varianceAxes、std/stddevAxes、median/medianAxes、quantile/quantileAxes/percentile/percentileAxes；同类 API 同时提供 PyTorch-style `dim/keepdim` 对象式别名（如 `sumDim/sumDims`、`meanDim/meanDims`、`varDim/varDims`、`stdDim/stdDims`、`argmaxDim/argminDim`、`topkDim/kthValueDim`、`softmaxDim/logSoftmaxDim`、`normDim/normDims`、`logsumexpDim/logsumexpDims`、累计 `*Dim`、nan-aware `nan*Dim/nan*Dims`、`countNonzeroDim/countNonzeroDims`、`allDim/allDims`、`anyDim/anyDims` 与 snake_case variants）；weighted mean/var/std/quantile/cov/corrcoef、cov/corrcoef、nan_to_num/nanToNumDefault 与 nan-aware 统计/cov/corrcoef、argmin、argmax、nanargmin、nanargmax、cumsum、cumprod、cummax、cummin。
- 排序/选择：任意 axis/dim 的 `sort/argsort`、descending variants、`sortWithIndices`、`partition/argpartition`、`kthValue/kth_value`、`topk`。
- 线性代数基础：dot、inner/vecdot/vdot、outer、cross、contractAxes、matmul/mm、matvec、bmm、norm、solve/inverse/det/eig/svd/qr/cholesky 等逐步补齐；数学/线性代数底层优先复用相邻 `../veyra` 库。
- 数学/神经网络常用函数：neg/abs/square/reciprocal/sign、nextAfter/ldexp/frexp、exp/exp2/log/log2/log10/lgamma/gammaln/sqrt/rsqrt/cbrt、log1p/expm1、floor/ceil/round/trunc、deg2rad/rad2deg、sinc/logit/expit、三角/反三角/双曲函数（含 asinh/acosh/atanh）、copysign/heaviside、isnan/isinf/isfinite/isnormal、relu、sigmoid、tanh、softmax、log_softmax、cross_entropy 相关基础。
- dtype 转换与类型提升规则：当前支持 `bool`、常用有符号/无符号整数、`isize/usize`、`BFloat16`、`f16/f32/f64`、`Complex64/Complex128`、对象式 dtype 查询（`dtypeName/dtype_name`、`dtypeTag/dtype_tag`、`dtypeByteSize/dtype_byte_size`、`dtypeBitSize/dtype_bit_size`、`isFloatDtype/isIntegerDtype/isSignedDtype/isUnsignedDtype/isComplexDtype/isBoolDtype/isRealDtype/isNumericDtype`、`canCastToDtype/can_cast_to_dtype`）、`canCastDType`、`promoteDType/resultDType`、`promoteType` 与一批 promoted mixed-dtype/complex/bf16 运算；后续继续补更完整 promotion。
- 设备抽象：先保持 CPU 正确；CUDA/GPU API 形态参考 CuPy/PyTorch，后续再接入真实后端。

API 取向：

- 数组 API 以类型/对象方法为主，例如 `Array(f64).zeros(...)`、`x.reshape(...)`、`x.softmax(axis)`、`x.matmul(y)`；不要新增重复的过程式数组包装。
- `axis` 与 `dim` 语义都要考虑；Zig 中如名称冲突可在实现中用 `axis_index` / `axis_opt`，文档可解释为 NumPy 的 axis / PyTorch 的 dim。
- 先保证 CPU 版本正确、可测试、API 稳定，再做 SIMD、BLAS、GPU 等性能后端。

### P1：SciPy 风格数值算法

在数组核心足够稳定后，再系统扩展 SciPy 类能力。

优先模块：

- `linalg`：更完整的矩阵分解、求解器、范数、特征值、SVD。
- `stats`：分布、相关性、假设检验、描述统计、zscore、概率密度/累积分布。
- `optimize`：root finding、最小化、least squares。
- `signal`：卷积、滤波、FFT 相关接口。
- `sparse`：稀疏矩阵格式与运算。

原则：SciPy 模块应建立在 Array/NDArray 核心之上，不要绕过核心数据结构单独设计一套数组表示。

### P2：DataFrame / 表格数据，类比 Pandas、Polars

Pandas/Polars 能力排在数组与 SciPy 之后。

后续扩展方向：

- Series/DataFrame 类型系统：nullable、categorical、string、date/time、list/struct dtype。
- 行列选择、过滤、排序、join、groupby、aggregation、pivot/melt。
- Polars 风格表达式 DSL 与 lazy query plan。
- CSV/JSON/Arrow/Parquet IO。
- 与 Array/NDArray 的互转：数值列转矩阵、数组结果回填列。

原则：DataFrame 不应抢占 Array/NDArray 核心的开发优先级；它应复用底层 array kernel 和 dtype 规则。

## 3. 已讨论并确认的 API 风格

- 使用方式尽量接近 Python 数值计算生态。
- 同类操作如 PyTorch 风格更自然，则更多参考 PyTorch，而不是机械照搬 NumPy。
- 顶层模块名暂用 `vectra` / `vx` 风格。
- 当前阶段允许 Zig 显式 allocator 与 `defer deinit()`，但应在示例中保持模式清晰。
- 错误处理使用 Zig error union，不隐藏内存/shape/device 错误。
- CUDA/CuPy/PyTorch-like device API 可以先占位（`Array`/`ArrayView.to/cpu/cuda`），但不能假装 GPU 已可用；未实现时应明确返回错误。

## 4. 当前已落地的基础

初始化阶段已优先实现一批 Array/NDArray 能力：

- 创建：`Array(T)` / `NDArray(T)` 类型方法，包括 `fromSlice`、`fromScalar`、`zeros`、`ones`、`full`、`empty`、`eye/identity/eyeRect`、`arange`、`linspace`、`logspace`、`geomspace`、`meshgrid(MeshGridIndexing.xy/ij)`、`rand`、`randn`、`randint`，Alea-backed `uniform/normal/bernoulli/exponential/gamma/beta/poisson/multinomial/dirichlet/lognormal/studentT/cauchy/laplace/weibull/halfNormal/chiSquared/chi/erlang/fisherF/triangular/arcsine/logistic/logLogistic/kumaraswamy/powerFunction/rayleigh/maxwell/pareto/gumbel/frechet/skewNormal/pert/inverseGaussian/normalInverseGaussian`，以及对象方法 `emptyLike/zerosLike/onesLike/fullLike`。
- 形状：对象/类型方法 `reshape/view`、`reshapeInfer/viewInfer`（单个 `-1` 推断维度）、`flatten/ravel`、`flattenAxes`、`unflatten`、`atLeast1d/atLeast2d/atLeast3d`、`squeeze/unsqueeze`、`transpose`、`matrixTranspose/matrix_transpose/mT`、`adjoint/mH/H_`、`matrixPower/matrix_power`、`permute`、`swapaxes`、`movedim`、`broadcastTo/broadcast_to/expand/expandAs`、`repeat`、`repeatInterleave/repeat_interleave/repeatInterleaveScalar/repeat_interleave_scalar`、`tile`、`slice/sliceAxis/slice1d`、`split/chunk`、`hstack/vstack/dstack/columnStack`、`flip/flipAxes/fliplr/flipud`、`roll/rollFlat/rollAxes`、`rot90`、`padConstant/padEdge/padReflect/padWrap/padSymmetric`；shape/device metadata 已包含 `num_dims`、`is_empty`、scalar/vector predicates、scalar/flat export helpers、storage span/sharing helpers、shape comparison helpers、Device/Array/ArrayView device queries；非复制视图基础包括 `ArrayView/NDArrayView`（非 owning、共享存储、保留 offset/stride，用于 slice/transpose/broadcast 等零拷贝视图）、`asView`、安全 `asStrided`、PyTorch-like `unfold`、`sliceAxisView/slice_axis_view/sliceView/slice_view`、`selectView/select_view/selectSignedView/select_signed_view/narrowView/narrow_view`、`narrowSigned/narrowSignedView/narrow_signed_view`、`unfoldView/unfold_view`、`permuteView/permute_view`、`swapaxesView/swapaxes_view`、`swapDimsView/swap_dims_view`、`movedimView/movedim_view`、`moveaxisView/moveaxis_view`、`moveaxesView/moveaxes_view/move_axes_view`、`transposeView/transpose_view/TView/T_view`、`matrixTransposeView/matrix_transpose_view/mTView/mT_view`、`matrixPower/diagonalView/diagonalAxesView`、zero-copy reshape/flatten aliases（`reshapeView/reshapeInferView`、`flattenView/flattenAxesView/flattenRangeView/flattenFromView`、`ravelView`、`unflattenView` 及 snake_case variants）、zero-copy squeeze/unsqueeze aliases（`squeezeView/squeezeDimView/squeezeAxesView`、`unsqueezeView/unsqueezeDimView/unsqueezeAxesView`、`expandDimsView`、`atLeast1dView/atLeast2dView/atLeast3dView` 及 snake_case variants）、`view/viewInfer`、zero-copy broadcast/expand aliases（`broadcastView/broadcast_view`、`broadcastToView/broadcast_to_view`、`expandView/expand_view`、`expandToView/expand_to_view`、`expandAs*`/`broadcastAs*` Array/View helpers）、`atLeast1d/atLeast2d/atLeast3d`、共享存储 mutation 与 `toArray/contiguous` materialization。
- 索引/搜索：`get/at`、`set/put`、scalar signed negative-index variants（`getSigned/get_signed/atSigned/at_signed`、`setSigned/set_signed/putSigned/put_signed`、`selectSigned/select_signed`）、`select`、`narrow/narrowSigned`、`take/indexSelect`、batch signed negative-index variants（`takeSigned/take_signed/takeSignedMode/take_signed_mode`、`indexSelectSigned/index_select_signed`、`takeAlongAxisSigned/take_along_axis_signed`、`gatherSigned/gather_signed`、`putFlatSigned`、`putFlatScalarSigned`）、`takeMode/takeSignedMode(IndexMode.raise/wrap/clip)`、`takeAlongAxis/putAlongAxis`、支持前缀形状坐标数组的坐标索引 `ravelCoords/unravelFlat/takeCoords/putCoords/putCoordsScalar`、支持广播坐标数组的 `ravelMultiIndex/takeMultiIndex/putMultiIndex/putMultiIndexScalar`、`gather`、`scatter/scatterScalar`、`scatterAdd/scatterReduce`、`scatterReduceScalar/scatterAddScalar`、`maskedSelect/masked_select`、`maskedFill/masked_fill`、`maskedScatter/masked_scatter`、`maskedPut/masked_put/maskedPutScalar/masked_put_scalar`、`putMask/put_mask/putMaskScalar/put_mask_scalar`、`copyWhere/copy_where/where/whereScalar/where_scalar`、对象式 mask 原地赋值（`maskedFillAssign`、`maskedCopyFrom`、`maskedCopyFromView`、`copyWhereAssign`、`copyWhereAssignView` 以及 `ArrayView.maskedFill/maskedCopyFrom*/copyWhereFrom*`）、`whereIndices`、`putFlat/put_flat/putFlatMode/putFlatScalar/put_flat_scalar/putFlatScalarMode`、`indexPut/index_put/indexPutScalar/index_put_scalar`、`compress`、`flatNonzero`、`nonzero/argwhere/countNonzero/countNonzeroAxis/count_nonzero`、`isin`、`searchsorted`、`bucketize`、`digitize`、`slice1d`。
- 广播与逐元素：`add/sub/mul/div/pow`、promoted mixed-dtype variants（`addPromote`、`subPromote`、`mulPromote`、`divPromote`、`maximumPromote`、`minimumPromote`）、complex helpers（`real`、`imag`、`conj/conjugate`、`magnitude/absComplex`、`angle/phase`、`isreal/iscomplex`、complex `exp/log/sqrt/sin/cos/tan/...` 与 complex finiteness predicates）、`floorDiv`、`mod/remainder`、scalar variants、`maximum/minimum`、`fmax/fmin`、`hypot`、`atan2`、`logAddExp/logaddexp`、`logAddExp2/logaddexp2`、`xlogy`、`copysign`、`heaviside`、`where/whereScalar/whereMask`、`eq/equal`、`ne/notEqual`、`gt/greater`、`ge/greaterEqual`、`lt/less`、`le/lessEqual`、scalar 比较（`equalScalar/greaterScalar/lessScalar/...`）、`logicalNot` 与 view-aware `logicalAnd/logicalOr/logicalXor`、scalar 逻辑、view-aware `isclose/isClose`/equal-NaN variants、`allclose/allClose`/equal-NaN variants；对象式原地赋值包括 `fill/fill_/zero_`、`copyFrom/copyFromView/copyFromArray/copy_`、`add/sub/mul/divAssign`、`add/sub/mul/divAssignView`、PyTorch-style mutating aliases（`add_/sub_/mul_/div_`、scalar `addScalar_/add_scalar_` 等）和 masked/copy-where mutating aliases（`masked_fill_`、`masked_copy_from`、`copy_where_`），可用于 `Array` 与 `ArrayView`。
- 数学/NN：`neg/negative`、`positive`、`abs/absolute/fabs`、`astype`、promoted mixed-dtype variants（`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`）、`square`、`reciprocal`、`sign/signbit`、`nextAfter/nextafter`、`ldexp`、`frexp`、`exp/exp2/expm1`、`log/log1p/log2/log10/lgamma/gammaln`、`sqrt/rsqrt/cbrt`、`floor`、`ceil`、`round`、`trunc`、`deg2rad/radians`、`rad2deg/degrees`、`sinc`、`logit`、`expit`、`sin`、`cos`、`tan`、`asin/arcsin`、`acos/arccos`、`atan/arctan`、`atan2/arctan2`、`hypot`、`copysign`、`heaviside`、`sinh`、`cosh`、`tanh`、`asinh/arcsinh`、`acosh/arccosh`、`atanh/arctanh`、`relu/leakyRelu`、`sigmoid`、`softplus`、`softsign`、`gelu`、`softmax`、`logsumexp`、`logcumsumexp`、`logSoftmax/log_softmax`、`clip/clamp`、`clipMin/clipMax/clampMin/clampMax`、`lerp`、`addcmul/addcdiv`、`fmax/fmin`、`isNan/isnan`、`isInf/isinf`、`isPosInf/isposinf`、`isNegInf/isneginf`、`isFinite/isfinite`、`isNormal/isnormal`。
- 归约/统计：对象方法 `sum/sumAxes/sumToSize`、`prod/prodAxes`、`min/amin/minAxes/aminAxes`、`max/amax/maxAxes/amaxAxes`、`ptp/ptpAxes`、`allAxis/allAxes/anyAxis/anyAxes`、`mean/meanAxes`、`variance/varianceAxes`、`stddev/stddevAxes`、`median/medianAxes`、`quantile/quantileAxes`、`percentile/percentileAxes`，并提供 `dim/keepdim` 风格别名（`sumDim/sumDims`、`prodDim/prodDims`、`minDim/aminDim`、`maxDim/amaxDim`、`ptpDim/ptpDims`、`meanDim/meanDims`、`varianceDim/varDim`、`stddevDim/stdDim`、`medianDim/medianDims`、`quantileDim/quantileDims`、`percentileDim/percentileDims` 及 snake_case）；`weightedMean`、`average`、`weightedVariance/weightedVar`、`weightedStddev/weightedStd`、`weightedQuantile`、`weightedMedian`、`weightedCov`、`weightedCorrcoef`、`cov`、`corrcoef`、`nanToNum/nan_to_num/nanToNumDefault/nan_to_num_default`、`nansum/nansumAxes/nansumDim/nansumDims`、`nanmean/nanmeanAxes/nanmeanDim/nanmeanDims`、`nanvar/nanvarAxes/nanvarDim/nanvarDims`、`nanstd/nanstdAxes/nanstdDim/nanstdDims`、`nanmin/nanminAxes/nanminDim/nanminDims`、`nanmax/nanmaxAxes/nanmaxDim/nanmaxDims`、`nanmedian/nanmedianAxes/nanmedianDim/nanmedianDims`、`nanquantile/nanquantileAxes/nanquantileDim/nanquantileDims`、`nanpercentile/nanpercentileAxes/nanpercentileDim/nanpercentileDims`、`nanCov`、`nanCorrcoef`、`norm/normAxes/normDim/normDims`、`logsumexp/logsumexpAxes/logsumexpDim/logsumexpDims`、`logcumsumexp`、`cumsum`、`cumprod`、`cummax`、`cummin`、`cumsumAxis/cumsumDim`、`cumprodAxis/cumprodDim`、`cummaxAxis/cummaxDim`、`cumminAxis/cumminDim`、`diff/diffWith/ediff1d`、`gradient`、`trapezoid/trapz`、`argmin`、`argmax`、`argminAxis/argmaxAxis/argminDim/argmaxDim`、`nanargmin/nanargmax`、`nanargminAxis/nanargmaxAxis/nanargminDim/nanargmaxDim`。
- ArrayView 支持更多对象式 materializing 包装：`softmax/logSoftmax`、`norm`、`sort/sortBy/sortDescending`、`argsort/argsortAxis`、`topk`、`matmul/matmulArray`、`bmm`、`matvec`、`dot`、`vdot/vecdot`、`inner`、`outer`、`cross`、`contractAxes`、`convolve1d/correlate1d/convolve2d/correlate2d`、`trace/traceOffset/traceAxes`、`diagonal/diagonalAxes/diag/diagflat`、`triu/tril`、对象式 linalg 包装（`det`、`inverse/inv`、`solve`、`cholesky`、`qr`、`lu`、`solveTriangular`、`svd`、`lstsq`、`singularValues`、`matrixRank`、`cond`、`pinv`、`matrixNorm`、`eigh`、`eigvalsh`）、complex helpers（`real`、`imag`、`conj`、`magnitude`、`angle/phase`、`isreal/iscomplex`），用于让非连续视图在常用计算路径中直接可用。
- ArrayView 也支持常用 dtype/逐元素/广播 materializing 包装：`astype`、promoted mixed-dtype variants（`addPromote/subPromote/mulPromote/divPromote/maximumPromote/minimumPromote`）、view-aware `neg/negative`、`positive`、`abs/absolute/fabs`、`square`、`reciprocal`、`sign/signbit`，并已将超越 unary 数学直接改为按视图 stride 遍历（`exp/exp2/expm1`、`log/log1p/log2/log10/lgamma/gammaln`、`sqrt/rsqrt/cbrt`、`floor/ceil/round/trunc`、`deg2rad/radians`、`rad2deg/degrees`、`sinc/logit/expit`、三角/双曲函数，含 `arcsin/arccos/arctan` 与 `asinh/acosh/atanh/arcsinh/arccosh/arctanh`）、activation unary（`relu/leakyRelu`、`sigmoid`、`softplus`、`softsign`、`gelu`）和 `clip`，`clipMin/clipMax/clampMin/clampMax`、direct view-aware scalar elementwise variants（`powScalar/floorDivScalar/modScalar`、`maximumScalar/minimumScalar`、`fmaxScalar/fminScalar`、`hypotScalar/atan2Scalar`、`logAddExpScalar/logAddExp2Scalar/xlogyScalar`、`nextAfterScalar/copysignScalar/heavisideScalar`、`ldexpScalar`）、`pow/floorDiv/mod/remainder`、`maximum/minimum`、`fmax/fmin`、`hypot/atan2`、`logAddExp/logAddExp2/xlogy`、`lerp`、`addcmul/addcdiv`、`nextAfter/copysign/heaviside`、`clipArray`、view-aware `isNan/isInf/isFinite/isNormal/signbit/isReal/iscomplex`、`isclose/iscloseScalar/allclose/allcloseScalar`、comparison aliases（`equal/greater/less`）、direct view-aware bool `logicalNot/logicalAnd/logicalOr/logicalXor`、scalar logical 和 Array logical wrappers。
- ArrayView 统计包装包括 view-aware `sum/prod/min/max/ptp/mean/variance/stddev` flat、单轴与多轴 reductions、`sumAxes/prodAxes/minAxes/aminAxes/maxAxes/amaxAxes/ptpAxes/meanAxes/varianceAxes/stddevAxes` 与对应 `sumDim/sumDims/...` PyTorch-style aliases、`min/amin`、`max/amax`、`ptp`、`variance/stddev`、view-aware `median/quantile/percentile` flat 与单轴 reductions、`medianAxes/medianDim`、`quantileAxes/quantileDim`、`percentileAxes/percentileDim`、`average`、weighted statistics（`weightedMean/weightedVariance/weightedVar/weightedStddev/weightedStd/weightedQuantile/weightedMedian`）、view-aware nan-aware flat 与单轴 reductions（`nanToNum/nan_to_num/nanToNumDefault/nan_to_num_default`、`nansum/nansumAxes/nansumDim`、`nanmean/nanmeanAxes/nanmeanDim`、`nanvar/nanvarAxes/nanvarDim`、`nanstd/nanstdAxes/nanstdDim`、`nanmin/nanminAxes/nanminDim`、`nanmax/nanmaxAxes/nanmaxDim`、`nanmedian/nanmedianDim`、`nanquantile/nanquantileDim`、`nanpercentile/nanpercentileDim`）、`logsumexp/logsumexpDim`、covariance/correlation wrappers（`cov/corrcoef/weightedCov/weightedCorrcoef/nanCov/nanCorrcoef`）、累计/积分操作（`cumsum/cumprod/cummax/cummin/logcumsumexp/cumsumAxis/cumsumDim/cumprodAxis/cumprodDim/cummaxAxis/cummaxDim/cumminAxis/cumminDim/logcumsumexpAxis/logcumsumexpDim/diff/gradient/trapezoid/trapz`）和 arg reductions（`argmin/argmax/argminAxis/argmaxAxis/argminDim/argmaxDim/nanargmin/nanargmax/nanargminAxis/nanargmaxAxis/nanargminDim/nanargmaxDim`）以及 bool `all/any/allAxis/anyAxis/allDim/anyDim`。
- ArrayView 也支持对象式 materializing 索引/搜索/形状包装：metadata/like/new allocation helpers、`repeat/repeatInterleave/repeatInterleaveScalar`、NumPy-like `tile`、`flip/flipAxes/fliplr/flipud`、`roll/rollFlat/rollAxes`、`rot90`、`padConstant/padEdge/padReflect/padWrap/padSymmetric`、`slice1d`、`split/splitWithSizes/splitAtIndices/chunk/unbind`、`take/takeSigned`、`takeAlongAxis/takeAlongAxisSigned/putAlongAxis`、`indexSelect/indexSelectSigned`、`gather/gatherSigned`、`putFlat/putFlatMode/putFlatScalar*`、`indexPut/indexPutScalar`、coordinate and multi-index helpers、`scatter/scatterScalar/scatterAdd/scatterReduce*`、`maskedSelect`、`where/whereScalar`、`compress`、`nonzero/argwhere/whereIndices/countNonzeroAxis/count_nonzero`、`unique/uniqueWithCounts`、`union1d/intersect1d/setdiff1d/setxor1d`、`bincount/bincountWeighted`、`histogram`、`searchsorted`、`bucketize`、`digitize`、`isin`。
- 排序/选择：`sort`、`sortBy`、`sortDescending`、`argsort`、`argsortAxis`、`argsortDescending`、`sortWithIndices`、`partition`、`argpartition`、`topk(sorted=true/false)`。
- 离散/计数/集合：`unique`、`uniqueWithCounts`、`union1d`、`intersect1d`、`setdiff1d`、`setxor1d`、`bincount`、`bincountWeighted`。
- 组合/矩阵/信号基础辅助：`cat/concatenate`、`stack`、`diag/diagflat/diagEmbed`、`diagonal/diagonalAxes/diagonalView/diagonalAxesView/fillDiagonal/diagonalScatter`、`trace/traceOffset/traceAxes`、`diagEmbed/diag_embed`、`triu/tril`、对象式矩阵谓词（`isDiagonalMatrix`、`isUpperTriangular`、`isLowerTriangular`、`isSymmetric`、`isHermitian`）、`matrixPower/matrix_power`、`det`、`inverse/inv`、`solve`、`cholesky`、`qr`、`lu`、`solveTriangular/solve_triangular`、`svd`、`lstsq`、`singularValues/singular_values`、`matrixRank/matrix_rank`、`cond`、`pinv`、`matrixNorm/matrix_norm`、`eigh`、`eigvalsh`、`dot`、`inner/vecdot/vdot`、`outer`、`cross`、`contractAxes`、支持 1D/2D/batched broadcasting 的广义 `matmul/mm`、`matvec/bmm`、1D/2D `convolve*` / `correlate*` 与 `ConvMode.full/same/valid`、real `rfft/irfft`、complex `fft/ifft`、`fftAxis/ifftAxis`、`fftAxes/ifftAxes`、`fft2/ifft2`；`linalg.matmul/matvec/det/solve/inverse/lu/solveTriangular/cholesky/qr/svd/lstsq/singularValues/matrixRank/cond/pinv/matrixNorm/eigh/eigvalsh`。
- 基础 linalg/stats/DataFrame 也有初版，但它们不是下一阶段最高优先级。

## 5. 后续每次开发的执行要求

- 优先补 Array/NDArray 测试，再补上层模块测试。
- 新 API 应同时覆盖：正常路径、shape mismatch、axis/dim 错误、空数组或边界情况。
- 任何新功能都应运行相关验证；常规至少运行：

```sh
zig fmt build.zig build.zig.zon src/*.zig
zig build test
```

- 如果目录是 git 仓库，完成变更后应只提交本次相关文件；如果不是 git 仓库则不提交。

## 6. Veyra 后端使用策略

用户明确要求：数学计算（尤其线性代数）可以使用相邻目录 `../veyra` 库，并应扫描其可复用能力。当前扫描结论：

- `veyra.dense`：提供 `Matrix/Vector`、`matmul`、`matvec`、`trace`、row/column sums/norms/statistics、triangular solve、GEMM/GEMV 及多种优化内核。
- `veyra.decomp`：提供 LU、Cholesky、LDLT、QR、SVD、Hessenberg/eigen 相关分解与 solve/inverse/condition number。
- `veyra.sparse`：提供 CSR/CSC/BSR 稀疏矩阵、转换、稀疏 matvec/matmat、稀疏三角求解和统计。
- `veyra.iterative`：提供 CG/PCG 等迭代求解能力。

后续策略：

- Array/NDArray API 保持用户友好的 NumPy/CuPy/PyTorch 风格；新文档和新 API 主名统一使用 `Array` / `NDArray`、`array` / `ndarray`。
- 底层 f64 dense linalg 优先委托 Veyra；当前对象式 `Array.det/inverse/solve/cholesky/qr/lu/solveTriangular/svd/lstsq/singularValues/matrixRank/cond/pinv/matrixNorm/eigh/eigvalsh` 与 `linalg.matmul/matvec/trace/det/solve/inverse/lu/solveTriangular/cholesky/qr/svd/lstsq/singularValues/matrixRank/cond/pinv/matrixNorm/eigh/eigvalsh` 已接入 Veyra-compatible 路径；非 f64 或 Veyra 暂无覆盖时保留 Vectra 泛型回退，暂无回退的高级分解会显式返回错误。
- SciPy-like `linalg/sparse/optimize` 扩展应优先检查 Veyra 是否已有对应算法，避免重复实现。
- 引入 Veyra 时必须保留 Vectra 层 shape/device/dtype 错误语义，并添加端到端测试。

## 7. Alea 随机后端与命名边界

用户明确要求：随机数处理可以使用相邻目录 `../alea`，该库已基本覆盖所需随机数相关功能；Vectra 不应自行重复实现随机分布内核。当前扫描结论：

- `alea.Rng` 与多种 deterministic engine：`DefaultPrng`、`ScalarPrng`、`FastPrng`、`Pcg64`、`Xoshiro*`、`ChaCha*` 等。
- `alea.distributions` 覆盖 uniform、Bernoulli、normal/log-normal、exponential、Poisson、gamma、beta、Student-t、Dirichlet、multinomial、half-normal、chi/chi-squared、Erlang、Fisher-F、triangular、arcsine、logistic/log-logistic、Kumaraswamy、power-function、Rayleigh、Maxwell、Pareto、Gumbel、Frechet、skew-normal、PERT、inverse-Gaussian、normal-inverse-Gaussian、weighted sampling、unit geometry 等大量分布，`alea.Rng` 还提供 shuffle、sampleWithoutReplacement、weightedIndex 等采样基础。
- Vectra 当前 `rand/randn/uniform/normal/randint/bernoulli/permutation/shuffle/shuffleInPlace/choice/choiceWeighted/exponential/gamma/beta/poisson/multinomial/dirichlet/lognormal/studentT/cauchy/laplace/weibull/halfNormal/chiSquared/chi/erlang/fisherF/triangular/arcsine/logistic/logLogistic/kumaraswamy/powerFunction/rayleigh/maxwell/pareto/gumbel/frechet/skewNormal/pert/inverseGaussian/normalInverseGaussian` 已接入 Alea seeded scalar streams、Alea 分布对象或 Alea 采样工具。

Array IO / serialization 当前支持：

- `toBytes/fromBytes`：裸数据字节 roundtrip。
- `toArchive/fromArchive`：包含 magic/version/dtype/rank/shape/data 的简单二进制归档格式。
- `saveArchive/saveArchiveToDir` 与 `loadArchive/loadArchiveFromDir`：基于 Zig 0.16 `std.Io.Dir` 的对象式文件归档读写。

命名边界：

- 本库不应引入自动微分/训练/推理框架式命名，因为这些能力属于相邻 `../forge`。
- 新文档和新 API 主名应使用 `Array` / `NDArray`、`array` / `ndarray`。
- 不再保留深度学习框架式数组命名作为兼容别名；历史调用应迁移到 `Array` / `array`。
- 自动微分、深度学习训练/推理相关能力应放到相邻 `../forge` 框架中处理。

## 8. Sparse / CSR 当前支持

当前已开始接入 SciPy-like sparse 能力，优先复用 `../veyra.sparse`：

- `CsrMatrix(T)`：Vectra 自有 CSR 所有权包装。
- `csrFromDense`：从 dense Array/NDArray 生成 CSR。
- `csrFromCompressed`：从 row_offsets / col_indices / values 构建 CSR。
- `CsrMatrix.toDense()`：CSR 转回 dense Array。
- `CsrMatrix.matvec()`：f64 路径复用 `veyra.csrMatvec`，其它 numeric dtype 保留泛型回退。
- `CsrMatrix.matmat()`：f64 路径复用 `veyra.csrMatmat`。
- `CsrMatrix.transpose()`：CSR 转置。
- `sum/absSum/frobeniusNorm/density`：基础 sparse 统计。
- `rowNnz/columnNnz`、`rowSums/columnSums`、`rowAbsSums/columnAbsSums`、`rowNorms/columnNorms`：CSR 行列统计，f64 路径优先复用 Veyra。
- `diagonal/trace/missingDiagonalCount/zeroDiagonalCount/bandwidth/structurallySymmetric/numericallySymmetric`：CSR 结构诊断，f64 路径优先复用 Veyra。
- `transposeMatvec/transposeMatmat`：CSR 转置乘法，f64 路径复用 Veyra。
- `solveTriangular`：CSR sparse triangular solve，支持 vector/matrix RHS，f64 路径复用 Veyra。

后续 sparse 扩展应继续优先检查 Veyra：CSC/BSR、sparse matmat、sparse triangular solve、iterative solvers。

CSC 当前支持：

- `CscMatrix(T)`：Vectra 自有 CSC 所有权包装。
- `cscFromDense` / `cscFromCompressed`：构建 CSC。
- `CscMatrix.toDense()` / `toCsr()`：dense/CSR 转换。
- `matvec/matmat`：f64 路径复用 `veyra.cscMatvec/cscMatmat`。
- `transposeMatvec/transposeMatmat`：CSC 转置乘法，f64 路径复用 Veyra。
- `rowNnz/columnNnz`、`rowSums/columnSums`、`rowNorms/columnNorms`、`density`：CSC 行列统计。
- `sum/frobeniusNorm`：基础 CSC 统计。
- `diagonal/trace/missingDiagonalCount/zeroDiagonalCount/bandwidth/structurallySymmetric/numericallySymmetric`：CSC 结构诊断，f64 路径优先复用 Veyra。
- `solveTriangular`：CSC sparse triangular solve，支持 vector/matrix RHS，f64 路径复用 Veyra。
