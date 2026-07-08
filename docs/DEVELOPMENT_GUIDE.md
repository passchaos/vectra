# Vectra 后续开发指导

本文件记录本项目初始化阶段和用户明确确认过的功能开发方向，后续实现、重构、测试和 API 设计应优先遵循这里的顺序与原则。

## 1. 总体目标

Vectra 目标是在 Zig 中实现一套完整的数据处理与数值计算库，能力长期对标 Python 生态中的：

- NumPy / CuPy
- PyTorch
- SciPy
- Pandas / Polars

接口应尽量让熟悉 Python 数值计算生态的用户容易迁移；但在同一件事情 NumPy 与 PyTorch 都有常见接口时，优先考虑更符合 PyTorch 使用习惯的链式/方法式 API，同时保留 NumPy 风格的顶层函数包装。

## 2. 功能开发优先级

后续开发必须按以下顺序推进，不要过早把主要精力放到 DataFrame 或上层分析功能上。

### P0：数组 / 张量核心，类比 NumPy、CuPy、PyTorch

这是当前最高优先级。

优先完善：

- 张量数据结构：shape、stride、dtype、device、内存布局、contiguous/non-contiguous view。
- 创建函数：`tensor`、`zeros`、`ones`、`empty`、`full`、`eye`、`arange`、`linspace`、随机初始化等。
- 索引与切片：整数索引、range/slice、bool mask、take/gather/scatter、advanced indexing。
- 形状变换：reshape/view、flatten/ravel、squeeze/unsqueeze、transpose/permute/movedim、broadcast、repeat/tile。
- 广播逐元素运算：加减乘除、幂、比较、逻辑运算、where、clip、maximum/minimum。
- 归约：sum、prod、min、max、mean、var、std、argmin、argmax、cumsum、cumprod。
- 线性代数基础：dot、matmul/mm、bmm、outer、norm、solve/inverse/det/eig/svd/qr/cholesky 等逐步补齐。
- 神经网络常用数学函数：relu、sigmoid、tanh、softmax、log_softmax、cross_entropy 相关基础。
- dtype 转换与类型提升规则：尽量接近 NumPy/PyTorch 直觉。
- 设备抽象：先保持 CPU 正确；CUDA/GPU API 形态参考 CuPy/PyTorch，后续再接入真实后端。

API 取向：

- 张量对象方法优先，例如 `x.reshape(...)`、`x.softmax(axis)`、`x.matmul(y)`。
- 同时提供顶层函数包装，例如 `vx.zeros(...)`、`vx.stack(...)`、`vx.cat(...)`。
- `axis` 与 `dim` 语义都要考虑；Zig 中如名称冲突可在实现中用 `axis_index` / `axis_opt`，文档可解释为 NumPy 的 axis / PyTorch 的 dim。
- 先保证 CPU 版本正确、可测试、API 稳定，再做 SIMD、BLAS、GPU 等性能后端。

### P1：SciPy 风格数值算法

在张量核心足够稳定后，再系统扩展 SciPy 类能力。

优先模块：

- `linalg`：更完整的矩阵分解、求解器、范数、特征值、SVD。
- `stats`：分布、相关性、假设检验、描述统计、zscore、概率密度/累积分布。
- `optimize`：root finding、最小化、least squares。
- `signal`：卷积、滤波、FFT 相关接口。
- `sparse`：稀疏矩阵格式与运算。

原则：SciPy 模块应建立在 Tensor/Array 核心之上，不要绕过核心数据结构单独设计一套数组表示。

### P2：DataFrame / 表格数据，类比 Pandas、Polars

Pandas/Polars 能力排在数组/张量与 SciPy 之后。

后续扩展方向：

- Series/DataFrame 类型系统：nullable、categorical、string、date/time、list/struct dtype。
- 行列选择、过滤、排序、join、groupby、aggregation、pivot/melt。
- Polars 风格表达式 DSL 与 lazy query plan。
- CSV/JSON/Arrow/Parquet IO。
- 与 Tensor 的互转：数值列转矩阵、张量结果回填列。

原则：DataFrame 不应抢占 Tensor 核心的开发优先级；它应复用底层 array kernel 和 dtype 规则。

## 3. 已讨论并确认的 API 风格

- 使用方式尽量接近 Python 数值计算生态。
- 同类操作如 PyTorch 风格更自然，则更多参考 PyTorch，而不是机械照搬 NumPy。
- 顶层模块名暂用 `vectra` / `vx` 风格。
- 当前阶段允许 Zig 显式 allocator 与 `defer deinit()`，但应在示例中保持模式清晰。
- 错误处理使用 Zig error union，不隐藏内存/shape/device 错误。
- CUDA/CuPy/PyTorch-like device API 可以先占位，但不能假装 GPU 已可用；未实现时应明确返回错误。

## 4. 当前已落地的基础

初始化阶段已优先实现一批 Tensor/Array 能力：

- 创建：`tensor`、`zeros`、`ones`、`full`、`empty`、`eye`、`arange`、`linspace`、`rand`、`randn`、`randint`。
- 形状：`reshape/view`、`flatten/ravel`、`squeeze/unsqueeze`、`transpose`、`permute`、`swapaxes`、`movedim`。
- 索引：`get/at`、`set/put`、`select`、`narrow`、`take/indexSelect`、`maskedSelect`、`slice1d`。
- 广播与逐元素：`add/sub/mul/div/pow`、scalar variants、`maximum/minimum`、`whereMask`、比较、`allclose`。
- 数学/NN：`exp`、`log`、`sqrt`、`sin`、`cos`、`tanh`、`relu`、`sigmoid`、`softmax`、`clip`。
- 归约：`sum`、`prod`、`min`、`max`、`mean`、`variance`、`stddev`、`cumsum`、`cumprod`、`argmin`、`argmax`。
- 组合：`cat/concatenate`、`stack`。
- 基础 linalg/stats/DataFrame 也有初版，但它们不是下一阶段最高优先级。

## 5. 后续每次开发的执行要求

- 优先补 Tensor/Array 测试，再补上层模块测试。
- 新 API 应同时覆盖：正常路径、shape mismatch、axis/dim 错误、空张量或边界情况。
- 任何新功能都应运行相关验证；常规至少运行：

```sh
zig fmt build.zig build.zig.zon src/*.zig
zig build test
```

- 如果目录是 git 仓库，完成变更后应只提交本次相关文件；如果不是 git 仓库则不提交。
