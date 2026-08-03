//! Vectra is an experimental Zig-native data processing and numerical computing
//! toolkit inspired by NumPy/CuPy/SciPy/Pandas/Polars, with PyTorch-style object
//! methods as the primary public array API.
//!
//! The current implementation provides a compact CPU core: typed arrays,
//! broadcasting arithmetic, reductions, linalg/stat helpers, typed Series,
//! heterogeneous DataFrame operations, sparse matrices, and CSV IO. CUDA/GPU is
//! available through the default Axiom backend when a CUDA device is
//! present; CUDA arrays own device-resident storage and dispatch supported f32/f64
//! operations without staging through host arrays.
//! `DeviceDataFrame` extends the same device model to fixed-width tabular
//! columns on CPU/CUDA/MPS while the existing heterogeneous `DataFrame` remains
//! the compact host/CSV-oriented API.
//! Public backend diagnostics intentionally flow through `axiom_backend` so
//! callers choose targets and inspect capability reports instead of binding to a
//! target-specific bridge module.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("array.zig");
const series_mod = @import("series.zig");
const dataframe_mod = if (build_options.enable_boltha) @import("dataframe.zig") else @import("dataframe_no_boltha.zig");
const einsum_mod = @import("einsum.zig");
const layered_array_mod = @import("layered_array.zig");
const forge_interop_mod = @import("forge_interop.zig");
pub const linalg = @import("linalg.zig");
pub const stats = @import("stats.zig");
pub const sparse = @import("sparse.zig");
pub const forge_interop = forge_interop_mod;
pub const ForgeInteropBoundary = forge_interop_mod.InteropBoundary;
pub const forgeArrayInteropBoundary = forge_interop_mod.forge_array_interop_boundary;
pub const forgeInteropBoundary = forge_interop_mod.forgeInteropBoundary;

pub const Array = array_mod.Array;
pub const NDArray = array_mod.NDArray;
pub const ArrayView = array_mod.ArrayView;
pub const NDArrayView = array_mod.NDArrayView;
pub const BFloat16 = array_mod.BFloat16;
pub const Complex64 = array_mod.Complex64;
pub const Complex128 = array_mod.Complex128;
pub const Device = array_mod.Device;
pub const DType = array_mod.DType;
pub const canCastDType = array_mod.canCastDType;
pub const promoteDType = array_mod.promoteDType;
pub const resultDType = array_mod.resultDType;
pub const promoteType = array_mod.promoteType;
pub const Slice = array_mod.Slice;
pub const ScatterReduce = array_mod.ScatterReduce;
pub const SearchSide = array_mod.SearchSide;
pub const IndexMode = array_mod.IndexMode;
pub const MeshGridIndexing = array_mod.MeshGridIndexing;
pub const ConvMode = array_mod.ConvMode;
pub const LossReduction = array_mod.LossReduction;
pub const ArrayError = array_mod.ArrayError;
pub const axiom_backend = @import("backends/axiom_backend.zig");
pub const DialectBackend = axiom_backend.DialectBackend;
pub const PreparedF32Matmul = axiom_backend.PreparedF32Matmul;
pub const PreparedF32TransposedMatmul = axiom_backend.PreparedF32TransposedMatmul;
pub const PreparedF64Matmul = axiom_backend.PreparedF64Matmul;
pub const PreparedF64TransposedMatmul = axiom_backend.PreparedF64TransposedMatmul;
pub const setDefaultDialectBackend = axiom_backend.setDefaultDialectBackend;
pub const defaultDialectBackend = axiom_backend.defaultDialectBackend;
pub const defaultExecutionTarget = axiom_backend.defaultExecutionTarget;
pub const executionTargetForDevice = axiom_backend.executionTargetForDevice;
pub const resetDefaultDialectBackend = axiom_backend.resetDefaultDialectBackend;
pub const cpuMatmulColumnMajorResult = axiom_backend.cpuMatmulColumnMajorResult;
pub const Series = series_mod.Series;
pub const DataFrame = dataframe_mod.DataFrame;
pub const Column = dataframe_mod.Column;
pub const ColumnDef = dataframe_mod.ColumnDef;
pub const DataError = dataframe_mod.DataError;
pub const DeviceDataFrame = dataframe_mod.DeviceDataFrame;
pub const DeviceLazyFrame = dataframe_mod.DeviceLazyFrame;
pub const DeviceLazyOp = dataframe_mod.DeviceLazyOp;
pub const DeviceLazySource = dataframe_mod.DeviceLazySource;
pub const DeviceLazyGroupByAggregation = dataframe_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyWeightedGroupByAggregation = dataframe_mod.DeviceLazyWeightedGroupByAggregation;
pub const DeviceLazyPairGroupByAggregation = dataframe_mod.DeviceLazyPairGroupByAggregation;
pub const DeviceLazyWeightedPairGroupByAggregation = dataframe_mod.DeviceLazyWeightedPairGroupByAggregation;
pub const DeviceLazyJoinKind = dataframe_mod.DeviceLazyJoinKind;
pub const DeviceParquetScan = dataframe_mod.DeviceParquetScan;
pub const DeviceParquetRangeFilter = dataframe_mod.DeviceParquetRangeFilter;
pub const DeviceParquetNullFilter = dataframe_mod.DeviceParquetNullFilter;
pub const ParquetRangePredicate = dataframe_mod.ParquetRangePredicate;
pub const DeviceDataFrameView = dataframe_mod.DeviceDataFrameView;
pub const DeviceColumn = dataframe_mod.DeviceColumn;
pub const DeviceColumnView = dataframe_mod.DeviceColumnView;
pub const DeviceColumnDef = dataframe_mod.DeviceColumnDef;
pub const DeviceTypedColumn = dataframe_mod.DeviceTypedColumn;
pub const DeviceDataError = dataframe_mod.DeviceDataError;
pub const ArrowInteropError = dataframe_mod.ArrowInteropError;
pub const ParquetInteropError = dataframe_mod.ParquetInteropError;
pub const DeviceDType = dataframe_mod.DeviceDType;
pub const DeviceDTypeClass = dataframe_mod.DeviceDTypeClass;
pub const DeviceScalar = dataframe_mod.DeviceScalar;
pub const Range = dataframe_mod.Range;
pub const DeviceColumnBinaryOp = dataframe_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = dataframe_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = dataframe_mod.DeviceColumnLogicalOp;
pub const DeviceGroupByAggregation = dataframe_mod.DeviceGroupByAggregation;
pub const DeviceSortOptions = dataframe_mod.DeviceSortOptions;
pub const DeviceClipOptions = dataframe_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = dataframe_mod.DeviceThresholdOptions;
pub const DeviceRollingOptions = dataframe_mod.DeviceRollingOptions;
pub const DeviceLagOptions = dataframe_mod.DeviceLagOptions;
pub const DeviceExpandingOptions = dataframe_mod.DeviceExpandingOptions;
pub const DeviceExpandingRankOptions = dataframe_mod.DeviceExpandingRankOptions;
pub const DeviceStandardizeOptions = dataframe_mod.DeviceStandardizeOptions;
pub const DeviceRobustOptions = dataframe_mod.DeviceRobustOptions;
pub const DeviceDrawdownOptions = dataframe_mod.DeviceDrawdownOptions;
pub const DeviceExtremaOptions = dataframe_mod.DeviceExtremaOptions;
pub const DeviceTrendOptions = dataframe_mod.DeviceTrendOptions;
pub const DeviceCrossoverOptions = dataframe_mod.DeviceCrossoverOptions;
pub const DeviceBucketOptions = dataframe_mod.DeviceBucketOptions;
pub const DeviceEmaOptions = dataframe_mod.DeviceEmaOptions;
pub const DeviceLinearFitOptions = dataframe_mod.DeviceLinearFitOptions;
pub const DeviceRollingCorrelationOptions = dataframe_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = dataframe_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = dataframe_mod.DeviceRollingRobustOptions;
pub const NullPlacement = dataframe_mod.NullPlacement;
pub const DeviceJoinOptions = dataframe_mod.DeviceJoinOptions;
pub const AsofStrategy = dataframe_mod.AsofStrategy;
pub const DeviceAsofOptions = dataframe_mod.DeviceAsofOptions;
pub const DeviceValidityEncoding = dataframe_mod.DeviceValidityEncoding;

pub const CsrMatrix = sparse.CsrMatrix;
pub const CscMatrix = sparse.CscMatrix;
pub const CooMatrix = sparse.CooMatrix;
pub const SparseError = sparse.SparseError;
pub const SparseProfile = sparse.SparseProfile;
pub const csrFromDense = sparse.csrFromDense;
pub const csrFromCompressed = sparse.csrFromCompressed;
pub const cscFromDense = sparse.cscFromDense;
pub const cscFromCompressed = sparse.cscFromCompressed;
pub const cooFromDense = sparse.cooFromDense;
pub const cooFromSlices = sparse.cooFromSlices;
pub const cooEye = sparse.cooEye;
pub const cooIdentity = sparse.cooIdentity;
pub const csrEye = sparse.csrEye;
pub const csrIdentity = sparse.csrIdentity;
pub const cscEye = sparse.cscEye;
pub const cscIdentity = sparse.cscIdentity;
pub const cooFromDiagonal = sparse.cooFromDiagonal;
pub const csrFromDiagonal = sparse.csrFromDiagonal;
pub const cscFromDiagonal = sparse.cscFromDiagonal;

pub const MatrixNormOrder = array_mod.MatrixNormOrder;
pub const Triangle = array_mod.Triangle;
pub const Diagonal = array_mod.Diagonal;
pub const QrResult = array_mod.QrResult;
pub const SvdResult = array_mod.SvdResult;
pub const EighResult = array_mod.EighResult;
pub const LuResult = array_mod.LuResult;

pub const LayoutOrder = layered_array_mod.LayoutOrder;
pub const RuntimeLayout = layered_array_mod.RuntimeLayout;
pub const StaticLayout = layered_array_mod.StaticLayout;
pub const StaticArray = layered_array_mod.StaticArray;
pub const SymbolicLayout = layered_array_mod.SymbolicLayout;
pub const SymbolicArray = layered_array_mod.SymbolicArray;
pub const DimExpr = layered_array_mod.DimExpr;
pub const DimBinding = layered_array_mod.DimBinding;
pub const dim = layered_array_mod.dim;
pub const symbol = layered_array_mod.symbol;
pub const dimAdd = layered_array_mod.dimAdd;
pub const dimSub = layered_array_mod.dimSub;
pub const dimMul = layered_array_mod.dimMul;
pub const AnyArray = layered_array_mod.AnyArray;
pub const CreationOptions = layered_array_mod.CreationOptions;
pub const Context = layered_array_mod.Context;
pub const cpu = Device.cpu;
pub const options = layered_array_mod.options;
pub const onDevice = layered_array_mod.onDevice;
pub const seeded = layered_array_mod.seeded;
pub const seededOn = layered_array_mod.seededOn;

pub fn cuda(index: usize) Device {
    return Device.cuda(index);
}

pub fn mps(index: usize) Device {
    return Device.mps(index);
}

pub fn withAllocator(allocator: @import("std").mem.Allocator) Context {
    return layered_array_mod.withAllocator(allocator);
}

pub fn withSeed(allocator: @import("std").mem.Allocator, seed: u64) Context {
    return layered_array_mod.withSeed(allocator, seed);
}

pub fn add(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.add(rhs);
}

pub fn matmul(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.matmul(rhs);
}

pub fn matmulOut(lhs: anytype, rhs: @TypeOf(lhs), out: @TypeOf(lhs)) ArrayError!void {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, out);
    return lhs.matmulOut(rhs, out);
}

pub fn matmulAddOut(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs), out: @TypeOf(lhs)) ArrayError!void {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    try requireSameDevice(lhs, out);
    return lhs.matmulAddOut(rhs, addend, out);
}

pub fn matmulAdd(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAdd(rhs, addend);
}

pub fn matmulAddSqrt(lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return lhs.matmulAddSqrt(rhs, addend);
}

pub fn einsumUnary(subscripts: []const u8, input: anytype) ArrayError!@TypeOf(input) {
    return einsum_mod.einsumUnary(subscripts, input);
}

pub fn einsum1(subscripts: []const u8, input: anytype) ArrayError!@TypeOf(input) {
    return einsum_mod.einsum1(subscripts, input);
}

pub fn einsum3(subscripts: []const u8, a: anytype, b: @TypeOf(a), c: @TypeOf(a)) ArrayError!@TypeOf(a) {
    return einsum_mod.einsum3(subscripts, a, b, c);
}

pub fn einsum(subscripts: []const u8, lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!@TypeOf(lhs) {
    return einsum_mod.einsum(subscripts, lhs, rhs);
}

pub fn tensordot(lhs: anytype, rhs: @TypeOf(lhs), axes_lhs: []const usize, axes_rhs: []const usize) ArrayError!@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    return lhs.contractAxes(rhs, axes_lhs, axes_rhs);
}

pub fn tensorDot(lhs: anytype, rhs: @TypeOf(lhs), axes_lhs: []const usize, axes_rhs: []const usize) ArrayError!@TypeOf(lhs) {
    return tensordot(lhs, rhs, axes_lhs, axes_rhs);
}

pub fn tryMatmulAddTarget(target: DialectBackend, lhs: anytype, rhs: @TypeOf(lhs), addend: @TypeOf(lhs)) ArrayError!?@TypeOf(lhs) {
    try requireSameDevice(lhs, rhs);
    try requireSameDevice(lhs, addend);
    return axiom_backend.executeMatmulAdd(@TypeOf(lhs).Scalar, target, lhs, rhs, addend);
}

fn requireSameDevice(lhs: anytype, rhs: @TypeOf(lhs)) ArrayError!void {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (!lhs.device.isAvailable()) return error.InvalidDevice;
}

pub fn sum(input: anytype, axis_opt: ?isize, keepdims: bool) ArrayError!@TypeOf(input) {
    return input.sum(axis_opt, keepdims);
}

pub fn addScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.addScalar(scalar);
}

pub fn subScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.subScalar(scalar);
}

pub fn mulScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.mulScalar(scalar);
}

pub fn divScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.divScalar(scalar);
}

pub fn rsubScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.rsubScalar(scalar);
}

pub fn rdivScalar(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.rdivScalar(scalar);
}

pub fn scalarSub(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.scalarSub(scalar);
}

pub fn scalarDiv(input: anytype, scalar: @TypeOf(input).Scalar) ArrayError!@TypeOf(input) {
    return input.scalarDiv(scalar);
}

pub fn addScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.addScalarPromote(U, scalar);
}

pub fn subScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.subScalarPromote(U, scalar);
}

pub fn mulScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.mulScalarPromote(U, scalar);
}

pub fn divScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.divScalarPromote(U, scalar);
}

pub fn rsubScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.rsubScalarPromote(U, scalar);
}

pub fn rdivScalarPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.rdivScalarPromote(U, scalar);
}

pub fn scalarSubPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.scalarSubPromote(U, scalar);
}

pub fn scalarDivPromote(input: anytype, comptime U: type, scalar: U) ArrayError!Array(promoteType(@TypeOf(input).Scalar, U)) {
    return input.scalarDivPromote(U, scalar);
}

test "top-level ops respect device dispatch" {
    const gpa = @import("std").testing.allocator;
    var lhs = try Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer lhs.deinit();
    var rhs = try Array(f32).fromSlice(gpa, &.{ 10, 20, 30, 40 }, &.{ 2, 2 });
    defer rhs.deinit();

    var sum_cpu = try add(lhs, rhs);
    defer sum_cpu.deinit();
    try @import("std").testing.expect(sum_cpu.device.isCpu());
    try @import("std").testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, sum_cpu.data);

    if (try tryMatmulAddTarget(.cpu, lhs, rhs, sum_cpu)) |target_add_value| {
        var target_add = target_add_value;
        defer target_add.deinit();
        try @import("std").testing.expect(target_add.device.isCpu());
        try @import("std").testing.expectEqualSlices(f32, &.{ 81, 122, 183, 264 }, target_add.data);
    }

    var cuda_rhs = try rhs.clone();
    defer cuda_rhs.deinit();
    cuda_rhs.device = Device.cuda(0);
    try @import("std").testing.expectError(error.InvalidDevice, add(lhs, cuda_rhs));
}

test {
    _ = array_mod;
    _ = axiom_backend;
    _ = series_mod;
    _ = dataframe_mod;
    _ = layered_array_mod;
    _ = forge_interop_mod;
    _ = linalg;
    _ = stats;
    _ = sparse;
}
