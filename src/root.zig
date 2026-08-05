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
pub const DeviceColumnSchema = dataframe_mod.DeviceColumnSchema;
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

pub const ArrowExport = if (build_options.enable_boltha) struct {
    pub const DataFrame = struct {
        pub const toArrowFields = dataframe_mod.DeviceDataFrame.toArrowFields;
        pub const toArrowSchema = dataframe_mod.DeviceDataFrame.toArrowSchema;
        pub const toArrowRecordBatch = dataframe_mod.DeviceDataFrame.toArrowRecordBatch;
        pub const toArrowTable = dataframe_mod.DeviceDataFrame.toArrowTable;
        pub const toParquetBytes = dataframe_mod.DeviceDataFrame.toParquetBytes;
        pub const fromArrowRecordBatch = dataframe_mod.DeviceDataFrame.fromArrowRecordBatch;
        pub const fromArrowRecordBatchProjection = dataframe_mod.DeviceDataFrame.fromArrowRecordBatchProjection;
        pub const fromArrowTable = dataframe_mod.DeviceDataFrame.fromArrowTable;
        pub const fromArrowTableProjection = dataframe_mod.DeviceDataFrame.fromArrowTableProjection;
        pub const fromParquetBytes = dataframe_mod.DeviceDataFrame.fromParquetBytes;
        pub const fromParquetBytesPruned = dataframe_mod.DeviceDataFrame.fromParquetBytesPruned;
    };

    pub const Column = struct {
        pub const arrowDataType = dataframe_mod.DeviceColumn.arrowDataType;
        pub const toArrowField = dataframe_mod.deviceColumnToArrowField;
        pub const toArrowArray = dataframe_mod.DeviceColumn.toArrowArray;
    };

    pub const ColumnSchema = struct {
        pub const arrowDataType = dataframe_mod.deviceColumnSchemaToArrowDataType;
        pub const toArrowField = dataframe_mod.deviceColumnSchemaToArrowField;
    };

    pub const ColumnView = struct {
        pub const arrowDataType = dataframe_mod.deviceColumnViewToArrowDataType;
        pub const toArrowField = dataframe_mod.deviceColumnViewToArrowField;
    };

    pub const DataFrameView = struct {
        pub const toArrowFields = dataframe_mod.deviceDataFrameViewToArrowFields;
        pub const toArrowSchema = dataframe_mod.deviceDataFrameViewToArrowSchema;
    };
} else struct {};

pub const DeviceDataFrameArrow = if (build_options.enable_boltha) ArrowExport.DataFrame else struct {};
pub const DeviceDataFrameViewArrow = if (build_options.enable_boltha) ArrowExport.DataFrameView else struct {};
pub const DeviceColumnArrow = if (build_options.enable_boltha) ArrowExport.Column else struct {};
pub const DeviceColumnViewArrow = if (build_options.enable_boltha) ArrowExport.ColumnView else struct {};
pub const DeviceColumnSchemaArrow = if (build_options.enable_boltha) ArrowExport.ColumnSchema else struct {};

pub const CsrMatrix = sparse.CsrMatrix;
pub const CscMatrix = sparse.CscMatrix;
pub const CooMatrix = sparse.CooMatrix;
pub const SparseError = sparse.SparseError;
pub const SparseProfile = sparse.SparseProfile;
pub const SparseDiffSummary = sparse.SparseDiffSummary;
pub const SparseResidualSummary = sparse.SparseResidualSummary;
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

test "no-boltha DeviceDataFrame metadata facade is source-compatible" {
    if (!build_options.enable_boltha) {
        const gpa = std.testing.allocator;
        const frame: DeviceDataFrame = .{};

        try std.testing.expectEqual(@as(usize, 0), frame.height());
        try std.testing.expectEqual(@as(usize, 0), frame.width());
        try std.testing.expectEqual(@as(usize, 0), frame.cellCount());
        try std.testing.expect(frame.isEmpty());
        try std.testing.expect(!frame.isDeviceBacked());
        try std.testing.expectEqualStrings("cpu", frame.deviceBackendName());
        try std.testing.expect(frame.columnNamesUnique());
        try std.testing.expect(!frame.hasDuplicateColumnNames());
        try std.testing.expect(frame.hasAllColumns(&.{}));
        try std.testing.expect(!frame.hasAnyColumn(&.{ "missing", "absent" }));

        const shape_value = frame.shape();
        try std.testing.expectEqual(@as(usize, 0), shape_value.rows);
        try std.testing.expectEqual(@as(usize, 0), shape_value.cols);
        try std.testing.expect(frame.hasShape(0, 0));
        try std.testing.expect(frame.columnIndex("missing") == null);
        try std.testing.expectError(error.ColumnNotFound, frame.columnDType("missing"));
        try std.testing.expectError(error.IndexOutOfBounds, frame.columnDTypeAt(0));
        try std.testing.expectError(error.FeatureUnavailable, frame.columnDTypes(gpa));
        try std.testing.expectError(error.FeatureUnavailable, frame.columnNullCounts(gpa));
        try std.testing.expectError(error.FeatureUnavailable, frame.columnSchemas(gpa));

        const column = DeviceColumn{ .i32 = undefined };
        const bool_column = DeviceColumn{ .bool = undefined };
        try std.testing.expectEqual(@as(usize, 0), column.len());
        try std.testing.expectEqual(column.len(), column.rowCount());
        try std.testing.expectEqual(column.len(), column.height());
        try std.testing.expectEqual(column.len(), column.nRows());
        try std.testing.expectEqual(@as(usize, 0), column.shape().rows);
        try std.testing.expect(column.shapeEquals(0));
        try std.testing.expect(column.hasShape(0));
        try std.testing.expect(column.isEmpty());
        try std.testing.expect(!column.isNonEmpty());
        try std.testing.expect(!column.hasRows());
        try std.testing.expectEqual(column.len(), column.cellCount());
        try std.testing.expectEqual(DeviceDType.i32, column.dtype());
        try std.testing.expectEqualStrings("i32", column.dtypeName());
        try std.testing.expectEqual(DeviceDType.i32.bitSize(), column.dtypeBitSize());
        try std.testing.expect(column.isNumeric());
        try std.testing.expect(column.isReal());
        try std.testing.expect(!column.isFloat());
        try std.testing.expect(column.isInteger());
        try std.testing.expect(column.isSignedInteger());
        try std.testing.expect(!column.isUnsignedInteger());
        try std.testing.expect(!column.isBool());
        try std.testing.expect(!column.isComplex());
        try std.testing.expect(column.isCpu());
        try std.testing.expect(!column.isDeviceBacked());
        try std.testing.expectEqualStrings("cpu", column.deviceBackendName());
        try std.testing.expectEqual(@as(usize, 0), column.memoryUsage());
        try std.testing.expect(column.sameDevice(bool_column));
        try std.testing.expect(column.sameLength(bool_column));
        try std.testing.expect(column.sameShape(bool_column));
        try std.testing.expect(column.lengthEquals(0));
        try std.testing.expect(!column.sameDType(bool_column));
        try std.testing.expect(column.sameNullability(bool_column));
        try std.testing.expect(!column.schemaEquals(bool_column));
        try std.testing.expect(column.sameStorage(bool_column));
        try std.testing.expect(column.schema("id").schemaEquals(column.schema("id")));

        const view_names = [_][]const u8{"id"};
        var view_columns = [_]DeviceColumnView{.{
            .dtype = .i32,
            .rows = 2,
            .device = .cpu,
            .data_ptr = 0x10,
            .data_nbytes = 2 * @sizeOf(i32),
        }};
        const view = DeviceDataFrameView{
            .allocator = gpa,
            .names = &view_names,
            .columns = &view_columns,
            .rows = 2,
            .device = .cpu,
        };
        try std.testing.expectEqual(@as(usize, 2), view.nRows());
        try std.testing.expectEqual(@as(usize, 1), view.nCols());
        try std.testing.expect(!view.isEmpty());
        try std.testing.expect(view.columnNamesUnique());
        try std.testing.expect(!view.hasDuplicateColumnNames());
        try std.testing.expectEqual(@as(usize, 0), view.duplicateColumnNameCount());
        try std.testing.expect(view.sameDevice(view));
        try std.testing.expect(view.sameShape(view));
        try std.testing.expect(view.sameStorage(view));
        try std.testing.expect(view.hasShape(2, 1));
        try std.testing.expect(view.sameHeight(view));
        try std.testing.expect(view.sameWidth(view));
        const view_dtypes = try view.columnDTypes(gpa);
        defer gpa.free(view_dtypes);
        try std.testing.expectEqualSlices(DeviceDType, &.{.i32}, view_dtypes);
        const view_dtype_names = try view.dtypeNames(gpa);
        defer gpa.free(view_dtype_names);
        try std.testing.expectEqualStrings("i32", view_dtype_names[0]);
        try std.testing.expectEqual(@as(usize, 1), view.numericColumnCount());
        try std.testing.expectEqual(@as(usize, 1), view.integerColumnCount());
        try std.testing.expectEqual(@as(usize, 1), view.signedIntegerColumnCount());
        const view_integer_mask = try view.columnIsIntegerMask(gpa);
        defer gpa.free(view_integer_mask);
        try std.testing.expectEqualSlices(bool, &.{true}, view_integer_mask);
        try std.testing.expect(view.hasColumn("id"));
        try std.testing.expect(view.hasAllColumns(&.{"id"}));
        try std.testing.expectEqual(DeviceDType.i32, try view.columnDType("id"));
        try std.testing.expectEqual(DeviceDType.i32, try view.columnDTypeAt(0));
        const duplicate_view_names = [_][]const u8{ "id", "id" };
        var duplicate_view_columns = [_]DeviceColumnView{ view_columns[0], view_columns[0] };
        const duplicate_view = DeviceDataFrameView{
            .allocator = gpa,
            .names = &duplicate_view_names,
            .columns = &duplicate_view_columns,
            .rows = 2,
            .device = .cpu,
        };
        try std.testing.expect(!duplicate_view.columnNamesUnique());
        try std.testing.expect(duplicate_view.hasDuplicateColumnNames());
        try std.testing.expectEqual(@as(usize, 1), duplicate_view.duplicateColumnNameCount());
        try std.testing.expect(!view.schemaEquals(duplicate_view));
        try std.testing.expect(!view.sameSchema(duplicate_view));
        const id_view = try view.columnView("id");
        try std.testing.expectEqual(DeviceDType.i32, id_view.dtype);
        try std.testing.expectEqual(@as(usize, 2), id_view.len());
        try std.testing.expectEqual(id_view.len(), id_view.rowCount());
        try std.testing.expectEqual(id_view.len(), id_view.height());
        try std.testing.expectEqual(id_view.len(), id_view.nRows());
        try std.testing.expectEqual(@as(usize, 2), id_view.shape().rows);
        try std.testing.expect(id_view.shapeEquals(2));
        try std.testing.expect(id_view.hasShape(2));
        try std.testing.expectEqual(@as(u64, 0x10), id_view.dataPtr());
        try std.testing.expect(!id_view.hasValidity());
        try std.testing.expect(id_view.validityPtr() == null);
        try std.testing.expectEqual(DeviceValidityEncoding.none, id_view.validityEncoding());
        try std.testing.expect(!id_view.isEmpty());
        try std.testing.expect(id_view.isNonEmpty());
        try std.testing.expect(id_view.hasRows());
        try std.testing.expectEqual(id_view.len(), id_view.cellCount());
        try std.testing.expectEqual(@as(usize, 0), id_view.nullCount());
        try std.testing.expectEqual(@as(usize, 2), id_view.validCount());
        try std.testing.expect(!id_view.anyNull());
        try std.testing.expect(!id_view.allNull());
        try std.testing.expect(id_view.anyValid());
        try std.testing.expect(id_view.allValid());
        try std.testing.expectEqual(@as(usize, 2 * @sizeOf(i32)), id_view.dataNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), id_view.memoryUsage());
        try std.testing.expectEqualStrings("i32", id_view.dtypeName());
        try std.testing.expect(id_view.isCpu());
        try std.testing.expectEqualStrings("cpu", id_view.deviceBackendName());
        try std.testing.expect(id_view.sameDevice(view_columns[0]));
        try std.testing.expect(id_view.sameLength(view_columns[0]));
        try std.testing.expect(id_view.sameShape(view_columns[0]));
        try std.testing.expect(id_view.lengthEquals(2));
        try std.testing.expect(id_view.sameDType(view_columns[0]));
        try std.testing.expect(id_view.sameNullability(view_columns[0]));
        try std.testing.expect(id_view.schemaEquals(view_columns[0]));
        try std.testing.expect(id_view.sameSchema(view_columns[0]));
        try std.testing.expect(id_view.schemaCompatible(view_columns[0]));
        try std.testing.expect(id_view.sameStorage(view_columns[0]));
        const view_null_counts = try view.columnNullCounts(gpa);
        defer gpa.free(view_null_counts);
        try std.testing.expectEqualSlices(usize, &.{0}, view_null_counts);
        const view_valid_counts = try view.columnValidCounts(gpa);
        defer gpa.free(view_valid_counts);
        try std.testing.expectEqualSlices(usize, &.{2}, view_valid_counts);
        try std.testing.expectEqual(@as(usize, 0), view.nullCount());
        try std.testing.expectEqual(@as(usize, 2), view.validCount());
        try std.testing.expectEqual(@as(usize, 2), view.cellCount());
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), view.nullRatio(), 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), view.validRatio(), 1e-12);
        const view_memory = try view.columnMemoryUsage(gpa);
        defer gpa.free(view_memory);
        try std.testing.expectEqualSlices(usize, &.{id_view.totalNbytes()}, view_memory);
        try std.testing.expectEqual(id_view.dataNbytes(), view.dataNbytes());
        try std.testing.expectEqual(id_view.validityNbytes(), view.validityNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), view.totalNbytes());
        try std.testing.expectEqual(view.totalNbytes(), view.estimatedSize());
        const id_schema = try view.columnSchema("id");
        try std.testing.expectEqual(DeviceDType.i32, id_schema.dtype);
        try std.testing.expectEqual(@as(usize, 2), id_schema.rows);
        try std.testing.expectEqual(@as(usize, 2), id_schema.len());
        try std.testing.expectEqual(id_schema.len(), id_schema.rowCount());
        try std.testing.expectEqual(id_schema.len(), id_schema.height());
        try std.testing.expectEqual(id_schema.len(), id_schema.nRows());
        try std.testing.expectEqual(@as(usize, 2), id_schema.shape().rows);
        try std.testing.expect(id_schema.shapeEquals(2));
        try std.testing.expect(id_schema.hasShape(2));
        try std.testing.expect(!id_schema.isEmpty());
        try std.testing.expect(id_schema.isNonEmpty());
        try std.testing.expect(id_schema.hasRows());
        try std.testing.expectEqual(id_schema.len(), id_schema.cellCount());
        try std.testing.expectEqualStrings("i32", id_schema.dtypeName());
        try std.testing.expect(id_schema.isNumeric());
        try std.testing.expect(id_schema.isInteger());
        try std.testing.expect(id_schema.isSignedInteger());
        try std.testing.expect(!id_schema.isBool());
        try std.testing.expect(id_schema.allValid());
        try std.testing.expect(!id_schema.anyNull());
        try std.testing.expect(!id_schema.allNull());
        try std.testing.expect(id_schema.anyValid());
        try std.testing.expectEqual(@as(usize, 0), id_schema.nullCount());
        try std.testing.expectEqual(@as(usize, 2), id_schema.validCount());
        try std.testing.expectEqual(id_view.dataNbytes(), id_schema.dataNbytes());
        try std.testing.expectEqual(id_view.validityNbytes(), id_schema.validityNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), id_schema.totalNbytes());
        try std.testing.expectEqual(id_view.totalNbytes(), id_schema.memoryUsage());
        try std.testing.expect(id_schema.isCpu());
        try std.testing.expect(!id_schema.isDeviceBacked());
        try std.testing.expectEqualStrings("cpu", id_schema.deviceBackendName());
        try std.testing.expect(id_schema.schemaEquals(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameSchema(try view.columnSchema("id")));
        try std.testing.expect(id_schema.schemaCompatible(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameDevice(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameLength(try view.columnSchema("id")));
        try std.testing.expect(id_schema.sameShape(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.lengthEquals(2));
        try std.testing.expect(id_schema.sameDType(try view.columnSchemaAt(0)));
        try std.testing.expect(id_schema.sameNullability(try view.columnSchema("id")));
        const schema = try view.schema(gpa);
        defer gpa.free(schema);
        try std.testing.expectEqual(@as(usize, 1), schema.len);
        try std.testing.expectEqual(DeviceDType.i32, schema[0].dtype);
        const view_alias = DeviceDataFrameView{
            .allocator = gpa,
            .names = &view_names,
            .columns = &view_columns,
            .rows = 2,
            .device = .cpu,
        };
        try std.testing.expect(view.schemaEquals(view_alias));
        try std.testing.expect(view.sameSchema(view_alias));
        try std.testing.expect(view.schemaCompatible(view_alias));
        try std.testing.expectError(error.ColumnNotFound, view.columnSchema("missing"));
        try std.testing.expectError(error.IndexOutOfBounds, view.columnSchemaAt(1));
        try std.testing.expect(std.mem.eql(u8, "id", try view.columnNameAt(0)));
        try std.testing.expectError(error.IndexOutOfBounds, view.columnAt(1));
    }
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
