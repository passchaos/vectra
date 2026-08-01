const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_core_mod = @import("dataframe_core.zig");
const dataframe_host_mod = @import("dataframe_host.zig");
const options_mod = @import("dataframe_options.zig");
const dataframe_view_mod = @import("dataframe_view.zig");
const dataframe_device_column_mod = @import("dataframe_device_column.zig");
const lazy_frame_mod = @import("dataframe_lazy_frame.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const boltha = @import("boltha");
const profile_methods_mod = @import("dataframe_profile_methods.zig");
const relation_methods_mod = @import("dataframe_relation_methods.zig");
const table_methods_mod = @import("dataframe_table_methods.zig");
const arrow_methods_mod = @import("dataframe_arrow_methods.zig");

pub const DataError = series_mod.DataError;
pub const DType = dataframe_host_mod.DType;
pub const Column = dataframe_host_mod.Column;
pub const ColumnDef = dataframe_host_mod.ColumnDef;
pub const DataFrame = dataframe_host_mod.DataFrame;
pub const dataframe = dataframe_host_mod.dataframe;
pub const DeviceDType = array_mod.DType;
pub const DeviceDataError = DataError || array_mod.ArrayError;
pub const ArrowInteropError = DeviceDataError || boltha.arrow.ArrayError || boltha.arrow.RecordBatchError || boltha.arrow.TableError;
pub const ParquetInteropError = ArrowInteropError || boltha.parquet.SimpleError;

/// Vectra's portable validity representation for device dataframe columns.
///
/// cuDF uses Arrow-compatible packed bitmasks.  Vectra starts one abstraction
/// level higher and keeps validity as a `Array(bool)` so the dataframe wrapper
/// can work across CPU, CUDA, and MPS storage immediately.  A future Arrow ABI
/// bridge can add a packed-bitmask view without changing the owning column/table
/// shape introduced here.
pub const DeviceValidityEncoding = options_mod.DeviceValidityEncoding;
pub const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
pub const DeviceDTypeClass = options_mod.DeviceDTypeClass;
pub const DeviceScalar = options_mod.DeviceScalar;
pub const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
pub const NullPlacement = options_mod.NullPlacement;
pub const DeviceSortOptions = options_mod.DeviceSortOptions;
pub const DeviceJoinOptions = options_mod.DeviceJoinOptions;
pub const AsofStrategy = options_mod.AsofStrategy;
pub const DeviceAsofOptions = options_mod.DeviceAsofOptions;
pub const DeviceRollingOptions = options_mod.DeviceRollingOptions;
pub const DeviceLagOptions = options_mod.DeviceLagOptions;
pub const DeviceExpandingOptions = options_mod.DeviceExpandingOptions;
pub const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
pub const DeviceStandardizeOptions = options_mod.DeviceStandardizeOptions;
pub const DeviceRobustOptions = options_mod.DeviceRobustOptions;
pub const DeviceDrawdownOptions = options_mod.DeviceDrawdownOptions;
pub const DeviceExtremaOptions = options_mod.DeviceExtremaOptions;
pub const DeviceTrendOptions = options_mod.DeviceTrendOptions;
pub const DeviceCrossoverOptions = options_mod.DeviceCrossoverOptions;
pub const DeviceBucketOptions = options_mod.DeviceBucketOptions;
pub const DeviceEmaOptions = options_mod.DeviceEmaOptions;
pub const DeviceLinearFitOptions = options_mod.DeviceLinearFitOptions;
pub const DeviceClipOptions = options_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
pub const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
pub const ParquetRangePredicate = options_mod.ParquetRangePredicate;
pub const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
pub const Range = options_mod.Range;

pub const DeviceColumnView = dataframe_view_mod.DeviceColumnView;
pub const DeviceDataFrameView = dataframe_view_mod.DeviceDataFrameView;

pub const DeviceTypedColumn = dataframe_device_column_mod.DeviceTypedColumn;

pub const DeviceColumn = dataframe_device_column_mod.DeviceColumn;
pub const DeviceColumnDef = dataframe_device_column_mod.DeviceColumnDef;

pub const DeviceColumnSchema = struct {
    name: []const u8,
    dtype: DeviceDType,
    rows: usize,
    nullable: bool,
    null_count: usize,
    valid_count: usize,
    data_nbytes: usize,
    validity_nbytes: usize,
    total_nbytes: usize,
    device: array_mod.Device,
};

pub const DeviceLazyGroupByAggregation = lazy_op_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyJoinKind = lazy_op_mod.DeviceLazyJoinKind;
pub const DeviceLazyOp = lazy_op_mod.DeviceLazyOp(DeviceDataFrame, DeviceColumn);

const lazy_frame_types = lazy_frame_mod.DeviceLazyTypes(DeviceDataFrame, DeviceColumnDef, DeviceColumn);
pub const DeviceLazySource = lazy_frame_types.DeviceLazySource;
pub const DeviceLazyFrame = lazy_frame_types.DeviceLazyFrame;
pub const DeviceParquetScan = lazy_frame_types.DeviceParquetScan;

/// Owning fixed-width dataframe that can keep every column on the same Vectra
/// device.
///
/// This is intentionally an owning/table wrapper rather than a CUDA-only API:
/// `.cpu`, `.cuda(index)`, and `.mps(index)` use the same metadata and column
/// invariants.  CUDA/MPS row slicing and host-mask filtering currently
/// materialize through host memory because Vectra has not yet grown
/// dataframe-specific gather/compact kernels; preserving the operation behind
/// this API gives those kernels a single integration point later.
pub const DeviceDataFrame = struct {
    allocator: std.mem.Allocator,
    names: [][]const u8,
    columns: []DeviceColumn,
    rows: usize,
    device: array_mod.Device,

    pub fn init(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
        return dataframe_core_mod.init(DeviceDataFrame, allocator, defs);
    }

    pub fn initEmpty(allocator: std.mem.Allocator, rows: usize, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return dataframe_core_mod.initEmpty(DeviceDataFrame, allocator, rows, device_value);
    }

    pub fn fromDataFrame(allocator: std.mem.Allocator, frame: DataFrame, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return dataframe_host_mod.deviceDataFrameFromDataFrame(DeviceDataFrame, DeviceColumnDef, DeviceColumn, allocator, frame, device_value);
    }

    pub fn deinit(self: *DeviceDataFrame) void {
        dataframe_core_mod.deinit(self);
    }

    pub fn clone(self: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
        return dataframe_core_mod.clone(DeviceDataFrame, self);
    }

    pub fn height(self: DeviceDataFrame) usize {
        return self.rows;
    }

    pub fn rowCount(self: DeviceDataFrame) usize {
        return self.height();
    }

    pub fn width(self: DeviceDataFrame) usize {
        return self.columns.len;
    }

    pub fn columnCount(self: DeviceDataFrame) usize {
        return self.width();
    }

    pub fn columnNames(self: DeviceDataFrame) []const []const u8 {
        return self.names;
    }

    pub fn columnDTypes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceDType {
        const out = try allocator.alloc(DeviceDType, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype();
        return out;
    }

    pub fn dtypes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceDType {
        return self.columnDTypes(allocator);
    }

    pub fn columnNullCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.nullCount();
        return out;
    }

    pub fn columnValidCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.validCount();
        return out;
    }

    fn ratioFromCount(count: usize, rows: usize) f64 {
        if (rows == 0) return std.math.nan(f64);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(rows));
    }

    pub fn columnNullRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]f64 {
        const out = try allocator.alloc(f64, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = ratioFromCount(column_value.nullCount(), self.rows);
        return out;
    }

    pub fn columnValidRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]f64 {
        const out = try allocator.alloc(f64, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = ratioFromCount(column_value.validCount(), self.rows);
        return out;
    }

    pub fn columnDistinctCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        errdefer allocator.free(out);
        for (self.names, out) |name, *slot| slot.* = try self.countDistinctColumn(name);
        return out;
    }

    pub fn columnNUniqueCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDistinctCounts(allocator);
    }

    pub fn columnNUnique(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDistinctCounts(allocator);
    }

    pub fn columnDuplicateCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        const distinct_counts = try self.columnDistinctCounts(allocator);
        errdefer allocator.free(distinct_counts);
        for (distinct_counts) |*slot| slot.* = self.rows - slot.*;
        return distinct_counts;
    }

    pub fn columnRepeatedCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDuplicateCounts(allocator);
    }

    pub fn columnDistinctRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]f64 {
        const counts = try self.columnDistinctCounts(allocator);
        defer allocator.free(counts);
        const out = try allocator.alloc(f64, counts.len);
        for (counts, out) |count, *slot| slot.* = ratioFromCount(count, self.rows);
        return out;
    }

    pub fn columnNUniqueRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]f64 {
        return self.columnDistinctRatios(allocator);
    }

    pub fn columnDuplicateRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]f64 {
        const distinct_ratios = try self.columnDistinctRatios(allocator);
        errdefer allocator.free(distinct_ratios);
        for (distinct_ratios) |*slot| slot.* = 1.0 - slot.*;
        return distinct_ratios;
    }

    pub fn columnIsUniqueMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        const counts = try self.columnDistinctCounts(allocator);
        defer allocator.free(counts);
        const out = try allocator.alloc(bool, self.columns.len);
        for (out, counts) |*slot, count| slot.* = count == self.rows;
        return out;
    }

    pub fn columnHasDuplicatesMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        const unique = try self.columnIsUniqueMask(allocator);
        errdefer allocator.free(unique);
        for (unique) |*slot| slot.* = !slot.*;
        return unique;
    }

    pub fn columnHasDuplicateValues(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnHasDuplicatesMask(allocator);
    }

    pub fn columnNullableMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        const out = try allocator.alloc(bool, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.nullable();
        return out;
    }

    pub fn columnHasNullsMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        const out = try allocator.alloc(bool, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.hasNulls();
        return out;
    }

    pub fn columnDataNbytes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dataNbytes();
        return out;
    }

    pub fn columnValidityNbytes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.validityNbytes();
        return out;
    }

    pub fn columnTotalNbytes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.totalNbytes();
        return out;
    }

    pub fn dataNbytes(self: DeviceDataFrame) usize {
        var total: usize = 0;
        for (self.columns) |column_value| total += column_value.dataNbytes();
        return total;
    }

    pub fn validityNbytes(self: DeviceDataFrame) usize {
        var total: usize = 0;
        for (self.columns) |column_value| total += column_value.validityNbytes();
        return total;
    }

    pub fn totalNbytes(self: DeviceDataFrame) usize {
        return self.dataNbytes() + self.validityNbytes();
    }

    pub fn columnSchemas(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceColumnSchema {
        const out = try allocator.alloc(DeviceColumnSchema, self.columns.len);
        for (self.names, self.columns, out) |name, column_value, *slot| {
            slot.* = .{
                .name = name,
                .dtype = column_value.dtype(),
                .rows = column_value.len(),
                .nullable = column_value.nullable(),
                .null_count = column_value.nullCount(),
                .valid_count = column_value.validCount(),
                .data_nbytes = column_value.dataNbytes(),
                .validity_nbytes = column_value.validityNbytes(),
                .total_nbytes = column_value.totalNbytes(),
                .device = column_value.device(),
            };
        }
        return out;
    }

    pub fn schemaSummary(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceColumnSchema {
        return self.columnSchemas(allocator);
    }

    pub fn isEmpty(self: DeviceDataFrame) bool {
        return self.rows == 0 or self.columns.len == 0;
    }

    pub fn isNonEmpty(self: DeviceDataFrame) bool {
        return !self.isEmpty();
    }

    pub fn hasRows(self: DeviceDataFrame) bool {
        return self.rows != 0;
    }

    pub fn hasColumns(self: DeviceDataFrame) bool {
        return self.columns.len != 0;
    }

    pub fn isCpu(self: DeviceDataFrame) bool {
        return self.device.isCpu();
    }

    pub fn isCuda(self: DeviceDataFrame) bool {
        return self.device.isCuda();
    }

    pub fn isMps(self: DeviceDataFrame) bool {
        return self.device.isMps();
    }

    pub fn isDeviceBacked(self: DeviceDataFrame) bool {
        return !self.isCpu();
    }

    pub fn deviceBackendName(self: DeviceDataFrame) []const u8 {
        return self.device.backendName();
    }

    pub fn sameDevice(self: DeviceDataFrame, other: DeviceDataFrame) bool {
        return self.device.sameDevice(other.device);
    }

    pub fn hasColumn(self: DeviceDataFrame, name: []const u8) bool {
        return self.columnIndex(name) != null;
    }

    pub fn hasAllColumns(self: DeviceDataFrame, names: []const []const u8) bool {
        for (names) |name| {
            if (!self.hasColumn(name)) return false;
        }
        return true;
    }

    pub fn hasAnyColumn(self: DeviceDataFrame, names: []const []const u8) bool {
        for (names) |name| {
            if (self.hasColumn(name)) return true;
        }
        return false;
    }

    pub fn shape(self: DeviceDataFrame) struct { rows: usize, cols: usize } {
        return dataframe_core_mod.shape(self);
    }

    pub fn sameShape(self: DeviceDataFrame, other: DeviceDataFrame) bool {
        return self.rows == other.rows and self.columns.len == other.columns.len;
    }

    pub fn shapeEquals(self: DeviceDataFrame, rows: usize, columns: usize) bool {
        return self.rows == rows and self.columns.len == columns;
    }

    pub fn hasShape(self: DeviceDataFrame, rows: usize, columns: usize) bool {
        return self.shapeEquals(rows, columns);
    }

    pub fn sameHeight(self: DeviceDataFrame, other: DeviceDataFrame) bool {
        return self.rows == other.rows;
    }

    pub fn sameWidth(self: DeviceDataFrame, other: DeviceDataFrame) bool {
        return self.columns.len == other.columns.len;
    }

    pub fn columnIndex(self: DeviceDataFrame, name: []const u8) ?usize {
        return dataframe_core_mod.columnIndex(self, name);
    }

    pub fn column(self: *const DeviceDataFrame, name: []const u8) DataError!*const DeviceColumn {
        return dataframe_core_mod.column(self, name);
    }

    pub fn columnAt(self: *const DeviceDataFrame, index: usize) DeviceDataError!*const DeviceColumn {
        return dataframe_core_mod.columnAt(self, index);
    }

    pub fn columnView(self: *const DeviceDataFrame, name: []const u8) DataError!DeviceColumnView {
        const column_value = try self.column(name);
        return column_value.view();
    }

    pub fn columnViewAt(self: *const DeviceDataFrame, index: usize) DeviceDataError!DeviceColumnView {
        const column_value = try self.columnAt(index);
        return column_value.view();
    }

    pub fn columnNameAt(self: DeviceDataFrame, index: usize) DeviceDataError![]const u8 {
        return dataframe_core_mod.columnNameAt(self, index);
    }

    pub fn columnDType(self: DeviceDataFrame, name: []const u8) DataError!DeviceDType {
        return dataframe_core_mod.columnDType(self, name);
    }

    pub fn columnDTypeAt(self: DeviceDataFrame, index: usize) DeviceDataError!DeviceDType {
        return dataframe_core_mod.columnDTypeAt(self, index);
    }

    pub const unaryColumnAbs = table_methods_mod.unaryColumnAbs;
    pub const withColumnAbs = table_methods_mod.withColumnAbs;
    pub const unaryColumnNeg = table_methods_mod.unaryColumnNeg;
    pub const withColumnNeg = table_methods_mod.withColumnNeg;
    pub const withColumnNegative = table_methods_mod.withColumnNegative;
    pub const unaryColumnSquare = table_methods_mod.unaryColumnSquare;
    pub const withColumnSquare = table_methods_mod.withColumnSquare;
    pub const unaryColumnReciprocal = table_methods_mod.unaryColumnReciprocal;
    pub const withColumnReciprocal = table_methods_mod.withColumnReciprocal;
    pub const unaryColumnSign = table_methods_mod.unaryColumnSign;
    pub const withColumnSign = table_methods_mod.withColumnSign;
    pub const unaryColumnSqrt = table_methods_mod.unaryColumnSqrt;
    pub const withColumnSqrt = table_methods_mod.withColumnSqrt;
    pub const unaryColumnRsqrt = table_methods_mod.unaryColumnRsqrt;
    pub const withColumnRsqrt = table_methods_mod.withColumnRsqrt;
    pub const unaryColumnCbrt = table_methods_mod.unaryColumnCbrt;
    pub const withColumnCbrt = table_methods_mod.withColumnCbrt;
    pub const unaryColumnFloor = table_methods_mod.unaryColumnFloor;
    pub const withColumnFloor = table_methods_mod.withColumnFloor;
    pub const unaryColumnCeil = table_methods_mod.unaryColumnCeil;
    pub const withColumnCeil = table_methods_mod.withColumnCeil;
    pub const unaryColumnRound = table_methods_mod.unaryColumnRound;
    pub const withColumnRound = table_methods_mod.withColumnRound;
    pub const unaryColumnTrunc = table_methods_mod.unaryColumnTrunc;
    pub const withColumnTrunc = table_methods_mod.withColumnTrunc;
    pub const unaryColumnDeg2rad = table_methods_mod.unaryColumnDeg2rad;
    pub const withColumnDeg2rad = table_methods_mod.withColumnDeg2rad;
    pub const unaryColumnRad2deg = table_methods_mod.unaryColumnRad2deg;
    pub const withColumnRad2deg = table_methods_mod.withColumnRad2deg;
    pub const unaryColumnExpit = table_methods_mod.unaryColumnExpit;
    pub const withColumnExpit = table_methods_mod.withColumnExpit;
    pub const unaryColumnLogit = table_methods_mod.unaryColumnLogit;
    pub const withColumnLogit = table_methods_mod.withColumnLogit;
    pub const unaryColumnSoftplus = table_methods_mod.unaryColumnSoftplus;
    pub const withColumnSoftplus = table_methods_mod.withColumnSoftplus;
    pub const unaryColumnLogsigmoid = table_methods_mod.unaryColumnLogsigmoid;
    pub const withColumnLogsigmoid = table_methods_mod.withColumnLogsigmoid;
    pub const unaryColumnRelu = table_methods_mod.unaryColumnRelu;
    pub const withColumnRelu = table_methods_mod.withColumnRelu;
    pub const unaryColumnLeakyRelu = table_methods_mod.unaryColumnLeakyRelu;
    pub const unaryColumnLeakyReluWithDeviceScalar = table_methods_mod.unaryColumnLeakyReluWithDeviceScalar;
    pub const withColumnLeakyRelu = table_methods_mod.withColumnLeakyRelu;
    pub const withColumnLeakyReluWithDeviceScalar = table_methods_mod.withColumnLeakyReluWithDeviceScalar;
    pub const unaryColumnRelu6 = table_methods_mod.unaryColumnRelu6;
    pub const withColumnRelu6 = table_methods_mod.withColumnRelu6;
    pub const unaryColumnPowScalar = table_methods_mod.unaryColumnPowScalar;
    pub const unaryColumnPowWithDeviceScalar = table_methods_mod.unaryColumnPowWithDeviceScalar;
    pub const withColumnPowScalar = table_methods_mod.withColumnPowScalar;
    pub const withColumnPowWithDeviceScalar = table_methods_mod.withColumnPowWithDeviceScalar;
    pub const unaryColumnFloorDivScalar = table_methods_mod.unaryColumnFloorDivScalar;
    pub const unaryColumnFloorDivWithDeviceScalar = table_methods_mod.unaryColumnFloorDivWithDeviceScalar;
    pub const withColumnFloorDivScalar = table_methods_mod.withColumnFloorDivScalar;
    pub const withColumnFloorDivWithDeviceScalar = table_methods_mod.withColumnFloorDivWithDeviceScalar;
    pub const unaryColumnModScalar = table_methods_mod.unaryColumnModScalar;
    pub const unaryColumnModWithDeviceScalar = table_methods_mod.unaryColumnModWithDeviceScalar;
    pub const withColumnModScalar = table_methods_mod.withColumnModScalar;
    pub const withColumnModWithDeviceScalar = table_methods_mod.withColumnModWithDeviceScalar;
    pub const unaryColumnRemainderScalar = table_methods_mod.unaryColumnRemainderScalar;
    pub const unaryColumnRemainderWithDeviceScalar = table_methods_mod.unaryColumnRemainderWithDeviceScalar;
    pub const withColumnRemainderScalar = table_methods_mod.withColumnRemainderScalar;
    pub const withColumnRemainderWithDeviceScalar = table_methods_mod.withColumnRemainderWithDeviceScalar;
    pub const unaryColumnLogAddExpScalar = table_methods_mod.unaryColumnLogAddExpScalar;
    pub const unaryColumnLogAddExpWithDeviceScalar = table_methods_mod.unaryColumnLogAddExpWithDeviceScalar;
    pub const withColumnLogAddExpScalar = table_methods_mod.withColumnLogAddExpScalar;
    pub const withColumnLogAddExpWithDeviceScalar = table_methods_mod.withColumnLogAddExpWithDeviceScalar;
    pub const unaryColumnLogAddExp2Scalar = table_methods_mod.unaryColumnLogAddExp2Scalar;
    pub const unaryColumnLogAddExp2WithDeviceScalar = table_methods_mod.unaryColumnLogAddExp2WithDeviceScalar;
    pub const withColumnLogAddExp2Scalar = table_methods_mod.withColumnLogAddExp2Scalar;
    pub const withColumnLogAddExp2WithDeviceScalar = table_methods_mod.withColumnLogAddExp2WithDeviceScalar;
    pub const unaryColumnXlogyScalar = table_methods_mod.unaryColumnXlogyScalar;
    pub const unaryColumnXlogyWithDeviceScalar = table_methods_mod.unaryColumnXlogyWithDeviceScalar;
    pub const withColumnXlogyScalar = table_methods_mod.withColumnXlogyScalar;
    pub const withColumnXlogyWithDeviceScalar = table_methods_mod.withColumnXlogyWithDeviceScalar;
    pub const unaryColumnFmaxScalar = table_methods_mod.unaryColumnFmaxScalar;
    pub const unaryColumnFmaxWithDeviceScalar = table_methods_mod.unaryColumnFmaxWithDeviceScalar;
    pub const withColumnFmaxScalar = table_methods_mod.withColumnFmaxScalar;
    pub const withColumnFmaxWithDeviceScalar = table_methods_mod.withColumnFmaxWithDeviceScalar;
    pub const unaryColumnFminScalar = table_methods_mod.unaryColumnFminScalar;
    pub const unaryColumnFminWithDeviceScalar = table_methods_mod.unaryColumnFminWithDeviceScalar;
    pub const withColumnFminScalar = table_methods_mod.withColumnFminScalar;
    pub const withColumnFminWithDeviceScalar = table_methods_mod.withColumnFminWithDeviceScalar;
    pub const unaryColumnHypotScalar = table_methods_mod.unaryColumnHypotScalar;
    pub const unaryColumnHypotWithDeviceScalar = table_methods_mod.unaryColumnHypotWithDeviceScalar;
    pub const withColumnHypotScalar = table_methods_mod.withColumnHypotScalar;
    pub const withColumnHypotWithDeviceScalar = table_methods_mod.withColumnHypotWithDeviceScalar;
    pub const unaryColumnAtan2Scalar = table_methods_mod.unaryColumnAtan2Scalar;
    pub const unaryColumnAtan2WithDeviceScalar = table_methods_mod.unaryColumnAtan2WithDeviceScalar;
    pub const withColumnAtan2Scalar = table_methods_mod.withColumnAtan2Scalar;
    pub const withColumnAtan2WithDeviceScalar = table_methods_mod.withColumnAtan2WithDeviceScalar;
    pub const unaryColumnNextAfterScalar = table_methods_mod.unaryColumnNextAfterScalar;
    pub const unaryColumnNextAfterWithDeviceScalar = table_methods_mod.unaryColumnNextAfterWithDeviceScalar;
    pub const withColumnNextAfterScalar = table_methods_mod.withColumnNextAfterScalar;
    pub const withColumnNextAfterWithDeviceScalar = table_methods_mod.withColumnNextAfterWithDeviceScalar;
    pub const unaryColumnCopysignScalar = table_methods_mod.unaryColumnCopysignScalar;
    pub const unaryColumnCopysignWithDeviceScalar = table_methods_mod.unaryColumnCopysignWithDeviceScalar;
    pub const withColumnCopysignScalar = table_methods_mod.withColumnCopysignScalar;
    pub const withColumnCopysignWithDeviceScalar = table_methods_mod.withColumnCopysignWithDeviceScalar;
    pub const unaryColumnHeavisideScalar = table_methods_mod.unaryColumnHeavisideScalar;
    pub const unaryColumnHeavisideWithDeviceScalar = table_methods_mod.unaryColumnHeavisideWithDeviceScalar;
    pub const withColumnHeavisideScalar = table_methods_mod.withColumnHeavisideScalar;
    pub const withColumnHeavisideWithDeviceScalar = table_methods_mod.withColumnHeavisideWithDeviceScalar;
    pub const unaryColumnLdexpScalar = table_methods_mod.unaryColumnLdexpScalar;
    pub const withColumnLdexpScalar = table_methods_mod.withColumnLdexpScalar;
    pub const unaryColumnThreshold = table_methods_mod.unaryColumnThreshold;
    pub const unaryColumnThresholdWithDeviceScalars = table_methods_mod.unaryColumnThresholdWithDeviceScalars;
    pub const withColumnThreshold = table_methods_mod.withColumnThreshold;
    pub const withColumnThresholdWithDeviceScalars = table_methods_mod.withColumnThresholdWithDeviceScalars;
    pub const unaryColumnHardtanh = table_methods_mod.unaryColumnHardtanh;
    pub const unaryColumnHardtanhWithDeviceScalars = table_methods_mod.unaryColumnHardtanhWithDeviceScalars;
    pub const withColumnHardtanh = table_methods_mod.withColumnHardtanh;
    pub const withColumnHardtanhWithDeviceScalars = table_methods_mod.withColumnHardtanhWithDeviceScalars;
    pub const unaryColumnMaximumScalar = table_methods_mod.unaryColumnMaximumScalar;
    pub const unaryColumnMaximumWithDeviceScalar = table_methods_mod.unaryColumnMaximumWithDeviceScalar;
    pub const withColumnMaximumScalar = table_methods_mod.withColumnMaximumScalar;
    pub const withColumnMaximumWithDeviceScalar = table_methods_mod.withColumnMaximumWithDeviceScalar;
    pub const unaryColumnMinimumScalar = table_methods_mod.unaryColumnMinimumScalar;
    pub const unaryColumnMinimumWithDeviceScalar = table_methods_mod.unaryColumnMinimumWithDeviceScalar;
    pub const withColumnMinimumScalar = table_methods_mod.withColumnMinimumScalar;
    pub const withColumnMinimumWithDeviceScalar = table_methods_mod.withColumnMinimumWithDeviceScalar;
    pub const unaryColumnClipMin = table_methods_mod.unaryColumnClipMin;
    pub const unaryColumnClipMinWithDeviceScalar = table_methods_mod.unaryColumnClipMinWithDeviceScalar;
    pub const withColumnClipMin = table_methods_mod.withColumnClipMin;
    pub const withColumnClipMinWithDeviceScalar = table_methods_mod.withColumnClipMinWithDeviceScalar;
    pub const unaryColumnClipMax = table_methods_mod.unaryColumnClipMax;
    pub const unaryColumnClipMaxWithDeviceScalar = table_methods_mod.unaryColumnClipMaxWithDeviceScalar;
    pub const withColumnClipMax = table_methods_mod.withColumnClipMax;
    pub const withColumnClipMaxWithDeviceScalar = table_methods_mod.withColumnClipMaxWithDeviceScalar;
    pub const unaryColumnHardshrink = table_methods_mod.unaryColumnHardshrink;
    pub const unaryColumnHardshrinkWithDeviceScalar = table_methods_mod.unaryColumnHardshrinkWithDeviceScalar;
    pub const withColumnHardshrink = table_methods_mod.withColumnHardshrink;
    pub const withColumnHardshrinkWithDeviceScalar = table_methods_mod.withColumnHardshrinkWithDeviceScalar;
    pub const unaryColumnSoftshrink = table_methods_mod.unaryColumnSoftshrink;
    pub const unaryColumnSoftshrinkWithDeviceScalar = table_methods_mod.unaryColumnSoftshrinkWithDeviceScalar;
    pub const withColumnSoftshrink = table_methods_mod.withColumnSoftshrink;
    pub const withColumnSoftshrinkWithDeviceScalar = table_methods_mod.withColumnSoftshrinkWithDeviceScalar;
    pub const unaryColumnTanhshrink = table_methods_mod.unaryColumnTanhshrink;
    pub const withColumnTanhshrink = table_methods_mod.withColumnTanhshrink;
    pub const unaryColumnElu = table_methods_mod.unaryColumnElu;
    pub const unaryColumnEluWithDeviceScalar = table_methods_mod.unaryColumnEluWithDeviceScalar;
    pub const withColumnElu = table_methods_mod.withColumnElu;
    pub const withColumnEluWithDeviceScalar = table_methods_mod.withColumnEluWithDeviceScalar;
    pub const unaryColumnCelu = table_methods_mod.unaryColumnCelu;
    pub const unaryColumnCeluWithDeviceScalar = table_methods_mod.unaryColumnCeluWithDeviceScalar;
    pub const withColumnCelu = table_methods_mod.withColumnCelu;
    pub const withColumnCeluWithDeviceScalar = table_methods_mod.withColumnCeluWithDeviceScalar;
    pub const unaryColumnSoftsign = table_methods_mod.unaryColumnSoftsign;
    pub const withColumnSoftsign = table_methods_mod.withColumnSoftsign;
    pub const unaryColumnHardsigmoid = table_methods_mod.unaryColumnHardsigmoid;
    pub const withColumnHardsigmoid = table_methods_mod.withColumnHardsigmoid;
    pub const unaryColumnHardswish = table_methods_mod.unaryColumnHardswish;
    pub const withColumnHardswish = table_methods_mod.withColumnHardswish;
    pub const unaryColumnSilu = table_methods_mod.unaryColumnSilu;
    pub const withColumnSilu = table_methods_mod.withColumnSilu;
    pub const unaryColumnSwish = table_methods_mod.unaryColumnSwish;
    pub const withColumnSwish = table_methods_mod.withColumnSwish;
    pub const unaryColumnMish = table_methods_mod.unaryColumnMish;
    pub const withColumnMish = table_methods_mod.withColumnMish;
    pub const unaryColumnGelu = table_methods_mod.unaryColumnGelu;
    pub const withColumnGelu = table_methods_mod.withColumnGelu;
    pub const unaryColumnSelu = table_methods_mod.unaryColumnSelu;
    pub const withColumnSelu = table_methods_mod.withColumnSelu;
    pub const unaryColumnExp = table_methods_mod.unaryColumnExp;
    pub const withColumnExp = table_methods_mod.withColumnExp;
    pub const unaryColumnExp2 = table_methods_mod.unaryColumnExp2;
    pub const withColumnExp2 = table_methods_mod.withColumnExp2;
    pub const unaryColumnExpm1 = table_methods_mod.unaryColumnExpm1;
    pub const withColumnExpm1 = table_methods_mod.withColumnExpm1;
    pub const unaryColumnSin = table_methods_mod.unaryColumnSin;
    pub const withColumnSin = table_methods_mod.withColumnSin;
    pub const unaryColumnCos = table_methods_mod.unaryColumnCos;
    pub const withColumnCos = table_methods_mod.withColumnCos;
    pub const unaryColumnTan = table_methods_mod.unaryColumnTan;
    pub const withColumnTan = table_methods_mod.withColumnTan;
    pub const unaryColumnAsin = table_methods_mod.unaryColumnAsin;
    pub const withColumnAsin = table_methods_mod.withColumnAsin;
    pub const unaryColumnAcos = table_methods_mod.unaryColumnAcos;
    pub const withColumnAcos = table_methods_mod.withColumnAcos;
    pub const unaryColumnAtan = table_methods_mod.unaryColumnAtan;
    pub const withColumnAtan = table_methods_mod.withColumnAtan;
    pub const unaryColumnSinh = table_methods_mod.unaryColumnSinh;
    pub const withColumnSinh = table_methods_mod.withColumnSinh;
    pub const unaryColumnCosh = table_methods_mod.unaryColumnCosh;
    pub const withColumnCosh = table_methods_mod.withColumnCosh;
    pub const unaryColumnTanh = table_methods_mod.unaryColumnTanh;
    pub const withColumnTanh = table_methods_mod.withColumnTanh;
    pub const unaryColumnAsinh = table_methods_mod.unaryColumnAsinh;
    pub const withColumnAsinh = table_methods_mod.withColumnAsinh;
    pub const unaryColumnAcosh = table_methods_mod.unaryColumnAcosh;
    pub const withColumnAcosh = table_methods_mod.withColumnAcosh;
    pub const unaryColumnAtanh = table_methods_mod.unaryColumnAtanh;
    pub const withColumnAtanh = table_methods_mod.withColumnAtanh;
    pub const unaryColumnLog = table_methods_mod.unaryColumnLog;
    pub const withColumnLog = table_methods_mod.withColumnLog;
    pub const unaryColumnLog1p = table_methods_mod.unaryColumnLog1p;
    pub const withColumnLog1p = table_methods_mod.withColumnLog1p;
    pub const unaryColumnLgamma = table_methods_mod.unaryColumnLgamma;
    pub const withColumnLgamma = table_methods_mod.withColumnLgamma;
    pub const unaryColumnSinc = table_methods_mod.unaryColumnSinc;
    pub const withColumnSinc = table_methods_mod.withColumnSinc;
    pub const unaryColumnLog2 = table_methods_mod.unaryColumnLog2;
    pub const withColumnLog2 = table_methods_mod.withColumnLog2;
    pub const unaryColumnLog10 = table_methods_mod.unaryColumnLog10;
    pub const withColumnLog10 = table_methods_mod.withColumnLog10;
    pub const binaryColumns = table_methods_mod.binaryColumns;
    pub const addColumns = table_methods_mod.addColumns;
    pub const subColumns = table_methods_mod.subColumns;
    pub const mulColumns = table_methods_mod.mulColumns;
    pub const divColumns = table_methods_mod.divColumns;
    pub const binaryColumnScalar = table_methods_mod.binaryColumnScalar;
    pub const binaryColumnScalarWithDeviceScalar = table_methods_mod.binaryColumnScalarWithDeviceScalar;
    pub const lerpColumnsScalar = table_methods_mod.lerpColumnsScalar;
    pub const lerpColumnsWithDeviceScalar = table_methods_mod.lerpColumnsWithDeviceScalar;
    pub const withColumnLerpScalar = table_methods_mod.withColumnLerpScalar;
    pub const withColumnLerpWithDeviceScalar = table_methods_mod.withColumnLerpWithDeviceScalar;
    pub const addcmulColumnsScalar = table_methods_mod.addcmulColumnsScalar;
    pub const addcmulColumnsWithDeviceScalar = table_methods_mod.addcmulColumnsWithDeviceScalar;
    pub const withColumnAddcmulScalar = table_methods_mod.withColumnAddcmulScalar;
    pub const withColumnAddcmulWithDeviceScalar = table_methods_mod.withColumnAddcmulWithDeviceScalar;
    pub const addcdivColumnsScalar = table_methods_mod.addcdivColumnsScalar;
    pub const addcdivColumnsWithDeviceScalar = table_methods_mod.addcdivColumnsWithDeviceScalar;
    pub const withColumnAddcdivScalar = table_methods_mod.withColumnAddcdivScalar;
    pub const withColumnAddcdivWithDeviceScalar = table_methods_mod.withColumnAddcdivWithDeviceScalar;
    pub const clipArrayColumns = table_methods_mod.clipArrayColumns;
    pub const withColumnClipArray = table_methods_mod.withColumnClipArray;
    pub const whereColumnScalar = table_methods_mod.whereColumnScalar;
    pub const whereColumnWithDeviceScalar = table_methods_mod.whereColumnWithDeviceScalar;
    pub const withColumnWhereScalar = table_methods_mod.withColumnWhereScalar;
    pub const withColumnWhereWithDeviceScalar = table_methods_mod.withColumnWhereWithDeviceScalar;
    pub const whereColumns = table_methods_mod.whereColumns;
    pub const withColumnWhere = table_methods_mod.withColumnWhere;
    pub const isinColumns = table_methods_mod.isinColumns;
    pub const isinColumnValuesWithDeviceColumn = table_methods_mod.isinColumnValuesWithDeviceColumn;
    pub const isinColumnValues = table_methods_mod.isinColumnValues;
    pub const withColumnIsIn = table_methods_mod.withColumnIsIn;
    pub const withColumnIsInInverted = table_methods_mod.withColumnIsInInverted;
    pub const withColumnIsin = table_methods_mod.withColumnIsin;
    pub const withColumnIsinInverted = table_methods_mod.withColumnIsinInverted;
    pub const withColumnIsInValues = table_methods_mod.withColumnIsInValues;
    pub const withColumnIsInValuesInverted = table_methods_mod.withColumnIsInValuesInverted;
    pub const withColumnIsinValues = table_methods_mod.withColumnIsinValues;
    pub const withColumnIsinValuesInverted = table_methods_mod.withColumnIsinValuesInverted;
    pub const maskedPutColumnScalar = table_methods_mod.maskedPutColumnScalar;
    pub const maskedPutColumnWithDeviceScalar = table_methods_mod.maskedPutColumnWithDeviceScalar;
    pub const withColumnMaskedPutScalar = table_methods_mod.withColumnMaskedPutScalar;
    pub const withColumnMaskedPutWithDeviceScalar = table_methods_mod.withColumnMaskedPutWithDeviceScalar;
    pub const withColumnPutMaskScalar = table_methods_mod.withColumnPutMaskScalar;
    pub const withColumnPutMaskWithDeviceScalar = table_methods_mod.withColumnPutMaskWithDeviceScalar;
    pub const putFlatColumnScalar = table_methods_mod.putFlatColumnScalar;
    pub const putFlatColumnWithDeviceScalar = table_methods_mod.putFlatColumnWithDeviceScalar;
    pub const putFlatColumns = table_methods_mod.putFlatColumns;
    pub const withColumnPutFlat = table_methods_mod.withColumnPutFlat;
    pub const withColumnPutFlatScalar = table_methods_mod.withColumnPutFlatScalar;
    pub const withColumnPutFlatWithDeviceScalar = table_methods_mod.withColumnPutFlatWithDeviceScalar;
    pub const withColumnIndexPut = table_methods_mod.withColumnIndexPut;
    pub const withColumnIndexPutScalar = table_methods_mod.withColumnIndexPutScalar;
    pub const withColumnIndexPutWithDeviceScalar = table_methods_mod.withColumnIndexPutWithDeviceScalar;
    pub const putFlatColumnScalarMode = table_methods_mod.putFlatColumnScalarMode;
    pub const putFlatColumnModeWithDeviceScalar = table_methods_mod.putFlatColumnModeWithDeviceScalar;
    pub const withColumnPutFlatScalarMode = table_methods_mod.withColumnPutFlatScalarMode;
    pub const withColumnPutFlatModeWithDeviceScalar = table_methods_mod.withColumnPutFlatModeWithDeviceScalar;
    pub const putFlatColumnScalarSigned = table_methods_mod.putFlatColumnScalarSigned;
    pub const putFlatColumnSignedWithDeviceScalar = table_methods_mod.putFlatColumnSignedWithDeviceScalar;
    pub const withColumnPutFlatScalarSigned = table_methods_mod.withColumnPutFlatScalarSigned;
    pub const withColumnPutFlatSignedWithDeviceScalar = table_methods_mod.withColumnPutFlatSignedWithDeviceScalar;
    pub const withColumnIndexPutScalarSigned = table_methods_mod.withColumnIndexPutScalarSigned;
    pub const withColumnIndexPutSignedWithDeviceScalar = table_methods_mod.withColumnIndexPutSignedWithDeviceScalar;
    pub const compareColumns = table_methods_mod.compareColumns;
    pub const compareColumnScalar = table_methods_mod.compareColumnScalar;
    pub const compareColumnScalarWithDeviceScalar = table_methods_mod.compareColumnScalarWithDeviceScalar;
    pub const betweenColumnScalar = table_methods_mod.betweenColumnScalar;
    pub const betweenColumnWithDeviceScalars = table_methods_mod.betweenColumnWithDeviceScalars;
    pub const withColumnBetween = table_methods_mod.withColumnBetween;
    pub const withColumnIsBetween = table_methods_mod.withColumnIsBetween;
    pub const withColumnBetweenClosed = table_methods_mod.withColumnBetweenClosed;
    pub const withColumnBetweenWithDeviceScalars = table_methods_mod.withColumnBetweenWithDeviceScalars;
    pub const withColumnBetweenExclusive = table_methods_mod.withColumnBetweenExclusive;
    pub const withColumnBetweenLeftClosed = table_methods_mod.withColumnBetweenLeftClosed;
    pub const withColumnBetweenRightClosed = table_methods_mod.withColumnBetweenRightClosed;
    pub const notBetweenColumnScalar = table_methods_mod.notBetweenColumnScalar;
    pub const notBetweenColumnWithDeviceScalars = table_methods_mod.notBetweenColumnWithDeviceScalars;
    pub const withColumnNotBetween = table_methods_mod.withColumnNotBetween;
    pub const withColumnOutside = table_methods_mod.withColumnOutside;
    pub const withColumnNotBetweenClosed = table_methods_mod.withColumnNotBetweenClosed;
    pub const withColumnNotBetweenWithDeviceScalars = table_methods_mod.withColumnNotBetweenWithDeviceScalars;
    pub const withColumnNotBetweenExclusive = table_methods_mod.withColumnNotBetweenExclusive;
    pub const withColumnNotBetweenLeftClosed = table_methods_mod.withColumnNotBetweenLeftClosed;
    pub const withColumnNotBetweenRightClosed = table_methods_mod.withColumnNotBetweenRightClosed;
    pub const iscloseColumnScalar = table_methods_mod.iscloseColumnScalar;
    pub const iscloseColumnScalarEqualNan = table_methods_mod.iscloseColumnScalarEqualNan;
    pub const iscloseColumnWithDeviceScalars = table_methods_mod.iscloseColumnWithDeviceScalars;
    pub const iscloseColumnWithDeviceScalarsEqualNan = table_methods_mod.iscloseColumnWithDeviceScalarsEqualNan;
    pub const withColumnIscloseScalar = table_methods_mod.withColumnIscloseScalar;
    pub const withColumnIscloseScalarEqualNan = table_methods_mod.withColumnIscloseScalarEqualNan;
    pub const withColumnIscloseWithDeviceScalars = table_methods_mod.withColumnIscloseWithDeviceScalars;
    pub const withColumnIscloseWithDeviceScalarsEqualNan = table_methods_mod.withColumnIscloseWithDeviceScalarsEqualNan;
    pub const allcloseColumnScalar = table_methods_mod.allcloseColumnScalar;
    pub const allcloseColumnScalarEqualNan = table_methods_mod.allcloseColumnScalarEqualNan;
    pub const allcloseColumnWithDeviceScalars = table_methods_mod.allcloseColumnWithDeviceScalars;
    pub const allcloseColumnWithDeviceScalarsEqualNan = table_methods_mod.allcloseColumnWithDeviceScalarsEqualNan;
    pub const countNonzeroColumn = table_methods_mod.countNonzeroColumn;
    pub const zeroCountColumn = table_methods_mod.zeroCountColumn;
    pub const countZeroColumn = table_methods_mod.countZeroColumn;
    pub const nanCountColumn = table_methods_mod.nanCountColumn;
    pub const infCountColumn = table_methods_mod.infCountColumn;
    pub const positiveInfCountColumn = table_methods_mod.positiveInfCountColumn;
    pub const negativeInfCountColumn = table_methods_mod.negativeInfCountColumn;
    pub const finiteCountColumn = table_methods_mod.finiteCountColumn;
    pub const nonFiniteCountColumn = table_methods_mod.nonFiniteCountColumn;
    pub const normalCountColumn = table_methods_mod.normalCountColumn;
    pub const subnormalCountColumn = table_methods_mod.subnormalCountColumn;
    pub const anyZeroColumn = table_methods_mod.anyZeroColumn;
    pub const allZeroColumn = table_methods_mod.allZeroColumn;
    pub const anyNonzeroColumn = table_methods_mod.anyNonzeroColumn;
    pub const anyNonZeroColumn = table_methods_mod.anyNonZeroColumn;
    pub const allNonzeroColumn = table_methods_mod.allNonzeroColumn;
    pub const allNonZeroColumn = table_methods_mod.allNonZeroColumn;
    pub const anyPositiveZeroColumn = table_methods_mod.anyPositiveZeroColumn;
    pub const allPositiveZeroColumn = table_methods_mod.allPositiveZeroColumn;
    pub const anyNegativeZeroColumn = table_methods_mod.anyNegativeZeroColumn;
    pub const allNegativeZeroColumn = table_methods_mod.allNegativeZeroColumn;
    pub const anyPositiveColumn = table_methods_mod.anyPositiveColumn;
    pub const allPositiveColumn = table_methods_mod.allPositiveColumn;
    pub const anyNegativeColumn = table_methods_mod.anyNegativeColumn;
    pub const allNegativeColumn = table_methods_mod.allNegativeColumn;
    pub const anySignBitColumn = table_methods_mod.anySignBitColumn;
    pub const allSignBitColumn = table_methods_mod.allSignBitColumn;
    pub const anyNanColumn = table_methods_mod.anyNanColumn;
    pub const anyNaNColumn = table_methods_mod.anyNaNColumn;
    pub const allNanColumn = table_methods_mod.allNanColumn;
    pub const allNaNColumn = table_methods_mod.allNaNColumn;
    pub const anyInfColumn = table_methods_mod.anyInfColumn;
    pub const allInfColumn = table_methods_mod.allInfColumn;
    pub const anyPositiveInfColumn = table_methods_mod.anyPositiveInfColumn;
    pub const allPositiveInfColumn = table_methods_mod.allPositiveInfColumn;
    pub const anyNegativeInfColumn = table_methods_mod.anyNegativeInfColumn;
    pub const allNegativeInfColumn = table_methods_mod.allNegativeInfColumn;
    pub const anyFiniteColumn = table_methods_mod.anyFiniteColumn;
    pub const allFiniteColumn = table_methods_mod.allFiniteColumn;
    pub const anyNonFiniteColumn = table_methods_mod.anyNonFiniteColumn;
    pub const allNonFiniteColumn = table_methods_mod.allNonFiniteColumn;
    pub const anyNormalColumn = table_methods_mod.anyNormalColumn;
    pub const allNormalColumn = table_methods_mod.allNormalColumn;
    pub const anySubnormalColumn = table_methods_mod.anySubnormalColumn;
    pub const allSubnormalColumn = table_methods_mod.allSubnormalColumn;
    pub const zeroRatioColumn = table_methods_mod.zeroRatioColumn;
    pub const nonzeroRatioColumn = table_methods_mod.nonzeroRatioColumn;
    pub const nonZeroRatioColumn = table_methods_mod.nonZeroRatioColumn;
    pub const positiveZeroCountColumn = table_methods_mod.positiveZeroCountColumn;
    pub const negativeZeroCountColumn = table_methods_mod.negativeZeroCountColumn;
    pub const positiveZeroRatioColumn = table_methods_mod.positiveZeroRatioColumn;
    pub const negativeZeroRatioColumn = table_methods_mod.negativeZeroRatioColumn;
    pub const positiveCountColumn = table_methods_mod.positiveCountColumn;
    pub const negativeCountColumn = table_methods_mod.negativeCountColumn;
    pub const signBitCountColumn = table_methods_mod.signBitCountColumn;
    pub const positiveRatioColumn = table_methods_mod.positiveRatioColumn;
    pub const negativeRatioColumn = table_methods_mod.negativeRatioColumn;
    pub const signBitRatioColumn = table_methods_mod.signBitRatioColumn;
    pub const nanRatioColumn = table_methods_mod.nanRatioColumn;
    pub const infRatioColumn = table_methods_mod.infRatioColumn;
    pub const positiveInfRatioColumn = table_methods_mod.positiveInfRatioColumn;
    pub const negativeInfRatioColumn = table_methods_mod.negativeInfRatioColumn;
    pub const finiteRatioColumn = table_methods_mod.finiteRatioColumn;
    pub const nonFiniteRatioColumn = table_methods_mod.nonFiniteRatioColumn;
    pub const normalRatioColumn = table_methods_mod.normalRatioColumn;
    pub const subnormalRatioColumn = table_methods_mod.subnormalRatioColumn;
    pub const firstZeroIndexColumn = table_methods_mod.firstZeroIndexColumn;
    pub const lastZeroIndexColumn = table_methods_mod.lastZeroIndexColumn;
    pub const firstPositiveZeroIndexColumn = table_methods_mod.firstPositiveZeroIndexColumn;
    pub const lastPositiveZeroIndexColumn = table_methods_mod.lastPositiveZeroIndexColumn;
    pub const firstNegativeZeroIndexColumn = table_methods_mod.firstNegativeZeroIndexColumn;
    pub const lastNegativeZeroIndexColumn = table_methods_mod.lastNegativeZeroIndexColumn;
    pub const firstNonzeroIndexColumn = table_methods_mod.firstNonzeroIndexColumn;
    pub const lastNonzeroIndexColumn = table_methods_mod.lastNonzeroIndexColumn;
    pub const firstPositiveIndexColumn = table_methods_mod.firstPositiveIndexColumn;
    pub const lastPositiveIndexColumn = table_methods_mod.lastPositiveIndexColumn;
    pub const firstNegativeIndexColumn = table_methods_mod.firstNegativeIndexColumn;
    pub const lastNegativeIndexColumn = table_methods_mod.lastNegativeIndexColumn;
    pub const firstSignBitIndexColumn = table_methods_mod.firstSignBitIndexColumn;
    pub const lastSignBitIndexColumn = table_methods_mod.lastSignBitIndexColumn;
    pub const firstNanIndexColumn = table_methods_mod.firstNanIndexColumn;
    pub const firstNaNIndexColumn = table_methods_mod.firstNaNIndexColumn;
    pub const lastNanIndexColumn = table_methods_mod.lastNanIndexColumn;
    pub const lastNaNIndexColumn = table_methods_mod.lastNaNIndexColumn;
    pub const firstInfIndexColumn = table_methods_mod.firstInfIndexColumn;
    pub const lastInfIndexColumn = table_methods_mod.lastInfIndexColumn;
    pub const firstPositiveInfIndexColumn = table_methods_mod.firstPositiveInfIndexColumn;
    pub const lastPositiveInfIndexColumn = table_methods_mod.lastPositiveInfIndexColumn;
    pub const firstNegativeInfIndexColumn = table_methods_mod.firstNegativeInfIndexColumn;
    pub const lastNegativeInfIndexColumn = table_methods_mod.lastNegativeInfIndexColumn;
    pub const firstFiniteIndexColumn = table_methods_mod.firstFiniteIndexColumn;
    pub const lastFiniteIndexColumn = table_methods_mod.lastFiniteIndexColumn;
    pub const firstNormalIndexColumn = table_methods_mod.firstNormalIndexColumn;
    pub const lastNormalIndexColumn = table_methods_mod.lastNormalIndexColumn;
    pub const firstSubnormalIndexColumn = table_methods_mod.firstSubnormalIndexColumn;
    pub const lastSubnormalIndexColumn = table_methods_mod.lastSubnormalIndexColumn;
    pub const firstNonFiniteIndexColumn = table_methods_mod.firstNonFiniteIndexColumn;
    pub const lastNonFiniteIndexColumn = table_methods_mod.lastNonFiniteIndexColumn;
    pub const firstValidIndexColumn = table_methods_mod.firstValidIndexColumn;
    pub const lastValidIndexColumn = table_methods_mod.lastValidIndexColumn;
    pub const firstNullIndexColumn = table_methods_mod.firstNullIndexColumn;
    pub const lastNullIndexColumn = table_methods_mod.lastNullIndexColumn;
    pub const countDistinctColumn = table_methods_mod.countDistinctColumn;
    pub const nUniqueColumn = table_methods_mod.nUniqueColumn;
    pub const nullCountColumn = table_methods_mod.nullCountColumn;
    pub const validCountColumn = table_methods_mod.validCountColumn;
    pub const anyNullColumn = table_methods_mod.anyNullColumn;
    pub const allNullColumn = table_methods_mod.allNullColumn;
    pub const anyValidColumn = table_methods_mod.anyValidColumn;
    pub const allValidColumn = table_methods_mod.allValidColumn;
    pub const nullRatioColumn = table_methods_mod.nullRatioColumn;
    pub const validRatioColumn = table_methods_mod.validRatioColumn;
    pub const modeColumn = table_methods_mod.modeColumn;
    pub const sumColumn = table_methods_mod.sumColumn;
    pub const prodColumn = table_methods_mod.prodColumn;
    pub const meanColumn = table_methods_mod.meanColumn;
    pub const quantileColumn = table_methods_mod.quantileColumn;
    pub const medianColumn = table_methods_mod.medianColumn;
    pub const varianceColumn = table_methods_mod.varianceColumn;
    pub const varColumn = table_methods_mod.varColumn;
    pub const stddevColumn = table_methods_mod.stddevColumn;
    pub const stdColumn = table_methods_mod.stdColumn;
    pub const semColumn = table_methods_mod.semColumn;
    pub const cvColumn = table_methods_mod.cvColumn;
    pub const skewnessColumn = table_methods_mod.skewnessColumn;
    pub const skewColumn = table_methods_mod.skewColumn;
    pub const kurtosisColumn = table_methods_mod.kurtosisColumn;
    pub const kurtColumn = table_methods_mod.kurtColumn;
    pub const meanAbsColumn = table_methods_mod.meanAbsColumn;
    pub const rmsColumn = table_methods_mod.rmsColumn;
    pub const l1NormColumn = table_methods_mod.l1NormColumn;
    pub const l2NormColumn = table_methods_mod.l2NormColumn;
    pub const geometricMeanColumn = table_methods_mod.geometricMeanColumn;
    pub const geoMeanColumn = table_methods_mod.geoMeanColumn;
    pub const harmonicMeanColumn = table_methods_mod.harmonicMeanColumn;
    pub const harmMeanColumn = table_methods_mod.harmMeanColumn;
    pub const madColumn = table_methods_mod.madColumn;
    pub const medianAbsDevColumn = table_methods_mod.medianAbsDevColumn;
    pub const iqrColumn = table_methods_mod.iqrColumn;
    pub const minColumn = table_methods_mod.minColumn;
    pub const maxColumn = table_methods_mod.maxColumn;
    pub const ptpColumn = table_methods_mod.ptpColumn;
    pub const argminColumn = table_methods_mod.argminColumn;
    pub const argmaxColumn = table_methods_mod.argmaxColumn;
    pub const anyColumn = table_methods_mod.anyColumn;
    pub const allColumn = table_methods_mod.allColumn;
    pub const anyTrueColumn = table_methods_mod.anyTrueColumn;
    pub const allTrueColumn = table_methods_mod.allTrueColumn;
    pub const anyFalseColumn = table_methods_mod.anyFalseColumn;
    pub const allFalseColumn = table_methods_mod.allFalseColumn;
    pub const countTrueColumn = table_methods_mod.countTrueColumn;
    pub const countFalseColumn = table_methods_mod.countFalseColumn;
    pub const trueRatioColumn = table_methods_mod.trueRatioColumn;
    pub const falseRatioColumn = table_methods_mod.falseRatioColumn;
    pub const firstTrueIndexColumn = table_methods_mod.firstTrueIndexColumn;
    pub const lastTrueIndexColumn = table_methods_mod.lastTrueIndexColumn;
    pub const firstFalseIndexColumn = table_methods_mod.firstFalseIndexColumn;
    pub const lastFalseIndexColumn = table_methods_mod.lastFalseIndexColumn;
    pub const logicalColumnScalar = table_methods_mod.logicalColumnScalar;
    pub const logicalNotColumn = table_methods_mod.logicalNotColumn;
    pub const notColumn = table_methods_mod.notColumn;
    pub const withColumnLogicalScalar = table_methods_mod.withColumnLogicalScalar;
    pub const withColumnLogicalAndScalar = table_methods_mod.withColumnLogicalAndScalar;
    pub const withColumnLogicalOrScalar = table_methods_mod.withColumnLogicalOrScalar;
    pub const withColumnLogicalXorScalar = table_methods_mod.withColumnLogicalXorScalar;
    pub const withColumnLogicalNot = table_methods_mod.withColumnLogicalNot;
    pub const withColumnNot = table_methods_mod.withColumnNot;
    pub const logicalColumns = table_methods_mod.logicalColumns;
    pub const withColumnLogical = table_methods_mod.withColumnLogical;
    pub const withColumnLogicalAnd = table_methods_mod.withColumnLogicalAnd;
    pub const withColumnLogicalOr = table_methods_mod.withColumnLogicalOr;
    pub const withColumnLogicalXor = table_methods_mod.withColumnLogicalXor;
    pub const filterColumnMask = table_methods_mod.filterColumnMask;
    pub const dropColumnMask = table_methods_mod.dropColumnMask;
    pub const filterColumn = table_methods_mod.filterColumn;
    pub const filterColumnScalarWithDeviceScalar = table_methods_mod.filterColumnScalarWithDeviceScalar;
    pub const filterColumnScalar = table_methods_mod.filterColumnScalar;
    pub const dropColumnScalarWithDeviceScalar = table_methods_mod.dropColumnScalarWithDeviceScalar;
    pub const dropColumnScalar = table_methods_mod.dropColumnScalar;
    pub const filterIsInColumn = table_methods_mod.filterIsInColumn;
    pub const filterNotInColumn = table_methods_mod.filterNotInColumn;
    pub const filterIsinColumn = table_methods_mod.filterIsinColumn;
    pub const filterIsInColumnInverted = table_methods_mod.filterIsInColumnInverted;
    pub const filterIsinColumnInverted = table_methods_mod.filterIsinColumnInverted;
    pub const filterIsInValues = table_methods_mod.filterIsInValues;
    pub const filterNotInValues = table_methods_mod.filterNotInValues;
    pub const filterIsinValues = table_methods_mod.filterIsinValues;
    pub const filterIsInValuesInverted = table_methods_mod.filterIsInValuesInverted;
    pub const filterIsinValuesInverted = table_methods_mod.filterIsinValuesInverted;
    pub const dropIsInColumn = table_methods_mod.dropIsInColumn;
    pub const dropNotInColumn = table_methods_mod.dropNotInColumn;
    pub const dropIsinColumn = table_methods_mod.dropIsinColumn;
    pub const dropIsInColumnInverted = table_methods_mod.dropIsInColumnInverted;
    pub const dropIsinColumnInverted = table_methods_mod.dropIsinColumnInverted;
    pub const dropIsInValues = table_methods_mod.dropIsInValues;
    pub const dropNotInValues = table_methods_mod.dropNotInValues;
    pub const dropIsinValues = table_methods_mod.dropIsinValues;
    pub const dropIsInValuesInverted = table_methods_mod.dropIsInValuesInverted;
    pub const dropIsinValuesInverted = table_methods_mod.dropIsinValuesInverted;
    pub const filterBetweenColumnWithDeviceScalars = table_methods_mod.filterBetweenColumnWithDeviceScalars;
    pub const filterBetweenColumnClosed = table_methods_mod.filterBetweenColumnClosed;
    pub const filterBetweenColumn = table_methods_mod.filterBetweenColumn;
    pub const filterOutsideColumnWithDeviceScalars = table_methods_mod.filterOutsideColumnWithDeviceScalars;
    pub const filterOutsideColumnClosed = table_methods_mod.filterOutsideColumnClosed;
    pub const filterOutsideColumn = table_methods_mod.filterOutsideColumn;
    pub const dropBetweenColumn = table_methods_mod.dropBetweenColumn;
    pub const dropOutsideColumn = table_methods_mod.dropOutsideColumn;
    pub const dropRowsByColumnMask = table_methods_mod.dropRowsByColumnMask;
    pub const whereIndicesColumn = table_methods_mod.whereIndicesColumn;
    pub const argwhereColumn = table_methods_mod.argwhereColumn;
    pub const toArrowSchema = arrow_methods_mod.toArrowSchema;
    pub const toArrowRecordBatch = arrow_methods_mod.toArrowRecordBatch;
    pub const toArrowTable = arrow_methods_mod.toArrowTable;
    pub const toParquetBytes = arrow_methods_mod.toParquetBytes;

    pub fn fromParquetBytes(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) ParquetInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromParquetBytes(DeviceDataFrame, DeviceColumnDef, allocator, bytes, device_value);
    }

    pub fn fromParquetBytesPruned(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        column_name: []const u8,
        predicate: ParquetRangePredicate,
        device_value: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromParquetBytesPruned(DeviceDataFrame, DeviceColumnDef, allocator, bytes, column_name, predicate, device_value);
    }

    pub fn fromArrowTable(allocator: std.mem.Allocator, table: boltha.arrow.Table, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowTable(DeviceDataFrame, DeviceColumnDef, allocator, table, device_value);
    }

    pub fn fromArrowTableProjection(
        allocator: std.mem.Allocator,
        table: boltha.arrow.Table,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowTableProjection(DeviceDataFrame, DeviceColumnDef, allocator, table, wanted_names, device_value);
    }

    pub fn fromArrowRecordBatch(allocator: std.mem.Allocator, batch: boltha.arrow.RecordBatch, device_value: array_mod.Device) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowRecordBatch(DeviceDataFrame, DeviceColumnDef, allocator, batch, device_value);
    }

    pub fn fromArrowRecordBatchProjection(
        allocator: std.mem.Allocator,
        batch: boltha.arrow.RecordBatch,
        wanted_names: []const []const u8,
        device_value: array_mod.Device,
    ) ArrowInteropError!DeviceDataFrame {
        return arrow_methods_mod.fromArrowRecordBatchProjection(DeviceDataFrame, DeviceColumnDef, allocator, batch, wanted_names, device_value);
    }

    pub const view = table_methods_mod.view;
    pub const select = table_methods_mod.select;
    pub const selectByColumnIndices = table_methods_mod.selectByColumnIndices;
    pub const selectColumnRange = table_methods_mod.selectColumnRange;
    pub const selectFirstColumns = table_methods_mod.selectFirstColumns;
    pub const selectLastColumns = table_methods_mod.selectLastColumns;
    pub const dropByColumnIndices = table_methods_mod.dropByColumnIndices;
    pub const dropColumnRange = table_methods_mod.dropColumnRange;
    pub const dropFirstColumns = table_methods_mod.dropFirstColumns;
    pub const dropLastColumns = table_methods_mod.dropLastColumns;
    pub const reverseColumns = table_methods_mod.reverseColumns;
    pub const sortColumnsByName = table_methods_mod.sortColumnsByName;
    pub const selectByNamePrefix = table_methods_mod.selectByNamePrefix;
    pub const selectByNameSuffix = table_methods_mod.selectByNameSuffix;
    pub const selectByNameContains = table_methods_mod.selectByNameContains;
    pub const selectByNameGlob = table_methods_mod.selectByNameGlob;
    pub const dropByNamePrefix = table_methods_mod.dropByNamePrefix;
    pub const dropByNameSuffix = table_methods_mod.dropByNameSuffix;
    pub const dropByNameContains = table_methods_mod.dropByNameContains;
    pub const dropByNameGlob = table_methods_mod.dropByNameGlob;
    pub const selectExcept = table_methods_mod.selectExcept;
    pub const selectAllExcept = table_methods_mod.selectAllExcept;
    pub const excludeColumns = table_methods_mod.excludeColumns;
    pub const selectByDTypes = table_methods_mod.selectByDTypes;
    pub const selectByDTypeClass = table_methods_mod.selectByDTypeClass;
    pub const dropByDTypes = table_methods_mod.dropByDTypes;
    pub const dropByDTypeClass = table_methods_mod.dropByDTypeClass;
    pub const selectNumeric = table_methods_mod.selectNumeric;
    pub const selectReal = table_methods_mod.selectReal;
    pub const selectFloat = table_methods_mod.selectFloat;
    pub const selectInteger = table_methods_mod.selectInteger;
    pub const selectBool = table_methods_mod.selectBool;
    pub const dropNumeric = table_methods_mod.dropNumeric;
    pub const dropReal = table_methods_mod.dropReal;
    pub const dropFloat = table_methods_mod.dropFloat;
    pub const dropInteger = table_methods_mod.dropInteger;
    pub const dropBool = table_methods_mod.dropBool;
    pub const selectNullableColumns = table_methods_mod.selectNullableColumns;
    pub const selectNonNullableColumns = table_methods_mod.selectNonNullableColumns;
    pub const selectColumnsWithNulls = table_methods_mod.selectColumnsWithNulls;
    pub const selectColumnsWithoutNulls = table_methods_mod.selectColumnsWithoutNulls;
    pub const dropNullableColumns = table_methods_mod.dropNullableColumns;
    pub const dropNonNullableColumns = table_methods_mod.dropNonNullableColumns;
    pub const dropColumnsWithNulls = table_methods_mod.dropColumnsWithNulls;
    pub const dropColumnsWithoutNulls = table_methods_mod.dropColumnsWithoutNulls;
    pub const selectColumnsWithNaNs = table_methods_mod.selectColumnsWithNaNs;
    pub const selectColumnsWithoutNaNs = table_methods_mod.selectColumnsWithoutNaNs;
    pub const dropColumnsWithNaNs = table_methods_mod.dropColumnsWithNaNs;
    pub const dropColumnsWithoutNaNs = table_methods_mod.dropColumnsWithoutNaNs;
    pub const selectColumnsWithInfs = table_methods_mod.selectColumnsWithInfs;
    pub const selectColumnsWithoutInfs = table_methods_mod.selectColumnsWithoutInfs;
    pub const dropColumnsWithInfs = table_methods_mod.dropColumnsWithInfs;
    pub const dropColumnsWithoutInfs = table_methods_mod.dropColumnsWithoutInfs;
    pub const selectColumnsWithPositiveInfs = table_methods_mod.selectColumnsWithPositiveInfs;
    pub const selectColumnsWithoutPositiveInfs = table_methods_mod.selectColumnsWithoutPositiveInfs;
    pub const dropColumnsWithPositiveInfs = table_methods_mod.dropColumnsWithPositiveInfs;
    pub const dropColumnsWithoutPositiveInfs = table_methods_mod.dropColumnsWithoutPositiveInfs;
    pub const selectColumnsWithNegativeInfs = table_methods_mod.selectColumnsWithNegativeInfs;
    pub const selectColumnsWithoutNegativeInfs = table_methods_mod.selectColumnsWithoutNegativeInfs;
    pub const dropColumnsWithNegativeInfs = table_methods_mod.dropColumnsWithNegativeInfs;
    pub const dropColumnsWithoutNegativeInfs = table_methods_mod.dropColumnsWithoutNegativeInfs;
    pub const selectColumnsWithZeros = table_methods_mod.selectColumnsWithZeros;
    pub const selectColumnsWithoutZeros = table_methods_mod.selectColumnsWithoutZeros;
    pub const dropColumnsWithZeros = table_methods_mod.dropColumnsWithZeros;
    pub const dropColumnsWithoutZeros = table_methods_mod.dropColumnsWithoutZeros;
    pub const selectColumnsWithPositiveZeros = table_methods_mod.selectColumnsWithPositiveZeros;
    pub const selectColumnsWithoutPositiveZeros = table_methods_mod.selectColumnsWithoutPositiveZeros;
    pub const dropColumnsWithPositiveZeros = table_methods_mod.dropColumnsWithPositiveZeros;
    pub const dropColumnsWithoutPositiveZeros = table_methods_mod.dropColumnsWithoutPositiveZeros;
    pub const selectColumnsWithNegativeZeros = table_methods_mod.selectColumnsWithNegativeZeros;
    pub const selectColumnsWithoutNegativeZeros = table_methods_mod.selectColumnsWithoutNegativeZeros;
    pub const dropColumnsWithNegativeZeros = table_methods_mod.dropColumnsWithNegativeZeros;
    pub const dropColumnsWithoutNegativeZeros = table_methods_mod.dropColumnsWithoutNegativeZeros;
    pub const selectColumnsWithNonZeros = table_methods_mod.selectColumnsWithNonZeros;
    pub const selectColumnsWithoutNonZeros = table_methods_mod.selectColumnsWithoutNonZeros;
    pub const dropColumnsWithNonZeros = table_methods_mod.dropColumnsWithNonZeros;
    pub const dropColumnsWithoutNonZeros = table_methods_mod.dropColumnsWithoutNonZeros;
    pub const selectColumnsWithPositives = table_methods_mod.selectColumnsWithPositives;
    pub const selectColumnsWithoutPositives = table_methods_mod.selectColumnsWithoutPositives;
    pub const dropColumnsWithPositives = table_methods_mod.dropColumnsWithPositives;
    pub const dropColumnsWithoutPositives = table_methods_mod.dropColumnsWithoutPositives;
    pub const selectColumnsWithSignBits = table_methods_mod.selectColumnsWithSignBits;
    pub const selectColumnsWithoutSignBits = table_methods_mod.selectColumnsWithoutSignBits;
    pub const dropColumnsWithSignBits = table_methods_mod.dropColumnsWithSignBits;
    pub const dropColumnsWithoutSignBits = table_methods_mod.dropColumnsWithoutSignBits;
    pub const selectColumnsWithNegatives = table_methods_mod.selectColumnsWithNegatives;
    pub const selectColumnsWithoutNegatives = table_methods_mod.selectColumnsWithoutNegatives;
    pub const dropColumnsWithNegatives = table_methods_mod.dropColumnsWithNegatives;
    pub const dropColumnsWithoutNegatives = table_methods_mod.dropColumnsWithoutNegatives;
    pub const selectColumnsWithFinites = table_methods_mod.selectColumnsWithFinites;
    pub const selectColumnsWithoutFinites = table_methods_mod.selectColumnsWithoutFinites;
    pub const dropColumnsWithFinites = table_methods_mod.dropColumnsWithFinites;
    pub const dropColumnsWithoutFinites = table_methods_mod.dropColumnsWithoutFinites;
    pub const selectColumnsWithNormals = table_methods_mod.selectColumnsWithNormals;
    pub const selectColumnsWithoutNormals = table_methods_mod.selectColumnsWithoutNormals;
    pub const dropColumnsWithNormals = table_methods_mod.dropColumnsWithNormals;
    pub const dropColumnsWithoutNormals = table_methods_mod.dropColumnsWithoutNormals;
    pub const selectColumnsWithSubnormals = table_methods_mod.selectColumnsWithSubnormals;
    pub const selectColumnsWithoutSubnormals = table_methods_mod.selectColumnsWithoutSubnormals;
    pub const dropColumnsWithSubnormals = table_methods_mod.dropColumnsWithSubnormals;
    pub const dropColumnsWithoutSubnormals = table_methods_mod.dropColumnsWithoutSubnormals;
    pub const selectColumnsWithNonFinites = table_methods_mod.selectColumnsWithNonFinites;
    pub const selectColumnsWithoutNonFinites = table_methods_mod.selectColumnsWithoutNonFinites;
    pub const dropColumnsWithNonFinites = table_methods_mod.dropColumnsWithNonFinites;
    pub const dropColumnsWithoutNonFinites = table_methods_mod.dropColumnsWithoutNonFinites;
    pub const withColumn = table_methods_mod.withColumn;
    pub const withColumnAt = table_methods_mod.withColumnAt;
    pub const withColumnBefore = table_methods_mod.withColumnBefore;
    pub const withColumnAfter = table_methods_mod.withColumnAfter;
    pub const copyColumn = table_methods_mod.copyColumn;
    pub const copyColumnAt = table_methods_mod.copyColumnAt;
    pub const copyColumnBefore = table_methods_mod.copyColumnBefore;
    pub const copyColumnAfter = table_methods_mod.copyColumnAfter;
    pub const castColumn = table_methods_mod.castColumn;
    pub const fillNullColumn = table_methods_mod.fillNullColumn;
    pub const fillNullColumnWithScalar = table_methods_mod.fillNullColumnWithScalar;
    pub const withColumnFillNull = table_methods_mod.withColumnFillNull;
    pub const withColumnFillNullScalar = table_methods_mod.withColumnFillNullScalar;
    pub const fillNullForwardColumn = table_methods_mod.fillNullForwardColumn;
    pub const fillNullBackwardColumn = table_methods_mod.fillNullBackwardColumn;
    pub const withColumnFillNullForward = table_methods_mod.withColumnFillNullForward;
    pub const withColumnFillNullBackward = table_methods_mod.withColumnFillNullBackward;
    pub const nullIfColumn = table_methods_mod.nullIfColumn;
    pub const nullIfColumnScalar = table_methods_mod.nullIfColumnScalar;
    pub const nullIfValuesColumnWithDeviceColumn = table_methods_mod.nullIfValuesColumnWithDeviceColumn;
    pub const nullIfValuesColumn = table_methods_mod.nullIfValuesColumn;
    pub const withColumnNullIf = table_methods_mod.withColumnNullIf;
    pub const withColumnNullIfScalar = table_methods_mod.withColumnNullIfScalar;
    pub const withColumnNullIfValuesWithDeviceColumn = table_methods_mod.withColumnNullIfValuesWithDeviceColumn;
    pub const withColumnNullIfValues = table_methods_mod.withColumnNullIfValues;
    pub const nullIfNaNColumn = table_methods_mod.nullIfNaNColumn;
    pub const withColumnNullIfNaN = table_methods_mod.withColumnNullIfNaN;
    pub const nullIfInfColumn = table_methods_mod.nullIfInfColumn;
    pub const withColumnNullIfInf = table_methods_mod.withColumnNullIfInf;
    pub const nullIfPositiveInfColumn = table_methods_mod.nullIfPositiveInfColumn;
    pub const withColumnNullIfPositiveInf = table_methods_mod.withColumnNullIfPositiveInf;
    pub const nullIfNegativeInfColumn = table_methods_mod.nullIfNegativeInfColumn;
    pub const withColumnNullIfNegativeInf = table_methods_mod.withColumnNullIfNegativeInf;
    pub const nullIfZeroColumn = table_methods_mod.nullIfZeroColumn;
    pub const withColumnNullIfZero = table_methods_mod.withColumnNullIfZero;
    pub const nullIfPositiveZeroColumn = table_methods_mod.nullIfPositiveZeroColumn;
    pub const withColumnNullIfPositiveZero = table_methods_mod.withColumnNullIfPositiveZero;
    pub const nullIfNegativeZeroColumn = table_methods_mod.nullIfNegativeZeroColumn;
    pub const withColumnNullIfNegativeZero = table_methods_mod.withColumnNullIfNegativeZero;
    pub const nullIfNonZeroColumn = table_methods_mod.nullIfNonZeroColumn;
    pub const withColumnNullIfNonZero = table_methods_mod.withColumnNullIfNonZero;
    pub const nullIfPositiveColumn = table_methods_mod.nullIfPositiveColumn;
    pub const withColumnNullIfPositive = table_methods_mod.withColumnNullIfPositive;
    pub const nullIfSignBitColumn = table_methods_mod.nullIfSignBitColumn;
    pub const withColumnNullIfSignBit = table_methods_mod.withColumnNullIfSignBit;
    pub const nullIfNegativeColumn = table_methods_mod.nullIfNegativeColumn;
    pub const withColumnNullIfNegative = table_methods_mod.withColumnNullIfNegative;
    pub const nullIfFiniteColumn = table_methods_mod.nullIfFiniteColumn;
    pub const withColumnNullIfFinite = table_methods_mod.withColumnNullIfFinite;
    pub const nullIfNormalColumn = table_methods_mod.nullIfNormalColumn;
    pub const withColumnNullIfNormal = table_methods_mod.withColumnNullIfNormal;
    pub const nullIfSubnormalColumn = table_methods_mod.nullIfSubnormalColumn;
    pub const withColumnNullIfSubnormal = table_methods_mod.withColumnNullIfSubnormal;
    pub const nullIfNonFiniteColumn = table_methods_mod.nullIfNonFiniteColumn;
    pub const withColumnNullIfNonFinite = table_methods_mod.withColumnNullIfNonFinite;
    pub const withColumnFillNaN = table_methods_mod.withColumnFillNaN;
    pub const withColumnFillNaNScalar = table_methods_mod.withColumnFillNaNScalar;
    pub const withColumnFillInf = table_methods_mod.withColumnFillInf;
    pub const withColumnFillInfScalar = table_methods_mod.withColumnFillInfScalar;
    pub const withColumnFillPositiveInf = table_methods_mod.withColumnFillPositiveInf;
    pub const withColumnFillPositiveInfScalar = table_methods_mod.withColumnFillPositiveInfScalar;
    pub const withColumnFillNegativeInf = table_methods_mod.withColumnFillNegativeInf;
    pub const withColumnFillNegativeInfScalar = table_methods_mod.withColumnFillNegativeInfScalar;
    pub const withColumnFillZero = table_methods_mod.withColumnFillZero;
    pub const withColumnFillZeroScalar = table_methods_mod.withColumnFillZeroScalar;
    pub const withColumnFillPositiveZero = table_methods_mod.withColumnFillPositiveZero;
    pub const withColumnFillPositiveZeroScalar = table_methods_mod.withColumnFillPositiveZeroScalar;
    pub const withColumnFillNegativeZero = table_methods_mod.withColumnFillNegativeZero;
    pub const withColumnFillNegativeZeroScalar = table_methods_mod.withColumnFillNegativeZeroScalar;
    pub const withColumnFillNonZero = table_methods_mod.withColumnFillNonZero;
    pub const withColumnFillNonZeroScalar = table_methods_mod.withColumnFillNonZeroScalar;
    pub const withColumnFillPositive = table_methods_mod.withColumnFillPositive;
    pub const withColumnFillPositiveScalar = table_methods_mod.withColumnFillPositiveScalar;
    pub const withColumnFillSignBit = table_methods_mod.withColumnFillSignBit;
    pub const withColumnFillSignBitScalar = table_methods_mod.withColumnFillSignBitScalar;
    pub const withColumnFillNegative = table_methods_mod.withColumnFillNegative;
    pub const withColumnFillNegativeScalar = table_methods_mod.withColumnFillNegativeScalar;
    pub const withColumnFillFinite = table_methods_mod.withColumnFillFinite;
    pub const withColumnFillFiniteScalar = table_methods_mod.withColumnFillFiniteScalar;
    pub const withColumnFillNormal = table_methods_mod.withColumnFillNormal;
    pub const withColumnFillNormalScalar = table_methods_mod.withColumnFillNormalScalar;
    pub const withColumnFillSubnormal = table_methods_mod.withColumnFillSubnormal;
    pub const withColumnFillSubnormalScalar = table_methods_mod.withColumnFillSubnormalScalar;
    pub const withColumnFillNonFinite = table_methods_mod.withColumnFillNonFinite;
    pub const withColumnFillNonFiniteScalar = table_methods_mod.withColumnFillNonFiniteScalar;
    pub const fillNaNColumn = table_methods_mod.fillNaNColumn;
    pub const fillNaNColumnWithScalar = table_methods_mod.fillNaNColumnWithScalar;
    pub const fillInfColumn = table_methods_mod.fillInfColumn;
    pub const fillInfColumnWithScalar = table_methods_mod.fillInfColumnWithScalar;
    pub const fillPositiveInfColumn = table_methods_mod.fillPositiveInfColumn;
    pub const fillPositiveInfColumnWithScalar = table_methods_mod.fillPositiveInfColumnWithScalar;
    pub const fillNegativeInfColumn = table_methods_mod.fillNegativeInfColumn;
    pub const fillNegativeInfColumnWithScalar = table_methods_mod.fillNegativeInfColumnWithScalar;
    pub const fillZeroColumn = table_methods_mod.fillZeroColumn;
    pub const fillZeroColumnWithScalar = table_methods_mod.fillZeroColumnWithScalar;
    pub const fillPositiveZeroColumn = table_methods_mod.fillPositiveZeroColumn;
    pub const fillPositiveZeroColumnWithScalar = table_methods_mod.fillPositiveZeroColumnWithScalar;
    pub const fillNegativeZeroColumn = table_methods_mod.fillNegativeZeroColumn;
    pub const fillNegativeZeroColumnWithScalar = table_methods_mod.fillNegativeZeroColumnWithScalar;
    pub const fillNonZeroColumn = table_methods_mod.fillNonZeroColumn;
    pub const fillNonZeroColumnWithScalar = table_methods_mod.fillNonZeroColumnWithScalar;
    pub const fillPositiveColumn = table_methods_mod.fillPositiveColumn;
    pub const fillPositiveColumnWithScalar = table_methods_mod.fillPositiveColumnWithScalar;
    pub const fillSignBitColumn = table_methods_mod.fillSignBitColumn;
    pub const fillSignBitColumnWithScalar = table_methods_mod.fillSignBitColumnWithScalar;
    pub const fillNegativeColumn = table_methods_mod.fillNegativeColumn;
    pub const fillNegativeColumnWithScalar = table_methods_mod.fillNegativeColumnWithScalar;
    pub const fillFiniteColumn = table_methods_mod.fillFiniteColumn;
    pub const fillFiniteColumnWithScalar = table_methods_mod.fillFiniteColumnWithScalar;
    pub const fillNormalColumn = table_methods_mod.fillNormalColumn;
    pub const fillNormalColumnWithScalar = table_methods_mod.fillNormalColumnWithScalar;
    pub const fillSubnormalColumn = table_methods_mod.fillSubnormalColumn;
    pub const fillSubnormalColumnWithScalar = table_methods_mod.fillSubnormalColumnWithScalar;
    pub const fillNonFiniteColumn = table_methods_mod.fillNonFiniteColumn;
    pub const fillNonFiniteColumnWithScalar = table_methods_mod.fillNonFiniteColumnWithScalar;
    pub const equals = table_methods_mod.equals;
    pub const frameEquals = table_methods_mod.frameEquals;
    pub const allClose = table_methods_mod.allClose;
    pub const allCloseEqualNan = table_methods_mod.allCloseEqualNan;
    pub const frameAllClose = table_methods_mod.frameAllClose;
    pub const schemaEquals = table_methods_mod.schemaEquals;
    pub const sameSchema = table_methods_mod.sameSchema;
    pub const schemaCompatible = table_methods_mod.schemaCompatible;
    pub const coalesceColumns = table_methods_mod.coalesceColumns;
    pub const coalesceColumnsMany = table_methods_mod.coalesceColumnsMany;
    pub const coalesceManyColumns = table_methods_mod.coalesceManyColumns;
    pub const coalesceFirstValidColumns = table_methods_mod.coalesceFirstValidColumns;
    pub const isNullColumn = table_methods_mod.isNullColumn;
    pub const isValidColumn = table_methods_mod.isValidColumn;
    pub const isNanColumn = table_methods_mod.isNanColumn;
    pub const isZeroColumn = table_methods_mod.isZeroColumn;
    pub const isPositiveZeroColumn = table_methods_mod.isPositiveZeroColumn;
    pub const isNegativeZeroColumn = table_methods_mod.isNegativeZeroColumn;
    pub const isNonZeroColumn = table_methods_mod.isNonZeroColumn;
    pub const isPositiveColumn = table_methods_mod.isPositiveColumn;
    pub const isSignBitColumn = table_methods_mod.isSignBitColumn;
    pub const isNegativeColumn = table_methods_mod.isNegativeColumn;
    pub const isFiniteColumn = table_methods_mod.isFiniteColumn;
    pub const isNormalColumn = table_methods_mod.isNormalColumn;
    pub const isSubnormalColumn = table_methods_mod.isSubnormalColumn;
    pub const isNonFiniteColumn = table_methods_mod.isNonFiniteColumn;
    pub const isInfColumn = table_methods_mod.isInfColumn;
    pub const isPositiveInfColumn = table_methods_mod.isPositiveInfColumn;
    pub const isNegativeInfColumn = table_methods_mod.isNegativeInfColumn;
    pub const withRowNullCount = table_methods_mod.withRowNullCount;
    pub const withRowValidCount = table_methods_mod.withRowValidCount;
    pub const withRowAnyNull = table_methods_mod.withRowAnyNull;
    pub const withRowAllNull = table_methods_mod.withRowAllNull;
    pub const withRowAnyValid = table_methods_mod.withRowAnyValid;
    pub const withRowAllValid = table_methods_mod.withRowAllValid;
    pub const withRowCumulativeAnyNull = table_methods_mod.withRowCumulativeAnyNull;
    pub const withRowCumAnyNull = table_methods_mod.withRowCumAnyNull;
    pub const withRowPrefixAnyNull = table_methods_mod.withRowPrefixAnyNull;
    pub const withRowCumulativeAllNull = table_methods_mod.withRowCumulativeAllNull;
    pub const withRowCumAllNull = table_methods_mod.withRowCumAllNull;
    pub const withRowPrefixAllNull = table_methods_mod.withRowPrefixAllNull;
    pub const withRowCumulativeAnyValid = table_methods_mod.withRowCumulativeAnyValid;
    pub const withRowCumAnyValid = table_methods_mod.withRowCumAnyValid;
    pub const withRowPrefixAnyValid = table_methods_mod.withRowPrefixAnyValid;
    pub const withRowCumulativeAllValid = table_methods_mod.withRowCumulativeAllValid;
    pub const withRowCumAllValid = table_methods_mod.withRowCumAllValid;
    pub const withRowPrefixAllValid = table_methods_mod.withRowPrefixAllValid;
    pub const withRowCumulativeNullCount = table_methods_mod.withRowCumulativeNullCount;
    pub const withRowCumNullCount = table_methods_mod.withRowCumNullCount;
    pub const withRowPrefixNullCount = table_methods_mod.withRowPrefixNullCount;
    pub const withRowCumulativeValidCount = table_methods_mod.withRowCumulativeValidCount;
    pub const withRowCumValidCount = table_methods_mod.withRowCumValidCount;
    pub const withRowPrefixValidCount = table_methods_mod.withRowPrefixValidCount;
    pub const withRowCumulativeNullRatio = table_methods_mod.withRowCumulativeNullRatio;
    pub const withRowCumNullRatio = table_methods_mod.withRowCumNullRatio;
    pub const withRowPrefixNullRatio = table_methods_mod.withRowPrefixNullRatio;
    pub const withRowCumulativeValidRatio = table_methods_mod.withRowCumulativeValidRatio;
    pub const withRowCumValidRatio = table_methods_mod.withRowCumValidRatio;
    pub const withRowPrefixValidRatio = table_methods_mod.withRowPrefixValidRatio;
    pub const withRowNullRatio = table_methods_mod.withRowNullRatio;
    pub const withRowValidRatio = table_methods_mod.withRowValidRatio;
    pub const withRowPairCount = table_methods_mod.withRowPairCount;
    pub const withRowFirstValidIndex = table_methods_mod.withRowFirstValidIndex;
    pub const withRowLastValidIndex = table_methods_mod.withRowLastValidIndex;
    pub const withRowFirstNullIndex = table_methods_mod.withRowFirstNullIndex;
    pub const withRowLastNullIndex = table_methods_mod.withRowLastNullIndex;
    pub const withRowCumulativeFirstValidIndex = table_methods_mod.withRowCumulativeFirstValidIndex;
    pub const withRowPrefixFirstValidIndex = table_methods_mod.withRowPrefixFirstValidIndex;
    pub const withRowCumulativeLastValidIndex = table_methods_mod.withRowCumulativeLastValidIndex;
    pub const withRowPrefixLastValidIndex = table_methods_mod.withRowPrefixLastValidIndex;
    pub const withRowCumulativeFirstNullIndex = table_methods_mod.withRowCumulativeFirstNullIndex;
    pub const withRowPrefixFirstNullIndex = table_methods_mod.withRowPrefixFirstNullIndex;
    pub const withRowCumulativeLastNullIndex = table_methods_mod.withRowCumulativeLastNullIndex;
    pub const withRowPrefixLastNullIndex = table_methods_mod.withRowPrefixLastNullIndex;
    pub const withRowWeightedMean = table_methods_mod.withRowWeightedMean;
    pub const withRowWeightedVariance = table_methods_mod.withRowWeightedVariance;
    pub const withRowWeightedVar = table_methods_mod.withRowWeightedVar;
    pub const withRowWeightedStddev = table_methods_mod.withRowWeightedStddev;
    pub const withRowWeightedStd = table_methods_mod.withRowWeightedStd;
    pub const withRowWeightedCovariance = table_methods_mod.withRowWeightedCovariance;
    pub const withRowWeightedCorrelation = table_methods_mod.withRowWeightedCorrelation;
    pub const withRowWeightedBeta = table_methods_mod.withRowWeightedBeta;
    pub const withRowWeightedQuantile = table_methods_mod.withRowWeightedQuantile;
    pub const withRowWeightedMedian = table_methods_mod.withRowWeightedMedian;
    pub const withRowWeightedIqr = table_methods_mod.withRowWeightedIqr;
    pub const withRowWeightedMad = table_methods_mod.withRowWeightedMad;
    pub const withRowWeightedMode = table_methods_mod.withRowWeightedMode;
    pub const withRowWeightedModeWeight = table_methods_mod.withRowWeightedModeWeight;
    pub const withRowWeightedModeRatio = table_methods_mod.withRowWeightedModeRatio;
    pub const withRowWeightedModeMargin = table_methods_mod.withRowWeightedModeMargin;
    pub const withRowWeightedModeMarginRatio = table_methods_mod.withRowWeightedModeMarginRatio;
    pub const withRowWeightedEntropy = table_methods_mod.withRowWeightedEntropy;
    pub const withRowWeightedGiniImpurity = table_methods_mod.withRowWeightedGiniImpurity;
    pub const withRowWeightedPerplexity = table_methods_mod.withRowWeightedPerplexity;
    pub const withRowWeightedInverseSimpson = table_methods_mod.withRowWeightedInverseSimpson;
    pub const withRowWeightedSimpsonConcentration = table_methods_mod.withRowWeightedSimpsonConcentration;
    pub const withRowWeightedEvenness = table_methods_mod.withRowWeightedEvenness;
    pub const withRowDot = table_methods_mod.withRowDot;
    pub const withRowCosineSimilarity = table_methods_mod.withRowCosineSimilarity;
    pub const withRowCosine = table_methods_mod.withRowCosine;
    pub const withRowSquaredEuclideanDistance = table_methods_mod.withRowSquaredEuclideanDistance;
    pub const withRowEuclideanDistance = table_methods_mod.withRowEuclideanDistance;
    pub const withRowManhattanDistance = table_methods_mod.withRowManhattanDistance;
    pub const withRowChebyshevDistance = table_methods_mod.withRowChebyshevDistance;
    pub const withRowCanberraDistance = table_methods_mod.withRowCanberraDistance;
    pub const withRowBrayCurtisDistance = table_methods_mod.withRowBrayCurtisDistance;
    pub const withRowMeanError = table_methods_mod.withRowMeanError;
    pub const withRowBias = table_methods_mod.withRowBias;
    pub const withRowMae = table_methods_mod.withRowMae;
    pub const withRowMse = table_methods_mod.withRowMse;
    pub const withRowRmse = table_methods_mod.withRowRmse;
    pub const withRowMape = table_methods_mod.withRowMape;
    pub const withRowSmape = table_methods_mod.withRowSmape;
    pub const withRowCovariance = table_methods_mod.withRowCovariance;
    pub const withRowCorrelation = table_methods_mod.withRowCorrelation;
    pub const withRowBeta = table_methods_mod.withRowBeta;
    pub const withRowArgMin = table_methods_mod.withRowArgMin;
    pub const withRowArgMax = table_methods_mod.withRowArgMax;
    pub const withRowCumulativeArgMin = table_methods_mod.withRowCumulativeArgMin;
    pub const withRowCumArgMin = table_methods_mod.withRowCumArgMin;
    pub const withRowPrefixArgMin = table_methods_mod.withRowPrefixArgMin;
    pub const withRowCumulativeArgMax = table_methods_mod.withRowCumulativeArgMax;
    pub const withRowCumArgMax = table_methods_mod.withRowCumArgMax;
    pub const withRowPrefixArgMax = table_methods_mod.withRowPrefixArgMax;
    pub const withRowQuantile = table_methods_mod.withRowQuantile;
    pub const withRowQuantileRange = table_methods_mod.withRowQuantileRange;
    pub const withRowTrimmedMean = table_methods_mod.withRowTrimmedMean;
    pub const withRowWinsorizedMean = table_methods_mod.withRowWinsorizedMean;
    pub const withRowMedian = table_methods_mod.withRowMedian;
    pub const withRowIqr = table_methods_mod.withRowIqr;
    pub const withRowInterdecileRange = table_methods_mod.withRowInterdecileRange;
    pub const withRowIdr = table_methods_mod.withRowIdr;
    pub const withRowMidhinge = table_methods_mod.withRowMidhinge;
    pub const withRowTrimean = table_methods_mod.withRowTrimean;
    pub const withRowBowleySkewness = table_methods_mod.withRowBowleySkewness;
    pub const withRowBowleySkew = table_methods_mod.withRowBowleySkew;
    pub const withRowQuartileCoeffDispersion = table_methods_mod.withRowQuartileCoeffDispersion;
    pub const withRowQcd = table_methods_mod.withRowQcd;
    pub const withRowKelleySkewness = table_methods_mod.withRowKelleySkewness;
    pub const withRowKelleySkew = table_methods_mod.withRowKelleySkew;
    pub const withRowMad = table_methods_mod.withRowMad;
    pub const withRowMedianAbsDev = table_methods_mod.withRowMedianAbsDev;
    pub const withRowMode = table_methods_mod.withRowMode;
    pub const withRowCumulativeMode = table_methods_mod.withRowCumulativeMode;
    pub const withRowCumMode = table_methods_mod.withRowCumMode;
    pub const withRowPrefixMode = table_methods_mod.withRowPrefixMode;
    pub const withRowCumulativeModeCount = table_methods_mod.withRowCumulativeModeCount;
    pub const withRowCumModeCount = table_methods_mod.withRowCumModeCount;
    pub const withRowPrefixModeCount = table_methods_mod.withRowPrefixModeCount;
    pub const withRowCumulativeModeRatio = table_methods_mod.withRowCumulativeModeRatio;
    pub const withRowCumModeRatio = table_methods_mod.withRowCumModeRatio;
    pub const withRowPrefixModeRatio = table_methods_mod.withRowPrefixModeRatio;
    pub const withRowCumulativeModeMargin = table_methods_mod.withRowCumulativeModeMargin;
    pub const withRowCumModeMargin = table_methods_mod.withRowCumModeMargin;
    pub const withRowPrefixModeMargin = table_methods_mod.withRowPrefixModeMargin;
    pub const withRowCumulativeModeMarginRatio = table_methods_mod.withRowCumulativeModeMarginRatio;
    pub const withRowCumModeMarginRatio = table_methods_mod.withRowCumModeMarginRatio;
    pub const withRowPrefixModeMarginRatio = table_methods_mod.withRowPrefixModeMarginRatio;
    pub const withRowEntropy = table_methods_mod.withRowEntropy;
    pub const withRowGiniImpurity = table_methods_mod.withRowGiniImpurity;
    pub const withRowPerplexity = table_methods_mod.withRowPerplexity;
    pub const withRowInverseSimpson = table_methods_mod.withRowInverseSimpson;
    pub const withRowSimpsonConcentration = table_methods_mod.withRowSimpsonConcentration;
    pub const withRowEvenness = table_methods_mod.withRowEvenness;
    pub const withRowModeCount = table_methods_mod.withRowModeCount;
    pub const withRowModeRatio = table_methods_mod.withRowModeRatio;
    pub const withRowModeMargin = table_methods_mod.withRowModeMargin;
    pub const withRowModeMarginRatio = table_methods_mod.withRowModeMarginRatio;
    pub const withRowCountDistinct = table_methods_mod.withRowCountDistinct;
    pub const withRowNUnique = table_methods_mod.withRowNUnique;
    pub const withRowCumulativeDistinctCount = table_methods_mod.withRowCumulativeDistinctCount;
    pub const withRowCumDistinctCount = table_methods_mod.withRowCumDistinctCount;
    pub const withRowPrefixDistinctCount = table_methods_mod.withRowPrefixDistinctCount;
    pub const withRowCumulativeNUnique = table_methods_mod.withRowCumulativeNUnique;
    pub const withRowPrefixNUnique = table_methods_mod.withRowPrefixNUnique;
    pub const withRowSum = table_methods_mod.withRowSum;
    pub const withRowMean = table_methods_mod.withRowMean;
    pub const withRowLogSumExp = table_methods_mod.withRowLogSumExp;
    pub const withRowLogsumexp = table_methods_mod.withRowLogsumexp;
    pub const withRowLogMeanExp = table_methods_mod.withRowLogMeanExp;
    pub const withRowLogmeanexp = table_methods_mod.withRowLogmeanexp;
    pub const withRowCentered = table_methods_mod.withRowCentered;
    pub const withRowDemean = table_methods_mod.withRowDemean;
    pub const withRowZScore = table_methods_mod.withRowZScore;
    pub const withRowZscore = table_methods_mod.withRowZscore;
    pub const withRowStandardize = table_methods_mod.withRowStandardize;
    pub const withRowRobustZScore = table_methods_mod.withRowRobustZScore;
    pub const withRowRobustZscore = table_methods_mod.withRowRobustZscore;
    pub const withRowMadZScore = table_methods_mod.withRowMadZScore;
    pub const withRowMadZscore = table_methods_mod.withRowMadZscore;
    pub const withRowAverageRank = table_methods_mod.withRowAverageRank;
    pub const withRowAverageRanks = table_methods_mod.withRowAverageRanks;
    pub const withRowAvgRank = table_methods_mod.withRowAvgRank;
    pub const withRowAvgRanks = table_methods_mod.withRowAvgRanks;
    pub const withRowFractionalRank = table_methods_mod.withRowFractionalRank;
    pub const withRowFractionalRanks = table_methods_mod.withRowFractionalRanks;
    pub const withRowOrdinalRank = table_methods_mod.withRowOrdinalRank;
    pub const withRowOrdinalRanks = table_methods_mod.withRowOrdinalRanks;
    pub const withRowDenseRank = table_methods_mod.withRowDenseRank;
    pub const withRowDenseRanks = table_methods_mod.withRowDenseRanks;
    pub const withRowCompetitionRank = table_methods_mod.withRowCompetitionRank;
    pub const withRowCompetitionRanks = table_methods_mod.withRowCompetitionRanks;
    pub const withRowMinRank = table_methods_mod.withRowMinRank;
    pub const withRowMinRanks = table_methods_mod.withRowMinRanks;
    pub const withRowPercentRank = table_methods_mod.withRowPercentRank;
    pub const withRowPercentRanks = table_methods_mod.withRowPercentRanks;
    pub const withRowPercentileRank = table_methods_mod.withRowPercentileRank;
    pub const withRowPercentileRanks = table_methods_mod.withRowPercentileRanks;
    pub const withRowCumeDist = table_methods_mod.withRowCumeDist;
    pub const withRowCumeDistribution = table_methods_mod.withRowCumeDistribution;
    pub const withRowCumulativeDistribution = table_methods_mod.withRowCumulativeDistribution;
    pub const withRowCumulativeSum = table_methods_mod.withRowCumulativeSum;
    pub const withRowCumsum = table_methods_mod.withRowCumsum;
    pub const withRowCumSum = table_methods_mod.withRowCumSum;
    pub const withRowPrefixSum = table_methods_mod.withRowPrefixSum;
    pub const withRowCumulativeMean = table_methods_mod.withRowCumulativeMean;
    pub const withRowCummean = table_methods_mod.withRowCummean;
    pub const withRowCumMean = table_methods_mod.withRowCumMean;
    pub const withRowPrefixMean = table_methods_mod.withRowPrefixMean;
    pub const withRowCumulativeAverage = table_methods_mod.withRowCumulativeAverage;
    pub const withRowCumAverage = table_methods_mod.withRowCumAverage;
    pub const withRowCumAvg = table_methods_mod.withRowCumAvg;
    pub const withRowPrefixAverage = table_methods_mod.withRowPrefixAverage;
    pub const withRowPrefixAvg = table_methods_mod.withRowPrefixAvg;
    pub const withRowCumulativeLogSumExp = table_methods_mod.withRowCumulativeLogSumExp;
    pub const withRowCumulativeLogsumexp = table_methods_mod.withRowCumulativeLogsumexp;
    pub const withRowCumLogSumExp = table_methods_mod.withRowCumLogSumExp;
    pub const withRowCumLogsumexp = table_methods_mod.withRowCumLogsumexp;
    pub const withRowPrefixLogSumExp = table_methods_mod.withRowPrefixLogSumExp;
    pub const withRowPrefixLogsumexp = table_methods_mod.withRowPrefixLogsumexp;
    pub const withRowCumulativeLogMeanExp = table_methods_mod.withRowCumulativeLogMeanExp;
    pub const withRowCumulativeLogmeanexp = table_methods_mod.withRowCumulativeLogmeanexp;
    pub const withRowCumLogMeanExp = table_methods_mod.withRowCumLogMeanExp;
    pub const withRowCumLogmeanexp = table_methods_mod.withRowCumLogmeanexp;
    pub const withRowPrefixLogMeanExp = table_methods_mod.withRowPrefixLogMeanExp;
    pub const withRowPrefixLogmeanexp = table_methods_mod.withRowPrefixLogmeanexp;
    pub const withRowCumulativeGeometricMean = table_methods_mod.withRowCumulativeGeometricMean;
    pub const withRowCumulativeGeoMean = table_methods_mod.withRowCumulativeGeoMean;
    pub const withRowCumGeometricMean = table_methods_mod.withRowCumGeometricMean;
    pub const withRowCumGeoMean = table_methods_mod.withRowCumGeoMean;
    pub const withRowPrefixGeometricMean = table_methods_mod.withRowPrefixGeometricMean;
    pub const withRowPrefixGeoMean = table_methods_mod.withRowPrefixGeoMean;
    pub const withRowCumulativeHarmonicMean = table_methods_mod.withRowCumulativeHarmonicMean;
    pub const withRowCumulativeHarmMean = table_methods_mod.withRowCumulativeHarmMean;
    pub const withRowCumHarmonicMean = table_methods_mod.withRowCumHarmonicMean;
    pub const withRowCumHarmMean = table_methods_mod.withRowCumHarmMean;
    pub const withRowPrefixHarmonicMean = table_methods_mod.withRowPrefixHarmonicMean;
    pub const withRowPrefixHarmMean = table_methods_mod.withRowPrefixHarmMean;
    pub const withRowCumulativeVariance = table_methods_mod.withRowCumulativeVariance;
    pub const withRowCumulativeVar = table_methods_mod.withRowCumulativeVar;
    pub const withRowCumVariance = table_methods_mod.withRowCumVariance;
    pub const withRowCumVar = table_methods_mod.withRowCumVar;
    pub const withRowPrefixVariance = table_methods_mod.withRowPrefixVariance;
    pub const withRowPrefixVar = table_methods_mod.withRowPrefixVar;
    pub const withRowCumulativeStddev = table_methods_mod.withRowCumulativeStddev;
    pub const withRowCumulativeStd = table_methods_mod.withRowCumulativeStd;
    pub const withRowCumStddev = table_methods_mod.withRowCumStddev;
    pub const withRowCumStd = table_methods_mod.withRowCumStd;
    pub const withRowPrefixStddev = table_methods_mod.withRowPrefixStddev;
    pub const withRowPrefixStd = table_methods_mod.withRowPrefixStd;
    pub const withRowCumulativeSem = table_methods_mod.withRowCumulativeSem;
    pub const withRowCumSem = table_methods_mod.withRowCumSem;
    pub const withRowPrefixSem = table_methods_mod.withRowPrefixSem;
    pub const withRowCumulativeCv = table_methods_mod.withRowCumulativeCv;
    pub const withRowCumCv = table_methods_mod.withRowCumCv;
    pub const withRowPrefixCv = table_methods_mod.withRowPrefixCv;
    pub const withRowCumulativeFano = table_methods_mod.withRowCumulativeFano;
    pub const withRowCumFano = table_methods_mod.withRowCumFano;
    pub const withRowPrefixFano = table_methods_mod.withRowPrefixFano;
    pub const withRowCumulativeIndexOfDispersion = table_methods_mod.withRowCumulativeIndexOfDispersion;
    pub const withRowCumIndexOfDispersion = table_methods_mod.withRowCumIndexOfDispersion;
    pub const withRowPrefixIndexOfDispersion = table_methods_mod.withRowPrefixIndexOfDispersion;
    pub const withRowCumulativeSkewness = table_methods_mod.withRowCumulativeSkewness;
    pub const withRowCumulativeSkew = table_methods_mod.withRowCumulativeSkew;
    pub const withRowCumSkewness = table_methods_mod.withRowCumSkewness;
    pub const withRowCumSkew = table_methods_mod.withRowCumSkew;
    pub const withRowPrefixSkewness = table_methods_mod.withRowPrefixSkewness;
    pub const withRowPrefixSkew = table_methods_mod.withRowPrefixSkew;
    pub const withRowCumulativeKurtosis = table_methods_mod.withRowCumulativeKurtosis;
    pub const withRowCumulativeKurt = table_methods_mod.withRowCumulativeKurt;
    pub const withRowCumKurtosis = table_methods_mod.withRowCumKurtosis;
    pub const withRowCumKurt = table_methods_mod.withRowCumKurt;
    pub const withRowPrefixKurtosis = table_methods_mod.withRowPrefixKurtosis;
    pub const withRowPrefixKurt = table_methods_mod.withRowPrefixKurt;
    pub const withRowCumulativeRms = table_methods_mod.withRowCumulativeRms;
    pub const withRowCumRms = table_methods_mod.withRowCumRms;
    pub const withRowPrefixRms = table_methods_mod.withRowPrefixRms;
    pub const withRowCumulativeMeanAbs = table_methods_mod.withRowCumulativeMeanAbs;
    pub const withRowCumulativeMeanAbsolute = table_methods_mod.withRowCumulativeMeanAbsolute;
    pub const withRowCumMeanAbs = table_methods_mod.withRowCumMeanAbs;
    pub const withRowCumMeanAbsolute = table_methods_mod.withRowCumMeanAbsolute;
    pub const withRowPrefixMeanAbs = table_methods_mod.withRowPrefixMeanAbs;
    pub const withRowPrefixMeanAbsolute = table_methods_mod.withRowPrefixMeanAbsolute;
    pub const withRowCumulativeMeanSquare = table_methods_mod.withRowCumulativeMeanSquare;
    pub const withRowCumulativeMeanSquared = table_methods_mod.withRowCumulativeMeanSquared;
    pub const withRowCumMeanSquare = table_methods_mod.withRowCumMeanSquare;
    pub const withRowCumMeanSquared = table_methods_mod.withRowCumMeanSquared;
    pub const withRowPrefixMeanSquare = table_methods_mod.withRowPrefixMeanSquare;
    pub const withRowPrefixMeanSquared = table_methods_mod.withRowPrefixMeanSquared;
    pub const withRowCumulativeMaxAbs = table_methods_mod.withRowCumulativeMaxAbs;
    pub const withRowCumulativeMaxAbsolute = table_methods_mod.withRowCumulativeMaxAbsolute;
    pub const withRowCumulativeLInfNorm = table_methods_mod.withRowCumulativeLInfNorm;
    pub const withRowCumulativeLinfNorm = table_methods_mod.withRowCumulativeLinfNorm;
    pub const withRowCumMaxAbs = table_methods_mod.withRowCumMaxAbs;
    pub const withRowCumMaxAbsolute = table_methods_mod.withRowCumMaxAbsolute;
    pub const withRowCumLInfNorm = table_methods_mod.withRowCumLInfNorm;
    pub const withRowCumLinfNorm = table_methods_mod.withRowCumLinfNorm;
    pub const withRowPrefixMaxAbs = table_methods_mod.withRowPrefixMaxAbs;
    pub const withRowPrefixMaxAbsolute = table_methods_mod.withRowPrefixMaxAbsolute;
    pub const withRowPrefixLInfNorm = table_methods_mod.withRowPrefixLInfNorm;
    pub const withRowPrefixLinfNorm = table_methods_mod.withRowPrefixLinfNorm;
    pub const withRowCumulativeMinAbs = table_methods_mod.withRowCumulativeMinAbs;
    pub const withRowCumulativeMinAbsolute = table_methods_mod.withRowCumulativeMinAbsolute;
    pub const withRowCumMinAbs = table_methods_mod.withRowCumMinAbs;
    pub const withRowCumMinAbsolute = table_methods_mod.withRowCumMinAbsolute;
    pub const withRowPrefixMinAbs = table_methods_mod.withRowPrefixMinAbs;
    pub const withRowPrefixMinAbsolute = table_methods_mod.withRowPrefixMinAbsolute;
    pub const withRowCumulativeL1Norm = table_methods_mod.withRowCumulativeL1Norm;
    pub const withRowCumL1Norm = table_methods_mod.withRowCumL1Norm;
    pub const withRowPrefixL1Norm = table_methods_mod.withRowPrefixL1Norm;
    pub const withRowCumulativeL2Norm = table_methods_mod.withRowCumulativeL2Norm;
    pub const withRowCumL2Norm = table_methods_mod.withRowCumL2Norm;
    pub const withRowPrefixL2Norm = table_methods_mod.withRowPrefixL2Norm;
    pub const withRowCumulativeProduct = table_methods_mod.withRowCumulativeProduct;
    pub const withRowCumprod = table_methods_mod.withRowCumprod;
    pub const withRowCumProd = table_methods_mod.withRowCumProd;
    pub const withRowPrefixProduct = table_methods_mod.withRowPrefixProduct;
    pub const withRowCumulativeMax = table_methods_mod.withRowCumulativeMax;
    pub const withRowCummax = table_methods_mod.withRowCummax;
    pub const withRowCumMax = table_methods_mod.withRowCumMax;
    pub const withRowPrefixMax = table_methods_mod.withRowPrefixMax;
    pub const withRowCumulativeMin = table_methods_mod.withRowCumulativeMin;
    pub const withRowCummin = table_methods_mod.withRowCummin;
    pub const withRowCumMin = table_methods_mod.withRowCumMin;
    pub const withRowPrefixMin = table_methods_mod.withRowPrefixMin;
    pub const withRowCumulativeRange = table_methods_mod.withRowCumulativeRange;
    pub const withRowCumRange = table_methods_mod.withRowCumRange;
    pub const withRowPrefixRange = table_methods_mod.withRowPrefixRange;
    pub const withRowCumulativePtp = table_methods_mod.withRowCumulativePtp;
    pub const withRowCumPtp = table_methods_mod.withRowCumPtp;
    pub const withRowPrefixPtp = table_methods_mod.withRowPrefixPtp;
    pub const withRowIqrOutlier = table_methods_mod.withRowIqrOutlier;
    pub const withRowIqrOutliers = table_methods_mod.withRowIqrOutliers;
    pub const withRowTukeyOutlier = table_methods_mod.withRowTukeyOutlier;
    pub const withRowTukeyOutliers = table_methods_mod.withRowTukeyOutliers;
    pub const withRowMaxIndicator = table_methods_mod.withRowMaxIndicator;
    pub const withRowMaxIndicators = table_methods_mod.withRowMaxIndicators;
    pub const withRowIsMax = table_methods_mod.withRowIsMax;
    pub const withRowMaxMask = table_methods_mod.withRowMaxMask;
    pub const withRowMinIndicator = table_methods_mod.withRowMinIndicator;
    pub const withRowMinIndicators = table_methods_mod.withRowMinIndicators;
    pub const withRowIsMin = table_methods_mod.withRowIsMin;
    pub const withRowMinMask = table_methods_mod.withRowMinMask;
    pub const withRowTukeyWinsorize = table_methods_mod.withRowTukeyWinsorize;
    pub const withRowTukeyWinsorized = table_methods_mod.withRowTukeyWinsorized;
    pub const withRowIqrWinsorize = table_methods_mod.withRowIqrWinsorize;
    pub const withRowIqrWinsorized = table_methods_mod.withRowIqrWinsorized;
    pub const withRowMinMaxScale = table_methods_mod.withRowMinMaxScale;
    pub const withRowMinmaxScale = table_methods_mod.withRowMinmaxScale;
    pub const withRowL2Normalize = table_methods_mod.withRowL2Normalize;
    pub const withRowL2Normalized = table_methods_mod.withRowL2Normalized;
    pub const withRowL1Normalize = table_methods_mod.withRowL1Normalize;
    pub const withRowL1Normalized = table_methods_mod.withRowL1Normalized;
    pub const withRowSumNormalize = table_methods_mod.withRowSumNormalize;
    pub const withRowProportion = table_methods_mod.withRowProportion;
    pub const withRowShare = table_methods_mod.withRowShare;
    pub const withRowMeanNormalize = table_methods_mod.withRowMeanNormalize;
    pub const withRowMeanNormalized = table_methods_mod.withRowMeanNormalized;
    pub const withRowMeanRatio = table_methods_mod.withRowMeanRatio;
    pub const withRowMaxAbsNormalize = table_methods_mod.withRowMaxAbsNormalize;
    pub const withRowMaxabsNormalize = table_methods_mod.withRowMaxabsNormalize;
    pub const withRowLInfNormalize = table_methods_mod.withRowLInfNormalize;
    pub const withRowLinfNormalize = table_methods_mod.withRowLinfNormalize;
    pub const withRowSoftmax = table_methods_mod.withRowSoftmax;
    pub const withRowLogSoftmax = table_methods_mod.withRowLogSoftmax;
    pub const withRowLogsoftmax = table_methods_mod.withRowLogsoftmax;
    pub const withRowSoftmin = table_methods_mod.withRowSoftmin;
    pub const withRowLogSoftmin = table_methods_mod.withRowLogSoftmin;
    pub const withRowLogsoftmin = table_methods_mod.withRowLogsoftmin;
    pub const withRowSoftmaxEntropy = table_methods_mod.withRowSoftmaxEntropy;
    pub const withRowSoftmaxPerplexity = table_methods_mod.withRowSoftmaxPerplexity;
    pub const withRowSoftmaxConfidence = table_methods_mod.withRowSoftmaxConfidence;
    pub const withRowSoftmaxMargin = table_methods_mod.withRowSoftmaxMargin;
    pub const withRowSoftmaxEvenness = table_methods_mod.withRowSoftmaxEvenness;
    pub const withRowSoftmaxNormalizedEntropy = table_methods_mod.withRowSoftmaxNormalizedEntropy;
    pub const withRowSoftmaxConcentration = table_methods_mod.withRowSoftmaxConcentration;
    pub const withRowSoftmaxNormalizedHhi = table_methods_mod.withRowSoftmaxNormalizedHhi;
    pub const withRowSoftmaxNormalizedHHI = table_methods_mod.withRowSoftmaxNormalizedHHI;
    pub const withRowSoftmaxNhhi = table_methods_mod.withRowSoftmaxNhhi;
    pub const withRowSoftmaxGiniImpurity = table_methods_mod.withRowSoftmaxGiniImpurity;
    pub const withRowSoftmaxGini = table_methods_mod.withRowSoftmaxGini;
    pub const withRowSoftmaxInverseSimpson = table_methods_mod.withRowSoftmaxInverseSimpson;
    pub const withRowSoftmaxSimpsonEvenness = table_methods_mod.withRowSoftmaxSimpsonEvenness;
    pub const withRowSoftmaxSimpsonEven = table_methods_mod.withRowSoftmaxSimpsonEven;
    pub const withRowLogitMargin = table_methods_mod.withRowLogitMargin;
    pub const withRowGeometricMean = table_methods_mod.withRowGeometricMean;
    pub const withRowGeoMean = table_methods_mod.withRowGeoMean;
    pub const withRowMagnitudeGeometricMean = table_methods_mod.withRowMagnitudeGeometricMean;
    pub const withRowAbsGeometricMean = table_methods_mod.withRowAbsGeometricMean;
    pub const withRowMagnitudeGeoMean = table_methods_mod.withRowMagnitudeGeoMean;
    pub const withRowAbsGeoMean = table_methods_mod.withRowAbsGeoMean;
    pub const withRowHarmonicMean = table_methods_mod.withRowHarmonicMean;
    pub const withRowHarmMean = table_methods_mod.withRowHarmMean;
    pub const withRowProd = table_methods_mod.withRowProd;
    pub const withRowMin = table_methods_mod.withRowMin;
    pub const withRowMax = table_methods_mod.withRowMax;
    pub const withRowPtp = table_methods_mod.withRowPtp;
    pub const withRowMagnitudePtp = table_methods_mod.withRowMagnitudePtp;
    pub const withRowAbsPtp = table_methods_mod.withRowAbsPtp;
    pub const withRowMagnitudePeakToPeak = table_methods_mod.withRowMagnitudePeakToPeak;
    pub const withRowAbsPeakToPeak = table_methods_mod.withRowAbsPeakToPeak;
    pub const withRowMidrange = table_methods_mod.withRowMidrange;
    pub const withRowMagnitudeMidrange = table_methods_mod.withRowMagnitudeMidrange;
    pub const withRowAbsMidrange = table_methods_mod.withRowAbsMidrange;
    pub const withRowRangeCoeff = table_methods_mod.withRowRangeCoeff;
    pub const withRowRangeCoefficient = table_methods_mod.withRowRangeCoefficient;
    pub const withRowMagnitudeRangeCoeff = table_methods_mod.withRowMagnitudeRangeCoeff;
    pub const withRowAbsRangeCoeff = table_methods_mod.withRowAbsRangeCoeff;
    pub const withRowMagnitudeRangeCoefficient = table_methods_mod.withRowMagnitudeRangeCoefficient;
    pub const withRowAbsRangeCoefficient = table_methods_mod.withRowAbsRangeCoefficient;
    pub const withRowMeanAbs = table_methods_mod.withRowMeanAbs;
    pub const withRowHhi = table_methods_mod.withRowHhi;
    pub const withRowHerfindahl = table_methods_mod.withRowHerfindahl;
    pub const withRowHerfindahlHirschman = table_methods_mod.withRowHerfindahlHirschman;
    pub const withRowMagnitudeNormalizedHhi = table_methods_mod.withRowMagnitudeNormalizedHhi;
    pub const withRowAbsNormalizedHhi = table_methods_mod.withRowAbsNormalizedHhi;
    pub const withRowMagnitudeSparsity = table_methods_mod.withRowMagnitudeSparsity;
    pub const withRowAbsSparsity = table_methods_mod.withRowAbsSparsity;
    pub const withRowMagnitudeInverseSimpson = table_methods_mod.withRowMagnitudeInverseSimpson;
    pub const withRowAbsInverseSimpson = table_methods_mod.withRowAbsInverseSimpson;
    pub const withRowMagnitudeSimpsonEvenness = table_methods_mod.withRowMagnitudeSimpsonEvenness;
    pub const withRowAbsSimpsonEvenness = table_methods_mod.withRowAbsSimpsonEvenness;
    pub const withRowMagnitudeDominance = table_methods_mod.withRowMagnitudeDominance;
    pub const withRowAbsDominance = table_methods_mod.withRowAbsDominance;
    pub const withRowMagnitudeDominanceMargin = table_methods_mod.withRowMagnitudeDominanceMargin;
    pub const withRowAbsDominanceMargin = table_methods_mod.withRowAbsDominanceMargin;
    pub const withRowMagnitudeEntropy = table_methods_mod.withRowMagnitudeEntropy;
    pub const withRowAbsEntropy = table_methods_mod.withRowAbsEntropy;
    pub const withRowMagnitudePerplexity = table_methods_mod.withRowMagnitudePerplexity;
    pub const withRowAbsPerplexity = table_methods_mod.withRowAbsPerplexity;
    pub const withRowMagnitudeEvenness = table_methods_mod.withRowMagnitudeEvenness;
    pub const withRowAbsEvenness = table_methods_mod.withRowAbsEvenness;
    pub const withRowMeanAbsDev = table_methods_mod.withRowMeanAbsDev;
    pub const withRowGiniMeanDiff = table_methods_mod.withRowGiniMeanDiff;
    pub const withRowGiniCoefficient = table_methods_mod.withRowGiniCoefficient;
    pub const withRowGiniCoeff = table_methods_mod.withRowGiniCoeff;
    pub const withRowMeanAbsDevRatio = table_methods_mod.withRowMeanAbsDevRatio;
    pub const withRowRms = table_methods_mod.withRowRms;
    pub const withRowL1Norm = table_methods_mod.withRowL1Norm;
    pub const withRowL2Norm = table_methods_mod.withRowL2Norm;
    pub const withRowVariance = table_methods_mod.withRowVariance;
    pub const withRowVar = table_methods_mod.withRowVar;
    pub const withRowMagnitudeVariance = table_methods_mod.withRowMagnitudeVariance;
    pub const withRowAbsVariance = table_methods_mod.withRowAbsVariance;
    pub const withRowMagnitudeVar = table_methods_mod.withRowMagnitudeVar;
    pub const withRowAbsVar = table_methods_mod.withRowAbsVar;
    pub const withRowStddev = table_methods_mod.withRowStddev;
    pub const withRowStd = table_methods_mod.withRowStd;
    pub const withRowMagnitudeStddev = table_methods_mod.withRowMagnitudeStddev;
    pub const withRowAbsStddev = table_methods_mod.withRowAbsStddev;
    pub const withRowMagnitudeStd = table_methods_mod.withRowMagnitudeStd;
    pub const withRowAbsStd = table_methods_mod.withRowAbsStd;
    pub const withRowSem = table_methods_mod.withRowSem;
    pub const withRowMagnitudeSem = table_methods_mod.withRowMagnitudeSem;
    pub const withRowAbsSem = table_methods_mod.withRowAbsSem;
    pub const withRowCv = table_methods_mod.withRowCv;
    pub const withRowMagnitudeCv = table_methods_mod.withRowMagnitudeCv;
    pub const withRowAbsCv = table_methods_mod.withRowAbsCv;
    pub const withRowMagnitudeFano = table_methods_mod.withRowMagnitudeFano;
    pub const withRowAbsFano = table_methods_mod.withRowAbsFano;
    pub const withRowMagnitudeIndexOfDispersion = table_methods_mod.withRowMagnitudeIndexOfDispersion;
    pub const withRowAbsIndexOfDispersion = table_methods_mod.withRowAbsIndexOfDispersion;
    pub const withRowFano = table_methods_mod.withRowFano;
    pub const withRowIndexOfDispersion = table_methods_mod.withRowIndexOfDispersion;
    pub const withRowSkewness = table_methods_mod.withRowSkewness;
    pub const withRowSkew = table_methods_mod.withRowSkew;
    pub const withRowMagnitudeSkewness = table_methods_mod.withRowMagnitudeSkewness;
    pub const withRowAbsSkewness = table_methods_mod.withRowAbsSkewness;
    pub const withRowMagnitudeSkew = table_methods_mod.withRowMagnitudeSkew;
    pub const withRowAbsSkew = table_methods_mod.withRowAbsSkew;
    pub const withRowKurtosis = table_methods_mod.withRowKurtosis;
    pub const withRowKurt = table_methods_mod.withRowKurt;
    pub const withRowMagnitudeKurtosis = table_methods_mod.withRowMagnitudeKurtosis;
    pub const withRowAbsKurtosis = table_methods_mod.withRowAbsKurtosis;
    pub const withRowMagnitudeKurt = table_methods_mod.withRowMagnitudeKurt;
    pub const withRowAbsKurt = table_methods_mod.withRowAbsKurt;
    pub const withRowTrueCount = table_methods_mod.withRowTrueCount;
    pub const withRowFalseCount = table_methods_mod.withRowFalseCount;
    pub const withRowCumulativeTrueCount = table_methods_mod.withRowCumulativeTrueCount;
    pub const withRowCumTrueCount = table_methods_mod.withRowCumTrueCount;
    pub const withRowPrefixTrueCount = table_methods_mod.withRowPrefixTrueCount;
    pub const withRowCumulativeFalseCount = table_methods_mod.withRowCumulativeFalseCount;
    pub const withRowCumFalseCount = table_methods_mod.withRowCumFalseCount;
    pub const withRowPrefixFalseCount = table_methods_mod.withRowPrefixFalseCount;
    pub const withRowCumulativeTrueRatio = table_methods_mod.withRowCumulativeTrueRatio;
    pub const withRowCumTrueRatio = table_methods_mod.withRowCumTrueRatio;
    pub const withRowPrefixTrueRatio = table_methods_mod.withRowPrefixTrueRatio;
    pub const withRowCumulativeFalseRatio = table_methods_mod.withRowCumulativeFalseRatio;
    pub const withRowCumFalseRatio = table_methods_mod.withRowCumFalseRatio;
    pub const withRowPrefixFalseRatio = table_methods_mod.withRowPrefixFalseRatio;
    pub const withRowAnyTrue = table_methods_mod.withRowAnyTrue;
    pub const withRowAllTrue = table_methods_mod.withRowAllTrue;
    pub const withRowAnyFalse = table_methods_mod.withRowAnyFalse;
    pub const withRowAllFalse = table_methods_mod.withRowAllFalse;
    pub const withRowCumulativeAnyTrue = table_methods_mod.withRowCumulativeAnyTrue;
    pub const withRowCumAnyTrue = table_methods_mod.withRowCumAnyTrue;
    pub const withRowPrefixAnyTrue = table_methods_mod.withRowPrefixAnyTrue;
    pub const withRowCumulativeAllTrue = table_methods_mod.withRowCumulativeAllTrue;
    pub const withRowCumAllTrue = table_methods_mod.withRowCumAllTrue;
    pub const withRowPrefixAllTrue = table_methods_mod.withRowPrefixAllTrue;
    pub const withRowCumulativeAnyFalse = table_methods_mod.withRowCumulativeAnyFalse;
    pub const withRowCumAnyFalse = table_methods_mod.withRowCumAnyFalse;
    pub const withRowPrefixAnyFalse = table_methods_mod.withRowPrefixAnyFalse;
    pub const withRowCumulativeAllFalse = table_methods_mod.withRowCumulativeAllFalse;
    pub const withRowCumAllFalse = table_methods_mod.withRowCumAllFalse;
    pub const withRowPrefixAllFalse = table_methods_mod.withRowPrefixAllFalse;
    pub const withRowFirstTrueIndex = table_methods_mod.withRowFirstTrueIndex;
    pub const withRowLastTrueIndex = table_methods_mod.withRowLastTrueIndex;
    pub const withRowFirstFalseIndex = table_methods_mod.withRowFirstFalseIndex;
    pub const withRowLastFalseIndex = table_methods_mod.withRowLastFalseIndex;
    pub const withRowCumulativeFirstTrueIndex = table_methods_mod.withRowCumulativeFirstTrueIndex;
    pub const withRowPrefixFirstTrueIndex = table_methods_mod.withRowPrefixFirstTrueIndex;
    pub const withRowCumulativeLastTrueIndex = table_methods_mod.withRowCumulativeLastTrueIndex;
    pub const withRowPrefixLastTrueIndex = table_methods_mod.withRowPrefixLastTrueIndex;
    pub const withRowCumulativeFirstFalseIndex = table_methods_mod.withRowCumulativeFirstFalseIndex;
    pub const withRowPrefixFirstFalseIndex = table_methods_mod.withRowPrefixFirstFalseIndex;
    pub const withRowCumulativeLastFalseIndex = table_methods_mod.withRowCumulativeLastFalseIndex;
    pub const withRowPrefixLastFalseIndex = table_methods_mod.withRowPrefixLastFalseIndex;
    pub const withRowTrueRatio = table_methods_mod.withRowTrueRatio;
    pub const withRowFalseRatio = table_methods_mod.withRowFalseRatio;
    pub const withRowNaNCount = table_methods_mod.withRowNaNCount;
    pub const withRowNaNRatio = table_methods_mod.withRowNaNRatio;
    pub const withRowNanRatio = table_methods_mod.withRowNanRatio;
    pub const withRowInfCount = table_methods_mod.withRowInfCount;
    pub const withRowInfRatio = table_methods_mod.withRowInfRatio;
    pub const withRowPositiveInfCount = table_methods_mod.withRowPositiveInfCount;
    pub const withRowNegativeInfCount = table_methods_mod.withRowNegativeInfCount;
    pub const withRowPositiveInfRatio = table_methods_mod.withRowPositiveInfRatio;
    pub const withRowNegativeInfRatio = table_methods_mod.withRowNegativeInfRatio;
    pub const withRowZeroCount = table_methods_mod.withRowZeroCount;
    pub const withRowZeroRatio = table_methods_mod.withRowZeroRatio;
    pub const withRowPositiveZeroCount = table_methods_mod.withRowPositiveZeroCount;
    pub const withRowNegativeZeroCount = table_methods_mod.withRowNegativeZeroCount;
    pub const withRowPositiveZeroRatio = table_methods_mod.withRowPositiveZeroRatio;
    pub const withRowNegativeZeroRatio = table_methods_mod.withRowNegativeZeroRatio;
    pub const withRowNonZeroCount = table_methods_mod.withRowNonZeroCount;
    pub const withRowNonZeroRatio = table_methods_mod.withRowNonZeroRatio;
    pub const withRowAnyZero = table_methods_mod.withRowAnyZero;
    pub const withRowAllZero = table_methods_mod.withRowAllZero;
    pub const withRowAnyNonZero = table_methods_mod.withRowAnyNonZero;
    pub const withRowAllNonZero = table_methods_mod.withRowAllNonZero;
    pub const withRowAnyPositiveZero = table_methods_mod.withRowAnyPositiveZero;
    pub const withRowAllPositiveZero = table_methods_mod.withRowAllPositiveZero;
    pub const withRowAnyNegativeZero = table_methods_mod.withRowAnyNegativeZero;
    pub const withRowAllNegativeZero = table_methods_mod.withRowAllNegativeZero;
    pub const withRowAnyPositive = table_methods_mod.withRowAnyPositive;
    pub const withRowAllPositive = table_methods_mod.withRowAllPositive;
    pub const withRowAnySignBit = table_methods_mod.withRowAnySignBit;
    pub const withRowAllSignBit = table_methods_mod.withRowAllSignBit;
    pub const withRowAnyNegative = table_methods_mod.withRowAnyNegative;
    pub const withRowAllNegative = table_methods_mod.withRowAllNegative;
    pub const withRowAnyNaN = table_methods_mod.withRowAnyNaN;
    pub const withRowAllNaN = table_methods_mod.withRowAllNaN;
    pub const withRowAnyInf = table_methods_mod.withRowAnyInf;
    pub const withRowAllInf = table_methods_mod.withRowAllInf;
    pub const withRowAnyPositiveInf = table_methods_mod.withRowAnyPositiveInf;
    pub const withRowAllPositiveInf = table_methods_mod.withRowAllPositiveInf;
    pub const withRowAnyNegativeInf = table_methods_mod.withRowAnyNegativeInf;
    pub const withRowAllNegativeInf = table_methods_mod.withRowAllNegativeInf;
    pub const withRowAnyFinite = table_methods_mod.withRowAnyFinite;
    pub const withRowAllFinite = table_methods_mod.withRowAllFinite;
    pub const withRowAnyNormal = table_methods_mod.withRowAnyNormal;
    pub const withRowAllNormal = table_methods_mod.withRowAllNormal;
    pub const withRowAnySubnormal = table_methods_mod.withRowAnySubnormal;
    pub const withRowAllSubnormal = table_methods_mod.withRowAllSubnormal;
    pub const withRowAnyNonFinite = table_methods_mod.withRowAnyNonFinite;
    pub const withRowAllNonFinite = table_methods_mod.withRowAllNonFinite;
    pub const withRowFirstNaNIndex = table_methods_mod.withRowFirstNaNIndex;
    pub const withRowFirstNanIndex = table_methods_mod.withRowFirstNanIndex;
    pub const withRowLastNaNIndex = table_methods_mod.withRowLastNaNIndex;
    pub const withRowLastNanIndex = table_methods_mod.withRowLastNanIndex;
    pub const withRowFirstInfIndex = table_methods_mod.withRowFirstInfIndex;
    pub const withRowLastInfIndex = table_methods_mod.withRowLastInfIndex;
    pub const withRowFirstPositiveInfIndex = table_methods_mod.withRowFirstPositiveInfIndex;
    pub const withRowLastPositiveInfIndex = table_methods_mod.withRowLastPositiveInfIndex;
    pub const withRowFirstNegativeInfIndex = table_methods_mod.withRowFirstNegativeInfIndex;
    pub const withRowLastNegativeInfIndex = table_methods_mod.withRowLastNegativeInfIndex;
    pub const withRowFirstPositiveZeroIndex = table_methods_mod.withRowFirstPositiveZeroIndex;
    pub const withRowLastPositiveZeroIndex = table_methods_mod.withRowLastPositiveZeroIndex;
    pub const withRowFirstNegativeZeroIndex = table_methods_mod.withRowFirstNegativeZeroIndex;
    pub const withRowLastNegativeZeroIndex = table_methods_mod.withRowLastNegativeZeroIndex;
    pub const withRowFirstSignBitIndex = table_methods_mod.withRowFirstSignBitIndex;
    pub const withRowLastSignBitIndex = table_methods_mod.withRowLastSignBitIndex;
    pub const withRowFirstFiniteIndex = table_methods_mod.withRowFirstFiniteIndex;
    pub const withRowLastFiniteIndex = table_methods_mod.withRowLastFiniteIndex;
    pub const withRowFirstNormalIndex = table_methods_mod.withRowFirstNormalIndex;
    pub const withRowLastNormalIndex = table_methods_mod.withRowLastNormalIndex;
    pub const withRowFirstSubnormalIndex = table_methods_mod.withRowFirstSubnormalIndex;
    pub const withRowLastSubnormalIndex = table_methods_mod.withRowLastSubnormalIndex;
    pub const withRowFirstNonFiniteIndex = table_methods_mod.withRowFirstNonFiniteIndex;
    pub const withRowFirstNonfiniteIndex = table_methods_mod.withRowFirstNonfiniteIndex;
    pub const withRowLastNonFiniteIndex = table_methods_mod.withRowLastNonFiniteIndex;
    pub const withRowLastNonfiniteIndex = table_methods_mod.withRowLastNonfiniteIndex;
    pub const withRowFirstZeroIndex = table_methods_mod.withRowFirstZeroIndex;
    pub const withRowLastZeroIndex = table_methods_mod.withRowLastZeroIndex;
    pub const withRowFirstNonZeroIndex = table_methods_mod.withRowFirstNonZeroIndex;
    pub const withRowFirstNonzeroIndex = table_methods_mod.withRowFirstNonzeroIndex;
    pub const withRowLastNonZeroIndex = table_methods_mod.withRowLastNonZeroIndex;
    pub const withRowLastNonzeroIndex = table_methods_mod.withRowLastNonzeroIndex;
    pub const withRowFirstPositiveIndex = table_methods_mod.withRowFirstPositiveIndex;
    pub const withRowLastPositiveIndex = table_methods_mod.withRowLastPositiveIndex;
    pub const withRowFirstNegativeIndex = table_methods_mod.withRowFirstNegativeIndex;
    pub const withRowLastNegativeIndex = table_methods_mod.withRowLastNegativeIndex;
    pub const withRowPositiveCount = table_methods_mod.withRowPositiveCount;
    pub const withRowPositiveRatio = table_methods_mod.withRowPositiveRatio;
    pub const withRowSignBitCount = table_methods_mod.withRowSignBitCount;
    pub const withRowSignBitRatio = table_methods_mod.withRowSignBitRatio;
    pub const withRowNegativeCount = table_methods_mod.withRowNegativeCount;
    pub const withRowNegativeRatio = table_methods_mod.withRowNegativeRatio;

    pub const withRowCumulativePositiveZeroCount = table_methods_mod.withRowCumulativePositiveZeroCount;
    pub const withRowCumPositiveZeroCount = table_methods_mod.withRowCumPositiveZeroCount;
    pub const withRowPrefixPositiveZeroCount = table_methods_mod.withRowPrefixPositiveZeroCount;
    pub const withRowCumulativePositiveZeroRatio = table_methods_mod.withRowCumulativePositiveZeroRatio;
    pub const withRowCumPositiveZeroRatio = table_methods_mod.withRowCumPositiveZeroRatio;
    pub const withRowPrefixPositiveZeroRatio = table_methods_mod.withRowPrefixPositiveZeroRatio;
    pub const withRowCumulativeNegativeZeroCount = table_methods_mod.withRowCumulativeNegativeZeroCount;
    pub const withRowCumNegativeZeroCount = table_methods_mod.withRowCumNegativeZeroCount;
    pub const withRowPrefixNegativeZeroCount = table_methods_mod.withRowPrefixNegativeZeroCount;
    pub const withRowCumulativeNegativeZeroRatio = table_methods_mod.withRowCumulativeNegativeZeroRatio;
    pub const withRowCumNegativeZeroRatio = table_methods_mod.withRowCumNegativeZeroRatio;
    pub const withRowPrefixNegativeZeroRatio = table_methods_mod.withRowPrefixNegativeZeroRatio;
    pub const withRowCumulativeSignBitCount = table_methods_mod.withRowCumulativeSignBitCount;
    pub const withRowCumSignBitCount = table_methods_mod.withRowCumSignBitCount;
    pub const withRowPrefixSignBitCount = table_methods_mod.withRowPrefixSignBitCount;
    pub const withRowCumulativeSignBitRatio = table_methods_mod.withRowCumulativeSignBitRatio;
    pub const withRowCumSignBitRatio = table_methods_mod.withRowCumSignBitRatio;
    pub const withRowPrefixSignBitRatio = table_methods_mod.withRowPrefixSignBitRatio;
    pub const withRowCumulativeAnyZero = table_methods_mod.withRowCumulativeAnyZero;
    pub const withRowCumAnyZero = table_methods_mod.withRowCumAnyZero;
    pub const withRowPrefixAnyZero = table_methods_mod.withRowPrefixAnyZero;
    pub const withRowCumulativeAllZero = table_methods_mod.withRowCumulativeAllZero;
    pub const withRowCumAllZero = table_methods_mod.withRowCumAllZero;
    pub const withRowPrefixAllZero = table_methods_mod.withRowPrefixAllZero;
    pub const withRowCumulativeAnyNonZero = table_methods_mod.withRowCumulativeAnyNonZero;
    pub const withRowCumAnyNonZero = table_methods_mod.withRowCumAnyNonZero;
    pub const withRowPrefixAnyNonZero = table_methods_mod.withRowPrefixAnyNonZero;
    pub const withRowCumulativeAllNonZero = table_methods_mod.withRowCumulativeAllNonZero;
    pub const withRowCumAllNonZero = table_methods_mod.withRowCumAllNonZero;
    pub const withRowPrefixAllNonZero = table_methods_mod.withRowPrefixAllNonZero;
    pub const withRowCumulativeAnyPositiveZero = table_methods_mod.withRowCumulativeAnyPositiveZero;
    pub const withRowCumAnyPositiveZero = table_methods_mod.withRowCumAnyPositiveZero;
    pub const withRowPrefixAnyPositiveZero = table_methods_mod.withRowPrefixAnyPositiveZero;
    pub const withRowCumulativeAllPositiveZero = table_methods_mod.withRowCumulativeAllPositiveZero;
    pub const withRowCumAllPositiveZero = table_methods_mod.withRowCumAllPositiveZero;
    pub const withRowPrefixAllPositiveZero = table_methods_mod.withRowPrefixAllPositiveZero;
    pub const withRowCumulativeAnyNegativeZero = table_methods_mod.withRowCumulativeAnyNegativeZero;
    pub const withRowCumAnyNegativeZero = table_methods_mod.withRowCumAnyNegativeZero;
    pub const withRowPrefixAnyNegativeZero = table_methods_mod.withRowPrefixAnyNegativeZero;
    pub const withRowCumulativeAllNegativeZero = table_methods_mod.withRowCumulativeAllNegativeZero;
    pub const withRowCumAllNegativeZero = table_methods_mod.withRowCumAllNegativeZero;
    pub const withRowPrefixAllNegativeZero = table_methods_mod.withRowPrefixAllNegativeZero;
    pub const withRowCumulativeAnyPositive = table_methods_mod.withRowCumulativeAnyPositive;
    pub const withRowCumAnyPositive = table_methods_mod.withRowCumAnyPositive;
    pub const withRowPrefixAnyPositive = table_methods_mod.withRowPrefixAnyPositive;
    pub const withRowCumulativeAllPositive = table_methods_mod.withRowCumulativeAllPositive;
    pub const withRowCumAllPositive = table_methods_mod.withRowCumAllPositive;
    pub const withRowPrefixAllPositive = table_methods_mod.withRowPrefixAllPositive;
    pub const withRowCumulativeAnySignBit = table_methods_mod.withRowCumulativeAnySignBit;
    pub const withRowCumAnySignBit = table_methods_mod.withRowCumAnySignBit;
    pub const withRowPrefixAnySignBit = table_methods_mod.withRowPrefixAnySignBit;
    pub const withRowCumulativeAllSignBit = table_methods_mod.withRowCumulativeAllSignBit;
    pub const withRowCumAllSignBit = table_methods_mod.withRowCumAllSignBit;
    pub const withRowPrefixAllSignBit = table_methods_mod.withRowPrefixAllSignBit;
    pub const withRowCumulativeAnyNegative = table_methods_mod.withRowCumulativeAnyNegative;
    pub const withRowCumAnyNegative = table_methods_mod.withRowCumAnyNegative;
    pub const withRowPrefixAnyNegative = table_methods_mod.withRowPrefixAnyNegative;
    pub const withRowCumulativeAllNegative = table_methods_mod.withRowCumulativeAllNegative;
    pub const withRowCumAllNegative = table_methods_mod.withRowCumAllNegative;
    pub const withRowPrefixAllNegative = table_methods_mod.withRowPrefixAllNegative;
    pub const withRowCumulativeAnyNaN = table_methods_mod.withRowCumulativeAnyNaN;
    pub const withRowCumAnyNaN = table_methods_mod.withRowCumAnyNaN;
    pub const withRowPrefixAnyNaN = table_methods_mod.withRowPrefixAnyNaN;
    pub const withRowCumulativeAllNaN = table_methods_mod.withRowCumulativeAllNaN;
    pub const withRowCumAllNaN = table_methods_mod.withRowCumAllNaN;
    pub const withRowPrefixAllNaN = table_methods_mod.withRowPrefixAllNaN;
    pub const withRowCumulativeAnyInf = table_methods_mod.withRowCumulativeAnyInf;
    pub const withRowCumAnyInf = table_methods_mod.withRowCumAnyInf;
    pub const withRowPrefixAnyInf = table_methods_mod.withRowPrefixAnyInf;
    pub const withRowCumulativeAllInf = table_methods_mod.withRowCumulativeAllInf;
    pub const withRowCumAllInf = table_methods_mod.withRowCumAllInf;
    pub const withRowPrefixAllInf = table_methods_mod.withRowPrefixAllInf;
    pub const withRowCumulativeAnyPositiveInf = table_methods_mod.withRowCumulativeAnyPositiveInf;
    pub const withRowCumAnyPositiveInf = table_methods_mod.withRowCumAnyPositiveInf;
    pub const withRowPrefixAnyPositiveInf = table_methods_mod.withRowPrefixAnyPositiveInf;
    pub const withRowCumulativeAllPositiveInf = table_methods_mod.withRowCumulativeAllPositiveInf;
    pub const withRowCumAllPositiveInf = table_methods_mod.withRowCumAllPositiveInf;
    pub const withRowPrefixAllPositiveInf = table_methods_mod.withRowPrefixAllPositiveInf;
    pub const withRowCumulativeAnyNegativeInf = table_methods_mod.withRowCumulativeAnyNegativeInf;
    pub const withRowCumAnyNegativeInf = table_methods_mod.withRowCumAnyNegativeInf;
    pub const withRowPrefixAnyNegativeInf = table_methods_mod.withRowPrefixAnyNegativeInf;
    pub const withRowCumulativeAllNegativeInf = table_methods_mod.withRowCumulativeAllNegativeInf;
    pub const withRowCumAllNegativeInf = table_methods_mod.withRowCumAllNegativeInf;
    pub const withRowPrefixAllNegativeInf = table_methods_mod.withRowPrefixAllNegativeInf;
    pub const withRowCumulativeAnyFinite = table_methods_mod.withRowCumulativeAnyFinite;
    pub const withRowCumAnyFinite = table_methods_mod.withRowCumAnyFinite;
    pub const withRowPrefixAnyFinite = table_methods_mod.withRowPrefixAnyFinite;
    pub const withRowCumulativeAllFinite = table_methods_mod.withRowCumulativeAllFinite;
    pub const withRowCumAllFinite = table_methods_mod.withRowCumAllFinite;
    pub const withRowPrefixAllFinite = table_methods_mod.withRowPrefixAllFinite;
    pub const withRowCumulativeAnyNormal = table_methods_mod.withRowCumulativeAnyNormal;
    pub const withRowCumAnyNormal = table_methods_mod.withRowCumAnyNormal;
    pub const withRowPrefixAnyNormal = table_methods_mod.withRowPrefixAnyNormal;
    pub const withRowCumulativeAllNormal = table_methods_mod.withRowCumulativeAllNormal;
    pub const withRowCumAllNormal = table_methods_mod.withRowCumAllNormal;
    pub const withRowPrefixAllNormal = table_methods_mod.withRowPrefixAllNormal;
    pub const withRowCumulativeAnySubnormal = table_methods_mod.withRowCumulativeAnySubnormal;
    pub const withRowCumAnySubnormal = table_methods_mod.withRowCumAnySubnormal;
    pub const withRowPrefixAnySubnormal = table_methods_mod.withRowPrefixAnySubnormal;
    pub const withRowCumulativeAllSubnormal = table_methods_mod.withRowCumulativeAllSubnormal;
    pub const withRowCumAllSubnormal = table_methods_mod.withRowCumAllSubnormal;
    pub const withRowPrefixAllSubnormal = table_methods_mod.withRowPrefixAllSubnormal;
    pub const withRowCumulativeAnyNonFinite = table_methods_mod.withRowCumulativeAnyNonFinite;
    pub const withRowCumAnyNonFinite = table_methods_mod.withRowCumAnyNonFinite;
    pub const withRowPrefixAnyNonFinite = table_methods_mod.withRowPrefixAnyNonFinite;
    pub const withRowCumulativeAllNonFinite = table_methods_mod.withRowCumulativeAllNonFinite;
    pub const withRowCumAllNonFinite = table_methods_mod.withRowCumAllNonFinite;
    pub const withRowPrefixAllNonFinite = table_methods_mod.withRowPrefixAllNonFinite;
    pub const withRowCumulativeNaNCount = table_methods_mod.withRowCumulativeNaNCount;
    pub const withRowCumNaNCount = table_methods_mod.withRowCumNaNCount;
    pub const withRowPrefixNaNCount = table_methods_mod.withRowPrefixNaNCount;
    pub const withRowCumulativeNaNRatio = table_methods_mod.withRowCumulativeNaNRatio;
    pub const withRowCumNaNRatio = table_methods_mod.withRowCumNaNRatio;
    pub const withRowPrefixNaNRatio = table_methods_mod.withRowPrefixNaNRatio;
    pub const withRowCumulativeInfCount = table_methods_mod.withRowCumulativeInfCount;
    pub const withRowCumInfCount = table_methods_mod.withRowCumInfCount;
    pub const withRowPrefixInfCount = table_methods_mod.withRowPrefixInfCount;
    pub const withRowCumulativeInfRatio = table_methods_mod.withRowCumulativeInfRatio;
    pub const withRowCumInfRatio = table_methods_mod.withRowCumInfRatio;
    pub const withRowPrefixInfRatio = table_methods_mod.withRowPrefixInfRatio;
    pub const withRowCumulativePositiveInfCount = table_methods_mod.withRowCumulativePositiveInfCount;
    pub const withRowCumPositiveInfCount = table_methods_mod.withRowCumPositiveInfCount;
    pub const withRowPrefixPositiveInfCount = table_methods_mod.withRowPrefixPositiveInfCount;
    pub const withRowCumulativePositiveInfRatio = table_methods_mod.withRowCumulativePositiveInfRatio;
    pub const withRowCumPositiveInfRatio = table_methods_mod.withRowCumPositiveInfRatio;
    pub const withRowPrefixPositiveInfRatio = table_methods_mod.withRowPrefixPositiveInfRatio;
    pub const withRowCumulativeNegativeInfCount = table_methods_mod.withRowCumulativeNegativeInfCount;
    pub const withRowCumNegativeInfCount = table_methods_mod.withRowCumNegativeInfCount;
    pub const withRowPrefixNegativeInfCount = table_methods_mod.withRowPrefixNegativeInfCount;
    pub const withRowCumulativeNegativeInfRatio = table_methods_mod.withRowCumulativeNegativeInfRatio;
    pub const withRowCumNegativeInfRatio = table_methods_mod.withRowCumNegativeInfRatio;
    pub const withRowPrefixNegativeInfRatio = table_methods_mod.withRowPrefixNegativeInfRatio;
    pub const withRowCumulativeFiniteCount = table_methods_mod.withRowCumulativeFiniteCount;
    pub const withRowCumFiniteCount = table_methods_mod.withRowCumFiniteCount;
    pub const withRowPrefixFiniteCount = table_methods_mod.withRowPrefixFiniteCount;
    pub const withRowCumulativeFiniteRatio = table_methods_mod.withRowCumulativeFiniteRatio;
    pub const withRowCumFiniteRatio = table_methods_mod.withRowCumFiniteRatio;
    pub const withRowPrefixFiniteRatio = table_methods_mod.withRowPrefixFiniteRatio;
    pub const withRowCumulativeNormalCount = table_methods_mod.withRowCumulativeNormalCount;
    pub const withRowCumNormalCount = table_methods_mod.withRowCumNormalCount;
    pub const withRowPrefixNormalCount = table_methods_mod.withRowPrefixNormalCount;
    pub const withRowCumulativeNormalRatio = table_methods_mod.withRowCumulativeNormalRatio;
    pub const withRowCumNormalRatio = table_methods_mod.withRowCumNormalRatio;
    pub const withRowPrefixNormalRatio = table_methods_mod.withRowPrefixNormalRatio;
    pub const withRowCumulativeSubnormalCount = table_methods_mod.withRowCumulativeSubnormalCount;
    pub const withRowCumSubnormalCount = table_methods_mod.withRowCumSubnormalCount;
    pub const withRowPrefixSubnormalCount = table_methods_mod.withRowPrefixSubnormalCount;
    pub const withRowCumulativeSubnormalRatio = table_methods_mod.withRowCumulativeSubnormalRatio;
    pub const withRowCumSubnormalRatio = table_methods_mod.withRowCumSubnormalRatio;
    pub const withRowPrefixSubnormalRatio = table_methods_mod.withRowPrefixSubnormalRatio;
    pub const withRowCumulativeNonFiniteCount = table_methods_mod.withRowCumulativeNonFiniteCount;
    pub const withRowCumNonFiniteCount = table_methods_mod.withRowCumNonFiniteCount;
    pub const withRowPrefixNonFiniteCount = table_methods_mod.withRowPrefixNonFiniteCount;
    pub const withRowCumulativeNonFiniteRatio = table_methods_mod.withRowCumulativeNonFiniteRatio;
    pub const withRowCumNonFiniteRatio = table_methods_mod.withRowCumNonFiniteRatio;
    pub const withRowPrefixNonFiniteRatio = table_methods_mod.withRowPrefixNonFiniteRatio;
    pub const withRowCumulativeZeroCount = table_methods_mod.withRowCumulativeZeroCount;
    pub const withRowCumZeroCount = table_methods_mod.withRowCumZeroCount;
    pub const withRowPrefixZeroCount = table_methods_mod.withRowPrefixZeroCount;
    pub const withRowCumulativeFirstNaNIndex = table_methods_mod.withRowCumulativeFirstNaNIndex;
    pub const withRowPrefixFirstNaNIndex = table_methods_mod.withRowPrefixFirstNaNIndex;
    pub const withRowCumulativeLastNaNIndex = table_methods_mod.withRowCumulativeLastNaNIndex;
    pub const withRowPrefixLastNaNIndex = table_methods_mod.withRowPrefixLastNaNIndex;
    pub const withRowCumulativeFirstInfIndex = table_methods_mod.withRowCumulativeFirstInfIndex;
    pub const withRowPrefixFirstInfIndex = table_methods_mod.withRowPrefixFirstInfIndex;
    pub const withRowCumulativeLastInfIndex = table_methods_mod.withRowCumulativeLastInfIndex;
    pub const withRowPrefixLastInfIndex = table_methods_mod.withRowPrefixLastInfIndex;
    pub const withRowCumulativeFirstPositiveInfIndex = table_methods_mod.withRowCumulativeFirstPositiveInfIndex;
    pub const withRowPrefixFirstPositiveInfIndex = table_methods_mod.withRowPrefixFirstPositiveInfIndex;
    pub const withRowCumulativeLastPositiveInfIndex = table_methods_mod.withRowCumulativeLastPositiveInfIndex;
    pub const withRowPrefixLastPositiveInfIndex = table_methods_mod.withRowPrefixLastPositiveInfIndex;
    pub const withRowCumulativeFirstNegativeInfIndex = table_methods_mod.withRowCumulativeFirstNegativeInfIndex;
    pub const withRowPrefixFirstNegativeInfIndex = table_methods_mod.withRowPrefixFirstNegativeInfIndex;
    pub const withRowCumulativeLastNegativeInfIndex = table_methods_mod.withRowCumulativeLastNegativeInfIndex;
    pub const withRowPrefixLastNegativeInfIndex = table_methods_mod.withRowPrefixLastNegativeInfIndex;
    pub const withRowCumulativeFirstFiniteIndex = table_methods_mod.withRowCumulativeFirstFiniteIndex;
    pub const withRowPrefixFirstFiniteIndex = table_methods_mod.withRowPrefixFirstFiniteIndex;
    pub const withRowCumulativeLastFiniteIndex = table_methods_mod.withRowCumulativeLastFiniteIndex;
    pub const withRowPrefixLastFiniteIndex = table_methods_mod.withRowPrefixLastFiniteIndex;
    pub const withRowCumulativeFirstNormalIndex = table_methods_mod.withRowCumulativeFirstNormalIndex;
    pub const withRowPrefixFirstNormalIndex = table_methods_mod.withRowPrefixFirstNormalIndex;
    pub const withRowCumulativeLastNormalIndex = table_methods_mod.withRowCumulativeLastNormalIndex;
    pub const withRowPrefixLastNormalIndex = table_methods_mod.withRowPrefixLastNormalIndex;
    pub const withRowCumulativeFirstSubnormalIndex = table_methods_mod.withRowCumulativeFirstSubnormalIndex;
    pub const withRowPrefixFirstSubnormalIndex = table_methods_mod.withRowPrefixFirstSubnormalIndex;
    pub const withRowCumulativeLastSubnormalIndex = table_methods_mod.withRowCumulativeLastSubnormalIndex;
    pub const withRowPrefixLastSubnormalIndex = table_methods_mod.withRowPrefixLastSubnormalIndex;
    pub const withRowCumulativeFirstNonFiniteIndex = table_methods_mod.withRowCumulativeFirstNonFiniteIndex;
    pub const withRowPrefixFirstNonFiniteIndex = table_methods_mod.withRowPrefixFirstNonFiniteIndex;
    pub const withRowCumulativeLastNonFiniteIndex = table_methods_mod.withRowCumulativeLastNonFiniteIndex;
    pub const withRowPrefixLastNonFiniteIndex = table_methods_mod.withRowPrefixLastNonFiniteIndex;
    pub const withRowCumulativeFirstZeroIndex = table_methods_mod.withRowCumulativeFirstZeroIndex;
    pub const withRowPrefixFirstZeroIndex = table_methods_mod.withRowPrefixFirstZeroIndex;
    pub const withRowCumulativeLastZeroIndex = table_methods_mod.withRowCumulativeLastZeroIndex;
    pub const withRowPrefixLastZeroIndex = table_methods_mod.withRowPrefixLastZeroIndex;
    pub const withRowCumulativeFirstPositiveZeroIndex = table_methods_mod.withRowCumulativeFirstPositiveZeroIndex;
    pub const withRowPrefixFirstPositiveZeroIndex = table_methods_mod.withRowPrefixFirstPositiveZeroIndex;
    pub const withRowCumulativeLastPositiveZeroIndex = table_methods_mod.withRowCumulativeLastPositiveZeroIndex;
    pub const withRowPrefixLastPositiveZeroIndex = table_methods_mod.withRowPrefixLastPositiveZeroIndex;
    pub const withRowCumulativeFirstNegativeZeroIndex = table_methods_mod.withRowCumulativeFirstNegativeZeroIndex;
    pub const withRowPrefixFirstNegativeZeroIndex = table_methods_mod.withRowPrefixFirstNegativeZeroIndex;
    pub const withRowCumulativeLastNegativeZeroIndex = table_methods_mod.withRowCumulativeLastNegativeZeroIndex;
    pub const withRowPrefixLastNegativeZeroIndex = table_methods_mod.withRowPrefixLastNegativeZeroIndex;
    pub const withRowCumulativeNonZeroCount = table_methods_mod.withRowCumulativeNonZeroCount;
    pub const withRowCumNonZeroCount = table_methods_mod.withRowCumNonZeroCount;
    pub const withRowPrefixNonZeroCount = table_methods_mod.withRowPrefixNonZeroCount;
    pub const withRowCumulativeFirstNonZeroIndex = table_methods_mod.withRowCumulativeFirstNonZeroIndex;
    pub const withRowCumulativeFirstNonzeroIndex = table_methods_mod.withRowCumulativeFirstNonzeroIndex;
    pub const withRowPrefixFirstNonZeroIndex = table_methods_mod.withRowPrefixFirstNonZeroIndex;
    pub const withRowPrefixFirstNonzeroIndex = table_methods_mod.withRowPrefixFirstNonzeroIndex;
    pub const withRowCumulativeLastNonZeroIndex = table_methods_mod.withRowCumulativeLastNonZeroIndex;
    pub const withRowCumulativeLastNonzeroIndex = table_methods_mod.withRowCumulativeLastNonzeroIndex;
    pub const withRowPrefixLastNonZeroIndex = table_methods_mod.withRowPrefixLastNonZeroIndex;
    pub const withRowPrefixLastNonzeroIndex = table_methods_mod.withRowPrefixLastNonzeroIndex;
    pub const withRowCumulativeFirstPositiveIndex = table_methods_mod.withRowCumulativeFirstPositiveIndex;
    pub const withRowPrefixFirstPositiveIndex = table_methods_mod.withRowPrefixFirstPositiveIndex;
    pub const withRowCumulativeLastPositiveIndex = table_methods_mod.withRowCumulativeLastPositiveIndex;
    pub const withRowPrefixLastPositiveIndex = table_methods_mod.withRowPrefixLastPositiveIndex;
    pub const withRowCumulativeFirstSignBitIndex = table_methods_mod.withRowCumulativeFirstSignBitIndex;
    pub const withRowPrefixFirstSignBitIndex = table_methods_mod.withRowPrefixFirstSignBitIndex;
    pub const withRowCumulativeLastSignBitIndex = table_methods_mod.withRowCumulativeLastSignBitIndex;
    pub const withRowPrefixLastSignBitIndex = table_methods_mod.withRowPrefixLastSignBitIndex;
    pub const withRowCumulativeFirstNegativeIndex = table_methods_mod.withRowCumulativeFirstNegativeIndex;
    pub const withRowPrefixFirstNegativeIndex = table_methods_mod.withRowPrefixFirstNegativeIndex;
    pub const withRowCumulativeLastNegativeIndex = table_methods_mod.withRowCumulativeLastNegativeIndex;
    pub const withRowPrefixLastNegativeIndex = table_methods_mod.withRowPrefixLastNegativeIndex;
    pub const withRowCumulativePositiveCount = table_methods_mod.withRowCumulativePositiveCount;
    pub const withRowCumPositiveCount = table_methods_mod.withRowCumPositiveCount;
    pub const withRowPrefixPositiveCount = table_methods_mod.withRowPrefixPositiveCount;
    pub const withRowCumulativeNegativeCount = table_methods_mod.withRowCumulativeNegativeCount;
    pub const withRowCumNegativeCount = table_methods_mod.withRowCumNegativeCount;
    pub const withRowPrefixNegativeCount = table_methods_mod.withRowPrefixNegativeCount;
    pub const withRowCumulativeZeroRatio = table_methods_mod.withRowCumulativeZeroRatio;
    pub const withRowCumZeroRatio = table_methods_mod.withRowCumZeroRatio;
    pub const withRowPrefixZeroRatio = table_methods_mod.withRowPrefixZeroRatio;
    pub const withRowCumulativeNonZeroRatio = table_methods_mod.withRowCumulativeNonZeroRatio;
    pub const withRowCumNonZeroRatio = table_methods_mod.withRowCumNonZeroRatio;
    pub const withRowPrefixNonZeroRatio = table_methods_mod.withRowPrefixNonZeroRatio;
    pub const withRowCumulativePositiveRatio = table_methods_mod.withRowCumulativePositiveRatio;
    pub const withRowCumPositiveRatio = table_methods_mod.withRowCumPositiveRatio;
    pub const withRowPrefixPositiveRatio = table_methods_mod.withRowPrefixPositiveRatio;
    pub const withRowCumulativeNegativeRatio = table_methods_mod.withRowCumulativeNegativeRatio;
    pub const withRowCumNegativeRatio = table_methods_mod.withRowCumNegativeRatio;
    pub const withRowPrefixNegativeRatio = table_methods_mod.withRowPrefixNegativeRatio;
    pub const withRowFiniteCount = table_methods_mod.withRowFiniteCount;
    pub const withRowFiniteRatio = table_methods_mod.withRowFiniteRatio;
    pub const withRowNormalCount = table_methods_mod.withRowNormalCount;
    pub const withRowNormalRatio = table_methods_mod.withRowNormalRatio;
    pub const withRowSubnormalCount = table_methods_mod.withRowSubnormalCount;
    pub const withRowSubnormalRatio = table_methods_mod.withRowSubnormalRatio;
    pub const withRowNonFiniteCount = table_methods_mod.withRowNonFiniteCount;
    pub const withRowNonFiniteRatio = table_methods_mod.withRowNonFiniteRatio;
    pub const withColumnLiteral = table_methods_mod.withColumnLiteral;
    pub const withColumnLiteralScalar = table_methods_mod.withColumnLiteralScalar;
    pub const withColumnLiteralAt = table_methods_mod.withColumnLiteralAt;
    pub const withColumnLiteralBefore = table_methods_mod.withColumnLiteralBefore;
    pub const withColumnLiteralAfter = table_methods_mod.withColumnLiteralAfter;
    pub const withColumnLiteralScalarAt = table_methods_mod.withColumnLiteralScalarAt;
    pub const withColumnLiteralScalarBefore = table_methods_mod.withColumnLiteralScalarBefore;
    pub const withColumnLiteralScalarAfter = table_methods_mod.withColumnLiteralScalarAfter;
    pub const withRowIndex = table_methods_mod.withRowIndex;
    pub const renameColumn = table_methods_mod.renameColumn;
    pub const renameColumns = table_methods_mod.renameColumns;
    pub const addColumnNamePrefix = table_methods_mod.addColumnNamePrefix;
    pub const addColumnNameSuffix = table_methods_mod.addColumnNameSuffix;
    pub const stripColumnNamePrefix = table_methods_mod.stripColumnNamePrefix;
    pub const removeColumnNamePrefix = table_methods_mod.removeColumnNamePrefix;
    pub const stripColumnNameSuffix = table_methods_mod.stripColumnNameSuffix;
    pub const removeColumnNameSuffix = table_methods_mod.removeColumnNameSuffix;
    pub const replaceColumnNamePrefix = table_methods_mod.replaceColumnNamePrefix;
    pub const replaceColumnNameSuffix = table_methods_mod.replaceColumnNameSuffix;
    pub const moveColumn = table_methods_mod.moveColumn;
    pub const moveColumnBefore = table_methods_mod.moveColumnBefore;
    pub const moveColumnAfter = table_methods_mod.moveColumnAfter;
    pub const dropColumns = table_methods_mod.dropColumns;
    pub const dropColumn = table_methods_mod.dropColumn;
    pub const dropNulls = table_methods_mod.dropNulls;
    pub const dropNullsOn = table_methods_mod.dropNullsOn;
    pub const dropNullsColumn = table_methods_mod.dropNullsColumn;
    pub const dropAllNulls = table_methods_mod.dropAllNulls;
    pub const dropAllNullsOn = table_methods_mod.dropAllNullsOn;
    pub const filterAllNulls = table_methods_mod.filterAllNulls;
    pub const filterAllNullsOn = table_methods_mod.filterAllNullsOn;
    pub const filterNullsColumn = table_methods_mod.filterNullsColumn;
    pub const dropNaNs = table_methods_mod.dropNaNs;
    pub const dropNaNsOn = table_methods_mod.dropNaNsOn;
    pub const dropNaNsColumn = table_methods_mod.dropNaNsColumn;
    pub const filterNaNsColumn = table_methods_mod.filterNaNsColumn;
    pub const dropInfs = table_methods_mod.dropInfs;
    pub const dropInfsOn = table_methods_mod.dropInfsOn;
    pub const dropInfsColumn = table_methods_mod.dropInfsColumn;
    pub const filterInfsColumn = table_methods_mod.filterInfsColumn;
    pub const dropPositiveInfs = table_methods_mod.dropPositiveInfs;
    pub const dropPositiveInfsOn = table_methods_mod.dropPositiveInfsOn;
    pub const dropPositiveInfsColumn = table_methods_mod.dropPositiveInfsColumn;
    pub const filterPositiveInfsColumn = table_methods_mod.filterPositiveInfsColumn;
    pub const dropNegativeInfs = table_methods_mod.dropNegativeInfs;
    pub const dropNegativeInfsOn = table_methods_mod.dropNegativeInfsOn;
    pub const dropNegativeInfsColumn = table_methods_mod.dropNegativeInfsColumn;
    pub const filterNegativeInfsColumn = table_methods_mod.filterNegativeInfsColumn;
    pub const dropZeros = table_methods_mod.dropZeros;
    pub const dropZerosOn = table_methods_mod.dropZerosOn;
    pub const dropZerosColumn = table_methods_mod.dropZerosColumn;
    pub const filterZerosColumn = table_methods_mod.filterZerosColumn;
    pub const dropPositiveZeros = table_methods_mod.dropPositiveZeros;
    pub const dropPositiveZerosOn = table_methods_mod.dropPositiveZerosOn;
    pub const dropPositiveZerosColumn = table_methods_mod.dropPositiveZerosColumn;
    pub const filterPositiveZerosColumn = table_methods_mod.filterPositiveZerosColumn;
    pub const dropNegativeZeros = table_methods_mod.dropNegativeZeros;
    pub const dropNegativeZerosOn = table_methods_mod.dropNegativeZerosOn;
    pub const dropNegativeZerosColumn = table_methods_mod.dropNegativeZerosColumn;
    pub const filterNegativeZerosColumn = table_methods_mod.filterNegativeZerosColumn;
    pub const dropNonZeros = table_methods_mod.dropNonZeros;
    pub const dropNonZerosOn = table_methods_mod.dropNonZerosOn;
    pub const dropNonZerosColumn = table_methods_mod.dropNonZerosColumn;
    pub const filterNonZerosColumn = table_methods_mod.filterNonZerosColumn;
    pub const dropPositives = table_methods_mod.dropPositives;
    pub const dropPositivesOn = table_methods_mod.dropPositivesOn;
    pub const dropPositivesColumn = table_methods_mod.dropPositivesColumn;
    pub const filterPositivesColumn = table_methods_mod.filterPositivesColumn;
    pub const dropSignBits = table_methods_mod.dropSignBits;
    pub const dropSignBitsOn = table_methods_mod.dropSignBitsOn;
    pub const dropSignBitsColumn = table_methods_mod.dropSignBitsColumn;
    pub const filterSignBitsColumn = table_methods_mod.filterSignBitsColumn;
    pub const dropNegatives = table_methods_mod.dropNegatives;
    pub const dropNegativesOn = table_methods_mod.dropNegativesOn;
    pub const dropNegativesColumn = table_methods_mod.dropNegativesColumn;
    pub const filterNegativesColumn = table_methods_mod.filterNegativesColumn;
    pub const dropFinites = table_methods_mod.dropFinites;
    pub const dropFinitesOn = table_methods_mod.dropFinitesOn;
    pub const dropFinitesColumn = table_methods_mod.dropFinitesColumn;
    pub const filterFinitesColumn = table_methods_mod.filterFinitesColumn;
    pub const dropNormals = table_methods_mod.dropNormals;
    pub const dropNormalsOn = table_methods_mod.dropNormalsOn;
    pub const dropNormalsColumn = table_methods_mod.dropNormalsColumn;
    pub const filterNormalsColumn = table_methods_mod.filterNormalsColumn;
    pub const dropSubnormals = table_methods_mod.dropSubnormals;
    pub const dropSubnormalsOn = table_methods_mod.dropSubnormalsOn;
    pub const dropSubnormalsColumn = table_methods_mod.dropSubnormalsColumn;
    pub const filterSubnormalsColumn = table_methods_mod.filterSubnormalsColumn;
    pub const dropNonFinites = table_methods_mod.dropNonFinites;
    pub const dropNonFinitesOn = table_methods_mod.dropNonFinitesOn;
    pub const dropNonFinitesColumn = table_methods_mod.dropNonFinitesColumn;
    pub const filterNonFinitesColumn = table_methods_mod.filterNonFinitesColumn;
    pub const head = table_methods_mod.head;
    pub const limit = table_methods_mod.limit;
    pub const firstRow = table_methods_mod.firstRow;
    pub const tail = table_methods_mod.tail;
    pub const lastRow = table_methods_mod.lastRow;
    pub const sliceRows = table_methods_mod.sliceRows;
    pub const sliceRowsLen = table_methods_mod.sliceRowsLen;
    pub const offset = table_methods_mod.offset;
    pub const sliceRowsSigned = table_methods_mod.sliceRowsSigned;
    pub const sliceSigned = table_methods_mod.sliceSigned;
    pub const sliceRowsSignedStep = table_methods_mod.sliceRowsSignedStep;
    pub const sliceSignedStep = table_methods_mod.sliceSignedStep;
    pub const dropRows = table_methods_mod.dropRows;
    pub const dropRowsMode = table_methods_mod.dropRowsMode;
    pub const dropRowsSigned = table_methods_mod.dropRowsSigned;
    pub const dropRowsSignedMode = table_methods_mod.dropRowsSignedMode;
    pub const dropRowRange = table_methods_mod.dropRowRange;
    pub const dropFirstRows = table_methods_mod.dropFirstRows;
    pub const dropLastRows = table_methods_mod.dropLastRows;
    pub const sliceRowsStep = table_methods_mod.sliceRowsStep;
    pub const sliceStep = table_methods_mod.sliceStep;
    pub const take = table_methods_mod.take;
    pub const takeMode = table_methods_mod.takeMode;
    pub const takeSigned = table_methods_mod.takeSigned;
    pub const takeSignedMode = table_methods_mod.takeSignedMode;
    pub const takeOptional = table_methods_mod.takeOptional;
    pub const takeOptionalRows = table_methods_mod.takeOptionalRows;
    pub const takeByColumn = table_methods_mod.takeByColumn;
    pub const takeByColumnMode = table_methods_mod.takeByColumnMode;
    pub const takeRowsByColumn = table_methods_mod.takeRowsByColumn;
    pub const takeRowsByColumnMode = table_methods_mod.takeRowsByColumnMode;
    pub const dropRowsByColumn = table_methods_mod.dropRowsByColumn;
    pub const dropRowsByColumnMode = table_methods_mod.dropRowsByColumnMode;
    pub const repeatRows = table_methods_mod.repeatRows;
    pub const tileRows = table_methods_mod.tileRows;
    pub const repeatRowsByColumn = table_methods_mod.repeatRowsByColumn;
    pub const sampleRows = table_methods_mod.sampleRows;
    pub const shuffleRows = table_methods_mod.shuffleRows;
    pub const sampleRowsFraction = table_methods_mod.sampleRowsFraction;
    pub const sampleFrac = table_methods_mod.sampleFrac;
    pub const sampleRowsWithReplacement = table_methods_mod.sampleRowsWithReplacement;
    pub const sampleRowsFractionWithReplacement = table_methods_mod.sampleRowsFractionWithReplacement;
    pub const sampleFracWithReplacement = table_methods_mod.sampleFracWithReplacement;
    pub const strideRows = table_methods_mod.strideRows;
    pub const reverseRows = table_methods_mod.reverseRows;
    pub const rollRows = table_methods_mod.rollRows;
    pub const shiftRows = table_methods_mod.shiftRows;
    pub const reverse = table_methods_mod.reverse;
    pub const concatRows = table_methods_mod.concatRows;
    pub const appendRows = table_methods_mod.appendRows;
    pub const vstack = table_methods_mod.vstack;
    pub const concatColumns = table_methods_mod.concatColumns;
    pub const appendColumns = table_methods_mod.appendColumns;
    pub const hstack = table_methods_mod.hstack;
    pub const distinctRows = table_methods_mod.distinctRows;
    pub const distinctRowsLast = table_methods_mod.distinctRowsLast;
    pub const distinctRowsNone = table_methods_mod.distinctRowsNone;
    pub const distinctOn = table_methods_mod.distinctOn;
    pub const distinctOnLast = table_methods_mod.distinctOnLast;
    pub const distinctOnNone = table_methods_mod.distinctOnNone;
    pub const withRowIsDuplicated = table_methods_mod.withRowIsDuplicated;
    pub const withRowIsUnique = table_methods_mod.withRowIsUnique;
    pub const dropDuplicates = table_methods_mod.dropDuplicates;
    pub const dropDuplicatesOn = table_methods_mod.dropDuplicatesOn;
    pub const dropDuplicatesLast = table_methods_mod.dropDuplicatesLast;
    pub const dropDuplicatesOnLast = table_methods_mod.dropDuplicatesOnLast;
    pub const dropDuplicatesNone = table_methods_mod.dropDuplicatesNone;
    pub const dropDuplicatesOnNone = table_methods_mod.dropDuplicatesOnNone;
    pub const uniqueRows = table_methods_mod.uniqueRows;
    pub const argsortBy = table_methods_mod.argsortBy;
    pub const isSortedBy = table_methods_mod.isSortedBy;
    pub const isSortedByColumn = table_methods_mod.isSortedByColumn;
    pub const sortBy = table_methods_mod.sortBy;
    pub const sortByColumn = table_methods_mod.sortByColumn;
    pub const argsortByColumns = table_methods_mod.argsortByColumns;
    pub const isSortedByColumns = table_methods_mod.isSortedByColumns;
    pub const sortByColumns = table_methods_mod.sortByColumns;
    pub const topKByColumns = table_methods_mod.topKByColumns;
    pub const topKBy = table_methods_mod.topKBy;
    pub const bottomKByColumns = table_methods_mod.bottomKByColumns;
    pub const bottomKBy = table_methods_mod.bottomKBy;
    pub const rankProfileBy = table_methods_mod.rankProfileBy;
    pub const rollingProfile = profile_methods_mod.rollingProfile;
    pub const rollingMomentProfile = profile_methods_mod.rollingMomentProfile;
    pub const rollingRangeProfile = profile_methods_mod.rollingRangeProfile;
    pub const rollingNormalizeProfile = profile_methods_mod.rollingNormalizeProfile;
    pub const expandingNormalizeProfile = profile_methods_mod.expandingNormalizeProfile;
    pub const rollingQuantileProfile = profile_methods_mod.rollingQuantileProfile;
    pub const expandingQuantileProfile = profile_methods_mod.expandingQuantileProfile;
    pub const rollingBoolProfile = profile_methods_mod.rollingBoolProfile;
    pub const rollingDrawdownProfile = profile_methods_mod.rollingDrawdownProfile;
    pub const rollingRobustProfile = profile_methods_mod.rollingRobustProfile;
    pub const rollingRankProfile = profile_methods_mod.rollingRankProfile;
    pub const lagProfile = profile_methods_mod.lagProfile;
    pub const leadProfile = profile_methods_mod.leadProfile;
    pub const clipProfile = profile_methods_mod.clipProfile;
    pub const rollingClipProfile = profile_methods_mod.rollingClipProfile;
    pub const expandingClipProfile = profile_methods_mod.expandingClipProfile;
    pub const thresholdProfile = profile_methods_mod.thresholdProfile;
    pub const rollingThresholdProfile = profile_methods_mod.rollingThresholdProfile;
    pub const expandingThresholdProfile = profile_methods_mod.expandingThresholdProfile;
    pub const expandingProfile = profile_methods_mod.expandingProfile;
    pub const expandingBoolProfile = profile_methods_mod.expandingBoolProfile;
    pub const expandingRankProfile = profile_methods_mod.expandingRankProfile;
    pub const expandingRobustProfile = profile_methods_mod.expandingRobustProfile;
    pub const expandingMomentProfile = profile_methods_mod.expandingMomentProfile;
    pub const standardizeProfile = profile_methods_mod.standardizeProfile;
    pub const robustProfile = profile_methods_mod.robustProfile;
    pub const drawdownProfile = profile_methods_mod.drawdownProfile;
    pub const extremaProfile = profile_methods_mod.extremaProfile;
    pub const trendProfile = profile_methods_mod.trendProfile;
    pub const changePointProfile = profile_methods_mod.changePointProfile;
    pub const rollingChangePointProfile = profile_methods_mod.rollingChangePointProfile;
    pub const expandingChangePointProfile = profile_methods_mod.expandingChangePointProfile;
    pub const rollingTrendProfile = profile_methods_mod.rollingTrendProfile;
    pub const expandingTrendProfile = profile_methods_mod.expandingTrendProfile;
    pub const signProfile = profile_methods_mod.signProfile;
    pub const rollingSignProfile = profile_methods_mod.rollingSignProfile;
    pub const expandingSignProfile = profile_methods_mod.expandingSignProfile;
    pub const crossoverProfile = profile_methods_mod.crossoverProfile;
    pub const rollingCrossoverProfile = profile_methods_mod.rollingCrossoverProfile;
    pub const expandingCrossoverProfile = profile_methods_mod.expandingCrossoverProfile;
    pub const bucketProfile = profile_methods_mod.bucketProfile;
    pub const emaProfile = profile_methods_mod.emaProfile;
    pub const linearFitProfile = profile_methods_mod.linearFitProfile;
    pub const errorProfile = profile_methods_mod.errorProfile;
    pub const rollingErrorProfile = profile_methods_mod.rollingErrorProfile;
    pub const expandingErrorProfile = profile_methods_mod.expandingErrorProfile;
    pub const classificationProfile = profile_methods_mod.classificationProfile;
    pub const rollingClassificationProfile = profile_methods_mod.rollingClassificationProfile;
    pub const expandingClassificationProfile = profile_methods_mod.expandingClassificationProfile;
    pub const boolTransitionProfile = profile_methods_mod.boolTransitionProfile;
    pub const rollingBoolTransitionProfile = profile_methods_mod.rollingBoolTransitionProfile;
    pub const expandingBoolTransitionProfile = profile_methods_mod.expandingBoolTransitionProfile;
    pub const rollingCorrelationProfile = profile_methods_mod.rollingCorrelationProfile;
    pub const expandingCorrelationProfile = profile_methods_mod.expandingCorrelationProfile;
    pub const expandingLinearFitProfile = profile_methods_mod.expandingLinearFitProfile;
    pub const rollingLinearFitProfile = profile_methods_mod.rollingLinearFitProfile;
    pub const validityProfile = profile_methods_mod.validityProfile;
    pub const rollingValidityProfile = profile_methods_mod.rollingValidityProfile;
    pub const expandingValidityProfile = profile_methods_mod.expandingValidityProfile;
    pub const groupByCount = relation_methods_mod.groupByCount;
    pub const valueCounts = relation_methods_mod.valueCounts;
    pub const valueCountsAs = relation_methods_mod.valueCountsAs;
    pub const valueCountsSorted = relation_methods_mod.valueCountsSorted;
    pub const valueCountsSortedAs = relation_methods_mod.valueCountsSortedAs;
    pub const groupBySum = relation_methods_mod.groupBySum;
    pub const groupByMin = relation_methods_mod.groupByMin;
    pub const groupByMax = relation_methods_mod.groupByMax;
    pub const groupByMean = relation_methods_mod.groupByMean;
    pub const groupByStats = relation_methods_mod.groupByStats;
    pub const groupByStatsOn = relation_methods_mod.groupByStatsOn;
    pub const groupByProfile = relation_methods_mod.groupByProfile;
    pub const groupByProfileOn = relation_methods_mod.groupByProfileOn;
    pub const innerJoin = relation_methods_mod.innerJoin;
    pub const innerJoinOn = relation_methods_mod.innerJoinOn;
    pub const leftJoin = relation_methods_mod.leftJoin;
    pub const leftJoinOn = relation_methods_mod.leftJoinOn;
    pub const fullJoin = relation_methods_mod.fullJoin;
    pub const fullJoinOn = relation_methods_mod.fullJoinOn;
    pub const semiJoin = relation_methods_mod.semiJoin;
    pub const semiJoinOn = relation_methods_mod.semiJoinOn;
    pub const antiJoin = relation_methods_mod.antiJoin;
    pub const antiJoinOn = relation_methods_mod.antiJoinOn;
    pub const asofJoin = relation_methods_mod.asofJoin;
    pub const filter = table_methods_mod.filter;
    pub const to = table_methods_mod.to;
    pub const cpu = table_methods_mod.cpu;
    pub const cuda = table_methods_mod.cuda;
    pub const mps = table_methods_mod.mps;
    pub fn toDataFrame(self: DeviceDataFrame) DeviceDataError!DataFrame {
        return dataframe_host_mod.deviceDataFrameToDataFrame(self);
    }
};

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
