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
pub const DeviceLazyWeightedGroupByAggregation = lazy_op_mod.DeviceLazyWeightedGroupByAggregation;
pub const DeviceLazyPairGroupByAggregation = lazy_op_mod.DeviceLazyPairGroupByAggregation;
pub const DeviceLazyWeightedPairGroupByAggregation = lazy_op_mod.DeviceLazyWeightedPairGroupByAggregation;
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

    pub fn columnNamesUnique(self: DeviceDataFrame) bool {
        for (self.names, 0..) |name, index| {
            for (self.names[0..index]) |previous| {
                if (std.mem.eql(u8, name, previous)) return false;
            }
        }
        return true;
    }

    pub fn hasDuplicateColumnNames(self: DeviceDataFrame) bool {
        return !self.columnNamesUnique();
    }

    pub fn duplicateColumnNameCount(self: DeviceDataFrame) usize {
        var duplicates: usize = 0;
        for (self.names, 0..) |name, index| {
            for (self.names[0..index]) |previous| {
                if (std.mem.eql(u8, name, previous)) {
                    duplicates += 1;
                    break;
                }
            }
        }
        return duplicates;
    }

    pub fn columnDTypes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceDType {
        const out = try allocator.alloc(DeviceDType, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype();
        return out;
    }

    pub fn dtypes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]DeviceDType {
        return self.columnDTypes(allocator);
    }

    pub fn columnDTypeNames(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![][]const u8 {
        const out = try allocator.alloc([]const u8, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype().name();
        return out;
    }

    pub fn dtypeNames(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![][]const u8 {
        return self.columnDTypeNames(allocator);
    }

    pub fn columnDTypeByteSizes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype().byteSize();
        return out;
    }

    pub fn columnDTypeBitSizes(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]usize {
        const out = try allocator.alloc(usize, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.dtype().bitSize();
        return out;
    }

    pub fn columnDTypeClassMask(self: DeviceDataFrame, allocator: std.mem.Allocator, class: DeviceDTypeClass) std.mem.Allocator.Error![]bool {
        const out = try allocator.alloc(bool, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = class.matches(column_value.dtype());
        return out;
    }

    pub fn columnDTypeClassCount(self: DeviceDataFrame, class: DeviceDTypeClass) usize {
        var count: usize = 0;
        for (self.columns) |column_value| {
            if (class.matches(column_value.dtype())) count += 1;
        }
        return count;
    }

    pub fn numericColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.numeric);
    }

    pub fn realColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.real);
    }

    pub fn floatColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.float);
    }

    pub fn integerColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.integer);
    }

    pub fn signedIntegerColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.signed_integer);
    }

    pub fn unsignedIntegerColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.unsigned_integer);
    }

    pub fn boolColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.bool);
    }

    pub fn complexColumnCount(self: DeviceDataFrame) usize {
        return self.columnDTypeClassCount(.complex);
    }

    pub fn columnIsNumericMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .numeric);
    }

    pub fn columnIsRealMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .real);
    }

    pub fn columnIsFloatMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .float);
    }

    pub fn columnIsIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .integer);
    }

    pub fn columnIsSignedIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .signed_integer);
    }

    pub fn columnIsUnsignedIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .unsigned_integer);
    }

    pub fn columnIsBoolMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .bool);
    }

    pub fn columnIsComplexMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        return self.columnDTypeClassMask(allocator, .complex);
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

    pub fn nullCount(self: DeviceDataFrame) usize {
        var count: usize = 0;
        for (self.columns) |column_value| count += column_value.nullCount();
        return count;
    }

    pub fn validCount(self: DeviceDataFrame) usize {
        var count: usize = 0;
        for (self.columns) |column_value| count += column_value.validCount();
        return count;
    }

    pub fn cellCount(self: DeviceDataFrame) usize {
        return self.rows * self.columns.len;
    }

    pub fn nullRatio(self: DeviceDataFrame) f64 {
        return ratioFromCount(self.nullCount(), self.cellCount());
    }

    pub fn validRatio(self: DeviceDataFrame) f64 {
        return ratioFromCount(self.validCount(), self.cellCount());
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

    pub fn nullableColumnCount(self: DeviceDataFrame) usize {
        var count: usize = 0;
        for (self.columns) |column_value| {
            if (column_value.nullable()) count += 1;
        }
        return count;
    }

    pub fn nonNullableColumnCount(self: DeviceDataFrame) usize {
        return self.columns.len - self.nullableColumnCount();
    }

    pub fn columnHasNullsMask(self: DeviceDataFrame, allocator: std.mem.Allocator) std.mem.Allocator.Error![]bool {
        const out = try allocator.alloc(bool, self.columns.len);
        for (self.columns, out) |column_value, *slot| slot.* = column_value.hasNulls();
        return out;
    }

    pub fn columnsWithNullsCount(self: DeviceDataFrame) usize {
        var count: usize = 0;
        for (self.columns) |column_value| {
            if (column_value.hasNulls()) count += 1;
        }
        return count;
    }

    pub fn columnsWithoutNullsCount(self: DeviceDataFrame) usize {
        return self.columns.len - self.columnsWithNullsCount();
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
    pub const distinctRowCount = table_methods_mod.distinctRowCount;
    pub const distinctRowCountOn = table_methods_mod.distinctRowCountOn;
    pub const uniqueRowCount = table_methods_mod.uniqueRowCount;
    pub const uniqueRowCountOn = table_methods_mod.uniqueRowCountOn;
    pub const distinctRowRatio = table_methods_mod.distinctRowRatio;
    pub const distinctRowRatioOn = table_methods_mod.distinctRowRatioOn;
    pub const uniqueRowRatio = table_methods_mod.uniqueRowRatio;
    pub const uniqueRowRatioOn = table_methods_mod.uniqueRowRatioOn;
    pub const duplicateRowCount = table_methods_mod.duplicateRowCount;
    pub const duplicateRowCountOn = table_methods_mod.duplicateRowCountOn;
    pub const duplicateRowRatio = table_methods_mod.duplicateRowRatio;
    pub const duplicateRowRatioOn = table_methods_mod.duplicateRowRatioOn;
    pub const hasDuplicateRows = table_methods_mod.hasDuplicateRows;
    pub const hasDuplicateRowsOn = table_methods_mod.hasDuplicateRowsOn;
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
    pub const groupByCountOn = relation_methods_mod.groupByCountOn;
    pub const groupByHeadRows = relation_methods_mod.groupByHeadRows;
    pub const groupByHeadRowsOn = relation_methods_mod.groupByHeadRowsOn;
    pub const groupByTailRows = relation_methods_mod.groupByTailRows;
    pub const groupByTailRowsOn = relation_methods_mod.groupByTailRowsOn;
    pub const groupBySliceRows = relation_methods_mod.groupBySliceRows;
    pub const groupBySliceRowsOn = relation_methods_mod.groupBySliceRowsOn;
    pub const groupBySliceRowsStep = relation_methods_mod.groupBySliceRowsStep;
    pub const groupBySliceRowsStepOn = relation_methods_mod.groupBySliceRowsStepOn;
    pub const groupBySliceRowsSigned = relation_methods_mod.groupBySliceRowsSigned;
    pub const groupBySliceRowsSignedOn = relation_methods_mod.groupBySliceRowsSignedOn;
    pub const groupBySliceRowsSignedStep = relation_methods_mod.groupBySliceRowsSignedStep;
    pub const groupBySliceRowsSignedStepOn = relation_methods_mod.groupBySliceRowsSignedStepOn;
    pub const groupByTopRows = relation_methods_mod.groupByTopRows;
    pub const groupByTopRowsOn = relation_methods_mod.groupByTopRowsOn;
    pub const groupByBottomRows = relation_methods_mod.groupByBottomRows;
    pub const groupByBottomRowsOn = relation_methods_mod.groupByBottomRowsOn;
    pub const groupByTopRowsByColumns = relation_methods_mod.groupByTopRowsByColumns;
    pub const groupByTopRowsByColumnsOn = relation_methods_mod.groupByTopRowsByColumnsOn;
    pub const groupByBottomRowsByColumns = relation_methods_mod.groupByBottomRowsByColumns;
    pub const groupByBottomRowsByColumnsOn = relation_methods_mod.groupByBottomRowsByColumnsOn;
    pub const withGroupId = relation_methods_mod.withGroupId;
    pub const withGroupIdOn = relation_methods_mod.withGroupIdOn;
    pub const withGroupIndex = relation_methods_mod.withGroupIndex;
    pub const withGroupIndexOn = relation_methods_mod.withGroupIndexOn;
    pub const withGroupFirstRowIndex = relation_methods_mod.withGroupFirstRowIndex;
    pub const withGroupFirstRowIndexOn = relation_methods_mod.withGroupFirstRowIndexOn;
    pub const withGroupLastRowIndex = relation_methods_mod.withGroupLastRowIndex;
    pub const withGroupLastRowIndexOn = relation_methods_mod.withGroupLastRowIndexOn;
    pub const withGroupIsFirstRow = relation_methods_mod.withGroupIsFirstRow;
    pub const withGroupIsFirstRowOn = relation_methods_mod.withGroupIsFirstRowOn;
    pub const withGroupIsLastRow = relation_methods_mod.withGroupIsLastRow;
    pub const withGroupIsLastRowOn = relation_methods_mod.withGroupIsLastRowOn;
    pub const withGroupIsSingleton = relation_methods_mod.withGroupIsSingleton;
    pub const withGroupIsSingletonOn = relation_methods_mod.withGroupIsSingletonOn;
    pub const withGroupIsDuplicated = relation_methods_mod.withGroupIsDuplicated;
    pub const withGroupIsDuplicatedOn = relation_methods_mod.withGroupIsDuplicatedOn;
    pub const withGroupCumeDist = relation_methods_mod.withGroupCumeDist;
    pub const withGroupCumeDistOn = relation_methods_mod.withGroupCumeDistOn;
    pub const withGroupCumulativeDistribution = relation_methods_mod.withGroupCumulativeDistribution;
    pub const withGroupCumulativeDistributionOn = relation_methods_mod.withGroupCumulativeDistributionOn;
    pub const withGroupPercentRank = relation_methods_mod.withGroupPercentRank;
    pub const withGroupPercentRankOn = relation_methods_mod.withGroupPercentRankOn;
    pub const withGroupPercentileRank = relation_methods_mod.withGroupPercentileRank;
    pub const withGroupPercentileRankOn = relation_methods_mod.withGroupPercentileRankOn;
    pub const withGroupReverseCumeDist = relation_methods_mod.withGroupReverseCumeDist;
    pub const withGroupReverseCumeDistOn = relation_methods_mod.withGroupReverseCumeDistOn;
    pub const withGroupReverseCumulativeDistribution = relation_methods_mod.withGroupReverseCumulativeDistribution;
    pub const withGroupReverseCumulativeDistributionOn = relation_methods_mod.withGroupReverseCumulativeDistributionOn;
    pub const withGroupReversePercentRank = relation_methods_mod.withGroupReversePercentRank;
    pub const withGroupReversePercentRankOn = relation_methods_mod.withGroupReversePercentRankOn;
    pub const withGroupReversePercentileRank = relation_methods_mod.withGroupReversePercentileRank;
    pub const withGroupReversePercentileRankOn = relation_methods_mod.withGroupReversePercentileRankOn;
    pub const withGroupLag = relation_methods_mod.withGroupLag;
    pub const withGroupLagOn = relation_methods_mod.withGroupLagOn;
    pub const withGroupLead = relation_methods_mod.withGroupLead;
    pub const withGroupLeadOn = relation_methods_mod.withGroupLeadOn;
    pub const withGroupFirstRowValue = relation_methods_mod.withGroupFirstRowValue;
    pub const withGroupFirstRowValueOn = relation_methods_mod.withGroupFirstRowValueOn;
    pub const withGroupLastRowValue = relation_methods_mod.withGroupLastRowValue;
    pub const withGroupLastRowValueOn = relation_methods_mod.withGroupLastRowValueOn;
    pub const withGroupNthRowValue = relation_methods_mod.withGroupNthRowValue;
    pub const withGroupNthRowValueOn = relation_methods_mod.withGroupNthRowValueOn;
    pub const withGroupNthValue = relation_methods_mod.withGroupNthValue;
    pub const withGroupNthValueOn = relation_methods_mod.withGroupNthValueOn;
    pub const withGroupFirstValidValue = relation_methods_mod.withGroupFirstValidValue;
    pub const withGroupFirstValidValueOn = relation_methods_mod.withGroupFirstValidValueOn;
    pub const withGroupLastValidValue = relation_methods_mod.withGroupLastValidValue;
    pub const withGroupLastValidValueOn = relation_methods_mod.withGroupLastValidValueOn;
    pub const withGroupNthValidValue = relation_methods_mod.withGroupNthValidValue;
    pub const withGroupNthValidValueOn = relation_methods_mod.withGroupNthValidValueOn;
    pub const withGroupFillNullForward = relation_methods_mod.withGroupFillNullForward;
    pub const withGroupFillNullForwardOn = relation_methods_mod.withGroupFillNullForwardOn;
    pub const withGroupFillNullBackward = relation_methods_mod.withGroupFillNullBackward;
    pub const withGroupFillNullBackwardOn = relation_methods_mod.withGroupFillNullBackwardOn;
    pub const withGroupCumulativeValidCount = relation_methods_mod.withGroupCumulativeValidCount;
    pub const withGroupCumulativeValidCountOn = relation_methods_mod.withGroupCumulativeValidCountOn;
    pub const withGroupCumulativeNullCount = relation_methods_mod.withGroupCumulativeNullCount;
    pub const withGroupCumulativeNullCountOn = relation_methods_mod.withGroupCumulativeNullCountOn;
    pub const withGroupCumValidCount = relation_methods_mod.withGroupCumValidCount;
    pub const withGroupCumValidCountOn = relation_methods_mod.withGroupCumValidCountOn;
    pub const withGroupCumNullCount = relation_methods_mod.withGroupCumNullCount;
    pub const withGroupCumNullCountOn = relation_methods_mod.withGroupCumNullCountOn;
    pub const withGroupCumulativeValidRatio = relation_methods_mod.withGroupCumulativeValidRatio;
    pub const withGroupCumulativeValidRatioOn = relation_methods_mod.withGroupCumulativeValidRatioOn;
    pub const withGroupCumulativeNullRatio = relation_methods_mod.withGroupCumulativeNullRatio;
    pub const withGroupCumulativeNullRatioOn = relation_methods_mod.withGroupCumulativeNullRatioOn;
    pub const withGroupCumValidRatio = relation_methods_mod.withGroupCumValidRatio;
    pub const withGroupCumValidRatioOn = relation_methods_mod.withGroupCumValidRatioOn;
    pub const withGroupCumNullRatio = relation_methods_mod.withGroupCumNullRatio;
    pub const withGroupCumNullRatioOn = relation_methods_mod.withGroupCumNullRatioOn;
    pub const withGroupCumulativeFirstValidIndex = relation_methods_mod.withGroupCumulativeFirstValidIndex;
    pub const withGroupCumulativeFirstValidIndexOn = relation_methods_mod.withGroupCumulativeFirstValidIndexOn;
    pub const withGroupCumulativeLastValidIndex = relation_methods_mod.withGroupCumulativeLastValidIndex;
    pub const withGroupCumulativeLastValidIndexOn = relation_methods_mod.withGroupCumulativeLastValidIndexOn;
    pub const withGroupCumulativeFirstNullIndex = relation_methods_mod.withGroupCumulativeFirstNullIndex;
    pub const withGroupCumulativeFirstNullIndexOn = relation_methods_mod.withGroupCumulativeFirstNullIndexOn;
    pub const withGroupCumulativeLastNullIndex = relation_methods_mod.withGroupCumulativeLastNullIndex;
    pub const withGroupCumulativeLastNullIndexOn = relation_methods_mod.withGroupCumulativeLastNullIndexOn;
    pub const withGroupCumFirstValidIndex = relation_methods_mod.withGroupCumFirstValidIndex;
    pub const withGroupCumFirstValidIndexOn = relation_methods_mod.withGroupCumFirstValidIndexOn;
    pub const withGroupCumLastValidIndex = relation_methods_mod.withGroupCumLastValidIndex;
    pub const withGroupCumLastValidIndexOn = relation_methods_mod.withGroupCumLastValidIndexOn;
    pub const withGroupCumFirstNullIndex = relation_methods_mod.withGroupCumFirstNullIndex;
    pub const withGroupCumFirstNullIndexOn = relation_methods_mod.withGroupCumFirstNullIndexOn;
    pub const withGroupCumLastNullIndex = relation_methods_mod.withGroupCumLastNullIndex;
    pub const withGroupCumLastNullIndexOn = relation_methods_mod.withGroupCumLastNullIndexOn;
    pub const withGroupCumulativeNaNCount = relation_methods_mod.withGroupCumulativeNaNCount;
    pub const withGroupCumulativeNaNCountOn = relation_methods_mod.withGroupCumulativeNaNCountOn;
    pub const withGroupCumulativeNaNRatio = relation_methods_mod.withGroupCumulativeNaNRatio;
    pub const withGroupCumulativeNaNRatioOn = relation_methods_mod.withGroupCumulativeNaNRatioOn;
    pub const withGroupCumulativeNanCount = relation_methods_mod.withGroupCumulativeNanCount;
    pub const withGroupCumulativeNanCountOn = relation_methods_mod.withGroupCumulativeNanCountOn;
    pub const withGroupCumulativeNanRatio = relation_methods_mod.withGroupCumulativeNanRatio;
    pub const withGroupCumulativeNanRatioOn = relation_methods_mod.withGroupCumulativeNanRatioOn;
    pub const withGroupCumulativeInfCount = relation_methods_mod.withGroupCumulativeInfCount;
    pub const withGroupCumulativeInfCountOn = relation_methods_mod.withGroupCumulativeInfCountOn;
    pub const withGroupCumulativeInfRatio = relation_methods_mod.withGroupCumulativeInfRatio;
    pub const withGroupCumulativeInfRatioOn = relation_methods_mod.withGroupCumulativeInfRatioOn;
    pub const withGroupCumulativePositiveInfCount = relation_methods_mod.withGroupCumulativePositiveInfCount;
    pub const withGroupCumulativePositiveInfCountOn = relation_methods_mod.withGroupCumulativePositiveInfCountOn;
    pub const withGroupCumulativePositiveInfRatio = relation_methods_mod.withGroupCumulativePositiveInfRatio;
    pub const withGroupCumulativePositiveInfRatioOn = relation_methods_mod.withGroupCumulativePositiveInfRatioOn;
    pub const withGroupCumulativeNegativeInfCount = relation_methods_mod.withGroupCumulativeNegativeInfCount;
    pub const withGroupCumulativeNegativeInfCountOn = relation_methods_mod.withGroupCumulativeNegativeInfCountOn;
    pub const withGroupCumulativeNegativeInfRatio = relation_methods_mod.withGroupCumulativeNegativeInfRatio;
    pub const withGroupCumulativeNegativeInfRatioOn = relation_methods_mod.withGroupCumulativeNegativeInfRatioOn;
    pub const withGroupCumulativeFiniteCount = relation_methods_mod.withGroupCumulativeFiniteCount;
    pub const withGroupCumulativeFiniteCountOn = relation_methods_mod.withGroupCumulativeFiniteCountOn;
    pub const withGroupCumulativeFiniteRatio = relation_methods_mod.withGroupCumulativeFiniteRatio;
    pub const withGroupCumulativeFiniteRatioOn = relation_methods_mod.withGroupCumulativeFiniteRatioOn;
    pub const withGroupCumulativeNormalCount = relation_methods_mod.withGroupCumulativeNormalCount;
    pub const withGroupCumulativeNormalCountOn = relation_methods_mod.withGroupCumulativeNormalCountOn;
    pub const withGroupCumulativeNormalRatio = relation_methods_mod.withGroupCumulativeNormalRatio;
    pub const withGroupCumulativeNormalRatioOn = relation_methods_mod.withGroupCumulativeNormalRatioOn;
    pub const withGroupCumulativeSubnormalCount = relation_methods_mod.withGroupCumulativeSubnormalCount;
    pub const withGroupCumulativeSubnormalCountOn = relation_methods_mod.withGroupCumulativeSubnormalCountOn;
    pub const withGroupCumulativeSubnormalRatio = relation_methods_mod.withGroupCumulativeSubnormalRatio;
    pub const withGroupCumulativeSubnormalRatioOn = relation_methods_mod.withGroupCumulativeSubnormalRatioOn;
    pub const withGroupCumulativeNonFiniteCount = relation_methods_mod.withGroupCumulativeNonFiniteCount;
    pub const withGroupCumulativeNonFiniteCountOn = relation_methods_mod.withGroupCumulativeNonFiniteCountOn;
    pub const withGroupCumulativeNonFiniteRatio = relation_methods_mod.withGroupCumulativeNonFiniteRatio;
    pub const withGroupCumulativeNonFiniteRatioOn = relation_methods_mod.withGroupCumulativeNonFiniteRatioOn;
    pub const withGroupCumulativeZeroCount = relation_methods_mod.withGroupCumulativeZeroCount;
    pub const withGroupCumulativeZeroCountOn = relation_methods_mod.withGroupCumulativeZeroCountOn;
    pub const withGroupCumulativeZeroRatio = relation_methods_mod.withGroupCumulativeZeroRatio;
    pub const withGroupCumulativeZeroRatioOn = relation_methods_mod.withGroupCumulativeZeroRatioOn;
    pub const withGroupCumulativePositiveZeroCount = relation_methods_mod.withGroupCumulativePositiveZeroCount;
    pub const withGroupCumulativePositiveZeroCountOn = relation_methods_mod.withGroupCumulativePositiveZeroCountOn;
    pub const withGroupCumulativePositiveZeroRatio = relation_methods_mod.withGroupCumulativePositiveZeroRatio;
    pub const withGroupCumulativePositiveZeroRatioOn = relation_methods_mod.withGroupCumulativePositiveZeroRatioOn;
    pub const withGroupCumulativeNegativeZeroCount = relation_methods_mod.withGroupCumulativeNegativeZeroCount;
    pub const withGroupCumulativeNegativeZeroCountOn = relation_methods_mod.withGroupCumulativeNegativeZeroCountOn;
    pub const withGroupCumulativeNegativeZeroRatio = relation_methods_mod.withGroupCumulativeNegativeZeroRatio;
    pub const withGroupCumulativeNegativeZeroRatioOn = relation_methods_mod.withGroupCumulativeNegativeZeroRatioOn;
    pub const withGroupCumulativeNonZeroCount = relation_methods_mod.withGroupCumulativeNonZeroCount;
    pub const withGroupCumulativeNonZeroCountOn = relation_methods_mod.withGroupCumulativeNonZeroCountOn;
    pub const withGroupCumulativeNonZeroRatio = relation_methods_mod.withGroupCumulativeNonZeroRatio;
    pub const withGroupCumulativeNonZeroRatioOn = relation_methods_mod.withGroupCumulativeNonZeroRatioOn;
    pub const withGroupCumulativePositiveCount = relation_methods_mod.withGroupCumulativePositiveCount;
    pub const withGroupCumulativePositiveCountOn = relation_methods_mod.withGroupCumulativePositiveCountOn;
    pub const withGroupCumulativePositiveRatio = relation_methods_mod.withGroupCumulativePositiveRatio;
    pub const withGroupCumulativePositiveRatioOn = relation_methods_mod.withGroupCumulativePositiveRatioOn;
    pub const withGroupCumulativeSignBitCount = relation_methods_mod.withGroupCumulativeSignBitCount;
    pub const withGroupCumulativeSignBitCountOn = relation_methods_mod.withGroupCumulativeSignBitCountOn;
    pub const withGroupCumulativeSignBitRatio = relation_methods_mod.withGroupCumulativeSignBitRatio;
    pub const withGroupCumulativeSignBitRatioOn = relation_methods_mod.withGroupCumulativeSignBitRatioOn;
    pub const withGroupCumulativeNegativeCount = relation_methods_mod.withGroupCumulativeNegativeCount;
    pub const withGroupCumulativeNegativeCountOn = relation_methods_mod.withGroupCumulativeNegativeCountOn;
    pub const withGroupCumulativeNegativeRatio = relation_methods_mod.withGroupCumulativeNegativeRatio;
    pub const withGroupCumulativeNegativeRatioOn = relation_methods_mod.withGroupCumulativeNegativeRatioOn;
    pub const withGroupCumNaNCount = relation_methods_mod.withGroupCumNaNCount;
    pub const withGroupCumNaNCountOn = relation_methods_mod.withGroupCumNaNCountOn;
    pub const withGroupCumNaNRatio = relation_methods_mod.withGroupCumNaNRatio;
    pub const withGroupCumNaNRatioOn = relation_methods_mod.withGroupCumNaNRatioOn;
    pub const withGroupCumNanCount = relation_methods_mod.withGroupCumNanCount;
    pub const withGroupCumNanCountOn = relation_methods_mod.withGroupCumNanCountOn;
    pub const withGroupCumNanRatio = relation_methods_mod.withGroupCumNanRatio;
    pub const withGroupCumNanRatioOn = relation_methods_mod.withGroupCumNanRatioOn;
    pub const withGroupCumInfCount = relation_methods_mod.withGroupCumInfCount;
    pub const withGroupCumInfCountOn = relation_methods_mod.withGroupCumInfCountOn;
    pub const withGroupCumInfRatio = relation_methods_mod.withGroupCumInfRatio;
    pub const withGroupCumInfRatioOn = relation_methods_mod.withGroupCumInfRatioOn;
    pub const withGroupCumPositiveInfCount = relation_methods_mod.withGroupCumPositiveInfCount;
    pub const withGroupCumPositiveInfCountOn = relation_methods_mod.withGroupCumPositiveInfCountOn;
    pub const withGroupCumPositiveInfRatio = relation_methods_mod.withGroupCumPositiveInfRatio;
    pub const withGroupCumPositiveInfRatioOn = relation_methods_mod.withGroupCumPositiveInfRatioOn;
    pub const withGroupCumNegativeInfCount = relation_methods_mod.withGroupCumNegativeInfCount;
    pub const withGroupCumNegativeInfCountOn = relation_methods_mod.withGroupCumNegativeInfCountOn;
    pub const withGroupCumNegativeInfRatio = relation_methods_mod.withGroupCumNegativeInfRatio;
    pub const withGroupCumNegativeInfRatioOn = relation_methods_mod.withGroupCumNegativeInfRatioOn;
    pub const withGroupCumFiniteCount = relation_methods_mod.withGroupCumFiniteCount;
    pub const withGroupCumFiniteCountOn = relation_methods_mod.withGroupCumFiniteCountOn;
    pub const withGroupCumFiniteRatio = relation_methods_mod.withGroupCumFiniteRatio;
    pub const withGroupCumFiniteRatioOn = relation_methods_mod.withGroupCumFiniteRatioOn;
    pub const withGroupCumNormalCount = relation_methods_mod.withGroupCumNormalCount;
    pub const withGroupCumNormalCountOn = relation_methods_mod.withGroupCumNormalCountOn;
    pub const withGroupCumNormalRatio = relation_methods_mod.withGroupCumNormalRatio;
    pub const withGroupCumNormalRatioOn = relation_methods_mod.withGroupCumNormalRatioOn;
    pub const withGroupCumSubnormalCount = relation_methods_mod.withGroupCumSubnormalCount;
    pub const withGroupCumSubnormalCountOn = relation_methods_mod.withGroupCumSubnormalCountOn;
    pub const withGroupCumSubnormalRatio = relation_methods_mod.withGroupCumSubnormalRatio;
    pub const withGroupCumSubnormalRatioOn = relation_methods_mod.withGroupCumSubnormalRatioOn;
    pub const withGroupCumNonFiniteCount = relation_methods_mod.withGroupCumNonFiniteCount;
    pub const withGroupCumNonFiniteCountOn = relation_methods_mod.withGroupCumNonFiniteCountOn;
    pub const withGroupCumNonFiniteRatio = relation_methods_mod.withGroupCumNonFiniteRatio;
    pub const withGroupCumNonFiniteRatioOn = relation_methods_mod.withGroupCumNonFiniteRatioOn;
    pub const withGroupCumZeroCount = relation_methods_mod.withGroupCumZeroCount;
    pub const withGroupCumZeroCountOn = relation_methods_mod.withGroupCumZeroCountOn;
    pub const withGroupCumZeroRatio = relation_methods_mod.withGroupCumZeroRatio;
    pub const withGroupCumZeroRatioOn = relation_methods_mod.withGroupCumZeroRatioOn;
    pub const withGroupCumPositiveZeroCount = relation_methods_mod.withGroupCumPositiveZeroCount;
    pub const withGroupCumPositiveZeroCountOn = relation_methods_mod.withGroupCumPositiveZeroCountOn;
    pub const withGroupCumPositiveZeroRatio = relation_methods_mod.withGroupCumPositiveZeroRatio;
    pub const withGroupCumPositiveZeroRatioOn = relation_methods_mod.withGroupCumPositiveZeroRatioOn;
    pub const withGroupCumNegativeZeroCount = relation_methods_mod.withGroupCumNegativeZeroCount;
    pub const withGroupCumNegativeZeroCountOn = relation_methods_mod.withGroupCumNegativeZeroCountOn;
    pub const withGroupCumNegativeZeroRatio = relation_methods_mod.withGroupCumNegativeZeroRatio;
    pub const withGroupCumNegativeZeroRatioOn = relation_methods_mod.withGroupCumNegativeZeroRatioOn;
    pub const withGroupCumNonZeroCount = relation_methods_mod.withGroupCumNonZeroCount;
    pub const withGroupCumNonZeroCountOn = relation_methods_mod.withGroupCumNonZeroCountOn;
    pub const withGroupCumNonZeroRatio = relation_methods_mod.withGroupCumNonZeroRatio;
    pub const withGroupCumNonZeroRatioOn = relation_methods_mod.withGroupCumNonZeroRatioOn;
    pub const withGroupCumPositiveCount = relation_methods_mod.withGroupCumPositiveCount;
    pub const withGroupCumPositiveCountOn = relation_methods_mod.withGroupCumPositiveCountOn;
    pub const withGroupCumPositiveRatio = relation_methods_mod.withGroupCumPositiveRatio;
    pub const withGroupCumPositiveRatioOn = relation_methods_mod.withGroupCumPositiveRatioOn;
    pub const withGroupCumSignBitCount = relation_methods_mod.withGroupCumSignBitCount;
    pub const withGroupCumSignBitCountOn = relation_methods_mod.withGroupCumSignBitCountOn;
    pub const withGroupCumSignBitRatio = relation_methods_mod.withGroupCumSignBitRatio;
    pub const withGroupCumSignBitRatioOn = relation_methods_mod.withGroupCumSignBitRatioOn;
    pub const withGroupCumNegativeCount = relation_methods_mod.withGroupCumNegativeCount;
    pub const withGroupCumNegativeCountOn = relation_methods_mod.withGroupCumNegativeCountOn;
    pub const withGroupCumNegativeRatio = relation_methods_mod.withGroupCumNegativeRatio;
    pub const withGroupCumNegativeRatioOn = relation_methods_mod.withGroupCumNegativeRatioOn;
    pub const withGroupCumulativeFirstNaNIndex = relation_methods_mod.withGroupCumulativeFirstNaNIndex;
    pub const withGroupCumulativeFirstNaNIndexOn = relation_methods_mod.withGroupCumulativeFirstNaNIndexOn;
    pub const withGroupCumulativeLastNaNIndex = relation_methods_mod.withGroupCumulativeLastNaNIndex;
    pub const withGroupCumulativeLastNaNIndexOn = relation_methods_mod.withGroupCumulativeLastNaNIndexOn;
    pub const withGroupCumulativeFirstNanIndex = relation_methods_mod.withGroupCumulativeFirstNanIndex;
    pub const withGroupCumulativeFirstNanIndexOn = relation_methods_mod.withGroupCumulativeFirstNanIndexOn;
    pub const withGroupCumulativeLastNanIndex = relation_methods_mod.withGroupCumulativeLastNanIndex;
    pub const withGroupCumulativeLastNanIndexOn = relation_methods_mod.withGroupCumulativeLastNanIndexOn;
    pub const withGroupCumulativeFirstInfIndex = relation_methods_mod.withGroupCumulativeFirstInfIndex;
    pub const withGroupCumulativeFirstInfIndexOn = relation_methods_mod.withGroupCumulativeFirstInfIndexOn;
    pub const withGroupCumulativeLastInfIndex = relation_methods_mod.withGroupCumulativeLastInfIndex;
    pub const withGroupCumulativeLastInfIndexOn = relation_methods_mod.withGroupCumulativeLastInfIndexOn;
    pub const withGroupCumulativeFirstPositiveInfIndex = relation_methods_mod.withGroupCumulativeFirstPositiveInfIndex;
    pub const withGroupCumulativeFirstPositiveInfIndexOn = relation_methods_mod.withGroupCumulativeFirstPositiveInfIndexOn;
    pub const withGroupCumulativeLastPositiveInfIndex = relation_methods_mod.withGroupCumulativeLastPositiveInfIndex;
    pub const withGroupCumulativeLastPositiveInfIndexOn = relation_methods_mod.withGroupCumulativeLastPositiveInfIndexOn;
    pub const withGroupCumulativeFirstNegativeInfIndex = relation_methods_mod.withGroupCumulativeFirstNegativeInfIndex;
    pub const withGroupCumulativeFirstNegativeInfIndexOn = relation_methods_mod.withGroupCumulativeFirstNegativeInfIndexOn;
    pub const withGroupCumulativeLastNegativeInfIndex = relation_methods_mod.withGroupCumulativeLastNegativeInfIndex;
    pub const withGroupCumulativeLastNegativeInfIndexOn = relation_methods_mod.withGroupCumulativeLastNegativeInfIndexOn;
    pub const withGroupCumulativeFirstFiniteIndex = relation_methods_mod.withGroupCumulativeFirstFiniteIndex;
    pub const withGroupCumulativeFirstFiniteIndexOn = relation_methods_mod.withGroupCumulativeFirstFiniteIndexOn;
    pub const withGroupCumulativeLastFiniteIndex = relation_methods_mod.withGroupCumulativeLastFiniteIndex;
    pub const withGroupCumulativeLastFiniteIndexOn = relation_methods_mod.withGroupCumulativeLastFiniteIndexOn;
    pub const withGroupCumulativeFirstNormalIndex = relation_methods_mod.withGroupCumulativeFirstNormalIndex;
    pub const withGroupCumulativeFirstNormalIndexOn = relation_methods_mod.withGroupCumulativeFirstNormalIndexOn;
    pub const withGroupCumulativeLastNormalIndex = relation_methods_mod.withGroupCumulativeLastNormalIndex;
    pub const withGroupCumulativeLastNormalIndexOn = relation_methods_mod.withGroupCumulativeLastNormalIndexOn;
    pub const withGroupCumulativeFirstSubnormalIndex = relation_methods_mod.withGroupCumulativeFirstSubnormalIndex;
    pub const withGroupCumulativeFirstSubnormalIndexOn = relation_methods_mod.withGroupCumulativeFirstSubnormalIndexOn;
    pub const withGroupCumulativeLastSubnormalIndex = relation_methods_mod.withGroupCumulativeLastSubnormalIndex;
    pub const withGroupCumulativeLastSubnormalIndexOn = relation_methods_mod.withGroupCumulativeLastSubnormalIndexOn;
    pub const withGroupCumulativeFirstNonFiniteIndex = relation_methods_mod.withGroupCumulativeFirstNonFiniteIndex;
    pub const withGroupCumulativeFirstNonFiniteIndexOn = relation_methods_mod.withGroupCumulativeFirstNonFiniteIndexOn;
    pub const withGroupCumulativeLastNonFiniteIndex = relation_methods_mod.withGroupCumulativeLastNonFiniteIndex;
    pub const withGroupCumulativeLastNonFiniteIndexOn = relation_methods_mod.withGroupCumulativeLastNonFiniteIndexOn;
    pub const withGroupCumulativeFirstZeroIndex = relation_methods_mod.withGroupCumulativeFirstZeroIndex;
    pub const withGroupCumulativeFirstZeroIndexOn = relation_methods_mod.withGroupCumulativeFirstZeroIndexOn;
    pub const withGroupCumulativeLastZeroIndex = relation_methods_mod.withGroupCumulativeLastZeroIndex;
    pub const withGroupCumulativeLastZeroIndexOn = relation_methods_mod.withGroupCumulativeLastZeroIndexOn;
    pub const withGroupCumulativeFirstPositiveZeroIndex = relation_methods_mod.withGroupCumulativeFirstPositiveZeroIndex;
    pub const withGroupCumulativeFirstPositiveZeroIndexOn = relation_methods_mod.withGroupCumulativeFirstPositiveZeroIndexOn;
    pub const withGroupCumulativeLastPositiveZeroIndex = relation_methods_mod.withGroupCumulativeLastPositiveZeroIndex;
    pub const withGroupCumulativeLastPositiveZeroIndexOn = relation_methods_mod.withGroupCumulativeLastPositiveZeroIndexOn;
    pub const withGroupCumulativeFirstNegativeZeroIndex = relation_methods_mod.withGroupCumulativeFirstNegativeZeroIndex;
    pub const withGroupCumulativeFirstNegativeZeroIndexOn = relation_methods_mod.withGroupCumulativeFirstNegativeZeroIndexOn;
    pub const withGroupCumulativeLastNegativeZeroIndex = relation_methods_mod.withGroupCumulativeLastNegativeZeroIndex;
    pub const withGroupCumulativeLastNegativeZeroIndexOn = relation_methods_mod.withGroupCumulativeLastNegativeZeroIndexOn;
    pub const withGroupCumulativeFirstNonZeroIndex = relation_methods_mod.withGroupCumulativeFirstNonZeroIndex;
    pub const withGroupCumulativeFirstNonZeroIndexOn = relation_methods_mod.withGroupCumulativeFirstNonZeroIndexOn;
    pub const withGroupCumulativeLastNonZeroIndex = relation_methods_mod.withGroupCumulativeLastNonZeroIndex;
    pub const withGroupCumulativeLastNonZeroIndexOn = relation_methods_mod.withGroupCumulativeLastNonZeroIndexOn;
    pub const withGroupCumulativeFirstPositiveIndex = relation_methods_mod.withGroupCumulativeFirstPositiveIndex;
    pub const withGroupCumulativeFirstPositiveIndexOn = relation_methods_mod.withGroupCumulativeFirstPositiveIndexOn;
    pub const withGroupCumulativeLastPositiveIndex = relation_methods_mod.withGroupCumulativeLastPositiveIndex;
    pub const withGroupCumulativeLastPositiveIndexOn = relation_methods_mod.withGroupCumulativeLastPositiveIndexOn;
    pub const withGroupCumulativeFirstSignBitIndex = relation_methods_mod.withGroupCumulativeFirstSignBitIndex;
    pub const withGroupCumulativeFirstSignBitIndexOn = relation_methods_mod.withGroupCumulativeFirstSignBitIndexOn;
    pub const withGroupCumulativeLastSignBitIndex = relation_methods_mod.withGroupCumulativeLastSignBitIndex;
    pub const withGroupCumulativeLastSignBitIndexOn = relation_methods_mod.withGroupCumulativeLastSignBitIndexOn;
    pub const withGroupCumulativeFirstNegativeIndex = relation_methods_mod.withGroupCumulativeFirstNegativeIndex;
    pub const withGroupCumulativeFirstNegativeIndexOn = relation_methods_mod.withGroupCumulativeFirstNegativeIndexOn;
    pub const withGroupCumulativeLastNegativeIndex = relation_methods_mod.withGroupCumulativeLastNegativeIndex;
    pub const withGroupCumulativeLastNegativeIndexOn = relation_methods_mod.withGroupCumulativeLastNegativeIndexOn;
    pub const withGroupCumFirstNaNIndex = relation_methods_mod.withGroupCumFirstNaNIndex;
    pub const withGroupCumFirstNaNIndexOn = relation_methods_mod.withGroupCumFirstNaNIndexOn;
    pub const withGroupCumLastNaNIndex = relation_methods_mod.withGroupCumLastNaNIndex;
    pub const withGroupCumLastNaNIndexOn = relation_methods_mod.withGroupCumLastNaNIndexOn;
    pub const withGroupCumFirstNanIndex = relation_methods_mod.withGroupCumFirstNanIndex;
    pub const withGroupCumFirstNanIndexOn = relation_methods_mod.withGroupCumFirstNanIndexOn;
    pub const withGroupCumLastNanIndex = relation_methods_mod.withGroupCumLastNanIndex;
    pub const withGroupCumLastNanIndexOn = relation_methods_mod.withGroupCumLastNanIndexOn;
    pub const withGroupCumFirstInfIndex = relation_methods_mod.withGroupCumFirstInfIndex;
    pub const withGroupCumFirstInfIndexOn = relation_methods_mod.withGroupCumFirstInfIndexOn;
    pub const withGroupCumLastInfIndex = relation_methods_mod.withGroupCumLastInfIndex;
    pub const withGroupCumLastInfIndexOn = relation_methods_mod.withGroupCumLastInfIndexOn;
    pub const withGroupCumFirstPositiveInfIndex = relation_methods_mod.withGroupCumFirstPositiveInfIndex;
    pub const withGroupCumFirstPositiveInfIndexOn = relation_methods_mod.withGroupCumFirstPositiveInfIndexOn;
    pub const withGroupCumLastPositiveInfIndex = relation_methods_mod.withGroupCumLastPositiveInfIndex;
    pub const withGroupCumLastPositiveInfIndexOn = relation_methods_mod.withGroupCumLastPositiveInfIndexOn;
    pub const withGroupCumFirstNegativeInfIndex = relation_methods_mod.withGroupCumFirstNegativeInfIndex;
    pub const withGroupCumFirstNegativeInfIndexOn = relation_methods_mod.withGroupCumFirstNegativeInfIndexOn;
    pub const withGroupCumLastNegativeInfIndex = relation_methods_mod.withGroupCumLastNegativeInfIndex;
    pub const withGroupCumLastNegativeInfIndexOn = relation_methods_mod.withGroupCumLastNegativeInfIndexOn;
    pub const withGroupCumFirstFiniteIndex = relation_methods_mod.withGroupCumFirstFiniteIndex;
    pub const withGroupCumFirstFiniteIndexOn = relation_methods_mod.withGroupCumFirstFiniteIndexOn;
    pub const withGroupCumLastFiniteIndex = relation_methods_mod.withGroupCumLastFiniteIndex;
    pub const withGroupCumLastFiniteIndexOn = relation_methods_mod.withGroupCumLastFiniteIndexOn;
    pub const withGroupCumFirstNormalIndex = relation_methods_mod.withGroupCumFirstNormalIndex;
    pub const withGroupCumFirstNormalIndexOn = relation_methods_mod.withGroupCumFirstNormalIndexOn;
    pub const withGroupCumLastNormalIndex = relation_methods_mod.withGroupCumLastNormalIndex;
    pub const withGroupCumLastNormalIndexOn = relation_methods_mod.withGroupCumLastNormalIndexOn;
    pub const withGroupCumFirstSubnormalIndex = relation_methods_mod.withGroupCumFirstSubnormalIndex;
    pub const withGroupCumFirstSubnormalIndexOn = relation_methods_mod.withGroupCumFirstSubnormalIndexOn;
    pub const withGroupCumLastSubnormalIndex = relation_methods_mod.withGroupCumLastSubnormalIndex;
    pub const withGroupCumLastSubnormalIndexOn = relation_methods_mod.withGroupCumLastSubnormalIndexOn;
    pub const withGroupCumFirstNonFiniteIndex = relation_methods_mod.withGroupCumFirstNonFiniteIndex;
    pub const withGroupCumFirstNonFiniteIndexOn = relation_methods_mod.withGroupCumFirstNonFiniteIndexOn;
    pub const withGroupCumLastNonFiniteIndex = relation_methods_mod.withGroupCumLastNonFiniteIndex;
    pub const withGroupCumLastNonFiniteIndexOn = relation_methods_mod.withGroupCumLastNonFiniteIndexOn;
    pub const withGroupCumFirstZeroIndex = relation_methods_mod.withGroupCumFirstZeroIndex;
    pub const withGroupCumFirstZeroIndexOn = relation_methods_mod.withGroupCumFirstZeroIndexOn;
    pub const withGroupCumLastZeroIndex = relation_methods_mod.withGroupCumLastZeroIndex;
    pub const withGroupCumLastZeroIndexOn = relation_methods_mod.withGroupCumLastZeroIndexOn;
    pub const withGroupCumFirstPositiveZeroIndex = relation_methods_mod.withGroupCumFirstPositiveZeroIndex;
    pub const withGroupCumFirstPositiveZeroIndexOn = relation_methods_mod.withGroupCumFirstPositiveZeroIndexOn;
    pub const withGroupCumLastPositiveZeroIndex = relation_methods_mod.withGroupCumLastPositiveZeroIndex;
    pub const withGroupCumLastPositiveZeroIndexOn = relation_methods_mod.withGroupCumLastPositiveZeroIndexOn;
    pub const withGroupCumFirstNegativeZeroIndex = relation_methods_mod.withGroupCumFirstNegativeZeroIndex;
    pub const withGroupCumFirstNegativeZeroIndexOn = relation_methods_mod.withGroupCumFirstNegativeZeroIndexOn;
    pub const withGroupCumLastNegativeZeroIndex = relation_methods_mod.withGroupCumLastNegativeZeroIndex;
    pub const withGroupCumLastNegativeZeroIndexOn = relation_methods_mod.withGroupCumLastNegativeZeroIndexOn;
    pub const withGroupCumFirstNonZeroIndex = relation_methods_mod.withGroupCumFirstNonZeroIndex;
    pub const withGroupCumFirstNonZeroIndexOn = relation_methods_mod.withGroupCumFirstNonZeroIndexOn;
    pub const withGroupCumLastNonZeroIndex = relation_methods_mod.withGroupCumLastNonZeroIndex;
    pub const withGroupCumLastNonZeroIndexOn = relation_methods_mod.withGroupCumLastNonZeroIndexOn;
    pub const withGroupCumFirstPositiveIndex = relation_methods_mod.withGroupCumFirstPositiveIndex;
    pub const withGroupCumFirstPositiveIndexOn = relation_methods_mod.withGroupCumFirstPositiveIndexOn;
    pub const withGroupCumLastPositiveIndex = relation_methods_mod.withGroupCumLastPositiveIndex;
    pub const withGroupCumLastPositiveIndexOn = relation_methods_mod.withGroupCumLastPositiveIndexOn;
    pub const withGroupCumFirstSignBitIndex = relation_methods_mod.withGroupCumFirstSignBitIndex;
    pub const withGroupCumFirstSignBitIndexOn = relation_methods_mod.withGroupCumFirstSignBitIndexOn;
    pub const withGroupCumLastSignBitIndex = relation_methods_mod.withGroupCumLastSignBitIndex;
    pub const withGroupCumLastSignBitIndexOn = relation_methods_mod.withGroupCumLastSignBitIndexOn;
    pub const withGroupCumFirstNegativeIndex = relation_methods_mod.withGroupCumFirstNegativeIndex;
    pub const withGroupCumFirstNegativeIndexOn = relation_methods_mod.withGroupCumFirstNegativeIndexOn;
    pub const withGroupCumLastNegativeIndex = relation_methods_mod.withGroupCumLastNegativeIndex;
    pub const withGroupCumLastNegativeIndexOn = relation_methods_mod.withGroupCumLastNegativeIndexOn;
    pub const withGroupCumulativeDistinctCount = relation_methods_mod.withGroupCumulativeDistinctCount;
    pub const withGroupCumulativeDistinctCountOn = relation_methods_mod.withGroupCumulativeDistinctCountOn;
    pub const withGroupCumulativeCountDistinct = relation_methods_mod.withGroupCumulativeCountDistinct;
    pub const withGroupCumulativeCountDistinctOn = relation_methods_mod.withGroupCumulativeCountDistinctOn;
    pub const withGroupCumulativeNUnique = relation_methods_mod.withGroupCumulativeNUnique;
    pub const withGroupCumulativeNUniqueOn = relation_methods_mod.withGroupCumulativeNUniqueOn;
    pub const withGroupCumulativeNunique = relation_methods_mod.withGroupCumulativeNunique;
    pub const withGroupCumulativeNuniqueOn = relation_methods_mod.withGroupCumulativeNuniqueOn;
    pub const withGroupCumDistinctCount = relation_methods_mod.withGroupCumDistinctCount;
    pub const withGroupCumDistinctCountOn = relation_methods_mod.withGroupCumDistinctCountOn;
    pub const withGroupCumCountDistinct = relation_methods_mod.withGroupCumCountDistinct;
    pub const withGroupCumCountDistinctOn = relation_methods_mod.withGroupCumCountDistinctOn;
    pub const withGroupCumNUnique = relation_methods_mod.withGroupCumNUnique;
    pub const withGroupCumNUniqueOn = relation_methods_mod.withGroupCumNUniqueOn;
    pub const withGroupCumNunique = relation_methods_mod.withGroupCumNunique;
    pub const withGroupCumNuniqueOn = relation_methods_mod.withGroupCumNuniqueOn;
    pub const withGroupCumulativeMode = relation_methods_mod.withGroupCumulativeMode;
    pub const withGroupCumulativeModeOn = relation_methods_mod.withGroupCumulativeModeOn;
    pub const withGroupCumulativeModeCount = relation_methods_mod.withGroupCumulativeModeCount;
    pub const withGroupCumulativeModeCountOn = relation_methods_mod.withGroupCumulativeModeCountOn;
    pub const withGroupCumulativeModeRatio = relation_methods_mod.withGroupCumulativeModeRatio;
    pub const withGroupCumulativeModeRatioOn = relation_methods_mod.withGroupCumulativeModeRatioOn;
    pub const withGroupCumulativeModeMargin = relation_methods_mod.withGroupCumulativeModeMargin;
    pub const withGroupCumulativeModeMarginOn = relation_methods_mod.withGroupCumulativeModeMarginOn;
    pub const withGroupCumulativeModeMarginRatio = relation_methods_mod.withGroupCumulativeModeMarginRatio;
    pub const withGroupCumulativeModeMarginRatioOn = relation_methods_mod.withGroupCumulativeModeMarginRatioOn;
    pub const withGroupCumMode = relation_methods_mod.withGroupCumMode;
    pub const withGroupCumModeOn = relation_methods_mod.withGroupCumModeOn;
    pub const withGroupCumModeCount = relation_methods_mod.withGroupCumModeCount;
    pub const withGroupCumModeCountOn = relation_methods_mod.withGroupCumModeCountOn;
    pub const withGroupCumModeRatio = relation_methods_mod.withGroupCumModeRatio;
    pub const withGroupCumModeRatioOn = relation_methods_mod.withGroupCumModeRatioOn;
    pub const withGroupCumModeMargin = relation_methods_mod.withGroupCumModeMargin;
    pub const withGroupCumModeMarginOn = relation_methods_mod.withGroupCumModeMarginOn;
    pub const withGroupCumModeMarginRatio = relation_methods_mod.withGroupCumModeMarginRatio;
    pub const withGroupCumModeMarginRatioOn = relation_methods_mod.withGroupCumModeMarginRatioOn;
    pub const withGroupCumulativeAny = relation_methods_mod.withGroupCumulativeAny;
    pub const withGroupCumulativeAnyOn = relation_methods_mod.withGroupCumulativeAnyOn;
    pub const withGroupCumulativeAll = relation_methods_mod.withGroupCumulativeAll;
    pub const withGroupCumulativeAllOn = relation_methods_mod.withGroupCumulativeAllOn;
    pub const withGroupCumulativeTrueCount = relation_methods_mod.withGroupCumulativeTrueCount;
    pub const withGroupCumulativeTrueCountOn = relation_methods_mod.withGroupCumulativeTrueCountOn;
    pub const withGroupCumulativeFalseCount = relation_methods_mod.withGroupCumulativeFalseCount;
    pub const withGroupCumulativeFalseCountOn = relation_methods_mod.withGroupCumulativeFalseCountOn;
    pub const withGroupCumulativeTrueRatio = relation_methods_mod.withGroupCumulativeTrueRatio;
    pub const withGroupCumulativeTrueRatioOn = relation_methods_mod.withGroupCumulativeTrueRatioOn;
    pub const withGroupCumulativeFalseRatio = relation_methods_mod.withGroupCumulativeFalseRatio;
    pub const withGroupCumulativeFalseRatioOn = relation_methods_mod.withGroupCumulativeFalseRatioOn;
    pub const withGroupCumAny = relation_methods_mod.withGroupCumAny;
    pub const withGroupCumAnyOn = relation_methods_mod.withGroupCumAnyOn;
    pub const withGroupCumAll = relation_methods_mod.withGroupCumAll;
    pub const withGroupCumAllOn = relation_methods_mod.withGroupCumAllOn;
    pub const withGroupCumTrueCount = relation_methods_mod.withGroupCumTrueCount;
    pub const withGroupCumTrueCountOn = relation_methods_mod.withGroupCumTrueCountOn;
    pub const withGroupCumFalseCount = relation_methods_mod.withGroupCumFalseCount;
    pub const withGroupCumFalseCountOn = relation_methods_mod.withGroupCumFalseCountOn;
    pub const withGroupCumTrueRatio = relation_methods_mod.withGroupCumTrueRatio;
    pub const withGroupCumTrueRatioOn = relation_methods_mod.withGroupCumTrueRatioOn;
    pub const withGroupCumFalseRatio = relation_methods_mod.withGroupCumFalseRatio;
    pub const withGroupCumFalseRatioOn = relation_methods_mod.withGroupCumFalseRatioOn;
    pub const withGroupCumulativeFirstTrueIndex = relation_methods_mod.withGroupCumulativeFirstTrueIndex;
    pub const withGroupCumulativeFirstTrueIndexOn = relation_methods_mod.withGroupCumulativeFirstTrueIndexOn;
    pub const withGroupCumulativeLastTrueIndex = relation_methods_mod.withGroupCumulativeLastTrueIndex;
    pub const withGroupCumulativeLastTrueIndexOn = relation_methods_mod.withGroupCumulativeLastTrueIndexOn;
    pub const withGroupCumulativeFirstFalseIndex = relation_methods_mod.withGroupCumulativeFirstFalseIndex;
    pub const withGroupCumulativeFirstFalseIndexOn = relation_methods_mod.withGroupCumulativeFirstFalseIndexOn;
    pub const withGroupCumulativeLastFalseIndex = relation_methods_mod.withGroupCumulativeLastFalseIndex;
    pub const withGroupCumulativeLastFalseIndexOn = relation_methods_mod.withGroupCumulativeLastFalseIndexOn;
    pub const withGroupCumFirstTrueIndex = relation_methods_mod.withGroupCumFirstTrueIndex;
    pub const withGroupCumFirstTrueIndexOn = relation_methods_mod.withGroupCumFirstTrueIndexOn;
    pub const withGroupCumLastTrueIndex = relation_methods_mod.withGroupCumLastTrueIndex;
    pub const withGroupCumLastTrueIndexOn = relation_methods_mod.withGroupCumLastTrueIndexOn;
    pub const withGroupCumFirstFalseIndex = relation_methods_mod.withGroupCumFirstFalseIndex;
    pub const withGroupCumFirstFalseIndexOn = relation_methods_mod.withGroupCumFirstFalseIndexOn;
    pub const withGroupCumLastFalseIndex = relation_methods_mod.withGroupCumLastFalseIndex;
    pub const withGroupCumLastFalseIndexOn = relation_methods_mod.withGroupCumLastFalseIndexOn;
    pub const withGroupCumulativeSum = relation_methods_mod.withGroupCumulativeSum;
    pub const withGroupCumulativeSumOn = relation_methods_mod.withGroupCumulativeSumOn;
    pub const withGroupCumSum = relation_methods_mod.withGroupCumSum;
    pub const withGroupCumSumOn = relation_methods_mod.withGroupCumSumOn;
    pub const withGroupCumulativeMean = relation_methods_mod.withGroupCumulativeMean;
    pub const withGroupCumulativeMeanOn = relation_methods_mod.withGroupCumulativeMeanOn;
    pub const withGroupCumMean = relation_methods_mod.withGroupCumMean;
    pub const withGroupCumMeanOn = relation_methods_mod.withGroupCumMeanOn;
    pub const withGroupCumulativeProduct = relation_methods_mod.withGroupCumulativeProduct;
    pub const withGroupCumulativeProductOn = relation_methods_mod.withGroupCumulativeProductOn;
    pub const withGroupCumProduct = relation_methods_mod.withGroupCumProduct;
    pub const withGroupCumProductOn = relation_methods_mod.withGroupCumProductOn;
    pub const withGroupCumProd = relation_methods_mod.withGroupCumProd;
    pub const withGroupCumProdOn = relation_methods_mod.withGroupCumProdOn;
    pub const withGroupCumulativeMin = relation_methods_mod.withGroupCumulativeMin;
    pub const withGroupCumulativeMinOn = relation_methods_mod.withGroupCumulativeMinOn;
    pub const withGroupCumulativeMax = relation_methods_mod.withGroupCumulativeMax;
    pub const withGroupCumulativeMaxOn = relation_methods_mod.withGroupCumulativeMaxOn;
    pub const withGroupCumMin = relation_methods_mod.withGroupCumMin;
    pub const withGroupCumMinOn = relation_methods_mod.withGroupCumMinOn;
    pub const withGroupCumMax = relation_methods_mod.withGroupCumMax;
    pub const withGroupCumMaxOn = relation_methods_mod.withGroupCumMaxOn;
    pub const withGroupCumulativeVariance = relation_methods_mod.withGroupCumulativeVariance;
    pub const withGroupCumulativeVarianceOn = relation_methods_mod.withGroupCumulativeVarianceOn;
    pub const withGroupCumulativeVar = relation_methods_mod.withGroupCumulativeVar;
    pub const withGroupCumulativeVarOn = relation_methods_mod.withGroupCumulativeVarOn;
    pub const withGroupCumVariance = relation_methods_mod.withGroupCumVariance;
    pub const withGroupCumVarianceOn = relation_methods_mod.withGroupCumVarianceOn;
    pub const withGroupCumVar = relation_methods_mod.withGroupCumVar;
    pub const withGroupCumVarOn = relation_methods_mod.withGroupCumVarOn;
    pub const withGroupCumulativeStddev = relation_methods_mod.withGroupCumulativeStddev;
    pub const withGroupCumulativeStddevOn = relation_methods_mod.withGroupCumulativeStddevOn;
    pub const withGroupCumulativeStd = relation_methods_mod.withGroupCumulativeStd;
    pub const withGroupCumulativeStdOn = relation_methods_mod.withGroupCumulativeStdOn;
    pub const withGroupCumStddev = relation_methods_mod.withGroupCumStddev;
    pub const withGroupCumStddevOn = relation_methods_mod.withGroupCumStddevOn;
    pub const withGroupCumStd = relation_methods_mod.withGroupCumStd;
    pub const withGroupCumStdOn = relation_methods_mod.withGroupCumStdOn;
    pub const withGroupCumulativeSem = relation_methods_mod.withGroupCumulativeSem;
    pub const withGroupCumulativeSemOn = relation_methods_mod.withGroupCumulativeSemOn;
    pub const withGroupCumulativeSEM = relation_methods_mod.withGroupCumulativeSEM;
    pub const withGroupCumulativeSEMOn = relation_methods_mod.withGroupCumulativeSEMOn;
    pub const withGroupCumSem = relation_methods_mod.withGroupCumSem;
    pub const withGroupCumSemOn = relation_methods_mod.withGroupCumSemOn;
    pub const withGroupCumulativeCv = relation_methods_mod.withGroupCumulativeCv;
    pub const withGroupCumulativeCvOn = relation_methods_mod.withGroupCumulativeCvOn;
    pub const withGroupCumulativeCV = relation_methods_mod.withGroupCumulativeCV;
    pub const withGroupCumulativeCVOn = relation_methods_mod.withGroupCumulativeCVOn;
    pub const withGroupCumCv = relation_methods_mod.withGroupCumCv;
    pub const withGroupCumCvOn = relation_methods_mod.withGroupCumCvOn;
    pub const withGroupCumulativeFano = relation_methods_mod.withGroupCumulativeFano;
    pub const withGroupCumulativeFanoOn = relation_methods_mod.withGroupCumulativeFanoOn;
    pub const withGroupCumFano = relation_methods_mod.withGroupCumFano;
    pub const withGroupCumFanoOn = relation_methods_mod.withGroupCumFanoOn;
    pub const withGroupCumulativeIndexOfDispersion = relation_methods_mod.withGroupCumulativeIndexOfDispersion;
    pub const withGroupCumulativeIndexOfDispersionOn = relation_methods_mod.withGroupCumulativeIndexOfDispersionOn;
    pub const withGroupCumIndexOfDispersion = relation_methods_mod.withGroupCumIndexOfDispersion;
    pub const withGroupCumIndexOfDispersionOn = relation_methods_mod.withGroupCumIndexOfDispersionOn;
    pub const withGroupCumulativeSkewness = relation_methods_mod.withGroupCumulativeSkewness;
    pub const withGroupCumulativeSkewnessOn = relation_methods_mod.withGroupCumulativeSkewnessOn;
    pub const withGroupCumulativeSkew = relation_methods_mod.withGroupCumulativeSkew;
    pub const withGroupCumulativeSkewOn = relation_methods_mod.withGroupCumulativeSkewOn;
    pub const withGroupCumSkewness = relation_methods_mod.withGroupCumSkewness;
    pub const withGroupCumSkewnessOn = relation_methods_mod.withGroupCumSkewnessOn;
    pub const withGroupCumSkew = relation_methods_mod.withGroupCumSkew;
    pub const withGroupCumSkewOn = relation_methods_mod.withGroupCumSkewOn;
    pub const withGroupCumulativeKurtosis = relation_methods_mod.withGroupCumulativeKurtosis;
    pub const withGroupCumulativeKurtosisOn = relation_methods_mod.withGroupCumulativeKurtosisOn;
    pub const withGroupCumulativeKurt = relation_methods_mod.withGroupCumulativeKurt;
    pub const withGroupCumulativeKurtOn = relation_methods_mod.withGroupCumulativeKurtOn;
    pub const withGroupCumKurtosis = relation_methods_mod.withGroupCumKurtosis;
    pub const withGroupCumKurtosisOn = relation_methods_mod.withGroupCumKurtosisOn;
    pub const withGroupCumKurt = relation_methods_mod.withGroupCumKurt;
    pub const withGroupCumKurtOn = relation_methods_mod.withGroupCumKurtOn;
    pub const withGroupCumulativeMeanAbs = relation_methods_mod.withGroupCumulativeMeanAbs;
    pub const withGroupCumulativeMeanAbsOn = relation_methods_mod.withGroupCumulativeMeanAbsOn;
    pub const withGroupCumulativeMeanAbsolute = relation_methods_mod.withGroupCumulativeMeanAbsolute;
    pub const withGroupCumulativeMeanAbsoluteOn = relation_methods_mod.withGroupCumulativeMeanAbsoluteOn;
    pub const withGroupCumMeanAbs = relation_methods_mod.withGroupCumMeanAbs;
    pub const withGroupCumMeanAbsOn = relation_methods_mod.withGroupCumMeanAbsOn;
    pub const withGroupCumMeanAbsolute = relation_methods_mod.withGroupCumMeanAbsolute;
    pub const withGroupCumMeanAbsoluteOn = relation_methods_mod.withGroupCumMeanAbsoluteOn;
    pub const withGroupCumulativeMeanSquare = relation_methods_mod.withGroupCumulativeMeanSquare;
    pub const withGroupCumulativeMeanSquareOn = relation_methods_mod.withGroupCumulativeMeanSquareOn;
    pub const withGroupCumulativeMeanSquared = relation_methods_mod.withGroupCumulativeMeanSquared;
    pub const withGroupCumulativeMeanSquaredOn = relation_methods_mod.withGroupCumulativeMeanSquaredOn;
    pub const withGroupCumulativeMeanSq = relation_methods_mod.withGroupCumulativeMeanSq;
    pub const withGroupCumulativeMeanSqOn = relation_methods_mod.withGroupCumulativeMeanSqOn;
    pub const withGroupCumMeanSquare = relation_methods_mod.withGroupCumMeanSquare;
    pub const withGroupCumMeanSquareOn = relation_methods_mod.withGroupCumMeanSquareOn;
    pub const withGroupCumMeanSquared = relation_methods_mod.withGroupCumMeanSquared;
    pub const withGroupCumMeanSquaredOn = relation_methods_mod.withGroupCumMeanSquaredOn;
    pub const withGroupCumMeanSq = relation_methods_mod.withGroupCumMeanSq;
    pub const withGroupCumMeanSqOn = relation_methods_mod.withGroupCumMeanSqOn;
    pub const withGroupCumulativeRms = relation_methods_mod.withGroupCumulativeRms;
    pub const withGroupCumulativeRmsOn = relation_methods_mod.withGroupCumulativeRmsOn;
    pub const withGroupCumulativeRMS = relation_methods_mod.withGroupCumulativeRMS;
    pub const withGroupCumulativeRMSOn = relation_methods_mod.withGroupCumulativeRMSOn;
    pub const withGroupCumRms = relation_methods_mod.withGroupCumRms;
    pub const withGroupCumRmsOn = relation_methods_mod.withGroupCumRmsOn;
    pub const withGroupCumRMS = relation_methods_mod.withGroupCumRMS;
    pub const withGroupCumRMSOn = relation_methods_mod.withGroupCumRMSOn;
    pub const withGroupCumulativeMaxAbs = relation_methods_mod.withGroupCumulativeMaxAbs;
    pub const withGroupCumulativeMaxAbsOn = relation_methods_mod.withGroupCumulativeMaxAbsOn;
    pub const withGroupCumulativeMaxAbsolute = relation_methods_mod.withGroupCumulativeMaxAbsolute;
    pub const withGroupCumulativeMaxAbsoluteOn = relation_methods_mod.withGroupCumulativeMaxAbsoluteOn;
    pub const withGroupCumMaxAbs = relation_methods_mod.withGroupCumMaxAbs;
    pub const withGroupCumMaxAbsOn = relation_methods_mod.withGroupCumMaxAbsOn;
    pub const withGroupCumMaxAbsolute = relation_methods_mod.withGroupCumMaxAbsolute;
    pub const withGroupCumMaxAbsoluteOn = relation_methods_mod.withGroupCumMaxAbsoluteOn;
    pub const withGroupCumulativeLInfNorm = relation_methods_mod.withGroupCumulativeLInfNorm;
    pub const withGroupCumulativeLInfNormOn = relation_methods_mod.withGroupCumulativeLInfNormOn;
    pub const withGroupCumulativeLinfNorm = relation_methods_mod.withGroupCumulativeLinfNorm;
    pub const withGroupCumulativeLinfNormOn = relation_methods_mod.withGroupCumulativeLinfNormOn;
    pub const withGroupCumLInfNorm = relation_methods_mod.withGroupCumLInfNorm;
    pub const withGroupCumLInfNormOn = relation_methods_mod.withGroupCumLInfNormOn;
    pub const withGroupCumLinfNorm = relation_methods_mod.withGroupCumLinfNorm;
    pub const withGroupCumLinfNormOn = relation_methods_mod.withGroupCumLinfNormOn;
    pub const withGroupCumulativeMinAbs = relation_methods_mod.withGroupCumulativeMinAbs;
    pub const withGroupCumulativeMinAbsOn = relation_methods_mod.withGroupCumulativeMinAbsOn;
    pub const withGroupCumulativeMinAbsolute = relation_methods_mod.withGroupCumulativeMinAbsolute;
    pub const withGroupCumulativeMinAbsoluteOn = relation_methods_mod.withGroupCumulativeMinAbsoluteOn;
    pub const withGroupCumMinAbs = relation_methods_mod.withGroupCumMinAbs;
    pub const withGroupCumMinAbsOn = relation_methods_mod.withGroupCumMinAbsOn;
    pub const withGroupCumMinAbsolute = relation_methods_mod.withGroupCumMinAbsolute;
    pub const withGroupCumMinAbsoluteOn = relation_methods_mod.withGroupCumMinAbsoluteOn;
    pub const withGroupCumulativeL1Norm = relation_methods_mod.withGroupCumulativeL1Norm;
    pub const withGroupCumulativeL1NormOn = relation_methods_mod.withGroupCumulativeL1NormOn;
    pub const withGroupCumL1Norm = relation_methods_mod.withGroupCumL1Norm;
    pub const withGroupCumL1NormOn = relation_methods_mod.withGroupCumL1NormOn;
    pub const withGroupCumulativeL2Norm = relation_methods_mod.withGroupCumulativeL2Norm;
    pub const withGroupCumulativeL2NormOn = relation_methods_mod.withGroupCumulativeL2NormOn;
    pub const withGroupCumL2Norm = relation_methods_mod.withGroupCumL2Norm;
    pub const withGroupCumL2NormOn = relation_methods_mod.withGroupCumL2NormOn;
    pub const withGroupCumulativeRange = relation_methods_mod.withGroupCumulativeRange;
    pub const withGroupCumulativeRangeOn = relation_methods_mod.withGroupCumulativeRangeOn;
    pub const withGroupCumulativePtp = relation_methods_mod.withGroupCumulativePtp;
    pub const withGroupCumulativePtpOn = relation_methods_mod.withGroupCumulativePtpOn;
    pub const withGroupCumulativePTP = relation_methods_mod.withGroupCumulativePTP;
    pub const withGroupCumulativePTPOn = relation_methods_mod.withGroupCumulativePTPOn;
    pub const withGroupCumulativePeakToPeak = relation_methods_mod.withGroupCumulativePeakToPeak;
    pub const withGroupCumulativePeakToPeakOn = relation_methods_mod.withGroupCumulativePeakToPeakOn;
    pub const withGroupCumRange = relation_methods_mod.withGroupCumRange;
    pub const withGroupCumRangeOn = relation_methods_mod.withGroupCumRangeOn;
    pub const withGroupCumPtp = relation_methods_mod.withGroupCumPtp;
    pub const withGroupCumPtpOn = relation_methods_mod.withGroupCumPtpOn;
    pub const withGroupCumPTP = relation_methods_mod.withGroupCumPTP;
    pub const withGroupCumPTPOn = relation_methods_mod.withGroupCumPTPOn;
    pub const withGroupCumPeakToPeak = relation_methods_mod.withGroupCumPeakToPeak;
    pub const withGroupCumPeakToPeakOn = relation_methods_mod.withGroupCumPeakToPeakOn;
    pub const withGroupCumulativeMidrange = relation_methods_mod.withGroupCumulativeMidrange;
    pub const withGroupCumulativeMidrangeOn = relation_methods_mod.withGroupCumulativeMidrangeOn;
    pub const withGroupCumMidrange = relation_methods_mod.withGroupCumMidrange;
    pub const withGroupCumMidrangeOn = relation_methods_mod.withGroupCumMidrangeOn;
    pub const withGroupCumulativeRangeCoeff = relation_methods_mod.withGroupCumulativeRangeCoeff;
    pub const withGroupCumulativeRangeCoeffOn = relation_methods_mod.withGroupCumulativeRangeCoeffOn;
    pub const withGroupCumulativeRangeCoefficient = relation_methods_mod.withGroupCumulativeRangeCoefficient;
    pub const withGroupCumulativeRangeCoefficientOn = relation_methods_mod.withGroupCumulativeRangeCoefficientOn;
    pub const withGroupCumRangeCoeff = relation_methods_mod.withGroupCumRangeCoeff;
    pub const withGroupCumRangeCoeffOn = relation_methods_mod.withGroupCumRangeCoeffOn;
    pub const withGroupCumRangeCoefficient = relation_methods_mod.withGroupCumRangeCoefficient;
    pub const withGroupCumRangeCoefficientOn = relation_methods_mod.withGroupCumRangeCoefficientOn;
    pub const withGroupCumulativeLogSumExp = relation_methods_mod.withGroupCumulativeLogSumExp;
    pub const withGroupCumulativeLogSumExpOn = relation_methods_mod.withGroupCumulativeLogSumExpOn;
    pub const withGroupCumulativeLogsumexp = relation_methods_mod.withGroupCumulativeLogsumexp;
    pub const withGroupCumulativeLogsumexpOn = relation_methods_mod.withGroupCumulativeLogsumexpOn;
    pub const withGroupCumLogSumExp = relation_methods_mod.withGroupCumLogSumExp;
    pub const withGroupCumLogSumExpOn = relation_methods_mod.withGroupCumLogSumExpOn;
    pub const withGroupCumLogsumexp = relation_methods_mod.withGroupCumLogsumexp;
    pub const withGroupCumLogsumexpOn = relation_methods_mod.withGroupCumLogsumexpOn;
    pub const withGroupCumulativeLogMeanExp = relation_methods_mod.withGroupCumulativeLogMeanExp;
    pub const withGroupCumulativeLogMeanExpOn = relation_methods_mod.withGroupCumulativeLogMeanExpOn;
    pub const withGroupCumulativeLogmeanexp = relation_methods_mod.withGroupCumulativeLogmeanexp;
    pub const withGroupCumulativeLogmeanexpOn = relation_methods_mod.withGroupCumulativeLogmeanexpOn;
    pub const withGroupCumLogMeanExp = relation_methods_mod.withGroupCumLogMeanExp;
    pub const withGroupCumLogMeanExpOn = relation_methods_mod.withGroupCumLogMeanExpOn;
    pub const withGroupCumLogmeanexp = relation_methods_mod.withGroupCumLogmeanexp;
    pub const withGroupCumLogmeanexpOn = relation_methods_mod.withGroupCumLogmeanexpOn;
    pub const withGroupCumulativeGeometricMean = relation_methods_mod.withGroupCumulativeGeometricMean;
    pub const withGroupCumulativeGeometricMeanOn = relation_methods_mod.withGroupCumulativeGeometricMeanOn;
    pub const withGroupCumulativeGeoMean = relation_methods_mod.withGroupCumulativeGeoMean;
    pub const withGroupCumulativeGeoMeanOn = relation_methods_mod.withGroupCumulativeGeoMeanOn;
    pub const withGroupCumGeometricMean = relation_methods_mod.withGroupCumGeometricMean;
    pub const withGroupCumGeometricMeanOn = relation_methods_mod.withGroupCumGeometricMeanOn;
    pub const withGroupCumGeoMean = relation_methods_mod.withGroupCumGeoMean;
    pub const withGroupCumGeoMeanOn = relation_methods_mod.withGroupCumGeoMeanOn;
    pub const withGroupCumulativeHarmonicMean = relation_methods_mod.withGroupCumulativeHarmonicMean;
    pub const withGroupCumulativeHarmonicMeanOn = relation_methods_mod.withGroupCumulativeHarmonicMeanOn;
    pub const withGroupCumulativeHarmMean = relation_methods_mod.withGroupCumulativeHarmMean;
    pub const withGroupCumulativeHarmMeanOn = relation_methods_mod.withGroupCumulativeHarmMeanOn;
    pub const withGroupCumHarmonicMean = relation_methods_mod.withGroupCumHarmonicMean;
    pub const withGroupCumHarmonicMeanOn = relation_methods_mod.withGroupCumHarmonicMeanOn;
    pub const withGroupCumHarmMean = relation_methods_mod.withGroupCumHarmMean;
    pub const withGroupCumHarmMeanOn = relation_methods_mod.withGroupCumHarmMeanOn;
    pub const withGroupCumulativeArgMin = relation_methods_mod.withGroupCumulativeArgMin;
    pub const withGroupCumulativeArgMinOn = relation_methods_mod.withGroupCumulativeArgMinOn;
    pub const withGroupCumArgMin = relation_methods_mod.withGroupCumArgMin;
    pub const withGroupCumArgMinOn = relation_methods_mod.withGroupCumArgMinOn;
    pub const withGroupCumulativeArgmin = relation_methods_mod.withGroupCumulativeArgmin;
    pub const withGroupCumulativeArgminOn = relation_methods_mod.withGroupCumulativeArgminOn;
    pub const withGroupCumArgmin = relation_methods_mod.withGroupCumArgmin;
    pub const withGroupCumArgminOn = relation_methods_mod.withGroupCumArgminOn;
    pub const withGroupCumulativeArgMax = relation_methods_mod.withGroupCumulativeArgMax;
    pub const withGroupCumulativeArgMaxOn = relation_methods_mod.withGroupCumulativeArgMaxOn;
    pub const withGroupCumArgMax = relation_methods_mod.withGroupCumArgMax;
    pub const withGroupCumArgMaxOn = relation_methods_mod.withGroupCumArgMaxOn;
    pub const withGroupCumulativeArgmax = relation_methods_mod.withGroupCumulativeArgmax;
    pub const withGroupCumulativeArgmaxOn = relation_methods_mod.withGroupCumulativeArgmaxOn;
    pub const withGroupCumArgmax = relation_methods_mod.withGroupCumArgmax;
    pub const withGroupCumArgmaxOn = relation_methods_mod.withGroupCumArgmaxOn;
    pub const withGroupRowNumber = relation_methods_mod.withGroupRowNumber;
    pub const withGroupRowNumberOn = relation_methods_mod.withGroupRowNumberOn;
    pub const withGroupCumCount = relation_methods_mod.withGroupCumCount;
    pub const withGroupCumCountOn = relation_methods_mod.withGroupCumCountOn;
    pub const withGroupSize = relation_methods_mod.withGroupSize;
    pub const withGroupSizeOn = relation_methods_mod.withGroupSizeOn;
    pub const withGroupCount = relation_methods_mod.withGroupCount;
    pub const withGroupCountOn = relation_methods_mod.withGroupCountOn;
    pub const withGroupReverseRowNumber = relation_methods_mod.withGroupReverseRowNumber;
    pub const withGroupReverseRowNumberOn = relation_methods_mod.withGroupReverseRowNumberOn;
    pub const withGroupReverseCumCount = relation_methods_mod.withGroupReverseCumCount;
    pub const withGroupReverseCumCountOn = relation_methods_mod.withGroupReverseCumCountOn;
    pub const valueCounts = relation_methods_mod.valueCounts;
    pub const valueCountsAs = relation_methods_mod.valueCountsAs;
    pub const valueCountsOn = relation_methods_mod.valueCountsOn;
    pub const valueCountsOnAs = relation_methods_mod.valueCountsOnAs;
    pub const valueCountsSorted = relation_methods_mod.valueCountsSorted;
    pub const valueCountsSortedAs = relation_methods_mod.valueCountsSortedAs;
    pub const valueCountsOnSorted = relation_methods_mod.valueCountsOnSorted;
    pub const valueCountsOnSortedAs = relation_methods_mod.valueCountsOnSortedAs;
    pub const valueCountsSortedOn = relation_methods_mod.valueCountsSortedOn;
    pub const valueCountsSortedOnAs = relation_methods_mod.valueCountsSortedOnAs;
    pub const groupBySum = relation_methods_mod.groupBySum;
    pub const groupBySumOn = relation_methods_mod.groupBySumOn;
    pub const groupByProd = relation_methods_mod.groupByProd;
    pub const groupByProduct = relation_methods_mod.groupByProduct;
    pub const groupByProdOn = relation_methods_mod.groupByProdOn;
    pub const groupByProductOn = relation_methods_mod.groupByProductOn;
    pub const groupByMin = relation_methods_mod.groupByMin;
    pub const groupByMinOn = relation_methods_mod.groupByMinOn;
    pub const groupByMax = relation_methods_mod.groupByMax;
    pub const groupByMaxOn = relation_methods_mod.groupByMaxOn;
    pub const groupByMean = relation_methods_mod.groupByMean;
    pub const groupByMeanOn = relation_methods_mod.groupByMeanOn;
    pub const groupByFirst = relation_methods_mod.groupByFirst;
    pub const groupByFirstOn = relation_methods_mod.groupByFirstOn;
    pub const groupByLast = relation_methods_mod.groupByLast;
    pub const groupByLastOn = relation_methods_mod.groupByLastOn;
    pub const groupByFirstRow = relation_methods_mod.groupByFirstRow;
    pub const groupByFirstRowOn = relation_methods_mod.groupByFirstRowOn;
    pub const groupByLastRow = relation_methods_mod.groupByLastRow;
    pub const groupByLastRowOn = relation_methods_mod.groupByLastRowOn;
    pub const groupByNth = relation_methods_mod.groupByNth;
    pub const groupByNthOn = relation_methods_mod.groupByNthOn;
    pub const groupByNthRow = relation_methods_mod.groupByNthRow;
    pub const groupByNthRowOn = relation_methods_mod.groupByNthRowOn;
    pub const groupByNthIndex = relation_methods_mod.groupByNthIndex;
    pub const groupByNthIndexOn = relation_methods_mod.groupByNthIndexOn;
    pub const groupByNthRowIndex = relation_methods_mod.groupByNthRowIndex;
    pub const groupByNthRowIndexOn = relation_methods_mod.groupByNthRowIndexOn;
    pub const groupByNUnique = relation_methods_mod.groupByNUnique;
    pub const groupByNUniqueOn = relation_methods_mod.groupByNUniqueOn;
    pub const groupByNunique = relation_methods_mod.groupByNunique;
    pub const groupByNuniqueOn = relation_methods_mod.groupByNuniqueOn;
    pub const groupByMode = relation_methods_mod.groupByMode;
    pub const groupByModeOn = relation_methods_mod.groupByModeOn;
    pub const groupByModeCount = relation_methods_mod.groupByModeCount;
    pub const groupByModeCountOn = relation_methods_mod.groupByModeCountOn;
    pub const groupByModeRatio = relation_methods_mod.groupByModeRatio;
    pub const groupByModeRatioOn = relation_methods_mod.groupByModeRatioOn;
    pub const groupByModeMargin = relation_methods_mod.groupByModeMargin;
    pub const groupByModeMarginOn = relation_methods_mod.groupByModeMarginOn;
    pub const groupByModeMarginRatio = relation_methods_mod.groupByModeMarginRatio;
    pub const groupByModeMarginRatioOn = relation_methods_mod.groupByModeMarginRatioOn;
    pub const groupByEntropy = relation_methods_mod.groupByEntropy;
    pub const groupByEntropyOn = relation_methods_mod.groupByEntropyOn;
    pub const groupByGiniImpurity = relation_methods_mod.groupByGiniImpurity;
    pub const groupByGiniImpurityOn = relation_methods_mod.groupByGiniImpurityOn;
    pub const groupByGini = relation_methods_mod.groupByGini;
    pub const groupByGiniOn = relation_methods_mod.groupByGiniOn;
    pub const groupByPerplexity = relation_methods_mod.groupByPerplexity;
    pub const groupByPerplexityOn = relation_methods_mod.groupByPerplexityOn;
    pub const groupByInverseSimpson = relation_methods_mod.groupByInverseSimpson;
    pub const groupByInverseSimpsonOn = relation_methods_mod.groupByInverseSimpsonOn;
    pub const groupBySimpsonConcentration = relation_methods_mod.groupBySimpsonConcentration;
    pub const groupBySimpsonConcentrationOn = relation_methods_mod.groupBySimpsonConcentrationOn;
    pub const groupByConcentration = relation_methods_mod.groupByConcentration;
    pub const groupByConcentrationOn = relation_methods_mod.groupByConcentrationOn;
    pub const groupByEvenness = relation_methods_mod.groupByEvenness;
    pub const groupByEvennessOn = relation_methods_mod.groupByEvennessOn;
    pub const groupByGiniMeanDiff = relation_methods_mod.groupByGiniMeanDiff;
    pub const groupByGiniMeanDiffOn = relation_methods_mod.groupByGiniMeanDiffOn;
    pub const groupByGiniCoefficient = relation_methods_mod.groupByGiniCoefficient;
    pub const groupByGiniCoefficientOn = relation_methods_mod.groupByGiniCoefficientOn;
    pub const groupByGiniCoeff = relation_methods_mod.groupByGiniCoeff;
    pub const groupByGiniCoeffOn = relation_methods_mod.groupByGiniCoeffOn;
    pub const groupByWeightedMean = relation_methods_mod.groupByWeightedMean;
    pub const groupByWeightedMeanOn = relation_methods_mod.groupByWeightedMeanOn;
    pub const groupByWeightedVariance = relation_methods_mod.groupByWeightedVariance;
    pub const groupByWeightedVarianceOn = relation_methods_mod.groupByWeightedVarianceOn;
    pub const groupByWeightedVar = relation_methods_mod.groupByWeightedVar;
    pub const groupByWeightedVarOn = relation_methods_mod.groupByWeightedVarOn;
    pub const groupByWeightedStddev = relation_methods_mod.groupByWeightedStddev;
    pub const groupByWeightedStddevOn = relation_methods_mod.groupByWeightedStddevOn;
    pub const groupByWeightedStd = relation_methods_mod.groupByWeightedStd;
    pub const groupByWeightedStdOn = relation_methods_mod.groupByWeightedStdOn;
    pub const groupByWeightedQuantile = relation_methods_mod.groupByWeightedQuantile;
    pub const groupByWeightedQuantileOn = relation_methods_mod.groupByWeightedQuantileOn;
    pub const groupByWeightedMedian = relation_methods_mod.groupByWeightedMedian;
    pub const groupByWeightedMedianOn = relation_methods_mod.groupByWeightedMedianOn;
    pub const groupByWeightedIqr = relation_methods_mod.groupByWeightedIqr;
    pub const groupByWeightedIqrOn = relation_methods_mod.groupByWeightedIqrOn;
    pub const groupByWeightedIQR = relation_methods_mod.groupByWeightedIQR;
    pub const groupByWeightedIQROn = relation_methods_mod.groupByWeightedIQROn;
    pub const groupByWeightedMad = relation_methods_mod.groupByWeightedMad;
    pub const groupByWeightedMadOn = relation_methods_mod.groupByWeightedMadOn;
    pub const groupByWeightedMAD = relation_methods_mod.groupByWeightedMAD;
    pub const groupByWeightedMADOn = relation_methods_mod.groupByWeightedMADOn;
    pub const groupByWeightedMode = relation_methods_mod.groupByWeightedMode;
    pub const groupByWeightedModeOn = relation_methods_mod.groupByWeightedModeOn;
    pub const groupByWeightedModeWeight = relation_methods_mod.groupByWeightedModeWeight;
    pub const groupByWeightedModeWeightOn = relation_methods_mod.groupByWeightedModeWeightOn;
    pub const groupByWeightedModeRatio = relation_methods_mod.groupByWeightedModeRatio;
    pub const groupByWeightedModeRatioOn = relation_methods_mod.groupByWeightedModeRatioOn;
    pub const groupByWeightedModeMargin = relation_methods_mod.groupByWeightedModeMargin;
    pub const groupByWeightedModeMarginOn = relation_methods_mod.groupByWeightedModeMarginOn;
    pub const groupByWeightedModeMarginRatio = relation_methods_mod.groupByWeightedModeMarginRatio;
    pub const groupByWeightedModeMarginRatioOn = relation_methods_mod.groupByWeightedModeMarginRatioOn;
    pub const groupByWeightedEntropy = relation_methods_mod.groupByWeightedEntropy;
    pub const groupByWeightedEntropyOn = relation_methods_mod.groupByWeightedEntropyOn;
    pub const groupByWeightedGiniImpurity = relation_methods_mod.groupByWeightedGiniImpurity;
    pub const groupByWeightedGiniImpurityOn = relation_methods_mod.groupByWeightedGiniImpurityOn;
    pub const groupByWeightedGini = relation_methods_mod.groupByWeightedGini;
    pub const groupByWeightedGiniOn = relation_methods_mod.groupByWeightedGiniOn;
    pub const groupByWeightedPerplexity = relation_methods_mod.groupByWeightedPerplexity;
    pub const groupByWeightedPerplexityOn = relation_methods_mod.groupByWeightedPerplexityOn;
    pub const groupByWeightedInverseSimpson = relation_methods_mod.groupByWeightedInverseSimpson;
    pub const groupByWeightedInverseSimpsonOn = relation_methods_mod.groupByWeightedInverseSimpsonOn;
    pub const groupByWeightedSimpsonConcentration = relation_methods_mod.groupByWeightedSimpsonConcentration;
    pub const groupByWeightedSimpsonConcentrationOn = relation_methods_mod.groupByWeightedSimpsonConcentrationOn;
    pub const groupByWeightedConcentration = relation_methods_mod.groupByWeightedConcentration;
    pub const groupByWeightedConcentrationOn = relation_methods_mod.groupByWeightedConcentrationOn;
    pub const groupByWeightedEvenness = relation_methods_mod.groupByWeightedEvenness;
    pub const groupByWeightedEvennessOn = relation_methods_mod.groupByWeightedEvennessOn;
    pub const groupByDot = relation_methods_mod.groupByDot;
    pub const groupByDotOn = relation_methods_mod.groupByDotOn;
    pub const groupByCosineSimilarity = relation_methods_mod.groupByCosineSimilarity;
    pub const groupByCosineSimilarityOn = relation_methods_mod.groupByCosineSimilarityOn;
    pub const groupByCosine = relation_methods_mod.groupByCosine;
    pub const groupByCosineOn = relation_methods_mod.groupByCosineOn;
    pub const groupBySquaredEuclideanDistance = relation_methods_mod.groupBySquaredEuclideanDistance;
    pub const groupBySquaredEuclideanDistanceOn = relation_methods_mod.groupBySquaredEuclideanDistanceOn;
    pub const groupByEuclideanDistance = relation_methods_mod.groupByEuclideanDistance;
    pub const groupByEuclideanDistanceOn = relation_methods_mod.groupByEuclideanDistanceOn;
    pub const groupByManhattanDistance = relation_methods_mod.groupByManhattanDistance;
    pub const groupByManhattanDistanceOn = relation_methods_mod.groupByManhattanDistanceOn;
    pub const groupByChebyshevDistance = relation_methods_mod.groupByChebyshevDistance;
    pub const groupByChebyshevDistanceOn = relation_methods_mod.groupByChebyshevDistanceOn;
    pub const groupByCanberraDistance = relation_methods_mod.groupByCanberraDistance;
    pub const groupByCanberraDistanceOn = relation_methods_mod.groupByCanberraDistanceOn;
    pub const groupByBrayCurtisDistance = relation_methods_mod.groupByBrayCurtisDistance;
    pub const groupByBrayCurtisDistanceOn = relation_methods_mod.groupByBrayCurtisDistanceOn;
    pub const groupByMeanError = relation_methods_mod.groupByMeanError;
    pub const groupByMeanErrorOn = relation_methods_mod.groupByMeanErrorOn;
    pub const groupByBias = relation_methods_mod.groupByBias;
    pub const groupByBiasOn = relation_methods_mod.groupByBiasOn;
    pub const groupByMae = relation_methods_mod.groupByMae;
    pub const groupByMaeOn = relation_methods_mod.groupByMaeOn;
    pub const groupByMse = relation_methods_mod.groupByMse;
    pub const groupByMseOn = relation_methods_mod.groupByMseOn;
    pub const groupByRmse = relation_methods_mod.groupByRmse;
    pub const groupByRmseOn = relation_methods_mod.groupByRmseOn;
    pub const groupByMape = relation_methods_mod.groupByMape;
    pub const groupByMapeOn = relation_methods_mod.groupByMapeOn;
    pub const groupBySmape = relation_methods_mod.groupBySmape;
    pub const groupBySmapeOn = relation_methods_mod.groupBySmapeOn;
    pub const groupByPairCount = relation_methods_mod.groupByPairCount;
    pub const groupByPairCountOn = relation_methods_mod.groupByPairCountOn;
    pub const groupByCovariance = relation_methods_mod.groupByCovariance;
    pub const groupByCovarianceOn = relation_methods_mod.groupByCovarianceOn;
    pub const groupByCov = relation_methods_mod.groupByCov;
    pub const groupByCovOn = relation_methods_mod.groupByCovOn;
    pub const groupByCorrelation = relation_methods_mod.groupByCorrelation;
    pub const groupByCorrelationOn = relation_methods_mod.groupByCorrelationOn;
    pub const groupByCorr = relation_methods_mod.groupByCorr;
    pub const groupByCorrOn = relation_methods_mod.groupByCorrOn;
    pub const groupByBeta = relation_methods_mod.groupByBeta;
    pub const groupByBetaOn = relation_methods_mod.groupByBetaOn;
    pub const groupByWeightedDot = relation_methods_mod.groupByWeightedDot;
    pub const groupByWeightedDotOn = relation_methods_mod.groupByWeightedDotOn;
    pub const groupByWeightedCosineSimilarity = relation_methods_mod.groupByWeightedCosineSimilarity;
    pub const groupByWeightedCosineSimilarityOn = relation_methods_mod.groupByWeightedCosineSimilarityOn;
    pub const groupByWeightedCosine = relation_methods_mod.groupByWeightedCosine;
    pub const groupByWeightedCosineOn = relation_methods_mod.groupByWeightedCosineOn;
    pub const groupByWeightedSquaredEuclideanDistance = relation_methods_mod.groupByWeightedSquaredEuclideanDistance;
    pub const groupByWeightedSquaredEuclideanDistanceOn = relation_methods_mod.groupByWeightedSquaredEuclideanDistanceOn;
    pub const groupByWeightedEuclideanDistance = relation_methods_mod.groupByWeightedEuclideanDistance;
    pub const groupByWeightedEuclideanDistanceOn = relation_methods_mod.groupByWeightedEuclideanDistanceOn;
    pub const groupByWeightedManhattanDistance = relation_methods_mod.groupByWeightedManhattanDistance;
    pub const groupByWeightedManhattanDistanceOn = relation_methods_mod.groupByWeightedManhattanDistanceOn;
    pub const groupByWeightedChebyshevDistance = relation_methods_mod.groupByWeightedChebyshevDistance;
    pub const groupByWeightedChebyshevDistanceOn = relation_methods_mod.groupByWeightedChebyshevDistanceOn;
    pub const groupByWeightedCanberraDistance = relation_methods_mod.groupByWeightedCanberraDistance;
    pub const groupByWeightedCanberraDistanceOn = relation_methods_mod.groupByWeightedCanberraDistanceOn;
    pub const groupByWeightedBrayCurtisDistance = relation_methods_mod.groupByWeightedBrayCurtisDistance;
    pub const groupByWeightedBrayCurtisDistanceOn = relation_methods_mod.groupByWeightedBrayCurtisDistanceOn;
    pub const groupByWeightedMeanError = relation_methods_mod.groupByWeightedMeanError;
    pub const groupByWeightedMeanErrorOn = relation_methods_mod.groupByWeightedMeanErrorOn;
    pub const groupByWeightedBias = relation_methods_mod.groupByWeightedBias;
    pub const groupByWeightedBiasOn = relation_methods_mod.groupByWeightedBiasOn;
    pub const groupByWeightedMae = relation_methods_mod.groupByWeightedMae;
    pub const groupByWeightedMaeOn = relation_methods_mod.groupByWeightedMaeOn;
    pub const groupByWeightedMse = relation_methods_mod.groupByWeightedMse;
    pub const groupByWeightedMseOn = relation_methods_mod.groupByWeightedMseOn;
    pub const groupByWeightedRmse = relation_methods_mod.groupByWeightedRmse;
    pub const groupByWeightedRmseOn = relation_methods_mod.groupByWeightedRmseOn;
    pub const groupByWeightedMape = relation_methods_mod.groupByWeightedMape;
    pub const groupByWeightedMapeOn = relation_methods_mod.groupByWeightedMapeOn;
    pub const groupByWeightedSmape = relation_methods_mod.groupByWeightedSmape;
    pub const groupByWeightedSmapeOn = relation_methods_mod.groupByWeightedSmapeOn;
    pub const groupByWeightedCovariance = relation_methods_mod.groupByWeightedCovariance;
    pub const groupByWeightedCovarianceOn = relation_methods_mod.groupByWeightedCovarianceOn;
    pub const groupByWeightedCov = relation_methods_mod.groupByWeightedCov;
    pub const groupByWeightedCovOn = relation_methods_mod.groupByWeightedCovOn;
    pub const groupByWeightedCorrelation = relation_methods_mod.groupByWeightedCorrelation;
    pub const groupByWeightedCorrelationOn = relation_methods_mod.groupByWeightedCorrelationOn;
    pub const groupByWeightedCorr = relation_methods_mod.groupByWeightedCorr;
    pub const groupByWeightedCorrOn = relation_methods_mod.groupByWeightedCorrOn;
    pub const groupByWeightedBeta = relation_methods_mod.groupByWeightedBeta;
    pub const groupByWeightedBetaOn = relation_methods_mod.groupByWeightedBetaOn;
    pub const groupByMeanAbsDev = relation_methods_mod.groupByMeanAbsDev;
    pub const groupByMeanAbsDevOn = relation_methods_mod.groupByMeanAbsDevOn;
    pub const groupByMeanAbsDevRatio = relation_methods_mod.groupByMeanAbsDevRatio;
    pub const groupByMeanAbsDevRatioOn = relation_methods_mod.groupByMeanAbsDevRatioOn;
    pub const groupByMedian = relation_methods_mod.groupByMedian;
    pub const groupByMedianOn = relation_methods_mod.groupByMedianOn;
    pub const groupByQuantile = relation_methods_mod.groupByQuantile;
    pub const groupByQuantileOn = relation_methods_mod.groupByQuantileOn;
    pub const groupByIqr = relation_methods_mod.groupByIqr;
    pub const groupByIqrOn = relation_methods_mod.groupByIqrOn;
    pub const groupByIQR = relation_methods_mod.groupByIQR;
    pub const groupByIQROn = relation_methods_mod.groupByIQROn;
    pub const groupByMad = relation_methods_mod.groupByMad;
    pub const groupByMadOn = relation_methods_mod.groupByMadOn;
    pub const groupByMAD = relation_methods_mod.groupByMAD;
    pub const groupByMADOn = relation_methods_mod.groupByMADOn;
    pub const groupByMedianAbsDev = relation_methods_mod.groupByMedianAbsDev;
    pub const groupByMedianAbsDevOn = relation_methods_mod.groupByMedianAbsDevOn;
    pub const groupByTrimmedMean = relation_methods_mod.groupByTrimmedMean;
    pub const groupByTrimmedMeanOn = relation_methods_mod.groupByTrimmedMeanOn;
    pub const groupByWinsorizedMean = relation_methods_mod.groupByWinsorizedMean;
    pub const groupByWinsorizedMeanOn = relation_methods_mod.groupByWinsorizedMeanOn;
    pub const groupByInterdecileRange = relation_methods_mod.groupByInterdecileRange;
    pub const groupByInterdecileRangeOn = relation_methods_mod.groupByInterdecileRangeOn;
    pub const groupByIdr = relation_methods_mod.groupByIdr;
    pub const groupByIdrOn = relation_methods_mod.groupByIdrOn;
    pub const groupByIDR = relation_methods_mod.groupByIDR;
    pub const groupByIDROn = relation_methods_mod.groupByIDROn;
    pub const groupByMidhinge = relation_methods_mod.groupByMidhinge;
    pub const groupByMidhingeOn = relation_methods_mod.groupByMidhingeOn;
    pub const groupByTrimean = relation_methods_mod.groupByTrimean;
    pub const groupByTrimeanOn = relation_methods_mod.groupByTrimeanOn;
    pub const groupByBowleySkewness = relation_methods_mod.groupByBowleySkewness;
    pub const groupByBowleySkewnessOn = relation_methods_mod.groupByBowleySkewnessOn;
    pub const groupByBowleySkew = relation_methods_mod.groupByBowleySkew;
    pub const groupByBowleySkewOn = relation_methods_mod.groupByBowleySkewOn;
    pub const groupByQuartileCoeffDispersion = relation_methods_mod.groupByQuartileCoeffDispersion;
    pub const groupByQuartileCoeffDispersionOn = relation_methods_mod.groupByQuartileCoeffDispersionOn;
    pub const groupByQcd = relation_methods_mod.groupByQcd;
    pub const groupByQcdOn = relation_methods_mod.groupByQcdOn;
    pub const groupByKelleySkewness = relation_methods_mod.groupByKelleySkewness;
    pub const groupByKelleySkewnessOn = relation_methods_mod.groupByKelleySkewnessOn;
    pub const groupByKelleySkew = relation_methods_mod.groupByKelleySkew;
    pub const groupByKelleySkewOn = relation_methods_mod.groupByKelleySkewOn;
    pub const groupByVariance = relation_methods_mod.groupByVariance;
    pub const groupByVarianceOn = relation_methods_mod.groupByVarianceOn;
    pub const groupByStddev = relation_methods_mod.groupByStddev;
    pub const groupByStddevOn = relation_methods_mod.groupByStddevOn;
    pub const groupByStd = relation_methods_mod.groupByStd;
    pub const groupByStdOn = relation_methods_mod.groupByStdOn;
    pub const groupBySem = relation_methods_mod.groupBySem;
    pub const groupBySemOn = relation_methods_mod.groupBySemOn;
    pub const groupBySEM = relation_methods_mod.groupBySEM;
    pub const groupBySEMOn = relation_methods_mod.groupBySEMOn;
    pub const groupByCv = relation_methods_mod.groupByCv;
    pub const groupByCvOn = relation_methods_mod.groupByCvOn;
    pub const groupByCV = relation_methods_mod.groupByCV;
    pub const groupByCVOn = relation_methods_mod.groupByCVOn;
    pub const groupByFano = relation_methods_mod.groupByFano;
    pub const groupByFanoOn = relation_methods_mod.groupByFanoOn;
    pub const groupByIndexOfDispersion = relation_methods_mod.groupByIndexOfDispersion;
    pub const groupByIndexOfDispersionOn = relation_methods_mod.groupByIndexOfDispersionOn;
    pub const groupBySkewness = relation_methods_mod.groupBySkewness;
    pub const groupBySkewnessOn = relation_methods_mod.groupBySkewnessOn;
    pub const groupByKurtosis = relation_methods_mod.groupByKurtosis;
    pub const groupByKurtosisOn = relation_methods_mod.groupByKurtosisOn;
    pub const groupBySkew = relation_methods_mod.groupBySkew;
    pub const groupBySkewOn = relation_methods_mod.groupBySkewOn;
    pub const groupByKurt = relation_methods_mod.groupByKurt;
    pub const groupByKurtOn = relation_methods_mod.groupByKurtOn;
    pub const groupByMagnitudeVariance = relation_methods_mod.groupByMagnitudeVariance;
    pub const groupByMagnitudeVarianceOn = relation_methods_mod.groupByMagnitudeVarianceOn;
    pub const groupByAbsVariance = relation_methods_mod.groupByAbsVariance;
    pub const groupByAbsVarianceOn = relation_methods_mod.groupByAbsVarianceOn;
    pub const groupByMagnitudeVar = relation_methods_mod.groupByMagnitudeVar;
    pub const groupByMagnitudeVarOn = relation_methods_mod.groupByMagnitudeVarOn;
    pub const groupByAbsVar = relation_methods_mod.groupByAbsVar;
    pub const groupByAbsVarOn = relation_methods_mod.groupByAbsVarOn;
    pub const groupByMagnitudeStddev = relation_methods_mod.groupByMagnitudeStddev;
    pub const groupByMagnitudeStddevOn = relation_methods_mod.groupByMagnitudeStddevOn;
    pub const groupByAbsStddev = relation_methods_mod.groupByAbsStddev;
    pub const groupByAbsStddevOn = relation_methods_mod.groupByAbsStddevOn;
    pub const groupByMagnitudeStd = relation_methods_mod.groupByMagnitudeStd;
    pub const groupByMagnitudeStdOn = relation_methods_mod.groupByMagnitudeStdOn;
    pub const groupByAbsStd = relation_methods_mod.groupByAbsStd;
    pub const groupByAbsStdOn = relation_methods_mod.groupByAbsStdOn;
    pub const groupByMagnitudeSem = relation_methods_mod.groupByMagnitudeSem;
    pub const groupByMagnitudeSemOn = relation_methods_mod.groupByMagnitudeSemOn;
    pub const groupByAbsSem = relation_methods_mod.groupByAbsSem;
    pub const groupByAbsSemOn = relation_methods_mod.groupByAbsSemOn;
    pub const groupByMagnitudeCv = relation_methods_mod.groupByMagnitudeCv;
    pub const groupByMagnitudeCvOn = relation_methods_mod.groupByMagnitudeCvOn;
    pub const groupByAbsCv = relation_methods_mod.groupByAbsCv;
    pub const groupByAbsCvOn = relation_methods_mod.groupByAbsCvOn;
    pub const groupByAbsCV = relation_methods_mod.groupByAbsCV;
    pub const groupByAbsCVOn = relation_methods_mod.groupByAbsCVOn;
    pub const groupByMagnitudeFano = relation_methods_mod.groupByMagnitudeFano;
    pub const groupByMagnitudeFanoOn = relation_methods_mod.groupByMagnitudeFanoOn;
    pub const groupByAbsFano = relation_methods_mod.groupByAbsFano;
    pub const groupByAbsFanoOn = relation_methods_mod.groupByAbsFanoOn;
    pub const groupByMagnitudeIndexOfDispersion = relation_methods_mod.groupByMagnitudeIndexOfDispersion;
    pub const groupByMagnitudeIndexOfDispersionOn = relation_methods_mod.groupByMagnitudeIndexOfDispersionOn;
    pub const groupByAbsIndexOfDispersion = relation_methods_mod.groupByAbsIndexOfDispersion;
    pub const groupByAbsIndexOfDispersionOn = relation_methods_mod.groupByAbsIndexOfDispersionOn;
    pub const groupByMagnitudeSkewness = relation_methods_mod.groupByMagnitudeSkewness;
    pub const groupByMagnitudeSkewnessOn = relation_methods_mod.groupByMagnitudeSkewnessOn;
    pub const groupByAbsSkewness = relation_methods_mod.groupByAbsSkewness;
    pub const groupByAbsSkewnessOn = relation_methods_mod.groupByAbsSkewnessOn;
    pub const groupByMagnitudeSkew = relation_methods_mod.groupByMagnitudeSkew;
    pub const groupByMagnitudeSkewOn = relation_methods_mod.groupByMagnitudeSkewOn;
    pub const groupByAbsSkew = relation_methods_mod.groupByAbsSkew;
    pub const groupByAbsSkewOn = relation_methods_mod.groupByAbsSkewOn;
    pub const groupByMagnitudeKurtosis = relation_methods_mod.groupByMagnitudeKurtosis;
    pub const groupByMagnitudeKurtosisOn = relation_methods_mod.groupByMagnitudeKurtosisOn;
    pub const groupByAbsKurtosis = relation_methods_mod.groupByAbsKurtosis;
    pub const groupByAbsKurtosisOn = relation_methods_mod.groupByAbsKurtosisOn;
    pub const groupByMagnitudeKurt = relation_methods_mod.groupByMagnitudeKurt;
    pub const groupByMagnitudeKurtOn = relation_methods_mod.groupByMagnitudeKurtOn;
    pub const groupByAbsKurt = relation_methods_mod.groupByAbsKurt;
    pub const groupByAbsKurtOn = relation_methods_mod.groupByAbsKurtOn;
    pub const groupByMeanAbs = relation_methods_mod.groupByMeanAbs;
    pub const groupByMeanAbsOn = relation_methods_mod.groupByMeanAbsOn;
    pub const groupByMeanSquare = relation_methods_mod.groupByMeanSquare;
    pub const groupByMeanSquareOn = relation_methods_mod.groupByMeanSquareOn;
    pub const groupByMeanSq = relation_methods_mod.groupByMeanSq;
    pub const groupByMeanSqOn = relation_methods_mod.groupByMeanSqOn;
    pub const groupByRms = relation_methods_mod.groupByRms;
    pub const groupByRmsOn = relation_methods_mod.groupByRmsOn;
    pub const groupByRMS = relation_methods_mod.groupByRMS;
    pub const groupByRMSOn = relation_methods_mod.groupByRMSOn;
    pub const groupByL1Norm = relation_methods_mod.groupByL1Norm;
    pub const groupByL1NormOn = relation_methods_mod.groupByL1NormOn;
    pub const groupByL2Norm = relation_methods_mod.groupByL2Norm;
    pub const groupByL2NormOn = relation_methods_mod.groupByL2NormOn;
    pub const groupByMaxAbs = relation_methods_mod.groupByMaxAbs;
    pub const groupByMaxAbsOn = relation_methods_mod.groupByMaxAbsOn;
    pub const groupByMinAbs = relation_methods_mod.groupByMinAbs;
    pub const groupByMinAbsOn = relation_methods_mod.groupByMinAbsOn;
    pub const groupByHhi = relation_methods_mod.groupByHhi;
    pub const groupByHhiOn = relation_methods_mod.groupByHhiOn;
    pub const groupByHerfindahl = relation_methods_mod.groupByHerfindahl;
    pub const groupByHerfindahlOn = relation_methods_mod.groupByHerfindahlOn;
    pub const groupByHerfindahlHirschman = relation_methods_mod.groupByHerfindahlHirschman;
    pub const groupByHerfindahlHirschmanOn = relation_methods_mod.groupByHerfindahlHirschmanOn;
    pub const groupByMagnitudeNormalizedHhi = relation_methods_mod.groupByMagnitudeNormalizedHhi;
    pub const groupByMagnitudeNormalizedHhiOn = relation_methods_mod.groupByMagnitudeNormalizedHhiOn;
    pub const groupByAbsNormalizedHhi = relation_methods_mod.groupByAbsNormalizedHhi;
    pub const groupByAbsNormalizedHhiOn = relation_methods_mod.groupByAbsNormalizedHhiOn;
    pub const groupByMagnitudeSparsity = relation_methods_mod.groupByMagnitudeSparsity;
    pub const groupByMagnitudeSparsityOn = relation_methods_mod.groupByMagnitudeSparsityOn;
    pub const groupByAbsSparsity = relation_methods_mod.groupByAbsSparsity;
    pub const groupByAbsSparsityOn = relation_methods_mod.groupByAbsSparsityOn;
    pub const groupByMagnitudeInverseSimpson = relation_methods_mod.groupByMagnitudeInverseSimpson;
    pub const groupByMagnitudeInverseSimpsonOn = relation_methods_mod.groupByMagnitudeInverseSimpsonOn;
    pub const groupByAbsInverseSimpson = relation_methods_mod.groupByAbsInverseSimpson;
    pub const groupByAbsInverseSimpsonOn = relation_methods_mod.groupByAbsInverseSimpsonOn;
    pub const groupByMagnitudeSimpsonEvenness = relation_methods_mod.groupByMagnitudeSimpsonEvenness;
    pub const groupByMagnitudeSimpsonEvennessOn = relation_methods_mod.groupByMagnitudeSimpsonEvennessOn;
    pub const groupByAbsSimpsonEvenness = relation_methods_mod.groupByAbsSimpsonEvenness;
    pub const groupByAbsSimpsonEvennessOn = relation_methods_mod.groupByAbsSimpsonEvennessOn;
    pub const groupByMagnitudeDominance = relation_methods_mod.groupByMagnitudeDominance;
    pub const groupByMagnitudeDominanceOn = relation_methods_mod.groupByMagnitudeDominanceOn;
    pub const groupByAbsDominance = relation_methods_mod.groupByAbsDominance;
    pub const groupByAbsDominanceOn = relation_methods_mod.groupByAbsDominanceOn;
    pub const groupByMagnitudeDominanceMargin = relation_methods_mod.groupByMagnitudeDominanceMargin;
    pub const groupByMagnitudeDominanceMarginOn = relation_methods_mod.groupByMagnitudeDominanceMarginOn;
    pub const groupByAbsDominanceMargin = relation_methods_mod.groupByAbsDominanceMargin;
    pub const groupByAbsDominanceMarginOn = relation_methods_mod.groupByAbsDominanceMarginOn;
    pub const groupByMagnitudeEntropy = relation_methods_mod.groupByMagnitudeEntropy;
    pub const groupByMagnitudeEntropyOn = relation_methods_mod.groupByMagnitudeEntropyOn;
    pub const groupByAbsEntropy = relation_methods_mod.groupByAbsEntropy;
    pub const groupByAbsEntropyOn = relation_methods_mod.groupByAbsEntropyOn;
    pub const groupByMagnitudePerplexity = relation_methods_mod.groupByMagnitudePerplexity;
    pub const groupByMagnitudePerplexityOn = relation_methods_mod.groupByMagnitudePerplexityOn;
    pub const groupByAbsPerplexity = relation_methods_mod.groupByAbsPerplexity;
    pub const groupByAbsPerplexityOn = relation_methods_mod.groupByAbsPerplexityOn;
    pub const groupByMagnitudeEvenness = relation_methods_mod.groupByMagnitudeEvenness;
    pub const groupByMagnitudeEvennessOn = relation_methods_mod.groupByMagnitudeEvennessOn;
    pub const groupByAbsEvenness = relation_methods_mod.groupByAbsEvenness;
    pub const groupByAbsEvennessOn = relation_methods_mod.groupByAbsEvennessOn;
    pub const groupByGeometricMean = relation_methods_mod.groupByGeometricMean;
    pub const groupByGeometricMeanOn = relation_methods_mod.groupByGeometricMeanOn;
    pub const groupByGeoMean = relation_methods_mod.groupByGeoMean;
    pub const groupByGeoMeanOn = relation_methods_mod.groupByGeoMeanOn;
    pub const groupByHarmonicMean = relation_methods_mod.groupByHarmonicMean;
    pub const groupByHarmonicMeanOn = relation_methods_mod.groupByHarmonicMeanOn;
    pub const groupByLogSumExp = relation_methods_mod.groupByLogSumExp;
    pub const groupByLogSumExpOn = relation_methods_mod.groupByLogSumExpOn;
    pub const groupByLogsumexp = relation_methods_mod.groupByLogsumexp;
    pub const groupByLogsumexpOn = relation_methods_mod.groupByLogsumexpOn;
    pub const groupByLogMeanExp = relation_methods_mod.groupByLogMeanExp;
    pub const groupByLogMeanExpOn = relation_methods_mod.groupByLogMeanExpOn;
    pub const groupByLogmeanexp = relation_methods_mod.groupByLogmeanexp;
    pub const groupByLogmeanexpOn = relation_methods_mod.groupByLogmeanexpOn;
    pub const groupByPtp = relation_methods_mod.groupByPtp;
    pub const groupByPtpOn = relation_methods_mod.groupByPtpOn;
    pub const groupByPTP = relation_methods_mod.groupByPTP;
    pub const groupByPTPOn = relation_methods_mod.groupByPTPOn;
    pub const groupByPeakToPeak = relation_methods_mod.groupByPeakToPeak;
    pub const groupByPeakToPeakOn = relation_methods_mod.groupByPeakToPeakOn;
    pub const groupByMidrange = relation_methods_mod.groupByMidrange;
    pub const groupByMidrangeOn = relation_methods_mod.groupByMidrangeOn;
    pub const groupByRangeCoeff = relation_methods_mod.groupByRangeCoeff;
    pub const groupByRangeCoeffOn = relation_methods_mod.groupByRangeCoeffOn;
    pub const groupByRangeCoefficient = relation_methods_mod.groupByRangeCoefficient;
    pub const groupByRangeCoefficientOn = relation_methods_mod.groupByRangeCoefficientOn;
    pub const groupByAny = relation_methods_mod.groupByAny;
    pub const groupByAnyOn = relation_methods_mod.groupByAnyOn;
    pub const groupByAll = relation_methods_mod.groupByAll;
    pub const groupByAllOn = relation_methods_mod.groupByAllOn;
    pub const groupByTrueCount = relation_methods_mod.groupByTrueCount;
    pub const groupByTrueCountOn = relation_methods_mod.groupByTrueCountOn;
    pub const groupByFalseCount = relation_methods_mod.groupByFalseCount;
    pub const groupByFalseCountOn = relation_methods_mod.groupByFalseCountOn;
    pub const groupByTrueRatio = relation_methods_mod.groupByTrueRatio;
    pub const groupByTrueRatioOn = relation_methods_mod.groupByTrueRatioOn;
    pub const groupByFalseRatio = relation_methods_mod.groupByFalseRatio;
    pub const groupByFalseRatioOn = relation_methods_mod.groupByFalseRatioOn;
    pub const groupByFirstTrueIndex = relation_methods_mod.groupByFirstTrueIndex;
    pub const groupByFirstTrueIndexOn = relation_methods_mod.groupByFirstTrueIndexOn;
    pub const groupByLastTrueIndex = relation_methods_mod.groupByLastTrueIndex;
    pub const groupByLastTrueIndexOn = relation_methods_mod.groupByLastTrueIndexOn;
    pub const groupByFirstFalseIndex = relation_methods_mod.groupByFirstFalseIndex;
    pub const groupByFirstFalseIndexOn = relation_methods_mod.groupByFirstFalseIndexOn;
    pub const groupByLastFalseIndex = relation_methods_mod.groupByLastFalseIndex;
    pub const groupByLastFalseIndexOn = relation_methods_mod.groupByLastFalseIndexOn;
    pub const groupByAnyValid = relation_methods_mod.groupByAnyValid;
    pub const groupByAnyValidOn = relation_methods_mod.groupByAnyValidOn;
    pub const groupByAllValid = relation_methods_mod.groupByAllValid;
    pub const groupByAllValidOn = relation_methods_mod.groupByAllValidOn;
    pub const groupByAnyNull = relation_methods_mod.groupByAnyNull;
    pub const groupByAnyNullOn = relation_methods_mod.groupByAnyNullOn;
    pub const groupByAllNull = relation_methods_mod.groupByAllNull;
    pub const groupByAllNullOn = relation_methods_mod.groupByAllNullOn;
    pub const groupByValidCount = relation_methods_mod.groupByValidCount;
    pub const groupByValidCountOn = relation_methods_mod.groupByValidCountOn;
    pub const groupByNullCount = relation_methods_mod.groupByNullCount;
    pub const groupByNullCountOn = relation_methods_mod.groupByNullCountOn;
    pub const groupByValidRatio = relation_methods_mod.groupByValidRatio;
    pub const groupByValidRatioOn = relation_methods_mod.groupByValidRatioOn;
    pub const groupByNullRatio = relation_methods_mod.groupByNullRatio;
    pub const groupByNullRatioOn = relation_methods_mod.groupByNullRatioOn;
    pub const groupByFirstValidIndex = relation_methods_mod.groupByFirstValidIndex;
    pub const groupByFirstValidIndexOn = relation_methods_mod.groupByFirstValidIndexOn;
    pub const groupByLastValidIndex = relation_methods_mod.groupByLastValidIndex;
    pub const groupByLastValidIndexOn = relation_methods_mod.groupByLastValidIndexOn;
    pub const groupByFirstNullIndex = relation_methods_mod.groupByFirstNullIndex;
    pub const groupByFirstNullIndexOn = relation_methods_mod.groupByFirstNullIndexOn;
    pub const groupByLastNullIndex = relation_methods_mod.groupByLastNullIndex;
    pub const groupByLastNullIndexOn = relation_methods_mod.groupByLastNullIndexOn;
    pub const groupByNaNCount = relation_methods_mod.groupByNaNCount;
    pub const groupByNaNCountOn = relation_methods_mod.groupByNaNCountOn;
    pub const groupByNaNRatio = relation_methods_mod.groupByNaNRatio;
    pub const groupByNaNRatioOn = relation_methods_mod.groupByNaNRatioOn;
    pub const groupByInfCount = relation_methods_mod.groupByInfCount;
    pub const groupByInfCountOn = relation_methods_mod.groupByInfCountOn;
    pub const groupByInfRatio = relation_methods_mod.groupByInfRatio;
    pub const groupByInfRatioOn = relation_methods_mod.groupByInfRatioOn;
    pub const groupByPositiveInfCount = relation_methods_mod.groupByPositiveInfCount;
    pub const groupByPositiveInfCountOn = relation_methods_mod.groupByPositiveInfCountOn;
    pub const groupByPositiveInfRatio = relation_methods_mod.groupByPositiveInfRatio;
    pub const groupByPositiveInfRatioOn = relation_methods_mod.groupByPositiveInfRatioOn;
    pub const groupByNegativeInfCount = relation_methods_mod.groupByNegativeInfCount;
    pub const groupByNegativeInfCountOn = relation_methods_mod.groupByNegativeInfCountOn;
    pub const groupByNegativeInfRatio = relation_methods_mod.groupByNegativeInfRatio;
    pub const groupByNegativeInfRatioOn = relation_methods_mod.groupByNegativeInfRatioOn;
    pub const groupByFirstNaNIndex = relation_methods_mod.groupByFirstNaNIndex;
    pub const groupByFirstNaNIndexOn = relation_methods_mod.groupByFirstNaNIndexOn;
    pub const groupByLastNaNIndex = relation_methods_mod.groupByLastNaNIndex;
    pub const groupByLastNaNIndexOn = relation_methods_mod.groupByLastNaNIndexOn;
    pub const groupByFirstInfIndex = relation_methods_mod.groupByFirstInfIndex;
    pub const groupByFirstInfIndexOn = relation_methods_mod.groupByFirstInfIndexOn;
    pub const groupByLastInfIndex = relation_methods_mod.groupByLastInfIndex;
    pub const groupByLastInfIndexOn = relation_methods_mod.groupByLastInfIndexOn;
    pub const groupByFirstPositiveInfIndex = relation_methods_mod.groupByFirstPositiveInfIndex;
    pub const groupByFirstPositiveInfIndexOn = relation_methods_mod.groupByFirstPositiveInfIndexOn;
    pub const groupByLastPositiveInfIndex = relation_methods_mod.groupByLastPositiveInfIndex;
    pub const groupByLastPositiveInfIndexOn = relation_methods_mod.groupByLastPositiveInfIndexOn;
    pub const groupByFirstNegativeInfIndex = relation_methods_mod.groupByFirstNegativeInfIndex;
    pub const groupByFirstNegativeInfIndexOn = relation_methods_mod.groupByFirstNegativeInfIndexOn;
    pub const groupByLastNegativeInfIndex = relation_methods_mod.groupByLastNegativeInfIndex;
    pub const groupByLastNegativeInfIndexOn = relation_methods_mod.groupByLastNegativeInfIndexOn;
    pub const groupByFiniteCount = relation_methods_mod.groupByFiniteCount;
    pub const groupByFiniteCountOn = relation_methods_mod.groupByFiniteCountOn;
    pub const groupByFiniteRatio = relation_methods_mod.groupByFiniteRatio;
    pub const groupByFiniteRatioOn = relation_methods_mod.groupByFiniteRatioOn;
    pub const groupByFirstFiniteIndex = relation_methods_mod.groupByFirstFiniteIndex;
    pub const groupByFirstFiniteIndexOn = relation_methods_mod.groupByFirstFiniteIndexOn;
    pub const groupByLastFiniteIndex = relation_methods_mod.groupByLastFiniteIndex;
    pub const groupByLastFiniteIndexOn = relation_methods_mod.groupByLastFiniteIndexOn;
    pub const groupByNormalCount = relation_methods_mod.groupByNormalCount;
    pub const groupByNormalCountOn = relation_methods_mod.groupByNormalCountOn;
    pub const groupByNormalRatio = relation_methods_mod.groupByNormalRatio;
    pub const groupByNormalRatioOn = relation_methods_mod.groupByNormalRatioOn;
    pub const groupByFirstNormalIndex = relation_methods_mod.groupByFirstNormalIndex;
    pub const groupByFirstNormalIndexOn = relation_methods_mod.groupByFirstNormalIndexOn;
    pub const groupByLastNormalIndex = relation_methods_mod.groupByLastNormalIndex;
    pub const groupByLastNormalIndexOn = relation_methods_mod.groupByLastNormalIndexOn;
    pub const groupBySubnormalCount = relation_methods_mod.groupBySubnormalCount;
    pub const groupBySubnormalCountOn = relation_methods_mod.groupBySubnormalCountOn;
    pub const groupBySubnormalRatio = relation_methods_mod.groupBySubnormalRatio;
    pub const groupBySubnormalRatioOn = relation_methods_mod.groupBySubnormalRatioOn;
    pub const groupByFirstSubnormalIndex = relation_methods_mod.groupByFirstSubnormalIndex;
    pub const groupByFirstSubnormalIndexOn = relation_methods_mod.groupByFirstSubnormalIndexOn;
    pub const groupByLastSubnormalIndex = relation_methods_mod.groupByLastSubnormalIndex;
    pub const groupByLastSubnormalIndexOn = relation_methods_mod.groupByLastSubnormalIndexOn;
    pub const groupByNonFiniteCount = relation_methods_mod.groupByNonFiniteCount;
    pub const groupByNonFiniteCountOn = relation_methods_mod.groupByNonFiniteCountOn;
    pub const groupByNonFiniteRatio = relation_methods_mod.groupByNonFiniteRatio;
    pub const groupByNonFiniteRatioOn = relation_methods_mod.groupByNonFiniteRatioOn;
    pub const groupByFirstNonFiniteIndex = relation_methods_mod.groupByFirstNonFiniteIndex;
    pub const groupByFirstNonFiniteIndexOn = relation_methods_mod.groupByFirstNonFiniteIndexOn;
    pub const groupByLastNonFiniteIndex = relation_methods_mod.groupByLastNonFiniteIndex;
    pub const groupByLastNonFiniteIndexOn = relation_methods_mod.groupByLastNonFiniteIndexOn;
    pub const groupByZeroCount = relation_methods_mod.groupByZeroCount;
    pub const groupByZeroCountOn = relation_methods_mod.groupByZeroCountOn;
    pub const groupByZeroRatio = relation_methods_mod.groupByZeroRatio;
    pub const groupByZeroRatioOn = relation_methods_mod.groupByZeroRatioOn;
    pub const groupByFirstZeroIndex = relation_methods_mod.groupByFirstZeroIndex;
    pub const groupByFirstZeroIndexOn = relation_methods_mod.groupByFirstZeroIndexOn;
    pub const groupByLastZeroIndex = relation_methods_mod.groupByLastZeroIndex;
    pub const groupByLastZeroIndexOn = relation_methods_mod.groupByLastZeroIndexOn;
    pub const groupByPositiveZeroCount = relation_methods_mod.groupByPositiveZeroCount;
    pub const groupByPositiveZeroCountOn = relation_methods_mod.groupByPositiveZeroCountOn;
    pub const groupByPositiveZeroRatio = relation_methods_mod.groupByPositiveZeroRatio;
    pub const groupByPositiveZeroRatioOn = relation_methods_mod.groupByPositiveZeroRatioOn;
    pub const groupByNegativeZeroCount = relation_methods_mod.groupByNegativeZeroCount;
    pub const groupByNegativeZeroCountOn = relation_methods_mod.groupByNegativeZeroCountOn;
    pub const groupByNegativeZeroRatio = relation_methods_mod.groupByNegativeZeroRatio;
    pub const groupByNegativeZeroRatioOn = relation_methods_mod.groupByNegativeZeroRatioOn;
    pub const groupByFirstPositiveZeroIndex = relation_methods_mod.groupByFirstPositiveZeroIndex;
    pub const groupByFirstPositiveZeroIndexOn = relation_methods_mod.groupByFirstPositiveZeroIndexOn;
    pub const groupByLastPositiveZeroIndex = relation_methods_mod.groupByLastPositiveZeroIndex;
    pub const groupByLastPositiveZeroIndexOn = relation_methods_mod.groupByLastPositiveZeroIndexOn;
    pub const groupByFirstNegativeZeroIndex = relation_methods_mod.groupByFirstNegativeZeroIndex;
    pub const groupByFirstNegativeZeroIndexOn = relation_methods_mod.groupByFirstNegativeZeroIndexOn;
    pub const groupByLastNegativeZeroIndex = relation_methods_mod.groupByLastNegativeZeroIndex;
    pub const groupByLastNegativeZeroIndexOn = relation_methods_mod.groupByLastNegativeZeroIndexOn;
    pub const groupByNonZeroCount = relation_methods_mod.groupByNonZeroCount;
    pub const groupByNonZeroCountOn = relation_methods_mod.groupByNonZeroCountOn;
    pub const groupByNonZeroRatio = relation_methods_mod.groupByNonZeroRatio;
    pub const groupByNonZeroRatioOn = relation_methods_mod.groupByNonZeroRatioOn;
    pub const groupByFirstNonZeroIndex = relation_methods_mod.groupByFirstNonZeroIndex;
    pub const groupByFirstNonZeroIndexOn = relation_methods_mod.groupByFirstNonZeroIndexOn;
    pub const groupByLastNonZeroIndex = relation_methods_mod.groupByLastNonZeroIndex;
    pub const groupByLastNonZeroIndexOn = relation_methods_mod.groupByLastNonZeroIndexOn;
    pub const groupByPositiveCount = relation_methods_mod.groupByPositiveCount;
    pub const groupByPositiveCountOn = relation_methods_mod.groupByPositiveCountOn;
    pub const groupByPositiveRatio = relation_methods_mod.groupByPositiveRatio;
    pub const groupByPositiveRatioOn = relation_methods_mod.groupByPositiveRatioOn;
    pub const groupByFirstPositiveIndex = relation_methods_mod.groupByFirstPositiveIndex;
    pub const groupByFirstPositiveIndexOn = relation_methods_mod.groupByFirstPositiveIndexOn;
    pub const groupByLastPositiveIndex = relation_methods_mod.groupByLastPositiveIndex;
    pub const groupByLastPositiveIndexOn = relation_methods_mod.groupByLastPositiveIndexOn;
    pub const groupBySignBitCount = relation_methods_mod.groupBySignBitCount;
    pub const groupBySignBitCountOn = relation_methods_mod.groupBySignBitCountOn;
    pub const groupBySignBitRatio = relation_methods_mod.groupBySignBitRatio;
    pub const groupBySignBitRatioOn = relation_methods_mod.groupBySignBitRatioOn;
    pub const groupByFirstSignBitIndex = relation_methods_mod.groupByFirstSignBitIndex;
    pub const groupByFirstSignBitIndexOn = relation_methods_mod.groupByFirstSignBitIndexOn;
    pub const groupByLastSignBitIndex = relation_methods_mod.groupByLastSignBitIndex;
    pub const groupByLastSignBitIndexOn = relation_methods_mod.groupByLastSignBitIndexOn;
    pub const groupByNegativeCount = relation_methods_mod.groupByNegativeCount;
    pub const groupByNegativeCountOn = relation_methods_mod.groupByNegativeCountOn;
    pub const groupByNegativeRatio = relation_methods_mod.groupByNegativeRatio;
    pub const groupByNegativeRatioOn = relation_methods_mod.groupByNegativeRatioOn;
    pub const groupByFirstNegativeIndex = relation_methods_mod.groupByFirstNegativeIndex;
    pub const groupByFirstNegativeIndexOn = relation_methods_mod.groupByFirstNegativeIndexOn;
    pub const groupByLastNegativeIndex = relation_methods_mod.groupByLastNegativeIndex;
    pub const groupByLastNegativeIndexOn = relation_methods_mod.groupByLastNegativeIndexOn;
    pub const groupByArgMin = relation_methods_mod.groupByArgMin;
    pub const groupByArgMinOn = relation_methods_mod.groupByArgMinOn;
    pub const groupByArgMax = relation_methods_mod.groupByArgMax;
    pub const groupByArgMaxOn = relation_methods_mod.groupByArgMaxOn;
    pub const groupByArgmin = relation_methods_mod.groupByArgmin;
    pub const groupByArgminOn = relation_methods_mod.groupByArgminOn;
    pub const groupByArgmax = relation_methods_mod.groupByArgmax;
    pub const groupByArgmaxOn = relation_methods_mod.groupByArgmaxOn;
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
