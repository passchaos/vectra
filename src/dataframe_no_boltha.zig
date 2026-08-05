const std = @import("std");
const array_mod = @import("array.zig");
const host_mod = @import("dataframe_no_boltha_host.zig");
const options_mod = @import("dataframe_no_boltha_options.zig");
const view_core_mod = @import("dataframe_view_core.zig");
const column_mod = @import("dataframe_no_boltha_column.zig");
const lazy_options_mod = @import("dataframe_no_boltha_lazy_options.zig");

pub const DataError = host_mod.DataError;
pub const DType = host_mod.DType;
pub const Column = host_mod.Column;
pub const ColumnDef = host_mod.ColumnDef;
pub const DataFrame = host_mod.DataFrame;
pub const dataframe = host_mod.dataframe;

pub const DeviceDType = array_mod.DType;
pub const DeviceDataError = DataError || array_mod.ArrayError;
pub const ArrowInteropError = DeviceDataError || error{FeatureUnavailable};
pub const ParquetInteropError = ArrowInteropError;

pub const DeviceValidityEncoding = options_mod.DeviceValidityEncoding;
pub const DeviceColumnBinaryOp = options_mod.DeviceColumnBinaryOp;
pub const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
pub const DeviceColumnLogicalOp = options_mod.DeviceColumnLogicalOp;
pub const DeviceDTypeClass = options_mod.DeviceDTypeClass;
pub const DeviceScalar = options_mod.DeviceScalar;
pub const DeviceGroupByAggregation = options_mod.DeviceGroupByAggregation;
pub const NullPlacement = options_mod.NullPlacement;
pub const DeviceSortOptions = options_mod.DeviceSortOptions;
pub const DeviceClipOptions = options_mod.DeviceClipOptions;
pub const DeviceThresholdOptions = options_mod.DeviceThresholdOptions;
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
pub const DeviceRollingCorrelationOptions = options_mod.DeviceRollingCorrelationOptions;
pub const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
pub const DeviceRollingRobustOptions = options_mod.DeviceRollingRobustOptions;
pub const DeviceJoinOptions = options_mod.DeviceJoinOptions;
pub const AsofStrategy = options_mod.AsofStrategy;
pub const DeviceAsofOptions = options_mod.DeviceAsofOptions;
pub const Range = options_mod.Range;
pub const ParquetRangePredicate = options_mod.ParquetRangePredicate;
pub const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
pub const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;

const view_types = view_core_mod.DeviceViewTypes(DeviceValidityEncoding, DeviceDTypeClass, DataError);
pub const DeviceDataFrameViewError = view_types.DeviceDataFrameViewError;
pub const DeviceColumnView = view_types.DeviceColumnView;
pub const DeviceDataFrameView = view_types.DeviceDataFrameView;
pub const DeviceTypedColumn = column_mod.DeviceTypedColumn;
pub const DeviceColumn = column_mod.DeviceColumn;
pub const DeviceColumnDef = column_mod.DeviceColumnDef;
pub const DeviceColumnSchema = column_mod.DeviceColumnSchema;
pub const DeviceLazyGroupByAggregation = lazy_options_mod.DeviceLazyGroupByAggregation;
pub const DeviceLazyWeightedGroupByAggregation = lazy_options_mod.DeviceLazyWeightedGroupByAggregation;
pub const DeviceLazyPairGroupByAggregation = lazy_options_mod.DeviceLazyPairGroupByAggregation;
pub const DeviceLazyWeightedPairGroupByAggregation = lazy_options_mod.DeviceLazyWeightedPairGroupByAggregation;
pub const DeviceLazyJoinKind = lazy_options_mod.DeviceLazyJoinKind;

pub const DeviceLazySource = union(enum) {
    unsupported: void,
};

pub const DeviceLazyFrame = struct {
    pub fn init(_: std.mem.Allocator, _: DeviceDataFrame) DeviceDataError!DeviceLazyFrame {
        return error.FeatureUnavailable;
    }

    pub fn scanParquetBytes(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceLazyFrame {
        return error.FeatureUnavailable;
    }

    pub fn filterIsInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
        return error.FeatureUnavailable;
    }

    pub fn filterNotInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
        return error.FeatureUnavailable;
    }
};

pub const DeviceParquetScan = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
        return error.FeatureUnavailable;
    }

    pub fn deinit(_: *DeviceParquetScan) void {}

    pub fn clone(_: DeviceParquetScan) ParquetInteropError!DeviceParquetScan {
        return error.FeatureUnavailable;
    }

    pub fn lazy(_: DeviceParquetScan) ParquetInteropError!DeviceLazyFrame {
        return error.FeatureUnavailable;
    }

    pub fn select(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
        return error.FeatureUnavailable;
    }

    pub fn whereRange(_: *DeviceParquetScan, _: []const u8, _: ParquetRangePredicate) ParquetInteropError!void {
        return error.FeatureUnavailable;
    }

    pub fn whereNull(_: *DeviceParquetScan, _: []const u8, _: bool) ParquetInteropError!void {
        return error.FeatureUnavailable;
    }

    pub fn collect(_: DeviceParquetScan) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn explain(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]u8 {
        return error.FeatureUnavailable;
    }
};

pub const DeviceDataFrame = struct {
    fn unavailableSlice(comptime T: type) DeviceDataError![]T {
        return error.FeatureUnavailable;
    }

    pub fn height(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn rowCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn nRows(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn width(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn cols(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn nCols(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnLabels(_: DeviceDataFrame) []const []const u8 {
        return &.{};
    }

    pub fn columnNames(_: DeviceDataFrame) []const []const u8 {
        return &.{};
    }

    pub fn columnNamesUnique(_: DeviceDataFrame) bool {
        return true;
    }

    pub fn hasDuplicateColumnNames(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn duplicateColumnNameCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnDTypes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceDType {
        return unavailableSlice(DeviceDType);
    }

    pub fn dtypes(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]DeviceDType {
        return self.columnDTypes(allocator);
    }

    pub fn columnDTypeNames(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![][]const u8 {
        return unavailableSlice([]const u8);
    }

    pub fn dtypeNames(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![][]const u8 {
        return self.columnDTypeNames(allocator);
    }

    pub fn columnDTypeByteSizes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnDTypeBitSizes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnDTypeClassMask(_: DeviceDataFrame, _: std.mem.Allocator, _: DeviceDTypeClass) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnDTypeClassCount(_: DeviceDataFrame, _: DeviceDTypeClass) usize {
        return 0;
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

    pub fn columnIsNumericMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .numeric);
    }

    pub fn columnIsRealMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .real);
    }

    pub fn columnIsFloatMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .float);
    }

    pub fn columnIsIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .integer);
    }

    pub fn columnIsSignedIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .signed_integer);
    }

    pub fn columnIsUnsignedIntegerMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .unsigned_integer);
    }

    pub fn columnIsBoolMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .bool);
    }

    pub fn columnIsComplexMask(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnDTypeClassMask(allocator, .complex);
    }

    pub fn columnNullCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnValidCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn nullCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn validCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn cellCount(_: DeviceDataFrame) usize {
        return 0;
    }

    fn ratioFromCount(count: usize, rows: usize) f64 {
        _ = count;
        if (rows == 0) return std.math.nan(f64);
        return 0.0;
    }

    pub fn nullRatio(_: DeviceDataFrame) f64 {
        return ratioFromCount(0, 0);
    }

    pub fn validRatio(_: DeviceDataFrame) f64 {
        return ratioFromCount(0, 0);
    }

    pub fn columnNullRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnValidRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnDistinctCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnNUniqueCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDistinctCounts(allocator);
    }

    pub fn columnNUnique(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDistinctCounts(allocator);
    }

    pub fn columnDuplicateCounts(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnRepeatedCounts(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]usize {
        return self.columnDuplicateCounts(allocator);
    }

    pub fn columnDistinctRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnNUniqueRatios(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]f64 {
        return self.columnDistinctRatios(allocator);
    }

    pub fn columnDuplicateRatios(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]f64 {
        return unavailableSlice(f64);
    }

    pub fn columnIsUniqueMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnHasDuplicatesMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnHasDuplicateValues(self: DeviceDataFrame, allocator: std.mem.Allocator) DeviceDataError![]bool {
        return self.columnHasDuplicatesMask(allocator);
    }

    pub fn columnNullableMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn nullableColumnCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn nonNullableColumnCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnHasNullsMask(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]bool {
        return unavailableSlice(bool);
    }

    pub fn columnsWithNullsCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnsWithoutNullsCount(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnDataNbytes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnDataMemoryUsage(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return error.FeatureUnavailable;
    }

    pub fn columnValidityNbytes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnValidityMemoryUsage(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return error.FeatureUnavailable;
    }

    pub fn columnTotalNbytes(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return unavailableSlice(usize);
    }

    pub fn columnMemoryUsage(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]usize {
        return error.FeatureUnavailable;
    }

    pub fn dataNbytes(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn dataMemoryUsage(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn validityNbytes(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn validityMemoryUsage(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn totalNbytes(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn memoryUsage(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn estimatedSize(_: DeviceDataFrame) usize {
        return 0;
    }

    pub fn columnSchemaAt(_: DeviceDataFrame, _: usize) DeviceDataError!DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn columnSchema(_: DeviceDataFrame, _: []const u8) DataError!DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn columnSchemas(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn schema(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn schemaSummary(_: DeviceDataFrame, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
        return error.FeatureUnavailable;
    }

    pub fn isEmpty(_: DeviceDataFrame) bool {
        return true;
    }

    pub fn isNonEmpty(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn hasRows(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn hasColumns(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn isCpu(_: DeviceDataFrame) bool {
        return true;
    }

    pub fn isCuda(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn isMps(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn isDeviceBacked(_: DeviceDataFrame) bool {
        return false;
    }

    pub fn deviceBackendName(_: DeviceDataFrame) []const u8 {
        return "cpu";
    }

    pub fn sameDevice(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn hasColumn(_: DeviceDataFrame, _: []const u8) bool {
        return false;
    }

    pub fn hasAllColumns(_: DeviceDataFrame, names: []const []const u8) bool {
        return names.len == 0;
    }

    pub fn hasAnyColumn(_: DeviceDataFrame, _: []const []const u8) bool {
        return false;
    }

    pub fn shape(_: DeviceDataFrame) struct { rows: usize, cols: usize } {
        return .{ .rows = 0, .cols = 0 };
    }

    pub fn sameShape(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn shapeEquals(_: DeviceDataFrame, rows: usize, columns: usize) bool {
        return rows == 0 and columns == 0;
    }

    pub fn hasShape(self: DeviceDataFrame, rows: usize, columns: usize) bool {
        return self.shapeEquals(rows, columns);
    }

    pub fn sameHeight(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn sameWidth(_: DeviceDataFrame, _: DeviceDataFrame) bool {
        return true;
    }

    pub fn columnIndex(_: DeviceDataFrame, _: []const u8) ?usize {
        return null;
    }

    pub fn column(_: *const DeviceDataFrame, _: []const u8) DataError!*const DeviceColumn {
        return error.ColumnNotFound;
    }

    pub fn columnAt(_: *const DeviceDataFrame, _: usize) DeviceDataError!*const DeviceColumn {
        return error.IndexOutOfBounds;
    }

    pub fn columnView(_: *const DeviceDataFrame, _: []const u8) DataError!DeviceColumnView {
        return error.ColumnNotFound;
    }

    pub fn columnViewAt(_: *const DeviceDataFrame, _: usize) DeviceDataError!DeviceColumnView {
        return error.IndexOutOfBounds;
    }

    pub fn columnNameAt(_: DeviceDataFrame, _: usize) DeviceDataError![]const u8 {
        return error.IndexOutOfBounds;
    }

    pub fn columnDType(_: DeviceDataFrame, _: []const u8) DataError!DeviceDType {
        return error.ColumnNotFound;
    }

    pub fn columnDTypeAt(_: DeviceDataFrame, _: usize) DeviceDataError!DeviceDType {
        return error.IndexOutOfBounds;
    }

    pub fn init(_: std.mem.Allocator, _: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn initEmpty(_: std.mem.Allocator, _: usize, _: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn fromDataFrame(_: std.mem.Allocator, _: DataFrame, _: array_mod.Device) DeviceDataError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn fromParquetBytes(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }

    pub fn fromParquetBytesPruned(
        _: std.mem.Allocator,
        _: []const u8,
        _: []const u8,
        _: ParquetRangePredicate,
        _: array_mod.Device,
    ) ParquetInteropError!DeviceDataFrame {
        return error.FeatureUnavailable;
    }
};

pub fn deviceDataFrame(allocator: std.mem.Allocator, defs: []const DeviceColumnDef) DeviceDataError!DeviceDataFrame {
    return DeviceDataFrame.init(allocator, defs);
}
