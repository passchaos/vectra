//! No-Boltha DeviceDataFrame facade stub.
//!
//! The main no-Boltha module instantiates this generic stub with its fallback
//! host/device types. Keeping the large metadata-only table facade here makes
//! the top-level fallback module easier to scan.

const std = @import("std");
const array_mod = @import("array.zig");

pub fn DeviceDataFrameType(
    comptime DataFrame: type,
    comptime DeviceColumn: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumnView: type,
    comptime DeviceColumnSchema: type,
    comptime DeviceDType: type,
    comptime DeviceDTypeClass: type,
    comptime ParquetRangePredicate: type,
    comptime DataError: type,
    comptime DeviceDataError: type,
    comptime ParquetInteropError: type,
) type {
    return struct {
        const Self = @This();
        fn unavailableSlice(comptime T: type) DeviceDataError![]T {
            return error.FeatureUnavailable;
        }

        pub fn height(_: Self) usize {
            return 0;
        }

        pub fn rowCount(_: Self) usize {
            return 0;
        }

        pub fn nRows(_: Self) usize {
            return 0;
        }

        pub fn width(_: Self) usize {
            return 0;
        }

        pub fn columnCount(_: Self) usize {
            return 0;
        }

        pub fn cols(_: Self) usize {
            return 0;
        }

        pub fn nCols(_: Self) usize {
            return 0;
        }

        pub fn columnLabels(_: Self) []const []const u8 {
            return &.{};
        }

        pub fn columnNames(_: Self) []const []const u8 {
            return &.{};
        }

        pub fn columnNamesUnique(_: Self) bool {
            return true;
        }

        pub fn hasDuplicateColumnNames(_: Self) bool {
            return false;
        }

        pub fn duplicateColumnNameCount(_: Self) usize {
            return 0;
        }

        pub fn columnDTypes(_: Self, _: std.mem.Allocator) DeviceDataError![]DeviceDType {
            return unavailableSlice(DeviceDType);
        }

        pub fn dtypes(self: Self, allocator: std.mem.Allocator) DeviceDataError![]DeviceDType {
            return self.columnDTypes(allocator);
        }

        pub fn columnDTypeNames(_: Self, _: std.mem.Allocator) DeviceDataError![][]const u8 {
            return unavailableSlice([]const u8);
        }

        pub fn dtypeNames(self: Self, allocator: std.mem.Allocator) DeviceDataError![][]const u8 {
            return self.columnDTypeNames(allocator);
        }

        pub fn columnDTypeByteSizes(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnDTypeBitSizes(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnDTypeClassMask(_: Self, _: std.mem.Allocator, _: DeviceDTypeClass) DeviceDataError![]bool {
            return unavailableSlice(bool);
        }

        pub fn columnDTypeClassCount(_: Self, _: DeviceDTypeClass) usize {
            return 0;
        }

        pub fn numericColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.numeric);
        }

        pub fn realColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.real);
        }

        pub fn floatColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.float);
        }

        pub fn integerColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.integer);
        }

        pub fn signedIntegerColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.signed_integer);
        }

        pub fn unsignedIntegerColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.unsigned_integer);
        }

        pub fn boolColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.bool);
        }

        pub fn complexColumnCount(self: Self) usize {
            return self.columnDTypeClassCount(.complex);
        }

        pub fn columnIsNumericMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .numeric);
        }

        pub fn columnIsRealMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .real);
        }

        pub fn columnIsFloatMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .float);
        }

        pub fn columnIsIntegerMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .integer);
        }

        pub fn columnIsSignedIntegerMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .signed_integer);
        }

        pub fn columnIsUnsignedIntegerMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .unsigned_integer);
        }

        pub fn columnIsBoolMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .bool);
        }

        pub fn columnIsComplexMask(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnDTypeClassMask(allocator, .complex);
        }

        pub fn columnNullCounts(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnValidCounts(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn nullCount(_: Self) usize {
            return 0;
        }

        pub fn validCount(_: Self) usize {
            return 0;
        }

        pub fn cellCount(_: Self) usize {
            return 0;
        }

        fn ratioFromCount(count: usize, rows: usize) f64 {
            _ = count;
            if (rows == 0) return std.math.nan(f64);
            return 0.0;
        }

        pub fn nullRatio(_: Self) f64 {
            return ratioFromCount(0, 0);
        }

        pub fn validRatio(_: Self) f64 {
            return ratioFromCount(0, 0);
        }

        pub fn columnNullRatios(_: Self, _: std.mem.Allocator) DeviceDataError![]f64 {
            return unavailableSlice(f64);
        }

        pub fn columnValidRatios(_: Self, _: std.mem.Allocator) DeviceDataError![]f64 {
            return unavailableSlice(f64);
        }

        pub fn columnDistinctCounts(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnNUniqueCounts(self: Self, allocator: std.mem.Allocator) DeviceDataError![]usize {
            return self.columnDistinctCounts(allocator);
        }

        pub fn columnNUnique(self: Self, allocator: std.mem.Allocator) DeviceDataError![]usize {
            return self.columnDistinctCounts(allocator);
        }

        pub fn columnDuplicateCounts(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnRepeatedCounts(self: Self, allocator: std.mem.Allocator) DeviceDataError![]usize {
            return self.columnDuplicateCounts(allocator);
        }

        pub fn columnDistinctRatios(_: Self, _: std.mem.Allocator) DeviceDataError![]f64 {
            return unavailableSlice(f64);
        }

        pub fn columnNUniqueRatios(self: Self, allocator: std.mem.Allocator) DeviceDataError![]f64 {
            return self.columnDistinctRatios(allocator);
        }

        pub fn columnDuplicateRatios(_: Self, _: std.mem.Allocator) DeviceDataError![]f64 {
            return unavailableSlice(f64);
        }

        pub fn columnIsUniqueMask(_: Self, _: std.mem.Allocator) DeviceDataError![]bool {
            return unavailableSlice(bool);
        }

        pub fn columnHasDuplicatesMask(_: Self, _: std.mem.Allocator) DeviceDataError![]bool {
            return unavailableSlice(bool);
        }

        pub fn columnHasDuplicateValues(self: Self, allocator: std.mem.Allocator) DeviceDataError![]bool {
            return self.columnHasDuplicatesMask(allocator);
        }

        pub fn columnNullableMask(_: Self, _: std.mem.Allocator) DeviceDataError![]bool {
            return unavailableSlice(bool);
        }

        pub fn nullableColumnCount(_: Self) usize {
            return 0;
        }

        pub fn nonNullableColumnCount(_: Self) usize {
            return 0;
        }

        pub fn columnHasNullsMask(_: Self, _: std.mem.Allocator) DeviceDataError![]bool {
            return unavailableSlice(bool);
        }

        pub fn columnsWithNullsCount(_: Self) usize {
            return 0;
        }

        pub fn columnsWithoutNullsCount(_: Self) usize {
            return 0;
        }

        pub fn columnDataNbytes(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnDataMemoryUsage(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return error.FeatureUnavailable;
        }

        pub fn columnValidityNbytes(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnValidityMemoryUsage(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return error.FeatureUnavailable;
        }

        pub fn columnTotalNbytes(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return unavailableSlice(usize);
        }

        pub fn columnMemoryUsage(_: Self, _: std.mem.Allocator) DeviceDataError![]usize {
            return error.FeatureUnavailable;
        }

        pub fn dataNbytes(_: Self) usize {
            return 0;
        }

        pub fn dataMemoryUsage(_: Self) usize {
            return 0;
        }

        pub fn validityNbytes(_: Self) usize {
            return 0;
        }

        pub fn validityMemoryUsage(_: Self) usize {
            return 0;
        }

        pub fn totalNbytes(_: Self) usize {
            return 0;
        }

        pub fn memoryUsage(_: Self) usize {
            return 0;
        }

        pub fn estimatedSize(_: Self) usize {
            return 0;
        }

        pub fn columnSchemaAt(_: Self, _: usize) DeviceDataError!DeviceColumnSchema {
            return error.FeatureUnavailable;
        }

        pub fn columnSchema(_: Self, _: []const u8) DataError!DeviceColumnSchema {
            return error.FeatureUnavailable;
        }

        pub fn columnSchemas(_: Self, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
            return error.FeatureUnavailable;
        }

        pub fn schema(_: Self, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
            return error.FeatureUnavailable;
        }

        pub fn schemaSummary(_: Self, _: std.mem.Allocator) DeviceDataError![]DeviceColumnSchema {
            return error.FeatureUnavailable;
        }

        pub fn isEmpty(_: Self) bool {
            return true;
        }

        pub fn isNonEmpty(_: Self) bool {
            return false;
        }

        pub fn hasRows(_: Self) bool {
            return false;
        }

        pub fn hasColumns(_: Self) bool {
            return false;
        }

        pub fn isCpu(_: Self) bool {
            return true;
        }

        pub fn isCuda(_: Self) bool {
            return false;
        }

        pub fn isMps(_: Self) bool {
            return false;
        }

        pub fn isDeviceBacked(_: Self) bool {
            return false;
        }

        pub fn isDeviceAvailable(_: Self) bool {
            return true;
        }

        pub fn deviceBackendName(_: Self) []const u8 {
            return "cpu";
        }

        pub fn deviceValue(_: Self) array_mod.Device {
            return .cpu;
        }

        pub fn deviceBackend(_: Self) array_mod.Backend {
            return .cpu;
        }

        pub fn deviceIndex(_: Self) usize {
            return 0;
        }

        pub fn sameDevice(_: Self, _: Self) bool {
            return true;
        }

        pub fn hasColumn(_: Self, _: []const u8) bool {
            return false;
        }

        pub fn hasAllColumns(_: Self, names: []const []const u8) bool {
            return names.len == 0;
        }

        pub fn hasAnyColumn(_: Self, _: []const []const u8) bool {
            return false;
        }

        pub fn shape(_: Self) struct { rows: usize, cols: usize } {
            return .{ .rows = 0, .cols = 0 };
        }

        pub fn sameShape(_: Self, _: Self) bool {
            return true;
        }

        pub fn sameStorage(_: Self, _: Self) bool {
            return true;
        }

        pub fn shapeEquals(_: Self, rows: usize, columns: usize) bool {
            return rows == 0 and columns == 0;
        }

        pub fn hasShape(self: Self, rows: usize, columns: usize) bool {
            return self.shapeEquals(rows, columns);
        }

        pub fn sameHeight(_: Self, _: Self) bool {
            return true;
        }

        pub fn sameWidth(_: Self, _: Self) bool {
            return true;
        }

        pub fn columnIndex(_: Self, _: []const u8) ?usize {
            return null;
        }

        pub fn column(_: *const Self, _: []const u8) DataError!*const DeviceColumn {
            return error.ColumnNotFound;
        }

        pub fn columnAt(_: *const Self, _: usize) DeviceDataError!*const DeviceColumn {
            return error.IndexOutOfBounds;
        }

        pub fn columnView(_: *const Self, _: []const u8) DataError!DeviceColumnView {
            return error.ColumnNotFound;
        }

        pub fn columnViewAt(_: *const Self, _: usize) DeviceDataError!DeviceColumnView {
            return error.IndexOutOfBounds;
        }

        pub fn columnNameAt(_: Self, _: usize) DeviceDataError![]const u8 {
            return error.IndexOutOfBounds;
        }

        pub fn columnDType(_: Self, _: []const u8) DataError!DeviceDType {
            return error.ColumnNotFound;
        }

        pub fn columnDTypeAt(_: Self, _: usize) DeviceDataError!DeviceDType {
            return error.IndexOutOfBounds;
        }

        pub fn init(_: std.mem.Allocator, _: []const DeviceColumnDef) DeviceDataError!Self {
            return error.FeatureUnavailable;
        }

        pub fn initEmpty(_: std.mem.Allocator, _: usize, _: array_mod.Device) DeviceDataError!Self {
            return error.FeatureUnavailable;
        }

        pub fn fromDataFrame(_: std.mem.Allocator, _: DataFrame, _: array_mod.Device) DeviceDataError!Self {
            return error.FeatureUnavailable;
        }

        pub fn fromParquetBytes(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!Self {
            return error.FeatureUnavailable;
        }

        pub fn fromParquetBytesPruned(
            _: std.mem.Allocator,
            _: []const u8,
            _: []const u8,
            _: ParquetRangePredicate,
            _: array_mod.Device,
        ) ParquetInteropError!Self {
            return error.FeatureUnavailable;
        }
    };
}
