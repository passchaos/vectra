//! No-Boltha lazy dataframe and Parquet scan stubs.
//!
//! This module is generic over the fallback dataframe/column types to avoid a
//! circular import with the main no-Boltha facade while preserving the same
//! public API shape.

const std = @import("std");
const array_mod = @import("array.zig");
const scan_summary_mod = @import("dataframe_parquet_scan_summary.zig");
const options_mod = @import("dataframe_no_boltha_options.zig");

const LazyParquetRangePredicate = options_mod.ParquetRangePredicate;
const LazyParquetNullFilter = options_mod.DeviceParquetNullFilter;

pub const OwnedLazyScanPushdownSummary = struct {
    allocator: std.mem.Allocator,
    value: DeviceParquetScanPushdownSummary = .{},

    pub fn deinit(self: *OwnedLazyScanPushdownSummary) void {
        self.* = undefined;
    }

    pub fn summary(self: OwnedLazyScanPushdownSummary) DeviceParquetScanPushdownSummary {
        return self.value;
    }
};

pub const LazyScanPushdown = struct {
    allocator: std.mem.Allocator,

    pub fn deinit(_: *LazyScanPushdown) void {}

    pub fn hasProjection(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn projectionColumnCount(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn projectionNames(_: LazyScanPushdown) []const []const u8 {
        return &.{};
    }

    pub fn projectionNameAt(_: LazyScanPushdown, _: usize) ?[]const u8 {
        return null;
    }

    pub fn projectionIndex(_: LazyScanPushdown, _: []const u8) ?usize {
        return null;
    }

    pub fn projectionContains(_: LazyScanPushdown, _: []const u8) bool {
        return false;
    }

    pub fn projectionNamesUnique(_: LazyScanPushdown) bool {
        return true;
    }

    pub fn hasDuplicateProjectionNames(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn duplicateProjectionNameCount(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn hasAllProjectionNames(_: LazyScanPushdown, _: []const []const u8) bool {
        return false;
    }

    pub fn hasAnyProjectionName(_: LazyScanPushdown, _: []const []const u8) bool {
        return false;
    }

    pub fn projectsColumn(_: LazyScanPushdown, _: []const u8) bool {
        return true;
    }

    pub fn hasRangePredicate(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn rangePredicateColumn(_: LazyScanPushdown) ?[]const u8 {
        return null;
    }

    pub fn rangePredicate(_: LazyScanPushdown) ?LazyParquetRangePredicate {
        return null;
    }

    pub fn rangePredicateDType(_: LazyScanPushdown) ?array_mod.DType {
        return null;
    }

    pub fn hasRangePredicateFor(_: LazyScanPushdown, _: []const u8) bool {
        return false;
    }

    pub fn hasNullPredicate(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn nullPredicateColumn(_: LazyScanPushdown) ?[]const u8 {
        return null;
    }

    pub fn nullPredicate(_: LazyScanPushdown) ?LazyParquetNullFilter {
        return null;
    }

    pub fn nullPredicateWantNulls(_: LazyScanPushdown) ?bool {
        return null;
    }

    pub fn hasNullPredicateFor(_: LazyScanPushdown, _: []const u8) bool {
        return false;
    }

    pub fn hasPredicate(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn predicateColumn(_: LazyScanPushdown) ?[]const u8 {
        return null;
    }

    pub fn hasPredicateFor(_: LazyScanPushdown, _: []const u8) bool {
        return false;
    }

    pub fn hasPushdown(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn isEmpty(_: LazyScanPushdown) bool {
        return true;
    }

    pub fn isNonEmpty(_: LazyScanPushdown) bool {
        return false;
    }

    pub fn projectionMetadataNbytes(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn rangePredicateMetadataNbytes(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn nullPredicateMetadataNbytes(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn predicateMetadataNbytes(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn pushdownMetadataNbytes(_: LazyScanPushdown) usize {
        return 0;
    }

    pub fn memoryUsage(self: LazyScanPushdown) usize {
        return self.pushdownMetadataNbytes();
    }

    pub fn estimatedSize(self: LazyScanPushdown) usize {
        return self.pushdownMetadataNbytes();
    }

    pub fn format(_: LazyScanPushdown, _: *std.Io.Writer) std.Io.Writer.Error!void {}

    pub fn explain(_: LazyScanPushdown, allocator: std.mem.Allocator) std.mem.Allocator.Error![]u8 {
        return allocator.dupe(u8, "none");
    }

    pub fn summary(_: LazyScanPushdown) DeviceParquetScanPushdownSummary {
        return .{};
    }

    pub fn summaryOwned(self: LazyScanPushdown, allocator: std.mem.Allocator) std.mem.Allocator.Error!OwnedLazyScanPushdownSummary {
        return .{ .allocator = allocator, .value = self.summary() };
    }
};

const DeviceColumnSchema = @import("dataframe_schema.zig").DeviceColumnSchema;
const DeviceParquetFileSummary = scan_summary_mod.DeviceParquetFileSummary;
const DeviceParquetScanSummary = scan_summary_mod.DeviceParquetScanSummary;
const DeviceParquetScanPushdownSummary = scan_summary_mod.DeviceParquetScanPushdownSummary;
const SourceRange = scan_summary_mod.SourceRange;

pub fn DeviceLazyParquetTypes(
    comptime DeviceDataFrame: type,
    comptime DeviceColumn: type,
    comptime ParquetRangePredicate: type,
    comptime DeviceDataError: type,
    comptime ParquetInteropError: type,
) type {
    return struct {
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

            pub fn scanParquetOwnedBytes(_: std.mem.Allocator, _: []u8, _: array_mod.Device) ParquetInteropError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn scanParquetFileInDir(_: std.mem.Allocator, _: std.Io.Dir, _: std.Io, _: []const u8, _: std.Io.Limit, _: array_mod.Device) ParquetInteropError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn scanParquetFile(_: std.mem.Allocator, _: std.Io, _: []const u8, _: std.Io.Limit, _: array_mod.Device) ParquetInteropError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn sourceName(_: *const DeviceLazyFrame) []const u8 {
                return "unsupported";
            }

            pub fn isDataFrameSource(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isParquetScanSource(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn opCount(_: *const DeviceLazyFrame) usize {
                return 0;
            }

            pub fn rawOpCount(self: *const DeviceLazyFrame) usize {
                return self.opCount();
            }

            pub fn optimizedOpCount(_: *const DeviceLazyFrame) DeviceDataError!usize {
                return 0;
            }

            pub fn scanPushdownSummary(_: *const DeviceLazyFrame) DeviceDataError!LazyScanPushdown {
                return .{ .allocator = std.heap.smp_allocator };
            }

            pub fn hasScanPushdown(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn usesScanPushdown(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn usesScanPushdownCollect(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn scanPushdownSummaryOwned(
                _: *const DeviceLazyFrame,
                allocator: std.mem.Allocator,
            ) DeviceDataError!OwnedLazyScanPushdownSummary {
                return .{ .allocator = allocator, .value = .{} };
            }

            pub fn isOptimizedNoOp(_: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn rowCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn height(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nRows(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn columnCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn width(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn cols(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nCols(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn cellCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn shape(_: *const DeviceLazyFrame) ParquetInteropError!struct { rows: usize, cols: usize } {
                return error.FeatureUnavailable;
            }

            pub fn hasRows(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn hasColumns(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn hasShape(_: *const DeviceLazyFrame, _: usize, _: usize) bool {
                return false;
            }

            pub fn shapeEquals(self: *const DeviceLazyFrame, rows: usize, columns: usize) bool {
                return self.hasShape(rows, columns);
            }

            pub fn sameHeight(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn sameWidth(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn sameShape(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isEmpty(_: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn isNonEmpty(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn columnNames(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn columnLabels(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn columnNamesUnique(_: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn hasDuplicateColumnNames(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn duplicateColumnNameCount(_: *const DeviceLazyFrame) usize {
                return 0;
            }

            pub fn columnNameAt(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: usize) ParquetInteropError!?[]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn columnIndex(_: *const DeviceLazyFrame, _: []const u8) ParquetInteropError!?usize {
                return error.FeatureUnavailable;
            }

            pub fn hasColumn(_: *const DeviceLazyFrame, _: []const u8) bool {
                return false;
            }

            pub fn hasAllColumns(_: *const DeviceLazyFrame, _: []const []const u8) bool {
                return false;
            }

            pub fn hasAnyColumn(_: *const DeviceLazyFrame, _: []const []const u8) bool {
                return false;
            }

            pub fn columnNullCounts(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnNullCountsProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnValidCounts(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnValidCountsProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn nullCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nullCountProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn validCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn validCountProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nullRatio(_: *const DeviceLazyFrame) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn nullRatioProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn validRatio(_: *const DeviceLazyFrame) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn validRatioProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn columnNullRatios(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn columnNullRatiosProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn columnValidRatios(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn columnValidRatiosProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn columnHasNullsMask(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn columnHasNullsMaskProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn columnsWithNullsCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn columnsWithNullsCountProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn columnsWithoutNullsCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn columnsWithoutNullsCountProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeNames(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeNamesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn dtypeNames(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn dtypeNamesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn columnDType(_: *const DeviceLazyFrame, _: []const u8) ParquetInteropError!array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeAt(_: *const DeviceLazyFrame, _: usize) ParquetInteropError!?array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeByteSizes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeByteSizesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeBitSizes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeBitSizesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeClassMask(
                _: *const DeviceLazyFrame,
                _: std.mem.Allocator,
                _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass,
            ) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeClassMaskProjection(
                _: *const DeviceLazyFrame,
                _: std.mem.Allocator,
                _: []const []const u8,
                _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass,
            ) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeClassCount(_: *const DeviceLazyFrame, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError!usize {
                return 0;
            }

            pub fn columnDTypeClassCountProjection(_: *const DeviceLazyFrame, _: []const []const u8, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn numericColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.numeric);
            }

            pub fn numericColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .numeric);
            }

            pub fn realColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.real);
            }

            pub fn realColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .real);
            }

            pub fn floatColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.float);
            }

            pub fn floatColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .float);
            }

            pub fn integerColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.integer);
            }

            pub fn integerColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .integer);
            }

            pub fn signedIntegerColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.signed_integer);
            }

            pub fn signedIntegerColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .signed_integer);
            }

            pub fn unsignedIntegerColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.unsigned_integer);
            }

            pub fn unsignedIntegerColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .unsigned_integer);
            }

            pub fn boolColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.bool);
            }

            pub fn boolColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .bool);
            }

            pub fn complexColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.complex);
            }

            pub fn complexColumnCountProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.columnDTypeClassCountProjection(names, .complex);
            }

            pub fn columnIsNumericMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .numeric);
            }

            pub fn columnIsRealMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .real);
            }

            pub fn columnIsFloatMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .float);
            }

            pub fn columnIsIntegerMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .integer);
            }

            pub fn columnIsSignedIntegerMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .signed_integer);
            }

            pub fn columnIsUnsignedIntegerMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .unsigned_integer);
            }

            pub fn columnIsBoolMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .bool);
            }

            pub fn columnIsComplexMask(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]bool {
                return self.columnDTypeClassMask(allocator, .complex);
            }

            pub fn columnNullableAt(_: *const DeviceLazyFrame, _: usize) ParquetInteropError!?bool {
                return error.FeatureUnavailable;
            }

            pub fn columnNullable(_: *const DeviceLazyFrame, _: []const u8) ParquetInteropError!bool {
                return error.FeatureUnavailable;
            }

            pub fn columnNullableMask(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn columnNullableMaskProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn nullableColumnCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return 0;
            }

            pub fn nullableColumnCountProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nonNullableColumnCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return 0;
            }

            pub fn nonNullableColumnCountProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn hasNullableColumns(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn allColumnsNullable(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn columnSchemaAt(_: *const DeviceLazyFrame, _: usize) ParquetInteropError!?DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn columnSchema(_: *const DeviceLazyFrame, _: []const u8) ParquetInteropError!DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn columnSchemas(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn columnSchemasProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn schema(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn schemaProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn schemaSummary(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn schemaSummaryProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn hasSchemaProjection(_: *const DeviceLazyFrame, _: []const []const u8) bool {
                return false;
            }

            pub fn schemaEqualsSchemas(_: *const DeviceLazyFrame, _: []const DeviceColumnSchema) bool {
                return false;
            }

            pub fn sameSchemaSchemas(self: *const DeviceLazyFrame, schemas: []const DeviceColumnSchema) bool {
                return self.schemaEqualsSchemas(schemas);
            }

            pub fn schemaCompatibleSchemas(self: *const DeviceLazyFrame, schemas: []const DeviceColumnSchema) bool {
                return self.schemaEqualsSchemas(schemas);
            }

            pub fn schemaEquals(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn sameSchema(self: *const DeviceLazyFrame, other: *const DeviceLazyFrame) bool {
                return self.schemaEquals(other);
            }

            pub fn schemaCompatible(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn columnDataNbytes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDataNbytesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDataMemoryUsage(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]usize {
                return self.columnDataNbytes(allocator);
            }

            pub fn columnDataMemoryUsageProjection(self: *const DeviceLazyFrame, allocator: std.mem.Allocator, names: []const []const u8) ParquetInteropError![]usize {
                return self.columnDataNbytesProjection(allocator, names);
            }

            pub fn columnValidityNbytes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnValidityNbytesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnValidityMemoryUsage(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]usize {
                return self.columnValidityNbytes(allocator);
            }

            pub fn columnValidityMemoryUsageProjection(self: *const DeviceLazyFrame, allocator: std.mem.Allocator, names: []const []const u8) ParquetInteropError![]usize {
                return self.columnValidityNbytesProjection(allocator, names);
            }

            pub fn columnTotalNbytes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnTotalNbytesProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnMemoryUsage(self: *const DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]usize {
                return self.columnTotalNbytes(allocator);
            }

            pub fn columnMemoryUsageProjection(self: *const DeviceLazyFrame, allocator: std.mem.Allocator, names: []const []const u8) ParquetInteropError![]usize {
                return self.columnTotalNbytesProjection(allocator, names);
            }

            pub fn dataNbytes(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn dataNbytesProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn dataMemoryUsage(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.dataNbytes();
            }

            pub fn dataMemoryUsageProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.dataNbytesProjection(names);
            }

            pub fn validityNbytes(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn validityNbytesProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn validityMemoryUsage(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.validityNbytes();
            }

            pub fn validityMemoryUsageProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.validityNbytesProjection(names);
            }

            pub fn totalNbytes(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn totalNbytesProjection(_: *const DeviceLazyFrame, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn memoryUsageProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.totalNbytesProjection(names);
            }

            pub fn estimatedSizeProjection(self: *const DeviceLazyFrame, names: []const []const u8) ParquetInteropError!usize {
                return self.totalNbytesProjection(names);
            }

            pub fn toArrowSchema(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn toArrowFields(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn hasArrowProjection(_: *const DeviceLazyFrame, _: []const []const u8) bool {
                return false;
            }

            pub fn toArrowSchemaProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn toArrowFieldsProjection(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn sourceNbytes(_: *const DeviceLazyFrame) usize {
                return 0;
            }

            pub fn sourceByteCount(self: *const DeviceLazyFrame) usize {
                return self.sourceNbytes();
            }

            pub fn nbytes(self: *const DeviceLazyFrame) usize {
                return self.sourceNbytes();
            }

            pub fn byteCount(self: *const DeviceLazyFrame) usize {
                return self.sourceNbytes();
            }

            pub fn hasBytes(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn ownedNbytes(_: *const DeviceLazyFrame) usize {
                return 0;
            }

            pub fn memoryUsage(self: *const DeviceLazyFrame) usize {
                return self.ownedNbytes();
            }

            pub fn estimatedSize(self: *const DeviceLazyFrame) usize {
                return self.ownedNbytes();
            }

            pub fn sameStorage(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn sharesStorage(self: *const DeviceLazyFrame, other: *const DeviceLazyFrame) bool {
                return self.sameStorage(other);
            }

            pub fn sameSource(self: *const DeviceLazyFrame, other: *const DeviceLazyFrame) bool {
                return self.sameStorage(other);
            }

            pub fn sharesSource(self: *const DeviceLazyFrame, other: *const DeviceLazyFrame) bool {
                return self.sameSource(other);
            }

            pub fn deviceValue(_: *const DeviceLazyFrame) array_mod.Device {
                return .cpu;
            }

            pub fn deviceBackend(_: *const DeviceLazyFrame) array_mod.Backend {
                return .cpu;
            }

            pub fn deviceBackendName(_: *const DeviceLazyFrame) []const u8 {
                return "cpu";
            }

            pub fn deviceIndex(_: *const DeviceLazyFrame) usize {
                return 0;
            }

            pub fn isCpu(_: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn isHostBacked(_: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn isCuda(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isCudaBacked(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isMps(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isMpsBacked(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isAcceleratorBacked(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isRemoteBacked(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isDeviceBacked(_: *const DeviceLazyFrame) bool {
                return false;
            }

            pub fn isDeviceAvailable(_: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn sameDevice(_: *const DeviceLazyFrame, _: *const DeviceLazyFrame) bool {
                return true;
            }

            pub fn filterIsInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
                return error.FeatureUnavailable;
            }

            pub fn filterNotInValuesColumn(_: *DeviceLazyFrame, _: []const u8, _: DeviceColumn) DeviceDataError!void {
                return error.FeatureUnavailable;
            }

            pub fn collect(_: DeviceLazyFrame) ParquetInteropError!DeviceDataFrame {
                return error.FeatureUnavailable;
            }

            pub fn explain(_: DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]u8 {
                return error.FeatureUnavailable;
            }

            pub fn explainSummary(self: DeviceLazyFrame, allocator: std.mem.Allocator) ParquetInteropError![]u8 {
                return self.explain(allocator);
            }
        };

        pub const DeviceParquetScan = struct {
            pub fn init(_: std.mem.Allocator, _: []const u8, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn initOwnedBytes(_: std.mem.Allocator, _: []u8, _: array_mod.Device) DeviceParquetScan {
                return .{};
            }

            pub fn fromFileInDir(_: std.mem.Allocator, _: std.Io.Dir, _: std.Io, _: []const u8, _: std.Io.Limit, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn fromFile(_: std.mem.Allocator, _: std.Io, _: []const u8, _: std.Io.Limit, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn deinit(_: *DeviceParquetScan) void {}

            pub fn moveBytes(_: *DeviceParquetScan) []u8 {
                return &.{};
            }

            pub fn clone(_: DeviceParquetScan) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn lazy(_: DeviceParquetScan) ParquetInteropError!DeviceLazyFrame {
                return error.FeatureUnavailable;
            }

            pub fn setDevice(_: *DeviceParquetScan, _: array_mod.Device) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn retarget(_: *DeviceParquetScan, _: array_mod.Device) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn to(_: DeviceParquetScan, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn withDevice(_: DeviceParquetScan, _: array_mod.Device) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn cpu(_: DeviceParquetScan) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn cuda(_: DeviceParquetScan, _: usize) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn mps(_: DeviceParquetScan, _: usize) ParquetInteropError!DeviceParquetScan {
                return error.FeatureUnavailable;
            }

            pub fn deviceValue(_: DeviceParquetScan) array_mod.Device {
                return .cpu;
            }

            pub fn deviceBackend(_: DeviceParquetScan) array_mod.Backend {
                return .cpu;
            }

            pub fn deviceBackendName(_: DeviceParquetScan) []const u8 {
                return "cpu";
            }

            pub fn deviceIndex(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn isCpu(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn isHostBacked(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn isCuda(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isCudaBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isMps(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isMpsBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isAcceleratorBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isRemoteBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isDeviceBacked(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn isDeviceAvailable(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn sameDevice(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return true;
            }

            pub fn sourceNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn sourcePtr(_: DeviceParquetScan) u64 {
                return 0;
            }

            pub fn dataPtr(_: DeviceParquetScan) u64 {
                return 0;
            }

            pub fn hasSourcePtr(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn sourceEndPtr(_: DeviceParquetScan) u64 {
                return 0;
            }

            pub fn sourceRange(_: DeviceParquetScan) SourceRange {
                return .{};
            }

            pub fn sharesSource(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return true;
            }

            pub fn sameSource(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return true;
            }

            pub fn sharesStorage(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return true;
            }

            pub fn sameStorage(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return true;
            }

            pub fn sourceMayOverlap(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn mayOverlap(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn sourceByteCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn nbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn byteCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn isEmpty(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn isNonEmpty(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasBytes(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn projectionMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn rangePredicateMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn nullPredicateMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn predicateMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn pushdownMetadataNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn ownedNbytes(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn memoryUsage(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn estimatedSize(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn parquetFileSummary(_: DeviceParquetScan) ParquetInteropError!DeviceParquetFileSummary {
                return error.FeatureUnavailable;
            }

            pub fn rowCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nRows(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn rowGroupCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn parquetColumnChunkCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn columnCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn width(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn cols(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nCols(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn cellCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn shape(_: DeviceParquetScan) ParquetInteropError!struct { rows: usize, cols: usize } {
                return error.FeatureUnavailable;
            }

            pub fn hasRows(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasColumns(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasShape(_: DeviceParquetScan, _: usize, _: usize) bool {
                return false;
            }

            pub fn sameHeight(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn sameWidth(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn sameShape(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn shapeEquals(_: DeviceParquetScan, _: usize, _: usize) bool {
                return false;
            }

            pub fn sameRowGroups(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn parquetTotalNbytes(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn parquetTotalCompressedNbytes(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn parquetTotalUncompressedNbytes(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn parquetCompressionRatio(_: DeviceParquetScan) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn parquetMetadataCoverageRatio(_: DeviceParquetScan) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn parquetPageIndexCoverageRatio(_: DeviceParquetScan) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn hasRowGroups(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn hasProjection(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn projectionColumnCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn projectionNames(_: DeviceParquetScan) []const []const u8 {
                return &.{};
            }

            pub fn projectionNameAt(_: DeviceParquetScan, _: usize) ?[]const u8 {
                return null;
            }

            pub fn projectionIndex(_: DeviceParquetScan, _: []const u8) ?usize {
                return null;
            }

            pub fn projectionContains(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn projectionNamesUnique(_: DeviceParquetScan) bool {
                return true;
            }

            pub fn hasDuplicateProjectionNames(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn duplicateProjectionNameCount(_: DeviceParquetScan) usize {
                return 0;
            }

            pub fn hasAllProjectionNames(_: DeviceParquetScan, _: []const []const u8) bool {
                return false;
            }

            pub fn hasAnyProjectionName(_: DeviceParquetScan, _: []const []const u8) bool {
                return false;
            }

            pub fn projectsColumn(_: DeviceParquetScan, _: []const u8) bool {
                return true;
            }

            pub fn hasPredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn predicateColumn(_: DeviceParquetScan) ?[]const u8 {
                return null;
            }

            pub fn hasPredicateFor(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn hasRangePredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn rangePredicateColumn(_: DeviceParquetScan) ?[]const u8 {
                return null;
            }

            pub fn rangePredicate(_: DeviceParquetScan) ?ParquetRangePredicate {
                return null;
            }

            pub fn rangePredicateDType(_: DeviceParquetScan) ?array_mod.DType {
                return null;
            }

            pub fn hasRangePredicateFor(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn hasNullPredicate(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn nullPredicateColumn(_: DeviceParquetScan) ?[]const u8 {
                return null;
            }

            pub fn nullPredicateWantNulls(_: DeviceParquetScan) ?bool {
                return null;
            }

            pub fn hasNullPredicateFor(_: DeviceParquetScan, _: []const u8) bool {
                return false;
            }

            pub fn arrowFieldDTypeAt(_: DeviceParquetScan, _: usize) ParquetInteropError!?array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDType(_: DeviceParquetScan, _: []const u8) ParquetInteropError!array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypes(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypesProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeNames(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeNamesProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeByteSizes(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeByteSizesProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeBitSizes(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeBitSizesProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeClassMask(_: DeviceParquetScan, _: std.mem.Allocator, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeClassMaskProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeClassCount(_: DeviceParquetScan, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeClassCountProjection(_: DeviceParquetScan, _: []const []const u8, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn numericArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn numericArrowFieldCountProjection(self: DeviceParquetScan, names: []const []const u8) ParquetInteropError!usize {
                return self.arrowFieldDTypeClassCountProjection(names, .numeric);
            }

            pub fn floatArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn floatArrowFieldCountProjection(self: DeviceParquetScan, names: []const []const u8) ParquetInteropError!usize {
                return self.arrowFieldDTypeClassCountProjection(names, .float);
            }

            pub fn integerArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn integerArrowFieldCountProjection(self: DeviceParquetScan, names: []const []const u8) ParquetInteropError!usize {
                return self.arrowFieldDTypeClassCountProjection(names, .integer);
            }

            pub fn boolArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn boolArrowFieldCountProjection(self: DeviceParquetScan, names: []const []const u8) ParquetInteropError!usize {
                return self.arrowFieldDTypeClassCountProjection(names, .bool);
            }

            pub fn arrowFieldNullableAt(_: DeviceParquetScan, _: usize) ParquetInteropError!?bool {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullable(_: DeviceParquetScan, _: []const u8) ParquetInteropError!bool {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullableMask(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullableMaskProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn nullableArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nullableArrowFieldCountProjection(_: DeviceParquetScan, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nonNullableArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nonNullableArrowFieldCountProjection(_: DeviceParquetScan, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullCount(_: DeviceParquetScan, _: []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldValidCount(_: DeviceParquetScan, _: []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullCounts(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullCountsProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldValidCounts(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldValidCountsProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullRatios(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldNullRatiosProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldValidRatios(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldValidRatiosProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowNullCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowNullCountProjection(_: DeviceParquetScan, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowValidCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowValidCountProjection(_: DeviceParquetScan, _: []const []const u8) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowNullRatio(_: DeviceParquetScan) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowNullRatioProjection(_: DeviceParquetScan, _: []const []const u8) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowValidRatio(_: DeviceParquetScan) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn arrowValidRatioProjection(_: DeviceParquetScan, _: []const []const u8) ParquetInteropError!f64 {
                return error.FeatureUnavailable;
            }

            pub fn hasNullableArrowFields(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn allArrowFieldsNullable(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn arrowColumnSchemaAt(_: DeviceParquetScan, _: usize) ParquetInteropError!?DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn arrowColumnSchema(_: DeviceParquetScan, _: []const u8) ParquetInteropError!DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn arrowColumnSchemas(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn arrowColumnSchemasProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn arrowSchemaSummary(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn arrowSchemaSummaryProjection(_: DeviceParquetScan, _: std.mem.Allocator, _: []const []const u8) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn arrowSchemaEquals(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn arrowSameSchema(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn arrowSchemaCompatible(_: DeviceParquetScan, _: DeviceParquetScan) bool {
                return false;
            }

            pub fn arrowSchemaEqualsSchemas(_: DeviceParquetScan, _: []const DeviceColumnSchema) bool {
                return false;
            }

            pub fn arrowSchemaEqualsFrame(_: DeviceParquetScan, _: anytype) bool {
                return false;
            }

            pub fn arrowSameSchemaFrame(_: DeviceParquetScan, _: anytype) bool {
                return false;
            }

            pub fn arrowSchemaCompatibleFrame(_: DeviceParquetScan, _: anytype) bool {
                return false;
            }

            pub fn hasPushdown(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn validateProjection(_: DeviceParquetScan) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn validatePredicate(_: DeviceParquetScan) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn validatePushdown(_: DeviceParquetScan) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn pushdownValid(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn validateCollect(_: DeviceParquetScan) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn collectValid(_: DeviceParquetScan) bool {
                return false;
            }

            pub fn pushdownSummary(_: DeviceParquetScan) DeviceParquetScanPushdownSummary {
                return .{};
            }

            pub fn summary(_: DeviceParquetScan) DeviceParquetScanSummary {
                return .{};
            }

            pub fn clearProjection(_: *DeviceParquetScan) void {}

            pub fn clearRangePredicate(_: *DeviceParquetScan) void {}

            pub fn clearNullPredicate(_: *DeviceParquetScan) void {}

            pub fn clearPredicate(_: *DeviceParquetScan) void {}

            pub fn clearPushdown(_: *DeviceParquetScan) void {}

            pub fn resetPushdown(_: *DeviceParquetScan) void {}

            pub fn select(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn appendSelect(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn dropSelected(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn selectAll(_: *DeviceParquetScan) void {}

            pub fn selectExcept(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn intersectSelect(_: *DeviceParquetScan, _: []const []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereRange(_: *DeviceParquetScan, _: []const u8, _: ParquetRangePredicate) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereMin(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereMax(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereBetween(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereGe(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereLe(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereGt(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereLt(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereEq(_: *DeviceParquetScan, _: []const u8, comptime _: type, _: anytype) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereBool(_: *DeviceParquetScan, _: []const u8, _: bool) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereNull(_: *DeviceParquetScan, _: []const u8, _: bool) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereIsNull(_: *DeviceParquetScan, _: []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereIsNotNull(_: *DeviceParquetScan, _: []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn whereNotNull(_: *DeviceParquetScan, _: []const u8) ParquetInteropError!void {
                return error.FeatureUnavailable;
            }

            pub fn collect(_: DeviceParquetScan) ParquetInteropError!DeviceDataFrame {
                return error.FeatureUnavailable;
            }

            pub fn explain(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]u8 {
                return error.FeatureUnavailable;
            }

            pub fn explainSummary(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]u8 {
                return error.FeatureUnavailable;
            }
        };
    };
}
