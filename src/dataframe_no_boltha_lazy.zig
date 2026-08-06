//! No-Boltha lazy dataframe and Parquet scan stubs.
//!
//! This module is generic over the fallback dataframe/column types to avoid a
//! circular import with the main no-Boltha facade while preserving the same
//! public API shape.

const std = @import("std");
const array_mod = @import("array.zig");
const scan_summary_mod = @import("dataframe_parquet_scan_summary.zig");

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

            pub fn columnNameAt(_: *const DeviceLazyFrame, _: std.mem.Allocator, _: usize) ParquetInteropError!?[]const u8 {
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

            pub fn columnDTypes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]array_mod.DType {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeNames(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn dtypeNames(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
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

            pub fn columnDTypeBitSizes(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeClassMask(
                _: *const DeviceLazyFrame,
                _: std.mem.Allocator,
                _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass,
            ) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn columnDTypeClassCount(_: *const DeviceLazyFrame, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError!usize {
                return 0;
            }

            pub fn numericColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.numeric);
            }

            pub fn floatColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.float);
            }

            pub fn integerColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.integer);
            }

            pub fn boolColumnCount(self: *const DeviceLazyFrame) ParquetInteropError!usize {
                return self.columnDTypeClassCount(.bool);
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

            pub fn nullableColumnCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return 0;
            }

            pub fn nonNullableColumnCount(_: *const DeviceLazyFrame) ParquetInteropError!usize {
                return 0;
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

            pub fn schema(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
            }

            pub fn schemaSummary(_: *const DeviceLazyFrame, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
                return error.FeatureUnavailable;
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

            pub fn arrowFieldDTypeNames(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![][]const u8 {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeByteSizes(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeBitSizes(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]usize {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeClassMask(_: DeviceParquetScan, _: std.mem.Allocator, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError![]bool {
                return error.FeatureUnavailable;
            }

            pub fn arrowFieldDTypeClassCount(_: DeviceParquetScan, _: @import("dataframe_no_boltha_options.zig").DeviceDTypeClass) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn numericArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn floatArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn integerArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn boolArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
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

            pub fn nullableArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
                return error.FeatureUnavailable;
            }

            pub fn nonNullableArrowFieldCount(_: DeviceParquetScan) ParquetInteropError!usize {
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

            pub fn arrowSchemaSummary(_: DeviceParquetScan, _: std.mem.Allocator) ParquetInteropError![]DeviceColumnSchema {
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
