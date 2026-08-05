//! Parquet byte scan state and collection helpers for `DeviceLazyFrame`.
//!
//! This module owns the scan lifecycle so the public dataframe facade can keep
//! only the type alias. The implementation still receives the concrete
//! dataframe/lazy-frame types as comptime parameters, preserving existing public
//! method signatures without creating a module import cycle.

const std = @import("std");
const array_mod = @import("../../array.zig");
const dataframe_arrow_mod = @import("../arrow.zig");
const lazy_format_mod = @import("../lazy.zig");
const names_mod = @import("../../dataframe_names.zig");
const options_mod = @import("../../dataframe_options.zig");
const scan_summary_mod = @import("../../dataframe_parquet_scan_summary.zig");
const schema_mod = @import("../../dataframe_schema.zig");
const scan_metadata_mod = @import("scan_metadata.zig");
const series_mod = @import("../../series.zig");
const boltha = @import("boltha");

const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
const DeviceColumnSchema = schema_mod.DeviceColumnSchema;
const DeviceParquetFileSummary = scan_summary_mod.DeviceParquetFileSummary;
const DeviceParquetScanSummary = scan_summary_mod.DeviceParquetScanSummary;
const DeviceParquetScanPushdownSummary = scan_summary_mod.DeviceParquetScanPushdownSummary;
const SourceRange = scan_summary_mod.SourceRange;
const ParquetRangePredicate = options_mod.ParquetRangePredicate;
const ParquetInteropError = dataframe_arrow_mod.ParquetInteropError;

pub fn DeviceParquetScan(
    comptime DeviceDataFrame: type,
    comptime DeviceLazyFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
) type {
    return struct {
        allocator: std.mem.Allocator,
        bytes: []u8,
        device: array_mod.Device,
        projection: ?[][]const u8 = null,
        range_predicate: ?DeviceParquetRangeFilter = null,
        null_predicate: ?DeviceParquetNullFilter = null,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, bytes: []const u8, device_value: array_mod.Device) std.mem.Allocator.Error!Self {
            return .{
                .allocator = allocator,
                .bytes = try allocator.dupe(u8, bytes),
                .device = device_value,
            };
        }

        pub fn deinit(self: *Self) void {
            self.clearPushdown();
            self.allocator.free(self.bytes);
            self.* = undefined;
        }

        pub fn clone(self: Self) std.mem.Allocator.Error!Self {
            var cloned = try Self.init(self.allocator, self.bytes, self.device);
            errdefer cloned.deinit();
            if (self.projection) |names| try cloned.select(names);
            if (self.range_predicate) |predicate| try cloned.whereRange(predicate.column, predicate.predicate);
            if (self.null_predicate) |predicate| try cloned.whereNull(predicate.column, predicate.want_nulls);
            return cloned;
        }

        pub fn lazy(self: Self) DeviceDataError!DeviceLazyFrame {
            return DeviceLazyFrame.initParquetScan(self.allocator, self);
        }

        fn requireDeviceAvailable(device_value: array_mod.Device) array_mod.ArrayError!void {
            if (!device_value.isAvailable()) return error.InvalidDevice;
        }

        /// Retarget the device used when `collect()` materializes decoded
        /// Arrow columns. The Parquet byte buffer itself remains host-owned;
        /// this mirrors lazy dataframe planning where scan bytes are metadata
        /// and device residency begins at the Arrow -> Vectra column boundary.
        pub fn setDevice(self: *Self, device_value: array_mod.Device) array_mod.ArrayError!void {
            try requireDeviceAvailable(device_value);
            self.device = device_value;
        }

        pub fn retarget(self: *Self, device_value: array_mod.Device) array_mod.ArrayError!void {
            try self.setDevice(device_value);
        }

        pub fn to(self: Self, device_value: array_mod.Device) array_mod.ArrayError!Self {
            try requireDeviceAvailable(device_value);
            var cloned = try self.clone();
            cloned.device = device_value;
            return cloned;
        }

        pub fn withDevice(self: Self, device_value: array_mod.Device) array_mod.ArrayError!Self {
            return self.to(device_value);
        }

        pub fn cpu(self: Self) array_mod.ArrayError!Self {
            return self.to(.cpu);
        }

        pub fn cuda(self: Self, index: usize) array_mod.ArrayError!Self {
            return self.to(array_mod.Device.cuda(index));
        }

        pub fn mps(self: Self, index: usize) array_mod.ArrayError!Self {
            return self.to(array_mod.Device.mps(index));
        }

        pub fn deviceValue(self: Self) array_mod.Device {
            return self.device;
        }

        pub fn deviceBackend(self: Self) array_mod.Backend {
            return self.device.backend;
        }

        pub fn deviceBackendName(self: Self) []const u8 {
            return self.device.backendName();
        }

        pub fn deviceIndex(self: Self) usize {
            return self.device.index;
        }

        pub fn isCpu(self: Self) bool {
            return self.device.isCpu();
        }

        pub fn isHostBacked(self: Self) bool {
            return self.isCpu();
        }

        pub fn isCuda(self: Self) bool {
            return self.device.isCuda();
        }

        pub fn isCudaBacked(self: Self) bool {
            return self.isCuda();
        }

        pub fn isMps(self: Self) bool {
            return self.device.isMps();
        }

        pub fn isMpsBacked(self: Self) bool {
            return self.isMps();
        }

        pub fn isAcceleratorBacked(self: Self) bool {
            return self.isCudaBacked() or self.isMpsBacked();
        }

        pub fn isRemoteBacked(self: Self) bool {
            return self.isAcceleratorBacked();
        }

        pub fn isDeviceBacked(self: Self) bool {
            return !self.isCpu();
        }

        pub fn isDeviceAvailable(self: Self) bool {
            return self.device.isAvailable();
        }

        pub fn sameDevice(self: Self, other: Self) bool {
            return self.device.sameDevice(other.device);
        }

        pub fn sourceNbytes(self: Self) usize {
            return self.bytes.len;
        }

        pub fn sourcePtr(self: Self) u64 {
            if (self.bytes.len == 0) return 0;
            return @intFromPtr(self.bytes.ptr);
        }

        pub fn dataPtr(self: Self) u64 {
            return self.sourcePtr();
        }

        pub fn hasSourcePtr(self: Self) bool {
            return self.sourcePtr() != 0;
        }

        pub fn sourceEndPtr(self: Self) u64 {
            return self.sourcePtr() + self.sourceNbytes();
        }

        pub fn sourceRange(self: Self) SourceRange {
            return .{
                .ptr = self.sourcePtr(),
                .nbytes = self.sourceNbytes(),
            };
        }

        pub fn sharesSource(self: Self, other: Self) bool {
            return self.sourceRange().sameRange(other.sourceRange());
        }

        pub fn sameSource(self: Self, other: Self) bool {
            return self.sharesSource(other);
        }

        pub fn sharesStorage(self: Self, other: Self) bool {
            return self.sharesSource(other);
        }

        pub fn sameStorage(self: Self, other: Self) bool {
            return self.sharesSource(other);
        }

        pub fn sourceMayOverlap(self: Self, other: Self) bool {
            return self.sourceRange().mayOverlap(other.sourceRange());
        }

        pub fn mayOverlap(self: Self, other: Self) bool {
            return self.sourceMayOverlap(other);
        }

        pub fn sourceByteCount(self: Self) usize {
            return self.sourceNbytes();
        }

        pub fn nbytes(self: Self) usize {
            return self.sourceNbytes();
        }

        pub fn byteCount(self: Self) usize {
            return self.sourceNbytes();
        }

        pub fn isEmpty(self: Self) bool {
            return self.bytes.len == 0;
        }

        pub fn isNonEmpty(self: Self) bool {
            return !self.isEmpty();
        }

        pub fn hasBytes(self: Self) bool {
            return !self.isEmpty();
        }

        // Scan pushdown state owns cloned column-name bytes on the host even
        // though the scan target may be CUDA/MPS.  These helpers intentionally
        // count only heap-owned payloads (the projection slice table plus
        // duplicated name strings), not inline option/enum fields embedded in
        // the scan struct itself.
        pub fn projectionMetadataNbytes(self: Self) usize {
            const names = self.projection orelse return 0;
            var total = names.len * @sizeOf([]const u8);
            for (names) |name| total += name.len;
            return total;
        }

        pub fn rangePredicateMetadataNbytes(self: Self) usize {
            return if (self.range_predicate) |predicate| predicate.column.len else 0;
        }

        pub fn nullPredicateMetadataNbytes(self: Self) usize {
            return if (self.null_predicate) |predicate| predicate.column.len else 0;
        }

        pub fn predicateMetadataNbytes(self: Self) usize {
            return self.rangePredicateMetadataNbytes() + self.nullPredicateMetadataNbytes();
        }

        pub fn pushdownMetadataNbytes(self: Self) usize {
            return self.projectionMetadataNbytes() + self.predicateMetadataNbytes();
        }

        pub fn ownedNbytes(self: Self) usize {
            return self.sourceNbytes() + self.pushdownMetadataNbytes();
        }

        pub fn memoryUsage(self: Self) usize {
            return self.ownedNbytes();
        }

        pub fn estimatedSize(self: Self) usize {
            return self.ownedNbytes();
        }

        pub fn parquetFileSummary(self: Self) ParquetInteropError!DeviceParquetFileSummary {
            const summary_value = try boltha.parquet.readFileSummary(self.allocator, self.bytes);
            return scan_metadata_mod.bolthaFileSummaryToDeviceSummary(summary_value);
        }

        pub fn rowCount(self: Self) ParquetInteropError!usize {
            return (try self.parquetFileSummary()).rowCount();
        }

        pub fn nRows(self: Self) ParquetInteropError!usize {
            return self.rowCount();
        }

        pub fn rowGroupCount(self: Self) ParquetInteropError!usize {
            return (try self.parquetFileSummary()).rowGroupCount();
        }

        pub fn parquetColumnChunkCount(self: Self) ParquetInteropError!usize {
            return (try self.parquetFileSummary()).columnChunkCount();
        }

        pub fn columnCount(self: Self) ParquetInteropError!usize {
            return self.arrowFieldCount();
        }

        pub fn width(self: Self) ParquetInteropError!usize {
            return self.columnCount();
        }

        pub fn cols(self: Self) ParquetInteropError!usize {
            return self.columnCount();
        }

        pub fn nCols(self: Self) ParquetInteropError!usize {
            return self.columnCount();
        }

        pub fn cellCount(self: Self) ParquetInteropError!usize {
            return (try self.rowCount()) * (try self.columnCount());
        }

        pub fn shape(self: Self) ParquetInteropError!struct { rows: usize, cols: usize } {
            return .{
                .rows = try self.rowCount(),
                .cols = try self.columnCount(),
            };
        }

        pub fn hasRows(self: Self) bool {
            return (self.rowCount() catch 0) != 0;
        }

        pub fn hasColumns(self: Self) bool {
            return (self.columnCount() catch 0) != 0;
        }

        pub fn hasShape(self: Self, rows: usize, columns: usize) bool {
            const current = self.shape() catch return false;
            return current.rows == rows and current.cols == columns;
        }

        pub fn sameHeight(self: Self, other: Self) bool {
            return (self.rowCount() catch return false) == (other.rowCount() catch return false);
        }

        pub fn sameWidth(self: Self, other: Self) bool {
            return (self.columnCount() catch return false) == (other.columnCount() catch return false);
        }

        pub fn sameShape(self: Self, other: Self) bool {
            return self.sameHeight(other) and self.sameWidth(other);
        }

        pub fn shapeEquals(self: Self, rows: usize, columns: usize) bool {
            return self.hasShape(rows, columns);
        }

        pub fn sameRowGroups(self: Self, other: Self) bool {
            return (self.rowGroupCount() catch return false) == (other.rowGroupCount() catch return false);
        }

        pub fn parquetTotalNbytes(self: Self) ParquetInteropError!usize {
            return (try self.parquetFileSummary()).totalNbytes();
        }

        pub fn parquetTotalCompressedNbytes(self: Self) ParquetInteropError!usize {
            return (try self.parquetFileSummary()).totalCompressedNbytes();
        }

        pub fn parquetTotalUncompressedNbytes(self: Self) ParquetInteropError!usize {
            return (try self.parquetFileSummary()).totalUncompressedNbytes();
        }

        pub fn parquetCompressionRatio(self: Self) ParquetInteropError!f64 {
            return (try self.parquetFileSummary()).compressionRatio();
        }

        pub fn parquetMetadataCoverageRatio(self: Self) ParquetInteropError!f64 {
            return (try self.parquetFileSummary()).metadataCoverageRatio();
        }

        pub fn parquetPageIndexCoverageRatio(self: Self) ParquetInteropError!f64 {
            return (try self.parquetFileSummary()).pageIndexCoverageRatio();
        }

        pub fn hasRowGroups(self: Self) bool {
            const summary_value = self.parquetFileSummary() catch return false;
            return summary_value.hasRowGroups();
        }

        pub fn hasProjection(self: Self) bool {
            return self.projection != null;
        }

        pub fn projectionColumnCount(self: Self) usize {
            return if (self.projection) |names| names.len else 0;
        }

        pub fn projectionNames(self: Self) []const []const u8 {
            return if (self.projection) |names| names else &.{};
        }

        pub fn projectionNameAt(self: Self, index: usize) ?[]const u8 {
            const names = self.projection orelse return null;
            if (index >= names.len) return null;
            return names[index];
        }

        pub fn projectionIndex(self: Self, name: []const u8) ?usize {
            const names = self.projection orelse return null;
            for (names, 0..) |candidate, index| {
                if (std.mem.eql(u8, candidate, name)) return index;
            }
            return null;
        }

        pub fn projectionContains(self: Self, name: []const u8) bool {
            return self.projectionIndex(name) != null;
        }

        pub fn projectionNamesUnique(self: Self) bool {
            const names = self.projection orelse return true;
            for (names, 0..) |name, index| {
                if (names_mod.nameInBorrowedList(name, names[0..index])) return false;
            }
            return true;
        }

        pub fn hasDuplicateProjectionNames(self: Self) bool {
            return !self.projectionNamesUnique();
        }

        pub fn duplicateProjectionNameCount(self: Self) usize {
            const names = self.projection orelse return 0;
            var count: usize = 0;
            for (names, 0..) |name, index| {
                if (names_mod.nameInBorrowedList(name, names[0..index])) count += 1;
            }
            return count;
        }

        pub fn hasAllProjectionNames(self: Self, names: []const []const u8) bool {
            for (names) |name| {
                if (!self.projectionContains(name)) return false;
            }
            return true;
        }

        pub fn hasAnyProjectionName(self: Self, names: []const []const u8) bool {
            for (names) |name| {
                if (self.projectionContains(name)) return true;
            }
            return false;
        }

        pub fn projectsColumn(self: Self, name: []const u8) bool {
            return self.projection == null or self.projectionContains(name);
        }

        pub fn hasPredicate(self: Self) bool {
            return self.hasRangePredicate() or self.hasNullPredicate();
        }

        pub fn predicateColumn(self: Self) ?[]const u8 {
            if (self.range_predicate) |predicate| return predicate.column;
            if (self.null_predicate) |predicate| return predicate.column;
            return null;
        }

        pub fn hasPredicateFor(self: Self, column: []const u8) bool {
            const active_column = self.predicateColumn() orelse return false;
            return std.mem.eql(u8, active_column, column);
        }

        pub fn hasRangePredicate(self: Self) bool {
            return self.range_predicate != null;
        }

        pub fn rangePredicateColumn(self: Self) ?[]const u8 {
            return if (self.range_predicate) |predicate| predicate.column else null;
        }

        pub fn rangePredicate(self: Self) ?ParquetRangePredicate {
            return if (self.range_predicate) |predicate| predicate.predicate else null;
        }

        pub fn rangePredicateDType(self: Self) ?array_mod.DType {
            const predicate = self.rangePredicate() orelse return null;
            return std.meta.activeTag(predicate);
        }

        pub fn hasRangePredicateFor(self: Self, column: []const u8) bool {
            const active_column = self.rangePredicateColumn() orelse return false;
            return std.mem.eql(u8, active_column, column);
        }

        pub fn hasNullPredicate(self: Self) bool {
            return self.null_predicate != null;
        }

        pub fn nullPredicateColumn(self: Self) ?[]const u8 {
            return if (self.null_predicate) |predicate| predicate.column else null;
        }

        pub fn nullPredicateWantNulls(self: Self) ?bool {
            return if (self.null_predicate) |predicate| predicate.want_nulls else null;
        }

        pub fn hasNullPredicateFor(self: Self, column: []const u8) bool {
            const active_column = self.nullPredicateColumn() orelse return false;
            return std.mem.eql(u8, active_column, column);
        }

        pub fn hasPushdown(self: Self) bool {
            return self.hasProjection() or self.hasRangePredicate() or self.hasNullPredicate();
        }

        pub fn validateProjection(self: Self) ParquetInteropError!void {
            if (self.projection) |names| {
                if (!self.projectionNamesUnique()) return error.ColumnNotFound;
                var schema = try self.toArrowSchemaProjection(self.allocator, names);
                defer schema.deinit(self.allocator);
            }
        }

        pub fn validatePredicate(self: Self) ParquetInteropError!void {
            const column = self.predicateColumn() orelse return;
            var full_schema = try boltha.parquet.readSchema(self.allocator, self.bytes);
            defer full_schema.deinit(self.allocator);
            const field = full_schema.fieldByName(column) orelse return error.ColumnNotFound;
            if (self.rangePredicateDType()) |predicate_dtype| {
                const predicate = self.rangePredicate().?;
                if (!scan_metadata_mod.rangePredicateBoundsValid(predicate)) return error.TypeMismatch;
                const field_dtype = try scan_metadata_mod.deviceDTypeFromArrowField(field.*);
                if (field_dtype != predicate_dtype) return error.TypeMismatch;
            }
            if (self.hasNullPredicate() and !field.nullable) return error.TypeMismatch;
        }

        pub fn validatePushdown(self: Self) ParquetInteropError!void {
            try self.validateProjection();
            try self.validatePredicate();
        }

        pub fn pushdownValid(self: Self) bool {
            self.validatePushdown() catch return false;
            return true;
        }

        pub fn validateCollect(self: Self) ParquetInteropError!void {
            try requireDeviceAvailable(self.device);
            try self.validatePushdown();
        }

        pub fn collectValid(self: Self) bool {
            self.validateCollect() catch return false;
            return true;
        }

        pub fn pushdownSummary(self: Self) DeviceParquetScanPushdownSummary {
            return .{
                .has_projection = self.hasProjection(),
                .projection_count = self.projectionColumnCount(),
                .projection_names = self.projectionNames(),
                .has_range_predicate = self.hasRangePredicate(),
                .range_predicate_column = self.rangePredicateColumn(),
                .range_predicate_dtype = self.rangePredicateDType(),
                .has_null_predicate = self.hasNullPredicate(),
                .null_predicate_column = self.nullPredicateColumn(),
                .null_predicate_want_nulls = self.nullPredicateWantNulls(),
                .projection_metadata_nbytes = self.projectionMetadataNbytes(),
                .range_predicate_metadata_nbytes = self.rangePredicateMetadataNbytes(),
                .null_predicate_metadata_nbytes = self.nullPredicateMetadataNbytes(),
                .predicate_metadata_nbytes = self.predicateMetadataNbytes(),
                .pushdown_metadata_nbytes = self.pushdownMetadataNbytes(),
            };
        }

        pub fn summary(self: Self) DeviceParquetScanSummary {
            return .{
                .device = self.deviceValue(),
                .source_ptr = self.sourcePtr(),
                .source_nbytes = self.sourceNbytes(),
                .owned_nbytes = self.ownedNbytes(),
                .pushdown = self.pushdownSummary(),
            };
        }

        pub const toArrowSchema = scan_metadata_mod.toArrowSchema;
        pub const toArrowSchemaProjection = scan_metadata_mod.toArrowSchemaProjection;
        pub const toArrowFields = scan_metadata_mod.toArrowFields;
        pub const toArrowFieldsProjection = scan_metadata_mod.toArrowFieldsProjection;
        pub const arrowFieldCount = scan_metadata_mod.arrowFieldCount;
        pub const arrowFieldNameAt = scan_metadata_mod.arrowFieldNameAt;
        pub const arrowFieldNames = scan_metadata_mod.arrowFieldNames;
        pub const arrowFieldIndex = scan_metadata_mod.arrowFieldIndex;
        pub const hasArrowField = scan_metadata_mod.hasArrowField;
        pub const hasAllArrowFields = scan_metadata_mod.hasAllArrowFields;
        pub const hasAnyArrowField = scan_metadata_mod.hasAnyArrowField;
        pub const arrowFieldDTypeAt = scan_metadata_mod.arrowFieldDTypeAt;
        pub const arrowFieldDType = scan_metadata_mod.arrowFieldDType;
        pub const arrowFieldDTypes = scan_metadata_mod.arrowFieldDTypes;
        pub const arrowFieldDTypeNames = scan_metadata_mod.arrowFieldDTypeNames;
        pub const arrowFieldDTypeByteSizes = scan_metadata_mod.arrowFieldDTypeByteSizes;
        pub const arrowFieldDTypeBitSizes = scan_metadata_mod.arrowFieldDTypeBitSizes;
        pub const arrowFieldDTypeClassMask = scan_metadata_mod.arrowFieldDTypeClassMask;
        pub const arrowFieldDTypeClassCount = scan_metadata_mod.arrowFieldDTypeClassCount;

        pub fn numericArrowFieldCount(self: Self) ParquetInteropError!usize {
            return self.arrowFieldDTypeClassCount(.numeric);
        }

        pub fn floatArrowFieldCount(self: Self) ParquetInteropError!usize {
            return self.arrowFieldDTypeClassCount(.float);
        }

        pub fn integerArrowFieldCount(self: Self) ParquetInteropError!usize {
            return self.arrowFieldDTypeClassCount(.integer);
        }

        pub fn boolArrowFieldCount(self: Self) ParquetInteropError!usize {
            return self.arrowFieldDTypeClassCount(.bool);
        }

        pub const arrowFieldNullableAt = scan_metadata_mod.arrowFieldNullableAt;
        pub const arrowFieldNullable = scan_metadata_mod.arrowFieldNullable;
        pub const arrowFieldNullableMask = scan_metadata_mod.arrowFieldNullableMask;
        pub const nullableArrowFieldCount = scan_metadata_mod.nullableArrowFieldCount;

        pub fn nonNullableArrowFieldCount(self: Self) ParquetInteropError!usize {
            return (try self.arrowFieldCount()) - try self.nullableArrowFieldCount();
        }

        pub fn hasNullableArrowFields(self: Self) bool {
            return (self.nullableArrowFieldCount() catch 0) != 0;
        }

        pub const allArrowFieldsNullable = scan_metadata_mod.allArrowFieldsNullable;
        pub const hasArrowProjection = scan_metadata_mod.hasArrowProjection;

        pub const arrowColumnSchemaAt = scan_metadata_mod.arrowColumnSchemaAt;
        pub const arrowColumnSchema = scan_metadata_mod.arrowColumnSchema;
        pub const arrowColumnSchemas = scan_metadata_mod.arrowColumnSchemas;
        pub const arrowSchemaSummary = scan_metadata_mod.arrowSchemaSummary;
        pub const arrowSchemaEquals = scan_metadata_mod.arrowSchemaEquals;
        pub const arrowSameSchema = scan_metadata_mod.arrowSameSchema;
        pub const arrowSchemaCompatible = scan_metadata_mod.arrowSchemaCompatible;
        pub const arrowSchemaEqualsSchemas = scan_metadata_mod.arrowSchemaEqualsSchemas;

        pub fn clearProjection(self: *Self) void {
            if (self.projection) |names| {
                freeNameList(self.allocator, names);
                self.projection = null;
            }
        }

        pub fn clearRangePredicate(self: *Self) void {
            if (self.range_predicate) |predicate| {
                self.allocator.free(predicate.column);
                self.range_predicate = null;
            }
        }

        pub fn clearNullPredicate(self: *Self) void {
            if (self.null_predicate) |predicate| {
                self.allocator.free(predicate.column);
                self.null_predicate = null;
            }
        }

        pub fn clearPredicate(self: *Self) void {
            self.clearRangePredicate();
            self.clearNullPredicate();
        }

        pub fn clearPushdown(self: *Self) void {
            self.clearProjection();
            self.clearPredicate();
        }

        pub fn resetPushdown(self: *Self) void {
            self.clearPushdown();
        }

        pub fn select(self: *Self, names: []const []const u8) std.mem.Allocator.Error!void {
            self.clearProjection();
            self.projection = try cloneNameList(self.allocator, names);
        }

        pub fn appendSelect(self: *Self, names: []const []const u8) std.mem.Allocator.Error!void {
            var builder: std.ArrayList([]const u8) = .empty;
            defer builder.deinit(self.allocator);
            if (self.projection) |current| {
                for (current) |name| try names_mod.appendOwnedNameUnique(self.allocator, &builder, name);
            }
            for (names) |name| try names_mod.appendOwnedNameUnique(self.allocator, &builder, name);
            const merged = try builder.toOwnedSlice(self.allocator);
            builder = .empty;
            self.clearProjection();
            self.projection = merged;
        }

        pub fn dropSelected(self: *Self, names: []const []const u8) std.mem.Allocator.Error!void {
            const current = self.projection orelse return;
            var builder: std.ArrayList([]const u8) = .empty;
            defer builder.deinit(self.allocator);
            for (current) |name| {
                if (!names_mod.nameInBorrowedList(name, names)) {
                    try names_mod.appendOwnedNameUnique(self.allocator, &builder, name);
                }
            }
            const kept = try builder.toOwnedSlice(self.allocator);
            builder = .empty;
            self.clearProjection();
            self.projection = kept;
        }

        pub fn selectAll(self: *Self) void {
            self.clearProjection();
        }

        pub fn selectExcept(self: *Self, names: []const []const u8) ParquetInteropError!void {
            const all_names = try self.arrowFieldNames(self.allocator);
            defer {
                for (all_names) |name| self.allocator.free(name);
                self.allocator.free(all_names);
            }
            var builder: std.ArrayList([]const u8) = .empty;
            defer builder.deinit(self.allocator);
            for (all_names) |name| {
                if (!names_mod.nameInBorrowedList(name, names)) {
                    try names_mod.appendOwnedNameUnique(self.allocator, &builder, name);
                }
            }
            const kept = try builder.toOwnedSlice(self.allocator);
            builder = .empty;
            self.clearProjection();
            self.projection = kept;
        }

        pub fn intersectSelect(self: *Self, names: []const []const u8) std.mem.Allocator.Error!void {
            const current = self.projection orelse {
                try self.select(names);
                return;
            };
            var builder: std.ArrayList([]const u8) = .empty;
            defer builder.deinit(self.allocator);
            for (current) |name| {
                if (names_mod.nameInBorrowedList(name, names)) {
                    try names_mod.appendOwnedNameUnique(self.allocator, &builder, name);
                }
            }
            const kept = try builder.toOwnedSlice(self.allocator);
            builder = .empty;
            self.clearProjection();
            self.projection = kept;
        }

        pub fn whereRange(self: *Self, column: []const u8, predicate: ParquetRangePredicate) std.mem.Allocator.Error!void {
            self.clearRangePredicate();
            self.clearNullPredicate();
            self.range_predicate = .{
                .column = try self.allocator.dupe(u8, column),
                .predicate = predicate,
            };
        }

        pub fn whereMin(self: *Self, column: []const u8, comptime T: type, min_value: T) std.mem.Allocator.Error!void {
            const tag = comptime array_mod.DType.of(T);
            try self.whereRange(column, @unionInit(ParquetRangePredicate, @tagName(tag), .{ .min = min_value }));
        }

        pub fn whereMax(self: *Self, column: []const u8, comptime T: type, max_value: T) std.mem.Allocator.Error!void {
            const tag = comptime array_mod.DType.of(T);
            try self.whereRange(column, @unionInit(ParquetRangePredicate, @tagName(tag), .{ .max = max_value }));
        }

        pub fn whereBetween(self: *Self, column: []const u8, comptime T: type, min_value: T, max_value: T) std.mem.Allocator.Error!void {
            const tag = comptime array_mod.DType.of(T);
            try self.whereRange(column, @unionInit(ParquetRangePredicate, @tagName(tag), .{ .min = min_value, .max = max_value }));
        }

        pub fn whereGe(self: *Self, column: []const u8, comptime T: type, value: T) std.mem.Allocator.Error!void {
            try self.whereMin(column, T, value);
        }

        pub fn whereLe(self: *Self, column: []const u8, comptime T: type, value: T) std.mem.Allocator.Error!void {
            try self.whereMax(column, T, value);
        }

        /// Alias for `whereGe`; Parquet statistics currently expose inclusive
        /// row-group bounds, so strict comparisons are conservatively lowered
        /// to inclusive pruning predicates and exact filtering remains a later
        /// dataframe/lazy execution step.
        pub fn whereGt(self: *Self, column: []const u8, comptime T: type, value: T) std.mem.Allocator.Error!void {
            try self.whereGe(column, T, value);
        }

        /// Alias for `whereLe`; see `whereGt` for the inclusive-bound rationale.
        pub fn whereLt(self: *Self, column: []const u8, comptime T: type, value: T) std.mem.Allocator.Error!void {
            try self.whereLe(column, T, value);
        }

        pub fn whereEq(self: *Self, column: []const u8, comptime T: type, value: T) std.mem.Allocator.Error!void {
            try self.whereBetween(column, T, value, value);
        }

        pub fn whereNull(self: *Self, column: []const u8, want_nulls: bool) std.mem.Allocator.Error!void {
            self.clearNullPredicate();
            self.clearRangePredicate();
            self.null_predicate = .{
                .column = try self.allocator.dupe(u8, column),
                .want_nulls = want_nulls,
            };
        }

        pub fn whereIsNull(self: *Self, column: []const u8) std.mem.Allocator.Error!void {
            try self.whereNull(column, true);
        }

        pub fn whereIsNotNull(self: *Self, column: []const u8) std.mem.Allocator.Error!void {
            try self.whereNull(column, false);
        }

        pub fn whereNotNull(self: *Self, column: []const u8) std.mem.Allocator.Error!void {
            try self.whereIsNotNull(column);
        }

        pub fn collect(self: Self) ParquetInteropError!DeviceDataFrame {
            try self.validateCollect();
            var table = if (self.range_predicate) |predicate|
                try dataframe_arrow_mod.readBolthaTableWithRangePruning(self.allocator, self.bytes, predicate.column, predicate.predicate)
            else if (self.null_predicate) |predicate|
                if (predicate.want_nulls)
                    try boltha.parquet.readTableWithNullPruning(self.allocator, self.bytes, predicate.column)
                else
                    try boltha.parquet.readTableWithNonNullPruning(self.allocator, self.bytes, predicate.column)
            else
                try boltha.parquet.readTable(self.allocator, self.bytes);
            defer table.deinit(self.allocator);

            if (self.projection) |names| {
                return dataframe_arrow_mod.fromArrowTableProjection(DeviceDataFrame, DeviceColumnDef, DeviceColumn, self.allocator, table, names, self.device);
            }
            return dataframe_arrow_mod.fromArrowTable(DeviceDataFrame, DeviceColumnDef, DeviceColumn, self.allocator, table, self.device);
        }

        pub fn explain(self: Self, allocator: std.mem.Allocator) (std.mem.Allocator.Error || std.Io.Writer.Error)![]u8 {
            var aw: std.Io.Writer.Allocating = .init(allocator);
            errdefer aw.deinit();
            try aw.writer.print("DeviceParquetScan(bytes={d}, device={s}", .{ self.bytes.len, self.device.backendName() });
            try aw.writer.print(", pushdown=", .{});
            try lazy_format_mod.formatLazyScanPushdown(&aw.writer, self);
            try aw.writer.print(")\n", .{});
            return aw.toOwnedSlice();
        }

        pub fn explainSummary(self: Self, allocator: std.mem.Allocator) (ParquetInteropError || std.Io.Writer.Error)![]u8 {
            const file_summary = try self.parquetFileSummary();
            var aw: std.Io.Writer.Allocating = .init(allocator);
            errdefer aw.deinit();
            try aw.writer.print(
                "DeviceParquetScanSummary(bytes={d}, rows={d}, cols={d}, row_groups={d}, device={s}, projection={d}, predicate={s}, valid={})\n",
                .{
                    self.sourceNbytes(),
                    file_summary.rowCount(),
                    try self.columnCount(),
                    file_summary.rowGroupCount(),
                    self.deviceBackendName(),
                    self.projectionColumnCount(),
                    self.predicateColumn() orelse "none",
                    self.collectValid(),
                },
            );
            return aw.toOwnedSlice();
        }
    };
}
