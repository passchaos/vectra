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
const scan_metadata_mod = @import("scan_metadata.zig");
const series_mod = @import("../../series.zig");
const boltha = @import("boltha");

const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
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

        pub fn toArrowSchema(self: Self, allocator: std.mem.Allocator) ParquetInteropError!boltha.arrow.Schema {
            var schema = try boltha.parquet.readSchema(allocator, self.bytes);
            errdefer schema.deinit(allocator);
            if (self.projection) |names| {
                var projected = try scan_metadata_mod.projectArrowSchemaByName(allocator, schema, names);
                errdefer projected.deinit(allocator);
                schema.deinit(allocator);
                return projected;
            }
            return schema;
        }

        pub fn toArrowSchemaProjection(self: Self, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError!boltha.arrow.Schema {
            var schema = try boltha.parquet.readSchema(allocator, self.bytes);
            defer schema.deinit(allocator);
            return scan_metadata_mod.projectArrowSchemaByName(allocator, schema, wanted_names);
        }

        pub fn toArrowFields(self: Self, allocator: std.mem.Allocator) ParquetInteropError![]boltha.arrow.Field {
            var schema = try self.toArrowSchema(allocator);
            defer schema.deinit(allocator);
            return scan_metadata_mod.cloneArrowFields(allocator, schema.fieldsView());
        }

        pub fn toArrowFieldsProjection(self: Self, allocator: std.mem.Allocator, wanted_names: []const []const u8) ParquetInteropError![]boltha.arrow.Field {
            var schema = try self.toArrowSchemaProjection(allocator, wanted_names);
            defer schema.deinit(allocator);
            return scan_metadata_mod.cloneArrowFields(allocator, schema.fieldsView());
        }

        pub fn arrowFieldCount(self: Self) ParquetInteropError!usize {
            var schema = try self.toArrowSchema(self.allocator);
            defer schema.deinit(self.allocator);
            return schema.fieldCount();
        }

        pub fn arrowFieldNameAt(self: Self, allocator: std.mem.Allocator, index: usize) ParquetInteropError!?[]const u8 {
            var schema = try self.toArrowSchema(allocator);
            defer schema.deinit(allocator);
            const field = schema.fieldAt(index) orelse return null;
            return try allocator.dupe(u8, field.name);
        }

        pub fn arrowFieldNames(self: Self, allocator: std.mem.Allocator) ParquetInteropError![][]const u8 {
            var schema = try self.toArrowSchema(allocator);
            defer schema.deinit(allocator);

            const names = try allocator.alloc([]const u8, schema.fields.len);
            var initialized: usize = 0;
            errdefer {
                for (names[0..initialized]) |name| allocator.free(name);
                allocator.free(names);
            }
            for (schema.fields, names) |field, *slot| {
                slot.* = try allocator.dupe(u8, field.name);
                initialized += 1;
            }
            return names;
        }

        pub fn arrowFieldIndex(self: Self, name: []const u8) ParquetInteropError!?usize {
            var schema = try self.toArrowSchema(self.allocator);
            defer schema.deinit(self.allocator);
            return schema.fieldIndexByName(name);
        }

        pub fn hasArrowField(self: Self, name: []const u8) bool {
            return (self.arrowFieldIndex(name) catch null) != null;
        }

        pub fn hasAllArrowFields(self: Self, wanted_names: []const []const u8) bool {
            for (wanted_names) |name| {
                if (!self.hasArrowField(name)) return false;
            }
            return true;
        }

        pub fn hasAnyArrowField(self: Self, wanted_names: []const []const u8) bool {
            for (wanted_names) |name| {
                if (self.hasArrowField(name)) return true;
            }
            return false;
        }

        pub fn arrowFieldDTypeAt(self: Self, index: usize) ParquetInteropError!?array_mod.DType {
            var schema = try self.toArrowSchema(self.allocator);
            defer schema.deinit(self.allocator);
            const field = schema.fieldAt(index) orelse return null;
            return try scan_metadata_mod.deviceDTypeFromArrowField(field.*);
        }

        pub fn arrowFieldDType(self: Self, name: []const u8) ParquetInteropError!array_mod.DType {
            var schema = try self.toArrowSchema(self.allocator);
            defer schema.deinit(self.allocator);
            const field = schema.fieldByName(name) orelse return error.ColumnNotFound;
            return try scan_metadata_mod.deviceDTypeFromArrowField(field.*);
        }

        pub fn arrowFieldDTypes(self: Self, allocator: std.mem.Allocator) ParquetInteropError![]array_mod.DType {
            var schema = try self.toArrowSchema(allocator);
            defer schema.deinit(allocator);
            const dtypes = try allocator.alloc(array_mod.DType, schema.fields.len);
            errdefer allocator.free(dtypes);
            for (schema.fields, dtypes) |field, *slot| slot.* = try scan_metadata_mod.deviceDTypeFromArrowField(field);
            return dtypes;
        }

        pub fn arrowFieldDTypeNames(self: Self, allocator: std.mem.Allocator) ParquetInteropError![][]const u8 {
            const dtypes = try self.arrowFieldDTypes(allocator);
            defer allocator.free(dtypes);
            const names = try allocator.alloc([]const u8, dtypes.len);
            for (dtypes, names) |dtype, *slot| slot.* = dtype.name();
            return names;
        }

        pub fn arrowFieldDTypeByteSizes(self: Self, allocator: std.mem.Allocator) ParquetInteropError![]usize {
            const dtypes = try self.arrowFieldDTypes(allocator);
            defer allocator.free(dtypes);
            const sizes = try allocator.alloc(usize, dtypes.len);
            for (dtypes, sizes) |dtype, *slot| slot.* = dtype.byteSize();
            return sizes;
        }

        pub fn arrowFieldDTypeBitSizes(self: Self, allocator: std.mem.Allocator) ParquetInteropError![]usize {
            const dtypes = try self.arrowFieldDTypes(allocator);
            defer allocator.free(dtypes);
            const sizes = try allocator.alloc(usize, dtypes.len);
            for (dtypes, sizes) |dtype, *slot| slot.* = dtype.bitSize();
            return sizes;
        }

        pub fn arrowFieldDTypeClassMask(self: Self, allocator: std.mem.Allocator, class: options_mod.DeviceDTypeClass) ParquetInteropError![]bool {
            const dtypes = try self.arrowFieldDTypes(allocator);
            defer allocator.free(dtypes);
            const mask = try allocator.alloc(bool, dtypes.len);
            for (dtypes, mask) |dtype, *slot| slot.* = class.matches(dtype);
            return mask;
        }

        pub fn arrowFieldDTypeClassCount(self: Self, class: options_mod.DeviceDTypeClass) ParquetInteropError!usize {
            const dtypes = try self.arrowFieldDTypes(self.allocator);
            defer self.allocator.free(dtypes);
            var count: usize = 0;
            for (dtypes) |dtype| {
                if (class.matches(dtype)) count += 1;
            }
            return count;
        }

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

        pub fn hasArrowProjection(self: Self, wanted_names: []const []const u8) bool {
            var schema = boltha.parquet.readSchema(self.allocator, self.bytes) catch return false;
            defer schema.deinit(self.allocator);
            for (wanted_names) |name| {
                if (schema.fieldIndexByName(name) == null) return false;
            }
            return true;
        }

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

        pub fn whereRange(self: *Self, column: []const u8, predicate: ParquetRangePredicate) std.mem.Allocator.Error!void {
            self.clearRangePredicate();
            self.clearNullPredicate();
            self.range_predicate = .{
                .column = try self.allocator.dupe(u8, column),
                .predicate = predicate,
            };
        }

        pub fn whereNull(self: *Self, column: []const u8, want_nulls: bool) std.mem.Allocator.Error!void {
            self.clearNullPredicate();
            self.clearRangePredicate();
            self.null_predicate = .{
                .column = try self.allocator.dupe(u8, column),
                .want_nulls = want_nulls,
            };
        }

        pub fn collect(self: Self) ParquetInteropError!DeviceDataFrame {
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
    };
}
