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
const series_mod = @import("../../series.zig");
const boltha = @import("boltha");

const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const DeviceParquetNullFilter = options_mod.DeviceParquetNullFilter;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
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
            self.allocator.free(self.bytes);
            if (self.projection) |names| freeNameList(self.allocator, names);
            if (self.range_predicate) |predicate| self.allocator.free(predicate.column);
            if (self.null_predicate) |predicate| self.allocator.free(predicate.column);
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

        pub fn hasProjection(self: Self) bool {
            return self.projection != null;
        }

        pub fn projectionColumnCount(self: Self) usize {
            return if (self.projection) |names| names.len else 0;
        }

        pub fn projectionNames(self: Self) []const []const u8 {
            return if (self.projection) |names| names else &.{};
        }

        pub fn hasRangePredicate(self: Self) bool {
            return self.range_predicate != null;
        }

        pub fn rangePredicateColumn(self: Self) ?[]const u8 {
            return if (self.range_predicate) |predicate| predicate.column else null;
        }

        pub fn hasNullPredicate(self: Self) bool {
            return self.null_predicate != null;
        }

        pub fn nullPredicateColumn(self: Self) ?[]const u8 {
            return if (self.null_predicate) |predicate| predicate.column else null;
        }

        pub fn hasPushdown(self: Self) bool {
            return self.hasProjection() or self.hasRangePredicate() or self.hasNullPredicate();
        }

        pub fn select(self: *Self, names: []const []const u8) std.mem.Allocator.Error!void {
            if (self.projection) |old| freeNameList(self.allocator, old);
            self.projection = try cloneNameList(self.allocator, names);
        }

        pub fn whereRange(self: *Self, column: []const u8, predicate: ParquetRangePredicate) std.mem.Allocator.Error!void {
            if (self.range_predicate) |old| self.allocator.free(old.column);
            if (self.null_predicate) |old| {
                self.allocator.free(old.column);
                self.null_predicate = null;
            }
            self.range_predicate = .{
                .column = try self.allocator.dupe(u8, column),
                .predicate = predicate,
            };
        }

        pub fn whereNull(self: *Self, column: []const u8, want_nulls: bool) std.mem.Allocator.Error!void {
            if (self.null_predicate) |old| self.allocator.free(old.column);
            if (self.range_predicate) |old| {
                self.allocator.free(old.column);
                self.range_predicate = null;
            }
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
