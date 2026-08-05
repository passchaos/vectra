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
