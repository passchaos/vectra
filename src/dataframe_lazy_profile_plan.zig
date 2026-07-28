//! Lazy single-column profile operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const series_mod = @import("series.zig");

const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn appendNameOutputOptions(frame: anytype, comptime tag_name: []const u8, name: []const u8, output_prefix: []const u8, options: anytype) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, .{
        .name = owned_name,
        .output_prefix = owned_prefix,
        .options = options,
    }));
}

pub fn appendNameOutput(frame: anytype, comptime tag_name: []const u8, name: []const u8, output_prefix: []const u8) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    const LazyOp = std.meta.Elem(@TypeOf(frame.ops.items));
    try frame.ops.append(frame.allocator, @unionInit(LazyOp, tag_name, .{
        .name = owned_name,
        .output_prefix = owned_prefix,
    }));
}
