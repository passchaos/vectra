//! Lazy sort/rank/slicing operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceSortOptions = options_mod.DeviceSortOptions;
const DeviceRollingRankOptions = options_mod.DeviceRollingRankOptions;
const DeviceExpandingRankOptions = options_mod.DeviceExpandingRankOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;

pub fn sortBy(frame: anytype, name: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
    try frame.ops.append(frame.allocator, .{ .sort_by = .{
        .name = try frame.allocator.dupe(u8, name),
        .options = options_value,
    } });
}

pub fn sortByColumns(frame: anytype, names: []const []const u8, options_values: []const DeviceSortOptions) DeviceDataError!void {
    if (names.len != options_values.len) return error.LengthMismatch;
    const owned_names = try frame.allocator.alloc([]const u8, names.len);
    errdefer frame.allocator.free(owned_names);
    var initialized: usize = 0;
    errdefer {
        for (owned_names[0..initialized]) |name| frame.allocator.free(name);
    }
    for (names, owned_names) |name, *slot| {
        slot.* = try frame.allocator.dupe(u8, name);
        initialized += 1;
    }
    const owned_options = try frame.allocator.dupe(DeviceSortOptions, options_values);
    errdefer frame.allocator.free(owned_options);
    try frame.ops.append(frame.allocator, .{ .sort_by_columns = .{
        .names = owned_names,
        .options = owned_options,
    } });
}

pub fn rankProfileBy(frame: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceSortOptions) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .rank_profile_by = .{
        .name = owned_name,
        .output_prefix = owned_prefix,
        .options = options_value,
    } });
}

pub fn rollingRankProfile(frame: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceRollingRankOptions) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .rolling_rank_profile = .{
        .name = owned_name,
        .output_prefix = owned_prefix,
        .options = options_value,
    } });
}

pub fn expandingRankProfile(frame: anytype, name: []const u8, output_prefix: []const u8, options_value: DeviceExpandingRankOptions) DeviceDataError!void {
    const owned_name = try frame.allocator.dupe(u8, name);
    errdefer frame.allocator.free(owned_name);
    const owned_prefix = try frame.allocator.dupe(u8, output_prefix);
    errdefer frame.allocator.free(owned_prefix);
    try frame.ops.append(frame.allocator, .{ .expanding_rank_profile = .{
        .name = owned_name,
        .output_prefix = owned_prefix,
        .options = options_value,
    } });
}
