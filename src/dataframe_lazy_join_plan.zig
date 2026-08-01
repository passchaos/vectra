//! Lazy join/row-set operation builders for DeviceLazyFrame.

const std = @import("std");
const array_mod = @import("array.zig");
const lazy_op_mod = @import("dataframe_lazy_op.zig");
const names_mod = @import("dataframe_names.zig");
const options_mod = @import("dataframe_options.zig");
const series_mod = @import("series.zig");

const DeviceLazyJoinKind = lazy_op_mod.DeviceLazyJoinKind;
const DeviceJoinOptions = options_mod.DeviceJoinOptions;
const DeviceAsofOptions = options_mod.DeviceAsofOptions;
const DeviceDataError = series_mod.DataError || array_mod.ArrayError;
const cloneNameList = names_mod.cloneNameList;
const freeNameList = names_mod.freeNameList;

pub fn joinOn(
    frame: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    kind: DeviceLazyJoinKind,
    options_value: DeviceJoinOptions,
) DeviceDataError!void {
    if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
    var owned_right = try right.clone();
    errdefer owned_right.deinit();
    const owned_left_keys = try cloneNameList(frame.allocator, left_key_names);
    errdefer freeNameList(frame.allocator, owned_left_keys);
    const owned_right_keys = try cloneNameList(frame.allocator, right_key_names);
    errdefer freeNameList(frame.allocator, owned_right_keys);
    const owned_suffix = try frame.allocator.dupe(u8, options_value.right_suffix);
    errdefer frame.allocator.free(owned_suffix);
    try frame.ops.append(frame.allocator, .{ .join_on = .{
        .kind = kind,
        .right = owned_right,
        .left_key_names = owned_left_keys,
        .right_key_names = owned_right_keys,
        .options = .{ .right_suffix = owned_suffix },
    } });
}

pub fn asofJoin(
    frame: anytype,
    right: anytype,
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: DeviceAsofOptions,
) DeviceDataError!void {
    var owned_right = try right.clone();
    errdefer owned_right.deinit();
    const owned_left_key = try frame.allocator.dupe(u8, left_key_name);
    errdefer frame.allocator.free(owned_left_key);
    const owned_right_key = try frame.allocator.dupe(u8, right_key_name);
    errdefer frame.allocator.free(owned_right_key);
    const owned_suffix = try frame.allocator.dupe(u8, options_value.right_suffix);
    errdefer frame.allocator.free(owned_suffix);
    try frame.ops.append(frame.allocator, .{ .asof_join = .{
        .right = owned_right,
        .left_key_name = owned_left_key,
        .right_key_name = owned_right_key,
        .options = .{
            .strategy = options_value.strategy,
            .right_suffix = owned_suffix,
        },
    } });
}

pub fn concatRows(frame: anytype, right: anytype) DeviceDataError!void {
    var owned_right = try right.clone();
    errdefer owned_right.deinit();
    try frame.ops.append(frame.allocator, .{ .concat_rows = owned_right });
}

pub fn distinctOn(frame: anytype, key_names: []const []const u8) DeviceDataError!void {
    if (key_names.len == 0) return error.LengthMismatch;
    try frame.ops.append(frame.allocator, .{ .distinct_on = try cloneNameList(frame.allocator, key_names) });
}

pub fn distinctOnLast(frame: anytype, key_names: []const []const u8) DeviceDataError!void {
    if (key_names.len == 0) return error.LengthMismatch;
    try frame.ops.append(frame.allocator, .{ .distinct_on_last = try cloneNameList(frame.allocator, key_names) });
}
