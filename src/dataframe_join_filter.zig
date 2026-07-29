//! Semi/anti/asof eager join implementations.

const std = @import("std");
const array_helpers = @import("dataframe_array.zig");
const join_concat_mod = @import("dataframe_join_concat.zig");
const join_indices_mod = @import("dataframe_join_indices.zig");
const join_validation = @import("dataframe_join_validation.zig");
const keys_mod = @import("dataframe_keys.zig");

const concatJoinedTables = join_concat_mod.concatJoinedTables;
const semiAntiJoinRowIndices = join_indices_mod.semiAntiJoinRowIndices;
const semiAntiJoinRowIndicesMulti = join_indices_mod.semiAntiJoinRowIndicesMulti;
const validateSingleJoinKeys = join_validation.validateSingleJoinKeys;
const validateMultiJoinKeys = join_validation.validateMultiJoinKeys;

fn semiAntiJoinIndices(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_name: []const u8,
    right_key_name: []const u8,
    keep_matches: bool,
) keys_mod.KeyMatchError![]usize {
    try validateSingleJoinKeys(left, right, left_key_name, right_key_name);
    const left_key = try left.column(left_key_name);
    const right_key = try right.column(right_key_name);
    return semiAntiJoinRowIndices(allocator, left_key.*, right_key.*, keep_matches);
}

fn semiAntiJoinIndicesOn(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    keep_matches: bool,
) keys_mod.KeyMatchError![]usize {
    try validateMultiJoinKeys(left, right, left_key_names, right_key_names);
    return semiAntiJoinRowIndicesMulti(allocator, left, right, left_key_names, right_key_names, keep_matches);
}

pub fn semiJoin(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_name: []const u8,
    right_key_name: []const u8,
) keys_mod.KeyMatchError!DeviceDataFrame {
    const indices = try semiAntiJoinIndices(left.allocator, left, right, left_key_name, right_key_name, true);
    defer left.allocator.free(indices);
    return left.take(indices);
}

pub fn semiJoinOn(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!DeviceDataFrame {
    const indices = try semiAntiJoinIndicesOn(left.allocator, left, right, left_key_names, right_key_names, true);
    defer left.allocator.free(indices);
    return left.take(indices);
}

pub fn antiJoin(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_name: []const u8,
    right_key_name: []const u8,
) keys_mod.KeyMatchError!DeviceDataFrame {
    const indices = try semiAntiJoinIndices(left.allocator, left, right, left_key_name, right_key_name, false);
    defer left.allocator.free(indices);
    return left.take(indices);
}

pub fn antiJoinOn(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!DeviceDataFrame {
    const indices = try semiAntiJoinIndicesOn(left.allocator, left, right, left_key_names, right_key_names, false);
    defer left.allocator.free(indices);
    return left.take(indices);
}

pub fn asofJoin(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: anytype,
) keys_mod.KeyMatchError!DeviceDataFrame {
    try validateSingleJoinKeys(left, right, left_key_name, right_key_name);
    const left_key = try left.column(left_key_name);
    const right_key = try right.column(right_key_name);

    const right_indices = try keys_mod.asofRightRowIndices(left.allocator, left_key.*, right_key.*, options_value.strategy);
    defer left.allocator.free(right_indices);
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, right_indices);
    defer right_rows.deinit();

    return concatJoinedTables(DeviceDataFrame, left.allocator, left, right_rows, right_key_name, .{ .right_suffix = options_value.right_suffix });
}
