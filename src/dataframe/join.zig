const std = @import("std");
const array_mod = @import("../array.zig");
const array_helpers = @import("../dataframe_array.zig");
const keys_mod = @import("../dataframe_keys.zig");
const join_concat_mod = @import("join/concat.zig");
const join_indices_mod = @import("join/indices.zig");
const join_filter_mod = @import("join/filter.zig");
const join_validation = @import("join/validation.zig");
const names_mod = @import("../dataframe_names.zig");

pub const JoinConcatError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
    InvalidDevice,
};

pub const JoinRowIndexPair = join_indices_mod.JoinRowIndexPair;
pub const innerJoinRowIndices = join_indices_mod.innerJoinRowIndices;
pub const innerJoinRowIndicesMulti = join_indices_mod.innerJoinRowIndicesMulti;
pub const leftJoinRowIndices = join_indices_mod.leftJoinRowIndices;
pub const leftJoinRowIndicesMulti = join_indices_mod.leftJoinRowIndicesMulti;
pub const fullJoinRowIndices = join_indices_mod.fullJoinRowIndices;
pub const fullJoinRowIndicesMulti = join_indices_mod.fullJoinRowIndicesMulti;
pub const semiAntiJoinRowIndices = join_indices_mod.semiAntiJoinRowIndices;
pub const semiAntiJoinRowIndicesMulti = join_indices_mod.semiAntiJoinRowIndicesMulti;

pub const concatJoinedTables = join_concat_mod.concatJoinedTables;
pub const concatJoinedTablesExcludingKeys = join_concat_mod.concatJoinedTablesExcludingKeys;
pub const concatFullJoinedTables = join_concat_mod.concatFullJoinedTables;
pub const concatFullJoinedTablesOn = join_concat_mod.concatFullJoinedTablesOn;

const validateSingleJoinKeys = join_validation.validateSingleJoinKeys;
const validateMultiJoinKeys = join_validation.validateMultiJoinKeys;

/// Execute an eager single-key inner join for a concrete DeviceDataFrame type.
///
/// The public dataframe facade forwards here so row-index construction,
/// optional-row materialization, and output-column stitching stay colocated with
/// the rest of the join implementation.
pub fn innerJoin(
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

    var pair = try innerJoinRowIndices(left.allocator, left_key.*, right_key.*);
    defer pair.deinit();

    var left_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, left, pair.left);
    defer left_rows.deinit();
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, pair.right);
    defer right_rows.deinit();

    return concatJoinedTables(DeviceDataFrame, left.allocator, left_rows, right_rows, right_key_name, options_value);
}

pub fn innerJoinOn(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: anytype,
) keys_mod.KeyMatchError!DeviceDataFrame {
    try validateMultiJoinKeys(left, right, left_key_names, right_key_names);

    var pair = try innerJoinRowIndicesMulti(left.allocator, left, right, left_key_names, right_key_names);
    defer pair.deinit();

    var left_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, left, pair.left);
    defer left_rows.deinit();
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, pair.right);
    defer right_rows.deinit();

    return concatJoinedTablesExcludingKeys(DeviceDataFrame, left.allocator, left_rows, right_rows, right_key_names, options_value);
}

pub fn leftJoin(
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

    var pair = try leftJoinRowIndices(left.allocator, left_key.*, right_key.*);
    defer pair.deinit();

    var left_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, left, pair.left);
    defer left_rows.deinit();
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, pair.right);
    defer right_rows.deinit();

    return concatJoinedTables(DeviceDataFrame, left.allocator, left_rows, right_rows, right_key_name, options_value);
}

pub fn leftJoinOn(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: anytype,
) keys_mod.KeyMatchError!DeviceDataFrame {
    try validateMultiJoinKeys(left, right, left_key_names, right_key_names);

    var pair = try leftJoinRowIndicesMulti(left.allocator, left, right, left_key_names, right_key_names);
    defer pair.deinit();

    var left_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, left, pair.left);
    defer left_rows.deinit();
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, pair.right);
    defer right_rows.deinit();

    return concatJoinedTablesExcludingKeys(DeviceDataFrame, left.allocator, left_rows, right_rows, right_key_names, options_value);
}

pub fn fullJoin(
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

    var pair = try fullJoinRowIndices(left.allocator, left_key.*, right_key.*);
    defer pair.deinit();

    var left_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, left, pair.left);
    defer left_rows.deinit();
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, pair.right);
    defer right_rows.deinit();

    return concatFullJoinedTables(DeviceDataFrame, left.allocator, left_rows, right_rows, left_key_name, right_key_name, options_value);
}

pub fn fullJoinOn(
    comptime DeviceDataFrame: type,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: anytype,
) keys_mod.KeyMatchError!DeviceDataFrame {
    try validateMultiJoinKeys(left, right, left_key_names, right_key_names);

    var pair = try fullJoinRowIndicesMulti(left.allocator, left, right, left_key_names, right_key_names);
    defer pair.deinit();

    var left_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, left, pair.left);
    defer left_rows.deinit();
    var right_rows = try array_helpers.takeOptionalRows(DeviceDataFrame, right, pair.right);
    defer right_rows.deinit();

    return concatFullJoinedTablesOn(DeviceDataFrame, left.allocator, left_rows, right_rows, left_key_names, right_key_names, options_value);
}

pub const semiJoin = join_filter_mod.semiJoin;
pub const semiJoinOn = join_filter_mod.semiJoinOn;
pub const antiJoin = join_filter_mod.antiJoin;
pub const antiJoinOn = join_filter_mod.antiJoinOn;
pub const asofJoin = join_filter_mod.asofJoin;
