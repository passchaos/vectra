const std = @import("std");
const array_mod = @import("array.zig");
const array_helpers = @import("dataframe_array.zig");
const keys_mod = @import("dataframe_keys.zig");
const names_mod = @import("dataframe_names.zig");

const coalesceJoinKeys = array_helpers.coalesceJoinKeys;
const initDeviceDataFrameFromOwnedColumns = array_helpers.initDeviceDataFrameFromOwnedColumns;
const leftKeyRightIndex = names_mod.leftKeyRightIndex;
const nameInBorrowedList = names_mod.nameInBorrowedList;
const nameNeedsSuffix = names_mod.nameNeedsSuffix;
const rightExcludedKeyCount = names_mod.rightExcludedKeyCount;
const rightKeyIndexInList = names_mod.rightKeyIndexInList;
const suffixedNameTemp = names_mod.suffixedNameTemp;

pub const JoinConcatError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
    InvalidDevice,
};

pub const JoinRowIndexPair = struct {
    allocator: std.mem.Allocator,
    left: []?usize,
    right: []?usize,

    pub fn deinit(self: *JoinRowIndexPair) void {
        self.allocator.free(self.left);
        self.allocator.free(self.right);
        self.* = undefined;
    }
};

pub fn innerJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.innerJoinRowIndices(JoinRowIndexPair, allocator, left, right);
}

pub fn innerJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.innerJoinRowIndicesMulti(JoinRowIndexPair, allocator, left, right, left_key_names, right_key_names);
}

pub fn leftJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.leftJoinRowIndices(JoinRowIndexPair, allocator, left, right);
}

pub fn leftJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.leftJoinRowIndicesMulti(JoinRowIndexPair, allocator, left, right, left_key_names, right_key_names);
}

pub fn fullJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.fullJoinRowIndices(JoinRowIndexPair, allocator, left, right);
}

pub fn fullJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!JoinRowIndexPair {
    return keys_mod.fullJoinRowIndicesMulti(JoinRowIndexPair, allocator, left, right, left_key_names, right_key_names);
}

pub fn semiAntiJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype, keep_matches: bool) keys_mod.KeyMatchError![]usize {
    return keys_mod.semiAntiJoinRowIndices(allocator, left, right, keep_matches);
}

pub fn semiAntiJoinRowIndicesMulti(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    keep_matches: bool,
) keys_mod.KeyMatchError![]usize {
    return keys_mod.semiAntiJoinRowIndicesMulti(allocator, left, right, left_key_names, right_key_names, keep_matches);
}

pub fn concatJoinedTables(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    right_key_name: []const u8,
    options_value: anytype,
) JoinConcatError!DeviceDataFrame {
    return concatJoinedTablesExcludingKeys(DeviceDataFrame, allocator, left, right, &.{right_key_name}, options_value);
}

pub fn concatJoinedTablesExcludingKeys(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    right_key_names: []const []const u8,
    options_value: anytype,
) JoinConcatError!DeviceDataFrame {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left.rows != right.rows) return error.LengthMismatch;

    const total_cols = left.columns.len + right.columns.len - rightExcludedKeyCount(right, right_key_names);
    var names = try allocator.alloc([]const u8, total_cols);
    defer allocator.free(names);
    var temporary_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (temporary_names.items) |name| allocator.free(name);
        temporary_names.deinit(allocator);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(left.columns));
    var columns = try allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (left.names, left.columns) |name, col| {
        names[initialized] = name;
        columns[initialized] = try col.clone();
        initialized += 1;
    }

    for (right.names, right.columns) |name, col| {
        if (nameInBorrowedList(name, right_key_names)) continue;
        if (nameNeedsSuffix(left, name)) {
            const suffixed = try suffixedNameTemp(allocator, name, options_value.right_suffix);
            errdefer allocator.free(suffixed);
            try temporary_names.append(allocator, suffixed);
            names[initialized] = suffixed;
        } else {
            names[initialized] = name;
        }
        columns[initialized] = try col.clone();
        initialized += 1;
    }

    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, names, columns, left.rows, left.device);
}

pub fn concatFullJoinedTables(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_name: []const u8,
    right_key_name: []const u8,
    options_value: anytype,
) JoinConcatError!DeviceDataFrame {
    return concatFullJoinedTablesOn(DeviceDataFrame, allocator, left, right, &.{left_key_name}, &.{right_key_name}, options_value);
}

pub fn concatFullJoinedTablesOn(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    left: DeviceDataFrame,
    right: DeviceDataFrame,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
    options_value: anytype,
) JoinConcatError!DeviceDataFrame {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left.rows != right.rows) return error.LengthMismatch;
    if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
    for (left_key_names, right_key_names) |left_name, right_name| {
        const left_key = try left.column(left_name);
        const right_key = try right.column(right_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
    }

    const total_cols = left.columns.len + right.columns.len - rightExcludedKeyCount(right, right_key_names);
    var names = try allocator.alloc([]const u8, total_cols);
    defer allocator.free(names);
    var temporary_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (temporary_names.items) |name| allocator.free(name);
        temporary_names.deinit(allocator);
    }
    const DeviceColumn = std.meta.Elem(@TypeOf(left.columns));
    var columns = try allocator.alloc(DeviceColumn, total_cols);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        allocator.free(columns);
    }

    for (left.names, left.columns, 0..) |name, col, i| {
        names[initialized] = name;
        columns[initialized] = if (leftKeyRightIndex(left, right, left_key_names, right_key_names, i)) |right_key_index|
            try coalesceJoinKeys(col, right.columns[right_key_index])
        else
            try col.clone();
        initialized += 1;
    }

    for (right.names, right.columns, 0..) |name, col, i| {
        if (rightKeyIndexInList(right, right_key_names, i)) continue;
        if (nameNeedsSuffix(left, name)) {
            const suffixed = try suffixedNameTemp(allocator, name, options_value.right_suffix);
            errdefer allocator.free(suffixed);
            try temporary_names.append(allocator, suffixed);
            names[initialized] = suffixed;
        } else {
            names[initialized] = name;
        }
        columns[initialized] = try col.clone();
        initialized += 1;
    }

    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, allocator, names, columns, left.rows, left.device);
}

fn validateSingleJoinKeys(left: anytype, right: anytype, left_key_name: []const u8, right_key_name: []const u8) keys_mod.KeyMatchError!void {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    const left_key = try left.column(left_key_name);
    const right_key = try right.column(right_key_name);
    if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
}

fn validateMultiJoinKeys(
    left: anytype,
    right: anytype,
    left_key_names: []const []const u8,
    right_key_names: []const []const u8,
) keys_mod.KeyMatchError!void {
    if (!left.device.sameDevice(right.device)) return error.InvalidDevice;
    if (left_key_names.len == 0 or left_key_names.len != right_key_names.len) return error.LengthMismatch;
    for (left_key_names, right_key_names) |left_name, right_name| {
        const left_key = try left.column(left_name);
        const right_key = try right.column(right_name);
        if (left_key.dtype() != right_key.dtype()) return error.TypeMismatch;
    }
}

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
