const std = @import("std");
const array_mod = @import("array.zig");
const array_helpers = @import("dataframe_array.zig");
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
