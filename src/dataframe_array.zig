const std = @import("std");
const array_mod = @import("array.zig");
const array_helpers_mod = @import("dataframe_array_helpers.zig");

const DeviceFrameArrayError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    InvalidCsv,
    EmptyDataFrame,
    UnsupportedType,
    InvalidDevice,
};

pub const requireCompatibleColumnArrays = array_helpers_mod.requireCompatibleColumnArrays;
pub const combineValidityMasks = array_helpers_mod.combineValidityMasks;
pub const zeroValue = array_helpers_mod.zeroValue;
pub const rowIndicesFromMask = array_helpers_mod.rowIndicesFromMask;
pub const sliceArray1d = array_helpers_mod.sliceArray1d;
pub const takeArray1d = array_helpers_mod.takeArray1d;
pub const concatTypedColumns = array_helpers_mod.concatTypedColumns;
pub const coalesceTypedJoinKeys = array_helpers_mod.coalesceTypedJoinKeys;
pub const concatDeviceColumns = array_helpers_mod.concatDeviceColumns;
pub const coalesceJoinKeys = array_helpers_mod.coalesceJoinKeys;
pub const initDeviceDataFrameFromOwnedColumns = array_helpers_mod.initDeviceDataFrameFromOwnedColumns;
pub const concatDeviceDataFramesRows = array_helpers_mod.concatDeviceDataFramesRows;
pub const takeOptionalRows = array_helpers_mod.takeOptionalRows;
pub const columnsRowsEqual = array_helpers_mod.columnsRowsEqual;
pub const columnsRowsEqualTyped = array_helpers_mod.columnsRowsEqualTyped;

pub fn select(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    wanted_names: []const []const u8,
) DeviceFrameArrayError!DeviceDataFrame {
    if (wanted_names.len == 0) return DeviceDataFrame.initEmpty(input.allocator, input.rows, input.device);
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, wanted_names.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (wanted_names, 0..) |name, i| {
        const source = try input.column(name);
        columns[i] = try source.clone();
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, wanted_names, columns, input.rows, input.device);
}

pub fn withColumn(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    name: []const u8,
    data: anytype,
) DeviceFrameArrayError!DeviceDataFrame {
    if (data.len() != input.rows) return error.LengthMismatch;
    if (!data.device().sameDevice(input.device)) return error.InvalidDevice;
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var source_names = try input.allocator.alloc([]const u8, input.columns.len + 1);
    defer input.allocator.free(source_names);
    for (input.names, 0..) |existing, i| source_names[i] = existing;
    source_names[input.columns.len] = name;

    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len + 1);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    columns[input.columns.len] = try data.clone();
    initialized += 1;
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, source_names, columns, input.rows, input.device);
}

pub fn view(
    comptime DeviceDataFrameView: type,
    comptime DeviceColumnView: type,
    input: anytype,
) DeviceFrameArrayError!DeviceDataFrameView {
    const columns = try input.allocator.alloc(DeviceColumnView, input.columns.len);
    errdefer input.allocator.free(columns);
    for (input.columns, columns) |col, *slot| slot.* = col.view();
    return .{
        .allocator = input.allocator,
        .names = input.names,
        .columns = columns,
        .rows = input.rows,
        .device = input.device,
    };
}

pub fn sliceRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    start: usize,
    stop: usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const end = @min(stop, input.rows);
    const begin = @min(start, end);
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.sliceRows(begin, end);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, end - begin, input.device);
}

pub fn takeRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const usize,
) DeviceFrameArrayError!DeviceDataFrame {
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.take(row_indices);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, row_indices.len, input.device);
}

pub fn filterRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    mask: []const bool,
) DeviceFrameArrayError!DeviceDataFrame {
    if (mask.len != input.rows) return error.LengthMismatch;
    const row_indices = try rowIndicesFromMask(input.allocator, mask);
    defer input.allocator.free(row_indices);
    return takeRows(DeviceDataFrame, input, row_indices);
}

pub fn toDevice(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    device_value: array_mod.Device,
) DeviceFrameArrayError!DeviceDataFrame {
    if (!device_value.isAvailable()) return error.InvalidDevice;
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.to(device_value);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, input.rows, device_value);
}
