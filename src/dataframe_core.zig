//! Core owning DeviceDataFrame lifecycle and metadata helpers.
//!
//! The public dataframe facade delegates here so ownership invariants are kept
//! together while `dataframe.zig` remains a compact API surface.

const std = @import("std");
const array_mod = @import("array.zig");
const dataframe_array_mod = @import("dataframe_array.zig");
const series_mod = @import("series.zig");

const DataError = series_mod.DataError;
const DeviceDataError = DataError || array_mod.ArrayError;

pub fn init(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    defs: anytype,
) DeviceDataError!DeviceDataFrame {
    if (defs.len == 0) return DeviceDataFrame.initEmpty(allocator, 0, .cpu);
    const rows = defs[0].data.len();
    const device_value = defs[0].data.device();
    for (defs) |def| {
        if (def.data.len() != rows) return error.LengthMismatch;
        if (!def.data.device().sameDevice(device_value)) return error.InvalidDevice;
    }

    var names = try allocator.alloc([]const u8, defs.len);
    errdefer allocator.free(names);
    const DeviceColumn = @TypeOf(defs[0].data);
    var columns = try allocator.alloc(DeviceColumn, defs.len);
    errdefer allocator.free(columns);

    var initialized: usize = 0;
    errdefer {
        for (0..initialized) |i| {
            allocator.free(names[i]);
            columns[i].deinit();
        }
    }

    for (defs, 0..) |def, i| {
        names[i] = try allocator.dupe(u8, def.name);
        columns[i] = try def.data.clone();
        initialized += 1;
    }

    return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows, .device = device_value };
}

pub fn initEmpty(comptime DeviceDataFrame: type, allocator: std.mem.Allocator, rows: usize, device_value: array_mod.Device) DeviceDataError!DeviceDataFrame {
    if (!device_value.isAvailable()) return error.InvalidDevice;
    return .{ .allocator = allocator, .names = &.{}, .columns = &.{}, .rows = rows, .device = device_value };
}

pub fn deinit(frame: anytype) void {
    for (frame.names) |name| frame.allocator.free(name);
    for (frame.columns) |*col| col.deinit();
    if (frame.names.len != 0) frame.allocator.free(frame.names);
    if (frame.columns.len != 0) frame.allocator.free(frame.columns);
    frame.* = undefined;
}

pub fn clone(comptime DeviceDataFrame: type, frame: DeviceDataFrame) DeviceDataError!DeviceDataFrame {
    const DeviceColumn = std.meta.Elem(@TypeOf(frame.columns));
    var columns = try frame.allocator.alloc(DeviceColumn, frame.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        frame.allocator.free(columns);
    }
    for (frame.columns, 0..) |col, i| {
        columns[i] = try col.clone();
        initialized += 1;
    }
    return dataframe_array_mod.initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, frame.allocator, frame.names, columns, frame.rows, frame.device);
}

pub fn shape(frame: anytype) struct { rows: usize, cols: usize } {
    return .{ .rows = frame.rows, .cols = frame.columns.len };
}

pub fn columnIndex(frame: anytype, name: []const u8) ?usize {
    for (frame.names, 0..) |existing, i| {
        if (std.mem.eql(u8, existing, name)) return i;
    }
    return null;
}

pub fn column(frame: anytype, name: []const u8) DataError!*const std.meta.Elem(@TypeOf(frame.columns)) {
    const idx = columnIndex(frame, name) orelse return error.ColumnNotFound;
    return &frame.columns[idx];
}

pub fn columnDType(frame: anytype, name: []const u8) DataError!array_mod.DType {
    const idx = columnIndex(frame, name) orelse return error.ColumnNotFound;
    return frame.columns[idx].dtype();
}
