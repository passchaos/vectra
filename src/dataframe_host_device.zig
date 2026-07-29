//! Host/device dataframe conversion helpers.
//!
//! Device dataframe types are declared by the Boltha-backed public facade, so
//! this module receives them as comptime parameters. That keeps the host-side
//! dataframe implementation independent of the device facade while preserving
//! existing conversion APIs.

const std = @import("std");
const series_mod = @import("series.zig");
const array_mod = @import("array.zig");
const dataframe_column_mod = @import("dataframe_column.zig");

pub const DataError = series_mod.DataError;
const freeColumn = dataframe_column_mod.freeColumn;

pub fn deviceDataFrameFromDataFrame(
    comptime DeviceDataFrame: type,
    comptime DeviceColumnDef: type,
    comptime DeviceColumn: type,
    allocator: std.mem.Allocator,
    frame: anytype,
    device_value: array_mod.Device,
) (DataError || array_mod.ArrayError)!DeviceDataFrame {
    if (!device_value.isAvailable()) return error.InvalidDevice;
    if (frame.columns.len == 0) return DeviceDataFrame.initEmpty(allocator, frame.rows, device_value);
    var defs = try allocator.alloc(DeviceColumnDef, frame.columns.len);
    defer allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |*def| def.data.deinit();
    }
    for (frame.names, frame.columns, 0..) |name, col, i| {
        defs[i].name = name;
        defs[i].data = switch (col) {
            .f64 => |values| try DeviceColumn.fromSlice(f64, allocator, values, device_value),
            .i64 => |values| try DeviceColumn.fromSlice(i64, allocator, values, device_value),
            .bool => |values| try DeviceColumn.fromSlice(bool, allocator, values, device_value),
            .string => return error.TypeUnsupported,
        };
        initialized += 1;
    }
    return DeviceDataFrame.init(allocator, defs);
}

pub fn deviceDataFrameToDataFrame(
    comptime DataFrame: type,
    comptime ColumnDef: type,
    frame: anytype,
) (DataError || array_mod.ArrayError)!DataFrame {
    var defs = try frame.allocator.alloc(ColumnDef, frame.columns.len);
    defer frame.allocator.free(defs);
    var initialized: usize = 0;
    defer {
        for (defs[0..initialized]) |def| freeColumn(frame.allocator, def.data);
    }

    for (frame.names, frame.columns, 0..) |name, col, i| {
        if (col.hasNulls()) return error.TypeUnsupported;
        defs[i].name = name;
        defs[i].data = switch (col) {
            .f64 => |typed| .{ .f64 = try typed.toOwnedSlice(frame.allocator) },
            .i64 => |typed| .{ .i64 = try typed.toOwnedSlice(frame.allocator) },
            .bool => |typed| .{ .bool = try typed.toOwnedSlice(frame.allocator) },
            else => return error.TypeUnsupported,
        };
        initialized += 1;
    }
    return DataFrame.init(frame.allocator, defs);
}
