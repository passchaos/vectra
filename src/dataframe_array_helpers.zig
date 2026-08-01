//! Low-level array/column helpers shared by dataframe operations.

const std = @import("std");
const array_mod = @import("array.zig");
const validity_mod = @import("dataframe_validity_core.zig");
const numeric_mod = @import("dataframe_numeric.zig");

const countNulls = validity_mod.countNulls;
const validityValues = validity_mod.validityValues;
const groupKeyEqual = numeric_mod.groupKeyEqual;

pub fn requireCompatibleColumnArrays(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!void {
    if (!lhs.device.sameDevice(rhs.device)) return error.InvalidDevice;
    if (lhs.shape.len != 1 or rhs.shape.len != 1 or lhs.shape[0] != rhs.shape[0]) return error.ShapeMismatch;
}

pub fn combineValidityMasks(
    _: std.mem.Allocator,
    lhs: ?array_mod.Array(bool),
    rhs: ?array_mod.Array(bool),
    rows: usize,
    device_value: array_mod.Device,
) array_mod.ArrayError!?array_mod.Array(bool) {
    if (lhs == null and rhs == null) return null;
    if (lhs) |mask| {
        if (!mask.device.sameDevice(device_value) or mask.shape.len != 1 or mask.shape[0] != rows) return error.InvalidDevice;
    }
    if (rhs) |mask| {
        if (!mask.device.sameDevice(device_value) or mask.shape.len != 1 or mask.shape[0] != rows) return error.InvalidDevice;
    }
    if (lhs == null) return try rhs.?.clone();
    if (rhs == null) return try lhs.?.clone();
    const lhs_values = try lhs.?.toOwnedSlice(lhs.?.allocator);
    defer lhs.?.allocator.free(lhs_values);
    const rhs_values = try rhs.?.toOwnedSlice(rhs.?.allocator);
    defer rhs.?.allocator.free(rhs_values);
    const out_values = try lhs.?.allocator.alloc(bool, rows);
    defer lhs.?.allocator.free(out_values);
    for (lhs_values, rhs_values, out_values) |left_valid, right_valid, *slot| {
        slot.* = left_valid and right_valid;
    }
    return try array_mod.Array(bool).fromSliceOn(lhs.?.allocator, out_values, &.{rows}, device_value);
}

pub fn zeroValue(comptime T: type) T {
    if (comptime T == bool) return false;
    if (comptime T == array_mod.BFloat16) return array_mod.BFloat16.fromF32(0);
    if (comptime T == array_mod.Complex64) return .{ .re = 0, .im = 0 };
    if (comptime T == array_mod.Complex128) return .{ .re = 0, .im = 0 };
    return switch (@typeInfo(T)) {
        .int, .float => 0,
        else => @compileError("zeroValue only supports primitive numeric Arrow values"),
    };
}

pub fn oneValue(comptime T: type) T {
    if (comptime T == bool) return true;
    if (comptime T == array_mod.BFloat16) return array_mod.BFloat16.fromF32(1);
    if (comptime T == array_mod.Complex64) return .{ .re = 1, .im = 0 };
    if (comptime T == array_mod.Complex128) return .{ .re = 1, .im = 0 };
    return switch (@typeInfo(T)) {
        .int, .float => 1,
        else => @compileError("oneValue only supports primitive numeric Arrow values"),
    };
}

pub fn rowIndicesFromMask(allocator: std.mem.Allocator, mask: []const bool) array_mod.ArrayError![]usize {
    var count: usize = 0;
    for (mask) |keep| {
        if (keep) count += 1;
    }
    const indices = try allocator.alloc(usize, count);
    var write: usize = 0;
    for (mask, 0..) |keep, i| {
        if (keep) {
            indices[write] = i;
            write += 1;
        }
    }
    return indices;
}

pub fn sliceArray1d(comptime T: type, values: array_mod.Array(T), start: usize, stop: usize) array_mod.ArrayError!array_mod.Array(T) {
    if (values.shape.len != 1) return error.InvalidShape;
    if (start > values.shape[0] or stop < start or stop > values.shape[0]) return error.IndexOutOfBounds;
    const host_values = try values.toOwnedSlice(values.allocator);
    defer values.allocator.free(host_values);
    return array_mod.Array(T).fromSliceOn(values.allocator, host_values[start..stop], &.{stop - start}, values.device);
}

pub fn takeArray1d(comptime T: type, values: array_mod.Array(T), row_indices: []const usize) array_mod.ArrayError!array_mod.Array(T) {
    if (values.shape.len != 1) return error.InvalidShape;
    const host_values = try values.toOwnedSlice(values.allocator);
    defer values.allocator.free(host_values);
    const out_values = try values.allocator.alloc(T, row_indices.len);
    defer values.allocator.free(out_values);
    for (row_indices, out_values) |idx, *slot| {
        if (idx >= host_values.len) return error.IndexOutOfBounds;
        slot.* = host_values[idx];
    }
    return array_mod.Array(T).fromSliceOn(values.allocator, out_values, &.{row_indices.len}, values.device);
}

pub fn concatTypedColumns(comptime T: type, first: anytype, second: anytype) array_mod.ArrayError!@TypeOf(first) {
    if (!first.device().sameDevice(second.device())) return error.InvalidDevice;
    const allocator = first.values.allocator;
    const first_values = try first.values.toOwnedSlice(allocator);
    defer allocator.free(first_values);
    const second_values = try second.values.toOwnedSlice(allocator);
    defer allocator.free(second_values);
    const values = try allocator.alloc(T, first_values.len + second_values.len);
    defer allocator.free(values);
    @memcpy(values[0..first_values.len], first_values);
    @memcpy(values[first_values.len..], second_values);

    const first_validity = try validityValues(first, allocator);
    defer if (first_validity) |validity| allocator.free(validity);
    const second_validity = try validityValues(second, allocator);
    defer if (second_validity) |validity| allocator.free(validity);

    if (first_validity == null and second_validity == null) return @TypeOf(first).fromSlice(allocator, values, first.device());
    const validity = try allocator.alloc(bool, values.len);
    defer allocator.free(validity);
    for (validity[0..first_values.len], 0..) |*slot, i| slot.* = if (first_validity) |mask| mask[i] else true;
    for (validity[first_values.len..], 0..) |*slot, i| slot.* = if (second_validity) |mask| mask[i] else true;
    return @TypeOf(first).fromSliceWithValidity(allocator, values, validity, first.device());
}

pub fn coalesceTypedJoinKeys(comptime T: type, left: anytype, right: anytype) (array_mod.ArrayError || error{LengthMismatch})!@TypeOf(left) {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    if (left.len() != right.len()) return error.LengthMismatch;
    const allocator = left.values.allocator;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    const values = try allocator.alloc(T, left_values.len);
    defer allocator.free(values);
    const validity = try allocator.alloc(bool, left_values.len);
    defer allocator.free(validity);
    for (values, validity, 0..) |*value_slot, *valid_slot, i| {
        const left_valid = if (maybe_left_validity) |mask| mask[i] else true;
        const right_valid = if (maybe_right_validity) |mask| mask[i] else true;
        if (left_valid) {
            value_slot.* = left_values[i];
            valid_slot.* = true;
        } else if (right_valid) {
            value_slot.* = right_values[i];
            valid_slot.* = true;
        } else {
            value_slot.* = zeroValue(T);
            valid_slot.* = false;
        }
    }
    if (countNulls(validity) == 0) return @TypeOf(left).fromSlice(allocator, values, left.device());
    return @TypeOf(left).fromSliceWithValidity(allocator, values, validity, left.device());
}

pub fn concatDeviceColumns(first: anytype, second: anytype) (array_mod.ArrayError || error{TypeMismatch})!@TypeOf(first) {
    if (first.dtype() != second.dtype()) return error.TypeMismatch;
    return switch (first) {
        .bool => |typed| .{ .bool = try concatTypedColumns(bool, typed, second.bool) },
        .i8 => |typed| .{ .i8 = try concatTypedColumns(i8, typed, second.i8) },
        .i16 => |typed| .{ .i16 = try concatTypedColumns(i16, typed, second.i16) },
        .i32 => |typed| .{ .i32 = try concatTypedColumns(i32, typed, second.i32) },
        .i64 => |typed| .{ .i64 = try concatTypedColumns(i64, typed, second.i64) },
        .u8 => |typed| .{ .u8 = try concatTypedColumns(u8, typed, second.u8) },
        .u16 => |typed| .{ .u16 = try concatTypedColumns(u16, typed, second.u16) },
        .u32 => |typed| .{ .u32 = try concatTypedColumns(u32, typed, second.u32) },
        .u64 => |typed| .{ .u64 = try concatTypedColumns(u64, typed, second.u64) },
        .usize => |typed| .{ .usize = try concatTypedColumns(usize, typed, second.usize) },
        .isize => |typed| .{ .isize = try concatTypedColumns(isize, typed, second.isize) },
        .f16 => |typed| .{ .f16 = try concatTypedColumns(f16, typed, second.f16) },
        .f32 => |typed| .{ .f32 = try concatTypedColumns(f32, typed, second.f32) },
        .f64 => |typed| .{ .f64 = try concatTypedColumns(f64, typed, second.f64) },
        .bf16 => |typed| .{ .bf16 = try concatTypedColumns(array_mod.BFloat16, typed, second.bf16) },
        .c64 => |typed| .{ .c64 = try concatTypedColumns(array_mod.Complex64, typed, second.c64) },
        .c128 => |typed| .{ .c128 = try concatTypedColumns(array_mod.Complex128, typed, second.c128) },
    };
}

pub fn coalesceJoinKeys(left: anytype, right: anytype) (array_mod.ArrayError || error{ LengthMismatch, TypeMismatch })!@TypeOf(left) {
    if (left.dtype() != right.dtype()) return error.TypeMismatch;
    return switch (left) {
        .bool => |typed| .{ .bool = try coalesceTypedJoinKeys(bool, typed, right.bool) },
        .i8 => |typed| .{ .i8 = try coalesceTypedJoinKeys(i8, typed, right.i8) },
        .i16 => |typed| .{ .i16 = try coalesceTypedJoinKeys(i16, typed, right.i16) },
        .i32 => |typed| .{ .i32 = try coalesceTypedJoinKeys(i32, typed, right.i32) },
        .i64 => |typed| .{ .i64 = try coalesceTypedJoinKeys(i64, typed, right.i64) },
        .u8 => |typed| .{ .u8 = try coalesceTypedJoinKeys(u8, typed, right.u8) },
        .u16 => |typed| .{ .u16 = try coalesceTypedJoinKeys(u16, typed, right.u16) },
        .u32 => |typed| .{ .u32 = try coalesceTypedJoinKeys(u32, typed, right.u32) },
        .u64 => |typed| .{ .u64 = try coalesceTypedJoinKeys(u64, typed, right.u64) },
        .usize => |typed| .{ .usize = try coalesceTypedJoinKeys(usize, typed, right.usize) },
        .isize => |typed| .{ .isize = try coalesceTypedJoinKeys(isize, typed, right.isize) },
        .f16 => |typed| .{ .f16 = try coalesceTypedJoinKeys(f16, typed, right.f16) },
        .f32 => |typed| .{ .f32 = try coalesceTypedJoinKeys(f32, typed, right.f32) },
        .f64 => |typed| .{ .f64 = try coalesceTypedJoinKeys(f64, typed, right.f64) },
        .bf16 => |typed| .{ .bf16 = try coalesceTypedJoinKeys(array_mod.BFloat16, typed, right.bf16) },
        .c64 => |typed| .{ .c64 = try coalesceTypedJoinKeys(array_mod.Complex64, typed, right.c64) },
        .c128 => |typed| .{ .c128 = try coalesceTypedJoinKeys(array_mod.Complex128, typed, right.c128) },
    };
}

pub fn initDeviceDataFrameFromOwnedColumns(
    comptime DeviceDataFrame: type,
    allocator: std.mem.Allocator,
    source_names: []const []const u8,
    columns: anytype,
    rows: usize,
    device_value: array_mod.Device,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })!DeviceDataFrame {
    if (source_names.len != columns.len) return error.LengthMismatch;
    for (columns) |col| {
        if (col.len() != rows) return error.LengthMismatch;
        if (!col.device().sameDevice(device_value)) return error.InvalidDevice;
    }

    var names = try allocator.alloc([]const u8, source_names.len);
    errdefer allocator.free(names);
    var initialized: usize = 0;
    errdefer {
        for (names[0..initialized]) |name| allocator.free(name);
    }
    for (source_names, names) |source, *slot| {
        slot.* = try allocator.dupe(u8, source);
        initialized += 1;
    }
    return .{ .allocator = allocator, .names = names, .columns = columns, .rows = rows, .device = device_value };
}

pub fn concatDeviceDataFramesRows(
    comptime DeviceDataFrame: type,
    first: DeviceDataFrame,
    second: DeviceDataFrame,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ LengthMismatch, InvalidDevice, ColumnNotFound, TypeMismatch })!DeviceDataFrame {
    if (!first.device.sameDevice(second.device)) return error.InvalidDevice;
    if (first.columns.len != second.columns.len) return error.LengthMismatch;
    for (first.names, second.names, first.columns, second.columns) |first_name, second_name, first_col, second_col| {
        if (!std.mem.eql(u8, first_name, second_name)) return error.ColumnNotFound;
        if (first_col.dtype() != second_col.dtype()) return error.TypeMismatch;
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(first.columns));
    var columns = try first.allocator.alloc(DeviceColumn, first.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        first.allocator.free(columns);
    }
    for (first.columns, second.columns, 0..) |first_col, second_col, i| {
        columns[i] = try concatDeviceColumns(first_col, second_col);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, first.allocator, first.names, columns, first.rows + second.rows, first.device);
}

pub fn concatDeviceDataFramesColumns(
    comptime DeviceDataFrame: type,
    first: DeviceDataFrame,
    second: DeviceDataFrame,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })!DeviceDataFrame {
    if (first.rows != second.rows) return error.LengthMismatch;
    if (!first.device.sameDevice(second.device)) return error.InvalidDevice;
    for (second.names, 0..) |name, index| {
        if (first.columnIndex(name) != null) return error.InvalidShape;
        for (second.names[0..index]) |previous| {
            if (std.mem.eql(u8, name, previous)) return error.InvalidShape;
        }
    }

    const DeviceColumn = std.meta.Elem(@TypeOf(first.columns));
    const total_columns = first.columns.len + second.columns.len;
    var source_names = try first.allocator.alloc([]const u8, total_columns);
    defer first.allocator.free(source_names);
    var columns = try first.allocator.alloc(DeviceColumn, total_columns);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        first.allocator.free(columns);
    }

    for (first.names, first.columns, 0..) |name, column, index| {
        source_names[index] = name;
        columns[index] = try column.clone();
        initialized += 1;
    }
    for (second.names, second.columns, 0..) |name, column, second_index| {
        const output_index = first.columns.len + second_index;
        source_names[output_index] = name;
        columns[output_index] = try column.clone();
        initialized += 1;
    }

    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, first.allocator, source_names, columns, first.rows, first.device);
}

pub fn takeOptionalRows(
    comptime DeviceDataFrame: type,
    input: DeviceDataFrame,
    row_indices: []const ?usize,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ LengthMismatch, InvalidDevice })!DeviceDataFrame {
    const DeviceColumn = std.meta.Elem(@TypeOf(input.columns));
    var columns = try input.allocator.alloc(DeviceColumn, input.columns.len);
    var initialized: usize = 0;
    errdefer {
        for (columns[0..initialized]) |*col| col.deinit();
        input.allocator.free(columns);
    }
    for (input.columns, 0..) |col, i| {
        columns[i] = try col.takeOptional(row_indices);
        initialized += 1;
    }
    return initDeviceDataFrameFromOwnedColumns(DeviceDataFrame, input.allocator, input.names, columns, row_indices.len, input.device);
}

pub fn columnsRowsEqual(
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_i: usize,
    right_i: usize,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ TypeMismatch, TypeUnsupported })!bool {
    if (left.dtype() != right.dtype()) return error.TypeMismatch;
    return switch (left) {
        .bool => |typed| columnsRowsEqualTyped(bool, allocator, typed, right.bool, left_i, right_i),
        .i8 => |typed| columnsRowsEqualTyped(i8, allocator, typed, right.i8, left_i, right_i),
        .i16 => |typed| columnsRowsEqualTyped(i16, allocator, typed, right.i16, left_i, right_i),
        .i32 => |typed| columnsRowsEqualTyped(i32, allocator, typed, right.i32, left_i, right_i),
        .i64 => |typed| columnsRowsEqualTyped(i64, allocator, typed, right.i64, left_i, right_i),
        .u8 => |typed| columnsRowsEqualTyped(u8, allocator, typed, right.u8, left_i, right_i),
        .u16 => |typed| columnsRowsEqualTyped(u16, allocator, typed, right.u16, left_i, right_i),
        .u32 => |typed| columnsRowsEqualTyped(u32, allocator, typed, right.u32, left_i, right_i),
        .u64 => |typed| columnsRowsEqualTyped(u64, allocator, typed, right.u64, left_i, right_i),
        .usize => |typed| columnsRowsEqualTyped(usize, allocator, typed, right.usize, left_i, right_i),
        .isize => |typed| columnsRowsEqualTyped(isize, allocator, typed, right.isize, left_i, right_i),
        .f16 => |typed| columnsRowsEqualTyped(f16, allocator, typed, right.f16, left_i, right_i),
        .f32 => |typed| columnsRowsEqualTyped(f32, allocator, typed, right.f32, left_i, right_i),
        .f64 => |typed| columnsRowsEqualTyped(f64, allocator, typed, right.f64, left_i, right_i),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn columnsRowsEqualTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    left_i: usize,
    right_i: usize,
) (std.mem.Allocator.Error || array_mod.ArrayError)!bool {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    if (left_i >= left.len() or right_i >= right.len()) return error.IndexOutOfBounds;
    const left_validity = try validityValues(left, allocator);
    defer if (left_validity) |validity| allocator.free(validity);
    const right_validity = try validityValues(right, allocator);
    defer if (right_validity) |validity| allocator.free(validity);
    if (left_validity) |validity| {
        if (!validity[left_i]) return false;
    }
    if (right_validity) |validity| {
        if (!validity[right_i]) return false;
    }
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    return groupKeyEqual(T, left_values[left_i], right_values[right_i]);
}
