const std = @import("std");
const array_mod = @import("array.zig");

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
