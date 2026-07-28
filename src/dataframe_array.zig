const std = @import("std");
const array_mod = @import("array.zig");
const validity_mod = @import("dataframe_validity.zig");
const countNulls = validity_mod.countNulls;
const validityValues = validity_mod.validityValues;

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
