//! Single-key as-of join row-index builders.

const std = @import("std");
const array_mod = @import("../../array.zig");
const numeric_mod = @import("../../dataframe_numeric.zig");
const validity_mod = @import("../validity/core.zig");

const validityValues = validity_mod.validityValues;
const compareSortValues = numeric_mod.compareSortValues;
const asofDistance = numeric_mod.asofDistance;

pub const KeyMatchError = std.mem.Allocator.Error || std.Io.Writer.Error || array_mod.ArrayError || error{
    LengthMismatch,
    ColumnNotFound,
    TypeMismatch,
    TypeUnsupported,
    UnsupportedType,
    InvalidCsv,
    EmptyDataFrame,
    InvalidDevice,
    IndexOutOfBounds,
};

pub fn asofRightRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype, strategy: anytype) KeyMatchError![]?usize {
    return switch (left) {
        .i8 => |typed| asofRightRowIndicesTyped(i8, allocator, typed, right.i8, strategy),
        .i16 => |typed| asofRightRowIndicesTyped(i16, allocator, typed, right.i16, strategy),
        .i32 => |typed| asofRightRowIndicesTyped(i32, allocator, typed, right.i32, strategy),
        .i64 => |typed| asofRightRowIndicesTyped(i64, allocator, typed, right.i64, strategy),
        .u8 => |typed| asofRightRowIndicesTyped(u8, allocator, typed, right.u8, strategy),
        .u16 => |typed| asofRightRowIndicesTyped(u16, allocator, typed, right.u16, strategy),
        .u32 => |typed| asofRightRowIndicesTyped(u32, allocator, typed, right.u32, strategy),
        .u64 => |typed| asofRightRowIndicesTyped(u64, allocator, typed, right.u64, strategy),
        .usize => |typed| asofRightRowIndicesTyped(usize, allocator, typed, right.usize, strategy),
        .isize => |typed| asofRightRowIndicesTyped(isize, allocator, typed, right.isize, strategy),
        .f16 => |typed| asofRightRowIndicesTyped(f16, allocator, typed, right.f16, strategy),
        .f32 => |typed| asofRightRowIndicesTyped(f32, allocator, typed, right.f32, strategy),
        .f64 => |typed| asofRightRowIndicesTyped(f64, allocator, typed, right.f64, strategy),
        .bool, .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn asofRightRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    strategy: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{ InvalidDevice, TypeUnsupported })![]?usize {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    const indices = try allocator.alloc(?usize, left_values.len);
    for (left_values, indices, 0..) |left_value, *slot, left_i| {
        slot.* = null;
        if (maybe_left_validity) |validity| {
            if (!validity[left_i]) continue;
        }
        var best: ?usize = null;
        for (right_values, 0..) |right_value, right_i| {
            if (maybe_right_validity) |validity| {
                if (!validity[right_i]) continue;
            }
            switch (strategy) {
                .previous => {
                    if (compareSortValues(T, right_value, left_value) <= 0 and (best == null or compareSortValues(T, right_value, right_values[best.?]) > 0)) best = right_i;
                },
                .next => {
                    if (compareSortValues(T, right_value, left_value) >= 0 and (best == null or compareSortValues(T, right_value, right_values[best.?]) < 0)) best = right_i;
                },
                .nearest => {
                    if (best == null or asofDistance(T, left_value, right_value) < asofDistance(T, left_value, right_values[best.?])) best = right_i;
                },
            }
        }
        slot.* = best;
    }
    return indices;
}
