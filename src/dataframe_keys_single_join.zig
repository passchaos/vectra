//! Single-key/asof row-index builders for dataframe joins.

const std = @import("std");
const array_mod = @import("array.zig");
const numeric_mod = @import("dataframe_numeric.zig");
const validity_mod = @import("dataframe_validity.zig");

const validityValues = validity_mod.validityValues;
const groupKeyEqual = numeric_mod.groupKeyEqual;
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

pub fn innerJoinRowIndicesTyped(
    comptime T: type,
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (left_values, 0..) |left_value, left_i| {
        if (maybe_left_validity) |validity| {
            if (!validity[left_i]) continue;
        }
        for (right_values, 0..) |right_value, right_i| {
            if (maybe_right_validity) |validity| {
                if (!validity[right_i]) continue;
            }
            if (groupKeyEqual(T, left_value, right_value)) {
                try left_indices.append(allocator, left_i);
                try right_indices.append(allocator, right_i);
            }
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

pub fn leftJoinRowIndicesTyped(
    comptime T: type,
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);

    for (left_values, 0..) |left_value, left_i| {
        var matched = false;
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    try left_indices.append(allocator, left_i);
                    try right_indices.append(allocator, right_i);
                    matched = true;
                }
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

pub fn fullJoinRowIndicesTyped(
    comptime T: type,
    comptime JoinRowIndexPair: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})!JoinRowIndexPair {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var left_indices: std.ArrayList(?usize) = .empty;
    errdefer left_indices.deinit(allocator);
    var right_indices: std.ArrayList(?usize) = .empty;
    errdefer right_indices.deinit(allocator);
    const right_matched = try allocator.alloc(bool, right_values.len);
    defer allocator.free(right_matched);
    @memset(right_matched, false);

    for (left_values, 0..) |left_value, left_i| {
        var matched = false;
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    try left_indices.append(allocator, left_i);
                    try right_indices.append(allocator, right_i);
                    right_matched[right_i] = true;
                    matched = true;
                }
            }
        }
        if (!matched) {
            try left_indices.append(allocator, left_i);
            try right_indices.append(allocator, null);
        }
    }

    for (right_values, 0..) |_, right_i| {
        if (maybe_right_validity) |validity| {
            if (!validity[right_i]) {
                try left_indices.append(allocator, null);
                try right_indices.append(allocator, right_i);
                continue;
            }
        }
        if (!right_matched[right_i]) {
            try left_indices.append(allocator, null);
            try right_indices.append(allocator, right_i);
        }
    }

    const owned_left = try left_indices.toOwnedSlice(allocator);
    left_indices = .empty;
    errdefer allocator.free(owned_left);
    const owned_right = try right_indices.toOwnedSlice(allocator);
    right_indices = .empty;
    return .{
        .allocator = allocator,
        .left = owned_left,
        .right = owned_right,
    };
}

pub fn innerJoinRowIndices(comptime JoinRowIndexPair: type, allocator: std.mem.Allocator, left: anytype, right: anytype) KeyMatchError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| innerJoinRowIndicesTyped(bool, JoinRowIndexPair, allocator, typed, right.bool),
        .i8 => |typed| innerJoinRowIndicesTyped(i8, JoinRowIndexPair, allocator, typed, right.i8),
        .i16 => |typed| innerJoinRowIndicesTyped(i16, JoinRowIndexPair, allocator, typed, right.i16),
        .i32 => |typed| innerJoinRowIndicesTyped(i32, JoinRowIndexPair, allocator, typed, right.i32),
        .i64 => |typed| innerJoinRowIndicesTyped(i64, JoinRowIndexPair, allocator, typed, right.i64),
        .u8 => |typed| innerJoinRowIndicesTyped(u8, JoinRowIndexPair, allocator, typed, right.u8),
        .u16 => |typed| innerJoinRowIndicesTyped(u16, JoinRowIndexPair, allocator, typed, right.u16),
        .u32 => |typed| innerJoinRowIndicesTyped(u32, JoinRowIndexPair, allocator, typed, right.u32),
        .u64 => |typed| innerJoinRowIndicesTyped(u64, JoinRowIndexPair, allocator, typed, right.u64),
        .usize => |typed| innerJoinRowIndicesTyped(usize, JoinRowIndexPair, allocator, typed, right.usize),
        .isize => |typed| innerJoinRowIndicesTyped(isize, JoinRowIndexPair, allocator, typed, right.isize),
        .f16 => |typed| innerJoinRowIndicesTyped(f16, JoinRowIndexPair, allocator, typed, right.f16),
        .f32 => |typed| innerJoinRowIndicesTyped(f32, JoinRowIndexPair, allocator, typed, right.f32),
        .f64 => |typed| innerJoinRowIndicesTyped(f64, JoinRowIndexPair, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn leftJoinRowIndices(comptime JoinRowIndexPair: type, allocator: std.mem.Allocator, left: anytype, right: anytype) KeyMatchError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| leftJoinRowIndicesTyped(bool, JoinRowIndexPair, allocator, typed, right.bool),
        .i8 => |typed| leftJoinRowIndicesTyped(i8, JoinRowIndexPair, allocator, typed, right.i8),
        .i16 => |typed| leftJoinRowIndicesTyped(i16, JoinRowIndexPair, allocator, typed, right.i16),
        .i32 => |typed| leftJoinRowIndicesTyped(i32, JoinRowIndexPair, allocator, typed, right.i32),
        .i64 => |typed| leftJoinRowIndicesTyped(i64, JoinRowIndexPair, allocator, typed, right.i64),
        .u8 => |typed| leftJoinRowIndicesTyped(u8, JoinRowIndexPair, allocator, typed, right.u8),
        .u16 => |typed| leftJoinRowIndicesTyped(u16, JoinRowIndexPair, allocator, typed, right.u16),
        .u32 => |typed| leftJoinRowIndicesTyped(u32, JoinRowIndexPair, allocator, typed, right.u32),
        .u64 => |typed| leftJoinRowIndicesTyped(u64, JoinRowIndexPair, allocator, typed, right.u64),
        .usize => |typed| leftJoinRowIndicesTyped(usize, JoinRowIndexPair, allocator, typed, right.usize),
        .isize => |typed| leftJoinRowIndicesTyped(isize, JoinRowIndexPair, allocator, typed, right.isize),
        .f16 => |typed| leftJoinRowIndicesTyped(f16, JoinRowIndexPair, allocator, typed, right.f16),
        .f32 => |typed| leftJoinRowIndicesTyped(f32, JoinRowIndexPair, allocator, typed, right.f32),
        .f64 => |typed| leftJoinRowIndicesTyped(f64, JoinRowIndexPair, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn fullJoinRowIndices(comptime JoinRowIndexPair: type, allocator: std.mem.Allocator, left: anytype, right: anytype) KeyMatchError!JoinRowIndexPair {
    return switch (left) {
        .bool => |typed| fullJoinRowIndicesTyped(bool, JoinRowIndexPair, allocator, typed, right.bool),
        .i8 => |typed| fullJoinRowIndicesTyped(i8, JoinRowIndexPair, allocator, typed, right.i8),
        .i16 => |typed| fullJoinRowIndicesTyped(i16, JoinRowIndexPair, allocator, typed, right.i16),
        .i32 => |typed| fullJoinRowIndicesTyped(i32, JoinRowIndexPair, allocator, typed, right.i32),
        .i64 => |typed| fullJoinRowIndicesTyped(i64, JoinRowIndexPair, allocator, typed, right.i64),
        .u8 => |typed| fullJoinRowIndicesTyped(u8, JoinRowIndexPair, allocator, typed, right.u8),
        .u16 => |typed| fullJoinRowIndicesTyped(u16, JoinRowIndexPair, allocator, typed, right.u16),
        .u32 => |typed| fullJoinRowIndicesTyped(u32, JoinRowIndexPair, allocator, typed, right.u32),
        .u64 => |typed| fullJoinRowIndicesTyped(u64, JoinRowIndexPair, allocator, typed, right.u64),
        .usize => |typed| fullJoinRowIndicesTyped(usize, JoinRowIndexPair, allocator, typed, right.usize),
        .isize => |typed| fullJoinRowIndicesTyped(isize, JoinRowIndexPair, allocator, typed, right.isize),
        .f16 => |typed| fullJoinRowIndicesTyped(f16, JoinRowIndexPair, allocator, typed, right.f16),
        .f32 => |typed| fullJoinRowIndicesTyped(f32, JoinRowIndexPair, allocator, typed, right.f32),
        .f64 => |typed| fullJoinRowIndicesTyped(f64, JoinRowIndexPair, allocator, typed, right.f64),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn semiAntiJoinRowIndices(allocator: std.mem.Allocator, left: anytype, right: anytype, keep_matches: bool) KeyMatchError![]usize {
    return switch (left) {
        .bool => |typed| semiAntiJoinRowIndicesTyped(bool, allocator, typed, right.bool, keep_matches),
        .i8 => |typed| semiAntiJoinRowIndicesTyped(i8, allocator, typed, right.i8, keep_matches),
        .i16 => |typed| semiAntiJoinRowIndicesTyped(i16, allocator, typed, right.i16, keep_matches),
        .i32 => |typed| semiAntiJoinRowIndicesTyped(i32, allocator, typed, right.i32, keep_matches),
        .i64 => |typed| semiAntiJoinRowIndicesTyped(i64, allocator, typed, right.i64, keep_matches),
        .u8 => |typed| semiAntiJoinRowIndicesTyped(u8, allocator, typed, right.u8, keep_matches),
        .u16 => |typed| semiAntiJoinRowIndicesTyped(u16, allocator, typed, right.u16, keep_matches),
        .u32 => |typed| semiAntiJoinRowIndicesTyped(u32, allocator, typed, right.u32, keep_matches),
        .u64 => |typed| semiAntiJoinRowIndicesTyped(u64, allocator, typed, right.u64, keep_matches),
        .usize => |typed| semiAntiJoinRowIndicesTyped(usize, allocator, typed, right.usize, keep_matches),
        .isize => |typed| semiAntiJoinRowIndicesTyped(isize, allocator, typed, right.isize, keep_matches),
        .f16 => |typed| semiAntiJoinRowIndicesTyped(f16, allocator, typed, right.f16, keep_matches),
        .f32 => |typed| semiAntiJoinRowIndicesTyped(f32, allocator, typed, right.f32, keep_matches),
        .f64 => |typed| semiAntiJoinRowIndicesTyped(f64, allocator, typed, right.f64, keep_matches),
        .bf16, .c64, .c128 => error.TypeUnsupported,
    };
}

pub fn semiAntiJoinRowIndicesTyped(
    comptime T: type,
    allocator: std.mem.Allocator,
    left: anytype,
    right: anytype,
    keep_matches: bool,
) (std.mem.Allocator.Error || array_mod.ArrayError || error{InvalidDevice})![]usize {
    if (!left.device().sameDevice(right.device())) return error.InvalidDevice;
    const left_values = try left.values.toOwnedSlice(allocator);
    defer allocator.free(left_values);
    const right_values = try right.values.toOwnedSlice(allocator);
    defer allocator.free(right_values);
    const maybe_left_validity = try validityValues(left, allocator);
    defer if (maybe_left_validity) |validity| allocator.free(validity);
    const maybe_right_validity = try validityValues(right, allocator);
    defer if (maybe_right_validity) |validity| allocator.free(validity);

    var indices: std.ArrayList(usize) = .empty;
    errdefer indices.deinit(allocator);
    for (left_values, 0..) |left_value, left_i| {
        const left_valid = if (maybe_left_validity) |validity| validity[left_i] else true;
        var matched = false;
        if (left_valid) {
            for (right_values, 0..) |right_value, right_i| {
                if (maybe_right_validity) |validity| {
                    if (!validity[right_i]) continue;
                }
                if (groupKeyEqual(T, left_value, right_value)) {
                    matched = true;
                    break;
                }
            }
        }
        if (matched == keep_matches) try indices.append(allocator, left_i);
    }
    return indices.toOwnedSlice(allocator);
}
