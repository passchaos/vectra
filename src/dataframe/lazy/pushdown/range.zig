//! Range-predicate helpers for Parquet scan pushdown.
//!
//! Keeping scalar/between/drop-filter range normalization here makes the main
//! lazy scan planner focus on dependency discovery and operation ordering.

const std = @import("std");
const options_mod = @import("../../../dataframe_options.zig");

const DeviceColumnCompareOp = options_mod.DeviceColumnCompareOp;
const DeviceParquetRangeFilter = options_mod.DeviceParquetRangeFilter;
const DeviceScalar = options_mod.DeviceScalar;
const ParquetRangePredicate = options_mod.ParquetRangePredicate;
const Range = options_mod.Range;

pub fn mergeRangePredicate(
    allocator: std.mem.Allocator,
    current: *?DeviceParquetRangeFilter,
    column_name: []const u8,
    predicate: ParquetRangePredicate,
) std.mem.Allocator.Error!void {
    if (current.*) |*existing| {
        if (!std.mem.eql(u8, existing.column, column_name)) return;
        if (intersectRangePredicates(existing.predicate, predicate)) |merged| {
            existing.predicate = merged;
        }
        return;
    }
    current.* = .{
        .column = try allocator.dupe(u8, column_name),
        .predicate = predicate,
    };
}

fn intersectRangePredicates(left: ParquetRangePredicate, right: ParquetRangePredicate) ?ParquetRangePredicate {
    if (std.meta.activeTag(left) != std.meta.activeTag(right)) return null;
    return switch (left) {
        .bool => |range| .{ .bool = intersectBoolRange(range, right.bool) orelse return null },
        .i8 => |range| .{ .i8 = intersectRange(i8, range, right.i8) orelse return null },
        .i16 => |range| .{ .i16 = intersectRange(i16, range, right.i16) orelse return null },
        .i32 => |range| .{ .i32 = intersectRange(i32, range, right.i32) orelse return null },
        .i64 => |range| .{ .i64 = intersectRange(i64, range, right.i64) orelse return null },
        .u8 => |range| .{ .u8 = intersectRange(u8, range, right.u8) orelse return null },
        .u16 => |range| .{ .u16 = intersectRange(u16, range, right.u16) orelse return null },
        .u32 => |range| .{ .u32 = intersectRange(u32, range, right.u32) orelse return null },
        .u64 => |range| .{ .u64 = intersectRange(u64, range, right.u64) orelse return null },
        .usize => |range| .{ .usize = intersectRange(usize, range, right.usize) orelse return null },
        .isize => |range| .{ .isize = intersectRange(isize, range, right.isize) orelse return null },
        .f16 => |range| .{ .f16 = intersectRange(f16, range, right.f16) orelse return null },
        .f32 => |range| .{ .f32 = intersectRange(f32, range, right.f32) orelse return null },
        .f64 => |range| .{ .f64 = intersectRange(f64, range, right.f64) orelse return null },
        .bf16, .c64, .c128 => null,
    };
}

fn intersectBoolRange(left: Range(bool), right: Range(bool)) ?Range(bool) {
    const min_value = maxOptionalBool(left.min, right.min);
    const max_value = minOptionalBool(left.max, right.max);
    if (min_value == true and max_value == false) return null;
    return .{ .min = min_value, .max = max_value };
}

fn intersectRange(comptime T: type, left: Range(T), right: Range(T)) ?Range(T) {
    const min_value = maxOptional(T, left.min, right.min);
    const max_value = minOptional(T, left.max, right.max);
    if (min_value) |min_unwrapped| {
        if (max_value) |max_unwrapped| {
            if (min_unwrapped > max_unwrapped) return null;
        }
    }
    return .{ .min = min_value, .max = max_value };
}

fn maxOptionalBool(left: ?bool, right: ?bool) ?bool {
    if (left) |left_value| {
        if (right) |right_value| return left_value or right_value;
        return left_value;
    }
    return right;
}

fn minOptionalBool(left: ?bool, right: ?bool) ?bool {
    if (left) |left_value| {
        if (right) |right_value| return left_value and right_value;
        return left_value;
    }
    return right;
}

fn maxOptional(comptime T: type, left: ?T, right: ?T) ?T {
    if (left) |left_value| {
        if (right) |right_value| return if (left_value >= right_value) left_value else right_value;
        return left_value;
    }
    return right;
}

fn minOptional(comptime T: type, left: ?T, right: ?T) ?T {
    if (left) |left_value| {
        if (right) |right_value| return if (left_value <= right_value) left_value else right_value;
        return left_value;
    }
    return right;
}

pub fn parquetRangePredicateFromScalar(scalar: DeviceScalar, op: DeviceColumnCompareOp) ?ParquetRangePredicate {
    return switch (scalar) {
        .bool => |value| blk: {
            const exact = switch (op) {
                .eq => value,
                .ne => !value,
                .gt, .ge, .lt, .le => break :blk null,
            };
            break :blk .{ .bool = .{ .min = exact, .max = exact } };
        },
        .i8 => |value| if (rangeFromScalarPredicate(i8, value, op)) |range| .{ .i8 = range } else null,
        .i16 => |value| if (rangeFromScalarPredicate(i16, value, op)) |range| .{ .i16 = range } else null,
        .i32 => |value| if (rangeFromScalarPredicate(i32, value, op)) |range| .{ .i32 = range } else null,
        .i64 => |value| if (rangeFromScalarPredicate(i64, value, op)) |range| .{ .i64 = range } else null,
        .u8 => |value| if (rangeFromScalarPredicate(u8, value, op)) |range| .{ .u8 = range } else null,
        .u16 => |value| if (rangeFromScalarPredicate(u16, value, op)) |range| .{ .u16 = range } else null,
        .u32 => |value| if (rangeFromScalarPredicate(u32, value, op)) |range| .{ .u32 = range } else null,
        .u64 => |value| if (rangeFromScalarPredicate(u64, value, op)) |range| .{ .u64 = range } else null,
        .usize => |value| if (rangeFromScalarPredicate(usize, value, op)) |range| .{ .usize = range } else null,
        .isize => |value| if (rangeFromScalarPredicate(isize, value, op)) |range| .{ .isize = range } else null,
        .f16 => |value| if (rangeFromScalarPredicate(f16, value, op)) |range| .{ .f16 = range } else null,
        .f32 => |value| if (rangeFromScalarPredicate(f32, value, op)) |range| .{ .f32 = range } else null,
        .f64 => |value| if (rangeFromScalarPredicate(f64, value, op)) |range| .{ .f64 = range } else null,
        .bf16, .c64, .c128 => null,
    };
}

pub fn parquetRangePredicateFromDroppedScalar(scalar: DeviceScalar, op: DeviceColumnCompareOp) ?ParquetRangePredicate {
    return parquetRangePredicateFromScalar(scalar, invertCompareOp(op));
}

fn invertCompareOp(op: DeviceColumnCompareOp) DeviceColumnCompareOp {
    return switch (op) {
        .eq => .ne,
        .ne => .eq,
        .gt => .le,
        .ge => .lt,
        .lt => .ge,
        .le => .gt,
    };
}

pub fn parquetRangePredicateFromBounds(lower: DeviceScalar, upper: DeviceScalar, lower_inclusive: bool, upper_inclusive: bool) ?ParquetRangePredicate {
    if (std.meta.activeTag(lower) != std.meta.activeTag(upper)) return null;
    return switch (lower) {
        .bool => |value| if (rangeFromBoolBounds(value, upper.bool, lower_inclusive, upper_inclusive)) |range| .{ .bool = range } else null,
        .i8 => |value| if (rangeFromIntegerBounds(i8, value, upper.i8, lower_inclusive, upper_inclusive)) |range| .{ .i8 = range } else null,
        .i16 => |value| if (rangeFromIntegerBounds(i16, value, upper.i16, lower_inclusive, upper_inclusive)) |range| .{ .i16 = range } else null,
        .i32 => |value| if (rangeFromIntegerBounds(i32, value, upper.i32, lower_inclusive, upper_inclusive)) |range| .{ .i32 = range } else null,
        .i64 => |value| if (rangeFromIntegerBounds(i64, value, upper.i64, lower_inclusive, upper_inclusive)) |range| .{ .i64 = range } else null,
        .u8 => |value| if (rangeFromIntegerBounds(u8, value, upper.u8, lower_inclusive, upper_inclusive)) |range| .{ .u8 = range } else null,
        .u16 => |value| if (rangeFromIntegerBounds(u16, value, upper.u16, lower_inclusive, upper_inclusive)) |range| .{ .u16 = range } else null,
        .u32 => |value| if (rangeFromIntegerBounds(u32, value, upper.u32, lower_inclusive, upper_inclusive)) |range| .{ .u32 = range } else null,
        .u64 => |value| if (rangeFromIntegerBounds(u64, value, upper.u64, lower_inclusive, upper_inclusive)) |range| .{ .u64 = range } else null,
        .usize => |value| if (rangeFromIntegerBounds(usize, value, upper.usize, lower_inclusive, upper_inclusive)) |range| .{ .usize = range } else null,
        .isize => |value| if (rangeFromIntegerBounds(isize, value, upper.isize, lower_inclusive, upper_inclusive)) |range| .{ .isize = range } else null,
        .f16 => |value| if (validFloatBounds(f16, value, upper.f16)) .{ .f16 = .{ .min = value, .max = upper.f16 } } else null,
        .f32 => |value| if (validFloatBounds(f32, value, upper.f32)) .{ .f32 = .{ .min = value, .max = upper.f32 } } else null,
        .f64 => |value| if (validFloatBounds(f64, value, upper.f64)) .{ .f64 = .{ .min = value, .max = upper.f64 } } else null,
        .bf16, .c64, .c128 => null,
    };
}

fn rangeFromBoolBounds(lower: bool, upper: bool, lower_inclusive: bool, upper_inclusive: bool) ?Range(bool) {
    const min_value = if (lower_inclusive) lower else if (!lower) true else return null;
    const max_value = if (upper_inclusive) upper else if (upper) false else return null;
    if (min_value and !max_value) return null;
    return .{ .min = min_value, .max = max_value };
}

fn rangeFromIntegerBounds(comptime T: type, lower: T, upper: T, lower_inclusive: bool, upper_inclusive: bool) ?Range(T) {
    const min_value = if (lower_inclusive) lower else nextInteger(T, lower) orelse return null;
    const max_value = if (upper_inclusive) upper else previousInteger(T, upper) orelse return null;
    if (min_value > max_value) return null;
    return .{ .min = min_value, .max = max_value };
}

fn nextInteger(comptime T: type, value: T) ?T {
    if (value == std.math.maxInt(T)) return null;
    return value + 1;
}

fn previousInteger(comptime T: type, value: T) ?T {
    if (value == std.math.minInt(T)) return null;
    return value - 1;
}

fn validFloatBounds(comptime T: type, lower: T, upper: T) bool {
    return !std.math.isNan(lower) and !std.math.isNan(upper) and lower <= upper;
}

fn rangeFromScalarPredicate(comptime T: type, value: T, op: DeviceColumnCompareOp) ?Range(T) {
    const type_info = comptime @typeInfo(T);
    if (type_info == .float and std.math.isNan(value)) return null;
    return switch (op) {
        .eq => .{ .min = value, .max = value },
        .gt => if (type_info == .int) .{ .min = nextInteger(T, value) orelse return null } else .{ .min = value },
        .ge => .{ .min = value },
        .lt => if (type_info == .int) .{ .max = previousInteger(T, value) orelse return null } else .{ .max = value },
        .le => .{ .max = value },
        .ne => null,
    };
}
