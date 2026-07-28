const std = @import("std");

pub fn castToF64(comptime T: type, value: T) f64 {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => @floatCast(value),
        .int, .comptime_int => @floatFromInt(value),
        else => @compileError("numeric dataframe profile requires numeric values"),
    };
}

pub fn isIntegerColumnType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => true,
        else => false,
    };
}

pub fn isOrderedColumnType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .comptime_int, .comptime_float => true,
        else => false,
    };
}

pub fn optionalCast(comptime T: type, value: anytype) ?T {
    const unwrapped = value orelse return null;
    return std.math.cast(T, unwrapped) orelse unreachable;
}

pub fn asofDistance(comptime T: type, lhs: T, rhs: T) f64 {
    return @abs(castToF64(T, lhs) - castToF64(T, rhs));
}

pub fn describeF64(allocator: std.mem.Allocator, v: []const f64) std.mem.Allocator.Error![]f64 {
    var out = try allocator.alloc(f64, 4);
    if (v.len == 0) {
        @memset(out, std.math.nan(f64));
        out[0] = 0;
        return out;
    }
    var total: f64 = 0;
    var min_v = v[0];
    var max_v = v[0];
    for (v) |x| {
        total += x;
        if (x < min_v) min_v = x;
        if (x > max_v) max_v = x;
    }
    out[0] = @floatFromInt(v.len);
    out[1] = total / @as(f64, @floatFromInt(v.len));
    out[2] = min_v;
    out[3] = max_v;
    return out;
}

pub fn describeI64(allocator: std.mem.Allocator, v: []const i64) std.mem.Allocator.Error![]f64 {
    const tmp = try allocator.alloc(f64, v.len);
    defer allocator.free(tmp);
    for (v, tmp) |x, *slot| slot.* = @floatFromInt(x);
    return describeF64(allocator, tmp);
}

pub fn compareSortValues(comptime T: type, lhs: T, rhs: T) i8 {
    if (comptime T == bool) {
        if (lhs == rhs) return 0;
        return if (!lhs and rhs) -1 else 1;
    }
    return switch (@typeInfo(T)) {
        .int, .comptime_int => if (lhs < rhs) -1 else if (rhs < lhs) 1 else 0,
        .float, .comptime_float => compareFloatSortValues(T, lhs, rhs),
        else => @compileError("sort requires bool or ordered numeric column values"),
    };
}

pub fn compareFloatSortValues(comptime T: type, lhs: T, rhs: T) i8 {
    const lhs_nan = std.math.isNan(lhs);
    const rhs_nan = std.math.isNan(rhs);
    if (lhs_nan != rhs_nan) return if (lhs_nan) 1 else -1;
    if (lhs_nan and rhs_nan) return 0;
    if (lhs < rhs) return -1;
    if (rhs < lhs) return 1;
    return 0;
}

pub fn groupKeyEqual(comptime T: type, lhs: T, rhs: T) bool {
    if (comptime @typeInfo(T) == .float) {
        const lhs_nan = std.math.isNan(lhs);
        const rhs_nan = std.math.isNan(rhs);
        return if (lhs_nan or rhs_nan) lhs_nan and rhs_nan else lhs == rhs;
    }
    return lhs == rhs;
}

pub fn findGroupIndex(comptime T: type, keys: []const T, value: T) ?usize {
    for (keys, 0..) |candidate, i| {
        if (groupKeyEqual(T, candidate, value)) return i;
    }
    return null;
}
