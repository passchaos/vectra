const std = @import("std");
const array_mod = @import("array.zig");

pub const StatsError = array_mod.ArrayError;

pub fn zscore(comptime T: type, x: array_mod.Array(T), dim: ?isize) StatsError!array_mod.Array(T) {
    if (@typeInfo(T) != .float) @compileError("zscore requires floating-point arrays");
    var mean = try x.mean(dim, true);
    defer mean.deinit();
    var std_t = try x.stddev(dim, true, 0);
    defer std_t.deinit();
    var centered = try x.sub(mean);
    defer centered.deinit();
    return centered.div(std_t);
}

pub fn normalize(comptime T: type, x: array_mod.Array(T), dim: ?isize) StatsError!array_mod.Array(T) {
    if (@typeInfo(T) != .float) @compileError("normalize requires floating-point arrays");
    var min_t = try x.min(dim, true);
    defer min_t.deinit();
    var max_t = try x.max(dim, true);
    defer max_t.deinit();
    var shifted = try x.sub(min_t);
    defer shifted.deinit();
    var range = try max_t.sub(min_t);
    defer range.deinit();
    return shifted.div(range);
}

pub fn pearsonr(comptime T: type, x: array_mod.Array(T), y: array_mod.Array(T)) StatsError!T {
    if (@typeInfo(T) != .float) @compileError("pearsonr requires floating-point arrays");
    if (x.data.len != y.data.len) return error.ShapeMismatch;
    if (x.data.len == 0) return error.EmptyArray;
    var mean_x: T = 0;
    var mean_y: T = 0;
    for (x.data, y.data) |a, b| {
        mean_x += a;
        mean_y += b;
    }
    const n: T = @floatFromInt(x.data.len);
    mean_x /= n;
    mean_y /= n;
    var num: T = 0;
    var dx2: T = 0;
    var dy2: T = 0;
    for (x.data, y.data) |a, b| {
        const dx = a - mean_x;
        const dy = b - mean_y;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    return num / std.math.sqrt(dx2 * dy2);
}

test "stats zscore pearson" {
    const gpa = std.testing.allocator;
    var x = try array_mod.array(f64, gpa, &.{ 1, 2, 3 }, &.{3});
    defer x.deinit();
    var z = try zscore(f64, x, null);
    defer z.deinit();
    var zsum = try z.sum(null, false);
    defer zsum.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0), zsum.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), try pearsonr(f64, x, x), 1e-12);
}
