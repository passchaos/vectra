const std = @import("std");
const tensor_mod = @import("tensor.zig");
const veyra = @import("veyra");

pub const LinalgError = tensor_mod.TensorError || error{ SingularMatrix, BackendFailure } || std.mem.Allocator.Error;

fn toVeyraMatrix(a: tensor_mod.Tensor(f64)) LinalgError!veyra.Matrix(f64) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    return veyra.Matrix(f64).fromSlice(a.allocator, a.shape[0], a.shape[1], .row_major, a.data) catch return error.BackendFailure;
}

fn fromVeyraMatrix(allocator: std.mem.Allocator, matrix: *const veyra.Matrix(f64)) LinalgError!tensor_mod.Tensor(f64) {
    return tensor_mod.Tensor(f64).fromSlice(allocator, matrix.data, &.{ matrix.rows, matrix.cols });
}

fn mapVeyraInverseError(err: anyerror) LinalgError {
    return switch (err) {
        error.Singular => error.SingularMatrix,
        error.DimensionMismatch => error.NonMatrixTensor,
        error.OutOfMemory => error.OutOfMemory,
        else => error.BackendFailure,
    };
}

pub fn eye(comptime T: type, allocator: std.mem.Allocator, n: usize) LinalgError!tensor_mod.Tensor(T) {
    var out = try tensor_mod.Tensor(T).zeros(allocator, &.{ n, n });
    for (0..n) |i| out.data[i * n + i] = 1;
    return out;
}

pub fn trace(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!T {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (T == f64 and a.shape[0] == a.shape[1]) {
        var matrix = try toVeyraMatrix(@as(tensor_mod.Tensor(f64), a));
        defer matrix.deinit();
        return veyra.trace(f64, matrix.asView()) catch return error.BackendFailure;
    }
    const n = @min(a.shape[0], a.shape[1]);
    var total: T = 0;
    for (0..n) |i| total += a.data[i * a.shape[1] + i];
    return total;
}

pub fn matmul(comptime T: type, a: tensor_mod.Tensor(T), b: tensor_mod.Tensor(T)) LinalgError!tensor_mod.Tensor(T) {
    if (T == f64) return matmulF64(@as(tensor_mod.Tensor(f64), a), @as(tensor_mod.Tensor(f64), b));
    return a.matmul(b);
}

fn matmulF64(a: tensor_mod.Tensor(f64), b: tensor_mod.Tensor(f64)) LinalgError!tensor_mod.Tensor(f64) {
    if (a.shape.len != 2 or b.shape.len != 2) return error.NonMatrixTensor;
    if (a.shape[1] != b.shape[0]) return error.ShapeMismatch;
    var lhs = try toVeyraMatrix(a);
    defer lhs.deinit();
    var rhs = try toVeyraMatrix(b);
    defer rhs.deinit();
    var out_matrix = veyra.Matrix(f64).zeros(a.allocator, a.shape[0], b.shape[1], .row_major) catch return error.BackendFailure;
    defer out_matrix.deinit();
    veyra.matmul(f64, lhs.asView(), rhs.asView(), out_matrix.asMut()) catch return error.BackendFailure;
    return fromVeyraMatrix(a.allocator, &out_matrix);
}

pub fn det(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!T {
    if (a.shape.len != 2 or a.shape[0] != a.shape[1]) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("det currently requires floating-point tensors");
    const n = a.shape[0];
    var m = try a.allocator.dupe(T, a.data);
    defer a.allocator.free(m);
    var sign: T = 1;
    var result: T = 1;
    for (0..n) |i| {
        var pivot = i;
        var pivot_abs = @abs(m[i * n + i]);
        for (i + 1..n) |r| {
            const candidate = @abs(m[r * n + i]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = r;
            }
        }
        if (pivot_abs == 0) return 0;
        if (pivot != i) {
            for (0..n) |c| std.mem.swap(T, &m[i * n + c], &m[pivot * n + c]);
            sign = -sign;
        }
        const diag = m[i * n + i];
        result *= diag;
        for (i + 1..n) |r| {
            const factor = m[r * n + i] / diag;
            for (i..n) |c| m[r * n + c] -= factor * m[i * n + c];
        }
    }
    return result * sign;
}

pub fn inverse(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!tensor_mod.Tensor(T) {
    if (a.shape.len != 2 or a.shape[0] != a.shape[1]) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("inverse currently requires floating-point tensors");
    if (T == f64) return inverseF64(@as(tensor_mod.Tensor(f64), a));

    const n = a.shape[0];
    var aug = try a.allocator.alloc(T, n * n * 2);
    defer a.allocator.free(aug);
    for (0..n) |r| {
        for (0..n) |c| {
            aug[r * (2 * n) + c] = a.data[r * n + c];
            aug[r * (2 * n) + n + c] = if (r == c) 1 else 0;
        }
    }
    for (0..n) |i| {
        var pivot = i;
        var pivot_abs = @abs(aug[i * (2 * n) + i]);
        for (i + 1..n) |r| {
            const candidate = @abs(aug[r * (2 * n) + i]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = r;
            }
        }
        if (pivot_abs == 0) return error.SingularMatrix;
        if (pivot != i) {
            for (0..2 * n) |c| std.mem.swap(T, &aug[i * (2 * n) + c], &aug[pivot * (2 * n) + c]);
        }
        const diag = aug[i * (2 * n) + i];
        for (0..2 * n) |c| aug[i * (2 * n) + c] /= diag;
        for (0..n) |r| {
            if (r == i) continue;
            const factor = aug[r * (2 * n) + i];
            for (0..2 * n) |c| aug[r * (2 * n) + c] -= factor * aug[i * (2 * n) + c];
        }
    }
    var out = try tensor_mod.Tensor(T).empty(a.allocator, &.{ n, n });
    for (0..n) |r| {
        for (0..n) |c| {
            out.data[r * n + c] = aug[r * (2 * n) + n + c];
        }
    }
    return out;
}

fn inverseF64(a: tensor_mod.Tensor(f64)) LinalgError!tensor_mod.Tensor(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var inv = veyra.inverse(f64, a.allocator, matrix.asView()) catch |err| return mapVeyraInverseError(err);
    defer inv.deinit();
    return fromVeyraMatrix(a.allocator, &inv);
}

pub fn solve(comptime T: type, a: tensor_mod.Tensor(T), b: tensor_mod.Tensor(T)) LinalgError!tensor_mod.Tensor(T) {
    var inv = try inverse(T, a);
    defer inv.deinit();
    return matmul(T, inv, b);
}

test "linalg inverse det solve" {
    const gpa = std.testing.allocator;
    var a = try tensor_mod.tensor(f64, gpa, &.{ 4, 7, 2, 6 }, &.{ 2, 2 });
    defer a.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 10), try det(f64, a), 1e-12);
    var inv = try inverse(f64, a);
    defer inv.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), inv.data[0], 1e-12);
    var ident = try matmul(f64, a, inv);
    defer ident.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), ident.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ident.data[1], 1e-12);
}

test "linalg f64 matmul uses Veyra-compatible path" {
    const gpa = std.testing.allocator;
    var a = try tensor_mod.tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try tensor_mod.tensor(f64, gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    var out = try matmul(f64, a, b);
    defer out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 58, 64, 139, 154 }, out.data);
    try std.testing.expectEqual(@as(f64, 212), try trace(f64, out));
}
