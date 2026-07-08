const std = @import("std");
const tensor_mod = @import("tensor.zig");

pub const LinalgError = tensor_mod.TensorError || error{SingularMatrix};

pub fn eye(comptime T: type, allocator: std.mem.Allocator, n: usize) LinalgError!tensor_mod.Tensor(T) {
    var out = try tensor_mod.Tensor(T).zeros(allocator, &.{ n, n });
    for (0..n) |i| out.data[i * n + i] = 1;
    return out;
}

pub fn trace(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!T {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    const n = @min(a.shape[0], a.shape[1]);
    var total: T = 0;
    for (0..n) |i| total += a.data[i * a.shape[1] + i];
    return total;
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

pub fn solve(comptime T: type, a: tensor_mod.Tensor(T), b: tensor_mod.Tensor(T)) LinalgError!tensor_mod.Tensor(T) {
    var inv = try inverse(T, a);
    defer inv.deinit();
    return inv.matmul(b);
}

test "linalg inverse det solve" {
    const gpa = std.testing.allocator;
    var a = try tensor_mod.tensor(f64, gpa, &.{ 4, 7, 2, 6 }, &.{ 2, 2 });
    defer a.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 10), try det(f64, a), 1e-12);
    var inv = try inverse(f64, a);
    defer inv.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), inv.data[0], 1e-12);
    var ident = try a.matmul(inv);
    defer ident.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), ident.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), ident.data[1], 1e-12);
}
