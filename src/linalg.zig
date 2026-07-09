const std = @import("std");
const array_mod = @import("array.zig");
const veyra = @import("veyra");

pub const LinalgError = array_mod.ArrayError || error{ SingularMatrix, NotPositiveDefinite, BackendFailure } || std.mem.Allocator.Error;

pub const MatrixNormOrder = array_mod.MatrixNormOrder;
pub const Triangle = array_mod.Triangle;
pub const Diagonal = array_mod.Diagonal;
pub const QrResult = array_mod.QrResult;
pub const SvdResult = array_mod.SvdResult;
pub const EighResult = array_mod.EighResult;
pub const LuResult = array_mod.LuResult;

fn toVeyraMatrix(a: array_mod.Array(f64)) LinalgError!veyra.Matrix(f64) {
    if (a.shape.len != 2) return error.NonMatrixArray;
    return veyra.Matrix(f64).fromSlice(a.allocator, a.shape[0], a.shape[1], .row_major, a.data) catch return error.BackendFailure;
}

fn fromVeyraMatrix(allocator: std.mem.Allocator, matrix: *const veyra.Matrix(f64)) LinalgError!array_mod.Array(f64) {
    return array_mod.Array(f64).fromSlice(allocator, matrix.data, &.{ matrix.rows, matrix.cols });
}

fn toVeyraVector(x: array_mod.Array(f64)) LinalgError!veyra.Vector(f64) {
    if (x.shape.len != 1) return error.NonVectorArray;
    return veyra.Vector(f64).fromSlice(x.allocator, x.data) catch return error.BackendFailure;
}

fn fromVeyraVector(allocator: std.mem.Allocator, vector: *const veyra.Vector(f64)) LinalgError!array_mod.Array(f64) {
    return array_mod.Array(f64).fromSlice(allocator, vector.data, &.{vector.len()});
}

pub fn eye(comptime T: type, allocator: std.mem.Allocator, n: usize) LinalgError!array_mod.Array(T) {
    var out = try array_mod.Array(T).zeros(allocator, &.{ n, n });
    for (0..n) |i| out.data[i * n + i] = 1;
    return out;
}

pub fn trace(comptime T: type, a: array_mod.Array(T)) LinalgError!T {
    if (a.shape.len != 2) return error.NonMatrixArray;
    if (T == f64 and a.shape[0] == a.shape[1]) {
        var matrix = try toVeyraMatrix(@as(array_mod.Array(f64), a));
        defer matrix.deinit();
        return veyra.trace(f64, matrix.asView()) catch return error.BackendFailure;
    }
    const n = @min(a.shape[0], a.shape[1]);
    var total: T = 0;
    for (0..n) |i| total += a.data[i * a.shape[1] + i];
    return total;
}

pub fn matmul(comptime T: type, a: array_mod.Array(T), b: array_mod.Array(T)) LinalgError!array_mod.Array(T) {
    if (T == f64) return matmulF64(@as(array_mod.Array(f64), a), @as(array_mod.Array(f64), b));
    return a.matmul(b);
}

fn matmulF64(a: array_mod.Array(f64), b: array_mod.Array(f64)) LinalgError!array_mod.Array(f64) {
    if (a.shape.len != 2 or b.shape.len != 2) return error.NonMatrixArray;
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

pub fn matvec(comptime T: type, a: array_mod.Array(T), x: array_mod.Array(T)) LinalgError!array_mod.Array(T) {
    if (a.shape.len != 2) return error.NonMatrixArray;
    if (x.shape.len != 1) return error.NonVectorArray;
    if (a.shape[1] != x.shape[0]) return error.ShapeMismatch;
    if (T == f64) return matvecF64(@as(array_mod.Array(f64), a), @as(array_mod.Array(f64), x));

    const out = try array_mod.Array(T).zeros(a.allocator, &.{a.shape[0]});
    for (0..a.shape[0]) |r| {
        var acc: T = 0;
        for (0..a.shape[1]) |c| acc += a.data[r * a.shape[1] + c] * x.data[c];
        out.data[r] = acc;
    }
    return out;
}

fn matvecF64(a: array_mod.Array(f64), x: array_mod.Array(f64)) LinalgError!array_mod.Array(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var vector = try toVeyraVector(x);
    defer vector.deinit();
    var out_vector = veyra.Vector(f64).zeros(a.allocator, a.shape[0]) catch return error.BackendFailure;
    defer out_vector.deinit();
    veyra.matvec(f64, matrix.asView(), vector.asView(), out_vector.asMut()) catch return error.BackendFailure;
    return fromVeyraVector(a.allocator, &out_vector);
}

pub fn cholesky(comptime T: type, a: array_mod.Array(T)) LinalgError!array_mod.Array(T) {
    return a.cholesky();
}

pub fn qr(comptime T: type, a: array_mod.Array(T)) LinalgError!QrResult(T) {
    return a.qr();
}

pub fn svd(comptime T: type, a: array_mod.Array(T), tolerance: T) LinalgError!SvdResult(T) {
    return a.svd(tolerance);
}

pub fn singularValues(comptime T: type, a: array_mod.Array(T), tolerance: T) LinalgError!array_mod.Array(T) {
    return a.singularValues(tolerance);
}

pub fn matrixRank(comptime T: type, a: array_mod.Array(T), tolerance: T) LinalgError!usize {
    return a.matrixRank(tolerance);
}

pub fn cond(comptime T: type, a: array_mod.Array(T), tolerance: T) LinalgError!T {
    return a.cond(tolerance);
}

pub fn pinv(comptime T: type, a: array_mod.Array(T), tolerance: T) LinalgError!array_mod.Array(T) {
    return a.pinv(tolerance);
}

pub fn matrixNorm(comptime T: type, a: array_mod.Array(T), order: MatrixNormOrder, tolerance: T) LinalgError!T {
    return a.matrixNorm(order, tolerance);
}

pub fn eigh(comptime T: type, a: array_mod.Array(T), max_sweeps: usize, tolerance: T) LinalgError!EighResult(T) {
    return a.eigh(max_sweeps, tolerance);
}

pub fn eigvalsh(comptime T: type, a: array_mod.Array(T), max_sweeps: usize, tolerance: T) LinalgError!array_mod.Array(T) {
    return a.eigvalsh(max_sweeps, tolerance);
}

pub fn lstsq(comptime T: type, a: array_mod.Array(T), b: array_mod.Array(T), tolerance: T) LinalgError!array_mod.Array(T) {
    return a.lstsq(b, tolerance);
}

pub fn lu(comptime T: type, a: array_mod.Array(T)) LinalgError!LuResult(T) {
    return a.lu();
}

pub fn solveTriangular(comptime T: type, a: array_mod.Array(T), b: array_mod.Array(T), triangle: Triangle, diagonal: Diagonal) LinalgError!array_mod.Array(T) {
    return a.solveTriangular(b, triangle, diagonal);
}

pub fn det(comptime T: type, a: array_mod.Array(T)) LinalgError!T {
    return a.det();
}

pub fn inverse(comptime T: type, a: array_mod.Array(T)) LinalgError!array_mod.Array(T) {
    return a.inverse();
}

pub fn solve(comptime T: type, a: array_mod.Array(T), b: array_mod.Array(T)) LinalgError!array_mod.Array(T) {
    return a.solve(b);
}

test "linalg inverse det solve" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 4, 7, 2, 6 }, &.{ 2, 2 });
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
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try array_mod.Array(f64).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    var out = try matmul(f64, a, b);
    defer out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, out.shape);
    try std.testing.expectEqualSlices(f64, &.{ 58, 64, 139, 154 }, out.data);
    try std.testing.expectEqual(@as(f64, 212), try trace(f64, out));
}

test "linalg matvec and cholesky use Veyra-compatible paths" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var x = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer x.deinit();
    var y = try matvec(f64, a, x);
    defer y.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 14, 32 }, y.data);

    var spd = try array_mod.Array(f64).fromSlice(gpa, &.{ 25, 15, -5, 15, 18, 0, -5, 0, 11 }, &.{ 3, 3 });
    defer spd.deinit();
    var l = try cholesky(f64, spd);
    defer l.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), l.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), l.data[3], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1), l.data[6], 1e-12);
    var lt = try l.transpose();
    defer lt.deinit();
    var reconstructed = try matmul(f64, l, lt);
    defer reconstructed.deinit();
    try std.testing.expect(try reconstructed.allclose(spd, 1e-12, 1e-12));
}

test "linalg qr reconstructs matrix" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer a.deinit();
    var factors = try qr(f64, a);
    defer factors.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, factors.q.shape);
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, factors.r.shape);
    var reconstructed = try matmul(f64, factors.q, factors.r);
    defer reconstructed.deinit();
    try std.testing.expect(try reconstructed.allclose(a, 1e-10, 1e-10));
}

test "linalg svd reconstructs matrix" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer a.deinit();
    var factors = try svd(f64, a, 1e-12);
    defer factors.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, factors.u.shape);
    try std.testing.expectEqualSlices(usize, &.{2}, factors.s.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, factors.vt.shape);

    var sigma = try array_mod.Array(f64).zeros(gpa, &.{ 2, 2 });
    defer sigma.deinit();
    sigma.data[0] = factors.s.data[0];
    sigma.data[3] = factors.s.data[1];
    var us = try matmul(f64, factors.u, sigma);
    defer us.deinit();
    var reconstructed = try matmul(f64, us, factors.vt);
    defer reconstructed.deinit();
    try std.testing.expect(try reconstructed.allclose(a, 1e-10, 1e-10));
}

test "linalg lstsq solves vector and matrix rhs" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer a.deinit();
    var b = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 2 }, &.{3});
    defer b.deinit();
    var x = try lstsq(f64, a, b, 1e-12);
    defer x.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), x.data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), x.data[1], 1e-10);

    var bm = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 2, 1, 2, 0 }, &.{ 3, 2 });
    defer bm.deinit();
    var xm = try lstsq(f64, a, bm, 1e-12);
    defer xm.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, xm.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), xm.data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), xm.data[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), xm.data[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), xm.data[3], 1e-10);
}

test "linalg singular values rank condition and pinv" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 3, 0, 0, 2 }, &.{ 2, 2 });
    defer a.deinit();

    var values = try singularValues(f64, a, 1e-12);
    defer values.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, values.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 3), values.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), values.data[1], 1e-12);
    try std.testing.expectEqual(@as(usize, 2), try matrixRank(f64, a, 1e-12));
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), try cond(f64, a, 1e-12), 1e-12);

    var p = try pinv(f64, a, 1e-12);
    defer p.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, p.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), p.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), p.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0), p.data[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), p.data[3], 1e-12);

    var rect = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer rect.deinit();
    var rect_p = try pinv(f64, rect, 1e-12);
    defer rect_p.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, rect_p.shape);
    var projected = try matmul(f64, rect_p, rect);
    defer projected.deinit();
    var ident = try eye(f64, gpa, 2);
    defer ident.deinit();
    try std.testing.expect(try projected.allclose(ident, 1e-10, 1e-10));
}

test "linalg matrix norms use Veyra-compatible paths" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, -2, 3, -4, 5, -6 }, &.{ 2, 3 });
    defer a.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(91.0)), try matrixNorm(f64, a, .fro, 1e-12), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9), try matrixNorm(f64, a, .one, 1e-12), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15), try matrixNorm(f64, a, .inf, 1e-12), 1e-12);
}

test "linalg symmetric eigen decomposition" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 1, 1, 2 }, &.{ 2, 2 });
    defer a.deinit();
    var result = try eigh(f64, a, 64, 1e-12);
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, result.values.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, result.vectors.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.values.data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3), result.values.data[1], 1e-10);

    var diag = try array_mod.Array(f64).zeros(gpa, &.{ 2, 2 });
    defer diag.deinit();
    diag.data[0] = result.values.data[0];
    diag.data[3] = result.values.data[1];
    var vd = try matmul(f64, result.vectors, diag);
    defer vd.deinit();
    var vt = try result.vectors.transpose();
    defer vt.deinit();
    var reconstructed = try matmul(f64, vd, vt);
    defer reconstructed.deinit();
    try std.testing.expect(try reconstructed.allclose(a, 1e-10, 1e-10));

    var values_only = try eigvalsh(f64, a, 64, 1e-12);
    defer values_only.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), values_only.data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3), values_only.data[1], 1e-10);
}

test "linalg lu reconstructs and det uses Veyra path" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 0, 2, 1, 2, 1, 1, 1, 1, 0 }, &.{ 3, 3 });
    defer a.deinit();
    var factors = try lu(f64, a);
    defer factors.deinit();
    var lu_product = try matmul(f64, factors.l, factors.u);
    defer lu_product.deinit();
    var reconstructed = try matmul(f64, factors.p, lu_product);
    defer reconstructed.deinit();
    try std.testing.expect(try reconstructed.allclose(a, 1e-10, 1e-10));
    try std.testing.expectApproxEqAbs(@as(f64, 3), try det(f64, a), 1e-12);
}

test "linalg solve uses LU for vector and matrix rhs" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f64).fromSlice(gpa, &.{ 3, 1, 1, 2 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try array_mod.Array(f64).fromSlice(gpa, &.{ 9, 8 }, &.{2});
    defer b.deinit();
    var x = try solve(f64, a, b);
    defer x.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2), x.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), x.data[1], 1e-12);

    var bm = try array_mod.Array(f64).fromSlice(gpa, &.{ 9, 4, 8, 5 }, &.{ 2, 2 });
    defer bm.deinit();
    var xm = try solve(f64, a, bm);
    defer xm.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, xm.shape);
    var ax = try matmul(f64, a, xm);
    defer ax.deinit();
    try std.testing.expect(try ax.allclose(bm, 1e-12, 1e-12));
}

test "linalg solveTriangular handles vector and matrix rhs" {
    const gpa = std.testing.allocator;
    var lower = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 0, 0, -1, 3, 0, 4, 2, 5 }, &.{ 3, 3 });
    defer lower.deinit();
    var rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 2, 25 }, &.{3});
    defer rhs.deinit();
    var x = try solveTriangular(f64, lower, rhs, .lower, .non_unit);
    defer x.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), x.data[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), x.data[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.8), x.data[2], 1e-12);
    var check = try matvec(f64, lower, x);
    defer check.deinit();
    try std.testing.expect(try check.allclose(rhs, 1e-12, 1e-12));

    var rhs_matrix = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 4, 2, 4, 25, 50 }, &.{ 3, 2 });
    defer rhs_matrix.deinit();
    var xm = try solveTriangular(f64, lower, rhs_matrix, .lower, .non_unit);
    defer xm.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, xm.shape);
    var check_m = try matmul(f64, lower, xm);
    defer check_m.deinit();
    try std.testing.expect(try check_m.allclose(rhs_matrix, 1e-12, 1e-12));

    var unit_upper = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, -1, 0, 1, 3, 0, 0, 1 }, &.{ 3, 3 });
    defer unit_upper.deinit();
    var rhs_upper = try array_mod.Array(f64).fromSlice(gpa, &.{ 5, 7, 2 }, &.{3});
    defer rhs_upper.deinit();
    var xu = try solveTriangular(f64, unit_upper, rhs_upper, .upper, .unit);
    defer xu.deinit();
    var check_u = try matvec(f64, unit_upper, xu);
    defer check_u.deinit();
    try std.testing.expect(try check_u.allclose(rhs_upper, 1e-12, 1e-12));
}
