const std = @import("std");
const tensor_mod = @import("tensor.zig");
const veyra = @import("veyra");

pub const LinalgError = tensor_mod.TensorError || error{ SingularMatrix, NotPositiveDefinite, BackendFailure } || std.mem.Allocator.Error;

pub const MatrixNormOrder = enum {
    fro,
    one,
    inf,
    two,
    nuclear,
};

fn toVeyraMatrix(a: tensor_mod.Tensor(f64)) LinalgError!veyra.Matrix(f64) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    return veyra.Matrix(f64).fromSlice(a.allocator, a.shape[0], a.shape[1], .row_major, a.data) catch return error.BackendFailure;
}

fn fromVeyraMatrix(allocator: std.mem.Allocator, matrix: *const veyra.Matrix(f64)) LinalgError!tensor_mod.Tensor(f64) {
    return tensor_mod.Tensor(f64).fromSlice(allocator, matrix.data, &.{ matrix.rows, matrix.cols });
}

fn toVeyraVector(x: tensor_mod.Tensor(f64)) LinalgError!veyra.Vector(f64) {
    if (x.shape.len != 1) return error.NonVectorTensor;
    return veyra.Vector(f64).fromSlice(x.allocator, x.data) catch return error.BackendFailure;
}

fn fromVeyraVector(allocator: std.mem.Allocator, vector: *const veyra.Vector(f64)) LinalgError!tensor_mod.Tensor(f64) {
    return tensor_mod.Tensor(f64).fromSlice(allocator, vector.data, &.{vector.len()});
}

pub fn QrResult(comptime T: type) type {
    return struct {
        q: tensor_mod.Tensor(T),
        r: tensor_mod.Tensor(T),

        pub fn deinit(self: *@This()) void {
            self.q.deinit();
            self.r.deinit();
            self.* = undefined;
        }
    };
}

pub fn SvdResult(comptime T: type) type {
    return struct {
        u: tensor_mod.Tensor(T),
        s: tensor_mod.Tensor(T),
        vt: tensor_mod.Tensor(T),

        pub fn deinit(self: *@This()) void {
            self.u.deinit();
            self.s.deinit();
            self.vt.deinit();
            self.* = undefined;
        }
    };
}

pub fn EighResult(comptime T: type) type {
    return struct {
        values: tensor_mod.Tensor(T),
        vectors: tensor_mod.Tensor(T),

        pub fn deinit(self: *@This()) void {
            self.values.deinit();
            self.vectors.deinit();
            self.* = undefined;
        }
    };
}

fn mapVeyraInverseError(err: anyerror) LinalgError {
    return switch (err) {
        error.Singular => error.SingularMatrix,
        error.DimensionMismatch => error.NonMatrixTensor,
        error.OutOfMemory => error.OutOfMemory,
        else => error.BackendFailure,
    };
}

fn mapVeyraError(err: anyerror) LinalgError {
    return switch (err) {
        error.Singular => error.SingularMatrix,
        error.NotPositiveDefinite => error.NotPositiveDefinite,
        error.DimensionMismatch => error.ShapeMismatch,
        error.IndexOutOfBounds => error.IndexOutOfBounds,
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

pub fn matvec(comptime T: type, a: tensor_mod.Tensor(T), x: tensor_mod.Tensor(T)) LinalgError!tensor_mod.Tensor(T) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (x.shape.len != 1) return error.NonVectorTensor;
    if (a.shape[1] != x.shape[0]) return error.ShapeMismatch;
    if (T == f64) return matvecF64(@as(tensor_mod.Tensor(f64), a), @as(tensor_mod.Tensor(f64), x));

    var out = try tensor_mod.Tensor(T).zeros(a.allocator, &.{a.shape[0]});
    for (0..a.shape[0]) |r| {
        var acc: T = 0;
        for (0..a.shape[1]) |c| acc += a.data[r * a.shape[1] + c] * x.data[c];
        out.data[r] = acc;
    }
    return out;
}

fn matvecF64(a: tensor_mod.Tensor(f64), x: tensor_mod.Tensor(f64)) LinalgError!tensor_mod.Tensor(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var vector = try toVeyraVector(x);
    defer vector.deinit();
    var out_vector = veyra.Vector(f64).zeros(a.allocator, a.shape[0]) catch return error.BackendFailure;
    defer out_vector.deinit();
    veyra.matvec(f64, matrix.asView(), vector.asView(), out_vector.asMut()) catch return error.BackendFailure;
    return fromVeyraVector(a.allocator, &out_vector);
}

pub fn cholesky(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!tensor_mod.Tensor(T) {
    if (a.shape.len != 2 or a.shape[0] != a.shape[1]) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("cholesky requires floating-point tensors");
    if (T == f64) return choleskyF64(@as(tensor_mod.Tensor(f64), a));

    const n = a.shape[0];
    var out = try tensor_mod.Tensor(T).zeros(a.allocator, &.{ n, n });
    for (0..n) |i| {
        for (0..i + 1) |j| {
            var sum: T = 0;
            for (0..j) |k| sum += out.data[i * n + k] * out.data[j * n + k];
            if (i == j) {
                const value = a.data[i * n + i] - sum;
                if (value <= 0) return error.NotPositiveDefinite;
                out.data[i * n + j] = std.math.sqrt(value);
            } else {
                const denom = out.data[j * n + j];
                if (denom == 0) return error.NotPositiveDefinite;
                out.data[i * n + j] = (a.data[i * n + j] - sum) / denom;
            }
        }
    }
    return out;
}

fn choleskyF64(a: tensor_mod.Tensor(f64)) LinalgError!tensor_mod.Tensor(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var factorization = veyra.cholesky(f64, a.allocator, matrix.asView()) catch |err| return mapVeyraError(err);
    defer factorization.deinit();
    var out = try tensor_mod.Tensor(f64).zeros(a.allocator, &.{ a.shape[0], a.shape[1] });
    errdefer out.deinit();
    const l = factorization.lView();
    for (0..a.shape[0]) |r| {
        for (0..r + 1) |c| out.data[r * a.shape[1] + c] = l.get(r, c);
    }
    return out;
}

pub fn qr(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!QrResult(T) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("qr requires floating-point tensors");
    if (T == f64) return qrF64(@as(tensor_mod.Tensor(f64), a));
    return qrReference(T, a);
}

fn qrF64(a: tensor_mod.Tensor(f64)) LinalgError!QrResult(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var factorization = veyra.qr(f64, a.allocator, matrix.asView()) catch |err| return mapVeyraError(err);
    defer factorization.deinit();

    var q_matrix = veyra.Matrix(f64).identity(a.allocator, a.shape[0], .row_major) catch return error.BackendFailure;
    defer q_matrix.deinit();
    var q_out_matrix = veyra.Matrix(f64).zeros(a.allocator, a.shape[0], a.shape[0], .row_major) catch return error.BackendFailure;
    defer q_out_matrix.deinit();
    factorization.applyQMatrix(q_matrix.asView(), q_out_matrix.asMut()) catch |err| return mapVeyraError(err);

    var q = try fromVeyraMatrix(a.allocator, &q_out_matrix);
    errdefer q.deinit();
    var r = try tensor_mod.Tensor(f64).zeros(a.allocator, &.{ a.shape[0], a.shape[1] });
    errdefer r.deinit();
    const rv = factorization.rView();
    for (0..a.shape[0]) |row| {
        for (row..a.shape[1]) |col| {
            if (row < rv.rows and col < rv.cols) r.data[row * a.shape[1] + col] = rv.get(row, col);
        }
    }
    return .{ .q = q, .r = r };
}

fn qrReference(comptime T: type, a: tensor_mod.Tensor(T)) LinalgError!QrResult(T) {
    const m = a.shape[0];
    const n = a.shape[1];
    var q = try tensor_mod.Tensor(T).zeros(a.allocator, &.{ m, m });
    errdefer q.deinit();
    var r = try tensor_mod.Tensor(T).zeros(a.allocator, &.{ m, n });
    errdefer r.deinit();

    // Classical Gram-Schmidt for the first n columns; complete remaining Q columns from basis vectors.
    for (0..m) |basis_col| {
        for (0..m) |row| q.data[row * m + basis_col] = if (row == basis_col) 1 else 0;
    }
    for (0..n) |j| {
        for (0..m) |row| q.data[row * m + j] = a.data[row * n + j];
        for (0..j) |i| {
            var dot: T = 0;
            for (0..m) |row| dot += q.data[row * m + i] * a.data[row * n + j];
            r.data[i * n + j] = dot;
            for (0..m) |row| q.data[row * m + j] -= dot * q.data[row * m + i];
        }
        var norm_sq: T = 0;
        for (0..m) |row| norm_sq += q.data[row * m + j] * q.data[row * m + j];
        if (norm_sq == 0) return error.SingularMatrix;
        const norm = std.math.sqrt(norm_sq);
        r.data[j * n + j] = norm;
        for (0..m) |row| q.data[row * m + j] /= norm;
    }
    return .{ .q = q, .r = r };
}

pub fn svd(comptime T: type, a: tensor_mod.Tensor(T), tolerance: T) LinalgError!SvdResult(T) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("svd requires floating-point tensors");
    if (T == f64) return svdF64(@as(tensor_mod.Tensor(f64), a), @as(f64, tolerance));
    return error.BackendFailure;
}

fn svdF64(a: tensor_mod.Tensor(f64), tolerance: f64) LinalgError!SvdResult(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var decomposition = veyra.svdViaEigen(f64, a.allocator, matrix.asView(), tolerance) catch |err| return mapVeyraError(err);
    defer decomposition.deinit();
    var u = try fromVeyraMatrix(a.allocator, &decomposition.u);
    errdefer u.deinit();
    var s_values = try fromVeyraVector(a.allocator, &decomposition.singular_values);
    errdefer s_values.deinit();
    var vt = try fromVeyraMatrix(a.allocator, &decomposition.vt);
    errdefer vt.deinit();
    return .{ .u = u, .s = s_values, .vt = vt };
}

pub fn singularValues(comptime T: type, a: tensor_mod.Tensor(T), tolerance: T) LinalgError!tensor_mod.Tensor(T) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("singularValues requires floating-point tensors");
    var factors = try svd(T, a, tolerance);
    defer factors.deinit();
    return factors.s.clone();
}

pub fn matrixRank(comptime T: type, a: tensor_mod.Tensor(T), tolerance: T) LinalgError!usize {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("matrixRank requires floating-point tensors");
    if (T == f64) return matrixRankF64(@as(tensor_mod.Tensor(f64), a), @as(f64, tolerance));
    var values = try singularValues(T, a, tolerance);
    defer values.deinit();
    var rank_value: usize = 0;
    for (values.data) |sigma| {
        if (sigma > tolerance) rank_value += 1;
    }
    return rank_value;
}

fn matrixRankF64(a: tensor_mod.Tensor(f64), tolerance: f64) LinalgError!usize {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var decomposition = veyra.svdViaEigen(f64, a.allocator, matrix.asView(), tolerance) catch |err| return mapVeyraError(err);
    defer decomposition.deinit();
    return decomposition.rank(tolerance);
}

pub fn cond(comptime T: type, a: tensor_mod.Tensor(T), tolerance: T) LinalgError!T {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("cond requires floating-point tensors");
    if (T == f64) return condF64(@as(tensor_mod.Tensor(f64), a), @as(f64, tolerance));
    var values = try singularValues(T, a, tolerance);
    defer values.deinit();
    if (values.data.len == 0) return error.InvalidShape;
    var max_sigma = values.data[0];
    var min_sigma: ?T = null;
    for (values.data) |sigma| {
        if (sigma > max_sigma) max_sigma = sigma;
        if (sigma > tolerance) min_sigma = if (min_sigma) |current| @min(current, sigma) else sigma;
    }
    const min_resolved = min_sigma orelse return error.SingularMatrix;
    return max_sigma / min_resolved;
}

fn condF64(a: tensor_mod.Tensor(f64), tolerance: f64) LinalgError!f64 {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var decomposition = veyra.svdViaEigen(f64, a.allocator, matrix.asView(), tolerance) catch |err| return mapVeyraError(err);
    defer decomposition.deinit();
    return decomposition.conditionNumber(tolerance) catch |err| return mapVeyraError(err);
}

pub fn pinv(comptime T: type, a: tensor_mod.Tensor(T), tolerance: T) LinalgError!tensor_mod.Tensor(T) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("pinv requires floating-point tensors");
    var factors = try svd(T, a, tolerance);
    defer factors.deinit();

    const k = factors.s.data.len;
    var sigma_inv = try tensor_mod.Tensor(T).zeros(a.allocator, &.{ k, k });
    defer sigma_inv.deinit();
    for (factors.s.data, 0..) |sigma, i| {
        if (sigma > tolerance) sigma_inv.data[i * k + i] = 1 / sigma;
    }
    var vt_t = try factors.vt.transpose();
    defer vt_t.deinit();
    var left = try matmul(T, vt_t, sigma_inv);
    defer left.deinit();
    var u_t = try factors.u.transpose();
    defer u_t.deinit();
    return matmul(T, left, u_t);
}

pub fn matrixNorm(comptime T: type, a: tensor_mod.Tensor(T), order: MatrixNormOrder, tolerance: T) LinalgError!T {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("matrixNorm requires floating-point tensors");
    if (T == f64) return matrixNormF64(@as(tensor_mod.Tensor(f64), a), order, @as(f64, tolerance));
    return matrixNormReference(T, a, order, tolerance);
}

fn matrixNormF64(a: tensor_mod.Tensor(f64), order: MatrixNormOrder, tolerance: f64) LinalgError!f64 {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    return switch (order) {
        .fro => veyra.frobeniusNorm(f64, matrix.asView()),
        .one => veyra.matrixOneNorm(f64, matrix.asView()),
        .inf => veyra.matrixInfNorm(f64, matrix.asView()),
        .two => blk: {
            var values = try singularValues(f64, a, tolerance);
            defer values.deinit();
            if (values.data.len == 0) break :blk 0;
            var max_value = values.data[0];
            for (values.data[1..]) |v| {
                if (v > max_value) max_value = v;
            }
            break :blk max_value;
        },
        .nuclear => blk: {
            var values = try singularValues(f64, a, tolerance);
            defer values.deinit();
            var total: f64 = 0;
            for (values.data) |v| total += v;
            break :blk total;
        },
    };
}

fn matrixNormReference(comptime T: type, a: tensor_mod.Tensor(T), order: MatrixNormOrder, tolerance: T) LinalgError!T {
    return switch (order) {
        .fro => blk: {
            var total: T = 0;
            for (a.data) |v| total += v * v;
            break :blk std.math.sqrt(total);
        },
        .one => blk: {
            var best: T = 0;
            for (0..a.shape[1]) |c| {
                var total: T = 0;
                for (0..a.shape[0]) |r| total += @abs(a.data[r * a.shape[1] + c]);
                if (total > best) best = total;
            }
            break :blk best;
        },
        .inf => blk: {
            var best: T = 0;
            for (0..a.shape[0]) |r| {
                var total: T = 0;
                for (0..a.shape[1]) |c| total += @abs(a.data[r * a.shape[1] + c]);
                if (total > best) best = total;
            }
            break :blk best;
        },
        .two, .nuclear => blk: {
            var values = try singularValues(T, a, tolerance);
            defer values.deinit();
            if (order == .two) {
                if (values.data.len == 0) break :blk 0;
                var max_value = values.data[0];
                for (values.data[1..]) |v| {
                    if (v > max_value) max_value = v;
                }
                break :blk max_value;
            }
            var total: T = 0;
            for (values.data) |v| total += v;
            break :blk total;
        },
    };
}

pub fn eigh(comptime T: type, a: tensor_mod.Tensor(T), max_sweeps: usize, tolerance: T) LinalgError!EighResult(T) {
    if (a.shape.len != 2 or a.shape[0] != a.shape[1]) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("eigh requires floating-point tensors");
    if (T == f64) return eighF64(@as(tensor_mod.Tensor(f64), a), max_sweeps, @as(f64, tolerance));
    return error.BackendFailure;
}

fn eighF64(a: tensor_mod.Tensor(f64), max_sweeps: usize, tolerance: f64) LinalgError!EighResult(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var eig = veyra.symmetricEigenJacobi(f64, a.allocator, matrix.asView(), max_sweeps, tolerance) catch |err| return mapVeyraError(err);
    defer eig.deinit();
    var values = try fromVeyraVector(a.allocator, &eig.eigenvalues);
    errdefer values.deinit();
    var vectors = try fromVeyraMatrix(a.allocator, &eig.eigenvectors);
    errdefer vectors.deinit();
    return .{ .values = values, .vectors = vectors };
}

pub fn eigvalsh(comptime T: type, a: tensor_mod.Tensor(T), max_sweeps: usize, tolerance: T) LinalgError!tensor_mod.Tensor(T) {
    var result = try eigh(T, a, max_sweeps, tolerance);
    defer result.deinit();
    return result.values.clone();
}

pub fn lstsq(comptime T: type, a: tensor_mod.Tensor(T), b: tensor_mod.Tensor(T), tolerance: T) LinalgError!tensor_mod.Tensor(T) {
    if (a.shape.len != 2) return error.NonMatrixTensor;
    if (@typeInfo(T) != .float) @compileError("lstsq requires floating-point tensors");
    if (T == f64) return lstsqF64(@as(tensor_mod.Tensor(f64), a), @as(tensor_mod.Tensor(f64), b), @as(f64, tolerance));

    // Basic fallback for vector RHS using QR. Matrix RHS can be added when needed.
    if (b.shape.len != 1) return error.NonVectorTensor;
    var factors = try qr(T, a);
    defer factors.deinit();
    var qt = try factors.q.transpose();
    defer qt.deinit();
    var y_full = try matvec(T, qt, b);
    defer y_full.deinit();
    const n = a.shape[1];
    var x = try tensor_mod.Tensor(T).zeros(a.allocator, &.{n});
    var i = n;
    while (i > 0) {
        i -= 1;
        var acc = y_full.data[i];
        for (i + 1..n) |j| acc -= factors.r.data[i * a.shape[1] + j] * x.data[j];
        const diag = factors.r.data[i * a.shape[1] + i];
        if (@abs(diag) <= tolerance) return error.SingularMatrix;
        x.data[i] = acc / diag;
    }
    return x;
}

fn lstsqF64(a: tensor_mod.Tensor(f64), b: tensor_mod.Tensor(f64), tolerance: f64) LinalgError!tensor_mod.Tensor(f64) {
    var matrix = try toVeyraMatrix(a);
    defer matrix.deinit();
    var decomposition = veyra.svdViaEigen(f64, a.allocator, matrix.asView(), tolerance) catch |err| return mapVeyraError(err);
    defer decomposition.deinit();

    if (b.shape.len == 1) {
        var rhs = try toVeyraVector(b);
        defer rhs.deinit();
        var dst = veyra.Vector(f64).zeros(a.allocator, a.shape[1]) catch return error.BackendFailure;
        defer dst.deinit();
        decomposition.solveLeastSquares(rhs.asView(), dst.asMut(), tolerance) catch |err| return mapVeyraError(err);
        return fromVeyraVector(a.allocator, &dst);
    }

    if (b.shape.len == 2) {
        if (b.shape[0] != a.shape[0]) return error.ShapeMismatch;
        var rhs = try toVeyraMatrix(b);
        defer rhs.deinit();
        var dst = veyra.Matrix(f64).zeros(a.allocator, a.shape[1], b.shape[1], .row_major) catch return error.BackendFailure;
        defer dst.deinit();
        decomposition.solveLeastSquaresMatrix(rhs.asView(), dst.asMut(), tolerance) catch |err| return mapVeyraError(err);
        return fromVeyraMatrix(a.allocator, &dst);
    }

    return error.InvalidShape;
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

test "linalg matvec and cholesky use Veyra-compatible paths" {
    const gpa = std.testing.allocator;
    var a = try tensor_mod.tensor(f64, gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var x = try tensor_mod.tensor(f64, gpa, &.{ 1, 2, 3 }, &.{3});
    defer x.deinit();
    var y = try matvec(f64, a, x);
    defer y.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 14, 32 }, y.data);

    var spd = try tensor_mod.tensor(f64, gpa, &.{ 25, 15, -5, 15, 18, 0, -5, 0, 11 }, &.{ 3, 3 });
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
    var a = try tensor_mod.tensor(f64, gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
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
    var a = try tensor_mod.tensor(f64, gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer a.deinit();
    var factors = try svd(f64, a, 1e-12);
    defer factors.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, factors.u.shape);
    try std.testing.expectEqualSlices(usize, &.{2}, factors.s.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, factors.vt.shape);

    var sigma = try tensor_mod.Tensor(f64).zeros(gpa, &.{ 2, 2 });
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
    var a = try tensor_mod.tensor(f64, gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
    defer a.deinit();
    var b = try tensor_mod.tensor(f64, gpa, &.{ 1, 2, 2 }, &.{3});
    defer b.deinit();
    var x = try lstsq(f64, a, b, 1e-12);
    defer x.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), x.data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), x.data[1], 1e-10);

    var bm = try tensor_mod.tensor(f64, gpa, &.{ 1, 2, 2, 1, 2, 0 }, &.{ 3, 2 });
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
    var a = try tensor_mod.tensor(f64, gpa, &.{ 3, 0, 0, 2 }, &.{ 2, 2 });
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

    var rect = try tensor_mod.tensor(f64, gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
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
    var a = try tensor_mod.tensor(f64, gpa, &.{ 1, -2, 3, -4, 5, -6 }, &.{ 2, 3 });
    defer a.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(91.0)), try matrixNorm(f64, a, .fro, 1e-12), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 9), try matrixNorm(f64, a, .one, 1e-12), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 15), try matrixNorm(f64, a, .inf, 1e-12), 1e-12);
}

test "linalg symmetric eigen decomposition" {
    const gpa = std.testing.allocator;
    var a = try tensor_mod.tensor(f64, gpa, &.{ 2, 1, 1, 2 }, &.{ 2, 2 });
    defer a.deinit();
    var result = try eigh(f64, a, 64, 1e-12);
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, result.values.shape);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, result.vectors.shape);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.values.data[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3), result.values.data[1], 1e-10);

    var diag = try tensor_mod.Tensor(f64).zeros(gpa, &.{ 2, 2 });
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
