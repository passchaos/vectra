//! Axiom CPU bridge for Vectra.
//!
//! This routes Vectra host arrays through Axiom's CPU lowering entry points,
//! which currently delegate dense GEMM to the sibling Veyra backend.

const std = @import("std");
const build_options = @import("vectra_build_options");
const array_mod = @import("../array.zig");

const axiom = if (build_options.enable_axiom_cpu_dispatch) @import("axiom") else struct {};

pub fn enabled() bool {
    return build_options.enable_axiom_cpu_dispatch;
}

pub const ElementwiseOp = enum {
    add,
    sub,
    mul,
    div,
};

pub fn tryAddF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryElementwise(f32, .add, lhs, rhs);
}
pub fn trySubF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryElementwise(f32, .sub, lhs, rhs);
}
pub fn tryMulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryElementwise(f32, .mul, lhs, rhs);
}
pub fn tryDivF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryElementwise(f32, .div, lhs, rhs);
}
pub fn tryAddF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryElementwise(f64, .add, lhs, rhs);
}
pub fn trySubF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryElementwise(f64, .sub, lhs, rhs);
}
pub fn tryMulF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryElementwise(f64, .mul, lhs, rhs);
}
pub fn tryDivF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryElementwise(f64, .div, lhs, rhs);
}

fn tryElementwise(comptime T: type, op: ElementwiseOp, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (comptime build_options.enable_axiom_cpu_dispatch) {
        if (!supportedSameShapeContiguous(T, lhs, rhs)) return null;
        var out = try array_mod.Array(T).empty(lhs.allocator, lhs.shape);
        errdefer out.deinit();
        const axiom_op: axiom.accelerator.TensorBinaryElementwiseOp = switch (op) {
            .add => .add,
            .sub => .sub,
            .mul => .mul,
            .div => .div,
        };
        const report = if (T == f32)
            axiom.accelerator.cpu_veyra.runElementwiseF32(axiom_op, lhs.data, rhs.data, out.data) catch {
                out.deinit();
                return null;
            }
        else
            axiom.accelerator.cpu_veyra.runElementwiseF64(axiom_op, lhs.data, rhs.data, out.data) catch {
                out.deinit();
                return null;
            };
        if (!report.ok()) {
            out.deinit();
            return null;
        }
        return out;
    } else {
        return null;
    }
}

pub fn tryMatmulF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryMatmulTyped(f32, lhs, rhs);
}

pub fn tryMatmulF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryMatmulTyped(f64, lhs, rhs);
}

pub fn tryMatvecF32(matrix: array_mod.Array(f32), vector: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryMatvecTyped(f32, matrix, vector);
}

pub fn tryMatvecF64(matrix: array_mod.Array(f64), vector: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryMatvecTyped(f64, matrix, vector);
}

pub fn tryVecmatF32(vector: array_mod.Array(f32), matrix: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryVecmatTyped(f32, vector, matrix);
}

pub fn tryVecmatF64(vector: array_mod.Array(f64), matrix: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryVecmatTyped(f64, vector, matrix);
}

pub fn tryDotF32(lhs: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?f32 {
    return tryDotTyped(f32, lhs, rhs);
}

pub fn tryDotF64(lhs: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?f64 {
    return tryDotTyped(f64, lhs, rhs);
}

pub fn tryTraceF32(matrix: array_mod.Array(f32), offset: isize) array_mod.ArrayError!?f32 {
    return tryTraceTyped(f32, matrix, offset);
}

pub fn tryTraceF64(matrix: array_mod.Array(f64), offset: isize) array_mod.ArrayError!?f64 {
    return tryTraceTyped(f64, matrix, offset);
}

pub fn tryDetF32(matrix: array_mod.Array(f32)) array_mod.ArrayError!?f32 {
    return tryDetTyped(f32, matrix);
}

pub fn tryDetF64(matrix: array_mod.Array(f64)) array_mod.ArrayError!?f64 {
    return tryDetTyped(f64, matrix);
}

pub fn tryInverseF32(matrix: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryInverseTyped(f32, matrix);
}

pub fn tryInverseF64(matrix: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryInverseTyped(f64, matrix);
}

pub fn trySolveF32(matrix: array_mod.Array(f32), rhs: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return trySolveTyped(f32, matrix, rhs);
}

pub fn trySolveF64(matrix: array_mod.Array(f64), rhs: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return trySolveTyped(f64, matrix, rhs);
}

pub fn tryCholeskyF32(matrix: array_mod.Array(f32)) array_mod.ArrayError!?array_mod.Array(f32) {
    return tryCholeskyTyped(f32, matrix);
}

pub fn tryCholeskyF64(matrix: array_mod.Array(f64)) array_mod.ArrayError!?array_mod.Array(f64) {
    return tryCholeskyTyped(f64, matrix);
}

pub fn QrResult(comptime T: type) type {
    return struct {
        q: array_mod.Array(T),
        r: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.q.deinit();
            self.r.deinit();
            self.* = undefined;
        }
    };
}

pub fn tryQrF32(matrix: array_mod.Array(f32)) array_mod.ArrayError!?QrResult(f32) {
    return tryQrTyped(f32, matrix);
}

pub fn tryQrF64(matrix: array_mod.Array(f64)) array_mod.ArrayError!?QrResult(f64) {
    return tryQrTyped(f64, matrix);
}

pub fn LuResult(comptime T: type) type {
    return struct {
        p: array_mod.Array(T),
        l: array_mod.Array(T),
        u: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.p.deinit();
            self.l.deinit();
            self.u.deinit();
            self.* = undefined;
        }
    };
}

pub fn tryLuF32(matrix: array_mod.Array(f32)) array_mod.ArrayError!?LuResult(f32) {
    return tryLuTyped(f32, matrix);
}

pub fn tryLuF64(matrix: array_mod.Array(f64)) array_mod.ArrayError!?LuResult(f64) {
    return tryLuTyped(f64, matrix);
}

pub fn SvdResult(comptime T: type) type {
    return struct {
        u: array_mod.Array(T),
        s: array_mod.Array(T),
        vt: array_mod.Array(T),

        pub fn deinit(self: *@This()) void {
            self.u.deinit();
            self.s.deinit();
            self.vt.deinit();
            self.* = undefined;
        }
    };
}

pub fn trySvdF32(matrix: array_mod.Array(f32), tolerance: f32) array_mod.ArrayError!?SvdResult(f32) {
    return trySvdTyped(f32, matrix, tolerance);
}

pub fn trySvdF64(matrix: array_mod.Array(f64), tolerance: f64) array_mod.ArrayError!?SvdResult(f64) {
    return trySvdTyped(f64, matrix, tolerance);
}

pub fn trySingularValuesF32(matrix: array_mod.Array(f32), tolerance: f32) array_mod.ArrayError!?array_mod.Array(f32) {
    return trySingularValuesTyped(f32, matrix, tolerance);
}

pub fn trySingularValuesF64(matrix: array_mod.Array(f64), tolerance: f64) array_mod.ArrayError!?array_mod.Array(f64) {
    return trySingularValuesTyped(f64, matrix, tolerance);
}

pub fn tryMatrixRankF32(matrix: array_mod.Array(f32), tolerance: f32) array_mod.ArrayError!?usize {
    return tryMatrixRankTyped(f32, matrix, tolerance);
}

pub fn tryMatrixRankF64(matrix: array_mod.Array(f64), tolerance: f64) array_mod.ArrayError!?usize {
    return tryMatrixRankTyped(f64, matrix, tolerance);
}

pub fn tryCondF32(matrix: array_mod.Array(f32), tolerance: f32) array_mod.ArrayError!?f32 {
    return tryCondTyped(f32, matrix, tolerance);
}

pub fn tryCondF64(matrix: array_mod.Array(f64), tolerance: f64) array_mod.ArrayError!?f64 {
    return tryCondTyped(f64, matrix, tolerance);
}

pub fn trySolveTriangularF32(
    matrix: array_mod.Array(f32),
    rhs: array_mod.Array(f32),
    triangle: array_mod.Triangle,
    diagonal: array_mod.Diagonal,
) array_mod.ArrayError!?array_mod.Array(f32) {
    return trySolveTriangularTyped(f32, matrix, rhs, triangle, diagonal);
}

pub fn trySolveTriangularF64(
    matrix: array_mod.Array(f64),
    rhs: array_mod.Array(f64),
    triangle: array_mod.Triangle,
    diagonal: array_mod.Diagonal,
) array_mod.ArrayError!?array_mod.Array(f64) {
    return trySolveTriangularTyped(f64, matrix, rhs, triangle, diagonal);
}

pub fn tryMatrixNormF32(matrix: array_mod.Array(f32), order: array_mod.MatrixNormOrder) array_mod.ArrayError!?f32 {
    return tryMatrixNormTyped(f32, matrix, order);
}

pub fn tryMatrixNormF64(matrix: array_mod.Array(f64), order: array_mod.MatrixNormOrder) array_mod.ArrayError!?f64 {
    return tryMatrixNormTyped(f64, matrix, order);
}

fn tryMatmulTyped(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatmul2dContiguous(T, lhs, rhs)) return null;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    const n = rhs.shape[1];
    var c = try array_mod.Array(T).zeros(lhs.allocator, &.{ m, n });
    defer c.deinit();
    var out = try array_mod.Array(T).empty(lhs.allocator, &.{ m, n });
    errdefer out.deinit();

    const spec = axiom.accelerator.TensorGemmSpec.rowMajor(
        .rowMajor("lhs", @intCast(@intFromPtr(lhs.data.ptr)), m, k),
        .rowMajor("rhs", @intCast(@intFromPtr(rhs.data.ptr)), k, n),
        .rowMajor("out", @intCast(@intFromPtr(out.data.ptr)), m, n),
    );
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runGemmF32(spec, lhs.data, rhs.data, c.data, out.data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runGemmF64(spec, lhs.data, rhs.data, c.data, out.data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryMatvecTyped(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatvec(T, matrix, vector)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    const vector_view = (try bufferView(T, vector, "vector")) orelse return null;
    var out = try array_mod.Array(T).empty(matrix.allocator, &.{matrix.shape[0]});
    errdefer out.deinit();
    const out_view = (try bufferView(T, out, "out")) orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runMatvecF32(matrix_view, vector_view, out_view, matrix.data, vector.data, out.data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runMatvecF64(matrix_view, vector_view, out_view, matrix.data, vector.data, out.data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryVecmatTyped(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedVecmat(T, vector, matrix)) return null;
    const vector_view = (try bufferView(T, vector, "vector")) orelse return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out = try array_mod.Array(T).empty(vector.allocator, &.{matrix.shape[1]});
    errdefer out.deinit();
    const out_view = (try bufferView(T, out, "out")) orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runVecmatF32(vector_view, matrix_view, out_view, vector.data, matrix.data, out.data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runVecmatF64(vector_view, matrix_view, out_view, vector.data, matrix.data, out.data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryDotTyped(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?T {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedDot(T, lhs, rhs)) return null;
    const lhs_view = (try bufferView(T, lhs, "lhs")) orelse return null;
    const rhs_view = (try bufferView(T, rhs, "rhs")) orelse return null;
    var out: T = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runDotF32(lhs_view, rhs_view, lhs.data, rhs.data, &out) catch return null
    else
        axiom.accelerator.cpu_veyra.runDotF64(lhs_view, rhs_view, lhs.data, rhs.data, &out) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn tryTraceTyped(comptime T: type, matrix: array_mod.Array(T), offset: isize) array_mod.ArrayError!?T {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedTrace(T, matrix)) return null;
    const view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out: T = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runTraceF32(view, offset, matrix.data, &out) catch return null
    else
        axiom.accelerator.cpu_veyra.runTraceF64(view, offset, matrix.data, &out) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn tryDetTyped(comptime T: type, matrix: array_mod.Array(T)) array_mod.ArrayError!?T {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedSquareMatrix(T, matrix)) return null;
    const view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out: T = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runDetF32(view, matrix.data, &out) catch return null
    else
        axiom.accelerator.cpu_veyra.runDetF64(view, matrix.data, &out) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn tryInverseTyped(comptime T: type, matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedSquareMatrix(T, matrix)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out = try array_mod.Array(T).empty(matrix.allocator, matrix.shape);
    errdefer out.deinit();
    const out_view = (try matrixView(T, out, "out")) orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runInverseF32(matrix_view, out_view, matrix.data, out.data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runInverseF64(matrix_view, out_view, matrix.data, out.data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn trySolveTyped(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedSolve(T, matrix, rhs)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    const rhs_view = (try matrixOrVectorColumnView(T, rhs, "rhs")) orelse return null;
    const out_shape: []const usize = if (rhs.shape.len == 1) &.{matrix.shape[1]} else &.{ matrix.shape[1], rhs.shape[1] };
    var out = try array_mod.Array(T).empty(matrix.allocator, out_shape);
    errdefer out.deinit();
    const out_view = (try matrixOrVectorColumnView(T, out, "out")) orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runSolveF32(matrix_view, rhs_view, out_view, matrix.data, rhs.data, out.data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runSolveF64(matrix_view, rhs_view, out_view, matrix.data, rhs.data, out.data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryCholeskyTyped(comptime T: type, matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedSquareMatrix(T, matrix)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out = try array_mod.Array(T).zeros(matrix.allocator, matrix.shape);
    errdefer out.deinit();
    const out_view = (try matrixView(T, out, "out")) orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runCholeskyF32(matrix_view, out_view, matrix.data, out.data) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runCholeskyF64(matrix_view, out_view, matrix.data, out.data) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryQrTyped(comptime T: type, matrix: array_mod.Array(T)) array_mod.ArrayError!?QrResult(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatrix(T, matrix)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var q = try array_mod.Array(T).empty(matrix.allocator, &.{ matrix.shape[0], matrix.shape[0] });
    errdefer q.deinit();
    var r = try array_mod.Array(T).empty(matrix.allocator, &.{ matrix.shape[0], matrix.shape[1] });
    errdefer r.deinit();
    const q_view = (try matrixView(T, q, "q")) orelse {
        q.deinit();
        r.deinit();
        return null;
    };
    const r_view = (try matrixView(T, r, "r")) orelse {
        q.deinit();
        r.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runQrF32(matrix_view, q_view, r_view, matrix.data, q.data, r.data) catch {
            q.deinit();
            r.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runQrF64(matrix_view, q_view, r_view, matrix.data, q.data, r.data) catch {
            q.deinit();
            r.deinit();
            return null;
        };
    if (!report.ok()) {
        q.deinit();
        r.deinit();
        return null;
    }
    return .{ .q = q, .r = r };
}

fn tryLuTyped(comptime T: type, matrix: array_mod.Array(T)) array_mod.ArrayError!?LuResult(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedSquareMatrix(T, matrix)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var p = try array_mod.Array(T).empty(matrix.allocator, matrix.shape);
    errdefer p.deinit();
    var l = try array_mod.Array(T).empty(matrix.allocator, matrix.shape);
    errdefer l.deinit();
    var u = try array_mod.Array(T).empty(matrix.allocator, matrix.shape);
    errdefer u.deinit();
    const p_view = (try matrixView(T, p, "p")) orelse {
        p.deinit();
        l.deinit();
        u.deinit();
        return null;
    };
    const l_view = (try matrixView(T, l, "l")) orelse {
        p.deinit();
        l.deinit();
        u.deinit();
        return null;
    };
    const u_view = (try matrixView(T, u, "u")) orelse {
        p.deinit();
        l.deinit();
        u.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runLuF32(matrix_view, p_view, l_view, u_view, matrix.data, p.data, l.data, u.data) catch {
            p.deinit();
            l.deinit();
            u.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runLuF64(matrix_view, p_view, l_view, u_view, matrix.data, p.data, l.data, u.data) catch {
            p.deinit();
            l.deinit();
            u.deinit();
            return null;
        };
    if (!report.ok()) {
        p.deinit();
        l.deinit();
        u.deinit();
        return null;
    }
    return .{ .p = p, .l = l, .u = u };
}

fn trySvdTyped(comptime T: type, matrix: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?SvdResult(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatrix(T, matrix)) return null;
    const factor_dim = @min(matrix.shape[0], matrix.shape[1]);
    if (factor_dim == 0) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var u = try array_mod.Array(T).empty(matrix.allocator, &.{ matrix.shape[0], factor_dim });
    errdefer u.deinit();
    var s = try array_mod.Array(T).empty(matrix.allocator, &.{factor_dim});
    errdefer s.deinit();
    var vt = try array_mod.Array(T).empty(matrix.allocator, &.{ factor_dim, matrix.shape[1] });
    errdefer vt.deinit();
    const u_view = (try matrixView(T, u, "u")) orelse {
        u.deinit();
        s.deinit();
        vt.deinit();
        return null;
    };
    const s_view = (try bufferView(T, s, "s")) orelse {
        u.deinit();
        s.deinit();
        vt.deinit();
        return null;
    };
    const vt_view = (try matrixView(T, vt, "vt")) orelse {
        u.deinit();
        s.deinit();
        vt.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runSvdF32(matrix_view, u_view, s_view, vt_view, matrix.data, u.data, s.data, vt.data, tolerance) catch {
            u.deinit();
            s.deinit();
            vt.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runSvdF64(matrix_view, u_view, s_view, vt_view, matrix.data, u.data, s.data, vt.data, tolerance) catch {
            u.deinit();
            s.deinit();
            vt.deinit();
            return null;
        };
    if (!report.ok()) {
        u.deinit();
        s.deinit();
        vt.deinit();
        return null;
    }
    return .{ .u = u, .s = s, .vt = vt };
}

fn trySingularValuesTyped(comptime T: type, matrix: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatrix(T, matrix)) return null;
    const factor_dim = @min(matrix.shape[0], matrix.shape[1]);
    if (factor_dim == 0) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var s = try array_mod.Array(T).empty(matrix.allocator, &.{factor_dim});
    errdefer s.deinit();
    const s_view = (try bufferView(T, s, "s")) orelse {
        s.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runSingularValuesF32(matrix_view, s_view, matrix.data, s.data, tolerance) catch {
            s.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runSingularValuesF64(matrix_view, s_view, matrix.data, s.data, tolerance) catch {
            s.deinit();
            return null;
        };
    if (!report.ok()) {
        s.deinit();
        return null;
    }
    return s;
}

fn tryMatrixRankTyped(comptime T: type, matrix: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?usize {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatrix(T, matrix)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out: usize = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runMatrixRankF32(matrix_view, matrix.data, tolerance, &out) catch return null
    else
        axiom.accelerator.cpu_veyra.runMatrixRankF64(matrix_view, matrix.data, tolerance, &out) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn tryCondTyped(comptime T: type, matrix: array_mod.Array(T), tolerance: T) array_mod.ArrayError!?T {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatrix(T, matrix)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out: T = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runConditionNumberF32(matrix_view, matrix.data, tolerance, &out) catch |err| return switch (err) {
            error.SingularMatrix => error.SingularMatrix,
            else => null,
        }
    else
        axiom.accelerator.cpu_veyra.runConditionNumberF64(matrix_view, matrix.data, tolerance, &out) catch |err| return switch (err) {
            error.SingularMatrix => error.SingularMatrix,
            else => null,
        };
    if (!report.ok()) return null;
    return out;
}

fn trySolveTriangularTyped(
    comptime T: type,
    matrix: array_mod.Array(T),
    rhs: array_mod.Array(T),
    triangle: array_mod.Triangle,
    diagonal: array_mod.Diagonal,
) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedSolve(T, matrix, rhs)) return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    const rhs_view = (try matrixOrVectorColumnView(T, rhs, "rhs")) orelse return null;
    const out_shape: []const usize = if (rhs.shape.len == 1) &.{matrix.shape[0]} else &.{ matrix.shape[0], rhs.shape[1] };
    var out = try array_mod.Array(T).empty(matrix.allocator, out_shape);
    errdefer out.deinit();
    const out_view = (try matrixOrVectorColumnView(T, out, "out")) orelse {
        out.deinit();
        return null;
    };
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runSolveTriangularF32(
            matrix_view,
            rhs_view,
            out_view,
            matrix.data,
            rhs.data,
            out.data,
            cpuTriangle(triangle),
            cpuDiagonal(diagonal),
        ) catch {
            out.deinit();
            return null;
        }
    else
        axiom.accelerator.cpu_veyra.runSolveTriangularF64(
            matrix_view,
            rhs_view,
            out_view,
            matrix.data,
            rhs.data,
            out.data,
            cpuTriangle(triangle),
            cpuDiagonal(diagonal),
        ) catch {
            out.deinit();
            return null;
        };
    if (!report.ok()) {
        out.deinit();
        return null;
    }
    return out;
}

fn tryMatrixNormTyped(comptime T: type, matrix: array_mod.Array(T), order: array_mod.MatrixNormOrder) array_mod.ArrayError!?T {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatrix(T, matrix)) return null;
    const axiom_order = cpuMatrixNormOrder(order) orelse return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    var out: T = 0;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runMatrixNormF32(matrix_view, matrix.data, axiom_order, &out, normTolerance(T)) catch return null
    else
        axiom.accelerator.cpu_veyra.runMatrixNormF64(matrix_view, matrix.data, axiom_order, &out, normTolerance(T)) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn supportedSameShapeContiguous(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and
        rhs.device.isCpu() and
        lhs.data.len != 0 and
        lhs.sameShape(rhs) and
        lhs.isContiguous() and
        rhs.isContiguous();
}

fn supportedMatmul2dContiguous(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and rhs.device.isCpu() and
        lhs.shape.len == 2 and rhs.shape.len == 2 and
        lhs.shape[1] == rhs.shape[0] and
        lhs.data.len != 0 and rhs.data.len != 0 and
        lhs.isContiguous() and rhs.isContiguous();
}

fn supportedMatvec(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) bool {
    return matrix.device.isCpu() and vector.device.isCpu() and
        matrix.shape.len == 2 and vector.shape.len == 1 and
        matrix.shape[1] == vector.shape[0] and
        matrix.data.len != 0 and vector.data.len != 0 and
        matrix.strides.len == 2 and vector.strides.len == 1;
}

fn supportedVecmat(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) bool {
    return vector.device.isCpu() and matrix.device.isCpu() and
        vector.shape.len == 1 and matrix.shape.len == 2 and
        vector.shape[0] == matrix.shape[0] and
        vector.data.len != 0 and matrix.data.len != 0 and
        vector.strides.len == 1 and matrix.strides.len == 2;
}

fn supportedDot(comptime T: type, lhs: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    return lhs.device.isCpu() and rhs.device.isCpu() and
        lhs.shape.len == 1 and rhs.shape.len == 1 and
        lhs.shape[0] == rhs.shape[0] and
        lhs.data.len != 0 and rhs.data.len != 0 and
        lhs.strides.len == 1 and rhs.strides.len == 1;
}

fn supportedTrace(comptime T: type, matrix: array_mod.Array(T)) bool {
    return supportedMatrix(T, matrix);
}

fn supportedMatrix(comptime T: type, matrix: array_mod.Array(T)) bool {
    return matrix.device.isCpu() and
        matrix.shape.len == 2 and
        matrix.data.len != 0 and
        matrix.strides.len == 2;
}

fn supportedSquareMatrix(comptime T: type, matrix: array_mod.Array(T)) bool {
    return supportedMatrix(T, matrix) and matrix.shape[0] == matrix.shape[1];
}

fn supportedSolve(comptime T: type, matrix: array_mod.Array(T), rhs: array_mod.Array(T)) bool {
    const rhs_rank_ok = rhs.shape.len == 1 or rhs.shape.len == 2;
    return supportedSquareMatrix(T, matrix) and
        rhs.device.isCpu() and
        rhs_rank_ok and
        rhs.shape[0] == matrix.shape[0] and
        rhs.data.len != 0 and
        (rhs.strides.len == 1 or rhs.strides.len == 2);
}

fn tensorElementType(comptime T: type) axiom.accelerator.TensorElementType {
    return if (T == f32) .f32 else .f64;
}

fn cpuTriangle(triangle: array_mod.Triangle) axiom.accelerator.cpu_veyra.CpuVeyraTriangle {
    return switch (triangle) {
        .lower => .lower,
        .upper => .upper,
    };
}

fn cpuDiagonal(diagonal: array_mod.Diagonal) axiom.accelerator.cpu_veyra.CpuVeyraDiagonal {
    return switch (diagonal) {
        .non_unit => .non_unit,
        .unit => .unit,
    };
}

fn cpuMatrixNormOrder(order: array_mod.MatrixNormOrder) ?axiom.accelerator.cpu_veyra.CpuVeyraMatrixNormOrder {
    return switch (order) {
        .fro => .fro,
        .one => .one,
        .inf => .inf,
        .two => .two,
        .nuclear => .nuclear,
    };
}

fn normTolerance(comptime T: type) T {
    return if (T == f32) 1e-5 else 1e-12;
}

fn matrixView(comptime T: type, matrix: array_mod.Array(T), name: []const u8) array_mod.ArrayError!?axiom.accelerator.TensorMatrixView {
    const row_stride = std.math.cast(isize, matrix.strides[0]) orelse return null;
    const col_stride = std.math.cast(isize, matrix.strides[1]) orelse return null;
    var view = axiom.accelerator.TensorMatrixView.strided(
        name,
        @intCast(@intFromPtr(matrix.data.ptr)),
        matrix.shape[0],
        matrix.shape[1],
        row_stride,
        col_stride,
    );
    view.element_type = tensorElementType(T);
    return view;
}

fn matrixOrVectorColumnView(comptime T: type, value: array_mod.Array(T), name: []const u8) array_mod.ArrayError!?axiom.accelerator.TensorMatrixView {
    if (value.shape.len == 2) return matrixView(T, value, name);
    if (value.shape.len != 1) return null;
    const row_stride = std.math.cast(isize, value.strides[0]) orelse return null;
    var view = axiom.accelerator.TensorMatrixView.strided(
        name,
        @intCast(@intFromPtr(value.data.ptr)),
        value.shape[0],
        1,
        row_stride,
        1,
    );
    view.element_type = tensorElementType(T);
    return view;
}

fn bufferView(comptime T: type, vector: array_mod.Array(T), name: []const u8) array_mod.ArrayError!?axiom.accelerator.TensorBufferView {
    const stride = std.math.cast(isize, vector.strides[0]) orelse return null;
    var view = axiom.accelerator.TensorBufferView.strided(
        name,
        @intCast(@intFromPtr(vector.data.ptr)),
        vector.shape[0],
        stride,
    );
    view.element_type = tensorElementType(T);
    return view;
}

test "Axiom CPU bridge dispatches GEMM by default" {
    const gpa = std.testing.allocator;
    var a = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try array_mod.Array(f32).fromSlice(gpa, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    const maybe = try tryMatmulF32(a, b);
    if (build_options.enable_axiom_cpu_dispatch) {
        try std.testing.expect(maybe != null);
        var out = maybe.?;
        defer out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, out.data);
    } else {
        try std.testing.expect(maybe == null);
    }
}

test "Axiom CPU bridge dispatches elementwise by default" {
    const gpa = std.testing.allocator;
    var lhs32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 8, 6, 4, 2 }, &.{4});
    defer lhs32.deinit();
    var rhs32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{4});
    defer rhs32.deinit();

    var lhs64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 8, 6, 4, 2 }, &.{4});
    defer lhs64.deinit();
    var rhs64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 3, 4, 2 }, &.{4});
    defer rhs64.deinit();

    const add = try tryAddF32(lhs32, rhs32);
    const div = try tryDivF64(lhs64, rhs64);
    if (build_options.enable_axiom_cpu_dispatch) {
        try std.testing.expect(add != null);
        var add_out = add.?;
        defer add_out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 9, 8, 7, 6 }, add_out.data);

        try std.testing.expect(div != null);
        var div_out = div.?;
        defer div_out.deinit();
        try std.testing.expectEqualSlices(f64, &.{ 4, 2, 1, 1 }, div_out.data);
    } else {
        try std.testing.expect(add == null);
        try std.testing.expect(div == null);
    }
}

test "Axiom CPU bridge dispatches vector ops and trace" {
    const gpa = std.testing.allocator;
    var matrix32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer matrix32.deinit();
    var vector32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer vector32.deinit();
    var lhs32 = try array_mod.Array(f32).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer lhs32.deinit();

    var matrix64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer matrix64.deinit();
    var vector64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3 }, &.{3});
    defer vector64.deinit();
    var lhs64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2 }, &.{2});
    defer lhs64.deinit();
    var square64 = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer square64.deinit();

    const mv32 = try tryMatvecF32(matrix32, vector32);
    const vt64 = try tryVecmatF64(lhs64, matrix64);
    const dot32 = try tryDotF32(vector32, vector32);
    const trace64 = try tryTraceF64(square64, 0);

    if (build_options.enable_axiom_cpu_dispatch) {
        try std.testing.expect(mv32 != null);
        var mv_out = mv32.?;
        defer mv_out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 14, 32 }, mv_out.data);

        try std.testing.expect(vt64 != null);
        var vt_out = vt64.?;
        defer vt_out.deinit();
        try std.testing.expectEqualSlices(f64, &.{ 9, 12, 15 }, vt_out.data);

        try std.testing.expect(dot32 != null);
        try std.testing.expectEqual(@as(f32, 14), dot32.?);
        try std.testing.expect(trace64 != null);
        try std.testing.expectEqual(@as(f64, 5), trace64.?);

        const mv64 = try tryMatvecF64(matrix64, vector64);
        try std.testing.expect(mv64 != null);
        var mv64_out = mv64.?;
        defer mv64_out.deinit();
        try std.testing.expectEqualSlices(f64, &.{ 14, 32 }, mv64_out.data);

        const vt32 = try tryVecmatF32(lhs32, matrix32);
        try std.testing.expect(vt32 != null);
        var vt32_out = vt32.?;
        defer vt32_out.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 9, 12, 15 }, vt32_out.data);

        var solve_matrix = try array_mod.Array(f64).fromSlice(gpa, &.{ 4, 7, 2, 6 }, &.{ 2, 2 });
        defer solve_matrix.deinit();
        var solve_rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 18, 16 }, &.{2});
        defer solve_rhs.deinit();
        const det = try tryDetF64(solve_matrix);
        try std.testing.expect(det != null);
        try std.testing.expectApproxEqAbs(@as(f64, 10), det.?, 1e-12);
        const inv = try tryInverseF64(solve_matrix);
        try std.testing.expect(inv != null);
        var inv_out = inv.?;
        defer inv_out.deinit();
        try std.testing.expectApproxEqAbs(@as(f64, 0.6), inv_out.data[0], 1e-12);
        const solved = try trySolveF64(solve_matrix, solve_rhs);
        try std.testing.expect(solved != null);
        var solved_out = solved.?;
        defer solved_out.deinit();
        try std.testing.expectApproxEqAbs(@as(f64, -0.4), solved_out.data[0], 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 2.8), solved_out.data[1], 1e-12);

        var spd = try array_mod.Array(f64).fromSlice(gpa, &.{ 25, 15, -5, 15, 18, 0, -5, 0, 11 }, &.{ 3, 3 });
        defer spd.deinit();
        const chol = try tryCholeskyF64(spd);
        try std.testing.expect(chol != null);
        var chol_out = chol.?;
        defer chol_out.deinit();
        try std.testing.expectApproxEqAbs(@as(f64, 5), chol_out.data[0], 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, 3), chol_out.data[3], 1e-12);
        try std.testing.expectApproxEqAbs(@as(f64, -1), chol_out.data[6], 1e-12);

        var rect = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, 1, 1, 2, 1, 3 }, &.{ 3, 2 });
        defer rect.deinit();
        const qr = try tryQrF64(rect);
        try std.testing.expect(qr != null);
        var qr_out = qr.?;
        defer qr_out.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, qr_out.q.shape);
        try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, qr_out.r.shape);
        var qr_reconstructed = try qr_out.q.matmul(qr_out.r);
        defer qr_reconstructed.deinit();
        try std.testing.expect(try qr_reconstructed.allclose(rect, 1e-10, 1e-10));

        const lu = try tryLuF64(solve_matrix);
        try std.testing.expect(lu != null);
        var lu_out = lu.?;
        defer lu_out.deinit();
        var lu_product = try lu_out.l.matmul(lu_out.u);
        defer lu_product.deinit();
        var lu_reconstructed = try lu_out.p.matmul(lu_product);
        defer lu_reconstructed.deinit();
        try std.testing.expect(try lu_reconstructed.allclose(solve_matrix, 1e-12, 1e-12));

        const svd = try trySvdF64(rect, 1e-12);
        try std.testing.expect(svd != null);
        var svd_out = svd.?;
        defer svd_out.deinit();
        try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, svd_out.u.shape);
        try std.testing.expectEqualSlices(usize, &.{2}, svd_out.s.shape);
        try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, svd_out.vt.shape);
        var sigma = try array_mod.Array(f64).zeros(gpa, &.{ 2, 2 });
        defer sigma.deinit();
        sigma.data[0] = svd_out.s.data[0];
        sigma.data[3] = svd_out.s.data[1];
        var us = try svd_out.u.matmul(sigma);
        defer us.deinit();
        var svd_reconstructed = try us.matmul(svd_out.vt);
        defer svd_reconstructed.deinit();
        try std.testing.expect(try svd_reconstructed.allclose(rect, 1e-10, 1e-10));
        const singular_values = try trySingularValuesF64(rect, 1e-12);
        try std.testing.expect(singular_values != null);
        var singular_values_out = singular_values.?;
        defer singular_values_out.deinit();
        try std.testing.expectEqualSlices(usize, &.{2}, singular_values_out.shape);
        try std.testing.expectApproxEqAbs(svd_out.s.data[0], singular_values_out.data[0], 1e-12);
        const rank = try tryMatrixRankF64(rect, 1e-12);
        try std.testing.expect(rank != null);
        try std.testing.expectEqual(@as(usize, 2), rank.?);
        const cond = try tryCondF64(rect, 1e-12);
        try std.testing.expect(cond != null);
        try std.testing.expectApproxEqAbs(singular_values_out.data[0] / singular_values_out.data[1], cond.?, 1e-12);

        var triangular = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 0, 0, -1, 3, 0, 4, 2, 5 }, &.{ 3, 3 });
        defer triangular.deinit();
        var triangular_rhs = try array_mod.Array(f64).fromSlice(gpa, &.{ 2, 2, 25 }, &.{3});
        defer triangular_rhs.deinit();
        const triangular_solution = try trySolveTriangularF64(triangular, triangular_rhs, .lower, .non_unit);
        try std.testing.expect(triangular_solution != null);
        var triangular_out = triangular_solution.?;
        defer triangular_out.deinit();
        try std.testing.expectApproxEqAbs(@as(f64, 3.8), triangular_out.data[2], 1e-12);

        var norm_source = try array_mod.Array(f64).fromSlice(gpa, &.{ 1, -2, 3, -4, 5, -6 }, &.{ 2, 3 });
        defer norm_source.deinit();
        const fro = try tryMatrixNormF64(norm_source, .fro);
        try std.testing.expect(fro != null);
        try std.testing.expectApproxEqAbs(@as(f64, @sqrt(91.0)), fro.?, 1e-12);
        const one_norm = try tryMatrixNormF64(norm_source, .one);
        try std.testing.expect(one_norm != null);
        try std.testing.expectApproxEqAbs(@as(f64, 9), one_norm.?, 1e-12);
        const inf_norm = try tryMatrixNormF64(norm_source, .inf);
        try std.testing.expect(inf_norm != null);
        try std.testing.expectApproxEqAbs(@as(f64, 15), inf_norm.?, 1e-12);
        const two_norm = try tryMatrixNormF64(rect, .two);
        try std.testing.expect(two_norm != null);
        try std.testing.expectApproxEqAbs(singular_values_out.data[0], two_norm.?, 1e-12);
        const nuclear_norm = try tryMatrixNormF64(rect, .nuclear);
        try std.testing.expect(nuclear_norm != null);
        try std.testing.expectApproxEqAbs(singular_values_out.data[0] + singular_values_out.data[1], nuclear_norm.?, 1e-12);
    } else {
        try std.testing.expect(mv32 == null);
        try std.testing.expect(vt64 == null);
        try std.testing.expect(dot32 == null);
        try std.testing.expect(trace64 == null);
    }
}
