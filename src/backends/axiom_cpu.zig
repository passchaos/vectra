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
            axiom.accelerator.cpu_veyra.runElementwiseF32(axiom_op, lhs.data, rhs.data, out.data) catch return null
        else
            axiom.accelerator.cpu_veyra.runElementwiseF64(axiom_op, lhs.data, rhs.data, out.data) catch return null;
        if (!report.ok()) return null;
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
        axiom.accelerator.cpu_veyra.runGemmF32(spec, lhs.data, rhs.data, c.data, out.data) catch return null
    else
        axiom.accelerator.cpu_veyra.runGemmF64(spec, lhs.data, rhs.data, c.data, out.data) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn tryMatvecTyped(comptime T: type, matrix: array_mod.Array(T), vector: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedMatvec(T, matrix, vector)) return null;
    var out = try array_mod.Array(T).empty(matrix.allocator, &.{matrix.shape[0]});
    errdefer out.deinit();
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    const vector_view = (try bufferView(T, vector, "vector")) orelse return null;
    const out_view = (try bufferView(T, out, "out")) orelse return null;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runMatvecF32(matrix_view, vector_view, out_view, matrix.data, vector.data, out.data) catch return null
    else
        axiom.accelerator.cpu_veyra.runMatvecF64(matrix_view, vector_view, out_view, matrix.data, vector.data, out.data) catch return null;
    if (!report.ok()) return null;
    return out;
}

fn tryVecmatTyped(comptime T: type, vector: array_mod.Array(T), matrix: array_mod.Array(T)) array_mod.ArrayError!?array_mod.Array(T) {
    if (!build_options.enable_axiom_cpu_dispatch) return null;
    if (!supportedVecmat(T, vector, matrix)) return null;
    var out = try array_mod.Array(T).empty(vector.allocator, &.{matrix.shape[1]});
    errdefer out.deinit();
    const vector_view = (try bufferView(T, vector, "vector")) orelse return null;
    const matrix_view = (try matrixView(T, matrix, "matrix")) orelse return null;
    const out_view = (try bufferView(T, out, "out")) orelse return null;
    const report = if (T == f32)
        axiom.accelerator.cpu_veyra.runVecmatF32(vector_view, matrix_view, out_view, vector.data, matrix.data, out.data) catch return null
    else
        axiom.accelerator.cpu_veyra.runVecmatF64(vector_view, matrix_view, out_view, vector.data, matrix.data, out.data) catch return null;
    if (!report.ok()) return null;
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
    return matrix.device.isCpu() and
        matrix.shape.len == 2 and
        matrix.data.len != 0 and
        matrix.strides.len == 2;
}

fn tensorElementType(comptime T: type) axiom.accelerator.TensorElementType {
    return if (T == f32) .f32 else .f64;
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
    } else {
        try std.testing.expect(mv32 == null);
        try std.testing.expect(vt64 == null);
        try std.testing.expect(dot32 == null);
        try std.testing.expect(trace64 == null);
    }
}
