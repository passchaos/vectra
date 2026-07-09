//! Optional Axiom CPU bridge for Vectra.
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

test "Axiom CPU bridge is disabled by default or dispatches GEMM" {
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

test "Axiom CPU bridge is disabled by default or dispatches elementwise" {
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
