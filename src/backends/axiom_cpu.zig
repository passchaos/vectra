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
