//! Smoke gate for opt-in automatic Array(f32) -> Axiom CUDA dispatch.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer lhs.deinit();
    var rhs = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4});
    defer rhs.deinit();

    var add = try lhs.add(rhs);
    defer add.deinit();
    var sub = try rhs.sub(lhs);
    defer sub.deinit();
    var mul = try lhs.mul(rhs);
    defer mul.deinit();

    var add_scalar = try rhs.addScalar(2.0);
    defer add_scalar.deinit();
    var mul_scalar = try rhs.mulScalar(2.0);
    defer mul_scalar.deinit();

    var mat_lhs = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2, 3,
        4, 5, 6,
    }, &.{ 2, 3 });
    defer mat_lhs.deinit();
    var mat_rhs = try vx.Array(f32).fromSlice(allocator, &.{
        7,  8,
        9,  10,
        11, 12,
    }, &.{ 3, 2 });
    defer mat_rhs.deinit();
    var matmul = try mat_lhs.matmul(mat_rhs);
    defer matmul.deinit();

    const add_ok = equalF32(add.data, &.{ 11, 22, 33, 44 });
    const sub_ok = equalF32(sub.data, &.{ 9, 18, 27, 36 });
    const mul_ok = equalF32(mul.data, &.{ 10, 40, 90, 160 });
    const add_scalar_ok = equalF32(add_scalar.data, &.{ 12, 22, 32, 42 });
    const mul_scalar_ok = equalF32(mul_scalar.data, &.{ 20, 40, 60, 80 });
    const matmul_ok = equalF32(matmul.data, &.{ 58, 64, 139, 154 });
    const ok = add_ok and sub_ok and mul_ok and add_scalar_ok and mul_scalar_ok and matmul_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_cuda_dispatch_smoke\",\"ok\":{},\"add_ok\":{},\"sub_ok\":{},\"mul_ok\":{},\"add_scalar_ok\":{},\"mul_scalar_ok\":{},\"matmul_ok\":{},\"add0\":{d},\"matmul3\":{d}}}\n",
        .{ ok, add_ok, sub_ok, mul_ok, add_scalar_ok, mul_scalar_ok, matmul_ok, add.data[0], matmul.data[3] },
    );
    try stdout.interface.flush();

    if (!ok) std.process.exit(1);
}

fn equalF32(actual: []const f32, expected: []const f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}
