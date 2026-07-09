//! Smoke gate for opt-in automatic Array(f32/f16/BFloat16) -> Axiom CUDA dispatch.

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
    var div = try rhs.div(lhs);
    defer div.deinit();

    var add_scalar = try rhs.addScalar(2.0);
    defer add_scalar.deinit();
    var mul_scalar = try rhs.mulScalar(2.0);
    defer mul_scalar.deinit();
    var div_scalar = try rhs.divScalar(2.0);
    defer div_scalar.deinit();
    var broadcast_scalar = try vx.Array(f32).fromSlice(allocator, &.{2.0}, &.{1});
    defer broadcast_scalar.deinit();
    var broadcast_add = try rhs.add(broadcast_scalar);
    defer broadcast_add.deinit();
    var broadcast_sub = try broadcast_scalar.sub(rhs);
    defer broadcast_sub.deinit();

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

    var lhs16 = try vx.Array(f16).fromSlice(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4) }, &.{ 2, 2 });
    defer lhs16.deinit();
    var rhs16 = try vx.Array(f16).fromSlice(allocator, &.{ @as(f16, 10), @as(f16, 20), @as(f16, 30), @as(f16, 40) }, &.{ 2, 2 });
    defer rhs16.deinit();
    var add16 = try lhs16.add(rhs16);
    defer add16.deinit();
    var matmul16 = try lhs16.matmul(rhs16);
    defer matmul16.deinit();

    var lhs_bf16 = try vx.Array(vx.BFloat16).fromSlice(allocator, &.{
        vx.BFloat16.fromF32(1),
        vx.BFloat16.fromF32(2),
        vx.BFloat16.fromF32(3),
        vx.BFloat16.fromF32(4),
    }, &.{ 2, 2 });
    defer lhs_bf16.deinit();
    var rhs_bf16 = try vx.Array(vx.BFloat16).fromSlice(allocator, &.{
        vx.BFloat16.fromF32(10),
        vx.BFloat16.fromF32(20),
        vx.BFloat16.fromF32(30),
        vx.BFloat16.fromF32(40),
    }, &.{ 2, 2 });
    defer rhs_bf16.deinit();
    var add_bf16 = try lhs_bf16.add(rhs_bf16);
    defer add_bf16.deinit();
    var matmul_bf16 = try lhs_bf16.matmul(rhs_bf16);
    defer matmul_bf16.deinit();

    const add_ok = equalF32(add.data, &.{ 11, 22, 33, 44 });
    const sub_ok = equalF32(sub.data, &.{ 9, 18, 27, 36 });
    const mul_ok = equalF32(mul.data, &.{ 10, 40, 90, 160 });
    const div_ok = equalF32(div.data, &.{ 10, 10, 10, 10 });
    const add_scalar_ok = equalF32(add_scalar.data, &.{ 12, 22, 32, 42 });
    const mul_scalar_ok = equalF32(mul_scalar.data, &.{ 20, 40, 60, 80 });
    const div_scalar_ok = equalF32(div_scalar.data, &.{ 5, 10, 15, 20 });
    const broadcast_add_ok = equalF32(broadcast_add.data, &.{ 12, 22, 32, 42 });
    const broadcast_sub_ok = equalF32(broadcast_sub.data, &.{ -8, -18, -28, -38 });
    const matmul_ok = equalF32(matmul.data, &.{ 58, 64, 139, 154 });
    const f16_add_ok = equalF16(add16.data, &.{ 11, 22, 33, 44 }, 0.02);
    const f16_matmul_ok = equalF16(matmul16.data, &.{ 70, 100, 150, 220 }, 0.25);
    const bf16_add_ok = equalBF16(add_bf16.data, &.{ 11, 22, 33, 44 }, 0.125);
    const bf16_matmul_ok = equalBF16(matmul_bf16.data, &.{ 70, 100, 150, 220 }, 0.5);
    const ok = add_ok and sub_ok and mul_ok and div_ok and add_scalar_ok and mul_scalar_ok and div_scalar_ok and broadcast_add_ok and broadcast_sub_ok and matmul_ok and f16_add_ok and f16_matmul_ok and bf16_add_ok and bf16_matmul_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_cuda_dispatch_smoke\",\"ok\":{},\"add_ok\":{},\"sub_ok\":{},\"mul_ok\":{},\"div_ok\":{},\"add_scalar_ok\":{},\"mul_scalar_ok\":{},\"div_scalar_ok\":{},\"broadcast_add_ok\":{},\"broadcast_sub_ok\":{},\"matmul_ok\":{},\"f16_add_ok\":{},\"f16_matmul_ok\":{},\"bf16_add_ok\":{},\"bf16_matmul_ok\":{},\"add0\":{d},\"matmul3\":{d}}}\n",
        .{ ok, add_ok, sub_ok, mul_ok, div_ok, add_scalar_ok, mul_scalar_ok, div_scalar_ok, broadcast_add_ok, broadcast_sub_ok, matmul_ok, f16_add_ok, f16_matmul_ok, bf16_add_ok, bf16_matmul_ok, add.data[0], matmul.data[3] },
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

fn equalF16(actual: []const f16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(@as(f32, @floatCast(a)) - e) > tolerance) return false;
    }
    return true;
}

fn equalBF16(actual: []const vx.BFloat16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a.toF32() - e) > tolerance) return false;
    }
    return true;
}
