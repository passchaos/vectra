const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var a32 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a32.deinit();
    var b32 = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b32.deinit();
    var out32 = try a32.matmul(b32);
    defer out32.deinit();
    var add32 = try a32.add(a32);
    defer add32.deinit();
    var sub32 = try a32.sub(a32);
    defer sub32.deinit();
    var mul32 = try a32.mul(a32);
    defer mul32.deinit();
    var div32 = try a32.div(a32);
    defer div32.deinit();
    var add_scalar32 = try a32.addScalar(2);
    defer add_scalar32.deinit();
    var sub_scalar32 = try a32.subScalar(2);
    defer sub_scalar32.deinit();
    var mul_scalar32 = try a32.mulScalar(2);
    defer mul_scalar32.deinit();
    var div_scalar32 = try a32.divScalar(2);
    defer div_scalar32.deinit();
    var matvec32_rhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer matvec32_rhs.deinit();
    var matvec32 = try a32.matvec(matvec32_rhs);
    defer matvec32.deinit();
    var dot32 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer dot32.deinit();
    var dot32_out = try dot32.dot(dot32);
    defer dot32_out.deinit();

    var a64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a64.deinit();
    var b64 = try vx.Array(f64).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b64.deinit();
    var out64 = try a64.matmul(b64);
    defer out64.deinit();
    var add64 = try a64.add(a64);
    defer add64.deinit();
    var sub64 = try a64.sub(a64);
    defer sub64.deinit();
    var mul64 = try a64.mul(a64);
    defer mul64.deinit();
    var div64 = try a64.div(a64);
    defer div64.deinit();
    var add_scalar64 = try a64.addScalar(2);
    defer add_scalar64.deinit();
    var sub_scalar64 = try a64.subScalar(2);
    defer sub_scalar64.deinit();
    var mul_scalar64 = try a64.mulScalar(2);
    defer mul_scalar64.deinit();
    var div_scalar64 = try a64.divScalar(2);
    defer div_scalar64.deinit();
    var broadcast_scalar64 = try vx.Array(f64).fromSlice(allocator, &.{2}, &.{1});
    defer broadcast_scalar64.deinit();
    var broadcast_sub64 = try broadcast_scalar64.sub(a64);
    defer broadcast_sub64.deinit();
    var broadcast_div64 = try a64.div(broadcast_scalar64);
    defer broadcast_div64.deinit();
    var matvec64_rhs = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer matvec64_rhs.deinit();
    var matvec64 = try vx.linalg.matvec(f64, a64, matvec64_rhs);
    defer matvec64.deinit();
    var vec64 = try vx.Array(f64).fromSlice(allocator, &.{ 1, 2 }, &.{2});
    defer vec64.deinit();
    var vecmat64 = try vec64.matmul(a64);
    defer vecmat64.deinit();
    const trace64 = try vx.linalg.trace(f64, out64);

    const matmul_ok = out32.data[0] == 58 and out32.data[3] == 154 and out64.data[0] == 58 and out64.data[3] == 154;
    const elementwise_ok = equalF32(add32.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF32(sub32.data, &.{ 0, 0, 0, 0, 0, 0 }) and
        equalF32(mul32.data, &.{ 1, 4, 9, 16, 25, 36 }) and
        equalF32(div32.data, &.{ 1, 1, 1, 1, 1, 1 }) and
        equalF64(add64.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF64(sub64.data, &.{ 0, 0, 0, 0, 0, 0 }) and
        equalF64(mul64.data, &.{ 1, 4, 9, 16, 25, 36 }) and
        equalF64(div64.data, &.{ 1, 1, 1, 1, 1, 1 });
    const scalar_ok = equalF32(add_scalar32.data, &.{ 3, 4, 5, 6, 7, 8 }) and
        equalF32(sub_scalar32.data, &.{ -1, 0, 1, 2, 3, 4 }) and
        equalF32(mul_scalar32.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF32(div_scalar32.data, &.{ 0.5, 1, 1.5, 2, 2.5, 3 }) and
        equalF64(add_scalar64.data, &.{ 3, 4, 5, 6, 7, 8 }) and
        equalF64(sub_scalar64.data, &.{ -1, 0, 1, 2, 3, 4 }) and
        equalF64(mul_scalar64.data, &.{ 2, 4, 6, 8, 10, 12 }) and
        equalF64(div_scalar64.data, &.{ 0.5, 1, 1.5, 2, 2.5, 3 }) and
        equalF64(broadcast_sub64.data, &.{ 1, 0, -1, -2, -3, -4 }) and
        equalF64(broadcast_div64.data, &.{ 0.5, 1, 1.5, 2, 2.5, 3 });
    const vector_ok = equalF32(matvec32.data, &.{ 14, 32 }) and
        dot32_out.data[0] == 14 and
        equalF64(matvec64.data, &.{ 14, 32 }) and
        equalF64(vecmat64.data, &.{ 9, 12, 15 }) and
        trace64 == 212;
    const ok = matmul_ok and elementwise_ok and scalar_ok and vector_ok;
    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print("{{\"kind\":\"vectra_axiom_cpu_dispatch_smoke\",\"enabled\":{},\"ok\":{},\"matmul_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"vector_ok\":{},\"f32_0\":{d},\"f64_3\":{d},\"add32_5\":{d},\"div64_0\":{d},\"sub_scalar64_0\":{d},\"matvec32_1\":{d},\"vecmat64_2\":{d},\"trace64\":{d}}}\n", .{ vx.axiom_cpu.enabled(), ok, matmul_ok, elementwise_ok, scalar_ok, vector_ok, out32.data[0], out64.data[3], add32.data[5], div64.data[0], sub_scalar64.data[0], matvec32.data[1], vecmat64.data[2], trace64 });
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

fn equalF64(actual: []const f64, expected: []const f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}
