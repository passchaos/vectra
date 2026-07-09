const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var a = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    const report = vx.axiom_backend.selectMatmul(f32, .prefer_cuda, a, b);
    var out = try vx.axiom_backend.matmul(f32, .prefer_axiom_cpu, a, b);
    defer out.deinit();
    var lhs32 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer lhs32.deinit();
    var rhs32 = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4});
    defer rhs32.deinit();
    const ew32_report = vx.axiom_backend.selectElementwise(f32, .add, .prefer_cuda, lhs32, rhs32);
    var ew32 = try vx.axiom_backend.elementwise(f32, .add, .prefer_cuda, lhs32, rhs32);
    defer ew32.deinit();

    var lhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 8, 6, 4, 2 }, &.{4});
    defer lhs64.deinit();
    var rhs64 = try vx.Array(f64).fromSlice(allocator, &.{ 2, 3, 4, 2 }, &.{4});
    defer rhs64.deinit();
    const ew64_report = vx.axiom_backend.selectElementwise(f64, .div, .prefer_axiom_cpu, lhs64, rhs64);
    var ew64 = try vx.axiom_backend.elementwise(f64, .div, .prefer_axiom_cpu, lhs64, rhs64);
    defer ew64.deinit();
    const scalar64_report = vx.axiom_backend.selectScalarElementwise(f64, .sub, .prefer_axiom_cpu, lhs64, 2.0, .rhs);
    var scalar64 = try vx.axiom_backend.elementwiseScalar(f64, .sub, .prefer_axiom_cpu, lhs64, 2.0, .rhs);
    defer scalar64.deinit();

    const matmul_ok = report.ok() and out.data[0] == 58 and out.data[3] == 154;
    const elementwise_ok = ew32_report.ok() and ew64_report.ok() and
        equalF32(ew32.data, &.{ 11, 22, 33, 44 }) and
        equalF64(ew64.data, &.{ 4, 2, 1, 1 });
    const scalar_ok = scalar64_report.ok() and equalF64(scalar64.data, &.{ 6, 4, 2, 0 });
    const ok = matmul_ok and elementwise_ok and scalar_ok;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_backend_policy_smoke\",\"ok\":{},\"matmul_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"selected\":\"{s}\",\"elementwise32_selected\":\"{s}\",\"elementwise64_selected\":\"{s}\",\"scalar64_selected\":\"{s}\",\"cpu_enabled\":{},\"cuda_enabled\":{},\"fingerprint\":{d},\"elementwise_fingerprint\":{d},\"scalar_fingerprint\":{d}}}\n",
        .{ ok, matmul_ok, elementwise_ok, scalar_ok, report.selected.label(), ew32_report.selected.label(), ew64_report.selected.label(), scalar64_report.selected.label(), report.axiom_cpu_enabled, report.axiom_cuda_enabled, report.fingerprint(), ew32_report.fingerprint() ^ ew64_report.fingerprint(), scalar64_report.fingerprint() },
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

fn equalF64(actual: []const f64, expected: []const f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}
