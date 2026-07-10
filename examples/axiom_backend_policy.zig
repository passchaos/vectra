//! Unified Axiom backend policy example.
//!
//! Axiom is the default backend policy for supported CPU and CUDA operations:
//!   zig build example-axiom-backend-policy

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer lhs.deinit();
    var rhs = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer rhs.deinit();

    const matmul_report = vx.axiom_backend.selectMatmul(f32, .prefer_cuda, lhs, rhs);
    var product = try vx.axiom_backend.matmul(f32, .prefer_cuda, lhs, rhs);
    defer product.deinit();

    var x = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer x.deinit();
    var y = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4});
    defer y.deinit();

    const add_report = vx.axiom_backend.selectElementwise(f32, .add, .prefer_cuda, x, y);
    var sum = try vx.axiom_backend.elementwise(f32, .add, .prefer_cuda, x, y);
    defer sum.deinit();

    const scalar_report = vx.axiom_backend.selectScalarElementwise(f32, .mul, .prefer_cuda, x, 3.0, .rhs);
    var scaled = try vx.axiom_backend.elementwiseScalar(f32, .mul, .prefer_cuda, x, 3.0, .rhs);
    defer scaled.deinit();

    try expectSlice(f32, product.data, &.{ 58, 64, 139, 154 });
    try expectSlice(f32, sum.data, &.{ 11, 22, 33, 44 });
    try expectSlice(f32, scaled.data, &.{ 3, 6, 9, 12 });

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        \\
        \\{{
        \\  "example": "axiom_backend_policy",
        \\  "cpu_dispatch_enabled": {},
        \\  "cuda_dispatch_enabled": {},
        \\  "matmul_route": "{s}",
        \\  "elementwise_route": "{s}",
        \\  "scalar_route": "{s}",
        \\  "matmul": [{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "sum": [{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "scaled": [{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "matmul_fingerprint": {d},
        \\  "elementwise_fingerprint": {d},
        \\  "scalar_fingerprint": {d},
        \\  "ok": true
        \\}}
        \\
    , .{
        matmul_report.axiom_cpu_enabled,
        matmul_report.axiom_cuda_enabled,
        matmul_report.selected.label(),
        add_report.selected.label(),
        scalar_report.selected.label(),
        product.data[0],
        product.data[1],
        product.data[2],
        product.data[3],
        sum.data[0],
        sum.data[1],
        sum.data[2],
        sum.data[3],
        scaled.data[0],
        scaled.data[1],
        scaled.data[2],
        scaled.data[3],
        matmul_report.fingerprint(),
        add_report.fingerprint(),
        scalar_report.fingerprint(),
    });
    try stdout.interface.flush();
}

fn expectSlice(comptime T: type, actual: []const T, expected: []const T) !void {
    if (actual.len != expected.len) return error.UnexpectedResult;
    for (actual, expected) |a, e| {
        if (a != e) return error.UnexpectedResult;
    }
}
