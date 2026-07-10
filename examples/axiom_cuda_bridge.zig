//! Explicit Axiom CUDA bridge example.
//!
//! Prints Axiom's smoke report and buffer-plan metadata:
//!   zig build example-axiom-cuda-bridge

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer lhs.deinit();
    var rhs = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4});
    defer rhs.deinit();

    const plan = vx.axiom_cuda.planArrayF32(lhs, "lhs");
    const smoke = vx.axiom_cuda.runSmoke(allocator);

    var out = try vx.axiom_cuda.tryAddF32(lhs, rhs);
    if (out) |*accelerated| {
        defer accelerated.deinit();
        try expectSlice(f32, accelerated.data, &.{ 11, 22, 33, 44 });
    } else {
        var fallback = try lhs.add(rhs);
        defer fallback.deinit();
        try expectSlice(f32, fallback.data, &.{ 11, 22, 33, 44 });
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        \\
        \\{{
        \\  "example": "axiom_cuda_bridge",
        \\  "enabled": {},
        \\  "smoke_status": "{s}",
        \\  "smoke_ok": {},
        \\  "dtype_support_count": {d},
        \\  "dtype_bridge_count": {d},
        \\  "dtype_native_seed_count": {d},
        \\  "buffer_plan_ok": {},
        \\  "buffer_plan_elements": {d},
        \\  "buffer_plan_fingerprint": {d},
        \\  "used_accelerated_add": {},
        \\  "ok": true
        \\}}
        \\
    , .{
        vx.axiom_cuda.enabled(),
        smoke.status.label(),
        smoke.ok(),
        smoke.dtype_support_count,
        smoke.dtype_bridge_count,
        smoke.dtype_native_seed_count,
        plan.ok,
        plan.logical_elements,
        plan.fingerprint,
        out != null,
    });
    try stdout.interface.flush();
}

fn expectSlice(comptime T: type, actual: []const T, expected: []const T) !void {
    if (actual.len != expected.len) return error.UnexpectedResult;
    for (actual, expected) |a, e| {
        if (a != e) return error.UnexpectedResult;
    }
}
