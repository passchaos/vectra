//! Explicit Axiom backend CUDA diagnostic example.
//!
//! Prints Axiom's facade-owned CUDA smoke and dtype metadata:
//!   zig build example-axiom-cuda-bridge

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer lhs.deinit();
    var rhs = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 40 }, &.{4});
    defer rhs.deinit();

    const smoke = vx.axiom_backend.cuda.runSmoke(allocator);
    const route = vx.axiom_backend.selectElementwise(f32, .add, .prefer_cuda, lhs, rhs);

    var out = try vx.axiom_backend.elementwise(f32, .add, .prefer_cuda, lhs, rhs);
    defer out.deinit();
    try expectSlice(f32, out.data, &.{ 11, 22, 33, 44 });

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
        \\  "dtype_support_fingerprint": {d},
        \\  "selected_route": "{s}",
        \\  "used_accelerated_add": {},
        \\  "ok": true
        \\}}
        \\
    , .{
        vx.axiom_backend.cuda.enabled(),
        smoke.status.label(),
        smoke.ok(),
        vx.axiom_backend.cuda.cudaDTypeSupportRecords().len,
        vx.axiom_backend.cuda.cudaDTypeBridgeCount(),
        vx.axiom_backend.cuda.cudaDTypeNativeSeedCount(),
        vx.axiom_backend.cuda.cudaDTypeSupportFingerprint(),
        route.selected.label(),
        route.selected == .axiom_cuda and smoke.status == .ran,
    });
    try stdout.interface.flush();
}

fn expectSlice(comptime T: type, actual: []const T, expected: []const T) !void {
    if (actual.len != expected.len) return error.UnexpectedResult;
    for (actual, expected) |a, e| {
        if (a != e) return error.UnexpectedResult;
    }
}
