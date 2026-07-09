//! Smoke gate for explicit Vectra -> Axiom CUDA device-buffer handle seed.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var host = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer host.deinit();

    var maybe_device = try vx.axiom_cuda.toDeviceF32(allocator, host);
    const status: []const u8 = if (maybe_device != null) "allocated" else if (vx.axiom_cuda.enabled()) "unavailable" else "disabled";
    var ok = !vx.axiom_cuda.enabled() or maybe_device != null;
    var fingerprint: u64 = 0;
    var bytes: usize = 0;
    if (maybe_device) |*device| {
        defer device.deinit();
        ok = device.ok();
        fingerprint = device.fingerprint();
        bytes = device.required_bytes;
    }

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_cuda_device_smoke\",\"enabled\":{},\"status\":\"{s}\",\"ok\":{},\"bytes\":{d},\"fingerprint\":{d}}}\n",
        .{ vx.axiom_cuda.enabled(), status, ok, bytes, fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
