const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var no_fallback_error_ok = !available;
    var fingerprint = report.fingerprint();

    if (available) {
        var lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 1, 2 }, vx.mps(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20, 30, 40 }, &.{ 1, 2, 2 }, vx.mps(0));
        defer rhs.deinit();
        no_fallback_error_ok = vx.axiom_backend.deviceHostFallbackEnabled() or std.meta.isError(tryMiddleBroadcast(lhs, rhs));
        fingerprint ^= lhs.numel() ^ (rhs.numel() << 8);
    }

    const ok = if (available)
        report.ok() and no_fallback_error_ok
    else
        !report.ok() and no_fallback_error_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_device_fallback_policy_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"device_host_fallback_enabled\":{},\"no_fallback_error_ok\":{},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, vx.axiom_backend.deviceHostFallbackEnabled(), no_fallback_error_ok, fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn tryMiddleBroadcast(lhs: vx.Array(f32), rhs: vx.Array(f32)) anyerror!void {
    var out = try lhs.add(rhs);
    defer out.deinit();
}
