const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var no_fallback_error_ok = !available;
    var fingerprint = report.fingerprint();

    if (available) {
        var input = try vx.Array(f32).fromSliceOn(allocator, &.{ 0.25, 0.5, 0.75, 1.0 }, &.{ 2, 2 }, vx.mps(0));
        defer input.deinit();
        no_fallback_error_ok = vx.axiom_backend.deviceHostFallbackEnabled() or std.meta.isError(tryUnsupportedUnary(input));
        fingerprint ^= input.numel();
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

fn tryUnsupportedUnary(input: vx.Array(f32)) anyerror!void {
    var out = try input.asin();
    defer out.deinit();
}
