const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var no_fallback_error_ok = !available;
    var view_policy_ok = !available;
    var fingerprint = report.fingerprint();

    if (available) {
        var input = try vx.Array(f32).fromSliceOn(allocator, &.{ 0.25, 0.5, 0.75, 1.0 }, &.{ 2, 2 }, vx.mps(0));
        defer input.deinit();
        no_fallback_error_ok = vx.axiom_backend.deviceHostFallbackEnabled() or std.meta.isError(tryUnsupportedUnary(input));
        view_policy_ok = vx.axiom_backend.deviceHostFallbackEnabled() or
            std.meta.isError(tryDeviceView(input)) and
                std.meta.isError(tryDeviceSliceView(input));
        fingerprint ^= input.numel();
    }

    const ok = if (available)
        report.ok() and no_fallback_error_ok and view_policy_ok
    else
        !report.ok() and no_fallback_error_ok and view_policy_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_device_fallback_policy_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"device_host_fallback_enabled\":{},\"no_fallback_error_ok\":{},\"view_policy_ok\":{},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, vx.axiom_backend.deviceHostFallbackEnabled(), no_fallback_error_ok, view_policy_ok, fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn tryUnsupportedUnary(input: vx.Array(f32)) anyerror!void {
    var out = try input.asin();
    defer out.deinit();
}

fn tryDeviceView(input: vx.Array(f32)) anyerror!void {
    var view = try input.asView();
    defer view.deinit();
}

fn tryDeviceSliceView(input: vx.Array(f32)) anyerror!void {
    var view = try input.sliceView(&.{ .{ .start = 0, .stop = 1, .step = 1 }, .{ .start = 0, .stop = 2, .step = 1 } });
    defer view.deinit();
}
