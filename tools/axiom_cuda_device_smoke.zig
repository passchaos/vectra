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

    var direct_storage_ok = !vx.axiom_cuda.enabled();
    var direct_add_ok = !vx.axiom_cuda.enabled();
    var direct_matmul_ok = !vx.axiom_cuda.enabled();
    var direct_matmul_add_ok = !vx.axiom_cuda.enabled();
    if (vx.Device.cuda(0).isAvailable()) {
        var lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, vx.cuda(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer rhs.deinit();
        var addend = try vx.Array(f32).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer addend.deinit();
        direct_storage_ok = lhs.device_storage != null and rhs.device_storage != null and lhs.data.len == 0 and rhs.data.len == 0;

        var sum = try lhs.add(rhs);
        defer sum.deinit();
        var sum_host = try sum.cpu();
        defer sum_host.deinit();
        direct_add_ok = sum.device.isCuda() and sum.device_storage != null and equalF32(sum_host.data, &.{ 2, 3, 4, 5 });

        var product = try lhs.matmul(rhs);
        defer product.deinit();
        var product_host = try product.cpu();
        defer product_host.deinit();
        direct_matmul_ok = product.device.isCuda() and product.device_storage != null and equalF32(product_host.data, &.{ 3, 3, 7, 7 });

        var fused = try vx.matmulAdd(lhs, rhs, addend);
        defer fused.deinit();
        var fused_host = try fused.cpu();
        defer fused_host.deinit();
        direct_matmul_add_ok = fused.device.isCuda() and fused.device_storage != null and equalF32(fused_host.data, &.{ 4, 4, 8, 8 });
    }
    ok = ok and direct_storage_ok and direct_add_ok and direct_matmul_ok and direct_matmul_add_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_cuda_device_smoke\",\"enabled\":{},\"status\":\"{s}\",\"ok\":{},\"bytes\":{d},\"fingerprint\":{d},\"direct_storage_ok\":{},\"direct_add_ok\":{},\"direct_matmul_ok\":{},\"direct_matmul_add_ok\":{}}}\n",
        .{ vx.axiom_cuda.enabled(), status, ok, bytes, fingerprint, direct_storage_ok, direct_add_ok, direct_matmul_ok, direct_matmul_add_ok },
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
