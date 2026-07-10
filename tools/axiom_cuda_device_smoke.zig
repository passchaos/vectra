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
    var chained_matmul_add_ok = !vx.axiom_cuda.enabled();
    var chained_matmul_sub_ok = !vx.axiom_cuda.enabled();
    var chained_sqrt_ok = !vx.axiom_cuda.enabled();
    var chained_exp_ok = !vx.axiom_cuda.enabled();
    var reversed_add_fusion_ok = !vx.axiom_cuda.enabled();
    var pending_fusion_status_ok = !vx.axiom_cuda.enabled();
    var bf16_chained_sqrt_ok = !vx.axiom_cuda.enabled();
    var bf16_chained_exp_ok = !vx.axiom_cuda.enabled();
    var bf16_scalar_mul_ok = !vx.axiom_cuda.enabled();
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
        direct_matmul_ok = product.device.isCuda() and equalF32(product_host.data, &.{ 3, 3, 7, 7 });

        var chained = try product.add(addend);
        defer chained.deinit();
        const chained_status_ok = chained.fusionStatus() == .cuda_matmul_add;
        var chained_host = try chained.cpu();
        defer chained_host.deinit();
        chained_matmul_add_ok = chained.device.isCuda() and equalF32(chained_host.data, &.{ 4, 4, 8, 8 });

        var reversed_chained = try addend.add(product);
        defer reversed_chained.deinit();
        var reversed_chained_host = try reversed_chained.cpu();
        defer reversed_chained_host.deinit();
        reversed_add_fusion_ok = reversed_chained.fusionStatus() == .cuda_matmul_add and equalF32(reversed_chained_host.data, &.{ 4, 4, 8, 8 });

        var chained_sub = try product.sub(addend);
        defer chained_sub.deinit();
        const chained_sub_status_ok = chained_sub.fusionStatus() == .cuda_matmul_sub;
        var chained_sub_host = try chained_sub.cpu();
        defer chained_sub_host.deinit();
        chained_matmul_sub_ok = chained_sub.device.isCuda() and equalF32(chained_sub_host.data, &.{ 2, 2, 6, 6 });

        var chained_sqrt = try chained.sqrt();
        defer chained_sqrt.deinit();
        const chained_sqrt_status_ok = chained_sqrt.fusionStatus() == .cuda_matmul_add_sqrt;
        var chained_sqrt_host = try chained_sqrt.cpu();
        defer chained_sqrt_host.deinit();
        chained_sqrt_ok = chained_sqrt.device.isCuda() and approxF32(chained_sqrt_host.data[0], 2.0, 0.01);

        var chained_add_exp = try chained.exp();
        defer chained_add_exp.deinit();
        const chained_add_exp_status_ok = chained_add_exp.fusionStatus() == .cuda_matmul_add_exp;
        var chained_add_exp_host = try chained_add_exp.cpu();
        defer chained_add_exp_host.deinit();
        const chained_add_exp_ok = chained_add_exp.device.isCuda() and approxF32(chained_add_exp_host.data[0], std.math.exp(@as(f32, 4.0)), 2.0);

        var chained_exp_input = try chained_sub.addScalar(1.0);
        defer chained_exp_input.deinit();
        var chained_exp = try chained_exp_input.exp();
        defer chained_exp.deinit();
        var chained_exp_host = try chained_exp.cpu();
        defer chained_exp_host.deinit();
        chained_exp_ok = chained_exp.device.isCuda() and approxF32(chained_exp_host.data[0], std.math.exp(@as(f32, 3.0)), 0.25);
        pending_fusion_status_ok = chained_status_ok and chained_sub_status_ok and chained_sqrt_status_ok and chained_add_exp_status_ok and chained_add_exp_ok;

        var fused = try vx.matmulAdd(lhs, rhs, addend);
        defer fused.deinit();
        var fused_host = try fused.cpu();
        defer fused_host.deinit();
        direct_matmul_add_ok = fused.device.isCuda() and fused.device_storage != null and equalF32(fused_host.data, &.{ 4, 4, 8, 8 });

        var bf16_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{
            vx.BFloat16.fromF32(1),
            vx.BFloat16.fromF32(2),
            vx.BFloat16.fromF32(3),
            vx.BFloat16.fromF32(4),
        }, &.{ 2, 2 }, vx.cuda(0));
        defer bf16_lhs.deinit();
        var bf16_rhs = try vx.Array(vx.BFloat16).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer bf16_rhs.deinit();
        var bf16_addend = try vx.Array(vx.BFloat16).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer bf16_addend.deinit();
        var bf16_product = try bf16_lhs.matmul(bf16_rhs);
        defer bf16_product.deinit();
        var bf16_chained = try bf16_product.add(bf16_addend);
        defer bf16_chained.deinit();
        const bf16_chained_status_ok = bf16_chained.fusionStatus() == .cuda_matmul_add;
        var bf16_sqrt = try bf16_chained.sqrt();
        defer bf16_sqrt.deinit();
        const bf16_sqrt_status_ok = bf16_sqrt.fusionStatus() == .cuda_matmul_add_sqrt;
        var bf16_sqrt_host = try bf16_sqrt.cpu();
        defer bf16_sqrt_host.deinit();
        bf16_chained_sqrt_ok = bf16_sqrt.device.isCuda() and approxF32(bf16_sqrt_host.data[0].toF32(), 2.0, 0.05);

        var bf16_exp = try bf16_chained.exp();
        defer bf16_exp.deinit();
        const bf16_exp_status_ok = bf16_exp.fusionStatus() == .cuda_matmul_add_exp;
        var bf16_exp_host = try bf16_exp.cpu();
        defer bf16_exp_host.deinit();
        bf16_chained_exp_ok = bf16_exp.device.isCuda() and approxF32(bf16_exp_host.data[0].toF32(), std.math.exp(@as(f32, 4.0)), 2.0);

        var bf16_scaled = try bf16_chained.mulScalar(vx.BFloat16.fromF32(0.25));
        defer bf16_scaled.deinit();
        var bf16_scaled_host = try bf16_scaled.cpu();
        defer bf16_scaled_host.deinit();
        bf16_scalar_mul_ok = bf16_scaled.device.isCuda() and bf16_scaled.device_storage != null and approxF32(bf16_scaled_host.data[0].toF32(), 1.0, 0.05);
        pending_fusion_status_ok = pending_fusion_status_ok and bf16_chained_status_ok and bf16_sqrt_status_ok and bf16_exp_status_ok;
    }
    ok = ok and direct_storage_ok and direct_add_ok and direct_matmul_ok and direct_matmul_add_ok and chained_matmul_add_ok and chained_matmul_sub_ok and chained_sqrt_ok and chained_exp_ok and reversed_add_fusion_ok and pending_fusion_status_ok and bf16_chained_sqrt_ok and bf16_chained_exp_ok and bf16_scalar_mul_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_cuda_device_smoke\",\"enabled\":{},\"status\":\"{s}\",\"ok\":{},\"bytes\":{d},\"fingerprint\":{d},\"direct_storage_ok\":{},\"direct_add_ok\":{},\"direct_matmul_ok\":{},\"direct_matmul_add_ok\":{},\"chained_matmul_add_ok\":{},\"chained_matmul_sub_ok\":{},\"chained_sqrt_ok\":{},\"chained_exp_ok\":{},\"reversed_add_fusion_ok\":{},\"pending_fusion_status_ok\":{},\"bf16_chained_sqrt_ok\":{},\"bf16_chained_exp_ok\":{},\"bf16_scalar_mul_ok\":{}}}\n",
        .{ vx.axiom_cuda.enabled(), status, ok, bytes, fingerprint, direct_storage_ok, direct_add_ok, direct_matmul_ok, direct_matmul_add_ok, chained_matmul_add_ok, chained_matmul_sub_ok, chained_sqrt_ok, chained_exp_ok, reversed_add_fusion_ok, pending_fusion_status_ok, bf16_chained_sqrt_ok, bf16_chained_exp_ok, bf16_scalar_mul_ok },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn approxF32(actual: f32, expected: f32, tolerance: f32) bool {
    return @abs(actual - expected) <= tolerance;
}

fn equalF32(actual: []const f32, expected: []const f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}
