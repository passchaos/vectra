const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var f32_ok = !available;
    var f16_ok = !available;
    var bf16_ok = !available;
    var fingerprint = report.fingerprint();

    if (available) {
        var inner_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 2, 3,
            4, 5, 6,
        }, &.{ 2, 3 }, vx.mps(0));
        defer inner_lhs.deinit();
        var inner_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            10, 20, 30,
            1,  1,  1,
        }, &.{ 2, 3 }, vx.mps(0));
        defer inner_rhs.deinit();
        var inner_out = try inner_lhs.inner(inner_rhs);
        defer inner_out.deinit();
        var inner_back = try inner_out.cpu();
        defer inner_back.deinit();
        var outer_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3 }, &.{3}, vx.mps(0));
        defer outer_lhs.deinit();
        var outer_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20 }, &.{2}, vx.mps(0));
        defer outer_rhs.deinit();
        var outer_out = try outer_lhs.outer(outer_rhs);
        defer outer_out.deinit();
        var outer_back = try outer_out.cpu();
        defer outer_back.deinit();
        f32_ok = inner_out.device.isMps() and inner_out.device_storage != null and
            outer_out.device.isMps() and outer_out.device_storage != null and
            std.mem.eql(usize, inner_back.shape, &.{ 2, 2 }) and
            std.mem.eql(usize, outer_back.shape, &.{ 3, 2 }) and
            closeF32(inner_back.data, &.{ 140, 6, 320, 15 }, 0.001) and
            closeF32(outer_back.data, &.{ 10, 20, 20, 40, 30, 60 }, 0.001);
        fingerprint ^= hashF32(inner_back.data) ^ hashF32(outer_back.data);

        var inner_lhs16 = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4), @as(f16, 5), @as(f16, 6) }, &.{ 2, 3 }, vx.mps(0));
        defer inner_lhs16.deinit();
        var inner_rhs16 = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20), @as(f16, 30), @as(f16, 1), @as(f16, 1), @as(f16, 1) }, &.{ 2, 3 }, vx.mps(0));
        defer inner_rhs16.deinit();
        var inner_out16 = try inner_lhs16.inner(inner_rhs16);
        defer inner_out16.deinit();
        var inner_back16 = try inner_out16.cpu();
        defer inner_back16.deinit();
        var outer_lhs16 = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3) }, &.{3}, vx.mps(0));
        defer outer_lhs16.deinit();
        var outer_rhs16 = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20) }, &.{2}, vx.mps(0));
        defer outer_rhs16.deinit();
        var outer_out16 = try outer_lhs16.outer(outer_rhs16);
        defer outer_out16.deinit();
        var outer_back16 = try outer_out16.cpu();
        defer outer_back16.deinit();
        f16_ok = inner_out16.device.isMps() and inner_out16.device_storage != null and
            outer_out16.device.isMps() and outer_out16.device_storage != null and
            std.mem.eql(usize, inner_back16.shape, &.{ 2, 2 }) and
            std.mem.eql(usize, outer_back16.shape, &.{ 3, 2 }) and
            closeF16(inner_back16.data, &.{ 140, 6, 320, 15 }, 0.5) and
            closeF16(outer_back16.data, &.{ 10, 20, 20, 40, 30, 60 }, 0.25);
        fingerprint ^= hashF16(inner_back16.data) ^ hashF16(outer_back16.data);

        var inner_lhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4), vx.BFloat16.fromF32(5), vx.BFloat16.fromF32(6) }, &.{ 2, 3 }, vx.mps(0));
        defer inner_lhs_bf16.deinit();
        var inner_rhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20), vx.BFloat16.fromF32(30), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1) }, &.{ 2, 3 }, vx.mps(0));
        defer inner_rhs_bf16.deinit();
        var inner_out_bf16 = try inner_lhs_bf16.inner(inner_rhs_bf16);
        defer inner_out_bf16.deinit();
        var inner_back_bf16 = try inner_out_bf16.cpu();
        defer inner_back_bf16.deinit();
        var outer_lhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3) }, &.{3}, vx.mps(0));
        defer outer_lhs_bf16.deinit();
        var outer_rhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20) }, &.{2}, vx.mps(0));
        defer outer_rhs_bf16.deinit();
        var outer_out_bf16 = try outer_lhs_bf16.outer(outer_rhs_bf16);
        defer outer_out_bf16.deinit();
        var outer_back_bf16 = try outer_out_bf16.cpu();
        defer outer_back_bf16.deinit();
        bf16_ok = inner_out_bf16.device.isMps() and inner_out_bf16.device_storage != null and
            outer_out_bf16.device.isMps() and outer_out_bf16.device_storage != null and
            std.mem.eql(usize, inner_back_bf16.shape, &.{ 2, 2 }) and
            std.mem.eql(usize, outer_back_bf16.shape, &.{ 3, 2 }) and
            closeBF16(inner_back_bf16.data, &.{ 140, 6, 320, 15 }, 1.0) and
            closeBF16(outer_back_bf16.data, &.{ 10, 20, 20, 40, 30, 60 }, 0.5);
        fingerprint ^= hashBF16(inner_back_bf16.data) ^ hashBF16(outer_back_bf16.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_inner_outer_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, f32_ok, f16_ok, bf16_ok, fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn closeF32(actual: []const f32, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a - e) > tolerance) return false;
    }
    return true;
}

fn closeF16(actual: []const f16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(@as(f32, a) - e) > tolerance) return false;
    }
    return true;
}

fn closeBF16(actual: []const vx.BFloat16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a.toF32() - e) > tolerance) return false;
    }
    return true;
}

fn hashF32(values: []const f32) u64 {
    var hasher = std.hash.Wyhash.init(0x4d50_5701_1e02_3d32);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}

fn hashF16(values: []const f16) u64 {
    var hasher = std.hash.Wyhash.init(0x4d50_5701_1e02_3d16);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &bytes, @bitCast(value), .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}

fn hashBF16(values: []const vx.BFloat16) u64 {
    var hasher = std.hash.Wyhash.init(0x4d50_5701_1e02_b3b1);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, values.len, .little);
    hasher.update(&len_bytes);
    for (values) |value| {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(u16, &bytes, value.bits, .little);
        hasher.update(&bytes);
    }
    return hasher.final();
}
