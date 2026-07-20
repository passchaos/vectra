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
        var lhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 2,
            3, 4,

            5, 6,
            7, 8,
        }, &.{ 2, 2, 2 }, vx.mps(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 0,
            0, 1,

            2, 1,
            1, 2,
        }, &.{ 2, 2, 2 }, vx.mps(0));
        defer rhs.deinit();
        var bmm = try lhs.bmm(rhs);
        defer bmm.deinit();
        var bmm_back = try bmm.cpu();
        defer bmm_back.deinit();
        var matmul = try lhs.matmul(rhs);
        defer matmul.deinit();
        var matmul_back = try matmul.cpu();
        defer matmul_back.deinit();
        f32_ok = bmm.device.isMps() and bmm.device_storage != null and
            matmul.device.isMps() and matmul.device_storage != null and
            std.mem.eql(usize, bmm_back.shape, &.{ 2, 2, 2 }) and
            std.mem.eql(usize, matmul_back.shape, &.{ 2, 2, 2 }) and
            closeF32(bmm_back.data, &.{ 1, 2, 3, 4, 16, 17, 22, 23 }, 0.001) and
            closeF32(matmul_back.data, bmm_back.data, 0.001);
        fingerprint ^= hashF32(bmm_back.data) ^ hashF32(matmul_back.data);

        var f16_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4), @as(f16, 5), @as(f16, 6), @as(f16, 7), @as(f16, 8) }, &.{ 2, 2, 2 }, vx.mps(0));
        defer f16_lhs.deinit();
        var f16_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 0), @as(f16, 0), @as(f16, 1), @as(f16, 2), @as(f16, 1), @as(f16, 1), @as(f16, 2) }, &.{ 2, 2, 2 }, vx.mps(0));
        defer f16_rhs.deinit();
        var f16_bmm = try f16_lhs.bmm(f16_rhs);
        defer f16_bmm.deinit();
        var f16_bmm_back = try f16_bmm.cpu();
        defer f16_bmm_back.deinit();
        f16_ok = f16_bmm.device.isMps() and f16_bmm.device_storage != null and
            std.mem.eql(usize, f16_bmm_back.shape, &.{ 2, 2, 2 }) and
            closeF16(f16_bmm_back.data, &.{ 1, 2, 3, 4, 16, 17, 22, 23 }, 0.125);
        fingerprint ^= hashF16(f16_bmm_back.data);

        var bf16_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4), vx.BFloat16.fromF32(5), vx.BFloat16.fromF32(6), vx.BFloat16.fromF32(7), vx.BFloat16.fromF32(8) }, &.{ 2, 2, 2 }, vx.mps(0));
        defer bf16_lhs.deinit();
        var bf16_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(0), vx.BFloat16.fromF32(0), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2) }, &.{ 2, 2, 2 }, vx.mps(0));
        defer bf16_rhs.deinit();
        var bf16_bmm = try bf16_lhs.bmm(bf16_rhs);
        defer bf16_bmm.deinit();
        var bf16_bmm_back = try bf16_bmm.cpu();
        defer bf16_bmm_back.deinit();
        bf16_ok = bf16_bmm.device.isMps() and bf16_bmm.device_storage != null and
            std.mem.eql(usize, bf16_bmm_back.shape, &.{ 2, 2, 2 }) and
            closeBF16(bf16_bmm_back.data, &.{ 1, 2, 3, 4, 16, 17, 22, 23 }, 0.25);
        fingerprint ^= hashBF16(bf16_bmm_back.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_bmm_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_3d32);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_3d16);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_b3b1);
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
