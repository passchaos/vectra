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
        var input = try vx.Array(f32).fromSliceOn(allocator, &.{ -2, -1, 0, 1, 2, 3 }, &.{ 2, 3 }, vx.mps(0));
        defer input.deinit();
        var gelu = try input.gelu();
        defer gelu.deinit();
        var gelu_back = try gelu.cpu();
        defer gelu_back.deinit();
        f32_ok = gelu.device.isMps() and gelu.device_storage != null and
            closeF32(gelu_back.data, &.{ geluExpected(-2.0), geluExpected(-1.0), 0, geluExpected(1.0), geluExpected(2.0), geluExpected(3.0) }, 0.001);
        fingerprint ^= hashF32(gelu_back.data);

        var f16_input = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, -2), @as(f16, -1), @as(f16, 0), @as(f16, 1), @as(f16, 2), @as(f16, 3) }, &.{ 2, 3 }, vx.mps(0));
        defer f16_input.deinit();
        var f16_gelu = try f16_input.gelu();
        defer f16_gelu.deinit();
        var f16_gelu_back = try f16_gelu.cpu();
        defer f16_gelu_back.deinit();
        f16_ok = f16_gelu.device.isMps() and f16_gelu.device_storage != null and
            closeF16(f16_gelu_back.data, &.{ geluExpected(-2.0), geluExpected(-1.0), 0, geluExpected(1.0), geluExpected(2.0), geluExpected(3.0) }, 0.05);
        fingerprint ^= hashF16(f16_gelu_back.data);

        var bf16_input = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(-2), vx.BFloat16.fromF32(-1), vx.BFloat16.fromF32(0), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3) }, &.{ 2, 3 }, vx.mps(0));
        defer bf16_input.deinit();
        var bf16_gelu = try bf16_input.gelu();
        defer bf16_gelu.deinit();
        var bf16_gelu_back = try bf16_gelu.cpu();
        defer bf16_gelu_back.deinit();
        bf16_ok = bf16_gelu.device.isMps() and bf16_gelu.device_storage != null and
            closeBF16(bf16_gelu_back.data, &.{ geluExpected(-2.0), geluExpected(-1.0), 0, geluExpected(1.0), geluExpected(2.0), geluExpected(3.0) }, 0.125);
        fingerprint ^= hashBF16(bf16_gelu_back.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_gelu_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, f32_ok, f16_ok, bf16_ok, fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn geluExpected(value: f32) f32 {
    const cubic = value * value * value;
    const arg = @sqrt(2.0 / std.math.pi) * (value + 0.044715 * cubic);
    return 0.5 * value * (1.0 + std.math.tanh(arg));
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_9e10);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_9e16);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_b9e1);
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
