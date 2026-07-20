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
        const lhs_values = [_]f32{
            1, 2,
            3, 4,
        };
        const rhs_values = [_]f32{
            1, 0,
            0, 1,

            2, 1,
            1, 2,
        };
        const lhs_batch_values = [_]f32{
            1, 2,
            3, 4,

            5, 6,
            7, 8,
        };
        const rhs_single_values = [_]f32{
            1, 0,
            0, 1,
        };
        const expected_lhs_broadcast = [_]f32{ 1, 2, 3, 4, 4, 5, 10, 11 };
        const expected_rhs_broadcast = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };

        var lhs = try vx.Array(f32).fromSliceOn(allocator, &lhs_values, &.{ 1, 2, 2 }, vx.mps(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).fromSliceOn(allocator, &rhs_values, &.{ 2, 2, 2 }, vx.mps(0));
        defer rhs.deinit();
        var lhs_broadcast = try lhs.matmul(rhs);
        defer lhs_broadcast.deinit();
        var lhs_broadcast_back = try lhs_broadcast.cpu();
        defer lhs_broadcast_back.deinit();
        var lhs_batch = try vx.Array(f32).fromSliceOn(allocator, &lhs_batch_values, &.{ 2, 2, 2 }, vx.mps(0));
        defer lhs_batch.deinit();
        var rhs_single = try vx.Array(f32).fromSliceOn(allocator, &rhs_single_values, &.{ 1, 2, 2 }, vx.mps(0));
        defer rhs_single.deinit();
        var rhs_broadcast = try lhs_batch.matmul(rhs_single);
        defer rhs_broadcast.deinit();
        var rhs_broadcast_back = try rhs_broadcast.cpu();
        defer rhs_broadcast_back.deinit();
        f32_ok = lhs_broadcast.device.isMps() and lhs_broadcast.device_storage != null and
            rhs_broadcast.device.isMps() and rhs_broadcast.device_storage != null and
            std.mem.eql(usize, lhs_broadcast_back.shape, &.{ 2, 2, 2 }) and
            std.mem.eql(usize, rhs_broadcast_back.shape, &.{ 2, 2, 2 }) and
            closeF32(lhs_broadcast_back.data, &expected_lhs_broadcast, 0.001) and
            closeF32(rhs_broadcast_back.data, &expected_rhs_broadcast, 0.001);
        fingerprint ^= hashF32(lhs_broadcast_back.data) ^ hashF32(rhs_broadcast_back.data);

        var lhs16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(lhs_values), &.{ 1, 2, 2 }, vx.mps(0));
        defer lhs16.deinit();
        var rhs16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(rhs_values), &.{ 2, 2, 2 }, vx.mps(0));
        defer rhs16.deinit();
        var lhs_broadcast16 = try lhs16.matmul(rhs16);
        defer lhs_broadcast16.deinit();
        var lhs_broadcast16_back = try lhs_broadcast16.cpu();
        defer lhs_broadcast16_back.deinit();
        var lhs_batch16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(lhs_batch_values), &.{ 2, 2, 2 }, vx.mps(0));
        defer lhs_batch16.deinit();
        var rhs_single16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(rhs_single_values), &.{ 1, 2, 2 }, vx.mps(0));
        defer rhs_single16.deinit();
        var rhs_broadcast16 = try lhs_batch16.matmul(rhs_single16);
        defer rhs_broadcast16.deinit();
        var rhs_broadcast16_back = try rhs_broadcast16.cpu();
        defer rhs_broadcast16_back.deinit();
        f16_ok = lhs_broadcast16.device.isMps() and lhs_broadcast16.device_storage != null and
            rhs_broadcast16.device.isMps() and rhs_broadcast16.device_storage != null and
            std.mem.eql(usize, lhs_broadcast16_back.shape, &.{ 2, 2, 2 }) and
            std.mem.eql(usize, rhs_broadcast16_back.shape, &.{ 2, 2, 2 }) and
            closeF16(lhs_broadcast16_back.data, &expected_lhs_broadcast, 0.25) and
            closeF16(rhs_broadcast16_back.data, &expected_rhs_broadcast, 0.25);
        fingerprint ^= hashF16(lhs_broadcast16_back.data) ^ hashF16(rhs_broadcast16_back.data);

        var lhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(lhs_values), &.{ 1, 2, 2 }, vx.mps(0));
        defer lhs_bf16.deinit();
        var rhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(rhs_values), &.{ 2, 2, 2 }, vx.mps(0));
        defer rhs_bf16.deinit();
        var lhs_broadcast_bf16 = try lhs_bf16.matmul(rhs_bf16);
        defer lhs_broadcast_bf16.deinit();
        var lhs_broadcast_bf16_back = try lhs_broadcast_bf16.cpu();
        defer lhs_broadcast_bf16_back.deinit();
        var lhs_batch_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(lhs_batch_values), &.{ 2, 2, 2 }, vx.mps(0));
        defer lhs_batch_bf16.deinit();
        var rhs_single_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(rhs_single_values), &.{ 1, 2, 2 }, vx.mps(0));
        defer rhs_single_bf16.deinit();
        var rhs_broadcast_bf16 = try lhs_batch_bf16.matmul(rhs_single_bf16);
        defer rhs_broadcast_bf16.deinit();
        var rhs_broadcast_bf16_back = try rhs_broadcast_bf16.cpu();
        defer rhs_broadcast_bf16_back.deinit();
        bf16_ok = lhs_broadcast_bf16.device.isMps() and lhs_broadcast_bf16.device_storage != null and
            rhs_broadcast_bf16.device.isMps() and rhs_broadcast_bf16.device_storage != null and
            std.mem.eql(usize, lhs_broadcast_bf16_back.shape, &.{ 2, 2, 2 }) and
            std.mem.eql(usize, rhs_broadcast_bf16_back.shape, &.{ 2, 2, 2 }) and
            closeBF16(lhs_broadcast_bf16_back.data, &expected_lhs_broadcast, 0.5) and
            closeBF16(rhs_broadcast_bf16_back.data, &expected_rhs_broadcast, 0.5);
        fingerprint ^= hashBF16(lhs_broadcast_bf16_back.data) ^ hashBF16(rhs_broadcast_bf16_back.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_broadcast_bmm_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, f32_ok, f16_ok, bf16_ok, fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn toF16(comptime values: anytype) [values.len]f16 {
    var out: [values.len]f16 = undefined;
    for (values, 0..) |value, i| out[i] = @floatCast(value);
    return out;
}

fn toBF16(comptime values: anytype) [values.len]vx.BFloat16 {
    var out: [values.len]vx.BFloat16 = undefined;
    for (values, 0..) |value, i| out[i] = vx.BFloat16.fromF32(value);
    return out;
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_b032);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_b016);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_b0b1);
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
