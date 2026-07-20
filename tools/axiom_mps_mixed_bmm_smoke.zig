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

            5, 6,
            7, 8,
        };
        const rhs_values = [_]f32{
            1, 0,
            0, 1,

            2, 1,
            1, 2,

            1, 1,
            0, 1,
        };
        const expected = [_]f32{
            1,  2,  3,  4,
            4,  5,  10, 11,
            1,  3,  3,  7,
            5,  6,  7,  8,
            16, 17, 22, 23,
            5,  11, 7,  15,
        };

        var lhs = try vx.Array(f32).fromSliceOn(allocator, &lhs_values, &.{ 2, 1, 2, 2 }, vx.mps(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).fromSliceOn(allocator, &rhs_values, &.{ 1, 3, 2, 2 }, vx.mps(0));
        defer rhs.deinit();
        var out = try lhs.matmul(rhs);
        defer out.deinit();
        var back = try out.cpu();
        defer back.deinit();
        var lhs_rank5 = try vx.Array(f32).fromSliceOn(allocator, &lhs_values, &.{ 2, 1, 1, 2, 2 }, vx.mps(0));
        defer lhs_rank5.deinit();
        var rhs_rank5 = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 0,
            0, 1,

            2, 1,
            1, 2,
        }, &.{ 1, 2, 1, 2, 2 }, vx.mps(0));
        defer rhs_rank5.deinit();
        var out_rank5 = try lhs_rank5.matmul(rhs_rank5);
        defer out_rank5.deinit();
        var back_rank5 = try out_rank5.cpu();
        defer back_rank5.deinit();
        var lhs_rank6 = try vx.Array(f32).fromSliceOn(allocator, &lhs_values, &.{ 2, 1, 1, 1, 2, 2 }, vx.mps(0));
        defer lhs_rank6.deinit();
        var rhs_rank6 = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 0,
            0, 1,

            2, 1,
            1, 2,
        }, &.{ 1, 2, 1, 1, 2, 2 }, vx.mps(0));
        defer rhs_rank6.deinit();
        var out_rank6 = try lhs_rank6.matmul(rhs_rank6);
        defer out_rank6.deinit();
        var back_rank6 = try out_rank6.cpu();
        defer back_rank6.deinit();
        const expected_rank5 = [_]f32{
            1,  2,  3,  4,
            4,  5,  10, 11,
            5,  6,  7,  8,
            16, 17, 22, 23,
        };
        const rhs_rank5_values = [_]f32{
            1, 0,
            0, 1,

            2, 1,
            1, 2,
        };
        f32_ok = out.device.isMps() and out.device_storage != null and
            std.mem.eql(usize, back.shape, &.{ 2, 3, 2, 2 }) and
            closeF32(back.data, &expected, 0.001) and
            out_rank5.device.isMps() and out_rank5.device_storage != null and
            std.mem.eql(usize, back_rank5.shape, &.{ 2, 2, 1, 2, 2 }) and
            closeF32(back_rank5.data, &expected_rank5, 0.001) and
            out_rank6.device.isMps() and out_rank6.device_storage != null and
            std.mem.eql(usize, back_rank6.shape, &.{ 2, 2, 1, 1, 2, 2 }) and
            closeF32(back_rank6.data, &expected_rank5, 0.001);
        fingerprint ^= hashF32(back.data) ^ hashF32(back_rank5.data) ^ hashF32(back_rank6.data);

        var lhs16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(lhs_values), &.{ 2, 1, 2, 2 }, vx.mps(0));
        defer lhs16.deinit();
        var rhs16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(rhs_values), &.{ 1, 3, 2, 2 }, vx.mps(0));
        defer rhs16.deinit();
        var out16 = try lhs16.matmul(rhs16);
        defer out16.deinit();
        var back16 = try out16.cpu();
        defer back16.deinit();
        var lhs_rank5_16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(lhs_values), &.{ 2, 1, 1, 2, 2 }, vx.mps(0));
        defer lhs_rank5_16.deinit();
        var rhs_rank5_16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(rhs_rank5_values), &.{ 1, 2, 1, 2, 2 }, vx.mps(0));
        defer rhs_rank5_16.deinit();
        var out_rank5_16 = try lhs_rank5_16.matmul(rhs_rank5_16);
        defer out_rank5_16.deinit();
        var back_rank5_16 = try out_rank5_16.cpu();
        defer back_rank5_16.deinit();
        var lhs_rank6_16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(lhs_values), &.{ 2, 1, 1, 1, 2, 2 }, vx.mps(0));
        defer lhs_rank6_16.deinit();
        var rhs_rank6_16 = try vx.Array(f16).fromSliceOn(allocator, &toF16(rhs_rank5_values), &.{ 1, 2, 1, 1, 2, 2 }, vx.mps(0));
        defer rhs_rank6_16.deinit();
        var out_rank6_16 = try lhs_rank6_16.matmul(rhs_rank6_16);
        defer out_rank6_16.deinit();
        var back_rank6_16 = try out_rank6_16.cpu();
        defer back_rank6_16.deinit();
        f16_ok = out16.device.isMps() and out16.device_storage != null and
            std.mem.eql(usize, back16.shape, &.{ 2, 3, 2, 2 }) and
            closeF16(back16.data, &expected, 0.25) and
            out_rank5_16.device.isMps() and out_rank5_16.device_storage != null and
            std.mem.eql(usize, back_rank5_16.shape, &.{ 2, 2, 1, 2, 2 }) and
            closeF16(back_rank5_16.data, &expected_rank5, 0.25) and
            out_rank6_16.device.isMps() and out_rank6_16.device_storage != null and
            std.mem.eql(usize, back_rank6_16.shape, &.{ 2, 2, 1, 1, 2, 2 }) and
            closeF16(back_rank6_16.data, &expected_rank5, 0.25);
        fingerprint ^= hashF16(back16.data) ^ hashF16(back_rank5_16.data) ^ hashF16(back_rank6_16.data);

        var lhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(lhs_values), &.{ 2, 1, 2, 2 }, vx.mps(0));
        defer lhs_bf16.deinit();
        var rhs_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(rhs_values), &.{ 1, 3, 2, 2 }, vx.mps(0));
        defer rhs_bf16.deinit();
        var out_bf16 = try lhs_bf16.matmul(rhs_bf16);
        defer out_bf16.deinit();
        var back_bf16 = try out_bf16.cpu();
        defer back_bf16.deinit();
        var lhs_rank5_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(lhs_values), &.{ 2, 1, 1, 2, 2 }, vx.mps(0));
        defer lhs_rank5_bf16.deinit();
        var rhs_rank5_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(rhs_rank5_values), &.{ 1, 2, 1, 2, 2 }, vx.mps(0));
        defer rhs_rank5_bf16.deinit();
        var out_rank5_bf16 = try lhs_rank5_bf16.matmul(rhs_rank5_bf16);
        defer out_rank5_bf16.deinit();
        var back_rank5_bf16 = try out_rank5_bf16.cpu();
        defer back_rank5_bf16.deinit();
        var lhs_rank6_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(lhs_values), &.{ 2, 1, 1, 1, 2, 2 }, vx.mps(0));
        defer lhs_rank6_bf16.deinit();
        var rhs_rank6_bf16 = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &toBF16(rhs_rank5_values), &.{ 1, 2, 1, 1, 2, 2 }, vx.mps(0));
        defer rhs_rank6_bf16.deinit();
        var out_rank6_bf16 = try lhs_rank6_bf16.matmul(rhs_rank6_bf16);
        defer out_rank6_bf16.deinit();
        var back_rank6_bf16 = try out_rank6_bf16.cpu();
        defer back_rank6_bf16.deinit();
        bf16_ok = out_bf16.device.isMps() and out_bf16.device_storage != null and
            std.mem.eql(usize, back_bf16.shape, &.{ 2, 3, 2, 2 }) and
            closeBF16(back_bf16.data, &expected, 0.5) and
            out_rank5_bf16.device.isMps() and out_rank5_bf16.device_storage != null and
            std.mem.eql(usize, back_rank5_bf16.shape, &.{ 2, 2, 1, 2, 2 }) and
            closeBF16(back_rank5_bf16.data, &expected_rank5, 0.5) and
            out_rank6_bf16.device.isMps() and out_rank6_bf16.device_storage != null and
            std.mem.eql(usize, back_rank6_bf16.shape, &.{ 2, 2, 1, 1, 2, 2 }) and
            closeBF16(back_rank6_bf16.data, &expected_rank5, 0.5);
        fingerprint ^= hashBF16(back_bf16.data) ^ hashBF16(back_rank5_bf16.data) ^ hashBF16(back_rank6_bf16.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_mixed_bmm_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_4f32);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_4f16);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_4fb1);
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
