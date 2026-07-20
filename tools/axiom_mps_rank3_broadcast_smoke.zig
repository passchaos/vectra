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
        var input = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 1, 2, 2 }, vx.mps(0));
        defer input.deinit();
        var bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20 }, &.{2}, vx.mps(0));
        defer bias.deinit();
        var div = try input.div(bias);
        defer div.deinit();
        var div_back = try div.cpu();
        defer div_back.deinit();
        var rsub = try bias.sub(input);
        defer rsub.deinit();
        var rsub_back = try rsub.cpu();
        defer rsub_back.deinit();
        var rdiv = try bias.div(input);
        defer rdiv.deinit();
        var rdiv_back = try rdiv.cpu();
        defer rdiv_back.deinit();
        var keepdims_bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20 }, &.{ 1, 1, 2 }, vx.mps(0));
        defer keepdims_bias.deinit();
        var keepdims_div = try input.div(keepdims_bias);
        defer keepdims_div.deinit();
        var keepdims_div_back = try keepdims_div.cpu();
        defer keepdims_div_back.deinit();
        var middle_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 1, 2 }, vx.mps(0));
        defer middle_lhs.deinit();
        var middle_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20, 30, 40 }, &.{ 1, 2, 2 }, vx.mps(0));
        defer middle_rhs.deinit();
        var middle_add = try middle_lhs.add(middle_rhs);
        defer middle_add.deinit();
        var middle_add_back = try middle_add.cpu();
        defer middle_add_back.deinit();
        var rank4_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 2,
            3, 4,

            5, 6,
            7, 8,
        }, &.{ 2, 1, 2, 2 }, vx.mps(0));
        defer rank4_lhs.deinit();
        var rank4_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            10, 20,
            30, 40,

            50, 60,
            70, 80,
        }, &.{ 1, 2, 2, 2 }, vx.mps(0));
        defer rank4_rhs.deinit();
        var rank4_add = try rank4_lhs.add(rank4_rhs);
        defer rank4_add.deinit();
        var rank4_add_back = try rank4_add.cpu();
        defer rank4_add_back.deinit();
        var rank5_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 2,
            3, 4,

            5, 6,
            7, 8,
        }, &.{ 2, 1, 1, 2, 2 }, vx.mps(0));
        defer rank5_lhs.deinit();
        var rank5_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{
            10, 20,
            30, 40,

            50, 60,
            70, 80,
        }, &.{ 1, 2, 1, 2, 2 }, vx.mps(0));
        defer rank5_rhs.deinit();
        var rank5_add = try rank5_lhs.add(rank5_rhs);
        defer rank5_add.deinit();
        var rank5_add_back = try rank5_add.cpu();
        defer rank5_add_back.deinit();
        f32_ok = div.device.isMps() and div.device_storage != null and
            rsub.device.isMps() and rsub.device_storage != null and
            keepdims_div.device.isMps() and keepdims_div.device_storage != null and
            middle_add.device.isMps() and middle_add.device_storage != null and
            rank4_add.device.isMps() and rank4_add.device_storage != null and
            rank5_add.device.isMps() and rank5_add.device_storage != null and
            std.mem.eql(usize, div_back.shape, &.{ 1, 2, 2 }) and
            closeF32(div_back.data, &.{ 0.1, 0.1, 0.3, 0.2 }, 0.001) and
            std.mem.eql(usize, rsub_back.shape, &.{ 1, 2, 2 }) and
            closeF32(rsub_back.data, &.{ 9, 18, 7, 16 }, 0.001) and
            rdiv.device.isMps() and rdiv.device_storage != null and
            std.mem.eql(usize, rdiv_back.shape, &.{ 1, 2, 2 }) and
            closeF32(rdiv_back.data, &.{ 10, 10, 3.3333333, 5 }, 0.001) and
            std.mem.eql(usize, keepdims_div_back.shape, &.{ 1, 2, 2 }) and
            closeF32(keepdims_div_back.data, &.{ 0.1, 0.1, 0.3, 0.2 }, 0.001) and
            std.mem.eql(usize, middle_add_back.shape, &.{ 2, 2, 2 }) and
            closeF32(middle_add_back.data, &.{ 11, 22, 31, 42, 13, 24, 33, 44 }, 0.001) and
            std.mem.eql(usize, rank4_add_back.shape, &.{ 2, 2, 2, 2 }) and
            closeF32(rank4_add_back.data, &.{
                11, 22, 33, 44,
                51, 62, 73, 84,
                15, 26, 37, 48,
                55, 66, 77, 88,
            }, 0.001) and
            std.mem.eql(usize, rank5_add_back.shape, &.{ 2, 2, 1, 2, 2 }) and
            closeF32(rank5_add_back.data, &.{
                11, 22, 33, 44,
                51, 62, 73, 84,
                15, 26, 37, 48,
                55, 66, 77, 88,
            }, 0.001);
        fingerprint ^= hashF32(div_back.data) ^ hashF32(rsub_back.data) ^ hashF32(rdiv_back.data) ^ hashF32(keepdims_div_back.data) ^ hashF32(middle_add_back.data) ^ hashF32(rank4_add_back.data) ^ hashF32(rank5_add_back.data);

        var f16_input = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4) }, &.{ 1, 2, 2 }, vx.mps(0));
        defer f16_input.deinit();
        var f16_bias = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20) }, &.{2}, vx.mps(0));
        defer f16_bias.deinit();
        var f16_div = try f16_input.div(f16_bias);
        defer f16_div.deinit();
        var f16_div_back = try f16_div.cpu();
        defer f16_div_back.deinit();
        var f16_rdiv = try f16_bias.div(f16_input);
        defer f16_rdiv.deinit();
        var f16_rdiv_back = try f16_rdiv.cpu();
        defer f16_rdiv_back.deinit();
        var f16_middle_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4) }, &.{ 2, 1, 2 }, vx.mps(0));
        defer f16_middle_lhs.deinit();
        var f16_middle_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20), @as(f16, 30), @as(f16, 40) }, &.{ 1, 2, 2 }, vx.mps(0));
        defer f16_middle_rhs.deinit();
        var f16_middle_add = try f16_middle_lhs.add(f16_middle_rhs);
        defer f16_middle_add.deinit();
        var f16_middle_add_back = try f16_middle_add.cpu();
        defer f16_middle_add_back.deinit();
        var f16_rank4_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{
            @as(f16, 1), @as(f16, 2),
            @as(f16, 3), @as(f16, 4),

            @as(f16, 5), @as(f16, 6),
            @as(f16, 7), @as(f16, 8),
        }, &.{ 2, 1, 2, 2 }, vx.mps(0));
        defer f16_rank4_lhs.deinit();
        var f16_rank4_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{
            @as(f16, 10), @as(f16, 20),
            @as(f16, 30), @as(f16, 40),

            @as(f16, 50), @as(f16, 60),
            @as(f16, 70), @as(f16, 80),
        }, &.{ 1, 2, 2, 2 }, vx.mps(0));
        defer f16_rank4_rhs.deinit();
        var f16_rank4_add = try f16_rank4_lhs.add(f16_rank4_rhs);
        defer f16_rank4_add.deinit();
        var f16_rank4_add_back = try f16_rank4_add.cpu();
        defer f16_rank4_add_back.deinit();
        var f16_rank5_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{
            @as(f16, 1), @as(f16, 2),
            @as(f16, 3), @as(f16, 4),

            @as(f16, 5), @as(f16, 6),
            @as(f16, 7), @as(f16, 8),
        }, &.{ 2, 1, 1, 2, 2 }, vx.mps(0));
        defer f16_rank5_lhs.deinit();
        var f16_rank5_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{
            @as(f16, 10), @as(f16, 20),
            @as(f16, 30), @as(f16, 40),

            @as(f16, 50), @as(f16, 60),
            @as(f16, 70), @as(f16, 80),
        }, &.{ 1, 2, 1, 2, 2 }, vx.mps(0));
        defer f16_rank5_rhs.deinit();
        var f16_rank5_add = try f16_rank5_lhs.add(f16_rank5_rhs);
        defer f16_rank5_add.deinit();
        var f16_rank5_add_back = try f16_rank5_add.cpu();
        defer f16_rank5_add_back.deinit();
        f16_ok = f16_div.device.isMps() and f16_div.device_storage != null and
            std.mem.eql(usize, f16_div_back.shape, &.{ 1, 2, 2 }) and
            closeF16(f16_div_back.data, &.{ 0.1, 0.1, 0.3, 0.2 }, 0.01) and
            f16_rdiv.device.isMps() and f16_rdiv.device_storage != null and
            std.mem.eql(usize, f16_rdiv_back.shape, &.{ 1, 2, 2 }) and
            closeF16(f16_rdiv_back.data, &.{ 10, 10, 3.3333333, 5 }, 0.02) and
            f16_middle_add.device.isMps() and f16_middle_add.device_storage != null and
            std.mem.eql(usize, f16_middle_add_back.shape, &.{ 2, 2, 2 }) and
            closeF16(f16_middle_add_back.data, &.{ 11, 22, 31, 42, 13, 24, 33, 44 }, 0.05) and
            f16_rank4_add.device.isMps() and f16_rank4_add.device_storage != null and
            std.mem.eql(usize, f16_rank4_add_back.shape, &.{ 2, 2, 2, 2 }) and
            closeF16(f16_rank4_add_back.data, &.{
                11, 22, 33, 44,
                51, 62, 73, 84,
                15, 26, 37, 48,
                55, 66, 77, 88,
            }, 0.05) and
            f16_rank5_add.device.isMps() and f16_rank5_add.device_storage != null and
            std.mem.eql(usize, f16_rank5_add_back.shape, &.{ 2, 2, 1, 2, 2 }) and
            closeF16(f16_rank5_add_back.data, &.{
                11, 22, 33, 44,
                51, 62, 73, 84,
                15, 26, 37, 48,
                55, 66, 77, 88,
            }, 0.05);
        fingerprint ^= hashF16(f16_div_back.data) ^ hashF16(f16_rdiv_back.data) ^ hashF16(f16_middle_add_back.data) ^ hashF16(f16_rank4_add_back.data) ^ hashF16(f16_rank5_add_back.data);

        var bf16_input = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4) }, &.{ 1, 2, 2 }, vx.mps(0));
        defer bf16_input.deinit();
        var bf16_bias = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20) }, &.{2}, vx.mps(0));
        defer bf16_bias.deinit();
        var bf16_div = try bf16_input.div(bf16_bias);
        defer bf16_div.deinit();
        var bf16_div_back = try bf16_div.cpu();
        defer bf16_div_back.deinit();
        var bf16_rdiv = try bf16_bias.div(bf16_input);
        defer bf16_rdiv.deinit();
        var bf16_rdiv_back = try bf16_rdiv.cpu();
        defer bf16_rdiv_back.deinit();
        var bf16_middle_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4) }, &.{ 2, 1, 2 }, vx.mps(0));
        defer bf16_middle_lhs.deinit();
        var bf16_middle_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20), vx.BFloat16.fromF32(30), vx.BFloat16.fromF32(40) }, &.{ 1, 2, 2 }, vx.mps(0));
        defer bf16_middle_rhs.deinit();
        var bf16_middle_add = try bf16_middle_lhs.add(bf16_middle_rhs);
        defer bf16_middle_add.deinit();
        var bf16_middle_add_back = try bf16_middle_add.cpu();
        defer bf16_middle_add_back.deinit();
        var bf16_rank4_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{
            vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2),
            vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4),

            vx.BFloat16.fromF32(5), vx.BFloat16.fromF32(6),
            vx.BFloat16.fromF32(7), vx.BFloat16.fromF32(8),
        }, &.{ 2, 1, 2, 2 }, vx.mps(0));
        defer bf16_rank4_lhs.deinit();
        var bf16_rank4_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{
            vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20),
            vx.BFloat16.fromF32(30), vx.BFloat16.fromF32(40),

            vx.BFloat16.fromF32(50), vx.BFloat16.fromF32(60),
            vx.BFloat16.fromF32(70), vx.BFloat16.fromF32(80),
        }, &.{ 1, 2, 2, 2 }, vx.mps(0));
        defer bf16_rank4_rhs.deinit();
        var bf16_rank4_add = try bf16_rank4_lhs.add(bf16_rank4_rhs);
        defer bf16_rank4_add.deinit();
        var bf16_rank4_add_back = try bf16_rank4_add.cpu();
        defer bf16_rank4_add_back.deinit();
        var bf16_rank5_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{
            vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2),
            vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4),

            vx.BFloat16.fromF32(5), vx.BFloat16.fromF32(6),
            vx.BFloat16.fromF32(7), vx.BFloat16.fromF32(8),
        }, &.{ 2, 1, 1, 2, 2 }, vx.mps(0));
        defer bf16_rank5_lhs.deinit();
        var bf16_rank5_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{
            vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20),
            vx.BFloat16.fromF32(30), vx.BFloat16.fromF32(40),

            vx.BFloat16.fromF32(50), vx.BFloat16.fromF32(60),
            vx.BFloat16.fromF32(70), vx.BFloat16.fromF32(80),
        }, &.{ 1, 2, 1, 2, 2 }, vx.mps(0));
        defer bf16_rank5_rhs.deinit();
        var bf16_rank5_add = try bf16_rank5_lhs.add(bf16_rank5_rhs);
        defer bf16_rank5_add.deinit();
        var bf16_rank5_add_back = try bf16_rank5_add.cpu();
        defer bf16_rank5_add_back.deinit();
        bf16_ok = bf16_div.device.isMps() and bf16_div.device_storage != null and
            std.mem.eql(usize, bf16_div_back.shape, &.{ 1, 2, 2 }) and
            closeBF16(bf16_div_back.data, &.{ 0.1, 0.1, 0.3, 0.2 }, 0.03) and
            bf16_rdiv.device.isMps() and bf16_rdiv.device_storage != null and
            std.mem.eql(usize, bf16_rdiv_back.shape, &.{ 1, 2, 2 }) and
            closeBF16(bf16_rdiv_back.data, &.{ 10, 10, 3.3333333, 5 }, 0.04) and
            bf16_middle_add.device.isMps() and bf16_middle_add.device_storage != null and
            std.mem.eql(usize, bf16_middle_add_back.shape, &.{ 2, 2, 2 }) and
            closeBF16(bf16_middle_add_back.data, &.{ 11, 22, 31, 42, 13, 24, 33, 44 }, 0.125) and
            bf16_rank4_add.device.isMps() and bf16_rank4_add.device_storage != null and
            std.mem.eql(usize, bf16_rank4_add_back.shape, &.{ 2, 2, 2, 2 }) and
            closeBF16(bf16_rank4_add_back.data, &.{
                11, 22, 33, 44,
                51, 62, 73, 84,
                15, 26, 37, 48,
                55, 66, 77, 88,
            }, 0.125) and
            bf16_rank5_add.device.isMps() and bf16_rank5_add.device_storage != null and
            std.mem.eql(usize, bf16_rank5_add_back.shape, &.{ 2, 2, 1, 2, 2 }) and
            closeBF16(bf16_rank5_add_back.data, &.{
                11, 22, 33, 44,
                51, 62, 73, 84,
                15, 26, 37, 48,
                55, 66, 77, 88,
            }, 0.125);
        fingerprint ^= hashBF16(bf16_div_back.data) ^ hashBF16(bf16_rdiv_back.data) ^ hashBF16(bf16_middle_add_back.data) ^ hashBF16(bf16_rank4_add_back.data) ^ hashBF16(bf16_rank5_add_back.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_rank3_broadcast_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_3b32);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_3b16);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_b3b1);
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
