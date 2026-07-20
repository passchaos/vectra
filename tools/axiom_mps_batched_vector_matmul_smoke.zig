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
        var matrix = try vx.Array(f32).fromSliceOn(allocator, &.{
            1, 2,
            3, 4,

            5, 6,
            7, 8,
        }, &.{ 2, 2, 2 }, vx.mps(0));
        defer matrix.deinit();
        var rhs_vec = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20 }, &.{2}, vx.mps(0));
        defer rhs_vec.deinit();
        var matvec = try matrix.matmul(rhs_vec);
        defer matvec.deinit();
        var matvec_back = try matvec.cpu();
        defer matvec_back.deinit();
        var vecmat = try rhs_vec.matmul(matrix);
        defer vecmat.deinit();
        var vecmat_back = try vecmat.cpu();
        defer vecmat_back.deinit();
        f32_ok = matvec.device.isMps() and matvec.device_storage != null and
            vecmat.device.isMps() and vecmat.device_storage != null and
            std.mem.eql(usize, matvec_back.shape, &.{ 2, 2 }) and
            std.mem.eql(usize, vecmat_back.shape, &.{ 2, 2 }) and
            closeF32(matvec_back.data, &.{ 50, 110, 170, 230 }, 0.001) and
            closeF32(vecmat_back.data, &.{ 70, 100, 190, 220 }, 0.001);
        fingerprint ^= hashF32(matvec_back.data) ^ hashF32(vecmat_back.data);

        var f16_matrix = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4), @as(f16, 5), @as(f16, 6), @as(f16, 7), @as(f16, 8) }, &.{ 2, 2, 2 }, vx.mps(0));
        defer f16_matrix.deinit();
        var f16_vec = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20) }, &.{2}, vx.mps(0));
        defer f16_vec.deinit();
        var f16_matvec = try f16_matrix.matmul(f16_vec);
        defer f16_matvec.deinit();
        var f16_matvec_back = try f16_matvec.cpu();
        defer f16_matvec_back.deinit();
        var f16_vecmat = try f16_vec.matmul(f16_matrix);
        defer f16_vecmat.deinit();
        var f16_vecmat_back = try f16_vecmat.cpu();
        defer f16_vecmat_back.deinit();
        f16_ok = f16_matvec.device.isMps() and f16_matvec.device_storage != null and
            f16_vecmat.device.isMps() and f16_vecmat.device_storage != null and
            std.mem.eql(usize, f16_matvec_back.shape, &.{ 2, 2 }) and
            std.mem.eql(usize, f16_vecmat_back.shape, &.{ 2, 2 }) and
            closeF16(f16_matvec_back.data, &.{ 50, 110, 170, 230 }, 0.25) and
            closeF16(f16_vecmat_back.data, &.{ 70, 100, 190, 220 }, 0.25);
        fingerprint ^= hashF16(f16_matvec_back.data) ^ hashF16(f16_vecmat_back.data);

        var bf16_matrix = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4), vx.BFloat16.fromF32(5), vx.BFloat16.fromF32(6), vx.BFloat16.fromF32(7), vx.BFloat16.fromF32(8) }, &.{ 2, 2, 2 }, vx.mps(0));
        defer bf16_matrix.deinit();
        var bf16_vec = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20) }, &.{2}, vx.mps(0));
        defer bf16_vec.deinit();
        var bf16_matvec = try bf16_matrix.matmul(bf16_vec);
        defer bf16_matvec.deinit();
        var bf16_matvec_back = try bf16_matvec.cpu();
        defer bf16_matvec_back.deinit();
        var bf16_vecmat = try bf16_vec.matmul(bf16_matrix);
        defer bf16_vecmat.deinit();
        var bf16_vecmat_back = try bf16_vecmat.cpu();
        defer bf16_vecmat_back.deinit();
        bf16_ok = bf16_matvec.device.isMps() and bf16_matvec.device_storage != null and
            bf16_vecmat.device.isMps() and bf16_vecmat.device_storage != null and
            std.mem.eql(usize, bf16_matvec_back.shape, &.{ 2, 2 }) and
            std.mem.eql(usize, bf16_vecmat_back.shape, &.{ 2, 2 }) and
            closeBF16(bf16_matvec_back.data, &.{ 50, 110, 170, 230 }, 0.5) and
            closeBF16(bf16_vecmat_back.data, &.{ 70, 100, 190, 220 }, 0.5);
        fingerprint ^= hashBF16(bf16_matvec_back.data) ^ hashBF16(bf16_vecmat_back.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_batched_vector_matmul_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"fingerprint\":{d}}}\n",
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_7032);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_7016);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_ba7c_70b1);
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
