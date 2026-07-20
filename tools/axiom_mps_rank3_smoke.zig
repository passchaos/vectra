const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var f32_ok = !available;
    var f16_ok = !available;
    var bf16_ok = !available;
    var f32_stats_ok = !available;
    var f16_stats_ok = !available;
    var bf16_stats_ok = !available;
    var fingerprint = report.fingerprint();

    if (available) {
        var input = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 1, 2, 2 }, vx.mps(0));
        defer input.deinit();
        var sum_last = try input.sum(2, false);
        defer sum_last.deinit();
        var sum_last_back = try sum_last.cpu();
        defer sum_last_back.deinit();
        var prod_last = try input.prod(2, false);
        defer prod_last.deinit();
        var prod_last_back = try prod_last.cpu();
        defer prod_last_back.deinit();
        var min_last = try input.min(2, true);
        defer min_last.deinit();
        var min_last_back = try min_last.cpu();
        defer min_last_back.deinit();
        var max_last = try input.max(2, true);
        defer max_last.deinit();
        var max_last_back = try max_last.cpu();
        defer max_last_back.deinit();
        f32_ok = sum_last.device.isMps() and sum_last.device_storage != null and
            prod_last.device.isMps() and prod_last.device_storage != null and
            min_last.device.isMps() and min_last.device_storage != null and
            max_last.device.isMps() and max_last.device_storage != null and
            std.mem.eql(usize, sum_last_back.shape, &.{ 1, 2 }) and
            closeF32(sum_last_back.data, &.{ 3, 7 }, 0.001) and
            std.mem.eql(usize, prod_last_back.shape, &.{ 1, 2 }) and
            closeF32(prod_last_back.data, &.{ 2, 12 }, 0.001) and
            std.mem.eql(usize, min_last_back.shape, &.{ 1, 2, 1 }) and
            closeF32(min_last_back.data, &.{ 1, 3 }, 0.001) and
            std.mem.eql(usize, max_last_back.shape, &.{ 1, 2, 1 }) and
            closeF32(max_last_back.data, &.{ 2, 4 }, 0.001);
        fingerprint ^= hashF32(sum_last_back.data) ^ hashF32(prod_last_back.data) ^ hashF32(min_last_back.data) ^ hashF32(max_last_back.data);

        var var_partial = try input.varianceAxes(&.{ 0, 2 }, false, 0.0);
        defer var_partial.deinit();
        var var_partial_back = try var_partial.cpu();
        defer var_partial_back.deinit();
        var std_partial_keep = try input.stddevAxes(&.{ 0, 2 }, true, 0.0);
        defer std_partial_keep.deinit();
        var std_partial_keep_back = try std_partial_keep.cpu();
        defer std_partial_keep_back.deinit();
        f32_stats_ok = var_partial.device.isMps() and var_partial.device_storage != null and
            std_partial_keep.device.isMps() and std_partial_keep.device_storage != null and
            std.mem.eql(usize, var_partial_back.shape, &.{2}) and
            closeF32(var_partial_back.data, &.{ 0.25, 0.25 }, 0.001) and
            std.mem.eql(usize, std_partial_keep_back.shape, &.{ 1, 2, 1 }) and
            closeF32(std_partial_keep_back.data, &.{ 0.5, 0.5 }, 0.001);
        fingerprint ^= hashF32(var_partial_back.data) ^ hashF32(std_partial_keep_back.data);

        var f16_input = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4) }, &.{ 1, 2, 2 }, vx.mps(0));
        defer f16_input.deinit();
        var f16_sum_last = try f16_input.sum(2, false);
        defer f16_sum_last.deinit();
        var f16_sum_last_back = try f16_sum_last.cpu();
        defer f16_sum_last_back.deinit();
        var f16_prod_last = try f16_input.prod(2, false);
        defer f16_prod_last.deinit();
        var f16_prod_last_back = try f16_prod_last.cpu();
        defer f16_prod_last_back.deinit();
        var f16_min_last = try f16_input.min(2, true);
        defer f16_min_last.deinit();
        var f16_min_last_back = try f16_min_last.cpu();
        defer f16_min_last_back.deinit();
        var f16_max_last = try f16_input.max(2, true);
        defer f16_max_last.deinit();
        var f16_max_last_back = try f16_max_last.cpu();
        defer f16_max_last_back.deinit();
        f16_ok = f16_sum_last.device.isMps() and f16_sum_last.device_storage != null and
            f16_prod_last.device.isMps() and f16_prod_last.device_storage != null and
            f16_min_last.device.isMps() and f16_min_last.device_storage != null and
            f16_max_last.device.isMps() and f16_max_last.device_storage != null and
            std.mem.eql(usize, f16_sum_last_back.shape, &.{ 1, 2 }) and
            closeF16(f16_sum_last_back.data, &.{ 3, 7 }, 0.05) and
            std.mem.eql(usize, f16_prod_last_back.shape, &.{ 1, 2 }) and
            closeF16(f16_prod_last_back.data, &.{ 2, 12 }, 0.05) and
            std.mem.eql(usize, f16_min_last_back.shape, &.{ 1, 2, 1 }) and
            closeF16(f16_min_last_back.data, &.{ 1, 3 }, 0.05) and
            std.mem.eql(usize, f16_max_last_back.shape, &.{ 1, 2, 1 }) and
            closeF16(f16_max_last_back.data, &.{ 2, 4 }, 0.05);
        fingerprint ^= hashF16(f16_sum_last_back.data) ^ hashF16(f16_prod_last_back.data) ^ hashF16(f16_min_last_back.data) ^ hashF16(f16_max_last_back.data);

        var f16_var_partial = try f16_input.varianceAxes(&.{ 0, 2 }, false, @as(f16, 0.0));
        defer f16_var_partial.deinit();
        var f16_var_partial_back = try f16_var_partial.cpu();
        defer f16_var_partial_back.deinit();
        var f16_std_partial_keep = try f16_input.stddevAxes(&.{ 0, 2 }, true, @as(f16, 0.0));
        defer f16_std_partial_keep.deinit();
        var f16_std_partial_keep_back = try f16_std_partial_keep.cpu();
        defer f16_std_partial_keep_back.deinit();
        f16_stats_ok = f16_var_partial.device.isMps() and f16_var_partial.device_storage != null and
            f16_std_partial_keep.device.isMps() and f16_std_partial_keep.device_storage != null and
            std.mem.eql(usize, f16_var_partial_back.shape, &.{2}) and
            closeF16(f16_var_partial_back.data, &.{ 0.25, 0.25 }, 0.05) and
            std.mem.eql(usize, f16_std_partial_keep_back.shape, &.{ 1, 2, 1 }) and
            closeF16(f16_std_partial_keep_back.data, &.{ 0.5, 0.5 }, 0.05);
        fingerprint ^= hashF16(f16_var_partial_back.data) ^ hashF16(f16_std_partial_keep_back.data);

        var bf16_input = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4) }, &.{ 1, 2, 2 }, vx.mps(0));
        defer bf16_input.deinit();
        var bf16_sum_last = try bf16_input.sum(2, false);
        defer bf16_sum_last.deinit();
        var bf16_sum_last_back = try bf16_sum_last.cpu();
        defer bf16_sum_last_back.deinit();
        var bf16_prod_last = try bf16_input.prod(2, false);
        defer bf16_prod_last.deinit();
        var bf16_prod_last_back = try bf16_prod_last.cpu();
        defer bf16_prod_last_back.deinit();
        var bf16_min_last = try bf16_input.min(2, true);
        defer bf16_min_last.deinit();
        var bf16_min_last_back = try bf16_min_last.cpu();
        defer bf16_min_last_back.deinit();
        var bf16_max_last = try bf16_input.max(2, true);
        defer bf16_max_last.deinit();
        var bf16_max_last_back = try bf16_max_last.cpu();
        defer bf16_max_last_back.deinit();
        bf16_ok = bf16_sum_last.device.isMps() and bf16_sum_last.device_storage != null and
            bf16_prod_last.device.isMps() and bf16_prod_last.device_storage != null and
            bf16_min_last.device.isMps() and bf16_min_last.device_storage != null and
            bf16_max_last.device.isMps() and bf16_max_last.device_storage != null and
            std.mem.eql(usize, bf16_sum_last_back.shape, &.{ 1, 2 }) and
            closeBF16(bf16_sum_last_back.data, &.{ 3, 7 }, 0.125) and
            std.mem.eql(usize, bf16_prod_last_back.shape, &.{ 1, 2 }) and
            closeBF16(bf16_prod_last_back.data, &.{ 2, 12 }, 0.125) and
            std.mem.eql(usize, bf16_min_last_back.shape, &.{ 1, 2, 1 }) and
            closeBF16(bf16_min_last_back.data, &.{ 1, 3 }, 0.125) and
            std.mem.eql(usize, bf16_max_last_back.shape, &.{ 1, 2, 1 }) and
            closeBF16(bf16_max_last_back.data, &.{ 2, 4 }, 0.125);
        fingerprint ^= hashBF16(bf16_sum_last_back.data) ^ hashBF16(bf16_prod_last_back.data) ^ hashBF16(bf16_min_last_back.data) ^ hashBF16(bf16_max_last_back.data);

        var bf16_var_partial = try bf16_input.varianceAxes(&.{ 0, 2 }, false, vx.BFloat16.fromF32(0.0));
        defer bf16_var_partial.deinit();
        var bf16_var_partial_back = try bf16_var_partial.cpu();
        defer bf16_var_partial_back.deinit();
        var bf16_std_partial_keep = try bf16_input.stddevAxes(&.{ 0, 2 }, true, vx.BFloat16.fromF32(0.0));
        defer bf16_std_partial_keep.deinit();
        var bf16_std_partial_keep_back = try bf16_std_partial_keep.cpu();
        defer bf16_std_partial_keep_back.deinit();
        bf16_stats_ok = bf16_var_partial.device.isMps() and bf16_var_partial.device_storage != null and
            bf16_std_partial_keep.device.isMps() and bf16_std_partial_keep.device_storage != null and
            std.mem.eql(usize, bf16_var_partial_back.shape, &.{2}) and
            closeBF16(bf16_var_partial_back.data, &.{ 0.25, 0.25 }, 0.125) and
            std.mem.eql(usize, bf16_std_partial_keep_back.shape, &.{ 1, 2, 1 }) and
            closeBF16(bf16_std_partial_keep_back.data, &.{ 0.5, 0.5 }, 0.125);
        fingerprint ^= hashBF16(bf16_var_partial_back.data) ^ hashBF16(bf16_std_partial_keep_back.data);
    }

    const ok = if (available)
        report.ok() and f32_ok and f16_ok and bf16_ok and f32_stats_ok and f16_stats_ok and bf16_stats_ok
    else
        !report.ok() and f32_ok and f16_ok and bf16_ok and f32_stats_ok and f16_stats_ok and bf16_stats_ok;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_rank3_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"f32_ok\":{},\"f16_ok\":{},\"bf16_ok\":{},\"f32_stats_ok\":{},\"f16_stats_ok\":{},\"bf16_stats_ok\":{},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, f32_ok, f16_ok, bf16_ok, f32_stats_ok, f16_stats_ok, bf16_stats_ok, fingerprint },
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_3d32);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_3d16);
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
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_b3d1);
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
