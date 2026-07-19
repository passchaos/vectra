const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var roundtrip_ok = !available;
    var copy_ok = !available;
    var fill_ok = !available;
    var elementwise_ok = !available;
    var scalar_ok = !available;
    var unary_ok = !available;
    var matmul_ok = !available;
    var transpose_ok = !available;
    var broadcast_ok = !available;
    var bytes: usize = 0;
    var fingerprint = report.fingerprint();

    if (available) {
        var host = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
        defer host.deinit();
        var device = try host.mps(0);
        defer device.deinit();
        bytes = if (device.device_storage) |storage| storage.bytes else 0;

        var back = try device.cpu();
        defer back.deinit();
        roundtrip_ok = device.device.isMps() and device.device_storage != null and equalF32(back.data, host.data);

        var clone = try device.clone();
        defer clone.deinit();
        var clone_back = try clone.cpu();
        defer clone_back.deinit();
        copy_ok = clone.device.isMps() and clone.device_storage != null and equalF32(clone_back.data, host.data);

        var filled = try vx.Array(f32).fullOn(allocator, &.{4}, 7.0, vx.mps(0));
        defer filled.deinit();
        var filled_back = try filled.cpu();
        defer filled_back.deinit();
        fill_ok = filled.device.isMps() and filled.device_storage != null and equalF32(filled_back.data, &.{ 7, 7, 7, 7 });

        var rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20, 30, 40 }, &.{ 2, 2 }, vx.mps(0));
        defer rhs.deinit();
        var add = try device.add(rhs);
        defer add.deinit();
        var add_back = try add.cpu();
        defer add_back.deinit();
        var div = try rhs.div(device);
        defer div.deinit();
        var div_back = try div.cpu();
        defer div_back.deinit();
        elementwise_ok = add.device.isMps() and add.device_storage != null and
            div.device.isMps() and div.device_storage != null and
            equalF32(add_back.data, &.{ 11, 22, 33, 44 }) and
            equalF32(div_back.data, &.{ 10, 10, 10, 10 });

        var scaled = try device.mulScalar(2.0);
        defer scaled.deinit();
        var scaled_back = try scaled.cpu();
        defer scaled_back.deinit();
        var rsub = try device.subScalar(10.0);
        defer rsub.deinit();
        var rsub_back = try rsub.cpu();
        defer rsub_back.deinit();
        scalar_ok = scaled.device.isMps() and scaled.device_storage != null and
            rsub.device.isMps() and rsub.device_storage != null and
            equalF32(scaled_back.data, &.{ 2, 4, 6, 8 }) and
            equalF32(rsub_back.data, &.{ -9, -8, -7, -6 });

        var square = try device.square();
        defer square.deinit();
        var square_back = try square.cpu();
        defer square_back.deinit();
        var sqrt = try square.sqrt();
        defer sqrt.deinit();
        var sqrt_back = try sqrt.cpu();
        defer sqrt_back.deinit();
        var exp_values = try device.exp();
        defer exp_values.deinit();
        var exp_back = try exp_values.cpu();
        defer exp_back.deinit();
        unary_ok = square.device.isMps() and square.device_storage != null and
            sqrt.device.isMps() and sqrt.device_storage != null and
            exp_values.device.isMps() and exp_values.device_storage != null and
            equalF32(square_back.data, &.{ 1, 4, 9, 16 }) and
            equalF32(sqrt_back.data, &.{ 1, 2, 3, 4 }) and
            closeF32(exp_back.data, &.{ std.math.exp(@as(f32, 1)), std.math.exp(@as(f32, 2)), std.math.exp(@as(f32, 3)), std.math.exp(@as(f32, 4)) }, 0.01);

        var mat_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 }, vx.mps(0));
        defer mat_lhs.deinit();
        var mat_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 }, vx.mps(0));
        defer mat_rhs.deinit();
        var mat_out = try mat_lhs.matmul(mat_rhs);
        defer mat_out.deinit();
        var mat_back = try mat_out.cpu();
        defer mat_back.deinit();
        matmul_ok = mat_out.device.isMps() and mat_out.device_storage != null and
            equalF32(mat_back.data, &.{ 58, 64, 139, 154 });

        var transposed = try mat_lhs.transpose();
        defer transposed.deinit();
        var transposed_back = try transposed.cpu();
        defer transposed_back.deinit();
        transpose_ok = transposed.device.isMps() and transposed.device_storage != null and
            equalF32(transposed_back.data, &.{ 1, 4, 2, 5, 3, 6 });

        var row_bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20, 30 }, &.{3}, vx.mps(0));
        defer row_bias.deinit();
        var row_added = try mat_lhs.add(row_bias);
        defer row_added.deinit();
        var row_added_back = try row_added.cpu();
        defer row_added_back.deinit();
        var col_bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 100, 200 }, &.{2}, vx.mps(0));
        defer col_bias.deinit();
        var col_added = try mat_lhs.add(col_bias);
        defer col_added.deinit();
        var col_added_back = try col_added.cpu();
        defer col_added_back.deinit();
        broadcast_ok = row_added.device.isMps() and row_added.device_storage != null and
            col_added.device.isMps() and col_added.device_storage != null and
            equalF32(row_added_back.data, &.{ 11, 22, 33, 14, 25, 36 }) and
            equalF32(col_added_back.data, &.{ 101, 102, 103, 204, 205, 206 });

        fingerprint ^= hashF32(back.data) ^ hashF32(clone_back.data) ^ hashF32(filled_back.data) ^ hashF32(add_back.data) ^ hashF32(div_back.data) ^ hashF32(scaled_back.data) ^ hashF32(rsub_back.data) ^ hashF32(square_back.data) ^ hashF32(sqrt_back.data) ^ hashF32(exp_back.data) ^ hashF32(mat_back.data) ^ hashF32(transposed_back.data) ^ hashF32(row_added_back.data) ^ hashF32(col_added_back.data);
    }

    const ok = if (available)
        report.ok() and roundtrip_ok and copy_ok and fill_ok and elementwise_ok and scalar_ok and unary_ok and matmul_ok and transpose_ok and broadcast_ok and bytes != 0
    else
        !report.ok() and roundtrip_ok and copy_ok and fill_ok and elementwise_ok and scalar_ok and unary_ok and matmul_ok and transpose_ok and broadcast_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_storage_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"roundtrip_ok\":{},\"copy_ok\":{},\"fill_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"unary_ok\":{},\"matmul_ok\":{},\"transpose_ok\":{},\"broadcast_ok\":{},\"bytes\":{d},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, roundtrip_ok, copy_ok, fill_ok, elementwise_ok, scalar_ok, unary_ok, matmul_ok, transpose_ok, broadcast_ok, bytes, fingerprint },
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

fn closeF32(actual: []const f32, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a - e) > tolerance) return false;
    }
    return true;
}

fn hashF32(values: []const f32) u64 {
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_0001);
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
