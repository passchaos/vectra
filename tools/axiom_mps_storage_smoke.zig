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
    var f16_elementwise_ok = !available;
    var f16_scalar_ok = !available;
    var f16_unary_ok = !available;
    var f16_matmul_ok = !available;
    var matmul_ok = !available;
    var matmul_add_ok = !available;
    var transpose_ok = !available;
    var broadcast_ok = !available;
    var reduction_ok = !available;
    var softmax_ok = !available;
    var log_softmax_ok = !available;
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
        var log_values = try device.log();
        defer log_values.deinit();
        var log_back = try log_values.cpu();
        defer log_back.deinit();
        var exp2_values = try device.exp2();
        defer exp2_values.deinit();
        var exp2_back = try exp2_values.cpu();
        defer exp2_back.deinit();
        var expm1_values = try device.expm1();
        defer expm1_values.deinit();
        var expm1_back = try expm1_values.cpu();
        defer expm1_back.deinit();
        var log1p_values = try device.log1p();
        defer log1p_values.deinit();
        var log1p_back = try log1p_values.cpu();
        defer log1p_back.deinit();
        var log2_values = try device.log2();
        defer log2_values.deinit();
        var log2_back = try log2_values.cpu();
        defer log2_back.deinit();
        var log10_values = try device.log10();
        defer log10_values.deinit();
        var log10_back = try log10_values.cpu();
        defer log10_back.deinit();
        var trig_host = try vx.Array(f32).fromSlice(allocator, &.{ 0.0, 0.5, 1.0, -0.5 }, &.{ 2, 2 });
        defer trig_host.deinit();
        var trig_device = try trig_host.mps(0);
        defer trig_device.deinit();
        var sin_values = try trig_device.sin();
        defer sin_values.deinit();
        var sin_back = try sin_values.cpu();
        defer sin_back.deinit();
        var cos_values = try trig_device.cos();
        defer cos_values.deinit();
        var cos_back = try cos_values.cpu();
        defer cos_back.deinit();
        var tan_values = try trig_device.tan();
        defer tan_values.deinit();
        var tan_back = try tan_values.cpu();
        defer tan_back.deinit();
        unary_ok = square.device.isMps() and square.device_storage != null and
            sqrt.device.isMps() and sqrt.device_storage != null and
            exp_values.device.isMps() and exp_values.device_storage != null and
            log_values.device.isMps() and log_values.device_storage != null and
            exp2_values.device.isMps() and exp2_values.device_storage != null and
            expm1_values.device.isMps() and expm1_values.device_storage != null and
            log1p_values.device.isMps() and log1p_values.device_storage != null and
            log2_values.device.isMps() and log2_values.device_storage != null and
            log10_values.device.isMps() and log10_values.device_storage != null and
            sin_values.device.isMps() and sin_values.device_storage != null and
            cos_values.device.isMps() and cos_values.device_storage != null and
            tan_values.device.isMps() and tan_values.device_storage != null and
            equalF32(square_back.data, &.{ 1, 4, 9, 16 }) and
            equalF32(sqrt_back.data, &.{ 1, 2, 3, 4 }) and
            closeF32(exp_back.data, &.{ std.math.exp(@as(f32, 1)), std.math.exp(@as(f32, 2)), std.math.exp(@as(f32, 3)), std.math.exp(@as(f32, 4)) }, 0.01) and
            closeF32(log_back.data, &.{ 0.0, std.math.log(f32, std.math.e, 2.0), std.math.log(f32, std.math.e, 3.0), std.math.log(f32, std.math.e, 4.0) }, 0.01) and
            closeF32(exp2_back.data, &.{ 2.0, 4.0, 8.0, 16.0 }, 0.01) and
            closeF32(expm1_back.data, &.{ std.math.exp(@as(f32, 1)) - 1.0, std.math.exp(@as(f32, 2)) - 1.0, std.math.exp(@as(f32, 3)) - 1.0, std.math.exp(@as(f32, 4)) - 1.0 }, 0.01) and
            closeF32(log1p_back.data, &.{ std.math.log(f32, std.math.e, 2.0), std.math.log(f32, std.math.e, 3.0), std.math.log(f32, std.math.e, 4.0), std.math.log(f32, std.math.e, 5.0) }, 0.01) and
            closeF32(log2_back.data, &.{ 0.0, 1.0, std.math.log2(@as(f32, 3.0)), 2.0 }, 0.01) and
            closeF32(log10_back.data, &.{ 0.0, std.math.log10(@as(f32, 2.0)), std.math.log10(@as(f32, 3.0)), std.math.log10(@as(f32, 4.0)) }, 0.01) and
            closeF32(sin_back.data, &.{ std.math.sin(@as(f32, 0.0)), std.math.sin(@as(f32, 0.5)), std.math.sin(@as(f32, 1.0)), std.math.sin(@as(f32, -0.5)) }, 0.01) and
            closeF32(cos_back.data, &.{ std.math.cos(@as(f32, 0.0)), std.math.cos(@as(f32, 0.5)), std.math.cos(@as(f32, 1.0)), std.math.cos(@as(f32, -0.5)) }, 0.01) and
            closeF32(tan_back.data, &.{ std.math.tan(@as(f32, 0.0)), std.math.tan(@as(f32, 0.5)), std.math.tan(@as(f32, 1.0)), std.math.tan(@as(f32, -0.5)) }, 0.03);

        var f16_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_lhs.deinit();
        var f16_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20), @as(f16, 30), @as(f16, 40) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_rhs.deinit();
        var f16_add = try f16_lhs.add(f16_rhs);
        defer f16_add.deinit();
        var f16_add_back = try f16_add.cpu();
        defer f16_add_back.deinit();
        var f16_div = try f16_rhs.div(f16_lhs);
        defer f16_div.deinit();
        var f16_div_back = try f16_div.cpu();
        defer f16_div_back.deinit();
        f16_elementwise_ok = f16_add.device.isMps() and f16_add.device_storage != null and
            f16_div.device.isMps() and f16_div.device_storage != null and
            closeF16(f16_add_back.data, &.{ 11, 22, 33, 44 }, 0.02) and
            closeF16(f16_div_back.data, &.{ 10, 10, 10, 10 }, 0.02);

        var f16_scaled = try f16_lhs.mulScalar(@as(f16, 2.0));
        defer f16_scaled.deinit();
        var f16_scaled_back = try f16_scaled.cpu();
        defer f16_scaled_back.deinit();
        var f16_rsub = try f16_lhs.subScalar(@as(f16, 10.0));
        defer f16_rsub.deinit();
        var f16_rsub_back = try f16_rsub.cpu();
        defer f16_rsub_back.deinit();
        f16_scalar_ok = f16_scaled.device.isMps() and f16_scaled.device_storage != null and
            f16_rsub.device.isMps() and f16_rsub.device_storage != null and
            closeF16(f16_scaled_back.data, &.{ 2, 4, 6, 8 }, 0.02) and
            closeF16(f16_rsub_back.data, &.{ -9, -8, -7, -6 }, 0.02);

        var f16_abs_source = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, -1), @as(f16, -2), @as(f16, 3), @as(f16, 4) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_abs_source.deinit();
        var f16_abs = try f16_abs_source.abs();
        defer f16_abs.deinit();
        var f16_abs_back = try f16_abs.cpu();
        defer f16_abs_back.deinit();
        var f16_sqrt = try f16_lhs.sqrt();
        defer f16_sqrt.deinit();
        var f16_sqrt_back = try f16_sqrt.cpu();
        defer f16_sqrt_back.deinit();
        var f16_exp = try f16_lhs.exp();
        defer f16_exp.deinit();
        var f16_exp_back = try f16_exp.cpu();
        defer f16_exp_back.deinit();
        f16_unary_ok = f16_abs.device.isMps() and f16_abs.device_storage != null and
            f16_sqrt.device.isMps() and f16_sqrt.device_storage != null and
            f16_exp.device.isMps() and f16_exp.device_storage != null and
            closeF16(f16_abs_back.data, &.{ 1, 2, 3, 4 }, 0.02) and
            closeF16(f16_sqrt_back.data, &.{ 1, std.math.sqrt(@as(f32, 2)), std.math.sqrt(@as(f32, 3)), 2 }, 0.03) and
            closeF16(f16_exp_back.data, &.{ std.math.exp(@as(f32, 1)), std.math.exp(@as(f32, 2)), std.math.exp(@as(f32, 3)), std.math.exp(@as(f32, 4)) }, 0.5);

        var f16_mat_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4), @as(f16, 5), @as(f16, 6) }, &.{ 2, 3 }, vx.mps(0));
        defer f16_mat_lhs.deinit();
        var f16_mat_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 7), @as(f16, 8), @as(f16, 9), @as(f16, 10), @as(f16, 11), @as(f16, 12) }, &.{ 3, 2 }, vx.mps(0));
        defer f16_mat_rhs.deinit();
        var f16_mat_out = try f16_mat_lhs.matmul(f16_mat_rhs);
        defer f16_mat_out.deinit();
        var f16_mat_back = try f16_mat_out.cpu();
        defer f16_mat_back.deinit();
        f16_matmul_ok = f16_mat_out.device.isMps() and f16_mat_out.device_storage != null and
            closeF16(f16_mat_back.data, &.{ 58, 64, 139, 154 }, 0.5);

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

        var mat_addend = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, -1, 2, -2 }, &.{ 2, 2 }, vx.mps(0));
        defer mat_addend.deinit();
        var mat_add_out = try vx.matmulAdd(mat_lhs, mat_rhs, mat_addend);
        defer mat_add_out.deinit();
        var mat_add_back = try mat_add_out.cpu();
        defer mat_add_back.deinit();
        var mat_scaled_add_out = (try vx.axiom_backend.executeMatmulAddScaled(f32, .mps, mat_lhs, mat_rhs, mat_addend, 2.0, -1.0)) orelse return error.BackendFailure;
        defer mat_scaled_add_out.deinit();
        var mat_scaled_add_back = try mat_scaled_add_out.cpu();
        defer mat_scaled_add_back.deinit();
        matmul_add_ok = mat_add_out.device.isMps() and mat_add_out.device_storage != null and
            mat_scaled_add_out.device.isMps() and mat_scaled_add_out.device_storage != null and
            equalF32(mat_add_back.data, &.{ 59, 63, 141, 152 }) and
            equalF32(mat_scaled_add_back.data, &.{ 115, 129, 276, 310 });

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

        var row_sum = try mat_lhs.sum(1, false);
        defer row_sum.deinit();
        var row_sum_back = try row_sum.cpu();
        defer row_sum_back.deinit();
        var col_max = try mat_lhs.max(0, false);
        defer col_max.deinit();
        var col_max_back = try col_max.cpu();
        defer col_max_back.deinit();
        var row_prod_keep = try mat_lhs.prod(1, true);
        defer row_prod_keep.deinit();
        var row_prod_keep_back = try row_prod_keep.cpu();
        defer row_prod_keep_back.deinit();
        reduction_ok = row_sum.device.isMps() and row_sum.device_storage != null and
            col_max.device.isMps() and col_max.device_storage != null and
            row_prod_keep.device.isMps() and row_prod_keep.device_storage != null and
            equalF32(row_sum_back.data, &.{ 6, 15 }) and
            equalF32(col_max_back.data, &.{ 4, 5, 6 }) and
            equalF32(row_prod_keep_back.data, &.{ 6, 120 });

        var softmax_row = try mat_lhs.softmax(1);
        defer softmax_row.deinit();
        var softmax_row_back = try softmax_row.cpu();
        defer softmax_row_back.deinit();
        var softmax_col = try mat_lhs.softmax(0);
        defer softmax_col.deinit();
        var softmax_col_back = try softmax_col.cpu();
        defer softmax_col_back.deinit();
        const row_denom = std.math.exp(@as(f32, -2)) + std.math.exp(@as(f32, -1)) + 1.0;
        const col_denom = std.math.exp(@as(f32, -3)) + 1.0;
        softmax_ok = softmax_row.device.isMps() and softmax_row.device_storage != null and
            softmax_col.device.isMps() and softmax_col.device_storage != null and
            closeF32(softmax_row_back.data, &.{ std.math.exp(@as(f32, -2)) / row_denom, std.math.exp(@as(f32, -1)) / row_denom, 1.0 / row_denom, std.math.exp(@as(f32, -2)) / row_denom, std.math.exp(@as(f32, -1)) / row_denom, 1.0 / row_denom }, 0.01) and
            closeF32(softmax_col_back.data, &.{ std.math.exp(@as(f32, -3)) / col_denom, std.math.exp(@as(f32, -3)) / col_denom, std.math.exp(@as(f32, -3)) / col_denom, 1.0 / col_denom, 1.0 / col_denom, 1.0 / col_denom }, 0.01);

        var log_softmax_row = try mat_lhs.logSoftmax(1);
        defer log_softmax_row.deinit();
        var log_softmax_row_back = try log_softmax_row.cpu();
        defer log_softmax_row_back.deinit();
        var log_softmax_col = try mat_lhs.logSoftmax(0);
        defer log_softmax_col.deinit();
        var log_softmax_col_back = try log_softmax_col.cpu();
        defer log_softmax_col_back.deinit();
        const row_log_denom = std.math.log(f32, std.math.e, row_denom);
        const col_log_denom = std.math.log(f32, std.math.e, col_denom);
        log_softmax_ok = log_softmax_row.device.isMps() and log_softmax_row.device_storage != null and
            log_softmax_col.device.isMps() and log_softmax_col.device_storage != null and
            closeF32(log_softmax_row_back.data, &.{ -2.0 - row_log_denom, -1.0 - row_log_denom, -row_log_denom, -2.0 - row_log_denom, -1.0 - row_log_denom, -row_log_denom }, 0.03) and
            closeF32(log_softmax_col_back.data, &.{ -3.0 - col_log_denom, -3.0 - col_log_denom, -3.0 - col_log_denom, -col_log_denom, -col_log_denom, -col_log_denom }, 0.03);

        fingerprint ^= hashF32(back.data) ^ hashF32(clone_back.data) ^ hashF32(filled_back.data) ^ hashF32(add_back.data) ^ hashF32(div_back.data) ^ hashF32(scaled_back.data) ^ hashF32(rsub_back.data) ^ hashF32(square_back.data) ^ hashF32(sqrt_back.data) ^ hashF32(exp_back.data) ^ hashF32(log_back.data) ^ hashF32(exp2_back.data) ^ hashF32(expm1_back.data) ^ hashF32(log1p_back.data) ^ hashF32(log2_back.data) ^ hashF32(log10_back.data) ^ hashF32(sin_back.data) ^ hashF32(cos_back.data) ^ hashF32(tan_back.data) ^ hashF16(f16_add_back.data) ^ hashF16(f16_div_back.data) ^ hashF16(f16_scaled_back.data) ^ hashF16(f16_rsub_back.data) ^ hashF16(f16_abs_back.data) ^ hashF16(f16_sqrt_back.data) ^ hashF16(f16_exp_back.data) ^ hashF16(f16_mat_back.data) ^ hashF32(mat_back.data) ^ hashF32(mat_add_back.data) ^ hashF32(mat_scaled_add_back.data) ^ hashF32(transposed_back.data) ^ hashF32(row_added_back.data) ^ hashF32(col_added_back.data) ^ hashF32(row_sum_back.data) ^ hashF32(col_max_back.data) ^ hashF32(row_prod_keep_back.data) ^ hashF32(softmax_row_back.data) ^ hashF32(softmax_col_back.data) ^ hashF32(log_softmax_row_back.data) ^ hashF32(log_softmax_col_back.data);
    }

    const ok = if (available)
        report.ok() and roundtrip_ok and copy_ok and fill_ok and elementwise_ok and scalar_ok and unary_ok and f16_elementwise_ok and f16_scalar_ok and f16_unary_ok and f16_matmul_ok and matmul_ok and matmul_add_ok and transpose_ok and broadcast_ok and reduction_ok and softmax_ok and log_softmax_ok and bytes != 0
    else
        !report.ok() and roundtrip_ok and copy_ok and fill_ok and elementwise_ok and scalar_ok and unary_ok and f16_elementwise_ok and f16_scalar_ok and f16_unary_ok and f16_matmul_ok and matmul_ok and matmul_add_ok and transpose_ok and broadcast_ok and reduction_ok and softmax_ok and log_softmax_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_storage_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\",\"roundtrip_ok\":{},\"copy_ok\":{},\"fill_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"unary_ok\":{},\"f16_elementwise_ok\":{},\"f16_scalar_ok\":{},\"f16_unary_ok\":{},\"f16_matmul_ok\":{},\"matmul_ok\":{},\"matmul_add_ok\":{},\"transpose_ok\":{},\"broadcast_ok\":{},\"reduction_ok\":{},\"softmax_ok\":{},\"log_softmax_ok\":{},\"bytes\":{d},\"fingerprint\":{d}}}\n",
        .{ ok, available, report.status.label(), report.backend_label, roundtrip_ok, copy_ok, fill_ok, elementwise_ok, scalar_ok, unary_ok, f16_elementwise_ok, f16_scalar_ok, f16_unary_ok, f16_matmul_ok, matmul_ok, matmul_add_ok, transpose_ok, broadcast_ok, reduction_ok, softmax_ok, log_softmax_ok, bytes, fingerprint },
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

fn closeF16(actual: []const f16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(@as(f32, a) - e) > tolerance) return false;
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

fn hashF16(values: []const f16) u64 {
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_0016);
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
