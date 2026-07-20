const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const available = vx.mps(0).isAvailable();
    const report = vx.axiom_backend.mpsDeviceReport(0);

    var roundtrip_ok = !available;
    var copy_ok = !available;
    var shape_ok = !available;
    var fill_ok = !available;
    var random_ok = !available;
    var elementwise_ok = !available;
    var scalar_ok = !available;
    var unary_ok = !available;
    var f16_elementwise_ok = !available;
    var f16_scalar_ok = !available;
    var f16_unary_ok = !available;
    var f16_matmul_ok = !available;
    var f16_matmul_add_ok = !available;
    var f16_vector_matmul_ok = !available;
    var f16_metric_ok = !available;
    var f16_transpose_ok = !available;
    var f16_broadcast_ok = !available;
    var f16_reduction_ok = !available;
    var f16_stats_ok = !available;
    var f16_softmax_ok = !available;
    var f16_log_softmax_ok = !available;
    var f16_softmin_ok = !available;
    var f16_log_softmin_ok = !available;
    var f16_activation_ok = !available;
    var f16_activation_compose_ok = !available;
    var f16_pow_ok = !available;
    var f16_loss_ok = !available;
    var bf16_elementwise_ok = !available;
    var bf16_scalar_ok = !available;
    var bf16_unary_ok = !available;
    var bf16_matmul_ok = !available;
    var bf16_matmul_add_ok = !available;
    var bf16_matmul_chain_ok = !available;
    var bf16_vector_matmul_ok = !available;
    var bf16_metric_ok = !available;
    var bf16_transpose_ok = !available;
    var bf16_broadcast_ok = !available;
    var bf16_reduction_ok = !available;
    var bf16_stats_ok = !available;
    var bf16_softmax_ok = !available;
    var bf16_log_softmax_ok = !available;
    var bf16_softmin_ok = !available;
    var bf16_log_softmin_ok = !available;
    var bf16_activation_ok = !available;
    var bf16_activation_compose_ok = !available;
    var bf16_pow_ok = !available;
    var bf16_loss_ok = !available;
    var matmul_ok = !available;
    var matmul_add_ok = !available;
    var matmul_chain_ok = !available;
    var vector_matmul_ok = !available;
    var metric_ok = !available;
    var transpose_ok = !available;
    var broadcast_ok = !available;
    var reduction_ok = !available;
    var stats_ok = !available;
    var minmax_ok = !available;
    var activation_ok = !available;
    var activation_compose_ok = !available;
    var pow_ok = !available;
    var ternary_ok = !available;
    var loss_ok = !available;
    var softmax_ok = !available;
    var log_softmax_ok = !available;
    var logsumexp_ok = !available;
    var softmin_ok = !available;
    var log_softmin_ok = !available;
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

        var reshaped = try device.reshape(&.{4});
        defer reshaped.deinit();
        var reshaped_back = try reshaped.cpu();
        defer reshaped_back.deinit();
        var reshaped_infer = try device.reshapeInfer(&.{ -1, 1 });
        defer reshaped_infer.deinit();
        var reshaped_infer_back = try reshaped_infer.cpu();
        defer reshaped_infer_back.deinit();
        var flattened = try device.flatten();
        defer flattened.deinit();
        var flattened_back = try flattened.cpu();
        defer flattened_back.deinit();
        shape_ok = reshaped.device.isMps() and reshaped.device_storage != null and
            reshaped_infer.device.isMps() and reshaped_infer.device_storage != null and
            flattened.device.isMps() and flattened.device_storage != null and
            std.mem.eql(usize, reshaped_back.shape, &.{4}) and
            equalF32(reshaped_back.data, &.{ 1, 2, 3, 4 }) and
            std.mem.eql(usize, reshaped_infer_back.shape, &.{ 4, 1 }) and
            equalF32(reshaped_infer_back.data, &.{ 1, 2, 3, 4 }) and
            std.mem.eql(usize, flattened_back.shape, &.{4}) and
            equalF32(flattened_back.data, &.{ 1, 2, 3, 4 });

        var filled = try vx.Array(f32).fullOn(allocator, &.{4}, 7.0, vx.mps(0));
        defer filled.deinit();
        var filled_back = try filled.cpu();
        defer filled_back.deinit();
        fill_ok = filled.device.isMps() and filled.device_storage != null and equalF32(filled_back.data, &.{ 7, 7, 7, 7 });

        var np = vx.withAllocator(allocator);
        var random_a = try np.randWith(f32, &.{8}, vx.seededOn(vx.mps(0), 1234));
        defer random_a.deinit();
        var random_b = try np.randWith(f32, &.{8}, vx.seededOn(vx.mps(0), 1234));
        defer random_b.deinit();
        var random_c = try np.randWith(f32, &.{8}, vx.seededOn(vx.mps(0), 5678));
        defer random_c.deinit();
        var random_a_back = try random_a.cpu();
        defer random_a_back.deinit();
        var random_b_back = try random_b.cpu();
        defer random_b_back.deinit();
        var random_c_back = try random_c.cpu();
        defer random_c_back.deinit();
        var random_f16_a = try np.randWith(f16, &.{8}, vx.seededOn(vx.mps(0), 1234));
        defer random_f16_a.deinit();
        var random_f16_b = try np.randWith(f16, &.{8}, vx.seededOn(vx.mps(0), 1234));
        defer random_f16_b.deinit();
        var random_f16_a_back = try random_f16_a.cpu();
        defer random_f16_a_back.deinit();
        var random_f16_b_back = try random_f16_b.cpu();
        defer random_f16_b_back.deinit();
        var random_bf16_a = try np.randWith(vx.BFloat16, &.{8}, vx.seededOn(vx.mps(0), 1234));
        defer random_bf16_a.deinit();
        var random_bf16_b = try np.randWith(vx.BFloat16, &.{8}, vx.seededOn(vx.mps(0), 1234));
        defer random_bf16_b.deinit();
        var random_bf16_a_back = try random_bf16_a.cpu();
        defer random_bf16_a_back.deinit();
        var random_bf16_b_back = try random_bf16_b.cpu();
        defer random_bf16_b_back.deinit();
        random_ok = random_a.device.isMps() and random_a.device_storage != null and
            random_b.device.isMps() and random_b.device_storage != null and
            random_c.device.isMps() and random_c.device_storage != null and
            random_f16_a.device.isMps() and random_f16_a.device_storage != null and
            random_bf16_a.device.isMps() and random_bf16_a.device_storage != null and
            equalF32(random_a_back.data, random_b_back.data) and
            !equalF32(random_a_back.data, random_c_back.data) and
            inUnitRangeF32(random_a_back.data) and
            equalF16(random_f16_a_back.data, random_f16_b_back.data) and
            inUnitRangeF16(random_f16_a_back.data) and
            equalBF16(random_bf16_a_back.data, random_bf16_b_back.data) and
            inUnitRangeBF16(random_bf16_a_back.data);

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
        var scalar_array = try vx.Array(f32).fromSliceOn(allocator, &.{2.0}, &.{1}, vx.mps(0));
        defer scalar_array.deinit();
        var array_scalar_mul = try device.mul(scalar_array);
        defer array_scalar_mul.deinit();
        var array_scalar_mul_back = try array_scalar_mul.cpu();
        defer array_scalar_mul_back.deinit();
        var array_scalar_div = try device.div(scalar_array);
        defer array_scalar_div.deinit();
        var array_scalar_div_back = try array_scalar_div.cpu();
        defer array_scalar_div_back.deinit();
        var scalar_array_rsub = try scalar_array.sub(device);
        defer scalar_array_rsub.deinit();
        var scalar_array_rsub_back = try scalar_array_rsub.cpu();
        defer scalar_array_rsub_back.deinit();
        var scalar_array_rdiv = try scalar_array.div(device);
        defer scalar_array_rdiv.deinit();
        var scalar_array_rdiv_back = try scalar_array_rdiv.cpu();
        defer scalar_array_rdiv_back.deinit();
        scalar_ok = scaled.device.isMps() and scaled.device_storage != null and
            rsub.device.isMps() and rsub.device_storage != null and
            array_scalar_mul.device.isMps() and array_scalar_mul.device_storage != null and
            array_scalar_div.device.isMps() and array_scalar_div.device_storage != null and
            scalar_array_rsub.device.isMps() and scalar_array_rsub.device_storage != null and
            scalar_array_rdiv.device.isMps() and scalar_array_rdiv.device_storage != null and
            equalF32(scaled_back.data, &.{ 2, 4, 6, 8 }) and
            equalF32(rsub_back.data, &.{ -9, -8, -7, -6 }) and
            equalF32(array_scalar_mul_back.data, &.{ 2, 4, 6, 8 }) and
            closeF32(array_scalar_div_back.data, &.{ 0.5, 1.0, 1.5, 2.0 }, 0.0001) and
            equalF32(scalar_array_rsub_back.data, &.{ 1, 0, -1, -2 }) and
            closeF32(scalar_array_rdiv_back.data, &.{ 2.0, 1.0, 2.0 / 3.0, 0.5 }, 0.0001);

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
        var f16_scalar_array = try vx.Array(f16).fromSliceOn(allocator, &.{@as(f16, 2.0)}, &.{1}, vx.mps(0));
        defer f16_scalar_array.deinit();
        var f16_array_scalar_mul = try f16_lhs.mul(f16_scalar_array);
        defer f16_array_scalar_mul.deinit();
        var f16_array_scalar_mul_back = try f16_array_scalar_mul.cpu();
        defer f16_array_scalar_mul_back.deinit();
        var f16_scalar_array_rsub = try f16_scalar_array.sub(f16_lhs);
        defer f16_scalar_array_rsub.deinit();
        var f16_scalar_array_rsub_back = try f16_scalar_array_rsub.cpu();
        defer f16_scalar_array_rsub_back.deinit();
        f16_scalar_ok = f16_scaled.device.isMps() and f16_scaled.device_storage != null and
            f16_rsub.device.isMps() and f16_rsub.device_storage != null and
            f16_array_scalar_mul.device.isMps() and f16_array_scalar_mul.device_storage != null and
            f16_scalar_array_rsub.device.isMps() and f16_scalar_array_rsub.device_storage != null and
            closeF16(f16_scaled_back.data, &.{ 2, 4, 6, 8 }, 0.02) and
            closeF16(f16_rsub_back.data, &.{ -9, -8, -7, -6 }, 0.02) and
            closeF16(f16_array_scalar_mul_back.data, &.{ 2, 4, 6, 8 }, 0.02) and
            closeF16(f16_scalar_array_rsub_back.data, &.{ 1, 0, -1, -2 }, 0.02);

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

        var f16_mat_addend = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, -1), @as(f16, 2), @as(f16, -2) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_mat_addend.deinit();
        var f16_mat_add = try vx.matmulAdd(f16_mat_lhs, f16_mat_rhs, f16_mat_addend);
        defer f16_mat_add.deinit();
        var f16_mat_add_back = try f16_mat_add.cpu();
        defer f16_mat_add_back.deinit();
        var f16_mat_scaled_add = (try vx.axiom_backend.executeMatmulAddScaled(f16, .mps, f16_mat_lhs, f16_mat_rhs, f16_mat_addend, 2.0, -1.0)) orelse return error.BackendFailure;
        defer f16_mat_scaled_add.deinit();
        var f16_mat_scaled_add_back = try f16_mat_scaled_add.cpu();
        defer f16_mat_scaled_add_back.deinit();
        f16_matmul_add_ok = f16_mat_add.device.isMps() and f16_mat_add.device_storage != null and
            f16_mat_scaled_add.device.isMps() and f16_mat_scaled_add.device_storage != null and
            closeF16(f16_mat_add_back.data, &.{ 59, 63, 141, 152 }, 0.5) and
            closeF16(f16_mat_scaled_add_back.data, &.{ 115, 129, 276, 310 }, 0.75);

        var f16_vec_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3) }, &.{3}, vx.mps(0));
        defer f16_vec_rhs.deinit();
        var f16_matvec = try f16_mat_lhs.matvec(f16_vec_rhs);
        defer f16_matvec.deinit();
        var f16_matvec_back = try f16_matvec.cpu();
        defer f16_matvec_back.deinit();
        var f16_vec_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2) }, &.{2}, vx.mps(0));
        defer f16_vec_lhs.deinit();
        var f16_vecmat = try f16_vec_lhs.matmul(f16_mat_lhs);
        defer f16_vecmat.deinit();
        var f16_vecmat_back = try f16_vecmat.cpu();
        defer f16_vecmat_back.deinit();
        var f16_dot = try f16_vec_rhs.dot(f16_vec_rhs);
        defer f16_dot.deinit();
        var f16_dot_back = try f16_dot.cpu();
        defer f16_dot_back.deinit();
        f16_vector_matmul_ok = f16_matvec.device.isMps() and f16_matvec.device_storage != null and
            f16_vecmat.device.isMps() and f16_vecmat.device_storage != null and
            f16_dot.device.isMps() and f16_dot.device_storage != null and
            closeF16(f16_matvec_back.data, &.{ 14, 32 }, 0.05) and
            closeF16(f16_vecmat_back.data, &.{ 9, 12, 15 }, 0.05) and
            closeF16(f16_dot_back.data, &.{14}, 0.05);

        var f16_norm = try f16_mat_lhs.norm(@as(f16, 2), 1, false);
        defer f16_norm.deinit();
        var f16_norm_back = try f16_norm.cpu();
        defer f16_norm_back.deinit();
        var f16_normalized = try f16_mat_lhs.normalize(@as(f16, 2), 1, @as(f16, 0.001));
        defer f16_normalized.deinit();
        var f16_normalized_back = try f16_normalized.cpu();
        defer f16_normalized_back.deinit();
        var f16_metric_other = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 0), @as(f16, 0), @as(f16, 1), @as(f16, 1), @as(f16, 2) }, &.{ 2, 3 }, vx.mps(0));
        defer f16_metric_other.deinit();
        var f16_cosine = try f16_mat_lhs.cosineSimilarity(f16_metric_other, 1, @as(f16, 0.001), false);
        defer f16_cosine.deinit();
        var f16_cosine_back = try f16_cosine.cpu();
        defer f16_cosine_back.deinit();
        var f16_distance = try f16_mat_lhs.pairwiseDistance(f16_metric_other, @as(f16, 2), 1, false);
        defer f16_distance.deinit();
        var f16_distance_back = try f16_distance.cpu();
        defer f16_distance_back.deinit();
        const f16_norm0 = std.math.sqrt(@as(f32, 14));
        const f16_norm1 = std.math.sqrt(@as(f32, 77));
        const f16_other_norm0 = @as(f32, 1);
        const f16_other_norm1 = std.math.sqrt(@as(f32, 6));
        f16_metric_ok = f16_norm.device.isMps() and f16_norm.device_storage != null and
            f16_normalized.device.isMps() and f16_normalized.device_storage != null and
            f16_cosine.device.isMps() and f16_cosine.device_storage != null and
            f16_distance.device.isMps() and f16_distance.device_storage != null and
            closeF16(f16_norm_back.data, &.{ f16_norm0, f16_norm1 }, 0.05) and
            closeF16(f16_normalized_back.data, &.{ 1.0 / f16_norm0, 2.0 / f16_norm0, 3.0 / f16_norm0, 4.0 / f16_norm1, 5.0 / f16_norm1, 6.0 / f16_norm1 }, 0.03) and
            closeF16(f16_cosine_back.data, &.{ 1.0 / (f16_norm0 * f16_other_norm0), 21.0 / (f16_norm1 * f16_other_norm1) }, 0.03) and
            closeF16(f16_distance_back.data, &.{ std.math.sqrt(@as(f32, 13)), std.math.sqrt(@as(f32, 41)) }, 0.05);

        var f16_transposed = try f16_mat_lhs.transpose();
        defer f16_transposed.deinit();
        var f16_transposed_back = try f16_transposed.cpu();
        defer f16_transposed_back.deinit();
        f16_transpose_ok = f16_transposed.device.isMps() and f16_transposed.device_storage != null and
            closeF16(f16_transposed_back.data, &.{ 1, 4, 2, 5, 3, 6 }, 0.02);

        var f16_row_bias = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 10), @as(f16, 20), @as(f16, 30) }, &.{3}, vx.mps(0));
        defer f16_row_bias.deinit();
        var f16_row_added = try f16_mat_lhs.add(f16_row_bias);
        defer f16_row_added.deinit();
        var f16_row_added_back = try f16_row_added.cpu();
        defer f16_row_added_back.deinit();
        var f16_row_sub = try f16_mat_lhs.sub(f16_row_bias);
        defer f16_row_sub.deinit();
        var f16_row_sub_back = try f16_row_sub.cpu();
        defer f16_row_sub_back.deinit();
        var f16_row_mul = try f16_mat_lhs.mul(f16_row_bias);
        defer f16_row_mul.deinit();
        var f16_row_mul_back = try f16_row_mul.cpu();
        defer f16_row_mul_back.deinit();
        var f16_row_div = try f16_mat_lhs.div(f16_row_bias);
        defer f16_row_div.deinit();
        var f16_row_div_back = try f16_row_div.cpu();
        defer f16_row_div_back.deinit();
        var f16_col_bias = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 100), @as(f16, 200) }, &.{2}, vx.mps(0));
        defer f16_col_bias.deinit();
        var f16_col_added = try f16_mat_lhs.add(f16_col_bias);
        defer f16_col_added.deinit();
        var f16_col_added_back = try f16_col_added.cpu();
        defer f16_col_added_back.deinit();
        var f16_col_sub = try f16_mat_lhs.sub(f16_col_bias);
        defer f16_col_sub.deinit();
        var f16_col_sub_back = try f16_col_sub.cpu();
        defer f16_col_sub_back.deinit();
        var f16_col_mul = try f16_mat_lhs.mul(f16_col_bias);
        defer f16_col_mul.deinit();
        var f16_col_mul_back = try f16_col_mul.cpu();
        defer f16_col_mul_back.deinit();
        var f16_col_div = try f16_mat_lhs.div(f16_col_bias);
        defer f16_col_div.deinit();
        var f16_col_div_back = try f16_col_div.cpu();
        defer f16_col_div_back.deinit();
        f16_broadcast_ok = f16_row_added.device.isMps() and f16_row_added.device_storage != null and
            f16_row_sub.device.isMps() and f16_row_sub.device_storage != null and
            f16_row_mul.device.isMps() and f16_row_mul.device_storage != null and
            f16_row_div.device.isMps() and f16_row_div.device_storage != null and
            f16_col_added.device.isMps() and f16_col_added.device_storage != null and
            f16_col_sub.device.isMps() and f16_col_sub.device_storage != null and
            f16_col_mul.device.isMps() and f16_col_mul.device_storage != null and
            f16_col_div.device.isMps() and f16_col_div.device_storage != null and
            closeF16(f16_row_added_back.data, &.{ 11, 22, 33, 14, 25, 36 }, 0.02) and
            closeF16(f16_row_sub_back.data, &.{ -9, -18, -27, -6, -15, -24 }, 0.02) and
            closeF16(f16_row_mul_back.data, &.{ 10, 40, 90, 40, 100, 180 }, 0.25) and
            closeF16(f16_row_div_back.data, &.{ 0.1, 0.1, 0.1, 0.4, 0.25, 0.2 }, 0.01) and
            closeF16(f16_col_added_back.data, &.{ 101, 102, 103, 204, 205, 206 }, 0.02) and
            closeF16(f16_col_sub_back.data, &.{ -99, -98, -97, -196, -195, -194 }, 0.02) and
            closeF16(f16_col_mul_back.data, &.{ 100, 200, 300, 800, 1000, 1200 }, 1.0) and
            closeF16(f16_col_div_back.data, &.{ 0.01, 0.02, 0.03, 0.02, 0.025, 0.03 }, 0.002);

        var f16_row_sum = try f16_mat_lhs.sum(1, false);
        defer f16_row_sum.deinit();
        var f16_row_sum_back = try f16_row_sum.cpu();
        defer f16_row_sum_back.deinit();
        var f16_col_max = try f16_mat_lhs.max(0, false);
        defer f16_col_max.deinit();
        var f16_col_max_back = try f16_col_max.cpu();
        defer f16_col_max_back.deinit();
        var f16_row_prod_keep = try f16_mat_lhs.prod(1, true);
        defer f16_row_prod_keep.deinit();
        var f16_row_prod_keep_back = try f16_row_prod_keep.cpu();
        defer f16_row_prod_keep_back.deinit();
        f16_reduction_ok = f16_row_sum.device.isMps() and f16_row_sum.device_storage != null and
            f16_col_max.device.isMps() and f16_col_max.device_storage != null and
            f16_row_prod_keep.device.isMps() and f16_row_prod_keep.device_storage != null and
            closeF16(f16_row_sum_back.data, &.{ 6, 15 }, 0.02) and
            closeF16(f16_col_max_back.data, &.{ 4, 5, 6 }, 0.02) and
            closeF16(f16_row_prod_keep_back.data, &.{ 6, 120 }, 0.5);

        var f16_row_mean = try f16_mat_lhs.mean(1, false);
        defer f16_row_mean.deinit();
        var f16_row_mean_back = try f16_row_mean.cpu();
        defer f16_row_mean_back.deinit();
        var f16_col_mean_keep = try f16_mat_lhs.mean(0, true);
        defer f16_col_mean_keep.deinit();
        var f16_col_mean_keep_back = try f16_col_mean_keep.cpu();
        defer f16_col_mean_keep_back.deinit();
        var f16_flat_var = try f16_mat_lhs.variance(null, false, @as(f16, 0.0));
        defer f16_flat_var.deinit();
        var f16_flat_var_back = try f16_flat_var.cpu();
        defer f16_flat_var_back.deinit();
        var f16_row_var = try f16_mat_lhs.variance(1, false, @as(f16, 0.0));
        defer f16_row_var.deinit();
        var f16_row_var_back = try f16_row_var.cpu();
        defer f16_row_var_back.deinit();
        var f16_col_std_keep = try f16_mat_lhs.stddev(0, true, @as(f16, 0.0));
        defer f16_col_std_keep.deinit();
        var f16_col_std_keep_back = try f16_col_std_keep.cpu();
        defer f16_col_std_keep_back.deinit();
        f16_stats_ok = f16_row_mean.device.isMps() and f16_row_mean.device_storage != null and
            f16_col_mean_keep.device.isMps() and f16_col_mean_keep.device_storage != null and
            f16_flat_var.device.isMps() and f16_flat_var.device_storage != null and
            f16_row_var.device.isMps() and f16_row_var.device_storage != null and
            f16_col_std_keep.device.isMps() and f16_col_std_keep.device_storage != null and
            closeF16(f16_row_mean_back.data, &.{ 2, 5 }, 0.02) and
            closeF16(f16_col_mean_keep_back.data, &.{ 2.5, 3.5, 4.5 }, 0.02) and
            closeF16(f16_flat_var_back.data, &.{35.0 / 12.0}, 0.02) and
            closeF16(f16_row_var_back.data, &.{ 2.0 / 3.0, 2.0 / 3.0 }, 0.02) and
            closeF16(f16_col_std_keep_back.data, &.{ 1.5, 1.5, 1.5 }, 0.02);

        var f16_shifted_for_max = try f16_mat_lhs.subScalar(@as(f16, 3.0));
        defer f16_shifted_for_max.deinit();
        var f16_scaled_for_max = try f16_mat_lhs.mulScalar(@as(f16, 0.1));
        defer f16_scaled_for_max.deinit();
        var f16_maximum = try f16_shifted_for_max.maximum(f16_scaled_for_max);
        defer f16_maximum.deinit();
        var f16_maximum_back = try f16_maximum.cpu();
        defer f16_maximum_back.deinit();
        var f16_minimum = try f16_shifted_for_max.minimum(f16_scaled_for_max);
        defer f16_minimum.deinit();
        var f16_minimum_back = try f16_minimum.cpu();
        defer f16_minimum_back.deinit();
        var f16_maximum_scalar = try f16_shifted_for_max.maximumScalar(@as(f16, 0.0));
        defer f16_maximum_scalar.deinit();
        var f16_maximum_scalar_back = try f16_maximum_scalar.cpu();
        defer f16_maximum_scalar_back.deinit();
        var f16_minimum_scalar = try f16_shifted_for_max.minimumScalar(@as(f16, 0.0));
        defer f16_minimum_scalar.deinit();
        var f16_minimum_scalar_back = try f16_minimum_scalar.cpu();
        defer f16_minimum_scalar_back.deinit();
        const f16_minmax_ok = f16_maximum.device.isMps() and f16_maximum.device_storage != null and
            f16_minimum.device.isMps() and f16_minimum.device_storage != null and
            f16_maximum_scalar.device.isMps() and f16_maximum_scalar.device_storage != null and
            f16_minimum_scalar.device.isMps() and f16_minimum_scalar.device_storage != null and
            closeF16(f16_maximum_back.data, &.{ 0.1, 0.2, 0.3, 1.0, 2.0, 3.0 }, 0.01) and
            closeF16(f16_minimum_back.data, &.{ -2.0, -1.0, 0.0, 0.4, 0.5, 0.6 }, 0.01) and
            closeF16(f16_maximum_scalar_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.01) and
            closeF16(f16_minimum_scalar_back.data, &.{ -2, -1, 0, 0, 0, 0 }, 0.01);

        var f16_relu = try f16_shifted_for_max.relu();
        defer f16_relu.deinit();
        var f16_relu_back = try f16_relu.cpu();
        defer f16_relu_back.deinit();
        var f16_threshold = try f16_shifted_for_max.threshold(@as(f16, 0.5), @as(f16, 0.5));
        defer f16_threshold.deinit();
        var f16_threshold_back = try f16_threshold.cpu();
        defer f16_threshold_back.deinit();
        var f16_clip = try f16_shifted_for_max.clip(@as(f16, -0.5), @as(f16, 0.5));
        defer f16_clip.deinit();
        var f16_clip_back = try f16_clip.cpu();
        defer f16_clip_back.deinit();
        var f16_relu6 = try f16_shifted_for_max.relu6();
        defer f16_relu6.deinit();
        var f16_relu6_back = try f16_relu6.cpu();
        defer f16_relu6_back.deinit();
        var f16_hardtanh = try f16_shifted_for_max.hardtanh(@as(f16, -0.75), @as(f16, 1.25));
        defer f16_hardtanh.deinit();
        var f16_hardtanh_back = try f16_hardtanh.cpu();
        defer f16_hardtanh_back.deinit();
        var f16_clip_min_values = try vx.Array(f16).fullOn(allocator, &.{ 2, 3 }, @as(f16, -0.25), vx.mps(0));
        defer f16_clip_min_values.deinit();
        var f16_clip_max_values = try vx.Array(f16).fullOn(allocator, &.{ 2, 3 }, @as(f16, 0.75), vx.mps(0));
        defer f16_clip_max_values.deinit();
        var f16_clip_array = try f16_shifted_for_max.clipArray(f16_clip_min_values, f16_clip_max_values);
        defer f16_clip_array.deinit();
        var f16_clip_array_back = try f16_clip_array.cpu();
        defer f16_clip_array_back.deinit();
        var f16_sigmoid = try f16_shifted_for_max.sigmoid();
        defer f16_sigmoid.deinit();
        var f16_sigmoid_back = try f16_sigmoid.cpu();
        defer f16_sigmoid_back.deinit();
        var f16_softsign = try f16_shifted_for_max.softsign();
        defer f16_softsign.deinit();
        var f16_softsign_back = try f16_softsign.cpu();
        defer f16_softsign_back.deinit();
        const sigmoid_neg2 = @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, 2.0)));
        const sigmoid_pos3 = @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -3.0)));
        const selu_scale = @as(f32, 1.0507009873554805);
        const selu_alpha = @as(f32, 1.6732632423543772);
        const selu_neg2 = selu_scale * selu_alpha * (std.math.exp(@as(f32, -2.0)) - 1.0);
        const selu_neg1 = selu_scale * selu_alpha * (std.math.exp(@as(f32, -1.0)) - 1.0);
        f16_activation_ok = f16_relu.device.isMps() and f16_relu.device_storage != null and
            f16_threshold.device.isMps() and f16_threshold.device_storage != null and
            f16_clip.device.isMps() and f16_clip.device_storage != null and
            f16_relu6.device.isMps() and f16_relu6.device_storage != null and
            f16_hardtanh.device.isMps() and f16_hardtanh.device_storage != null and
            f16_clip_array.device.isMps() and f16_clip_array.device_storage != null and
            f16_sigmoid.device.isMps() and f16_sigmoid.device_storage != null and
            f16_softsign.device.isMps() and f16_softsign.device_storage != null and
            closeF16(f16_relu_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.01) and
            closeF16(f16_threshold_back.data, &.{ 0.5, 0.5, 0.5, 1, 2, 3 }, 0.01) and
            closeF16(f16_clip_back.data, &.{ -0.5, -0.5, 0, 0.5, 0.5, 0.5 }, 0.01) and
            closeF16(f16_relu6_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.01) and
            closeF16(f16_hardtanh_back.data, &.{ -0.75, -0.75, 0, 1, 1.25, 1.25 }, 0.01) and
            closeF16(f16_clip_array_back.data, &.{ -0.25, -0.25, 0, 0.75, 0.75, 0.75 }, 0.01) and
            closeF16(f16_sigmoid_back.data, &.{ sigmoid_neg2, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.e), 0.5, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0))), @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -2.0))), sigmoid_pos3 }, 0.02) and
            closeF16(f16_softsign_back.data, &.{ -2.0 / 3.0, -0.5, 0, 0.5, 2.0 / 3.0, 0.75 }, 0.02);

        var f16_rsqrt = try f16_mat_lhs.rsqrt();
        defer f16_rsqrt.deinit();
        var f16_rsqrt_back = try f16_rsqrt.cpu();
        defer f16_rsqrt_back.deinit();
        var f16_leaky = try f16_shifted_for_max.leakyRelu(@as(f16, 0.1));
        defer f16_leaky.deinit();
        var f16_leaky_back = try f16_leaky.cpu();
        defer f16_leaky_back.deinit();
        var f16_silu = try f16_shifted_for_max.silu();
        defer f16_silu.deinit();
        var f16_silu_back = try f16_silu.cpu();
        defer f16_silu_back.deinit();
        var f16_hardsigmoid = try f16_shifted_for_max.hardsigmoid();
        defer f16_hardsigmoid.deinit();
        var f16_hardsigmoid_back = try f16_hardsigmoid.cpu();
        defer f16_hardsigmoid_back.deinit();
        var f16_hardswish = try f16_shifted_for_max.hardswish();
        defer f16_hardswish.deinit();
        var f16_hardswish_back = try f16_hardswish.cpu();
        defer f16_hardswish_back.deinit();
        var f16_softshrink = try f16_shifted_for_max.softshrink(@as(f16, 0.5));
        defer f16_softshrink.deinit();
        var f16_softshrink_back = try f16_softshrink.cpu();
        defer f16_softshrink_back.deinit();
        var f16_elu = try f16_shifted_for_max.elu(@as(f16, 1.0));
        defer f16_elu.deinit();
        var f16_elu_back = try f16_elu.cpu();
        defer f16_elu_back.deinit();
        var f16_celu = try f16_shifted_for_max.celu(@as(f16, 2.0));
        defer f16_celu.deinit();
        var f16_celu_back = try f16_celu.cpu();
        defer f16_celu_back.deinit();
        var f16_selu = try f16_shifted_for_max.selu();
        defer f16_selu.deinit();
        var f16_selu_back = try f16_selu.cpu();
        defer f16_selu_back.deinit();
        var f16_tanh = try f16_shifted_for_max.tanh();
        defer f16_tanh.deinit();
        var f16_tanh_back = try f16_tanh.cpu();
        defer f16_tanh_back.deinit();
        var f16_tanhshrink = try f16_shifted_for_max.tanhshrink();
        defer f16_tanhshrink.deinit();
        var f16_tanhshrink_back = try f16_tanhshrink.cpu();
        defer f16_tanhshrink_back.deinit();
        f16_activation_compose_ok = f16_rsqrt.device.isMps() and f16_rsqrt.device_storage != null and
            f16_leaky.device.isMps() and f16_leaky.device_storage != null and
            f16_silu.device.isMps() and f16_silu.device_storage != null and
            f16_hardsigmoid.device.isMps() and f16_hardsigmoid.device_storage != null and
            f16_hardswish.device.isMps() and f16_hardswish.device_storage != null and
            f16_softshrink.device.isMps() and f16_softshrink.device_storage != null and
            f16_elu.device.isMps() and f16_elu.device_storage != null and
            f16_celu.device.isMps() and f16_celu.device_storage != null and
            f16_selu.device.isMps() and f16_selu.device_storage != null and
            f16_tanh.device.isMps() and f16_tanh.device_storage != null and
            f16_tanhshrink.device.isMps() and f16_tanhshrink.device_storage != null and
            closeF16(f16_rsqrt_back.data, &.{ 1.0, 1.0 / std.math.sqrt(@as(f32, 2.0)), 1.0 / std.math.sqrt(@as(f32, 3.0)), 0.5, 1.0 / std.math.sqrt(@as(f32, 5.0)), 1.0 / std.math.sqrt(@as(f32, 6.0)) }, 0.02) and
            closeF16(f16_leaky_back.data, &.{ -0.2, -0.1, 0, 1, 2, 3 }, 0.02) and
            closeF16(f16_silu_back.data, &.{ -2.0 * sigmoid_neg2, -1.0 / (@as(f32, 1.0) + std.math.e), 0, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0))), @as(f32, 2.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -2.0))), @as(f32, 3.0) * sigmoid_pos3 }, 0.03) and
            closeF16(f16_hardsigmoid_back.data, &.{ 1.0 / 6.0, 2.0 / 6.0, 0.5, 4.0 / 6.0, 5.0 / 6.0, 1.0 }, 0.02) and
            closeF16(f16_hardswish_back.data, &.{ -2.0 / 6.0, -2.0 / 6.0, 0, 4.0 / 6.0, 10.0 / 6.0, 3.0 }, 0.03) and
            closeF16(f16_softshrink_back.data, &.{ -1.5, -0.5, 0, 0.5, 1.5, 2.5 }, 0.02) and
            closeF16(f16_elu_back.data, &.{ std.math.exp(@as(f32, -2.0)) - 1.0, std.math.exp(@as(f32, -1.0)) - 1.0, 0, 1, 2, 3 }, 0.03) and
            closeF16(f16_celu_back.data, &.{ 2.0 * (std.math.exp(@as(f32, -1.0)) - 1.0), 2.0 * (std.math.exp(@as(f32, -0.5)) - 1.0), 0, 1, 2, 3 }, 0.03) and
            closeF16(f16_selu_back.data, &.{ selu_neg2, selu_neg1, 0, selu_scale, 2.0 * selu_scale, 3.0 * selu_scale }, 0.03) and
            closeF16(f16_tanh_back.data, &.{ std.math.tanh(@as(f32, -2.0)), std.math.tanh(@as(f32, -1.0)), 0, std.math.tanh(@as(f32, 1.0)), std.math.tanh(@as(f32, 2.0)), std.math.tanh(@as(f32, 3.0)) }, 0.03) and
            closeF16(f16_tanhshrink_back.data, &.{ -2.0 - std.math.tanh(@as(f32, -2.0)), -1.0 - std.math.tanh(@as(f32, -1.0)), 0, 1.0 - std.math.tanh(@as(f32, 1.0)), 2.0 - std.math.tanh(@as(f32, 2.0)), 3.0 - std.math.tanh(@as(f32, 3.0)) }, 0.03);

        var f16_pow_zero = try f16_mat_lhs.powScalar(@as(f16, 0));
        defer f16_pow_zero.deinit();
        var f16_pow_zero_back = try f16_pow_zero.cpu();
        defer f16_pow_zero_back.deinit();
        var f16_pow_one = try f16_mat_lhs.powScalar(@as(f16, 1));
        defer f16_pow_one.deinit();
        var f16_pow_one_back = try f16_pow_one.cpu();
        defer f16_pow_one_back.deinit();
        var f16_pow_recip = try f16_mat_lhs.powScalar(@as(f16, -1));
        defer f16_pow_recip.deinit();
        var f16_pow_recip_back = try f16_pow_recip.cpu();
        defer f16_pow_recip_back.deinit();
        var f16_pow_sqrt = try f16_mat_lhs.powScalar(@as(f16, 0.5));
        defer f16_pow_sqrt.deinit();
        var f16_pow_sqrt_back = try f16_pow_sqrt.cpu();
        defer f16_pow_sqrt_back.deinit();
        var f16_pow_rsqrt = try f16_mat_lhs.powScalar(@as(f16, -0.5));
        defer f16_pow_rsqrt.deinit();
        var f16_pow_rsqrt_back = try f16_pow_rsqrt.cpu();
        defer f16_pow_rsqrt_back.deinit();
        var f16_pow_square = try f16_mat_lhs.powScalar(@as(f16, 2));
        defer f16_pow_square.deinit();
        var f16_pow_square_back = try f16_pow_square.cpu();
        defer f16_pow_square_back.deinit();
        var f16_pow_cube = try f16_mat_lhs.powScalar(@as(f16, 3));
        defer f16_pow_cube.deinit();
        var f16_pow_cube_back = try f16_pow_cube.cpu();
        defer f16_pow_cube_back.deinit();
        f16_pow_ok = f16_pow_zero.device.isMps() and f16_pow_zero.device_storage != null and
            f16_pow_one.device.isMps() and f16_pow_one.device_storage != null and
            f16_pow_recip.device.isMps() and f16_pow_recip.device_storage != null and
            f16_pow_sqrt.device.isMps() and f16_pow_sqrt.device_storage != null and
            f16_pow_rsqrt.device.isMps() and f16_pow_rsqrt.device_storage != null and
            f16_pow_square.device.isMps() and f16_pow_square.device_storage != null and
            f16_pow_cube.device.isMps() and f16_pow_cube.device_storage != null and
            closeF16(f16_pow_zero_back.data, &.{ 1, 1, 1, 1, 1, 1 }, 0.02) and
            closeF16(f16_pow_one_back.data, &.{ 1, 2, 3, 4, 5, 6 }, 0.02) and
            closeF16(f16_pow_recip_back.data, &.{ 1, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0 }, 0.01) and
            closeF16(f16_pow_sqrt_back.data, &.{ 1, std.math.sqrt(@as(f32, 2)), std.math.sqrt(@as(f32, 3)), 2, std.math.sqrt(@as(f32, 5)), std.math.sqrt(@as(f32, 6)) }, 0.03) and
            closeF16(f16_pow_rsqrt_back.data, &.{ 1, 1.0 / std.math.sqrt(@as(f32, 2)), 1.0 / std.math.sqrt(@as(f32, 3)), 0.5, 1.0 / std.math.sqrt(@as(f32, 5)), 1.0 / std.math.sqrt(@as(f32, 6)) }, 0.03) and
            closeF16(f16_pow_square_back.data, &.{ 1, 4, 9, 16, 25, 36 }, 0.125) and
            closeF16(f16_pow_cube_back.data, &.{ 1, 8, 27, 64, 125, 216 }, 0.5);

        var f16_ternary_base = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 2), @as(f16, 3), @as(f16, 4) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_ternary_base.deinit();
        var f16_ternary_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 1), @as(f16, 1), @as(f16, 1) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_ternary_lhs.deinit();
        var f16_ternary_rhs = try vx.Array(f16).fromSliceOn(allocator, &.{ @as(f16, 1), @as(f16, 1.5), @as(f16, 2), @as(f16, 2.5) }, &.{ 2, 2 }, vx.mps(0));
        defer f16_ternary_rhs.deinit();
        var f16_addcmul = try f16_ternary_base.addcmul(f16_ternary_lhs, f16_ternary_rhs, @as(f16, 2.0));
        defer f16_addcmul.deinit();
        var f16_addcmul_back = try f16_addcmul.cpu();
        defer f16_addcmul_back.deinit();
        var f16_addcdiv = try f16_ternary_base.addcdiv(f16_ternary_lhs, f16_ternary_rhs, @as(f16, 0.5));
        defer f16_addcdiv.deinit();
        var f16_addcdiv_back = try f16_addcdiv.cpu();
        defer f16_addcdiv_back.deinit();
        var f16_lerp_scalar = try f16_ternary_base.lerpScalar(f16_ternary_rhs, @as(f16, 0.5));
        defer f16_lerp_scalar.deinit();
        var f16_lerp_scalar_back = try f16_lerp_scalar.cpu();
        defer f16_lerp_scalar_back.deinit();
        var f16_lerp_array = try f16_ternary_base.lerp(f16_ternary_lhs, f16_ternary_rhs);
        defer f16_lerp_array.deinit();
        var f16_lerp_array_back = try f16_lerp_array.cpu();
        defer f16_lerp_array_back.deinit();
        const f16_ternary_ok = f16_addcmul.device.isMps() and f16_addcmul.device_storage != null and
            f16_addcdiv.device.isMps() and f16_addcdiv.device_storage != null and
            f16_lerp_scalar.device.isMps() and f16_lerp_scalar.device_storage != null and
            f16_lerp_array.device.isMps() and f16_lerp_array.device_storage != null and
            closeF16(f16_addcmul_back.data, &.{ 3, 5, 7, 9 }, 0.02) and
            closeF16(f16_addcdiv_back.data, &.{ 1.5, 2.3333333, 3.25, 4.2 }, 0.02) and
            closeF16(f16_lerp_scalar_back.data, &.{ 1, 1.75, 2.5, 3.25 }, 0.02) and
            closeF16(f16_lerp_array_back.data, &.{ 1, 0.5, -1, -3.5 }, 0.02);

        var f16_loss_target = try vx.Array(f16).zerosOn(allocator, &.{ 2, 3 }, vx.mps(0));
        defer f16_loss_target.deinit();
        var f16_mse = try f16_shifted_for_max.mseLoss(f16_loss_target, .none);
        defer f16_mse.deinit();
        var f16_mse_back = try f16_mse.cpu();
        defer f16_mse_back.deinit();
        var f16_l1 = try f16_shifted_for_max.l1Loss(f16_loss_target, .none);
        defer f16_l1.deinit();
        var f16_l1_back = try f16_l1.cpu();
        defer f16_l1_back.deinit();
        var f16_smooth_l1 = try f16_shifted_for_max.smoothL1Loss(f16_loss_target, @as(f16, 1), .none);
        defer f16_smooth_l1.deinit();
        var f16_smooth_l1_back = try f16_smooth_l1.cpu();
        defer f16_smooth_l1_back.deinit();
        var f16_huber = try f16_shifted_for_max.huberLoss(f16_loss_target, @as(f16, 1), .none);
        defer f16_huber.deinit();
        var f16_huber_back = try f16_huber.cpu();
        defer f16_huber_back.deinit();
        f16_loss_ok = f16_mse.device.isMps() and f16_mse.device_storage != null and
            f16_l1.device.isMps() and f16_l1.device_storage != null and
            f16_smooth_l1.device.isMps() and f16_smooth_l1.device_storage != null and
            f16_huber.device.isMps() and f16_huber.device_storage != null and
            closeF16(f16_mse_back.data, &.{ 4, 1, 0, 1, 4, 9 }, 0.02) and
            closeF16(f16_l1_back.data, &.{ 2, 1, 0, 1, 2, 3 }, 0.02) and
            closeF16(f16_smooth_l1_back.data, &.{ 1.5, 0.5, 0, 0.5, 1.5, 2.5 }, 0.02) and
            closeF16(f16_huber_back.data, &.{ 1.5, 0.5, 0, 0.5, 1.5, 2.5 }, 0.02);

        var f16_softmax_row = try f16_mat_lhs.softmax(1);
        defer f16_softmax_row.deinit();
        var f16_softmax_row_back = try f16_softmax_row.cpu();
        defer f16_softmax_row_back.deinit();
        var f16_softmax_col = try f16_mat_lhs.softmax(0);
        defer f16_softmax_col.deinit();
        var f16_softmax_col_back = try f16_softmax_col.cpu();
        defer f16_softmax_col_back.deinit();
        const f16_row_denom = std.math.exp(@as(f32, -2)) + std.math.exp(@as(f32, -1)) + 1.0;
        const f16_col_denom = std.math.exp(@as(f32, -3)) + 1.0;
        f16_softmax_ok = f16_softmax_row.device.isMps() and f16_softmax_row.device_storage != null and
            f16_softmax_col.device.isMps() and f16_softmax_col.device_storage != null and
            closeF16(f16_softmax_row_back.data, &.{ std.math.exp(@as(f32, -2)) / f16_row_denom, std.math.exp(@as(f32, -1)) / f16_row_denom, 1.0 / f16_row_denom, std.math.exp(@as(f32, -2)) / f16_row_denom, std.math.exp(@as(f32, -1)) / f16_row_denom, 1.0 / f16_row_denom }, 0.01) and
            closeF16(f16_softmax_col_back.data, &.{ std.math.exp(@as(f32, -3)) / f16_col_denom, std.math.exp(@as(f32, -3)) / f16_col_denom, std.math.exp(@as(f32, -3)) / f16_col_denom, 1.0 / f16_col_denom, 1.0 / f16_col_denom, 1.0 / f16_col_denom }, 0.01);

        var f16_log_softmax_row = try f16_mat_lhs.logSoftmax(1);
        defer f16_log_softmax_row.deinit();
        var f16_log_softmax_row_back = try f16_log_softmax_row.cpu();
        defer f16_log_softmax_row_back.deinit();
        var f16_log_softmax_col = try f16_mat_lhs.logSoftmax(0);
        defer f16_log_softmax_col.deinit();
        var f16_log_softmax_col_back = try f16_log_softmax_col.cpu();
        defer f16_log_softmax_col_back.deinit();
        const f16_row_log_denom = std.math.log(f32, std.math.e, f16_row_denom);
        const f16_col_log_denom = std.math.log(f32, std.math.e, f16_col_denom);
        f16_log_softmax_ok = f16_log_softmax_row.device.isMps() and f16_log_softmax_row.device_storage != null and
            f16_log_softmax_col.device.isMps() and f16_log_softmax_col.device_storage != null and
            closeF16(f16_log_softmax_row_back.data, &.{ -2.0 - f16_row_log_denom, -1.0 - f16_row_log_denom, -f16_row_log_denom, -2.0 - f16_row_log_denom, -1.0 - f16_row_log_denom, -f16_row_log_denom }, 0.03) and
            closeF16(f16_log_softmax_col_back.data, &.{ -3.0 - f16_col_log_denom, -3.0 - f16_col_log_denom, -3.0 - f16_col_log_denom, -f16_col_log_denom, -f16_col_log_denom, -f16_col_log_denom }, 0.03);

        var f16_softmin_row = try f16_mat_lhs.softmin(1);
        defer f16_softmin_row.deinit();
        var f16_softmin_row_back = try f16_softmin_row.cpu();
        defer f16_softmin_row_back.deinit();
        var f16_softmin_col = try f16_mat_lhs.softmin(0);
        defer f16_softmin_col.deinit();
        var f16_softmin_col_back = try f16_softmin_col.cpu();
        defer f16_softmin_col_back.deinit();
        const f16_softmin_row_denom = 1.0 + std.math.exp(@as(f32, -1)) + std.math.exp(@as(f32, -2));
        const f16_softmin_col_denom = 1.0 + std.math.exp(@as(f32, -3));
        f16_softmin_ok = f16_softmin_row.device.isMps() and f16_softmin_row.device_storage != null and
            f16_softmin_col.device.isMps() and f16_softmin_col.device_storage != null and
            closeF16(f16_softmin_row_back.data, &.{ 1.0 / f16_softmin_row_denom, std.math.exp(@as(f32, -1)) / f16_softmin_row_denom, std.math.exp(@as(f32, -2)) / f16_softmin_row_denom, 1.0 / f16_softmin_row_denom, std.math.exp(@as(f32, -1)) / f16_softmin_row_denom, std.math.exp(@as(f32, -2)) / f16_softmin_row_denom }, 0.01) and
            closeF16(f16_softmin_col_back.data, &.{ 1.0 / f16_softmin_col_denom, 1.0 / f16_softmin_col_denom, 1.0 / f16_softmin_col_denom, std.math.exp(@as(f32, -3)) / f16_softmin_col_denom, std.math.exp(@as(f32, -3)) / f16_softmin_col_denom, std.math.exp(@as(f32, -3)) / f16_softmin_col_denom }, 0.01);

        var f16_log_softmin_row = try f16_mat_lhs.logSoftmin(1);
        defer f16_log_softmin_row.deinit();
        var f16_log_softmin_row_back = try f16_log_softmin_row.cpu();
        defer f16_log_softmin_row_back.deinit();
        var f16_log_softmin_col = try f16_mat_lhs.logSoftmin(0);
        defer f16_log_softmin_col.deinit();
        var f16_log_softmin_col_back = try f16_log_softmin_col.cpu();
        defer f16_log_softmin_col_back.deinit();
        const f16_softmin_row_log_denom = std.math.log(f32, std.math.e, f16_softmin_row_denom);
        const f16_softmin_col_log_denom = std.math.log(f32, std.math.e, f16_softmin_col_denom);
        f16_log_softmin_ok = f16_log_softmin_row.device.isMps() and f16_log_softmin_row.device_storage != null and
            f16_log_softmin_col.device.isMps() and f16_log_softmin_col.device_storage != null and
            closeF16(f16_log_softmin_row_back.data, &.{ -f16_softmin_row_log_denom, -1.0 - f16_softmin_row_log_denom, -2.0 - f16_softmin_row_log_denom, -f16_softmin_row_log_denom, -1.0 - f16_softmin_row_log_denom, -2.0 - f16_softmin_row_log_denom }, 0.03) and
            closeF16(f16_log_softmin_col_back.data, &.{ -f16_softmin_col_log_denom, -f16_softmin_col_log_denom, -f16_softmin_col_log_denom, -3.0 - f16_softmin_col_log_denom, -3.0 - f16_softmin_col_log_denom, -3.0 - f16_softmin_col_log_denom }, 0.03);

        var bf16_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_lhs.deinit();
        var bf16_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20), vx.BFloat16.fromF32(30), vx.BFloat16.fromF32(40) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_rhs.deinit();
        var bf16_add = try bf16_lhs.add(bf16_rhs);
        defer bf16_add.deinit();
        var bf16_add_back = try bf16_add.cpu();
        defer bf16_add_back.deinit();
        var bf16_div = try bf16_rhs.div(bf16_lhs);
        defer bf16_div.deinit();
        var bf16_div_back = try bf16_div.cpu();
        defer bf16_div_back.deinit();
        bf16_elementwise_ok = bf16_add.device.isMps() and bf16_add.device_storage != null and
            bf16_div.device.isMps() and bf16_div.device_storage != null and
            closeBF16(bf16_add_back.data, &.{ 11, 22, 33, 44 }, 0.125) and
            closeBF16(bf16_div_back.data, &.{ 10, 10, 10, 10 }, 0.125);

        var bf16_scaled = try bf16_lhs.mulScalar(vx.BFloat16.fromF32(2.0));
        defer bf16_scaled.deinit();
        var bf16_scaled_back = try bf16_scaled.cpu();
        defer bf16_scaled_back.deinit();
        var bf16_rsub = try bf16_lhs.subScalar(vx.BFloat16.fromF32(10.0));
        defer bf16_rsub.deinit();
        var bf16_rsub_back = try bf16_rsub.cpu();
        defer bf16_rsub_back.deinit();
        var bf16_scalar_array = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{vx.BFloat16.fromF32(2.0)}, &.{1}, vx.mps(0));
        defer bf16_scalar_array.deinit();
        var bf16_array_scalar_mul = try bf16_lhs.mul(bf16_scalar_array);
        defer bf16_array_scalar_mul.deinit();
        var bf16_array_scalar_mul_back = try bf16_array_scalar_mul.cpu();
        defer bf16_array_scalar_mul_back.deinit();
        var bf16_scalar_array_rsub = try bf16_scalar_array.sub(bf16_lhs);
        defer bf16_scalar_array_rsub.deinit();
        var bf16_scalar_array_rsub_back = try bf16_scalar_array_rsub.cpu();
        defer bf16_scalar_array_rsub_back.deinit();
        bf16_scalar_ok = bf16_scaled.device.isMps() and bf16_scaled.device_storage != null and
            bf16_rsub.device.isMps() and bf16_rsub.device_storage != null and
            bf16_array_scalar_mul.device.isMps() and bf16_array_scalar_mul.device_storage != null and
            bf16_scalar_array_rsub.device.isMps() and bf16_scalar_array_rsub.device_storage != null and
            closeBF16(bf16_scaled_back.data, &.{ 2, 4, 6, 8 }, 0.125) and
            closeBF16(bf16_rsub_back.data, &.{ -9, -8, -7, -6 }, 0.125) and
            closeBF16(bf16_array_scalar_mul_back.data, &.{ 2, 4, 6, 8 }, 0.125) and
            closeBF16(bf16_scalar_array_rsub_back.data, &.{ 1, 0, -1, -2 }, 0.125);

        var bf16_abs_source = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(-1), vx.BFloat16.fromF32(-2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_abs_source.deinit();
        var bf16_abs = try bf16_abs_source.abs();
        defer bf16_abs.deinit();
        var bf16_abs_back = try bf16_abs.cpu();
        defer bf16_abs_back.deinit();
        var bf16_sqrt = try bf16_lhs.sqrt();
        defer bf16_sqrt.deinit();
        var bf16_sqrt_back = try bf16_sqrt.cpu();
        defer bf16_sqrt_back.deinit();
        var bf16_exp = try bf16_lhs.exp();
        defer bf16_exp.deinit();
        var bf16_exp_back = try bf16_exp.cpu();
        defer bf16_exp_back.deinit();
        bf16_unary_ok = bf16_abs.device.isMps() and bf16_abs.device_storage != null and
            bf16_sqrt.device.isMps() and bf16_sqrt.device_storage != null and
            bf16_exp.device.isMps() and bf16_exp.device_storage != null and
            closeBF16(bf16_abs_back.data, &.{ 1, 2, 3, 4 }, 0.125) and
            closeBF16(bf16_sqrt_back.data, &.{ 1, std.math.sqrt(@as(f32, 2)), std.math.sqrt(@as(f32, 3)), 2 }, 0.125) and
            closeBF16(bf16_exp_back.data, &.{ std.math.exp(@as(f32, 1)), std.math.exp(@as(f32, 2)), std.math.exp(@as(f32, 3)), std.math.exp(@as(f32, 4)) }, 1.0);

        var bf16_mat_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4), vx.BFloat16.fromF32(5), vx.BFloat16.fromF32(6) }, &.{ 2, 3 }, vx.mps(0));
        defer bf16_mat_lhs.deinit();
        var bf16_mat_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(7), vx.BFloat16.fromF32(8), vx.BFloat16.fromF32(9), vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(11), vx.BFloat16.fromF32(12) }, &.{ 3, 2 }, vx.mps(0));
        defer bf16_mat_rhs.deinit();
        var bf16_mat_out = try bf16_mat_lhs.matmul(bf16_mat_rhs);
        defer bf16_mat_out.deinit();
        var bf16_mat_back = try bf16_mat_out.cpu();
        defer bf16_mat_back.deinit();
        bf16_matmul_ok = bf16_mat_out.device.isMps() and bf16_mat_out.device_storage != null and
            closeBF16(bf16_mat_back.data, &.{ 58, 64, 139, 154 }, 0.5);

        var bf16_mat_addend = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(-1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(-2) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_mat_addend.deinit();
        var bf16_mat_add = try vx.matmulAdd(bf16_mat_lhs, bf16_mat_rhs, bf16_mat_addend);
        defer bf16_mat_add.deinit();
        var bf16_mat_add_back = try bf16_mat_add.cpu();
        defer bf16_mat_add_back.deinit();
        var bf16_mat_scaled_add = (try vx.axiom_backend.executeMatmulAddScaled(vx.BFloat16, .mps, bf16_mat_lhs, bf16_mat_rhs, bf16_mat_addend, 2.0, -1.0)) orelse return error.BackendFailure;
        defer bf16_mat_scaled_add.deinit();
        var bf16_mat_scaled_add_back = try bf16_mat_scaled_add.cpu();
        defer bf16_mat_scaled_add_back.deinit();
        bf16_matmul_add_ok = bf16_mat_add.device.isMps() and bf16_mat_add.device_storage != null and
            bf16_mat_scaled_add.device.isMps() and bf16_mat_scaled_add.device_storage != null and
            closeBF16(bf16_mat_add_back.data, &.{ 59, 63, 141, 152 }, 0.5) and
            closeBF16(bf16_mat_scaled_add_back.data, &.{ 115, 129, 276, 310 }, 1.0);

        var bf16_mat_chain_add = try bf16_mat_out.add(bf16_mat_addend);
        defer bf16_mat_chain_add.deinit();
        var bf16_mat_chain_add_back = try bf16_mat_chain_add.cpu();
        defer bf16_mat_chain_add_back.deinit();
        var bf16_mat_chain_sqrt = try bf16_mat_chain_add.sqrt();
        defer bf16_mat_chain_sqrt.deinit();
        var bf16_mat_chain_sqrt_back = try bf16_mat_chain_sqrt.cpu();
        defer bf16_mat_chain_sqrt_back.deinit();
        var bf16_mat_chain_exp_input = try bf16_mat_addend.sub(bf16_mat_out);
        defer bf16_mat_chain_exp_input.deinit();
        var bf16_mat_chain_exp_input_shifted = try bf16_mat_chain_exp_input.addScalar(vx.BFloat16.fromF32(60.0));
        defer bf16_mat_chain_exp_input_shifted.deinit();
        var bf16_mat_chain_exp = try bf16_mat_chain_exp_input_shifted.exp();
        defer bf16_mat_chain_exp.deinit();
        var bf16_mat_chain_exp_back = try bf16_mat_chain_exp.cpu();
        defer bf16_mat_chain_exp_back.deinit();
        bf16_matmul_chain_ok = bf16_mat_chain_add.device.isMps() and bf16_mat_chain_add.device_storage != null and
            bf16_mat_chain_sqrt.device.isMps() and bf16_mat_chain_sqrt.device_storage != null and
            bf16_mat_chain_exp_input.device.isMps() and bf16_mat_chain_exp_input.device_storage != null and
            bf16_mat_chain_exp_input_shifted.device.isMps() and bf16_mat_chain_exp_input_shifted.device_storage != null and
            bf16_mat_chain_exp.device.isMps() and bf16_mat_chain_exp.device_storage != null and
            closeBF16(bf16_mat_chain_add_back.data, &.{ 59, 63, 141, 152 }, 0.5) and
            closeBF16(bf16_mat_chain_sqrt_back.data, &.{ std.math.sqrt(@as(f32, 59)), std.math.sqrt(@as(f32, 63)), std.math.sqrt(@as(f32, 141)), std.math.sqrt(@as(f32, 152)) }, 0.125) and
            closeBF16(bf16_mat_chain_exp_back.data, &.{ std.math.exp(@as(f32, 3)), std.math.exp(@as(f32, -5)), std.math.exp(@as(f32, -77)), std.math.exp(@as(f32, -96)) }, 0.25);

        var bf16_vec_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3) }, &.{3}, vx.mps(0));
        defer bf16_vec_rhs.deinit();
        var bf16_matvec = try bf16_mat_lhs.matvec(bf16_vec_rhs);
        defer bf16_matvec.deinit();
        var bf16_matvec_back = try bf16_matvec.cpu();
        defer bf16_matvec_back.deinit();
        var bf16_vec_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2) }, &.{2}, vx.mps(0));
        defer bf16_vec_lhs.deinit();
        var bf16_vecmat = try bf16_vec_lhs.matmul(bf16_mat_lhs);
        defer bf16_vecmat.deinit();
        var bf16_vecmat_back = try bf16_vecmat.cpu();
        defer bf16_vecmat_back.deinit();
        var bf16_dot = try bf16_vec_rhs.dot(bf16_vec_rhs);
        defer bf16_dot.deinit();
        var bf16_dot_back = try bf16_dot.cpu();
        defer bf16_dot_back.deinit();
        bf16_vector_matmul_ok = bf16_matvec.device.isMps() and bf16_matvec.device_storage != null and
            bf16_vecmat.device.isMps() and bf16_vecmat.device_storage != null and
            bf16_dot.device.isMps() and bf16_dot.device_storage != null and
            closeBF16(bf16_matvec_back.data, &.{ 14, 32 }, 0.125) and
            closeBF16(bf16_vecmat_back.data, &.{ 9, 12, 15 }, 0.125) and
            closeBF16(bf16_dot_back.data, &.{14}, 0.125);

        var bf16_norm = try bf16_mat_lhs.norm(vx.BFloat16.fromF32(2), 1, false);
        defer bf16_norm.deinit();
        var bf16_norm_back = try bf16_norm.cpu();
        defer bf16_norm_back.deinit();
        var bf16_normalized = try bf16_mat_lhs.normalize(vx.BFloat16.fromF32(2), 1, vx.BFloat16.fromF32(0.001));
        defer bf16_normalized.deinit();
        var bf16_normalized_back = try bf16_normalized.cpu();
        defer bf16_normalized_back.deinit();
        var bf16_metric_other = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(0), vx.BFloat16.fromF32(0), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2) }, &.{ 2, 3 }, vx.mps(0));
        defer bf16_metric_other.deinit();
        var bf16_cosine = try bf16_mat_lhs.cosineSimilarity(bf16_metric_other, 1, vx.BFloat16.fromF32(0.001), false);
        defer bf16_cosine.deinit();
        var bf16_cosine_back = try bf16_cosine.cpu();
        defer bf16_cosine_back.deinit();
        var bf16_distance = try bf16_mat_lhs.pairwiseDistance(bf16_metric_other, vx.BFloat16.fromF32(2), 1, false);
        defer bf16_distance.deinit();
        var bf16_distance_back = try bf16_distance.cpu();
        defer bf16_distance_back.deinit();
        const bf16_norm0 = std.math.sqrt(@as(f32, 14));
        const bf16_norm1 = std.math.sqrt(@as(f32, 77));
        const bf16_other_norm0 = @as(f32, 1);
        const bf16_other_norm1 = std.math.sqrt(@as(f32, 6));
        bf16_metric_ok = bf16_norm.device.isMps() and bf16_norm.device_storage != null and
            bf16_normalized.device.isMps() and bf16_normalized.device_storage != null and
            bf16_cosine.device.isMps() and bf16_cosine.device_storage != null and
            bf16_distance.device.isMps() and bf16_distance.device_storage != null and
            closeBF16(bf16_norm_back.data, &.{ bf16_norm0, bf16_norm1 }, 0.125) and
            closeBF16(bf16_normalized_back.data, &.{ 1.0 / bf16_norm0, 2.0 / bf16_norm0, 3.0 / bf16_norm0, 4.0 / bf16_norm1, 5.0 / bf16_norm1, 6.0 / bf16_norm1 }, 0.05) and
            closeBF16(bf16_cosine_back.data, &.{ 1.0 / (bf16_norm0 * bf16_other_norm0), 21.0 / (bf16_norm1 * bf16_other_norm1) }, 0.05) and
            closeBF16(bf16_distance_back.data, &.{ std.math.sqrt(@as(f32, 13)), std.math.sqrt(@as(f32, 41)) }, 0.125);

        var bf16_transposed = try bf16_mat_lhs.transpose();
        defer bf16_transposed.deinit();
        var bf16_transposed_back = try bf16_transposed.cpu();
        defer bf16_transposed_back.deinit();
        bf16_transpose_ok = bf16_transposed.device.isMps() and bf16_transposed.device_storage != null and
            closeBF16(bf16_transposed_back.data, &.{ 1, 4, 2, 5, 3, 6 }, 0.125);

        var bf16_row_bias = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20), vx.BFloat16.fromF32(30) }, &.{3}, vx.mps(0));
        defer bf16_row_bias.deinit();
        var bf16_row_added = try bf16_mat_lhs.add(bf16_row_bias);
        defer bf16_row_added.deinit();
        var bf16_row_added_back = try bf16_row_added.cpu();
        defer bf16_row_added_back.deinit();
        var bf16_row_sub = try bf16_mat_lhs.sub(bf16_row_bias);
        defer bf16_row_sub.deinit();
        var bf16_row_sub_back = try bf16_row_sub.cpu();
        defer bf16_row_sub_back.deinit();
        var bf16_row_mul = try bf16_mat_lhs.mul(bf16_row_bias);
        defer bf16_row_mul.deinit();
        var bf16_row_mul_back = try bf16_row_mul.cpu();
        defer bf16_row_mul_back.deinit();
        var bf16_row_div = try bf16_mat_lhs.div(bf16_row_bias);
        defer bf16_row_div.deinit();
        var bf16_row_div_back = try bf16_row_div.cpu();
        defer bf16_row_div_back.deinit();
        var bf16_col_bias = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(100), vx.BFloat16.fromF32(200) }, &.{2}, vx.mps(0));
        defer bf16_col_bias.deinit();
        var bf16_col_added = try bf16_mat_lhs.add(bf16_col_bias);
        defer bf16_col_added.deinit();
        var bf16_col_added_back = try bf16_col_added.cpu();
        defer bf16_col_added_back.deinit();
        var bf16_col_sub = try bf16_mat_lhs.sub(bf16_col_bias);
        defer bf16_col_sub.deinit();
        var bf16_col_sub_back = try bf16_col_sub.cpu();
        defer bf16_col_sub_back.deinit();
        var bf16_col_mul = try bf16_mat_lhs.mul(bf16_col_bias);
        defer bf16_col_mul.deinit();
        var bf16_col_mul_back = try bf16_col_mul.cpu();
        defer bf16_col_mul_back.deinit();
        var bf16_col_div = try bf16_mat_lhs.div(bf16_col_bias);
        defer bf16_col_div.deinit();
        var bf16_col_div_back = try bf16_col_div.cpu();
        defer bf16_col_div_back.deinit();
        bf16_broadcast_ok = bf16_row_added.device.isMps() and bf16_row_added.device_storage != null and
            bf16_row_sub.device.isMps() and bf16_row_sub.device_storage != null and
            bf16_row_mul.device.isMps() and bf16_row_mul.device_storage != null and
            bf16_row_div.device.isMps() and bf16_row_div.device_storage != null and
            bf16_col_added.device.isMps() and bf16_col_added.device_storage != null and
            bf16_col_sub.device.isMps() and bf16_col_sub.device_storage != null and
            bf16_col_mul.device.isMps() and bf16_col_mul.device_storage != null and
            bf16_col_div.device.isMps() and bf16_col_div.device_storage != null and
            closeBF16(bf16_row_added_back.data, &.{ 11, 22, 33, 14, 25, 36 }, 0.125) and
            closeBF16(bf16_row_sub_back.data, &.{ -9, -18, -27, -6, -15, -24 }, 0.125) and
            closeBF16(bf16_row_mul_back.data, &.{ 10, 40, 90, 40, 100, 180 }, 0.5) and
            closeBF16(bf16_row_div_back.data, &.{ 0.1, 0.1, 0.1, 0.4, 0.25, 0.2 }, 0.02) and
            closeBF16(bf16_col_added_back.data, &.{ 101, 102, 103, 204, 205, 206 }, 0.125) and
            closeBF16(bf16_col_sub_back.data, &.{ -99, -98, -97, -196, -195, -194 }, 0.125) and
            closeBF16(bf16_col_mul_back.data, &.{ 100, 200, 300, 800, 1000, 1200 }, 2.0) and
            closeBF16(bf16_col_div_back.data, &.{ 0.01, 0.02, 0.03, 0.02, 0.025, 0.03 }, 0.005);

        var bf16_row_sum = try bf16_mat_lhs.sum(1, false);
        defer bf16_row_sum.deinit();
        var bf16_row_sum_back = try bf16_row_sum.cpu();
        defer bf16_row_sum_back.deinit();
        var bf16_col_max = try bf16_mat_lhs.max(0, false);
        defer bf16_col_max.deinit();
        var bf16_col_max_back = try bf16_col_max.cpu();
        defer bf16_col_max_back.deinit();
        var bf16_col_min = try bf16_mat_lhs.min(0, false);
        defer bf16_col_min.deinit();
        var bf16_col_min_back = try bf16_col_min.cpu();
        defer bf16_col_min_back.deinit();
        var bf16_row_prod_keep = try bf16_mat_lhs.prod(1, true);
        defer bf16_row_prod_keep.deinit();
        var bf16_row_prod_keep_back = try bf16_row_prod_keep.cpu();
        defer bf16_row_prod_keep_back.deinit();
        bf16_reduction_ok = bf16_row_sum.device.isMps() and bf16_row_sum.device_storage != null and
            bf16_col_max.device.isMps() and bf16_col_max.device_storage != null and
            bf16_col_min.device.isMps() and bf16_col_min.device_storage != null and
            bf16_row_prod_keep.device.isMps() and bf16_row_prod_keep.device_storage != null and
            closeBF16(bf16_row_sum_back.data, &.{ 6, 15 }, 0.125) and
            closeBF16(bf16_col_max_back.data, &.{ 4, 5, 6 }, 0.125) and
            closeBF16(bf16_col_min_back.data, &.{ 1, 2, 3 }, 0.125) and
            closeBF16(bf16_row_prod_keep_back.data, &.{ 6, 120 }, 0.5);

        var bf16_row_mean = try bf16_mat_lhs.mean(1, false);
        defer bf16_row_mean.deinit();
        var bf16_row_mean_back = try bf16_row_mean.cpu();
        defer bf16_row_mean_back.deinit();
        var bf16_col_mean_keep = try bf16_mat_lhs.mean(0, true);
        defer bf16_col_mean_keep.deinit();
        var bf16_col_mean_keep_back = try bf16_col_mean_keep.cpu();
        defer bf16_col_mean_keep_back.deinit();
        var bf16_flat_var = try bf16_mat_lhs.variance(null, false, vx.BFloat16.fromF32(0.0));
        defer bf16_flat_var.deinit();
        var bf16_flat_var_back = try bf16_flat_var.cpu();
        defer bf16_flat_var_back.deinit();
        var bf16_row_var = try bf16_mat_lhs.variance(1, false, vx.BFloat16.fromF32(0.0));
        defer bf16_row_var.deinit();
        var bf16_row_var_back = try bf16_row_var.cpu();
        defer bf16_row_var_back.deinit();
        var bf16_col_std_keep = try bf16_mat_lhs.stddev(0, true, vx.BFloat16.fromF32(0.0));
        defer bf16_col_std_keep.deinit();
        var bf16_col_std_keep_back = try bf16_col_std_keep.cpu();
        defer bf16_col_std_keep_back.deinit();
        bf16_stats_ok = bf16_row_mean.device.isMps() and bf16_row_mean.device_storage != null and
            bf16_col_mean_keep.device.isMps() and bf16_col_mean_keep.device_storage != null and
            bf16_flat_var.device.isMps() and bf16_flat_var.device_storage != null and
            bf16_row_var.device.isMps() and bf16_row_var.device_storage != null and
            bf16_col_std_keep.device.isMps() and bf16_col_std_keep.device_storage != null and
            closeBF16(bf16_row_mean_back.data, &.{ 2, 5 }, 0.125) and
            closeBF16(bf16_col_mean_keep_back.data, &.{ 2.5, 3.5, 4.5 }, 0.125) and
            closeBF16(bf16_flat_var_back.data, &.{35.0 / 12.0}, 0.125) and
            closeBF16(bf16_row_var_back.data, &.{ 2.0 / 3.0, 2.0 / 3.0 }, 0.125) and
            closeBF16(bf16_col_std_keep_back.data, &.{ 1.5, 1.5, 1.5 }, 0.125);

        var bf16_shifted_for_max = try bf16_mat_lhs.subScalar(vx.BFloat16.fromF32(3.0));
        defer bf16_shifted_for_max.deinit();
        var bf16_scaled_for_max = try bf16_mat_lhs.mulScalar(vx.BFloat16.fromF32(0.1));
        defer bf16_scaled_for_max.deinit();
        var bf16_maximum = try bf16_shifted_for_max.maximum(bf16_scaled_for_max);
        defer bf16_maximum.deinit();
        var bf16_maximum_back = try bf16_maximum.cpu();
        defer bf16_maximum_back.deinit();
        var bf16_minimum = try bf16_shifted_for_max.minimum(bf16_scaled_for_max);
        defer bf16_minimum.deinit();
        var bf16_minimum_back = try bf16_minimum.cpu();
        defer bf16_minimum_back.deinit();
        var bf16_maximum_scalar = try bf16_shifted_for_max.maximumScalar(vx.BFloat16.fromF32(0.0));
        defer bf16_maximum_scalar.deinit();
        var bf16_maximum_scalar_back = try bf16_maximum_scalar.cpu();
        defer bf16_maximum_scalar_back.deinit();
        var bf16_minimum_scalar = try bf16_shifted_for_max.minimumScalar(vx.BFloat16.fromF32(0.0));
        defer bf16_minimum_scalar.deinit();
        var bf16_minimum_scalar_back = try bf16_minimum_scalar.cpu();
        defer bf16_minimum_scalar_back.deinit();
        const bf16_minmax_ok = bf16_maximum.device.isMps() and bf16_maximum.device_storage != null and
            bf16_minimum.device.isMps() and bf16_minimum.device_storage != null and
            bf16_maximum_scalar.device.isMps() and bf16_maximum_scalar.device_storage != null and
            bf16_minimum_scalar.device.isMps() and bf16_minimum_scalar.device_storage != null and
            closeBF16(bf16_maximum_back.data, &.{ 0.1, 0.2, 0.3, 1.0, 2.0, 3.0 }, 0.125) and
            closeBF16(bf16_minimum_back.data, &.{ -2.0, -1.0, 0.0, 0.4, 0.5, 0.6 }, 0.125) and
            closeBF16(bf16_maximum_scalar_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_minimum_scalar_back.data, &.{ -2, -1, 0, 0, 0, 0 }, 0.125);

        var bf16_relu = try bf16_shifted_for_max.relu();
        defer bf16_relu.deinit();
        var bf16_relu_back = try bf16_relu.cpu();
        defer bf16_relu_back.deinit();
        var bf16_threshold = try bf16_shifted_for_max.threshold(vx.BFloat16.fromF32(0.5), vx.BFloat16.fromF32(0.5));
        defer bf16_threshold.deinit();
        var bf16_threshold_back = try bf16_threshold.cpu();
        defer bf16_threshold_back.deinit();
        var bf16_clip = try bf16_shifted_for_max.clip(vx.BFloat16.fromF32(-0.5), vx.BFloat16.fromF32(0.5));
        defer bf16_clip.deinit();
        var bf16_clip_back = try bf16_clip.cpu();
        defer bf16_clip_back.deinit();
        var bf16_relu6 = try bf16_shifted_for_max.relu6();
        defer bf16_relu6.deinit();
        var bf16_relu6_back = try bf16_relu6.cpu();
        defer bf16_relu6_back.deinit();
        var bf16_hardtanh = try bf16_shifted_for_max.hardtanh(vx.BFloat16.fromF32(-0.75), vx.BFloat16.fromF32(1.25));
        defer bf16_hardtanh.deinit();
        var bf16_hardtanh_back = try bf16_hardtanh.cpu();
        defer bf16_hardtanh_back.deinit();
        var bf16_clip_min_values = try vx.Array(vx.BFloat16).fullOn(allocator, &.{ 2, 3 }, vx.BFloat16.fromF32(-0.25), vx.mps(0));
        defer bf16_clip_min_values.deinit();
        var bf16_clip_max_values = try vx.Array(vx.BFloat16).fullOn(allocator, &.{ 2, 3 }, vx.BFloat16.fromF32(0.75), vx.mps(0));
        defer bf16_clip_max_values.deinit();
        var bf16_clip_array = try bf16_shifted_for_max.clipArray(bf16_clip_min_values, bf16_clip_max_values);
        defer bf16_clip_array.deinit();
        var bf16_clip_array_back = try bf16_clip_array.cpu();
        defer bf16_clip_array_back.deinit();
        var bf16_sigmoid = try bf16_shifted_for_max.sigmoid();
        defer bf16_sigmoid.deinit();
        var bf16_sigmoid_back = try bf16_sigmoid.cpu();
        defer bf16_sigmoid_back.deinit();
        var bf16_softsign = try bf16_shifted_for_max.softsign();
        defer bf16_softsign.deinit();
        var bf16_softsign_back = try bf16_softsign.cpu();
        defer bf16_softsign_back.deinit();
        bf16_activation_ok = bf16_relu.device.isMps() and bf16_relu.device_storage != null and
            bf16_threshold.device.isMps() and bf16_threshold.device_storage != null and
            bf16_clip.device.isMps() and bf16_clip.device_storage != null and
            bf16_relu6.device.isMps() and bf16_relu6.device_storage != null and
            bf16_hardtanh.device.isMps() and bf16_hardtanh.device_storage != null and
            bf16_clip_array.device.isMps() and bf16_clip_array.device_storage != null and
            bf16_sigmoid.device.isMps() and bf16_sigmoid.device_storage != null and
            bf16_softsign.device.isMps() and bf16_softsign.device_storage != null and
            closeBF16(bf16_relu_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_threshold_back.data, &.{ 0.5, 0.5, 0.5, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_clip_back.data, &.{ -0.5, -0.5, 0, 0.5, 0.5, 0.5 }, 0.125) and
            closeBF16(bf16_relu6_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_hardtanh_back.data, &.{ -0.75, -0.75, 0, 1, 1.25, 1.25 }, 0.125) and
            closeBF16(bf16_clip_array_back.data, &.{ -0.25, -0.25, 0, 0.75, 0.75, 0.75 }, 0.125) and
            closeBF16(bf16_sigmoid_back.data, &.{ sigmoid_neg2, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.e), 0.5, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0))), @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -2.0))), sigmoid_pos3 }, 0.05) and
            closeBF16(bf16_softsign_back.data, &.{ -2.0 / 3.0, -0.5, 0, 0.5, 2.0 / 3.0, 0.75 }, 0.05);

        var bf16_rsqrt = try bf16_mat_lhs.rsqrt();
        defer bf16_rsqrt.deinit();
        var bf16_rsqrt_back = try bf16_rsqrt.cpu();
        defer bf16_rsqrt_back.deinit();
        var bf16_leaky = try bf16_shifted_for_max.leakyRelu(vx.BFloat16.fromF32(0.1));
        defer bf16_leaky.deinit();
        var bf16_leaky_back = try bf16_leaky.cpu();
        defer bf16_leaky_back.deinit();
        var bf16_silu = try bf16_shifted_for_max.silu();
        defer bf16_silu.deinit();
        var bf16_silu_back = try bf16_silu.cpu();
        defer bf16_silu_back.deinit();
        var bf16_hardsigmoid = try bf16_shifted_for_max.hardsigmoid();
        defer bf16_hardsigmoid.deinit();
        var bf16_hardsigmoid_back = try bf16_hardsigmoid.cpu();
        defer bf16_hardsigmoid_back.deinit();
        var bf16_hardswish = try bf16_shifted_for_max.hardswish();
        defer bf16_hardswish.deinit();
        var bf16_hardswish_back = try bf16_hardswish.cpu();
        defer bf16_hardswish_back.deinit();
        var bf16_softshrink = try bf16_shifted_for_max.softshrink(vx.BFloat16.fromF32(0.5));
        defer bf16_softshrink.deinit();
        var bf16_softshrink_back = try bf16_softshrink.cpu();
        defer bf16_softshrink_back.deinit();
        var bf16_elu = try bf16_shifted_for_max.elu(vx.BFloat16.fromF32(1.0));
        defer bf16_elu.deinit();
        var bf16_elu_back = try bf16_elu.cpu();
        defer bf16_elu_back.deinit();
        var bf16_celu = try bf16_shifted_for_max.celu(vx.BFloat16.fromF32(2.0));
        defer bf16_celu.deinit();
        var bf16_celu_back = try bf16_celu.cpu();
        defer bf16_celu_back.deinit();
        var bf16_selu = try bf16_shifted_for_max.selu();
        defer bf16_selu.deinit();
        var bf16_selu_back = try bf16_selu.cpu();
        defer bf16_selu_back.deinit();
        var bf16_tanh = try bf16_shifted_for_max.tanh();
        defer bf16_tanh.deinit();
        var bf16_tanh_back = try bf16_tanh.cpu();
        defer bf16_tanh_back.deinit();
        var bf16_tanhshrink = try bf16_shifted_for_max.tanhshrink();
        defer bf16_tanhshrink.deinit();
        var bf16_tanhshrink_back = try bf16_tanhshrink.cpu();
        defer bf16_tanhshrink_back.deinit();
        bf16_activation_compose_ok = bf16_rsqrt.device.isMps() and bf16_rsqrt.device_storage != null and
            bf16_leaky.device.isMps() and bf16_leaky.device_storage != null and
            bf16_silu.device.isMps() and bf16_silu.device_storage != null and
            bf16_hardsigmoid.device.isMps() and bf16_hardsigmoid.device_storage != null and
            bf16_hardswish.device.isMps() and bf16_hardswish.device_storage != null and
            bf16_softshrink.device.isMps() and bf16_softshrink.device_storage != null and
            bf16_elu.device.isMps() and bf16_elu.device_storage != null and
            bf16_celu.device.isMps() and bf16_celu.device_storage != null and
            bf16_selu.device.isMps() and bf16_selu.device_storage != null and
            bf16_tanh.device.isMps() and bf16_tanh.device_storage != null and
            bf16_tanhshrink.device.isMps() and bf16_tanhshrink.device_storage != null and
            closeBF16(bf16_rsqrt_back.data, &.{ 1.0, 1.0 / std.math.sqrt(@as(f32, 2.0)), 1.0 / std.math.sqrt(@as(f32, 3.0)), 0.5, 1.0 / std.math.sqrt(@as(f32, 5.0)), 1.0 / std.math.sqrt(@as(f32, 6.0)) }, 0.05) and
            closeBF16(bf16_leaky_back.data, &.{ -0.2, -0.1, 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_silu_back.data, &.{ -2.0 * sigmoid_neg2, -1.0 / (@as(f32, 1.0) + std.math.e), 0, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0))), @as(f32, 2.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -2.0))), @as(f32, 3.0) * sigmoid_pos3 }, 0.08) and
            closeBF16(bf16_hardsigmoid_back.data, &.{ 1.0 / 6.0, 2.0 / 6.0, 0.5, 4.0 / 6.0, 5.0 / 6.0, 1.0 }, 0.05) and
            closeBF16(bf16_hardswish_back.data, &.{ -2.0 / 6.0, -2.0 / 6.0, 0, 4.0 / 6.0, 10.0 / 6.0, 3.0 }, 0.125) and
            closeBF16(bf16_softshrink_back.data, &.{ -1.5, -0.5, 0, 0.5, 1.5, 2.5 }, 0.125) and
            closeBF16(bf16_elu_back.data, &.{ std.math.exp(@as(f32, -2.0)) - 1.0, std.math.exp(@as(f32, -1.0)) - 1.0, 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_celu_back.data, &.{ 2.0 * (std.math.exp(@as(f32, -1.0)) - 1.0), 2.0 * (std.math.exp(@as(f32, -0.5)) - 1.0), 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_selu_back.data, &.{ selu_neg2, selu_neg1, 0, selu_scale, 2.0 * selu_scale, 3.0 * selu_scale }, 0.125) and
            closeBF16(bf16_tanh_back.data, &.{ std.math.tanh(@as(f32, -2.0)), std.math.tanh(@as(f32, -1.0)), 0, std.math.tanh(@as(f32, 1.0)), std.math.tanh(@as(f32, 2.0)), std.math.tanh(@as(f32, 3.0)) }, 0.125) and
            closeBF16(bf16_tanhshrink_back.data, &.{ -2.0 - std.math.tanh(@as(f32, -2.0)), -1.0 - std.math.tanh(@as(f32, -1.0)), 0, 1.0 - std.math.tanh(@as(f32, 1.0)), 2.0 - std.math.tanh(@as(f32, 2.0)), 3.0 - std.math.tanh(@as(f32, 3.0)) }, 0.125);

        var bf16_pow_zero = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(0));
        defer bf16_pow_zero.deinit();
        var bf16_pow_zero_back = try bf16_pow_zero.cpu();
        defer bf16_pow_zero_back.deinit();
        var bf16_pow_one = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(1));
        defer bf16_pow_one.deinit();
        var bf16_pow_one_back = try bf16_pow_one.cpu();
        defer bf16_pow_one_back.deinit();
        var bf16_pow_recip = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(-1));
        defer bf16_pow_recip.deinit();
        var bf16_pow_recip_back = try bf16_pow_recip.cpu();
        defer bf16_pow_recip_back.deinit();
        var bf16_pow_sqrt = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(0.5));
        defer bf16_pow_sqrt.deinit();
        var bf16_pow_sqrt_back = try bf16_pow_sqrt.cpu();
        defer bf16_pow_sqrt_back.deinit();
        var bf16_pow_rsqrt = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(-0.5));
        defer bf16_pow_rsqrt.deinit();
        var bf16_pow_rsqrt_back = try bf16_pow_rsqrt.cpu();
        defer bf16_pow_rsqrt_back.deinit();
        var bf16_pow_square = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(2));
        defer bf16_pow_square.deinit();
        var bf16_pow_square_back = try bf16_pow_square.cpu();
        defer bf16_pow_square_back.deinit();
        var bf16_pow_cube = try bf16_mat_lhs.powScalar(vx.BFloat16.fromF32(3));
        defer bf16_pow_cube.deinit();
        var bf16_pow_cube_back = try bf16_pow_cube.cpu();
        defer bf16_pow_cube_back.deinit();
        bf16_pow_ok = bf16_pow_zero.device.isMps() and bf16_pow_zero.device_storage != null and
            bf16_pow_one.device.isMps() and bf16_pow_one.device_storage != null and
            bf16_pow_recip.device.isMps() and bf16_pow_recip.device_storage != null and
            bf16_pow_sqrt.device.isMps() and bf16_pow_sqrt.device_storage != null and
            bf16_pow_rsqrt.device.isMps() and bf16_pow_rsqrt.device_storage != null and
            bf16_pow_square.device.isMps() and bf16_pow_square.device_storage != null and
            bf16_pow_cube.device.isMps() and bf16_pow_cube.device_storage != null and
            closeBF16(bf16_pow_zero_back.data, &.{ 1, 1, 1, 1, 1, 1 }, 0.05) and
            closeBF16(bf16_pow_one_back.data, &.{ 1, 2, 3, 4, 5, 6 }, 0.125) and
            closeBF16(bf16_pow_recip_back.data, &.{ 1, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0 }, 0.05) and
            closeBF16(bf16_pow_sqrt_back.data, &.{ 1, std.math.sqrt(@as(f32, 2)), std.math.sqrt(@as(f32, 3)), 2, std.math.sqrt(@as(f32, 5)), std.math.sqrt(@as(f32, 6)) }, 0.08) and
            closeBF16(bf16_pow_rsqrt_back.data, &.{ 1, 1.0 / std.math.sqrt(@as(f32, 2)), 1.0 / std.math.sqrt(@as(f32, 3)), 0.5, 1.0 / std.math.sqrt(@as(f32, 5)), 1.0 / std.math.sqrt(@as(f32, 6)) }, 0.05) and
            closeBF16(bf16_pow_square_back.data, &.{ 1, 4, 9, 16, 25, 36 }, 0.5) and
            closeBF16(bf16_pow_cube_back.data, &.{ 1, 8, 27, 64, 125, 216 }, 1.0);

        var bf16_ternary_base = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(3), vx.BFloat16.fromF32(4) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_ternary_base.deinit();
        var bf16_ternary_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_ternary_lhs.deinit();
        var bf16_ternary_rhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(1), vx.BFloat16.fromF32(1.5), vx.BFloat16.fromF32(2), vx.BFloat16.fromF32(2.5) }, &.{ 2, 2 }, vx.mps(0));
        defer bf16_ternary_rhs.deinit();
        var bf16_addcmul = try bf16_ternary_base.addcmul(bf16_ternary_lhs, bf16_ternary_rhs, vx.BFloat16.fromF32(2.0));
        defer bf16_addcmul.deinit();
        var bf16_addcmul_back = try bf16_addcmul.cpu();
        defer bf16_addcmul_back.deinit();
        var bf16_addcdiv = try bf16_ternary_base.addcdiv(bf16_ternary_lhs, bf16_ternary_rhs, vx.BFloat16.fromF32(0.5));
        defer bf16_addcdiv.deinit();
        var bf16_addcdiv_back = try bf16_addcdiv.cpu();
        defer bf16_addcdiv_back.deinit();
        var bf16_lerp_scalar = try bf16_ternary_base.lerpScalar(bf16_ternary_rhs, vx.BFloat16.fromF32(0.5));
        defer bf16_lerp_scalar.deinit();
        var bf16_lerp_scalar_back = try bf16_lerp_scalar.cpu();
        defer bf16_lerp_scalar_back.deinit();
        var bf16_lerp_array = try bf16_ternary_base.lerp(bf16_ternary_lhs, bf16_ternary_rhs);
        defer bf16_lerp_array.deinit();
        var bf16_lerp_array_back = try bf16_lerp_array.cpu();
        defer bf16_lerp_array_back.deinit();
        const bf16_ternary_ok = bf16_addcmul.device.isMps() and bf16_addcmul.device_storage != null and
            bf16_addcdiv.device.isMps() and bf16_addcdiv.device_storage != null and
            bf16_lerp_scalar.device.isMps() and bf16_lerp_scalar.device_storage != null and
            bf16_lerp_array.device.isMps() and bf16_lerp_array.device_storage != null and
            closeBF16(bf16_addcmul_back.data, &.{ 3, 5, 7, 9 }, 0.125) and
            closeBF16(bf16_addcdiv_back.data, &.{ 1.5, 2.3333333, 3.25, 4.2 }, 0.125) and
            closeBF16(bf16_lerp_scalar_back.data, &.{ 1, 1.75, 2.5, 3.25 }, 0.125) and
            closeBF16(bf16_lerp_array_back.data, &.{ 1, 0.5, -1, -3.5 }, 0.125);

        var bf16_loss_target = try vx.Array(vx.BFloat16).zerosOn(allocator, &.{ 2, 3 }, vx.mps(0));
        defer bf16_loss_target.deinit();
        var bf16_mse = try bf16_shifted_for_max.mseLoss(bf16_loss_target, .none);
        defer bf16_mse.deinit();
        var bf16_mse_back = try bf16_mse.cpu();
        defer bf16_mse_back.deinit();
        var bf16_l1 = try bf16_shifted_for_max.l1Loss(bf16_loss_target, .none);
        defer bf16_l1.deinit();
        var bf16_l1_back = try bf16_l1.cpu();
        defer bf16_l1_back.deinit();
        var bf16_smooth_l1 = try bf16_shifted_for_max.smoothL1Loss(bf16_loss_target, vx.BFloat16.fromF32(1), .none);
        defer bf16_smooth_l1.deinit();
        var bf16_smooth_l1_back = try bf16_smooth_l1.cpu();
        defer bf16_smooth_l1_back.deinit();
        var bf16_huber = try bf16_shifted_for_max.huberLoss(bf16_loss_target, vx.BFloat16.fromF32(1), .none);
        defer bf16_huber.deinit();
        var bf16_huber_back = try bf16_huber.cpu();
        defer bf16_huber_back.deinit();
        bf16_loss_ok = bf16_mse.device.isMps() and bf16_mse.device_storage != null and
            bf16_l1.device.isMps() and bf16_l1.device_storage != null and
            bf16_smooth_l1.device.isMps() and bf16_smooth_l1.device_storage != null and
            bf16_huber.device.isMps() and bf16_huber.device_storage != null and
            closeBF16(bf16_mse_back.data, &.{ 4, 1, 0, 1, 4, 9 }, 0.125) and
            closeBF16(bf16_l1_back.data, &.{ 2, 1, 0, 1, 2, 3 }, 0.125) and
            closeBF16(bf16_smooth_l1_back.data, &.{ 1.5, 0.5, 0, 0.5, 1.5, 2.5 }, 0.125) and
            closeBF16(bf16_huber_back.data, &.{ 1.5, 0.5, 0, 0.5, 1.5, 2.5 }, 0.125);

        var bf16_softmax_row = try bf16_mat_lhs.softmax(1);
        defer bf16_softmax_row.deinit();
        var bf16_softmax_row_back = try bf16_softmax_row.cpu();
        defer bf16_softmax_row_back.deinit();
        var bf16_softmax_col = try bf16_mat_lhs.softmax(0);
        defer bf16_softmax_col.deinit();
        var bf16_softmax_col_back = try bf16_softmax_col.cpu();
        defer bf16_softmax_col_back.deinit();
        const bf16_row_denom = std.math.exp(@as(f32, -2)) + std.math.exp(@as(f32, -1)) + 1.0;
        const bf16_col_denom = std.math.exp(@as(f32, -3)) + 1.0;
        bf16_softmax_ok = bf16_softmax_row.device.isMps() and bf16_softmax_row.device_storage != null and
            bf16_softmax_col.device.isMps() and bf16_softmax_col.device_storage != null and
            closeBF16(bf16_softmax_row_back.data, &.{ std.math.exp(@as(f32, -2)) / bf16_row_denom, std.math.exp(@as(f32, -1)) / bf16_row_denom, 1.0 / bf16_row_denom, std.math.exp(@as(f32, -2)) / bf16_row_denom, std.math.exp(@as(f32, -1)) / bf16_row_denom, 1.0 / bf16_row_denom }, 0.02) and
            closeBF16(bf16_softmax_col_back.data, &.{ std.math.exp(@as(f32, -3)) / bf16_col_denom, std.math.exp(@as(f32, -3)) / bf16_col_denom, std.math.exp(@as(f32, -3)) / bf16_col_denom, 1.0 / bf16_col_denom, 1.0 / bf16_col_denom, 1.0 / bf16_col_denom }, 0.02);

        var bf16_log_softmax_row = try bf16_mat_lhs.logSoftmax(1);
        defer bf16_log_softmax_row.deinit();
        var bf16_log_softmax_row_back = try bf16_log_softmax_row.cpu();
        defer bf16_log_softmax_row_back.deinit();
        var bf16_log_softmax_col = try bf16_mat_lhs.logSoftmax(0);
        defer bf16_log_softmax_col.deinit();
        var bf16_log_softmax_col_back = try bf16_log_softmax_col.cpu();
        defer bf16_log_softmax_col_back.deinit();
        const bf16_row_log_denom = std.math.log(f32, std.math.e, bf16_row_denom);
        const bf16_col_log_denom = std.math.log(f32, std.math.e, bf16_col_denom);
        bf16_log_softmax_ok = bf16_log_softmax_row.device.isMps() and bf16_log_softmax_row.device_storage != null and
            bf16_log_softmax_col.device.isMps() and bf16_log_softmax_col.device_storage != null and
            closeBF16(bf16_log_softmax_row_back.data, &.{ -2.0 - bf16_row_log_denom, -1.0 - bf16_row_log_denom, -bf16_row_log_denom, -2.0 - bf16_row_log_denom, -1.0 - bf16_row_log_denom, -bf16_row_log_denom }, 0.04) and
            closeBF16(bf16_log_softmax_col_back.data, &.{ -3.0 - bf16_col_log_denom, -3.0 - bf16_col_log_denom, -3.0 - bf16_col_log_denom, -bf16_col_log_denom, -bf16_col_log_denom, -bf16_col_log_denom }, 0.04);

        var bf16_softmin_row = try bf16_mat_lhs.softmin(1);
        defer bf16_softmin_row.deinit();
        var bf16_softmin_row_back = try bf16_softmin_row.cpu();
        defer bf16_softmin_row_back.deinit();
        var bf16_softmin_col = try bf16_mat_lhs.softmin(0);
        defer bf16_softmin_col.deinit();
        var bf16_softmin_col_back = try bf16_softmin_col.cpu();
        defer bf16_softmin_col_back.deinit();
        const bf16_softmin_row_denom = 1.0 + std.math.exp(@as(f32, -1)) + std.math.exp(@as(f32, -2));
        const bf16_softmin_col_denom = 1.0 + std.math.exp(@as(f32, -3));
        bf16_softmin_ok = bf16_softmin_row.device.isMps() and bf16_softmin_row.device_storage != null and
            bf16_softmin_col.device.isMps() and bf16_softmin_col.device_storage != null and
            closeBF16(bf16_softmin_row_back.data, &.{ 1.0 / bf16_softmin_row_denom, std.math.exp(@as(f32, -1)) / bf16_softmin_row_denom, std.math.exp(@as(f32, -2)) / bf16_softmin_row_denom, 1.0 / bf16_softmin_row_denom, std.math.exp(@as(f32, -1)) / bf16_softmin_row_denom, std.math.exp(@as(f32, -2)) / bf16_softmin_row_denom }, 0.02) and
            closeBF16(bf16_softmin_col_back.data, &.{ 1.0 / bf16_softmin_col_denom, 1.0 / bf16_softmin_col_denom, 1.0 / bf16_softmin_col_denom, std.math.exp(@as(f32, -3)) / bf16_softmin_col_denom, std.math.exp(@as(f32, -3)) / bf16_softmin_col_denom, std.math.exp(@as(f32, -3)) / bf16_softmin_col_denom }, 0.02);

        var bf16_log_softmin_row = try bf16_mat_lhs.logSoftmin(1);
        defer bf16_log_softmin_row.deinit();
        var bf16_log_softmin_row_back = try bf16_log_softmin_row.cpu();
        defer bf16_log_softmin_row_back.deinit();
        var bf16_log_softmin_col = try bf16_mat_lhs.logSoftmin(0);
        defer bf16_log_softmin_col.deinit();
        var bf16_log_softmin_col_back = try bf16_log_softmin_col.cpu();
        defer bf16_log_softmin_col_back.deinit();
        const bf16_softmin_row_log_denom = std.math.log(f32, std.math.e, bf16_softmin_row_denom);
        const bf16_softmin_col_log_denom = std.math.log(f32, std.math.e, bf16_softmin_col_denom);
        bf16_log_softmin_ok = bf16_log_softmin_row.device.isMps() and bf16_log_softmin_row.device_storage != null and
            bf16_log_softmin_col.device.isMps() and bf16_log_softmin_col.device_storage != null and
            closeBF16(bf16_log_softmin_row_back.data, &.{ -bf16_softmin_row_log_denom, -1.0 - bf16_softmin_row_log_denom, -2.0 - bf16_softmin_row_log_denom, -bf16_softmin_row_log_denom, -1.0 - bf16_softmin_row_log_denom, -2.0 - bf16_softmin_row_log_denom }, 0.04) and
            closeBF16(bf16_log_softmin_col_back.data, &.{ -bf16_softmin_col_log_denom, -bf16_softmin_col_log_denom, -bf16_softmin_col_log_denom, -3.0 - bf16_softmin_col_log_denom, -3.0 - bf16_softmin_col_log_denom, -3.0 - bf16_softmin_col_log_denom }, 0.04);

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

        var mat_chain_add = try mat_out.add(mat_addend);
        defer mat_chain_add.deinit();
        var mat_chain_add_back = try mat_chain_add.cpu();
        defer mat_chain_add_back.deinit();
        var mat_chain_radd = try mat_addend.add(mat_out);
        defer mat_chain_radd.deinit();
        var mat_chain_radd_back = try mat_chain_radd.cpu();
        defer mat_chain_radd_back.deinit();
        var mat_chain_rsub = try mat_addend.sub(mat_out);
        defer mat_chain_rsub.deinit();
        var mat_chain_rsub_back = try mat_chain_rsub.cpu();
        defer mat_chain_rsub_back.deinit();
        var mat_chain_sqrt = try mat_chain_add.sqrt();
        defer mat_chain_sqrt.deinit();
        var mat_chain_sqrt_back = try mat_chain_sqrt.cpu();
        defer mat_chain_sqrt_back.deinit();
        var mat_chain_exp_input_shifted = try mat_chain_rsub.addScalar(60.0);
        defer mat_chain_exp_input_shifted.deinit();
        var mat_chain_exp = try mat_chain_exp_input_shifted.exp();
        defer mat_chain_exp.deinit();
        var mat_chain_exp_back = try mat_chain_exp.cpu();
        defer mat_chain_exp_back.deinit();
        matmul_chain_ok = mat_chain_add.device.isMps() and mat_chain_add.device_storage != null and
            mat_chain_radd.device.isMps() and mat_chain_radd.device_storage != null and
            mat_chain_rsub.device.isMps() and mat_chain_rsub.device_storage != null and
            mat_chain_sqrt.device.isMps() and mat_chain_sqrt.device_storage != null and
            mat_chain_exp_input_shifted.device.isMps() and mat_chain_exp_input_shifted.device_storage != null and
            mat_chain_exp.device.isMps() and mat_chain_exp.device_storage != null and
            equalF32(mat_chain_add_back.data, &.{ 59, 63, 141, 152 }) and
            equalF32(mat_chain_radd_back.data, &.{ 59, 63, 141, 152 }) and
            equalF32(mat_chain_rsub_back.data, &.{ -57, -65, -137, -156 }) and
            closeF32(mat_chain_sqrt_back.data, &.{ std.math.sqrt(@as(f32, 59)), std.math.sqrt(@as(f32, 63)), std.math.sqrt(@as(f32, 141)), std.math.sqrt(@as(f32, 152)) }, 0.001) and
            closeF32(mat_chain_exp_back.data, &.{ std.math.exp(@as(f32, 3)), std.math.exp(@as(f32, -5)), std.math.exp(@as(f32, -77)), std.math.exp(@as(f32, -96)) }, 0.001);

        var vec_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3 }, &.{3}, vx.mps(0));
        defer vec_rhs.deinit();
        var matvec = try mat_lhs.matvec(vec_rhs);
        defer matvec.deinit();
        var matvec_back = try matvec.cpu();
        defer matvec_back.deinit();
        var vec_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2 }, &.{2}, vx.mps(0));
        defer vec_lhs.deinit();
        var vecmat = try vec_lhs.matmul(mat_lhs);
        defer vecmat.deinit();
        var vecmat_back = try vecmat.cpu();
        defer vecmat_back.deinit();
        var dot = try vec_rhs.dot(vec_rhs);
        defer dot.deinit();
        var dot_back = try dot.cpu();
        defer dot_back.deinit();
        vector_matmul_ok = matvec.device.isMps() and matvec.device_storage != null and
            vecmat.device.isMps() and vecmat.device_storage != null and
            dot.device.isMps() and dot.device_storage != null and
            equalF32(matvec_back.data, &.{ 14, 32 }) and
            equalF32(vecmat_back.data, &.{ 9, 12, 15 }) and
            equalF32(dot_back.data, &.{14});

        var norm = try mat_lhs.norm(2.0, 1, false);
        defer norm.deinit();
        var norm_back = try norm.cpu();
        defer norm_back.deinit();
        var normalized = try mat_lhs.normalize(2.0, 1, 0.001);
        defer normalized.deinit();
        var normalized_back = try normalized.cpu();
        defer normalized_back.deinit();
        var metric_other = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 0, 0, 1, 1, 2 }, &.{ 2, 3 }, vx.mps(0));
        defer metric_other.deinit();
        var cosine = try mat_lhs.cosineSimilarity(metric_other, 1, 0.001, false);
        defer cosine.deinit();
        var cosine_back = try cosine.cpu();
        defer cosine_back.deinit();
        var distance = try mat_lhs.pairwiseDistance(metric_other, 2.0, 1, false);
        defer distance.deinit();
        var distance_back = try distance.cpu();
        defer distance_back.deinit();
        const norm0 = std.math.sqrt(@as(f32, 14));
        const norm1 = std.math.sqrt(@as(f32, 77));
        const other_norm0 = @as(f32, 1);
        const other_norm1 = std.math.sqrt(@as(f32, 6));
        metric_ok = norm.device.isMps() and norm.device_storage != null and
            normalized.device.isMps() and normalized.device_storage != null and
            cosine.device.isMps() and cosine.device_storage != null and
            distance.device.isMps() and distance.device_storage != null and
            closeF32(norm_back.data, &.{ norm0, norm1 }, 0.001) and
            closeF32(normalized_back.data, &.{ 1.0 / norm0, 2.0 / norm0, 3.0 / norm0, 4.0 / norm1, 5.0 / norm1, 6.0 / norm1 }, 0.001) and
            closeF32(cosine_back.data, &.{ 1.0 / (norm0 * other_norm0), 21.0 / (norm1 * other_norm1) }, 0.001) and
            closeF32(distance_back.data, &.{ std.math.sqrt(@as(f32, 13)), std.math.sqrt(@as(f32, 41)) }, 0.001);

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
        var row_sub = try mat_lhs.sub(row_bias);
        defer row_sub.deinit();
        var row_sub_back = try row_sub.cpu();
        defer row_sub_back.deinit();
        var row_mul = try mat_lhs.mul(row_bias);
        defer row_mul.deinit();
        var row_mul_back = try row_mul.cpu();
        defer row_mul_back.deinit();
        var row_div = try mat_lhs.div(row_bias);
        defer row_div.deinit();
        var row_div_back = try row_div.cpu();
        defer row_div_back.deinit();
        var col_bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 100, 200 }, &.{2}, vx.mps(0));
        defer col_bias.deinit();
        var col_added = try mat_lhs.add(col_bias);
        defer col_added.deinit();
        var col_added_back = try col_added.cpu();
        defer col_added_back.deinit();
        var col_sub = try mat_lhs.sub(col_bias);
        defer col_sub.deinit();
        var col_sub_back = try col_sub.cpu();
        defer col_sub_back.deinit();
        var col_mul = try mat_lhs.mul(col_bias);
        defer col_mul.deinit();
        var col_mul_back = try col_mul.cpu();
        defer col_mul_back.deinit();
        var col_div = try mat_lhs.div(col_bias);
        defer col_div.deinit();
        var col_div_back = try col_div.cpu();
        defer col_div_back.deinit();
        broadcast_ok = row_added.device.isMps() and row_added.device_storage != null and
            row_sub.device.isMps() and row_sub.device_storage != null and
            row_mul.device.isMps() and row_mul.device_storage != null and
            row_div.device.isMps() and row_div.device_storage != null and
            col_added.device.isMps() and col_added.device_storage != null and
            col_sub.device.isMps() and col_sub.device_storage != null and
            col_mul.device.isMps() and col_mul.device_storage != null and
            col_div.device.isMps() and col_div.device_storage != null and
            equalF32(row_added_back.data, &.{ 11, 22, 33, 14, 25, 36 }) and
            equalF32(row_sub_back.data, &.{ -9, -18, -27, -6, -15, -24 }) and
            equalF32(row_mul_back.data, &.{ 10, 40, 90, 40, 100, 180 }) and
            closeF32(row_div_back.data, &.{ 0.1, 0.1, 0.1, 0.4, 0.25, 0.2 }, 0.0001) and
            equalF32(col_added_back.data, &.{ 101, 102, 103, 204, 205, 206 }) and
            equalF32(col_sub_back.data, &.{ -99, -98, -97, -196, -195, -194 }) and
            equalF32(col_mul_back.data, &.{ 100, 200, 300, 800, 1000, 1200 }) and
            closeF32(col_div_back.data, &.{ 0.01, 0.02, 0.03, 0.02, 0.025, 0.03 }, 0.0001);

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
        var flat_sum = try mat_lhs.sum(null, false);
        defer flat_sum.deinit();
        var flat_sum_back = try flat_sum.cpu();
        defer flat_sum_back.deinit();
        var flat_prod_keep = try mat_lhs.prod(null, true);
        defer flat_prod_keep.deinit();
        var flat_prod_keep_back = try flat_prod_keep.cpu();
        defer flat_prod_keep_back.deinit();
        var flat_min = try mat_lhs.min(null, false);
        defer flat_min.deinit();
        var flat_min_back = try flat_min.cpu();
        defer flat_min_back.deinit();
        var flat_max_keep = try mat_lhs.max(null, true);
        defer flat_max_keep.deinit();
        var flat_max_keep_back = try flat_max_keep.cpu();
        defer flat_max_keep_back.deinit();
        var all_axes_sum = try mat_lhs.sumAxes(&.{ 0, 1 }, false);
        defer all_axes_sum.deinit();
        var all_axes_sum_back = try all_axes_sum.cpu();
        defer all_axes_sum_back.deinit();
        var all_axes_max_keep = try mat_lhs.maxAxes(&.{ 0, 1 }, true);
        defer all_axes_max_keep.deinit();
        var all_axes_max_keep_back = try all_axes_max_keep.cpu();
        defer all_axes_max_keep_back.deinit();
        var row_ptp = try mat_lhs.ptp(1, false);
        defer row_ptp.deinit();
        var row_ptp_back = try row_ptp.cpu();
        defer row_ptp_back.deinit();
        var flat_ptp_keep = try mat_lhs.ptp(null, true);
        defer flat_ptp_keep.deinit();
        var flat_ptp_keep_back = try flat_ptp_keep.cpu();
        defer flat_ptp_keep_back.deinit();
        var all_axes_ptp = try mat_lhs.ptpAxes(&.{ 0, 1 }, false);
        defer all_axes_ptp.deinit();
        var all_axes_ptp_back = try all_axes_ptp.cpu();
        defer all_axes_ptp_back.deinit();
        reduction_ok = row_sum.device.isMps() and row_sum.device_storage != null and
            col_max.device.isMps() and col_max.device_storage != null and
            row_prod_keep.device.isMps() and row_prod_keep.device_storage != null and
            flat_sum.device.isMps() and flat_sum.device_storage != null and
            flat_prod_keep.device.isMps() and flat_prod_keep.device_storage != null and
            flat_min.device.isMps() and flat_min.device_storage != null and
            flat_max_keep.device.isMps() and flat_max_keep.device_storage != null and
            all_axes_sum.device.isMps() and all_axes_sum.device_storage != null and
            all_axes_max_keep.device.isMps() and all_axes_max_keep.device_storage != null and
            row_ptp.device.isMps() and row_ptp.device_storage != null and
            flat_ptp_keep.device.isMps() and flat_ptp_keep.device_storage != null and
            all_axes_ptp.device.isMps() and all_axes_ptp.device_storage != null and
            equalF32(row_sum_back.data, &.{ 6, 15 }) and
            equalF32(col_max_back.data, &.{ 4, 5, 6 }) and
            std.mem.eql(usize, row_prod_keep_back.shape, &.{ 2, 1 }) and
            equalF32(row_prod_keep_back.data, &.{ 6, 120 }) and
            std.mem.eql(usize, flat_sum_back.shape, &.{}) and
            equalF32(flat_sum_back.data, &.{21}) and
            std.mem.eql(usize, flat_prod_keep_back.shape, &.{ 1, 1 }) and
            equalF32(flat_prod_keep_back.data, &.{720}) and
            std.mem.eql(usize, flat_min_back.shape, &.{}) and
            equalF32(flat_min_back.data, &.{1}) and
            std.mem.eql(usize, flat_max_keep_back.shape, &.{ 1, 1 }) and
            equalF32(flat_max_keep_back.data, &.{6}) and
            std.mem.eql(usize, all_axes_sum_back.shape, &.{}) and
            equalF32(all_axes_sum_back.data, &.{21}) and
            std.mem.eql(usize, all_axes_max_keep_back.shape, &.{ 1, 1 }) and
            equalF32(all_axes_max_keep_back.data, &.{6}) and
            std.mem.eql(usize, row_ptp_back.shape, &.{2}) and
            equalF32(row_ptp_back.data, &.{ 2, 2 }) and
            std.mem.eql(usize, flat_ptp_keep_back.shape, &.{ 1, 1 }) and
            equalF32(flat_ptp_keep_back.data, &.{5}) and
            std.mem.eql(usize, all_axes_ptp_back.shape, &.{}) and
            equalF32(all_axes_ptp_back.data, &.{5});

        var row_mean = try mat_lhs.mean(1, false);
        defer row_mean.deinit();
        var row_mean_back = try row_mean.cpu();
        defer row_mean_back.deinit();
        var col_mean_keep = try mat_lhs.mean(0, true);
        defer col_mean_keep.deinit();
        var col_mean_keep_back = try col_mean_keep.cpu();
        defer col_mean_keep_back.deinit();
        var flat_var = try mat_lhs.variance(null, false, 0.0);
        defer flat_var.deinit();
        var flat_var_back = try flat_var.cpu();
        defer flat_var_back.deinit();
        var row_var = try mat_lhs.variance(1, false, 0.0);
        defer row_var.deinit();
        var row_var_back = try row_var.cpu();
        defer row_var_back.deinit();
        var col_std_keep = try mat_lhs.stddev(0, true, 0.0);
        defer col_std_keep.deinit();
        var col_std_keep_back = try col_std_keep.cpu();
        defer col_std_keep_back.deinit();
        stats_ok = f16_stats_ok and bf16_stats_ok and
            row_mean.device.isMps() and row_mean.device_storage != null and
            col_mean_keep.device.isMps() and col_mean_keep.device_storage != null and
            flat_var.device.isMps() and flat_var.device_storage != null and
            row_var.device.isMps() and row_var.device_storage != null and
            col_std_keep.device.isMps() and col_std_keep.device_storage != null and
            closeF32(row_mean_back.data, &.{ 2, 5 }, 0.001) and
            closeF32(col_mean_keep_back.data, &.{ 2.5, 3.5, 4.5 }, 0.001) and
            closeF32(flat_var_back.data, &.{35.0 / 12.0}, 0.001) and
            closeF32(row_var_back.data, &.{ 2.0 / 3.0, 2.0 / 3.0 }, 0.001) and
            closeF32(col_std_keep_back.data, &.{ 1.5, 1.5, 1.5 }, 0.001);

        var shifted_for_max = try mat_lhs.subScalar(3.0);
        defer shifted_for_max.deinit();
        var scaled_for_max = try mat_lhs.mulScalar(0.1);
        defer scaled_for_max.deinit();
        var maximum = try shifted_for_max.maximum(scaled_for_max);
        defer maximum.deinit();
        var maximum_back = try maximum.cpu();
        defer maximum_back.deinit();
        var minimum = try shifted_for_max.minimum(scaled_for_max);
        defer minimum.deinit();
        var minimum_back = try minimum.cpu();
        defer minimum_back.deinit();
        var maximum_scalar = try shifted_for_max.maximumScalar(0.0);
        defer maximum_scalar.deinit();
        var maximum_scalar_back = try maximum_scalar.cpu();
        defer maximum_scalar_back.deinit();
        var minimum_scalar = try shifted_for_max.minimumScalar(0.0);
        defer minimum_scalar.deinit();
        var minimum_scalar_back = try minimum_scalar.cpu();
        defer minimum_scalar_back.deinit();
        minmax_ok = f16_minmax_ok and bf16_minmax_ok and
            maximum.device.isMps() and maximum.device_storage != null and
            minimum.device.isMps() and minimum.device_storage != null and
            maximum_scalar.device.isMps() and maximum_scalar.device_storage != null and
            minimum_scalar.device.isMps() and minimum_scalar.device_storage != null and
            closeF32(maximum_back.data, &.{ 0.1, 0.2, 0.3, 1.0, 2.0, 3.0 }, 0.001) and
            closeF32(minimum_back.data, &.{ -2.0, -1.0, 0.0, 0.4, 0.5, 0.6 }, 0.001) and
            closeF32(maximum_scalar_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.001) and
            closeF32(minimum_scalar_back.data, &.{ -2, -1, 0, 0, 0, 0 }, 0.001);

        var relu = try shifted_for_max.relu();
        defer relu.deinit();
        var relu_back = try relu.cpu();
        defer relu_back.deinit();
        var threshold = try shifted_for_max.threshold(0.5, 0.5);
        defer threshold.deinit();
        var threshold_back = try threshold.cpu();
        defer threshold_back.deinit();
        var clip = try shifted_for_max.clip(-0.5, 0.5);
        defer clip.deinit();
        var clip_back = try clip.cpu();
        defer clip_back.deinit();
        var relu6 = try shifted_for_max.relu6();
        defer relu6.deinit();
        var relu6_back = try relu6.cpu();
        defer relu6_back.deinit();
        var hardtanh = try shifted_for_max.hardtanh(-0.75, 1.25);
        defer hardtanh.deinit();
        var hardtanh_back = try hardtanh.cpu();
        defer hardtanh_back.deinit();
        var clip_min_values = try vx.Array(f32).fullOn(allocator, &.{ 2, 3 }, -0.25, vx.mps(0));
        defer clip_min_values.deinit();
        var clip_max_values = try vx.Array(f32).fullOn(allocator, &.{ 2, 3 }, 0.75, vx.mps(0));
        defer clip_max_values.deinit();
        var clip_array = try shifted_for_max.clipArray(clip_min_values, clip_max_values);
        defer clip_array.deinit();
        var clip_array_back = try clip_array.cpu();
        defer clip_array_back.deinit();
        var sigmoid = try shifted_for_max.sigmoid();
        defer sigmoid.deinit();
        var sigmoid_back = try sigmoid.cpu();
        defer sigmoid_back.deinit();
        var softsign = try shifted_for_max.softsign();
        defer softsign.deinit();
        var softsign_back = try softsign.cpu();
        defer softsign_back.deinit();
        activation_ok = f16_activation_ok and bf16_activation_ok and
            relu.device.isMps() and relu.device_storage != null and
            threshold.device.isMps() and threshold.device_storage != null and
            clip.device.isMps() and clip.device_storage != null and
            relu6.device.isMps() and relu6.device_storage != null and
            hardtanh.device.isMps() and hardtanh.device_storage != null and
            clip_array.device.isMps() and clip_array.device_storage != null and
            sigmoid.device.isMps() and sigmoid.device_storage != null and
            softsign.device.isMps() and softsign.device_storage != null and
            closeF32(relu_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.001) and
            closeF32(threshold_back.data, &.{ 0.5, 0.5, 0.5, 1, 2, 3 }, 0.001) and
            closeF32(clip_back.data, &.{ -0.5, -0.5, 0, 0.5, 0.5, 0.5 }, 0.001) and
            closeF32(relu6_back.data, &.{ 0, 0, 0, 1, 2, 3 }, 0.001) and
            closeF32(hardtanh_back.data, &.{ -0.75, -0.75, 0, 1, 1.25, 1.25 }, 0.001) and
            closeF32(clip_array_back.data, &.{ -0.25, -0.25, 0, 0.75, 0.75, 0.75 }, 0.001) and
            closeF32(sigmoid_back.data, &.{ sigmoid_neg2, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.e), 0.5, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0))), @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -2.0))), sigmoid_pos3 }, 0.01) and
            closeF32(softsign_back.data, &.{ -2.0 / 3.0, -0.5, 0, 0.5, 2.0 / 3.0, 0.75 }, 0.01);

        var rsqrt = try mat_lhs.rsqrt();
        defer rsqrt.deinit();
        var rsqrt_back = try rsqrt.cpu();
        defer rsqrt_back.deinit();
        var leaky = try shifted_for_max.leakyRelu(0.1);
        defer leaky.deinit();
        var leaky_back = try leaky.cpu();
        defer leaky_back.deinit();
        var silu = try shifted_for_max.silu();
        defer silu.deinit();
        var silu_back = try silu.cpu();
        defer silu_back.deinit();
        var hardsigmoid = try shifted_for_max.hardsigmoid();
        defer hardsigmoid.deinit();
        var hardsigmoid_back = try hardsigmoid.cpu();
        defer hardsigmoid_back.deinit();
        var hardswish = try shifted_for_max.hardswish();
        defer hardswish.deinit();
        var hardswish_back = try hardswish.cpu();
        defer hardswish_back.deinit();
        var softshrink = try shifted_for_max.softshrink(0.5);
        defer softshrink.deinit();
        var softshrink_back = try softshrink.cpu();
        defer softshrink_back.deinit();
        var elu = try shifted_for_max.elu(1.0);
        defer elu.deinit();
        var elu_back = try elu.cpu();
        defer elu_back.deinit();
        var celu = try shifted_for_max.celu(2.0);
        defer celu.deinit();
        var celu_back = try celu.cpu();
        defer celu_back.deinit();
        var selu = try shifted_for_max.selu();
        defer selu.deinit();
        var selu_back = try selu.cpu();
        defer selu_back.deinit();
        var tanh = try shifted_for_max.tanh();
        defer tanh.deinit();
        var tanh_back = try tanh.cpu();
        defer tanh_back.deinit();
        var tanhshrink = try shifted_for_max.tanhshrink();
        defer tanhshrink.deinit();
        var tanhshrink_back = try tanhshrink.cpu();
        defer tanhshrink_back.deinit();
        activation_compose_ok = f16_activation_compose_ok and bf16_activation_compose_ok and
            rsqrt.device.isMps() and rsqrt.device_storage != null and
            leaky.device.isMps() and leaky.device_storage != null and
            silu.device.isMps() and silu.device_storage != null and
            hardsigmoid.device.isMps() and hardsigmoid.device_storage != null and
            hardswish.device.isMps() and hardswish.device_storage != null and
            softshrink.device.isMps() and softshrink.device_storage != null and
            elu.device.isMps() and elu.device_storage != null and
            celu.device.isMps() and celu.device_storage != null and
            selu.device.isMps() and selu.device_storage != null and
            tanh.device.isMps() and tanh.device_storage != null and
            tanhshrink.device.isMps() and tanhshrink.device_storage != null and
            closeF32(rsqrt_back.data, &.{ 1.0, 1.0 / std.math.sqrt(@as(f32, 2.0)), 1.0 / std.math.sqrt(@as(f32, 3.0)), 0.5, 1.0 / std.math.sqrt(@as(f32, 5.0)), 1.0 / std.math.sqrt(@as(f32, 6.0)) }, 0.001) and
            closeF32(leaky_back.data, &.{ -0.2, -0.1, 0, 1, 2, 3 }, 0.001) and
            closeF32(silu_back.data, &.{ -2.0 * sigmoid_neg2, -1.0 / (@as(f32, 1.0) + std.math.e), 0, @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0))), @as(f32, 2.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -2.0))), @as(f32, 3.0) * sigmoid_pos3 }, 0.01) and
            closeF32(hardsigmoid_back.data, &.{ 1.0 / 6.0, 2.0 / 6.0, 0.5, 4.0 / 6.0, 5.0 / 6.0, 1.0 }, 0.001) and
            closeF32(hardswish_back.data, &.{ -2.0 / 6.0, -2.0 / 6.0, 0, 4.0 / 6.0, 10.0 / 6.0, 3.0 }, 0.001) and
            closeF32(softshrink_back.data, &.{ -1.5, -0.5, 0, 0.5, 1.5, 2.5 }, 0.001) and
            closeF32(elu_back.data, &.{ std.math.exp(@as(f32, -2.0)) - 1.0, std.math.exp(@as(f32, -1.0)) - 1.0, 0, 1, 2, 3 }, 0.01) and
            closeF32(celu_back.data, &.{ 2.0 * (std.math.exp(@as(f32, -1.0)) - 1.0), 2.0 * (std.math.exp(@as(f32, -0.5)) - 1.0), 0, 1, 2, 3 }, 0.01) and
            closeF32(selu_back.data, &.{ selu_neg2, selu_neg1, 0, selu_scale, 2.0 * selu_scale, 3.0 * selu_scale }, 0.001) and
            closeF32(tanh_back.data, &.{ std.math.tanh(@as(f32, -2.0)), std.math.tanh(@as(f32, -1.0)), 0, std.math.tanh(@as(f32, 1.0)), std.math.tanh(@as(f32, 2.0)), std.math.tanh(@as(f32, 3.0)) }, 0.001) and
            closeF32(tanhshrink_back.data, &.{ -2.0 - std.math.tanh(@as(f32, -2.0)), -1.0 - std.math.tanh(@as(f32, -1.0)), 0, 1.0 - std.math.tanh(@as(f32, 1.0)), 2.0 - std.math.tanh(@as(f32, 2.0)), 3.0 - std.math.tanh(@as(f32, 3.0)) }, 0.001);

        var pow_zero = try mat_lhs.powScalar(0);
        defer pow_zero.deinit();
        var pow_zero_back = try pow_zero.cpu();
        defer pow_zero_back.deinit();
        var pow_one = try mat_lhs.powScalar(1);
        defer pow_one.deinit();
        var pow_one_back = try pow_one.cpu();
        defer pow_one_back.deinit();
        var pow_recip = try mat_lhs.powScalar(-1);
        defer pow_recip.deinit();
        var pow_recip_back = try pow_recip.cpu();
        defer pow_recip_back.deinit();
        var pow_sqrt = try mat_lhs.powScalar(0.5);
        defer pow_sqrt.deinit();
        var pow_sqrt_back = try pow_sqrt.cpu();
        defer pow_sqrt_back.deinit();
        var pow_rsqrt = try mat_lhs.powScalar(-0.5);
        defer pow_rsqrt.deinit();
        var pow_rsqrt_back = try pow_rsqrt.cpu();
        defer pow_rsqrt_back.deinit();
        var pow_square = try mat_lhs.powScalar(2);
        defer pow_square.deinit();
        var pow_square_back = try pow_square.cpu();
        defer pow_square_back.deinit();
        var pow_cube = try mat_lhs.powScalar(3);
        defer pow_cube.deinit();
        var pow_cube_back = try pow_cube.cpu();
        defer pow_cube_back.deinit();
        pow_ok = f16_pow_ok and bf16_pow_ok and
            pow_zero.device.isMps() and pow_zero.device_storage != null and
            pow_one.device.isMps() and pow_one.device_storage != null and
            pow_recip.device.isMps() and pow_recip.device_storage != null and
            pow_sqrt.device.isMps() and pow_sqrt.device_storage != null and
            pow_rsqrt.device.isMps() and pow_rsqrt.device_storage != null and
            pow_square.device.isMps() and pow_square.device_storage != null and
            pow_cube.device.isMps() and pow_cube.device_storage != null and
            closeF32(pow_zero_back.data, &.{ 1, 1, 1, 1, 1, 1 }, 0.001) and
            closeF32(pow_one_back.data, &.{ 1, 2, 3, 4, 5, 6 }, 0.001) and
            closeF32(pow_recip_back.data, &.{ 1, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0 }, 0.001) and
            closeF32(pow_sqrt_back.data, &.{ 1, std.math.sqrt(@as(f32, 2)), std.math.sqrt(@as(f32, 3)), 2, std.math.sqrt(@as(f32, 5)), std.math.sqrt(@as(f32, 6)) }, 0.001) and
            closeF32(pow_rsqrt_back.data, &.{ 1, 1.0 / std.math.sqrt(@as(f32, 2)), 1.0 / std.math.sqrt(@as(f32, 3)), 0.5, 1.0 / std.math.sqrt(@as(f32, 5)), 1.0 / std.math.sqrt(@as(f32, 6)) }, 0.001) and
            closeF32(pow_square_back.data, &.{ 1, 4, 9, 16, 25, 36 }, 0.001) and
            closeF32(pow_cube_back.data, &.{ 1, 8, 27, 64, 125, 216 }, 0.001);

        var ternary_base = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, vx.mps(0));
        defer ternary_base.deinit();
        var ternary_lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 1, 1, 1 }, &.{ 2, 2 }, vx.mps(0));
        defer ternary_lhs.deinit();
        var ternary_rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 1.5, 2, 2.5 }, &.{ 2, 2 }, vx.mps(0));
        defer ternary_rhs.deinit();
        var addcmul = try ternary_base.addcmul(ternary_lhs, ternary_rhs, 2.0);
        defer addcmul.deinit();
        var addcmul_back = try addcmul.cpu();
        defer addcmul_back.deinit();
        var addcdiv = try ternary_base.addcdiv(ternary_lhs, ternary_rhs, 0.5);
        defer addcdiv.deinit();
        var addcdiv_back = try addcdiv.cpu();
        defer addcdiv_back.deinit();
        var lerp_scalar = try ternary_base.lerpScalar(ternary_rhs, 0.5);
        defer lerp_scalar.deinit();
        var lerp_scalar_back = try lerp_scalar.cpu();
        defer lerp_scalar_back.deinit();
        var lerp_array = try ternary_base.lerp(ternary_lhs, ternary_rhs);
        defer lerp_array.deinit();
        var lerp_array_back = try lerp_array.cpu();
        defer lerp_array_back.deinit();
        ternary_ok = f16_ternary_ok and bf16_ternary_ok and
            addcmul.device.isMps() and addcmul.device_storage != null and
            addcdiv.device.isMps() and addcdiv.device_storage != null and
            lerp_scalar.device.isMps() and lerp_scalar.device_storage != null and
            lerp_array.device.isMps() and lerp_array.device_storage != null and
            closeF32(addcmul_back.data, &.{ 3, 5, 7, 9 }, 0.001) and
            closeF32(addcdiv_back.data, &.{ 1.5, 2.3333333, 3.25, 4.2 }, 0.001) and
            closeF32(lerp_scalar_back.data, &.{ 1, 1.75, 2.5, 3.25 }, 0.001) and
            closeF32(lerp_array_back.data, &.{ 1, 0.5, -1, -3.5 }, 0.001);

        var loss_target = try vx.Array(f32).zerosOn(allocator, &.{ 2, 3 }, vx.mps(0));
        defer loss_target.deinit();
        var mse = try shifted_for_max.mseLoss(loss_target, .none);
        defer mse.deinit();
        var mse_back = try mse.cpu();
        defer mse_back.deinit();
        var l1 = try shifted_for_max.l1Loss(loss_target, .none);
        defer l1.deinit();
        var l1_back = try l1.cpu();
        defer l1_back.deinit();
        var smooth_l1 = try shifted_for_max.smoothL1Loss(loss_target, 1.0, .none);
        defer smooth_l1.deinit();
        var smooth_l1_back = try smooth_l1.cpu();
        defer smooth_l1_back.deinit();
        var huber = try shifted_for_max.huberLoss(loss_target, 1.0, .none);
        defer huber.deinit();
        var huber_back = try huber.cpu();
        defer huber_back.deinit();
        loss_ok = f16_loss_ok and bf16_loss_ok and
            mse.device.isMps() and mse.device_storage != null and
            l1.device.isMps() and l1.device_storage != null and
            smooth_l1.device.isMps() and smooth_l1.device_storage != null and
            huber.device.isMps() and huber.device_storage != null and
            closeF32(mse_back.data, &.{ 4, 1, 0, 1, 4, 9 }, 0.001) and
            closeF32(l1_back.data, &.{ 2, 1, 0, 1, 2, 3 }, 0.001) and
            closeF32(smooth_l1_back.data, &.{ 1.5, 0.5, 0, 0.5, 1.5, 2.5 }, 0.001) and
            closeF32(huber_back.data, &.{ 1.5, 0.5, 0, 0.5, 1.5, 2.5 }, 0.001);

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

        var logsumexp_row = try mat_lhs.logsumexp(1, false);
        defer logsumexp_row.deinit();
        var logsumexp_row_back = try logsumexp_row.cpu();
        defer logsumexp_row_back.deinit();
        var logsumexp_col_keep = try mat_lhs.logsumexp(0, true);
        defer logsumexp_col_keep.deinit();
        var logsumexp_col_keep_back = try logsumexp_col_keep.cpu();
        defer logsumexp_col_keep_back.deinit();
        var logsumexp_all = try mat_lhs.logsumexpAxes(&.{ 0, 1 }, false);
        defer logsumexp_all.deinit();
        var logsumexp_all_back = try logsumexp_all.cpu();
        defer logsumexp_all_back.deinit();
        const all_log_denom = std.math.log(f32, std.math.e, std.math.exp(@as(f32, -5)) + std.math.exp(@as(f32, -4)) + std.math.exp(@as(f32, -3)) + std.math.exp(@as(f32, -2)) + std.math.exp(@as(f32, -1)) + 1.0);
        logsumexp_ok = logsumexp_row.device.isMps() and logsumexp_row.device_storage != null and
            logsumexp_col_keep.device.isMps() and logsumexp_col_keep.device_storage != null and
            logsumexp_all.device.isMps() and logsumexp_all.device_storage != null and
            std.mem.eql(usize, logsumexp_row_back.shape, &.{2}) and
            closeF32(logsumexp_row_back.data, &.{ 3.0 + row_log_denom, 6.0 + row_log_denom }, 0.03) and
            std.mem.eql(usize, logsumexp_col_keep_back.shape, &.{ 1, 3 }) and
            closeF32(logsumexp_col_keep_back.data, &.{ 4.0 + col_log_denom, 5.0 + col_log_denom, 6.0 + col_log_denom }, 0.03) and
            std.mem.eql(usize, logsumexp_all_back.shape, &.{}) and
            closeF32(logsumexp_all_back.data, &.{6.0 + all_log_denom}, 0.03);

        var softmin_row = try mat_lhs.softmin(1);
        defer softmin_row.deinit();
        var softmin_row_back = try softmin_row.cpu();
        defer softmin_row_back.deinit();
        var softmin_col = try mat_lhs.softmin(0);
        defer softmin_col.deinit();
        var softmin_col_back = try softmin_col.cpu();
        defer softmin_col_back.deinit();
        const softmin_row_denom = 1.0 + std.math.exp(@as(f32, -1)) + std.math.exp(@as(f32, -2));
        const softmin_col_denom = 1.0 + std.math.exp(@as(f32, -3));
        softmin_ok = f16_softmin_ok and bf16_softmin_ok and
            softmin_row.device.isMps() and softmin_row.device_storage != null and
            softmin_col.device.isMps() and softmin_col.device_storage != null and
            closeF32(softmin_row_back.data, &.{ 1.0 / softmin_row_denom, std.math.exp(@as(f32, -1)) / softmin_row_denom, std.math.exp(@as(f32, -2)) / softmin_row_denom, 1.0 / softmin_row_denom, std.math.exp(@as(f32, -1)) / softmin_row_denom, std.math.exp(@as(f32, -2)) / softmin_row_denom }, 0.001) and
            closeF32(softmin_col_back.data, &.{ 1.0 / softmin_col_denom, 1.0 / softmin_col_denom, 1.0 / softmin_col_denom, std.math.exp(@as(f32, -3)) / softmin_col_denom, std.math.exp(@as(f32, -3)) / softmin_col_denom, std.math.exp(@as(f32, -3)) / softmin_col_denom }, 0.001);

        var log_softmin_row = try mat_lhs.logSoftmin(1);
        defer log_softmin_row.deinit();
        var log_softmin_row_back = try log_softmin_row.cpu();
        defer log_softmin_row_back.deinit();
        var log_softmin_col = try mat_lhs.logSoftmin(0);
        defer log_softmin_col.deinit();
        var log_softmin_col_back = try log_softmin_col.cpu();
        defer log_softmin_col_back.deinit();
        const softmin_row_log_denom = std.math.log(f32, std.math.e, softmin_row_denom);
        const softmin_col_log_denom = std.math.log(f32, std.math.e, softmin_col_denom);
        log_softmin_ok = f16_log_softmin_ok and bf16_log_softmin_ok and
            log_softmin_row.device.isMps() and log_softmin_row.device_storage != null and
            log_softmin_col.device.isMps() and log_softmin_col.device_storage != null and
            closeF32(log_softmin_row_back.data, &.{ -softmin_row_log_denom, -1.0 - softmin_row_log_denom, -2.0 - softmin_row_log_denom, -softmin_row_log_denom, -1.0 - softmin_row_log_denom, -2.0 - softmin_row_log_denom }, 0.001) and
            closeF32(log_softmin_col_back.data, &.{ -softmin_col_log_denom, -softmin_col_log_denom, -softmin_col_log_denom, -3.0 - softmin_col_log_denom, -3.0 - softmin_col_log_denom, -3.0 - softmin_col_log_denom }, 0.001);

        fingerprint ^= hashF32(back.data) ^ hashF32(clone_back.data) ^ hashF32(reshaped_back.data) ^ hashF32(reshaped_infer_back.data) ^ hashF32(flattened_back.data) ^ hashF32(filled_back.data) ^ hashF32(random_a_back.data) ^ hashF16(random_f16_a_back.data) ^ hashBF16(random_bf16_a_back.data) ^ hashF32(add_back.data) ^ hashF32(div_back.data) ^ hashF32(scaled_back.data) ^ hashF32(rsub_back.data) ^ hashF32(array_scalar_div_back.data) ^ hashF32(scalar_array_rdiv_back.data) ^ hashF32(square_back.data) ^ hashF32(sqrt_back.data) ^ hashF32(exp_back.data) ^ hashF32(log_back.data) ^ hashF32(exp2_back.data) ^ hashF32(expm1_back.data) ^ hashF32(log1p_back.data) ^ hashF32(log2_back.data) ^ hashF32(log10_back.data) ^ hashF32(sin_back.data) ^ hashF32(cos_back.data) ^ hashF32(tan_back.data) ^ hashF16(f16_add_back.data) ^ hashF16(f16_div_back.data) ^ hashF16(f16_scaled_back.data) ^ hashF16(f16_rsub_back.data) ^ hashF16(f16_abs_back.data) ^ hashF16(f16_sqrt_back.data) ^ hashF16(f16_exp_back.data) ^ hashF16(f16_mat_back.data) ^ hashF16(f16_mat_add_back.data) ^ hashF16(f16_mat_scaled_add_back.data) ^ hashF16(f16_matvec_back.data) ^ hashF16(f16_vecmat_back.data) ^ hashF16(f16_dot_back.data) ^ hashF16(f16_norm_back.data) ^ hashF16(f16_normalized_back.data) ^ hashF16(f16_cosine_back.data) ^ hashF16(f16_distance_back.data) ^ hashF16(f16_transposed_back.data) ^ hashF16(f16_row_added_back.data) ^ hashF16(f16_row_sub_back.data) ^ hashF16(f16_row_mul_back.data) ^ hashF16(f16_row_div_back.data) ^ hashF16(f16_col_added_back.data) ^ hashF16(f16_col_sub_back.data) ^ hashF16(f16_col_mul_back.data) ^ hashF16(f16_col_div_back.data) ^ hashF16(f16_row_sum_back.data) ^ hashF16(f16_col_max_back.data) ^ hashF16(f16_row_prod_keep_back.data) ^ hashF16(f16_maximum_back.data) ^ hashF16(f16_minimum_back.data) ^ hashF16(f16_maximum_scalar_back.data) ^ hashF16(f16_minimum_scalar_back.data) ^ hashF16(f16_relu_back.data) ^ hashF16(f16_threshold_back.data) ^ hashF16(f16_clip_back.data) ^ hashF16(f16_relu6_back.data) ^ hashF16(f16_hardtanh_back.data) ^ hashF16(f16_clip_array_back.data) ^ hashF16(f16_sigmoid_back.data) ^ hashF16(f16_softsign_back.data) ^ hashF16(f16_rsqrt_back.data) ^ hashF16(f16_leaky_back.data) ^ hashF16(f16_silu_back.data) ^ hashF16(f16_hardsigmoid_back.data) ^ hashF16(f16_hardswish_back.data) ^ hashF16(f16_softshrink_back.data) ^ hashF16(f16_elu_back.data) ^ hashF16(f16_celu_back.data) ^ hashF16(f16_pow_zero_back.data) ^ hashF16(f16_pow_one_back.data) ^ hashF16(f16_pow_recip_back.data) ^ hashF16(f16_pow_sqrt_back.data) ^ hashF16(f16_pow_rsqrt_back.data) ^ hashF16(f16_pow_square_back.data) ^ hashF16(f16_pow_cube_back.data) ^ hashF16(f16_addcmul_back.data) ^ hashF16(f16_addcdiv_back.data) ^ hashF16(f16_lerp_scalar_back.data) ^ hashF16(f16_lerp_array_back.data) ^ hashF16(f16_mse_back.data) ^ hashF16(f16_l1_back.data) ^ hashF16(f16_smooth_l1_back.data) ^ hashF16(f16_huber_back.data) ^ hashF16(f16_softmax_row_back.data) ^ hashF16(f16_softmax_col_back.data) ^ hashF16(f16_log_softmax_row_back.data) ^ hashF16(f16_log_softmax_col_back.data) ^ hashF16(f16_softmin_row_back.data) ^ hashF16(f16_softmin_col_back.data) ^ hashF16(f16_log_softmin_row_back.data) ^ hashF16(f16_log_softmin_col_back.data) ^ hashBF16(bf16_add_back.data) ^ hashBF16(bf16_div_back.data) ^ hashBF16(bf16_scaled_back.data) ^ hashBF16(bf16_rsub_back.data) ^ hashBF16(bf16_abs_back.data) ^ hashBF16(bf16_sqrt_back.data) ^ hashBF16(bf16_exp_back.data) ^ hashBF16(bf16_mat_back.data) ^ hashBF16(bf16_mat_add_back.data) ^ hashBF16(bf16_mat_scaled_add_back.data) ^ hashBF16(bf16_matvec_back.data) ^ hashBF16(bf16_vecmat_back.data) ^ hashBF16(bf16_dot_back.data) ^ hashBF16(bf16_norm_back.data) ^ hashBF16(bf16_normalized_back.data) ^ hashBF16(bf16_cosine_back.data) ^ hashBF16(bf16_distance_back.data) ^ hashBF16(bf16_transposed_back.data) ^ hashBF16(bf16_row_added_back.data) ^ hashBF16(bf16_row_sub_back.data) ^ hashBF16(bf16_row_mul_back.data) ^ hashBF16(bf16_row_div_back.data) ^ hashBF16(bf16_col_added_back.data) ^ hashBF16(bf16_col_sub_back.data) ^ hashBF16(bf16_col_mul_back.data) ^ hashBF16(bf16_col_div_back.data) ^ hashBF16(bf16_row_sum_back.data) ^ hashBF16(bf16_col_max_back.data) ^ hashBF16(bf16_col_min_back.data) ^ hashBF16(bf16_row_prod_keep_back.data) ^ hashBF16(bf16_maximum_back.data) ^ hashBF16(bf16_minimum_back.data) ^ hashBF16(bf16_maximum_scalar_back.data) ^ hashBF16(bf16_minimum_scalar_back.data) ^ hashBF16(bf16_relu_back.data) ^ hashBF16(bf16_threshold_back.data) ^ hashBF16(bf16_clip_back.data) ^ hashBF16(bf16_relu6_back.data) ^ hashBF16(bf16_hardtanh_back.data) ^ hashBF16(bf16_clip_array_back.data) ^ hashBF16(bf16_sigmoid_back.data) ^ hashBF16(bf16_softsign_back.data) ^ hashBF16(bf16_rsqrt_back.data) ^ hashBF16(bf16_leaky_back.data) ^ hashBF16(bf16_silu_back.data) ^ hashBF16(bf16_hardsigmoid_back.data) ^ hashBF16(bf16_hardswish_back.data) ^ hashBF16(bf16_softshrink_back.data) ^ hashBF16(bf16_elu_back.data) ^ hashBF16(bf16_celu_back.data) ^ hashBF16(bf16_pow_zero_back.data) ^ hashBF16(bf16_pow_one_back.data) ^ hashBF16(bf16_pow_recip_back.data) ^ hashBF16(bf16_pow_sqrt_back.data) ^ hashBF16(bf16_pow_rsqrt_back.data) ^ hashBF16(bf16_pow_square_back.data) ^ hashBF16(bf16_pow_cube_back.data) ^ hashBF16(bf16_addcmul_back.data) ^ hashBF16(bf16_addcdiv_back.data) ^ hashBF16(bf16_lerp_scalar_back.data) ^ hashBF16(bf16_lerp_array_back.data) ^ hashBF16(bf16_mse_back.data) ^ hashBF16(bf16_l1_back.data) ^ hashBF16(bf16_smooth_l1_back.data) ^ hashBF16(bf16_huber_back.data) ^ hashBF16(bf16_softmax_row_back.data) ^ hashBF16(bf16_softmax_col_back.data) ^ hashBF16(bf16_log_softmax_row_back.data) ^ hashBF16(bf16_log_softmax_col_back.data) ^ hashBF16(bf16_softmin_row_back.data) ^ hashBF16(bf16_softmin_col_back.data) ^ hashBF16(bf16_log_softmin_row_back.data) ^ hashBF16(bf16_log_softmin_col_back.data) ^ hashF32(mat_back.data) ^ hashF32(mat_add_back.data) ^ hashF32(mat_scaled_add_back.data) ^ hashF32(matvec_back.data) ^ hashF32(vecmat_back.data) ^ hashF32(dot_back.data) ^ hashF32(norm_back.data) ^ hashF32(normalized_back.data) ^ hashF32(cosine_back.data) ^ hashF32(distance_back.data) ^ hashF32(transposed_back.data) ^ hashF32(row_added_back.data) ^ hashF32(row_sub_back.data) ^ hashF32(row_mul_back.data) ^ hashF32(row_div_back.data) ^ hashF32(col_added_back.data) ^ hashF32(col_sub_back.data) ^ hashF32(col_mul_back.data) ^ hashF32(col_div_back.data) ^ hashF32(row_sum_back.data) ^ hashF32(col_max_back.data) ^ hashF32(row_prod_keep_back.data) ^ hashF32(maximum_back.data) ^ hashF32(minimum_back.data) ^ hashF32(maximum_scalar_back.data) ^ hashF32(minimum_scalar_back.data) ^ hashF32(relu_back.data) ^ hashF32(threshold_back.data) ^ hashF32(clip_back.data) ^ hashF32(relu6_back.data) ^ hashF32(hardtanh_back.data) ^ hashF32(clip_array_back.data) ^ hashF32(sigmoid_back.data) ^ hashF32(softsign_back.data) ^ hashF32(rsqrt_back.data) ^ hashF32(leaky_back.data) ^ hashF32(silu_back.data) ^ hashF32(hardsigmoid_back.data) ^ hashF32(hardswish_back.data) ^ hashF32(softshrink_back.data) ^ hashF32(elu_back.data) ^ hashF32(celu_back.data) ^ hashF32(pow_zero_back.data) ^ hashF32(pow_one_back.data) ^ hashF32(pow_recip_back.data) ^ hashF32(pow_sqrt_back.data) ^ hashF32(pow_rsqrt_back.data) ^ hashF32(pow_square_back.data) ^ hashF32(pow_cube_back.data) ^ hashF32(addcmul_back.data) ^ hashF32(addcdiv_back.data) ^ hashF32(lerp_scalar_back.data) ^ hashF32(lerp_array_back.data) ^ hashF32(mse_back.data) ^ hashF32(l1_back.data) ^ hashF32(smooth_l1_back.data) ^ hashF32(huber_back.data) ^ hashF32(softmax_row_back.data) ^ hashF32(softmax_col_back.data) ^ hashF32(log_softmax_row_back.data) ^ hashF32(log_softmax_col_back.data) ^ hashF32(softmin_row_back.data) ^ hashF32(softmin_col_back.data) ^ hashF32(log_softmin_row_back.data) ^ hashF32(log_softmin_col_back.data);
        fingerprint ^= hashF32(flat_sum_back.data) ^ hashF32(flat_prod_keep_back.data) ^ hashF32(flat_min_back.data) ^ hashF32(flat_max_keep_back.data) ^ hashF32(all_axes_sum_back.data) ^ hashF32(all_axes_max_keep_back.data) ^ hashF32(row_ptp_back.data) ^ hashF32(flat_ptp_keep_back.data) ^ hashF32(all_axes_ptp_back.data) ^ hashF32(logsumexp_row_back.data) ^ hashF32(logsumexp_col_keep_back.data) ^ hashF32(logsumexp_all_back.data);
        fingerprint ^= hashF32(mat_chain_add_back.data) ^ hashF32(mat_chain_radd_back.data) ^ hashF32(mat_chain_rsub_back.data) ^ hashF32(mat_chain_sqrt_back.data) ^ hashF32(mat_chain_exp_back.data);
        fingerprint ^= hashBF16(bf16_mat_chain_add_back.data) ^ hashBF16(bf16_mat_chain_sqrt_back.data) ^ hashBF16(bf16_mat_chain_exp_back.data);
        fingerprint ^= hashF16(f16_row_mean_back.data) ^ hashF16(f16_col_mean_keep_back.data) ^ hashF16(f16_flat_var_back.data) ^ hashF16(f16_row_var_back.data) ^ hashF16(f16_col_std_keep_back.data) ^ hashBF16(bf16_row_mean_back.data) ^ hashBF16(bf16_col_mean_keep_back.data) ^ hashBF16(bf16_flat_var_back.data) ^ hashBF16(bf16_row_var_back.data) ^ hashBF16(bf16_col_std_keep_back.data) ^ hashF32(row_mean_back.data) ^ hashF32(col_mean_keep_back.data) ^ hashF32(flat_var_back.data) ^ hashF32(row_var_back.data) ^ hashF32(col_std_keep_back.data);
        fingerprint ^= hashF16(f16_selu_back.data) ^ hashF16(f16_tanh_back.data) ^ hashF16(f16_tanhshrink_back.data) ^ hashBF16(bf16_selu_back.data) ^ hashBF16(bf16_tanh_back.data) ^ hashBF16(bf16_tanhshrink_back.data) ^ hashF32(selu_back.data) ^ hashF32(tanh_back.data) ^ hashF32(tanhshrink_back.data);
    }

    const ok_without_stats = if (available)
        report.ok() and roundtrip_ok and copy_ok and shape_ok and fill_ok and random_ok and elementwise_ok and scalar_ok and unary_ok and f16_elementwise_ok and f16_scalar_ok and f16_unary_ok and f16_matmul_ok and f16_matmul_add_ok and f16_vector_matmul_ok and f16_metric_ok and f16_transpose_ok and f16_broadcast_ok and f16_reduction_ok and f16_softmax_ok and f16_log_softmax_ok and f16_softmin_ok and f16_log_softmin_ok and f16_activation_ok and f16_activation_compose_ok and f16_pow_ok and f16_loss_ok and bf16_elementwise_ok and bf16_scalar_ok and bf16_unary_ok and bf16_matmul_ok and bf16_matmul_add_ok and bf16_matmul_chain_ok and bf16_vector_matmul_ok and bf16_metric_ok and bf16_transpose_ok and bf16_broadcast_ok and bf16_reduction_ok and bf16_softmax_ok and bf16_log_softmax_ok and bf16_softmin_ok and bf16_log_softmin_ok and bf16_activation_ok and bf16_activation_compose_ok and bf16_pow_ok and bf16_loss_ok and matmul_ok and matmul_add_ok and matmul_chain_ok and vector_matmul_ok and metric_ok and transpose_ok and broadcast_ok and reduction_ok and minmax_ok and activation_ok and activation_compose_ok and pow_ok and ternary_ok and loss_ok and softmax_ok and log_softmax_ok and logsumexp_ok and softmin_ok and log_softmin_ok and bytes != 0
    else
        !report.ok() and roundtrip_ok and copy_ok and shape_ok and fill_ok and random_ok and elementwise_ok and scalar_ok and unary_ok and f16_elementwise_ok and f16_scalar_ok and f16_unary_ok and f16_matmul_ok and f16_matmul_add_ok and f16_vector_matmul_ok and f16_metric_ok and f16_transpose_ok and f16_broadcast_ok and f16_reduction_ok and f16_softmax_ok and f16_log_softmax_ok and f16_softmin_ok and f16_log_softmin_ok and f16_activation_ok and f16_activation_compose_ok and f16_pow_ok and f16_loss_ok and bf16_elementwise_ok and bf16_scalar_ok and bf16_unary_ok and bf16_matmul_ok and bf16_matmul_add_ok and bf16_matmul_chain_ok and bf16_vector_matmul_ok and bf16_metric_ok and bf16_transpose_ok and bf16_broadcast_ok and bf16_reduction_ok and bf16_softmax_ok and bf16_log_softmax_ok and bf16_softmin_ok and bf16_log_softmin_ok and bf16_activation_ok and bf16_activation_compose_ok and bf16_pow_ok and bf16_loss_ok and matmul_ok and matmul_add_ok and matmul_chain_ok and vector_matmul_ok and metric_ok and transpose_ok and broadcast_ok and reduction_ok and minmax_ok and activation_ok and activation_compose_ok and pow_ok and ternary_ok and loss_ok and softmax_ok and log_softmax_ok and logsumexp_ok and softmin_ok and log_softmin_ok;
    const ok = ok_without_stats and f16_stats_ok and bf16_stats_ok and stats_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_mps_storage_smoke\",\"ok\":{},\"available\":{},\"status\":\"{s}\",\"backend\":\"{s}\"",
        .{ ok, available, report.status.label(), report.backend_label },
    );
    try stdout.interface.print(
        ",\"roundtrip_ok\":{},\"copy_ok\":{},\"shape_ok\":{},\"fill_ok\":{},\"random_ok\":{},\"elementwise_ok\":{},\"scalar_ok\":{},\"unary_ok\":{},\"f16_elementwise_ok\":{},\"f16_scalar_ok\":{},\"f16_unary_ok\":{},\"f16_matmul_ok\":{},\"f16_matmul_add_ok\":{},\"f16_vector_matmul_ok\":{},\"f16_metric_ok\":{},\"f16_transpose_ok\":{},\"f16_broadcast_ok\":{},\"f16_reduction_ok\":{},\"f16_stats_ok\":{},\"f16_softmax_ok\":{},\"f16_log_softmax_ok\":{},\"f16_softmin_ok\":{},\"f16_log_softmin_ok\":{},\"f16_activation_ok\":{},\"f16_activation_compose_ok\":{},\"f16_pow_ok\":{},\"f16_loss_ok\":{}",
        .{ roundtrip_ok, copy_ok, shape_ok, fill_ok, random_ok, elementwise_ok, scalar_ok, unary_ok, f16_elementwise_ok, f16_scalar_ok, f16_unary_ok, f16_matmul_ok, f16_matmul_add_ok, f16_vector_matmul_ok, f16_metric_ok, f16_transpose_ok, f16_broadcast_ok, f16_reduction_ok, f16_stats_ok, f16_softmax_ok, f16_log_softmax_ok, f16_softmin_ok, f16_log_softmin_ok, f16_activation_ok, f16_activation_compose_ok, f16_pow_ok, f16_loss_ok },
    );
    try stdout.interface.print(
        ",\"bf16_elementwise_ok\":{},\"bf16_scalar_ok\":{},\"bf16_unary_ok\":{},\"bf16_matmul_ok\":{},\"bf16_matmul_add_ok\":{},\"bf16_matmul_chain_ok\":{},\"bf16_vector_matmul_ok\":{},\"bf16_metric_ok\":{},\"bf16_transpose_ok\":{},\"bf16_broadcast_ok\":{},\"bf16_reduction_ok\":{},\"bf16_stats_ok\":{},\"bf16_softmax_ok\":{},\"bf16_log_softmax_ok\":{},\"bf16_softmin_ok\":{},\"bf16_log_softmin_ok\":{},\"bf16_activation_ok\":{},\"bf16_activation_compose_ok\":{},\"bf16_pow_ok\":{},\"bf16_loss_ok\":{}",
        .{ bf16_elementwise_ok, bf16_scalar_ok, bf16_unary_ok, bf16_matmul_ok, bf16_matmul_add_ok, bf16_matmul_chain_ok, bf16_vector_matmul_ok, bf16_metric_ok, bf16_transpose_ok, bf16_broadcast_ok, bf16_reduction_ok, bf16_stats_ok, bf16_softmax_ok, bf16_log_softmax_ok, bf16_softmin_ok, bf16_log_softmin_ok, bf16_activation_ok, bf16_activation_compose_ok, bf16_pow_ok, bf16_loss_ok },
    );
    try stdout.interface.print(
        ",\"matmul_ok\":{},\"matmul_add_ok\":{},\"matmul_chain_ok\":{},\"vector_matmul_ok\":{},\"metric_ok\":{},\"transpose_ok\":{},\"broadcast_ok\":{},\"reduction_ok\":{},\"stats_ok\":{},\"minmax_ok\":{},\"activation_ok\":{},\"activation_compose_ok\":{},\"pow_ok\":{},\"ternary_ok\":{},\"loss_ok\":{},\"softmax_ok\":{},\"log_softmax_ok\":{},\"logsumexp_ok\":{},\"softmin_ok\":{},\"log_softmin_ok\":{}",
        .{ matmul_ok, matmul_add_ok, matmul_chain_ok, vector_matmul_ok, metric_ok, transpose_ok, broadcast_ok, reduction_ok, stats_ok, minmax_ok, activation_ok, activation_compose_ok, pow_ok, ternary_ok, loss_ok, softmax_ok, log_softmax_ok, logsumexp_ok, softmin_ok, log_softmin_ok },
    );
    try stdout.interface.print(
        ",\"bytes\":{d},\"fingerprint\":{d}}}\n",
        .{ bytes, fingerprint },
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

fn equalF16(actual: []const f16, expected: []const f16) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}

fn equalBF16(actual: []const vx.BFloat16, expected: []const vx.BFloat16) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a.bits != e.bits) return false;
    }
    return true;
}

fn inUnitRangeF32(values: []const f32) bool {
    for (values) |value| {
        if (!(value >= 0 and value < 1)) return false;
    }
    return true;
}

fn inUnitRangeF16(values: []const f16) bool {
    for (values) |value| {
        if (!(value >= 0 and value < 1)) return false;
    }
    return true;
}

fn inUnitRangeBF16(values: []const vx.BFloat16) bool {
    for (values) |value| {
        const f32_value = value.toF32();
        if (!(f32_value >= 0 and f32_value < 1)) return false;
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

fn closeBF16(actual: []const vx.BFloat16, expected: []const f32, tolerance: f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (@abs(a.toF32() - e) > tolerance) return false;
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

fn hashBF16(values: []const vx.BFloat16) u64 {
    var hasher = std.hash.Wyhash.init(0x4d50_5701_2a11_bf16);
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
