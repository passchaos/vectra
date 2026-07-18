//! Smoke gate for explicit Vectra -> Axiom CUDA device-buffer handle seed.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var host = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{4});
    defer host.deinit();

    var maybe_device = try vx.axiom_cuda.toDeviceF32(allocator, host);
    const status: []const u8 = if (maybe_device != null) "allocated" else if (vx.axiom_cuda.enabled()) "unavailable" else "disabled";
    var ok = !vx.axiom_cuda.enabled() or maybe_device != null;
    var fingerprint: u64 = 0;
    var bytes: usize = 0;
    if (maybe_device) |*device| {
        defer device.deinit();
        ok = device.ok();
        fingerprint = device.fingerprint();
        bytes = device.required_bytes;
    }

    var direct_storage_ok = !vx.axiom_cuda.enabled();
    var direct_add_ok = !vx.axiom_cuda.enabled();
    var direct_square_ok = !vx.axiom_cuda.enabled();
    var direct_unary_scalar_ok = !vx.axiom_cuda.enabled();
    var direct_reduction_ok = !vx.axiom_cuda.enabled();
    var direct_broadcast_ok = !vx.axiom_cuda.enabled();
    var direct_transpose_ok = !vx.axiom_cuda.enabled();
    var direct_softmax_ok = !vx.axiom_cuda.enabled();
    var direct_log_softmax_ok = !vx.axiom_cuda.enabled();
    var direct_ternary_ok = !vx.axiom_cuda.enabled();
    var direct_matmul_ok = !vx.axiom_cuda.enabled();
    var direct_matmul_add_ok = !vx.axiom_cuda.enabled();
    var scaled_matmul_add_ok = !vx.axiom_cuda.enabled();
    var chained_matmul_add_ok = !vx.axiom_cuda.enabled();
    var chained_matmul_sub_ok = !vx.axiom_cuda.enabled();
    var chained_sqrt_ok = !vx.axiom_cuda.enabled();
    var chained_exp_ok = !vx.axiom_cuda.enabled();
    var reversed_add_fusion_ok = !vx.axiom_cuda.enabled();
    var reversed_sub_fusion_ok = !vx.axiom_cuda.enabled();
    var pending_fusion_status_ok = !vx.axiom_cuda.enabled();
    var bf16_chained_sqrt_ok = !vx.axiom_cuda.enabled();
    var bf16_chained_exp_ok = !vx.axiom_cuda.enabled();
    var bf16_scalar_mul_ok = !vx.axiom_cuda.enabled();
    var bf16_broadcast_ok = !vx.axiom_cuda.enabled();
    var bf16_reduction_ok = !vx.axiom_cuda.enabled();
    var bf16_transpose_ok = !vx.axiom_cuda.enabled();
    var bf16_softmax_ok = !vx.axiom_cuda.enabled();
    var bf16_log_softmax_ok = !vx.axiom_cuda.enabled();
    var f16_activation_ok = !vx.axiom_cuda.enabled();
    var f16_broadcast_ok = !vx.axiom_cuda.enabled();
    var f16_reduction_ok = !vx.axiom_cuda.enabled();
    var f16_transpose_ok = !vx.axiom_cuda.enabled();
    var f16_softmax_ok = !vx.axiom_cuda.enabled();
    var f16_log_softmax_ok = !vx.axiom_cuda.enabled();
    var f64_matmul_ok = !vx.axiom_cuda.enabled();
    var f64_elementwise_ok = !vx.axiom_cuda.enabled();
    var f64_transpose_ok = !vx.axiom_cuda.enabled();
    var f64_broadcast_ok = !vx.axiom_cuda.enabled();
    var f64_reduction_ok = !vx.axiom_cuda.enabled();
    var f64_softmax_ok = !vx.axiom_cuda.enabled();
    var f64_log_softmax_ok = !vx.axiom_cuda.enabled();
    var f64_matmul_add_ok = !vx.axiom_cuda.enabled();
    var elementwise_binary_memref_fingerprint: u64 = 0;
    var elementwise_unary_memref_fingerprint: u64 = 0;
    var gemm_memref_fingerprint: u64 = 0;
    var reduction_memref_fingerprint: u64 = 0;
    var broadcast_memref_fingerprint: u64 = 0;
    var transpose_memref_fingerprint: u64 = 0;
    var softmax_memref_fingerprint: u64 = 0;
    var log_softmax_memref_fingerprint: u64 = 0;
    if (vx.Device.cuda(0).isAvailable()) {
        var lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, vx.cuda(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer rhs.deinit();
        var addend = try vx.Array(f32).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer addend.deinit();
        direct_storage_ok = lhs.device_storage != null and rhs.device_storage != null and lhs.data.len == 0 and rhs.data.len == 0;

        var sum = try lhs.add(rhs);
        defer sum.deinit();
        var sum_host = try sum.cpu();
        defer sum_host.deinit();
        direct_add_ok = sum.device.isCuda() and sum.device_storage != null and equalF32(sum_host.data, &.{ 2, 3, 4, 5 });

        var squared = try lhs.square();
        defer squared.deinit();
        var squared_host = try squared.cpu();
        defer squared_host.deinit();
        const elementwise_binary_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        elementwise_binary_memref_fingerprint = elementwise_binary_report.memref_spec_fingerprint;
        direct_square_ok = squared.device.isCuda() and squared.device_storage != null and
            elementwise_binary_report.valid() and
            std.mem.eql(u8, elementwise_binary_report.operation, "elementwise_binary") and
            equalF32(squared_host.data, &.{ 1, 4, 9, 16 });

        var row_sum = try lhs.sum(1, false);
        defer row_sum.deinit();
        var row_sum_host = try row_sum.cpu();
        defer row_sum_host.deinit();
        var col_sum_keep = try lhs.sum(0, true);
        defer col_sum_keep.deinit();
        var col_sum_keep_host = try col_sum_keep.cpu();
        defer col_sum_keep_host.deinit();
        var row_prod = try lhs.prod(1, false);
        defer row_prod.deinit();
        var row_prod_host = try row_prod.cpu();
        defer row_prod_host.deinit();
        var col_min = try lhs.min(0, false);
        defer col_min.deinit();
        var col_min_host = try col_min.cpu();
        defer col_min_host.deinit();
        var col_max = try lhs.max(0, false);
        defer col_max.deinit();
        var col_max_host = try col_max.cpu();
        defer col_max_host.deinit();
        const reduction_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        reduction_memref_fingerprint = reduction_report.memref_spec_fingerprint;
        direct_reduction_ok = row_sum.device.isCuda() and row_sum.device_storage != null and
            reduction_report.valid() and
            std.mem.eql(u8, reduction_report.operation, "reduction2d") and
            equalF32(row_sum_host.data, &.{ 3, 7 }) and
            col_sum_keep.device.isCuda() and col_sum_keep.device_storage != null and
            std.mem.eql(usize, col_sum_keep_host.shape, &.{ 1, 2 }) and
            equalF32(col_sum_keep_host.data, &.{ 4, 6 }) and
            row_prod.device.isCuda() and row_prod.device_storage != null and
            equalF32(row_prod_host.data, &.{ 2, 12 }) and
            col_min.device.isCuda() and col_min.device_storage != null and
            equalF32(col_min_host.data, &.{ 1, 2 }) and
            col_max.device.isCuda() and col_max.device_storage != null and
            equalF32(col_max_host.data, &.{ 3, 4 });

        var row_bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20 }, &.{2}, vx.cuda(0));
        defer row_bias.deinit();
        var row_broadcast = try lhs.add(row_bias);
        defer row_broadcast.deinit();
        var row_broadcast_host = try row_broadcast.cpu();
        defer row_broadcast_host.deinit();
        var column_bias = try vx.Array(f32).fromSliceOn(allocator, &.{ 100, 200 }, &.{ 2, 1 }, vx.cuda(0));
        defer column_bias.deinit();
        var column_broadcast = try lhs.add(column_bias);
        defer column_broadcast.deinit();
        var column_broadcast_host = try column_broadcast.cpu();
        defer column_broadcast_host.deinit();
        const broadcast_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        broadcast_memref_fingerprint = broadcast_report.memref_spec_fingerprint;
        direct_broadcast_ok = row_broadcast.device.isCuda() and row_broadcast.device_storage != null and
            broadcast_report.valid() and
            std.mem.eql(u8, broadcast_report.operation, "broadcast_add2d") and
            equalF32(row_broadcast_host.data, &.{ 11, 22, 13, 24 }) and
            column_broadcast.device.isCuda() and column_broadcast.device_storage != null and
            equalF32(column_broadcast_host.data, &.{ 101, 102, 203, 204 });

        var transposed = try lhs.transpose();
        defer transposed.deinit();
        var transposed_host = try transposed.cpu();
        defer transposed_host.deinit();
        const transpose_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        transpose_memref_fingerprint = transpose_report.memref_spec_fingerprint;
        direct_transpose_ok = transposed.device.isCuda() and transposed.device_storage != null and
            transpose_report.valid() and
            std.mem.eql(u8, transpose_report.operation, "transpose2d") and
            std.mem.eql(usize, transposed_host.shape, &.{ 2, 2 }) and
            equalF32(transposed_host.data, &.{ 1, 3, 2, 4 });

        var softmax_row = try lhs.softmax(1);
        defer softmax_row.deinit();
        var softmax_row_host = try softmax_row.cpu();
        defer softmax_row_host.deinit();
        var softmax_col = try lhs.softmax(0);
        defer softmax_col.deinit();
        var softmax_col_host = try softmax_col.cpu();
        defer softmax_col_host.deinit();
        const softmax_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        softmax_memref_fingerprint = softmax_report.memref_spec_fingerprint;
        const row_denom = std.math.exp(@as(f32, -1)) + 1.0;
        const col_denom = std.math.exp(@as(f32, -2)) + 1.0;
        direct_softmax_ok = softmax_row.device.isCuda() and softmax_row.device_storage != null and
            softmax_report.valid() and
            std.mem.eql(u8, softmax_report.operation, "softmax2d") and
            approxF32(softmax_row_host.data[0], std.math.exp(@as(f32, -1)) / row_denom, 0.01) and
            approxF32(softmax_row_host.data[1], 1.0 / row_denom, 0.01) and
            approxF32(softmax_row_host.data[2], std.math.exp(@as(f32, -1)) / row_denom, 0.01) and
            approxF32(softmax_row_host.data[3], 1.0 / row_denom, 0.01) and
            softmax_col.device.isCuda() and softmax_col.device_storage != null and
            approxF32(softmax_col_host.data[0], std.math.exp(@as(f32, -2)) / col_denom, 0.01) and
            approxF32(softmax_col_host.data[2], 1.0 / col_denom, 0.01);

        var log_softmax_row = try lhs.logSoftmax(1);
        defer log_softmax_row.deinit();
        var log_softmax_row_host = try log_softmax_row.cpu();
        defer log_softmax_row_host.deinit();
        var log_softmax_col = try lhs.logSoftmax(0);
        defer log_softmax_col.deinit();
        var log_softmax_col_host = try log_softmax_col.cpu();
        defer log_softmax_col_host.deinit();
        const log_softmax_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        log_softmax_memref_fingerprint = log_softmax_report.memref_spec_fingerprint;
        const row_log_denom = std.math.log(f32, std.math.e, row_denom);
        const col_log_denom = std.math.log(f32, std.math.e, col_denom);
        direct_log_softmax_ok = log_softmax_row.device.isCuda() and log_softmax_row.device_storage != null and
            log_softmax_report.valid() and
            std.mem.eql(u8, log_softmax_report.operation, "log_softmax2d") and
            approxF32(log_softmax_row_host.data[0], -1.0 - row_log_denom, 0.03) and
            approxF32(log_softmax_row_host.data[1], -row_log_denom, 0.03) and
            approxF32(log_softmax_row_host.data[2], -1.0 - row_log_denom, 0.03) and
            approxF32(log_softmax_row_host.data[3], -row_log_denom, 0.03) and
            log_softmax_col.device.isCuda() and log_softmax_col.device_storage != null and
            approxF32(log_softmax_col_host.data[0], -2.0 - col_log_denom, 0.03) and
            approxF32(log_softmax_col_host.data[2], -col_log_denom, 0.03);

        var negated = try lhs.neg();
        defer negated.deinit();
        var negated_host = try negated.cpu();
        defer negated_host.deinit();
        var abs_negated = try negated.abs();
        defer abs_negated.deinit();
        var abs_negated_host = try abs_negated.cpu();
        defer abs_negated_host.deinit();
        const elementwise_unary_report = vx.axiom_cuda.lastCudaDeviceMemRefReport();
        elementwise_unary_memref_fingerprint = elementwise_unary_report.memref_spec_fingerprint;
        var reciprocal = try lhs.reciprocal();
        defer reciprocal.deinit();
        var reciprocal_host = try reciprocal.cpu();
        defer reciprocal_host.deinit();
        var pow_zero = try lhs.powScalar(0);
        defer pow_zero.deinit();
        var pow_zero_host = try pow_zero.cpu();
        defer pow_zero_host.deinit();
        var pow_recip = try lhs.powScalar(-1);
        defer pow_recip.deinit();
        var pow_recip_host = try pow_recip.cpu();
        defer pow_recip_host.deinit();
        var pow_sqrt = try lhs.powScalar(0.5);
        defer pow_sqrt.deinit();
        var pow_sqrt_host = try pow_sqrt.cpu();
        defer pow_sqrt_host.deinit();
        var pow_rsqrt = try lhs.powScalar(-0.5);
        defer pow_rsqrt.deinit();
        var pow_rsqrt_host = try pow_rsqrt.cpu();
        defer pow_rsqrt_host.deinit();
        var pow_cube = try lhs.powScalar(3);
        defer pow_cube.deinit();
        var pow_cube_host = try pow_cube.cpu();
        defer pow_cube_host.deinit();
        var rsqrt = try lhs.rsqrt();
        defer rsqrt.deinit();
        var rsqrt_host = try rsqrt.cpu();
        defer rsqrt_host.deinit();
        var shifted_for_relu = try lhs.subScalar(3);
        defer shifted_for_relu.deinit();
        var relu_out = try shifted_for_relu.relu();
        defer relu_out.deinit();
        var relu_host = try relu_out.cpu();
        defer relu_host.deinit();
        var threshold_zero_out = try shifted_for_relu.threshold(0, 0);
        defer threshold_zero_out.deinit();
        var threshold_zero_host = try threshold_zero_out.cpu();
        defer threshold_zero_host.deinit();
        var threshold_clamp_out = try shifted_for_relu.threshold(0.5, 0.5);
        defer threshold_clamp_out.deinit();
        var threshold_clamp_host = try threshold_clamp_out.cpu();
        defer threshold_clamp_host.deinit();
        var sigmoid_out = try shifted_for_relu.sigmoid();
        defer sigmoid_out.deinit();
        var sigmoid_host = try sigmoid_out.cpu();
        defer sigmoid_host.deinit();
        var silu_out = try shifted_for_relu.silu();
        defer silu_out.deinit();
        var silu_host = try silu_out.cpu();
        defer silu_host.deinit();
        var clipped_out = try shifted_for_relu.clip(-0.5, 0.5);
        defer clipped_out.deinit();
        var clipped_host = try clipped_out.cpu();
        defer clipped_host.deinit();
        var clip_min_values = try vx.Array(f32).fullOn(allocator, &.{ 2, 2 }, -0.25, vx.cuda(0));
        defer clip_min_values.deinit();
        var clip_max_values = try vx.Array(f32).fullOn(allocator, &.{ 2, 2 }, 0.75, vx.cuda(0));
        defer clip_max_values.deinit();
        var clipped_array_out = try shifted_for_relu.clipArray(clip_min_values, clip_max_values);
        defer clipped_array_out.deinit();
        var clipped_array_host = try clipped_array_out.cpu();
        defer clipped_array_host.deinit();
        var relu6_out = try shifted_for_relu.relu6();
        defer relu6_out.deinit();
        var relu6_host = try relu6_out.cpu();
        defer relu6_host.deinit();
        var hardsigmoid_out = try shifted_for_relu.hardsigmoid();
        defer hardsigmoid_out.deinit();
        var hardsigmoid_host = try hardsigmoid_out.cpu();
        defer hardsigmoid_host.deinit();
        var hardswish_out = try shifted_for_relu.hardswish();
        defer hardswish_out.deinit();
        var hardswish_host = try hardswish_out.cpu();
        defer hardswish_host.deinit();
        var softsign_out = try shifted_for_relu.softsign();
        defer softsign_out.deinit();
        var softsign_host = try softsign_out.cpu();
        defer softsign_host.deinit();
        var softshrink_out = try shifted_for_relu.softshrink(0.5);
        defer softshrink_out.deinit();
        var softshrink_host = try softshrink_out.cpu();
        defer softshrink_host.deinit();
        var loss_target = try vx.Array(f32).zerosOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer loss_target.deinit();
        var mse_out = try shifted_for_relu.mseLoss(loss_target, .none);
        defer mse_out.deinit();
        var mse_host = try mse_out.cpu();
        defer mse_host.deinit();
        var l1_out = try shifted_for_relu.l1Loss(loss_target, .none);
        defer l1_out.deinit();
        var l1_host = try l1_out.cpu();
        defer l1_host.deinit();
        var smooth_l1_out = try shifted_for_relu.smoothL1Loss(loss_target, 1.0, .none);
        defer smooth_l1_out.deinit();
        var smooth_l1_host = try smooth_l1_out.cpu();
        defer smooth_l1_host.deinit();
        var huber_out = try shifted_for_relu.huberLoss(loss_target, 1.0, .none);
        defer huber_out.deinit();
        var huber_host = try huber_out.cpu();
        defer huber_host.deinit();
        var elu_out = try shifted_for_relu.elu(1.0);
        defer elu_out.deinit();
        var elu_host = try elu_out.cpu();
        defer elu_host.deinit();
        var celu_out = try shifted_for_relu.celu(2.0);
        defer celu_out.deinit();
        var celu_host = try celu_out.cpu();
        defer celu_host.deinit();
        var scaled_for_max = try shifted_for_relu.mulScalar(0.1);
        defer scaled_for_max.deinit();
        var leaky_relu_out = try shifted_for_relu.leakyRelu(0.1);
        defer leaky_relu_out.deinit();
        var leaky_relu_host = try leaky_relu_out.cpu();
        defer leaky_relu_host.deinit();
        var maximum_out = try shifted_for_relu.maximum(scaled_for_max);
        defer maximum_out.deinit();
        var maximum_host = try maximum_out.cpu();
        defer maximum_host.deinit();
        var minimum_out = try shifted_for_relu.minimum(scaled_for_max);
        defer minimum_out.deinit();
        var minimum_host = try minimum_out.cpu();
        defer minimum_host.deinit();
        const sigmoid_neg2 = @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, 2.0)));
        const sigmoid_pos1 = @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, -1.0)));
        direct_unary_scalar_ok = negated.device.isCuda() and negated.device_storage != null and
            equalF32(negated_host.data, &.{ -1, -2, -3, -4 }) and
            abs_negated.device.isCuda() and abs_negated.device_storage != null and
            elementwise_unary_report.valid() and
            std.mem.eql(u8, elementwise_unary_report.operation, "elementwise_unary") and
            equalF32(abs_negated_host.data, &.{ 1, 2, 3, 4 }) and
            reciprocal.device.isCuda() and reciprocal.device_storage != null and
            approxF32(reciprocal_host.data[0], 1.0, 1e-6) and
            approxF32(reciprocal_host.data[3], 0.25, 1e-6) and
            pow_zero.device.isCuda() and pow_zero.device_storage != null and
            equalF32(pow_zero_host.data, &.{ 1, 1, 1, 1 }) and
            pow_recip.device.isCuda() and pow_recip.device_storage != null and
            approxF32(pow_recip_host.data[3], 0.25, 1e-6) and
            pow_sqrt.device.isCuda() and pow_sqrt.device_storage != null and
            approxF32(pow_sqrt_host.data[3], 2, 0.01) and
            pow_rsqrt.device.isCuda() and pow_rsqrt.device_storage != null and
            approxF32(pow_rsqrt_host.data[3], 0.5, 0.01) and
            pow_cube.device.isCuda() and pow_cube.device_storage != null and
            equalF32(pow_cube_host.data, &.{ 1, 8, 27, 64 }) and
            rsqrt.device.isCuda() and rsqrt.device_storage != null and
            approxF32(rsqrt_host.data[0], 1.0, 0.01) and
            approxF32(rsqrt_host.data[3], 0.5, 0.01) and
            relu_out.device.isCuda() and relu_out.device_storage != null and
            equalF32(relu_host.data, &.{ 0, 0, 0, 1 }) and
            threshold_zero_out.device.isCuda() and threshold_zero_out.device_storage != null and
            equalF32(threshold_zero_host.data, &.{ 0, 0, 0, 1 }) and
            threshold_clamp_out.device.isCuda() and threshold_clamp_out.device_storage != null and
            equalF32(threshold_clamp_host.data, &.{ 0.5, 0.5, 0.5, 1 }) and
            sigmoid_out.device.isCuda() and sigmoid_out.device_storage != null and
            approxF32(sigmoid_host.data[0], sigmoid_neg2, 0.01) and
            approxF32(sigmoid_host.data[3], sigmoid_pos1, 0.01) and
            silu_out.device.isCuda() and silu_out.device_storage != null and
            approxF32(silu_host.data[0], -2.0 * sigmoid_neg2, 0.01) and
            approxF32(silu_host.data[3], sigmoid_pos1, 0.01) and
            clipped_out.device.isCuda() and clipped_out.device_storage != null and
            equalF32(clipped_host.data, &.{ -0.5, -0.5, 0, 0.5 }) and
            clipped_array_out.device.isCuda() and clipped_array_out.device_storage != null and
            equalF32(clipped_array_host.data, &.{ -0.25, -0.25, 0, 0.75 }) and
            relu6_out.device.isCuda() and relu6_out.device_storage != null and
            equalF32(relu6_host.data, &.{ 0, 0, 0, 1 }) and
            hardsigmoid_out.device.isCuda() and hardsigmoid_out.device_storage != null and
            approxF32(hardsigmoid_host.data[0], @as(f32, 1.0) / 6.0, 0.01) and
            approxF32(hardsigmoid_host.data[3], @as(f32, 4.0) / 6.0, 0.01) and
            hardswish_out.device.isCuda() and hardswish_out.device_storage != null and
            approxF32(hardswish_host.data[0], -2.0 / 6.0, 0.01) and
            approxF32(hardswish_host.data[3], @as(f32, 4.0) / 6.0, 0.01) and
            softsign_out.device.isCuda() and softsign_out.device_storage != null and
            approxF32(softsign_host.data[0], -2.0 / 3.0, 0.01) and
            approxF32(softsign_host.data[3], 0.5, 0.01) and
            softshrink_out.device.isCuda() and softshrink_out.device_storage != null and
            approxF32(softshrink_host.data[0], -1.5, 0.01) and
            approxF32(softshrink_host.data[3], 0.5, 0.01) and
            mse_out.device.isCuda() and mse_out.device_storage != null and
            equalF32(mse_host.data, &.{ 4, 1, 0, 1 }) and
            l1_out.device.isCuda() and l1_out.device_storage != null and
            equalF32(l1_host.data, &.{ 2, 1, 0, 1 }) and
            smooth_l1_out.device.isCuda() and smooth_l1_out.device_storage != null and
            equalF32(smooth_l1_host.data, &.{ 1.5, 0.5, 0, 0.5 }) and
            huber_out.device.isCuda() and huber_out.device_storage != null and
            equalF32(huber_host.data, &.{ 1.5, 0.5, 0, 0.5 }) and
            elu_out.device.isCuda() and elu_out.device_storage != null and
            approxF32(elu_host.data[0], std.math.exp(@as(f32, -2.0)) - 1.0, 0.01) and
            approxF32(elu_host.data[3], 1.0, 0.01) and
            celu_out.device.isCuda() and celu_out.device_storage != null and
            approxF32(celu_host.data[0], 2.0 * (std.math.exp(@as(f32, -1.0)) - 1.0), 0.01) and
            approxF32(celu_host.data[3], 1.0, 0.01) and
            leaky_relu_out.device.isCuda() and leaky_relu_out.device_storage != null and
            approxF32(leaky_relu_host.data[0], -0.2, 0.01) and
            approxF32(leaky_relu_host.data[3], 1.0, 0.01) and
            maximum_out.device.isCuda() and maximum_out.device_storage != null and
            approxF32(maximum_host.data[0], -0.2, 0.01) and
            approxF32(maximum_host.data[3], 1.0, 0.01) and
            minimum_out.device.isCuda() and minimum_out.device_storage != null and
            approxF32(minimum_host.data[0], -2.0, 0.01) and
            approxF32(minimum_host.data[3], 0.1, 0.01);

        var addcmul_out = try addend.addcmul(lhs, rhs, 2.0);
        defer addcmul_out.deinit();
        var addcmul_host = try addcmul_out.cpu();
        defer addcmul_host.deinit();
        var addcdiv_out = try addend.addcdiv(lhs, rhs, 0.5);
        defer addcdiv_out.deinit();
        var addcdiv_host = try addcdiv_out.cpu();
        defer addcdiv_host.deinit();
        var lerp_scalar_out = try addend.lerpScalar(lhs, 0.5);
        defer lerp_scalar_out.deinit();
        var lerp_scalar_host = try lerp_scalar_out.cpu();
        defer lerp_scalar_host.deinit();
        var lerp_array_out = try addend.lerp(lhs, rhs);
        defer lerp_array_out.deinit();
        var lerp_array_host = try lerp_array_out.cpu();
        defer lerp_array_host.deinit();
        direct_ternary_ok = addcmul_out.device.isCuda() and addcmul_out.device_storage != null and
            equalF32(addcmul_host.data, &.{ 3, 5, 7, 9 }) and
            addcdiv_out.device.isCuda() and addcdiv_out.device_storage != null and
            approxF32(addcdiv_host.data[0], 1.5, 0.01) and
            approxF32(addcdiv_host.data[3], 3.0, 0.01) and
            lerp_scalar_out.device.isCuda() and lerp_scalar_out.device_storage != null and
            equalF32(lerp_scalar_host.data, &.{ 1, 1.5, 2, 2.5 }) and
            lerp_array_out.device.isCuda() and lerp_array_out.device_storage != null and
            equalF32(lerp_array_host.data, &.{ 1, 2, 3, 4 });

        var product = try lhs.matmul(rhs);
        defer product.deinit();
        var product_host = try product.cpu();
        defer product_host.deinit();
        const gemm_report = vx.axiom_cuda.lastCudaDeviceGemmReport();
        gemm_memref_fingerprint = gemm_report.memref_spec_fingerprint;
        direct_matmul_ok = product.device.isCuda() and
            gemm_report.valid() and
            gemm_report.memref_spec_fingerprint != 0 and
            equalF32(product_host.data, &.{ 3, 3, 7, 7 });

        var chained = try product.add(addend);
        defer chained.deinit();
        const chained_status_ok = chained.fusionStatus() == .cuda_matmul_add;
        var chained_host = try chained.cpu();
        defer chained_host.deinit();
        chained_matmul_add_ok = chained.device.isCuda() and equalF32(chained_host.data, &.{ 4, 4, 8, 8 });

        var reversed_chained = try addend.add(product);
        defer reversed_chained.deinit();
        var reversed_chained_host = try reversed_chained.cpu();
        defer reversed_chained_host.deinit();
        reversed_add_fusion_ok = reversed_chained.fusionStatus() == .cuda_matmul_add and equalF32(reversed_chained_host.data, &.{ 4, 4, 8, 8 });

        var reversed_sub_chained = try addend.sub(product);
        defer reversed_sub_chained.deinit();
        var reversed_sub_chained_host = try reversed_sub_chained.cpu();
        defer reversed_sub_chained_host.deinit();
        reversed_sub_fusion_ok = reversed_sub_chained.fusionStatus() == .cuda_matmul_rsub and equalF32(reversed_sub_chained_host.data, &.{ -2, -2, -6, -6 });
        var reversed_sub_exp = try reversed_sub_chained.exp();
        defer reversed_sub_exp.deinit();
        var reversed_sub_exp_host = try reversed_sub_exp.cpu();
        defer reversed_sub_exp_host.deinit();
        reversed_sub_fusion_ok = reversed_sub_fusion_ok and reversed_sub_exp.fusionStatus() == .cuda_matmul_rsub_exp and approxF32(reversed_sub_exp_host.data[0], std.math.exp(@as(f32, -2.0)), 0.01);

        var chained_sub = try product.sub(addend);
        defer chained_sub.deinit();
        const chained_sub_status_ok = chained_sub.fusionStatus() == .cuda_matmul_sub;
        var chained_sub_host = try chained_sub.cpu();
        defer chained_sub_host.deinit();
        chained_matmul_sub_ok = chained_sub.device.isCuda() and equalF32(chained_sub_host.data, &.{ 2, 2, 6, 6 });

        var chained_sqrt = try chained.sqrt();
        defer chained_sqrt.deinit();
        const chained_sqrt_status_ok = chained_sqrt.fusionStatus() == .cuda_matmul_add_sqrt;
        var chained_sqrt_host = try chained_sqrt.cpu();
        defer chained_sqrt_host.deinit();
        chained_sqrt_ok = chained_sqrt.device.isCuda() and approxF32(chained_sqrt_host.data[0], 2.0, 0.01);

        var chained_add_exp = try chained.exp();
        defer chained_add_exp.deinit();
        const chained_add_exp_status_ok = chained_add_exp.fusionStatus() == .cuda_matmul_add_exp;
        var chained_add_exp_host = try chained_add_exp.cpu();
        defer chained_add_exp_host.deinit();
        const chained_add_exp_ok = chained_add_exp.device.isCuda() and approxF32(chained_add_exp_host.data[0], std.math.exp(@as(f32, 4.0)), 2.0);

        var chained_exp_input = try chained_sub.addScalar(1.0);
        defer chained_exp_input.deinit();
        var chained_exp = try chained_exp_input.exp();
        defer chained_exp.deinit();
        var chained_exp_host = try chained_exp.cpu();
        defer chained_exp_host.deinit();
        chained_exp_ok = chained_exp.device.isCuda() and approxF32(chained_exp_host.data[0], std.math.exp(@as(f32, 3.0)), 0.25);
        pending_fusion_status_ok = chained_status_ok and chained_sub_status_ok and chained_sqrt_status_ok and chained_add_exp_status_ok and chained_add_exp_ok;

        var fused = try vx.matmulAdd(lhs, rhs, addend);
        defer fused.deinit();
        var fused_host = try fused.cpu();
        defer fused_host.deinit();
        direct_matmul_add_ok = fused.device.isCuda() and fused.device_storage != null and equalF32(fused_host.data, &.{ 4, 4, 8, 8 });
        var scaled_fused = (try vx.axiom_backend.executeMatmulAddScaled(f32, .cuda, lhs, rhs, addend, 2.0, -1.0)) orelse return error.BackendFailure;
        defer scaled_fused.deinit();
        var scaled_fused_host = try scaled_fused.cpu();
        defer scaled_fused_host.deinit();
        scaled_matmul_add_ok = scaled_fused.device.isCuda() and scaled_fused.device_storage != null and equalF32(scaled_fused_host.data, &.{ 5, 5, 13, 13 });

        var bf16_lhs = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{
            vx.BFloat16.fromF32(1),
            vx.BFloat16.fromF32(2),
            vx.BFloat16.fromF32(3),
            vx.BFloat16.fromF32(4),
        }, &.{ 2, 2 }, vx.cuda(0));
        defer bf16_lhs.deinit();
        var bf16_rhs = try vx.Array(vx.BFloat16).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer bf16_rhs.deinit();
        var bf16_addend = try vx.Array(vx.BFloat16).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer bf16_addend.deinit();
        var bf16_product = try bf16_lhs.matmul(bf16_rhs);
        defer bf16_product.deinit();
        var bf16_chained = try bf16_product.add(bf16_addend);
        defer bf16_chained.deinit();
        const bf16_chained_status_ok = bf16_chained.fusionStatus() == .cuda_matmul_add;
        var bf16_sqrt = try bf16_chained.sqrt();
        defer bf16_sqrt.deinit();
        const bf16_sqrt_status_ok = bf16_sqrt.fusionStatus() == .cuda_matmul_add_sqrt;
        var bf16_sqrt_host = try bf16_sqrt.cpu();
        defer bf16_sqrt_host.deinit();
        bf16_chained_sqrt_ok = bf16_sqrt.device.isCuda() and approxF32(bf16_sqrt_host.data[0].toF32(), 2.0, 0.05);

        var bf16_exp = try bf16_chained.exp();
        defer bf16_exp.deinit();
        const bf16_exp_status_ok = bf16_exp.fusionStatus() == .cuda_matmul_add_exp;
        var bf16_exp_host = try bf16_exp.cpu();
        defer bf16_exp_host.deinit();
        bf16_chained_exp_ok = bf16_exp.device.isCuda() and approxF32(bf16_exp_host.data[0].toF32(), std.math.exp(@as(f32, 4.0)), 2.0);

        var bf16_scaled = try bf16_chained.mulScalar(vx.BFloat16.fromF32(0.25));
        defer bf16_scaled.deinit();
        var bf16_scaled_host = try bf16_scaled.cpu();
        defer bf16_scaled_host.deinit();
        var bf16_row_bias = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(10), vx.BFloat16.fromF32(20) }, &.{2}, vx.cuda(0));
        defer bf16_row_bias.deinit();
        var bf16_row_broadcast = try bf16_lhs.add(bf16_row_bias);
        defer bf16_row_broadcast.deinit();
        var bf16_row_broadcast_host = try bf16_row_broadcast.cpu();
        defer bf16_row_broadcast_host.deinit();
        var bf16_column_bias = try vx.Array(vx.BFloat16).fromSliceOn(allocator, &.{ vx.BFloat16.fromF32(100), vx.BFloat16.fromF32(200) }, &.{ 2, 1 }, vx.cuda(0));
        defer bf16_column_bias.deinit();
        var bf16_column_broadcast = try bf16_lhs.add(bf16_column_bias);
        defer bf16_column_broadcast.deinit();
        var bf16_column_broadcast_host = try bf16_column_broadcast.cpu();
        defer bf16_column_broadcast_host.deinit();
        bf16_broadcast_ok = bf16_row_broadcast.device.isCuda() and bf16_row_broadcast.device_storage != null and
            approxF32(bf16_row_broadcast_host.data[0].toF32(), 11, 0.05) and
            approxF32(bf16_row_broadcast_host.data[1].toF32(), 22, 0.05) and
            approxF32(bf16_row_broadcast_host.data[2].toF32(), 13, 0.05) and
            approxF32(bf16_row_broadcast_host.data[3].toF32(), 24, 0.05) and
            bf16_column_broadcast.device.isCuda() and bf16_column_broadcast.device_storage != null and
            approxF32(bf16_column_broadcast_host.data[0].toF32(), 101, 0.05) and
            approxF32(bf16_column_broadcast_host.data[1].toF32(), 102, 0.05) and
            approxF32(bf16_column_broadcast_host.data[2].toF32(), 203, 0.05) and
            approxF32(bf16_column_broadcast_host.data[3].toF32(), 204, 0.05);
        var bf16_shifted = try bf16_lhs.subScalar(vx.BFloat16.fromF32(3));
        defer bf16_shifted.deinit();
        var bf16_relu = try bf16_shifted.relu();
        defer bf16_relu.deinit();
        var bf16_relu_host = try bf16_relu.cpu();
        defer bf16_relu_host.deinit();
        var bf16_sigmoid = try bf16_shifted.sigmoid();
        defer bf16_sigmoid.deinit();
        var bf16_sigmoid_host = try bf16_sigmoid.cpu();
        defer bf16_sigmoid_host.deinit();
        var bf16_softsign = try bf16_shifted.softsign();
        defer bf16_softsign.deinit();
        var bf16_softsign_host = try bf16_softsign.cpu();
        defer bf16_softsign_host.deinit();
        var bf16_clip = try bf16_shifted.clip(vx.BFloat16.fromF32(-0.5), vx.BFloat16.fromF32(0.5));
        defer bf16_clip.deinit();
        var bf16_clip_host = try bf16_clip.cpu();
        defer bf16_clip_host.deinit();
        var bf16_pow_zero = try bf16_lhs.powScalar(vx.BFloat16.fromF32(0));
        defer bf16_pow_zero.deinit();
        var bf16_pow_zero_host = try bf16_pow_zero.cpu();
        defer bf16_pow_zero_host.deinit();
        var bf16_pow_recip = try bf16_lhs.powScalar(vx.BFloat16.fromF32(-1));
        defer bf16_pow_recip.deinit();
        var bf16_pow_recip_host = try bf16_pow_recip.cpu();
        defer bf16_pow_recip_host.deinit();
        var bf16_pow_sqrt = try bf16_lhs.powScalar(vx.BFloat16.fromF32(0.5));
        defer bf16_pow_sqrt.deinit();
        var bf16_pow_sqrt_host = try bf16_pow_sqrt.cpu();
        defer bf16_pow_sqrt_host.deinit();
        var bf16_pow_rsqrt = try bf16_lhs.powScalar(vx.BFloat16.fromF32(-0.5));
        defer bf16_pow_rsqrt.deinit();
        var bf16_pow_rsqrt_host = try bf16_pow_rsqrt.cpu();
        defer bf16_pow_rsqrt_host.deinit();
        var bf16_pow_cube = try bf16_lhs.powScalar(vx.BFloat16.fromF32(3));
        defer bf16_pow_cube.deinit();
        var bf16_pow_cube_host = try bf16_pow_cube.cpu();
        defer bf16_pow_cube_host.deinit();
        bf16_scalar_mul_ok = bf16_scaled.device.isCuda() and bf16_scaled.device_storage != null and approxF32(bf16_scaled_host.data[0].toF32(), 1.0, 0.05) and
            bf16_relu.device.isCuda() and bf16_relu.device_storage != null and
            approxF32(bf16_relu_host.data[0].toF32(), 0, 0.05) and
            approxF32(bf16_relu_host.data[3].toF32(), 1, 0.05) and
            bf16_sigmoid.device.isCuda() and bf16_sigmoid.device_storage != null and
            approxF32(bf16_sigmoid_host.data[0].toF32(), @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, 2.0))), 0.05) and
            bf16_softsign.device.isCuda() and bf16_softsign.device_storage != null and
            approxF32(bf16_softsign_host.data[0].toF32(), -2.0 / 3.0, 0.05) and
            bf16_clip.device.isCuda() and bf16_clip.device_storage != null and
            approxF32(bf16_clip_host.data[0].toF32(), -0.5, 0.05) and
            approxF32(bf16_clip_host.data[3].toF32(), 0.5, 0.05) and
            bf16_pow_zero.device.isCuda() and bf16_pow_zero.device_storage != null and
            approxF32(bf16_pow_zero_host.data[0].toF32(), 1, 0.05) and
            bf16_pow_recip.device.isCuda() and bf16_pow_recip.device_storage != null and
            approxF32(bf16_pow_recip_host.data[3].toF32(), 0.25, 0.05) and
            bf16_pow_sqrt.device.isCuda() and bf16_pow_sqrt.device_storage != null and
            approxF32(bf16_pow_sqrt_host.data[3].toF32(), 2, 0.05) and
            bf16_pow_rsqrt.device.isCuda() and bf16_pow_rsqrt.device_storage != null and
            approxF32(bf16_pow_rsqrt_host.data[3].toF32(), 0.5, 0.05) and
            bf16_pow_cube.device.isCuda() and bf16_pow_cube.device_storage != null and
            approxF32(bf16_pow_cube_host.data[3].toF32(), 64, 0.5);
        var bf16_transpose = try bf16_lhs.transpose();
        defer bf16_transpose.deinit();
        var bf16_transpose_host = try bf16_transpose.cpu();
        defer bf16_transpose_host.deinit();
        bf16_transpose_ok = bf16_transpose.device.isCuda() and bf16_transpose.device_storage != null and
            std.mem.eql(usize, bf16_transpose_host.shape, &.{ 2, 2 }) and
            approxF32(bf16_transpose_host.data[0].toF32(), 1, 0.05) and
            approxF32(bf16_transpose_host.data[1].toF32(), 3, 0.05) and
            approxF32(bf16_transpose_host.data[2].toF32(), 2, 0.05) and
            approxF32(bf16_transpose_host.data[3].toF32(), 4, 0.05);

        var bf16_softmax = try bf16_lhs.softmax(1);
        defer bf16_softmax.deinit();
        var bf16_softmax_host = try bf16_softmax.cpu();
        defer bf16_softmax_host.deinit();
        const bf16_softmax_denom = std.math.exp(@as(f32, -1)) + 1.0;
        bf16_softmax_ok = bf16_softmax.device.isCuda() and bf16_softmax.device_storage != null and
            approxF32(bf16_softmax_host.data[0].toF32(), std.math.exp(@as(f32, -1)) / bf16_softmax_denom, 0.05) and
            approxF32(bf16_softmax_host.data[1].toF32(), 1.0 / bf16_softmax_denom, 0.05);
        var bf16_log_softmax = try bf16_lhs.logSoftmax(1);
        defer bf16_log_softmax.deinit();
        var bf16_log_softmax_host = try bf16_log_softmax.cpu();
        defer bf16_log_softmax_host.deinit();
        const bf16_log_softmax_denom = std.math.log(f32, std.math.e, bf16_softmax_denom);
        bf16_log_softmax_ok = bf16_log_softmax.device.isCuda() and bf16_log_softmax.device_storage != null and
            approxF32(bf16_log_softmax_host.data[0].toF32(), -1.0 - bf16_log_softmax_denom, 0.08) and
            approxF32(bf16_log_softmax_host.data[1].toF32(), -bf16_log_softmax_denom, 0.08);

        var bf16_row_sum = try bf16_lhs.sum(1, false);
        defer bf16_row_sum.deinit();
        var bf16_row_sum_host = try bf16_row_sum.cpu();
        defer bf16_row_sum_host.deinit();
        var bf16_col_sum_keep = try bf16_lhs.sum(0, true);
        defer bf16_col_sum_keep.deinit();
        var bf16_col_sum_keep_host = try bf16_col_sum_keep.cpu();
        defer bf16_col_sum_keep_host.deinit();
        var bf16_row_prod = try bf16_lhs.prod(1, false);
        defer bf16_row_prod.deinit();
        var bf16_row_prod_host = try bf16_row_prod.cpu();
        defer bf16_row_prod_host.deinit();
        var bf16_col_min = try bf16_lhs.min(0, false);
        defer bf16_col_min.deinit();
        var bf16_col_min_host = try bf16_col_min.cpu();
        defer bf16_col_min_host.deinit();
        var bf16_col_max = try bf16_lhs.max(0, false);
        defer bf16_col_max.deinit();
        var bf16_col_max_host = try bf16_col_max.cpu();
        defer bf16_col_max_host.deinit();
        bf16_reduction_ok = bf16_row_sum.device.isCuda() and bf16_row_sum.device_storage != null and
            approxF32(bf16_row_sum_host.data[0].toF32(), 3, 0.05) and
            approxF32(bf16_row_sum_host.data[1].toF32(), 7, 0.05) and
            bf16_col_sum_keep.device.isCuda() and bf16_col_sum_keep.device_storage != null and
            std.mem.eql(usize, bf16_col_sum_keep_host.shape, &.{ 1, 2 }) and
            approxF32(bf16_col_sum_keep_host.data[0].toF32(), 4, 0.05) and
            approxF32(bf16_col_sum_keep_host.data[1].toF32(), 6, 0.05) and
            bf16_row_prod.device.isCuda() and bf16_row_prod.device_storage != null and
            approxF32(bf16_row_prod_host.data[0].toF32(), 2, 0.05) and
            approxF32(bf16_row_prod_host.data[1].toF32(), 12, 0.05) and
            bf16_col_min.device.isCuda() and bf16_col_min.device_storage != null and
            approxF32(bf16_col_min_host.data[0].toF32(), 1, 0.05) and
            approxF32(bf16_col_min_host.data[1].toF32(), 2, 0.05) and
            bf16_col_max.device.isCuda() and bf16_col_max.device_storage != null and
            approxF32(bf16_col_max_host.data[0].toF32(), 3, 0.05) and
            approxF32(bf16_col_max_host.data[1].toF32(), 4, 0.05);
        pending_fusion_status_ok = pending_fusion_status_ok and bf16_chained_status_ok and bf16_sqrt_status_ok and bf16_exp_status_ok;

        var f16_lhs = try vx.Array(f16).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, vx.cuda(0));
        defer f16_lhs.deinit();
        var f16_row_bias = try vx.Array(f16).fromSliceOn(allocator, &.{ 10, 20 }, &.{2}, vx.cuda(0));
        defer f16_row_bias.deinit();
        var f16_row_broadcast = try f16_lhs.add(f16_row_bias);
        defer f16_row_broadcast.deinit();
        var f16_row_broadcast_host = try f16_row_broadcast.cpu();
        defer f16_row_broadcast_host.deinit();
        var f16_column_bias = try vx.Array(f16).fromSliceOn(allocator, &.{ 100, 200 }, &.{ 2, 1 }, vx.cuda(0));
        defer f16_column_bias.deinit();
        var f16_column_broadcast = try f16_lhs.add(f16_column_bias);
        defer f16_column_broadcast.deinit();
        var f16_column_broadcast_host = try f16_column_broadcast.cpu();
        defer f16_column_broadcast_host.deinit();
        f16_broadcast_ok = f16_row_broadcast.device.isCuda() and f16_row_broadcast.device_storage != null and
            approxF16(f16_row_broadcast_host.data[0], 11, 0.05) and
            approxF16(f16_row_broadcast_host.data[1], 22, 0.05) and
            approxF16(f16_row_broadcast_host.data[2], 13, 0.05) and
            approxF16(f16_row_broadcast_host.data[3], 24, 0.05) and
            f16_column_broadcast.device.isCuda() and f16_column_broadcast.device_storage != null and
            approxF16(f16_column_broadcast_host.data[0], 101, 0.05) and
            approxF16(f16_column_broadcast_host.data[1], 102, 0.05) and
            approxF16(f16_column_broadcast_host.data[2], 203, 0.05) and
            approxF16(f16_column_broadcast_host.data[3], 204, 0.05);
        var f16_transpose = try f16_lhs.transpose();
        defer f16_transpose.deinit();
        var f16_transpose_host = try f16_transpose.cpu();
        defer f16_transpose_host.deinit();
        f16_transpose_ok = f16_transpose.device.isCuda() and f16_transpose.device_storage != null and
            std.mem.eql(usize, f16_transpose_host.shape, &.{ 2, 2 }) and
            approxF16(f16_transpose_host.data[0], 1, 0.05) and
            approxF16(f16_transpose_host.data[1], 3, 0.05) and
            approxF16(f16_transpose_host.data[2], 2, 0.05) and
            approxF16(f16_transpose_host.data[3], 4, 0.05);
        var f16_softmax = try f16_lhs.softmax(1);
        defer f16_softmax.deinit();
        var f16_softmax_host = try f16_softmax.cpu();
        defer f16_softmax_host.deinit();
        const f16_softmax_denom = std.math.exp(@as(f32, -1)) + 1.0;
        f16_softmax_ok = f16_softmax.device.isCuda() and f16_softmax.device_storage != null and
            approxF16(f16_softmax_host.data[0], std.math.exp(@as(f32, -1)) / f16_softmax_denom, 0.05) and
            approxF16(f16_softmax_host.data[1], 1.0 / f16_softmax_denom, 0.05);
        var f16_log_softmax = try f16_lhs.logSoftmax(1);
        defer f16_log_softmax.deinit();
        var f16_log_softmax_host = try f16_log_softmax.cpu();
        defer f16_log_softmax_host.deinit();
        const f16_log_softmax_denom = std.math.log(f32, std.math.e, f16_softmax_denom);
        f16_log_softmax_ok = f16_log_softmax.device.isCuda() and f16_log_softmax.device_storage != null and
            approxF16(f16_log_softmax_host.data[0], -1.0 - f16_log_softmax_denom, 0.08) and
            approxF16(f16_log_softmax_host.data[1], -f16_log_softmax_denom, 0.08);
        var f16_shifted = try f16_lhs.subScalar(@as(f16, 3));
        defer f16_shifted.deinit();
        var f16_relu = try f16_shifted.relu();
        defer f16_relu.deinit();
        var f16_relu_host = try f16_relu.cpu();
        defer f16_relu_host.deinit();
        var f16_sigmoid = try f16_shifted.sigmoid();
        defer f16_sigmoid.deinit();
        var f16_sigmoid_host = try f16_sigmoid.cpu();
        defer f16_sigmoid_host.deinit();
        var f16_softsign = try f16_shifted.softsign();
        defer f16_softsign.deinit();
        var f16_softsign_host = try f16_softsign.cpu();
        defer f16_softsign_host.deinit();
        var f16_clip = try f16_shifted.clip(@as(f16, -0.5), @as(f16, 0.5));
        defer f16_clip.deinit();
        var f16_clip_host = try f16_clip.cpu();
        defer f16_clip_host.deinit();
        var f16_pow_zero = try f16_lhs.powScalar(@as(f16, 0));
        defer f16_pow_zero.deinit();
        var f16_pow_zero_host = try f16_pow_zero.cpu();
        defer f16_pow_zero_host.deinit();
        var f16_pow_recip = try f16_lhs.powScalar(@as(f16, -1));
        defer f16_pow_recip.deinit();
        var f16_pow_recip_host = try f16_pow_recip.cpu();
        defer f16_pow_recip_host.deinit();
        var f16_pow_sqrt = try f16_lhs.powScalar(@as(f16, 0.5));
        defer f16_pow_sqrt.deinit();
        var f16_pow_sqrt_host = try f16_pow_sqrt.cpu();
        defer f16_pow_sqrt_host.deinit();
        var f16_pow_rsqrt = try f16_lhs.powScalar(@as(f16, -0.5));
        defer f16_pow_rsqrt.deinit();
        var f16_pow_rsqrt_host = try f16_pow_rsqrt.cpu();
        defer f16_pow_rsqrt_host.deinit();
        var f16_pow_cube = try f16_lhs.powScalar(@as(f16, 3));
        defer f16_pow_cube.deinit();
        var f16_pow_cube_host = try f16_pow_cube.cpu();
        defer f16_pow_cube_host.deinit();
        var f16_loss_target = try vx.Array(f16).zerosOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer f16_loss_target.deinit();
        var f16_mse = try f16_shifted.mseLoss(f16_loss_target, .none);
        defer f16_mse.deinit();
        var f16_mse_host = try f16_mse.cpu();
        defer f16_mse_host.deinit();
        var f16_l1 = try f16_shifted.l1Loss(f16_loss_target, .none);
        defer f16_l1.deinit();
        var f16_l1_host = try f16_l1.cpu();
        defer f16_l1_host.deinit();
        var f16_smooth_l1 = try f16_shifted.smoothL1Loss(f16_loss_target, @as(f16, 1), .none);
        defer f16_smooth_l1.deinit();
        var f16_smooth_l1_host = try f16_smooth_l1.cpu();
        defer f16_smooth_l1_host.deinit();
        var f16_huber = try f16_shifted.huberLoss(f16_loss_target, @as(f16, 1), .none);
        defer f16_huber.deinit();
        var f16_huber_host = try f16_huber.cpu();
        defer f16_huber_host.deinit();
        var f16_row_sum = try f16_lhs.sum(1, false);
        defer f16_row_sum.deinit();
        var f16_row_sum_host = try f16_row_sum.cpu();
        defer f16_row_sum_host.deinit();
        var f16_col_sum_keep = try f16_lhs.sum(0, true);
        defer f16_col_sum_keep.deinit();
        var f16_col_sum_keep_host = try f16_col_sum_keep.cpu();
        defer f16_col_sum_keep_host.deinit();
        var f16_row_prod = try f16_lhs.prod(1, false);
        defer f16_row_prod.deinit();
        var f16_row_prod_host = try f16_row_prod.cpu();
        defer f16_row_prod_host.deinit();
        var f16_col_min = try f16_lhs.min(0, false);
        defer f16_col_min.deinit();
        var f16_col_min_host = try f16_col_min.cpu();
        defer f16_col_min_host.deinit();
        var f16_col_max = try f16_lhs.max(0, false);
        defer f16_col_max.deinit();
        var f16_col_max_host = try f16_col_max.cpu();
        defer f16_col_max_host.deinit();
        f16_reduction_ok = f16_row_sum.device.isCuda() and f16_row_sum.device_storage != null and
            approxF16(f16_row_sum_host.data[0], 3, 0.05) and
            approxF16(f16_row_sum_host.data[1], 7, 0.05) and
            f16_col_sum_keep.device.isCuda() and f16_col_sum_keep.device_storage != null and
            std.mem.eql(usize, f16_col_sum_keep_host.shape, &.{ 1, 2 }) and
            approxF16(f16_col_sum_keep_host.data[0], 4, 0.05) and
            approxF16(f16_col_sum_keep_host.data[1], 6, 0.05) and
            f16_row_prod.device.isCuda() and f16_row_prod.device_storage != null and
            approxF16(f16_row_prod_host.data[0], 2, 0.05) and
            approxF16(f16_row_prod_host.data[1], 12, 0.05) and
            f16_col_min.device.isCuda() and f16_col_min.device_storage != null and
            approxF16(f16_col_min_host.data[0], 1, 0.05) and
            approxF16(f16_col_min_host.data[1], 2, 0.05) and
            f16_col_max.device.isCuda() and f16_col_max.device_storage != null and
            approxF16(f16_col_max_host.data[0], 3, 0.05) and
            approxF16(f16_col_max_host.data[1], 4, 0.05);

        f16_activation_ok = f16_relu.device.isCuda() and f16_relu.device_storage != null and
            approxF16(f16_relu_host.data[0], 0, 0.05) and
            approxF16(f16_relu_host.data[3], 1, 0.05) and
            f16_sigmoid.device.isCuda() and f16_sigmoid.device_storage != null and
            approxF16(f16_sigmoid_host.data[0], @as(f32, 1.0) / (@as(f32, 1.0) + std.math.exp(@as(f32, 2.0))), 0.05) and
            f16_softsign.device.isCuda() and f16_softsign.device_storage != null and
            approxF16(f16_softsign_host.data[0], -2.0 / 3.0, 0.05) and
            f16_clip.device.isCuda() and f16_clip.device_storage != null and
            approxF16(f16_clip_host.data[0], -0.5, 0.05) and
            approxF16(f16_clip_host.data[3], 0.5, 0.05) and
            f16_pow_zero.device.isCuda() and f16_pow_zero.device_storage != null and
            approxF16(f16_pow_zero_host.data[0], 1, 0.05) and
            f16_pow_recip.device.isCuda() and f16_pow_recip.device_storage != null and
            approxF16(f16_pow_recip_host.data[3], 0.25, 0.05) and
            f16_pow_sqrt.device.isCuda() and f16_pow_sqrt.device_storage != null and
            approxF16(f16_pow_sqrt_host.data[3], 2, 0.05) and
            f16_pow_rsqrt.device.isCuda() and f16_pow_rsqrt.device_storage != null and
            approxF16(f16_pow_rsqrt_host.data[3], 0.5, 0.05) and
            f16_pow_cube.device.isCuda() and f16_pow_cube.device_storage != null and
            approxF16(f16_pow_cube_host.data[3], 64, 0.5) and
            f16_mse.device.isCuda() and f16_mse.device_storage != null and
            approxF16(f16_mse_host.data[0], 4, 0.05) and
            f16_l1.device.isCuda() and f16_l1.device_storage != null and
            approxF16(f16_l1_host.data[0], 2, 0.05) and
            f16_smooth_l1.device.isCuda() and f16_smooth_l1.device_storage != null and
            approxF16(f16_smooth_l1_host.data[0], 1.5, 0.05) and
            f16_huber.device.isCuda() and f16_huber.device_storage != null and
            approxF16(f16_huber_host.data[0], 1.5, 0.05);

        var f64_lhs = try vx.Array(f64).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, vx.cuda(0));
        defer f64_lhs.deinit();
        var f64_rhs = try vx.Array(f64).onesOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer f64_rhs.deinit();
        var f64_product = try f64_lhs.matmul(f64_rhs);
        defer f64_product.deinit();
        var f64_product_host = try f64_product.cpu();
        defer f64_product_host.deinit();
        var f64_target_product = (try vx.axiom_backend.executeMatmul(f64, .cuda, f64_lhs, f64_rhs)) orelse return error.BackendFailure;
        defer f64_target_product.deinit();
        var f64_target_host = try f64_target_product.cpu();
        defer f64_target_host.deinit();
        f64_matmul_ok = f64_product.device.isCuda() and
            f64_product.fusionStatus() == .cuda_matmul and
            equalF64(f64_product_host.data, &.{ 3, 3, 7, 7 }) and
            f64_target_product.device.isCuda() and
            f64_target_product.device_storage != null and
            equalF64(f64_target_host.data, &.{ 3, 3, 7, 7 });

        var f64_transpose = try f64_lhs.transpose();
        defer f64_transpose.deinit();
        var f64_transpose_host = try f64_transpose.cpu();
        defer f64_transpose_host.deinit();
        f64_transpose_ok = f64_transpose.device.isCuda() and f64_transpose.device_storage != null and
            std.mem.eql(usize, f64_transpose_host.shape, &.{ 2, 2 }) and
            equalF64(f64_transpose_host.data, &.{ 1, 3, 2, 4 });

        var f64_row_bias = try vx.Array(f64).fromSliceOn(allocator, &.{ 10, 20 }, &.{2}, vx.cuda(0));
        defer f64_row_bias.deinit();
        var f64_row_broadcast = try f64_lhs.add(f64_row_bias);
        defer f64_row_broadcast.deinit();
        var f64_row_broadcast_host = try f64_row_broadcast.cpu();
        defer f64_row_broadcast_host.deinit();
        var f64_column_bias = try vx.Array(f64).fromSliceOn(allocator, &.{ 100, 200 }, &.{ 2, 1 }, vx.cuda(0));
        defer f64_column_bias.deinit();
        var f64_column_broadcast = try f64_lhs.add(f64_column_bias);
        defer f64_column_broadcast.deinit();
        var f64_column_broadcast_host = try f64_column_broadcast.cpu();
        defer f64_column_broadcast_host.deinit();
        f64_broadcast_ok = f64_row_broadcast.device.isCuda() and f64_row_broadcast.device_storage != null and
            equalF64(f64_row_broadcast_host.data, &.{ 11, 22, 13, 24 }) and
            f64_column_broadcast.device.isCuda() and f64_column_broadcast.device_storage != null and
            equalF64(f64_column_broadcast_host.data, &.{ 101, 102, 203, 204 });

        var f64_row_sum = try f64_lhs.sum(1, false);
        defer f64_row_sum.deinit();
        var f64_row_sum_host = try f64_row_sum.cpu();
        defer f64_row_sum_host.deinit();
        var f64_col_sum_keep = try f64_lhs.sum(0, true);
        defer f64_col_sum_keep.deinit();
        var f64_col_sum_keep_host = try f64_col_sum_keep.cpu();
        defer f64_col_sum_keep_host.deinit();
        var f64_row_prod = try f64_lhs.prod(1, false);
        defer f64_row_prod.deinit();
        var f64_row_prod_host = try f64_row_prod.cpu();
        defer f64_row_prod_host.deinit();
        var f64_col_min = try f64_lhs.min(0, false);
        defer f64_col_min.deinit();
        var f64_col_min_host = try f64_col_min.cpu();
        defer f64_col_min_host.deinit();
        var f64_col_max = try f64_lhs.max(0, false);
        defer f64_col_max.deinit();
        var f64_col_max_host = try f64_col_max.cpu();
        defer f64_col_max_host.deinit();
        f64_reduction_ok = f64_row_sum.device.isCuda() and f64_row_sum.device_storage != null and
            equalF64(f64_row_sum_host.data, &.{ 3, 7 }) and
            f64_col_sum_keep.device.isCuda() and f64_col_sum_keep.device_storage != null and
            std.mem.eql(usize, f64_col_sum_keep_host.shape, &.{ 1, 2 }) and
            equalF64(f64_col_sum_keep_host.data, &.{ 4, 6 }) and
            f64_row_prod.device.isCuda() and f64_row_prod.device_storage != null and
            equalF64(f64_row_prod_host.data, &.{ 2, 12 }) and
            f64_col_min.device.isCuda() and f64_col_min.device_storage != null and
            equalF64(f64_col_min_host.data, &.{ 1, 2 }) and
            f64_col_max.device.isCuda() and f64_col_max.device_storage != null and
            equalF64(f64_col_max_host.data, &.{ 3, 4 });

        var f64_softmax_row = try f64_lhs.softmax(1);
        defer f64_softmax_row.deinit();
        var f64_softmax_row_host = try f64_softmax_row.cpu();
        defer f64_softmax_row_host.deinit();
        var f64_softmax_col = try f64_lhs.softmax(0);
        defer f64_softmax_col.deinit();
        var f64_softmax_col_host = try f64_softmax_col.cpu();
        defer f64_softmax_col_host.deinit();
        const f64_row_denom = std.math.exp(@as(f64, -1)) + 1.0;
        const f64_col_denom = std.math.exp(@as(f64, -2)) + 1.0;
        f64_softmax_ok = f64_softmax_row.device.isCuda() and f64_softmax_row.device_storage != null and
            approxF64(f64_softmax_row_host.data[0], std.math.exp(@as(f64, -1)) / f64_row_denom, 0.01) and
            approxF64(f64_softmax_row_host.data[1], 1.0 / f64_row_denom, 0.01) and
            f64_softmax_col.device.isCuda() and f64_softmax_col.device_storage != null and
            approxF64(f64_softmax_col_host.data[0], std.math.exp(@as(f64, -2)) / f64_col_denom, 0.01) and
            approxF64(f64_softmax_col_host.data[2], 1.0 / f64_col_denom, 0.01);

        var f64_log_softmax_row = try f64_lhs.logSoftmax(1);
        defer f64_log_softmax_row.deinit();
        var f64_log_softmax_row_host = try f64_log_softmax_row.cpu();
        defer f64_log_softmax_row_host.deinit();
        var f64_log_softmax_col = try f64_lhs.logSoftmax(0);
        defer f64_log_softmax_col.deinit();
        var f64_log_softmax_col_host = try f64_log_softmax_col.cpu();
        defer f64_log_softmax_col_host.deinit();
        const f64_row_log_denom = std.math.log(f64, std.math.e, f64_row_denom);
        const f64_col_log_denom = std.math.log(f64, std.math.e, f64_col_denom);
        f64_log_softmax_ok = f64_log_softmax_row.device.isCuda() and f64_log_softmax_row.device_storage != null and
            approxF64(f64_log_softmax_row_host.data[0], -1.0 - f64_row_log_denom, 0.03) and
            approxF64(f64_log_softmax_row_host.data[1], -f64_row_log_denom, 0.03) and
            f64_log_softmax_col.device.isCuda() and f64_log_softmax_col.device_storage != null and
            approxF64(f64_log_softmax_col_host.data[0], -2.0 - f64_col_log_denom, 0.03) and
            approxF64(f64_log_softmax_col_host.data[2], -f64_col_log_denom, 0.03);

        var f64_sum = try f64_lhs.add(f64_rhs);
        defer f64_sum.deinit();
        var f64_sum_host = try f64_sum.cpu();
        defer f64_sum_host.deinit();
        var f64_scaled = try f64_lhs.mulScalar(0.5);
        defer f64_scaled.deinit();
        var f64_scaled_host = try f64_scaled.cpu();
        defer f64_scaled_host.deinit();
        var f64_neg = try f64_lhs.neg();
        defer f64_neg.deinit();
        var f64_neg_host = try f64_neg.cpu();
        defer f64_neg_host.deinit();
        var f64_abs = try f64_neg.abs();
        defer f64_abs.deinit();
        var f64_abs_host = try f64_abs.cpu();
        defer f64_abs_host.deinit();
        var f64_square = try f64_lhs.square();
        defer f64_square.deinit();
        var f64_square_host = try f64_square.cpu();
        defer f64_square_host.deinit();
        var f64_pow_zero = try f64_lhs.powScalar(0);
        defer f64_pow_zero.deinit();
        var f64_pow_zero_host = try f64_pow_zero.cpu();
        defer f64_pow_zero_host.deinit();
        var f64_pow_recip = try f64_lhs.powScalar(-1);
        defer f64_pow_recip.deinit();
        var f64_pow_recip_host = try f64_pow_recip.cpu();
        defer f64_pow_recip_host.deinit();
        var f64_pow_sqrt = try f64_lhs.powScalar(0.5);
        defer f64_pow_sqrt.deinit();
        var f64_pow_sqrt_host = try f64_pow_sqrt.cpu();
        defer f64_pow_sqrt_host.deinit();
        var f64_pow_rsqrt = try f64_lhs.powScalar(-0.5);
        defer f64_pow_rsqrt.deinit();
        var f64_pow_rsqrt_host = try f64_pow_rsqrt.cpu();
        defer f64_pow_rsqrt_host.deinit();
        var f64_pow_cube = try f64_lhs.powScalar(3);
        defer f64_pow_cube.deinit();
        var f64_pow_cube_host = try f64_pow_cube.cpu();
        defer f64_pow_cube_host.deinit();
        var f64_sqrt = try f64_product.sqrt();
        defer f64_sqrt.deinit();
        var f64_sqrt_host = try f64_sqrt.cpu();
        defer f64_sqrt_host.deinit();
        var f64_rsqrt = try f64_lhs.rsqrt();
        defer f64_rsqrt.deinit();
        var f64_rsqrt_host = try f64_rsqrt.cpu();
        defer f64_rsqrt_host.deinit();
        var f64_exp = try f64_sum.exp();
        defer f64_exp.deinit();
        var f64_exp_host = try f64_exp.cpu();
        defer f64_exp_host.deinit();
        var f64_shifted = try f64_lhs.subScalar(3);
        defer f64_shifted.deinit();
        var f64_relu = try f64_shifted.relu();
        defer f64_relu.deinit();
        var f64_relu_host = try f64_relu.cpu();
        defer f64_relu_host.deinit();
        var f64_threshold = try f64_shifted.threshold(0.5, 0.5);
        defer f64_threshold.deinit();
        var f64_threshold_host = try f64_threshold.cpu();
        defer f64_threshold_host.deinit();
        var f64_sigmoid = try f64_shifted.sigmoid();
        defer f64_sigmoid.deinit();
        var f64_sigmoid_host = try f64_sigmoid.cpu();
        defer f64_sigmoid_host.deinit();
        var f64_softsign = try f64_shifted.softsign();
        defer f64_softsign.deinit();
        var f64_softsign_host = try f64_softsign.cpu();
        defer f64_softsign_host.deinit();
        var f64_clip = try f64_shifted.clip(-0.5, 0.5);
        defer f64_clip.deinit();
        var f64_clip_host = try f64_clip.cpu();
        defer f64_clip_host.deinit();
        var f64_scaled_for_max = try f64_shifted.mulScalar(0.1);
        defer f64_scaled_for_max.deinit();
        var f64_maximum = try f64_shifted.maximum(f64_scaled_for_max);
        defer f64_maximum.deinit();
        var f64_maximum_host = try f64_maximum.cpu();
        defer f64_maximum_host.deinit();
        var f64_leaky = try f64_shifted.leakyRelu(0.1);
        defer f64_leaky.deinit();
        var f64_leaky_host = try f64_leaky.cpu();
        defer f64_leaky_host.deinit();
        var f64_lerp = try f64_rhs.lerpScalar(f64_lhs, 0.5);
        defer f64_lerp.deinit();
        var f64_lerp_host = try f64_lerp.cpu();
        defer f64_lerp_host.deinit();
        var f64_addcmul = try f64_rhs.addcmul(f64_lhs, f64_rhs, 2.0);
        defer f64_addcmul.deinit();
        var f64_addcmul_host = try f64_addcmul.cpu();
        defer f64_addcmul_host.deinit();
        var f64_addcdiv = try f64_rhs.addcdiv(f64_lhs, f64_rhs, 0.5);
        defer f64_addcdiv.deinit();
        var f64_addcdiv_host = try f64_addcdiv.cpu();
        defer f64_addcdiv_host.deinit();
        var f64_silu = try f64_shifted.silu();
        defer f64_silu.deinit();
        var f64_silu_host = try f64_silu.cpu();
        defer f64_silu_host.deinit();
        var f64_hardsigmoid = try f64_shifted.hardsigmoid();
        defer f64_hardsigmoid.deinit();
        var f64_hardsigmoid_host = try f64_hardsigmoid.cpu();
        defer f64_hardsigmoid_host.deinit();
        var f64_hardswish = try f64_shifted.hardswish();
        defer f64_hardswish.deinit();
        var f64_hardswish_host = try f64_hardswish.cpu();
        defer f64_hardswish_host.deinit();
        var f64_relu6 = try f64_shifted.relu6();
        defer f64_relu6.deinit();
        var f64_relu6_host = try f64_relu6.cpu();
        defer f64_relu6_host.deinit();
        var f64_clip_min_values = try vx.Array(f64).fullOn(allocator, &.{ 2, 2 }, -0.25, vx.cuda(0));
        defer f64_clip_min_values.deinit();
        var f64_clip_max_values = try vx.Array(f64).fullOn(allocator, &.{ 2, 2 }, 0.75, vx.cuda(0));
        defer f64_clip_max_values.deinit();
        var f64_clip_array = try f64_shifted.clipArray(f64_clip_min_values, f64_clip_max_values);
        defer f64_clip_array.deinit();
        var f64_clip_array_host = try f64_clip_array.cpu();
        defer f64_clip_array_host.deinit();
        var f64_elu = try f64_shifted.elu(1.0);
        defer f64_elu.deinit();
        var f64_elu_host = try f64_elu.cpu();
        defer f64_elu_host.deinit();
        var f64_celu = try f64_shifted.celu(2.0);
        defer f64_celu.deinit();
        var f64_celu_host = try f64_celu.cpu();
        defer f64_celu_host.deinit();
        var f64_softshrink = try f64_shifted.softshrink(0.5);
        defer f64_softshrink.deinit();
        var f64_softshrink_host = try f64_softshrink.cpu();
        defer f64_softshrink_host.deinit();
        var f64_loss_target = try vx.Array(f64).zerosOn(allocator, &.{ 2, 2 }, vx.cuda(0));
        defer f64_loss_target.deinit();
        var f64_mse = try f64_shifted.mseLoss(f64_loss_target, .none);
        defer f64_mse.deinit();
        var f64_mse_host = try f64_mse.cpu();
        defer f64_mse_host.deinit();
        var f64_l1 = try f64_shifted.l1Loss(f64_loss_target, .none);
        defer f64_l1.deinit();
        var f64_l1_host = try f64_l1.cpu();
        defer f64_l1_host.deinit();
        var f64_smooth_l1 = try f64_shifted.smoothL1Loss(f64_loss_target, 1.0, .none);
        defer f64_smooth_l1.deinit();
        var f64_smooth_l1_host = try f64_smooth_l1.cpu();
        defer f64_smooth_l1_host.deinit();
        var f64_huber = try f64_shifted.huberLoss(f64_loss_target, 1.0, .none);
        defer f64_huber.deinit();
        var f64_huber_host = try f64_huber.cpu();
        defer f64_huber_host.deinit();
        f64_elementwise_ok = f64_sum.device.isCuda() and f64_sum.device_storage != null and
            equalF64(f64_sum_host.data, &.{ 2, 3, 4, 5 }) and
            f64_scaled.device.isCuda() and f64_scaled.device_storage != null and
            equalF64(f64_scaled_host.data, &.{ 0.5, 1, 1.5, 2 }) and
            f64_neg.device.isCuda() and f64_neg.device_storage != null and
            equalF64(f64_neg_host.data, &.{ -1, -2, -3, -4 }) and
            f64_abs.device.isCuda() and f64_abs.device_storage != null and
            equalF64(f64_abs_host.data, &.{ 1, 2, 3, 4 }) and
            f64_square.device.isCuda() and f64_square.device_storage != null and
            equalF64(f64_square_host.data, &.{ 1, 4, 9, 16 }) and
            f64_pow_zero.device.isCuda() and f64_pow_zero.device_storage != null and
            equalF64(f64_pow_zero_host.data, &.{ 1, 1, 1, 1 }) and
            f64_pow_recip.device.isCuda() and f64_pow_recip.device_storage != null and
            approxF64(f64_pow_recip_host.data[3], 0.25, 1e-12) and
            f64_pow_sqrt.device.isCuda() and f64_pow_sqrt.device_storage != null and
            approxF64(f64_pow_sqrt_host.data[3], 2, 1e-12) and
            f64_pow_rsqrt.device.isCuda() and f64_pow_rsqrt.device_storage != null and
            approxF64(f64_pow_rsqrt_host.data[3], 0.5, 1e-12) and
            f64_pow_cube.device.isCuda() and f64_pow_cube.device_storage != null and
            equalF64(f64_pow_cube_host.data, &.{ 1, 8, 27, 64 }) and
            f64_sqrt.device.isCuda() and
            approxF64(f64_sqrt_host.data[0], std.math.sqrt(@as(f64, 3)), 1e-12) and
            f64_rsqrt.device.isCuda() and f64_rsqrt.device_storage != null and
            approxF64(f64_rsqrt_host.data[0], 1.0, 1e-12) and
            approxF64(f64_rsqrt_host.data[3], 0.5, 1e-12) and
            f64_exp.device.isCuda() and
            approxF64(f64_exp_host.data[0], std.math.exp(@as(f64, 2)), 1e-12) and
            f64_relu.device.isCuda() and f64_relu.device_storage != null and
            equalF64(f64_relu_host.data, &.{ 0, 0, 0, 1 }) and
            f64_threshold.device.isCuda() and f64_threshold.device_storage != null and
            equalF64(f64_threshold_host.data, &.{ 0.5, 0.5, 0.5, 1 }) and
            f64_sigmoid.device.isCuda() and f64_sigmoid.device_storage != null and
            approxF64(f64_sigmoid_host.data[0], @as(f64, 1.0) / (@as(f64, 1.0) + std.math.exp(@as(f64, 2.0))), 1e-12) and
            f64_softsign.device.isCuda() and f64_softsign.device_storage != null and
            approxF64(f64_softsign_host.data[0], -2.0 / 3.0, 1e-12) and
            f64_clip.device.isCuda() and f64_clip.device_storage != null and
            equalF64(f64_clip_host.data, &.{ -0.5, -0.5, 0, 0.5 }) and
            f64_maximum.device.isCuda() and f64_maximum.device_storage != null and
            approxF64(f64_maximum_host.data[0], -0.2, 1e-12) and
            approxF64(f64_maximum_host.data[3], 1.0, 1e-12) and
            f64_leaky.device.isCuda() and f64_leaky.device_storage != null and
            approxF64(f64_leaky_host.data[0], -0.2, 1e-12) and
            f64_lerp.device.isCuda() and f64_lerp.device_storage != null and
            equalF64(f64_lerp_host.data, &.{ 1, 1.5, 2, 2.5 }) and
            f64_addcmul.device.isCuda() and f64_addcmul.device_storage != null and
            equalF64(f64_addcmul_host.data, &.{ 3, 5, 7, 9 }) and
            f64_addcdiv.device.isCuda() and f64_addcdiv.device_storage != null and
            equalF64(f64_addcdiv_host.data, &.{ 1.5, 2, 2.5, 3 }) and
            f64_silu.device.isCuda() and f64_silu.device_storage != null and
            approxF64(f64_silu_host.data[0], -2.0 / (1.0 + std.math.exp(@as(f64, 2.0))), 1e-12) and
            f64_hardsigmoid.device.isCuda() and f64_hardsigmoid.device_storage != null and
            approxF64(f64_hardsigmoid_host.data[0], 1.0 / 6.0, 1e-12) and
            f64_hardswish.device.isCuda() and f64_hardswish.device_storage != null and
            approxF64(f64_hardswish_host.data[0], -2.0 / 6.0, 1e-12) and
            f64_relu6.device.isCuda() and f64_relu6.device_storage != null and
            equalF64(f64_relu6_host.data, &.{ 0, 0, 0, 1 }) and
            f64_clip_array.device.isCuda() and f64_clip_array.device_storage != null and
            equalF64(f64_clip_array_host.data, &.{ -0.25, -0.25, 0, 0.75 }) and
            f64_elu.device.isCuda() and f64_elu.device_storage != null and
            approxF64(f64_elu_host.data[0], std.math.exp(@as(f64, -2.0)) - 1.0, 1e-12) and
            f64_celu.device.isCuda() and f64_celu.device_storage != null and
            approxF64(f64_celu_host.data[0], 2.0 * (std.math.exp(@as(f64, -1.0)) - 1.0), 1e-12) and
            f64_softshrink.device.isCuda() and f64_softshrink.device_storage != null and
            approxF64(f64_softshrink_host.data[0], -1.5, 1e-12) and
            f64_mse.device.isCuda() and f64_mse.device_storage != null and
            equalF64(f64_mse_host.data, &.{ 4, 1, 0, 1 }) and
            f64_l1.device.isCuda() and f64_l1.device_storage != null and
            equalF64(f64_l1_host.data, &.{ 2, 1, 0, 1 }) and
            f64_smooth_l1.device.isCuda() and f64_smooth_l1.device_storage != null and
            equalF64(f64_smooth_l1_host.data, &.{ 1.5, 0.5, 0, 0.5 }) and
            f64_huber.device.isCuda() and f64_huber.device_storage != null and
            equalF64(f64_huber_host.data, &.{ 1.5, 0.5, 0, 0.5 });

        var f64_fused = try vx.matmulAdd(f64_lhs, f64_rhs, f64_rhs);
        defer f64_fused.deinit();
        var f64_fused_host = try f64_fused.cpu();
        defer f64_fused_host.deinit();
        var f64_scaled_fused = (try vx.axiom_backend.executeMatmulAddScaled(f64, .cuda, f64_lhs, f64_rhs, f64_rhs, 2.0, -1.0)) orelse return error.BackendFailure;
        defer f64_scaled_fused.deinit();
        var f64_scaled_fused_host = try f64_scaled_fused.cpu();
        defer f64_scaled_fused_host.deinit();
        var f64_chained = try f64_product.add(f64_rhs);
        defer f64_chained.deinit();
        var f64_chained_host = try f64_chained.cpu();
        defer f64_chained_host.deinit();
        f64_matmul_add_ok = f64_fused.device.isCuda() and
            f64_fused.device_storage != null and
            equalF64(f64_fused_host.data, &.{ 4, 4, 8, 8 }) and
            f64_scaled_fused.device.isCuda() and
            f64_scaled_fused.device_storage != null and
            equalF64(f64_scaled_fused_host.data, &.{ 5, 5, 13, 13 }) and
            f64_chained.device.isCuda() and
            f64_chained.fusionStatus() == .cuda_matmul_add and
            equalF64(f64_chained_host.data, &.{ 4, 4, 8, 8 });
    }
    const cuda_available = vx.Device.cuda(0).isAvailable();
    const memref_fingerprints_ok = !cuda_available or
        (elementwise_binary_memref_fingerprint != 0 and
            elementwise_unary_memref_fingerprint != 0 and
            gemm_memref_fingerprint != 0 and
            reduction_memref_fingerprint != 0 and
            broadcast_memref_fingerprint != 0 and
            transpose_memref_fingerprint != 0 and
            softmax_memref_fingerprint != 0 and
            log_softmax_memref_fingerprint != 0);
    ok = ok and memref_fingerprints_ok and direct_storage_ok and direct_add_ok and direct_square_ok and direct_unary_scalar_ok and direct_reduction_ok and direct_broadcast_ok and direct_transpose_ok and direct_softmax_ok and direct_log_softmax_ok and direct_ternary_ok and direct_matmul_ok and direct_matmul_add_ok and scaled_matmul_add_ok and chained_matmul_add_ok and chained_matmul_sub_ok and chained_sqrt_ok and chained_exp_ok and reversed_add_fusion_ok and reversed_sub_fusion_ok and pending_fusion_status_ok and bf16_chained_sqrt_ok and bf16_chained_exp_ok and bf16_scalar_mul_ok and bf16_broadcast_ok and bf16_reduction_ok and bf16_transpose_ok and bf16_softmax_ok and bf16_log_softmax_ok and f16_activation_ok and f16_broadcast_ok and f16_reduction_ok and f16_transpose_ok and f16_softmax_ok and f16_log_softmax_ok and f64_matmul_ok and f64_elementwise_ok and f64_transpose_ok and f64_broadcast_ok and f64_reduction_ok and f64_softmax_ok and f64_log_softmax_ok and f64_matmul_add_ok;

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    // Zig's std.Io formatter intentionally caps each call at 32 arguments, so
    // keep the smoke JSON evidence split across contiguous writes as coverage grows.
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_cuda_device_smoke\",\"enabled\":{},\"status\":\"{s}\",\"ok\":{},\"bytes\":{d},\"fingerprint\":{d},\"direct_storage_ok\":{},\"direct_add_ok\":{},\"direct_square_ok\":{},\"direct_unary_scalar_ok\":{},\"direct_reduction_ok\":{},\"direct_broadcast_ok\":{},\"direct_transpose_ok\":{},\"direct_softmax_ok\":{},\"direct_log_softmax_ok\":{},\"direct_ternary_ok\":{},\"direct_matmul_ok\":{},\"direct_matmul_add_ok\":{},\"scaled_matmul_add_ok\":{},\"chained_matmul_add_ok\":{},\"chained_matmul_sub_ok\":{},\"chained_sqrt_ok\":{},\"chained_exp_ok\":{},\"reversed_add_fusion_ok\":{},\"reversed_sub_fusion_ok\":{},\"pending_fusion_status_ok\":{},\"bf16_chained_sqrt_ok\":{},\"bf16_chained_exp_ok\":{},\"bf16_scalar_mul_ok\":{},\"bf16_broadcast_ok\":{},\"bf16_reduction_ok\":{},\"bf16_transpose_ok\":{},\"bf16_softmax_ok\":{}",
        .{ vx.axiom_cuda.enabled(), status, ok, bytes, fingerprint, direct_storage_ok, direct_add_ok, direct_square_ok, direct_unary_scalar_ok, direct_reduction_ok, direct_broadcast_ok, direct_transpose_ok, direct_softmax_ok, direct_log_softmax_ok, direct_ternary_ok, direct_matmul_ok, direct_matmul_add_ok, scaled_matmul_add_ok, chained_matmul_add_ok, chained_matmul_sub_ok, chained_sqrt_ok, chained_exp_ok, reversed_add_fusion_ok, reversed_sub_fusion_ok, pending_fusion_status_ok, bf16_chained_sqrt_ok, bf16_chained_exp_ok, bf16_scalar_mul_ok, bf16_broadcast_ok, bf16_reduction_ok, bf16_transpose_ok, bf16_softmax_ok },
    );
    try stdout.interface.print(
        ",\"bf16_log_softmax_ok\":{},\"f16_activation_ok\":{},\"f16_broadcast_ok\":{},\"f16_reduction_ok\":{},\"f16_transpose_ok\":{},\"f16_softmax_ok\":{},\"f16_log_softmax_ok\":{},\"f64_matmul_ok\":{},\"f64_elementwise_ok\":{},\"f64_transpose_ok\":{},\"f64_broadcast_ok\":{},\"f64_reduction_ok\":{},\"f64_softmax_ok\":{},\"f64_log_softmax_ok\":{},\"f64_matmul_add_ok\":{}",
        .{ bf16_log_softmax_ok, f16_activation_ok, f16_broadcast_ok, f16_reduction_ok, f16_transpose_ok, f16_softmax_ok, f16_log_softmax_ok, f64_matmul_ok, f64_elementwise_ok, f64_transpose_ok, f64_broadcast_ok, f64_reduction_ok, f64_softmax_ok, f64_log_softmax_ok, f64_matmul_add_ok },
    );
    try stdout.interface.print(
        ",\"memref_fingerprints_ok\":{},\"elementwise_binary_memref_fingerprint\":{d},\"elementwise_unary_memref_fingerprint\":{d},\"gemm_memref_fingerprint\":{d},\"reduction_memref_fingerprint\":{d},\"broadcast_memref_fingerprint\":{d},\"transpose_memref_fingerprint\":{d},\"softmax_memref_fingerprint\":{d},\"log_softmax_memref_fingerprint\":{d}}}\n",
        .{ memref_fingerprints_ok, elementwise_binary_memref_fingerprint, elementwise_unary_memref_fingerprint, gemm_memref_fingerprint, reduction_memref_fingerprint, broadcast_memref_fingerprint, transpose_memref_fingerprint, softmax_memref_fingerprint, log_softmax_memref_fingerprint },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn approxF32(actual: f32, expected: f32, tolerance: f32) bool {
    return @abs(actual - expected) <= tolerance;
}

fn approxF16(actual: f16, expected: f32, tolerance: f32) bool {
    return approxF32(@as(f32, @floatCast(actual)), expected, tolerance);
}

fn equalF32(actual: []const f32, expected: []const f32) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}

fn equalF64(actual: []const f64, expected: []const f64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |a, e| {
        if (a != e) return false;
    }
    return true;
}

fn approxF64(actual: f64, expected: f64, tolerance: f64) bool {
    return @abs(actual - expected) <= tolerance;
}
