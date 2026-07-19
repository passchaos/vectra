const std = @import("std");
const builtin = @import("builtin");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer lhs.deinit();
    var rhs = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer rhs.deinit();
    var row_bias = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30 }, &.{3});
    defer row_bias.deinit();
    var col_bias = try vx.Array(f32).fromSlice(allocator, &.{ 100, 200 }, &.{2});
    defer col_bias.deinit();
    var lhs_cuda_lowering = try cudaLoweringArray(f32, lhs);
    defer lhs_cuda_lowering.deinit();
    var rhs_cuda_lowering = try cudaLoweringArray(f32, rhs);
    defer rhs_cuda_lowering.deinit();
    var row_bias_cuda_lowering = try cudaLoweringArray(f32, row_bias);
    defer row_bias_cuda_lowering.deinit();

    const cpu_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs, rhs, .cpu);
    const cuda_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs, rhs, .cuda);
    const mps_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs, rhs, .mps);
    const mps_runtime = vx.axiom_backend.mpsDeviceReport(0);
    vx.setDefaultDialectBackend(.cuda);
    const default_cuda_report = try vx.axiom_backend.lowerMatmulDialectDefault(f32, lhs, rhs);
    const elementwise_cuda_report = try vx.axiom_backend.lowerElementwiseDialectDefault(f32, .add, lhs, lhs);
    const reduction_cuda_report = try vx.axiom_backend.lowerReductionDialectDefault(f32, lhs, .sum, 1);
    const reduction_cuda_runtime = vx.axiom_backend.reductionRuntimeCapability(.cuda);
    const broadcast_cuda_report = try vx.axiom_backend.lowerBroadcastAddDialectDefault(f32, lhs, row_bias, .row);
    const broadcast_cuda_runtime = vx.axiom_backend.broadcastAddRuntimeCapability(.cuda);
    const unary_cuda_report = try vx.axiom_backend.lowerUnaryDialectDefault(f32, lhs, .square);
    const unary_cuda_runtime = vx.axiom_backend.unaryRuntimeCapability(.cuda, .square);
    const unary_log_cuda_report = try vx.axiom_backend.lowerUnaryDialectDefault(f32, lhs, .log);
    const unary_log_cuda_runtime = vx.axiom_backend.unaryRuntimeCapability(.cuda, .log);
    const unary_log_cpu_runtime = vx.axiom_backend.unaryRuntimeCapability(.cpu, .log);
    const transpose_cuda_report = try vx.axiom_backend.lowerTransposeDialectDefault(f32, lhs);
    const transpose_cuda_runtime = vx.axiom_backend.transposeRuntimeCapability(.cuda);
    const device_matmul_cuda_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs_cuda_lowering, rhs_cuda_lowering, .cuda);
    const device_elementwise_cuda_report = try vx.axiom_backend.lowerElementwiseDialect(f32, .add, lhs_cuda_lowering, lhs_cuda_lowering, .cuda);
    const device_reduction_cuda_report = try vx.axiom_backend.lowerReductionDialect(f32, lhs_cuda_lowering, .sum, 1, .cuda);
    const device_broadcast_cuda_report = try vx.axiom_backend.lowerBroadcastAddDialect(f32, lhs_cuda_lowering, row_bias_cuda_lowering, .row, .cuda);
    const device_unary_cuda_report = try vx.axiom_backend.lowerUnaryDialect(f32, lhs_cuda_lowering, .square, .cuda);
    const device_transpose_cuda_report = try vx.axiom_backend.lowerTransposeDialect(f32, lhs_cuda_lowering, .cuda);
    vx.setDefaultDialectBackend(.mps);
    const default_mps_report = try vx.axiom_backend.lowerMatmulDialectDefault(f32, lhs, rhs);
    const elementwise_mps_report = try vx.axiom_backend.lowerElementwiseDialectDefault(f32, .mul, lhs, lhs);
    const reduction_mps_report = try vx.axiom_backend.lowerReductionDialectDefault(f32, lhs, .max, 0);
    const reduction_mps_runtime = vx.axiom_backend.reductionRuntimeCapability(.mps);
    const broadcast_mps_report = try vx.axiom_backend.lowerBroadcastAddDialectDefault(f32, lhs, col_bias, .column);
    const broadcast_mps_runtime = vx.axiom_backend.broadcastAddRuntimeCapability(.mps);
    const unary_mps_report = try vx.axiom_backend.lowerUnaryDialectDefault(f32, lhs, .cube);
    const unary_mps_runtime = vx.axiom_backend.unaryRuntimeCapability(.mps, .log);
    const unary_mps_planned_runtime = vx.axiom_backend.unaryRuntimeCapability(.mps, .cube);
    const transpose_mps_report = try vx.axiom_backend.lowerTransposeDialectDefault(f32, lhs);
    const transpose_mps_runtime = vx.axiom_backend.transposeRuntimeCapability(.mps);
    const softmax_mps_runtime = vx.axiom_backend.softmaxRuntimeCapability(.mps);
    const log_softmax_mps_runtime = vx.axiom_backend.logSoftmaxRuntimeCapability(.mps);
    vx.resetDefaultDialectBackend();
    const platform_default_backend = vx.defaultDialectBackend();
    const expected_platform_default_backend: vx.DialectBackend = if (builtin.os.tag == .macos) .mps else .cpu;
    const expected_mps_runtime_status: vx.axiom_backend.MpsRuntimeAbiStatus = if (builtin.os.tag == .macos) .available else .unavailable;
    const ok = cpu_report.ok() and cuda_report.ok() and mps_report.ok() and
        default_cuda_report.ok() and default_mps_report.ok() and
        elementwise_cuda_report.ok() and elementwise_mps_report.ok() and
        reduction_cuda_report.ok() and reduction_mps_report.ok() and
        broadcast_cuda_report.ok() and broadcast_mps_report.ok() and
        unary_cuda_report.ok() and unary_mps_report.ok() and
        transpose_cuda_report.ok() and transpose_mps_report.ok() and
        cpu_report.status == .lowered_cpu and
        cuda_report.status == .lowered_cuda and
        mps_report.status == .planned_mps and
        default_cuda_report.status == .lowered_cuda and
        default_mps_report.status == .planned_mps and
        elementwise_cuda_report.status == .lowered_cuda and
        elementwise_mps_report.status == .planned_mps and
        reduction_cuda_report.status == .lowered_cuda and
        reduction_cuda_runtime.status == .executable and
        reduction_mps_report.status == .planned_mps and
        reduction_mps_runtime.status == .executable and
        broadcast_cuda_report.status == .lowered_cuda and
        broadcast_cuda_runtime.status == .executable and
        broadcast_mps_report.status == .planned_mps and
        broadcast_mps_runtime.status == .executable and
        unary_cuda_report.status == .lowered_cuda and
        unary_cuda_runtime.status == .executable and
        unary_log_cuda_report.status == .lowered_cuda and
        unary_log_cuda_runtime.status == .executable and
        unary_log_cpu_runtime.status == .executable and
        unary_mps_report.status == .planned_mps and
        unary_mps_runtime.status == .executable and
        unary_mps_planned_runtime.status == .planned and
        transpose_cuda_report.status == .lowered_cuda and
        transpose_cuda_runtime.status == .executable and
        device_matmul_cuda_report.status == .lowered_cuda and
        device_elementwise_cuda_report.status == .lowered_cuda and
        device_reduction_cuda_report.status == .lowered_cuda and
        device_broadcast_cuda_report.status == .lowered_cuda and
        device_unary_cuda_report.status == .lowered_cuda and
        device_transpose_cuda_report.status == .lowered_cuda and
        transpose_mps_report.status == .planned_mps and
        transpose_mps_runtime.status == .executable and
        softmax_mps_runtime.status == .executable and
        log_softmax_mps_runtime.status == .executable and
        reduction_mps_runtime.executable() and
        broadcast_mps_runtime.executable() and
        unary_mps_runtime.executable() and
        !unary_mps_planned_runtime.executable() and
        transpose_mps_runtime.executable() and
        softmax_mps_runtime.executable() and
        log_softmax_mps_runtime.executable() and
        std.mem.eql(u8, mps_report.launch_backend, "mps_planned") and
        std.mem.eql(u8, elementwise_mps_report.launch_backend, "mps_planned") and
        std.mem.eql(u8, reduction_mps_report.launch_backend, "mps_planned") and
        mps_runtime.status == expected_mps_runtime_status and
        platform_default_backend == expected_platform_default_backend and
        cpu_report.registration.ok() and
        cuda_report.cuda_tile_projection_fingerprint != 0;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_dialect_lowering_smoke\",\"ok\":{},\"platform_default_backend\":\"{s}\",\"cpu_status\":\"{s}\",\"cuda_status\":\"{s}\",\"mps_status\":\"{s}\",\"dialects\":{d},\"ops\":{d},\"memref_ops\":{d},\"linalg_ops\":{d},\"gpu_ops\":{d},\"cuda_tile\":{d},\"default_cuda_status\":\"{s}\",\"default_mps_status\":\"{s}\",\"elementwise_cuda_status\":\"{s}\",\"elementwise_mps_status\":\"{s}\",\"reduction_cuda_status\":\"{s}\",\"reduction_cuda_runtime_status\":\"{s}\",\"reduction_cuda_runtime_fingerprint\":{d},\"reduction_mps_status\":\"{s}\",\"reduction_mps_runtime_status\":\"{s}\",\"reduction_mps_runtime_fingerprint\":{d},\"mps_launch_backend\":\"{s}\",\"mps_runtime_status\":\"{s}\",\"mps_runtime_fingerprint\":{d},\"broadcast_cuda_status\":\"{s}\",\"broadcast_cuda_runtime_status\":\"{s}\",\"broadcast_cuda_runtime_fingerprint\":{d},\"broadcast_mps_status\":\"{s}\",\"broadcast_mps_runtime_status\":\"{s}\",\"broadcast_mps_runtime_fingerprint\":{d}",
        .{
            ok,
            platform_default_backend.label(),
            cpu_report.status.label(),
            cuda_report.status.label(),
            mps_report.status.label(),
            cuda_report.registration.dialect_count,
            cuda_report.registration.operation_count,
            cuda_report.registration.memref_operation_count,
            cuda_report.registration.linalg_operation_count,
            cuda_report.registration.gpu_operation_count,
            cuda_report.cuda_tile_projection_fingerprint,
            default_cuda_report.status.label(),
            default_mps_report.status.label(),
            elementwise_cuda_report.status.label(),
            elementwise_mps_report.status.label(),
            reduction_cuda_report.status.label(),
            reduction_cuda_runtime.status.label(),
            reduction_cuda_runtime.fingerprint(),
            reduction_mps_report.status.label(),
            reduction_mps_runtime.status.label(),
            reduction_mps_runtime.fingerprint(),
            mps_report.launch_backend,
            mps_runtime.status.label(),
            mps_runtime.fingerprint(),
            broadcast_cuda_report.status.label(),
            broadcast_cuda_runtime.status.label(),
            broadcast_cuda_runtime.fingerprint(),
            broadcast_mps_report.status.label(),
            broadcast_mps_runtime.status.label(),
            broadcast_mps_runtime.fingerprint(),
        },
    );
    try stdout.interface.print(
        ",\"unary_cuda_status\":\"{s}\",\"unary_cuda_runtime_status\":\"{s}\",\"unary_cuda_runtime_fingerprint\":{d},\"unary_log_cuda_status\":\"{s}\",\"unary_log_cuda_runtime_status\":\"{s}\",\"unary_log_cpu_runtime_status\":\"{s}\",\"unary_mps_status\":\"{s}\",\"unary_mps_runtime_status\":\"{s}\",\"unary_mps_planned_runtime_status\":\"{s}\",\"unary_mps_runtime_fingerprint\":{d},\"transpose_cuda_status\":\"{s}\",\"transpose_cuda_runtime_status\":\"{s}\",\"transpose_cuda_runtime_fingerprint\":{d},\"device_matmul_cuda_status\":\"{s}\",\"device_elementwise_cuda_status\":\"{s}\",\"device_reduction_cuda_status\":\"{s}\",\"device_broadcast_cuda_status\":\"{s}\",\"device_unary_cuda_status\":\"{s}\",\"device_transpose_cuda_status\":\"{s}\",\"transpose_mps_status\":\"{s}\",\"transpose_mps_runtime_status\":\"{s}\",\"transpose_mps_runtime_fingerprint\":{d},\"softmax_mps_runtime_status\":\"{s}\",\"log_softmax_mps_runtime_status\":\"{s}\",\"fingerprint\":{d}}}\n",
        .{
            unary_cuda_report.status.label(),
            unary_cuda_runtime.status.label(),
            unary_cuda_runtime.fingerprint(),
            unary_log_cuda_report.status.label(),
            unary_log_cuda_runtime.status.label(),
            unary_log_cpu_runtime.status.label(),
            unary_mps_report.status.label(),
            unary_mps_runtime.status.label(),
            unary_mps_planned_runtime.status.label(),
            unary_mps_runtime.fingerprint(),
            transpose_cuda_report.status.label(),
            transpose_cuda_runtime.status.label(),
            transpose_cuda_runtime.fingerprint(),
            device_matmul_cuda_report.status.label(),
            device_elementwise_cuda_report.status.label(),
            device_reduction_cuda_report.status.label(),
            device_broadcast_cuda_report.status.label(),
            device_unary_cuda_report.status.label(),
            device_transpose_cuda_report.status.label(),
            transpose_mps_report.status.label(),
            transpose_mps_runtime.status.label(),
            transpose_mps_runtime.fingerprint(),
            softmax_mps_runtime.status.label(),
            log_softmax_mps_runtime.status.label(),
            cuda_report.fingerprint(),
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn cudaLoweringArray(comptime T: type, input: vx.Array(T)) !vx.Array(T) {
    if (vx.Device.cuda(0).isAvailable()) return input.cuda(0);
    var tagged = try input.clone();
    tagged.device = vx.cuda(0);
    return tagged;
}
