const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer lhs.deinit();
    var rhs = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer rhs.deinit();

    const cpu_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs, rhs, .cpu);
    const cuda_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs, rhs, .cuda);
    const mps_report = try vx.axiom_backend.lowerMatmulDialect(f32, lhs, rhs, .mps);
    const mps_runtime = vx.axiom_backend.mpsDeviceReport(0);
    vx.setDefaultDialectBackend(.cuda);
    const default_cuda_report = try vx.axiom_backend.lowerMatmulDialectDefault(f32, lhs, rhs);
    const elementwise_cuda_report = try vx.axiom_backend.lowerElementwiseDialectDefault(f32, .add, lhs, lhs);
    const reduction_cuda_report = try vx.axiom_backend.lowerReductionDialectDefault(f32, lhs, .sum, 1);
    vx.setDefaultDialectBackend(.mps);
    const default_mps_report = try vx.axiom_backend.lowerMatmulDialectDefault(f32, lhs, rhs);
    const elementwise_mps_report = try vx.axiom_backend.lowerElementwiseDialectDefault(f32, .mul, lhs, lhs);
    const reduction_mps_report = try vx.axiom_backend.lowerReductionDialectDefault(f32, lhs, .max, 0);
    vx.resetDefaultDialectBackend();
    const ok = cpu_report.ok() and cuda_report.ok() and mps_report.ok() and
        default_cuda_report.ok() and default_mps_report.ok() and
        elementwise_cuda_report.ok() and elementwise_mps_report.ok() and
        reduction_cuda_report.ok() and reduction_mps_report.ok() and
        cpu_report.status == .lowered_cpu and
        cuda_report.status == .lowered_cuda and
        mps_report.status == .planned_mps and
        default_cuda_report.status == .lowered_cuda and
        default_mps_report.status == .planned_mps and
        elementwise_cuda_report.status == .lowered_cuda and
        elementwise_mps_report.status == .planned_mps and
        reduction_cuda_report.status == .lowered_cuda and
        reduction_mps_report.status == .planned_mps and
        std.mem.eql(u8, mps_report.launch_backend, "mps_planned") and
        std.mem.eql(u8, elementwise_mps_report.launch_backend, "mps_planned") and
        std.mem.eql(u8, reduction_mps_report.launch_backend, "mps_planned") and
        mps_runtime.status == .planned and
        !mps_runtime.ok() and
        vx.defaultDialectBackend() == .cpu and
        cpu_report.registration.ok() and
        cuda_report.cuda_tile_projection_fingerprint != 0;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_dialect_lowering_smoke\",\"ok\":{},\"cpu_status\":\"{s}\",\"cuda_status\":\"{s}\",\"mps_status\":\"{s}\",\"dialects\":{d},\"ops\":{d},\"memref_ops\":{d},\"linalg_ops\":{d},\"gpu_ops\":{d},\"cuda_tile\":{d},\"default_cuda_status\":\"{s}\",\"default_mps_status\":\"{s}\",\"elementwise_cuda_status\":\"{s}\",\"elementwise_mps_status\":\"{s}\",\"reduction_cuda_status\":\"{s}\",\"reduction_mps_status\":\"{s}\",\"mps_launch_backend\":\"{s}\",\"mps_runtime_status\":\"{s}\",\"mps_runtime_fingerprint\":{d},\"fingerprint\":{d}}}\n",
        .{
            ok,
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
            reduction_mps_report.status.label(),
            mps_report.launch_backend,
            mps_runtime.status.label(),
            mps_runtime.fingerprint(),
            cuda_report.fingerprint(),
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
