//! Smoke gate for Vectra view descriptors crossing Axiom GEMM layout planning.
//!
//! This deliberately exercises Axiom's memref lowering plan rather than running
//! a Vectra materialized matmul.  The goal is to prove that transposed and
//! non-row-major ArrayView layouts remain visible to Axiom as pack/unpack
//! bufferization work instead of being erased into a raw pointer/shape ABI.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var lhs_base = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2, 3,
        4, 5, 6,
    }, &.{ 2, 3 });
    defer lhs_base.deinit();
    var rhs_base = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2,
        3, 4,
        5, 6,
    }, &.{ 3, 2 });
    defer rhs_base.deinit();
    var out_base = try vx.Array(f32).zeros(allocator, &.{ 2, 2 });
    defer out_base.deinit();

    var lhs = try lhs_base.asView();
    defer lhs.deinit();
    var rhs = try rhs_base.asView();
    defer rhs.deinit();
    var out = try out_base.asView();
    defer out.deinit();

    const executable = try vx.axiom_backend.planGemmMemRefLowering(f32, lhs, rhs, out);

    var lhs_transposed_storage = try vx.Array(f32).fromSlice(allocator, &.{
        1, 4,
        2, 5,
        3, 6,
    }, &.{ 3, 2 });
    defer lhs_transposed_storage.deinit();
    var lhs_transposed = try lhs_transposed_storage.transposeView();
    defer lhs_transposed.deinit();
    const pack_a = try vx.axiom_backend.planGemmMemRefLowering(f32, lhs_transposed, rhs, out);

    var rhs_transposed_storage = try vx.Array(f32).fromSlice(allocator, &.{
        1, 3, 5,
        2, 4, 6,
    }, &.{ 2, 3 });
    defer rhs_transposed_storage.deinit();
    var rhs_transposed = try rhs_transposed_storage.transposeView();
    defer rhs_transposed.deinit();
    const pack_b = try vx.axiom_backend.planGemmMemRefLowering(f32, lhs, rhs_transposed, out);

    var out_transposed_storage = try vx.Array(f32).zeros(allocator, &.{ 2, 2 });
    defer out_transposed_storage.deinit();
    var out_transposed = try out_transposed_storage.transposeView();
    defer out_transposed.deinit();
    const unpack_c = try vx.axiom_backend.planGemmMemRefLowering(f32, lhs, rhs, out_transposed);
    const bufferized = try vx.axiom_backend.computeGemmMemRefBufferizedReference(f32, allocator, lhs_transposed, rhs_transposed, out_transposed);
    const device_bufferized = try vx.axiom_backend.planGemmMemRefDeviceBufferization(f32, lhs_transposed, rhs_transposed, out_transposed);

    var lhs64_storage = try vx.Array(f64).fromSlice(allocator, &.{
        1, 4,
        2, 5,
        3, 6,
    }, &.{ 3, 2 });
    defer lhs64_storage.deinit();
    var rhs64_storage = try vx.Array(f64).fromSlice(allocator, &.{
        1, 3, 5,
        2, 4, 6,
    }, &.{ 2, 3 });
    defer rhs64_storage.deinit();
    var out64_storage = try vx.Array(f64).zeros(allocator, &.{ 2, 2 });
    defer out64_storage.deinit();
    var lhs64_transposed = try lhs64_storage.transposeView();
    defer lhs64_transposed.deinit();
    var rhs64_transposed = try rhs64_storage.transposeView();
    defer rhs64_transposed.deinit();
    var out64_transposed = try out64_storage.transposeView();
    defer out64_transposed.deinit();
    const device_bufferized64 = try vx.axiom_backend.planGemmMemRefDeviceBufferization(f64, lhs64_transposed, rhs64_transposed, out64_transposed);

    var broadcast_scalar = try vx.Array(f32).fromSlice(allocator, &.{1}, &.{1});
    defer broadcast_scalar.deinit();
    var broadcast_lhs = try broadcast_scalar.asStrided(&.{ 2, 3 }, &.{ 0, 0 }, 0);
    defer broadcast_lhs.deinit();
    const broadcast_rejected = blk: {
        const plan = try vx.axiom_backend.planGemmMemRefLowering(f32, broadcast_lhs, rhs, out);
        break :blk !plan.ok() and plan.status == .unsupported_broadcast;
    };

    const ok = executable.ok() and
        executable.executable() and
        executable.status == .executable_row_major_ld and
        executable.fingerprint() != 0 and
        pack_a.ok() and
        pack_a.needsBufferization() and
        pack_a.a_requires_pack and
        pack_a.status == .needs_pack_a and
        pack_b.ok() and
        pack_b.needsBufferization() and
        pack_b.b_requires_pack and
        pack_b.status == .needs_pack_b and
        unpack_c.ok() and
        unpack_c.needsBufferization() and
        unpack_c.c_requires_unpack and
        unpack_c.status == .needs_unpack_c and
        bufferized.ok() and
        bufferized.a_packed and
        bufferized.b_packed and
        bufferized.c_unpacked and
        bufferized.output_fingerprint != 0 and
        device_bufferized.ok() and
        device_bufferized.usesDeviceCopyPackRuntime() and
        device_bufferized.deviceRuntimeExecutable() and
        device_bufferized.status == .executable_device_copy_pack_unpack and
        device_bufferized64.ok() and
        device_bufferized64.usesDeviceCopyPackRuntime() and
        device_bufferized64.deviceRuntimeExecutable() and
        device_bufferized64.status == .executable_device_copy_pack_unpack and
        std.mem.eql(f32, out_transposed_storage.data, &.{ 22, 49, 28, 64 }) and
        broadcast_rejected;

    var stdout_buffer: [1536]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axiom_gemm_layout_smoke\",\"ok\":{},\"executable_status\":\"{s}\",\"pack_a_status\":\"{s}\",\"pack_b_status\":\"{s}\",\"unpack_c_status\":\"{s}\",\"bufferized_ok\":{},\"device_bufferized_status\":\"{s}\",\"device_bufferized64_status\":\"{s}\",\"bufferized_fp\":{d},\"device_bufferized_fp\":{d},\"device_bufferized64_fp\":{d},\"bufferized_output_fp\":{d},\"broadcast_rejected\":{},\"executable_fp\":{d},\"pack_a_fp\":{d},\"pack_b_fp\":{d},\"unpack_c_fp\":{d}}}\n",
        .{
            ok,
            executable.status.label(),
            pack_a.status.label(),
            pack_b.status.label(),
            unpack_c.status.label(),
            bufferized.ok(),
            device_bufferized.status.label(),
            device_bufferized64.status.label(),
            bufferized.fingerprint(),
            device_bufferized.fingerprint(),
            device_bufferized64.fingerprint(),
            bufferized.output_fingerprint,
            broadcast_rejected,
            executable.fingerprint(),
            pack_a.fingerprint(),
            pack_b.fingerprint(),
            unpack_c.fingerprint(),
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
