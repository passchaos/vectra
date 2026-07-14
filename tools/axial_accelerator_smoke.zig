const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const kernel = vx.axial_cuda.saxpyKernelFingerprint();
    const launch = vx.axial_cuda.saxpyLaunchFingerprint(0x1000, 0x2000, 64, 2.0);
    const registry = try vx.CudaKernelRegistry.init().addDecl(vx.axial_cuda.axial.cuda.SaxpyKernel);
    const x_slice = try vx.CudaDeviceSlice(f32).external(0x1000, 64, 0);
    const y_slice = try vx.CudaDeviceSlice(f32).external(0x2000, 64, 0);
    const typed_slice = try vx.CudaDeviceSlice(f64).external(0x3000, 4, 0);
    const config = try vx.cudaLaunchConfig(.{ .x = 2, .y = 1, .z = 1 }, .{ .x = 64, .y = 1, .z = 1 }, 0, 0);
    const wrapped_launch = try vx.CudaKernel(vx.axial_cuda.axial.cuda.SaxpyKernel).launchWith(config, &.{
        x_slice.arg("x"),
        y_slice.arg("y"),
        .scalar("n", @as(i32, 64)),
        .scalar("alpha", @as(f32, 2.0)),
    });
    var cuda_attempted = false;
    var cuda_launched = false;
    var route = vx.axial_cuda.lastReport();

    if (vx.cuda(0).isAvailable()) {
        cuda_attempted = true;
        var lhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 1, 2, 3, 4 }, &.{4}, vx.cuda(0));
        defer lhs.deinit();
        var rhs = try vx.Array(f32).fromSliceOn(allocator, &.{ 10, 20, 30, 40 }, &.{4}, vx.cuda(0));
        defer rhs.deinit();
        if (try vx.axial_cuda.tryDeviceBinaryF32(.add, lhs, rhs)) |out| {
            var owned = out;
            defer owned.deinit();
            route = vx.axial_cuda.lastReport();
            cuda_launched = route.ok();
        }
    }

    const metadata_ok = vx.axial_cuda.enabled() and kernel != 0 and launch != 0 and registry.ok() and wrapped_launch.ok() and typed_slice.valid();
    const ok = metadata_ok and (!cuda_attempted or cuda_launched or route.status == .planned or route.status == .unavailable);
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axial_accelerator_smoke\",\"ok\":{},\"metadata_ok\":{},\"cuda_attempted\":{},\"cuda_launched\":{},\"kernel\":{d},\"launch\":{d},\"registry\":{d},\"wrapped_launch\":{d},\"typed_slice\":{d},\"route\":\"{s}\",\"status\":\"{s}\",\"route_fingerprint\":{d}}}\n",
        .{ ok, metadata_ok, cuda_attempted, cuda_launched, kernel, launch, registry.fingerprint(), wrapped_launch.fingerprint(), typed_slice.fingerprint(), route.route.label(), route.status.label(), route.fingerprint() },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
