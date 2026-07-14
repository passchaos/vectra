const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const kernel = vx.axial_cuda.saxpyKernelFingerprint();
    const launch = vx.axial_cuda.saxpyLaunchFingerprint(0x1000, 0x2000, 64, 2.0);
    const context_device = vx.CudaDeviceIdentity.primary(0, "vectra-smoke-gpu", 9, 0);
    const context_streams = try vx.CudaStreamPool.roundRobin(0, 31, 4);
    const context = try (try vx.CudaExecutionContext.init(context_device, context_streams)).withMemoryPool(.init(0, 0x7100, 1 << 20));
    const allocation = try context.allocAsyncPlan(4096);
    const registry = try (try vx.CudaKernelRegistry.init().addDecl(vx.axial_cuda.axial.cuda.SaxpyKernel)).addDecl(vx.axial_cuda.axial.cuda.SaxpyWideKernel);
    const x_slice = try vx.CudaDeviceSlice(f32).external(0x1000, 64, 0);
    const y_slice = try vx.CudaDeviceSlice(f32).external(0x2000, 64, 0);
    const family_x = try vx.CudaDeviceSlice(f32).external(0x5000, 1024, 0);
    const family_y = try vx.CudaDeviceSlice(f32).external(0x6000, 1024, 0);
    const typed_slice = try vx.CudaDeviceSlice(f64).external(0x3000, 4, 0);
    const tensor = try vx.CudaTensor(f32, 2).fromSlice(try vx.CudaDeviceSlice(f32).external(0x7000, 32 * 64, 0), .{ 32, 64 });
    const tensor_view = try tensor.view(1, .{32 * 64});
    const tensor_partition = try tensor.partition(.{ 8, 16 });
    const tensor_mapped = try tensor.mappedPartition(.{ 8, 16 }, 2, .{ 4, 1 }, 8);
    const tensor_grid = try tensor_partition.grid();
    const tensor_mapped_grid = try tensor_mapped.grid();
    const tensor_creation = try vx.CudaTensorCreationPlan(f32, 2).fromHost(context, .{ 32, 64 }, 0x8000, 0xbead);
    const config = try vx.cudaLaunchConfig(.{ .x = 2, .y = 1, .z = 1 }, .{ .x = 64, .y = 1, .z = 1 }, 0, 0);
    const args = try (try (try (try vx.CudaArgumentList.init().slice("x", x_slice)).slice("y", y_slice)).scalar("n", @as(i32, 64))).scalar("alpha", @as(f32, 2.0));
    const wrapped_launch = try registry.launchWith("axial_saxpy", config, args);
    const call_launch = try vx.CudaKernel(vx.axial_cuda.axial.cuda.SaxpyKernel).call1D(64, 128, .{ x_slice, y_slice, @as(i32, 64), @as(f32, 2.0) });
    const top_level_call = try vx.cudaCallWith(vx.axial_cuda.axial.cuda.SaxpyKernel, config, .{ x_slice, y_slice, @as(i32, 64), @as(f32, 2.0) });
    var schedule = try vx.CudaSchedulingPolicy.roundRobin(0, 13, 2);
    const lazy_op = try vx.CudaDeviceOperation.fromCall1D(vx.axial_cuda.axial.cuda.SaxpyKernel, 64, 128, .{ x_slice, y_slice, @as(i32, 64), @as(f32, 2.0) });
    const scheduled_op = try lazy_op.scheduledNext(&schedule);
    const graph_next = try vx.CudaDeviceOperation.fromCall1D(vx.axial_cuda.axial.cuda.SaxpyKernel, 64, 128, .{ x_slice, y_slice, @as(i32, 64), @as(f32, 3.0) });
    const graph_chain = try (try scheduled_op.then(graph_next)).scheduledOn(.explicit(0, 21));
    const graph_capture = try vx.CudaGraphCapture.fromChain(graph_chain, .relaxed);
    const graph_exec = try vx.CudaGraphExecutable.instantiate(graph_capture);
    const graph_copy = vx.CudaMemoryCopyOperation.hostToDevice("input", x_slice.buffer.device_ptr, 0xfeed, 256);
    const graph_updates = (try vx.CudaMemoryOperationChain.init(graph_copy)).scheduledOn(.explicit(0, 21));
    const graph_update = try graph_exec.updateMemory(graph_updates);
    const graph_launch = (try graph_exec.launch(.explicit(0, 22))).asDeviceOperation();
    const partition = try x_slice.partition1D(64, 16);
    const partition_grid = try vx.cudaInferPartitionLaunchGrid(.{partition});
    const partition_metadata = partition.argMetadata("x");
    const family_args = try (try (try (try vx.CudaArgumentList.init().slice("x", family_x)).slice("y", family_y)).scalar("n", @as(i32, 1024))).scalar("alpha", @as(f32, 2.0));
    const family_problem: vx.KernelProblem = .{ .name = "saxpy", .m = 1024, .n = 1, .k = 1, .target_arch = "sm_89" };
    const family_launch = try registry.launchFamily(vx.axial_cuda.axial.cuda.saxpyKernelFamily(), family_problem, .auto, null, family_args);
    const module_artifact = vx.CudaModuleArtifact.externalCubin("vectra_axial_saxpy.cubin", "sm_89", registry.fingerprint(), 4096);
    const module_bundle = vx.CudaModuleBundle.init("vectra_axial_saxpy", "sm_89", registry, module_artifact);
    const loaded_module = try module_bundle.load(0x7151);
    const module_launch = try loaded_module.launch("axial_saxpy", args);
    const module_family_launch = try loaded_module.launchFamily(vx.axial_cuda.axial.cuda.saxpyKernelFamily(), family_problem, .auto, null, family_args);
    const index_space = vx.CudaThreadIndexSpace.oneD("thread.index_1d");
    const index_witness = try vx.CudaThreadIndexWitness.fromSpace(index_space, "thread.index_1d()");
    const disjoint = try vx.CudaDisjointDeviceSlice.init("out", "f32", index_space, true);
    const shared = try vx.CudaSharedMemoryRegion.init("tile", "f32", 128, 64);
    var intrinsics = try vx.CudaDeviceIntrinsicSet.init().add(vx.axial_cuda.axial.device_model.threadIndexIntrinsic());
    intrinsics = try intrinsics.add(vx.axial_cuda.axial.device_model.warpShuffleIntrinsic());
    intrinsics = try intrinsics.add(vx.axial_cuda.axial.device_model.deviceAtomicIntrinsic());
    const device_contract: vx.CudaKernelDeviceContract = .{
        .index_space = index_space,
        .disjoint_slice = disjoint,
        .shared_region = shared,
        .barrier = .{ .kind = .sync_threads, .scope = .block },
        .intrinsic_set = intrinsics,
    };
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

    const metadata_ok = vx.axial_cuda.enabled() and kernel != 0 and launch != 0 and context.valid() and allocation.ok() and registry.ok() and args.ok() and wrapped_launch.ok() and call_launch.ok() and top_level_call.ok() and scheduled_op.ok() and scheduled_op.scheduled() and graph_capture.ok() and graph_exec.ok() and graph_update.ok() and graph_launch.ok() and tensor.valid() and tensor_view.valid() and tensor_partition.valid() and tensor_mapped.valid() and tensor_creation.ok() and tensor_grid.x == 4 and tensor_grid.y == 4 and tensor_mapped_grid.x == 8 and partition.valid() and partition_metadata.valid() and partition_grid.x == 4 and family_launch.ok() and loaded_module.ok() and module_launch.ok() and module_family_launch.ok() and disjoint.accepts(index_witness) and intrinsics.supportedOn(70) and device_contract.valid() and typed_slice.valid();
    const ok = metadata_ok and (!cuda_attempted or cuda_launched or route.status == .planned or route.status == .unavailable);
    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_axial_accelerator_smoke\",\"ok\":{},\"metadata_ok\":{},\"cuda_attempted\":{},\"cuda_launched\":{},\"context\":{d},\"allocation\":{d},\"registry\":{d},\"wrapped_launch\":{d},\"call_launch\":{d},\"lazy_op\":{d},\"lazy_stream\":{d},\"graph\":{d},\"graph_update\":{d},\"graph_ops\":{d},\"tensor\":{d},\"tensor_creation\":{d},\"tensor_grid_x\":{d},\"tensor_mapped\":{d},\"tensor_mapped_grid_x\":{d},\"partition\":{d},\"partition_grid_x\":{d},\"family_launch\":{d},\"family_variant\":\"{s}\",\"module\":{d},\"module_launch\":{d},\"module_family\":{d},\"intrinsics\":{d},\"device_contract\":{d},\"typed_slice\":{d},\"route\":\"{s}\",\"status\":\"{s}\"}}\n",
        .{ ok, metadata_ok, cuda_attempted, cuda_launched, context.fingerprint(), allocation.fingerprint(), registry.fingerprint(), wrapped_launch.fingerprint(), call_launch.fingerprint(), scheduled_op.fingerprint(), scheduled_op.stream.stream_id, graph_launch.fingerprint(), graph_update.fingerprint(), graph_capture.operation_count, tensor.fingerprint(), tensor_creation.fingerprint(), tensor_grid.x, tensor_mapped.fingerprint(), tensor_mapped_grid.x, partition_metadata.fingerprint(), partition_grid.x, family_launch.fingerprint(), family_launch.selection.variant.id, loaded_module.fingerprint(), module_launch.fingerprint(), module_family_launch.fingerprint(), intrinsics.fingerprint(), device_contract.fingerprint(), typed_slice.fingerprint(), route.route.label(), route.status.label() },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}
