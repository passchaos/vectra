//! PyTorch-like random GEMM + add example.
//!
//! Production shape: M = 4096 * 4, N = 4096, K = 4096.
//! Default mode is a dry-run plan; pass `--smoke` for a tiny execution or
//! `--execute` for the production shape.

const std = @import("std");
const vx = @import("vectra");

const production: Shape = .{ .m = 4096 * 4, .n = 4096, .k = 4096 };
const smoke: Shape = .{ .m = 8, .n = 4, .k = 6 };

const Shape = struct { m: usize, n: usize, k: usize };
const Mode = enum { dry_run, smoke, execute };

pub fn main(init: std.process.Init) !void {
    const args = try parseArgs(init);
    const shape = if (args.mode == .execute) production else smoke;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);

    try printPlan(&stdout.interface, args.mode, shape);
    if (args.mode == .dry_run) return stdout.interface.flush();

    const np = vx.withAllocator(std.heap.smp_allocator);

    // CPU tensors run the CPU path.
    var a = try np.randWith(vx.seeded(f32, 0x4096_0001), &.{ shape.m, shape.k });
    defer a.deinit();
    var b = try np.randWith(vx.seeded(f32, 0x4096_0002), &.{ shape.k, shape.n });
    defer b.deinit();
    var c = try np.randWith(vx.seeded(f32, 0x4096_0003), &.{ shape.m, shape.n });
    defer c.deinit();
    var y = try vx.matmulAdd(a, b, c);
    defer y.deinit();
    try printResult(&stdout.interface, "cpu", "direct_cpu", y);

    // CUDA tensors use the same operation and dispatch through CUDA when enabled.
    if (vx.axiom_cuda.enabled()) {
        const gpu = vx.cuda(0);
        var a_cuda = try np.randWith(vx.seededOn(f32, gpu, 0x4096_0001), &.{ shape.m, shape.k });
        defer a_cuda.deinit();
        var b_cuda = try np.randWith(vx.seededOn(f32, gpu, 0x4096_0002), &.{ shape.k, shape.n });
        defer b_cuda.deinit();
        var c_cuda = try np.randWith(vx.seededOn(f32, gpu, 0x4096_0003), &.{ shape.m, shape.n });
        defer c_cuda.deinit();
        var y_cuda = try vx.matmulAdd(a_cuda, b_cuda, c_cuda);
        defer y_cuda.deinit();
        try printResult(&stdout.interface, "cuda", "axiom_cuda", y_cuda);
    } else if (args.require_cuda) {
        return error.CudaDisabled;
    } else {
        try stdout.interface.print("{{\"backend\":\"cuda\",\"skipped\":true,\"ok\":true}}\n", .{});
    }

    try stdout.interface.flush();
}

const Args = struct {
    mode: Mode = .dry_run,
    require_cuda: bool = false,
};

fn parseArgs(init: std.process.Init) !Args {
    var parsed: Args = .{};
    var it = std.process.Args.Iterator.init(init.minimal.args);
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dry-run")) {
            parsed.mode = .dry_run;
        } else if (std.mem.eql(u8, arg, "--smoke")) {
            parsed.mode = .smoke;
        } else if (std.mem.eql(u8, arg, "--execute")) {
            parsed.mode = .execute;
        } else if (std.mem.eql(u8, arg, "--require-cuda")) {
            parsed.require_cuda = true;
        } else if (std.mem.startsWith(u8, arg, "--backend")) {
            // Accepted for compatibility with older invocations; this example now
            // always runs CPU and tries CUDA when the build enables it.
            if (std.mem.eql(u8, arg, "--backend")) _ = it.next() orelse return error.MissingBackend;
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}

fn printPlan(writer: *std.Io.Writer, mode: Mode, shape: Shape) !void {
    try writer.print(
        "{{\"example\":\"large_matmul_add\",\"mode\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"expression\":\"Y = rand(M,K) @ rand(K,N) + rand(M,N)\",\"cuda_enabled\":{},\"dry_run\":{}}}\n",
        .{ @tagName(mode), shape.m, shape.n, shape.k, vx.axiom_cuda.enabled(), mode == .dry_run },
    );
}

fn printResult(writer: *std.Io.Writer, backend: []const u8, route: []const u8, y: vx.Array(f32)) !void {
    try writer.print(
        "{{\"backend\":\"{s}\",\"route\":\"{s}\",\"shape\":[{d},{d}],\"first\":{d:.6},\"sample_checksum\":{d:.6},\"ok\":true}}\n",
        .{ backend, route, y.shape[0], y.shape[1], y.data[0], sampleChecksum(y.data) },
    );
}

fn sampleChecksum(values: []const f32) f64 {
    const sample_count = @min(values.len, 1024);
    var total: f64 = 0;
    for (values[0..sample_count]) |value| total += value;
    if (values.len > sample_count) {
        for (values[values.len - sample_count ..]) |value| total += value;
    }
    return total;
}
