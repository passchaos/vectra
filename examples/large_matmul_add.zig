//! PyTorch-like random GEMM + add example.
//!
//! Execute shape defaults to a CUDA stress run; keep the default `--dry-run`
//! planning mode or pass `--backend=cpu --smoke` for an interactive CPU run.
//! Default mode is a dry-run plan; pass `--smoke` for a tiny execution or
//! `--execute` for the production shape.

const std = @import("std");
const vx = @import("vectra");

const production: Shape = .{ .m = 4096 * 4, .n = 4096, .k = 4096 };
const smoke: Shape = .{ .m = 8, .n = 4, .k = 6 };

const Shape = struct { m: usize, n: usize, k: usize };
const Mode = enum { dry_run, smoke, execute };
const Backend = enum { cpu, cuda, both };

pub fn main(init: std.process.Init) !void {
    const args = try parseArgs(init);
    const shape = if (args.mode == .execute) production else smoke;
    const warmup = args.warmup orelse if (args.mode == .execute) @as(usize, 2) else 0;
    const iters = args.iters orelse if (args.mode == .execute) @as(usize, 3) else 1;
    if (iters == 0) return error.InvalidIterations;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);

    try printPlan(&stdout.interface, args.mode, args.backend, shape, warmup, iters);
    if (args.mode == .dry_run) return stdout.interface.flush();

    var np = vx.withAllocator(std.heap.smp_allocator);

    if (args.backend == .cpu or args.backend == .both) {
        // CPU tensors run the CPU path.
        var a = try np.ones(f32, &.{ shape.m, shape.k });
        defer a.deinit();
        var b = try np.ones(f32, &.{ shape.k, shape.n });
        defer b.deinit();
        var c = try np.ones(f32, &.{ shape.m, shape.n });
        defer c.deinit();

        for (0..warmup) |_| {
            var warm = try vx.matmulAdd(a, b, c);
            warm.deinit();
        }
        const begin = std.Io.Timestamp.now(init.io, .real);
        var y: ?vx.Array(f32) = null;
        defer if (y) |*out| out.deinit();
        for (0..iters) |_| {
            if (y) |*out| out.deinit();
            y = try vx.matmulAdd(a, b, c);
        }
        const elapsed_us = begin.untilNow(init.io, .real).toMicroseconds();

        std.debug.print("cpu avg cost: {d:.3}us (iters={d}, warmup={d})\n", .{ avgUs(elapsed_us, iters), iters, warmup });
        try printResult(&stdout.interface, "cpu", "axiom_cpu", y.?, elapsed_us, iters);
    }

    // CUDA tensors use the same operation and dispatch through CUDA when enabled.
    const gpu = vx.cuda(0);
    if ((args.backend == .cuda or args.backend == .both) and gpu.isAvailable()) {
        var a_cuda = try np.onesWith(vx.onDevice(f32, gpu), &.{ shape.m, shape.k });
        defer a_cuda.deinit();
        var b_cuda = try np.onesWith(vx.onDevice(f32, gpu), &.{ shape.k, shape.n });
        defer b_cuda.deinit();
        var c_cuda = try np.onesWith(vx.onDevice(f32, gpu), &.{ shape.m, shape.n });
        defer c_cuda.deinit();

        for (0..warmup) |_| {
            var warm = try vx.matmulAdd(a_cuda, b_cuda, c_cuda);
            warm.deinit();
        }
        const cuda_begin = std.Io.Timestamp.now(init.io, .real);
        var y_cuda: ?vx.Array(f32) = null;
        defer if (y_cuda) |*out| out.deinit();
        for (0..iters) |_| {
            if (y_cuda) |*out| out.deinit();
            y_cuda = try vx.matmulAdd(a_cuda, b_cuda, c_cuda);
        }
        const elapsed_us = cuda_begin.untilNow(init.io, .real).toMicroseconds();

        std.debug.print("cuda avg cost: {d:.3}us (iters={d}, warmup={d})\n", .{ avgUs(elapsed_us, iters), iters, warmup });

        try printResult(&stdout.interface, "cuda", "axiom_cuda", y_cuda.?, elapsed_us, iters);
    } else if (args.backend == .cuda and args.require_cuda) {
        return error.CudaDisabled;
    } else if (args.backend == .cuda or args.backend == .both) {
        try stdout.interface.print("{{\"backend\":\"cuda\",\"skipped\":true,\"ok\":true}}\n", .{});
    }

    try stdout.interface.flush();
}

const Args = struct {
    mode: Mode = .dry_run,
    backend: Backend = .both,
    require_cuda: bool = false,
    warmup: ?usize = null,
    iters: ?usize = null,
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
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            parsed.warmup = try parsePositiveOrZero(arg["--warmup=".len..]);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            parsed.warmup = try parsePositiveOrZero(it.next() orelse return error.MissingWarmup);
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            parsed.iters = try parsePositiveOrZero(arg["--iters=".len..]);
        } else if (std.mem.eql(u8, arg, "--iters")) {
            parsed.iters = try parsePositiveOrZero(it.next() orelse return error.MissingIterations);
        } else if (std.mem.startsWith(u8, arg, "--backend=")) {
            parsed.backend = parseBackend(arg["--backend=".len..]) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            parsed.backend = parseBackend(it.next() orelse return error.MissingBackend) orelse return error.InvalidBackend;
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}

fn parsePositiveOrZero(value: []const u8) !usize {
    return std.fmt.parseInt(usize, value, 10) catch return error.InvalidCount;
}

fn parseBackend(value: []const u8) ?Backend {
    inline for (.{ Backend.cpu, Backend.cuda, Backend.both }) |backend| {
        if (std.mem.eql(u8, value, @tagName(backend))) return backend;
    }
    return null;
}

fn printPlan(writer: *std.Io.Writer, mode: Mode, backend: Backend, shape: Shape, warmup: usize, iters: usize) !void {
    try writer.print(
        "{{\"example\":\"large_matmul_add\",\"mode\":\"{s}\",\"backend\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"expression\":\"Y = ones(M,K) @ ones(K,N) + ones(M,N)\",\"axiom_enabled\":{},\"cuda_available\":{},\"warmup\":{d},\"iters\":{d},\"dry_run\":{}}}\n",
        .{ @tagName(mode), @tagName(backend), shape.m, shape.n, shape.k, vx.axiom_cuda.enabled(), vx.cuda(0).isAvailable(), warmup, iters, mode == .dry_run },
    );
}

fn printResult(writer: *std.Io.Writer, backend: []const u8, route: []const u8, y: vx.Array(f32), elapsed_us: i64, iters: usize) !void {
    const sample = try y.toOwnedSlice(y.allocator);
    defer y.allocator.free(sample);
    try writer.print(
        "{{\"backend\":\"{s}\",\"route\":\"{s}\",\"shape\":[{d},{d}],\"iters\":{d},\"elapsed_us\":{d},\"avg_us\":{d:.3},\"first\":{d:.6},\"sample_checksum\":{d:.6},\"ok\":true}}\n",
        .{ backend, route, y.shape[0], y.shape[1], iters, elapsed_us, avgUs(elapsed_us, iters), sample[0], sampleChecksum(sample) },
    );
}

fn avgUs(elapsed_us: i64, iters: usize) f64 {
    return @as(f64, @floatFromInt(elapsed_us)) / @as(f64, @floatFromInt(iters));
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
