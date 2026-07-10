//! PyTorch-like random GEMM + add example.
//!
//! Execute shape defaults to a CUDA stress run; keep the default `--dry-run`
//! planning mode or pass `--backend=cpu --smoke` for an interactive CPU run.
//! Default mode is a dry-run plan; pass `--smoke` for a tiny execution or
//! `--execute` for the production shape. Use `--dtype=all` to compare f32/f64/
//! f16/BFloat16 where the selected device supports them.

const std = @import("std");
const vx = @import("vectra");

const production: Shape = .{ .m = 4096 * 4, .n = 4096, .k = 4096 };
const smoke: Shape = .{ .m = 8, .n = 4, .k = 6 };

const Shape = struct { m: usize, n: usize, k: usize };
const Mode = enum { dry_run, smoke, execute };
const Backend = enum { cpu, cuda, both };
const DTypeFilter = enum { all, f32, f64, f16, bf16 };
const BenchOp = enum {
    matmul,
    matmul_add,
    matmul_then_add,
    matmul_then_sub,
    matmul_then_add_sqrt,
    matmul_then_add_exp,

    fn label(op: BenchOp) []const u8 {
        return @tagName(op);
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try parseArgs(init);
    var shape = if (args.mode == .execute) production else smoke;
    if (args.m) |m| shape.m = m;
    if (args.n) |n| shape.n = n;
    if (args.k) |k| shape.k = k;
    const warmup = args.warmup orelse if (args.mode == .execute) @as(usize, 2) else 0;
    const iters = args.iters orelse if (args.mode == .execute) @as(usize, 3) else 1;
    if (iters == 0) return error.InvalidIterations;
    if (shape.m == 0 or shape.n == 0 or shape.k == 0) return error.InvalidShape;
    const dtype_filter = args.dtype orelse if (args.mode == .execute) DTypeFilter.f32 else DTypeFilter.all;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);

    try printPlan(&stdout.interface, args.mode, args.backend, dtype_filter, shape, warmup, iters, args.retain_outputs);
    if (args.mode == .dry_run) return stdout.interface.flush();

    const allocator = std.heap.smp_allocator;
    if (args.backend == .cpu or args.backend == .both) {
        if (dtypeIncluded(dtype_filter, .f32)) try runCpuDtype(f32, "f32", init, &stdout.interface, allocator, shape, warmup, iters, args.retain_outputs);
        if (dtypeIncluded(dtype_filter, .f64)) try runCpuDtype(f64, "f64", init, &stdout.interface, allocator, shape, warmup, iters, args.retain_outputs);
        if (dtypeIncluded(dtype_filter, .f16)) try runCpuDtype(f16, "f16", init, &stdout.interface, allocator, shape, warmup, iters, args.retain_outputs);
        if (dtypeIncluded(dtype_filter, .bf16)) try runCpuDtype(vx.BFloat16, "bf16", init, &stdout.interface, allocator, shape, warmup, iters, args.retain_outputs);
    }

    const gpu = vx.cuda(0);
    if (args.backend == .cuda or args.backend == .both) {
        if (gpu.isAvailable()) {
            if (dtypeIncluded(dtype_filter, .f32)) try runCudaF32(init, &stdout.interface, allocator, gpu, shape, warmup, iters, args.retain_outputs);
            if (dtypeIncluded(dtype_filter, .f64)) try printSkipped(&stdout.interface, "cuda", "f64", "matmul", "cuda_matmul_f64_not_exposed");
            if (dtypeIncluded(dtype_filter, .f64)) try printSkipped(&stdout.interface, "cuda", "f64", "matmul_add", "cuda_matmul_add_f64_not_exposed");
            if (dtypeIncluded(dtype_filter, .f64)) try printSkipped(&stdout.interface, "cuda", "f64", "matmul_then_add", "cuda_matmul_f64_not_exposed");
            if (dtypeIncluded(dtype_filter, .f64)) try printSkipped(&stdout.interface, "cuda", "f64", "matmul_then_sub", "cuda_matmul_f64_not_exposed");
            if (dtypeIncluded(dtype_filter, .f64)) try printSkipped(&stdout.interface, "cuda", "f64", "matmul_then_add_sqrt", "cuda_matmul_f64_not_exposed");
            if (dtypeIncluded(dtype_filter, .f64)) try printSkipped(&stdout.interface, "cuda", "f64", "matmul_then_add_exp", "cuda_matmul_f64_not_exposed");
            if (dtypeIncluded(dtype_filter, .f16)) try runCudaHostDtype(f16, "f16", init, &stdout.interface, allocator, shape, warmup, iters, args.retain_outputs, args.allow_slow_typed_cuda);
            if (dtypeIncluded(dtype_filter, .bf16)) try runCudaBf16(init, &stdout.interface, allocator, gpu, shape, warmup, iters, args.retain_outputs);
        } else if (args.backend == .cuda and args.require_cuda) {
            return error.CudaDisabled;
        } else {
            try stdout.interface.print("{{\"backend\":\"cuda\",\"skipped\":true,\"reason\":\"cuda_unavailable\",\"ok\":true}}\n", .{});
        }
    }

    try stdout.interface.flush();
}

const Args = struct {
    mode: Mode = .dry_run,
    backend: Backend = .both,
    dtype: ?DTypeFilter = null,
    require_cuda: bool = false,
    retain_outputs: bool = false,
    allow_slow_typed_cuda: bool = false,
    warmup: ?usize = null,
    iters: ?usize = null,
    m: ?usize = null,
    n: ?usize = null,
    k: ?usize = null,
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
        } else if (std.mem.eql(u8, arg, "--retain-outputs")) {
            parsed.retain_outputs = true;
        } else if (std.mem.eql(u8, arg, "--allow-slow-typed-cuda")) {
            parsed.allow_slow_typed_cuda = true;
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
        } else if (std.mem.startsWith(u8, arg, "--dtype=")) {
            parsed.dtype = parseDType(arg["--dtype=".len..]) orelse return error.InvalidDType;
        } else if (std.mem.eql(u8, arg, "--dtype")) {
            parsed.dtype = parseDType(it.next() orelse return error.MissingDType) orelse return error.InvalidDType;
        } else if (std.mem.startsWith(u8, arg, "--m=")) {
            parsed.m = try parsePositive(arg["--m=".len..]);
        } else if (std.mem.eql(u8, arg, "--m")) {
            parsed.m = try parsePositive(it.next() orelse return error.MissingM);
        } else if (std.mem.startsWith(u8, arg, "--n=")) {
            parsed.n = try parsePositive(arg["--n=".len..]);
        } else if (std.mem.eql(u8, arg, "--n")) {
            parsed.n = try parsePositive(it.next() orelse return error.MissingN);
        } else if (std.mem.startsWith(u8, arg, "--k=")) {
            parsed.k = try parsePositive(arg["--k=".len..]);
        } else if (std.mem.eql(u8, arg, "--k")) {
            parsed.k = try parsePositive(it.next() orelse return error.MissingK);
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}

fn parsePositiveOrZero(value: []const u8) !usize {
    return std.fmt.parseInt(usize, value, 10) catch return error.InvalidCount;
}

fn parsePositive(value: []const u8) !usize {
    const parsed = std.fmt.parseInt(usize, value, 10) catch return error.InvalidCount;
    if (parsed == 0) return error.InvalidShape;
    return parsed;
}

fn parseBackend(value: []const u8) ?Backend {
    inline for (.{ Backend.cpu, Backend.cuda, Backend.both }) |backend| {
        if (std.mem.eql(u8, value, @tagName(backend))) return backend;
    }
    return null;
}

fn parseDType(value: []const u8) ?DTypeFilter {
    if (std.mem.eql(u8, value, "float32")) return .f32;
    if (std.mem.eql(u8, value, "float64")) return .f64;
    if (std.mem.eql(u8, value, "float16") or std.mem.eql(u8, value, "half")) return .f16;
    if (std.mem.eql(u8, value, "bfloat16")) return .bf16;
    inline for (.{ DTypeFilter.all, DTypeFilter.f32, DTypeFilter.f64, DTypeFilter.f16, DTypeFilter.bf16 }) |dtype| {
        if (std.mem.eql(u8, value, @tagName(dtype))) return dtype;
    }
    return null;
}

fn dtypeIncluded(filter: DTypeFilter, dtype: DTypeFilter) bool {
    return filter == .all or filter == dtype;
}

fn runCpuDtype(
    comptime T: type,
    dtype_name: []const u8,
    init: std.process.Init,
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    shape: Shape,
    warmup: usize,
    iters: usize,
    retain_outputs: bool,
) !void {
    var np = vx.withAllocator(allocator);
    var a = try np.ones(T, &.{ shape.m, shape.k });
    defer a.deinit();
    var b = try np.ones(T, &.{ shape.k, shape.n });
    defer b.deinit();
    var c = try np.ones(T, &.{ shape.m, shape.n });
    defer c.deinit();
    const route = vx.axiom_backend.selectMatmul(T, .prefer_axiom_cpu, a, b).selected.label();
    try runBenchmark(T, init, writer, allocator, "cpu", route, dtype_name, .matmul, a, b, null, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cpu", route, dtype_name, .matmul_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cpu", route, dtype_name, .matmul_then_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cpu", route, dtype_name, .matmul_then_sub, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cpu", route, dtype_name, .matmul_then_add_sqrt, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cpu", route, dtype_name, .matmul_then_add_exp, a, b, c, warmup, iters, retain_outputs);
}

fn runCudaF32(
    init: std.process.Init,
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    gpu: vx.Device,
    shape: Shape,
    warmup: usize,
    iters: usize,
    retain_outputs: bool,
) !void {
    var np = vx.withAllocator(allocator);
    var a = try np.onesWith(vx.onDevice(f32, gpu), &.{ shape.m, shape.k });
    defer a.deinit();
    var b = try np.onesWith(vx.onDevice(f32, gpu), &.{ shape.k, shape.n });
    defer b.deinit();
    var c = try np.onesWith(vx.onDevice(f32, gpu), &.{ shape.m, shape.n });
    defer c.deinit();
    try runBenchmark(f32, init, writer, allocator, "cuda", "axiom_cuda_device", "f32", .matmul, a, b, null, warmup, iters, retain_outputs);
    try runBenchmark(f32, init, writer, allocator, "cuda", "axiom_cuda_device", "f32", .matmul_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(f32, init, writer, allocator, "cuda", "axiom_cuda_device_auto_fused", "f32", .matmul_then_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(f32, init, writer, allocator, "cuda", "axiom_cuda_device_auto_fused", "f32", .matmul_then_sub, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(f32, init, writer, allocator, "cuda", "axiom_cuda_device_auto_fused_plus_unary", "f32", .matmul_then_add_sqrt, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(f32, init, writer, allocator, "cuda", "axiom_cuda_device_auto_fused_plus_unary", "f32", .matmul_then_add_exp, a, b, c, warmup, iters, retain_outputs);
}

fn runCudaBf16(
    init: std.process.Init,
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    gpu: vx.Device,
    shape: Shape,
    warmup: usize,
    iters: usize,
    retain_outputs: bool,
) !void {
    const value = vx.BFloat16.fromF32(0.01);
    var a = try vx.Array(vx.BFloat16).fullOn(allocator, &.{ shape.m, shape.k }, value, gpu);
    defer a.deinit();
    var b = try vx.Array(vx.BFloat16).fullOn(allocator, &.{ shape.k, shape.n }, value, gpu);
    defer b.deinit();
    var c = try vx.Array(vx.BFloat16).fullOn(allocator, &.{ shape.m, shape.n }, value, gpu);
    defer c.deinit();
    try runBenchmark(vx.BFloat16, init, writer, allocator, "cuda", "axiom_cuda_device_bf16_cublas", "bf16", .matmul, a, b, null, warmup, iters, retain_outputs);
    try runBenchmark(vx.BFloat16, init, writer, allocator, "cuda", "axiom_cuda_device_bf16_cublas", "bf16", .matmul_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(vx.BFloat16, init, writer, allocator, "cuda", "axiom_cuda_device_bf16_auto_fused", "bf16", .matmul_then_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(vx.BFloat16, init, writer, allocator, "cuda", "axiom_cuda_device_bf16_auto_fused", "bf16", .matmul_then_sub, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(vx.BFloat16, init, writer, allocator, "cuda", "axiom_cuda_device_bf16_auto_fused_plus_unary", "bf16", .matmul_then_add_sqrt, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(vx.BFloat16, init, writer, allocator, "cuda", "axiom_cuda_device_bf16_auto_fused_plus_unary", "bf16", .matmul_then_add_exp, a, b, c, warmup, iters, retain_outputs);
}

fn runCudaHostDtype(
    comptime T: type,
    dtype_name: []const u8,
    init: std.process.Init,
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    shape: Shape,
    warmup: usize,
    iters: usize,
    retain_outputs: bool,
    allow_slow_typed_cuda: bool,
) !void {
    if (!allow_slow_typed_cuda and slowTypedCudaShape(shape)) {
        try printSkipped(writer, "cuda", dtype_name, "matmul", "typed_host_cuda_path_slow_for_shape_use_smaller_shape_or_allow_slow_typed_cuda");
        try printSkipped(writer, "cuda", dtype_name, "matmul_add", "typed_host_cuda_path_slow_for_shape_use_smaller_shape_or_allow_slow_typed_cuda");
        try printSkipped(writer, "cuda", dtype_name, "matmul_then_add", "typed_host_cuda_path_slow_for_shape_use_smaller_shape_or_allow_slow_typed_cuda");
        try printSkipped(writer, "cuda", dtype_name, "matmul_then_sub", "typed_host_cuda_path_slow_for_shape_use_smaller_shape_or_allow_slow_typed_cuda");
        try printSkipped(writer, "cuda", dtype_name, "matmul_then_add_sqrt", "typed_host_cuda_path_slow_for_shape_use_smaller_shape_or_allow_slow_typed_cuda");
        try printSkipped(writer, "cuda", dtype_name, "matmul_then_add_exp", "typed_host_cuda_path_slow_for_shape_use_smaller_shape_or_allow_slow_typed_cuda");
        return;
    }
    var np = vx.withAllocator(allocator);
    var a = try np.ones(T, &.{ shape.m, shape.k });
    defer a.deinit();
    var b = try np.ones(T, &.{ shape.k, shape.n });
    defer b.deinit();
    var c = try np.ones(T, &.{ shape.m, shape.n });
    defer c.deinit();
    try runBenchmark(T, init, writer, allocator, "cuda", "axiom_cuda_host_typed", dtype_name, .matmul, a, b, null, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cuda", "axiom_cuda_host_typed", dtype_name, .matmul_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cuda", "axiom_cuda_host_typed", dtype_name, .matmul_then_add, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cuda", "axiom_cuda_host_typed", dtype_name, .matmul_then_sub, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cuda", "axiom_cuda_host_typed", dtype_name, .matmul_then_add_sqrt, a, b, c, warmup, iters, retain_outputs);
    try runBenchmark(T, init, writer, allocator, "cuda", "axiom_cuda_host_typed", dtype_name, .matmul_then_add_exp, a, b, c, warmup, iters, retain_outputs);
}

fn runBenchmark(
    comptime T: type,
    init: std.process.Init,
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    backend: []const u8,
    route: []const u8,
    dtype_name: []const u8,
    op: BenchOp,
    a: vx.Array(T),
    b: vx.Array(T),
    c: ?vx.Array(T),
    warmup: usize,
    iters: usize,
    retain_outputs: bool,
) !void {
    for (0..warmup) |_| {
        var warm = try computeOp(T, op, a, b, c);
        warm.deinit();
    }

    if (retain_outputs) {
        const begin = std.Io.Timestamp.now(init.io, .real);
        var retained = try allocator.alloc(?vx.Array(T), iters);
        defer allocator.free(retained);
        @memset(retained, null);
        defer {
            for (retained) |*maybe_out| {
                if (maybe_out.*) |*out| out.deinit();
            }
        }
        for (0..iters) |i| retained[i] = try computeOp(T, op, a, b, c);
        const elapsed_us = begin.untilNow(init.io, .real).toMicroseconds();
        try printResult(T, writer, backend, route, dtype_name, op.label(), retained[iters - 1].?, elapsed_us, iters, retain_outputs);
    } else {
        const begin = std.Io.Timestamp.now(init.io, .real);

        var y: ?vx.Array(T) = null;
        defer if (y) |*out| out.deinit();
        for (0..iters) |_| {
            // if (y) |*out| out.deinit();
            y = try computeOp(T, op, a, b, c);
        }

        const elapsed_us = begin.untilNow(init.io, .real).toMicroseconds();
        try printResult(T, writer, backend, route, dtype_name, op.label(), y.?, elapsed_us, iters, retain_outputs);
    }
}

fn computeOp(comptime T: type, op: BenchOp, a: vx.Array(T), b: vx.Array(T), c: ?vx.Array(T)) vx.ArrayError!vx.Array(T) {
    return switch (op) {
        .matmul => blk: {
            var product = try vx.matmul(a, b);
            if (product.device.isCuda()) {
                var host = try product.cpu();
                host.deinit();
            }
            break :blk product;
        },
        .matmul_add => vx.matmulAdd(a, b, c orelse return error.InvalidShape),
        .matmul_then_add => blk: {
            var product = try vx.matmul(a, b);
            defer product.deinit();
            break :blk try product.add(c orelse return error.InvalidShape);
        },
        .matmul_then_sub => blk: {
            var product = try vx.matmul(a, b);
            defer product.deinit();
            break :blk try product.sub(c orelse return error.InvalidShape);
        },
        .matmul_then_add_sqrt => blk: {
            var product = try vx.matmul(a, b);
            defer product.deinit();
            var added = try product.add(c orelse return error.InvalidShape);
            defer added.deinit();
            break :blk try added.sqrt();
        },
        .matmul_then_add_exp => blk: {
            var product = try vx.matmul(a, b);
            defer product.deinit();
            var added = try product.add(c orelse return error.InvalidShape);
            defer added.deinit();
            if (comptime T == vx.BFloat16) {
                if (added.device.isCuda()) break :blk try added.exp();
            }
            const k = if (a.shape.len == 0) return error.NonMatrixArray else a.shape[a.shape.len - 1];
            var normalized = try added.mulScalar(valueFromF64(T, 1.0 / @as(f64, @floatFromInt(k + 1))));
            defer normalized.deinit();
            break :blk try normalized.exp();
        },
    };
}

fn printPlan(writer: *std.Io.Writer, mode: Mode, backend: Backend, dtype_filter: DTypeFilter, shape: Shape, warmup: usize, iters: usize, retain_outputs: bool) !void {
    try writer.print(
        "{{\"example\":\"large_matmul_add\",\"mode\":\"{s}\",\"backend\":\"{s}\",\"dtype\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"expressions\":[\"Y=A@B\",\"Y=A@B+C\",\"tmp=A@B;Y=tmp+C\",\"tmp=A@B;Y=tmp-C\",\"tmp=A@B;Y=sqrt(tmp+C)\",\"tmp=A@B;Y=exp((tmp+C)/(K+1))\"],\"axiom_enabled\":{},\"cuda_available\":{},\"warmup\":{d},\"iters\":{d},\"retain_outputs\":{},\"dry_run\":{}}}\n",
        .{ @tagName(mode), @tagName(backend), @tagName(dtype_filter), shape.m, shape.n, shape.k, vx.axiom_cuda.enabled(), vx.cuda(0).isAvailable(), warmup, iters, retain_outputs, mode == .dry_run },
    );
}

fn slowTypedCudaShape(shape: Shape) bool {
    const ops = shape.m * shape.n * shape.k;
    return ops > 256 * 256 * 256;
}

fn printSkipped(writer: *std.Io.Writer, backend: []const u8, dtype_name: []const u8, op: []const u8, reason: []const u8) !void {
    try writer.print(
        "{{\"backend\":\"{s}\",\"dtype\":\"{s}\",\"op\":\"{s}\",\"skipped\":true,\"reason\":\"{s}\",\"ok\":true}}\n",
        .{ backend, dtype_name, op, reason },
    );
}

fn printResult(comptime T: type, writer: *std.Io.Writer, backend: []const u8, route: []const u8, dtype_name: []const u8, op: []const u8, y: vx.Array(T), elapsed_us: i64, iters: usize, retain_outputs: bool) !void {
    const sample = try y.toOwnedSlice(y.allocator);
    defer y.allocator.free(sample);
    try writer.print(
        "{{\"backend\":\"{s}\",\"route\":\"{s}\",\"dtype\":\"{s}\",\"op\":\"{s}\",\"shape\":[{d},{d}],\"iters\":{d},\"elapsed_us\":{d},\"avg_us\":{d:.3},\"retain_outputs\":{},\"first\":{d:.6},\"sample_checksum\":{d:.6},\"ok\":true}}\n",
        .{ backend, route, dtype_name, op, y.shape[0], y.shape[1], iters, elapsed_us, avgUs(elapsed_us, iters), retain_outputs, valueAsF64(T, sample[0]), sampleChecksum(T, sample) },
    );
}

fn avgUs(elapsed_us: i64, iters: usize) f64 {
    return @as(f64, @floatFromInt(elapsed_us)) / @as(f64, @floatFromInt(iters));
}

fn valueAsF64(comptime T: type, value: T) f64 {
    if (T == vx.BFloat16) return value.toF64();
    return @floatCast(value);
}

fn valueFromF64(comptime T: type, value: f64) T {
    if (T == vx.BFloat16) return vx.BFloat16.fromF64(value);
    return @floatCast(value);
}

fn sampleChecksum(comptime T: type, values: []const T) f64 {
    const sample_count = @min(values.len, 1024);
    var total: f64 = 0;
    for (values[0..sample_count]) |value| total += valueAsF64(T, value);
    if (values.len > sample_count) {
        for (values[values.len - sample_count ..]) |value| total += valueAsF64(T, value);
    }
    return total;
}
