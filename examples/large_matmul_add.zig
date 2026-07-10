//! Large random GEMM + bias-add example with explicit CPU and CUDA paths.
//!
//! The production shape is intentionally large:
//!   M = 4096 * 4, N = 4096, K = 4096
//!
//! By default this example prints a dry-run plan so `zig build examples` remains
//! safe.  Use `--smoke` for a tiny executable check, or `--execute` to allocate
//! the full random matrices and run the selected backend explicitly.
//!
//! Examples:
//!   zig build example-large-matmul-add
//!   zig build example-large-matmul-add -- --smoke --backend=both
//!   zig build example-large-matmul-add -- --execute --backend=cpu
//!   zig build -Daxiom-cuda=true example-large-matmul-add -- --execute --backend=cuda --require-cuda

const std = @import("std");
const vx = @import("vectra");

const production_dims: Dimensions = .{
    .m = 4096 * 4,
    .n = 4096,
    .k = 4096,
};

const smoke_dims: Dimensions = .{
    .m = 8,
    .n = 4,
    .k = 6,
};

const Dimensions = struct {
    m: usize,
    n: usize,
    k: usize,

    fn outputElements(self: Dimensions) usize {
        return self.m * self.n;
    }

    fn workingSetBytes(self: Dimensions) u128 {
        // A[M,K], B[K,N], C[M,N], product[M,N], output[M,N].
        const m: u128 = @intCast(self.m);
        const n: u128 = @intCast(self.n);
        const k: u128 = @intCast(self.k);
        const f32_bytes: u128 = @sizeOf(f32);
        return (m * k + k * n + 3 * m * n) * f32_bytes;
    }

    fn fusedFlopEstimate(self: Dimensions) u128 {
        // Dense GEMM multiply-add cost plus the final matrix add.
        const m: u128 = @intCast(self.m);
        const n: u128 = @intCast(self.n);
        const k: u128 = @intCast(self.k);
        return 2 * m * n * k + m * n;
    }
};

const Mode = enum {
    dry_run,
    smoke,
    execute,

    fn label(mode: Mode) []const u8 {
        return @tagName(mode);
    }
};

const Backend = enum {
    cpu,
    cuda,
    both,

    fn label(backend: Backend) []const u8 {
        return @tagName(backend);
    }
};

const Args = struct {
    mode: Mode = .dry_run,
    backend: Backend = .both,
    require_cuda: bool = false,
};

const Matrices = struct {
    a: vx.Array(f32),
    b: vx.Array(f32),
    c: vx.Array(f32),

    fn deinit(self: *Matrices) void {
        self.a.deinit();
        self.b.deinit();
        self.c.deinit();
        self.* = undefined;
    }
};

const Summary = struct {
    backend: []const u8,
    route: []const u8,
    m: usize,
    n: usize,
    k: usize,
    first: f32,
    checksum: f64,
    output_elements: usize,

    fn writeJson(self: Summary, writer: *std.Io.Writer) !void {
        try writer.print(
            "{{\"kind\":\"large_matmul_add_result\",\"backend\":\"{s}\",\"route\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"output_elements\":{d},\"first\":{d:.6},\"sample_checksum\":{d:.6},\"ok\":true}}\n",
            .{
                self.backend,
                self.route,
                self.m,
                self.n,
                self.k,
                self.output_elements,
                self.first,
                self.checksum,
            },
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const args = parseArgs(init) catch |err| {
        var stderr_buffer: [1024]u8 = undefined;
        var stderr = std.Io.File.stderr().writerStreaming(init.io, &stderr_buffer);
        try stderr.interface.print("invalid large-matmul-add args: {s}\n", .{@errorName(err)});
        try stderr.interface.print("usage: zig build example-large-matmul-add -- [--dry-run|--smoke|--execute] [--backend=cpu|cuda|both] [--require-cuda]\n", .{});
        try stderr.interface.flush();
        std.process.exit(2);
    };

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);

    const dims = switch (args.mode) {
        .dry_run, .execute => production_dims,
        .smoke => smoke_dims,
    };

    try writePlan(&stdout.interface, args, dims);
    if (args.mode == .dry_run) {
        try stdout.interface.flush();
        return;
    }

    if (args.backend == .cpu or args.backend == .both) {
        const cpu_summary = try runCpu(std.heap.smp_allocator, dims);
        try cpu_summary.writeJson(&stdout.interface);
    }

    if (args.backend == .cuda or args.backend == .both) {
        if (!vx.axiom_cuda.enabled()) {
            if (args.require_cuda) return error.CudaDisabled;
            try stdout.interface.print(
                "{{\"kind\":\"large_matmul_add_result\",\"backend\":\"cuda\",\"route\":\"disabled\",\"m\":{d},\"n\":{d},\"k\":{d},\"ok\":true,\"skipped\":true}}\n",
                .{ dims.m, dims.n, dims.k },
            );
        } else {
            const cuda_summary = try runCuda(std.heap.smp_allocator, dims);
            try cuda_summary.writeJson(&stdout.interface);
        }
    }

    try stdout.interface.flush();
}

fn parseArgs(init: std.process.Init) !Args {
    var parsed: Args = .{};
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dry-run")) {
            parsed.mode = .dry_run;
        } else if (std.mem.eql(u8, arg, "--smoke")) {
            parsed.mode = .smoke;
        } else if (std.mem.eql(u8, arg, "--execute")) {
            parsed.mode = .execute;
        } else if (std.mem.eql(u8, arg, "--cpu")) {
            parsed.backend = .cpu;
        } else if (std.mem.eql(u8, arg, "--cuda")) {
            parsed.backend = .cuda;
        } else if (std.mem.eql(u8, arg, "--both")) {
            parsed.backend = .both;
        } else if (std.mem.eql(u8, arg, "--require-cuda")) {
            parsed.require_cuda = true;
        } else if (std.mem.startsWith(u8, arg, "--backend=")) {
            parsed.backend = parseBackend(arg["--backend=".len..]) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const value = args.next() orelse return error.MissingBackend;
            parsed.backend = parseBackend(value) orelse return error.InvalidBackend;
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}

fn parseBackend(value: []const u8) ?Backend {
    inline for (.{ Backend.cpu, Backend.cuda, Backend.both }) |backend| {
        if (std.mem.eql(u8, value, backend.label())) return backend;
    }
    return null;
}

fn writePlan(writer: *std.Io.Writer, args: Args, dims: Dimensions) !void {
    try writer.print(
        "{{\"example\":\"large_matmul_add\",\"mode\":\"{s}\",\"backend\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"expression\":\"Y = random(M,K) * random(K,N) + random(M,N)\",\"working_set_bytes\":{d},\"fused_flop_estimate\":{d},\"cuda_enabled\":{},\"cpu_policy\":\"force_direct_cpu\",\"cuda_policy\":\"explicit_vx.axiom_cuda\",\"dry_run\":{}}}\n",
        .{
            args.mode.label(),
            args.backend.label(),
            dims.m,
            dims.n,
            dims.k,
            dims.workingSetBytes(),
            dims.fusedFlopEstimate(),
            vx.axiom_cuda.enabled(),
            args.mode == .dry_run,
        },
    );
}

fn randomMatrices(allocator: std.mem.Allocator, dims: Dimensions) !Matrices {
    const np = vx.withAllocator(allocator);
    const a_shape = [_]usize{ dims.m, dims.k };
    const b_shape = [_]usize{ dims.k, dims.n };
    const c_shape = [_]usize{ dims.m, dims.n };

    var a = try np.rand(f32, a_shape[0..], 0x4096_0001);
    errdefer a.deinit();
    var b = try np.rand(f32, b_shape[0..], 0x4096_0002);
    errdefer b.deinit();
    var c = try np.rand(f32, c_shape[0..], 0x4096_0003);
    errdefer c.deinit();

    return .{ .a = a, .b = b, .c = c };
}

fn runCpu(allocator: std.mem.Allocator, dims: Dimensions) !Summary {
    var matrices = try randomMatrices(allocator, dims);
    defer matrices.deinit();

    var product = try vx.matmul(matrices.a, matrices.b);
    defer product.deinit();

    var output = try vx.add(product, matrices.c);
    defer output.deinit();

    return summarize("cpu", vx.axiom_backend.BackendRoute.direct_cpu.label(), dims, output);
}

fn runCuda(allocator: std.mem.Allocator, dims: Dimensions) !Summary {
    if (!vx.axiom_cuda.enabled()) return error.CudaDisabled;

    var matrices = try randomMatrices(allocator, dims);
    defer matrices.deinit();

    const report = vx.axiom_backend.selectMatmul(f32, .prefer_cuda, matrices.a, matrices.b);
    if (report.selected != .axiom_cuda) return error.CudaNotSelected;

    const maybe_product = try vx.axiom_cuda.tryMatmulF32(matrices.a, matrices.b);
    var product = maybe_product orelse return error.CudaExecutionUnavailable;
    defer product.deinit();

    const maybe_output = try vx.axiom_cuda.tryAddF32(product, matrices.c);
    var output = maybe_output orelse return error.CudaExecutionUnavailable;
    defer output.deinit();

    return summarize("cuda", report.selected.label(), dims, output);
}

fn summarize(backend: []const u8, route: []const u8, dims: Dimensions, output: vx.Array(f32)) Summary {
    return .{
        .backend = backend,
        .route = route,
        .m = dims.m,
        .n = dims.n,
        .k = dims.k,
        .first = if (output.data.len == 0) 0 else output.data[0],
        .checksum = sampleChecksum(output.data),
        .output_elements = dims.outputElements(),
    };
}

fn sampleChecksum(values: []const f32) f64 {
    if (values.len == 0) return 0;
    const sample_count = @min(values.len, 1024);
    var total: f64 = 0;
    for (values[0..sample_count]) |value| total += value;
    if (values.len > sample_count) {
        const tail = values[values.len - sample_count ..];
        for (tail) |value| total += value;
    }
    return total;
}
