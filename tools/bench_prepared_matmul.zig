const std = @import("std");
const vx = @import("vectra");

const Shape = struct {
    m: usize = 1024,
    n: usize = 1024,
    k: usize = 1024,
};

const Options = struct {
    shape: Shape = .{},
    iters: usize = 3,
    dtype: DType = .f32,
    transposed_prepared: bool = false,
};

const DType = enum {
    f32,
    f64,
};

fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
}

fn fill(comptime T: type, data: []T) void {
    for (data, 0..) |*slot, index| {
        slot.* = @floatCast(@as(f64, @floatFromInt((index * 17 + 11) % 97)) * 0.015625 + 0.125);
    }
}

fn parseShape(text: []const u8, shape: *Shape) void {
    var parts = std.mem.splitScalar(u8, text, 'x');
    const m_text = parts.next() orelse return;
    const n_text = parts.next() orelse return;
    const k_text = parts.next() orelse return;
    if (parts.next() != null) return;
    const m = std.fmt.parseUnsigned(usize, m_text, 10) catch return;
    const n = std.fmt.parseUnsigned(usize, n_text, 10) catch return;
    const k = std.fmt.parseUnsigned(usize, k_text, 10) catch return;
    if (m != 0 and n != 0 and k != 0) shape.* = .{ .m = m, .n = n, .k = k };
}

fn parseOptions(init: std.process.Init) Options {
    var options: Options = .{};
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--shape=")) {
            parseShape(arg["--shape=".len..], &options.shape);
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            const value = std.fmt.parseUnsigned(usize, arg["--iters=".len..], 10) catch continue;
            if (value != 0) options.iters = value;
        } else if (std.mem.eql(u8, arg, "--dtype=f32")) {
            options.dtype = .f32;
        } else if (std.mem.eql(u8, arg, "--dtype=f64")) {
            options.dtype = .f64;
        } else if (std.mem.eql(u8, arg, "--transposed-prepared")) {
            options.transposed_prepared = true;
        }
    }
    return options;
}

fn gflops(shape: Shape, elapsed_ns: i128) f64 {
    if (elapsed_ns <= 0) return 0;
    const ops = 2.0 * @as(f64, @floatFromInt(shape.m)) * @as(f64, @floatFromInt(shape.n)) * @as(f64, @floatFromInt(shape.k));
    return ops / @as(f64, @floatFromInt(elapsed_ns));
}

fn maxAbsDiff(comptime T: type, lhs: []const T, rhs: []const T) f64 {
    var result: f64 = 0;
    for (lhs, rhs) |a, b| {
        result = @max(result, @abs(@as(f64, @floatCast(a)) - @as(f64, @floatCast(b))));
    }
    return result;
}

pub fn main(init: std.process.Init) !void {
    const options = parseOptions(init);
    switch (options.dtype) {
        .f32 => try run(f32, init, options),
        .f64 => try runF64(init, options),
    }
}

fn PreparedMatmul(comptime T: type) type {
    return switch (T) {
        f32 => vx.PreparedF32Matmul,
        f64 => vx.PreparedF64Matmul,
        else => @compileError("unsupported prepared matmul dtype"),
    };
}

fn run(comptime T: type, init: std.process.Init, options: Options) !void {
    const shape = options.shape;
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(T).empty(allocator, &.{ shape.m, shape.k });
    defer lhs.deinit();
    var rhs = try vx.Array(T).empty(allocator, &.{ shape.k, shape.n });
    defer rhs.deinit();
    var prepared_out = try vx.Array(T).empty(allocator, &.{ shape.m, shape.n });
    defer prepared_out.deinit();
    var normal_out = try vx.Array(T).empty(allocator, &.{ shape.m, shape.n });
    defer normal_out.deinit();
    fill(T, lhs.data);
    fill(T, rhs.data);

    var prepared = try PreparedMatmul(T).init(allocator, lhs, rhs);
    defer prepared.deinit();
    var prepared_transposed = if (options.transposed_prepared and T == f32)
        try vx.PreparedF32TransposedMatmul.init(allocator, lhs, rhs)
    else
        null;
    defer if (prepared_transposed) |*value| value.deinit();
    var prepared_column_out = try prepared.matmulColumnMajor();
    defer prepared_column_out.deinit();
    try prepared.matmulOut(prepared_out);
    try prepared.matmulColumnMajorOut(prepared_column_out);
    if (prepared_transposed) |*value| try value.matmulOut(prepared_out);
    try lhs.matmulOut(rhs, normal_out);

    var prepared_ns: i128 = 0;
    var prepared_column_ns: i128 = 0;
    var normal_ns: i128 = 0;
    var sink: T = 0;
    for (0..options.iters) |_| {
        var start = nowNs(init.io);
        if (prepared_transposed) |*value| {
            try value.matmulOut(prepared_out);
            prepared_ns += nowNs(init.io) - start;
            sink += prepared_out.data[0];
        } else {
            try prepared.matmulOut(prepared_out);
            prepared_ns += nowNs(init.io) - start;
            sink += prepared_out.data[0];
        }

        start = nowNs(init.io);
        try prepared.matmulColumnMajorOut(prepared_column_out);
        prepared_column_ns += nowNs(init.io) - start;
        sink += prepared_column_out.data[0];

        start = nowNs(init.io);
        try lhs.matmulOut(rhs, normal_out);
        normal_ns += nowNs(init.io) - start;
        sink += normal_out.data[0];
    }
    std.mem.doNotOptimizeAway(sink);

    const denom: i128 = @intCast(options.iters);
    const prepared_avg = @divTrunc(prepared_ns, denom);
    const prepared_column_avg = @divTrunc(prepared_column_ns, denom);
    const normal_avg = @divTrunc(normal_ns, denom);
    if (prepared_transposed == null) try prepared_column_out.copyToSlice(prepared_out.data);
    std.debug.print(
        "vectra_prepared_matmul dtype={s} shape={d}x{d}x{d} iters={d} transposed_prepared={} prepared_ns={} prepared_gflops={d:.3} prepared_column_ns={} prepared_column_gflops={d:.3} prepared_32x8_ns={} prepared_32x8_gflops={d:.3} normal_ns={} normal_gflops={d:.3} ratio={d:.3} column_ratio={d:.3} prepared_32x8_ratio={d:.3} max_diff={d:.6} prepared_32x8_diff={d:.6}\n",
        .{
            if (T == f32) "f32" else "f64",
            shape.m,
            shape.n,
            shape.k,
            options.iters,
            options.transposed_prepared and T == f32,
            prepared_avg,
            gflops(shape, prepared_avg),
            prepared_column_avg,
            gflops(shape, prepared_column_avg),
            0,
            0.0,
            normal_avg,
            gflops(shape, normal_avg),
            gflops(shape, prepared_avg) / gflops(shape, normal_avg),
            gflops(shape, prepared_column_avg) / gflops(shape, normal_avg),
            0.0,
            maxAbsDiff(T, prepared_out.data, normal_out.data),
            0.0,
        },
    );
}

fn runF64(init: std.process.Init, options: Options) !void {
    const shape = options.shape;
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(f64).empty(allocator, &.{ shape.m, shape.k });
    defer lhs.deinit();
    var rhs = try vx.Array(f64).empty(allocator, &.{ shape.k, shape.n });
    defer rhs.deinit();
    var prepared_out = try vx.Array(f64).empty(allocator, &.{ shape.m, shape.n });
    defer prepared_out.deinit();
    var normal_out = try vx.Array(f64).empty(allocator, &.{ shape.m, shape.n });
    defer normal_out.deinit();
    fill(f64, lhs.data);
    fill(f64, rhs.data);

    var prepared = try vx.PreparedF64Matmul.init(allocator, lhs, rhs);
    defer prepared.deinit();
    var prepared_transposed = if (options.transposed_prepared)
        try vx.PreparedF64TransposedMatmul.init(allocator, lhs, rhs)
    else
        null;
    defer if (prepared_transposed) |*value| value.deinit();
    var prepared_column_out = try prepared.matmulColumnMajor();
    defer prepared_column_out.deinit();
    var prepared_32x8_out = try prepared.matmulColumnMajor();
    defer prepared_32x8_out.deinit();
    try prepared.matmulOut(prepared_out);
    try prepared.matmulColumnMajorOut(prepared_column_out);
    try prepared.matmulColumnMajor32x8DiagnosticOut(prepared_32x8_out);
    if (prepared_transposed) |*value| try value.matmulOut(prepared_out);
    try lhs.matmulOut(rhs, normal_out);

    var prepared_ns: i128 = 0;
    var prepared_column_ns: i128 = 0;
    var prepared_32x8_ns: i128 = 0;
    var normal_ns: i128 = 0;
    var sink: f64 = 0;
    for (0..options.iters) |_| {
        var start = nowNs(init.io);
        if (prepared_transposed) |*value| {
            try value.matmulOut(prepared_out);
        } else {
            try prepared.matmulOut(prepared_out);
        }
        prepared_ns += nowNs(init.io) - start;
        sink += prepared_out.data[0];

        start = nowNs(init.io);
        try prepared.matmulColumnMajorOut(prepared_column_out);
        prepared_column_ns += nowNs(init.io) - start;
        sink += prepared_column_out.data[0];

        start = nowNs(init.io);
        try prepared.matmulColumnMajor32x8DiagnosticOut(prepared_32x8_out);
        prepared_32x8_ns += nowNs(init.io) - start;
        sink += prepared_32x8_out.data[0];

        start = nowNs(init.io);
        try lhs.matmulOut(rhs, normal_out);
        normal_ns += nowNs(init.io) - start;
        sink += normal_out.data[0];
    }
    std.mem.doNotOptimizeAway(sink);

    const denom: i128 = @intCast(options.iters);
    const prepared_avg = @divTrunc(prepared_ns, denom);
    const prepared_column_avg = @divTrunc(prepared_column_ns, denom);
    const prepared_32x8_avg = @divTrunc(prepared_32x8_ns, denom);
    const normal_avg = @divTrunc(normal_ns, denom);
    if (prepared_transposed == null) try prepared_column_out.copyToSlice(prepared_out.data);
    var row_major_32x8 = try vx.Array(f64).empty(allocator, &.{ shape.m, shape.n });
    defer row_major_32x8.deinit();
    try prepared_32x8_out.copyToSlice(row_major_32x8.data);
    std.debug.print(
        "vectra_prepared_matmul dtype=f64 shape={d}x{d}x{d} iters={d} transposed_prepared={} prepared_ns={} prepared_gflops={d:.3} prepared_column_ns={} prepared_column_gflops={d:.3} prepared_32x8_ns={} prepared_32x8_gflops={d:.3} normal_ns={} normal_gflops={d:.3} ratio={d:.3} column_ratio={d:.3} prepared_32x8_ratio={d:.3} max_diff={d:.6} prepared_32x8_diff={d:.6}\n",
        .{
            shape.m,
            shape.n,
            shape.k,
            options.iters,
            options.transposed_prepared,
            prepared_avg,
            gflops(shape, prepared_avg),
            prepared_column_avg,
            gflops(shape, prepared_column_avg),
            prepared_32x8_avg,
            gflops(shape, prepared_32x8_avg),
            normal_avg,
            gflops(shape, normal_avg),
            gflops(shape, prepared_avg) / gflops(shape, normal_avg),
            gflops(shape, prepared_column_avg) / gflops(shape, normal_avg),
            gflops(shape, prepared_32x8_avg) / gflops(shape, normal_avg),
            maxAbsDiff(f64, prepared_out.data, normal_out.data),
            maxAbsDiff(f64, row_major_32x8.data, normal_out.data),
        },
    );
}
