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

const TimingStats = struct {
    min_ns: i128,
    median_ns: i128,
    avg_ns: i128,
};

fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
}

fn timingStats(allocator: std.mem.Allocator, timings: []const i128) !TimingStats {
    const sorted = try allocator.dupe(i128, timings);
    defer allocator.free(sorted);
    std.mem.sort(i128, sorted, {}, std.sort.asc(i128));
    var total: i128 = 0;
    for (timings) |timing| total += timing;
    return .{
        .min_ns = sorted[0],
        .median_ns = sorted[sorted.len / 2],
        .avg_ns = @divTrunc(total, @as(i128, @intCast(timings.len))),
    };
}

fn printStatsFields(comptime label: []const u8, shape: Shape, stats: TimingStats) void {
    std.debug.print(
        " {s}_ns={}/{}/{} {s}_gflops={d:.3}/{d:.3}/{d:.3}",
        .{
            label,
            stats.min_ns,
            stats.median_ns,
            stats.avg_ns,
            label,
            gflops(shape, stats.min_ns),
            gflops(shape, stats.median_ns),
            gflops(shape, stats.avg_ns),
        },
    );
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

    const prepared_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(prepared_timings);
    const prepared_column_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(prepared_column_timings);
    const normal_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(normal_timings);
    var sink: T = 0;
    for (0..options.iters) |iteration| {
        var start = nowNs(init.io);
        if (prepared_transposed) |*value| {
            try value.matmulOut(prepared_out);
            prepared_timings[iteration] = nowNs(init.io) - start;
            sink += prepared_out.data[0];
        } else {
            try prepared.matmulOut(prepared_out);
            prepared_timings[iteration] = nowNs(init.io) - start;
            sink += prepared_out.data[0];
        }

        start = nowNs(init.io);
        try prepared.matmulColumnMajorOut(prepared_column_out);
        prepared_column_timings[iteration] = nowNs(init.io) - start;
        sink += prepared_column_out.data[0];

        start = nowNs(init.io);
        try lhs.matmulOut(rhs, normal_out);
        normal_timings[iteration] = nowNs(init.io) - start;
        sink += normal_out.data[0];
    }
    std.mem.doNotOptimizeAway(sink);

    const prepared_stats = try timingStats(allocator, prepared_timings);
    const prepared_column_stats = try timingStats(allocator, prepared_column_timings);
    const normal_stats = try timingStats(allocator, normal_timings);
    if (prepared_transposed == null) try prepared_column_out.copyToSlice(prepared_out.data);
    std.debug.print(
        "vectra_prepared_matmul dtype={s} shape={d}x{d}x{d} iters={d} transposed_prepared={}",
        .{
            if (T == f32) "f32" else "f64",
            shape.m,
            shape.n,
            shape.k,
            options.iters,
            options.transposed_prepared and T == f32,
        },
    );
    printStatsFields("prepared", shape, prepared_stats);
    printStatsFields("prepared_column", shape, prepared_column_stats);
    std.debug.print(" prepared_32x8_ns=0 prepared_32x8_gflops=0.000", .{});
    printStatsFields("normal", shape, normal_stats);
    std.debug.print(
        " ratio={d:.3} column_ratio={d:.3} prepared_32x8_ratio=0.000 max_diff={d:.6} prepared_32x8_diff=0.000000\n",
        .{
            gflops(shape, prepared_stats.median_ns) / gflops(shape, normal_stats.median_ns),
            gflops(shape, prepared_column_stats.median_ns) / gflops(shape, normal_stats.median_ns),
            maxAbsDiff(T, prepared_out.data, normal_out.data),
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

    const prepared_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(prepared_timings);
    const prepared_column_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(prepared_column_timings);
    const prepared_32x8_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(prepared_32x8_timings);
    const normal_timings = try allocator.alloc(i128, options.iters);
    defer allocator.free(normal_timings);
    var sink: f64 = 0;
    for (0..options.iters) |iteration| {
        var start = nowNs(init.io);
        if (prepared_transposed) |*value| {
            try value.matmulOut(prepared_out);
        } else {
            try prepared.matmulOut(prepared_out);
        }
        prepared_timings[iteration] = nowNs(init.io) - start;
        sink += prepared_out.data[0];

        start = nowNs(init.io);
        try prepared.matmulColumnMajorOut(prepared_column_out);
        prepared_column_timings[iteration] = nowNs(init.io) - start;
        sink += prepared_column_out.data[0];

        start = nowNs(init.io);
        try prepared.matmulColumnMajor32x8DiagnosticOut(prepared_32x8_out);
        prepared_32x8_timings[iteration] = nowNs(init.io) - start;
        sink += prepared_32x8_out.data[0];

        start = nowNs(init.io);
        try lhs.matmulOut(rhs, normal_out);
        normal_timings[iteration] = nowNs(init.io) - start;
        sink += normal_out.data[0];
    }
    std.mem.doNotOptimizeAway(sink);

    const prepared_stats = try timingStats(allocator, prepared_timings);
    const prepared_column_stats = try timingStats(allocator, prepared_column_timings);
    const prepared_32x8_stats = try timingStats(allocator, prepared_32x8_timings);
    const normal_stats = try timingStats(allocator, normal_timings);
    if (prepared_transposed == null) try prepared_column_out.copyToSlice(prepared_out.data);
    var row_major_32x8 = try vx.Array(f64).empty(allocator, &.{ shape.m, shape.n });
    defer row_major_32x8.deinit();
    try prepared_32x8_out.copyToSlice(row_major_32x8.data);
    std.debug.print(
        "vectra_prepared_matmul dtype=f64 shape={d}x{d}x{d} iters={d} transposed_prepared={}",
        .{
            shape.m,
            shape.n,
            shape.k,
            options.iters,
            options.transposed_prepared,
        },
    );
    printStatsFields("prepared", shape, prepared_stats);
    printStatsFields("prepared_column", shape, prepared_column_stats);
    printStatsFields("prepared_32x8", shape, prepared_32x8_stats);
    printStatsFields("normal", shape, normal_stats);
    std.debug.print(
        " ratio={d:.3} column_ratio={d:.3} prepared_32x8_ratio={d:.3} max_diff={d:.6} prepared_32x8_diff={d:.6}\n",
        .{
            gflops(shape, prepared_stats.median_ns) / gflops(shape, normal_stats.median_ns),
            gflops(shape, prepared_column_stats.median_ns) / gflops(shape, normal_stats.median_ns),
            gflops(shape, prepared_32x8_stats.median_ns) / gflops(shape, normal_stats.median_ns),
            maxAbsDiff(f64, prepared_out.data, normal_out.data),
            maxAbsDiff(f64, row_major_32x8.data, normal_out.data),
        },
    );
}
