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
};

fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
}

fn fill(data: []f32) void {
    for (data, 0..) |*slot, index| {
        slot.* = @as(f32, @floatFromInt((index * 17 + 11) % 97)) * 0.015625 + 0.125;
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
        }
    }
    return options;
}

fn gflops(shape: Shape, elapsed_ns: i128) f64 {
    if (elapsed_ns <= 0) return 0;
    const ops = 2.0 * @as(f64, @floatFromInt(shape.m)) * @as(f64, @floatFromInt(shape.n)) * @as(f64, @floatFromInt(shape.k));
    return ops / @as(f64, @floatFromInt(elapsed_ns));
}

fn maxAbsDiff(lhs: []const f32, rhs: []const f32) f32 {
    var result: f32 = 0;
    for (lhs, rhs) |a, b| {
        result = @max(result, @abs(a - b));
    }
    return result;
}

pub fn main(init: std.process.Init) !void {
    const options = parseOptions(init);
    const shape = options.shape;
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(f32).empty(allocator, &.{ shape.m, shape.k });
    defer lhs.deinit();
    var rhs = try vx.Array(f32).empty(allocator, &.{ shape.k, shape.n });
    defer rhs.deinit();
    var prepared_out = try vx.Array(f32).empty(allocator, &.{ shape.m, shape.n });
    defer prepared_out.deinit();
    var normal_out = try vx.Array(f32).empty(allocator, &.{ shape.m, shape.n });
    defer normal_out.deinit();
    fill(lhs.data);
    fill(rhs.data);

    var prepared = try vx.PreparedF32Matmul.init(allocator, lhs, rhs);
    defer prepared.deinit();
    var prepared_column_out = try prepared.matmulColumnMajor();
    defer prepared_column_out.deinit();
    try prepared.matmulOut(prepared_out);
    try prepared.matmulColumnMajorOut(prepared_column_out);
    try lhs.matmulOut(rhs, normal_out);

    var prepared_ns: i128 = 0;
    var prepared_column_ns: i128 = 0;
    var normal_ns: i128 = 0;
    var sink: f32 = 0;
    for (0..options.iters) |_| {
        var start = nowNs(init.io);
        try prepared.matmulOut(prepared_out);
        prepared_ns += nowNs(init.io) - start;
        sink += prepared_out.data[0];

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
    try prepared_column_out.copyToSlice(prepared_out.data);
    std.debug.print(
        "vectra_prepared_matmul_f32 shape={d}x{d}x{d} iters={d} prepared_ns={} prepared_gflops={d:.3} prepared_column_ns={} prepared_column_gflops={d:.3} normal_ns={} normal_gflops={d:.3} ratio={d:.3} column_ratio={d:.3} max_diff={d:.6}\n",
        .{
            shape.m,
            shape.n,
            shape.k,
            options.iters,
            prepared_avg,
            gflops(shape, prepared_avg),
            prepared_column_avg,
            gflops(shape, prepared_column_avg),
            normal_avg,
            gflops(shape, normal_avg),
            gflops(shape, prepared_avg) / gflops(shape, normal_avg),
            gflops(shape, prepared_column_avg) / gflops(shape, normal_avg),
            maxAbsDiff(prepared_out.data, normal_out.data),
        },
    );
}
