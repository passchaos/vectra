const std = @import("std");
const vx = @import("vectra");

const Shape = struct {
    m: usize = 16,
    n: usize = 1024,
    k: usize = 16,
};

const Options = struct {
    shape: Shape = .{},
    iters: usize = 16,
};

fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
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
    if (m == 0 or n == 0 or k == 0) return;
    shape.* = .{ .m = m, .n = n, .k = k };
}

fn parseOptions(init: std.process.Init) Options {
    var options: Options = .{};
    var args = std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa) catch return options;
    defer args.deinit();
    _ = args.skip();

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

fn fill(comptime T: type, data: []T) void {
    for (data, 0..) |*slot, index| {
        const value: T = @floatFromInt((index * 17 + 11) % 97);
        slot.* = value * @as(T, 0.015625) + @as(T, 0.125);
    }
}

fn gflops(shape: Shape, elapsed_ns: i128) f64 {
    if (elapsed_ns <= 0) return 0;
    const ops = 2.0 * @as(f64, @floatFromInt(shape.m)) * @as(f64, @floatFromInt(shape.n)) * @as(f64, @floatFromInt(shape.k));
    return ops / @as(f64, @floatFromInt(elapsed_ns));
}

fn benchType(comptime T: type, init: std.process.Init, options: Options) !void {
    const shape = options.shape;
    const allocator = std.heap.smp_allocator;

    var lhs = try vx.Array(T).empty(allocator, &.{ shape.m, shape.k });
    defer lhs.deinit();
    var rhs = try vx.Array(T).empty(allocator, &.{ shape.k, shape.n });
    defer rhs.deinit();
    var addend = try vx.Array(T).full(allocator, &.{ shape.m, shape.n }, @as(T, 0.5));
    defer addend.deinit();
    fill(T, lhs.data);
    fill(T, rhs.data);

    var warm_column = (try vx.cpuMatmulColumnMajorResult(T, lhs, rhs)) orelse return error.Unsupported;
    warm_column.deinit();
    var warm_matmul = try lhs.matmul(rhs);
    warm_matmul.deinit();
    var warm_add = try lhs.matmulAdd(rhs, addend);
    warm_add.deinit();
    var warm_add_base = try lhs.matmulAdd(rhs, addend);
    var warm_add_sqrt = try warm_add_base.sqrt();
    warm_add_sqrt.deinit();
    warm_add_base.deinit();

    var column_ns: i128 = 0;
    var clone_materialize_ns: i128 = 0;
    var copy_to_slice_ns: i128 = 0;
    var copy_add_ns: i128 = 0;
    var copy_add_sqrt_ns: i128 = 0;
    var matmul_ns: i128 = 0;
    var matmul_add_ns: i128 = 0;
    var matmul_add_sqrt_ns: i128 = 0;
    var sink: T = 0;
    const scratch = try allocator.alloc(T, shape.m * shape.n);
    defer allocator.free(scratch);

    for (0..options.iters) |_| {
        var start = nowNs(init.io);
        var column = (try vx.cpuMatmulColumnMajorResult(T, lhs, rhs)) orelse return error.Unsupported;
        column_ns += nowNs(init.io) - start;

        start = nowNs(init.io);
        var materialized = try column.clone();
        clone_materialize_ns += nowNs(init.io) - start;
        sink += materialized.data[0];
        materialized.deinit();

        start = nowNs(init.io);
        try column.copyToSlice(scratch);
        copy_to_slice_ns += nowNs(init.io) - start;
        sink += scratch[0];

        start = nowNs(init.io);
        try column.copyToSlice(scratch);
        for (scratch, addend.data) |*slot, add_value| {
            slot.* += add_value;
        }
        copy_add_ns += nowNs(init.io) - start;
        sink += scratch[0];

        start = nowNs(init.io);
        try column.copyToSlice(scratch);
        for (scratch, addend.data) |*slot, add_value| {
            slot.* = @sqrt(slot.* + add_value);
        }
        copy_add_sqrt_ns += nowNs(init.io) - start;
        sink += scratch[0];
        column.deinit();

        start = nowNs(init.io);
        var matmul = try lhs.matmul(rhs);
        matmul_ns += nowNs(init.io) - start;
        sink += matmul.data[0];
        matmul.deinit();

        start = nowNs(init.io);
        var matmul_add = try lhs.matmulAdd(rhs, addend);
        matmul_add_ns += nowNs(init.io) - start;
        sink += matmul_add.data[0];
        matmul_add.deinit();

        start = nowNs(init.io);
        var matmul_add_base = try lhs.matmulAdd(rhs, addend);
        var matmul_add_sqrt = try matmul_add_base.sqrt();
        matmul_add_sqrt_ns += nowNs(init.io) - start;
        sink += matmul_add_sqrt.data[0];
        matmul_add_sqrt.deinit();
        matmul_add_base.deinit();
    }
    std.mem.doNotOptimizeAway(sink);

    const stdout = std.debug;
    const denom: i128 = @intCast(options.iters);
    stdout.print(
        "vectra_matmul_materialize dtype={s} shape={d}x{d}x{d} iters={d} column_ns={} column_gflops={d:.3} clone_materialize_ns={} copy_to_slice_ns={} copy_add_ns={} copy_add_sqrt_ns={} matmul_ns={} matmul_add_ns={} matmul_add_sqrt_ns={}\n",
        .{
            @typeName(T),
            shape.m,
            shape.n,
            shape.k,
            options.iters,
            @divTrunc(column_ns, denom),
            gflops(shape, @divTrunc(column_ns, denom)),
            @divTrunc(clone_materialize_ns, denom),
            @divTrunc(copy_to_slice_ns, denom),
            @divTrunc(copy_add_ns, denom),
            @divTrunc(copy_add_sqrt_ns, denom),
            @divTrunc(matmul_ns, denom),
            @divTrunc(matmul_add_ns, denom),
            @divTrunc(matmul_add_sqrt_ns, denom),
        },
    );
}

pub fn main(init: std.process.Init) !void {
    const options = parseOptions(init);
    try benchType(f64, init, options);
}
