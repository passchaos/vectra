//! Local Array performance smoke benchmark.
//!
//! Run with `zig build bench --release=fast` from the repository root.
//! Compare with `tools/bench_numpy_torch.py` under matching CPU thread settings.

const std = @import("std");
const vx = @import("vectra");

const ArrayF64 = vx.Array(f64);
const ArrayI32 = vx.Array(i32);

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

fn fillF64(a: ArrayF64) void {
    for (a.data, 0..) |*slot, i| {
        const base: f64 = @floatFromInt(i % 1024);
        slot.* = base * 0.001 + 0.25;
    }
}

fn fillI32(a: ArrayI32) void {
    for (a.data, 0..) |*slot, i| slot.* = @intCast(i % 1024);
}

fn benchAddArray(io: std.Io, a: ArrayF64, b: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.add(b);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchAddScalar(io: std.Io, a: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.addScalar(1.25);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchMulArray(io: std.Io, a: ArrayF64, b: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.mul(b);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchGtScalar(io: std.Io, a: ArrayF64, iters: usize) !f64 {
    var sink = false;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.gtScalar(0.5);
        sink = sink != out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchSumAll(io: std.Io, a: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.sum(null, false);
        sink += out.data[0];
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchMeanAll(io: std.Io, a: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.mean(null, false);
        sink += out.data[0];
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchPromotedAdd(io: std.Io, a: ArrayI32, b: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.addPromote(f64, b);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchStridedAddScalar(io: std.Io, a: ArrayF64, iters: usize) !f64 {
    var view = try a.slice1d(.{ .start = 0, .stop = null, .step = 2 });
    defer view.deinit();
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try view.addScalar(1.25);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchStridedAddArray(io: std.Io, a: ArrayF64, b: ArrayF64, iters: usize) !f64 {
    var lhs = try a.slice1d(.{ .start = 0, .stop = null, .step = 2 });
    defer lhs.deinit();
    var rhs = try b.slice1d(.{ .start = 0, .stop = null, .step = 2 });
    defer rhs.deinit();
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try lhs.add(rhs);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

fn benchMatmul(io: std.Io, a: ArrayF64, b: ArrayF64, iters: usize) !f64 {
    var sink: f64 = 0;
    const start = nowNs(io);
    for (0..iters) |_| {
        var out = try a.matmul(b);
        sink += out.data[0];
        std.mem.doNotOptimizeAway(out.data.ptr);
        out.deinit();
    }
    std.mem.doNotOptimizeAway(sink);
    const elapsed: f64 = @floatFromInt(nowNs(io) - start);
    return elapsed / @as(f64, @floatFromInt(iters));
}

pub fn main() !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const allocator = std.heap.smp_allocator;
    const n: usize = 1_000_000;

    var a = try ArrayF64.empty(allocator, &.{n});
    defer a.deinit();
    var b = try ArrayF64.empty(allocator, &.{n});
    defer b.deinit();
    var ai = try ArrayI32.empty(allocator, &.{n});
    defer ai.deinit();
    fillF64(a);
    fillF64(b);
    fillI32(ai);

    var warm_add = try a.add(b);
    warm_add.deinit();
    var warm_scalar = try a.addScalar(1.25);
    warm_scalar.deinit();
    var warm_mul = try a.mul(b);
    warm_mul.deinit();
    var warm_gt = try a.gtScalar(0.5);
    warm_gt.deinit();
    var warm_sum = try a.sum(null, false);
    warm_sum.deinit();
    var warm_mean = try a.mean(null, false);
    warm_mean.deinit();
    var warm_promote = try ai.addPromote(f64, b);
    warm_promote.deinit();

    const m: usize = 256;
    var ma = try ArrayF64.empty(allocator, &.{ m, m });
    defer ma.deinit();
    var mb = try ArrayF64.empty(allocator, &.{ m, m });
    defer mb.deinit();
    fillF64(ma);
    fillF64(mb);
    var warm_mm = try ma.matmul(mb);
    warm_mm.deinit();

    std.debug.print("bench,items,ns_per_op\n", .{});
    std.debug.print("add_array_f64,{d},{d:.3}\n", .{ n, try benchAddArray(io, a, b, 120) });
    std.debug.print("add_scalar_f64,{d},{d:.3}\n", .{ n, try benchAddScalar(io, a, 120) });
    std.debug.print("mul_array_f64,{d},{d:.3}\n", .{ n, try benchMulArray(io, a, b, 120) });
    std.debug.print("gt_scalar_f64,{d},{d:.3}\n", .{ n, try benchGtScalar(io, a, 120) });
    std.debug.print("sum_all_f64,{d},{d:.3}\n", .{ n, try benchSumAll(io, a, 240) });
    std.debug.print("mean_all_f64,{d},{d:.3}\n", .{ n, try benchMeanAll(io, a, 120) });
    std.debug.print("promoted_add_i32_f64,{d},{d:.3}\n", .{ n, try benchPromotedAdd(io, ai, b, 120) });
    std.debug.print("strided_add_scalar_f64,{d},{d:.3}\n", .{ n / 2, try benchStridedAddScalar(io, a, 120) });
    std.debug.print("strided_add_array_f64,{d},{d:.3}\n", .{ n / 2, try benchStridedAddArray(io, a, b, 120) });
    std.debug.print("matmul_f64,{d}x{d},{d:.3}\n", .{ m, m, try benchMatmul(io, ma, mb, 12) });
}
