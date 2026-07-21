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

pub fn main(init: std.process.Init) !void {
    var np = vx.withAllocator(init.gpa);

    // const device = vx.Device.cuda(0);
    const device = vx.Device.cpu;

    var a = try np.randWith(f32, &.{ production.m, production.k }, .{ .device = device });
    defer a.deinit();

    var b = try np.randWith(f32, &.{ production.k, production.n }, .{ .device = device });
    defer b.deinit();

    // Build the addend at its final value in CUDA storage.  In-place scalar
    // assignment currently goes through host ArrayView semantics, so using
    // `fullOn` avoids accidentally requesting a host view of a CUDA array.
    var c = try vx.Array(f32).fullOn(init.gpa, &.{ production.m, production.n }, 50.0, device);
    defer c.deinit();

    for (0..20000) |_| {
        const begin = std.Io.Timestamp.now(init.io, .real);
        var h = try a.matmul(b);
        defer h.deinit();

        var i = try h.add(c);
        defer i.deinit();

        var ii = i;
        // var ii = try i.exp();
        // defer ii.deinit();

        var j = try ii.materializeAndSynchronize();
        defer j.deinit();

        const cost = std.Io.Timestamp.untilNow(begin, init.io, .real);

        // std.debug.print("a first: {f} h: {f} j: {f}\n", .{ a, h, j });
        std.debug.print("device: {} matmul cost: {}\n", .{ device, cost.toMicroseconds() });
    }
}
