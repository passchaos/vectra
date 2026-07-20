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

    const device = vx.Device.cuda(0);

    var a = try np.randWith(vx.BFloat16, &.{ production.m, production.k }, .{ .device = device });
    defer a.deinit();

    var b = try np.randWith(vx.BFloat16, &.{ production.k, production.n }, .{ .device = device });
    defer b.deinit();

    const begin = std.Io.Timestamp.now(init.io, .real);
    var c = try a.matmul(b);

    defer c.deinit();

    var d = try c.add(c);
    defer d.deinit();

    var e = try d.materializeAndSynchronize();
    defer e.deinit();

    const cost = std.Io.Timestamp.untilNow(begin, init.io, .real);

    std.debug.print("a first: {f} c: {f}\n", .{ a, c });
    std.debug.print("device: {} matmul cost: {}\n", .{ device, cost.toMilliseconds() });
}
