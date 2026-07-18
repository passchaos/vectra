//! Smoke gate for a bounded NumPy/PyTorch-style einsum front-end.
//!
//! Vectra does not claim full NumPy einsum syntax yet.  This smoke verifies
//! representative contraction strings that lower to existing Array primitives,
//! which in turn keep using the Axiom backend selection paths.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var a = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    var mm = try vx.einsum("ij,jk->ik", a, b);
    defer mm.deinit();

    var v1 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer v1.deinit();
    var v2 = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30 }, &.{3});
    defer v2.deinit();
    var dot = try vx.einsum("i,i->", v1, v2);
    defer dot.deinit();
    var outer = try vx.einsum("i,j->ij", v1, v2);
    defer outer.deinit();
    var matvec = try vx.einsum("ij,j->i", a, v1);
    defer matvec.deinit();

    const unsupported_rejected = blk: {
        var bad = vx.einsum("ij->ji", a, b) catch |err| {
            break :blk err == error.InvalidShape;
        };
        bad.deinit();
        break :blk false;
    };

    const ok = eql(f32, mm.data, &.{ 58, 64, 139, 154 }) and
        std.mem.eql(usize, mm.shape, &.{ 2, 2 }) and
        std.mem.eql(usize, dot.shape, &.{}) and
        eql(f32, dot.data, &.{140}) and
        std.mem.eql(usize, outer.shape, &.{ 3, 3 }) and
        eql(f32, outer.data, &.{ 10, 20, 30, 20, 40, 60, 30, 60, 90 }) and
        std.mem.eql(usize, matvec.shape, &.{2}) and
        eql(f32, matvec.data, &.{ 14, 32 }) and
        unsupported_rejected;

    var stdout_buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_einsum_smoke\",\"ok\":{},\"matmul_ok\":{},\"dot_ok\":{},\"outer_ok\":{},\"matvec_ok\":{},\"unsupported_rejected\":{}}}\n",
        .{
            ok,
            eql(f32, mm.data, &.{ 58, 64, 139, 154 }),
            eql(f32, dot.data, &.{140}),
            eql(f32, outer.data, &.{ 10, 20, 30, 20, 40, 60, 30, 60, 90 }),
            eql(f32, matvec.data, &.{ 14, 32 }),
            unsupported_rejected,
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}
