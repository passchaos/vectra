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
    var mm_implicit = try vx.einsum("ij,jk", a, b);
    defer mm_implicit.deinit();

    var v1 = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3 }, &.{3});
    defer v1.deinit();
    var v2 = try vx.Array(f32).fromSlice(allocator, &.{ 10, 30, 50 }, &.{3});
    defer v2.deinit();
    var dot = try vx.einsum("i,i->", v1, v2);
    defer dot.deinit();
    var dot_implicit = try vx.einsum("i,i", v1, v2);
    defer dot_implicit.deinit();
    var outer = try vx.einsum("i,j->ij", v1, v2);
    defer outer.deinit();
    var outer_implicit = try vx.einsum("i,j", v1, v2);
    defer outer_implicit.deinit();
    var matvec = try vx.einsum("ij,j->i", a, v1);
    defer matvec.deinit();
    var vecmat = try vx.einsum("i,ij->j", v1, b);
    defer vecmat.deinit();
    var transposed_outer = try vx.einsum("i,j->ji", v1, v2);
    defer transposed_outer.deinit();
    var tensor3 = try vx.Array(f32).fromSlice(allocator, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    }, &.{ 2, 2, 3 });
    defer tensor3.deinit();
    var contract_rhs = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2,
        3, 4,
        5, 6,
    }, &.{ 3, 2 });
    defer contract_rhs.deinit();
    var generic_contract = try vx.einsum("abc,cd->abd", tensor3, contract_rhs);
    defer generic_contract.deinit();
    var batch_lhs = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }, &.{ 2, 2, 2 });
    defer batch_lhs.deinit();
    var batch_rhs = try vx.Array(f32).fromSlice(allocator, &.{
        1, 0, 0, 1,
        1, 1, 1, 1,
    }, &.{ 2, 2, 2 });
    defer batch_rhs.deinit();
    var batched = try vx.einsum("bij,bjk->bik", batch_lhs, batch_rhs);
    defer batched.deinit();
    var batched_implicit = try vx.einsum("bij,bjk", batch_lhs, batch_rhs);
    defer batched_implicit.deinit();

    const unsupported_rejected = blk: {
        var bad = vx.einsum("ij->ji", a, b) catch |err| {
            break :blk err == error.InvalidShape;
        };
        bad.deinit();
        break :blk false;
    };

    const ok = eql(f32, mm.data, &.{ 58, 64, 139, 154 }) and
        eql(f32, mm_implicit.data, &.{ 58, 64, 139, 154 }) and
        std.mem.eql(usize, mm.shape, &.{ 2, 2 }) and
        std.mem.eql(usize, dot.shape, &.{}) and
        eql(f32, dot.data, &.{220}) and
        eql(f32, dot_implicit.data, &.{220}) and
        std.mem.eql(usize, outer.shape, &.{ 3, 3 }) and
        eql(f32, outer.data, &.{ 10, 30, 50, 20, 60, 100, 30, 90, 150 }) and
        eql(f32, outer_implicit.data, &.{ 10, 30, 50, 20, 60, 100, 30, 90, 150 }) and
        std.mem.eql(usize, matvec.shape, &.{2}) and
        eql(f32, matvec.data, &.{ 14, 32 }) and
        std.mem.eql(usize, vecmat.shape, &.{2}) and
        eql(f32, vecmat.data, &.{ 58, 64 }) and
        std.mem.eql(usize, transposed_outer.shape, &.{ 3, 3 }) and
        eql(f32, transposed_outer.data, &.{ 10, 20, 30, 30, 60, 90, 50, 100, 150 }) and
        std.mem.eql(usize, generic_contract.shape, &.{ 2, 2, 2 }) and
        eql(f32, generic_contract.data, &.{ 22, 28, 49, 64, 76, 100, 103, 136 }) and
        std.mem.eql(usize, batched.shape, &.{ 2, 2, 2 }) and
        eql(f32, batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }) and
        eql(f32, batched_implicit.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }) and
        unsupported_rejected;

    var stdout_buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_einsum_smoke\",\"ok\":{},\"matmul_ok\":{},\"implicit_output_ok\":{},\"dot_ok\":{},\"outer_ok\":{},\"matvec_ok\":{},\"vecmat_ok\":{},\"reordered_ok\":{},\"generic_contract_ok\":{},\"batched_matmul_ok\":{},\"unsupported_rejected\":{}}}\n",
        .{
            ok,
            eql(f32, mm.data, &.{ 58, 64, 139, 154 }),
            eql(f32, mm_implicit.data, &.{ 58, 64, 139, 154 }) and eql(f32, dot_implicit.data, &.{220}) and eql(f32, outer_implicit.data, &.{ 10, 30, 50, 20, 60, 100, 30, 90, 150 }) and eql(f32, batched_implicit.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }),
            eql(f32, dot.data, &.{220}),
            eql(f32, outer.data, &.{ 10, 30, 50, 20, 60, 100, 30, 90, 150 }),
            eql(f32, matvec.data, &.{ 14, 32 }),
            eql(f32, vecmat.data, &.{ 58, 64 }),
            eql(f32, transposed_outer.data, &.{ 10, 20, 30, 30, 60, 90, 50, 100, 150 }),
            eql(f32, generic_contract.data, &.{ 22, 28, 49, 64, 76, 100, 103, 136 }),
            eql(f32, batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }),
            unsupported_rejected,
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}
