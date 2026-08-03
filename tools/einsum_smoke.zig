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
    var ellipsis_batched = try vx.einsum("...ij,...jk->...ik", batch_lhs, batch_rhs);
    defer ellipsis_batched.deinit();
    var ellipsis_batched_implicit = try vx.einsum("...ij,...jk", batch_lhs, batch_rhs);
    defer ellipsis_batched_implicit.deinit();
    var batch_vec = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 1, 1 }, &.{ 2, 2 });
    defer batch_vec.deinit();
    var ellipsis_matvec = try vx.einsum("...ij,...j->...i", batch_lhs, batch_vec);
    defer ellipsis_matvec.deinit();
    var ellipsis_matvec_implicit = try vx.einsum("...ij,...j", batch_lhs, batch_vec);
    defer ellipsis_matvec_implicit.deinit();
    var ellipsis_vecmat = try vx.einsum("...i,...ij->...j", batch_vec, batch_rhs);
    defer ellipsis_vecmat.deinit();
    var ellipsis_vecmat_implicit = try vx.einsum("...i,...ij", batch_vec, batch_rhs);
    defer ellipsis_vecmat_implicit.deinit();
    var batch_dot_lhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer batch_dot_lhs.deinit();
    var batch_dot_rhs = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30, 1, 1, 1 }, &.{ 2, 3 });
    defer batch_dot_rhs.deinit();
    var ellipsis_dot = try vx.einsum("...i,...i->...", batch_dot_lhs, batch_dot_rhs);
    defer ellipsis_dot.deinit();
    var ellipsis_dot_implicit = try vx.einsum("...i,...i", batch_dot_lhs, batch_dot_rhs);
    defer ellipsis_dot_implicit.deinit();
    var rank4_lhs = try vx.Array(f32).fromSlice(allocator, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    }, &.{ 2, 2, 2, 2 });
    defer rank4_lhs.deinit();
    var rank4_rhs = try vx.Array(f32).fromSlice(allocator, &.{
        1, 0, 0, 1,
        1, 1, 1, 1,
        2, 0, 0, 2,
        0, 1, 1, 0,
    }, &.{ 2, 2, 2, 2 });
    defer rank4_rhs.deinit();
    var rank4_batched = try vx.einsum("abij,abjk->abik", rank4_lhs, rank4_rhs);
    defer rank4_batched.deinit();
    var rank4_batched_implicit = try vx.einsum("abij,abjk", rank4_lhs, rank4_rhs);
    defer rank4_batched_implicit.deinit();
    var unary_identity = try vx.einsumUnary("ij->ij", a);
    defer unary_identity.deinit();
    var unary_transpose = try vx.einsumUnary("ij->ji", a);
    defer unary_transpose.deinit();
    var unary_row_sum = try vx.einsumUnary("ij->i", a);
    defer unary_row_sum.deinit();
    var unary_col_sum = try vx.einsumUnary("ij->j", a);
    defer unary_col_sum.deinit();
    var square = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer square.deinit();
    var unary_diag = try vx.einsumUnary("ii->i", square);
    defer unary_diag.deinit();
    var unary_trace = try vx.einsumUnary("ii->", square);
    defer unary_trace.deinit();
    var unary_trace_implicit = try vx.einsumUnary("ii", square);
    defer unary_trace_implicit.deinit();
    var batched_square = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 2, 2 });
    defer batched_square.deinit();
    var batched_diag = try vx.einsumUnary("bii->bi", batched_square);
    defer batched_diag.deinit();
    var batched_diag_reordered = try vx.einsumUnary("bii->ib", batched_square);
    defer batched_diag_reordered.deinit();
    var batched_trace = try vx.einsumUnary("bii->b", batched_square);
    defer batched_trace.deinit();
    var batched_trace_implicit = try vx.einsumUnary("bii", batched_square);
    defer batched_trace_implicit.deinit();
    var chain_rhs = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer chain_rhs.deinit();
    var ternary_chain = try vx.einsum3("ij,jk,kl->il", a, b, chain_rhs);
    defer ternary_chain.deinit();
    var ternary_chain_implicit = try vx.einsum3("ij,jk,kl", a, b, chain_rhs);
    defer ternary_chain_implicit.deinit();
    var rank4_ternary_chain = try vx.einsum3("abij,abjk,abkl->abil", rank4_lhs, rank4_rhs, rank4_rhs);
    defer rank4_ternary_chain.deinit();
    var rank4_ternary_chain_implicit = try vx.einsum3("abij,abjk,abkl", rank4_lhs, rank4_rhs, rank4_rhs);
    defer rank4_ternary_chain_implicit.deinit();
    var same_label_elementwise = try vx.einsum("ij,ij->ij", a, a);
    defer same_label_elementwise.deinit();
    var same_label_rows = try vx.einsum("ij,ij->i", a, a);
    defer same_label_rows.deinit();
    var same_label_all = try vx.einsum("ij,ij->", a, a);
    defer same_label_all.deinit();
    var same_label_reordered = try vx.einsum("ij,ij->ji", a, a);
    defer same_label_reordered.deinit();

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
        eql(f32, ellipsis_batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }) and
        eql(f32, ellipsis_batched_implicit.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }) and
        std.mem.eql(usize, ellipsis_matvec.shape, &.{ 2, 2 }) and
        eql(f32, ellipsis_matvec.data, &.{ 50, 110, 11, 15 }) and
        eql(f32, ellipsis_matvec_implicit.data, &.{ 50, 110, 11, 15 }) and
        std.mem.eql(usize, ellipsis_vecmat.shape, &.{ 2, 2 }) and
        eql(f32, ellipsis_vecmat.data, &.{ 10, 20, 2, 2 }) and
        eql(f32, ellipsis_vecmat_implicit.data, &.{ 10, 20, 2, 2 }) and
        std.mem.eql(usize, ellipsis_dot.shape, &.{2}) and
        eql(f32, ellipsis_dot.data, &.{ 140, 15 }) and
        eql(f32, ellipsis_dot_implicit.data, &.{ 140, 15 }) and
        std.mem.eql(usize, rank4_batched.shape, &.{ 2, 2, 2, 2 }) and
        eql(f32, rank4_batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15, 18, 20, 22, 24, 14, 13, 16, 15 }) and
        eql(f32, rank4_batched_implicit.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15, 18, 20, 22, 24, 14, 13, 16, 15 }) and
        eql(f32, unary_identity.data, &.{ 1, 2, 3, 4, 5, 6 }) and
        std.mem.eql(usize, unary_transpose.shape, &.{ 3, 2 }) and
        eql(f32, unary_transpose.data, &.{ 1, 4, 2, 5, 3, 6 }) and
        eql(f32, unary_row_sum.data, &.{ 6, 15 }) and
        eql(f32, unary_col_sum.data, &.{ 5, 7, 9 }) and
        eql(f32, unary_diag.data, &.{ 1, 4 }) and
        std.mem.eql(usize, unary_trace.shape, &.{}) and
        eql(f32, unary_trace.data, &.{5}) and
        eql(f32, unary_trace_implicit.data, &.{5}) and
        std.mem.eql(usize, batched_diag.shape, &.{ 2, 2 }) and
        eql(f32, batched_diag.data, &.{ 1, 4, 5, 8 }) and
        std.mem.eql(usize, batched_diag_reordered.shape, &.{ 2, 2 }) and
        eql(f32, batched_diag_reordered.data, &.{ 1, 5, 4, 8 }) and
        eql(f32, batched_trace.data, &.{ 5, 13 }) and
        eql(f32, batched_trace_implicit.data, &.{ 5, 13 }) and
        std.mem.eql(usize, ternary_chain.shape, &.{ 2, 2 }) and
        eql(f32, ternary_chain.data, &.{ 250, 372, 601, 894 }) and
        eql(f32, ternary_chain_implicit.data, &.{ 250, 372, 601, 894 }) and
        std.mem.eql(usize, rank4_ternary_chain.shape, &.{ 2, 2, 2, 2 }) and
        eql(f32, rank4_ternary_chain.data, &.{ 1, 2, 3, 4, 22, 22, 30, 30, 36, 40, 44, 48, 13, 14, 15, 16 }) and
        eql(f32, rank4_ternary_chain_implicit.data, &.{ 1, 2, 3, 4, 22, 22, 30, 30, 36, 40, 44, 48, 13, 14, 15, 16 }) and
        eql(f32, same_label_elementwise.data, &.{ 1, 4, 9, 16, 25, 36 }) and
        eql(f32, same_label_rows.data, &.{ 14, 77 }) and
        eql(f32, same_label_all.data, &.{91}) and
        std.mem.eql(usize, same_label_reordered.shape, &.{ 3, 2 }) and
        eql(f32, same_label_reordered.data, &.{ 1, 16, 4, 25, 9, 36 }) and
        unsupported_rejected;

    var stdout_buffer: [1536]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_einsum_smoke\",\"ok\":{},\"matmul_ok\":{},\"implicit_output_ok\":{},\"dot_ok\":{},\"outer_ok\":{},\"matvec_ok\":{},\"vecmat_ok\":{},\"reordered_ok\":{},\"generic_contract_ok\":{},\"batched_matmul_ok\":{},\"rank4_batched_matmul_ok\":{},\"unary_ok\":{},\"ternary_chain_ok\":{},\"rank4_ternary_chain_ok\":{},\"same_label_binary_ok\":{},\"ellipsis_batched_matmul_ok\":{},\"ellipsis_matvec_ok\":{},\"ellipsis_vecmat_ok\":{},\"ellipsis_dot_ok\":{},\"unsupported_rejected\":{}}}\n",
        .{
            ok,
            eql(f32, mm.data, &.{ 58, 64, 139, 154 }),
            eql(f32, mm_implicit.data, &.{ 58, 64, 139, 154 }) and eql(f32, dot_implicit.data, &.{220}) and eql(f32, outer_implicit.data, &.{ 10, 30, 50, 20, 60, 100, 30, 90, 150 }) and eql(f32, batched_implicit.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }) and eql(f32, ellipsis_batched_implicit.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }),
            eql(f32, dot.data, &.{220}),
            eql(f32, outer.data, &.{ 10, 30, 50, 20, 60, 100, 30, 90, 150 }),
            eql(f32, matvec.data, &.{ 14, 32 }),
            eql(f32, vecmat.data, &.{ 58, 64 }),
            eql(f32, transposed_outer.data, &.{ 10, 20, 30, 30, 60, 90, 50, 100, 150 }),
            eql(f32, generic_contract.data, &.{ 22, 28, 49, 64, 76, 100, 103, 136 }),
            eql(f32, batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }),
            eql(f32, rank4_batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15, 18, 20, 22, 24, 14, 13, 16, 15 }),
            eql(f32, unary_identity.data, &.{ 1, 2, 3, 4, 5, 6 }) and eql(f32, unary_transpose.data, &.{ 1, 4, 2, 5, 3, 6 }) and eql(f32, unary_row_sum.data, &.{ 6, 15 }) and eql(f32, unary_col_sum.data, &.{ 5, 7, 9 }) and eql(f32, unary_diag.data, &.{ 1, 4 }) and eql(f32, unary_trace.data, &.{5}) and eql(f32, unary_trace_implicit.data, &.{5}) and eql(f32, batched_diag.data, &.{ 1, 4, 5, 8 }) and eql(f32, batched_diag_reordered.data, &.{ 1, 5, 4, 8 }) and eql(f32, batched_trace.data, &.{ 5, 13 }) and eql(f32, batched_trace_implicit.data, &.{ 5, 13 }),
            eql(f32, ternary_chain.data, &.{ 250, 372, 601, 894 }) and eql(f32, ternary_chain_implicit.data, &.{ 250, 372, 601, 894 }),
            eql(f32, rank4_ternary_chain.data, &.{ 1, 2, 3, 4, 22, 22, 30, 30, 36, 40, 44, 48, 13, 14, 15, 16 }) and eql(f32, rank4_ternary_chain_implicit.data, &.{ 1, 2, 3, 4, 22, 22, 30, 30, 36, 40, 44, 48, 13, 14, 15, 16 }),
            eql(f32, same_label_elementwise.data, &.{ 1, 4, 9, 16, 25, 36 }) and eql(f32, same_label_rows.data, &.{ 14, 77 }) and eql(f32, same_label_all.data, &.{91}) and eql(f32, same_label_reordered.data, &.{ 1, 16, 4, 25, 9, 36 }),
            eql(f32, ellipsis_batched.data, &.{ 1, 2, 3, 4, 11, 11, 15, 15 }),
            eql(f32, ellipsis_matvec.data, &.{ 50, 110, 11, 15 }),
            eql(f32, ellipsis_vecmat.data, &.{ 10, 20, 2, 2 }),
            eql(f32, ellipsis_dot.data, &.{ 140, 15 }),
            unsupported_rejected,
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}
