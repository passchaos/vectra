//! Smoke gate for NumPy/PyTorch-style general contraction helpers.
//!
//! `tensordot` is a front-end helper over Array.contractAxes; execution remains
//! on existing Array/Axiom matmul/contraction paths.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var a = try vx.Array(f32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try vx.Array(f32).fromSlice(allocator, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 3, 2 });
    defer b.deinit();
    var matrix_contract = try vx.tensordot(a, b, &.{1}, &.{0});
    defer matrix_contract.deinit();
    var matrix_contract_alias = try vx.tensorDot(a, b, &.{1}, &.{0});
    defer matrix_contract_alias.deinit();

    var tensor_a = try vx.Array(f32).fromSlice(allocator, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    }, &.{ 2, 2, 3 });
    defer tensor_a.deinit();
    var tensor_b = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2,
        3, 4,
        5, 6,
    }, &.{ 3, 2 });
    defer tensor_b.deinit();
    var rank3_contract = try vx.tensordot(tensor_a, tensor_b, &.{2}, &.{0});
    defer rank3_contract.deinit();

    var multi_a = try vx.Array(f32).fromSlice(allocator, &.{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    }, &.{ 2, 2, 2 });
    defer multi_a.deinit();
    var multi_b = try vx.Array(f32).fromSlice(allocator, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    }, &.{ 2, 2, 3 });
    defer multi_b.deinit();
    var multi_axis_contract = try vx.tensordot(multi_a, multi_b, &.{ 1, 2 }, &.{ 0, 1 });
    defer multi_axis_contract.deinit();

    const mismatch_rejected = blk: {
        var bad = vx.tensordot(a, b, &.{0}, &.{0}) catch |err| {
            break :blk err == error.ShapeMismatch;
        };
        bad.deinit();
        break :blk false;
    };

    const ok =
        std.mem.eql(usize, matrix_contract.shape, &.{ 2, 2 }) and
        eql(f32, matrix_contract.data, &.{ 58, 64, 139, 154 }) and
        eql(f32, matrix_contract_alias.data, &.{ 58, 64, 139, 154 }) and
        std.mem.eql(usize, rank3_contract.shape, &.{ 2, 2, 2 }) and
        eql(f32, rank3_contract.data, &.{ 22, 28, 49, 64, 76, 100, 103, 136 }) and
        std.mem.eql(usize, multi_axis_contract.shape, &.{ 2, 3 }) and
        eql(f32, multi_axis_contract.data, &.{ 70, 80, 90, 158, 184, 210 }) and
        mismatch_rejected;

    var stdout_buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_contraction_smoke\",\"ok\":{},\"matrix_ok\":{},\"rank3_ok\":{},\"multi_axis_ok\":{},\"alias_ok\":{},\"mismatch_rejected\":{}}}\n",
        .{
            ok,
            eql(f32, matrix_contract.data, &.{ 58, 64, 139, 154 }),
            eql(f32, rank3_contract.data, &.{ 22, 28, 49, 64, 76, 100, 103, 136 }),
            eql(f32, multi_axis_contract.data, &.{ 70, 80, 90, 158, 184, 210 }),
            eql(f32, matrix_contract_alias.data, &.{ 58, 64, 139, 154 }),
            mismatch_rejected,
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}
