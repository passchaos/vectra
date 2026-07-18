//! Smoke gate for NumPy/PyTorch-style indexing, gather/scatter, and masking.

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var a = try vx.Array(i32).fromSlice(allocator, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();

    var indices = try vx.Array(usize).fromSlice(allocator, &.{ 2, 0, 1, 2 }, &.{ 2, 2 });
    defer indices.deinit();
    var gathered = try a.gather(1, indices);
    defer gathered.deinit();
    var taken_along = try a.takeAlongAxis(indices, 1);
    defer taken_along.deinit();

    var base = try vx.Array(i32).zeros(allocator, &.{ 2, 3 });
    defer base.deinit();
    var scatter_src = try vx.Array(i32).fromSlice(allocator, &.{ 9, 8, 7, 6 }, &.{ 2, 2 });
    defer scatter_src.deinit();
    var scattered = try base.scatter(1, indices, scatter_src);
    defer scattered.deinit();
    var scatter_added = try base.scatterAdd(1, indices, scatter_src);
    defer scatter_added.deinit();
    var put_axis = try base.putAlongAxis(indices, scatter_src, 1);
    defer put_axis.deinit();

    var mask = try vx.Array(bool).fromSlice(allocator, &.{ true, false, true, false, true, false }, &.{ 2, 3 });
    defer mask.deinit();
    var other = try vx.Array(i32).full(allocator, &.{ 2, 3 }, -1);
    defer other.deinit();
    var where_out = try a.where(mask, other);
    defer where_out.deinit();
    var where_scalar = try a.whereScalar(mask, -9);
    defer where_scalar.deinit();
    var masked = try a.maskedSelect(mask);
    defer masked.deinit();
    var filled = try a.maskedFill(mask, 42);
    defer filled.deinit();

    var flat_indices = try vx.Array(usize).fromSlice(allocator, &.{ 0, 3, 5 }, &.{3});
    defer flat_indices.deinit();
    var index_put_values = try vx.Array(i32).fromSlice(allocator, &.{ 10, 20, 30 }, &.{3});
    defer index_put_values.deinit();
    var index_put = try a.indexPut(flat_indices, index_put_values);
    defer index_put.deinit();

    const mismatch_rejected = blk: {
        var bad_indices = try vx.Array(usize).fromSlice(allocator, &.{ 0, 1, 2 }, &.{3});
        defer bad_indices.deinit();
        var bad = a.gather(1, bad_indices) catch |err| {
            break :blk err == error.ShapeMismatch;
        };
        bad.deinit();
        break :blk false;
    };

    const ok =
        eql(i32, gathered.data, &.{ 3, 1, 5, 6 }) and
        eql(i32, taken_along.data, gathered.data) and
        eql(i32, scattered.data, &.{ 8, 0, 9, 0, 7, 6 }) and
        eql(i32, scatter_added.data, &.{ 8, 0, 9, 0, 7, 6 }) and
        eql(i32, put_axis.data, scattered.data) and
        eql(i32, where_out.data, &.{ 1, -1, 3, -1, 5, -1 }) and
        eql(i32, where_scalar.data, &.{ 1, -9, 3, -9, 5, -9 }) and
        eql(i32, masked.data, &.{ 1, 3, 5 }) and
        eql(i32, filled.data, &.{ 42, 2, 42, 4, 42, 6 }) and
        eql(i32, index_put.data, &.{ 10, 2, 3, 20, 5, 30 }) and
        mismatch_rejected;

    var stdout_buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        "{{\"kind\":\"vectra_indexing_smoke\",\"ok\":{},\"gather_ok\":{},\"scatter_ok\":{},\"where_ok\":{},\"masked_ok\":{},\"index_put_ok\":{},\"mismatch_rejected\":{}}}\n",
        .{
            ok,
            eql(i32, gathered.data, &.{ 3, 1, 5, 6 }),
            eql(i32, scattered.data, &.{ 8, 0, 9, 0, 7, 6 }),
            eql(i32, where_out.data, &.{ 1, -1, 3, -1, 5, -1 }),
            eql(i32, masked.data, &.{ 1, 3, 5 }),
            eql(i32, index_put.data, &.{ 10, 2, 3, 20, 5, 30 }),
            mismatch_rejected,
        },
    );
    try stdout.interface.flush();
    if (!ok) std.process.exit(1);
}

fn eql(comptime T: type, actual: []const T, expected: []const T) bool {
    return std.mem.eql(T, actual, expected);
}
