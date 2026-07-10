//! Basic Vectra Array usage with broadcasting, reductions, and dtype metadata.
//!
//! Run with:
//!   zig build example-basic-array

const std = @import("std");
const vx = @import("vectra");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;

    var x = try vx.Array(f32).fromSlice(
        allocator,
        &.{ 1, 2, 3, 4, 5, 6 },
        &.{ 2, 3 },
    );
    defer x.deinit();

    var bias = try vx.Array(f32).fromSlice(allocator, &.{ 10, 20, 30 }, &.{3});
    defer bias.deinit();

    const broadcast_shape = try x.broadcastShape(bias);
    defer allocator.free(broadcast_shape);

    var y = try x.add(bias);
    defer y.deinit();

    var row_sums = try y.sum(1, false);
    defer row_sums.deinit();

    var scaled = try y.mulScalar(0.5);
    defer scaled.deinit();

    try expectSlice(f32, y.data, &.{ 11, 22, 33, 14, 25, 36 });
    try expectSlice(f32, row_sums.data, &.{ 66, 75 });
    try expectSlice(f32, scaled.data, &.{ 5.5, 11, 16.5, 7, 12.5, 18 });

    var stdout_buffer: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &stdout_buffer);
    try stdout.interface.print(
        \\
        \\{{
        \\  "example": "basic_array",
        \\  "dtype": "{s}",
        \\  "shape": [{d},{d}],
        \\  "bias_shape": [{d}],
        \\  "broadcast_shape": [{d},{d}],
        \\  "y": [{d:.1},{d:.1},{d:.1},{d:.1},{d:.1},{d:.1}],
        \\  "row_sums": [{d:.1},{d:.1}],
        \\  "scaled_first": {d:.1},
        \\  "ok": true
        \\}}
        \\
    , .{
        x.dtypeName(),
        x.shape[0],
        x.shape[1],
        bias.shape[0],
        broadcast_shape[0],
        broadcast_shape[1],
        y.data[0],
        y.data[1],
        y.data[2],
        y.data[3],
        y.data[4],
        y.data[5],
        row_sums.data[0],
        row_sums.data[1],
        scaled.data[0],
    });
    try stdout.interface.flush();
}

fn expectSlice(comptime T: type, actual: []const T, expected: []const T) !void {
    if (actual.len != expected.len) return error.UnexpectedResult;
    for (actual, expected) |a, e| {
        if (a != e) return error.UnexpectedResult;
    }
}
